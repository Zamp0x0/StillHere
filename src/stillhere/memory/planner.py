"""QueryPlanner: one user message in, a ``QueryPlan`` out (Day 7, Query Understanding V2).

``SlotRetriever.retrieve`` calls ``plan`` first and then runs every atom on its own. Planning is deterministic,
local and reads nothing from the store. Steps:

1. **Parts.** The NFKC text is cut at whitespace (bubbles) and punctuation. A part starting with ``还是`` stays with
   the part before it (one alternative question). A part asks something (``question``) when it ends with "?" or has
   a question word, or (``request``) a request word and a fact it asks for; words that only look like questions
   (``几乎``, ``没什么``) do not count. Other parts are ``statement`` s. An asked part that joins two owned facts with
   a conjunction (``你生日和你妈妈生日分别是几月``) is cut into one part per owner.
2. **Which parts become atoms.** A message of one part, a message whose parts all are statements, and a query with
   an explicit ``subject`` are read whole, as one unit (the Day 6 reading, so keyword lookups such as ``十月`` keep
   working). Otherwise every asked part is a unit and remarks (``我最近也在找工作``) are left out.
3. **Intent** (``SlotIntentResolver``): attribute cue words (``住`` -> ``home_city``) and a closed set of phrases
   for asking what someone likes to do (``喜欢做什么``, ``喜欢哪些活动``, a final ``喜欢什么``) -> ``hobby``.
   ``喜欢`` + a person (``你喜欢我吗``) is a social question memory does not record (``unsupported``); together
   with another intent in one part it is ``unknown``. ``喜欢吃``/``喜欢这首歌`` name an object, not an activity, and
   stay keyword lookups. No cue: ``keyword``.
4. **Owner** of each unit (the Day 7 Phase 3 ownership rules, applied per unit): the nearest owner reference before
   the first noun or intent match, with only closed-class words between (``你妈妈住哪`` -> ``unowned``); plural,
   quoted or unidentified references are ``ambiguous``. Two references joined by a conjunction (``你和我分别住哪``,
   ``你和我谁生日更早``) give one atom per owner. A unit with no owner reference takes the owner of the asked part
   before it in the same message (``你喜欢做什么？在哪上班？``); an owner found without a reference before the
   asked fact (a pronoun after it, a context-only ``她``) sends the unit back to reading the whole message, which
   is what keeps ``猫叫什么来着？我忘了`` the user's question. Never: a subject from another atom's attribute.
5. **Ellipsis and dialogue focus.** A unit that is only an owner and/or a time word with ``呢``/``也是吗``
   (``那你呢``, ``我呢``, ``你爸爸呢``, ``以前呢``) takes its attribute from the atom before it in the same message,
   else from the ``DialogueFocus`` of the context: the newest turn and, when that is the persona's reply, the user
   turn before it. The focus is invalid (nothing is carried) without speaker roles, with two candidate topics or
   two attributes, when a cue is quoted, or when the persona's newest turn asked about something else. Only the
   attribute (and the kind of thing it belonged to, ``猫``) is carried: the owner comes from the unit's own
   pronoun, else from the atom before it, else from the focus turn read with its speaker (the user's ``我`` is the
   user, the persona's ``我`` the persona), else the atom is ``ambiguous``; it never falls back to "the only subject
   with that attribute". The focus is derived again for every message and never stored.
6. **Time** (Phase 5B: ``stillhere.memory.temporal``). Each unit's time words are read first. A calendar time
   (``去年``, ``2026年3月10号``, ``今年三月初``) makes the scope ``at`` and puts its ``TimeExpression`` on the atom; a
   time no calendar places (``前阵子``, ``2026年2月30号``) makes it ``unresolved`` (nothing is returned). Those spans
   are taken out of the text before owners and intents are read, so ``你今年三月初住哪`` is the persona's home, not an
   unowned fact. A month without a year counts only before the asked fact or with a time suffix, and never in a
   birthday question (``你生日是十月还是十一月``: the months are the values). Without calendar words: explicit, else
   the Day 5 open scopes (现在 / 以前 / 一直), else the time of the asked part before it in the same message
   (``你以前住哪，做什么工作``, ``你去年住哪，做什么工作``); a life period is unplaced whatever the scope says.
   Stage K: ``那时候`` alone, after a part with a calendar time, points back to that time (``你前年住在哪？那时候在哪
   工作？`` asks about 前年), not to the open past.
7. **Merge, number, hash.** Atoms with the same owner and attributes merge (``你住哪？以前呢？`` is one atom over
   all time); an atom with a calendar time never merges (``你住哪？去年呢？`` asks two times). Atoms are numbered ``q1..qn`` in message order and the plan id hashes the input and the atoms.
8. **Optional semantic parser.** Only when a parser is configured and the plan has an asked part with no
   attribute. Its answer is a candidate: ``validate_candidate`` keeps only attributes from the vocabulary for parts
   the local rules left without one, and an owner hint may only make an atom more cautious (``unowned`` or
   ``ambiguous``), never assign a subject. Timeouts, bad JSON or a rejected candidate leave the local plan unchanged
   and add a note.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from datetime import tzinfo

from stillhere.contracts import (AtomicQuery, DialogueFocus, IntentStatus, MemoryQuery, QueryDependency, QueryPart,
                                 QueryPlan, SpeechAct, SubjectStatus, TimeExpression, TimeKind, TimeScope,
                                 canonical_json)
from stillhere.contracts.planning import DependencyKind, FocusStatus

from .search import normalize
from .temporal import (CALENDAR_TZ, CURRENT_CUES, DURATION_CUES, HISTORICAL_CUES, LIFE_PERIOD_CUES, RELATIVE_PAST_CUES,
                       TIME_WORDS, find_times)

PLANNER_NAME = "local_rules_v1"


def _alternatives(words) -> str:
    return "|".join(re.escape(w) for w in sorted(set(words), key=lambda w: (-len(w), w)) if w)


@dataclass(frozen=True)
class SlotCue:
    attribute: str
    cues: Tuple[str, ...]


# Attribute vocabulary of the store. A slot not listed here is still reachable through keywords.
DEFAULT_SLOT_CUES: Tuple[SlotCue, ...] = (
    SlotCue("home_city", ("住", "搬家", "搬到", "搬去", "哪个城市", "家在")),
    SlotCue("occupation", ("工作", "上班", "职业", "做什么的", "干什么的", "做哪行")),
    SlotCue("name", ("名字", "叫什么", "叫啥", "名叫")),
    SlotCue("birthday_month", ("生日",)),
    SlotCue("dislikes_food", ("不吃", "忌口", "讨厌吃", "不爱吃")),
    SlotCue("hobby", ("爱好", "兴趣", "业余")),
)
# Nouns naming a non-person subject.
DEFAULT_SUBJECT_NOUNS: Mapping[str, str] = {"猫咪": "cat", "小猫": "cat", "猫": "cat", "小狗": "dog", "狗": "dog"}

# The time word lists (HISTORICAL_CUES, CURRENT_CUES, DURATION_CUES, RELATIVE_PAST_CUES, LIFE_PERIOD_CUES,
# TIME_WORDS) live in .temporal since Phase 5B and are imported above.
PERSON_PRONOUNS = "她他"
THING_PRONOUN = "它"
PERSONA, USER = "persona", "user"
# Query parts: bubbles are joined with spaces and NFKC turns full-width ，？！ into ASCII.
_PART = re.compile(r"[^\s,.!?;:。、~]+")
QUESTION_WORDS = ("什么", "啥", "哪", "几", "多少", "谁", "怎么", "怎样", "如何", "干嘛", "吗", "呢", "还是", "是不是",
                  "有没有", "来着", "是否", "能不能", "会不会", "对不对", "要不要")
REQUEST_WORDS = ("告诉", "说说", "讲讲", "聊聊", "提醒", "帮我", "查查", "查一下", "想知道", "想问", "问问", "问一下",
                 "请问", "介绍")
NOT_QUESTION = ("几乎", "几近", "好几", "无几", "未几", "没什么", "不怎么", "怎么也", "怎么都", "什么都", "什么也",
                "多多少少", "哪怕", "谁都", "谁也")
PLURAL_REFS = ("我们", "咱们", "你们", "大家", "她们", "他们", "它们", "咱俩", "我俩", "你俩", "她俩", "他俩", "咱")
# The only words allowed between an owner reference and the slot cue or noun it owns: "你家那只猫", "你现在在哪个城市",
# "你做什么工作", "你几月生日". Any other word ("你妈妈住", "你男朋友做", "你家那位住") may make the fact someone else's.
# Stage K: modal adverbs too ("你可能在做什么工作", "你应该已经在咖啡馆上班了吧" are the persona's own job).
_OWNER_GAP = re.compile("(?:" + "|".join(sorted({
    "可能", "应该", "大概", "好像", "也许", "或许", "估计", "已经",
    "的", "家", "那", "这", "只", "个", "在", "还", "也", "都", "又", "就", "是", "到底", "究竟", "平时", "平常", "一般",
    "通常", "周末", "平日", "每天", "有空", "一直", "最", "最近", "小时候", "本来", "自己", "真的", "具体", "分别", "各自",
    "做", "干", "去", "回", "有", "要", "想", "打算", "准备", "会", "能", "哪", "哪里", "哪儿", "哪个", "什么", "啥", "几",
    "多少", "怎么", "呢", "吗", "啊", "呀", "吧", "嘛", "年", "月", "月份", "号", "天", "岁", "时候", *TIME_WORDS},
    key=lambda w: (-len(w), w))) + ")*")
_CONJUNCTION = re.compile("和|跟|与|及|以及|还有|同")
# Kinship and social words: in a query with no slot cue or noun (where _OWNER_GAP has nothing to measure) they mark
# the fact as someone else's ("你妹妹几岁", "你妈妈喜欢香菜吗").
RELATION_WORDS = ("妈", "爸", "父", "母", "哥", "姐", "弟", "妹", "爷", "奶", "姥", "外公", "外婆", "叔", "姨", "舅", "姑",
                  "婶", "伯", "儿子", "女儿", "孩子", "老公", "老婆", "丈夫", "妻子", "对象", "男朋友", "女朋友", "男友",
                  "女友", "朋友", "同事", "同学", "室友", "舍友", "老师", "老板", "领导", "邻居", "家人", "亲戚", "闺蜜",
                  "兄弟", "前任", "那位", "那口子")
_QUOTES = (("“", "”"), ("「", "」"), ("『", "』"), ('"', '"'))
_OWNER_OF = {"I": USER, "you": PERSONA, "name": PERSONA}

# Asking what someone likes to do: 喜欢 + do/play + what, 喜欢 + which/what + activity, or a final 喜欢什么.
_ACTIVITY_LIKE = re.compile("喜欢(?:做|干|玩)(?:些什么|什么|啥|哪些)|喜欢(?:哪些|什么|啥)(?:活动|运动|事情|事)|喜欢干嘛")
_LIKE_WHAT_END = re.compile("喜欢(?:什么|啥)(?=[呢呀啊吗嘛哦]*$)")
_LIKE = re.compile("喜欢")
# The answer to an elliptical part ("那你呢", "你爸爸呢", "以前呢") is the question asked just before it.
_ELLIPSIS_END = ("也是这样吗", "也这样吗", "也一样吗", "也一样", "也是吗", "也是", "呢")
_KIN_PREFIX = "表堂亲大小老"
_RELATION_RUN = re.compile(f"(?:[{_KIN_PREFIX}])?(?:{_alternatives(RELATION_WORDS)})+")
# Comparisons between two joined owners: 你和我谁生日更早.
_COMPARE = re.compile("谁(?:的)?.*?(更早|更晚|更大|更小|更远|更近|更多|更少|早|晚|先|后|大|小|远|近|多|少)")
_RELATION_OF = {"更早": "earlier", "早": "earlier", "先": "earlier", "更晚": "later", "晚": "later", "后": "later",
                "更大": "greater", "大": "greater", "更小": "smaller", "小": "smaller", "更远": "farther",
                "远": "farther", "更近": "nearer", "近": "nearer", "更多": "more", "多": "more", "更少": "less",
                "少": "less"}
# Comparisons with 比 (Phase 5B): 你生日比我早吗, 我的生日比你晚还是早. The second owner comes right after 比 and
# the relation after it.
_VERSUS_RELATION = re.compile("更早|更晚|更大|更小|更远|更近|更多|更少|早|晚|先|后|大|小|远|近|多|少")
# Comparisons the records can decide: the month of a birthday is ordered.
COMPARABLE = {("birthday_month", "earlier"), ("birthday_month", "later")}


class _Ambiguous(Exception):
    def __init__(self, reason: str = "unidentified") -> None:
        super().__init__(reason)
        self.reason = reason


class _Unowned(Exception):
    """The asked fact belongs to someone with no subject in memory (``你妈妈``): no slots, no records."""

    def __init__(self, word: str = "relation") -> None:
        super().__init__(word)
        self.word = word


def _find(text: str, word: str) -> int:
    return text.find(normalize(word))


def _points_back(part: str) -> bool:
    """Stage K: the part's only time word is ``那时候`` (a historical cue that, after a dated part, names that part's
    time again); a part with any other time word has its own time."""
    words = {w for w in TIME_WORDS if _find(part, w) >= 0}
    return bool(words) and words <= {"那时候"}


def _asks(part: str, words: Sequence[str]) -> bool:
    for word in NOT_QUESTION:
        part = part.replace(word, " ")
    return any(w in part for w in words)


def _quoted(text: str, pos: int) -> bool:
    before = text[:pos]
    for start, end in _QUOTES:
        if before.count(start) % 2 if start == end else before.rfind(start) > before.rfind(end):
            return True
    return False


def _pronoun(text: str, chars: str) -> Optional[str]:
    """'singular' or 'plural' when one of ``chars`` occurs as a pronoun (plural: followed by 们), else None."""
    found = None
    for i, c in enumerate(text):
        if c in chars:
            if text[i + 1:i + 2] == "们":
                return "plural"
            found = "singular"
    return found


# ---------------------------------------------------------------- slot intents

@dataclass(frozen=True)
class IntentMatch:
    attribute: str
    start: int
    words: Tuple[str, ...]     # the cue words or the phrase that matched
    rule: str                  # "cue" or "phrase"


@dataclass(frozen=True)
class SlotIntent:
    status: IntentStatus
    matches: Tuple[IntentMatch, ...] = ()      # one per attribute, in order of first appearance
    ambiguity: Optional[str] = None
    at: Optional[int] = None                   # where an unsupported or unknown intent is named (its 喜欢)

    @property
    def attributes(self) -> Tuple[str, ...]:
        return tuple(m.attribute for m in self.matches)

    @property
    def anchor(self) -> Optional[int]:
        return min((m.start for m in self.matches), default=self.at)

    @property
    def words(self) -> Tuple[str, ...]:
        return tuple(w for m in self.matches for w in m.words)

    @property
    def source(self) -> Optional[str]:
        if not self.matches:
            return None
        return "phrase" if any(m.rule == "phrase" for m in self.matches) else "cue"


class SlotIntentResolver:
    """Which memory attributes a clause asks about. Cue words come from the slot vocabulary; the only phrases are
    the ones that ask what someone likes to do (``hobby``). ``喜欢`` + a person is social (``unsupported``), and
    mixed with another intent in the same clause ``unknown``; an unsupported intent never takes the nearest slot."""

    def __init__(self, slot_cues: Sequence[SlotCue] = DEFAULT_SLOT_CUES, persona_names: Sequence[str] = ()) -> None:
        self._slot_cues = tuple(slot_cues)
        self._attributes = tuple(c.attribute for c in self._slot_cues)
        people = ["我", "你", "她", "他", "谁", "人家", *persona_names, *RELATION_WORDS]
        self._social = re.compile("喜欢(?:" + _alternatives(people) + ")")

    @property
    def attributes(self) -> Tuple[str, ...]:
        return self._attributes

    def resolve(self, text: str) -> SlotIntent:
        """``text`` is normalized; a clause, or several (a context turn), each read on its own."""
        matches: List[Tuple[int, int, str, Tuple[str, ...], str]] = []
        social: Optional[int] = None
        likes = socials = 0
        for clause in _PART.finditer(text):
            c, base = clause.group(), clause.start()
            found = self._clause(c)
            likes += len(_LIKE.findall(c))
            at = [m.start() for m in self._social.finditer(c)]
            socials += len(at)
            if at and social is None:
                social = base + at[0]
            matches += [(base + pos, order, attr, words, rule) for pos, order, attr, words, rule in found]
        if social is not None and (matches or likes > socials):   # 你喜欢我，还是喜欢攀岩: a person and something else
            return SlotIntent(IntentStatus.UNKNOWN, ambiguity="unknown:mixed_social", at=social)
        if social is not None:
            return SlotIntent(IntentStatus.UNSUPPORTED, ambiguity="unsupported:social", at=social)
        out: Dict[str, IntentMatch] = {}
        for pos, _, attr, words, rule in sorted(matches):
            if attr in out:
                prev = out[attr]
                out[attr] = IntentMatch(attr, prev.start, tuple(dict.fromkeys(prev.words + words)),
                                        "phrase" if "phrase" in (prev.rule, rule) else "cue")
            else:
                out[attr] = IntentMatch(attr, pos, words, rule)
        if not out:
            return SlotIntent(IntentStatus.KEYWORD)
        return SlotIntent(IntentStatus.SLOT, tuple(sorted(out.values(), key=lambda m: (m.start, m.attribute))))

    def _clause(self, c: str) -> List[Tuple[int, int, str, Tuple[str, ...], str]]:
        found = []
        for order, cue in enumerate(self._slot_cues):
            hit = [(pos, w) for w in cue.cues for pos in [_find(c, w)] if pos >= 0]
            if hit:
                found.append((min(p for p, _ in hit), order, cue.attribute, tuple(w for _, w in sorted(hit)), "cue"))
        if "hobby" in self._attributes and not any(f[2] == "hobby" for f in found):
            m = _ACTIVITY_LIKE.search(c) or _LIKE_WHAT_END.search(c)
            if m is not None:
                found.append((m.start(), self._attributes.index("hobby"), "hobby", (m.group(),), "phrase"))
        return found


# ---------------------------------------------------------------- the planner

@dataclass
class _Owner:
    status: SubjectStatus
    subject: Optional[str] = None        # resolved: the subject; for a person owner before nouns are applied too
    source: Optional[str] = None
    by_ref: bool = False                 # an owner reference before the asked fact decided it
    ambiguity: Optional[str] = None
    owner: Optional[str] = None          # "persona" / "user": whose it is (a noun subject belongs to the persona)


@dataclass
class _Draft:
    parts: List[int]
    span: str
    act: SpeechAct
    owner: _Owner
    intent: SlotIntent
    intent_source: Optional[str]
    attributes: Tuple[str, ...]
    noun: Optional[str]
    scope: TimeScope
    scope_source: str
    period: bool
    time: Optional[TimeExpression] = None   # Phase 5B: scope "at" or "unresolved"
    reason: List[str] = field(default_factory=list)
    intent_ambiguity: Optional[str] = None
    group: Optional[int] = None          # joined owners of one part share a group
    compare: Optional[Tuple[Optional[str], bool]] = None   # (relation, supported) for a comparison group


class QueryPlanner:
    """Deterministic local planner. ``persona_names`` maps a persona id to the names it is called by; ``parser``
    is an optional semantic parser (``stillhere.memory.semantic_parser``), asked at most once per message
    (``max_parser_calls`` 1; 0 turns it off) and only about asked parts the rules leave without an attribute."""

    def __init__(self, *, persona_names: Optional[Mapping[str, Sequence[str]]] = None,
                 slot_cues: Sequence[SlotCue] = DEFAULT_SLOT_CUES,
                 subject_nouns: Mapping[str, str] = DEFAULT_SUBJECT_NOUNS, parser: Any = None,
                 max_parser_calls: int = 1, calendar_tz: tzinfo = CALENDAR_TZ) -> None:
        self._names = {k: tuple(normalize(n) for n in v) for k, v in (persona_names or {}).items()}
        self._slot_cues = tuple(slot_cues)
        self._nouns = sorted(((normalize(k), v) for k, v in subject_nouns.items()), key=lambda kv: -len(kv[0]))
        self._thing_subjects = frozenset(subject_nouns.values())
        self._resolvers: Dict[str, SlotIntentResolver] = {}
        self._parser = parser
        if max_parser_calls not in (0, 1):     # one request covers every asked part of a message
            raise ValueError("max_parser_calls must be 0 (off) or 1")
        self._max_parser_calls = max_parser_calls
        self._tz = calendar_tz
        self.name = PLANNER_NAME

    @property
    def thing_subjects(self) -> frozenset:
        return self._thing_subjects

    def resolver(self, persona_id: str) -> SlotIntentResolver:
        if persona_id not in self._resolvers:
            self._resolvers[persona_id] = SlotIntentResolver(self._slot_cues, self._names.get(persona_id, ()))
        return self._resolvers[persona_id]

    # ------------------------------------------------------------------ public

    def plan(self, query: MemoryQuery) -> QueryPlan:
        """Raises ValueError for a message with no words (the retriever answers it with no_match)."""
        text = normalize(query.text)
        parts = self._parts(query.persona_id, text)
        if not parts:
            raise ValueError("the message has no words to plan")
        focus = self._dialogue_focus(query)
        drafts = self._drafts(query, text, parts, focus)
        drafts = self._merge(drafts)
        plan = self._build(query, text, parts, focus, drafts)
        if self._parser is not None and self._max_parser_calls and plan.confidence == "low":
            plan = self._consult(query, plan)
        return plan

    # ------------------------------------------------------------------ parts

    def _parts(self, persona_id: str, text: str) -> List[QueryPart]:
        raw: List[List] = []
        for m in _PART.finditer(text):
            after = _PART.search(text, m.end())
            mark = "?" in text[m.end():after.start() if after else len(text)]
            if raw and m.group().startswith("还是"):          # 你喜欢我，还是喜欢攀岩: one alternative question
                raw[-1][1], raw[-1][2] = m.end(), mark
            else:
                raw.append([m.start(), m.end(), mark])
        spans: List[Tuple[int, int, SpeechAct]] = []
        for start, end, mark in raw:
            p = text[start:end]
            if mark or _asks(p, QUESTION_WORDS):
                act = SpeechAct.QUESTION
            elif _asks(p, REQUEST_WORDS) and self._anchor(persona_id, p) is not None:
                act = SpeechAct.REQUEST
            else:
                act = SpeechAct.STATEMENT
            if act is SpeechAct.STATEMENT:
                spans.append((start, end, act))
            else:
                spans += [(s, e, act) for s, e in self._split_conjoined(persona_id, text, start, end)]
        return [QueryPart(index=i, text=text[s:e], start=s, end=e, act=act) for i, (s, e, act) in enumerate(spans)]

    def _split_conjoined(self, persona_id: str, text: str, start: int, end: int) -> List[Tuple[int, int]]:
        """Cut ``你生日和你妈妈生日分别是几月`` at the conjunction: each side names its own owner and asked fact."""
        p = text[start:end]
        for m in _CONJUNCTION.finditer(p):
            left, right = p[:m.start()], p[m.end():]
            refs = self._refs(persona_id, right)
            if (left and self._anchor(persona_id, left) is not None and refs and refs[0][0] == 0
                    and self._anchor(persona_id, right) is not None):
                return [(start, start + m.start())] + self._split_conjoined(persona_id, text, start + m.end(), end)
        return [(start, end)]

    # ------------------------------------------------------------------ vocabulary helpers

    def _noun_spans(self, text: str) -> List[Tuple[int, str]]:
        """(start, subject) of each subject noun, in text order."""
        found: List[Tuple[int, str]] = []
        taken = [False] * len(text)
        for word, subj in self._nouns:            # longest first, so 猫咪 is not also counted as 猫
            start = text.find(word)
            while start >= 0:
                if not any(taken[start:start + len(word)]):
                    found.append((start, subj))
                    for i in range(start, start + len(word)):
                        taken[i] = True
                start = text.find(word, start + 1)
        return sorted(found)

    def _nouns_in(self, text: str) -> List[str]:
        out: List[str] = []
        for _, subj in self._noun_spans(text):
            if subj not in out:
                out.append(subj)
        return out

    def _named(self, persona_id: str, text: str) -> bool:
        return any(n and n in text for n in self._names.get(persona_id, ()))

    def _refs(self, persona_id: str, text: str) -> List[Tuple[int, int, str]]:
        """Owner references in text order: (start, end, kind), kind "I" (我), "you" (你), "name" (the persona's
        name), "person" (她/他), "thing" (它) or "plural" (我们, 你们, ...)."""
        found: List[Tuple[int, int, str]] = []
        taken = [False] * len(text)
        words = [(w, "plural") for w in PLURAL_REFS] + [(n, "name") for n in self._names.get(persona_id, ()) if n]
        for word, kind in words:
            start = text.find(word)
            while start >= 0:
                if not any(taken[start:start + len(word)]):
                    found.append((start, start + len(word), kind))
                    for i in range(start, start + len(word)):
                        taken[i] = True
                start = text.find(word, start + 1)
        kinds = {"我": "I", "你": "you", "她": "person", "他": "person", "它": "thing"}
        found += [(i, i + 1, kinds[c]) for i, c in enumerate(text) if c in kinds and not taken[i]]
        return sorted(found)

    def _anchor(self, persona_id: str, text: str, spans: Optional[List[Tuple[int, str]]] = None) -> Optional[int]:
        """Where the asked fact is named: the first subject noun, else the first intent match."""
        spans = self._noun_spans(text) if spans is None else spans
        if spans:
            return spans[0][0]
        return self.resolver(persona_id).resolve(text).anchor

    def _relation_follows(self, text: str, end: int) -> bool:
        rest = text[end:]
        heads = (rest, rest.lstrip("的"), rest.lstrip("的家"), rest.lstrip("的家那这个"))
        return any(h.startswith(w) for h in heads for w in RELATION_WORDS)

    def _relation_after(self, text: str, end: int) -> str:
        """The kinship or social word after an owner reference (妈妈, 男朋友), for the reason of an unowned atom."""
        m = _RELATION_RUN.match(text[end:].lstrip("的家那这个"))
        return m.group() if m else "relation"

    def _relation_in_clause(self, text: str, end: int) -> bool:
        clause = _PART.match(text, end)
        return clause is not None and any(w in clause.group() for w in RELATION_WORDS)

    def _possessed(self, persona_id: str, text: str) -> bool:
        """True when the persona's name owns someone here ("乔沫的妈妈"): 她 may be that person."""
        intent_words = [normalize(w) for cue in self._slot_cues for w in cue.cues]
        for name in self._names.get(persona_id, ()):
            i = text.find(name) if name else -1
            while i >= 0:
                end = i + len(name)
                head = text[end + 1:] if text[end:end + 1] == "的" else None
                if self._relation_follows(text, end) or (head is not None and not (
                        any(head.startswith(w) for w, _ in self._nouns)
                        or any(head.startswith(w) for w in intent_words))):
                    return True
                i = text.find(name, end)
        return False

    # ------------------------------------------------------------------ time

    def _calendar(self, query: MemoryQuery, q: str) -> Tuple[Optional[TimeExpression], List[Tuple[int, int]]]:
        """The calendar time of a unit (None if it names none) and the spans of its time words. Two different
        calendar times in one unit cannot be told apart: the unit's time is then unresolved."""
        anchor = self._anchor(query.persona_id, q)
        birthday = "birthday_month" in self.resolver(query.persona_id).resolve(q).attributes
        found = [m for m in find_times(q, query.as_of, self._tz)
                 if m.expression.kind in (TimeKind.ABSOLUTE_POINT, TimeKind.ABSOLUTE_INTERVAL,
                                          TimeKind.RELATIVE_INTERVAL, TimeKind.UNRESOLVED_TIME)
                 and m.scope_for(anchor) and not (birthday and not m.anchored)]   # 你生日是十月还是十一月: values
        if not found:
            return None, []
        spans = [(m.start, m.end) for m in found]
        times = list(dict.fromkeys((m.expression.kind, m.expression.start, m.expression.end) for m in found))
        if len(times) == 1:
            return found[0].expression, spans
        first = found[0].expression
        return TimeExpression(kind=TimeKind.UNRESOLVED_TIME, source_text=" ".join(m.expression.source_text
                                                                                     for m in found),
                              as_of=first.as_of, timezone=first.timezone, granularity="unknown",
                              resolution_status="unresolved"), spans

    @staticmethod
    def _masked(q: str, spans: Sequence[Tuple[int, int]]) -> str:
        """The unit without its calendar time words, for reading owners and intents."""
        if not spans:
            return q
        keep = [True] * len(q)
        for start, end in spans:
            for i in range(start, end):
                keep[i] = False
        return "".join(c for c, k in zip(q, keep) if k)

    def _time(self, query: MemoryQuery, q: str) -> Tuple[TimeScope, str, bool, Optional[TimeExpression], str]:
        """(scope, source, life period, calendar time, the unit with its calendar time words taken out)."""
        time, spans = self._calendar(query, q)
        masked = self._masked(q, spans)
        if query.time_scope is not None or time is None:
            scope, source, period = self._scope(query, q)
            return scope, source, period, None, masked
        period = any(_find(q, w) >= 0 for w in LIFE_PERIOD_CUES)
        return (TimeScope.AT if time.placed else TimeScope.UNRESOLVED), "query", period, time, masked

    def _scope(self, query: MemoryQuery, q: str) -> Tuple[TimeScope, str, bool]:
        """(scope, source, whether a life period no record places is asked about)."""
        period = any(_find(q, w) >= 0 for w in LIFE_PERIOD_CUES)   # unplaced whatever the scope says
        if query.time_scope is not None:
            return query.time_scope, "explicit", period
        past = any(_find(q, w) >= 0 for w in HISTORICAL_CUES + RELATIVE_PAST_CUES + LIFE_PERIOD_CUES)
        now = any(_find(q, w) >= 0 for w in CURRENT_CUES)
        if (past and now) or any(_find(q, w) >= 0 for w in DURATION_CUES):
            return TimeScope.ALL, "query", period
        if past:
            return TimeScope.HISTORICAL, "query", period
        if now:
            return TimeScope.CURRENT, "query", period
        return TimeScope.CURRENT, "default", period

    # ------------------------------------------------------------------ owners

    def owner_of(self, query: MemoryQuery, q: str) -> Tuple[Optional[str], Optional[str], bool]:
        """(subject, source, by_ref) of a text read as one unit; raises _Ambiguous / _Unowned."""
        if query.subject is not None:
            return query.subject, "explicit", False
        spans = self._noun_spans(q)
        anchor = self._anchor(query.persona_id, q, spans)
        refs = self._refs(query.persona_id, q)
        before = [r for r in refs if anchor is not None and r[1] <= anchor]
        nouns = self._nouns_in(q)
        if not before:
            subject, source = self._legacy_subject(query, q, nouns, refs, anchor)
            return subject, source, False
        if len(nouns) > 1:
            raise _Ambiguous("two_nouns")
        start, end, kind = before[-1]
        if len(before) > 1 and _CONJUNCTION.fullmatch(q[before[-2][1]:start]) and (
                _OWNER_OF.get(before[-2][2], before[-2][2]) != _OWNER_OF.get(kind, kind)):
            raise _Ambiguous("joined")                    # 你和我: two owners (the planner splits them first)
        if kind == "plural":
            raise _Ambiguous("plural")                    # 我们家猫
        if _quoted(q, start):
            raise _Ambiguous("quoted")                    # someone else's quoted 我
        if _OWNER_GAP.fullmatch(q[end:anchor]) is None:
            raise _Unowned(self._relation_after(q, end))  # 你妈妈住哪: not the persona's own fact
        if kind == "I":
            subject, source, by_ref = USER, "query", True
        elif kind in ("you", "name"):
            subject, source, by_ref = (nouns[0] if nouns else PERSONA), "query", True
        else:
            subject, source = self._third_person(query, q, kind == "person", kind == "thing", nouns)
            by_ref = False
        if self._another_owner_after(query.persona_id, q, anchor, USER if subject == USER else PERSONA):
            raise _Ambiguous("two_owners")                # 你生日我生日: two owners in one clause
        return subject, source, by_ref

    def _another_owner_after(self, persona_id: str, q: str, anchor: int, owner: str) -> bool:
        """True when a reference after the anchor owns another slot cue or noun for someone else: "我" in
        "你生日我生日", "你妈妈" in "你生日你妈妈生日", or 她 (who is not known) in "你生日她生日".
        "我忘了名字" owns nothing (忘了 is not an owner gap)."""
        for start, end, later in self._refs(persona_id, q):
            if start < anchor:
                continue
            rest = q[end:]
            pos = self._anchor(persona_id, rest)
            if pos is None:
                continue
            if self._relation_follows(q, end):
                return True
            if later == "person" and self._named(persona_id, q) and not self._possessed(persona_id, q):
                later = "name"                            # 乔沫的爱好是什么？她住哪？: 她 is the named persona
            same = later == "thing" or _OWNER_OF.get(later) == owner     # 它 points back at the noun
            if not same and _OWNER_GAP.fullmatch(rest[:pos]) is not None:
                return True
        return False

    def _legacy_subject(self, query: MemoryQuery, q: str, nouns: List[str], refs, anchor: Optional[int]
                        ) -> Tuple[Optional[str], Optional[str]]:
        """The order used before Day 7, for a text with no owner reference before its slot cue or noun ("住哪",
        "生日你是几月", keyword queries): 我, then 她/他/它, a subject noun, the persona's name or 你. A plural with a
        slot cue or noun is ambiguous, and a kinship word in an owner's clause makes the fact someone else's."""
        if anchor is not None and any(kind == "plural" for _, _, kind in refs):
            raise _Ambiguous("plural")                    # 我们都住哪: no single subject answers it
        for _, end, kind in refs:
            if kind in _OWNER_OF and self._relation_in_clause(q, end):
                raise _Unowned(self._relation_after(q, end))   # 你表哥吃香菜吗: a relation in the owner's clause
        if _pronoun(q, "我") == "singular":
            return USER, "query"
        person, thing = _pronoun(q, PERSON_PRONOUNS), _pronoun(q, THING_PRONOUN)
        if len(nouns) > 1 or "plural" in (person, thing):
            raise _Ambiguous("plural" if "plural" in (person, thing) else "two_nouns")
        if person or thing:
            return self._third_person(query, q, bool(person), bool(thing), nouns)
        if nouns:
            return nouns[0], "query"
        if self._named(query.persona_id, q) or _pronoun(q, "你") == "singular":
            return PERSONA, "query"
        return None, None

    def _third_person(self, query: MemoryQuery, q: str, person: bool, thing: bool, nouns: List[str]
                      ) -> Tuple[str, str]:
        source = "query"
        if person and self._named(query.persona_id, q):
            referent = PERSONA
        elif thing and not person and nouns:
            referent = nouns[0]
        else:
            referent, source = self._from_context(query, person=person), "context"
        if referent is None:
            raise _Ambiguous("pronoun")
        if person and nouns:                  # 她的猫: once 她 is known, the noun is the subject
            return nouns[0], source
        return referent, source

    def _from_context(self, query: MemoryQuery, *, person: bool) -> Optional[str]:
        """The referent in the newest context turn that names a compatible one, owned as that turn's speaker means
        it: in the user's turn 我 is the user and 你 the persona, in the persona's turn the other way round; turns
        without a role are read as the user's. None when that turn names several, or does not say whose it is."""
        roles = query.context_roles or ("user",) * len(query.context)
        for turn, role in reversed(tuple(zip(query.context, roles))):
            t = normalize(turn)
            if person:
                if not self._named(query.persona_id, t):
                    continue
                return None if self._possessed(query.persona_id, t) else PERSONA
            spans = self._noun_spans(t)
            nouns = self._nouns_in(t)
            if not nouns:
                continue
            if len(nouns) > 1:
                return None
            return self._context_owner(query.persona_id, t, spans[0][0], role, nouns[0])
        return None

    def _context_owner(self, persona_id: str, text: str, pos: int, role: str, noun: str) -> Optional[str]:
        """Whose the noun at ``pos`` is in a context turn spoken by ``role``: the noun's subject when the persona's,
        the user when the user's; None when not said; raises _Unowned for someone else's."""
        before = [r for r in self._refs(persona_id, text) if r[1] <= pos]
        if not before:
            return None                                   # 猫叫汤圆: whose cat is not said
        start, end, kind = before[-1]
        if kind in ("plural", "thing") or _quoted(text, start):
            return None
        if _OWNER_GAP.fullmatch(text[end:pos]) is None:
            raise _Unowned(self._relation_after(text, end))   # 我妈家的猫
        if kind == "person":
            return noun if self._named(persona_id, text) and not self._possessed(persona_id, text) else None
        speaker = PERSONA if role == "persona" else USER
        owner = {"I": speaker, "you": USER if speaker == PERSONA else PERSONA, "name": PERSONA}[kind]
        return noun if owner == PERSONA else USER

    def _turn_owner(self, persona_id: str, text: str, pos: Optional[int], role: str) -> Optional[str]:
        """Whose fact at ``pos`` a focus turn spoken by ``role`` states or asks: PERSONA or USER, read with the
        speaker (the user's 我 is the user, the persona's 我 the persona); None when not said, a third person, a
        plural, a quotation or someone else's (我妈家的猫)."""
        if pos is None:
            return None
        before = [r for r in self._refs(persona_id, text) if r[1] <= pos]
        if not before:
            return None
        start, end, kind = before[-1]
        if kind in ("plural", "thing", "person") or _quoted(text, start):
            return None
        if _OWNER_GAP.fullmatch(text[end:pos]) is None:
            return None
        speaker = PERSONA if role == "persona" else USER
        return {"I": speaker, "you": USER if speaker == PERSONA else PERSONA, "name": PERSONA}[kind]

    def _resolve_owner(self, query: MemoryQuery, q: str) -> _Owner:
        try:
            subject, source, by_ref = self.owner_of(query, q)
        except _Ambiguous as exc:
            return _Owner(SubjectStatus.AMBIGUOUS, ambiguity=f"ambiguous:{exc.reason}")
        except _Unowned as exc:
            return _Owner(SubjectStatus.UNOWNED, ambiguity=f"unowned:{exc.word}")
        if subject is None:
            return _Owner(SubjectStatus.OPEN)
        owner = USER if subject == USER else PERSONA
        return _Owner(SubjectStatus.RESOLVED, subject, source, by_ref, owner=owner)

    def _joined(self, query: MemoryQuery, q: str) -> Optional[List[_Owner]]:
        """Two owner references joined by a conjunction right before the asked fact (你和我分别住哪, 我跟你的生日,
        你和我谁生日更早): one owner each, or None when the text is not like that."""
        if query.subject is not None:
            return None
        pid = query.persona_id
        spans = self._noun_spans(q)
        anchor = self._anchor(pid, q, spans)
        if anchor is None:
            return None
        before = [r for r in self._refs(pid, q) if r[1] <= anchor]
        if len(before) < 2 or not _CONJUNCTION.fullmatch(q[before[-2][1]:before[-1][0]]):
            return None
        if len(before) > 2 and _CONJUNCTION.fullmatch(q[before[-3][1]:before[-2][0]]):
            return [_Owner(SubjectStatus.AMBIGUOUS, ambiguity="ambiguous:three_owners")]
        nouns = self._nouns_in(q)
        if len(nouns) > 1:
            return [_Owner(SubjectStatus.AMBIGUOUS, ambiguity="ambiguous:two_nouns")]
        gap = re.sub("^(?:之间)?谁(?:的)?", "", q[before[-1][1]:anchor])
        return [self._side(query, q, before[-2], nouns, None), self._side(query, q, before[-1], nouns, gap)]

    def _versus(self, query: MemoryQuery, q: str) -> Optional[Tuple[List[_Owner], Optional[str]]]:
        """Two owners compared with 比 (你生日比我早还是比我晚, 我的生日比你晚吗): one owner before 比, the other
        right after it, and the first relation word after that; None when the text is not like that."""
        if query.subject is not None:
            return None
        m = re.search("比(?!较)", q)
        if m is None:
            return None
        refs = self._refs(query.persona_id, q)
        before = [r for r in refs if r[1] <= m.start()]
        after = [r for r in refs if r[0] == m.end()]
        if len(before) != 1 or not after or {before[0][2], after[0][2]} != {"I", "you"}:
            return None
        rel = _VERSUS_RELATION.search(q, after[0][1])
        nouns = self._nouns_in(q)
        sides = [self._side(query, q, before[0], nouns, None), self._side(query, q, after[0], nouns, None)]
        return sides, None if rel is None else _RELATION_OF[rel.group(0)]

    def _side(self, query: MemoryQuery, q: str, ref: Tuple[int, int, str], nouns: List[str], gap: Optional[str]
              ) -> _Owner:
        start, end, kind = ref
        if kind == "plural" or _quoted(q, start):
            return _Owner(SubjectStatus.AMBIGUOUS, ambiguity="ambiguous:" + ("plural" if kind == "plural" else "quoted"))
        if self._relation_follows(q, end) or (gap is not None and _OWNER_GAP.fullmatch(gap) is None):
            return _Owner(SubjectStatus.UNOWNED, ambiguity=f"unowned:{self._relation_after(q, end)}")
        if kind == "I":
            return _Owner(SubjectStatus.RESOLVED, USER, "query", True, owner=USER)
        if kind in ("you", "name"):
            return _Owner(SubjectStatus.RESOLVED, nouns[0] if nouns else PERSONA, "query", True, owner=PERSONA)
        try:
            subject, source = self._third_person(query, q, kind == "person", kind == "thing", nouns)
        except _Ambiguous as exc:
            return _Owner(SubjectStatus.AMBIGUOUS, ambiguity=f"ambiguous:{exc.reason}")
        return _Owner(SubjectStatus.RESOLVED, subject, source, False, owner=USER if subject == USER else PERSONA)

    # ------------------------------------------------------------------ ellipsis and dialogue focus

    def _ellipsis(self, persona_id: str, part: str, timed: bool = False) -> bool:
        """True for a part that is only an owner, a relation and/or a time word with 呢 / 也是吗. ``timed``: a
        calendar time was taken out of ``part`` (``去年呢`` arrives as ``呢``)."""
        owners = _alternatives(["你", "我", "她", "他", "它", *PLURAL_REFS, *self._names.get(persona_id, ())])
        relation = f"(?:[{_KIN_PREFIX}])?(?:{_alternatives(RELATION_WORDS)})+"
        pattern = (f"(?:那么|那)?(?P<owner>{owners})?(?:的|家)?(?P<rel>{relation})?(?P<time>{_alternatives(TIME_WORDS)})?"
                   f"(?:{_alternatives(_ELLIPSIS_END)})")
        m = re.fullmatch(pattern, "".join(_PART.findall(part)))     # "那你呢？" is still "那你呢"
        return m is not None and (timed or any(m.group(g) for g in ("owner", "rel", "time")))

    def _dialogue_focus(self, query: MemoryQuery) -> Optional[DialogueFocus]:
        """The attribute the last turns were about, or why there is none. None without context."""
        if not query.context:
            return None
        if not query.context_roles:
            return DialogueFocus(status=FocusStatus.INVALID, reason="no_roles")
        turns = list(zip(query.context, query.context_roles))
        newest = len(turns) - 1
        window = [newest]
        if turns[newest][1] == "persona" and newest >= 1 and turns[newest - 1][1] == "user":
            window.append(newest - 1)                     # the user's turn and the persona's reply to it
        resolver = self.resolver(query.persona_id)
        found = []
        for i in window:
            t, role = normalize(turns[i][0]), turns[i][1]
            offset = len(turns) - i
            intent = resolver.resolve(t)
            asks = "?" in t or _asks(t, QUESTION_WORDS)
            if any(_quoted(t, m.start) for m in intent.matches):
                return DialogueFocus(status=FocusStatus.INVALID, reason="quoted")
            if intent.status in (IntentStatus.UNKNOWN, IntentStatus.UNSUPPORTED):
                return DialogueFocus(status=FocusStatus.INVALID, reason="unclear")
            if offset == 1 and role == "persona" and asks and not intent.matches:
                return DialogueFocus(status=FocusStatus.INVALID, reason="switched")   # the persona asked about else
            if intent.matches:
                found.append((offset, role, t, intent, SpeechAct.QUESTION if asks else SpeechAct.STATEMENT))
        attrs = list(dict.fromkeys(a for *_, intent, _ in found for a in intent.attributes))
        if not attrs:
            return DialogueFocus(status=FocusStatus.INVALID, reason="no_topic")
        if len(attrs) > 1:
            return DialogueFocus(status=FocusStatus.INVALID, reason="multiple")
        offset, role, t, intent, act = found[0]           # the newest turn that names it
        spans = self._noun_spans(t)
        anchor = spans[0][0] if spans else intent.anchor
        owner = self._turn_owner(query.persona_id, t, anchor, role)
        nouns = self._nouns_in(t)
        return DialogueFocus(status=FocusStatus.ACTIVE, reason="carried", attribute=attrs[0],
                             noun_subject=nouns[0] if len(nouns) == 1 else None, turn_offset=offset, speaker=role,
                             act=act, owner=owner if owner in (PERSONA, USER) else None)

    # ------------------------------------------------------------------ drafts

    def _drafts(self, query: MemoryQuery, text: str, parts: List[QueryPart], focus: Optional[DialogueFocus]
                ) -> List[_Draft]:
        asked = [p for p in parts if p.act is not SpeechAct.STATEMENT]
        whole = query.subject is not None or len(parts) <= 1 or not asked
        if whole:
            act = parts[0].act if len(parts) == 1 else (SpeechAct.QUESTION if asked else SpeechAct.STATEMENT)
            units = [(list(range(len(parts))), text, act)]
        else:
            units = [([p.index], p.text, p.act) for p in asked]
        drafts: List[_Draft] = []
        prev: List[_Draft] = []                           # the drafts of the unit before
        for group, (indices, unit, act) in enumerate(units):
            made = self._unit(query, text, indices, unit, act, whole, prev, focus, group)
            drafts += made
            prev = made
        return drafts

    def _unit(self, query: MemoryQuery, text: str, indices: List[int], unit: str, act: SpeechAct, whole: bool,
              prev: List[_Draft], focus: Optional[DialogueFocus], group: int) -> List[_Draft]:
        pid = query.persona_id
        span = unit
        scope, scope_source, period, time, unit = self._time(query, unit)   # owners and intents: without dates
        back = time is None and bool(prev) and prev[0].time is not None and _points_back(span)   # 前年…？那时候…？
        if (scope_source == "default" or back) and not whole and prev and prev[0].scope_source == "query":
            scope, scope_source, period, time = prev[0].scope, "query", prev[0].period, prev[0].time   # 你以前住哪，做什么工作
        intent = self.resolver(pid).resolve(unit)
        nouns = self._nouns_in(unit)
        noun = nouns[0] if len(nouns) == 1 else None
        base = dict(parts=indices, span=span, act=act, intent=intent, noun=noun, scope=scope,
                    scope_source=scope_source, period=period, time=time, group=group)
        if intent.status is IntentStatus.KEYWORD and not nouns and self._ellipsis(pid, unit, timed=unit != span):
            return [self._elliptical(query, unit, base, prev, focus)]
        reason = ["intent:" + (intent.source or intent.status.value) + (":" + "+".join(intent.words)
                                                                          if intent.words else "")]
        joined = self._joined(query, unit)
        versus = self._versus(query, unit) if joined is None and intent.attributes else None
        if versus is not None:
            joined = versus[0]
        if joined is not None:
            compare = None
            if len(joined) == 2 and ("谁" in unit or versus is not None):
                if versus is not None:
                    relation = versus[1]
                else:
                    m = _COMPARE.search(unit)
                    relation = _RELATION_OF.get(m.group(1)) if m is not None else None
                supported = bool(intent.attributes) and relation is not None and all(
                    (a, relation) in COMPARABLE for a in intent.attributes)
                compare = (relation, supported)
            return [_Draft(owner=o, intent_source=intent.source, attributes=intent.attributes,
                           reason=reason + ["owner:joined"], intent_ambiguity=intent.ambiguity, compare=compare,
                           **base) for o in joined]
        owner = self._resolve_owner(query, unit)
        undecided = owner.status is SubjectStatus.OPEN or (
            owner.status is SubjectStatus.RESOLVED and not owner.by_ref and owner.subject != USER
            and owner.source != "explicit")
        if undecided and not whole:
            if not self._refs(pid, unit) and prev:
                owner = self._inherit(prev, noun)         # 你喜欢做什么？在哪上班？
                reason.append("owner:previous_atom")
            else:
                owner = self._resolve_owner(query, self._time(query, text)[4])   # not decided in the part: the message
                reason.append("owner:whole_message")
        else:
            reason.append("owner:part")
        return [_Draft(owner=owner, intent_source=intent.source, attributes=intent.attributes, reason=reason,
                       intent_ambiguity=intent.ambiguity, **base)]

    @staticmethod
    def _inherit(prev: List[_Draft], noun: Optional[str]) -> _Owner:
        """The owner of the unit before (one owner only: after 你和我… it is not known which one is meant)."""
        if len(prev) != 1:
            return _Owner(SubjectStatus.AMBIGUOUS, ambiguity="ambiguous:joined_previous")
        owner = prev[0].owner
        if owner.status is not SubjectStatus.RESOLVED:
            return replace(owner, by_ref=False)
        subject = noun if owner.owner == PERSONA and noun else owner.owner
        return _Owner(SubjectStatus.RESOLVED, subject, owner.source, False, owner=owner.owner)

    def _elliptical(self, query: MemoryQuery, unit: str, base: Dict[str, Any], prev: List[_Draft],
                    focus: Optional[DialogueFocus]) -> _Draft:
        reason = ["intent:ellipsis"]
        noun: Optional[str] = None
        attributes: Tuple[str, ...] = ()
        intent_source: Optional[str] = None
        ambiguity: Optional[str] = None
        topics = {d.attributes for d in prev if d.attributes}
        if topics:
            if len(topics) == 1 and len(next(iter(topics))) == 1:
                attributes, intent_source, noun = next(iter(topics)), "previous", prev[0].noun
            else:
                ambiguity = "unknown:ellipsis_multiple"
            reason.append("topic:previous_atom")
        elif focus is not None and focus.status is FocusStatus.ACTIVE:
            attributes, intent_source, noun = (focus.attribute,), "focus", focus.noun_subject
            reason.append(f"topic:focus@{focus.speaker}-{focus.turn_offset}")
        else:
            ambiguity = f"unknown:ellipsis_{focus.reason if focus is not None else 'no_topic'}"
        if query.subject is not None:
            owner = _Owner(SubjectStatus.RESOLVED, query.subject, "explicit", False,
                           owner=USER if query.subject == USER else PERSONA)
            implied = noun
            if self._refs(query.persona_id, unit):          # the unit's own reference must agree with it
                own = self._resolve_owner(replace(query, subject=None), unit)
                implied = (noun if noun is not None and own.owner == PERSONA else own.subject
                           ) if own.status is SubjectStatus.RESOLVED else ""
            if implied is not None and implied != query.subject:
                owner = _Owner(SubjectStatus.AMBIGUOUS, ambiguity="ambiguous:focus_subject_conflict")
        elif self._refs(query.persona_id, unit):
            owner = self._resolve_owner(query, unit)          # 那你呢 / 我呢 / 你爸爸呢 / 那她呢
            reason.append("owner:ellipsis_pronoun")
        elif prev:
            owner = self._inherit(prev, None)
            reason.append("owner:previous_atom")
        elif focus is not None and focus.status is FocusStatus.ACTIVE and focus.owner is not None:
            owner = _Owner(SubjectStatus.RESOLVED, focus.owner, "context", False, owner=focus.owner)
            reason.append("owner:focus_turn")
        else:
            owner = _Owner(SubjectStatus.AMBIGUOUS, ambiguity="ambiguous:ellipsis_owner")
        if owner.status is SubjectStatus.OPEN:
            owner = _Owner(SubjectStatus.AMBIGUOUS, ambiguity="ambiguous:ellipsis_owner")
        if owner.status is SubjectStatus.RESOLVED and noun is not None and owner.source != "explicit":
            owner = replace(owner, subject=noun if owner.owner == PERSONA else USER)
        intent = SlotIntent(IntentStatus.SLOT) if attributes else SlotIntent(IntentStatus.UNKNOWN, ambiguity=ambiguity)
        base = dict(base, intent=intent, noun=noun)
        return _Draft(owner=owner, intent_source=intent_source, attributes=attributes, reason=reason,
                      intent_ambiguity=ambiguity, **base)

    # ------------------------------------------------------------------ merge and build

    @staticmethod
    def _merge(drafts: List[_Draft]) -> List[_Draft]:
        """Atoms asking the same owner the same attributes become one (``你住哪？以前呢？``: one atom, all time)."""
        out: List[_Draft] = []
        for d in drafts:
            key = (d.owner.status, d.owner.subject, d.attributes)
            same = next((o for o in out if d.attributes and not d.period and not o.period and d.compare is None
                         and o.compare is None and d.time is None and o.time is None
                         and (o.owner.status, o.owner.subject, o.attributes) == key
                         and d.owner.status in (SubjectStatus.RESOLVED, SubjectStatus.OPEN)), None)
            if same is None:
                out.append(d)
                continue
            same.parts = sorted(set(same.parts) | set(d.parts))
            same.span = f"{same.span} {d.span}"
            if same.scope is not d.scope:
                same.scope, same.scope_source = TimeScope.ALL, "query"
            same.reason = same.reason + [f"merged:{'+'.join(d.reason)}"]
        return out

    def _build(self, query: MemoryQuery, text: str, parts: List[QueryPart], focus: Optional[DialogueFocus],
               drafts: List[_Draft], planner: Optional[str] = None, notes: Tuple[str, ...] = ()) -> QueryPlan:
        atoms: List[AtomicQuery] = []
        for n, d in enumerate(drafts, start=1):
            atoms.append(self._atom(f"q{n}", d))
        deps: List[QueryDependency] = []
        groups: Dict[int, List[str]] = {}
        for a, d in zip(atoms, drafts):
            if d.compare is not None:
                groups.setdefault(d.group, []).append(a.query_id)
        for g, ids in groups.items():
            relation, supported = next(d.compare for d in drafts if d.group == g and d.compare is not None)
            kind = DependencyKind.DEPENDENT_COMPARE if relation is not None or supported else DependencyKind.UNKNOWN
            if len(ids) >= 2:
                deps.append(QueryDependency(kind=kind, query_ids=tuple(ids), relation=relation, supported=supported,
                                            reason="compare" if kind is DependencyKind.DEPENDENT_COMPARE else "who"))
        low = any(a.act is not SpeechAct.STATEMENT and a.intent in (IntentStatus.KEYWORD, IntentStatus.UNKNOWN)
                  for a in atoms)
        payload = {"persona_id": query.persona_id, "text": text, "context": list(query.context),
                   "context_roles": list(query.context_roles), "subject": query.subject,
                   "time_scope": query.time_scope.value if query.time_scope else None,
                   "atoms": [a.to_dict() for a in atoms], "dependencies": [d.to_dict() for d in deps],
                   "planner": planner or self.name}
        plan_id = "plan_" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]
        return QueryPlan(plan_id=plan_id, persona_id=query.persona_id, text=text, planner=planner or self.name,
                         parts=tuple(parts), atoms=tuple(atoms), dependencies=tuple(deps), focus=focus,
                         confidence="low" if low else "high", notes=notes)

    @staticmethod
    def _atom(query_id: str, d: _Draft) -> AtomicQuery:
        o = d.owner
        intent = d.intent.status
        attributes = d.attributes if intent is IntentStatus.SLOT else ()
        ambiguity = o.ambiguity if o.status in (SubjectStatus.AMBIGUOUS, SubjectStatus.UNOWNED) else None
        if intent in (IntentStatus.UNSUPPORTED, IntentStatus.UNKNOWN):
            ambiguity = ambiguity or d.intent_ambiguity or d.intent.ambiguity or f"{intent.value}:intent"
        retrievable = (o.status in (SubjectStatus.RESOLVED, SubjectStatus.OPEN)
                       and intent in (IntentStatus.SLOT, IntentStatus.KEYWORD) and not d.period
                       and d.scope is not TimeScope.UNRESOLVED)
        if d.period and not ambiguity:
            ambiguity = "life_period"
        if d.scope is TimeScope.UNRESOLVED and not ambiguity:
            ambiguity = "time_unresolved"
        resolved = o.status is SubjectStatus.RESOLVED
        return AtomicQuery(query_id=query_id, parts=tuple(d.parts), span=d.span, speaker="user", act=d.act,
                           subject=o.subject if resolved else None, subject_status=o.status,
                           subject_source=o.source if resolved else None, attributes=attributes, intent=intent,
                           intent_source=d.intent_source if intent is IntentStatus.SLOT else None,
                           time_scope=d.scope, time_scope_source=d.scope_source, period_unplaced=d.period,
                           retrievable=retrievable, matched=d.intent.words if intent is IntentStatus.SLOT else (),
                           reason="; ".join(d.reason), ambiguity=ambiguity, time=d.time)

    # ------------------------------------------------------------------ optional semantic parser

    def _consult(self, query: MemoryQuery, plan: QueryPlan) -> QueryPlan:
        """Ask the configured parser about asked parts without an attribute; keep only what ``validate_candidate``
        accepts. Any failure keeps the local plan and records why."""
        asked = [a for a in plan.atoms if a.act is not SpeechAct.STATEMENT and a.intent is IntentStatus.KEYWORD]
        if not asked:
            return plan
        spans = [a.span for a in asked]
        vocabulary = self.resolver(query.persona_id).attributes
        try:
            candidate = self._parser.propose(spans, vocabulary)
            accepted = validate_candidate(candidate, len(spans), vocabulary)
        except CandidateRejected as exc:
            return replace(plan, notes=plan.notes + (f"parser:rejected:{exc}",))
        except Exception as exc:          # the parser is optional: any failure falls back to the local plan
            return replace(plan, notes=plan.notes + (f"parser:{type(exc).__name__}",))
        if not accepted:
            return replace(plan, notes=plan.notes + ("parser:no_change",))
        atoms = list(plan.atoms)
        for index, attribute, owner_hint in accepted:
            a = asked[index]
            change: Dict[str, Any] = dict(intent=IntentStatus.SLOT, attributes=(attribute,), intent_source="parser",
                                          matched=("parser",), reason=a.reason + f"; parser:{attribute}")
            if owner_hint in ("other", "unclear") and a.subject_status in (SubjectStatus.RESOLVED, SubjectStatus.OPEN):
                status = SubjectStatus.UNOWNED if owner_hint == "other" else SubjectStatus.AMBIGUOUS
                change.update(subject=None, subject_source=None, subject_status=status,
                              ambiguity=f"{status.value}:parser_hint")
            elif owner_hint in (PERSONA, USER) and a.subject_status is SubjectStatus.RESOLVED and (
                    (USER if a.subject == USER else PERSONA) != owner_hint):
                change.update(subject=None, subject_source=None, subject_status=SubjectStatus.AMBIGUOUS,
                              ambiguity="ambiguous:parser_disagrees")
            status = change.get("subject_status", a.subject_status)
            change["retrievable"] = (status in (SubjectStatus.RESOLVED, SubjectStatus.OPEN) and not a.period_unplaced
                                     and a.time_scope is not TimeScope.UNRESOLVED)
            atoms[atoms.index(a)] = replace(a, **change)
        model = getattr(self._parser, "model", "parser")
        planner = f"{self.name}+parser:{model}"
        low = any(a.act is not SpeechAct.STATEMENT and a.intent in (IntentStatus.KEYWORD, IntentStatus.UNKNOWN)
                  for a in atoms)
        payload = {"base": plan.plan_id, "atoms": [a.to_dict() for a in atoms], "planner": planner}
        plan_id = "plan_" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]
        return replace(plan, plan_id=plan_id, planner=planner, atoms=tuple(atoms), confidence="low" if low else "high",
                       notes=plan.notes + (f"parser:accepted:{len(accepted)}",))



class CandidateRejected(ValueError):
    """A parser's candidate plan failed validation; the local plan is kept."""


_CANDIDATE_KEYS = {"atoms"}
_ATOM_KEYS = {"part", "attribute", "owner"}
_OWNER_HINTS = (PERSONA, USER, "other", "unclear")


def validate_candidate(candidate: Any, n_parts: int, vocabulary: Sequence[str]
                       ) -> List[Tuple[int, str, str]]:
    """Strict check of a parser's candidate: ``{"atoms": [{"part": i, "attribute": a|null, "owner": o}]}`` and
    nothing else. ``part`` indexes the parts that were sent, ``attribute`` must be in the vocabulary (null: none),
    ``owner`` one of persona/user/other/unclear. Memory ids, times, subjects or any other key reject the whole
    candidate. Returns (part, attribute, owner) for the entries that name an attribute."""
    if not isinstance(candidate, dict) or set(candidate) != _CANDIDATE_KEYS or not isinstance(candidate["atoms"], list):
        raise CandidateRejected("schema")
    if len(candidate["atoms"]) > max(1, 2 * n_parts):
        raise CandidateRejected("too_many_atoms")
    out: List[Tuple[int, str, str]] = []
    seen = set()
    for entry in candidate["atoms"]:
        if not isinstance(entry, dict) or set(entry) != _ATOM_KEYS:
            raise CandidateRejected("atom_schema")
        part, attribute, owner = entry["part"], entry["attribute"], entry["owner"]
        if isinstance(part, bool) or not isinstance(part, int) or not 0 <= part < n_parts:
            raise CandidateRejected("part")
        if owner not in _OWNER_HINTS:
            raise CandidateRejected("owner")
        if attribute is None:
            continue
        if not isinstance(attribute, str) or attribute not in vocabulary:
            raise CandidateRejected("attribute")
        if part in seen:
            raise CandidateRejected("duplicate_part")      # one attribute per part: anything else is undecided
        seen.add(part)
        out.append((part, attribute, owner))
    return out
