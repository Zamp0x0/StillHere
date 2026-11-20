"""Persona recall helpers for the Fact Reader route: whose fact a question asks about, and date-scoped conversation recall.

In persona chat the ASKER is the other person (the conversation counterpart): "你" is the persona, "我" the asker. The Fact Reader's
own frame is the owner's ("本人"). Two failure modes of the plain route were found on the local GPU (FINAL_LOCAL_GPU_SMOKE):

1. **Counterpart questions** ("我以前在哪读书来着？"): the evidence line was retrieved ("对方：我在澳洲读书呢"), but the reader's system
   prompt says '我' is the owner while the counterpart's own line also says '我' - the reader abstained. Fix: a deterministic subject
   resolution, and a perspective note in the pack's system-provided 【检索说明】 (the FACT prompt itself is unchanged). The answer must
   still cite a line of the right speaker (``attribution_ok``): recall is not bought by dropping attribution.
2. **Date-scoped questions** ("2025年6月1日我们聊了什么？"): top-k retrieval returns a handful of mixed-date items, and the reader is
   (rightly) not a summariser. Fix: parse the date (the existing ``temporal_anchor``; plus whole months), read EVERY message of that
   local-time window from the evidence archive in order, summarise it in segments with cited message ids, drop any point whose
   citations / numbers / wording are not in the cited lines, and only then let the persona say it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

TZ = timezone(timedelta(hours=8))
PERSONA_SELF, COUNTERPART, THIRD_PARTY, JOINT, UNKNOWN_SUBJECT = "PERSONA_SELF", "CONVERSATION_COUNTERPART", "THIRD_PARTY", "JOINT", "UNKNOWN_SUBJECT"
KIN = ("妹妹", "姐姐", "哥哥", "弟弟", "爸爸", "妈妈", "父亲", "母亲", "爷爷", "奶奶", "外公", "外婆", "姥姥", "姥爷", "老爸", "老妈", "爸", "妈",
       "叔叔", "阿姨", "舅舅", "姑姑", "表哥", "表姐", "表弟", "表妹", "堂哥", "堂姐", "堂弟", "堂妹", "儿子", "女儿", "老公", "老婆",
       "男朋友", "女朋友", "对象", "同事", "室友", "同学", "朋友", "闺蜜", "兄弟", "老板", "领导", "老师")
KIN_RE = "|".join(sorted(KIN, key=len, reverse=True))
DATE_TOPIC = re.compile(r"聊了|聊过|聊的|聊啥|聊什么|聊天|说了|说过|说的|说啥|说什么|谈了|谈过|讲了|讲过|发生了什么|干了什么|做了什么|干嘛了|发了什么")
MONTH = re.compile(r"(?:(\d{4})年)?(\d{1,2})月份?(?!\s*\d{1,2}\s*[日号])")
SUMMARY_VERSION = "stillhere.persona.date_summary/1.0"
SEGMENT = 80                 # messages per summarisation segment
MAX_POINTS = 8
SUMMARY_SYSTEM = """你是聊天记录整理助手。下面是“本人”和“对方”在某一时间段内的聊天记录（按时间先后，每条有编号，冒号前是说话人）。
请只根据这些记录，列出这段时间聊到的主要话题和关键事实：
1. 每条写成一句完整的话，写清是谁说的、是谁的事（本人 / 对方 / 双方 / 第三方的名字）。
2. 每条必须给出支持它的记录编号（如 D03），只写真正支持它的编号。
3. 不要加入记录里没有的内容，不要推测，不要评价；数字、时间、地点照抄记录。
4. 没有实际内容的寒暄（嗯、好、哈哈、表情、图片）不用列。
只输出一个 JSON 对象：{"points": [{"text": "...", "who": "本人|对方|双方|第三方", "evidence": ["D01"]}]}，最多 6 条；没有可列的内容就输出 {"points": []}。"""
_CJK = re.compile(r"[一-鿿]")


# ---------------------------------------------------------------- subject resolution
@dataclass
class Subject:
    kind: str
    owner: Optional[str] = None          # for THIRD_PARTY: whose relative / friend ("PERSONA_SELF" / "CONVERSATION_COUNTERPART"), if said
    name: Optional[str] = None           # the third party as named in the question
    basis: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "owner": self.owner, "name": self.name, "basis": self.basis}


def resolve_subject_v1(question: str) -> Subject:
    """Whose fact the counterpart's question asks about. Deterministic, pronoun based: '你的妹妹' -> THIRD_PARTY of the persona, '你' ->
    PERSONA_SELF, '我' -> CONVERSATION_COUNTERPART, '我们 / 咱们' -> JOINT, a kin word alone -> THIRD_PARTY, else UNKNOWN_SUBJECT."""
    q = question or ""
    m = re.search(rf"(你们|我们|咱们|你|我)?的?({KIN_RE})", q)
    if m and m.group(2):
        owner = {"你": PERSONA_SELF, "我": COUNTERPART}.get(m.group(1) or "")
        return Subject(THIRD_PARTY, owner, m.group(2), f"kin word '{m.group(2)}'" + (f" of '{m.group(1)}'" if m.group(1) else ""))
    if re.search(r"我们|咱们|咱俩|我俩|你们", q):
        return Subject(JOINT, basis="'我们' (both people)")
    has_you, has_me = "你" in q, "我" in q
    if has_you and not has_me:
        return Subject(PERSONA_SELF, basis="'你' = the persona")
    if has_me and not has_you:
        return Subject(COUNTERPART, basis="'我' = the asker (the conversation counterpart)")
    if has_you and has_me:                    # "你还记得我以前…": the fact is the asker's when '我' follows the memory verb
        k = re.search(r"你(还)?(记得|知道|晓得|记不记得)", q)
        if k and "我" in q[k.end():]:
            return Subject(COUNTERPART, basis="'你记得 我…' = the asker's own fact")
        return Subject(PERSONA_SELF, basis="'你' and '我' (persona first)")
    return Subject(UNKNOWN_SUBJECT, basis="no pronoun or kin word")


def _resolve_subject_p1(question: str) -> Subject:
    """Whose fact the counterpart's question asks about. Deterministic, pronoun based: '你的妹妹' -> THIRD_PARTY of the persona, '你' ->
    PERSONA_SELF, '我' -> CONVERSATION_COUNTERPART, '我们 / 咱们' -> JOINT, a kin word alone -> THIRD_PARTY, else UNKNOWN_SUBJECT."""
    q = question or ""
    m = re.search(rf"(你们|我们|咱们|你|我)?的?({KIN_RE})", q)
    if m and m.group(2):
        owner = {"你": PERSONA_SELF, "我": COUNTERPART}.get(m.group(1) or "")
        return Subject(THIRD_PARTY, owner, m.group(2), f"kin word '{m.group(2)}'" + (f" of '{m.group(1)}'" if m.group(1) else ""))
    if re.search(r"我们|咱们|咱俩|我俩|你们", q):
        return Subject(JOINT, basis="'我们' (both people)")
    has_you, has_me = "你" in q, "我" in q
    if has_you and not has_me:
        return Subject(PERSONA_SELF, basis="'你' = the persona")
    if has_me and not has_you:
        return Subject(COUNTERPART, basis="'我' = the asker (the conversation counterpart)")
    if has_you and has_me:                    # "你还记得我以前…": the fact is the asker's when '我' follows the memory verb
        k = re.search(r"你(还)?(记得|知道|晓得|记不记得)", q)
        if k and "我" in q[k.end():]:
            return Subject(COUNTERPART, basis="'你记得 我…' = the asker's own fact")
        # P1 (2026-11-19): who SAID / WANTED / DID it, read from the question form. "我那次说… 你…" asks for the asker's own words, so the
        # cited lines are the counterpart's; the old 'persona first' default rejected 4 correct answers of the V3 run (known failure).
        bare = DESCRIPTOR.sub("", q)
        asker, persona = bool(ASKER_ACT.search(bare)), bool(PERSONA_ACT.search(bare))
        # never switched to the OTHER speaker: an ambiguous mixed question becomes JOINT (both people's lines may be cited; the
        # perspective note still asks the reader to keep who said / did what apart). Single-pronoun questions are unchanged.
        if asker:
            return Subject(JOINT, basis="'我…说 / 想 / 怕 / 让你…' with '你': the asker's own words or intent are asked about")
        if not persona and bare != q:
            return Subject(JOINT, basis="'你给我…的 X' names a shared thing; no one's act is asked about")
        return Subject(PERSONA_SELF, basis="'你' and '我' (persona first)")
    return Subject(UNKNOWN_SUBJECT, basis="no pronoun or kin word")


READER_VERSIONS = ("v1", "p1")      # v1: the reader as evaluated in FAST / V2 / V3 (default); p1: the 2026-11-19 fixes (V4 decides)


def resolve_subject(question: str, version: str = "v1") -> Subject:
    """``version``: "v1" = the frozen rules (``resolve_subject_v1``); "p1" = + who said / wanted / did it in mixed 你 + 我 questions."""
    if version not in READER_VERSIONS:
        raise ValueError(f"unknown reader version {version!r}")
    return resolve_subject_v1(question) if version == "v1" else _resolve_subject_p1(question)


# the speaker's own act in the question: subject pronoun (not an object: "跟我说的") + (time word) + a verb of saying / wanting / intending /
# doing for the other, in one clause; memory verbs (记得 / 知道) are never acts. A relative clause naming a thing ("你给我做的吊坠") is a
# descriptor, not the asked act, and is removed first.
_TIME = r"(?:当时|那次|那天|之前|以前|上次|一直|一开始|后来|原本|本来|还|也|又|都|就|是)*"
_ACT = r"(?:跟你|跟我|对你|对我|和你|和我)?(?:说|讲|问|告诉|答应|回答|解释|让|叫|求|骂|怕|想|觉得|担心|希望|打算|准备|决定|送|给|请|发)"
_NOT_OBJECT = r"(?<![跟对和给让叫向帮替被把])"
ASKER_ACT = re.compile(_NOT_OBJECT + r"我" + _TIME + r"(?:怎么)?" + _ACT)
PERSONA_ACT = re.compile(_NOT_OBJECT + r"你" + _TIME + r"(?!记得|知道|晓得|记不记得)(?:怎么)?" + _ACT)
DESCRIPTOR = re.compile(r"[你我](?:给|送|帮|替)[你我][^，。？?！!]{0,6}的")


SPEAKER_NOTE = "证据中每条消息前面的“本人”“对方”是说话人；消息原文里的“我”指说这句话的人自己。"


# p1: a relative named by a speaker ("对方：我弟在喂猫") is THAT speaker's relative (V3 known failure: the reader answered "对方在喂")
SPEAKER_NOTE_P1 = SPEAKER_NOTE + "原文里的“我妈 / 我弟 / 我姐”等，是说这句话的人自己的亲属，不是说话人本人；回答时要写清是谁的亲属。"


def perspective_notes(subject: Subject, version: str = "v1") -> List[str]:
    """Notes for the pack's 【检索说明】 (system-provided, not answers): who is who in the evidence, and whose fact is asked about."""
    notes = [SPEAKER_NOTE if version == "v1" else SPEAKER_NOTE_P1]
    if subject.kind == COUNTERPART:
        notes.append("这个问题问的是“对方”（聊天对象）自己的事：请依据“对方：”开头、由对方自己说出的记录回答，回答里称“对方”；本人说的自己的事不能当作对方的。")
    elif subject.kind == PERSONA_SELF:
        notes.append("这个问题问的是“本人”自己的事：只依据本人自己说出的记录（“本人：”开头）回答；对方说的对方自己的经历不能算作本人的。")
    elif subject.kind == THIRD_PARTY:
        whose = {PERSONA_SELF: "本人的", COUNTERPART: "对方的"}.get(subject.owner or "", "")
        notes.append(f"这个问题问的是第三方（{whose}{subject.name or '某人'}）的事，不是本人或对方自己的事；请依据提到这个人的记录回答。")
    elif subject.kind == JOINT:
        notes.append("这个问题问的是本人和对方两个人之间的事，两人的记录都可以用，但要分清是谁说的、是谁的事。")
    return notes


def attribution_ok(subject: Subject, cited_speakers: Sequence[Sequence[str]]) -> bool:
    """The deterministic attribution gate after the reader: a PERSONA_SELF answer must cite at least one line the persona said, a
    COUNTERPART answer one the counterpart said. Other kinds (third party / joint / unknown) are not restricted by speaker."""
    need = {PERSONA_SELF: "本人", COUNTERPART: "对方"}.get(subject.kind)
    if need is None:
        return True
    return any(need in (s or ()) for s in cited_speakers)


# ---------------------------------------------------------------- date-scoped recall
@dataclass
class DateScope:
    start: datetime
    end: datetime                 # half-open, local time
    granularity: str              # "day" / "month"
    span: str

    def label(self) -> str:
        d = self.start             # no strftime: it cannot take non-ASCII formats on Windows before Python 3.12
        return f"{d.year:04d}年{d.month:02d}月{d.day:02d}日" if self.granularity == "day" else f"{d.year:04d}年{d.month:02d}月"

    def to_dict(self) -> dict:
        return {"start": self.start.isoformat(), "end": self.end.isoformat(), "granularity": self.granularity, "span": self.span, "timezone": "UTC+08:00"}


def date_scope(question: str, ask_time: datetime) -> Optional[DateScope]:
    """An explicit day (the existing EF1 ``temporal_anchor`` parser: '2025年6月1日', '6月1号', '昨天', ...) or a whole month ('2025年6月')."""
    from stillhere.retrieval.temporal_anchor import parse_anchor
    at = ask_time if ask_time.tzinfo else ask_time.replace(tzinfo=TZ)
    a = parse_anchor(question, at, TZ)
    if a.usable and a.granularity == "day" and a.start is not None:
        return DateScope(a.start, a.end, "day", " ".join(a.source_span))
    m = MONTH.search(question)
    if m:
        month = int(m.group(2))
        if 1 <= month <= 12:
            year = int(m.group(1)) if m.group(1) else (at.year if month <= at.month else at.year - 1)
            start = datetime(year, month, 1, tzinfo=TZ)
            end = datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=TZ)
            return DateScope(start, end, "month", m.group(0))
    return None


def is_date_question(question: str, ask_time: datetime) -> bool:
    return bool(DATE_TOPIC.search(question or "")) and date_scope(question, ask_time) is not None


def _utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def window_messages(db_path, persona_id: str, scope: DateScope, *, before: Optional[datetime] = None) -> List[dict]:
    """Every archived message of the persona's conversation inside the local-time window (and before ``before``), oldest first, with
    its message id, speaker label (本人 / 对方) and text. Read-only on the evidence archive (no second memory system)."""
    import sqlite3
    hi = min(scope.end, before) if before else scope.end
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT message_id, speaker_id, timestamp_utc, content, conversation_id FROM messages WHERE target_persona_id = ? "
                           "AND timestamp_utc >= ? AND timestamp_utc < ? ORDER BY timestamp_utc, message_id",
                           (persona_id, _utc(scope.start), _utc(hi))).fetchall()
    finally:
        con.close()
    out = []
    for mid, spk, ts, content, cid in rows:
        t = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(TZ)
        out.append({"message_id": mid, "speaker": "本人" if spk == persona_id else "对方", "time": t.strftime("%Y-%m-%d %H:%M"),
                    "text": content, "conversation_id": cid})
    return out


def _bigrams(s: str) -> set:
    t = "".join(_CJK.findall(s or ""))
    return {t[i:i + 2] for i in range(len(t) - 1)}


def _numbers(s: str) -> set:
    return set(re.findall(r"\d+(?:\.\d+)?", s or ""))


@dataclass
class DateSummary:
    scope: DateScope
    messages: int
    points: List[dict] = field(default_factory=list)
    dropped: List[dict] = field(default_factory=list)
    segments: int = 0
    status: str = ""                  # ANSWERED / NO_MESSAGES / NOTHING_SUBSTANTIVE / SUMMARY_UNGROUNDED / SUMMARY_FAILED

    @property
    def answerable(self) -> bool:
        return self.status in ("ANSWERED", "NOTHING_SUBSTANTIVE")

    def answer(self, label: bool = True) -> str:
        """The grounded answer; ``label=False`` leaves out the date prefix (the date is the question's, not a fact the reply must repeat)."""
        head = self.scope.label() if label else "那天"
        if self.status == "NO_MESSAGES":
            return f"{head}没有聊天记录。"
        if self.status == "NOTHING_SUBSTANTIVE":
            return f"{head}只有几句简短的回复、图片或表情，没有聊到具体的事。"
        if not self.answerable:
            return "根据现有记录，无法确定。"
        return ("聊到：" if not label else f"{head}聊到：") + "；".join(p["text"].rstrip("。；;.") for p in self.points) + "。"

    def to_dict(self) -> dict:
        return {"scope": self.scope.to_dict(), "messages": self.messages, "segments": self.segments, "status": self.status, "points": self.points,
                "dropped": self.dropped, "evidence_message_ids": sorted({m for p in self.points for m in p["message_ids"]}), "version": SUMMARY_VERSION}


TRIVIAL = re.compile(r"^(\[.*\]|[嗯哦噢啊哈呵好行对是的吗呢吧嘛啦呀哟诶欸~～!！?？。，,.\s]*|ok|okk|好的|好滴|收到|晚安|早|早安)$", re.I)


def summarize_window(scope: DateScope, msgs: Sequence[dict], complete: Callable[[list], dict]) -> DateSummary:
    """Segmented, grounded summary of one window. ``complete(messages) -> {"text"}`` is the local model (greedy). A point survives only
    if it cites ids of this window, its numbers occur in the cited lines, and it shares wording (2-char units) with them."""
    s = DateSummary(scope, len(msgs))
    if not msgs:
        s.status = "NO_MESSAGES"
        return s
    subst = [m for m in msgs if not TRIVIAL.match((m["text"] or "").strip())]
    if not subst:
        s.status = "NOTHING_SUBSTANTIVE"
        return s
    seen = set()
    for a in range(0, len(subst), SEGMENT):
        seg = subst[a:a + SEGMENT]
        ids = {f"D{i + 1:02d}": m for i, m in enumerate(seg)}
        body = "\n".join(f"[{k}] {m['time']} {m['speaker']}：{m['text']}" for k, m in ids.items())
        s.segments += 1
        try:
            raw = complete([{"role": "system", "content": SUMMARY_SYSTEM},
                            {"role": "user", "content": f"【时间段】{scope.label()}（UTC+8）\n【聊天记录】\n{body}"}])["text"]
            m = re.search(r"\{.*\}", raw or "", re.S)
            pts = json.loads(m.group(0)).get("points") if m else None
        except (ValueError, TypeError, AttributeError, KeyError):
            pts = None
        if not isinstance(pts, list):
            s.dropped.append({"segment": s.segments, "reason": "SUMMARY_FAILED (no valid JSON)"})
            continue
        for p in pts:
            if not isinstance(p, dict) or not isinstance(p.get("text"), str) or not p["text"].strip():
                continue
            ev = [e for e in (p.get("evidence") or []) if isinstance(e, str) and e in ids]
            cited = " ".join(ids[e]["text"] for e in ev)
            reason = None
            if not ev:
                reason = "no valid evidence id"
            elif _numbers(p["text"]) - _numbers(cited) - _numbers(scope.label()):
                reason = "number not in the cited lines"
            elif _bigrams(p["text"]) and not (_bigrams(p["text"]) & _bigrams(cited)):
                reason = "no wording shared with the cited lines"
            key = p["text"].strip()
            if reason:
                s.dropped.append({"segment": s.segments, "text": key[:80], "reason": reason})
            elif key not in seen:
                seen.add(key)
                s.points.append({"text": key, "who": p.get("who"), "message_ids": [ids[e]["message_id"] for e in ev]})
    s.points = s.points[:MAX_POINTS]
    if s.points:
        s.status = "ANSWERED"
    elif not s.dropped:
        s.status = "NOTHING_SUBSTANTIVE"          # the model found nothing beyond small talk in real messages: a grounded answer
    elif all("SUMMARY_FAILED" in d.get("reason", "") for d in s.dropped):
        s.status = "SUMMARY_FAILED"
    else:
        s.status = "SUMMARY_UNGROUNDED"
    return s
