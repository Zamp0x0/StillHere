"""The one versioned prompt schema of the Universal Style Adapter (T2-D2): training, validation, test fixtures and inference all call
``build_style_conditioned_messages``. Memory decides WHAT to say (Grounded Content); the adapter decides HOW (Style Profile + Examples).

Output: chat messages ``[system, user]`` for the official Qwen3 chat template (``enable_thinking=False``); the target reply is never part of
them, it is only the assistant turn in training (``to_sharegpt``).

Sections of the system message, always present and always in this order, so a missing input has an explicit meaning instead of a gap:

* 【说话方式】 the Style Profile (traits from the profiler) or the missing-profile line;
* 【说话方式示例】 Style Examples (the person's own earlier messages; style only, never facts of this turn) or the missing line;
* 【记忆事实】 Grounded Content: one line per fact with its status (current / past / corrected / conflicting / at the time asked), time,
  source and confidence; unknown items; or the no-memory line. Built by hand or from the Core retriever's ``MemoryContextView``
  (``grounded_from_memory_view``), with the same statuses the Core memory block uses;
* 【生成要求】 generation constraints: the fixed rules plus optional extra rules.

The user message is the conversation so far (``对方：…`` / ``我：…``), optionally after a relation line.

Every free-text field is checked with the rules of ``stillhere.data.templates`` (kept local, the style layer depends on contracts only):
special tokens and reserved tags are refused, so text inside a fact or an example can never act as a chat-template marker.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Mapping, Optional, Sequence

from stillhere.style.profiler import render as render_profile


class PromptError(ValueError):
    """A field cannot be rendered safely into the style prompt."""


TemplateError = PromptError  # the name the T2 scripts use
# Same refusal rules as stillhere.data.templates (kept local: the style layer may depend on contracts only)
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|<>\s]{1,64}\|>")
_RESERVED_TAGS = ("<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>")

PROMPT_SCHEMA_VERSION = "stillhere.style.prompt/2.0"

ROLE_LABEL = {
    "current": "当前",
    "past": "过去，现在已不成立",
    "corrected": "已更正",
    "unverified": "冲突，未确认",
    "at_time": "所问时间内成立",
}
CONFIDENCE_LABEL = {"high": "高", "medium": "中", "low": "低"}

INTRO = ("你要以一位特定说话者的身份回复对方。“说话方式”和“示例”只决定怎么说；"
         "说什么只能来自“记忆事实”和对话本身。")
NO_PROFILE = "- （没有可用的说话方式描述：用自然、简洁的口吻）"
NO_EXAMPLES = "（没有可用的示例）"
EXAMPLES_NOTE = "（此人过去在其他场合说的话，只示范说话方式；其中提到的经历和事实不属于本次对话，不能当作事实使用）"
NO_MEMORY = ("（没有提供记忆事实：可以对对话内容表达态度、看法或追问，但不要编造具体的个人经历、数字、时间、价格或产品型号）")
UNKNOWN_LINE = "- [未知] {item}：没有记录，不要猜；需要时自然地说不清楚或问一下"
BASE_CONSTRAINTS = (
    "记忆事实里标为“当前”的内容要准确表达，不要改数字、时间和否定关系；",
    "“已更正”的事实只用更正后的说法；“冲突，未确认”的不要替对方选一个，要说明不确定；“过去”的不要说成现在；",
    "不要照抄示例里的句子，也不要把示例里的经历当成这次的事实；",
    "不要新增记忆事实之外的具体个人经历、身份、数字或时间；",
    "顺着对方这次说的话回应，回复可以分成几条消息，每条一行。",
)


def _check(text: str, where: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise TemplateError(f"{where} must be a non-empty string")
    if _SPECIAL_TOKEN_RE.search(text) or any(tag in text for tag in _RESERVED_TAGS):
        raise TemplateError(f"{where} contains a special-token or reserved marker")
    return text.strip()


def no_memory() -> dict:
    """Grounded Content when Memory returned nothing (or was not asked)."""
    return {"status": "no_memory", "facts": [], "unknown": []}


def grounded(facts: Sequence[Mapping] = (), unknown: Sequence[str] = ()) -> dict:
    """Grounded Content from explicit facts. A fact: ``statement`` (required), ``role`` (current | past | corrected | unverified | at_time;
    default current), optional ``time``, ``source``, ``confidence`` (high | medium | low), and for ``corrected`` the wrong earlier
    statement in ``replaces``."""
    out = []
    for i, f in enumerate(facts):
        role = f.get("role", "current")
        if role not in ROLE_LABEL:
            raise TemplateError(f"fact[{i}] role {role!r} is not one of {sorted(ROLE_LABEL)}")
        if f.get("confidence") not in (None, *CONFIDENCE_LABEL):
            raise TemplateError(f"fact[{i}] confidence {f.get('confidence')!r}")
        out.append({k: v for k, v in {"statement": _check(f["statement"], f"fact[{i}].statement"), "role": role, "time": f.get("time"), "source": f.get("source"),
                                      "confidence": f.get("confidence"), "replaces": f.get("replaces")}.items() if v is not None})
    unk = [_check(u, f"unknown[{i}]") for i, u in enumerate(unknown)]
    return {"status": "facts" if out or unk else "no_memory", "facts": out, "unknown": unk}


def _fact_of(rec, role: str, replaces: Optional[str] = None) -> dict:
    conf = getattr(rec, "confidence", 1.0)
    t = None
    if rec.valid_from or rec.valid_to:
        t = f"{rec.valid_from.date().isoformat() if rec.valid_from else '?'} ~ {rec.valid_to.date().isoformat() if rec.valid_to else '现在'}"
    return {"statement": rec.statement, "role": role, "time": t, "source": rec.provenance[0].source_id if rec.provenance else None,
            "confidence": "high" if conf >= 0.8 else "medium" if conf >= 0.5 else "low", "replaces": replaces}


def grounded_from_memory_view(view, records) -> dict:
    """Grounded Content from the Core retriever's ``MemoryContextView`` and the request's ``memory_context`` records (the Core contract:
    a view's facts name records by ``memory_id``; only a CORRECTED fact carries its own, superseded, record).

    * CURRENT / PAST / UNVERIFIED (conflict) / AT_TIME facts keep the Core role and the record's own statement, valid time, first provenance
      source and confidence;
    * a CORRECTED fact is the mis-recorded entry: its replacement (``corrected_by``) is shown as ``corrected`` with the wrong statement in
      ``replaces``; the wrong entry is never shown as a fact;
    * ``no_match`` / an empty view -> no memory; ``ambiguous_subject`` -> one unknown item; memory_block_v2 queries with no record (or no
      record for the time asked) -> unknown items. Nothing is invented."""
    by_id = {r.memory_id: r for r in (records.values() if hasattr(records, "values") else records)}
    status = getattr(view.status, "value", view.status)
    if status == "ambiguous_subject":
        return grounded([], ["对方说的是谁或哪件事"])
    facts, unknown, seen = [], [], set()
    for slot in getattr(view, "slots", ()):
        replaced = {cf.corrected_by: cf.record.statement for cf in slot.facts if getattr(cf.role, "value", cf.role) == "corrected" and cf.record is not None}
        for cf in slot.facts:
            role = getattr(cf.role, "value", cf.role)
            if role == "corrected" or cf.memory_id in seen:
                continue
            rec = by_id.get(cf.memory_id)
            if rec is None:
                raise TemplateError(f"memory view names {cf.memory_id!r}, which is not in the request's records")
            seen.add(cf.memory_id)
            facts.append(_fact_of(rec, "corrected" if cf.memory_id in replaced else role, replaced.get(cf.memory_id)))
    for q in getattr(view, "queries", ()):
        suff = getattr(q.sufficiency, "value", q.sufficiency)
        if suff in ("no_record", "time_unknown"):
            unknown.append(f"{q.subject or '对方问到的'}的{'、'.join(q.attributes) or '这件事'}")
    return grounded(facts, unknown)


def render_grounded(g: Optional[Mapping]) -> str:
    if not g or g.get("status") == "no_memory" or not (g.get("facts") or g.get("unknown")):
        return NO_MEMORY
    lines = []
    for f in g.get("facts", []):
        meta = [x for x in (f"时间：{f['time']}" if f.get("time") else None, f"来源：{f['source']}" if f.get("source") else None,
                            f"可信度：{CONFIDENCE_LABEL[f['confidence']]}" if f.get("confidence") else None) if x]
        text = f["statement"] + (f"（原先记成“{f['replaces']}”，那是记错了）" if f.get("role") == "corrected" and f.get("replaces") else "")
        lines.append(f"- [{ROLE_LABEL[f['role']]}] {text}" + (f"（{'；'.join(meta)}）" if meta else ""))
    lines += [UNKNOWN_LINE.format(item=u) for u in g.get("unknown", [])]
    return "\n".join(lines)


def render_examples(examples: Optional[Sequence[str]]) -> str:
    ex = [_check(e.replace("\n", " / "), f"example[{i}]") for i, e in enumerate(examples or [])]
    return EXAMPLES_NOTE + "\n" + "\n".join(f"例{i}：{e}" for i, e in enumerate(ex, 1)) if ex else NO_EXAMPLES


def render_conversation(conversation: Sequence[Mapping], relation: Optional[str] = None) -> str:
    if not conversation:
        raise TemplateError("the conversation needs at least one message")
    lines = [f"（关系：{_check(relation, 'relation')}）"] if relation else []
    for i, m in enumerate(conversation):
        who = {"other": "对方", "self": "我"}.get(m.get("speaker"))
        if who is None:
            raise TemplateError(f"conversation[{i}] speaker must be 'other' or 'self'")
        lines.append(f"{who}：{_check(m['text'], f'conversation[{i}]')}")
    return "\n".join(lines)


def build_style_conditioned_messages(conversation: Sequence[Mapping], *, grounded_content: Optional[Mapping] = None,
                                     style_profile: Optional[Mapping] = None, style_examples: Optional[Sequence[str]] = None,
                                     constraints: Iterable[str] = (), relation: Optional[str] = None) -> List[dict]:
    """The chat for one reply. ``grounded_content``: from ``grounded`` / ``no_memory`` / ``grounded_from_memory_view`` (None = no memory).
    ``style_profile``: a profiler profile (None = missing). ``style_examples``: the person's earlier messages (None or empty = missing).
    ``constraints``: extra generation rules appended to the fixed ones."""
    profile = render_profile(style_profile) if style_profile and style_profile.get("traits") else NO_PROFILE
    extra = [_check(c, "constraint") for c in constraints]
    system = "\n".join([INTRO, "【说话方式】", profile, "【说话方式示例】", render_examples(style_examples), "【记忆事实】", render_grounded(grounded_content),
                        "【生成要求】", *[f"{i}. {c}" for i, c in enumerate([*BASE_CONSTRAINTS, *extra], 1)]])
    return [{"role": "system", "content": system}, {"role": "user", "content": render_conversation(conversation, relation)}]


def to_sharegpt(messages: Sequence[Mapping], target: str, **meta) -> dict:
    """LLaMA-Factory ShareGPT row: the system and user text of ``messages`` and the supervised reply (the only assistant turn)."""
    if [m["role"] for m in messages] != ["system", "user"]:
        raise TemplateError("expected [system, user] messages from build_style_conditioned_messages")
    reply = _check(target, "target")
    return {**meta, "schema": PROMPT_SCHEMA_VERSION, "system": messages[0]["content"],
            "conversations": [{"from": "human", "value": messages[1]["content"]}, {"from": "gpt", "value": reply}]}
