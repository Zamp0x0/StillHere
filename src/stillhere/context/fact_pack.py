"""EF1 fact evidence pack (``stillhere.context.fact_pack/1.0``): the evidence a fact reader receives, with every source kept.

    build(api, candidate_set, counter=..., ...) -> FactPack

* items: the top-K evidence units of the EF1 ranking (K from the retrieval config: 40 by default), each with its ORIGINAL text (every turn of
  a dialogue), its source type, event date, recorded / known times, the persons speaking, the retrieval channels and ranks, and the
  structured memory records the channels linked to it (with their status at the ask time: 当前 / 过去 / 已更正 / 已撤回, validity);
* no fixed item cap: K units are all rendered unless the model context would overflow; then the lowest-ranked units outside the protected
  set (inside an event window, best per sub-question) are moved to ``dropped`` with the reason ``context_window`` - nothing is cut silently;
* deduplication only removes an exact copy (same text, same day); units of different days are never merged (repeated events are count
  evidence);
* order: chronological (event time), so ordering, offset, update and conflict questions read in time order; ``rank`` is kept per item;
* privacy: ``audience="owner"`` (the persona querying their own history - the memory-assistant setting) shows the owner's own records;
  any other audience must pass a ``redactor(text, at) -> None | redacted_text`` (``stillhere.reader.fact.third_party_redactor`` wraps
  ``session.disclosure.PolicyResolver``); without one a non-owner pack is refused.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence

from stillhere.memory import reflection as r2
from stillhere.memory.labels import SOURCE_LABELS, speaker_label

PACK_SCHEMA = "stillhere.context.fact_pack/1.0"
CONTEXT_WINDOW = 32768                  # Qwen3-8B native context (config max_position_embeddings 40960 = 32768 + output headroom)
OUTPUT_TOKENS = {"default": 512, "long": 1024}
LONG_FORMS = ("ENUMERATED_LIST", "ORDERING", "RANGE_SUMMARY")     # stillhere.retrieval.ef1.shape flags that need a long answer


def output_tokens_for(flags: Sequence[str]) -> int:
    flags = getattr(flags, "flags", flags)
    return OUTPUT_TOKENS["long"] if any(f in flags for f in LONG_FORMS) else OUTPUT_TOKENS["default"]


@dataclass
class PackMemory:
    memory_id: str
    statement: str
    status: str
    valid_from: Optional[str]
    valid_to: Optional[str]
    recorded_at: Optional[str]


@dataclass
class PackItem:
    eid: str
    unit_key: str
    rank: int                          # position in the EF1 ranking (1 = best)
    score: float
    source_type: str
    source_label: str
    source_id: str
    event_time: Optional[str]          # time of the record (the message time), local
    recorded_at: Optional[str]
    speakers: List[str]
    text: str                          # original text, every turn of the unit, "说话人：内容" per line
    message_ids: List[str]
    channels: Dict[str, int]
    memories: List[PackMemory] = field(default_factory=list)
    in_windows: List[str] = field(default_factory=list)
    redacted: bool = False
    kind: str = "record"               # record (raw source) | memory_only (a stored fact with no message provenance)


@dataclass
class FactPack:
    schema: str
    persona_id: str
    question: str
    body: str
    ask_time: str
    k: int
    items: List[PackItem]
    notes: List[str]
    subqueries: List[str]
    shape: List[str]
    dropped: List[dict] = field(default_factory=list)
    dedup: List[dict] = field(default_factory=list)
    tokens: Dict[str, int] = field(default_factory=dict)
    token_mode: str = "ESTIMATED"
    context_window: int = CONTEXT_WINDOW
    output_tokens: int = 512
    audience: str = "owner"
    timings_ms: Dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "FactPack":
        items = [PackItem(**{**i, "memories": [PackMemory(**m) for m in i.get("memories", [])]}) for i in d.get("items", [])]
        return cls(**{**d, "items": items})

    def source_map(self) -> Dict[str, dict]:
        return {i.eid: {"unit_key": i.unit_key, "message_ids": i.message_ids, "memory_ids": [m.memory_id for m in i.memories], "source_id": i.source_id}
                for i in self.items}

    def evidence_text(self) -> str:
        """Every text a claim may rest on (item texts, linked memory statements, notes): the grounding check's evidence scope."""
        parts = [i.text for i in self.items] + [m.statement for i in self.items for m in i.memories] + list(self.notes)
        return "\n".join(parts)


# ---------------------------------------------------------------- rendering
def render_item(i: PackItem) -> str:
    head = f"[{i.eid}] {(i.event_time or '时间未知')[:16]}｜{i.source_label}"
    lines = [head] + ["    " + ln for ln in i.text.splitlines() if ln.strip()]
    for m in i.memories:
        span = ""
        if m.valid_from or m.valid_to:
            span = f"；有效期 {(m.valid_from or '?')[:10]} ~ {(m.valid_to or '现在')[:10]}"
        lines.append(f"    （系统整理的记忆：{m.statement}；状态：{m.status}{span}）")
    return "\n".join(lines)


def render_evidence(pack: FactPack) -> str:
    head = f"【记忆证据】（提问时间：{pack.ask_time[:10]}；按时间先后排列，共 {len(pack.items)} 条；每条是一条原始记录，括号里是系统据此整理的记忆）"
    body = "\n".join(render_item(i) for i in pack.items) if pack.items else "（没有检索到相关记录）"
    return head + "\n" + body


def render_notes(pack: FactPack) -> str:
    lines = list(pack.notes)
    if pack.subqueries:
        lines.append("问题包含以下几项，请逐项核对证据：" + "；".join(f"({i + 1}) {s}" for i, s in enumerate(pack.subqueries)))
    if pack.dropped:
        lines.append(f"因上下文长度限制，另有 {len(pack.dropped)} 条排名较低的记录没有列出。")
    return ("【检索说明】（系统根据问题文字和日期计算得出，不是答案）\n" + "\n".join(f"- {l}" for l in lines)) if lines else ""


# ---------------------------------------------------------------- building
def _local(dt: Optional[datetime], tz) -> Optional[str]:
    return dt.astimezone(tz).isoformat(timespec="minutes") if dt else None


def _memory(api, persona_id: str, mid: str, at: datetime) -> Optional[PackMemory]:
    r = api.get(persona_id, mid, known_at=at)
    if r is None or r.recorded_at > at:
        return None
    return PackMemory(r.memory_id, r.statement, r2._status(r, at), r.valid_from.isoformat() if r.valid_from else None,
                      r.valid_to.isoformat() if r.valid_to else None, r.recorded_at.isoformat() if r.recorded_at else None)


def build(api, cs, *, names: Optional[Dict[str, str]] = None, counter=None, audience: str = "owner", redactor=None,
          context_window: int = CONTEXT_WINDOW, system_tokens: int = 0, tz=timezone(timedelta(hours=8))) -> FactPack:
    """``cs``: an EF1 ``CandidateSet``. ``counter``: a token counter (``session.tokens``); without one the counts are ESTIMATED.
    ``system_tokens``: tokens of the reader's fixed instructions (counted by the caller with the same counter)."""
    t0 = time.perf_counter()
    if audience != "owner" and redactor is None:
        raise ValueError("a pack for another audience needs a redactor (disclosure policy)")
    pid, at = cs.persona_id, cs.query.ask_time
    names = names or {}
    top = cs.top
    keys = [c.key for c in top]
    msgs = api.store.messages_by_locator(pid, [k for k in keys if not k.startswith("mem:")], as_of=at, known_at=at)
    by_unit: Dict[str, list] = {}
    for m in msgs:
        by_unit.setdefault(m.locator or m.message_id, []).append(m)
    for k in keys:                                      # units keyed by a message id (a message without a locator)
        if k not in by_unit and not k.startswith("mem:"):
            m = api.get_message(pid, k)
            if m is not None and m.timestamp <= at and (m.recorded_at is None or m.recorded_at <= at):
                by_unit[k] = [m]
    windows = cs.intent.windows
    items: List[PackItem] = []
    dedup: List[dict] = []
    seen_text: Dict[tuple, str] = {}
    for rank, c in enumerate(top, 1):
        mems = [pm for pm in (_memory(api, pid, mid, at) for mid in c.memories) if pm is not None]
        ms = by_unit.get(c.key, [])
        if c.key.startswith("mem:"):
            if not mems:
                continue
            text = "；".join(m.statement for m in mems)
            when = mems[0].recorded_at
            item = PackItem("", c.key, rank, c.score, "memory", "系统整理的记忆（无原始记录）", "memory", when and _local(datetime.fromisoformat(when), tz),
                            when, [], text, [], dict(c.ranks), mems, kind="memory_only")
        elif ms:
            first = ms[0]
            text = "\n".join(f"{speaker_label(pid, m.speaker_id, names)}：{m.content}" for m in ms)
            item = PackItem("", c.key, rank, c.score, first.source_type, SOURCE_LABELS.get(first.source_type, first.source_type), first.source_id,
                            _local(first.timestamp, tz), first.recorded_at.isoformat() if first.recorded_at else None,
                            list(dict.fromkeys(speaker_label(pid, m.speaker_id, names) for m in ms)), text, [m.message_id for m in ms], dict(c.ranks), mems)
        else:
            continue                                    # not visible at the ask time (never shown)
        day = (item.event_time or "")[:10]
        sig = (day, item.text)
        if sig in seen_text:
            dedup.append({"unit_key": c.key, "duplicate_of": seen_text[sig], "reason": "same text, same day"})
            continue
        seen_text[sig] = c.key
        if day:
            d = datetime.fromisoformat(day).date()
            item.in_windows = [w.role for w in windows if w.contains(d)]
        if audience != "owner":
            red = redactor(item.text + " " + " ".join(m.statement for m in mems), at)
            if red is not None:
                item.text, item.memories, item.redacted = red, [], True
        items.append(item)
    items.sort(key=lambda i: ((i.event_time or "9999"), i.rank))
    for n, i in enumerate(items, 1):
        i.eid = f"E{n:02d}" if len(items) < 100 else f"E{n:03d}"
    pack = FactPack(PACK_SCHEMA, pid, cs.question, cs.query.body, at.isoformat(), cs.k, items, list(cs.intent.notes), list(cs.subqueries),
                    list(cs.shape.flags), [], dedup, audience=audience, context_window=context_window, output_tokens=output_tokens_for(cs.shape.flags))
    _fit(pack, counter, system_tokens)
    pack.timings_ms["build"] = round((time.perf_counter() - t0) * 1000, 2)
    return pack


def _count(counter, text: str) -> int:
    if counter is None:
        return int(len(text) * 1.05) + 1
    return counter(text)


def _fit(pack: FactPack, counter, system_tokens: int) -> None:
    """Token accounting with the real tokenizer when given; drop (and record) the lowest-ranked unprotected items only if the prompt
    would not fit ``context_window - output_tokens``."""
    pack.token_mode = getattr(counter, "mode", "ESTIMATED") if counter is not None else "ESTIMATED"
    budget = pack.context_window - pack.output_tokens - 64          # chat template and safety margin

    def measure() -> Dict[str, int]:
        ev = _count(counter, render_evidence(pack))
        notes = _count(counter, render_notes(pack))
        q = _count(counter, pack.body)
        return {"system": system_tokens, "evidence": ev, "notes": notes, "question": q, "total_input": system_tokens + ev + notes + q + 32,
                "reserved_output": pack.output_tokens, "context_window": pack.context_window}

    t = measure()
    if t["total_input"] > budget:
        protected = {i.eid for i in pack.items if any(r in ("target", "mention", "range") for r in i.in_windows)}
        if pack.subqueries:
            best_sub = sorted((i for i in pack.items if "sub_msg" in i.channels), key=lambda i: i.channels["sub_msg"])[:2 * len(pack.subqueries)]
            protected |= {i.eid for i in best_sub}
        size = {i.eid: _count(counter, render_item(i)) + 1 for i in pack.items}
        # first the lowest-ranked unprotected items; only if that is not enough, the lowest-ranked protected ones (a long range summary
        # whose whole window does not fit). Every removal is recorded; the prompt says how many records were left out.
        order = [i for i in sorted(pack.items, key=lambda i: -i.rank) if i.eid not in protected] + \
                [i for i in sorted(pack.items, key=lambda i: -i.rank) if i.eid in protected]
        excess = t["total_input"] - budget
        for i in order:
            if excess <= 0:
                break
            pack.items.remove(i)
            pack.dropped.append({"unit_key": i.unit_key, "rank": i.rank, "tokens": size[i.eid],
                                 "reason": "context_window" if i.eid not in protected else "context_window_protected"})
            excess -= size[i.eid]
        t = measure()
        while t["total_input"] > budget and pack.items:                  # the per-item estimate was short: remove one more at a time
            i = max(pack.items, key=lambda x: (x.eid not in protected, x.rank))
            pack.items.remove(i)
            pack.dropped.append({"unit_key": i.unit_key, "rank": i.rank, "reason": "context_window" if i.eid not in protected else "context_window_protected"})
            t = measure()
    t["fits"] = t["total_input"] <= budget
    pack.tokens = t
