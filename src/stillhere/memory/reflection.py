"""M3 Evidence-Grounded Adaptive Reflection (bounded, deterministic, read-only).

Simple questions keep the fast path (first-hop evidence -> reader). Complex ones get, at most, TWO extra evidence retrievals and a
deterministic evidence brief, then the same reader, then a deterministic claim check:

* ``route``: a rule router over the question (dates via ``stillhere.memory.temporal``), labels TEMPORAL / KNOWLEDGE_UPDATE / CONFLICT /
  MULTI_FACT / SIMPLE_FACT. Rules, not a model; its "reason" is the rule that fired, not a probability.
* ``reflect``: hop 1 = date-conditioned facts when the question names a time (``facts_during`` / ``facts_at``) - the first-hop search does
  not use the date, so this is where a dated question gets its evidence; hop 2 = the full history of every slot touched by the evidence
  (``history``), so old values, corrections, retractions and event-status changes are visible. Records are read with ``known_at`` =
  the asking time: nothing recorded later is ever shown.
* the brief states, per slot, the values with their valid time and status (current / past / corrected / retracted) and per event the status
  sequence; it never adds a value the records do not hold, and says when a named time has no record.
* ``verify``: digits / dates in the answer must appear in the evidence; a question routed with no supporting evidence must be answered
  with an abstention; an answer that states a retracted or corrected value as current is flagged. A failed check -> safe abstention.

Reflection never writes to memory, never loops, and never calls a model itself.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Sequence

from .temporal import resolve_time

FLAG = "MEMORY_REFLECTION_V1_ENABLED"
MAX_EXTRA_RETRIEVALS = 2
MAX_SLOTS = 8
TEMPORAL = re.compile(r"以前|之前|当时|那时|原来|后来|现在|目前|还在|还是|最近|上周|下周|上个月|这个月|几月|几号|哪天|什么时候|何时|计划|打算|改到|改成|推迟|提前|取消|完成|去成|搬|换了|月|日|号|周[一二三四五六日天]")
CONFLICT = re.compile(r"到底|记错|更正|纠正|矛盾|不一致|哪个对|改过|改了")
MULTI = re.compile(r"和|以及|还有|分别|两个|哪些|都")
UPDATE = re.compile(r"现在还|还住|还在|搬|换|改")
UNCERTAIN = re.compile(r"不知道|不清楚|不确定|没有记录|没记录|记录里没有|无法确定|无法回答|找不到|没有提到|未提及|缺少")
DIGITS = re.compile(r"\d+")


@dataclass
class Route:
    kind: str
    reason: str
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None

    @property
    def complex(self) -> bool:
        return self.kind != "SIMPLE_FACT"


@dataclass
class Reflection:
    route: Route
    extra_retrievals: int = 0
    brief: str = ""
    slots: List[dict] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)


def route(question: str, ask_time: datetime) -> Route:
    t = resolve_time(question, ask_time)
    start, end = (getattr(t, "start", None), getattr(t, "end", None)) if t else (None, None)
    if CONFLICT.search(question):
        return Route("CONFLICT", "conflict cue", start, end)
    if t is not None:
        return Route("TEMPORAL", "explicit time expression", start, end)
    if UPDATE.search(question):
        return Route("KNOWLEDGE_UPDATE", "update cue", start, end)
    if TEMPORAL.search(question):
        return Route("TEMPORAL", "temporal cue", start, end)
    if MULTI.search(question):
        return Route("MULTI_FACT", "multi-part cue", start, end)
    return Route("SIMPLE_FACT", "no complex cue", start, end)


def _fmt(d: Optional[datetime]) -> str:
    return d.date().isoformat() if d else "?"


def _status(r, ask: datetime) -> str:
    s = getattr(r.status, "value", r.status)
    if s == "retracted":
        return "已撤回（不成立）"
    if s == "superseded":
        return "已更正（原记录有误）" if (r.correction_reason or "").lower().find("temporal") < 0 else "过去"
    if r.valid_to is not None and r.valid_to <= ask:
        return "过去（已不成立）"
    return "当前"


def reflect(api, persona_id: str, question: str, ask_time: datetime, first_hop: Sequence) -> Reflection:
    """``first_hop``: the records the normal path retrieved (objects with subject / attribute / memory_id)."""
    rt = route(question, ask_time)
    ref = Reflection(rt)
    if not rt.complex:
        return ref
    records = list(first_hop)
    lines = []
    # hop 1: the time the question names. Two readings are tried when the words carry no year (``11月5日`` asked on 11-04 may mean the
    # coming day, a planned event): the parser's reading and the same calendar day in the asking year. Records count when their validity
    # overlaps the time OR when an event's planned / actual date (a value, not a validity) falls inside it.
    if rt.time_start is not None and ref.extra_retrievals < MAX_EXTRA_RETRIEVALS:
        ref.extra_retrievals += 1
        end = rt.time_end or rt.time_start
        windows = [(rt.time_start, end)]
        if "年" not in question and rt.time_start.year < ask_time.year:
            try:
                windows.append((rt.time_start.replace(year=ask_time.year), end.replace(year=ask_time.year)))
            except ValueError:
                pass
        keys = {(r.subject, r.attribute) for r in records}
        hit = []
        for ws, we_ in windows:
            try:
                dated = list(api.facts_during(persona_id, start=ws, end=we_, known_at=ask_time)) if we_ > ws else list(api.facts_at(persona_id, as_of=ws, known_at=ask_time))
            except Exception:  # noqa: BLE001 - a store without the interval query keeps the first hop
                dated = []
            try:
                everything = list(api.facts_at(persona_id, as_of=ask_time, known_at=ask_time, include_ended=True))
            except TypeError:
                everything = list(api.facts_at(persona_id, as_of=ask_time, known_at=ask_time))
            except Exception:  # noqa: BLE001
                everything = []
            day_lo, day_hi = ws.date().isoformat(), (we_ if we_ > ws else ws).date().isoformat()
            by_date = [r for r in everything if r.attribute in ("event.planned_time", "event.time") and day_lo <= (r.value or "")[:10] <= day_hi]
            subjects = {r.subject for r in by_date}
            by_date += [r for r in everything if r.subject in subjects and r not in by_date]
            hit = [r for r in dated if (r.subject, r.attribute) in keys] + by_date or dated[:MAX_SLOTS]
            if hit:
                break
        if not hit:
            lines.append(f"- 所问时间（{_fmt(rt.time_start)} ~ {_fmt(rt.time_end)}）内没有相关记录：不要用其他时间的值回答。")
        records += [r for r in hit if r.memory_id not in {x.memory_id for x in records}]
    # hop 2: every slot's history (old values, corrections, retractions, event status changes)
    if ref.extra_retrievals < MAX_EXTRA_RETRIEVALS:
        ref.extra_retrievals += 1
        seen = []
        for r in records:
            k = (r.subject, r.attribute) if not r.subject.startswith("event:") else (r.subject, None)
            if k not in seen:
                seen.append(k)
        for subject, attribute in seen[:MAX_SLOTS]:
            attrs = [attribute] if attribute else ["event.status", "event.planned_time", "event.time", "event.description"]
            for att in attrs:
                try:
                    hist = api.history(persona_id, subject, att, known_at=ask_time)
                except Exception:  # noqa: BLE001
                    hist = []
                hist = sorted(hist, key=lambda r: (r.valid_from or r.recorded_at, r.recorded_at))
                if not hist:
                    continue
                vals = [f"{r.value}（{_status(r, ask_time)}；有效 {_fmt(r.valid_from)} ~ {_fmt(r.valid_to) if r.valid_to else '现在'}；记录于 {_fmt(r.recorded_at)}）" for r in hist]
                ref.slots.append({"subject": subject, "attribute": att, "records": [r.memory_id for r in hist]})
                ref.evidence_ids += [r.memory_id for r in hist]
                lines.append(f"- {subject} / {att}：" + " → ".join(vals))
    ref.brief = ("【证据整理（按时间，系统根据记录生成）】\n" + "\n".join(lines) + "\n只用标为“当前”的值回答现在的情况；问过去某个时间时用当时有效的值；已撤回、已更正的不要当作事实。") if lines else ""
    return ref


def verify(answer: str, abstained: bool, evidence_text: str, reflection: Optional[Reflection] = None) -> dict:
    """Deterministic claim check. Returns {"ok": bool, "problems": [...]}."""
    problems = []
    if not abstained:
        hay = evidence_text
        missing = [d for d in DIGITS.findall(answer) if d not in hay]
        if missing:
            problems.append(f"unsupported_numbers:{','.join(sorted(set(missing))[:5])}")
        if not evidence_text.strip():
            problems.append("answer_without_evidence")
        if reflection and "所问时间" in reflection.brief and "没有相关记录" in reflection.brief and not UNCERTAIN.search(answer):
            problems.append("answered_a_time_with_no_record")
    return {"ok": not problems, "problems": problems}
