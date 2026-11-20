"""Batch 2.1 writer precision rules (M2.2.1), on top of M2.2 (``m22``), behind ``MEMORY_WRITER_M221_ENABLED`` (needs M2.2 on; off = M2.2 exactly).

Found on the Batch 2 runs (DEV v2 world and the now-inspected v3 world), each rule acts only on what the candidate cites and on existing memory:

R5  reference resolution: ``refers_to`` that names one ATTRIBUTE record of an event (``w.…``) is resolved to the event (``event:…``), so a
    reschedule / completion / retraction updates that event instead of creating a new one.
R6  negated preference: ``likes X`` with negative polarity ("我现在不吃辣了") is ``dislikes X`` (before verification).
R4b reschedule: a PLANNED / SCHEDULED candidate with a new planned date (or marked as a correction) links to the one open event of the same
    actor and activity (M2.2's R4 covered SCHEDULED only).
R8  not-a-fact: a candidate whose cited wording is a suggestion / question ("要不…？"), a hypothetical ("如果", "要是…就"), advice
    ("医生建议", "让我少…"), a wish ("我想去…" with no plan word, no status change such as 原本 / 后来 / 没去成, on an undecided plan) or hearsay about the persona ("阿雷说我…", "人家都说") is not written.
R10 third-party minimisation (privacy): for other people only job / workplace / home / school / relationship and their own events are kept;
    their preferences, pets, possessions, habits, health, age and birthday are not the persona's memory.
R11 retraction linking: an event retraction that does not resolve to an event is linked to the one event of the same activity.
R7  one occurrence, one event (batch level): an extra event with no definite status (a cause / consequence, "演唱会延期", "回不了杭州") cited
    from the same message as an event that has a status is not written.
R12 a workplace / home / school whose value is only a generic word ("公司（调岗）") says nothing and is not stored.
R9  a routine cited from the same message as an event of the same activity ("报名了摄影班，每周三上课") is a detail of that event.
"""
from __future__ import annotations

import os
import re
from dataclasses import replace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from . import m22

FLAG = "MEMORY_WRITER_M221_ENABLED"
DEFINITE = ("PLANNED", "SCHEDULED", "IN_PROGRESS", "COMPLETED", "CANCELLED")
THIRD_PARTY_KEEP = ("job", "works_at", "lives_in", "school", "relationship", "event")
SUGGESTION = re.compile(r"要不|不如|要么|[？?]\s*$")
HYPOTHETICAL = re.compile(r"如果|要是|假如|万一|的话|假设")
ADVICE = re.compile(r"建议|让我|叫我|劝我|要我|医生说|应该|最好")
WISH = re.compile(r"想去|想要|希望|梦想")
PLAN_WORD = re.compile(r"打算|准备|计划|要去|报名|预约|约了|订了|定了|决定")
HEARSAY = re.compile(r"(说过?我|都说|人家说|听说|据说|别人说|他们说)")
STATE_CHANGE = re.compile(r"原本|本来|后来|改成|改到|取消|没去成|没去|退了|延期")
GENERIC = {"公司", "一家公司", "单位", "学校", "家", "老家"}


def enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    e = env if env is not None else os.environ
    return m22.enabled(e) and e.get(FLAG, "").strip().lower() in ("1", "true", "yes", "on")


def prenormalize(c):
    """R6 (before verification)."""
    if c.predicate == "likes" and (c.polarity or "positive") == "negative":
        return replace(c, predicate="dislikes", polarity="positive")
    return c


def _own(c, msgs, persona_id: Optional[str]):
    """The cited messages the persona wrote, when there are any (an assistant's suggestion the persona accepts is not the persona's
    hypothetical); otherwise every cited message."""
    cited = [msgs[m] for m in c.source_message_ids if msgs.get(m) is not None]
    own = [m for m in cited if persona_id is None or getattr(m, "speaker_id", persona_id) == persona_id]
    return own or cited


def _context(c, msgs: Mapping[str, object], persona_id: Optional[str] = None, sentence: bool = False) -> List[str]:
    """Each cited span with up to 10 characters before it in its message. ``sentence=False``: never across a clause boundary (，；) -
    hotfix after the b21 confirmation run: "朋友都说…好吃，我自己也挺喜欢做饭" must not read the first clause's "都说" as hearsay about the
    second. ``sentence=True``: up to the sentence start (。！？) - a hypothetical spans its clauses ("如果下个月有空，我想去西藏")."""
    out = []
    stops = "。！!？?" if sentence else "，,。；;！!？?"
    for m in _own(c, msgs, persona_id):
        text = getattr(m, "content", "")
        for s in c.source_spans or [""]:
            i = text.find(s) if s else -1
            if i < 0:
                out.append(text)
                continue
            head = text[: i] if sentence else text[max(0, i - 10): i]
            cut = max(head.rfind(ch) for ch in stops)
            out.append((head[cut + 1:] if cut >= 0 else head) + s)
    return out or [""]


def not_a_fact(c, msgs, persona_id: Optional[str] = None) -> Optional[str]:
    """R8. Returns the reason, or None. Reads only the persona's own cited messages when there are any."""
    ctx = _context(c, msgs, persona_id)
    sent = _context(c, msgs, persona_id, sentence=True)
    whole = " ".join(getattr(m, "content", "") for m in _own(c, msgs, persona_id))
    if any(HYPOTHETICAL.search(x) for x in sent):
        return "R8 hypothetical"
    if c.predicate in ("event", "routine") and any(SUGGESTION.search(x) for x in sent + [whole.strip()]) and not PLAN_WORD.search(" ".join(sent)):
        return "R8 suggestion / question"
    if c.subject == "self" and any(ADVICE.search(x) for x in ctx):
        return "R8 advice"
    if (c.predicate == "event" and c.event_status in (None, "PLANNED", "UNKNOWN") and any(WISH.search(x) for x in ctx)
            and not any(PLAN_WORD.search(x) or STATE_CHANGE.search(x) for x in ctx + [whole])):
        return "R8 wish"
    if c.subject == "self" and any(HEARSAY.search(x) for x in ctx):
        return "R8 hearsay about the persona"
    return None


def event_index(existing: Sequence[Tuple[str, str, str, str]]):
    """(memory id -> event subject, [event field dicts of every event, any status])."""
    id2subject, ev = {}, {}
    for mid, subject, attribute, value in existing:
        if subject.startswith("event:"):
            id2subject[mid] = subject
            ev.setdefault(subject, {})[attribute] = value
    return id2subject, [{"subject": s, **f} for s, f in ev.items()]


def _same(c, e) -> bool:
    d = e.get("event.description", "")
    return m22.same_activity(c.object, d) or m22.same_activity(c.event_key or "", d) or (bool(c.activity) and m22.same_activity(c.activity, d))


def normalize(c, msgs, persona_id: str, people: set, events: Sequence[dict], existing: Sequence[Tuple[str, str, str, str]]):
    """M2.2's rules, then R5 / R4b / R8 / R10 / R11. Returns (candidate or None, notes)."""
    id2subject, all_events = event_index(existing)
    notes: List[str] = []
    # R5 before M2.2's R4 so the resolved event is kept
    if c.refers_to and not c.refers_to.startswith("event:") and c.refers_to in id2subject:
        c = replace(c, refers_to=id2subject[c.refers_to])
        notes.append(f"R5 refers_to resolved to {c.refers_to}")
    why = not_a_fact(c, msgs, persona_id)
    if why:
        return None, notes + [why]
    c2, n2 = m22.normalize(c, msgs, persona_id, people, events)
    notes += n2
    if c2 is None:
        return None, notes
    c = c2
    if c.subject.startswith("person:") and c.predicate not in THIRD_PARTY_KEEP:
        return None, notes + [f"R10 third-party {c.predicate} not stored"]
    if c.predicate in ("works_at", "lives_in", "school") and re.sub(r"[（(].*?[)）]|\W", "", c.object or "") in GENERIC:
        return None, notes + [f"R12 generic value {c.object!r} not stored"]
    if c.predicate == "event" and not (c.refers_to or "").startswith("event:"):
        actor = c.subject
        if c.retracts:
            hits = [e for e in all_events if (e.get("event.actor") or "self") == actor and _same(c, e)]
            if len(hits) == 1:
                c = replace(c, refers_to=hits[0]["subject"])
                notes.append(f"R11 retraction linked to {hits[0]['subject']}")
        elif c.event_status in ("PLANNED", "SCHEDULED") and (c.planned_time or c.corrects or c.type == "CORRECTION"):
            hits = [e for e in events if e.get("event.status") in ("PLANNED", "SCHEDULED") and (e.get("event.actor") or "self") == actor and _same(c, e)]
            if len(hits) == 1:
                c = replace(c, refers_to=hits[0]["subject"])
                notes.append(f"R4b reschedule linked to {hits[0]['subject']}")
    return c, notes


def batch_filter(verdicts) -> List[Tuple[int, str]]:
    """R7 / R9 over the committable verdicts of one group. Returns [(index, reason)] to drop."""
    live = [(i, v.candidate) for i, v in enumerate(verdicts) if v.committable]
    drop: Dict[int, str] = {}
    events = [(i, c) for i, c in live if c.predicate == "event"]
    for i, c in events:
        if c.event_status in DEFINITE:
            continue
        for j, d in events:
            if j != i and d.event_status in DEFINITE and set(c.source_message_ids) & set(d.source_message_ids) and d.subject == c.subject:
                drop[i] = "R7 detail / cause of an event in the same message"
                break
    for i, c in live:
        if c.predicate != "routine":
            continue
        for j, d in events:
            if set(c.source_message_ids) & set(d.source_message_ids) and d.subject == c.subject and (
                    m22.same_activity(c.object, d.object) or (d.activity and d.activity in c.object) or _shared_core(c.object, d.object)):
                drop[i] = "R9 routine is a detail of an event in the same message"
                break
    return sorted(drop.items())


def _shared_core(a: str, b: str) -> bool:
    """A common substring of >= 3 characters ("社区摄影班")."""
    a, b = re.sub(r"\W", "", a or ""), re.sub(r"\W", "", b or "")
    return any(a[i:i + 3] in b for i in range(max(0, len(a) - 2)))
