"""Batch 2 first-hop evidence retrieval (B2): the legacy lexical search plus three cheap, deterministic channels, merged and deduplicated.

Behind ``MEMORY_RETRIEVAL_B2_ENABLED`` (off: callers keep ``MemoryAPI.search``). Read-only; every read uses ``known_at`` = the asking time.

1. predicate aliases: question cues -> stored attributes (``不吃`` -> dislikes, ``做什么的`` -> job / works_at, ``狗`` -> pet ...); every record of
   those slots (current and ended) is a candidate. This fixes first-hop misses where the question and the stored statement share no words.
2. subject awareness: a person named in the question, or a relation word the persona has a stated relationship for ("我妹妹" -> 小雨) -> that person's records (``person:<name>``) and
   events whose actor is that person. Records about OTHER people are only taken by this channel, so "my" questions are not answered from a
   friend's facts.
3. date awareness: a date / month in the question (``resolve_time``; two readings when no year: the parser's and the asking year) -> facts
   valid in that window and events whose planned or actual date (a value, not a validity) falls in it. Dates are only used to SELECT evidence:
   a date in the question is never taken as the date something happened. Batch 2.1: windows come from ``query_windows`` (relative
   weekdays / days / weekends via ``relative_days``; the window end is exclusive, the old inclusive end matched the following day).

Events are grouped: an event subject contributes all its fields (description, actor, status, times) as one unit, so status and date travel with
the description. Order: subject, date, alias (slots) channels, then legacy hits, then the persona's own events (most similar description
first); duplicates removed by memory id; ``k`` evidence UNITS returned
(an event unit counts once).
"""
from __future__ import annotations

import os
import re
import threading
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from . import relative_days
from .temporal import resolve_time

FLAG = "MEMORY_RETRIEVAL_B2_ENABLED"
ALIASES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (r"不吃|不喝|忌口|过敏|讨厌|不喜欢|不爱", ("dislikes",)),
    (r"喜欢|爱好|爱吃|爱喝|运动|饮品|喝什么|吃什么|口味|吃辣|能吃|还吃", ("likes", "dislikes", "routine")),
    (r"做什么的|做什么工作|干什么的|职业|工作|上班|退休前|教什么|教哪|什么老师|当什么|哪一行|干哪行", ("job", "works_at")),   # post-retest: 教什么 etc.
    (r"公司|单位|在哪上班|在哪工作", ("works_at",)),
    (r"住|家在|哪里人|哪个区|哪个城市|搬", ("lives_in",)),
    (r"宠物|狗|猫|鹦鹉|仓鼠|养了", ("pet",)),
    (r"学校|专业|大学|读书|上学|读的", ("school",)),
    (r"老公|老婆|丈夫|妻子|儿子|女儿|弟弟|妹妹|哥哥|姐姐|爸|妈|朋友|同事|叫什么", ("relationship",)),
    (r"睡眠|睡得|心情|情绪|精力|压力|体重", ("state",)),
    (r"每天|平时|习惯|早上|晚上|一般", ("routine",)),
    (r"计划|安排|打算|准备|哪天|什么时候|几号|去了吗|去成|考试|面试|体检|旅行|旅游|出差|比赛|马拉松|拔|报名|取消|改到|完成|发生", ("event",)),
)


def enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    return (env if env is not None else os.environ).get(FLAG, "").strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Evidence:
    records: List = field(default_factory=list)          # MemoryRecord objects, grouped by unit, in rank order
    units: List[Tuple[str, List[str]]] = field(default_factory=list)   # (unit key, memory ids)
    channels: Dict[str, int] = field(default_factory=dict)


def _bigrams(s: str) -> set:
    s = re.sub(r"\W", "", s or "")
    return {s[i:i + 2] for i in range(len(s) - 1)}


def attributes_for(question: str) -> List[str]:
    out: List[str] = []
    for pat, attrs in ALIASES:
        if re.search(pat, question):
            out += [a for a in attrs if a not in out]
    return out


_ALL_RECORDS_MEMO: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_ALL_RECORDS_LOCK = threading.Lock()
ALL_RECORDS_MEMO_SIZE = 4


def _all_records(api, persona_id: str, known_at: datetime) -> List:
    """Every active / superseded record of the persona as believed at ``known_at``. One SEARCH asks for the same list three times (B2,
    and twice in the D4 evidence view) and a benchmark day asks it for every question at one ask time: the list is memoised per store,
    keyed by (persona, known_at, the store's ``read_generation``), so any write, rollback or outside commit recomputes it. A fresh list is
    returned each time (its records are frozen)."""
    store = api.store
    gen = store.read_generation() if hasattr(store, "read_generation") else None
    if gen is None:
        return _all_records_uncached(api, persona_id, known_at)
    key = (persona_id, known_at.isoformat(), gen)
    with _ALL_RECORDS_LOCK:
        memo = _ALL_RECORDS_MEMO.setdefault(store, {})
        hit = memo.get(key)
    if hit is not None:
        return list(hit)
    out = _all_records_uncached(api, persona_id, known_at)
    if store.read_generation() == gen:            # nothing changed while computing: safe to keep
        with _ALL_RECORDS_LOCK:
            memo = _ALL_RECORDS_MEMO.setdefault(store, {})
            memo[key] = tuple(out)
            while len(memo) > ALL_RECORDS_MEMO_SIZE:
                memo.pop(next(iter(memo)))
    return out


def _all_records_uncached(api, persona_id: str, known_at: datetime) -> List:
    sql = "SELECT memory_id FROM memories WHERE persona_id = ? ORDER BY recorded_at_utc, memory_id"
    read = getattr(api.store, "_read", None)
    if read is None:
        rows = api.store._conn.execute(sql, (persona_id,)).fetchall()
    else:
        # under the store's lock: the connection is shared by threads, and two threads stepping the same cached prepared statement at
        # once got an empty or a duplicated id list (the flaky tests/test_read_cache.py concurrency test, 13 / 60 runs before this fix)
        with read() as conn:
            rows = conn.execute(sql, (persona_id,)).fetchall()
    out = []
    for (mid,) in rows:
        r = api.get(persona_id, mid, known_at=known_at)
        if r is not None and r.recorded_at <= known_at and getattr(r.status, "value", r.status) in ("active", "superseded"):
            out.append(r)
    return out


def query_windows(question: str, ask_time: datetime) -> List[Tuple[datetime, datetime, str]]:
    """The day windows a question names, in search order: [(start, end_exclusive, rule)] (Batch 2.1).

    A relative day / weekday / weekend (``relative_days``; both readings of an ambiguous ``下周X``) comes first; otherwise the core parser's
    reading and, for a date without a year read into an earlier year, the same date in the asking year. Anchored on ``ask_time`` only."""
    w = relative_days.resolve(question, ask_time)
    if w is not None:
        return [(r.start, r.end, r.rule) for r in (w.readings or [w])]
    t = resolve_time(question, ask_time)
    if t is None or getattr(t, "start", None) is None:
        return []
    end = t.end or (t.start + timedelta(days=1))
    out = [(t.start, end, "parser")]
    if "年" not in question and t.start.year < ask_time.year:
        try:
            out.append((t.start.replace(year=ask_time.year), end.replace(year=ask_time.year), "asking-year reading"))
        except ValueError:
            pass
    return out


def people_in(question: str, records: Sequence) -> List[str]:
    names = set()
    for r in records:
        if r.attribute == "relationship" and r.value:
            names.add(r.value.split()[-1])
        if r.subject.startswith("person:"):
            names.add(r.subject.split(":", 1)[1])
        if r.attribute == "event.actor" and r.value.startswith("person:"):
            names.add(r.value.split(":", 1)[1])
    hits = {n for n in names if n and n in question}
    # a relation word in the question ("my sister") resolves to the person the persona stated that relation with ("妹妹 小雨")
    for r in records:
        if r.subject == "persona" and r.attribute == "relationship" and r.value and len(r.value.split()) >= 2:
            rel, name = r.value.split()[0], r.value.split()[-1]
            if rel and rel in question:
                hits.add(name)
    return sorted(hits)


def retrieve(api, persona_id: str, question: str, ask_time: datetime, k: int = 10, legacy_k: int = 10) -> Evidence:
    recs = _all_records(api, persona_id, ask_time)
    by_subject: Dict[str, List] = {}
    for r in recs:
        by_subject.setdefault(r.subject, []).append(r)
    ev = Evidence()
    order: List[str] = []           # unit keys in rank order

    def add_unit(key: str, channel: str):
        if key not in order:
            order.append(key)
            ev.channels[channel] = ev.channels.get(channel, 0) + 1

    def event_actor(subject: str) -> str:
        acts = [r.value for r in by_subject.get(subject, []) if r.attribute == "event.actor"]
        return acts[-1] if acts else "self"

    named = people_in(question, recs)
    attrs = attributes_for(question)
    # 2. subject channel (first: a named person is the most specific cue)
    for name in named:
        for subj, rs in by_subject.items():
            if subj == f"person:{name}":
                for r in rs:
                    add_unit(r.memory_id, "subject")
            elif subj.startswith("event:") and event_actor(subj) == f"person:{name}":
                add_unit(subj, "subject")
        for r in recs:
            if r.attribute == "relationship" and r.value.endswith(name):
                add_unit(r.memory_id, "subject")
    # 3. date channel
    for ws, we_, _rule in query_windows(question, ask_time):
        lo, hi = ws.date().isoformat(), (we_ - timedelta(microseconds=1)).date().isoformat() if we_ > ws else ws.date().isoformat()
        hit = False
        for r in recs:
            if r.attribute in ("event.planned_time", "event.time") and lo <= (r.value or "")[:10] <= hi:
                add_unit(r.subject, "date")
                hit = True
        if hit:
            break
    # 1. alias channel (persona's own slots; other people's only through the subject channel)
    for r in recs:
        if r.subject == "persona" and (r.attribute in attrs or (r.attribute.startswith("state:") and "state" in attrs)):
            add_unit(r.memory_id, "alias")
    # legacy lexical hits (before the generic event channel, which can be large)
    for h in api.search(persona_id, question, as_of=ask_time, known_at=ask_time, k=legacy_k, include_ended=True):
        if not h.memory_id:
            continue
        r = api.get(persona_id, h.memory_id, known_at=ask_time)
        if r is None:
            continue
        add_unit(r.subject if r.subject.startswith("event:") else r.memory_id, "legacy")
    # generic event channel: the persona's own events, most similar description first, then most recent
    if "event" in attrs:
        qb = _bigrams(question)
        own = []
        for subj, rs in by_subject.items():
            if subj.startswith("event:") and event_actor(subj) == "self":
                desc = " ".join(r.value or "" for r in rs if r.attribute in ("event.description", "event.activity"))
                own.append((-len(qb & _bigrams(desc)), -max(r.recorded_at.timestamp() for r in rs), subj))
        for _o, _t, subj in sorted(own):
            add_unit(subj, "alias_event")
    ids = {r.memory_id: r for r in recs}
    for key in order[:k]:
        if key.startswith("event:"):
            rs = sorted(by_subject.get(key, []), key=lambda r: (r.recorded_at, r.attribute))
        else:
            rs = [ids[key]] if key in ids else [api.get(persona_id, key, known_at=ask_time)]
        rs = [r for r in rs if r is not None]
        ev.units.append((key, [r.memory_id for r in rs]))
        ev.records += rs
    return ev


def legacy(api, persona_id: str, question: str, ask_time: datetime, k: int = 10) -> Evidence:
    """The old first hop (as used by M3 R2), in the same Evidence shape, for side-by-side evaluation."""
    ev = Evidence()
    for h in api.search(persona_id, question, as_of=ask_time, known_at=ask_time, k=k, include_ended=True):
        if h.memory_id:
            r = api.get(persona_id, h.memory_id, known_at=ask_time)
            if r is not None:
                ev.units.append((r.memory_id, [r.memory_id]))
                ev.records.append(r)
    ev.channels["legacy"] = len(ev.units)
    return ev
