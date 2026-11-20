"""Temporal boundaries of a question: which side of a date the asked time lies on, and whether that day itself counts.

``memory.temporal.find_times`` places a date as a half-open calendar interval ``[start, end)`` in +08:00. This module reads the
words around it and turns them into one comparison:

=================  ===========================  ==================================
operator           bounds (half-open)           example
=================  ===========================  ==================================
``BEFORE``         ``t < start``                11月5日之前, 早于, before
``ON_OR_BEFORE``   ``t < end``                  截至11月5日, 11月5日为止, 不晚于, by, on or before
``ON``             ``start <= t < end``         11月5日当天, 11月5日 (a date with no relation word)
``AFTER``          ``t >= end``                 11月5日之后, 晚于, after
``ON_OR_AFTER``    ``t >= start``               自11月5日起, 11月5日以来, 不早于, since, on or after
``BETWEEN``        ``start1 <= t < end2``       11月3日到11月5日之间, between (both days included)
=================  ===========================  ==================================

The interval is the whole calendar unit named: ``11月之前`` is before 1 November, ``11月为止`` runs to the end of November.

Explicit words win: ``（含当天）``/``包括当天`` make the target day count, ``不含当天``/``不包括当天`` make it not count.

Some wordings stay **AMBIGUOUS**, never forced into a comparison: ``前后``/``左右`` (around), ``到``/``直到``/``until`` without
``为止`` (inclusive or not), several dates without a range word, a date ``find_times`` cannot place (vague, invalid, future). No
date at all is ``NONE``. Comparing an event to an AMBIGUOUS or NONE boundary gives ``UNKNOWN``, as does an event time too coarse to
decide (an event known only to the month, against a day boundary inside that month).

Which time is compared is the caller's choice and is labelled: said, occurred, scheduled (planned), valid from, recorded. The
system's knowledge time is never compared here: ``known_at`` already decides what is visible at all.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple

from stillhere.contracts.planning import INTERVAL_KINDS
from stillhere.memory.search import normalize
from stillhere.memory.temporal import CALENDAR_TZ, find_times

BEFORE, ON_OR_BEFORE, ON, AFTER, ON_OR_AFTER, BETWEEN = "BEFORE", "ON_OR_BEFORE", "ON", "AFTER", "ON_OR_AFTER", "BETWEEN"
RESOLVED, AMBIGUOUS, UNKNOWN, NONE = "RESOLVED", "AMBIGUOUS", "UNKNOWN", "NONE"
SATISFIES, VIOLATES = "SATISFIES", "VIOLATES"

# relation words: (text, operator or AMBIGUOUS). Longest first wins at a position.
_AFTER_WORDS = (("前后", AMBIGUOUS), ("左右", AMBIGUOUS), ("之前", BEFORE), ("以前", BEFORE), ("为止", ON_OR_BEFORE), ("当天", ON), ("当日", ON),
                ("那天", ON), ("这天", ON), ("之后", AFTER), ("以后", AFTER), ("以来", ON_OR_AFTER), ("开始", ON_OR_AFTER), ("起", ON_OR_AFTER),
                ("前", BEFORE), ("后", AFTER))
_BEFORE_WORDS = (("不晚于", ON_OR_BEFORE), ("不迟于", ON_OR_BEFORE), ("不早于", ON_OR_AFTER), ("截止到", ON_OR_BEFORE), ("截至", ON_OR_BEFORE),
                 ("截止", ON_OR_BEFORE), ("早于", BEFORE), ("先于", BEFORE), ("晚于", AFTER), ("迟于", AFTER), ("自从", ON_OR_AFTER), ("直到", AMBIGUOUS),
                 ("自", ON_OR_AFTER), ("从", ON_OR_AFTER), ("到", AMBIGUOUS))
_ENGLISH = ((r"\bon or before\b|\bno later than\b|\bby\b", ON_OR_BEFORE), (r"\bon or after\b|\bno earlier than\b|\bsince\b", ON_OR_AFTER),
            (r"\bbefore\b|\bprior to\b", BEFORE), (r"\bafter\b", AFTER), (r"\buntil\b|\btill\b", AMBIGUOUS), (r"\bon\b", ON))
_RANGE = re.compile(r"^\s*(?:到|至|~|-|—|to)\s*$")          # 和 / 与 / and list dates, they do not span them
_INCLUDE_DAY = ("含当天", "包括当天", "包含当天", "含当日", "包括当日")
_EXCLUDE_DAY = ("不含当天", "不包括当天", "不包含当天", "不含当日", "不包括当日")
_DAY_SENSITIVE = (BEFORE, ON_OR_BEFORE, AFTER, ON_OR_AFTER)


@dataclass(frozen=True)
class Boundary:
    status: str
    operator: Optional[str] = None
    lower: Optional[datetime] = None      # inclusive
    upper: Optional[datetime] = None      # exclusive
    target_start: Optional[datetime] = None
    target_end: Optional[datetime] = None
    granularity: Optional[str] = None
    reason: str = ""
    span: str = ""
    description: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in d.items()}


def _label(t: datetime, granularity: str) -> str:
    local = t.astimezone(CALENDAR_TZ)
    if granularity == "year":
        return f"{local.year}年"
    if granularity in ("month",):
        return f"{local.year}年{local.month}月"
    return f"{local.year}-{local.month:02d}-{local.day:02d}"


def _last_day(end: datetime, granularity: str) -> str:
    return _label(end - timedelta(microseconds=1), granularity if granularity in ("year", "month") else "day")


def _describe(op: str, start: datetime, end: datetime, granularity: str, end2: Optional[Tuple[datetime, str]] = None) -> str:
    a, unit = _label(start, granularity), ("当年" if granularity == "year" else "当月" if granularity == "month" else "当天")
    if op == BEFORE:
        return f"严格早于{a}：不包括{a}{unit}"
    if op == ON_OR_BEFORE:
        return f"截至{a}：包括{a}{unit}及之前"
    if op == AFTER:
        return f"晚于{a}：不包括{a}{unit}"
    if op == ON_OR_AFTER:
        return f"{a}{unit}及之后"
    if op == BETWEEN and end2:
        return f"{a}至{_last_day(*end2)}：包括两端"
    return f"{a}{unit}（{start.astimezone(CALENDAR_TZ).isoformat(timespec='minutes')} 至 {end.astimezone(CALENDAR_TZ).isoformat(timespec='minutes')}，不含后者）"


def _match_word(text: str, pos: int, words, *, forward: bool) -> Optional[Tuple[str, str]]:
    if forward:
        while pos < len(text) and text[pos].isspace():      # "2026-11-05 之前": a space never turns a boundary into "on"
            pos += 1
    for w, op in words:
        if forward and text.startswith(w, pos):
            return w, op
        if not forward and text[:pos].rstrip().endswith(w):
            return w, op
    return None


def parse_boundary(question: str, as_of: Optional[datetime]) -> Boundary:
    """The temporal boundary of a question, or AMBIGUOUS / UNKNOWN / NONE with the reason. Deterministic; no model."""
    text = normalize(question)
    mentions = find_times(text, as_of)
    dated = [m for m in mentions if m.expression.kind in INTERVAL_KINDS and m.expression.start is not None]
    unplaced = [m for m in mentions if m.expression.resolution_status in ("unresolved", "invalid", "future")]
    if not dated:
        if unplaced:
            return Boundary(UNKNOWN, reason=f"time cannot be placed ({unplaced[0].expression.resolution_status})", span=text[unplaced[0].start:unplaced[0].end])
        return Boundary(NONE, reason="no date in the question")
    lower_text = text.lower()
    include_day = any(w in text for w in _INCLUDE_DAY) and not any(w in text for w in _EXCLUDE_DAY)
    exclude_day = any(w in text for w in _EXCLUDE_DAY)

    if len(dated) >= 2:
        a, b = dated[0], dated[1]
        between = text[a.end:b.start]
        if len(dated) == 2 and (_RANGE.match(between) or "之间" in text[b.end:b.end + 3] or "between" in lower_text):
            ea, eb = a.expression, b.expression
            if eb.end <= ea.start:
                return Boundary(AMBIGUOUS, reason="range ends before it starts", span=text[a.start:b.end])
            return Boundary(RESOLVED, BETWEEN, ea.start, eb.end, ea.start, eb.end, ea.granularity, span=text[a.start:b.end],
                            description=_describe(BETWEEN, ea.start, ea.end, ea.granularity, (eb.end, eb.granularity)))
        return Boundary(AMBIGUOUS, reason="several dates without a range word", span=text[a.start:dated[-1].end])

    m = dated[0]
    e = m.expression
    after = _match_word(text, m.end, _AFTER_WORDS, forward=True)
    before = _match_word(text, m.start, _BEFORE_WORDS, forward=False)
    english = next((op for rx, op in _ENGLISH if re.search(rx + r"\s*$", lower_text[:m.start])), None)
    if before and before[0] in ("到", "直到") and after and after[0] == "为止":
        before = None                                          # 到……为止: inclusive
    found = [x[1] for x in (before, after) if x] + ([english] if english else [])
    ops = set(found)
    span = text[max(0, m.start - 4):m.end + 3]
    if AMBIGUOUS in ops:
        word = (after if after and after[1] == AMBIGUOUS else before)[0] if (after and after[1] == AMBIGUOUS) or (before and before[1] == AMBIGUOUS) else "until"
        return Boundary(AMBIGUOUS, reason=f"'{word}' does not say whether {_label(e.start, e.granularity)} itself counts", span=span,
                        target_start=e.start, target_end=e.end, granularity=e.granularity)
    if len(ops) > 1:
        if ops == {ON_OR_AFTER} or ops <= {ON_OR_AFTER, ON}:   # 从11月5日起 / 自11月5日以来
            ops = {ON_OR_AFTER}
        else:
            return Boundary(AMBIGUOUS, reason=f"conflicting relation words {sorted(ops)}", span=span, target_start=e.start, target_end=e.end, granularity=e.granularity)
    op = next(iter(ops)) if ops else ON
    if op in _DAY_SENSITIVE and (include_day or exclude_day):
        if include_day:
            op = {BEFORE: ON_OR_BEFORE, AFTER: ON_OR_AFTER}.get(op, op)
        else:
            op = {ON_OR_BEFORE: BEFORE, ON_OR_AFTER: AFTER}.get(op, op)
    lower, upper = {BEFORE: (None, e.start), ON_OR_BEFORE: (None, e.end), ON: (e.start, e.end), AFTER: (e.end, None), ON_OR_AFTER: (e.start, None)}[op]
    inferred = "（年份按提问时间推断）" if e.resolution_status == "inferred_year" else ""
    return Boundary(RESOLVED, op, lower, upper, e.start, e.end, e.granularity, reason="year inferred" if inferred else "", span=span,
                    description=_describe(op, e.start, e.end, e.granularity) + inferred)


def relation(boundary: Boundary, start: Optional[datetime], end: Optional[datetime] = None) -> str:
    """Where an event lies against the boundary: SATISFIES, VIOLATES or UNKNOWN. ``[start, end)`` is the event's time at its
    precision (a point when ``end`` is None). Only a RESOLVED boundary and a known time can decide."""
    if boundary.status != RESOLVED or start is None:
        return UNKNOWN
    end = end or start + timedelta(microseconds=1)
    lo, hi = boundary.lower, boundary.upper
    if (lo is None or start >= lo) and (hi is None or end <= hi):
        return SATISFIES
    if (lo is not None and end <= lo) or (hi is not None and start >= hi):
        return VIOLATES
    return UNKNOWN


def day_of(t: datetime) -> Tuple[datetime, datetime]:
    """The calendar day around ``t`` (+08:00), for a time known only to the day."""
    local = t.astimezone(CALENDAR_TZ)
    start = local.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1)
