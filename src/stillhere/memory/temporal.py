"""Chinese time words -> ``TimeExpression`` (Day 7 Phase 5B, Temporal Query Engine).

The planner reads every clause with ``find_times`` before it looks for owners and slots, so a date between the
owner and the slot words (``你今年三月初住哪``) is a time, not a gap that makes the fact someone else's. Each
mention becomes a ``TimeExpression``:

* **Calendar intervals** (``absolute_point``, ``absolute_interval``, ``relative_interval``) are half-open,
  ``[start, end)``, in the calendar timezone (``CALENDAR_TZ``, a fixed +08:00 offset: no tz database is needed, so
  Windows without ``tzdata`` reads dates the same way). A day is local midnight to local midnight. Relative words
  are counted from the query's ``as_of`` read in that calendar, never from the machine's clock. An interval that
  runs past ``as_of`` ends at ``as_of`` (memory knows nothing later); one that starts after it is ``future``.
* **A date without its year** (``3月10号``, ``3月初``) takes the latest year in which it does not start after
  ``as_of`` (``inferred_year``).
* **Parts of a month**: ``初``/``上旬`` days 1-10, ``中``/``中旬`` 11-20, ``底``/``末``/``下旬`` 21 to the end.
  **Parts of a year**: ``年初`` January-February, ``年底``/``年末`` November-December. ``上周`` is the previous
  Monday-to-Monday week; ``上上周``, ``上上个月`` and ``大前年`` go back two weeks, two months and three years.
* **Numeric dates** (``2026-03-10``, ``2026/3/14``, ``2026.3``) read like ``2026年3月10号``.
* **A number of units ago** (``三年前``, ``两个月前``, ``两周前``, ``三天前``): the whole calendar year, month,
  Monday-to-Monday week or day that many units before ``as_of``. It is a rough time read as its whole calendar
  unit, so a value that changed inside it shows as segmented or partial, never as one picked value.
* **Open scopes** (``现在``, ``以前``, ``一直``) and **life periods** (``小时候``) have no interval; the planner keeps
  its earlier rules for them.
* **Unresolved**: an impossible date (``2026年2月30号``, ``13月``), a vague time (``前阵子``, ``前段时间``), a relative
  word without ``as_of``, or a future date. Nothing is looked up for it and nothing falls back to the current value.

A month or a month-day without a year is only a time when it comes before the asked fact or is followed by a time
suffix (``那会儿``, ``的时候``): in ``你生日是十月还是十一月`` the months are the values asked about. Years, relative
words and vague times are times wherever they stand. Standard library only.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from typing import List, Optional, Tuple

from stillhere.contracts.planning import TimeExpression, TimeKind

from .search import normalize

CALENDAR_TZ = timezone(timedelta(hours=8), "+08:00")

HISTORICAL_CUES = ("以前", "之前", "原来", "过去", "曾经", "从前", "早先", "当初", "那时候")
CURRENT_CUES = ("现在", "目前", "如今", "当前", "眼下")
# A stretch of time ("一直住北川吗"): what held before is evidence too.
DURATION_CUES = ("一直", "一向", "向来", "从来", "始终", "从小")
# Calendar times before now. Since Phase 5B 去年/前年/上个月 are calendar intervals (find_times); the vague ones
# (前几年, 那年) still read as the open past.
RELATIVE_PAST_CUES = ("去年", "前年", "前几年", "几年前", "上个月", "那年", "当年")
# Life periods: no record places the persona's childhood or school years in time, so none can answer for them.
LIFE_PERIOD_CUES = ("小时候", "童年", "儿时", "幼时", "少年时", "青春期", "年轻时", "年轻的时候", "学生时代") + tuple(
    s + t for s in ("小学", "初中", "高中", "中学", "大学", "上大学", "上学", "读书") for t in ("时", "的时候", "那会儿"))
TIME_WORDS = HISTORICAL_CUES + CURRENT_CUES + DURATION_CUES + RELATIVE_PAST_CUES + LIFE_PERIOD_CUES
# Times nothing on a calendar places.
VAGUE_TIMES = ("前阵子", "前段时间", "那段时间", "有一年", "某一年", "某年", "某天", "那天")

_NUM = "[0-9]{1,2}|[零〇一二两三四五六七八九十]{1,3}"
_YEAR = "[0-9]{4}|[零〇一二三四五六七八九]{4}"
_PART = "上旬|中旬|下旬|初|中|底|末"
_SUFFIX = "(?:那会儿|那会|那时候|那时|的时候|时候|期间|那阵子|那段时间|时)?"
_MONTH_TAIL = f"(?P<month>{_NUM})月份?(?:(?P<day>{_NUM})[号日]|(?P<part>{_PART}))?"
_ABSOLUTE = re.compile(f"(?P<year>{_YEAR})年(?:{_MONTH_TAIL}|(?P<ypart>初|底|末))?(?P<suffix>{_SUFFIX})")
_RELATIVE_YEAR = re.compile(f"(?P<rel>今年|去年|大前年|前年)(?:{_MONTH_TAIL}|年?(?P<ypart>初|底|末))?(?P<suffix>{_SUFFIX})")
_RELATIVE_MONTH = re.compile(f"(?P<rel>上上个?月|上个?月|这个月|本月)(?P<part>{_PART})?(?P<suffix>{_SUFFIX})")
_RELATIVE_WEEK = re.compile(f"(?P<rel>上上周|上上个?星期|上上个?礼拜|上周|上个?星期|上个?礼拜|这周|本周)(?P<suffix>{_SUFFIX})")
_NUMERIC_DATE = re.compile(f"(?P<year>[0-9]{{4}})(?P<sep>[-/.])(?P<month>[0-9]{{1,2}})(?:(?P=sep)(?P<day>[0-9]{{1,2}}))?"
                           f"[号日]?(?P<suffix>{_SUFFIX})")
_AGO = re.compile(f"(?P<n>{_NUM})(?P<unit>个月|个星期|个礼拜|年|周|星期|礼拜|天)(?:以前|之前|前)(?P<suffix>{_SUFFIX})")
_MONTH_DAY = re.compile(f"{_MONTH_TAIL}(?P<suffix>{_SUFFIX})")
_DIGITS = {c: i for i, c in enumerate("零一二三四五六七八九")} | {"〇": 0, "两": 2}
_PART_DAYS = {"初": (1, 11), "上旬": (1, 11), "中": (11, 21), "中旬": (11, 21)}     # 底/末/下旬: 21 to the end


@dataclass(frozen=True)
class TimeMention:
    start: int                 # offsets into the normalized text, the time suffix included
    end: int
    expression: TimeExpression
    anchored: bool             # a time wherever it stands (a year, a relative word, a vague time, a cue word)
    suffixed: bool = False     # followed by 那会儿, 的时候, ...

    def scope_for(self, anchor: Optional[int]) -> bool:
        """Whether this mention sets the time of a clause whose asked fact is named at ``anchor``."""
        return self.anchored or self.suffixed or anchor is None or self.end <= anchor


def _number(text: str) -> Optional[int]:
    if text.isdigit():
        return int(text)
    if "十" in text:
        tens, _, ones = text.partition("十")
        if (tens and tens not in _DIGITS) or (ones and ones not in _DIGITS) or len(tens) > 1 or len(ones) > 1:
            return None
        return (_DIGITS[tens] if tens else 1) * 10 + (_DIGITS[ones] if ones else 0)
    return _DIGITS[text] if len(text) == 1 and text in _DIGITS else None


def _year(text: str) -> int:
    return int(text) if text.isdigit() else int("".join(str(_DIGITS[c]) for c in text))


def _offset(tz: tzinfo) -> str:
    total = int(tz.utcoffset(None).total_seconds()) // 60
    sign = "+" if total >= 0 else "-"
    return f"{sign}{abs(total) // 60:02d}:{abs(total) % 60:02d}"


def _month_start(year: int, month: int, tz: tzinfo) -> datetime:
    while month > 12:
        year, month = year + 1, month - 12
    while month < 1:
        year, month = year - 1, month + 12
    return datetime(year, month, 1, tzinfo=tz)


class _Builder:
    def __init__(self, text: str, as_of: Optional[datetime], tz: tzinfo) -> None:
        self.text, self.as_of, self.tz = text, as_of, tz
        self.local = as_of.astimezone(tz) if as_of is not None else None

    def make(self, kind: TimeKind, granularity: str, status: str = "resolved", start: Optional[datetime] = None,
             end: Optional[datetime] = None) -> TimeExpression:
        if start is not None and self.as_of is not None:
            if start > self.as_of:
                return self.unresolved("future")
            if end > self.as_of:
                end = self.as_of.astimezone(self.tz)
        return TimeExpression(kind=kind, source_text=self.text, as_of=self.as_of, timezone=_offset(self.tz),
                              granularity=granularity, resolution_status=status, start=start, end=end)

    def unresolved(self, status: str = "unresolved") -> TimeExpression:
        return TimeExpression(kind=TimeKind.UNRESOLVED_TIME, source_text=self.text, as_of=self.as_of,
                              timezone=_offset(self.tz), granularity="unknown", resolution_status=status)

    def month_or_part(self, kind: TimeKind, year: int, month_text: str, day_text: Optional[str],
                      part: Optional[str], status: str) -> TimeExpression:
        month = _number(month_text)
        if month is None or not 1 <= month <= 12:
            return self.unresolved("invalid")
        first = _month_start(year, month, self.tz)
        if day_text is not None:
            day = _number(day_text)
            if day is None or not 1 <= day <= calendar.monthrange(year, month)[1]:
                return self.unresolved("invalid")
            start = first.replace(day=day)
            return self.make(TimeKind.ABSOLUTE_POINT if kind is TimeKind.ABSOLUTE_INTERVAL else kind, "day", status,
                             start, start + timedelta(days=1))
        if part is not None:
            lo, hi = _PART_DAYS.get(part, (21, None))
            end = first.replace(day=hi) if hi else _month_start(year, month + 1, self.tz)
            return self.make(kind, "part_of_month", status, first.replace(day=lo), end)
        return self.make(kind, "month", status, first, _month_start(year, month + 1, self.tz))

    def year_or_part(self, kind: TimeKind, year: int, ypart: Optional[str]) -> TimeExpression:
        if ypart == "初":
            return self.make(kind, "part_of_year", "resolved", _month_start(year, 1, self.tz),
                             _month_start(year, 3, self.tz))
        if ypart in ("底", "末"):
            return self.make(kind, "part_of_year", "resolved", _month_start(year, 11, self.tz),
                             _month_start(year + 1, 1, self.tz))
        return self.make(kind, "year", "resolved", _month_start(year, 1, self.tz), _month_start(year + 1, 1, self.tz))


def _absolute(m: re.Match, b: _Builder) -> TimeExpression:
    year = _year(m.group("year"))
    if m.group("month"):
        return b.month_or_part(TimeKind.ABSOLUTE_INTERVAL, year, m.group("month"), m.group("day"), m.group("part"),
                               "resolved")
    return b.year_or_part(TimeKind.ABSOLUTE_INTERVAL, year, m.group("ypart"))


def _relative_year(m: re.Match, b: _Builder) -> TimeExpression:
    if b.local is None:
        return b.unresolved()
    year = b.local.year - {"今年": 0, "去年": 1, "前年": 2, "大前年": 3}[m.group("rel")]
    if m.group("month"):
        return b.month_or_part(TimeKind.RELATIVE_INTERVAL, year, m.group("month"), m.group("day"), m.group("part"),
                               "resolved")
    return b.year_or_part(TimeKind.RELATIVE_INTERVAL, year, m.group("ypart"))


def _relative_month(m: re.Match, b: _Builder) -> TimeExpression:
    if b.local is None:
        return b.unresolved()
    rel = m.group("rel")
    back = 2 if rel.startswith("上上") else 1 if rel.startswith("上") else 0
    first = _month_start(b.local.year, b.local.month - back, b.tz)
    return b.month_or_part(TimeKind.RELATIVE_INTERVAL, first.year, str(first.month), None, m.group("part"),
                           "resolved")


def _relative_week(m: re.Match, b: _Builder) -> TimeExpression:
    if b.local is None:
        return b.unresolved()
    monday = datetime(b.local.year, b.local.month, b.local.day, tzinfo=b.tz) - timedelta(days=b.local.weekday())
    rel = m.group("rel")
    monday -= timedelta(days=14 if rel.startswith("上上") else 7 if rel.startswith("上") else 0)
    return b.make(TimeKind.RELATIVE_INTERVAL, "week", "resolved", monday, monday + timedelta(days=7))


def _numeric_date(m: re.Match, b: _Builder) -> TimeExpression:
    return b.month_or_part(TimeKind.ABSOLUTE_INTERVAL, int(m.group("year")), m.group("month"), m.group("day"), None,
                           "resolved")


def _ago(m: re.Match, b: _Builder) -> TimeExpression:
    if b.local is None:
        return b.unresolved()
    n = _number(m.group("n"))
    if not n:
        return b.unresolved("invalid")
    unit = m.group("unit")
    if unit == "年":
        return b.year_or_part(TimeKind.RELATIVE_INTERVAL, b.local.year - n, None)
    if unit == "个月":
        first = _month_start(b.local.year, b.local.month - n, b.tz)
        return b.month_or_part(TimeKind.RELATIVE_INTERVAL, first.year, str(first.month), None, None, "resolved")
    today = datetime(b.local.year, b.local.month, b.local.day, tzinfo=b.tz)
    if unit == "天":
        day = today - timedelta(days=n)
        return b.make(TimeKind.RELATIVE_INTERVAL, "day", "resolved", day, day + timedelta(days=1))
    monday = today - timedelta(days=b.local.weekday() + 7 * n)
    return b.make(TimeKind.RELATIVE_INTERVAL, "week", "resolved", monday, monday + timedelta(days=7))


def _month_day(m: re.Match, b: _Builder) -> TimeExpression:
    if b.local is None:
        return b.unresolved()
    month = _number(m.group("month"))
    if month is None or not 1 <= month <= 12:
        return b.unresolved("invalid")
    for year in (b.local.year, b.local.year - 1):     # the latest year in which it does not start after as_of
        day = _number(m.group("day")) if m.group("day") else 1
        if day is not None and 1 <= day <= calendar.monthrange(year, month)[1] and \
                datetime(year, month, day, tzinfo=b.tz) <= b.local:
            return b.month_or_part(TimeKind.ABSOLUTE_INTERVAL, year, m.group("month"), m.group("day"),
                                   m.group("part"), "inferred_year")
    return b.month_or_part(TimeKind.ABSOLUTE_INTERVAL, b.local.year - 1, m.group("month"), m.group("day"),
                           m.group("part"), "inferred_year")


_CALENDAR = ((_ABSOLUTE, _absolute, True), (_NUMERIC_DATE, _numeric_date, True), (_RELATIVE_YEAR, _relative_year, True),
             (_RELATIVE_MONTH, _relative_month, True), (_RELATIVE_WEEK, _relative_week, True), (_AGO, _ago, True),
             (_MONTH_DAY, _month_day, False))
_WORDS: Tuple[Tuple[str, TimeKind, str], ...] = tuple(sorted(
    [(w, TimeKind.UNRESOLVED_TIME, "unknown") for w in VAGUE_TIMES]
    + [(w, TimeKind.LIFE_PERIOD, "life_period") for w in LIFE_PERIOD_CUES]
    + [(w, TimeKind.CURRENT, "open") for w in CURRENT_CUES]
    + [(w, TimeKind.HISTORICAL, "open") for w in HISTORICAL_CUES + ("前几年", "几年前", "那年", "当年")]
    + [(w, TimeKind.ALL, "open") for w in DURATION_CUES], key=lambda x: -len(x[0])))


def find_times(text: str, as_of: Optional[datetime], tz: tzinfo = CALENDAR_TZ) -> List[TimeMention]:
    """Every time mention in ``text`` (normalized here), left to right, longest first at each position."""
    text = normalize(text)
    out: List[TimeMention] = []
    pos = 0
    while pos < len(text):
        best = None
        for pattern, build, anchored in _CALENDAR:
            m = pattern.match(text, pos)
            if m is not None and m.end() > pos and (best is None or m.end() > best[0].end()):
                best = (m, build, anchored)
        if best is not None:
            m, build, anchored = best
            out.append(TimeMention(m.start(), m.end(), build(m, _Builder(m.group(), as_of, tz)), anchored,
                                   bool(m.group("suffix"))))
            pos = m.end()
            continue
        word = next((w for w in _WORDS if text.startswith(w[0], pos)), None)
        if word is not None:
            w, kind, granularity = word
            b = _Builder(w, as_of, tz)
            expr = b.unresolved() if kind is TimeKind.UNRESOLVED_TIME else b.make(kind, granularity)
            out.append(TimeMention(pos, pos + len(w), expr, True))
            pos += len(w)
            continue
        pos += 1
    return out


def resolve_time(text: str, as_of: Optional[datetime], tz: tzinfo = CALENDAR_TZ) -> Optional[TimeExpression]:
    """The expression of the first time mention in ``text``, or None."""
    found = find_times(text, as_of, tz)
    return found[0].expression if found else None
