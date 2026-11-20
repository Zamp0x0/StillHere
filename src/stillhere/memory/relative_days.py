"""Relative day and weekday expressions for QUESTIONS (Batch 2.1), anchored on the asking time only, never the machine clock.

``temporal.resolve_time`` (frozen core) reads past-oriented calendar intervals for the memory planner: ``这周`` is the current week cut at the
asking time, and it has no ``明天``, ``这周六``, ``下周六`` or ``周末``. A question about a plan needs those, forward as well as backward:

* calendar: fixed +08:00 (``temporal.CALENDAR_TZ``); a week runs Monday to Sunday (``周日`` / ``周天`` / ``星期天`` = day 7 of its week);
* ``今天 / 明天 / 后天 / 大后天 / 昨天 / 前天``: that day;
* ``这周X / 本周X / 这个星期X``: day X of the asking week (may be before or after the asking day; never "the next X");
* ``下周X / 下个星期X``: day X of the following week. AMBIGUITY (recorded, not hidden): said early in a week, some speakers mean the coming X
  of the same week; the window flags it and ``readings`` gives both, so the caller can search both and say so;
* ``上周X``: day X of the previous week; ``这周末 / 本周末 / 周末`` = Saturday-Sunday of the asking week, ``下周末`` / ``上周末`` the next / previous;
* month and year boundaries follow from date arithmetic (``下周一`` asked on 2026-12-30 is 2027-01-04).

Returns half-open day windows ``[start, end)`` in the calendar timezone.

Batch 2.1 final: fixed-date public holidays (``holiday``). 国庆 / 国庆节 / 十一长假 = 10-01 .. 10-07, 五一 / 劳动节 = 05-01 .. 05-05, 元旦 = 01-01,
with an optional position (第一天 / 第N天 / 最后一天 / 头一天). The occurrence is the one nearest the asking time (asked on 10-13 -> this
October; asked in September -> the coming one). LIMITS (recorded): the State Council sets each year's exact span and make-up workdays, and
this uses the usual span; lunar holidays (春节, 中秋, 端午) and 清明 are not resolved (no lunar calendar here) and fall through to the parser.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

from .temporal import CALENDAR_TZ

DAY = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "日": 7, "天": 7, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7}
_WEEKDAY = re.compile(r"(?P<rel>这个?|本|下个?|上个?)?(?:周|星期|礼拜)(?P<d>[一二三四五六日天1-7])")
_WEEKEND = re.compile(r"(?P<rel>这个?|本|下个?|上个?)?(?:周末)")
_HOLIDAY = re.compile(r"(?P<h>国庆节?|十一长假|十一假期|五一|劳动节|元旦)(?:节|假期|长假|放假)?(?:的)?(?P<pos>第一天|头一天|第[一二三四五六七1-7]天|最后一天)?")
_HOLIDAYS = {"国庆": (10, 1, 7), "十一": (10, 1, 7), "五一": (5, 1, 5), "劳动": (5, 1, 5), "元旦": (1, 1, 1)}
_CN = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7}
_DAYWORD = {"大后天": 3, "后天": 2, "明天": 1, "今天": 0, "昨天": -1, "前天": -2}


@dataclass
class Window:
    start: datetime
    end: datetime
    text: str
    rule: str
    ambiguous: bool = False
    readings: List["Window"] = field(default_factory=list)

    def days(self) -> str:
        last = self.end - timedelta(microseconds=1)
        return self.start.date().isoformat() if last.date() == self.start.date() else f"{self.start.date().isoformat()} ~ {last.date().isoformat()}"


def _day(d: datetime) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=CALENDAR_TZ)


def holiday(question: str, ask_time: datetime) -> Optional[Window]:
    """A fixed-date public holiday (optionally one day of it) as a day window, the occurrence nearest the asking time; None otherwise."""
    m = _HOLIDAY.search(question)
    if not m:
        return None
    month, day, n = next(v for k, v in _HOLIDAYS.items() if m.group("h").startswith(k))
    local = ask_time.astimezone(CALENDAR_TZ)
    starts = [datetime(y, month, day, tzinfo=CALENDAR_TZ) for y in (local.year - 1, local.year, local.year + 1)]
    start = min(starts, key=lambda d: abs((d + timedelta(days=n / 2) - local).total_seconds()))
    pos = m.group("pos")
    if not pos:
        return Window(start, start + timedelta(days=n), m.group(0), f"holiday {m.group('h')} ({n} days, usual span)")
    k = n if pos == "最后一天" else 1 if pos in ("第一天", "头一天") else int(_CN.get(pos[1], pos[1]))
    if k > n:
        return None
    d = start + timedelta(days=k - 1)
    return Window(d, d + timedelta(days=1), m.group(0), f"holiday {m.group('h')} day {k} of {n} (usual span)")


def resolve(question: str, ask_time: datetime) -> Optional[Window]:
    """The first relative day / weekday / weekend expression of ``question`` as a day window, or None."""
    local = ask_time.astimezone(CALENDAR_TZ)
    today = _day(local)
    monday = today - timedelta(days=local.weekday())
    best = None
    for w, n in sorted(_DAYWORD.items(), key=lambda x: -len(x[0])):
        i = question.find(w)
        if i >= 0 and (best is None or i < best[0]):
            d = today + timedelta(days=n)
            best = (i, Window(d, d + timedelta(days=1), w, f"dayword {n:+d}"))
    for m in _WEEKDAY.finditer(question):
        rel = m.group("rel") or "这"
        shift = 7 if rel.startswith("下") else -7 if rel.startswith("上") else 0
        d = monday + timedelta(days=shift + DAY[m.group("d")] - 1)
        win = Window(d, d + timedelta(days=1), m.group(0), f"weekday {'next' if shift > 0 else 'previous' if shift < 0 else 'this'} week (Mon-Sun)")
        if shift > 0:
            same = monday + timedelta(days=DAY[m.group("d")] - 1)
            if same >= today:          # "下周六" said before this week's Saturday: some speakers mean the coming one
                win.ambiguous = True
                win.readings = [Window(d, d + timedelta(days=1), m.group(0), "next-week reading"), Window(same, same + timedelta(days=1), m.group(0), "coming-day reading")]
        if best is None or m.start() < best[0]:
            best = (m.start(), win)
        break
    for m in _WEEKEND.finditer(question):
        rel = m.group("rel") or "这"
        shift = 7 if rel.startswith("下") else -7 if rel.startswith("上") else 0
        sat = monday + timedelta(days=shift + 5)
        if best is None or m.start() < best[0]:
            best = (m.start(), Window(sat, sat + timedelta(days=2), m.group(0), "weekend (Sat-Sun)"))
        break
    if best is None:
        return holiday(question, ask_time)
    return best[1]
