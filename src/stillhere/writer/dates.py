"""The calendar dates a span of text can mean, counted from the time the message was said (+08:00). Deterministic; no model.

Used to check a candidate's dates, never to invent them: a candidate date is supported only if the span names it. Plans need
future dates, which ``memory.temporal.find_times`` deliberately leaves unresolved (memory knows nothing after ``as_of``), so this
reader covers both directions: 今天/明天/后天/昨天/前天, 下周三/这周五/上周一/周末, 下个月/这个月/上个月, 11月10日/10号,
2026年11月10日 and 2026-11-10. A month without a year takes the year that puts it nearest the message.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Set, Tuple

from stillhere.memory.temporal import CALENDAR_TZ

_NUM = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
WEEKDAY = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6, "1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6}
RELATIVE_DAYS = {"大前天": -3, "前天": -2, "昨天": -1, "昨晚": -1, "昨日": -1, "今天": 0, "今晚": 0, "今早": 0, "今日": 0, "刚才": 0, "刚刚": 0, "现在": 0,
                 "明天": 1, "明早": 1, "明晚": 1, "明日": 1, "后天": 2, "大后天": 3}
DAYPART_CUES = {"早上": ("早上", "早晨", "清晨", "今早", "明早", "一早"), "上午": ("上午",), "中午": ("中午", "午饭"), "下午": ("下午",),
                "晚上": ("晚上", "今晚", "昨晚", "明晚", "晚饭后", "睡前", "夜晚"), "夜里": ("夜里", "半夜", "凌晨")}


def _n(text: str) -> int:
    if text.isdigit():
        return int(text)
    if len(text) == 1:
        return _NUM[text]
    if text.startswith("十"):
        return 10 + (_NUM[text[1]] if len(text) > 1 else 0)
    if text.endswith("十"):
        return _NUM[text[0]] * 10
    return _NUM[text[0]] * 10 + _NUM[text[-1]]


_D = r"[0-9]{1,2}|[一二三四五六七八九十]{1,3}"
_FULL = re.compile(rf"(?P<y>[0-9]{{4}})\s*年\s*(?P<m>{_D})\s*月\s*(?P<d>{_D})\s*[日号]?|(?P<iy>[0-9]{{4}})[-/.](?P<im>[0-9]{{1,2}})[-/.](?P<id>[0-9]{{1,2}})")
_MONTH_DAY = re.compile(rf"(?<![年0-9])(?P<m>{_D})\s*月\s*(?P<d>{_D})\s*[日号]")
_DAY_ONLY = re.compile(rf"(?<![月0-9])(?P<d>{_D})\s*号")
_WEEK = re.compile(r"(?P<rel>下下|下|这|本|上)?(?:个)?(?:周|星期|礼拜)(?P<w>[一二三四五六日天1-7])")
_WEEKEND = re.compile(r"(?P<rel>下|这|本|上)?(?:个)?周末")
_MONTH_REL = re.compile(r"(?P<rel>下|这|本|上)个?月(?!\s*[0-9一二三四五六七八九十]+\s*[日号])")


def _nearest_year(ref: date, month: int, day: int) -> date:
    options = []
    for y in (ref.year - 1, ref.year, ref.year + 1):
        try:
            options.append(date(y, month, day))
        except ValueError:
            pass
    return min(options, key=lambda d: abs((d - ref).days))


def dates_in(text: str, said_at: datetime) -> Tuple[Set[date], Set[Tuple[int, int]]]:
    """(days, months) the text can name. Empty when it names none: then a candidate date has no support in it."""
    ref = said_at.astimezone(CALENDAR_TZ).date()
    days: Set[date] = set()
    months: Set[Tuple[int, int]] = set()
    for m in _FULL.finditer(text):
        try:
            if m.group("y"):
                days.add(date(int(m.group("y")), _n(m.group("m")), _n(m.group("d"))))
            else:
                days.add(date(int(m.group("iy")), int(m.group("im")), int(m.group("id"))))
        except (ValueError, KeyError):
            pass
    for m in _MONTH_DAY.finditer(text):
        try:
            days.add(_nearest_year(ref, _n(m.group("m")), _n(m.group("d"))))
        except (ValueError, KeyError):
            pass
    for m in _DAY_ONLY.finditer(text):
        try:
            days.add(_nearest_year(ref, ref.month, _n(m.group("d"))))
        except (ValueError, KeyError):
            pass
    for word, delta in RELATIVE_DAYS.items():
        if word in text:
            days.add(ref + timedelta(days=delta))
    monday = ref - timedelta(days=ref.weekday())
    for m in _WEEK.finditer(text):
        rel, w = m.group("rel"), WEEKDAY[m.group("w")]
        if rel in ("下",):
            days.add(monday + timedelta(days=7 + w))
        elif rel == "下下":
            days.add(monday + timedelta(days=14 + w))
        elif rel == "上":
            days.add(monday + timedelta(days=-7 + w))
        elif rel in ("这", "本"):
            days.add(monday + timedelta(days=w))
        else:                                   # a bare 周三: this week's, or next week's once this week's has passed
            d = monday + timedelta(days=w)
            days.add(d if d >= ref else d + timedelta(days=7))
    for m in _WEEKEND.finditer(text):
        shift = {"下": 7, "上": -7}.get(m.group("rel"), 0)
        sat = monday + timedelta(days=5 + shift)
        days.update({sat, sat + timedelta(days=1)})
    for m in _MONTH_REL.finditer(text):
        shift = {"下": 1, "上": -1}.get(m.group("rel"), 0)
        y, mo = ref.year, ref.month + shift
        if mo == 13:
            y, mo = y + 1, 1
        if mo == 0:
            y, mo = y - 1, 12
        months.add((y, mo))
    return days, months


def dayparts_in(text: str) -> Set[str]:
    return {part for part, cues in DAYPART_CUES.items() if any(c in text for c in cues)}


def parse_iso(value: str):
    """(date, precision) of a candidate's ISO value: 2026-11-10, 2026-11, 2026-11-10T21:00."""
    value = (value or "").strip()
    try:
        if re.fullmatch(r"\d{4}-\d{2}", value):
            y, m = value.split("-")
            return (int(y), int(m)), "month"
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return date.fromisoformat(value), "day"
        return datetime.fromisoformat(value).date(), "exact"
    except ValueError:
        return None, None


def supported_date(value: str, text: str, said_at: datetime) -> bool:
    got, precision = parse_iso(value)
    if got is None:
        return False
    days, months = dates_in(text, said_at)
    if precision == "month":
        return got in months or any((d.year, d.month) == got for d in days)
    return got in days
