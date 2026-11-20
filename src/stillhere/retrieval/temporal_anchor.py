"""Reads an explicit date out of a question, for the date candidate channel. It reads only the question and the time it was asked.

Calendar reading comes from ``memory.temporal``. This module adds what a search needs on top: ranges, offsets from a named date, and day words relative to
the question. Anything less certain returns no window, and the search then runs exactly as it would have without this module.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from typing import List, Optional, Tuple

from stillhere.memory.search import normalize
from stillhere.memory.temporal import TimeMention, find_times

OK = "OK"
NO_ANCHOR = "NO_ANCHOR"
UNSUPPORTED = "UNSUPPORTED_ANCHOR"
ABSTAIN = "ABSTAIN_PARSE"
INVALID = "INVALID_DATE"
MAX_RANGE_DAYS = 31

_NUM = "[0-9]{1,3}|[零一二两三四五六七八九十]{1,3}"
_DIGITS = {c: i for i, c in enumerate("零一二三四五六七八九")} | {"两": 2}
_RANGE_TAIL = re.compile(f"\\s*(?:至|到|~|～|-|—|－)\\s*(?P<day>{_NUM})[日号]")
_RANGE_JOIN = re.compile(r"^\s*(?:至|到|~|～|-|—|－)\s*$")
_FIXED_OFFSET = {"前一天": -1, "前1天": -1, "后一天": 1, "后1天": 1, "第二天": 1, "次日": 1, "隔天": 1, "翌日": 1}
_FIXED = re.compile("|".join(_FIXED_OFFSET))
_COUNTED = re.compile(f"(?P<n>{_NUM})(?P<unit>天|周|个星期|星期)(?:之|以)?(?P<dir>前|后)")
_ASK_DAYS = {"大前天": -3, "前天": -2, "昨天": -1, "昨日": -1, "昨晚": -1, "今天": 0, "今日": 0, "今早": 0, "今晚": 0}
_ASK_WORD = re.compile("|".join(sorted(_ASK_DAYS, key=len, reverse=True)))
_AFTER_EVENT = re.compile(r"(?:的|之)$")
_VAGUE = re.compile(r"月份|[0-9一二三四五六七八九十]月|上旬|中旬|下旬|月初|月底|月末|年初|年底|年末|去年|今年|前年|上个?月|这个月|本月|上周|上个?星期|这周|本周|最近|近期|前阵子|前段时间|前几天|那天|当天|那时|"
                    r"夏天|春天|秋天|冬天|春节|前一天|后一天|第二天|次日|之前|之后|以前|以后|天前|天后|周前|周后")
_PLAN = re.compile(r"打算|计划|准备|将要|想要|想约")


@dataclass(frozen=True)
class TemporalAnchor:
    parse_status: str
    anchor_type: Optional[str] = None
    start: Optional[datetime] = None
    end: Optional[datetime] = None                # half-open, in the question's calendar
    timezone: Optional[str] = None
    granularity: Optional[str] = None
    confidence: float = 0.0
    source_span: Tuple[str, ...] = ()
    resolution_basis: Optional[str] = None
    query_without_time: Optional[str] = None

    @property
    def usable(self) -> bool:
        return self.parse_status == OK

    def to_dict(self) -> dict:
        return {"parse_status": self.parse_status, "anchor_type": self.anchor_type, "start": self.start.isoformat() if self.start else None, "end": self.end.isoformat() if self.end else None, "timezone": self.timezone, "granularity": self.granularity, "confidence": self.confidence, "source_span": list(self.source_span), "resolution_basis": self.resolution_basis}


def _number(text: str) -> Optional[int]:
    if text.isdigit():
        return int(text)
    if "十" in text:
        tens, _, ones = text.partition("十")
        if (tens and tens not in _DIGITS) or (ones and ones not in _DIGITS):
            return None
        return (_DIGITS[tens] if tens else 1) * 10 + (_DIGITS[ones] if ones else 0)
    return _DIGITS.get(text)


def _calendar(tz: Optional[tzinfo], asked: datetime) -> timezone:
    """The fixed offset of the asking moment in the given zone, because ``memory.temporal`` works in fixed offsets."""
    local = asked.astimezone(tz) if tz is not None else asked
    return timezone(local.utcoffset())


def _day(moment: datetime) -> datetime:
    return moment.replace(hour=0, minute=0, second=0, microsecond=0)


def _anchor(kind: str, start: datetime, days: int, tz: timezone, confidence: float, spans: Tuple[str, ...], basis: str, text: str) -> TemporalAnchor:
    rest = text
    for span in spans:
        rest = rest.replace(span, " ")
    offset = tz.utcoffset(None)
    sign = "+" if offset >= timedelta(0) else "-"
    minutes = abs(int(offset.total_seconds())) // 60
    return TemporalAnchor(OK, kind, start, start + timedelta(days=days), f"{sign}{minutes // 60:02d}:{minutes % 60:02d}", "day" if days == 1 else "days", confidence, spans, basis, rest)


def _offset_after(text: str, position: int) -> Tuple[Optional[int], Optional[str]]:
    """A single offset phrase after the named date, in days. Several different ones make it unclear, so none is used."""
    tail = text[position:]
    found = [(_FIXED_OFFSET[m.group()], m.group()) for m in _FIXED.finditer(tail)]
    for m in _COUNTED.finditer(tail):
        n = _number(m.group("n"))
        if n:
            found.append(((n * (1 if m.group("unit") == "天" else 7)) * (-1 if m.group("dir") == "前" else 1), m.group()))
    if len({x[0] for x in found}) != 1:
        return None, None
    return found[0]


def _day_mentions(mentions: List[TimeMention]) -> List[TimeMention]:
    """Calendar dates only. "Three days ago" is also a day to ``memory.temporal``, but its meaning depends on what comes before it."""
    return [m for m in mentions if m.expression.granularity == "day" and m.expression.start is not None and not _COUNTED.search(m.expression.source_text)]


def parse_anchor(query: str, asked: Optional[datetime], tz: Optional[tzinfo] = None) -> TemporalAnchor:
    """``asked`` is the moment of the question. Without a timezone the question's own offset is the calendar."""
    if asked is None:
        return TemporalAnchor(ABSTAIN)
    calendar = _calendar(tz, asked)
    text = normalize(query)
    mentions = find_times(text, asked, calendar)
    invalid = [m for m in mentions if m.expression.resolution_status == "invalid"]
    days = _day_mentions(mentions)
    if invalid:
        return TemporalAnchor(INVALID)
    if any(m.expression.resolution_status == "future" for m in mentions) and not days:
        return TemporalAnchor(ABSTAIN)
    if days:
        return _from_dates(text, days, mentions, asked, calendar)
    return _from_ask_words(text, asked, calendar)


def _from_dates(text: str, days: List[TimeMention], mentions: List[TimeMention], asked: datetime, tz: timezone) -> TemporalAnchor:
    first = days[0]
    start = first.expression.start
    inferred = first.expression.resolution_status == "inferred_year"
    if inferred and _PLAN.search(text) and start.year < asked.astimezone(tz).year:
        return TemporalAnchor(ABSTAIN)
    kind, confidence, basis = ("MONTH_DAY", 0.9, "latest_year_not_after_question") if inferred else ("FULL_DATE", 1.0, "explicit_date")
    span = text[first.start:first.end]
    if len(days) == 2:
        between = text[first.end:days[1].start]
        if not _RANGE_JOIN.match(between):
            return TemporalAnchor(ABSTAIN) if days[1].expression.start != start else _single(text, first, start, kind, confidence, basis, tz)
        width = (days[1].expression.start - start).days + 1
        return _range(text[first.start:days[1].end], start, width, tz, text, inferred)
    if len(days) > 2:
        return TemporalAnchor(ABSTAIN)
    tail = _RANGE_TAIL.match(text, first.end)
    if tail:
        day = _number(tail.group("day"))
        if day is None:
            return TemporalAnchor(INVALID)
        try:
            last = start.replace(day=day)
        except ValueError:
            return TemporalAnchor(INVALID)
        return _range(text[first.start:tail.end()], start, (last - start).days + 1, tz, text, inferred)
    return _single(text, first, start, kind, confidence, basis, tz)


def _single(text: str, first: TimeMention, start: datetime, kind: str, confidence: float, basis: str, tz: timezone) -> TemporalAnchor:
    span = text[first.start:first.end]
    shift, phrase = _offset_after(text, first.end)
    if shift is not None:
        return _anchor("DATE_OFFSET", start + timedelta(days=shift), 1, tz, min(confidence, 0.9), (span, phrase), f"{basis}; offset {shift} days", text)
    return _anchor(kind, start, 1, tz, confidence, (span,), basis, text)


def _range(span: str, start: datetime, width: int, tz: timezone, text: str, inferred: bool) -> TemporalAnchor:
    if width < 1:
        return TemporalAnchor(ABSTAIN)
    if width > MAX_RANGE_DAYS:
        return TemporalAnchor(UNSUPPORTED)
    return _anchor("DATE_RANGE", start, width, tz, 0.9 if inferred else 1.0, (span,), "explicit_range", text)


def _from_ask_words(text: str, asked: datetime, tz: timezone) -> TemporalAnchor:
    today = _day(asked.astimezone(tz))
    word = _ASK_WORD.search(text)
    if word and not _AFTER_EVENT.search(text[:word.start()]):
        return _anchor("ASK_RELATIVE", today + timedelta(days=_ASK_DAYS[word.group()]), 1, tz, 0.9, (word.group(),), "relative_to_question_date", text)
    counted = _COUNTED.search(text)
    if counted and counted.group("unit") == "天" and counted.group("dir") == "前" and not _AFTER_EVENT.search(text[:counted.start()]):
        n = _number(counted.group("n"))
        if n:
            return _anchor("ASK_RELATIVE", today - timedelta(days=n), 1, tz, 0.9, (counted.group(),), "relative_to_question_date", text)
    return TemporalAnchor(UNSUPPORTED if _VAGUE.search(text) else NO_ANCHOR)
