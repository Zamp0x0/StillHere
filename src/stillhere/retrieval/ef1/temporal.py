"""Event-time understanding for EF1: which calendar time does the question ask about (never the time it is asked).

``resolve(body, ask_time)`` reads only the question body (the ask-time prefix is removed by ``query.prepare`` first) and returns a
``TimeIntent``:

* ``mentions``  - every date / month / range / year / quarter written in the body, with the year resolved;
* ``windows``   - the day windows evidence should come from: ``mention`` (a written date or range), ``anchor`` + ``target`` (an offset
                  from a written date: "3月10日…34天后" -> anchor 03-10, target 04-13), ``target_alt`` (the other reading of an ordinal day:
                  "后的第N天" is read as anchor + N, the inclusive reading anchor + N - 1 is kept as a low-weight alternative), ``range``;
* ``lookups``   - offsets from an EVENT with no written date ("投票站开放前5天"): the retriever finds the event first, then computes the window;
* ``intervals`` - day differences between two written dates (for "间隔多少天" questions), computed here so the reader does not do arithmetic;
* ``notes``     - the same facts as short sentences for the evidence context (no answer is inferred).

Year rule: an explicit year wins; 今年/去年/前年/明年 are relative to the ask year; a month-day without a year is read in the ask year, or the
previous year if that date would lie after the ask date and the question does not talk about a plan. Every window is clipped to the ask
date unless it is a plan window (a plan check asks about a future day, evidence still comes only from what is known at the ask time).
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple

from .query import normalize

_CN = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_NUM = r"(?:\d{1,4}|[零〇一二两三四五六七八九十百]{1,6})"
_EN_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m} | {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
_PLAN = re.compile(r"打算|计划|准备|将要|要去|想去|想约|安排|预约|下周|下个月|明天|后天")
_REL_YEAR = {"今年": 0, "去年": -1, "前年": -2, "明年": 1}


def cn_number(text: str) -> Optional[int]:
    """'34' / '三十四' / '十' / '一百零五' / '两' -> int."""
    t = (text or "").strip()
    if not t:
        return None
    if t.isdigit():
        return int(t)
    total, section, num = 0, 0, 0
    for ch in t:
        if ch in _CN:
            num = _CN[ch]
        elif ch == "十":
            section += (num or 1) * 10
            num = 0
        elif ch == "百":
            section += (num or 1) * 100
            num = 0
        else:
            return None
    return total + section + num


@dataclass(frozen=True)
class Mention:
    start: date
    end: date                 # exclusive
    kind: str                 # day | month | half_month | range | year | quarter | week
    text: str
    pos: int
    end_pos: int
    explicit_year: bool


@dataclass(frozen=True)
class Window:
    start: date
    end: date                 # exclusive
    role: str                 # mention | anchor | target | target_alt | range | plan
    why: str
    weight: float = 1.0

    def contains(self, d: date) -> bool:
        return self.start <= d < self.end


@dataclass(frozen=True)
class Lookup:
    phrase: str               # the event the offset counts from
    days: int                 # signed offset (after = +, before = -)
    ordinal: bool             # "第N天" form (the inclusive reading N-1 is kept as an alternative)
    text: str


@dataclass
class TimeIntent:
    ask_date: date
    mentions: List[Mention] = field(default_factory=list)
    windows: List[Window] = field(default_factory=list)
    lookups: List[Lookup] = field(default_factory=list)
    intervals: List[Tuple[date, date, int]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    plan: bool = False

    @property
    def has_time(self) -> bool:
        return bool(self.windows or self.lookups)

    def to_json(self) -> dict:
        return {"ask_date": self.ask_date.isoformat(), "plan": self.plan,
                "mentions": [{"start": m.start.isoformat(), "end": m.end.isoformat(), "kind": m.kind, "text": m.text, "explicit_year": m.explicit_year}
                             for m in self.mentions],
                "windows": [{"start": w.start.isoformat(), "end": w.end.isoformat(), "role": w.role, "why": w.why, "weight": w.weight} for w in self.windows],
                "lookups": [{"phrase": l.phrase, "days": l.days, "ordinal": l.ordinal, "text": l.text} for l in self.lookups],
                "intervals": [[a.isoformat(), b.isoformat(), n] for a, b, n in self.intervals], "notes": list(self.notes)}


# ---------------------------------------------------------------- calendar helpers
def _safe(y: int, m: int, d: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _month_end(y: int, m: int) -> date:
    return date(y + (m == 12), 1 if m == 12 else m + 1, 1)


def add_months(d: date, n: int) -> date:
    y, m = divmod(d.month - 1 + n, 12)
    y, m = d.year + y, m + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def zh(d: date) -> str:
    return f"{d.year}年{d.month}月{d.day}日"


# ---------------------------------------------------------------- mention patterns (on the NFKC text; positions match the body)
_P_ISO = re.compile(r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)")
_P_YMD = re.compile(rf"(?:(\d{{4}})年|(今年|去年|前年|明年))?\s*({_NUM})月({_NUM})[日号]")
_P_EN = re.compile(r"\b(" + "|".join(sorted(_EN_MONTHS, key=len, reverse=True)) + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?\b", re.IGNORECASE)
_P_EN2 = re.compile(r"\b(\d{1,2})\s+(" + "|".join(sorted(_EN_MONTHS, key=len, reverse=True)) + r")\.?(?:\s+(\d{4}))?\b", re.IGNORECASE)
_P_MONTH = re.compile(rf"(?:(\d{{4}})年|(今年|去年|前年|明年))?\s*({_NUM})月(?:份)?(?!\s*{_NUM}\s*[日号])(?P<part>上半月|下半月|上旬|中旬|下旬|初|中|底|末)?")
_P_RANGE_DAY_TAIL = re.compile(rf"\s*(?:至|到|~|～|-|—|－)\s*(?:({_NUM})月)?({_NUM})[日号]")
_P_QUARTER = re.compile(r"(?:(\d{4})年|(今年|去年|前年|明年))?\s*(?:第([一二三四1-4])季度|[Qq]([1-4]))")
_P_YEAR = re.compile(r"(?:(\d{4})年(?!\s*\d{1,2}\s*月)|(今年|去年|前年)(?!\s*\d{1,2}\s*月)(?=.{0,6}(?:全年|一年|这一年|整年|以来|一整年|所有|总共|一共|回顾|总结|里|内|中|期间|哪些|几|多少|做了|经历|发生))|(全年|这一年|一整年|整整一年))")
_P_RELMONTH = re.compile(r"(这个月|本月|上个月|上月|下个月)")


def _year_for(explicit: Optional[str], rel: Optional[str], m: int, d: int, ask: date, plan: bool) -> Tuple[int, bool]:
    if explicit:
        return int(explicit), True
    if rel:
        return ask.year + _REL_YEAR[rel], True
    y = ask.year
    probe = _safe(y, m, min(d, 28)) or date(y, m, 1)
    if probe > ask and not plan:
        y -= 1
    return y, False


def find_mentions(body: str, ask: date, plan: bool = False) -> List[Mention]:
    s = normalize(body)
    out: List[Mention] = []
    taken: List[Tuple[int, int]] = []

    def free(a: int, b: int) -> bool:
        return all(b <= x or a >= y for x, y in taken)

    def add(m: Mention) -> None:
        out.append(m)
        taken.append((m.pos, m.end_pos))

    for r in _P_ISO.finditer(s):
        d = _safe(int(r.group(1)), int(r.group(2)), int(r.group(3)))
        if d and free(r.start(), r.end()):
            add(Mention(d, d + timedelta(days=1), "day", r.group(0), r.start(), r.end(), True))
    for r in _P_YMD.finditer(s):
        if not free(r.start(), r.end()):
            continue
        mo, dd = cn_number(r.group(3)), cn_number(r.group(4))
        if not mo or not dd or not 1 <= mo <= 12:
            continue
        y, ex = _year_for(r.group(1), r.group(2), mo, dd, ask, plan)
        d = _safe(y, mo, dd)
        if d is None:
            continue
        tail = _P_RANGE_DAY_TAIL.match(s, r.end())
        if tail:                                            # "5月10日到12日" / "5月10日至6月2日"
            mo2 = cn_number(tail.group(1)) if tail.group(1) else mo
            d2 = _safe(y + (1 if mo2 and mo2 < mo else 0), mo2 or mo, cn_number(tail.group(2)) or 0)
            if d2 and d2 >= d:
                add(Mention(d, d2 + timedelta(days=1), "range", s[r.start():tail.end()], r.start(), tail.end(), ex))
                continue
        add(Mention(d, d + timedelta(days=1), "day", r.group(0), r.start(), r.end(), ex))
    for pat, order in ((_P_EN, "md"), (_P_EN2, "dm")):
        for r in pat.finditer(s):
            if not free(r.start(), r.end()):
                continue
            mon = _EN_MONTHS[(r.group(1) if order == "md" else r.group(2)).lower()]
            dd = int(r.group(2) if order == "md" else r.group(1))
            y, ex = _year_for(r.group(3), None, mon, dd, ask, plan)
            d = _safe(y, mon, dd)
            if d:
                add(Mention(d, d + timedelta(days=1), "day", r.group(0), r.start(), r.end(), ex))
    # two day mentions joined by 到/至/~ form one range ("3月1日到3月20日", "2025-01-30 至 2025-02-03")
    out.sort(key=lambda m: m.pos)
    merged: List[Mention] = []
    for m in out:
        if merged and merged[-1].kind == "day" and m.kind == "day" and re.fullmatch(r"\s*(?:至|到|~|～|-|—|－|和)\s*", s[merged[-1].end_pos:m.pos] or "x") \
                and s[merged[-1].end_pos:m.pos].strip() != "和" and m.start >= merged[-1].start:
            a = merged.pop()
            merged.append(Mention(a.start, m.end, "range", s[a.pos:m.end_pos], a.pos, m.end_pos, a.explicit_year or m.explicit_year))
        else:
            merged.append(m)
    out = merged
    taken = [(m.pos, m.end_pos) for m in out]
    for r in _P_QUARTER.finditer(s):
        if not free(r.start(), r.end()):
            continue
        q = cn_number(r.group(3)) if r.group(3) else int(r.group(4))
        y = int(r.group(1)) if r.group(1) else ask.year + _REL_YEAR.get(r.group(2) or "今年", 0)
        a = date(y, 3 * (q - 1) + 1, 1)
        add(Mention(a, _month_end(y, 3 * q), "quarter", r.group(0), r.start(), r.end(), bool(r.group(1) or r.group(2))))
    for r in _P_MONTH.finditer(s):
        if not free(r.start(), r.end()):
            continue
        mo = cn_number(r.group(3))
        if not mo or not 1 <= mo <= 12:
            continue
        y, ex = _year_for(r.group(1), r.group(2), mo, 1, ask, plan)
        a, b = date(y, mo, 1), _month_end(y, mo)
        part, kind = r.group("part"), "month"
        if part in ("上半月",):
            b, kind = date(y, mo, 16), "half_month"
        elif part in ("下半月",):
            a, kind = date(y, mo, 16), "half_month"
        elif part in ("上旬", "初"):
            b, kind = date(y, mo, 11), "half_month"
        elif part in ("中旬", "中"):
            a, b, kind = date(y, mo, 11), date(y, mo, 21), "half_month"
        elif part in ("下旬", "底", "末"):
            a, kind = date(y, mo, 21), "half_month"
        add(Mention(a, b, kind, r.group(0), r.start(), r.end(), ex))
    # a bare half / part of a month ("10月上半月和下半月") refers to the month mentioned last before it
    for r in re.finditer(r"(上半月|下半月|上旬|中旬|下旬)", s):
        if not free(r.start(), r.end()):
            continue
        prev = [m for m in out if m.end_pos <= r.start() and m.kind in ("month", "half_month")]
        if not prev:
            continue
        y, mo = prev[-1].start.year, prev[-1].start.month
        a, b = {"上半月": (date(y, mo, 1), date(y, mo, 16)), "下半月": (date(y, mo, 16), _month_end(y, mo)),
                "上旬": (date(y, mo, 1), date(y, mo, 11)), "中旬": (date(y, mo, 11), date(y, mo, 21)),
                "下旬": (date(y, mo, 21), _month_end(y, mo))}[r.group(1)]
        add(Mention(a, b, "half_month", r.group(0), r.start(), r.end(), prev[-1].explicit_year))
    for r in _P_RELMONTH.finditer(s):
        if not free(r.start(), r.end()):
            continue
        base = date(ask.year, ask.month, 1)
        a = add_months(base, {"这个月": 0, "本月": 0, "上个月": -1, "上月": -1, "下个月": 1}[r.group(1)])
        add(Mention(a, _month_end(a.year, a.month), "month", r.group(0), r.start(), r.end(), True))
    for r in _P_YEAR.finditer(s):
        if not free(r.start(), r.end()):
            continue
        if r.group(1):
            y = int(r.group(1))
        elif r.group(2):
            y = ask.year + _REL_YEAR[r.group(2)]
        else:
            y = ask.year
        add(Mention(date(y, 1, 1), date(y + 1, 1, 1), "year", r.group(0), r.start(), r.end(), True))
    return sorted(out, key=lambda m: m.pos)


# ---------------------------------------------------------------- offsets
_UNIT_DAYS = {"天": 1, "日": 1, "周": 7, "星期": 7, "个星期": 7, "礼拜": 7, "个礼拜": 7}
_OFF_ORDINAL = re.compile(rf"(?:之后|以后|后)的?第\s*({_NUM})\s*天")
_OFF_COUNT = re.compile(rf"({_NUM})\s*(天|日|周|个星期|星期|个礼拜|礼拜|个月|月)\s*(?:之后|以后|后|过后)")
_OFF_AFTER_LEAD = re.compile(rf"(?:之后|以后|过了)\s*的?\s*({_NUM})\s*(天|日|周|个星期|星期|个月)(?!\s*(?:之后|以后|后|前|之前))")
_OFF_BEFORE = re.compile(rf"(?:之前|以前|前)的?第?\s*({_NUM})\s*天|({_NUM})\s*(天|日|周|个星期|星期|个月)\s*(?:之前|以前|前)")
_OFF_FIXED = re.compile(r"(第二天|次日|隔天|翌日|后一天|第三天|前一天|前一晚|当天|同一天|当日)")
_FIXED_DAYS = {"第二天": 1, "次日": 1, "隔天": 1, "翌日": 1, "后一天": 1, "第三天": 2, "前一天": -1, "前一晚": -1, "当天": 0, "同一天": 0, "当日": 0}


def _offsets(s: str, start: int, stop: int):
    """Offsets written between ``start`` and ``stop`` (the text after an anchor up to the next mention): [(days, ordinal, text)]."""
    seg = s[start:stop]
    found = []
    for r in _OFF_ORDINAL.finditer(seg):
        n = cn_number(r.group(1))
        if n is not None:
            # "后的第二天" is the idiom for the NEXT day (+1); "后的第N天" for a larger N counts N days on (+N), with N-1 kept as the
            # alternative reading by ``resolve``
            found.append((r.start(), 1 if n == 2 else n, n != 2, r.group(0), "1"))
    for r in _OFF_COUNT.finditer(seg):
        n = cn_number(r.group(1))
        if n is None or any(a <= r.start() < a + len(t) for a, _n, _o, t, _u in found):
            continue
        unit = r.group(2)
        found.append((r.start(), n, False, r.group(0), "m" if unit in ("个月", "月") else str(_UNIT_DAYS.get(unit, 1))))
    for r in _OFF_AFTER_LEAD.finditer(seg):          # "3月10日之后34天" (the after-word before the number)
        n = cn_number(r.group(1))
        if n is None or any(a <= r.start() < a + len(t) for a, _n, _o, t, _u in found):
            continue
        unit = r.group(2)
        found.append((r.start(), n, False, r.group(0), "m" if unit == "个月" else str(_UNIT_DAYS.get(unit, 1))))
    for r in _OFF_BEFORE.finditer(seg):
        if r.group(1):
            n, unit, ordinal = cn_number(r.group(1)), "1", True
        else:
            n, ordinal = cn_number(r.group(2)), False
            unit = "m" if r.group(3) in ("个月",) else str(_UNIT_DAYS.get(r.group(3), 1))
        if n is not None:
            found.append((r.start(), -n, ordinal, r.group(0), unit))
    for r in _OFF_FIXED.finditer(seg):
        if not any(a <= r.start() < a + len(t) for a, _n, _o, t, _u in found):
            found.append((r.start(), _FIXED_DAYS[r.group(1)], False, r.group(0), "1"))
    found.sort()
    return found


def _apply(d: date, n: int, unit: str) -> date:
    if unit == "m":
        return add_months(d, n)
    return d + timedelta(days=n * int(unit))


# ---------------------------------------------------------------- the intent
_INTERVAL = re.compile(r"间隔|相隔|隔了|相差|相距|多少天|几天|多久|多长时间|持续|过了多少|之间")
_ANCHOR_SPLIT = re.compile(r"[，,。；;？?！!]")


def resolve(body: str, ask_time: datetime) -> TimeIntent:
    ask = ask_time.date()
    s = normalize(body)
    plan = bool(_PLAN.search(s))
    ms = find_mentions(body, ask, plan)
    ti = TimeIntent(ask_date=ask, mentions=ms, plan=plan)
    clip = (lambda d: d) if plan else (lambda d: min(d, ask + timedelta(days=1)))
    for i, m in enumerate(ms):
        stop = ms[i + 1].pos if i + 1 < len(ms) else len(s)
        offs = _offsets(s, m.end_pos, stop) if m.kind == "day" else []
        if offs:
            ti.windows.append(Window(m.start, m.end, "anchor", f"{m.text}（锚点）", 0.7))
            for _p, n, ordinal, text, unit in offs:
                t = _apply(m.start, n, unit)
                ti.windows.append(Window(t, t + timedelta(days=1), "target", f"{m.text}{text} = {zh(t)}", 1.0))
                ti.notes.append(f"日期计算：{zh(m.start)}{text}是{zh(t)}" + ("（按“起始日之后再数 N 天”计算）" if ordinal else ""))
                if ordinal and unit == "1" and n != 0:
                    alt = m.start + timedelta(days=n - 1 if n > 0 else n + 1)
                    ti.windows.append(Window(alt, alt + timedelta(days=1), "target_alt", f"{m.text}{text}（含起始日的另一种算法）= {zh(alt)}", 0.4))
                    ti.notes.append(f"另一种算法（把起始日算作第 1 天）：{zh(alt)}")
        else:
            if m.start <= ask or plan:
                ti.windows.append(Window(m.start, clip(m.end) if m.start <= ask else m.end, "range" if m.kind != "day" else "mention", m.text, 1.0))
    # offsets from an event with no written date before them ("投票站开放前5天", "体检后的第3天")
    first = ms[0].pos if ms else len(s)
    for _p, n, ordinal, text, unit in _offsets(s, 0, first):
        clause = _ANCHOR_SPLIT.split(s[:s.find(text, _p) if s.find(text, _p) >= 0 else _p])[-1].strip() if text in s else ""
        clause = re.sub(r"^(我|在|那次|那天|当|自从)", "", clause).strip()
        if len(clause) >= 2 and unit != "m":
            ti.lookups.append(Lookup(clause[-24:], n * int(unit), ordinal, text))
    # intervals between written days (computed, never left to the reader)
    days = [m for m in ms if m.kind == "day"]
    if _INTERVAL.search(s) and len(days) >= 2:
        for a, b in zip(days, days[1:]):
            n = (b.start - a.start).days
            ti.intervals.append((a.start, b.start, n))
            ti.notes.append(f"日期间隔：{zh(a.start)}到{zh(b.start)}相差 {abs(n)} 天（不含起始日；含首尾两天则为 {abs(n) + 1} 天）")
    if plan and any(w.start > ask for w in ti.windows):
        ti.notes.append(f"所问日期在提问时间（{zh(ask)}）之后：证据只能来自提问时已知的记录（已有的日程、计划、周期性安排）。")
    for m in ms:
        if m.kind in ("month", "half_month", "quarter", "year", "range"):
            ti.notes.append(f"时间范围：{m.text} = {zh(m.start)} 至 {zh(m.end - timedelta(days=1))}")
    return ti
