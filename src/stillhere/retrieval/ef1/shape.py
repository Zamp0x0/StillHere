"""Question form from its words only (never a benchmark label), and rule-based sub-questions.

A question can have several flags at once (an ordered list of events is ENUMERATED_LIST + ORDERING). The flags decide the candidate pool size
and which notes the evidence context carries; they never choose an answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from .query import normalize

ENUMERATED_LIST = "ENUMERATED_LIST"
ORDERING = "ORDERING"
PLAN_CHECK = "PLAN_CHECK"
COUNT = "COUNT"
RANGE_SUMMARY = "RANGE_SUMMARY"
INTERVAL = "INTERVAL"
RELATIVE_OFFSET = "RELATIVE_OFFSET"
CAUSAL = "CAUSAL"
STATE_NOW = "STATE_NOW"          # "现在/目前/还…吗": the state valid at the ask time (knowledge update)
SINGLE = "SINGLE"

_ITEM = re.compile(r"(?:^|\n|\s)\s*(\d{1,2})\s*[.、．)）]\s*([^\n]+)")
_ORDER = re.compile(r"按(?:照)?(?:时间)?(?:先后)?顺序|排(?:个|一下)?序|先后顺序|时间顺序|哪个(?:先|在前)|谁先")
_PLAN = re.compile(r"(?:安排|计划|打算|准备|想约|想去|约你)[^？?。]{0,40}(?:怎么样|如何|合适|可以吗|行吗|好吗|行不行|靠谱|听起来|不错吧|没问题吧)|"
                   r"(?:怎么样|合适吗|可以吗|行吗)[？?]?\s*$")
_COUNT = re.compile(r"几次|多少次|次数|几回|多少回|几趟|多少趟|几天|多少天|一共|总共|总计|合计|几件|几个|多少个|多少笔|几笔")
_RANGE = re.compile(r"规律|模式|习惯|变化|趋势|转折|回顾|总结|梳理|全年|这一年|一整年|每个月|每周|上半月|下半月|上半年|下半年|期间|阶段|演变|分别属于|怎样的")
_INTERVAL = re.compile(r"间隔|相隔|隔了|相差|相距|多久|多长时间|持续了")
_OFFSET = re.compile(r"(?:之后|以后|后|之前|以前|前)的?第\s*[\d一二两三四五六七八九十百]+\s*天|[\d一二两三四五六七八九十百]+\s*(?:天|周|个星期|个月)\s*(?:之后|以后|后|之前|以前|前)")
_CAUSAL = re.compile(r"为什么|为何|原因|因为什么|怎么会|导致|由于|是什么让")
_NOW = re.compile(r"现在|目前|如今|当前|最近|还在|还是|是否(?:还|已经)|已经.{0,6}了吗|最新")


@dataclass
class Shape:
    flags: List[str] = field(default_factory=list)
    items: List[str] = field(default_factory=list)          # the enumerated items as written (for sub-questions)

    def has(self, flag: str) -> bool:
        return flag in self.flags

    @property
    def primary(self) -> str:
        for f in (ENUMERATED_LIST, RANGE_SUMMARY, PLAN_CHECK, COUNT, INTERVAL, RELATIVE_OFFSET, ORDERING, CAUSAL, STATE_NOW):
            if f in self.flags:
                return f
        return SINGLE


def list_items(body: str) -> List[str]:
    s = normalize(body)
    items = [(int(n), t.strip()) for n, t in _ITEM.findall(s)]
    # a real list counts 1, 2, 3 ...; stray digits ("5公里") do not
    seq = [t for i, (n, t) in enumerate(items) if n == i + 1]
    return seq if len(seq) >= 2 else []


def detect(body: str) -> Shape:
    s = normalize(body)
    flags: List[str] = []
    items = list_items(body)
    if items:
        flags.append(ENUMERATED_LIST)
    if _ORDER.search(s):
        flags.append(ORDERING)
    if _PLAN.search(s):
        flags.append(PLAN_CHECK)
    if _COUNT.search(s):
        flags.append(COUNT)
    if _RANGE.search(s):
        flags.append(RANGE_SUMMARY)
    if _INTERVAL.search(s):
        flags.append(INTERVAL)
    if _OFFSET.search(s):
        flags.append(RELATIVE_OFFSET)
    if _CAUSAL.search(s):
        flags.append(CAUSAL)
    if _NOW.search(s):
        flags.append(STATE_NOW)
    return Shape(flags or [SINGLE], items)


_SPLIT_AND = re.compile(r"[，,；;。？?]")


def subqueries(body: str, shape: Shape) -> List[str]:
    """Rule-based sub-questions: every enumerated item; for a plan check, the planned activity; clauses of a multi-part question.
    The full body is always searched as well (by the caller), so a sub-question can only add evidence."""
    out: List[str] = []
    if shape.items:
        out += [re.sub(r"^\s*\d+\s*[.、．)）]\s*", "", it).strip() for it in shape.items]
    elif re.search(r"分别|以及|各自|和.{1,12}(?:分别|各)", normalize(body)):
        out += [c.strip() for c in _SPLIT_AND.split(normalize(body)) if len(c.strip()) >= 4]
    return [q for q in dict.fromkeys(out) if q]
