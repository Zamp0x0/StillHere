"""M3.1 reflection: R2's bounded, deterministic reflection on top of the B2 first hop, with an evidence context the reader can use directly.

Behind ``MEMORY_REFLECTION_M31_ENABLED`` (off: R2 is unchanged). Read-only, no loops, never calls a model; the reader is the same locked one.
What it keeps from R2: the rule router, at most two extra retrievals (date hop, slot-history hop), statuses with valid / recorded times,
``known_at`` = the asking time everywhere, the deterministic claim check and the safe abstention fallback.

What M3.1 adds (every line is derived from stored records only, never from the question's expected answer):

* evidence UNITS (B2): an event is shown as one line with its actor, its description(s), its status sequence and its dates, so a reader
  sees "planned 09-20 -> rescheduled 09-24 -> completed" together instead of loose rows;
* attribution: each unit is labelled with whose fact it is (本人 / person X); when the question is about the persona and a unit is about
  someone else, the brief says so (a friend's trip is not the persona's trip);
* negation and preference: ``dislikes`` / negated statements are shown as negative facts, so "do I like X?" can be answered "no";
* multi-part questions: the brief asks for each part and says which parts have no record;
* counting questions (几次 / 多少次 / 几回): the matching events of the window are listed with dates and status, so the reader counts
  records instead of guessing; the brief does not compute the number;
* unknown: no evidence unit at all -> the brief says there is no record (the claim check then requires an abstention);
* claim check (R2's) plus: an answer that is not an abstention needs at least one evidence unit.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Mapping, Optional

from . import reflection as r2
from . import retrieval_b2 as b2

FLAG = "MEMORY_REFLECTION_M31_ENABLED"
COUNT = re.compile(r"几次|多少次|几回|多少回|次数")
MULTI = re.compile(r"分别|以及|和.*(分别|各|都)|，.*(什么|哪|几|吗)")
SELF_Q = re.compile(r"我")
STATUS_ZH = {"PLANNED": "计划中", "SCHEDULED": "已预约", "COMPLETED": "已完成", "CANCELLED": "已取消", "IN_PROGRESS": "进行中", "POSTPONED": "已推迟"}
NEG_ATTRS = ("dislikes",)


def enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    return (env if env is not None else os.environ).get(FLAG, "").strip().lower() in ("1", "true", "yes", "on")


@dataclass
class M31:
    route: r2.Route
    evidence: b2.Evidence
    context: str = ""
    brief: str = ""
    extra_retrievals: int = 0
    evidence_ids: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    ask_time: Optional[datetime] = None
    windows: List[tuple] = field(default_factory=list)


def _owner(subject: str, rs) -> str:
    if subject.startswith("person:"):
        return subject.split(":", 1)[1]
    if subject.startswith("event:"):
        acts = [r.value for r in rs if r.attribute == "event.actor"]
        if acts and acts[-1].startswith("person:"):
            return acts[-1].split(":", 1)[1]
    return "本人"


def _zh_date(v: str) -> str:
    """2026-10-12 -> 10月12日（2026-10-12）; a time part is kept."""
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})(.*)", v or "")
    if not m:
        return v or ""
    return f"{int(m.group(2))}月{int(m.group(3))}日{m.group(4)}（{m.group(1)}-{m.group(2)}-{m.group(3)}）"


def _event_line(subject: str, rs, ask: datetime) -> str:
    rs = sorted(rs, key=lambda r: (r.recorded_at, r.valid_from or r.recorded_at, r.memory_id))
    descs = []
    for r in rs:
        if r.attribute == "event.description" and r.value not in descs:
            descs.append(r.value)
    sts = [r for r in rs if r.attribute == "event.status"]
    statuses = [f"{STATUS_ZH.get(r.value, r.value)}（{r2._status(r, ask)}；记录于 {r2._fmt(r.recorded_at)}）" for r in sts]
    current = [r for r in sts if r2._status(r, ask) == "当前"]
    planned = [f"{_zh_date(r.value)}（{r2._status(r, ask)}）" for r in rs if r.attribute == "event.planned_time"]
    done = sorted({r.value for r in rs if r.attribute == "event.time"})
    parts = ["事件：" + " / ".join(descs or ["（无描述）"]), f"当事人：{_owner(subject, rs)}"]
    if current:
        parts.append("当前状态：" + STATUS_ZH.get(current[-1].value, current[-1].value))
    elif not sts:      # Batch 2.1: say what the record does NOT hold, so "was it cancelled?" can be answered from it
        parts.append("当前状态：未记录（没有取消、改期或完成的记录）")
    if len(statuses) > 1:
        parts.append("状态记录：" + " → ".join(statuses))
    if planned:
        parts.append("计划时间：" + "，".join(planned))
    if done:
        parts.append("发生时间：" + "，".join(_zh_date(d) for d in done))
    return "；".join(parts)


def _fact_line(r, ask: datetime) -> str:
    owner = r.subject.split(":", 1)[1] if r.subject.startswith("person:") else "本人"
    neg = "【否定/不喜欢】" if r.attribute in NEG_ATTRS else ""
    return (f"{neg}{r.statement}（关于：{owner}；{r2._status(r, ask)}；有效 {r2._fmt(r.valid_from)} ~ {r2._fmt(r.valid_to) if r.valid_to else '现在'}；"
            f"记录于 {r2._fmt(r.recorded_at)}）")


def render(ev: b2.Evidence, ask: datetime) -> str:
    by_id = {r.memory_id: r for r in ev.records}
    lines = []
    for i, (key, ids) in enumerate(ev.units, 1):
        rs = [by_id[m] for m in ids if m in by_id]
        if not rs:
            continue
        lines.append(f"[{i}] " + (_event_line(key, rs, ask) if key.startswith("event:") else _fact_line(rs[0], ask)))
    return "\n".join(lines) or "（没有检索到记录）"


def reflect(api, persona_id: str, question: str, ask_time: datetime, k: int = 10) -> M31:
    rt = r2.route(question, ask_time)
    ev = b2.retrieve(api, persona_id, question, ask_time, k=k)
    out = M31(rt, ev, ask_time=ask_time)
    out.evidence_ids = [r.memory_id for r in ev.records]
    out.context = render(ev, ask_time)
    lines: List[str] = []
    by_id = {r.memory_id: r for r in ev.records}
    units = [(key, [by_id[m] for m in ids if m in by_id]) for key, ids in ev.units]
    if not units:
        lines.append("- 没有找到与问题相关的记录：应回答无法确定，不要猜。")
        out.notes.append("no_evidence")
    # attribution
    named = b2.people_in(question, ev.records)
    if SELF_Q.search(question):
        others = sorted({_owner(key, rs) for key, rs in units if _owner(key, rs) != "本人"} - set(named))
        if others:
            lines.append(f"- 注意：证据里有关于他人（{'、'.join(others)}）的记录，那是他们的事，不是你本人的；问题问的是你本人。")
            out.notes.append("attribution")
    # negation / preference
    if any(r.attribute in NEG_ATTRS for _k, rs in units for r in rs):
        lines.append("- 标为【否定/不喜欢】的记录表示本人不喜欢或不吃/不喝该事物，可以据此作否定回答。")
        out.notes.append("negation")
    # a date the question names with no event on that date: say so (a "did I ... on <date>?" question can then be answered from the records)
    # Batch 2.1 (bug A): the note names exactly the windows B2 searched (relative weekdays included; a date without a year in both readings,
    # the asking-year reading first), never the parser's first reading alone
    wins = b2.query_windows(question, ask_time)
    out.windows = [(ws.date().isoformat(), (we_ - timedelta(microseconds=1)).date().isoformat(), rule) for ws, we_, rule in wins]
    if any(r in ("next-week reading", "coming-day reading") for _a, _b, r in out.windows):
        shown = "；".join(f"{a_}（{'下一周的那天' if r == 'next-week reading' else '本周即将到来的那天'}）" for a_, _b, r in out.windows)
        lines.append(f"- 问题里的日期有两种理解：{shown}。两种都查了；回答时说明按哪一种理解。")
        out.notes.append("ambiguous_date")
    if wins and "date" not in ev.channels:
        ordered = sorted(out.windows, key=lambda w: (not w[0].startswith(str(ask_time.year)), w[0]))
        shown = "、".join(a_ if a_ == b_ else f"{a_} ~ {b_}" for a_, b_, _r in ordered)
        undated = [str(i) for i, (key, rs) in enumerate(units, 1) if key.startswith("event:") and not any(r.attribute in ("event.planned_time", "event.time") for r in rs)]
        if undated:   # Batch 2.1 (c67): an event with no stored date may still be the one asked about; do not imply nothing happened
            lines.append(f"- 所问日期（{shown}）内没有带日期的事件记录；证据里没有记录日期的事件（编号 {'、'.join(undated)}）是否就是所问的事，请按内容判断。")
            out.notes.append("no_dated_event_on_date")
        else:
            lines.append(f"- 所问日期（{shown}）内没有任何事件记录。")
            out.notes.append("no_event_on_date")
    # Batch 2.1 last fix: a corrected-away value shown next to its correction. Without this note (and with no other note) the brief was empty,
    # so the reader saw "已更正" only inside the record line and listed both values ("你养了两只鹦鹉，豆豆和球球", b21g-confirm-7 g01 / g06 / g11)
    facts = [(i, rs[0]) for i, (key, rs) in enumerate(units, 1) if rs and not key.startswith("event:")]
    for i, r in facts:
        if not r2._status(r, ask_time).startswith("已更正"):
            continue
        now = [(j, x) for j, x in facts if j != i and x.subject == r.subject and x.attribute == r.attribute and r2._status(x, ask_time) == "当前"
               and (x.supersedes == r.memory_id or getattr(r, "superseded_by", None) == x.memory_id)]
        if now:
            j, x = now[0]
            lines.append(f"- 编号 {i} 的“{r.value}”已被更正为编号 {j} 的“{x.value}”：回答现在的情况只用更正后的值，不要把更正前的值当作现在的事实。")
        else:
            lines.append(f"- 编号 {i} 的“{r.value}”已被更正（原记录有误），不是现在的事实。")
        if "correction" not in out.notes:
            out.notes.append("correction")
    # counting
    if COUNT.search(question):
        evs = [(key, rs) for key, rs in units if key.startswith("event:")]
        if evs:
            lines.append("- 计数题：请逐条核对上面的事件，只数状态为已完成、发生时间在所问时间范围内、且是同一类活动的事件；不要把计划中或已取消的算进去。")
            out.notes.append("count")
    # multi-part
    if MULTI.search(question):
        lines.append("- 问题包含多个部分：逐一回答；哪一部分没有记录就说明没有记录。")
        out.notes.append("multi")
    # R2 hops on the B2 evidence (date hop is already part of B2; the slot-history hop shows old values, corrections, retractions)
    if rt.complex:
        refl = r2.reflect(api, persona_id, question, ask_time, [r for r in ev.records if not r.subject.startswith("event:")])
        out.extra_retrievals = refl.extra_retrievals
        out.evidence_ids += [m for m in refl.evidence_ids if m not in out.evidence_ids]
        # event histories are already in the event units; keep the fact-slot histories (old values, corrections, retractions) and R2's
        # "no record at the asked time" note
        hist = [l for l in refl.brief.splitlines() if l.startswith("- ") and not l.startswith("- event:")]
        lines += hist
    if lines:
        out.brief = ("【证据整理（系统根据记录生成）】\n" + "\n".join(lines) +
                     "\n只用标为“当前”的值回答现在的情况；问过去某个时间时用当时有效的值；已撤回、已更正、已取消的不要当作事实。")
    return out


def allowed_numbers_text(m: M31, question: str = "") -> str:
    """Batch 2.1 (bug B): text whose numbers an answer may repeat without a memory record behind them: the question itself, the asking date
    (year, month, day, padded and unpadded) and the dates of the windows the question names. Only numbers the model introduces beyond these
    and the evidence are checked."""
    parts = [question]
    if m.ask_time is not None:
        a = m.ask_time
        parts += [str(a.year), f"{a.month}", f"{a.month:02d}", f"{a.day}", f"{a.day:02d}", a.date().isoformat()]
    for lo, hi, _r in m.windows:
        for d in (lo, hi):
            y, mo, dd = d.split("-")
            parts += [d, y, str(int(mo)), mo, str(int(dd)), dd]
    return " ".join(parts)


def verify(answer: str, abstained: bool, m: M31, question: str = "") -> Dict:
    # numbers the question / asking date / named windows contain may be repeated in the answer (they are not claims about memory)
    check = r2.verify(answer, abstained, m.context + "\n" + m.brief + "\n" + allowed_numbers_text(m, question), None)
    if not abstained and "no_evidence" in m.notes and not r2.UNCERTAIN.search(answer):
        check["problems"].append("answer_without_evidence")
        check["ok"] = False
    return check
