"""Memory P2 (2026-11-19): evidence requirements for multi-hop / temporal questions, one bounded second pass, a read-only temporal view,
and an evidence-coverage note. It WRAPS the unchanged V2 path (``memory_v2.pack_v2``); with P2 off nothing here runs.

Why (root-cause audit of the LifeBench 于晓薇 pilot, 37 wrong V2D answers, ``research/p2_final_memory/P2_ROOT_CAUSE_AUDIT_CN.md``): 23 were
multi-hop evidence gaps that a single retrieval of the whole question cannot close - a target day reached by date arithmetic ("52 天后"),
one item of an enumerated list, a second explicit date, a month-long pattern, the days after an anchor event; 6 were a key record of an
already-found day that was not retrieved. The answer model then abstained on the missing part (correctly, given the evidence).

    plan(question, at)          -> requirements, each a short retrieval sub-query with an optional date window (never an answer)
    pack_p2(api, pid, q, at, cfg, p2cfg, dense, names) -> (FactPack, meta)
        1. the V2 pack of the whole question (exactly the OLD first hop)
        2. ONE second pass: for each requirement (at most ``max_requirements``) a small pack - lexical / dense candidates inside its
           window first, then the time-window channel (records of that window ranked by word overlap) - nothing after the ask time
        3. merge: dedupe by message id, interleave the first hop's anchors with each requirement's best anchor (so a budget cut keeps
           every requirement represented), renumber
        meta["p2"]: the plan, per-requirement coverage, the second-pass trace and timings

Deterministic rules only (no LLM call). Detected forms: date offset (N 天后 / N 天前 / 前一天 / 第二天) from an explicit anchor date,
enumerated items (1. … 2. …), several explicit dates, a month pattern (M 月 … 模式 / 变化 / 哪些 / 每周), the span after an anchor date
(… 后 … 变化 / 调整 / 时间线), an advisory question about an explicit date (… 怎么样 / 合适 / 有空 / 可以吗) and "最近 / 近期". A question
with none of these is left to the first hop alone.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

P2_VERSION = "stillhere.memory.p2/2026-11-19.2"
TZ = timezone(timedelta(hours=8))

FULL_DATE = re.compile(r"(?:(\d{4})\s*年\s*)?(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]")
ISO_DATE = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")
SAME_MONTH_DAY = re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?\s*(?:和|、|与|及|至|到)\s*(\d{1,2})\s*[日号]")
MONTH_ONLY = re.compile(r"(?:(\d{4})\s*年\s*)?(\d{1,2})\s*月(?:份)?(?!\s*\d{1,2}\s*[日号])")
OFFSET_AFTER = re.compile(r"(?:过了|再过|又过了|(?:之?后|以后)的?)\s*(\d{1,3})\s*天|(\d{1,3})\s*天\s*(?:之?后|以后|过后)")
OFFSET_BEFORE = re.compile(r"(?:之前的|以前的|前的)\s*(\d{1,3})\s*天|(\d{1,3})\s*天\s*(?:之?前|以前)")
OFFSET_ONE = re.compile(r"(前一天|前天|后一天|第二天|次日|隔天)")
ITEM = re.compile(r"(?m)^\s*(\d{1,2})\s*[.、．)]\s*(.+?)\s*$")
MONTH_PATTERN_CUE = re.compile(r"模式|变化|新的|哪些|固定|每周|每天|习惯|偏好|发展")
SPAN_CUE = re.compile(r"(?:之后|以后|后)[^。？?]{0,12}(?:变化|调整|时间线|恢复|怎样|如何|发展)")
ADVISORY_CUE = re.compile(r"怎么样|合适吗|合适|可以吗|有空|行吗|安排")
RECENT_CUE = re.compile(r"最近|近期|这段时间")
FEW_DAYS_BEFORE = re.compile(r"几天前|前几天")
ASK_PREFIX = re.compile(r"^\s*[（(]\s*提问时间[:：][^)）]*[)）]\s*")


@dataclass
class P2Config:
    max_requirements: int = 6
    sub_k: int = 3                     # anchors per requirement (each gets +-1 neighbours)
    window_k: int = 5                  # anchors for a date-window requirement
    neighbors: Tuple[int, int] = (1, 1)
    pool: int = 200                    # fused candidates inspected for in-window hits
    recent_days: int = 30
    advisory_days: int = 21
    span_days: int = 7
    few_days: int = 7
    item_pool: int = 40                # candidate pool per channel for an item requirement (no window: the items are short phrases)
    item_lead: int = 3                 # item anchors enter the merge after the first hop's top anchors (the first hop already saw every item)


@dataclass
class Requirement:
    rid: str
    kind: str                          # offset_target | item | date | month | span | advisory | recent | before_anchor | same_day
    label: str                         # shown in the coverage note (question words only)
    query: str
    start: Optional[date] = None
    end: Optional[date] = None         # exclusive
    k: int = 3
    offset_days: Optional[int] = None  # anchor_event: the target day is (the anchor record's day + offset_days)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"] = self.start.isoformat() if self.start else None
        d["end"] = self.end.isoformat() if self.end else None
        return d


def _mk(y: Optional[str], m: str, d: str, at: date) -> Optional[date]:
    try:
        x = date(int(y) if y else at.year, int(m), int(d))
    except ValueError:
        return None
    if not y and x > at:
        try:
            x = date(at.year - 1, int(m), int(d))
        except ValueError:
            return None
    return x


def explicit_dates(text: str, at: date) -> List[Tuple[int, date]]:
    """(position, date) of every explicit calendar date in the question, oldest mention first; "8月9日和10日" gives both days."""
    out: Dict[int, date] = {}
    for m in SAME_MONTH_DAY.finditer(text):
        a, b = _mk(None, m.group(1), m.group(2), at), _mk(None, m.group(1), m.group(3), at)
        if a:
            out[m.start()] = a
        if b:
            out[m.end() - 1] = b
    for m in FULL_DATE.finditer(text):
        d = _mk(m.group(1), m.group(2), m.group(3), at)
        if d and not any(abs(p - m.start()) < 3 for p in out):
            out[m.start()] = d
    for m in ISO_DATE.finditer(text):
        d = _mk(m.group(1), m.group(2), m.group(3), at)
        if d:
            out[m.start()] = d
    return sorted(out.items())


def _strip(text: str) -> str:
    t = ASK_PREFIX.sub("", text or "")
    t = FULL_DATE.sub(" ", t)
    t = ISO_DATE.sub(" ", t)
    t = OFFSET_AFTER.sub(" ", t)
    t = OFFSET_BEFORE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def _label(text: str, n: int = 18) -> str:
    t = re.sub(r"\s+", "", text)
    return t[:n] + ("…" if len(t) > n else "")


def plan(question: str, at: datetime, cfg: Optional[P2Config] = None) -> dict:
    cfg = cfg or P2Config()
    q = ASK_PREFIX.sub("", question or "").strip()
    today = at.astimezone(TZ).date()
    reqs: List[Requirement] = []
    kinds: List[str] = []
    core = _strip(q)

    def add(kind, label, query, start=None, end=None, k=None):
        if start and end:
            end = min(end, today + timedelta(days=1))
            if start >= end:
                return
        if len(reqs) < cfg.max_requirements:
            reqs.append(Requirement(f"R{len(reqs) + 1}", kind, label, query, start, end, k or (cfg.window_k if start else cfg.sub_k)))

    items = [(int(n), t) for n, t in ITEM.findall(q)]
    if len(items) >= 2:
        kinds.append("enumerated_items")
        for n, t in items:
            add("item", f"第{n}项：{_label(t)}", t)
    dates = explicit_dates(q, today)
    off_after, off_before, off_one = OFFSET_AFTER.search(q), OFFSET_BEFORE.search(q), OFFSET_ONE.search(q)
    off = off_after or off_before or off_one
    if off:
        if off_after:
            n = int(off_after.group(1) or off_after.group(2))
        elif off_before:
            n = -int(off_before.group(1) or off_before.group(2))
        else:
            n = -2 if off_one.group(1) == "前天" else (-1 if off_one.group(1) == "前一天" else 1)
        before = [(pos, d) for pos, d in dates if pos < off.start()]
        between = q[before[-1][0]:off.start()] if before else q[:off.start()]
        month_between = [m for m in MONTH_ONLY.finditer(between)]
        if before and not month_between:                     # the anchor is the explicit date right before the offset phrase
            kinds.append("date_offset")
            anchor = before[-1][1]
            target = anchor + timedelta(days=n)
            add("offset_target", f"{anchor.isoformat()} {'+' if n > 0 else ''}{n} 天 = {target.isoformat()}", core, target, target + timedelta(days=1))
            add("date", f"{anchor.isoformat()} 当天", core, anchor, anchor + timedelta(days=1))
        else:                                                # the anchor is an event: find it first (bounded two-hop)
            kinds.append("date_offset_event_anchor")
            seg = q[(before[-1][0] if before else 0):off.start()]
            seg = re.sub(r"^\d{1,2}\s*月\s*\d{1,2}\s*[日号][^，,]*[，,]", "", seg)
            mw = month_between[-1] if month_between else None
            start = end = None
            if mw:
                mo = int(mw.group(2))
                y = int(mw.group(1)) if mw.group(1) else today.year
                start = date(y, mo, 1) if 1 <= mo <= 12 else None
                if start:
                    end = date(start.year + (start.month == 12), start.month % 12 + 1, 1)
            add("anchor_event", f"锚点事件：{_label(_strip(seg))}", _strip(seg) or core, start, end, k=1)
            reqs[-1].offset_days = n if reqs and reqs[-1].kind == "anchor_event" else None
        if dates and kinds[-1] == "date_offset_event_anchor":
            for _, d in dates[:1]:
                add("date", f"{d.isoformat()} 当天", core, d, d + timedelta(days=1))
    elif len({d for _, d in dates}) >= 2 and not items and not off:
        kinds.append("several_dates")
        for _, d in dates:
            add("date", f"{d.isoformat()} 当天", core, d, d + timedelta(days=1))
    elif len(dates) == 1 and not items and not off:
        d = dates[0][1]
        if ADVISORY_CUE.search(q):
            kinds.append("advisory_on_date")
            add("advisory", f"{d.isoformat()} 之前 {cfg.advisory_days} 天的状态", core, d - timedelta(days=cfg.advisory_days), d + timedelta(days=1))
        elif FEW_DAYS_BEFORE.search(q):
            kinds.append("before_anchor")
            add("before_anchor", f"{d.isoformat()} 之前几天", core, d - timedelta(days=cfg.few_days), d)
            add("same_day", f"{d.isoformat()} 当天", core, d, d + timedelta(days=1))
        elif SPAN_CUE.search(q):
            kinds.append("span_after_anchor")
            add("same_day", f"{d.isoformat()} 当天", core, d, d + timedelta(days=1))
            add("span", f"{d.isoformat()} 之后 {cfg.span_days} 天", core, d + timedelta(days=1), d + timedelta(days=1 + cfg.span_days), k=cfg.window_k + 2)
        else:
            kinds.append("single_date")
            add("same_day", f"{d.isoformat()} 当天", core, d, d + timedelta(days=1))
    if not dates and not items:
        mm = [m for m in MONTH_ONLY.finditer(q)]
        if mm and MONTH_PATTERN_CUE.search(q):
            kinds.append("month_pattern")
            for m in mm[:2]:
                y = int(m.group(1)) if m.group(1) else today.year
                mo = int(m.group(2))
                if not 1 <= mo <= 12:
                    continue
                start = date(y, mo, 1)
                if start > today:
                    start = date(y - 1, mo, 1)
                end = date(start.year + (start.month == 12), start.month % 12 + 1, 1)
                add("month", f"{start.year}年{start.month}月", core, start, end, k=cfg.window_k + 3)
        elif RECENT_CUE.search(q):
            kinds.append("recent")
            add("recent", f"最近 {cfg.recent_days} 天", core, today - timedelta(days=cfg.recent_days), today + timedelta(days=1))
    return {"version": P2_VERSION, "kinds": kinds, "requirements": reqs, "simple": not reqs}


# ------------------------------------------------------------------ second pass
def _unit_day(api, pid: str, key: str) -> Optional[date]:
    row = api.store._conn.execute("SELECT timestamp FROM messages WHERE target_persona_id = ? AND (locator = ? OR message_id = ?) LIMIT 1",
                                  (pid, key, key)).fetchone()
    return datetime.fromisoformat(row[0]).date() if row else None


def requirement_pack(api, pid: str, req: Requirement, at: datetime, v2cfg, p2cfg: P2Config, *, dense=None, names=None, cache=None):
    """A small pack for one requirement. Window: fused (lexical + dense) candidates inside it first, then the time-window channel.
    ``cache`` (one per question) shares the global retrieval between requirements with the same query and the unit days."""
    import dataclasses
    from stillhere.context import fact_pack as FP
    from stillhere.retrieval.ef1.channels import time_window
    from stillhere.retrieval.ef1.fusion import Candidate
    from stillhere.retrieval.ef1.pipeline import retrieve
    from stillhere.retrieval.ef1.temporal import Window
    from . import memory_v2 as V2
    sub = V2.V2Config(router=v2cfg.router, k=req.k, date_mode="off", slack_days=0, neighbors=(0, 0), use_dense=v2cfg.use_dense,
                      dense_weight=v2cfg.dense_weight, strip_qform=v2cfg.strip_qform)
    cache = cache if cache is not None else {}
    ecfg = V2.ef1_config(req.query, at, sub, dense is not None)
    if req.start is None:                                    # an item: short phrase, no window -> a smaller pool per channel
        ecfg = dataclasses.replace(ecfg, candidate_k=p2cfg.item_pool)
    ck = ("retrieve", req.query, req.start is None)
    if ck not in cache:
        cache[ck] = retrieve(api, pid, req.query, at, ecfg, dense=dense if ecfg.use_dense else None)
    cs = dataclasses.replace(cache[ck], candidates=list(cache[ck].candidates))
    days = cache.setdefault("days", {})
    if req.start is not None:
        inside = []
        for c in cs.candidates[: p2cfg.pool]:
            if c.key not in days:
                days[c.key] = None if c.key.startswith("mem:") else _unit_day(api, pid, c.key)
            d = days[c.key]
            if d is not None and req.start <= d < req.end:
                inside.append(c)
        tw = time_window(api, pid, [Window(req.start, req.end, "target", "p2 requirement window", 1.0)], req.query, at, k=p2cfg.pool)
        have = {c.key for c in inside}
        for key in tw.keys:
            if key not in have:
                inside.append(Candidate(key, float(tw.scores.get(key, 0.0))))
                have.add(key)
        cs.candidates = inside
    cs.k = min(req.k, len(cs.candidates))
    pk = FP.build(api, cs, names=names)
    V2.expand_pack(api, pk, pid, at, p2cfg.neighbors[0], p2cfg.neighbors[1], names or {})
    return pk


def _late(items, at: datetime) -> List[str]:
    limit = at.astimezone(TZ).isoformat()[:19]
    return [i.eid for i in items if (i.event_time or "")[:19] > limit]


def pack_p2(api, pid: str, question: str, at: datetime, v2cfg, p2cfg: Optional[P2Config] = None, *, dense=None, names=None):
    import time
    from . import memory_v2 as V2
    p2cfg = p2cfg or P2Config()
    t0 = time.perf_counter()
    first, meta = V2.pack_v2(api, pid, question, at, v2cfg, dense=dense, names=names)
    pl = plan(question, at, p2cfg)
    coverage, trace = [], []
    if pl["simple"]:
        meta["p2"] = {"version": P2_VERSION, "kinds": [], "requirements": [], "coverage": [], "second_pass": False,
                      "ms": round((time.perf_counter() - t0) * 1000, 1)}
        return first, meta
    groups: List[List] = []                     # per source: list of anchor groups (items of one rank), best first

    def ranked_groups(pk):
        by: Dict[int, list] = {}
        for it in pk.items:
            by.setdefault(it.rank, []).append(it)
        return [by[r] for r in sorted(by)]
    groups.append(ranked_groups(first))
    seen = {m for it in first.items for m in (it.message_ids or [])}
    reqs = list(pl["requirements"])
    cache: dict = {}
    lead = [0]                                  # merge offset per source (the first hop starts at round 0)
    for req in list(reqs):
        if req.kind == "anchor_event" and req.offset_days is not None:   # hop 1: the anchor event's day; hop 2: the target day
            apk = requirement_pack(api, pid, req, at, v2cfg, p2cfg, dense=dense, names=names, cache=cache)
            days = sorted({(it.event_time or "")[:10] for it in apk.items if it.rank == 1 and it.event_time})
            if days:
                ad = date.fromisoformat(days[0])
                tgt = ad + timedelta(days=req.offset_days)
                if tgt <= at.astimezone(TZ).date() and len(reqs) < p2cfg.max_requirements:
                    reqs.append(Requirement(f"R{len(reqs) + 1}", "offset_target", f"锚点 {ad.isoformat()} {'+' if req.offset_days > 0 else ''}{req.offset_days} 天 = {tgt.isoformat()}",
                                            req.query if False else _strip(question), tgt, tgt + timedelta(days=1), p2cfg.window_k))
    pl["requirements"] = reqs
    for req in reqs:
        t1 = time.perf_counter()
        pk = requirement_pack(api, pid, req, at, v2cfg, p2cfg, dense=dense, names=names, cache=cache)
        late = _late(pk.items, at)
        if late:
            raise AssertionError(f"P2 requirement {req.rid} returned items after the ask time: {late}")
        gs = ranked_groups(pk)
        new_gs = []
        for g in gs:
            fresh = [it for it in g if not set(it.message_ids or []) <= seen]
            for it in fresh:
                seen.update(it.message_ids or [])
            if fresh:
                new_gs.append(fresh)
        hits = sum(len(g) for g in gs)
        coverage.append({"rid": req.rid, "kind": req.kind, "label": req.label, "records_found": hits, "new_records": sum(len(g) for g in new_gs),
                         "covered": hits > 0, "window": [req.start.isoformat() if req.start else None, req.end.isoformat() if req.end else None]})
        trace.append({"rid": req.rid, "candidates": len(pk.items), "ms": round((time.perf_counter() - t1) * 1000, 1)})
        groups.append(new_gs)
        lead.append(p2cfg.item_lead if req.kind == "item" else 0)
    # interleave: first-hop anchor 1, every windowed requirement's best anchor, first-hop anchor 2, ...; item requirements join from round
    # ``item_lead`` on (the first hop searched with every item in the query, so its top anchors stay ahead of them)
    merged, rank = [], 0
    depth = max(len(g) + o for g, o in zip(groups, lead))
    for i in range(depth):
        for src, gs in enumerate(groups):
            j = i - lead[src]
            if 0 <= j < len(gs):
                rank += 1
                for it in gs[j]:
                    it.rank = rank
                    it.channels = dict(it.channels or {}, p2_source=src)
                    merged.append(it)
    merged.sort(key=lambda it: ((it.event_time or "9999"), it.rank))
    for n, it in enumerate(merged, 1):
        it.eid = f"E{n:02d}" if len(merged) < 100 else f"E{n:03d}"
    first.items = merged
    first.notes.append(f"P2: {len(pl['requirements'])} evidence requirements ({', '.join(pl['kinds'])})")
    meta["p2"] = {"version": P2_VERSION, "kinds": pl["kinds"], "requirements": [r.to_dict() for r in pl["requirements"]], "coverage": coverage,
                  "second_pass": True, "trace": trace, "ms": round((time.perf_counter() - t0) * 1000, 1)}
    return first, meta


# ------------------------------------------------------------------ P2-B / P2-C: temporal view and coverage note (read-only)
CAL_START = re.compile(r"日程的开始时间是\s*(\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2})?)")


def event_time_note(text: str, record_date: Optional[str]) -> Optional[str]:
    """A calendar record states when the planned event starts; when that differs from the day the record was written, say so. Only what
    the record itself says (no inference): '日程记录（写于 X）：计划开始 Y；是否实际发生需要其他记录确认'."""
    m = CAL_START.search(text or "")
    if not m:
        return None
    start = m.group(1)
    if record_date and start[:10] == record_date[:10]:
        return f"日程/计划记录：计划开始 {start}"
    return f"日程/计划记录（写于 {record_date or '未知'}）：计划开始 {start}；是否实际发生需要其他记录确认"


def coverage_note(coverage: List[dict]) -> str:
    """The structured coverage the reader sees before answering. Labels come from the question only; never a fact."""
    if not coverage:
        return ""
    lines = ["【证据覆盖】（系统按问题拆出的检索要求；只说明找没找到记录，不是答案）"]
    for c in coverage:
        state = f"找到 {c['records_found']} 条相关记录" if c["covered"] else "没有找到记录"
        lines.append(f"- {c['rid']} {c['label']}：{state}")
    return "\n".join(lines)
