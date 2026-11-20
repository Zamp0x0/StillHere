"""EF1 retrieval pipeline: one call per question, read-only, ``as_of = known_at = ask_time`` in every channel.

    retrieve(api, persona_id, question, ask_time, config, dense=None) -> CandidateSet

``EF1Config.top_k`` (default 40) is the number of evidence units handed on to the context builder for an ordinary question; questions whose
form needs more events (an enumerated list, an ordering, a count, a range summary) use ``expanded_top_k`` / ``range_top_k``. The pool each
channel retrieves (``candidate_k``) is larger than ``top_k``. All sizes are configuration, recorded in the trace, never hidden.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Mapping, Optional

from . import channels as ch
from . import shape as sh
from .fusion import Candidate, fuse
from .query import PreparedQuery, lexical_body, prepare
from .temporal import TimeIntent, Window, resolve, zh

DEFAULT_WEIGHTS = {"msg_bm25": 1.0, "sub_msg": 1.0, "dense": 1.0, "sub_dense": 0.6, "time_window": 0.8, "mem_bm25": 0.5, "mem_struct": 0.3}


@dataclass(frozen=True)
class EF1Config:
    top_k: int = 40                    # evidence units for an ordinary question (the user's decision: 40 by default)
    expanded_top_k: int = 80           # enumerated list / ordering / count / plan check
    range_top_k: int = 160             # range summaries (a month, a half month, a quarter, a year)
    candidate_k: int = 200             # pool per channel
    sub_k: int = 12                    # per sub-question (enumerated items)
    struct_k: int = 20                 # cap of the structured channel
    strategy: str = "rrf"              # "rrf" or one channel name (for comparisons)
    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    use_dense: bool = True             # used only when a dense channel is supplied
    use_memory: bool = True
    use_time: bool = True
    adaptive_k: bool = True
    strip_question_form: bool = False  # lexical channels search ``query.lexical_body`` (question form removed); dense keeps the body

    def k_for(self, shape: sh.Shape, intent: TimeIntent) -> int:
        if not self.adaptive_k:
            return self.top_k
        if shape.has(sh.RANGE_SUMMARY) and any(m.kind in ("month", "half_month", "quarter", "year", "range") for m in intent.mentions):
            return self.range_top_k
        if any(shape.has(f) for f in (sh.ENUMERATED_LIST, sh.ORDERING, sh.COUNT, sh.PLAN_CHECK, sh.RANGE_SUMMARY)):
            return self.expanded_top_k
        return self.top_k


@dataclass
class CandidateSet:
    persona_id: str
    question: str
    query: PreparedQuery
    intent: TimeIntent
    shape: sh.Shape
    subqueries: List[str]
    k: int
    candidates: List[Candidate]                 # the whole fused pool, best first
    channel_notes: Dict[str, str] = field(default_factory=dict)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    channel_results: list = field(default_factory=list, repr=False)     # the raw channel outputs (not serialised; for diagnostics)

    @property
    def top(self) -> List[Candidate]:
        return self.candidates[:self.k]

    def trace(self) -> dict:
        return {"schema": "stillhere.retrieval.ef1.trace/1.0", "body": self.query.body, "prefix_date": self.query.prefix_date.isoformat() if self.query.prefix_date else None,
                "warnings": list(self.query.warnings), "time": self.intent.to_json(), "shape": list(self.shape.flags), "subqueries": list(self.subqueries),
                "k": self.k, "pool": len(self.candidates), "channel_notes": dict(self.channel_notes), "timings_ms": dict(self.timings_ms),
                "config": dict(self.config),
                "top": [{"key": c.key, "score": round(c.score, 6), "ranks": dict(c.ranks), "memories": list(c.memories)} for c in self.top]}


def _resolve_lookups(api, persona_id: str, intent: TimeIntent, at: datetime) -> None:
    """Offsets from an event with no written date: find the event (raw-message bm25, top 1, visible at the ask time), take its date and
    add anchor + target windows. Uncertain by nature: marked as such in the notes."""
    for lk in intent.lookups:
        hits = api.search_messages(persona_id, lk.phrase, as_of=at, known_at=at, k=1)
        if not hits:
            intent.notes.append(f"未能找到“{lk.phrase}”这件事的记录，无法计算“{lk.text}”的日期。")
            continue
        d = hits[0].timestamp.astimezone(at.tzinfo).date()
        t = d + timedelta(days=lk.days)
        intent.windows.append(Window(d, d + timedelta(days=1), "anchor", f"检索到的锚点事件“{lk.phrase}”", 0.6))
        intent.windows.append(Window(t, t + timedelta(days=1), "target", f"锚点 {zh(d)}{lk.text} = {zh(t)}", 0.9))
        intent.notes.append(f"日期计算（锚点来自检索，可能不准）：“{lk.phrase}”记录在{zh(d)}，{lk.text}是{zh(t)}。")


def retrieve(api, persona_id: str, question: str, ask_time: datetime, config: Optional[EF1Config] = None,
             dense: Optional[ch.DenseChannel] = None, units: Optional[ch.UnitIndex] = None) -> CandidateSet:
    cfg = config or EF1Config()
    t0 = time.perf_counter()
    q = prepare(question, ask_time)
    intent = resolve(q.body, ask_time)
    shape = sh.detect(q.body)
    subs = sh.subqueries(q.body, shape)
    timings: Dict[str, float] = {"prepare": (time.perf_counter() - t0) * 1000}
    if intent.lookups and cfg.use_time:
        t = time.perf_counter()
        _resolve_lookups(api, persona_id, intent, ask_time)
        timings["lookups"] = (time.perf_counter() - t) * 1000
    units = units or ch.UnitIndex(api, persona_id)
    lex = lexical_body(q.body) if cfg.strip_question_form else q.body
    results: List[ch.ChannelResult] = []
    results.append(ch.msg_bm25(api, persona_id, lex, ask_time, cfg.candidate_k))
    if cfg.use_memory:
        results.append(ch.mem_bm25(api, persona_id, lex, ask_time, cfg.candidate_k, units))
        results.append(ch.mem_struct(api, persona_id, q.body, ask_time, cfg.struct_k, units))
    if cfg.use_time and intent.windows:
        results.append(ch.time_window(api, persona_id, intent.windows, lex, ask_time, cfg.candidate_k))
    if dense is not None and cfg.use_dense:
        results.append(dense(persona_id, q.body, ask_time, cfg.candidate_k))
    if subs:
        # every sub-question searched on its own; interleaved so each one is represented near the top
        per = [ch.msg_bm25(api, persona_id, s, ask_time, cfg.sub_k, name="sub_msg") for s in subs]
        if dense is not None and cfg.use_dense:
            per_d = [dense(persona_id, s, ask_time, cfg.sub_k) for s in subs]
        else:
            per_d = []
        for name, group in (("sub_msg", per), ("sub_dense", per_d)):
            if not group:
                continue
            merged = ch.ChannelResult(name)
            for i in range(max(len(r.keys) for r in group) if group else 0):
                for r in group:
                    if i < len(r.keys) and r.keys[i] not in merged.scores:
                        merged.keys.append(r.keys[i])
                        merged.scores[r.keys[i]] = 1.0 / (i + 1)
            merged.seconds = sum(r.seconds for r in group)
            results.append(merged)
    for r in results:
        timings[r.name] = round(r.seconds * 1000, 2)
    t = time.perf_counter()
    fused = fuse(results, strategy=cfg.strategy, weights=cfg.weights)
    timings["fusion"] = (time.perf_counter() - t) * 1000
    timings["total"] = (time.perf_counter() - t0) * 1000
    notes = {r.name: (r.note or f"{len(r.keys)} units") for r in results}
    return CandidateSet(persona_id, question, q, intent, shape, subs, cfg.k_for(shape, intent), fused, notes,
                        {k: round(v, 2) for k, v in timings.items()},
                        {**{k: v for k, v in asdict(cfg).items() if k != "weights"}, "weights": dict(cfg.weights)}, results)
