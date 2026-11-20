"""StillHere Context Orchestrator V1: one deterministic assembly of everything a reader sees for one question.

    question
      -> persona / time scope (MemoryAPI.resolve_times: as_of alone means known_at = as_of)
      -> temporal boundary of the question (boundary.parse_boundary)
      -> budget plan (budget.plan: fixed parts first, then sections by profile)
      -> pinned records -> working context -> recent conversation (whole messages, newest first, earlier corrections kept)
      -> long-term evidence: ranked hits from the existing retriever / reranker -> units (whole rounds) -> validation -> dedupe
         -> selection in rank order: whole unit, else an explicit CONTEXT_OVERFLOW part, else excluded with its reason
      -> integrity check (leaks, ids, budget) -> assembled text, provenance map, trace

It reads memory only through ``MemoryAPI`` and writes nothing. It does not retrieve: hits come in ranked, so the V3.2 pool and
its reranker stay exactly as they are. No model is called.
"""
from __future__ import annotations

import time
from typing import Dict, List, Mapping, Optional, Sequence

from stillhere.contracts import MemoryStatus

import stillhere.context.boundary as bd
import stillhere.context.budget as bg
import stillhere.context.compaction as cp
import stillhere.context.render as rd
import stillhere.context.units as un
import stillhere.context.validation as va
from .attribution import round_direction
from .contracts import (BUDGET_INSUFFICIENT, CONTEXT_OVERFLOW, DERIVED_EXCLUDED, EVIDENCE_OVER_BUDGET, LATER_MESSAGES_HIDDEN, OK, ORDER_ANOMALY,
                        RECENT_TRIMMED, SCOPE_ERROR, TEMPORAL_BOUNDARY_AMBIGUOUS, TOP_EVIDENCE_EXCLUDED, ContextRequest, ContextResult)

VERSION = "context-orchestrator-v1"
TRUNCATION_POLICY = ("whole units in rank order; a unit that does not fit is shown in part only with CONTEXT_OVERFLOW (hits, final message, "
                     "state changes kept; every omitted message listed), else excluded with reason; fixed parts never dropped")
CORRECTION_CUES = ("记错", "更正", "纠正", "改正", "写错", "弄错", "搞错", "记混", "不是", "其实", "改成")


class ContextOrchestrator:
    def __init__(self, api, *, speakers: Optional[Mapping[str, str]] = None, counter: Optional[bg.TokenCounter] = None,
                 round_types=un.ROUND_SOURCE_TYPES) -> None:
        self.api = api
        self.speakers = dict(speakers or {})
        self.count = counter or bg.TokenCounter()
        self.round_types = round_types

    # ------------------------------------------------------------------ helpers

    def _text(self, sections: Dict[str, List[str]]) -> str:
        return rd.join_sections([(n, sections[n]) for n in ("pinned", "working", "recent", "boundary", "derived", "evidence")])

    def _fits(self, fixed: int, total: int, sections: Dict[str, List[str]]) -> bool:
        return fixed + self.count(self._text(sections)) <= total

    def _section_cost(self, sections: Dict[str, List[str]], name: str) -> int:
        return self.count("\n".join(sections[name])) if sections[name] else 0

    # ------------------------------------------------------------------ assembly

    def assemble(self, req: ContextRequest) -> ContextResult:
        started = time.perf_counter()
        api, persona = self.api, req.persona_id
        as_of, known_at = api.resolve_times(req.as_of, req.known_at)
        trace: List[dict] = [{"step": "scope", "persona_id": persona, "as_of": as_of.isoformat(), "known_at": known_at.isoformat(), "version": VERSION}]
        warnings: List[str] = []
        excluded: Dict[str, str] = {}
        provenance: Dict[str, dict] = {}
        omitted: Dict[str, List[str]] = {}
        boundary = bd.parse_boundary(req.question, as_of)
        trace.append({"step": "temporal_boundary", **{k: v for k, v in boundary.to_dict().items() if k != "span"}})
        fixed = req.instruction_tokens + self.count(req.question)
        profile = bg.profile_of(req.question, has_boundary=boundary.status == bd.RESOLVED, has_conversation=bool(req.conversation_id))
        budget = bg.plan(req.input_token_budget, req.output_token_reserve, fixed, profile)
        trace.append({"step": "budget", **budget.to_dict()})
        sections: Dict[str, List[str]] = {n: [] for n in ("pinned", "working", "recent", "boundary", "derived", "evidence")}

        def result(status: str) -> ContextResult:
            text = self._text(sections)
            used = fixed + self.count(text)
            included = [m for p in provenance.values() for m in p.get("message_ids", [])]
            return ContextResult(status, text, [p["key"] for c, p in provenance.items() if p["section"] == "evidence"], list(dict.fromkeys(included)), excluded,
                                 provenance, used, {**budget.to_dict(), "used": used, "remaining": budget.input_total - used}, self.count.mode, TRUNCATION_POLICY,
                                 list(dict.fromkeys(warnings)), trace, boundary.to_dict(), omitted, round((time.perf_counter() - started) * 1000, 3))

        if fixed > req.input_token_budget:
            warnings.append(BUDGET_INSUFFICIENT)
            trace.append({"step": "refused", "reason": "instructions and question exceed the input budget"})
            return result(BUDGET_INSUFFICIENT)
        total = req.input_token_budget

        # pinned records: confirmed, current at as_of, believed at known_at
        for n, mid in enumerate(req.pinned_memory_ids, 1):
            record = api.get(persona, mid, known_at=known_at)
            if record is None or record.status is not MemoryStatus.ACTIVE or (record.valid_to and record.valid_to <= as_of) or (record.valid_from and record.valid_from > as_of):
                excluded[mid] = "pinned_not_current"
                continue
            cid = f"P{n:02d}"
            sections["pinned"].append(f"[{cid}] {record.statement}")
            if self._section_cost(sections, "pinned") > budget.cap("pinned") or not self._fits(fixed, total, sections):
                sections["pinned"].pop()
                excluded[mid] = "budget:pinned"
                continue
            provenance[cid] = {"key": mid, "section": "pinned", "memory_ids": [mid], "source_ids": list(dict.fromkeys(p.source_id for p in record.provenance)), "complete": True}

        # working context: session only
        for n, w in enumerate(req.working, 1):
            cid = f"W{n:02d}"
            sections["working"].append(f"[{cid}] {w.text}")
            if self._section_cost(sections, "working") > budget.cap("working") or not self._fits(fixed, total, sections):
                sections["working"].pop()
                excluded[w.key] = "budget:working"
                continue
            provenance[cid] = {"key": w.key, "section": "working", "source_ids": list(w.source_ids), "persisted": False, "complete": True}

        # recent conversation: newest whole messages first; the latest is guaranteed; earlier corrections pulled back in
        recent_ids: set = set()
        if req.conversation_id:
            raw, hidden = api.conversation(persona, req.conversation_id, as_of=as_of, known_at=known_at)
            participants = [m.speaker_id for m in raw]
            msgs = [un.UnitMessage(m.message_id, i, m.speaker_id, *_dir(m.speaker_id, participants, persona, self.speakers), m.timestamp, m.recorded_at, m.content,
                                   m.source_id, m.source_type, m.locator, m.target_persona_id) for i, m in enumerate(raw, 1)]
            msgs = [m for m in msgs if not va.message_reason(api, persona, m, as_of, known_at)]
            if req.recent_max_turns is not None:
                msgs = msgs[-req.recent_max_turns:] if req.recent_max_turns > 0 else []
            chosen: List[un.UnitMessage] = []
            for i, m in enumerate(reversed(msgs)):
                trial = sorted(chosen + [m], key=lambda x: x.position)
                sections["recent"] = [rd.recent_line(f"R{j:02d}", x) for j, x in enumerate(trial, 1)]
                within = self._section_cost(sections, "recent") <= budget.cap("recent") or i == 0
                if within and self._fits(fixed, total, sections):
                    chosen = trial
                else:
                    break
            for m in msgs:                         # earlier corrections the newest-first pass did not reach
                if m in chosen or not any(c in m.content for c in CORRECTION_CUES):
                    continue
                trial = sorted(chosen + [m], key=lambda x: x.position)
                sections["recent"] = [rd.recent_line(f"R{j:02d}", x) for j, x in enumerate(trial, 1)]
                if self._fits(fixed, total, sections):        # a correction is not cut for the section cap, only for the total
                    chosen = trial
            sections["recent"] = [rd.recent_line(f"R{j:02d}", x) for j, x in enumerate(chosen, 1)]
            for j, m in enumerate(chosen, 1):
                provenance[f"R{j:02d}"] = {"key": m.message_id, "section": "recent", "message_ids": [m.message_id], "source_ids": [m.source_id], "complete": True}
            recent_ids = {m.message_id for m in chosen}
            if len(chosen) < len(msgs):
                warnings.append(RECENT_TRIMMED)
            trace.append({"step": "recent", "conversation_id": req.conversation_id, "visible": len(msgs), "shown": len(chosen), "hidden_later": hidden})

        # temporal boundary line
        if boundary.status == bd.RESOLVED:
            sections["boundary"] = [f"问题的时间范围：{boundary.description}（按 +08:00 日历）"]
        elif boundary.status in (bd.AMBIGUOUS, bd.UNKNOWN):
            sections["boundary"] = ["问题的时间范围不明确：问题里的时间说法无法确定是否包含该日当天，或无法定位到日历日期"]
            warnings.append(TEMPORAL_BOUNDARY_AMBIGUOUS)
        if sections["boundary"] and not self._fits(fixed, total, sections):
            sections["boundary"] = []

        # long-term evidence
        keys = list(req.hit_keys) or [h.memory_id or h.message_id for h in req.hits]
        units, dropped, utrace = un.expand(api, persona, req.hits, keys, as_of, known_at, self.speakers, self.round_types)
        excluded.update(dropped)
        trace += utrace
        checked = []
        for u in units:
            reason = va.check_unit(api, persona, u, as_of, known_at)
            if reason:
                excluded[u.key] = reason
            else:
                checked.append(u)
        checked, dup = cp.dedupe_units(checked, recent_ids)
        excluded.update(dup)
        ranked = checked[:req.top_k]
        for u in checked[req.top_k:]:
            excluded[u.key] = "beyond_top_k"
        first_rank_key = units[0].key if units else None
        n = 0
        partial: List[tuple] = []                  # (index in evidence, context id, unit): rounds shown in part, extended at the end
        for u in ranked:
            cid = f"E{n + 1:02d}"
            if u.hidden_later:
                warnings.append(LATER_MESSAGES_HIDDEN)
            if u.order_anomaly:
                warnings.append(ORDER_ANOMALY)
            sections["evidence"].append(rd.render_unit(cid, u))
            if self._fits(fixed, total, sections):
                shown = u.messages
            else:
                sections["evidence"].pop()
                shown = None
                if u.kind == "round":
                    def fits(sub, cid=cid, u=u):
                        sections["evidence"].append(rd.render_unit(cid, u, sub))
                        ok = self._fits(fixed, total, sections)
                        sections["evidence"].pop()
                        return ok
                    shown = cp.overflow_selection(u, fits)
                if shown is None:
                    excluded[u.key] = "budget:evidence"
                    warnings.append(EVIDENCE_OVER_BUDGET)
                    trace.append({"step": "exclude", "key": u.key, "reason": "budget", "kind": u.kind, "messages": len(u.messages)})
                    continue
                sections["evidence"].append(rd.render_unit(cid, u, shown))
                partial.append((len(sections["evidence"]) - 1, cid, u))
                warnings.append(CONTEXT_OVERFLOW)
            n += 1
            rel = bd.relation(boundary, *(bd.day_of(u.event_time) if u.day_precision and u.event_time else (u.event_time, None)))
            provenance[cid] = {"key": u.key, "section": "evidence", "kind": u.kind, "hit_keys": list(u.hit_keys), "rank": u.rank,
                               "message_ids": [m.message_id for m in shown] if u.kind != "memory" else [], "memory_ids": [u.memory_id] if u.memory_id else [],
                               "source_ids": list(u.source_ids), "complete": u.kind == "memory" or len(shown) == len(u.messages),
                               "messages_total": len(u.messages), "time_kind": u.time_kind, "boundary_relation": rel,
                               "senders": sorted({m.sender for m in shown}) if u.kind != "memory" else [], "recipients": sorted({m.recipient for m in shown}) if u.kind != "memory" else []}

        # second pass: partial rounds take what the other units left, in rank order
        for idx, cid, u in partial:
            current = provenance[cid]["message_ids"]
            start = [m for m in u.messages if m.message_id in current]

            def fits(sub, idx=idx, cid=cid, u=u):
                before = sections["evidence"][idx]
                sections["evidence"][idx] = rd.render_unit(cid, u, sub)
                ok = self._fits(fixed, total, sections)
                sections["evidence"][idx] = before
                return ok
            shown = cp.overflow_selection(u, fits, essential_only=False, start=start)
            sections["evidence"][idx] = rd.render_unit(cid, u, shown)
            provenance[cid].update(message_ids=[m.message_id for m in shown], complete=len(shown) == len(u.messages),
                                   senders=sorted({m.sender for m in shown}), recipients=sorted({m.recipient for m in shown}))
            if len(shown) < len(u.messages):
                omitted[u.key] = [m.message_id for m in u.messages if m not in shown]
            trace.append({"step": "overflow", "key": u.key, "shown": len(shown), "total": len(u.messages), "omitted": omitted.get(u.key, [])})

        # derived items (observation / reflection): a slot only in V1
        for d in req.derived:
            if not req.allow_derived:
                excluded[d.key] = "derived_disabled_v1"
                warnings.append(DERIVED_EXCLUDED)
                continue
            shown_keys = {p["key"] for p in provenance.values() if p["section"] == "evidence"}
            if not d.supporting_evidence_ids or not set(d.supporting_evidence_ids) <= shown_keys:
                excluded[d.key] = "derived_without_shown_support"
                warnings.append(DERIVED_EXCLUDED)
                continue
            cid = f"D{len(sections['derived']) + 1:02d}"
            cites = "、".join(c for c, p in provenance.items() if p["key"] in d.supporting_evidence_ids)
            sections["derived"].append(f"[{cid}] （{d.kind}，依据 {cites}）{d.text}")
            if not self._fits(fixed, total, sections):
                sections["derived"].pop()
                excluded[d.key] = "budget:derived"
                continue
            provenance[cid] = {"key": d.key, "section": "derived", "supporting": list(d.supporting_evidence_ids), "authority": "derived_below_raw", "complete": True}

        status = OK
        if units and first_rank_key not in {p["key"] for p in provenance.values()}:
            warnings.append(TOP_EVIDENCE_EXCLUDED)
            if excluded.get(first_rank_key, "").startswith("budget"):
                status = BUDGET_INSUFFICIENT
        text = self._text(sections)
        used = fixed + self.count(text)
        counts = va.integrity(api, persona, provenance, text, as_of, known_at, total, used)
        trace.append({"step": "integrity", **counts})
        if any(counts.values()):
            status = SCOPE_ERROR
        trace.append({"step": "done", "status": status, "evidence_shown": n, "excluded": len(excluded), "used_tokens": used, "token_mode": self.count.mode})
        out = result(status)
        if status == SCOPE_ERROR:                  # never hand a leaking context to a reader
            out.assembled_context = ""
        return out


def _dir(speaker_id, participants, persona, speakers):
    d = round_direction(speaker_id, participants, persona, speakers)
    return d.sender, d.recipient
