"""Checks run before anything reaches the reader. Each re-reads through ``MemoryAPI``; none trusts the retriever's copy.

``check_unit`` (per unit, before selection) and ``check_messages`` (the recent conversation) return a reason to exclude, or
None. ``integrity`` runs on the assembled result and reports counts that must all be zero; a non-zero count is a bug in this
package, and the orchestrator refuses the result (status SCOPE_ERROR) rather than send it.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Dict, Iterable, List, Optional

from stillhere.contracts import MemoryStatus
from stillhere.memory.sqlite_store import PersonaScopeError

from .units import Unit, UnitMessage


def message_reason(api, persona_id: str, m: UnitMessage, as_of: datetime, known_at: datetime) -> Optional[str]:
    if m.persona_id != persona_id:
        return "persona_scope"
    try:
        stored = api.get_message(persona_id, m.message_id)
    except PersonaScopeError:
        return "persona_scope"
    if stored is None or stored.content != m.content:
        return "provenance_mismatch"
    if stored.timestamp > as_of or (stored.recorded_at or stored.timestamp) > known_at:
        return "temporal_scope"
    if api.get_source(persona_id, stored.source_id) is None:
        return "provenance_missing"
    return None


def check_unit(api, persona_id: str, u: Unit, as_of: datetime, known_at: datetime) -> Optional[str]:
    if u.persona_id != persona_id:
        return "persona_scope"
    if u.kind == "memory":
        record = api.get(persona_id, u.memory_id, known_at=known_at)
        if record is None or record.recorded_at > known_at:
            return "temporal_scope"
        if record.status is not MemoryStatus.ACTIVE:
            return f"not_current:{record.status.value}"
        if record.statement != u.statement:
            return "provenance_mismatch"
        if any(api.get_source(persona_id, s) is None for s in u.source_ids):
            return "provenance_missing"
        return None
    if not u.messages:
        return "empty_unit"
    for m in u.messages:
        reason = message_reason(api, persona_id, m, as_of, known_at)
        if reason:
            return reason
    return None


def integrity(api, persona_id: str, provenance: Dict[str, dict], text: str, as_of: datetime, known_at: datetime,
              budget_total: int, used: int) -> Dict[str, int]:
    """Counts that must be zero: leaks, broken ids, missing attributions, budget."""
    counts = Counter()
    ids_seen = Counter()
    for cid, p in provenance.items():
        if f"[{cid}]" not in text:
            counts["context_id_not_rendered"] += 1
        for mid in p.get("message_ids", []):
            ids_seen[mid] += 1
            try:
                m = api.get_message(persona_id, mid)
            except PersonaScopeError:
                counts["persona_leakage"] += 1
                continue
            if m is None:
                counts["evidence_id_broken"] += 1
                continue
            if m.timestamp > as_of or (m.recorded_at or m.timestamp) > known_at:
                counts["temporal_leakage"] += 1
        for rid in p.get("memory_ids", []):
            try:
                r = api.get(persona_id, rid, known_at=known_at)
            except PersonaScopeError:
                counts["persona_leakage"] += 1
                continue
            if r is None:
                counts["temporal_leakage"] += 1
            elif r.status is not MemoryStatus.ACTIVE:
                counts["retracted_as_current"] += 1
    counts["duplicate_message_in_section"] = sum(1 for cid_dupes in _per_section_dupes(provenance) for _ in cid_dupes)
    if used > budget_total:
        counts["token_budget_violation"] += 1
    return dict(counts)


def _per_section_dupes(provenance: Dict[str, dict]) -> Iterable[List[str]]:
    by_section: Dict[str, Counter] = {}
    for p in provenance.values():
        c = by_section.setdefault(p["section"], Counter())
        c.update(p.get("message_ids", []))
    for c in by_section.values():
        yield [mid for mid, n in c.items() if n > 1]
