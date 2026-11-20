"""Observations into the M1 Context Contract. Never as facts: as ``DerivedItem`` (authority DERIVED_OBSERVATION), each with the
memories it rests on, which are passed as ordinary evidence hits. The orchestrator shows an observation only when it also shows
that evidence, labels it as not an original record, and keeps it below the raw evidence."""
from __future__ import annotations

from datetime import datetime
from typing import List, Tuple

from stillhere.context.contracts import DerivedItem
from stillhere.contracts import EvidenceHit, HitKind


def for_context(api, engine, persona_id: str, known_at: datetime, as_of: datetime) -> Tuple[List[EvidenceHit], List[DerivedItem]]:
    hits, items, seen = [], [], set()
    for key, row in sorted(engine.current(persona_id, known_at).items()):
        if row["status"] != "active" or (row["valid_to"] and datetime.fromisoformat(row["valid_to"]) <= as_of):
            continue
        for mid in row["support"]:
            if mid in seen:
                continue
            r = api.get(persona_id, mid, known_at=known_at)
            if r is None or not r.is_active:
                break
            seen.add(mid)
            hits.append(EvidenceHit(kind=HitKind.MEMORY, persona_id=persona_id, text=r.statement, source_id=r.provenance[0].source_id, score=1.0, retriever="observation_support",
                                    memory_id=r.memory_id, subject_id=r.subject, valid_from=r.valid_from, valid_to=r.valid_to, recorded_at=r.recorded_at, locator=r.provenance[0].locator))
        else:
            items.append(DerivedItem(key=row["observation_id"], kind="observation", text=row["claim"], supporting_evidence_ids=tuple(row["support"]),
                                     contradicting_evidence_ids=tuple(row["contradict"]), valid_from=datetime.fromisoformat(row["valid_from"]) if row["valid_from"] else None,
                                     valid_to=datetime.fromisoformat(row["valid_to"]) if row["valid_to"] else None, confidence=row["heuristic_support"], version=str(row["version"])))
    return hits, items
