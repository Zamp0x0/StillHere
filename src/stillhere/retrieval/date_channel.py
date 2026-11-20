"""The date candidate channel: records dated inside a question's window, the ones closest to its words first.

The channel only adds candidates. Scope comes first, so a record that may not be seen at query time is never considered. It reads record times from the scope
table and text from the documents, both of which every store has. No event link, source format or dataset field is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Tuple

import numpy as np

from stillhere.memory.search import query_units

from .dense import NO_LIMIT, NO_START, Document, ScopeTable, clock_of
from .temporal_anchor import TemporalAnchor


@dataclass(frozen=True)
class TimeAnchorConfig:
    date_k: int = 20


def records_in_window(scope: ScopeTable, persona_id: str, mask: np.ndarray, anchor: TemporalAnchor) -> np.ndarray:
    """Rows dated inside the half-open window. An interval counts when it overlaps; a record with only a start counts when it starts inside; undated records do not count."""
    t = scope.tables[persona_id]
    low, high = clock_of(anchor.start), clock_of(anchor.end)
    start, end = t["start"], t["end"]
    dated = start != NO_START
    point = (end == NO_LIMIT) & (start >= low)
    span = (end != NO_LIMIT) & ((end > low) | (start >= low))
    return np.flatnonzero(mask & dated & (start < high) & (point | span))


def date_candidates(scope: ScopeTable, documents: Mapping[str, Document], persona_id: str, mask: np.ndarray, anchor: TemporalAnchor, k: int) -> Tuple[List[Tuple[str, int]], int]:
    """(best ``k`` records with their count of matched word pairs, records in the window). A record matching nothing can still fill the budget."""
    rows = records_in_window(scope, persona_id, mask, anchor)
    table = scope.tables[persona_id]
    ids = table.get("ids")
    if ids is None:
        ids = table["ids"] = list(table["row"])
    terms = set(query_units(anchor.query_without_time or "").bigrams)
    scored = []
    for row in rows.tolist():
        record = ids[row]
        document = documents.get(record)
        hits = sum(t in document.text for t in terms) if document else 0
        scored.append((-hits, int(table["start"][row]), record))
    scored.sort()
    return [(record, -hits) for hits, _, record in scored[:k]], len(scored)
