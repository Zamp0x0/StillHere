"""One bounded hop from first-hop seeds along stored relations. Nothing expands twice, so there is no loop to run away.

A neighbour must pass the same scope mask as any first-hop record before it is even considered: a legal seed proves
nothing about its neighbours. Every score is bounded by the seed's own score, so a large group cannot outrank what led to it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .dense import ScopeTable
from .event_graph import EventGraph

RELATION_STRENGTH = {"same_event": 0.9, "tree_root": 0.7, "conversation": 0.5, "entity": 0.5, "same_day": 0.2}


@dataclass(frozen=True)
class ExpansionConfig:
    relations: Tuple[str, ...] = ("same_event",)
    seeds: int = 10
    per_seed: int = 5
    global_budget: int = 50
    reserve: int = 0                  # final places kept for expanded evidence; 0 ranks everything together
    max_group: int = 300              # a larger group says too little about any member


@dataclass(frozen=True)
class Expanded:
    key: str
    record_id: str
    score: float
    seed_key: str
    relation: str
    provenance: str


def expand(graph: EventGraph, scope: ScopeTable, mask: np.ndarray, persona_id: str, seeds: Sequence[Tuple[str, float]], present: Set[str],
           relevance: Mapping[str, float], config: ExpansionConfig) -> List[Expanded]:
    """The best legal neighbours of the seeds that the first hop did not already find. ``relevance`` is the question's own
    keyword score of a record, 0 to 1. A neighbour reached twice keeps its best path."""
    best: Dict[str, Expanded] = {}
    for seed_key, seed_score in seeds:
        found: Dict[str, Expanded] = {}
        for relation in config.relations:
            strength = RELATION_STRENGTH[relation]
            for group in graph.groups_of(seed_key, relation):
                members = graph.groups[group]
                if len(members) > config.max_group:
                    continue
                for m in members:
                    if m.key == seed_key or m.key in present or not scope.visible(persona_id, m.record_id, mask):
                        continue
                    score = seed_score * strength * (0.5 + 0.5 * relevance.get(m.record_id, 0.0))
                    if m.key not in found or score > found[m.key].score:
                        found[m.key] = Expanded(m.key, m.record_id, score, seed_key, relation, m.provenance)
        for item in sorted(found.values(), key=lambda e: (-e.score, e.key))[:config.per_seed]:
            if item.key not in best or item.score > best[item.key].score:
                best[item.key] = item
    return sorted(best.values(), key=lambda e: (-e.score, e.key))[:config.global_budget]
