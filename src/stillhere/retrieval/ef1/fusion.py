"""Rank fusion for EF1: weighted reciprocal-rank fusion (RRF) on evidence units, plus the single-channel orders for fair comparison.

    score(u) = sum_c  w_c / (c0 + rank_c(u))        rank_c = 1-based position of unit u in channel c (absent: no term)

``strategy = "rrf"`` uses the weights; ``strategy = "<channel>"`` (e.g. ``"msg_bm25"``) returns that channel's own order, so a fused default
has to beat each single channel on the same questions before it is adopted (Phase 1: a plain unweighted RRF lost to raw-message bm25 at
K = 10). Ties break by the best single-channel rank, then by key, so the order is deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence

from .channels import ChannelResult


@dataclass
class Candidate:
    key: str
    score: float
    ranks: Dict[str, int] = field(default_factory=dict)          # channel -> 1-based rank
    memories: List[str] = field(default_factory=list)            # memory ids linked by any channel


def fuse(results: Sequence[ChannelResult], *, strategy: str = "rrf", weights: Mapping[str, float] = None, c0: float = 60.0) -> List[Candidate]:
    weights = dict(weights or {})
    cands: Dict[str, Candidate] = {}
    for res in results:
        for i, key in enumerate(res.keys, 1):
            c = cands.setdefault(key, Candidate(key, 0.0))
            c.ranks[res.name] = min(i, c.ranks.get(res.name, i))
            for mid in res.memories.get(key, []):
                if mid not in c.memories:
                    c.memories.append(mid)
    if strategy == "rrf":
        for c in cands.values():
            c.score = sum(weights.get(ch, 0.0) / (c0 + r) for ch, r in c.ranks.items())
        pool = [c for c in cands.values() if c.score > 0]
        return sorted(pool, key=lambda c: (-c.score, min(c.ranks.values()), c.key))
    # single channel order (others only contribute linked memories)
    pool = [c for c in cands.values() if strategy in c.ranks]
    for c in pool:
        c.score = 1.0 / c.ranks[strategy]
    return sorted(pool, key=lambda c: (c.ranks[strategy], c.key))
