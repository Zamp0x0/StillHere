"""Named configurations, one pipeline. Chosen on development questions only; docs/HYBRID_RETRIEVAL_V2.md has the evidence.

Rule fixed before the choice: Coverage@10 within 0.01 of the best is a tie, then Hit@10, then event coverage, then latency.
QUALITY keeps the reranker off: on the same candidates it raised no Coverage@10 and added about two seconds. FAST calls no API.
"""
from __future__ import annotations

from dataclasses import replace

from .date_channel import TimeAnchorConfig
from .expansion import ExpansionConfig
from .hybrid import RetrievalConfig

FLASH = "qwen3.7-text-embedding-flash"

QUALITY = RetrievalConfig("quality", lexical="optimized", dense=True, lexical_k=50, dense_k=50, fusion="rrf", event_weight=0.5,
                          embedding_model=FLASH, parallel=True)
FAST = RetrievalConfig("fast", lexical="optimized", dense=False, lexical_k=50, dense_k=0, fusion="rrf", event_weight=0.1,
                       embedding_model=None)


# Optional. On development questions it raised candidate recall; in the final top 10 it gains and loses evidence (BUILD-4.1.3).
FAST_MULTIHOP = replace(FAST, name="fast_multihop", expansion=ExpansionConfig(("same_event",), seeds=20, per_seed=5, global_budget=50, reserve=3))


# Optional. FAST plus one candidate list for dates the question names; without such a date it is FAST (docs/TIME_ANCHORED_RETRIEVAL_PROTOCOL.md).
FAST_TIME = replace(FAST, name="fast_time", time_anchor=TimeAnchorConfig(date_k=20))


def experiment_base() -> RetrievalConfig:
    """The candidate generator the reranker and event experiments share."""
    return RetrievalConfig("hybrid_base")


def get(name: str) -> RetrievalConfig:
    profiles = {"quality": QUALITY, "fast": FAST, "fast_multihop": FAST_MULTIHOP, "fast_time": FAST_TIME}
    if name not in profiles:
        raise KeyError(f"no profile {name}")
    return profiles[name]
