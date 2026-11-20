"""Scoring of hybrid retrieval: the BUILD-4 evidence keys and scorer, applied to any retrieval configuration.

Gold enters only after a retrieval has returned, in ``score_question``. ``HybridRetriever`` is given the question, the
persona and the times; a question's gold fields are never passed to it.
"""
from __future__ import annotations

import random
import statistics
from dataclasses import asdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from stillhere.memory import MemoryAPI
from stillhere.retrieval.hybrid import HybridRetriever, Retrieval, RetrievalConfig
from stillhere.retrieval.service import evidence_hit

from . import memory_bench as mb

TOP_KS = (5, 10)
CANDIDATE_KS = (10, 20, 50, 100)


def resolved_gold(q: mb.Question, gold_map: mb.GoldMap) -> Tuple[List[str], Dict[str, Optional[str]]]:
    resolved = [gold_map.resolve(key) for key in q.gold]
    gold = list(dict.fromkeys(k for k, _ in resolved if k))
    links = {new: dict(q.gold_links).get(old) for old, (new, _) in zip(q.gold, resolved) if new}
    return gold, links


def score_question(api: MemoryAPI, q: mb.Question, r: Retrieval, gold_map: mb.GoldMap, when) -> dict:
    keys = [c.key for c in r.final]
    gold, links = resolved_gold(q, gold_map)
    items = [mb.Retrieved(c.key, "hybrid", c.fusion_score, evidence_hit(api, c)) for c in r.final]
    row = {"qa_id": q.qa_id, "persona_id": q.persona_id, "primary_type": q.primary_type, "gold_count": len(gold),
           "scorable": bool(gold), "timings_ms": {k: round(v, 3) for k, v in r.timings_ms.items()}, "rerank_failed": r.rerank_failed,
           "temporal_violations": len(mb.temporal_violations(items, when)) + r.removed_by_validation,
           "persona_violations": sum(1 for i in items if i.hit.persona_id != q.persona_id or not i.key.startswith(q.persona_id + ":")),
           "source_errors": sum(1 for i in items if mb.locator_error(i)), "retrieved": keys,
           "channels": {c.key: list(c.retrieval_channels) for c in r.final}}
    if gold:
        scored = mb.score_list(keys, gold, links, q.gold_events, list(TOP_KS))
        first = next((i for i, k in enumerate(keys[:10], 1) if k in set(gold)), None)
        top = set(keys[:10])
        facts = [g for g in gold if g.split(":")[1] not in ("sms", "agent_chat")]
        messages = [g for g in gold if g not in facts]
        row.update(scores=asdict(scored), mrr=1.0 / first if first else 0.0,
                   coverage_facts=sum(g in top for g in facts) / len(facts) if facts else None,
                   coverage_messages=sum(g in top for g in messages) / len(messages) if messages else None)
    return row


def run(retriever: HybridRetriever, config: RetrievalConfig, questions: Sequence[mb.Question], gold_map: mb.GoldMap) -> List[dict]:
    rows = []
    for q in questions:
        when, _ = mb.query_times(q.ask_time)
        rows.append(score_question(retriever.api, q, retriever.retrieve(config, q.persona_id, q.question, when, when), gold_map, when))
    return rows


def _mean(values: Sequence[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return round(statistics.fmean(values), 4) if values else None


def metrics(rows: Sequence[dict]) -> dict:
    s = [r for r in rows if r["scorable"]]
    return {"questions": len(rows), "scorable": len(s),
            "hit@5": _mean([float(r["scores"]["hit"][5]) for r in s]), "hit@10": _mean([float(r["scores"]["hit"][10]) for r in s]),
            "coverage@5": _mean([r["scores"]["coverage"][5] for r in s]), "coverage@10": _mean([r["scores"]["coverage"][10] for r in s]),
            "full_recall@10": _mean([float(r["scores"]["full"][10]) for r in s]),
            "event_hit@10": _mean([float(r["scores"]["event_hit"][10]) for r in s]),
            "event_coverage@10": _mean([r["scores"]["event_coverage"][10] for r in s]), "mrr@10": _mean([r["mrr"] for r in s]),
            "coverage_gold_facts@10": _mean([r["coverage_facts"] for r in s]),
            "coverage_gold_messages@10": _mean([r["coverage_messages"] for r in s])}


def safety(rows: Sequence[dict]) -> dict:
    return {"temporal_violations": sum(r["temporal_violations"] for r in rows), "persona_violations": sum(r["persona_violations"] for r in rows),
            "source_errors": sum(r["source_errors"] for r in rows), "rerank_failures": sum(r["rerank_failed"] for r in rows)}


def breakdown(rows: Sequence[dict]) -> dict:
    groups: Dict[str, Dict[str, List[dict]]] = {"by_type": {}, "by_persona": {}, "by_hops": {}}
    for r in rows:
        groups["by_type"].setdefault(r["primary_type"], []).append(r)
        groups["by_persona"].setdefault(r["persona_id"], []).append(r)
        hops = "single" if r["primary_type"] == "Single_hop" else "multi"
        groups["by_hops"].setdefault(hops, []).append(r)
    return {name: {k: metrics(v) for k, v in sorted(g.items())} for name, g in groups.items()}


def latency(rows: Sequence[dict]) -> dict:
    names = sorted({k for r in rows for k in r["timings_ms"]})
    return {n: mb.latency_stats([r["timings_ms"][n] for r in rows if n in r["timings_ms"]]) for n in names}


def paired_bootstrap(a: Sequence[dict], b: Sequence[dict], field: str = "coverage", k: int = 10, *, rounds: int = 2000, seed: int = 1) -> dict:
    """Mean difference a - b by question, with a 95% interval. A hint about small-sample noise, not a guarantee."""
    x = [(float(p["scores"][field][k]), float(q["scores"][field][k])) for p, q in zip(a, b) if p["scorable"]]
    rng = random.Random(seed)
    n = len(x)
    diffs = [u - v for u, v in x]
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(rounds))
    return {"mean_difference": round(statistics.fmean(diffs), 4), "ci95": [round(means[int(0.025 * rounds)], 4), round(means[int(0.975 * rounds)], 4)], "questions": n}


def pareto(rows: Sequence[Mapping]) -> List[str]:
    """Names not dominated on coverage (higher), p95 and cost per query (lower)."""
    keep = []
    for a in rows:
        dominated = any(b is not a and b["coverage"] >= a["coverage"] and b["p95_ms"] <= a["p95_ms"] and b["cost_per_query"] <= a["cost_per_query"]
                        and (b["coverage"] > a["coverage"] or b["p95_ms"] < a["p95_ms"] or b["cost_per_query"] < a["cost_per_query"]) for b in rows)
        if not dominated:
            keep.append(a["name"])
    return keep


def candidate_recall(lists: Mapping[str, Sequence[str]], gold: Sequence[str], ks: Sequence[int] = CANDIDATE_KS) -> Dict[str, dict]:
    out = {}
    for name, keys in lists.items():
        out[name] = {}
        for k in ks:
            top = set(keys[:k])
            found = sum(g in top for g in gold)
            out[name][k] = (found > 0, found / len(gold), found)
    return out
