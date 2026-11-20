"""CPU retrieval benchmark on frozen real-history questions (Memory V2, RQ2). No LLM: it measures what the Reader would SEE.

Same questions, same gold (``evidence_message_ids`` of the frozen QA), same ask time and the same unit budget ``k`` for every variant.
Per question and variant: the ranked pack items (each a list of message ids), then

- ``recall``      |gold ∩ shown| / |gold|                       (= FAST's ``gold_retrieved``)
- ``hit``         any gold message shown                          (first-hop success)
- ``mrr``         1 / rank of the first item holding a gold message (0 if none)
- ``ndcg``        binary item relevance (item holds a gold message), cut at ``k``
- ``shown_msgs`` / ``shown_chars``: the evidence cost, reported next to recall (more neighbours = more recall AND more text)
- ``ms``          wall time of the retrieval (+ pack build)

A variant is a function ``(question, ask_time) -> (route, [[message ids per item]], shown_chars, meta)``. Gold is never read by a
variant. Aggregates: mean / CI by paired bootstrap against a baseline variant (``statistics.paired``-style, fixed seed).
"""
from __future__ import annotations

import math
import random
import statistics
import time
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

Variant = Callable[[str, object], Tuple[str, List[List[str]], int, dict]]


def score(items: Sequence[Sequence[str]], gold: Sequence[str], k: int) -> dict:
    g = set(gold)
    shown = [m for it in items[:k] for m in it]
    rel = [1 if g & set(it) else 0 for it in items[:k]]
    first = next((i for i, r in enumerate(rel, 1) if r), None)
    dcg = sum(r / math.log2(i + 1) for i, r in enumerate(rel, 1))
    ideal_n = min(k, len(g)) or 1
    idcg = sum(1 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return {"recall": (len(g & set(shown)) / len(g)) if g else None, "hit": int(bool(g & set(shown))) if g else None,
            "mrr": (1.0 / first) if first else 0.0, "ndcg": dcg / idcg if g else None, "shown_msgs": len(shown)}


def paired_ci(a: Mapping[str, float], b: Mapping[str, float], *, boot: int = 5000, seed: int = 20261118) -> dict:
    keys = sorted(k for k in set(a) & set(b) if a[k] is not None and b[k] is not None)
    if not keys:
        return {"n": 0}
    d = [b[k] - a[k] for k in keys]
    r = random.Random(seed)
    ms = sorted(sum(d[r.randrange(len(d))] for _ in d) / len(d) for _ in range(boot))
    return {"n": len(d), "diff": round(sum(d) / len(d), 4), "ci95": [round(ms[int(.025 * boot)], 4), round(ms[int(.975 * boot) - 1], 4)],
            "win": sum(x > 0 for x in d), "tie": sum(x == 0 for x in d), "loss": sum(x < 0 for x in d)}


def run(questions: Sequence[Mapping], variants: Mapping[str, Variant], ask_time, *, k: int = 8) -> dict:
    """``questions``: frozen QA rows (question_id, question, category, answerability, evidence_message_ids)."""
    per: Dict[str, Dict[str, dict]] = {v: {} for v in variants}
    for q in questions:
        gold = q.get("evidence_message_ids") or []
        for name, fn in variants.items():
            t0 = time.perf_counter()
            route, items, chars, meta = fn(q["question"], ask_time)
            ms = (time.perf_counter() - t0) * 1000
            s = score(items, gold, k) if gold else {"recall": None, "hit": None, "mrr": None, "ndcg": None, "shown_msgs": sum(len(i) for i in items[:k])}
            s.update(route=route, shown_chars=chars, ms=round(ms, 1), category=q.get("category"), answerable=q.get("answerability") == "ANSWERABLE",
                     n_gold=len(gold), meta=meta)
            per[name][q["question_id"]] = s
    return {"k": k, "per_question": per, "summary": summarize(per)}


def _agg(rows: List[dict]) -> dict:
    def mean(key):
        xs = [r[key] for r in rows if r.get(key) is not None]
        return round(statistics.fmean(xs), 4) if xs else None
    ms = sorted(r["ms"] for r in rows)
    return {"n": len(rows), "recall": mean("recall"), "hit": mean("hit"), "mrr": mean("mrr"), "ndcg": mean("ndcg"),
            "shown_msgs": mean("shown_msgs"), "shown_chars": mean("shown_chars"),
            "memory_route_rate": round(sum(r["route"] != "chat" for r in rows) / len(rows), 4) if rows else None,
            "ms_p50": ms[len(ms) // 2] if ms else None, "ms_p95": ms[min(len(ms) - 1, int(.95 * len(ms)))] if ms else None}


def summarize(per: Mapping[str, Mapping[str, dict]]) -> dict:
    out = {}
    for v, rows in per.items():
        ans = [r for r in rows.values() if r["answerable"] and r["n_gold"]]
        cats = sorted({r["category"] for r in ans})
        out[v] = {"answerable": _agg(ans), "by_category": {c: _agg([r for r in ans if r["category"] == c]) for c in cats},
                  "non_date": _agg([r for r in ans if r["category"] != "DATE_SUMMARY"])}
    return out


def compare(result: dict, base: str, metric: str = "recall", subset: Optional[Callable[[dict], bool]] = None) -> Dict[str, dict]:
    per = result["per_question"]
    keep = subset or (lambda r: r["answerable"] and r["n_gold"])
    a = {q: r[metric] for q, r in per[base].items() if keep(r)}
    return {v: paired_ci(a, {q: r[metric] for q, r in rows.items() if keep(r)}) for v, rows in per.items() if v != base}
