"""Metrics from the raw records only (recomputable offline from the Results ZIP).

Accuracy follows the official judge: one binary label per question (majority of 3). Micro = correct / questions; per-type accuracy is
multi-label (a question counts in every type it lists, as ``LLMJudge.evaluate`` does); Macro = the mean over the 9 official types that
have at least one question (an absent type is N/A, never 0).

Evidence measures (none of them is answer accuracy):
* ``strict_id_recall`` (StillHere groups): share of the question's gold record ids present in the KEPT context entries.
* ``batch_provenance_recall`` (Mem0): share of gold ids that were in an ADD batch from which a KEPT memory came; weaker by design
  (a batch of 10 records produces several memories); reported separately, never compared with the strict recall.
* ``semantic_coverage`` (every memory group, one rule): a gold record counts as covered when >= 50 % of its distinct character bigrams
  appear in the kept context text. A crude, symmetric text-overlap heuristic: Mem0's rewritten facts are penalised for paraphrase,
  StillHere's raw records are not; read it as a rough signal only.
"""
from __future__ import annotations

import random
import statistics
from typing import Dict, List, Mapping, Optional, Sequence

from . import protocol as P

UA = "Unanswerable"


def _q(xs: Sequence[float], p: float) -> Optional[float]:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 3)


def bigrams(t: str) -> set:
    t = "".join(ch for ch in (t or "") if not ch.isspace())
    return {t[i:i + 2] for i in range(len(t) - 1)}


def covered(gold_text: str, context_text: str, threshold: float = 0.5) -> bool:
    g = bigrams(gold_text)
    return bool(g) and len(g & bigrams(context_text)) / len(g) >= threshold


def per_question(qa: Mapping, ans: Optional[Mapping], verdict: Optional[Mapping], ctx: Optional[Mapping], gold_texts: Mapping[str, List[str]],
                 group: str) -> dict:
    types = [str(t) for t in (qa.get("question_type") or [])]
    ev = list(qa.get("evidence") or [])
    row = {"question_id": qa["question_id"], "types": types, "unanswerable": UA in types, "is_correct": bool(verdict and verdict["is_correct"]),
           "judged": verdict is not None, "answered": bool(ans and ans.get("status") == "OK"), "abstained": bool(ans and ans.get("abstained")),
           "weighted_score": verdict.get("weighted_score") if verdict else None, "unanimous": verdict.get("unanimous") if verdict else None,
           "evidence_tokens": (ctx or {}).get("tokens"), "entries_kept": (ctx or {}).get("entries_kept"), "entries_dropped": (ctx or {}).get("entries_dropped"),
           "retrieval_ms": (ctx or {}).get("retrieval_ms"), "answer_latency_ms": (ans or {}).get("latency_ms"),
           "answer_input_tokens": (ans or {}).get("input_tokens"), "answer_output_tokens": (ans or {}).get("output_tokens"), "answer_cost_usd": (ans or {}).get("cost_usd")}
    if ctx and ev and group != "M0_NO_MEMORY":
        kept_ids = {i for e in ctx.get("kept", []) for i in e.get("ids", [])}
        from .timeline import gold_id
        kept_gold = {gold_id(i.split(":t", 1)[0]) for i in kept_ids if "." not in i}
        row["strict_id_recall" if group.startswith("M1") or group.startswith("M2") else "batch_provenance_recall"] = \
            round(sum(1 for g in ev if g in kept_gold) / len(ev), 4)
        text = ctx.get("text") or ""
        cov = [any(covered(t, text) for t in gold_texts.get(g, [])) for g in ev]
        row["semantic_coverage"] = round(sum(cov) / len(cov), 4)
    return row


def group_metrics(rows: Sequence[Mapping]) -> dict:
    n = len(rows)
    corr = sum(r["is_correct"] for r in rows)
    by_type: Dict[str, dict] = {}
    for t in P.OFFICIAL_TYPES:
        rs = [r for r in rows if t in r["types"]]
        by_type[t] = {"n": len(rs), "correct": sum(r["is_correct"] for r in rs), "accuracy": round(sum(r["is_correct"] for r in rs) / len(rs), 4) if rs else None}
    present = [v["accuracy"] for v in by_type.values() if v["accuracy"] is not None]
    ans = [r for r in rows if not r["unanswerable"]]
    ua = [r for r in rows if r["unanswerable"]]

    def mean(k, rs=rows):
        xs = [r[k] for r in rs if r.get(k) is not None]
        return round(statistics.mean(xs), 4) if xs else None
    return {
        "n": n, "judged": sum(r["judged"] for r in rows), "answered": sum(r["answered"] for r in rows),
        "micro_accuracy": round(corr / n, 4) if n else None, "correct": corr,
        "macro_accuracy": round(statistics.mean(present), 4) if present else None, "macro_types_present": len(present),
        "macro_types_na": [t for t, v in by_type.items() if v["accuracy"] is None], "by_type": by_type,
        "answerable": {"n": len(ans), "correct": sum(r["is_correct"] for r in ans), "accuracy": round(sum(r["is_correct"] for r in ans) / len(ans), 4) if ans else None},
        "unanswerable": {"n": len(ua), "correct": sum(r["is_correct"] for r in ua), "accuracy": round(sum(r["is_correct"] for r in ua) / len(ua), 4) if ua else None},
        "false_abstention": {"count": sum(r["abstained"] for r in ans), "of": len(ans)},
        "unsupported_claim": {"count": sum((not r["is_correct"]) and (not r["abstained"]) for r in rows), "of": n,
                              "on_unanswerable": sum((not r["is_correct"]) and (not r["abstained"]) for r in ua)},
        "temporal_accuracy": by_type["Temporal"]["accuracy"], "multi_hop_accuracy": by_type["Multi_hop"]["accuracy"],
        "weighted_score_mean": mean("weighted_score"), "judge_unanimous_share": mean("unanimous"),
        "evidence": {"strict_id_recall_mean": mean("strict_id_recall"), "batch_provenance_recall_mean": mean("batch_provenance_recall"),
                     "semantic_coverage_mean": mean("semantic_coverage"), "evidence_tokens_mean": mean("evidence_tokens"),
                     "evidence_tokens_p95": _q([r["evidence_tokens"] for r in rows], .95), "entries_dropped_total": sum(r.get("entries_dropped") or 0 for r in rows)},
        "latency_ms": {"retrieval_p50": _q([r["retrieval_ms"] for r in rows], .5), "retrieval_p95": _q([r["retrieval_ms"] for r in rows], .95),
                       "answer_p50": _q([r["answer_latency_ms"] for r in rows], .5), "answer_p95": _q([r["answer_latency_ms"] for r in rows], .95),
                       "end_to_end_p50": _q([(r["retrieval_ms"] or 0) + (r["answer_latency_ms"] or 0) for r in rows], .5)},
        "answer_tokens": {"input": sum(r.get("answer_input_tokens") or 0 for r in rows), "output": sum(r.get("answer_output_tokens") or 0 for r in rows)},
        "answer_cost_usd": round(sum(r.get("answer_cost_usd") or 0 for r in rows), 6),
    }


def paired(a: Sequence[Mapping], b: Sequence[Mapping], *, resamples: int = P.BOOTSTRAP["resamples"], seed: int = P.BOOTSTRAP["seed"]) -> dict:
    """A - B on the same questions: W / T / L on the binary label and a paired bootstrap 95 % CI of the accuracy difference."""
    ib = {r["question_id"]: r for r in b}
    pairs = [(int(r["is_correct"]), int(ib[r["question_id"]]["is_correct"])) for r in a if r["question_id"] in ib]
    n = len(pairs)
    d = [x - y for x, y in pairs]
    rng = random.Random(seed)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)) if n else []
    lo, hi = (boots[int(0.025 * resamples)], boots[int(0.975 * resamples) - 1]) if boots else (None, None)
    return {"n": n, "diff": round(sum(d) / n, 4) if n else None, "ci95": [round(lo, 4), round(hi, 4)] if boots else None,
            "wins": sum(1 for x in d if x > 0), "ties": sum(1 for x in d if x == 0), "losses": sum(1 for x in d if x < 0),
            "significant": bool(boots) and (lo > 0 or hi < 0)}
