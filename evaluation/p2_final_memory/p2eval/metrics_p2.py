"""P2 metrics and the registered adoption rule, from raw records only (recomputable offline from the Results ZIP)."""
from __future__ import annotations

import math
import random
import statistics
from typing import Dict, List, Mapping, Optional, Sequence

from lbpilot import timeline as TL

from . import protocol as P

UA = "Unanswerable"


def majority(recs: Sequence[Mapping]) -> Optional[bool]:
    if len(recs) != 3 or any(r.get("status") != "OK" for r in recs):
        return None
    return sum(bool(r["is_correct"]) for r in recs) * 2 >= 3


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return float(f"{min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n):.3g}")


def bootstrap(d: List[int], resamples=P.BOOTSTRAP["resamples"], seed=P.BOOTSTRAP["seed"]):
    if not d:
        return None
    rng = random.Random(seed)
    n = len(d)
    bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return [round(bs[int(0.025 * resamples)], 4), round(bs[int(0.975 * resamples) - 1], 4)]


def rows_for(qs, answers: Mapping, judgments: Mapping, contexts: Mapping, key) -> List[dict]:
    out = []
    for q in qs:
        k = key(q["question_id"])
        recs = [judgments.get(k + (r,)) for r in range(3)]
        recs = [r for r in recs if r]
        a, c = answers.get(k), contexts.get(k)
        ev = q.get("evidence") or []
        kept = set()
        if c:
            kept = {TL.gold_id(x.split(":t", 1)[0]) for e in c.get("kept", []) for x in e.get("ids", []) if "." not in x}
        out.append({"question_id": q["question_id"], "types": q["question_type"], "ua": UA in q["question_type"], "correct": majority(recs),
                    "abstained": bool(a and a.get("abstained")), "answered": bool(a and a.get("status") == "OK"),
                    "recall": (sum(1 for g in ev if g in kept) / len(ev)) if (ev and c) else None, "tokens": c.get("tokens") if c else None,
                    "retrieval_ms": c.get("retrieval_ms") if c else None, "future": c.get("future_messages_in_store") if c else None,
                    "input_tokens": (a or {}).get("input_tokens"), "output_tokens": (a or {}).get("output_tokens")})
    return out


def group(rows: Sequence[Mapping]) -> dict:
    ok = [r for r in rows if r["correct"] is not None]
    ans = [r for r in ok if not r["ua"]]
    ua = [r for r in ok if r["ua"]]

    def acc(rs):
        return {"correct": sum(r["correct"] for r in rs), "n": len(rs), "accuracy": round(sum(r["correct"] for r in rs) / len(rs), 4) if rs else None}

    def typ(t):
        return acc([r for r in ok if t in r["types"]])
    rec = [r["recall"] for r in rows if r["recall"] is not None]
    tok = [r["tokens"] for r in rows if r["tokens"] is not None]
    lat = [r["retrieval_ms"] for r in rows if r["retrieval_ms"] is not None]
    return {"complete": len(ok) == len(rows) == 100, "judged": len(ok), "micro": acc(ok), "answerable": acc(ans), "unanswerable": acc(ua),
            "multi_hop": typ("Multi_hop"), "temporal": typ("Temporal"), "knowledge_update": typ("Knowledge_update"),
            "false_abstention": {"count": sum(1 for r in ans if r["abstained"]), "n": len(ans)},
            "unsupported": {"count": sum(1 for r in ok if not r["correct"] and not r["abstained"]), "n": len(ok)},
            "gold_recall_mean": round(statistics.mean(rec), 4) if rec else None, "evidence_tokens_median": statistics.median(tok) if tok else None,
            "retrieval_ms_p50": statistics.median(lat) if lat else None, "future_leaks": sum(1 for r in rows if r["future"] not in (None, 0))}


def paired(new: Sequence[Mapping], old: Sequence[Mapping], *, answerable_only: bool) -> dict:
    o = {r["question_id"]: r for r in old}
    pairs = [(int(r["correct"]), int(o[r["question_id"]]["correct"])) for r in new
             if r["question_id"] in o and r["correct"] is not None and o[r["question_id"]]["correct"] is not None and (not answerable_only or not r["ua"])]
    d = [a - b for a, b in pairs]
    b, c = sum(1 for x in d if x > 0), sum(1 for x in d if x < 0)
    ci = bootstrap(d)
    return {"n": len(d), "diff": round(sum(d) / len(d), 4) if d else None, "ci95": ci, "wins": b, "ties": len(d) - b - c, "losses": c,
            "mcnemar_p": mcnemar(b, c), "significant": bool(ci) and (ci[0] > 0 or ci[1] < 0)}


def adoption(old: Mapping, new: Mapping, prim: Mapping) -> dict:
    a = {"A_complete": old["complete"] and new["complete"],
         "B_answerable_not_lower": new["answerable"]["correct"] >= old["answerable"]["correct"],
         "C_unsupported": new["unsupported"]["count"] <= old["unsupported"]["count"] + 2,
         "D_unanswerable_safety": new["unanswerable"]["correct"] >= old["unanswerable"]["correct"] - 1,
         "E_no_leakage": old["future_leaks"] == 0 and new["future_leaks"] == 0,
         "F_targeted_gain": ((new["multi_hop"]["correct"] > old["multi_hop"]["correct"] or new["temporal"]["correct"] > old["temporal"]["correct"]
                              or new["false_abstention"]["count"] < old["false_abstention"]["count"]) and new["micro"]["correct"] >= old["micro"]["correct"]),
         "G_cost_latency": (new["evidence_tokens_median"] or 0) <= 4000 and (new["retrieval_ms_p50"] or 0) <= 3 * max(1.0, old["retrieval_ms_p50"] or 1.0)}
    decision = "ADOPT_P2" if all(a.values()) else ("NO_DECISION_INCOMPLETE" if not a["A_complete"] else "KEEP_OLD")
    return {"checks": a, "decision": decision, "primary_significant": prim.get("significant", False),
            "wording": "significant improvement" if (decision == "ADOPT_P2" and prim.get("significant")) else
            ("adopted on the registered rule; improvement not statistically significant" if decision == "ADOPT_P2" else "not adopted")}


def compute(inputs: Mapping, answers: Mapping, judgments: Mapping, contexts: Mapping, known_old: Optional[Mapping] = None) -> dict:
    out = {"schema": "stillhere.p2_final_memory.metrics/1.0", "protocol_sha256": P.P2_PROTOCOL_SHA256}
    if "NEWP" in inputs:
        qs = inputs["NEWP"]["qs"]
        r_old = rows_for(qs, answers, judgments, contexts, lambda i: ("NEWP", "OLD_V2D", i))
        r_new = rows_for(qs, answers, judgments, contexts, lambda i: ("NEWP", "NEW_P2_V2D", i))
        g_old, g_new = group(r_old), group(r_new)
        prim = paired(r_new, r_old, answerable_only=True)
        out["new_person"] = {"OLD_V2D": g_old, "NEW_P2_V2D": g_new, "primary_answerable": prim, "micro_pair": paired(r_new, r_old, answerable_only=False),
                             "adoption": adoption(g_old, g_new, prim), "per_question": {"OLD_V2D": r_old, "NEW_P2_V2D": r_new}}
    if "KNOWN" in inputs and known_old:
        qs = inputs["KNOWN"]["qs"]
        r_new = rows_for(qs, answers, judgments, contexts, lambda i: ("KNOWN", "NEW_P2_V2D", i))
        r_old = rows_for(qs, known_old["answers"], known_old["judgments"], known_old["contexts"], lambda i: ("M2_STILLHERE_V2D", i))
        out["known_set"] = {"label": "KNOWN_SET_REGRESSION (used for the root-cause audit; not a decision set)", "OLD_V2D_stored": group(r_old),
                            "NEW_P2_V2D": group(r_new), "answerable_pair": paired(r_new, r_old, answerable_only=True),
                            "micro_pair": paired(r_new, r_old, answerable_only=False),
                            "note": "OLD was answered / judged in the pilot run; the judge model alias is the same but the serving version may differ"}
    return out
