#!/usr/bin/env python3
"""Independent final audit of the P2 Colab results (CPU only; no API call).

Reads ONLY the raw records of the Results ZIP (contexts / answers / judge / usage logs / STATE / manifest of the input package), the
pilot's stored M2_STILLHERE_V2D records (known-set OLD) and the official dataset (questions, gold evidence). Every metric is recomputed
here with its own code (majority vote, paired bootstrap, exact McNemar) and compared with the run's own P2_METRICS.json; the run's metric
module is NOT imported. Writes the aggregate audit (public, no question text / answers) and the per-question detail (private).

    python evaluation/p2_final_memory/final_audit.py --results <raw dir> --zip <Results ZIP> --pilot-raw <pilot raw dir>
        --dataset <official json> --input-zip <P2 input zip> --public-out research/p2_final_memory/final_audit --private-out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "evaluation" / "lifebench_pilot"), str(ROOT / "evaluation" / "p2_final_memory")]

from lbpilot import answer as LA  # noqa: E402
from lbpilot import select as LS  # noqa: E402
from lbpilot import timeline as TL  # noqa: E402
from p2eval import protocol as P  # noqa: E402
from p2eval import select_new as SN  # noqa: E402

_spec = importlib.util.spec_from_file_location("pilot_final_audit", ROOT / "evaluation" / "lifebench_pilot" / "final_audit.py")
PFA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PFA)                       # svg helpers only

UA = "Unanswerable"
SEED, RESAMPLES = 20261119, 10000


def rows(p: Path):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 1)


def mcnemar_exact(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def boot(d):
    rng = random.Random(SEED)
    n = len(d)
    bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(RESAMPLES))
    return [round(bs[int(0.025 * RESAMPLES)], 4), round(bs[int(0.975 * RESAMPLES) - 1], 4)]


def gold_kept(ctx):
    return {TL.gold_id(x.split(":t", 1)[0]) for e in ctx.get("kept", []) for x in e.get("ids", []) if "." not in x}


def per_question(qs, ctx, ans, jud, key):
    out = []
    for q in qs:
        k = key(q["question_id"])
        js = [jud.get(k + (r,)) for r in range(3)]
        ok = all(j is not None and j.get("status") == "OK" for j in js)
        votes = sum(1 for j in js if j and str(j.get("is_correct")) == "True") if ok else None
        a, c = ans.get(k), ctx.get(k)
        ev = q.get("evidence") or []
        kept = gold_kept(c) if c else set()
        text = (a or {}).get("answer") or ""
        out.append({
            "question_id": q["question_id"], "types": q["question_type"], "ua": UA in q["question_type"],
            "correct": (votes * 2 >= 3) if votes is not None else None, "votes": votes,
            "abstained": str((a or {}).get("abstained")) == "True", "abstain_recomputed": LA.is_abstention(text) if a else None,
            "answer_status": (a or {}).get("status"), "gold_n": len(ev), "gold_kept": sorted(g for g in ev if g in kept),
            "recall": (sum(1 for g in ev if g in kept) / len(ev)) if (ev and c) else None,
            "tokens": int(c["tokens"]) if c else None, "entries_in": c.get("entries_in") if c else None, "entries_dropped": c.get("entries_dropped") if c else None,
            "retrieval_ms": float(c["retrieval_ms"]) if c and c.get("retrieval_ms") is not None else None,
            "future": c.get("future_messages_in_store") if c else None, "ask_time": c.get("ask_time") if c else None,
            "added_through": c.get("added_through") if c else None,
            "p2_kinds": ((c or {}).get("meta") or {}).get("p2_kinds"), "second_pass": ((c or {}).get("meta") or {}).get("second_pass"),
            "p2_requirements": len(((c or {}).get("meta") or {}).get("p2_requirements") or []),
            "coverage_uncovered": sum(1 for ln in (c or {}).get("text", "").splitlines() if ln.startswith("- R") and ln.endswith("没有找到记录")),
            "temporal_notes": (c or {}).get("text", "").count("日程/计划记录"),
            "answer_latency_ms": float(a["latency_ms"]) if a and a.get("latency_ms") is not None else None,
            "input_tokens": int(a["input_tokens"]) if a and a.get("input_tokens") is not None else None,
            "output_tokens": int(a["output_tokens"]) if a and a.get("output_tokens") is not None else None,
            "answer_cost_usd": float(a["cost_usd"]) if a and a.get("cost_usd") is not None else None,
            "judge_cost_usd": round(sum(float(j.get("cost_usd") or 0) for j in js if j), 8),
        })
    return out


def summarise(rs):
    ok = [r for r in rs if r["correct"] is not None]
    ans = [r for r in ok if not r["ua"]]
    ua = [r for r in ok if r["ua"]]

    def acc(xs):
        return {"correct": sum(r["correct"] for r in xs), "n": len(xs)}

    def typ(t):
        return acc([r for r in ok if t in r["types"]])
    rec = [r["recall"] for r in rs if r["recall"] is not None]
    rl = [r["retrieval_ms"] for r in rs if r["retrieval_ms"] is not None]
    al = [r["answer_latency_ms"] for r in rs if r["answer_latency_ms"] is not None]
    return {
        "records": len(rs), "judged": len(ok), "complete": len(ok) == len(rs) == 100,
        "micro": acc(ok), "answerable": acc(ans), "unanswerable": acc(ua),
        "multi_hop": typ("Multi_hop"), "temporal": typ("Temporal"), "knowledge_update": typ("Knowledge_update"), "single_hop": typ("Single_hop"),
        "conflict": typ("Conflict"), "causal": typ("Causal"), "pattern": typ("Pattern_recognition(Non-declarative)"), "hidden": typ("Hidden_info"),
        "false_abstention": {"count": sum(1 for r in ans if r["abstained"]), "n": len(ans)},
        "correct_abstention": {"count": sum(1 for r in ua if r["abstained"] and r["correct"]), "n": len(ua)},
        "unsupported": {"count": sum(1 for r in ok if not r["correct"] and not r["abstained"]), "n": len(ok)},
        "wrong_attribution": "UNAVAILABLE (the official judge gives CORRECT/WRONG + point hits, no attribution label)",
        "wrong_temporal_claims": "UNAVAILABLE (no temporal-claim label in the judge output)",
        "gold_recall_mean": round(statistics.mean(rec), 4) if rec else None, "gold_full_recall": sum(1 for x in rec if x == 1),
        "gold_hit_any": sum(1 for x in rec if x > 0), "gold_questions": len(rec),
        "tokens_median": statistics.median([r["tokens"] for r in rs if r["tokens"] is not None]),
        "tokens_max": max(r["tokens"] for r in rs if r["tokens"] is not None),
        "contexts_with_budget_drop": sum(1 for r in rs if str(r["entries_dropped"]) not in ("0", "None")),
        "retrieval_ms": {"p50": pct(rl, .5), "p95": pct(rl, .95)}, "answer_latency_ms": {"p50": pct(al, .5), "p95": pct(al, .95)},
        "input_tokens": sum(r["input_tokens"] or 0 for r in rs), "output_tokens": sum(r["output_tokens"] or 0 for r in rs),
        "answer_cost_usd_estimate": round(sum(r["answer_cost_usd"] or 0 for r in rs), 4), "judge_cost_usd_estimate": round(sum(r["judge_cost_usd"] for r in rs), 4),
        "future_leaks": sum(1 for r in rs if str(r["future"]) not in ("0", "None")),
        "abstain_flag_mismatch": sum(1 for r in rs if r["abstain_recomputed"] is not None and r["abstain_recomputed"] != r["abstained"]),
    }


def pair(new, old, answerable_only):
    o = {r["question_id"]: r for r in old}
    d, flips = [], {"win": [], "loss": []}
    for r in new:
        x = o.get(r["question_id"])
        if not x or r["correct"] is None or x["correct"] is None or (answerable_only and r["ua"]):
            continue
        v = int(r["correct"]) - int(x["correct"])
        d.append(v)
        if v:
            flips["win" if v > 0 else "loss"].append(r["question_id"])
    b, c = d.count(1), d.count(-1)
    ci = boot(d)
    return {"n": len(d), "diff": round(sum(d) / len(d), 4), "ci95": ci, "wins": b, "ties": len(d) - b - c, "losses": c,
            "mcnemar_exact_p": round(mcnemar_exact(b, c), 4), "significant": ci[0] > 0 or ci[1] < 0}, flips


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ("--results", "--zip", "--pilot-raw", "--dataset", "--input-zip", "--public-out", "--private-out"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    raw, pub, priv = Path(a.results), Path(a.public_out), Path(a.private_out)
    pub.mkdir(parents=True, exist_ok=True)
    priv.mkdir(parents=True, exist_ok=True)
    checks = {}

    # ---------------------------------------------------------------- integrity
    z = zipfile.ZipFile(a.zip)
    checks["zip_sha256"] = sha_file(Path(a.zip))
    checks["zip_crc_ok"] = z.testzip() is None
    sums = dict(l.split("  ", 1)[::-1] for l in (raw / "SHA256SUMS.txt").read_text().splitlines() if l.strip())
    checks["sha256sums"] = {"files": len(sums), "mismatch": [n for n, s in sums.items() if sha_file(raw / n) != s]}
    st = json.loads((raw / "STATE.json").read_text(encoding="utf-8"))
    checks["state_protocol_sha256"] = st["protocol_sha256"]
    checks["protocol_matches_registered"] = st["protocol_sha256"] == P.P2_PROTOCOL_SHA256
    checks["stages"] = {k: v["status"] for k, v in st["stages"].items()}
    iz = zipfile.ZipFile(a.input_zip)
    man = json.loads(iz.read("P2_MANIFEST.json"))
    checks["input_package"] = {"zip_sha256": sha_file(Path(a.input_zip)), "git_commit": man["git_commit"], "protocol_sha256": man["protocol_sha256"],
                               "old_fingerprint_source_commit": man["old_fingerprint"]["source_commit"], "old_tree_sha256": man["old_fingerprint"]["tree_sha256"],
                               "new_p2_module_sha256": man["new_p2"]["sha256"], "answer_prompt_p2_sha256": man["new_p2"]["answer_prompt_sha256"]}
    data = LS.load_dataset(a.dataset)
    checks["dataset_sha256"] = sha_file(Path(a.dataset))
    checks["dataset_matches_protocol"] = checks["dataset_sha256"] == P.DATASET["sha256"]
    FN, FK = json.loads(iz.read("FROZEN_NEW_100QA.json")), json.loads(iz.read("FROZEN_KNOWN_100QA.json"))
    checks["frozen_new_qa_sha256"] = {"manifest": FN["qa_sha256"], "recomputed": SN.freeze(data)["qa_sha256"]}
    checks["frozen_known_qa_sha256"] = {"manifest": FK["qa_sha256"], "recomputed": LS.freeze_data(data)["qa_sha256"]}
    pn, qn = SN.selected(data, FN)
    pk, qk = LS.selected(data, FK)
    checks["new_person"] = FN["person"]["sample_id"]
    checks["timelines"] = {"NEWP": TL.sha(TL.build(pn, until=max(q["ask_time"][:10] for q in qn))) == man["timelines"]["NEWP"],
                           "KNOWN": TL.sha(TL.build(pk, until=max(q["ask_time"][:10] for q in qk))) == man["timelines"]["KNOWN"]}

    C, A, J = rows(raw / "contexts.jsonl"), rows(raw / "answers.jsonl"), rows(raw / "judge.jsonl")
    ctx = {(r["experiment"], r["group"], r["question_id"]): r for r in C}
    ans = {(r["experiment"], r["group"], r["question_id"]): r for r in A}
    jud = {(r["experiment"], r["group"], r["question_id"], int(r["repeat"])): r for r in J}
    checks["duplicates"] = {"contexts": len(C) - len(ctx), "answers": len(A) - len(ans), "judge": len(J) - len(jud)}
    checks["counts"] = {"contexts": dict(Counter(f"{r['experiment']}:{r['group']}" for r in C)), "answers": dict(Counter(f"{r['experiment']}:{r['group']}" for r in A)),
                        "judge": dict(Counter(f"{r['experiment']}:{r['group']}" for r in J))}
    want = {("NEWP", g, q["question_id"]) for g in ("OLD_V2D", "NEW_P2_V2D") for q in qn} | {("KNOWN", "NEW_P2_V2D", q["question_id"]) for q in qk}
    checks["question_ids_exact"] = {"contexts": set(ctx) == want, "answers": set(ans) == want,
                                    "judge": {k[:3] for k in jud} == want and all(len([r for r in range(3) if k + (r,) in jud]) == 3 for k in want)}
    asks = {q["question_id"]: q["ask_time"][:10] for q in qn + qk}
    checks["ask_time_matches_dataset"] = all(r["ask_time"][:10] == asks[r["question_id"]] for r in C)
    checks["store_not_ahead_of_ask"] = (all(r["added_through"] <= r["ask_time"][:10] for r in C) if all("added_through" in r for r in C)
                                        else "added_through not recorded per context in the P2 runner; time safety = future_messages_in_store (below)")
    checks["future_messages_in_store_nonzero"] = sum(1 for r in C if r.get("future_messages_in_store") not in (0, None))
    checks["context_status_not_ok"] = sum(1 for r in C if r.get("status") != "OK")
    checks["tokens_over_budget"] = sum(1 for r in C if int(r["tokens"]) > P.EVIDENCE["token_budget"])
    checks["answer_status"] = dict(Counter(r["status"] for r in A))
    checks["answer_prompts"] = {g: sorted({(r["prompt_version"], r["prompt_sha256"]) for r in A if r["group"] == g}) for g in ("OLD_V2D", "NEW_P2_V2D")}
    checks["answer_models"] = dict(Counter(f"{r['requested_model']} -> {r['returned_model']}" for r in A))
    checks["answer_fingerprints"] = dict(Counter(r.get("fingerprint") for r in A))
    checks["answer_retries"] = {"attempts>1": sum(1 for r in A if int(r.get("attempts") or 1) > 1), "max_attempts": max(int(r.get("attempts") or 1) for r in A)}
    checks["answer_finish"] = dict(Counter(r.get("finish_reason") for r in A))
    checks["judge_status"] = dict(Counter(r["status"] for r in J))
    checks["judge_prompt_sha256"] = sorted({r["judge_prompt_sha256"] for r in J})
    from lbpilot import judge as LJ
    checks["judge_prompt_matches_official"] = checks["judge_prompt_sha256"] == [LJ.OFFICIAL_PROMPT_SHA256]
    checks["judge_models"] = dict(Counter(f"{r['requested_model']} -> {r['returned_model']}" for r in J))
    checks["judge_parse_attempts>1"] = sum(1 for r in J if int(r.get("parse_attempts") or 1) > 1)
    checks["judge_retries>1"] = sum(1 for r in J if int(r.get("attempts") or 1) > 1)
    # pilot OLD prompt is the same v1 prompt
    PA = rows(Path(a.pilot_raw) / "answers.jsonl")
    pilot_v1 = sorted({r.get("prompt_sha256") for r in PA if r["group"] == "M2_STILLHERE_V2D"} - {None})
    checks["pilot_v1_prompt_sha256"] = pilot_v1 or "not recorded in the pilot answers"
    usage = {n: rows(raw / "logs" / f"api_usage_{n}.jsonl") for n in ("answer", "judge")}
    checks["usage_log"] = {n: {"calls": len(u), "errors": sum(1 for x in u if x.get("status") not in (None, "OK", 200, "200")),
                               "cost_usd_estimate": round(sum(float(x.get("cost_usd") or 0) for x in u), 4)} for n, u in usage.items()}

    # ---------------------------------------------------------------- recompute
    old_new = per_question(qn, ctx, ans, jud, lambda i: ("NEWP", "OLD_V2D", i))
    new_new = per_question(qn, ctx, ans, jud, lambda i: ("NEWP", "NEW_P2_V2D", i))
    new_known = per_question(qk, ctx, ans, jud, lambda i: ("KNOWN", "NEW_P2_V2D", i))
    PC = {(r["group"], r["question_id"]): r for r in rows(Path(a.pilot_raw) / "contexts.jsonl")}
    PJ = {(r["group"], r["question_id"], int(r["repeat"])): r for r in rows(Path(a.pilot_raw) / "judge.jsonl")}
    PAk = {(r["group"], r["question_id"]): r for r in PA}
    old_known = per_question(qk, PC, PAk, PJ, lambda i: ("M2_STILLHERE_V2D", i))
    S = {"new_person": {"OLD_V2D": summarise(old_new), "NEW_P2_V2D": summarise(new_new)},
         "known_set": {"OLD_V2D_stored_pilot": summarise(old_known), "NEW_P2_V2D": summarise(new_known),
                       "label": "KNOWN_SET_REGRESSION — NOT INDEPENDENT VALIDATION"}}
    flips = {}
    for name, n_, o_ in (("new_person", new_new, old_new), ("known_set", new_known, old_known)):
        S[name]["answerable_pair"], flips[name] = pair(n_, o_, True)
        S[name]["micro_pair"], _ = pair(n_, o_, False)
        for t, lab in (("Multi_hop", "multi_hop_pair"), ("Temporal", "temporal_pair")):
            S[name][lab], _ = pair([r for r in n_ if t in r["types"]], [r for r in o_ if t in r["types"]], False)
        rd = [(r["recall"] - o["recall"]) for r, o in zip(n_, o_) if r["recall"] is not None and o["recall"] is not None]
        S[name]["recall_pair"] = {"n": len(rd), "diff": round(statistics.mean(rd), 4), "ci95": boot(rd), "better": sum(1 for x in rd if x > 1e-9),
                                  "worse": sum(1 for x in rd if x < -1e-9)}
        # second-pass recovery: gold records in NEW's context that OLD's context (same store, same first hop) did not have
        rec, lost = 0, 0
        for r, o in zip(n_, o_):
            rec += len(set(r["gold_kept"]) - set(o["gold_kept"]))
            lost += len(set(o["gold_kept"]) - set(r["gold_kept"]))
        S[name]["gold_records_recovered_by_p2"], S[name]["gold_records_lost_vs_old"] = rec, lost
        S[name]["second_pass_rate"] = {"count": sum(1 for r in n_ if r["second_pass"]), "n": len(n_)}
        S[name]["p2_llm_calls_in_retrieval"] = 0
        # by P2 form (NEW's plan) : correctness + recall, OLD vs NEW
        by = defaultdict(lambda: {"n": 0, "old_correct": 0, "new_correct": 0, "old_recall": [], "new_recall": []})
        for r, o in zip(n_, o_):
            for kd in (r["p2_kinds"] or ["simple"]):
                b = by[kd]
                b["n"] += 1
                b["old_correct"] += int(bool(o["correct"]))
                b["new_correct"] += int(bool(r["correct"]))
                if r["recall"] is not None and o["recall"] is not None:
                    b["old_recall"].append(o["recall"])
                    b["new_recall"].append(r["recall"])
        S[name]["by_p2_form"] = {k: {"n": v["n"], "old_correct": v["old_correct"], "new_correct": v["new_correct"],
                                     "old_recall": round(statistics.mean(v["old_recall"]), 3) if v["old_recall"] else None,
                                     "new_recall": round(statistics.mean(v["new_recall"]), 3) if v["new_recall"] else None} for k, v in sorted(by.items())}
        # reader use: answerable questions whose context held ALL gold records — correct?
        full = [(r, o) for r, o in zip(n_, o_) if not r["ua"] and r["gold_n"] and len(r["gold_kept"]) == r["gold_n"]]
        S[name]["new_full_evidence_answerable"] = {"n": len(full), "correct": sum(1 for r, _ in full if r["correct"]),
                                                   "abstained": sum(1 for r, _ in full if r["abstained"])}
        S[name]["coverage_note"] = {"contexts_with_note": sum(1 for r in n_ if r["p2_requirements"]),
                                    "requirements": sum(r["p2_requirements"] for r in n_), "uncovered_requirements": sum(r["coverage_uncovered"] for r in n_),
                                    "contexts_with_temporal_note": sum(1 for r in n_ if r["temporal_notes"])}
    # recompute vs the run's own metrics
    m = json.loads((raw / "P2_METRICS.json").read_text(encoding="utf-8"))
    cmp = {}
    for side, mine, theirs in (("new_person", S["new_person"], m["new_person"]), ("known_set", S["known_set"], m["known_set"])):
        pairs_ = [("OLD_V2D", "OLD_V2D") if side == "new_person" else ("OLD_V2D_stored_pilot", "OLD_V2D_stored"), ("NEW_P2_V2D", "NEW_P2_V2D")]
        for mg, tg in pairs_:
            for k in ("micro", "answerable", "unanswerable", "multi_hop", "temporal", "knowledge_update"):
                cmp[f"{side}.{mg}.{k}"] = mine[mg][k]["correct"] == theirs[tg][k]["correct"] and mine[mg][k]["n"] == theirs[tg][k]["n"]
            for k in ("false_abstention", "unsupported"):
                cmp[f"{side}.{mg}.{k}"] = mine[mg][k]["count"] == theirs[tg][k]["count"]
            cmp[f"{side}.{mg}.gold_recall_mean"] = mine[mg]["gold_recall_mean"] == theirs[tg]["gold_recall_mean"]
        tp = theirs["primary_answerable"] if side == "new_person" else theirs["answerable_pair"]
        mp = mine["answerable_pair"]
        cmp[f"{side}.answerable_pair"] = (mp["wins"], mp["losses"], mp["ci95"], mp["diff"]) == (tp["wins"], tp["losses"], tp["ci95"], tp["diff"])
    checks["recomputed_equals_run_metrics"] = {"all": all(cmp.values()), "mismatch": [k for k, v in cmp.items() if not v]}

    # ---------------------------------------------------------------- registered adoption rule (re-evaluated here)
    o, n = S["new_person"]["OLD_V2D"], S["new_person"]["NEW_P2_V2D"]
    rule = {"A_complete": o["complete"] and n["complete"],
            "B_answerable_not_lower": n["answerable"]["correct"] >= o["answerable"]["correct"],
            "C_unsupported_le_old_plus_2": n["unsupported"]["count"] <= o["unsupported"]["count"] + 2,
            "D_unanswerable_ge_old_minus_1": n["unanswerable"]["correct"] >= o["unanswerable"]["correct"] - 1,
            "E_no_leakage": o["future_leaks"] == 0 and n["future_leaks"] == 0,
            "F_targeted_gain_and_micro_not_lower": (n["multi_hop"]["correct"] > o["multi_hop"]["correct"] or n["temporal"]["correct"] > o["temporal"]["correct"]
                                                    or n["false_abstention"]["count"] < o["false_abstention"]["count"]) and n["micro"]["correct"] >= o["micro"]["correct"],
            "G_tokens_and_latency": n["tokens_median"] <= 4000 and n["retrieval_ms"]["p50"] <= 3 * o["retrieval_ms"]["p50"]}
    decision = {"rule": rule, "registered_decision": "ADOPT_P2" if all(rule.values()) else ("NO_DECISION_INCOMPLETE" if not rule["A_complete"] else "KEEP_OLD"),
                "run_decision": m["new_person"]["adoption"]["decision"], "primary_significant": S["new_person"]["answerable_pair"]["significant"]}
    out = {"schema": "stillhere.p2_final_audit/1.0", "protocol_sha256": P.P2_PROTOCOL_SHA256, "checks": checks, "metrics": S, "decision": decision}
    (pub / "P2_FINAL_AUDIT_METRICS.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (priv / "P2_FINAL_PER_QUESTION.json").write_text(json.dumps({"new_person": {"OLD_V2D": old_new, "NEW_P2_V2D": new_new}, "known_set": {"OLD_V2D_stored_pilot": old_known,
                                                                 "NEW_P2_V2D": new_known}, "flips": flips}, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------------------------------------------------------------- figures (aggregates only, real values, 0-based axes)
    fig = pub / "figures"
    fig.mkdir(exist_ok=True)

    def r_(d):
        return d["correct"] / d["n"] if d["n"] else None
    labs = ["micro", "answerable", "multi_hop", "temporal", "knowledge_update", "unanswerable"]
    (fig / "p2_new_person_accuracy.svg").write_text(PFA.svg_bars("Fresh person 于晓雯 (independent, n=100): accuracy", labs,
                                                                 {"OLD_V2D": [r_(o[k]) for k in labs], "NEW_P2_V2D": [r_(n[k]) for k in labs]}), encoding="utf-8")
    ko, kn = S["known_set"]["OLD_V2D_stored_pilot"], S["known_set"]["NEW_P2_V2D"]
    (fig / "p2_known_set_accuracy.svg").write_text(PFA.svg_bars("Known set 于晓薇 (KNOWN_SET_REGRESSION, not independent): accuracy", labs,
                                                                {"OLD_V2D (pilot)": [r_(ko[k]) for k in labs], "NEW_P2_V2D": [r_(kn[k]) for k in labs]}), encoding="utf-8")
    ab = ["false abstention", "unsupported", "gold full recall"]
    (fig / "p2_abstention_recall_counts.svg").write_text(PFA.svg_bars("Fresh person: counts (lower is better for the first two)", ab,
                                                                      {"OLD_V2D": [o["false_abstention"]["count"], o["unsupported"]["count"], o["gold_full_recall"]],
                                                                       "NEW_P2_V2D": [n["false_abstention"]["count"], n["unsupported"]["count"], n["gold_full_recall"]]},
                                                                      ymax=40, fmt="{:.0f}"), encoding="utf-8")
    npair, kpair = S["new_person"], S["known_set"]
    items = [("fresh: answerable (primary)", npair["answerable_pair"]["diff"], npair["answerable_pair"]["ci95"]),
             ("fresh: micro", npair["micro_pair"]["diff"], npair["micro_pair"]["ci95"]),
             ("fresh: multi-hop", npair["multi_hop_pair"]["diff"], npair["multi_hop_pair"]["ci95"]),
             ("fresh: temporal", npair["temporal_pair"]["diff"], npair["temporal_pair"]["ci95"]),
             ("fresh: gold recall", npair["recall_pair"]["diff"], npair["recall_pair"]["ci95"]),
             ("known (not indep.): answerable", kpair["answerable_pair"]["diff"], kpair["answerable_pair"]["ci95"])]
    (fig / "p2_paired_ci.svg").write_text(PFA.svg_ci("NEW - OLD, paired bootstrap 95% CI (10,000 resamples)", items, lo=-0.4, hi=0.6, height=320), encoding="utf-8")
    wtl = ["fresh answerable", "fresh micro", "known answerable"]
    (fig / "p2_win_tie_loss.svg").write_text(PFA.svg_bars("Win / tie / loss (NEW vs OLD, per question)", wtl,
                                                          {"win": [npair["answerable_pair"]["wins"], npair["micro_pair"]["wins"], kpair["answerable_pair"]["wins"]],
                                                           "tie": [npair["answerable_pair"]["ties"], npair["micro_pair"]["ties"], kpair["answerable_pair"]["ties"]],
                                                           "loss": [npair["answerable_pair"]["losses"], npair["micro_pair"]["losses"], kpair["answerable_pair"]["losses"]]},
                                                          ymax=100, fmt="{:.0f}"), encoding="utf-8")
    (fig / "p2_latency_tokens.svg").write_text(PFA.svg_bars("Fresh person: retrieval latency (ms) on Colab L4", ["retrieval p50", "retrieval p95", "answer p50", "answer p95"],
                                                            {"OLD_V2D": [o["retrieval_ms"]["p50"], o["retrieval_ms"]["p95"], o["answer_latency_ms"]["p50"], o["answer_latency_ms"]["p95"]],
                                                             "NEW_P2_V2D": [n["retrieval_ms"]["p50"], n["retrieval_ms"]["p95"], n["answer_latency_ms"]["p50"], n["answer_latency_ms"]["p95"]]},
                                                            ymax=max(5000, (n["retrieval_ms"]["p95"] or 0) * 1.1, (n["answer_latency_ms"]["p95"] or 0) * 1.1), fmt="{:.0f}"), encoding="utf-8")
    print(json.dumps({"checks": {k: v for k, v in checks.items() if k not in ("counts",)}, "decision": decision}, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
