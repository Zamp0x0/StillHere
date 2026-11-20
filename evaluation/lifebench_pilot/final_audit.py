#!/usr/bin/env python3
"""Independent final audit of a LifeBench 100-QA pilot Results ZIP (CPU only; no API call, no model).

    python evaluation/lifebench_pilot/final_audit.py --results <Results ZIP> --dataset <official json> --eval-dir <LifeBench_eval checkout>
        --private-out <private dir> --public-out research/lifebench_pilot/final_audit

Independent of the run's own code path: the majority vote, micro / macro, per-type accuracy, judge agreement, paired bootstrap and the
exact McNemar test are implemented HERE from the raw JSONL records (answers, judge, contexts, logs). ``lbpilot.runner.compute`` (the
run's code) is then executed on the same records and both must agree with the exported METRICS.json; any difference is a FAIL.

Public outputs (aggregates only, no question / answer text): LIFEBENCH_FINAL_METRICS.json, AUDIT_CHECKS.json, figures/*.svg.
Private outputs (question, gold, the four answers, judge votes, failure attribution): PER_QUESTION_AUDIT.csv / .jsonl.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evaluation" / "lifebench_pilot"))

from lbpilot import protocol as P  # noqa: E402
from lbpilot import select as SEL  # noqa: E402
from lbpilot import timeline as TL  # noqa: E402

GROUPS = list(P.GROUP_ORDER)
SHORT = {"M0_NO_MEMORY": "No Memory", "M1_STILLHERE_V2L": "StillHere V2L", "M2_STILLHERE_V2D": "StillHere V2D", "M3_MEM0_OSS": "Mem0 OSS"}
PAIRS = [("M1_STILLHERE_V2L", "M0_NO_MEMORY"), ("M2_STILLHERE_V2D", "M1_STILLHERE_V2L"), ("M2_STILLHERE_V2D", "M0_NO_MEMORY"),
         ("M2_STILLHERE_V2D", "M3_MEM0_OSS"), ("M3_MEM0_OSS", "M0_NO_MEMORY"), ("M1_STILLHERE_V2L", "M3_MEM0_OSS")]
PRICE = {"input_miss": 0.30, "input_hit": 0.006, "output": 1.20}          # USD / 1M, DeepSeek peak (off-peak is half): an UPPER bound
SEED, RESAMPLES = 20261119, 10000


def rows(p: Path):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def usd(inp, hit, out):
    inp, hit, out = inp or 0, hit or 0, out or 0
    return (inp - hit) / 1e6 * PRICE["input_miss"] + hit / 1e6 * PRICE["input_hit"] + out / 1e6 * PRICE["output"]


def q(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 2)


def mcnemar_exact(b: int, c: int):
    """Two-sided exact McNemar (binomial on the discordant pairs, p = 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(f"{min(1.0, 2 * tail):.3g}")


def bootstrap(d, resamples=RESAMPLES, seed=SEED):
    rng = random.Random(seed)
    n = len(d)
    bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return [round(bs[int(0.025 * resamples)], 4), round(bs[int(0.975 * resamples) - 1], 4)]


# ------------------------------------------------------------------ svg charts (aggregates only)
def svg_bars(title, labels, series, *, ymax=1.0, fmt="{:.0%}", width=720, height=360, colors=("#4C72B0", "#55A868", "#C44E52", "#8172B2")):
    """Grouped bar chart. ``series``: {name: [values per label]}."""
    pad_l, pad_b, pad_t = 50, 80, 40
    W, H = width - pad_l - 20, height - pad_b - pad_t
    n, m = len(labels), len(series)
    gw = W / max(1, n)
    bw = gw * 0.8 / max(1, m)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="sans-serif" font-size="11">',
           f'<rect width="100%" height="100%" fill="white"/><text x="{width/2}" y="20" text-anchor="middle" font-size="14">{title}</text>']
    for t in range(0, 6):
        v = ymax * t / 5
        y = pad_t + H - H * t / 5
        out.append(f'<line x1="{pad_l}" x2="{pad_l+W}" y1="{y}" y2="{y}" stroke="#ddd"/><text x="{pad_l-5}" y="{y+4}" text-anchor="end">{fmt.format(v)}</text>')
    for i, lab in enumerate(labels):
        for j, (name, vals) in enumerate(series.items()):
            v = vals[i]
            if v is None:
                continue
            h = H * max(0.0, v) / ymax
            x = pad_l + i * gw + gw * 0.1 + j * bw
            out.append(f'<rect x="{x:.1f}" y="{pad_t+H-h:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{colors[j % len(colors)]}"/>')
            out.append(f'<text x="{x+bw/2:.1f}" y="{pad_t+H-h-3:.1f}" text-anchor="middle" font-size="9">{fmt.format(v)}</text>')
        out.append(f'<text x="{pad_l+i*gw+gw/2:.1f}" y="{pad_t+H+14}" text-anchor="middle" font-size="10">{lab}</text>')
    for j, name in enumerate(series):
        out.append(f'<rect x="{pad_l+j*150}" y="{height-22}" width="10" height="10" fill="{colors[j % len(colors)]}"/>'
                   f'<text x="{pad_l+j*150+14}" y="{height-13}">{name}</text>')
    out.append("</svg>")
    return "\n".join(out)


def svg_ci(title, items, *, lo=-0.2, hi=0.6, width=720, height=300):
    """Forest plot: items = [(label, diff, [lo, hi])]."""
    pad_l, pad_r, pad_t = 230, 30, 40
    W = width - pad_l - pad_r
    row = (height - pad_t - 30) / max(1, len(items))
    X = lambda v: pad_l + (v - lo) / (hi - lo) * W  # noqa: E731
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="sans-serif" font-size="11">',
           f'<rect width="100%" height="100%" fill="white"/><text x="{width/2}" y="20" text-anchor="middle" font-size="14">{title}</text>',
           f'<line x1="{X(0)}" x2="{X(0)}" y1="{pad_t-5}" y2="{height-25}" stroke="#999" stroke-dasharray="4"/>']
    for t in range(int(lo * 10), int(hi * 10) + 1, 1):
        v = t / 10
        out.append(f'<text x="{X(v):.1f}" y="{height-10}" text-anchor="middle" font-size="9">{v:+.1f}</text>')
    for i, (lab, d, ci) in enumerate(items):
        y = pad_t + row * i + row / 2
        out.append(f'<text x="{pad_l-8}" y="{y+4}" text-anchor="end">{lab}</text>')
        out.append(f'<line x1="{X(ci[0]):.1f}" x2="{X(ci[1]):.1f}" y1="{y}" y2="{y}" stroke="#333" stroke-width="2"/>')
        out.append(f'<circle cx="{X(d):.1f}" cy="{y}" r="4" fill="#C44E52"/><text x="{X(ci[1])+6:.1f}" y="{y+4}" font-size="9">{d:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]</text>')
    out.append("</svg>")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--private-out", required=True)
    ap.add_argument("--public-out", required=True)
    ap.add_argument("--release-manifest", default=str(ROOT / "releases" / "lifebench_pilot_100qa" / "BENCHMARK_MANIFEST.json"))
    a = ap.parse_args()
    priv, pub = Path(a.private_out), Path(a.public_out)
    raw = priv / "raw"
    (pub / "figures").mkdir(parents=True, exist_ok=True)
    checks = {}

    # ---------------------------------------------------------------- A. integrity
    zb = Path(a.results).read_bytes()
    zsha = hashlib.sha256(zb).hexdigest()
    with zipfile.ZipFile(a.results) as z:
        checks["zip_crc_ok"] = z.testzip() is None
        names = z.namelist()
        if not raw.is_dir():
            z.extractall(raw)
    sums = {l.split("  ", 1)[1].strip(): l.split("  ", 1)[0] for l in (raw / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
    checks["inner_sha_mismatches"] = [n for n, h in sums.items() if hashlib.sha256((raw / n).read_bytes()).hexdigest() != h]
    checks["zip_sha256"] = zsha
    checks["files"] = len(names)
    st = json.loads((raw / "STATE.json").read_text(encoding="utf-8"))
    env = json.loads((raw / "ENVIRONMENT.json").read_text(encoding="utf-8"))
    audit = json.loads((raw / "MEM0_RUNTIME_AUDIT.json").read_text(encoding="utf-8"))
    rel = json.loads(Path(a.release_manifest).read_text(encoding="utf-8"))
    checks["protocol_sha256_run"] = st["protocol_sha256"]
    checks["protocol_matches_current_code"] = st["protocol_sha256"] == P.PROTOCOL_SHA256 == rel["protocol_sha256"] == env["protocol_sha256"]
    checks["stages"] = {k: v["status"] for k, v in st["stages"].items()}
    checks["qa_sha256_run"] = st["stages"]["06_frozen_100qa"]["qa_sha256"]
    checks["timeline_sha256_run"] = st["stages"]["07_timeline"]["sha256"]
    checks["qa_matches_release"] = checks["qa_sha256_run"] == rel["frozen"]["qa_sha256"]
    checks["timeline_matches_release"] = checks["timeline_sha256_run"] == rel["timeline_sha256"]
    data = SEL.load_dataset(a.dataset)                                        # verifies the dataset sha (revision 3d06052d)
    frozen = SEL.freeze_data(data)
    person, qs = SEL.selected(data, frozen)
    events = TL.build(person, until=frozen["ask_time_range"][1])
    checks["dataset_sha256"] = P.DATASET["sha256"]
    checks["recomputed_qa_sha256_matches"] = frozen["qa_sha256"] == checks["qa_sha256_run"]
    checks["recomputed_timeline_sha256_matches"] = TL.sha(events) == checks["timeline_sha256_run"]
    checks["person"] = frozen["person"]["sample_id"]
    checks["events"] = len(events)
    qa = {x["question_id"]: x for x in qs}
    ids = list(qa)
    ans_all, jud_all, ctx_all = rows(raw / "answers.jsonl"), rows(raw / "judge.jsonl"), rows(raw / "contexts.jsonl")
    ans = {(r["group"], r["question_id"]): r for r in ans_all}
    jud = defaultdict(dict)
    for r in jud_all:
        jud[(r["group"], r["question_id"])][r["repeat"]] = r
    ctx = {(r["group"], r["question_id"]): r for r in ctx_all}
    comp = {}
    for g in GROUPS:
        comp[g] = {"answers_ok": sum(1 for i in ids if ans.get((g, i), {}).get("status") == "OK"),
                   "answers_empty": sum(1 for i in ids if not (ans.get((g, i), {}).get("answer") or "").strip()),
                   "judge_ok": sum(1 for i in ids for r in range(3) if jud[(g, i)].get(r, {}).get("status") == "OK"),
                   "contexts_ok": sum(1 for i in ids if ctx.get((g, i), {}).get("status") == "OK")}
    checks["completeness"] = comp
    checks["raw_lines"] = {"answers": len(ans_all), "judge": len(jud_all), "contexts": len(ctx_all)}
    checks["duplicate_keys"] = {"answers": len(ans_all) - len(ans), "judge": len(jud_all) - sum(len(v) for v in jud.values()), "contexts": len(ctx_all) - len(ctx)}
    checks["unknown_question_ids"] = sorted({r["question_id"] for r in ans_all + jud_all + ctx_all} - set(ids))
    checks["judge_request_ids_unique"] = len({r["request_id"] for r in jud_all}) == len(jud_all)
    checks["answer_request_ids_unique"] = len({r["request_id"] for r in ans_all}) == len(ans_all)
    checks["answer_models"] = dict(Counter(f"{r['requested_model']} -> {r['returned_model']}" for r in ans_all))
    checks["judge_models"] = dict(Counter(f"{r['requested_model']} -> {r['returned_model']}" for r in jud_all))
    checks["answer_prompt_versions"] = dict(Counter(r["prompt_version"] for r in ans_all))
    checks["judge_prompt_sha256"] = dict(Counter(r["judge_prompt_sha256"] for r in jud_all))
    checks["answer_attempts"] = dict(Counter(r["attempts"] for r in ans_all))
    checks["judge_parse_attempts"] = dict(Counter(r["parse_attempts"] for r in jud_all))
    checks["answer_context_sha_matches"] = sum(1 for k, r in ans.items() if ctx[k]["context_sha256"] == r["context_sha256"]
                                              and hashlib.sha256(ctx[k]["text"].encode()).hexdigest() == r["context_sha256"])
    # identical-repeat check: three judge calls are separate requests; identical raw text is expected at temperature 0 and is reported
    checks["judge_identical_raw_triplets"] = sum(1 for v in jud.values() if len({x["raw"] for x in v.values()}) == 1)

    # ---------------------------------------------------------------- time visibility / leakage
    day_of = {e["event_id"]: e["date"] for e in events}
    late = Counter()
    mem0_empty = 0
    for (g, i), c in ctx.items():
        ask = qa[i]["ask_time"][:10]
        if c.get("future_messages_in_store") not in (None, 0):
            late[f"{g}:store"] += 1
        for e in c["kept"]:
            if e.get("date") and e["date"] > ask:
                late[f"{g}:entry_date"] += 1
            for x in e.get("ids", []):
                ev = x.split(".")[-1] if x.startswith("lb2.") else x
                d = day_of.get(ev)
                if d and d > ask:
                    late[f"{g}:source_id"] += 1
        if g == "M3_MEM0_OSS" and c["entries_in"] == 0:
            mem0_empty += 1
        if g == "M0_NO_MEMORY" and (c["entries_in"] != 0 or "【记忆记录】" in c["text"]):
            late["M0:context_not_empty"] += 1
    checks["leakage_counts"] = dict(late)
    checks["mem0_empty_searches"] = mem0_empty
    checks["max_evidence_tokens"] = max(c["tokens"] for c in ctx_all)
    checks["over_budget"] = sum(1 for c in ctx_all if c["tokens"] > P.EVIDENCE["token_budget"])
    checks["same_query_all_groups"] = all(len({ctx[(g, i)]["query"] for g in GROUPS}) == 1 for i in ids)
    checks["same_ask_time_all_groups"] = all(len({ctx[(g, i)]["ask_time"] for g in GROUPS}) == 1 for i in ids)
    px = rows(raw / "logs" / "mem0_llm_proxy.jsonl")
    checks["mem0_proxy"] = {"calls": len(px), "status": dict(Counter(str(r.get("status")) for r in px)), "returned_models": dict(Counter(r.get("returned_model") for r in px)),
                            "violations": sum(1 for r in px if r.get("status") == "MODEL_VIOLATION")}
    checks["mem0_runtime"] = {k: audit[k] for k in ("mem0ai_version", "llm_class", "llm_model", "embedder_class", "embedding_dims", "vector_store_class",
                                                    "openai_api_key_set", "telemetry_env", "infer", "add_batch_size", "search_top_k")}
    checks["mem0_state"] = {"add": st.get("mem0_add"), "stats": st.get("mem0_stats"), "memories_final": st.get("mem0_memories_final")}
    checks["environment"] = env

    # ---------------------------------------------------------------- B. independent recomputation
    def majority(g, i):
        v = jud[(g, i)]
        votes = [bool(v[r]["is_correct"]) for r in range(3)]
        return sum(votes) >= 2, votes
    per = {g: {} for g in GROUPS}
    agree = {g: Counter() for g in GROUPS}
    for g in GROUPS:
        for i in ids:
            ok, votes = majority(g, i)
            per[g][i] = ok
            agree[g]["3:0" if len(set(votes)) == 1 else "2:1"] += 1
    met = {}
    for g in GROUPS:
        corr = sum(per[g].values())
        bytype = {}
        for t in P.OFFICIAL_TYPES:
            sub = [i for i in ids if t in (qa[i]["question_type"] or [])]
            bytype[t] = {"n": len(sub), "correct": sum(per[g][i] for i in sub), "accuracy": round(sum(per[g][i] for i in sub) / len(sub), 4) if sub else None}
        present = [v["accuracy"] for v in bytype.values() if v["accuracy"] is not None]
        uaq = [i for i in ids if "Unanswerable" in qa[i]["question_type"]]
        anq = [i for i in ids if i not in uaq]
        abst = {i: bool(ans[(g, i)].get("abstained")) for i in ids}
        cg = [ctx[(g, i)] for i in ids]
        au = [ans[(g, i)] for i in ids]
        ju = [jud[(g, i)][r] for i in ids for r in range(3)]
        met[g] = {
            "micro": {"correct": corr, "n": len(ids), "accuracy": round(corr / len(ids), 4), "missing_or_failed": len(ids) - comp[g]["answers_ok"]},
            "macro": {"accuracy": round(sum(present) / len(present), 4), "types_present": len(present), "na": [t for t, v in bytype.items() if v["accuracy"] is None]},
            "by_type": bytype,
            "answerable": {"correct": sum(per[g][i] for i in anq), "n": len(anq)}, "unanswerable": {"correct": sum(per[g][i] for i in uaq), "n": len(uaq)},
            "false_abstention": {"count": sum(abst[i] for i in anq), "n": len(anq)},
            "unsupported_claim": {"count": sum(1 for i in ids if not per[g][i] and not abst[i]), "n": len(ids),
                                  "on_unanswerable": sum(1 for i in uaq if not per[g][i] and not abst[i])},
            "judge_agreement": {"unanimous": agree[g]["3:0"], "split_2_1": agree[g]["2:1"]},
            "evidence_tokens": {"mean": round(sum(c["tokens"] for c in cg) / len(cg), 1), "p95": q([c["tokens"] for c in cg], .95), "max": max(c["tokens"] for c in cg),
                                "entries_kept_mean": round(sum(c["entries_kept"] for c in cg) / len(cg), 1), "contexts_truncated": sum(1 for c in cg if c["entries_dropped"] > 0)},
            "latency_ms": {"retrieval_p50": q([c["retrieval_ms"] for c in cg], .5), "retrieval_p95": q([c["retrieval_ms"] for c in cg], .95),
                           "answer_p50": q([r["latency_ms"] for r in au], .5), "answer_p95": q([r["latency_ms"] for r in au], .95),
                           "end_to_end_p50": q([(c["retrieval_ms"] or 0) + (r["latency_ms"] or 0) for c, r in zip(cg, au)], .5)},
            "answer_tokens": {"input": sum(r["input_tokens"] for r in au), "cache_hit": sum(r.get("cache_hit_tokens") or 0 for r in au), "output": sum(r["output_tokens"] for r in au)},
            "judge_tokens": {"input": sum(r["input_tokens"] for r in ju), "cache_hit": sum(r.get("cache_hit_tokens") or 0 for r in ju), "output": sum(r["output_tokens"] for r in ju)},
        }
        met[g]["answer_cost_usd_upper"] = round(usd(met[g]["answer_tokens"]["input"], met[g]["answer_tokens"]["cache_hit"], met[g]["answer_tokens"]["output"]), 4)
        met[g]["judge_cost_usd_upper"] = round(usd(met[g]["judge_tokens"]["input"], met[g]["judge_tokens"]["cache_hit"], met[g]["judge_tokens"]["output"]), 4)
    # memory construction
    mp = st.get("mem0_proxy") or {}
    build = {"M0_NO_MEMORY": {"llm_calls": 0, "cost_usd_upper": 0.0, "seconds": 0.0},
             "M1_STILLHERE_V2L": {"llm_calls": 0, "cost_usd_upper": 0.0, "seconds": round(st["timing"]["stillhere_add_s"], 1)},
             "M2_STILLHERE_V2D": {"llm_calls": 0, "cost_usd_upper": 0.0, "seconds": round(st["timing"]["stillhere_add_s"] + st["stages"]["09_stillhere_dense_index"]["seconds"], 1),
                                  "note": "+ dense index over the whole timeline on the L4 (7,097 documents, embedding)"},
             "M3_MEM0_OSS": {"llm_calls": mp.get("calls"), "input_tokens": mp.get("input_tokens"), "cache_hit_tokens": mp.get("cache_hit_tokens"),
                             "output_tokens": mp.get("output_tokens"), "cost_usd_upper": round(usd(mp.get("input_tokens"), mp.get("cache_hit_tokens"), mp.get("output_tokens")), 4),
                             "seconds": round(st["timing"]["mem0_add_s"], 1), "memories": st.get("mem0_memories_final"),
                             "note": "Mem0 ADD = LLM extraction + embedding + Qdrant insert; 872 proxy calls for 871 batches (1 re-sent request after a client timeout)"}}
    for g in GROUPS:
        met[g]["memory_build"] = build[g]
        met[g]["total_cost_usd_upper"] = round(met[g]["answer_cost_usd_upper"] + met[g]["judge_cost_usd_upper"] + build[g]["cost_usd_upper"], 4)
        met[g]["system_cost_usd_upper"] = round(met[g]["answer_cost_usd_upper"] + build[g]["cost_usd_upper"], 4)

    # pairwise
    pairs = {}
    for A_, B_ in PAIRS:
        d = [int(per[A_][i]) - int(per[B_][i]) for i in ids]
        b = sum(1 for x in d if x > 0)
        c = sum(1 for x in d if x < 0)
        ci = bootstrap(d)
        by = {}
        for t in P.OFFICIAL_TYPES:
            sub = [i for i in ids if t in qa[i]["question_type"]]
            if sub:
                by[t] = {"n": len(sub), "diff": round(sum(int(per[A_][i]) - int(per[B_][i]) for i in sub) / len(sub), 4),
                         "wins": sum(1 for i in sub if per[A_][i] and not per[B_][i]), "losses": sum(1 for i in sub if per[B_][i] and not per[A_][i])}
        pairs[f"{A_}__minus__{B_}"] = {"diff_pp": round(100 * sum(d) / len(d), 1), "ci95_pp": [round(100 * ci[0], 1), round(100 * ci[1], 1)],
                                       "wins": b, "ties": len(d) - b - c, "losses": c, "mcnemar_exact_p": mcnemar_exact(b, c),
                                       "significant_ci": ci[0] > 0 or ci[1] < 0, "by_type": by}

    # cross-check: the run's METRICS.json and the run's own compute() on the same records
    exported = json.loads((raw / "METRICS.json").read_text(encoding="utf-8"))
    cross = {}
    for g in GROUPS:
        e = exported["groups"][g]
        cross[g] = {"micro_equal": abs(e["micro_accuracy"] - met[g]["micro"]["accuracy"]) < 1e-9, "macro_equal": abs(e["macro_accuracy"] - met[g]["macro"]["accuracy"]) < 1e-4,
                    "by_type_equal": all(e["by_type"][t]["correct"] == met[g]["by_type"][t]["correct"] for t in P.OFFICIAL_TYPES)}
    from lbpilot import judge as J
    from lbpilot import runner as R
    official = J.Official(Path(a.eval_dir))
    checks["official_prompt_sha_matches"] = official.prompt_sha256 == J.OFFICIAL_PROMPT_SHA256
    again = R.compute(qs, events, {(r["group"], r["question_id"]): r for r in ctx_all}, ans, {(r["group"], r["question_id"], r["repeat"]): r for r in jud_all}, official)
    cross["runner_compute_equals_export"] = again["groups"] == exported["groups"] and again["pairs"] == exported["pairs"]
    checks["cross_check"] = cross

    # ---------------------------------------------------------------- E. per-question audit (private) + failure attribution
    from lbpilot import metrics as MT
    gold_text = defaultdict(list)
    for e in events:
        gold_text[e["gold_id"]].append(e)

    def coverage(g, i):
        ev = qa[i].get("evidence") or []
        if not ev or g == "M0_NO_MEMORY":
            return None
        c = ctx[(g, i)]
        ask = qa[i]["ask_time"][:10]
        if g in ("M1_STILLHERE_V2L", "M2_STILLHERE_V2D"):
            kept = {TL.gold_id(x.split(":t", 1)[0]) for e in c["kept"] for x in e.get("ids", []) if "." not in x}
            return sum(1 for x in ev if x in kept) / len(ev)
        return sum(1 for x in ev if any(MT.covered(t["content"], c["text"]) for t in gold_text.get(x, []) if t["date"] <= ask)) / len(ev)

    for g in GROUPS:                                                      # official weighted score (point hits by the same majority) + evidence
        ws, cov_all = [], []
        for i in ids:
            sp = qa[i].get("score_points") or []
            tot = sum(float((p or {}).get("score", 0) or 0) for p in sp)
            if sp and tot > 0:
                got = 0.0
                for k, p_ in enumerate(sp):
                    hits = 0
                    for r in range(3):
                        h = jud[(g, i)][r].get("point_hits") or []
                        if k < len(h):
                            hits += int(h[k].get("hit", False) if isinstance(h[k], dict) else bool(h[k]))
                    if hits * 2 >= 3:
                        got += float((p_ or {}).get("score", 0) or 0)
                ws.append(got / tot)
            c_ = coverage(g, i)
            if c_ is not None:
                cov_all.append(c_)
        met[g]["weighted_score_mean"] = round(sum(ws) / len(ws), 4) if ws else None
        met[g]["evidence_coverage_mean"] = round(sum(cov_all) / len(cov_all), 4) if cov_all else None
        met[g]["evidence_coverage_kind"] = {"M0_NO_MEMORY": None, "M1_STILLHERE_V2L": "strict record-id recall", "M2_STILLHERE_V2D": "strict record-id recall",
                                            "M3_MEM0_OSS": "semantic bigram coverage (heuristic; penalises paraphrased facts)"}[g]
    attribution = {g: Counter() for g in GROUPS}
    table = []
    for i in ids:
        x = qa[i]
        rec = {"question_id": i, "types": "|".join(x["question_type"]), "ask_time": x["ask_time"], "question": x["question"], "gold": x["answer"],
               "evidence": "|".join(x.get("evidence") or [])}
        for g in GROUPS:
            ok, votes = majority(g, i)
            cov = coverage(g, i)
            a_ = ans[(g, i)]
            ua = "Unanswerable" in x["question_type"]
            if ok:
                why = "CORRECT"
            elif ua:
                why = "FABRICATED_ON_UNANSWERABLE"
            elif g == "M0_NO_MEMORY":
                why = "NO_MEMORY_ABSTAINED" if a_.get("abstained") else "NO_MEMORY_GUESS"
            elif cov is None:
                why = "UNRESOLVED_NO_GOLD_IDS"
            elif cov == 0:
                why = "RETRIEVAL_MISSING"
            elif cov < 1:
                why = "PARTIAL_EVIDENCE"
            elif a_.get("abstained"):
                why = "INCORRECT_ABSTENTION_WITH_EVIDENCE"
            elif any(t in x["question_type"] for t in ("Temporal", "Knowledge_update", "Conflict")):
                why = "TEMPORAL_OR_UPDATE_REASONING"
            else:
                why = "READER_REASONING"
            if not ok and votes.count(True) == 1:
                why += "+JUDGE_SPLIT"
            attribution[g][why] += 1
            rec.update({f"{g}:answer": a_["answer"], f"{g}:votes": "".join("1" if v else "0" for v in votes), f"{g}:correct": int(ok),
                        f"{g}:coverage": None if cov is None else round(cov, 3), f"{g}:abstained": int(bool(a_.get("abstained"))), f"{g}:attribution": why})
        table.append(rec)
    priv.mkdir(parents=True, exist_ok=True)
    with open(priv / "PER_QUESTION_AUDIT.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    (priv / "PER_QUESTION_AUDIT.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in table), encoding="utf-8")
    contrast = {}
    for name, cond in (("V2D_right_V2L_wrong", lambda i: per["M2_STILLHERE_V2D"][i] and not per["M1_STILLHERE_V2L"][i]),
                       ("V2L_right_V2D_wrong", lambda i: per["M1_STILLHERE_V2L"][i] and not per["M2_STILLHERE_V2D"][i]),
                       ("Mem0_right_both_StillHere_wrong", lambda i: per["M3_MEM0_OSS"][i] and not per["M1_STILLHERE_V2L"][i] and not per["M2_STILLHERE_V2D"][i]),
                       ("StillHere_V2D_right_Mem0_wrong", lambda i: per["M2_STILLHERE_V2D"][i] and not per["M3_MEM0_OSS"][i]),
                       ("Mem0_right_V2D_wrong", lambda i: per["M3_MEM0_OSS"][i] and not per["M2_STILLHERE_V2D"][i]),
                       ("all_four_wrong", lambda i: not any(per[g][i] for g in GROUPS)),
                       ("all_memory_groups_wrong", lambda i: not any(per[g][i] for g in GROUPS[1:])),
                       ("no_memory_right", lambda i: per["M0_NO_MEMORY"][i]),
                       ("no_memory_right_on_answerable", lambda i: per["M0_NO_MEMORY"][i] and "Unanswerable" not in qa[i]["question_type"])):
        sel = [i for i in ids if cond(i)]
        contrast[name] = {"count": len(sel), "types": dict(Counter(qa[i]["question_type"][0] for i in sel))}
    split_any = sum(1 for i in ids if any(len(set(majority(g, i)[1])) > 1 for g in GROUPS))

    # ---------------------------------------------------------------- outputs
    final = {"schema": "stillhere.lifebench_pilot.final_audit/1.0", "scope": P.SCOPE, "results_zip_sha256": zsha, "protocol_sha256": st["protocol_sha256"],
             "dataset": {"repo": P.DATASET["repo"], "revision": P.DATASET["revision"], "sha256": P.DATASET["sha256"]}, "person": checks["person"],
             "questions": len(ids), "events": len(events), "price_basis": "DeepSeek deepseek-flash peak USD/1M (input miss 0.30, hit 0.006, output 1.20); off-peak is half: UPPER bounds",
             "bootstrap": {"resamples": RESAMPLES, "seed": SEED, "unit": "question (paired)"}, "groups": met, "pairs": pairs,
             "failure_attribution": {g: dict(v) for g, v in attribution.items()}, "contrasts": contrast, "questions_with_any_judge_split": split_any}
    integrity_ok = (checks["zip_crc_ok"] and not checks["inner_sha_mismatches"] and checks["protocol_matches_current_code"] and checks["recomputed_qa_sha256_matches"]
                    and checks["recomputed_timeline_sha256_matches"] and all(v == "PASS" for v in checks["stages"].values())
                    and all(c["answers_ok"] == 100 and c["judge_ok"] == 300 and c["contexts_ok"] == 100 and c["answers_empty"] == 0 for c in comp.values())
                    and not any(checks["duplicate_keys"].values()) and not checks["unknown_question_ids"] and not checks["leakage_counts"] and checks["over_budget"] == 0
                    and checks["mem0_proxy"]["violations"] == 0 and all(v for g in GROUPS for v in cross[g].values()) and cross["runner_compute_equals_export"]
                    and checks["official_prompt_sha_matches"])
    checks["LIFEBENCH_INTEGRITY"] = "PASS" if integrity_ok else "FAIL"
    final["LIFEBENCH_INTEGRITY"] = checks["LIFEBENCH_INTEGRITY"]
    (pub / "LIFEBENCH_FINAL_METRICS.json").write_text(json.dumps(final, ensure_ascii=False, indent=1), encoding="utf-8")
    (pub / "AUDIT_CHECKS.json").write_text(json.dumps(checks, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    labels = [SHORT[g] for g in GROUPS]
    figs = pub / "figures"
    figs.joinpath("01_micro_macro.svg").write_text(svg_bars("Micro / Macro accuracy (100 QA, judge x3 majority)", labels,
                                                            {"Micro": [met[g]["micro"]["accuracy"] for g in GROUPS], "Macro": [met[g]["macro"]["accuracy"] for g in GROUPS]}), encoding="utf-8")
    tl = [t.replace("Pattern_recognition(Non-declarative)", "Pattern").replace("Knowledge_update", "K-update") for t in P.OFFICIAL_TYPES]
    figs.joinpath("02_by_type.svg").write_text(svg_bars("Accuracy by official question type (multi-label; n in parentheses)",
                                                        [f"{t} ({met[GROUPS[0]]['by_type'][o]['n']})" for t, o in zip(tl, P.OFFICIAL_TYPES)],
                                                        {SHORT[g]: [met[g]["by_type"][t]["accuracy"] for t in P.OFFICIAL_TYPES] for g in GROUPS}, width=1100), encoding="utf-8")
    figs.joinpath("03_paired_ci.svg").write_text(svg_ci("Paired accuracy difference, 95% bootstrap CI (10,000 resamples)",
                                                        [(k.replace("__minus__", " - ").replace("M1_STILLHERE_", "").replace("M2_STILLHERE_", "").replace("M0_NO_MEMORY", "NoMem")
                                                          .replace("M3_MEM0_OSS", "Mem0"), v["diff_pp"] / 100, [v["ci95_pp"][0] / 100, v["ci95_pp"][1] / 100]) for k, v in pairs.items()]),
                                                 encoding="utf-8")
    figs.joinpath("04_cost.svg").write_text(svg_bars("System cost per 100 QA, USD upper bound (memory build + answers; judge excluded)", labels,
                                                     {"memory build": [build[g]["cost_usd_upper"] for g in GROUPS], "answers": [met[g]["answer_cost_usd_upper"] for g in GROUPS]},
                                                     ymax=max(1.5, max(met[g]["system_cost_usd_upper"] for g in GROUPS) * 1.1), fmt="${:.2f}"), encoding="utf-8")
    figs.joinpath("05_latency.svg").write_text(svg_bars("Latency p50 (ms): retrieval vs answer", labels,
                                                        {"retrieval": [met[g]["latency_ms"]["retrieval_p50"] for g in GROUPS], "answer": [met[g]["latency_ms"]["answer_p50"] for g in GROUPS]},
                                                        ymax=max(met[g]["latency_ms"]["answer_p50"] for g in GROUPS) * 1.2, fmt="{:.0f}"), encoding="utf-8")
    figs.joinpath("06_judge_agreement.svg").write_text(svg_bars("Judge agreement per question (3 runs)", labels,
                                                                {"unanimous 3:0": [met[g]["judge_agreement"]["unanimous"] / 100 for g in GROUPS],
                                                                 "split 2:1": [met[g]["judge_agreement"]["split_2_1"] / 100 for g in GROUPS]}), encoding="utf-8")
    cats = sorted({k for v in attribution.values() for k in v if k != "CORRECT"})
    figs.joinpath("07_failure_attribution.svg").write_text(svg_bars("Wrong answers by attributed cause (rule-based, see report)", [c.replace("_", " ")[:22] for c in cats],
                                                                     {SHORT[g]: [attribution[g].get(c, 0) / 100 for c in cats] for g in GROUPS}, ymax=0.6, width=1300,
                                                                     fmt="{:.0%}"), encoding="utf-8")
    print(json.dumps({"LIFEBENCH_INTEGRITY": checks["LIFEBENCH_INTEGRITY"], "micro": {g: met[g]["micro"] for g in GROUPS},
                      "macro": {g: met[g]["macro"]["accuracy"] for g in GROUPS}, "pairs": {k: {x: v[x] for x in ("diff_pp", "ci95_pp", "wins", "ties", "losses", "mcnemar_exact_p")} for k, v in pairs.items()},
                      "cross_check": cross, "leakage": checks["leakage_counts"]}, ensure_ascii=False, indent=1))
    return 0 if integrity_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
