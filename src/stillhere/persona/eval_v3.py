"""Third-round Colab evaluation: the Reader on the AI-verified NEW gold set (``AI_VERIFIED_NEW_GOLD``), memory groups only.

Purpose: separate "retrieval found the evidence" from "the Reader answered correctly" on questions nobody tuned on (DEV was used for
tuning, the HOLDOUT for the registered decision). The gold is AI-verified (two AI review passes), NOT human gold: every report says so.

Groups (frozen configurations; the same Answer model, Reader prompt, judge rubric, ask time and evidence archive for all):

    M1                     V1 baseline (router v1, lexical EF1, k = 8)
    V2L                    the current default (MEMORY_V2_RULE: ADOPT_V2 / V2L)
    V2D                    V2 hybrid + windows (the original V2D reference)
    BEST_STABLE_V2         ``memory_v2.FUTURE_CANDIDATES["BEST_STABLE_V2"]``: V2L + question-form removal
    BEST_EVIDENCE_FOCUSED  ``memory_v2.FUTURE_CANDIDATES["BEST_EVIDENCE_FOCUSED"]``: V2D_K40 (40 hybrid anchors, no windows)

Rule (registered 2026-11-19, before any answer exists): a candidate replaces V2L only if, on the new set, its judge-score difference to V2L
has a 95 % CI lower bound > -2 AND a mean > 0, its unsupported-claim rate is not higher and its attribution / temporal error count is not
higher. Otherwise V2L stays. Because the gold is AI-verified, the verdict is ENGINEERING evidence; it does not make the set a test set for
further tuning (no re-tuning on these questions).
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from . import colab_eval as CE
from . import eval_v2 as EV2
from . import judge as J
from . import qa_builder as QB
from . import store as S

SCHEMA = "stillhere.persona.eval_v3/1.0"
GOLD_NAME = "AI_VERIFIED_NEW_GOLD"


def _specs() -> Dict[str, dict]:
    from . import memory_v2 as V2
    fc = V2.FUTURE_CANDIDATES
    best_stable = {"memory": "production", "pipeline": "v2", "adapter": None, "dense": bool(fc["BEST_STABLE_V2"]["use_dense"]),
                   "strip_qform": bool(fc["BEST_STABLE_V2"]["strip_qform"])}
    best_ev = {"memory": "production", "pipeline": "v2", "adapter": None, "dense": bool(fc["BEST_EVIDENCE_FOCUSED"]["use_dense"]),
               "k": int(fc["BEST_EVIDENCE_FOCUSED"]["k"]), "neighbors": tuple(fc["BEST_EVIDENCE_FOCUSED"]["neighbors"])}
    return {"M1": dict(EV2.MEMORY_SPECS["M1"]), "V2L": dict(EV2.MEMORY_SPECS["V2L"]), "V2D": dict(EV2.MEMORY_SPECS["V2D"]),
            "BEST_STABLE_V2": best_stable, "BEST_EVIDENCE_FOCUSED": best_ev}


MEMORY_SPECS_V3: Dict[str, dict] = _specs()
PAIRS = (("V2L", "M1"), ("BEST_STABLE_V2", "V2L"), ("BEST_EVIDENCE_FOCUSED", "V2L"), ("BEST_EVIDENCE_FOCUSED", "BEST_STABLE_V2"), ("V2D", "V2L"),
         ("BEST_EVIDENCE_FOCUSED", "V2D"))
CANDIDATES = ("BEST_STABLE_V2", "BEST_EVIDENCE_FOCUSED")
RULE_V3 = ("registered 2026-11-19 before any answer: a candidate replaces V2L iff (score diff to V2L: CI95 lower > -2 and mean > 0) and "
           "unsupported-claim rate <= V2L and attribution / temporal errors <= V2L; else V2L stays. AI-verified gold: engineering evidence only.")
SUBJECTS = {"self": "PERSONA_SELF", "counterpart": "CONVERSATION_COUNTERPART", "joint": "JOINT"}


def defaults(cfg: Mapping) -> dict:
    base = {"eval_mode": "V3", "memory_groups": list(MEMORY_SPECS_V3), "seed": 20261119, "judge_concurrency": 4, "judge_per_second": 4.0,
            "dense_device": "cuda", "dense_batch": 64, "memory_build": "evidence_only"}
    return {**base, **dict(cfg)}


def to_eval_rows(gold_rows: List[Mapping], evidence_text: Mapping[str, Mapping]) -> List[dict]:
    """AI_VERIFIED_NEW_GOLD rows -> the frozen-QA layout the runtime / judge read (``qa_builder`` / ``colab_eval.memory_units``)."""
    out = []
    for g in gold_rows:
        subj = g["subject_id"]
        ev = g.get("evidence_message_ids") or []
        facts = [f"{evidence_text[m]['time_local'][:16].replace('T', ' ')} {evidence_text[m]['speaker']}：{evidence_text[m]['text']}" for m in ev if m in evidence_text]
        out.append({"question_id": g["question_id"], "question": g["question"], "category": g["category"], "answerability": g["answerability"],
                    "subject": SUBJECTS.get(subj, "THIRD_PARTY" if subj.startswith("third_party") else "UNKNOWN_SUBJECT"),
                    "evidence_message_ids": list(ev), "reference_answer": g["answer"], "reference_facts": facts,
                    "unanswerable_basis": g["answer"] if g["answerability"] != "ANSWERABLE" else None,
                    "time_scope": {"ask_time": g["ask_time"], "evidence_dates": sorted({t[:10] for t in g.get("relevant_timestamps") or []})},
                    "verification_status": "AI_VERIFIED", "gold_source": g.get("gold_source", "AI_VERIFIED")})
    return out


def seed_frozen(run: CE.Run, *, gold_jsonl: Path, gold_manifest: Path, eval_rows_jsonl: Path) -> dict:
    """Copy the frozen AI-verified set into this run (never overwritten with different bytes); the sha must equal the gold manifest."""
    R = Path(run.run_dir)
    man = json.loads(Path(gold_manifest).read_text(encoding="utf-8"))
    gold_b = Path(gold_jsonl).read_bytes()
    if hashlib.sha256(gold_b).hexdigest() != man["sha256"] or man.get("gold_source") != "AI_VERIFIED" or man.get("is_human_gold") is not False:
        raise RuntimeError("GOLD_CHANGED: not the frozen AI_VERIFIED_NEW_GOLD set of the manifest")
    rows_b = Path(eval_rows_jsonl).read_bytes()
    ids_gold = [json.loads(l)["question_id"] for l in gold_b.decode("utf-8").splitlines() if l.strip()]
    ids_rows = [json.loads(l)["question_id"] for l in rows_b.decode("utf-8").splitlines() if l.strip()]
    if ids_gold != ids_rows:
        raise RuntimeError("EVAL_ROWS_MISMATCH: the evaluation rows are not the frozen gold questions")
    ask = sorted({json.loads(l)["ask_time"] for l in gold_b.decode("utf-8").splitlines() if l.strip()})
    if len(ask) != 1:
        raise RuntimeError("one ask time per evaluation set")
    qsha = EV2._put(R / "qa" / "REAL_HISTORY_QA_200.jsonl", rows_b)          # the loader's file name; content = the new set
    EV2._put(R / "qa" / "QA_FROZEN.sha256", f"{qsha}  REAL_HISTORY_QA_200.jsonl\n".encode())
    EV2._put(R / "qa" / "QA_BUILD_REPORT.json", json.dumps({"ask_time": ask[0], "source": GOLD_NAME, "gold_sha256": man["sha256"]}, indent=1).encode())
    EV2._put(R / "reports" / "NEW_GOLD_MANIFEST.json", Path(gold_manifest).read_bytes())
    run.status("V3 STAGE 06 frozen AI-verified gold", "PASS", gold_sha256=man["sha256"], questions=len(ids_gold), ask_time=ask[0])
    return {"gold_sha256": man["sha256"], "rows_sha256": qsha, "questions": len(ids_gold)}


def stage_memory(run: CE.Run, *, dense=None, groups: Optional[List[str]] = None) -> dict:
    return EV2.stage_memory(run, dense=dense, groups=groups or list(MEMORY_SPECS_V3), specs=MEMORY_SPECS_V3)


def stage_judge(run: CE.Run, env: Mapping[str, str], transport=None) -> dict:
    client, model = CE.judge_client(run, env, transport)
    cal = J.calibrate(client, model, run.p("judge", "CALIBRATION_SCORES.jsonl"), log=run.log)
    CE._write(run.p("reports", "JUDGE_CALIBRATION_REPORT.json"), cal)
    units = CE.memory_units(run)
    CE._write(run.p("judge", "memory_unit_key.json"), {u["unit_id"]: {"question_id": u["question_id"], "group": u["_group"]} for u in units})
    res = J.run(client, model, "memory", [{k: v for k, v in u.items() if not k.startswith("_")} for u in units], run.p("judge", "MEMORY_DEEPSEEK_SCORES.jsonl"),
                seed=run.manifest["run_salt"], concurrency=int(run.cfg["judge_concurrency"]), per_second=float(run.cfg["judge_per_second"]), log=run.log)
    out = {"calibration": f"{cal['passed']}/{cal['total']}", "memory": {"ok": res["ok"], "failed": res["failed"]}}
    run.status("V3 STAGE 14-15 judge", "PASS" if res["failed"] == 0 else "PARTIAL", **out)
    return out


def _abstained(a: Mapping) -> bool:
    return a.get("final_check") in ("UNCERTAINTY_KEPT", "FALLBACK_MINIMAL_UNCERTAIN") or a.get("reader_fail") == "READER_ABSTAINED" or CE._abstains(a.get("text"))


def group_metrics(rs: Mapping[str, dict], ans: Mapping[str, dict], by_q: Mapping[str, dict]) -> dict:
    """Judge-based answer metrics next to retrieval (gold evidence recall) and cost, for one group."""
    A = [q for q in rs if by_q[q]["answerability"] == "ANSWERABLE"]
    U = [q for q in rs if by_q[q]["answerability"] != "ANSWERABLE"]
    j = lambda q, k: rs[q]["judgement"].get(k)                                         # noqa: E731
    rate = lambda xs, f: round(sum(1 for x in xs if f(x)) / len(xs), 4) if xs else None  # noqa: E731
    correct = lambda q: j(q, "factual_correctness") >= 4                                # noqa: E731
    found = [q for q in A if (ans.get(q, {}).get("gold_retrieved") or 0) > 0]
    missed = [q for q in A if q not in found]
    tok = lambda k: [((ans.get(q) or {}).get("tokens") or {}).get(k) for q in rs]      # noqa: E731
    mean = lambda xs: round(sum(x for x in xs if x is not None) / max(1, sum(x is not None for x in xs)), 1) if any(x is not None for x in xs) else None  # noqa: E731
    lat = sorted(x for x in tok("reader_latency_s") if x is not None)
    by_cat = {}
    for c in sorted({by_q[q]["category"] for q in rs}):
        qq = [q for q in rs if by_q[q]["category"] == c]
        by_cat[c] = {"n": len(qq), "score": J.describe([rs[q]["score_100"] for q in qq]).get("mean"),
                     "answer_accuracy": rate([q for q in qq if q in A], correct) if any(q in A for q in qq) else None}
    return {"n": len(rs), "answer_accuracy": rate(A, correct),
            "grounded_accuracy": rate(A, lambda q: correct(q) and j(q, "evidence_grounding") >= 4),
            "gold_evidence_recall": round(sum(ans.get(q, {}).get("gold_retrieved") or 0 for q in A) / len(A), 4) if A else None,
            "accuracy_when_gold_retrieved": rate(found, correct), "accuracy_when_gold_missed": rate(missed, correct),
            "n_gold_retrieved": len(found), "n_gold_missed": len(missed),
            "false_abstention_rate": rate(A, lambda q: _abstained(ans.get(q, {}))),
            "correct_abstention_rate_unanswerable": rate(U, lambda q: _abstained(ans.get(q, {}))),
            "unsupported_claim_rate": rate(list(rs), lambda q: bool(j(q, "fabricated_private_fact"))),
            "attribution_temporal_errors": sum(1 for q in rs if j(q, "temporal_attribution") <= 2),
            "attribution_temporal_ok_rate": rate(list(rs), lambda q: j(q, "temporal_attribution") >= 4),
            "attribution_gate_rejections": sum(1 for q in rs if (ans.get(q) or {}).get("reader_fail") == "SUBJECT_RESOLUTION_FAIL"),
            "input_tokens_reader": mean(tok("reader_prompt")), "output_tokens_reader": mean(tok("reader_new")),
            "output_tokens_reply": mean(tok("reply_new")), "reader_latency_p50_s": lat[len(lat) // 2] if lat else None,
            "reader_latency_p95_s": lat[min(len(lat) - 1, int(.95 * len(lat)))] if lat else None, "by_category": by_cat}


def decision(groups: Mapping[str, dict], paired: Mapping[str, dict], n_questions: Optional[int] = None) -> dict:
    """``n_questions``: the frozen set size; a group (or V2L) with fewer scored answers is INCOMPLETE and gets no verdict (the first V3 run
    gave BEST_EVIDENCE_FOCUSED a verdict from 7 / 35 items after 28 generations failed with a disk I/O error)."""
    base = groups.get("V2L")
    out = {}
    for c in CANDIDATES:
        p, g = paired.get(f"{c}-V2L"), groups.get(c)
        if not (p and g and base) or p.get("n", 0) == 0:
            out[c] = {"verdict": "NOT_EVALUATED"}
            continue
        if n_questions is not None and (g.get("n", 0) < n_questions or base.get("n", 0) < n_questions or p.get("n", 0) < n_questions):
            out[c] = {"verdict": "INCOMPLETE", "scored": g.get("n", 0), "of": n_questions, "paired": p}
            continue
        lo = (p.get("ci95") or [None, None])[0]
        ok = (lo is not None and lo > -2 and p.get("mean_diff", p.get("diff", 0)) > 0
              and (g["unsupported_claim_rate"] or 0) <= (base["unsupported_claim_rate"] or 0)
              and g["attribution_temporal_errors"] <= base["attribution_temporal_errors"])
        out[c] = {"verdict": "REPLACES_V2L" if ok else "V2L_STAYS", "paired": p}
    win = [c for c in CANDIDATES if out[c]["verdict"] == "REPLACES_V2L"]
    pending = [c for c in CANDIDATES if out[c]["verdict"] in ("INCOMPLETE", "NOT_EVALUATED")]
    rec = win[0] if len(win) == 1 else ("COMPARE_" + "_VS_".join(win) if win else ("V2L_PENDING_" + "_".join(pending) if pending else "V2L"))
    return {"rule": RULE_V3, "per_candidate": out, "recommendation": rec,
            "gold": "AI_VERIFIED (not human)"}


def analysis(run: CE.Run) -> dict:
    mem = CE._scores(run, "memory", "memory_unit_key.json")
    rows, _ = QB.load_frozen(Path(run.run_dir) / "qa")
    by_q = {q["question_id"]: q for q in rows}
    mv = lambda g: {k: r["score_100"] for k, r in mem.get(g, {}).items()}              # noqa: E731
    groups = {g: {**J.describe(list(mv(g).values())), **group_metrics(mem[g], CE._qa_answers(run, g), by_q)} for g in mem}
    paired = {f"{b}-{a}": J.paired(mv(a), mv(b)) for b, a in PAIRS if a in mem and b in mem}
    rep = {"schema": SCHEMA, "gold": GOLD_NAME, "gold_source": "AI_VERIFIED", "is_human_gold": False, "n_questions": len(rows),
           "by_answerability": dict(Counter(q["answerability"] for q in rows)), "groups": groups, "paired": paired,
           "decision": decision(groups, paired, len(rows)), "notes": ["retrieval recall is not answer accuracy: compare gold_evidence_recall with answer_accuracy",
                                                           "AI-verified gold; one judge pass; small n per category"]}
    CE._write(run.p("reports", "ANALYSIS_V3.json"), rep)
    run.status("V3 STAGE 17 analysis", "PASS", recommendation=rep["decision"]["recommendation"])
    return rep


EXPORT = ("STAGE_STATUS.json", "STAGE_TIMINGS.json", "reports/ANALYSIS_V3.json", "reports/JUDGE_CALIBRATION_REPORT.json", "reports/NEW_GOLD_MANIFEST.json",
          "reports/DENSE_INDEX_REPORT.json", "judge/MEMORY_DEEPSEEK_SCORES.jsonl", "judge/memory_unit_key.json", "judge/CALIBRATION_SCORES.jsonl",
          "judge/ledger.jsonl", "qa/QA_BUILD_REPORT.json", "qa/QA_FROZEN.sha256")


def stage_export(run: CE.Run, env_report: Mapping) -> Path:
    from . import package as P
    R = Path(run.run_dir)
    out = R / "export" / "StillHere_V3_Results.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    entries = [(n, (R / n).read_bytes()) for n in EXPORT if (R / n).is_file()]
    for f in sorted((R / "memory_qa").rglob("answers.jsonl")) if (R / "memory_qa").is_dir() else []:
        entries.append((f.relative_to(R).as_posix(), f.read_bytes()))
    usage = J.make_ledger_summary(run.p("judge", "ledger.jsonl")) if run.p("judge", "ledger.jsonl").is_file() else {}
    man = {"schema": SCHEMA, "run_dir": str(R), "manifest": {k: run.manifest.get(k) for k in ("package_id", "input_zip_sha256", "git_commit", "base", "gold")},
           "cfg": run.cfg, "environment": dict(env_report), "judge_usage": usage, "gold_source": "AI_VERIFIED", "human_review": "NONE",
           "contains_sqlite": False, "created_at": S.now_iso()}
    entries.append(("RUN_MANIFEST_V3.json", json.dumps(man, ensure_ascii=False, indent=1, default=str).encode("utf-8")))
    sums = "".join(f"{hashlib.sha256(b).hexdigest()}  {n}\n" for n, b in entries)
    out.write_bytes(P.det_zip(entries + [("SHA256SUMS.txt", sums.encode("utf-8"))]))
    run.status("V3 STAGE 19 export", "PASS", zip=str(out), files=len(entries) + 1)
    return out
