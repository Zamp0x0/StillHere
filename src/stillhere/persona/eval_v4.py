"""Fourth-round (final) Colab evaluation: the P1 Reader fixes on a NEW frozen AI-verified set (V4), base model only.

Groups (all on the same frozen questions, base model revision, decoding, judge rubric, ask time and archive):

    V2L_OLD   V2L retrieval + the reader as evaluated in FAST / V2 / V3 (``reader_version`` "v1": subject rules v1, router v1, notes v1)
    V2L_NEW   V2L retrieval + the P1 reader (``reader_version`` "p1": mixed 你 + 我 subject rule, router past-speech cue, speaker note)
    V2D_NEW   V2D hybrid retrieval (dense + windows) + the P1 reader

The old reader is the SAME code behind a version switch (``persona.recall.resolve_subject_v1``, ``memory_v2.route_v2(version="v1")``), so
V2L_OLD is a real snapshot, not the new reader relabelled. Every group is CORE: if any is incomplete after the retries, the analysis
reports INCOMPLETE and makes no ADOPT decision (V3: 28 / 35 answers of one group failed and were silently left out of the run).

Rule (registered 2026-11-19, before any V4 answer): V2L_NEW replaces V2L_OLD iff all groups complete AND judge-score diff CI95 lower > -2
AND mean > 0 AND unsupported-claim rate not higher AND attribution / temporal errors not higher. V2D_NEW replaces V2L_NEW only by the same
test against V2L_NEW. Otherwise the older / simpler configuration stays. AI-verified gold: engineering evidence, not a human-gold result.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from . import colab_eval as CE
from . import eval_v2 as EV2
from . import eval_v3 as EV3
from . import judge as J
from . import qa_builder as QB
from . import store as S

SCHEMA = "stillhere.persona.eval_v4/1.0"
MEMORY_SPECS_V4: Dict[str, dict] = {
    "V2L_OLD": {"memory": "production", "pipeline": "v2", "dense": False, "adapter": None, "reader_version": "v1"},
    "V2L_NEW": {"memory": "production", "pipeline": "v2", "dense": False, "adapter": None, "reader_version": "p1"},
    "V2D_NEW": {"memory": "production", "pipeline": "v2", "dense": True, "adapter": None, "reader_version": "p1"},
}
CORE = tuple(MEMORY_SPECS_V4)
PAIRS = (("V2L_NEW", "V2L_OLD"), ("V2D_NEW", "V2L_NEW"), ("V2D_NEW", "V2L_OLD"))
MAX_MEMORY_PASSES = 3
RULE_V4 = ("registered 2026-11-19 before any V4 answer: V2L_NEW replaces V2L_OLD iff every group is complete and (judge diff CI95 lower > -2 "
           "and mean > 0) and unsupported-claim rate <= V2L_OLD and attribution / temporal errors <= V2L_OLD; V2D_NEW replaces V2L_NEW by the "
           "same test; else the older configuration stays. AI-verified gold: engineering evidence only.")


def reader_versions() -> dict:
    """What OLD and NEW are, as source hashes (the manifest records them; a later change is visible)."""
    from stillhere.reader import fact as RF
    from . import memory_v2 as V2
    from . import recall as R
    h = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()                     # noqa: E731
    common = {"fact_system_sha256": h(RF.FACT_SYSTEM), "attribution_ok_sha256": h(inspect.getsource(R.attribution_ok))}
    return {"v1": {**common, "subject_rules_sha256": h(inspect.getsource(R.resolve_subject_v1)), "speaker_note_sha256": h(R.SPEAKER_NOTE),
                   "router": "route_v2(version='v1')"},
            "p1": {**common, "subject_rules_sha256": h(inspect.getsource(R._resolve_subject_p1)), "speaker_note_sha256": h(R.SPEAKER_NOTE_P1),
                   "router": "route_v2(version='p1'): + past-speech cue", "router_cue": V2.PAST_SPEECH_CUE.pattern}}


def defaults(cfg: Mapping) -> dict:
    return {**EV3.defaults({}), "eval_mode": "V4", "memory_groups": list(MEMORY_SPECS_V4), "seed": 20261119, **dict(cfg)}


seed_frozen = EV3.seed_frozen
stage_judge = EV3.stage_judge


def completeness(run: CE.Run) -> dict:
    rows, _ = QB.load_frozen(Path(run.run_dir) / "qa")
    ids = {q["question_id"] for q in rows}
    out = {}
    for g in MEMORY_SPECS_V4:
        ans = CE._jsonl(Path(run.run_dir) / "memory_qa" / g / "answers.jsonl")
        ok = {r["question_id"] for r in ans if r.get("status") == "OK"}
        errs = Counter((r.get("error") or "")[:80] for r in ans if r.get("status") != "OK" and r["question_id"] not in ok)
        out[g] = {"ok": len(ok & ids), "of": len(ids), "missing": sorted(ids - ok), "errors": dict(errs), "complete": ids <= ok}
    return out


def stage_memory(run: CE.Run, *, dense=None) -> dict:
    """All groups, up to ``MAX_MEMORY_PASSES`` passes (each pass only retries what is not OK: no generation is repeated). Returns the
    completeness table; the stage status is INCOMPLETE (never PASS) while any CORE group misses an answer."""
    comp = {}
    for _ in range(MAX_MEMORY_PASSES):
        EV2.stage_memory(run, dense=dense, groups=list(MEMORY_SPECS_V4), specs=MEMORY_SPECS_V4)
        comp = completeness(run)
        if all(comp[g]["complete"] for g in CORE):
            break
    ok = all(comp[g]["complete"] for g in CORE)
    run.status("V4 STAGE 13 memory completeness", "PASS" if ok else "INCOMPLETE", **{g: f"{v['ok']}/{v['of']}" for g, v in comp.items()})
    CE._write(run.p("reports", "V4_COMPLETENESS.json"), comp)
    return comp


def group_metrics(rs: Mapping[str, dict], ans: Mapping[str, dict], by_q: Mapping[str, dict]) -> dict:
    m = EV3.group_metrics(rs, ans, by_q)
    A = [q for q in rs if by_q[q]["answerability"] == "ANSWERABLE"]
    produced = [q for q in A if (ans.get(q) or {}).get("reader_answer") or (ans.get(q) or {}).get("rejected_answer")]
    kept = [q for q in produced if (ans.get(q) or {}).get("final_check") in ("ANSWER_KEPT", "FALLBACK_TO_READER_ANSWER")]
    gate_rej_with_gold = [q for q in A if (ans.get(q) or {}).get("reader_fail") == "SUBJECT_RESOLUTION_FAIL" and ((ans.get(q) or {}).get("gold_retrieved") or 0) > 0]
    third = [q for q in A if str(by_q[q].get("subject")) == "THIRD_PARTY"]
    m.update(valid_answer_preservation={"num": len(kept), "den": len(produced), "rate": round(len(kept) / len(produced), 4) if produced else None},
             attribution_false_rejection_with_gold=len(gate_rej_with_gold),
             third_party_attribution_errors=sum(1 for q in third if rs[q]["judgement"].get("temporal_attribution", 5) <= 2 and not EV3._abstained(ans.get(q, {}))),
             reader_versions=dict(Counter(str((ans.get(q) or {}).get("reader_version")) for q in rs)))
    return m


def decision(groups: Mapping[str, dict], paired: Mapping[str, dict], comp: Mapping[str, dict]) -> dict:
    incomplete = [g for g in CORE if not (comp.get(g) or {}).get("complete")]
    if incomplete:
        return {"rule": RULE_V4, "verdict": "INCOMPLETE", "incomplete_groups": incomplete, "recommendation": "NO_DECISION (resume the run)",
                "gold": "AI_VERIFIED (not human)"}

    def better(new, old):
        p, g, b = paired.get(f"{new}-{old}") or {}, groups.get(new) or {}, groups.get(old) or {}
        lo = (p.get("ci95") or [None, None])[0]
        ok = (lo is not None and lo > -2 and (p.get("mean_diff") or 0) > 0 and (g.get("unsupported_claim_rate") or 0) <= (b.get("unsupported_claim_rate") or 0)
              and (g.get("attribution_temporal_errors") or 0) <= (b.get("attribution_temporal_errors") or 0))
        return {"replaces": ok, "paired": p}
    reader = better("V2L_NEW", "V2L_OLD")
    retrieval = better("V2D_NEW", "V2L_NEW")
    rec = ("V2D_NEW" if retrieval["replaces"] else "V2L_NEW") if reader["replaces"] else "V2L_OLD"
    return {"rule": RULE_V4, "verdict": "COMPLETE", "reader_p1": reader, "dense_over_lexical": retrieval, "recommendation": rec,
            "gold": "AI_VERIFIED (not human)"}


def analysis(run: CE.Run) -> dict:
    mem = CE._scores(run, "memory", "memory_unit_key.json")
    rows, _ = QB.load_frozen(Path(run.run_dir) / "qa")
    by_q = {q["question_id"]: q for q in rows}
    comp = completeness(run)
    mv = lambda g: {k: r["score_100"] for k, r in mem.get(g, {}).items()}              # noqa: E731
    groups = {g: {**J.describe(list(mv(g).values())), **group_metrics(mem[g], CE._qa_answers(run, g), by_q), "complete": comp[g]["complete"]} for g in mem}
    paired = {f"{b}-{a}": J.paired(mv(a), mv(b)) for b, a in PAIRS if a in mem and b in mem}
    rep = {"schema": SCHEMA, "gold_source": "AI_VERIFIED", "is_human_gold": False, "n_questions": len(rows), "completeness": comp,
           "by_answerability": dict(Counter(q["answerability"] for q in rows)), "groups": groups, "paired": paired,
           "decision": decision(groups, paired, comp), "reader_versions": reader_versions(),
           "notes": ["retrieval recall is not answer accuracy", "AI-verified gold; one judge pass; small n per category",
                     "the V3 known-failure questions are NOT in this set (V4 is independent of V3 / FAST)"]}
    CE._write(run.p("reports", "ANALYSIS_V4.json"), rep)
    run.status("V4 STAGE 17 analysis", "PASS" if rep["decision"]["verdict"] == "COMPLETE" else "INCOMPLETE", recommendation=rep["decision"]["recommendation"])
    return rep


def stage_writer_diagnostics(run: CE.Run, tmp: Path) -> dict:
    """OPTIONAL (never blocks): the public synthetic chat through the real Writer, with the wedding diagnostics (P2). Synthetic data only."""
    try:
        res = CE.synthetic_gold(run, tmp)
    except Exception as e:  # noqa: BLE001 - optional stage: recorded, never fatal
        run.status("V4 STAGE 07w writer diagnostics (optional)", "FAIL", error=f"{type(e).__name__}: {str(e)[:200]}")
        return {"status": "FAIL"}
    CE._write(run.p("reports", "WRITER_SYNTHETIC_DIAGNOSTICS.json"), res)
    run.status("V4 STAGE 07w writer diagnostics (optional)", "PASS", writer=f"{res['summary']['writer_pass']}/{res['summary']['writer_total']}")
    return res


EXPORT = ("STAGE_STATUS.json", "STAGE_TIMINGS.json", "reports/ANALYSIS_V4.json", "reports/V4_COMPLETENESS.json", "reports/JUDGE_CALIBRATION_REPORT.json",
          "reports/NEW_GOLD_MANIFEST.json", "reports/DENSE_INDEX_REPORT.json", "reports/WRITER_SYNTHETIC_DIAGNOSTICS.json", "judge/MEMORY_DEEPSEEK_SCORES.jsonl",
          "judge/memory_unit_key.json", "judge/CALIBRATION_SCORES.jsonl", "judge/ledger.jsonl", "qa/QA_BUILD_REPORT.json", "qa/QA_FROZEN.sha256")


def stage_export(run: CE.Run, env_report: Mapping) -> Path:
    from . import package as P
    R = Path(run.run_dir)
    out = R / "export" / "StillHere_V4_Results.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    entries = [(n, (R / n).read_bytes()) for n in EXPORT if (R / n).is_file()]
    for f in sorted((R / "memory_qa").rglob("answers.jsonl")) if (R / "memory_qa").is_dir() else []:
        entries.append((f.relative_to(R).as_posix(), f.read_bytes()))
    usage = J.make_ledger_summary(run.p("judge", "ledger.jsonl")) if run.p("judge", "ledger.jsonl").is_file() else {}
    comp = completeness(run)
    man = {"schema": SCHEMA, "run_dir": str(R), "manifest": {k: run.manifest.get(k) for k in ("package_id", "input_zip_sha256", "git_commit", "base", "gold", "reader_versions")},
           "cfg": run.cfg, "environment": dict(env_report), "judge_usage": usage, "gold_source": "AI_VERIFIED", "human_review": "NONE",
           "completeness": comp, "complete": all(v["complete"] for v in comp.values()), "contains_sqlite": False, "created_at": S.now_iso()}
    entries.append(("RUN_MANIFEST_V4.json", json.dumps(man, ensure_ascii=False, indent=1, default=str).encode("utf-8")))
    sums = "".join(f"{hashlib.sha256(b).hexdigest()}  {n}\n" for n, b in entries)
    out.write_bytes(P.det_zip(entries + [("SHA256SUMS.txt", sums.encode("utf-8"))]))
    run.status("V4 STAGE 19 export", "PASS", zip=str(out), files=len(entries) + 1, complete=man["complete"])
    return out
