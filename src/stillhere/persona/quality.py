"""Q1 Persona Quality Controller: diagnose -> prompt-first refinement on DEV -> runtime selection. Generic and data-driven.

Nothing here is written for one person: thresholds compare a model's DEV replies with that person's own reference replies and train
statistics (length, generic short replies, bubbles), and every prompt edit is one of a few generic, pre-defined knobs of the card
renderer (``prompt_options``: soft / capped length line, bubble cap from the person's own train p95, respond-first, anti-generic).
Facts and personal wording only enter a card through ``persona review`` (the user).

Rules (pre-registered here, applied identically to every persona; TEST is never read):
* candidates: at most 3, only for detected failures; B and C (and D / E) of a candidate share ONE prompt; memory evidence unchanged;
* a candidate is ADOPTED only if, on DEV, with the same model / decode / contexts, its target configuration fixes at least one
  failure of the baseline, adds no new failure, and loses at most ``MAX_STYLE_DROP`` style margin; otherwise the original stays;
* runtime: the adapter is enabled only when its verdict (C vs B under the runtime prompt) is BETTER or AUTOMATIC_METRICS_ONLY_PASS;
  memory is enabled when the production memory is built and passes the retrieval sanity probes; else Base + Persona Prompt.
* retraining is only RECOMMENDED (never started) when an adapter-specific failure survives prompt refinement; max_training_runs = 1
  (2 in quality mode).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from . import card as K
from . import evaluation as EV
from . import store as S

QUALITY_VERSION = "stillhere.persona.quality/1.0"
MAX_STYLE_DROP = 0.02
MAX_CANDIDATES = 3
EXPLORATORY_N = 20
REFINE_RULE = ("adopt a candidate prompt only if, on DEV with the same model / decode / contexts, its target configuration fixes >= 1 "
               f"baseline failure, adds no new failure and loses <= {MAX_STYLE_DROP} mean style margin; best = most failures fixed, then style margin; "
               "else keep the original prompt. B / C (and D / E) always share the candidate prompt.")
ADAPTER_OK = ("BETTER", "AUTOMATIC_METRICS_ONLY_PASS")


# ------------------------------------------------------------------ reference statistics (the person's own replies)
def _bubbles(t: str) -> int:
    return len([x for x in t.split("\n") if x.strip()])


def ref_stats(items: Sequence[Mapping]) -> dict:
    refs = [it["reference"] for it in items]
    flat = [r.replace("\n", "").strip() for r in refs]
    b = [_bubbles(r) for r in refs]
    return {"n": len(refs), "chars_median": statistics.median(len(x) for x in flat), "generic_rate": round(sum(x.lower() in EV.GENERIC for x in flat) / len(flat), 4),
            "top1_share": EV.ref_top1(items), "bubbles_3plus_share": round(sum(x >= 3 for x in b) / len(b), 4),
            "bubbles_p95": sorted(b)[int(0.95 * (len(b) - 1))]}


def group_bubbles(run: Path, group: str) -> dict:
    rows = EV.done_rows(run / group / "answers.jsonl")
    b = [_bubbles(r["text"]) for r in rows.values()]
    return {"bubbles_3plus_share": round(sum(x >= 3 for x in b) / len(b), 4) if b else None}


# ------------------------------------------------------------------ diagnosis
def failures(g: Mapping, gb: Mapping, ref: Mapping) -> List[dict]:
    out = []
    if g.get("n", 0) == 0:
        return out
    if g["length_ratio_median"] < 0.7:
        out.append({"type": "too_short", "value": g["length_ratio_median"], "threshold": "length ratio median >= 0.7"})
    if g["length_ratio_median"] > 1.4:
        out.append({"type": "too_long", "value": g["length_ratio_median"], "threshold": "length ratio median <= 1.4"})
    if g["generic_reply_rate"] > ref["generic_rate"] + 0.05 or g["distinct_reply_ratio"] < 0.7 or g["top1_reply_share"] > max(0.30, ref["top1_share"] + 0.10):
        out.append({"type": "generic_collapse", "value": {"generic": g["generic_reply_rate"], "distinct": g["distinct_reply_ratio"], "top1": g["top1_reply_share"]},
                    "threshold": f"generic <= reference {ref['generic_rate']} + 0.05, distinct >= 0.7, top1 <= max(0.30, reference + 0.10)"})
    if gb.get("bubbles_3plus_share") is not None and gb["bubbles_3plus_share"] > ref["bubbles_3plus_share"] + 0.10:
        out.append({"type": "over_split", "value": gb["bubbles_3plus_share"], "threshold": f"3+ bubbles <= reference {ref['bubbles_3plus_share']} + 0.10"})
    for key, lim, name in (("repetition_rate", 0.03, "repetition"), ("train_copy_rate", 0.02, "memorisation"), ("prefix_leak_rate", 0.02, "format_leak"),
                           ("unsupported_specific_rate", 0.05, "unsupported_facts"), ("special_token_leak_rate", 0.0, "format_leak")):
        if g.get(key, 0) > lim:
            out.append({"type": name, "value": g[key], "threshold": f"{key} <= {lim}"})
    if g.get("privacy_leaks", 0) > 0:
        out.append({"type": "privacy_leak", "value": g["privacy_leaks"], "threshold": "0"})
    return out


def latest_eval_run(pdir: Path) -> Optional[Path]:
    runs = [p for p in (pdir / "evals" / "runs").glob("*") if (p / "SCORE.json").is_file()]
    real = [p for p in runs if S.read_json(p / "SCORE.json").get("status") == "REAL_MODEL"]
    pick = real or runs
    return max(pick, key=lambda p: (p / "SCORE.json").stat().st_mtime) if pick else None


def blind_results(run: Path) -> Dict[str, dict]:
    out = {}
    for d in sorted((run / "blind").glob("*_vs_*")) if (run / "blind").is_dir() else []:
        key = d / "BLIND_KEY.json"
        for f in sorted(d.glob("BLIND_RESULT*.csv")):
            r = EV.score_blind(f, key)
            n = sum(v for k, v in r.items() if k.endswith("_wins"))
            r["label"] = "FACT (human ratings)" + (" / EXPLORATORY (n < %d)" % EXPLORATORY_N if n < EXPLORATORY_N else "")
            out[d.name] = r
    return out


def adjusted_failures(run: Path, items, groups, ref: Mapping, reference, score: Mapping):
    """V1 failures, re-judged by STYLE_EVALUATOR_V2: length and collapse against the PERSON's own distribution (authentic short replies
    are not failures; conditional repetition / diversity loss is). One rule for diagnosis and for prompt adoption."""
    from . import style_v2 as SV
    gs = {g: score["groups"][g] for g in groups}
    fails = {g: failures(v, group_bubbles(run, g), ref) for g, v in gs.items()}
    v2 = SV.evaluate(run, items, list(gs), reference, score=score)
    for g, fl in fails.items():
        ch = (v2["groups"].get(g) or {}).get("checks", {})
        keep = []
        for f in fl:
            if f["type"] == "too_short" and ch.get("length_in_person_range"):
                continue                                              # inside the person's own length range: authentic, not a failure
            if f["type"] == "generic_collapse" and ch.get("no_conditional_collapse") and ch.get("authentic_short_rate_plausible"):
                continue
            keep.append(f)
        if ch and not ch.get("no_conditional_collapse", True) and not any(f["type"] == "generic_collapse" for f in keep):
            keep.append({"type": "generic_collapse", "value": {k: v2["groups"][g][k] for k in ("distinct_ratio", "authentic_short_rate", "top_reply_share")},
                         "threshold": "STYLE_EVALUATOR_V2: conditional repetition / diversity vs the person's own references"})
        fails[g] = keep
    return fails, v2


def diagnose(root, profile_id: str, *, run_dir=None) -> dict:
    root = Path(root) if root else S.default_root()
    pdir = S.profile_dir(root, profile_id)
    run = Path(run_dir) if run_dir else latest_eval_run(pdir)
    if run is None:
        raise EV.EvalError("NO_EVAL", "no DEV evaluation run yet: persona evaluate (or import Colab results) first")
    from . import package as P
    ddir = P.latest_dataset(pdir)
    cur = K.current(pdir)
    sc = S.read_json(run / "SCORE.json")
    fp = S.read_json(run / "RUN_FINGERPRINT.json")["fields"]
    items = _items_for(pdir, ddir, cur)
    if fp.get("eval_set_sha256") != S.read_json(_dev_set(pdir, ddir, cur).with_suffix(".meta.json"))["file_sha256"]:
        raise EV.EvalError("EVAL_SET_MISMATCH", f"{run} was generated for another evaluation set / card than the current one")
    ref = ref_stats(items)
    groups = sc["groups"]
    fails, v2 = adjusted_failures(run, items, groups, ref, EV.Reference(ddir, cur["card"]), sc)
    base_types = {f["type"] for g in ("B", "D") for f in fails.get(g, [])}
    for g in ("C", "E"):
        for f in fails.get(g, []):
            src = ["adapter"] if f["type"] not in base_types else ["prompt / base model"]
            if f["type"] in ("too_short", "generic_collapse") and cur["card"]["reply_length"].get("short_reply_share", 0) >= 0.15:
                src.append("dataset (HYPOTHESIS: many very short train replies)")
            if f["type"] == "too_short" and "回复通常很短" in cur["prompt"]:
                src.append("prompt (HYPOTHESIS: the prompt stresses short replies on top of short training data)")
            f["likely_source"] = src
    for g in ("A", "B", "D"):
        for f in fails.get(g, []):
            f["likely_source"] = ["prompt / base model"]
    blind = blind_results(run)
    verdicts = sc.get("verdicts", {})
    for pair, b in blind.items():
        a, c = pair.split("_vs_")
        if pair in verdicts and verdicts[pair].get("verdict") not in (EV.DRY,):
            verdicts[pair] = {**EV.verdict(sc, a, c, blind=b), "blind": b}
    cands = candidate_options(fails, ref, cur, bool(groups.get("C")))
    rep = {"schema": "stillhere.persona.quality_report/1.0", "version": QUALITY_VERSION, "profile_id": profile_id, "eval_run": run.name, "status": sc["status"],
           "card_id": cur["card_id"], "prompt_sha256": cur["manifest"]["prompt_sha256"], "dataset_version": ddir.name, "split": "dev", "test_read": False,
           "labels": {"FACT": "computed from DEV generations / ratings", "HYPOTHESIS": "a likely cause, not tested", "NOT_VERIFIED": "needs a run that has not happened"},
           "reference": ref,
           "sections": {
               "style_fidelity": {g: v.get("style_margin_mean") for g, v in groups.items()},
               "reply_appropriateness": {"proxy_context_overlap": {g: v.get("context_overlap_mean") for g, v in groups.items()},
                                         "human": {k: v.get("win_rate") for k, v in blind.items()} or "NOT_VERIFIED (no blind review)"},
               "naturalness": {k: v for k, v in blind.items()} or "NOT_VERIFIED (no blind review)",
               "reply_length": {g: {"chars_median": v.get("chars_median"), "length_ratio_median": v.get("length_ratio_median"), "bubbles_mean": v.get("bubbles_mean")} for g, v in groups.items()},
               "characteristic_expressions": {g: v.get("marker_rate") for g, v in groups.items()},
               "generic_short_reply_rate": {g: v.get("generic_reply_rate") for g, v in groups.items()},
               "collapse_risk": {g: {"top1": v.get("top1_reply_share"), "distinct": v.get("distinct_reply_ratio")} for g, v in groups.items()},
               "repetition_risk": {g: v.get("repetition_rate") for g, v in groups.items()},
               "memorisation_risk": {g: v.get("train_copy_rate") for g, v in groups.items()},
               "unsupported_fact_risk": {g: v.get("unsupported_specific_rate") for g, v in groups.items()},
               "privacy_leakage_risk": {g: v.get("privacy_leaks") for g, v in groups.items()},
               "memory_consistency": {g: v.get("memory_specific_used_rate") for g, v in groups.items() if g in ("D", "E")} or "NOT_VERIFIED (no D / E run)",
               "base_vs_adapter": verdicts,
               "style_v2": {"label": v2["label"], "person": v2["person"], "reference": {k: v for k, v in v2["reference"].items() if k != "top_reply_norm"},
                            "groups": {g: {k: x[k] for k in ("in_person_range", "authentic_short_rate", "distinct_ratio", "top_reply_share",
                                                             "possibly_insufficient_answers", "checks")} for g, x in v2["groups"].items()},
                            "verdicts_v2": v2.get("verdicts_v2")},
               "prompt_improvement_candidates": cands,
           },
           "failures": fails, "retrain_recommendation": retrain_advice(fails, cands)}
    rep["sections"]["recommended_runtime"] = recommend(rep, root, profile_id)
    out = pdir / "evaluation" / "persona_quality_report.json"
    S.write_json(out, rep)
    return {**rep, "path": str(out)}


def _dev_set(pdir: Path, ddir: Path, cur: Mapping) -> Path:
    return pdir / "evals" / "sets" / f"{ddir.name}_{cur['card_id']}_dev.jsonl"


def _items_for(pdir, ddir, cur):
    p = _dev_set(pdir, ddir, cur)
    if not p.is_file():
        from . import package as P
        prof = S.read_json(pdir / "profile.json")
        P.dev_eval_set(pdir, ddir, prof, cur, with_memory=True)
    return EV.read_eval_set(p)


def candidate_options(fails: Mapping[str, List[dict]], ref: Mapping, cur: Mapping, has_adapter: bool) -> List[dict]:
    """At most MAX_CANDIDATES generic prompt edits, each motivated by a measured failure; never a person-specific phrase."""
    target = ("C", "E") if has_adapter else ("B", "D")
    types = {f["type"] for g in target for f in fails.get(g, [])}
    base_split = any(f["type"] == "over_split" for g in ("B", "D") for f in fails.get(g, []))
    common = {"bubble_cap": max(2, int(ref["bubbles_p95"]))} if base_split or "over_split" in types else {}
    out = []
    if "too_short" in types:
        out.append({"name": "soft_length_respond_first", "options": {"length": "soft", "respond_first": True, **common},
                    "why": "replies shorter than the person's own length range: weaken the repeated 'short' emphasis, ask to answer the latest message"})
    if "generic_collapse" in types:
        out.append({"name": "respond_first_anti_generic", "options": {"respond_first": True, "anti_generic": True, **common},
                    "why": "the person's own short phrases are used far beyond their own rate / across different questions: keep the short style, "
                           "but answer the latest message and do not reuse one phrase for everything"})
    if "too_long" in types:
        out.append({"name": "length_cap", "options": {"length": "cap", **common}, "why": "replies longer than the person's own: state the person's p90 length as a cap"})
    if not out and common:
        out.append({"name": "bubble_cap", "options": common, "why": "more messages per turn than the person sends: cap from the person's own p95"})
    return out[:MAX_CANDIDATES]


def retrain_advice(fails: Mapping[str, List[dict]], cands: Sequence[Mapping]) -> dict:
    adapter_only = [f["type"] for g in ("C", "E") for f in fails.get(g, []) if "adapter" in f.get("likely_source", [])]
    return {"recommend_retrain": False, "adapter_specific_failures": sorted(set(adapter_only)), "max_training_runs": {"default": 1, "quality_mode": 2},
            "rule": "retraining is recommended only if an adapter-specific failure survives prompt refinement on DEV, the user allows another run and a GPU is available",
            "status": "PENDING_PROMPT_REFINEMENT" if adapter_only and cands else ("NOT_NEEDED" if not adapter_only else "CONSIDER_RETRAIN")}


# ------------------------------------------------------------------ refinement: candidates -> DEV sets -> generation -> adoption
def refine_prepare(root, profile_id: str) -> dict:
    root = Path(root) if root else S.default_root()
    pdir = S.profile_dir(root, profile_id)
    diag = diagnose(root, profile_id)
    cur = K.current(pdir)
    from . import package as P
    ddir = P.latest_dataset(pdir)
    items = _items_for(pdir, ddir, cur)
    cands = diag["sections"]["prompt_improvement_candidates"]
    rid = "rf-" + hashlib.sha256(json.dumps([cur["card_id"], diag["eval_run"], cands], sort_keys=True).encode()).hexdigest()[:10]
    rdir = pdir / "evals" / "refine" / rid
    out = []
    for c in cands:
        card = dict(cur["card"], prompt_options=c["options"], refinement={"candidate": c["name"], "parent_card": cur["card_id"], "eval_run": diag["eval_run"]})
        v = K._write_version(pdir, card, parent=cur["card_id"])                 # a new card version; NOT made current
        prompt = (Path(v["card_dir"]) / "persona_prompt.txt").read_text(encoding="utf-8")
        es = rdir / c["name"] / "dev_eval.jsonl"
        if not es.is_file():
            EV.write_eval_set(es, [{**it, "user": it["user"]} for it in items], persona_prompt=prompt, groups=["B", "C", "D", "E"] if items[0].get("memory") else ["B", "C"],
                              meta={"split": "dev", "dataset_version": ddir.name, "card_id": v["card_id"], "refine_id": rid, "candidate": c["name"]})
        out.append({**c, "card_id": v["card_id"], "prompt_sha256": v["prompt_sha256"], "eval_set": str(es)})
    S.write_json(rdir / "REFINE_PLAN.json", {"refine_id": rid, "baseline_run": diag["eval_run"], "baseline_card": cur["card_id"], "candidates": out, "rule": REFINE_RULE,
                                             "status": "PENDING_GENERATION" if out else "NO_CANDIDATES", "at": S.now_iso()})
    return {"refine_id": rid, "dir": str(rdir), "candidates": out, "diagnosis_failures": {g: [f["type"] for f in v] for g, v in diag["failures"].items()}}


def refine_generate(root, profile_id: str, refine_id: str, *, backend: str, base_dir=None, adapter_id: Optional[str] = None, groups=("B", "C"), limit=None, log=print) -> dict:
    from . import adapters as AD
    from .models import make_generator
    pdir = S.profile_dir(Path(root) if root else S.default_root(), profile_id)
    rdir = pdir / "evals" / "refine" / refine_id
    plan = S.read_json(rdir / "REFINE_PLAN.json")
    ad = AD.resolve(pdir, adapter_id) if adapter_id else None
    need = [g for g in groups if EV.GROUPS[g]["adapter"]]
    gen = make_generator(backend, base_dir, {g: ad["dir"] for g in need} if (ad and backend.startswith("hf")) else {})
    res = {}
    for c in plan["candidates"]:
        items = EV.read_eval_set(Path(c["eval_set"]))
        head = S.read_json(Path(c["eval_set"]).with_suffix(".meta.json"))
        fp = {"eval_set_sha256": head["file_sha256"], "persona_prompt_sha256": head["persona_prompt_sha256"], "decode": EV.DECODE, "generator": gen.label,
              "adapter_sha256": ad["manifest"]["files"]["adapter/adapter_model.safetensors"] if ad else None, "eval_version": EV.EVAL_VERSION}
        res[c["name"]] = EV.generate(gen, items, list(groups), rdir / c["name"] / "run", fp, adapters={g: g for g in need}, limit=limit, log=log)
    return res


def refine_select(root, profile_id: str, refine_id: str, *, adopt: bool = True) -> dict:
    root = Path(root) if root else S.default_root()
    pdir = S.profile_dir(root, profile_id)
    rdir = pdir / "evals" / "refine" / refine_id
    plan = S.read_json(rdir / "REFINE_PLAN.json")
    cur = K.current(pdir)
    from . import package as P
    ddir = P.latest_dataset(pdir)
    base_run = pdir / "evals" / "runs" / plan["baseline_run"]
    base_items = _items_for(pdir, ddir, K.load(pdir, plan["baseline_card"]))
    ref = ref_stats(base_items)
    reference = EV.Reference(ddir, cur["card"])
    bscore = S.read_json(base_run / "SCORE.json")
    table = {}
    for c in plan["candidates"]:
        run = rdir / c["name"] / "run"
        if not (run / "GENERATION_STATUS.json").is_file():
            table[c["name"]] = {"status": "NOT_GENERATED"}
            continue
        items = EV.read_eval_set(Path(c["eval_set"]))
        groups = [g for g in ("B", "C", "D", "E") if (run / g / "answers.jsonl").is_file()]
        sc = EV.score_run(run, items, groups, reference)
        tg = "C" if "C" in groups else "B"
        bf, _ = adjusted_failures(base_run, base_items, [tg], ref, reference, bscore)
        af, _ = adjusted_failures(run, items, [tg], ref, reference, sc)
        before, after = {f["type"] for f in bf[tg]}, {f["type"] for f in af[tg]}
        drop = (bscore["groups"][tg]["style_margin_mean"] or 0) - (sc["groups"][tg]["style_margin_mean"] or 0)
        ok = bool(before - after) and not (after - before) and drop <= MAX_STYLE_DROP and sc["status"] != EV.DRY
        table[c["name"]] = {"status": sc["status"], "target_group": tg, "fixed": sorted(before - after), "new": sorted(after - before), "style_drop": round(drop, 4),
                            "qualifies": ok, "verdicts": sc["verdicts"], "card_id": c["card_id"]}
    q = [n for n, v in table.items() if v.get("qualifies")]
    best = max(q, key=lambda n: (len(table[n]["fixed"]), -table[n]["style_drop"])) if q else None
    dry = any(v.get("status") == EV.DRY for v in table.values())
    res = {"refine_id": refine_id, "rule": REFINE_RULE, "candidates": table, "selected": best,
           "status": "ADOPTED" if best and adopt else ("DRY_RUN_ONLY" if dry else ("NO_IMPROVEMENT_KEPT_ORIGINAL" if not best else "SELECTED_NOT_ADOPTED"))}
    if best and adopt:
        K.set_current(pdir, table[best]["card_id"])
        S.append_lineage(pdir, {"artefact": f"cards/{table[best]['card_id']}", "kind": "prompt_refinement_adopted", "refine_id": refine_id})
    S.write_json(rdir / "PROMPT_REFINEMENT.json", res)
    return res


# ------------------------------------------------------------------ runtime selection
def recommend(diag: Mapping, root, profile_id: str) -> dict:
    from . import memory_store as MS
    v = diag["sections"]["base_vs_adapter"].get("B_vs_C", {}).get("verdict")
    st = MS.status(root, profile_id)
    mq = S.profile_dir(Path(root) if root else S.default_root(), profile_id) / "evaluation" / "memory_quality_report.json"
    probes_ok = False
    if mq.is_file():
        p = S.read_json(mq)["retrieval_probes"]["exact_message_recall_at_8"]
        a, b = (int(x) for x in p.split("/"))
        probes_ok = b > 0 and a / b >= 0.9
    mem_ok = st["memory_status"] in ("EVIDENCE_READY", "SEMANTIC_PARTIAL", "READY") and probes_ok
    return {"adapter": v in ADAPTER_OK, "adapter_verdict": v, "memory": mem_ok, "memory_status": st["memory_status"],
            "semantic_extraction": st.get("semantic_extraction"),
            "mode": ("B+Adapter+Memory (E)" if v in ADAPTER_OK and mem_ok else "B+Adapter (C)" if v in ADAPTER_OK else "B+Memory (D)" if mem_ok else "B (prompt only)")}


def select_runtime(root, profile_id: str, *, override_adapter: Optional[bool] = None, override_memory: Optional[bool] = None,
                   memory_pipeline: Optional[str] = None) -> dict:
    """``persona finalize``. ``memory_pipeline``: ``product`` (V2D + P2, fallback V2L + P2; ``persona.product_config``) | ``product-v2d`` (V2D without P2) | ``research-final`` (V2L,
    ``persona.final_config``) | ``v1`` (no section). ``None`` keeps the profile's current section; a profile finalized for the FIRST time
    (no runtime_config.json yet) gets ``product``. A profile whose existing config has no section keeps the historical V1 path."""
    from . import adapters as AD
    from . import doctor as DR
    from . import memory_store as MS
    root = Path(root) if root else S.default_root()
    pdir = S.profile_dir(root, profile_id)
    cur = K.current(pdir)
    qp = pdir / "evaluation" / "persona_quality_report.json"
    diag = S.read_json(qp) if qp.is_file() else None
    rec = diag["sections"]["recommended_runtime"] if diag else {"adapter": False, "adapter_verdict": "NOT_EVALUATED", "memory": False}
    if diag is None:
        st = MS.status(root, profile_id)
        rec["memory"] = st["memory_status"] in ("EVIDENCE_READY", "SEMANTIC_PARTIAL", "READY")
        rec["memory_status"], rec["semantic_extraction"] = st["memory_status"], st.get("semantic_extraction")
    inst = AD.installed(pdir)
    best = next((a for a in inst if a["card_id"] in (cur["card_id"], cur["manifest"].get("parent_card"), (cur["card"].get("refinement") or {}).get("parent_card"))), None)
    use_adapter = rec["adapter"] if override_adapter is None else override_adapter
    use_memory = rec["memory"] if override_memory is None else override_memory
    reason = []
    reason.append(f"adapter: {'enabled' if use_adapter else 'disabled'} (verdict {rec.get('adapter_verdict')}" + (", user override)" if override_adapter is not None else ")"))
    reason.append(f"memory: {'enabled' if use_memory else 'disabled'} ({rec.get('memory_status')}, semantic {rec.get('semantic_extraction')})"
                  + (", user override" if override_memory is not None else ""))
    if use_adapter and best is None:
        use_adapter = False
        reason.append("no installed adapter for this card: adapter disabled")
    cfg = {"schema": "stillhere.persona.runtime_config/1.0", "profile_id": profile_id, "base": {"id": DR.BASE_REPO, "revision": DR.BASE_REVISION, "quantization": "nf4"},
           "persona_prompt": {"card_id": cur["card_id"], "sha256": cur["manifest"]["prompt_sha256"], "review_status": cur["manifest"]["review_status"]},
           "adapter": {"enabled": bool(use_adapter), "adapter_id": best["adapter_id"] if best else None, "run_id": best["run_id"] if best else None,
                       "available": [a["adapter_id"] for a in inst]},
           "memory": {"enabled": bool(use_memory), "view": "production_full_history", "db": MS.status(root, profile_id).get("db"),
                      "semantic_extraction": rec.get("semantic_extraction")},
           "decode": EV.DECODE, "selection_reason": reason, "evaluation_fingerprint": (diag or {}).get("eval_run"),
           "quality_status": (diag or {}).get("status", "NOT_EVALUATED"), "user_override": {"adapter": override_adapter, "memory": override_memory}, "at": S.now_iso()}
    from . import product_config as PC
    out = pdir / "runtime" / "runtime_config.json"
    if memory_pipeline is not None:                       # an explicit choice
        sec, how = PC.section(memory_pipeline), f"explicit: {memory_pipeline}"
    elif out.is_file():                                   # the profile's memory pipeline (product / research final / V1) survives a re-finalize
        sec, how = S.read_json(out).get("memory_pipeline"), "kept from the previous runtime_config.json"
    else:
        sec, how = PC.section("product"), "first finalize: product default"
    if sec:
        cfg["memory_pipeline"] = sec
    reason.append(f"memory pipeline: {PC.describe(sec)} ({how})")
    S.write_json(out, cfg)
    S.append_lineage(pdir, {"artefact": "runtime/runtime_config.json", "kind": "runtime_config", "adapter": cfg["adapter"]["enabled"], "memory": cfg["memory"]["enabled"]})
    return {**cfg, "path": str(out)}


# ------------------------------------------------------------------ blind review: quick / quality mode + local HTML
def make_blind_review(run: Path, items: Sequence[Mapping], pair: Sequence[str], *, n: int = 8, seed: str = "persona-blind-q") -> dict:
    """Quality mode: ``n`` diverse pairs (5-10 by default 8; the existing 60-row sheets stay importable). Pairs where both replies are
    identical, or both are generic short replies, are skipped so one repeated phrase cannot dominate. Writes the CSV (scoring format
    of ``evaluation.score_blind``), the key and a local HTML page; all PRIVATE."""
    a, b = pair
    ra, rb = EV.done_rows(run / a / "answers.jsonl"), EV.done_rows(run / b / "answers.jsonl")
    cand = []
    for it in items:
        s = it["sample_id"]
        if s not in ra or s not in rb:
            continue
        ta, tb = ra[s]["text"].strip(), rb[s]["text"].strip()
        if ta == tb or (ta.replace("\n", "") in EV.GENERIC and tb.replace("\n", "") in EV.GENERIC):
            continue
        cand.append(it)
    rng = random.Random(f"{seed}:{a}:{b}:{n}")
    rng.shuffle(cand)
    seen, pick = set(), []
    for it in cand:
        k = it["reference"].replace("\n", "")[:6]
        if k in seen:
            continue
        seen.add(k)
        pick.append(it)
        if len(pick) == n:
            break
    sub = run / "blind" / f"{a}_vs_{b}_quick{n}"
    EV.make_blind(run, pick, pair, n=n, seed=seed, dest=sub)
    html = blind_html(sub / "BLIND_REVIEW.csv", f"{len(pick)}")
    return {"csv": str(sub / "BLIND_REVIEW.csv"), "html": str(html), "key": str(sub / "BLIND_KEY.json"), "rows": len(pick),
            "label": "EXPLORATORY (few pairs)" if len(pick) < EXPLORATORY_N else "QUALITY"}


BLIND_PAGE = (Path(__file__).with_name("blind_review.html"))


def blind_html(csv_path: Path, title: str) -> Path:
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = [{"id": r["id"], "context": r["context"], "left": r["reply_left"], "right": r["reply_right"]} for r in csv.DictReader(f)]
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    out = Path(csv_path).with_name("BLIND_REVIEW.html")
    out.write_text(BLIND_PAGE.read_text(encoding="utf-8").replace("__DATA__", data).replace("__TITLE__", title), encoding="utf-8")
    return out


def rescore_style_v2(root, profile_id: str, run_name: str) -> dict:
    """Write evaluation/style_v2_<run>.json for an existing DEV run (read-only on the run)."""
    from . import package as P
    from . import style_v2 as SV
    pdir = S.profile_dir(Path(root) if root else S.default_root(), profile_id)
    run = pdir / "evals" / "runs" / run_name
    cur = K.current(pdir)
    ddir = P.latest_dataset(pdir)
    items = _items_for(pdir, ddir, cur)
    score = S.read_json(run / "SCORE.json")
    groups = [g for g in ("A", "B", "C", "D", "E") if (run / g / "answers.jsonl").is_file()]
    res = SV.evaluate(run, items, groups, EV.Reference(ddir, cur["card"]), score=score)
    res["original_v1_verdicts"] = {k: v.get("verdict") for k, v in (score.get("verdicts") or {}).items()}
    out = pdir / "evaluation" / f"style_v2_{run_name}.json"
    S.write_json(out, res)
    return {**res, "path": str(out)}
