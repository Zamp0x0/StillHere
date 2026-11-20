"""Import the final Colab L4 results ZIP into the local profile (``persona import-colab-results``).

Checks before anything is written: every file against SHA256SUMS; the profile; the adapter sha256 against the local adapter; both
prompt card ids / prompt sha256 against the local cards; the QA set sha256; the judge report; the memory snapshot schema; the
snapshot's evidence archive equal to the local one (same message ids and content hashes); every memory traced to archived messages;
valid_from <= valid_to; no memory event claiming a human review (the Colab run writes ``auto_verified;accepted_by:...`` only); the
exported VERIFIED rows marked ``human_reviewed: false`` and equal in number to the snapshot's memories (only committed candidates).

Then: the local store is backed up and replaced by the snapshot (only if the local store holds no semantic memory yet, or exactly the
same content: idempotent), the reports go to ``evaluation/colab_final/<run>/``, and ``LATEST.json`` points Persona Studio at them.
Cards, ``cards/CURRENT.json``, ``runtime_config.json``, frozen evaluations and the sealed test are never touched; the prompt / adapter
recommendation is shown, not applied.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from . import store as S

REQUIRED_COMMON = ("RUN_MANIFEST.json", "SHA256SUMS.txt", "REAL_HISTORY_QA_200.jsonl", "QA_BUILD_REPORT.json", "JUDGE_USAGE_REPORT.json",
                   "COMBINED_EVALUATION_REPORT.json", "RUNTIME_RECOMMENDATION.json", "ABLATION_REPORT.json", "MEMORY_QA_GENERATIONS.jsonl", "ACADEMIC_DEMO_CASES.md")
REQUIRED_MODE = {"FULL": ("memory_snapshot.sqlite", "VERIFIED_MEMORIES.jsonl", "MEMORY_CANDIDATES.jsonl"),
                 "FAST": ("EVAL_MODE.json", "ENVIRONMENT_REPORT.json", "FINAL_SUMMARY_CN.md", "STYLE_DEV_GENERATIONS.jsonl", "STYLE_DEEPSEEK_SCORES.jsonl",
                          "STYLE_V2_REPORT.json", "MEMORY_DEEPSEEK_SCORES.jsonl", "WRITER_AUDIT_100.jsonl", "WRITER_AUDIT_REPORT.json", "STILLHERE_ABLATION_RESULTS.md",
                          "REAL_CASE_STUDIES_20_30.md", "ACADEMIC_RESULTS.md", "INTERVIEW_SUMMARY_EN_CN.md", "ERROR_QUEUE.json", "VERIFIED_MEMORIES.jsonl")}
FAST_LABELS = ("Evidence-only Memory Evaluation", "Semantic Memory Full Build Not Executed", "Experimental Evaluation Results")
MEMORY_GROUPS = ["M0", "M1", "M2"]


class ImportError_(S.StoreError):
    pass


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _archive_sha(db: Path, pid: str) -> str:
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT message_id, content_sha256 FROM messages WHERE target_persona_id = ? ORDER BY message_id", (pid,)).fetchall()
    finally:
        con.close()
    return _sha(json.dumps(rows).encode("utf-8"))


def _memory_sha(db: Path) -> tuple:
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT memory_id, subject, attribute, value, status, valid_from_utc, valid_to_utc FROM memories ORDER BY memory_id").fetchall()
    finally:
        con.close()
    return len(rows), _sha(json.dumps(rows, ensure_ascii=False, default=str).encode("utf-8"))


def check_snapshot(db: Path, pid: str, local_db: Path) -> dict:
    """Deterministic checks of the Colab-built store (raises ImportError_ on the first broken rule)."""
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for t in ("messages", "memories", "provenance", "memory_events"):
            if t not in tables:
                raise ImportError_("MEMORY_SCHEMA", f"the snapshot has no {t} table")
        persons = {r[0] for r in con.execute("SELECT DISTINCT persona_id FROM memories")}
        if persons - {pid}:
            raise ImportError_("PROFILE_MISMATCH", "the snapshot holds memories of another persona")
        sources = {r[0] for r in con.execute("SELECT DISTINCT source_id FROM messages WHERE target_persona_id = ?", (pid,))}
        no_prov = con.execute("SELECT COUNT(*) FROM memories m WHERE NOT EXISTS (SELECT 1 FROM provenance p WHERE p.memory_id = m.memory_id)").fetchone()[0]
        bad_src = con.execute(f"SELECT COUNT(*) FROM provenance WHERE source_id NOT IN ({','.join('?' * len(sources)) or 'NULL'})", tuple(sources)).fetchone()[0]
        if no_prov or bad_src:
            raise ImportError_("SOURCE_IDS", f"{no_prov} memories without provenance, {bad_src} provenance rows outside the archive")
        bad_time = con.execute("SELECT COUNT(*) FROM memories WHERE valid_from_utc IS NOT NULL AND valid_to_utc IS NOT NULL AND valid_to_utc < valid_from_utc").fetchone()[0]
        if bad_time:
            raise ImportError_("TEMPORAL", f"{bad_time} memories end before they start")
        reviewers = {r[0] for r in con.execute("SELECT DISTINCT reviewer FROM memory_events WHERE reviewer IS NOT NULL")}
        human = [r for r in reviewers if not str(r).startswith("auto_verified")]
        if human:
            raise ImportError_("REVIEW_INTEGRITY", "the snapshot records reviews that are not auto-verification (a human review cannot come from Colab)")
        n_mem = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    finally:
        con.close()
    if _archive_sha(db, pid) != _archive_sha(local_db, pid):
        raise ImportError_("ARCHIVE_MISMATCH", "the snapshot was built from another evidence archive than the local one")
    return {"memories": n_mem, "reviewers": sorted(map(str, reviewers))}


def import_results(root, profile_id: str, zip_path, *, replace_memory: bool = False) -> dict:
    from . import adapters as AD
    from . import card as K
    root = Path(root) if root else S.default_root()
    pdir = S.profile_dir(root, profile_id)
    zip_path = Path(zip_path)
    zsha = _sha(zip_path.read_bytes())
    ledger_path = pdir / "evaluation" / "colab_final" / "IMPORTS.json"
    ledger = S.read_json(ledger_path) if ledger_path.is_file() else {}
    if zsha in ledger:
        return {"status": "ALREADY_IMPORTED", **ledger[zsha]}
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        if any(".." in n.split("/") or n.startswith("/") or ":" in n for n in names):
            raise ImportError_("ZIP_UNSAFE", "unsafe member path")
        mode = json.loads(z.read("EVAL_MODE.json")).get("eval_mode", "FULL") if "EVAL_MODE.json" in names else "FULL"
        if mode not in REQUIRED_MODE:
            raise ImportError_("EVAL_MODE", f"unknown evaluation mode {mode!r}")
        from .colab_eval import export_names
        ren = export_names(mode)
        nm = lambda canonical: ren.get(canonical, canonical)   # noqa: E731
        missing = [nm(n) for n in REQUIRED_COMMON + REQUIRED_MODE[mode] if nm(n) not in names]
        if missing:
            raise ImportError_("RESULTS_INCOMPLETE", f"missing {missing}")
        sums = dict((l.split(None, 1)[1].strip(), l.split(None, 1)[0]) for l in z.read("SHA256SUMS.txt").decode("utf-8").splitlines() if l.strip())
        bad = [n for n in names if n != "SHA256SUMS.txt" and sums.get(n) != _sha(z.read(n))]
        if bad or set(sums) - set(names):
            raise ImportError_("SHA256", f"files do not match SHA256SUMS: {bad[:5]}")
        man = json.loads(z.read("RUN_MANIFEST.json"))
        if man.get("profile_id") != profile_id:
            raise ImportError_("PROFILE_MISMATCH", f"results of {man.get('profile_id')}, not {profile_id}")
        pkg = man.get("package") or {}
        ad = AD.resolve(pdir, pkg["adapter"]["adapter_id"])
        if _sha((Path(ad["dir"]) / "adapter_model.safetensors").read_bytes()) != pkg["adapter"]["sha256"]:
            raise ImportError_("ADAPTER_MISMATCH", "the evaluated adapter is not the local one")
        for k in ("old", "new"):
            c = K.load(pdir, pkg["prompts"][k]["card_id"])
            if c["manifest"]["prompt_sha256"] != pkg["prompts"][k]["prompt_sha256"]:
                raise ImportError_("PROMPT_MISMATCH", f"the {k} prompt differs from the local card")
        if man.get("memory_groups") != MEMORY_GROUPS:
            raise ImportError_("ABLATION_GROUPS", f"expected memory groups {MEMORY_GROUPS}, got {man.get('memory_groups')}")
        gen_groups = {json.loads(l)["group"] for l in z.read(nm("MEMORY_QA_GENERATIONS.jsonl")).decode("utf-8").splitlines() if l.strip()}
        if gen_groups - set(MEMORY_GROUPS):
            raise ImportError_("ABLATION_GROUPS", f"unknown memory groups in the generations: {sorted(gen_groups - set(MEMORY_GROUPS))}")
        if _sha(z.read(nm("REAL_HISTORY_QA_200.jsonl"))) != man.get("qa_sha256"):
            raise ImportError_("QA_SHA", "the QA set does not match the run manifest")
        usage = json.loads(z.read("JUDGE_USAGE_REPORT.json"))
        if not usage.get("judge_model") or not usage.get("rubric_version"):
            raise ImportError_("JUDGE_REPORT", "the judge report does not record the model / rubric")
        for l in z.read(nm("VERIFIED_MEMORIES.jsonl")).decode("utf-8").splitlines():
            if l.strip() and json.loads(l).get("human_reviewed") is not False:
                raise ImportError_("REVIEW_INTEGRITY", "an exported memory claims a human review")
        if mode == "FAST":
            return _import_fast(pdir, z, names, man, usage, zsha, ledger, ledger_path)
        stored = set()
        for l in z.read("VERIFIED_MEMORIES.jsonl").decode("utf-8").splitlines():
            if l.strip():
                stored.add(json.loads(l)["memory_id"])
        with tempfile.TemporaryDirectory() as td:
            snap = Path(td) / "memory_snapshot.sqlite"
            snap.write_bytes(z.read("memory_snapshot.sqlite"))
            local_db = pdir / "memory" / "memory.sqlite"
            chk = check_snapshot(snap, profile_id, local_db)
            local_n, local_sha = _memory_sha(local_db)
            snap_n, snap_sha = _memory_sha(snap)
            if snap_n != len(stored):
                raise ImportError_("MEMORY_EXPORT_MISMATCH", "VERIFIED_MEMORIES does not list the snapshot's memories")
            if local_n and local_sha != snap_sha and not replace_memory:
                raise ImportError_("LOCAL_MEMORY_DIFFERS", f"the local store already holds {local_n} other memories; pass --replace-memory to replace it (it is backed up)")
            run = Path(man.get("run_dir") or "run").name
            dest = pdir / "evaluation" / "colab_final" / run
            dest.mkdir(parents=True, exist_ok=True)
            for n in names:
                if n != "memory_snapshot.sqlite":
                    (dest / n).parent.mkdir(parents=True, exist_ok=True)
                    (dest / n).write_bytes(z.read(n))
            memory_action = "UNCHANGED (same content)"
            if local_sha != snap_sha:
                bdir = pdir / "memory" / "backups"
                bdir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                shutil.copyfile(local_db, bdir / f"memory-{stamp}-before-colab-import.sqlite")
                src, dst = sqlite3.connect(str(snap)), sqlite3.connect(str(local_db))
                try:
                    src.backup(dst)
                finally:
                    dst.close()
                    src.close()
                memory_action = f"REPLACED (backup memory/backups/memory-{stamp}-before-colab-import.sqlite)"
    rec = {"run": run, "imported_at": S.now_iso(), "results_zip_sha256": zsha, "memories": chk["memories"], "memory_action": memory_action,
           "reports_dir": str(dest), "judge_ok_units": usage.get("ok_total"), "judge_planned_units": usage.get("planned_total"),
           "note": "machine judge scores and auto-verified memories; nothing here is a human review. Prompt / adapter recommendations are not applied."}
    ledger[zsha] = rec
    S.write_json(ledger_path, ledger)
    S.write_json(pdir / "evaluation" / "colab_final" / "LATEST.json", rec)
    S.append_lineage(pdir, {"artefact": "memory/memory.sqlite", "kind": "colab_final_import", "results_zip_sha256": zsha, "memories": chk["memories"]})
    return {"status": "IMPORTED", **rec}


def _import_fast(pdir: Path, z, names, man, usage, zsha: str, ledger: dict, ledger_path: Path) -> dict:
    """FAST results: reports only. Evidence-only memory evaluation; no structured semantic memory was built, so the local memory store
    is NOT touched and the 20-session writer-audit candidates are never imported."""
    if man.get("eval_mode") != "FAST" or not set(FAST_LABELS) <= set(man.get("labels") or []):
        raise ImportError_("EVAL_MODE", "a FAST result must carry the evidence-only labels")
    if "memory_snapshot.sqlite" in names:
        raise ImportError_("EVAL_MODE", "a FAST result may not contain a memory snapshot")
    run = Path(man.get("run_dir") or "run").name
    dest = pdir / "evaluation" / "colab_final" / run
    dest.mkdir(parents=True, exist_ok=True)
    for n in names:
        (dest / n).parent.mkdir(parents=True, exist_ok=True)
        (dest / n).write_bytes(z.read(n))
    rec = {"run": run, "eval_mode": "FAST", "labels": list(FAST_LABELS), "imported_at": S.now_iso(), "results_zip_sha256": zsha, "memories": None,
           "memory_action": "NOT_APPLICABLE (FAST: evidence-only memory evaluation; the semantic memory full build was not executed; writer-audit candidates are not imported)",
           "reports_dir": str(dest), "judge_ok_units": usage.get("ok_total"), "judge_planned_units": usage.get("planned_total"), "judge_target_units": usage.get("target_total"),
           "note": "experimental evaluation results (machine judge); not a production semantic memory and not a human review. Recommendations are not applied."}
    ledger[zsha] = rec
    S.write_json(ledger_path, ledger)
    S.write_json(pdir / "evaluation" / "colab_final" / "LATEST.json", rec)
    S.append_lineage(pdir, {"artefact": f"evaluation/colab_final/{run}", "kind": "colab_fast_import", "results_zip_sha256": zsha})
    return {"status": "IMPORTED", **rec}


def latest(root, profile_id: str) -> dict:
    """What Persona Studio shows: memory totals, reader / style / memory / combined results and the recommendations (machine judged)."""
    pdir = S.profile_dir(Path(root) if root else S.default_root(), profile_id)
    p = pdir / "evaluation" / "colab_final" / "LATEST.json"
    if not p.is_file():
        return {"imported": False}
    rec = S.read_json(p)
    d = Path(rec["reports_dir"])
    from .colab_eval import export_names
    ren = export_names(rec.get("eval_mode", "FULL"))
    rd = lambda n: S.read_json(d / ren.get(n, n)) if (d / ren.get(n, n)).is_file() else None   # noqa: E731
    comb, recm = rd("COMBINED_EVALUATION_REPORT.json") or {}, rd("RUNTIME_RECOMMENDATION.json") or {}
    cand = {}
    cpath = d / ren.get("MEMORY_CANDIDATES.jsonl", "MEMORY_CANDIDATES.jsonl")
    if cpath.is_file():
        for l in cpath.read_text(encoding="utf-8").splitlines():
            if l.strip():
                s = json.loads(l)["status"]
                cand[s] = cand.get(s, 0) + 1
    return {"imported": True, "run": rec["run"], "eval_mode": rec.get("eval_mode", "FULL"), "labels": rec.get("labels"), "imported_at": rec["imported_at"],
            "memory": {"active_memories": rec["memories"], "auto_verified": cand.get("AUTO_VERIFIED", 0),
            "pending_review_ambiguous": cand.get("AMBIGUOUS", 0), "rejected": cand.get("REJECTED", 0), "human_reviewed": 0},
            "style": {g: v["deepseek_style_score"] for g, v in (comb.get("style") or {}).items()},
            "memory_scores": {g: v["deepseek_memory_score"] for g, v in (comb.get("memory") or {}).items()}, "combined": comb.get("combined"),
            "reader": (comb.get("reader") or {}), "writer": comb.get("writer"),
            "ablation": {k: {"absolute_diff_points": v.get("absolute_diff_points"), "ci95": (v.get("paired") or {}).get("ci95"),
                             "relative_improvement_pct": v.get("relative_improvement_pct")} for k, v in ((comb.get("ablation") or {}).get("comparisons") or {}).items()}, "recommendation": {"prompt": recm.get("prompt"), "adapter": {k: v for k, v in (recm.get("adapter") or {}).items() if k != "evidence"}},
            "label": ("FAST: evidence-only memory evaluation; semantic memory full build not executed; experimental results; " if rec.get("eval_mode") == "FAST" else "")
                     + "DeepSeek machine judge + deterministic verification; not human review"}
