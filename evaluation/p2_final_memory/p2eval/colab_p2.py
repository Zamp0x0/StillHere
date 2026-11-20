"""Colab glue for StillHere_P2_Final_Memory_Eval_Colab.ipynb (reuses the pilot's helpers for key, client, dataset, official judge)."""
from __future__ import annotations

import hashlib
import json
import os
import tarfile
from pathlib import Path

from lbpilot import colab as LC
from lbpilot import context as CX
from lbpilot import judge as J
from lbpilot import stillhere_memory as SM
from lbpilot.records import write_json_atomic

from . import metrics_p2 as M
from . import protocol as P

environment, read_key, chat_client, preflight, dataset, official_eval, embedding_model, token_counter = (
    LC.environment, LC.read_key, LC.chat_client, LC.preflight, LC.dataset, LC.official_eval, LC.embedding_model, LC.token_counter)


def verify_old_fingerprint(code_root: Path, manifest: dict) -> dict:
    """Every file of the OLD path must be byte-identical to the one fingerprinted at build time (= the pilot's code)."""
    bad = []
    for rel, h in manifest["old_fingerprint"]["files"].items():
        p = Path(code_root) / rel
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != h:
            bad.append(rel)
    if bad:
        raise RuntimeError(f"OLD_FINGERPRINT_MISMATCH: {bad}")
    return {"files": len(manifest["old_fingerprint"]["files"]), "tree_sha256": manifest["old_fingerprint"]["tree_sha256"]}


def dense_index(run, experiment: str, embedder, *, scratch: Path) -> dict:
    inp = run.inputs[experiment]
    cache = run.run_dir / "cache" / f"dense_{experiment}.tar"
    idx_dir, idx_store = Path(scratch) / f"dense_{experiment}", Path(scratch) / f"index_store_{experiment}.sqlite"
    if not (idx_dir / "index" / "embedding_index_manifest.json").is_file() and cache.is_file():
        with tarfile.open(cache) as t:
            t.extractall(scratch, filter="data")
    res = SM.build_dense(idx_store, idx_dir, inp["events"], inp["slug"], inp["name"], inp["assistant"], embedder, dataset_sha256=P.DATASET["sha256"],
                         locator=P.DATASET["url"], batch_size=64)
    if not cache.is_file() and (idx_dir / "index" / "embedding_index_manifest.json").is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tar.partial")
        with tarfile.open(tmp, "w") as t:
            t.add(idx_dir, arcname=idx_dir.name)
            t.add(idx_store, arcname=idx_store.name)
        os.replace(tmp, cache)
    run.stage(f"dense_index_{experiment}", "PASS" if res.get("status") == "COMPLETE" else "INCOMPLETE", build_status=res.get("status"),
              **{k: v for k, v in res.items() if k in ("documents", "index_sha256", "seconds", "messages", "reused")})
    return res


def load_known_old(pkg: Path) -> dict:
    """The pilot's stored M2_STILLHERE_V2D records of the known set (copied into the package, sha-checked by SHA256SUMS)."""
    base = Path(pkg) / "old_baseline"

    def rows(n):
        return [json.loads(l) for l in open(base / n, encoding="utf-8") if l.strip()]
    ans = {(r["group"], r["question_id"]): r for r in rows("answers_M2.jsonl")}
    ctx = {(r["group"], r["question_id"]): r for r in rows("contexts_M2.jsonl")}
    jud = {(r["group"], r["question_id"], r["repeat"]): r for r in rows("judge_M2.jsonl")}
    return {"answers": ans, "contexts": ctx, "judgments": jud}


def calibrate(run, official: J.Official, chat) -> dict:
    res = []
    for c in J.calibration_items():
        r = chat(official.messages(c["qa"], c["answer"]))
        v = official.parse(r.text)
        res.append({"id": c["id"], "expect": c["expect"], "got": None if v is None else v["is_correct"], "ok": v is not None and v["is_correct"] == c["expect"], **r.usage()})
    write_json_atomic(run.run_dir / "JUDGE_CALIBRATION.json", res)
    return run.stage("judge_calibration", "PASS" if all(x["ok"] for x in res) else "INCOMPLETE", passed=sum(x["ok"] for x in res), of=len(res))


def metrics(run, known_old: dict) -> dict:
    out = M.compute(run.inputs, run.answers.done(), run.judgments.done(), run.contexts.done(), known_old=known_old)
    out["run"] = {"stages": {k: v["status"] for k, v in run.state["stages"].items()}, "spent_usd_peak_estimate": run.spent()}
    write_json_atomic(run.run_dir / "P2_METRICS.json", out)
    np = out.get("new_person", {})
    run.stage("metrics", "PASS" if np.get("adoption", {}).get("decision") in ("ADOPT_P2", "KEEP_OLD") else "INCOMPLETE",
              decision=np.get("adoption", {}).get("decision"))
    return out


def report(out: dict) -> str:
    L = ["# StillHere P2 final memory evaluation", ""]
    np = out.get("new_person")
    if np:
        o, n, p = np["OLD_V2D"], np["NEW_P2_V2D"], np["primary_answerable"]
        L += ["## Fresh person (decision set)", "", "| metric | OLD_V2D | NEW_P2_V2D |", "|---|---:|---:|"]
        for k in ("micro", "answerable", "unanswerable", "multi_hop", "temporal", "knowledge_update"):
            L.append(f"| {k} | {o[k]['correct']}/{o[k]['n']} | {n[k]['correct']}/{n[k]['n']} |")
        L += [f"| false abstention | {o['false_abstention']['count']} | {n['false_abstention']['count']} |",
              f"| unsupported | {o['unsupported']['count']} | {n['unsupported']['count']} |",
              f"| gold recall | {o['gold_recall_mean']} | {n['gold_recall_mean']} |", "",
              f"Primary (answerable, NEW - OLD): {p['diff']} CI95 {p['ci95']} W/T/L {p['wins']}/{p['ties']}/{p['losses']} McNemar p {p['mcnemar_p']}",
              f"Decision: **{np['adoption']['decision']}** ({np['adoption']['wording']})", ""]
    ks = out.get("known_set")
    if ks:
        L += ["## Known set (regression only)", "", f"NEW answerable {ks['NEW_P2_V2D']['answerable']} vs stored OLD {ks['OLD_V2D_stored']['answerable']}; pair {ks['answerable_pair']}"]
    return "\n".join(L) + "\n"


def smoke(run_dir: Path, scratch: Path, *, fixture: Path, key: str, embedder, official, count) -> dict:
    """A few-cent check before the formal run: synthetic fixture, OLD and NEW retrieval, one real answer each, one judge call each."""
    from lbpilot import answer as A
    from lbpilot import timeline as TL
    from stillhere.retrieval import local_embedding as LE
    from . import answer_p2 as A2
    from . import stillhere_p2 as SP
    import shutil
    work, sdir = Path(scratch) / "smoke", Path(run_dir) / "smoke"
    shutil.rmtree(work, ignore_errors=True)
    sdir.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(fixture).read_text(encoding="utf-8"))
    person = data[0]
    ev = TL.build(person)
    conv = person["conversation"]
    fx = hashlib.sha256(Path(fixture).read_bytes()).hexdigest()
    sh = SM.StillHereMemory(work / "run.sqlite", "smoke", conv["speaker_a"], conv["speaker_b"], dataset_sha256=fx, locator="smoke")
    for d, evs in sorted(TL.by_date(ev).items()):
        sh.add_day(evs)
    SM.build_dense(work / "ix.sqlite", work / "dense", ev, "smoke", conv["speaker_a"], conv["speaker_b"], embedder, dataset_sha256=fx, locator="smoke")
    dense, why = SM.open_dense(work / "dense", LE.LocalQueryEncoder(embedder))
    chat = chat_client(key, sdir, "smoke", max_tokens=1024)
    rows = []
    for q in person["qa"][:2]:
        at = SM.ask_at(q["ask_time"])
        query = TL.search_query(q["question"])
        old = CX.render(sh.search(query, at, dense=dense)["entries"], budget=4000, count=count)
        new = SP.render(SP.search(sh, query, at, dense=dense), count)
        for g, ctx, fn in (("OLD_V2D", old, A.messages), ("NEW_P2_V2D", new, A2.messages)):
            a = chat.chat(fn(conv["speaker_a"], q["question"], q["ask_time"], ctx["text"]))
            v = official.parse(chat.chat(official.messages(q, a.text)).text)
            rows.append({"group": g, "question_id": q["question_id"], "entries": ctx["entries_kept"], "tokens": ctx["tokens"], "answer_model": a.returned_model,
                         "judge_parsed": v is not None})
    sh.close()
    out = {"rows": rows, "dense": why, "status": "PASS" if why == "OK" and all(r["judge_parsed"] for r in rows) else "FAIL"}
    write_json_atomic(sdir / "SMOKE_REPORT.json", out)
    return out
