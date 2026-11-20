"""Colab glue for the notebook ``StillHere_LifeBench_100QA_Mem0_Colab.ipynb``: one function per stage, so the notebook cells stay short.
Everything durable goes to ``RUN_DIR`` on Google Drive; databases and models live on the Colab disk (``/content``)."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Optional

from . import context as CX
from . import protocol as P
from .records import write_json_atomic

EMB_DIR = Path("/content/models/Qwen3-Embedding-0.6B")


def sh(cmd: str) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def environment(run_dir: Path) -> dict:
    gpu = sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
    import importlib.metadata as md
    vers = {}
    for p in ("mem0ai", "qdrant-client", "sentence-transformers", "transformers", "torch", "openai", "numpy", "aiohttp"):
        try:
            vers[p] = md.version(p)
        except md.PackageNotFoundError:
            vers[p] = None
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "gpu": gpu, "packages": vers, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "protocol_sha256": P.PROTOCOL_SHA256}
    write_json_atomic(Path(run_dir) / "ENVIRONMENT.json", env)
    return env


def read_key(secret_file: Optional[Path] = None) -> str:
    """DEEPSEEK_API_KEY from Colab Secrets, else from a private env file on Drive. Never printed."""
    try:
        from google.colab import userdata  # type: ignore
        k = userdata.get("DEEPSEEK_API_KEY")
        if k:
            return k
    except Exception:  # noqa: BLE001
        pass
    if secret_file and Path(secret_file).is_file():
        for line in Path(secret_file).read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("DEEPSEEK_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("DEEPSEEK_API_KEY not found: add it in Colab > Secrets (key icon) and allow this notebook, or put it in the private env file")


def chat_client(key: str, run_dir: Path, role: str, *, max_tokens: int):
    from stillhere.providers import llm as PLM
    cfg = PLM.ProviderConfig(provider="deepseek_api", model=P.ANSWER["model"], temperature=0.0, max_output_tokens=max_tokens,
                             max_attempts=int(P.ANSWER["max_attempts"]), timeout_s=120.0)
    return PLM.ChatClient(cfg, env={"DEEPSEEK_API_KEY": key}, usage_log=PLM.UsageLog(Path(run_dir) / "logs" / f"api_usage_{role}.jsonl"), role=role)


def preflight(client) -> dict:
    from stillhere.providers import llm as PLM
    r = PLM.check_connection(client)
    if not r["ok"]:
        raise RuntimeError("DeepSeek preflight returned no text")
    return r


def dataset(data_dir: Path, drive_copy: Optional[Path] = None) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / Path(P.DATASET["file"]).name
    from .select import sha256_file
    if out.is_file() and sha256_file(out) == P.DATASET["sha256"]:
        return out
    if drive_copy and Path(drive_copy).is_file():
        shutil.copy2(drive_copy, out)
    else:
        tmp = out.with_suffix(".partial")
        urllib.request.urlretrieve(P.DATASET["url"], tmp)
        os.replace(tmp, out)
    got = sha256_file(out)
    if got != P.DATASET["sha256"]:
        raise RuntimeError(f"DATASET_SHA_MISMATCH {got}")
    return out


def official_eval(dest: Path) -> Path:
    dest = Path(dest)
    if not (dest / ".git").is_dir():
        print(sh(f"git clone -q {P.OFFICIAL_EVAL['repo']} {dest}"))
    print(sh(f"cd {dest} && git checkout -q {P.OFFICIAL_EVAL['commit']}"))
    head = sh(f"cd {dest} && git rev-parse HEAD")
    if head != P.OFFICIAL_EVAL["commit"]:
        raise RuntimeError(f"LifeBench_eval is at {head}, expected {P.OFFICIAL_EVAL['commit']}")
    return dest


def embedding_model(dest: Path = EMB_DIR) -> Path:
    from huggingface_hub import snapshot_download
    rev = P.STILLHERE["embedding"]["revision"]
    p = snapshot_download(P.STILLHERE["embedding"]["id"], revision=rev, local_dir=str(dest))
    return Path(p)


def token_counter(model_dir: Path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir))

    def count(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])
    count.mode = "QWEN3_EMBEDDING_TOKENIZER"
    return count


def dense_index(run, embedder, *, scratch: Path) -> dict:
    """Build (or restore from Drive) the StillHere V2D index over the whole timeline; the time sidecar masks later rows at search."""
    from . import stillhere_memory as SM
    inp = run.inputs
    cache = run.run_dir / "cache" / "dense.tar"
    idx_dir, idx_store = Path(scratch) / "dense", Path(scratch) / "index_store.sqlite"
    if not (idx_dir / "index" / "embedding_index_manifest.json").is_file() and cache.is_file():
        with tarfile.open(cache) as t:
            t.extractall(scratch, filter="data")
    res = SM.build_dense(idx_store, idx_dir, inp["events"], inp["slug"], inp["name"], inp["assistant"], embedder, dataset_sha256=P.DATASET["sha256"],
                         locator=P.DATASET["url"], batch_size=64)
    if not cache.is_file() and (idx_dir / "index" / "embedding_index_manifest.json").is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tar.partial")
        with tarfile.open(tmp, "w") as t:
            t.add(idx_dir, arcname="dense")
            t.add(idx_store, arcname="index_store.sqlite")
        os.replace(tmp, cache)
    st = "PASS" if res.get("status") == "COMPLETE" else "INCOMPLETE"
    run.stage("09_stillhere_dense_index", st, build_status=res.get("status"),
              **{k: v for k, v in res.items() if k in ("documents", "chunks", "index_sha256", "seconds", "messages", "reused")})
    return res


def report(run_dir: Path) -> str:
    m = json.loads((Path(run_dir) / "METRICS.json").read_text(encoding="utf-8"))
    lines = [f"# LifeBench 2.0 pilot ({P.SCOPE})", "", f"formal: {m['formal']} (every group complete)", "",
             "| group | micro | macro | answerable | unanswerable | false abstention | unsupported | evidence tokens | strict id recall | semantic coverage |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g, x in m["groups"].items():
        e = x["evidence"]
        lines.append(f"| {g} | {x['micro_accuracy']} | {x['macro_accuracy']} | {x['answerable']['correct']}/{x['answerable']['n']} | "
                     f"{x['unanswerable']['correct']}/{x['unanswerable']['n']} | {x['false_abstention']['count']}/{x['false_abstention']['of']} | "
                     f"{x['unsupported_claim']['count']}/{x['unsupported_claim']['of']} | {e['evidence_tokens_mean']} | {e['strict_id_recall_mean']} | {e['semantic_coverage_mean']} |")
    lines += ["", "| pair (A - B) | diff | 95% CI | W/T/L | decidable |", "|---|---:|---|---|---|"]
    for k, p in m["pairs"].items():
        lines.append(f"| {k.replace('__vs__', ' - ')} | {p['diff']} | {p['ci95']} | {p['wins']}/{p['ties']}/{p['losses']} | {p['decidable']} |")
    txt = "\n".join(lines) + "\n\nThis is a one-persona pilot; it is not the official LifeBench ranking.\n"
    (Path(run_dir) / "REPORT.md").write_text(txt, encoding="utf-8")
    return txt


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


__all__ = ["CX", "environment", "read_key", "chat_client", "preflight", "dataset", "official_eval", "embedding_model", "token_counter", "dense_index", "report"]


def smoke(run_dir: Path, scratch: Path, *, fixture: Path, key: str, embedder, emb_dir: Path, official, count,
          expect_embedder: str = "HuggingFaceEmbedding") -> dict:
    """A few-cent end-to-end check BEFORE the formal run: synthetic fixture, 3 days, 2 questions, the REAL Mem0 SDK, the REAL DeepSeek
    answer / judge calls, the real embedder. Writes ``smoke/SMOKE_REPORT.json``; the formal stores are never touched."""
    from . import answer as A
    from . import judge as J
    from . import llm_proxy as LP
    from . import mem0_memory as MM
    from . import stillhere_memory as SM
    from . import timeline as TL
    sdir, work = Path(run_dir) / "smoke", Path(scratch) / "smoke"
    shutil.rmtree(work, ignore_errors=True)
    sdir.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(fixture).read_text(encoding="utf-8"))
    fx_sha = sha256(Path(fixture))
    person = data[0]
    ev = TL.build(person)
    days = sorted(TL.by_date(ev))[:3]
    evs = [e for e in ev if e["date"] in days]
    qs = [q for q in person["qa"] if q["ask_time"][:10] == days[-1]][:2]
    conv = person["conversation"]
    px = LP.CountingProxy(expected_model=P.MEM0["llm"]["model"], log_path=sdir / "mem0_llm_proxy.jsonl").start()
    out = {"days": days, "events": len(evs), "questions": [q["question_id"] for q in qs]}
    try:
        sh_ = SM.StillHereMemory(work / "run.sqlite", "smoke", conv["speaker_a"], conv["speaker_b"], dataset_sha256=fx_sha, locator="smoke")
        SM.build_dense(work / "index.sqlite", work / "dense", evs, "smoke", conv["speaker_a"], conv["speaker_b"], embedder, dataset_sha256=fx_sha, locator="smoke")
        from stillhere.retrieval import local_embedding as LE
        dense, why = SM.open_dense(work / "dense", LE.LocalQueryEncoder(embedder))
        emb = {"provider": "huggingface", "config": {"model": str(emb_dir), "embedding_dims": 1024, "model_kwargs": {"device": "cuda"}}}
        m = MM.Mem0Memory(work / "mem0", user_id="smoke", proxy_url=px.url, api_key=key, collection="smoke", embedder=emb)
        out["mem0_audit"] = m.runtime_audit()
        for d in days:
            day = [e for e in evs if e["date"] == d]
            sh_.add_day(day)
            out.setdefault("mem0_add", []).append(m.add_day(day))
        out["mem0_memories"] = m.count()
        chat = chat_client(key, sdir, "smoke", max_tokens=1024)
        rows = []
        for q in qs:
            at = SM.ask_at(q["ask_time"])
            query = TL.search_query(q["question"])
            res = {"M0_NO_MEMORY": {"entries": []}, "M1_STILLHERE_V2L": sh_.search(query, at), "M2_STILLHERE_V2D": sh_.search(query, at, dense=dense),
                   "M3_MEM0_OSS": m.search(query)}
            for g, r in res.items():
                ctx = CX.render(r["entries"], budget=int(P.EVIDENCE["token_budget"]), count=count)
                a = chat.chat(A.messages(conv["speaker_a"], q["question"], q["ask_time"], ctx["text"]))
                j = chat.chat(official.messages(q, a.text))
                v = official.parse(j.text)
                rows.append({"group": g, "question_id": q["question_id"], "entries": ctx["entries_kept"], "tokens": ctx["tokens"], "answer_model": a.returned_model,
                             "judge_parsed": v is not None, "judge_correct": None if v is None else v["is_correct"]})
        out["rows"] = rows
        out["dense"] = why
        sh_.close()
        m.close()
    finally:
        px.stop()
    out["proxy"] = px.summary()
    ok = (out["proxy"]["calls"] > 0 and out["proxy"]["model_violations"] == 0 and out["mem0_memories"] > 0 and all(r["judge_parsed"] for r in out["rows"])
          and out["dense"] == "OK" and out["mem0_audit"]["llm_class"] == "DeepSeekLLM" and out["mem0_audit"]["embedder_class"] == expect_embedder)
    out["status"] = "PASS" if ok else "FAIL"
    write_json_atomic(sdir / "SMOKE_REPORT.json", out)
    return out
