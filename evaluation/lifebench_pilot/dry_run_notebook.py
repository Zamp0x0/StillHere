#!/usr/bin/env python3
"""Offline Run-All dry run of the release notebook (CPU; no GPU, no network, no API key, no cost).

Every code cell of ``releases/lifebench_pilot_100qa/StillHere_LifeBench_100QA_Mem0_Colab.ipynb`` is executed in order, in one namespace,
against the REAL input ZIP, the REAL frozen 100 QA of the official dataset (local copy), the REAL official judge code (local checkout)
and the REAL Mem0 SDK + StillHere memory. Replaced, and only these: Google Drive / Colab secrets (a temp directory), pip (skipped),
the DeepSeek API (an in-process fake OpenAI-compatible server: chat for Mem0's extractor, the answers and the judge; embeddings for
Mem0), the embedding model (a deterministic bigram embedder) and the tokenizer (the char estimate). Then the notebook is run AGAIN
(a reconnect) to show that nothing is requested twice. Output: DRY_RUN_REPORT.json. A dry run proves the plumbing, never a score.

    <mem0 env python> evaluation/lifebench_pilot/dry_run_notebook.py --dataset <official json> --eval-dir <LifeBench_eval checkout> --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REL = ROOT / "releases" / "lifebench_pilot_100qa"
NB = REL / "StillHere_LifeBench_100QA_Mem0_Colab.ipynb"
COUNTS = {"chat_mem0": 0, "chat_answer": 0, "chat_judge": 0, "chat_other": 0, "embeddings": 0}


def vec(t):
    v = [0.0] * 1024
    for i in range(len(t) - 1):
        v[int(hashlib.sha256(t[i:i + 2].encode()).hexdigest()[:6], 16) % 1024] += 1.0
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        raw = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/embeddings"):
            COUNTS["embeddings"] += 1
            inp = b["input"] if isinstance(b["input"], list) else [b["input"]]
            return self._send({"object": "list", "model": b["model"], "data": [{"object": "embedding", "index": i, "embedding": vec(t)} for i, t in enumerate(inp)],
                               "usage": {"prompt_tokens": 1, "total_tokens": 1}})
        sys_ = b["messages"][0]["content"] if b["messages"][0]["role"] == "system" else ""
        user = b["messages"][-1]["content"]
        if b.get("response_format") == {"type": "json_object"}:                 # Mem0's extractor
            COUNTS["chat_mem0"] += 1
            new = re.findall(r"\(Conversation date: [^)]*\) [^:\n]*: ([^\n\"]{4,})", user)[-10:]
            text = json.dumps({"memory": [{"text": x.strip()[:120]} for x in new[:3]]}, ensure_ascii=False)
        elif "评分员" in sys_:
            COUNTS["chat_judge"] += 1
            gen = user.split("待评的生成答案：", 1)[1].split("\n", 1)[0] if "待评的生成答案：" in user else ""
            ok = "无法确定" not in gen
            text = json.dumps({"reasoning": "dry run", "point_hits": [], "label": "CORRECT" if ok else "WRONG"}, ensure_ascii=False)
        elif "个人记忆记录" in sys_:
            COUNTS["chat_answer"] += 1
            text = "根据现有记录无法确定" if "没有检索到相关记忆" in user else "（干跑答案）"
        else:
            COUNTS["chat_other"] += 1
            text = "OK"
        return self._send({"id": "dry", "object": "chat.completion", "model": b.get("model"), "system_fingerprint": "dry",
                           "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 100, "completion_tokens": 10}})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    shutil.rmtree(out, ignore_errors=True)
    content, drive = out / "content", out / "content" / "drive" / "MyDrive" / "StillHere" / "lifebench_pilot_100qa"
    (drive / "input").mkdir(parents=True)
    shutil.copy2(REL / "StillHere_LifeBench_Mem0_100QA_Input.zip", drive / "input")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    fake_url = f"http://127.0.0.1:{srv.server_address[1]}"
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda p: None)
    colab.userdata = types.SimpleNamespace(get=lambda k: "dry-run-key")
    import google.protobuf  # noqa: F401 - load the REAL "google" namespace package first (qdrant-client needs google.protobuf)
    import google
    google.colab = colab
    sys.modules["google.colab"] = colab
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]

    def patch(ns):
        C = ns["C"]
        from lbpilot import llm_proxy as LP
        from lbpilot import mem0_memory as MM
        from stillhere.providers import llm as PLM
        from stillhere.retrieval import local_embedding as LE
        sys.path.insert(0, str(ROOT / "tests"))
        from test_lifebench_pilot import FakeEmbedder
        PLM.DEEPSEEK_BASE_URL = fake_url                                         # the answer / judge client
        orig_url = PLM.ProviderConfig.url
        PLM.ProviderConfig.url = property(lambda self: fake_url + "/chat/completions" if self.provider == "deepseek_api" else orig_url.fget(self))
        orig_init = LP.CountingProxy.__init__

        def proxy_init(self, *, expected_model, upstream="https://api.deepseek.com", log_path=None, disable_thinking=True, timeout_s=180.0):
            orig_init(self, expected_model=expected_model, upstream=fake_url, log_path=log_path, disable_thinking=disable_thinking, timeout_s=timeout_s)
        LP.CountingProxy.__init__ = proxy_init
        m_init = MM.Mem0Memory.__init__

        def mem0_init(self, work_dir, **kw):
            kw["embedder"] = {"provider": "openai", "config": {"model": "dry", "api_key": "dry", "openai_base_url": fake_url + "/v1", "embedding_dims": 1024}}
            kw["device"] = "cpu"
            m_init(self, work_dir, **kw)
        MM.Mem0Memory.__init__ = mem0_init
        C.dataset = lambda d, drive_copy=None: Path(a.dataset)
        C.official_eval = lambda d: Path(a.eval_dir)
        C.embedding_model = lambda dest=None: Path(a.out) / "fake_embedding_model"
        def est(text):
            return C.CX.estimate_tokens(text)
        est.mode = "CHAR_ESTIMATE_DRY_RUN"
        C.token_counter = lambda d: est
        LE.get_embedder = lambda d, **kw: FakeEmbedder()
        LE.LocalQueryEncoder = lambda e: (lambda t: FakeEmbedder()._vec(t))
        C.sh = lambda cmd: "dry-run: " + cmd[:40]
        orig_smoke = C.smoke

        def smoke(*args, **kw):
            return orig_smoke(*args, **kw, expect_embedder="OpenAIEmbedding")        # the dry run's fake embedder (see the module doc)
        C.smoke = smoke

    def run_all(tag):
        ns = {"__name__": "__main__"}
        for i, src in enumerate(cells):
            src = src.replace("/content", str(content).replace("\\", "/"))
            src = src.replace('subprocess.run(f"pip install -q -r', 'subprocess.run(f"echo skip pip install -q -r')
            src = src.replace('subprocess.run("nvidia-smi', 'subprocess.run("echo nvidia-smi')
            src = src.replace('AUDIT["embedder_class"] == "HuggingFaceEmbedding"', 'AUDIT["embedder_class"] == "OpenAIEmbedding"')   # dry-run embedder
            exec(compile(src, f"<{tag} cell {i}>", "exec"), ns)
            if "from lbpilot import colab as C" in src:
                patch(ns)
        return ns
    ns1 = run_all("first")
    first = dict(COUNTS)
    for k in ("PROXY",):
        ns1[k].stop()
    ns1["STILL"].close()
    ns1["MEM0"].close()
    ns2 = run_all("again")
    again = {k: COUNTS[k] - first[k] for k in COUNTS}
    ns2["PROXY"].stop()
    run_dir = Path(str(ns2["RUN_DIR"]))
    m = json.loads((run_dir / "METRICS.json").read_text(encoding="utf-8"))
    st = json.loads((run_dir / "STATE.json").read_text(encoding="utf-8"))
    rep = {"first_run_requests": first, "second_run_new_requests": again, "stages": {k: v["status"] for k, v in st["stages"].items()},
           "formal": m["formal"], "completeness": {g: x["completeness"] for g, x in m["groups"].items()},
           "mem0_add": st.get("mem0_add"), "mem0_memories_final": st.get("mem0_memories_final"), "added_through": st.get("added_through"),
           "smoke": json.loads((run_dir / "smoke" / "SMOKE_REPORT.json").read_text(encoding="utf-8"))["status"],
           "exports": sorted(p.name for p in (drive / "exports").iterdir()),
           "note": "DRY RUN with fake API / embedder: plumbing only; the numbers in METRICS.json are meaningless"}
    (out / "DRY_RUN_REPORT.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    srv.shutdown()
    ok = rep["formal"] and not any(again[k] for k in ("chat_answer", "chat_judge")) and rep["smoke"] == "PASS"
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
