#!/usr/bin/env python3
"""Offline Run-All dry run of the P2 release notebook (CPU; no GPU, no network, no API key, no cost).

Every code cell runs in order against the REAL input ZIP, the REAL frozen question sets, the REAL official judge code (local checkout)
and the REAL StillHere OLD / P2 code; replaced: Drive / Colab secrets (a temp dir), pip (skipped), the DeepSeek API (an in-process fake
server), the embedding model (a deterministic bigram embedder) and the tokenizer (char estimate). The notebook is then run AGAIN to show
that nothing is requested twice. Output: DRY_RUN_REPORT.json. Plumbing only; the numbers are meaningless.

    python evaluation/p2_final_memory/dry_run_notebook.py --dataset <official json> --eval-dir <LifeBench_eval checkout> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REL = ROOT / "releases" / "p2_final_memory"
NB = REL / "StillHere_P2_Final_Memory_Eval_Colab.ipynb"
COUNTS = {"answer": 0, "judge": 0, "other": 0}


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        sys_ = b["messages"][0]["content"] if b["messages"][0]["role"] == "system" else ""
        user = b["messages"][-1]["content"]
        if "评分员" in sys_:
            COUNTS["judge"] += 1
            gen = user.split("待评的生成答案：", 1)[1].split("\n", 1)[0] if "待评的生成答案：" in user else ""
            text = json.dumps({"reasoning": "dry", "point_hits": [], "label": "CORRECT" if "无法确定" not in gen else "WRONG"}, ensure_ascii=False)
        elif "个人记忆记录" in sys_:
            COUNTS["answer"] += 1
            text = "根据现有记录无法确定" if "没有检索到相关记忆" in user else "（干跑答案）"
        else:
            COUNTS["other"] += 1
            text = "OK"
        raw = json.dumps({"id": "dry", "model": b.get("model"), "system_fingerprint": "dry", "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                          "finish_reason": "stop"}], "usage": {"prompt_tokens": 100, "completion_tokens": 10}}, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    shutil.rmtree(out, ignore_errors=True)
    content = out / "content"
    drive = content / "drive" / "MyDrive" / "StillHere" / "p2_final_memory"
    (drive / "input").mkdir(parents=True)
    shutil.copy2(REL / "StillHere_P2_Final_Memory_Eval_Input.zip", drive / "input")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    fake_url = f"http://127.0.0.1:{srv.server_address[1]}"
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda p: None)
    colab.userdata = types.SimpleNamespace(get=lambda k: "dry-run-key")
    try:
        import google  # noqa: F401
    except ImportError:
        sys.modules["google"] = types.ModuleType("google")
    sys.modules["google"].colab = colab
    sys.modules["google.colab"] = colab
    cells = ["".join(c["source"]) for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]

    def patch(ns):
        C = ns["C"]
        from lbpilot import colab as LC
        from stillhere.providers import llm as PLM
        from stillhere.retrieval import local_embedding as LE
        sys.path.insert(0, str(ROOT / "tests"))
        from test_lifebench_pilot import FakeEmbedder
        orig = PLM.ProviderConfig.url
        PLM.ProviderConfig.url = property(lambda self: fake_url + "/chat/completions" if self.provider == "deepseek_api" else orig.fget(self))
        C.dataset = LC.dataset = lambda d, drive_copy=None: Path(a.dataset)
        C.official_eval = LC.official_eval = lambda d: Path(a.eval_dir)
        C.embedding_model = lambda dest=None: Path(a.out) / "fake_embedding_model"

        def est(text):
            from lbpilot import context as CX
            return CX.estimate_tokens(text)
        est.mode = "CHAR_ESTIMATE_DRY_RUN"
        C.token_counter = lambda d: est
        LE.get_embedder = lambda d, **kw: FakeEmbedder()
        LE.LocalQueryEncoder = lambda e: (lambda t: FakeEmbedder()._vec(t))
        LC.sh = lambda cmd: "dry-run"

    def run_all(tag):
        ns = {"__name__": "__main__"}
        for i, src in enumerate(cells):
            src = src.replace("/content", str(content).replace("\\", "/"))
            src = src.replace('subprocess.run(f"pip install -q -r', 'subprocess.run(f"echo skip pip install -q -r')
            src = src.replace('subprocess.run("nvidia-smi', 'subprocess.run("echo nvidia-smi')
            exec(compile(src, f"<{tag} cell {i}>", "exec"), ns)
            if "from p2eval import colab_p2 as C" in src:
                patch(ns)
        return ns
    run_all("first")
    first = dict(COUNTS)
    ns = run_all("again")
    again = {k: COUNTS[k] - first[k] for k in COUNTS}
    run_dir = Path(str(ns["RUN_DIR"]))
    st = json.loads((run_dir / "STATE.json").read_text(encoding="utf-8"))
    m = json.loads((run_dir / "P2_METRICS.json").read_text(encoding="utf-8"))
    rep = {"first_run_requests": first, "second_run_new_requests": again, "stages": {k: v["status"] for k, v in st["stages"].items()},
           "decision": m["new_person"]["adoption"]["decision"], "complete": [m["new_person"]["OLD_V2D"]["complete"], m["new_person"]["NEW_P2_V2D"]["complete"]],
           "known_set_new_judged": m["known_set"]["NEW_P2_V2D"]["judged"], "leaks": [m["new_person"][g]["future_leaks"] for g in ("OLD_V2D", "NEW_P2_V2D")],
           "exports": sorted(p.name for p in (drive / "exports").iterdir()), "note": "DRY RUN with fake API / embedder: plumbing only"}
    (out / "DRY_RUN_REPORT.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    srv.shutdown()
    return 0 if (all(rep["complete"]) and again["answer"] == 0 and again["judge"] == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
