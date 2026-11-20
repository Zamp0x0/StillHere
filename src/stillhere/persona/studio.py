"""Persona Studio: a minimal LOCAL web UI over the same service API (standard library only; no login, no deployment).

    stillhere persona studio [--port 8765] [--backend fake|hf] [--base-dir <Qwen3-8B>]

Binds 127.0.0.1 only and answers only requests whose Host is localhost / 127.0.0.1 (no DNS rebinding). Every action calls the Python
services (pipeline, card review, memory_store, quality, runtime); the page holds no cleaning / memory logic of its own. Private chat
text is shown only to the local browser.
"""
from __future__ import annotations

import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import parse_qs, urlparse

from . import store as S

PAGE = Path(__file__).with_name("studio.html")


class Studio:
    def __init__(self, root=None, backend: Optional[str] = None, base_dir: Optional[str] = None):
        self.root = Path(root) if root else S.default_root()
        self.backend, self.base_dir = backend, base_dir
        self.jobs: Dict[str, dict] = {}
        self.runtimes: Dict[str, object] = {}
        self.lock = threading.Lock()

    # ---------------------------------------------------------------- actions (plain dict in, plain dict out)
    def profiles(self, _q):
        out = []
        for p in sorted(self.root.glob("*/profile.json")) if self.root.is_dir() else []:
            d = S.read_json(p)
            out.append({"profile_id": d["profile_id"], "display_name": d["target"].get("display_name"), "role": d["target"]["role"], "created_at": d.get("created_at")})
        from . import config as CF
        return {"profiles": out, "backend": self.backend or CF.load().backend}

    def speakers(self, q):
        from . import service as svc
        return {"speakers": svc.speakers(q["input"])}

    def build(self, q):
        from . import pipeline as PL
        jid = uuid.uuid4().hex[:8]
        job = {"id": jid, "status": "RUNNING", "log": []}
        self.jobs[jid] = job

        def work():
            try:
                job["result"] = PL.run(q["input"], speaker=str(q["speaker"]), root=self.root, target=q.get("target", "cpu-only"),
                                       confirm_cloud=bool(q.get("confirm_cloud")), base_dir=q.get("base_dir") or self.base_dir, log=job["log"].append)
                job["status"] = "DONE"
            except Exception as e:  # noqa: BLE001
                job.update(status="FAILED", error=f"{type(e).__name__}: {e}")
        threading.Thread(target=work, daemon=True).start()
        return {"job": jid}

    def job(self, q):
        j = self.jobs.get(q["id"])
        return {k: v for k, v in (j or {"status": "UNKNOWN"}).items()}

    def state(self, q):
        from . import pipeline as PL
        return PL.state(self.root, q["profile"])

    def card(self, q):
        from . import card as K
        cur = K.current(S.profile_dir(self.root, q["profile"]))
        if cur is None:
            return {"card": None}
        return {"card_id": cur["card_id"], "manifest": cur["manifest"], "prompt": cur["prompt"], "editable": {k: cur["card"].get(k) for k in K.EDITABLE},
                "summary": {k: cur["card"].get(k) for k in ("reply_length", "common_expressions", "frequent_short_replies", "examples", "language")}}

    def card_edit(self, q):
        from . import card as K
        pdir = S.profile_dir(self.root, q["profile"])
        if q.get("approve"):
            r = K.review(pdir, approve=True)
        else:
            cur = K.current(pdir)
            r = K.review(pdir, edited={**cur["card"], **{k: v for k, v in q["edits"].items() if k in K.EDITABLE}})
        self.runtimes.pop(q["profile"], None)
        return {"card_id": r["card_id"], "review_status": r["review_status"], "prompt_sha256": r["prompt_sha256"]}

    def memory(self, q):
        from . import memory_store as MS
        pdir = S.profile_dir(self.root, q["profile"])
        rep = pdir / "evaluation" / "memory_quality_report.json"
        return {"status": MS.status(self.root, q["profile"]), "report": S.read_json(rep) if rep.is_file() else None}

    def memory_query(self, q):
        from datetime import datetime
        from . import memory_store as MS
        at = datetime.fromisoformat(q["ask_time"]) if q.get("ask_time") else None
        r = MS.query(self.root, q["profile"], q["q"], ask_time=at)
        return {"items": r["items"], "ask_time": r["ask_time"]}

    def training(self, q):
        from . import adapters as AD
        pdir = S.profile_dir(self.root, q["profile"])
        pk = []
        for d in sorted((pdir / "packages").glob("p*-*")):
            if (d / "RELEASE_MANIFEST.json").is_file():
                rm = S.read_json(d / "RELEASE_MANIFEST.json")
                pk.append({"package_id": d.name, "kind": rm.get("kind", "training"), "target": rm.get("target"), "dir": str(d),
                           "notebook": next((str(x) for x in d.glob("*.ipynb")), None)})
        return {"packages": pk, "adapters": AD.installed(pdir), "gpu_note": "Qwen3-8B QLoRA trains on Colab L4 (or a >= 20 GiB GPU); this machine is checked by persona doctor"}

    def evaluation(self, q):
        pdir = S.profile_dir(self.root, q["profile"])
        qr, rc = pdir / "evaluation" / "persona_quality_report.json", pdir / "runtime" / "runtime_config.json"
        rep = S.read_json(qr) if qr.is_file() else None
        return {"quality": {k: rep[k] for k in ("status", "eval_run", "failures", "retrain_recommendation")} | {"sections": {k: rep["sections"][k] for k in (
            "base_vs_adapter", "reply_length", "generic_short_reply_rate", "prompt_improvement_candidates", "recommended_runtime")}} if rep else None,
                "runtime": S.read_json(rc) if rc.is_file() else None}

    def final_eval(self, q):
        from . import results_import as RI
        return RI.latest(self.root, q["profile"])

    def runtime_select(self, q):
        from . import quality as Q
        self.runtimes.pop(q["profile"], None)
        return Q.select_runtime(self.root, q["profile"], override_adapter=q.get("adapter"), override_memory=q.get("memory"))

    def chat(self, q):
        from .runtime import PersonaRuntime
        with self.lock:
            rt = self.runtimes.get(q["profile"])
            if rt is None:
                rt = PersonaRuntime.from_config(self.root, q["profile"], backend=self.backend, base_dir=self.base_dir)
                self.runtimes[q["profile"]] = rt
            r = rt.reply([(h["role"], h["text"]) for h in q["history"]])
        return {k: r.get(k) for k in ("text", "mode", "adapter", "memory", "generator", "synthetic", "loop_guard", "redacted", "prompt_sha256", "evidence_chars")}

    ROUTES = {"GET": {"/api/profiles": "profiles", "/api/state": "state", "/api/card": "card", "/api/memory": "memory", "/api/training": "training",
                      "/api/evaluation": "evaluation", "/api/job": "job", "/api/final_eval": "final_eval"},
              "POST": {"/api/speakers": "speakers", "/api/build": "build", "/api/card": "card_edit", "/api/memory/query": "memory_query",
                       "/api/runtime": "runtime_select", "/api/chat": "chat"}}

    def handle(self, method: str, path: str, query: dict, body: dict):
        name = self.ROUTES.get(method, {}).get(path)
        if name is None:
            return 404, {"error": "not found"}
        try:
            return 200, getattr(self, name)({**query, **body})
        except Exception as e:  # noqa: BLE001 - shown in the page
            return 400, {"error": f"{type(e).__name__}: {e}"}


def handler_for(studio: Studio):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _ok_host(self) -> bool:
            host = (self.headers.get("Host") or "").split(":")[0]
            return host in ("127.0.0.1", "localhost")

        def _send(self, code: int, obj=None, html: Optional[bytes] = None):
            b = html if html is not None else json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8" if html is not None else "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)

        def _go(self, method):
            if not self._ok_host():
                return self._send(403, {"error": "local only"})
            u = urlparse(self.path)
            if method == "GET" and u.path in ("/", "/index.html"):
                return self._send(200, html=PAGE.read_bytes())
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            body = {}
            if method == "POST":
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            code, obj = studio.handle(method, u.path, q, body)
            self._send(code, obj)

        def do_GET(self):
            self._go("GET")

        def do_POST(self):
            self._go("POST")
    return H


def serve(root=None, port: int = 8765, backend: Optional[str] = None, base_dir: Optional[str] = None):
    studio = Studio(root, backend, base_dir)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler_for(studio))
    return httpd, studio
