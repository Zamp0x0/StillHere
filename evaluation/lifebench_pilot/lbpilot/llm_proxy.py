"""A local, counting pass-through for Mem0's LLM calls (Mem0's own DeepSeek provider points its ``deepseek_base_url`` here).

    client (mem0 DeepSeekLLM / openai SDK) -> http://127.0.0.1:<port>/chat/completions -> https://api.deepseek.com/chat/completions

What it does, and nothing else:
* refuses any request whose ``model`` is not the expected one (HTTP 400, recorded as MODEL_VIOLATION): no silent fallback model;
* sets ``thinking: {"type": "disabled"}`` (the same non-thinking mode every other call of this pilot uses; recorded in the protocol);
* forwards the request body otherwise unchanged with the caller's Authorization header, and returns the provider's answer unchanged;
* appends one JSONL row per call: requested / returned model, tokens, latency, HTTP status (no text, no key).

``upstream`` can be replaced (tests use a fake in-process server).
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional


class CountingProxy:
    def __init__(self, *, expected_model: str, upstream: str = "https://api.deepseek.com", log_path: Optional[Path] = None,
                 disable_thinking: bool = True, timeout_s: float = 180.0, deadline_s: float = 300.0):
        """``deadline_s``: a hard wall-clock limit per upstream call. It must stay below the caller's own HTTP timeout (the openai SDK
        that Mem0 uses waits 600 s), so a slow upstream answer becomes a clean 504 the caller retries, instead of an answer written to a
        socket the caller already closed (BrokenPipe). ``timeout_s`` is the socket-level timeout of each read."""
        self.expected, self.upstream, self.disable_thinking, self.timeout = expected_model, upstream.rstrip("/"), disable_thinking, timeout_s
        self.deadline = deadline_s
        self.log_path = Path(log_path) if log_path else None
        self.lock = threading.Lock()
        self.calls = 0
        self.violations = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0, "cache_hit_tokens": 0, "errors": 0, "deadline_504": 0, "client_gone": 0}
        proxy = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                proxy._handle(self, body)
        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):          # never a traceback in the notebook; counted instead
                with proxy.lock:
                    proxy.usage["errors"] += 1
        self.server = Server(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> "CountingProxy":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def _row(self, row: dict) -> None:
        with self.lock:
            if self.log_path:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _reply(self, h, status: int, raw: bytes) -> bool:
        try:
            h.send_response(status)
            h.send_header("Content-Type", "application/json")
            h.send_header("Content-Length", str(len(raw)))
            h.end_headers()
            h.wfile.write(raw)
            return True
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            with self.lock:                                            # the caller gave up first; its own retry sends a new request
                self.usage["client_gone"] += 1
            self._row({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "status": "CLIENT_GONE"})
            return False

    def _fetch(self, out: bytes, headers: dict):
        try:
            r = urllib.request.urlopen(urllib.request.Request(self.upstream + "/chat/completions", data=out, headers=headers, method="POST"), timeout=self.timeout)
            return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read() or b"{}"
        except Exception as e:  # noqa: BLE001 - the caller (mem0) sees a 502 and its own retry logic applies
            return 502, json.dumps({"error": {"message": f"upstream {type(e).__name__}"}}).encode()

    def _handle(self, h, body: bytes) -> None:
        t0 = time.perf_counter()
        at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        path = h.path.split("?", 1)[0]
        if not path.endswith("/chat/completions"):
            self._row({"at": at, "status": "BAD_PATH", "path": path})
            return self._reply(h, 404, b'{"error": {"message": "only /chat/completions is proxied"}}')
        try:
            req = json.loads(body)
        except ValueError:
            return self._reply(h, 400, b'{"error": {"message": "bad json"}}')
        model = req.get("model")
        if model != self.expected:
            with self.lock:
                self.violations += 1
            self._row({"at": at, "status": "MODEL_VIOLATION", "requested_model": model})
            return self._reply(h, 400, json.dumps({"error": {"message": f"model {model!r} refused: only {self.expected!r} is allowed"}}).encode())
        if self.disable_thinking:
            req["thinking"] = {"type": "disabled"}
        out = json.dumps(req, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": h.headers.get("Authorization", "")}
        box: dict = {}
        worker = threading.Thread(target=lambda: box.update(res=self._fetch(out, headers)), daemon=True)
        worker.start()
        worker.join(self.deadline)
        if "res" in box:
            status, raw = box["res"]
        else:                                                          # still waiting upstream: answer before the caller times out
            with self.lock:
                self.usage["deadline_504"] += 1
            status, raw = 504, json.dumps({"error": {"message": f"upstream did not answer within {self.deadline:.0f} s"}}).encode()
        lat = round((time.perf_counter() - t0) * 1000, 1)
        row = {"at": at, "status": status, "requested_model": model, "latency_ms": lat, "messages": len(req.get("messages") or [])}
        try:
            data = json.loads(raw)
            u = data.get("usage") or {}
            row.update(returned_model=data.get("model"), input_tokens=u.get("prompt_tokens"), output_tokens=u.get("completion_tokens"),
                       cache_hit_tokens=u.get("prompt_cache_hit_tokens"), finish_reason=((data.get("choices") or [{}])[0] or {}).get("finish_reason"))
        except (ValueError, AttributeError):
            pass
        with self.lock:
            self.calls += 1
            if status != 200:
                self.usage["errors"] += 1
            for k in ("input_tokens", "output_tokens", "cache_hit_tokens"):
                self.usage[k] += int(row.get(k) or 0)
        self._row(row)
        return self._reply(h, status, raw)

    def summary(self) -> dict:
        with self.lock:
            return {"calls": self.calls, "model_violations": self.violations, **self.usage, "expected_model": self.expected,
                    "thinking_disabled": self.disable_thinking, "upstream": self.upstream}
