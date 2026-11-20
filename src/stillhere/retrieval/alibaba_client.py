"""DashScope embedding, rerank and chat calls, with a persistent ledger and a hard budget. Standard library only.

Every attempt is written to an append-only ledger before the next one starts, so running again never resets the spend.
A request is refused when its conservative estimate could pass the budget. A timeout may still have been billed, so it
counts at the estimate and is marked uncertain. The key is read from the environment or ``.env`` and is never logged:
everything an error shows goes through ``redact``.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import threading
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

PRICES_CNY_PER_MILLION = {"qwen3.7-text-embedding-flash": 0.125, "text-embedding-v4": 0.5, "qwen3-rerank": 0.5,
                          "qwen3.7-text-rerank": 0.5}
CHAT_PRICES_CNY_PER_MILLION = {"qwen3.7-flash-2026-07-15": (0.2, 0.8)}     # (input, output), requests of at most 32K input tokens
CHAT_MAX_INPUT_TOKENS = 32000
BATCH_LIMITS = {"qwen3.7-text-embedding-flash": 20, "text-embedding-v4": 10}
RERANK_MAX_DOCUMENTS = 500
MAX_ATTEMPTS = 4
TIMEOUT_SECONDS = 60.0
BUDGET_CEILING_CNY = 10.0
_SECRET = re.compile(r"(sk-[A-Za-z0-9_\-]{6,}|Bearer\s+[A-Za-z0-9_\-\.]+)")


class ApiError(RuntimeError):
    """A failed call. The message holds a status and a code, never the request or the key."""


class BudgetExceeded(ApiError):
    pass


class OfflineCall(ApiError):
    """A paid call was needed while running offline."""


def redact(text: object) -> str:
    return _SECRET.sub("[redacted]", str(text))


def read_env(path: Path = Path(".env")) -> Dict[str, str]:
    """``.env`` values, overridden by the process environment. Nothing here is printed."""
    values: Dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                name, value = line.split("=", 1)
                values[name.strip()] = value.strip().strip('"').strip("'")
    for name in ("DASHSCOPE_API_KEY", "STILLHERE_EMBEDDING_MODEL", "STILLHERE_RERANK_MODEL", "STILLHERE_DASHSCOPE_REGION",
                 "STILLHERE_API_BUDGET_CNY", "STILLHERE_DASHSCOPE_WORKSPACE_ID"):
        if os.environ.get(name):
            values[name] = os.environ[name]
    return values


def base_url(env: Mapping[str, str]) -> str:
    """The workspace host when one is configured, else the shared host, which the base URL page says still works."""
    region = env.get("STILLHERE_DASHSCOPE_REGION", "cn-beijing")
    if region != "cn-beijing":
        raise ApiError(f"region {region} is not configured")
    workspace = env.get("STILLHERE_DASHSCOPE_WORKSPACE_ID")
    return f"https://{workspace}.cn-beijing.maas.aliyuncs.com/api/v1" if workspace else "https://dashscope.aliyuncs.com/api/v1"


def estimate_tokens(chars: int, items: int = 1) -> int:
    """A conservative upper estimate: never less than one token per character, plus a margin per item."""
    return int(chars * 1.1) + 16 * items


class Ledger:
    """Append-only JSONL. ``spent`` is rebuilt from the file on start, so a new run continues the old total."""

    def __init__(self, path: Path, budget_cny: float) -> None:
        self.path = Path(path)
        self.budget = min(float(budget_cny), BUDGET_CEILING_CNY)
        self._lock = threading.Lock()
        self.spent = 0.0
        self.pending = 0.0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.spent += json.loads(line)["cost_cny"]

    def reserve(self, cost: float) -> None:
        with self._lock:
            if self.spent + self.pending + cost > self.budget:
                raise BudgetExceeded(f"budget {self.budget} CNY: spent {self.spent:.4f}, would add {cost:.4f}")
            self.pending += cost

    def settle(self, reserved: float, entry: Mapping) -> None:
        with self._lock:
            self.pending -= reserved
            self.spent += entry["cost_cny"]
            line = json.dumps({**entry, "cumulative_cny": round(self.spent, 6),
                               "at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def summary(self) -> dict:
        requests = tokens = output_tokens = failures = uncertain = 0
        by_model: Dict[str, dict] = {}
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                e = json.loads(line)
                requests += 1
                tokens += e["input_tokens"] or 0
                output_tokens += e.get("output_tokens") or 0
                failures += not e["success"]
                uncertain += bool(e["billing_uncertain"])
                m = by_model.setdefault(e["model_id"], {"requests": 0, "input_tokens": 0, "cost_cny": 0.0})
                m["requests"] += 1
                m["input_tokens"] += e["input_tokens"] or 0
                m["cost_cny"] = round(m["cost_cny"] + e["cost_cny"], 6)
        return {"requests": requests, "input_tokens": tokens, "output_tokens": output_tokens, "failed": failures, "billing_uncertain": uncertain,
                "estimated_cost_cny": round(self.spent, 6), "budget_cny": self.budget,
                "remaining_cny": round(self.budget - self.spent, 6), "by_model": by_model}


_local = threading.local()


def _http(url: str, headers: Mapping[str, str], body: bytes, timeout: float) -> Tuple[int, bytes]:
    """POST over one kept-alive connection per thread: a new TLS handshake costs more than the call itself."""
    parts = urllib.parse.urlsplit(url)
    connection = getattr(_local, "connection", None)
    if connection is None or connection.host != parts.hostname:
        connection = _local.connection = http.client.HTTPSConnection(parts.hostname, timeout=timeout)
    for retry in (True, False):
        try:
            connection.request("POST", parts.path, body=body, headers=dict(headers))
            response = connection.getresponse()
            return response.status, response.read()
        except (http.client.RemoteDisconnected, http.client.CannotSendRequest, http.client.BadStatusLine, ConnectionResetError, BrokenPipeError):
            connection.close()                      # a closed idle connection was not billed: nothing was answered
            if not retry:
                raise
    raise OSError("unreachable")


@dataclass
class EmbedResult:
    vectors: List[List[float]]
    tokens: int
    latency_ms: float


@dataclass
class RerankResult:
    scores: List[Tuple[int, float]]    # (document index, relevance), best first
    tokens: int
    latency_ms: float


@dataclass
class ChatResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str


class DashScopeClient:
    def __init__(self, env: Mapping[str, str], ledger: Ledger, *, transport: Callable = _http, offline: bool = False,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self._key = env.get("DASHSCOPE_API_KEY", "")
        self._base = base_url(env)
        self.ledger = ledger
        self.offline = offline
        self._transport = transport
        self._sleep = sleep
        if not self._key and not offline:
            raise ApiError("DASHSCOPE_API_KEY is not set")

    def _post(self, path: str, payload: Mapping, *, model: str, kind: str, upper_tokens: int, cache_key: str, upper_output: int = 0) -> Tuple[dict, float]:
        if self.offline:
            raise OfflineCall(f"{kind} {model}: not cached and running offline")
        price, output_price = CHAT_PRICES_CNY_PER_MILLION.get(model) or (PRICES_CNY_PER_MILLION[model], 0.0)
        reserved = upper_tokens / 1e6 * price + upper_output / 1e6 * output_price
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        last = "no attempt"
        for attempt in range(MAX_ATTEMPTS):
            self.ledger.reserve(reserved)
            started = time.perf_counter()
            entry = {"model_id": model, "request_type": kind, "request_id": None, "input_tokens": None, "cost_cny": 0.0,
                     "success": False, "billing_uncertain": False, "cache_key": cache_key, "attempt": attempt}
            try:
                status, raw = self._transport(path if path.startswith("https://") else f"{self._base}{path}", headers, body, TIMEOUT_SECONDS)
            except OSError as exc:                 # a timeout or a dropped connection may have been billed
                entry.update(billing_uncertain=True, cost_cny=reserved, error=type(exc).__name__)
                self.ledger.settle(reserved, entry)
                last = type(exc).__name__
                self._sleep(2 ** attempt)
                continue
            latency = (time.perf_counter() - started) * 1000
            if status == 200:
                data = json.loads(raw)
                usage = data.get("usage") or {}
                tokens = usage.get("prompt_tokens") if output_price else usage.get("total_tokens")
                output = usage.get("completion_tokens")
                cost = reserved if tokens is None else tokens / 1e6 * price + (output or 0) / 1e6 * output_price
                if output is not None:
                    entry["output_tokens"] = output
                request_id = str(data.get("request_id") or data.get("id") or "")
                entry.update(request_id=request_id[:8] or None, input_tokens=tokens, cost_cny=cost, success=True,
                             billing_uncertain=tokens is None, latency_ms=round(latency, 1))
                self.ledger.settle(reserved, entry)
                return data, latency
            code = ""
            try:
                code = str(json.loads(raw).get("code") or "")
            except ValueError:
                pass
            entry["error"] = redact(f"HTTP {status} {code}")
            self.ledger.settle(reserved, entry)
            last = entry["error"]
            if status == 429 or status >= 500:
                self._sleep(2 ** attempt)
                continue
            raise ApiError(last)
        raise ApiError(f"gave up after {MAX_ATTEMPTS} attempts: {last}")

    def embed(self, model: str, texts: Sequence[str], *, text_type: str, dimension: int) -> EmbedResult:
        """One synchronous request. This is not the Batch API, and it is billed at the full price."""
        if len(texts) > BATCH_LIMITS[model]:
            raise ValueError(f"{model} takes at most {BATCH_LIMITS[model]} texts in one request")
        payload = {"model": model, "input": {"texts": list(texts)},
                   "parameters": {"dimension": dimension, "text_type": text_type, "output_type": "dense"}}
        upper = estimate_tokens(sum(len(t) for t in texts), len(texts))
        key = hashlib.sha256("\x1f".join(texts).encode("utf-8")).hexdigest()[:16]
        data, latency = self._post("/services/embeddings/text-embedding/text-embedding", payload, model=model,
                                   kind=f"embedding:{text_type}", upper_tokens=upper, cache_key=key)
        items = (data.get("output") or data).get("embeddings") or []
        if len(items) != len(texts):
            raise ApiError(f"{model} returned {len(items)} vectors for {len(texts)} texts")
        ordered = sorted(items, key=lambda e: e.get("text_index", e.get("index", 0)))
        vectors = [e["embedding"] for e in ordered]
        if any(len(v) != dimension for v in vectors):
            raise ApiError(f"{model} returned the wrong dimension")
        return EmbedResult(vectors, int((data.get("usage") or {}).get("total_tokens") or 0), latency)

    def chat(self, model: str, messages: Sequence[Mapping[str, str]], *, max_tokens: int, temperature: float = 0.0, seed: Optional[int] = None) -> ChatResult:
        """A non-thinking completion. The reservation assumes the conservative input estimate and the full ``max_tokens``."""
        upper = estimate_tokens(sum(len(m["content"]) for m in messages), len(messages))
        if upper > CHAT_MAX_INPUT_TOKENS:
            raise ValueError(f"the listed price holds up to {CHAT_MAX_INPUT_TOKENS} input tokens")
        payload = {"model": model, "messages": list(messages), "temperature": temperature, "max_tokens": max_tokens, "enable_thinking": False}
        if seed is not None:
            payload["seed"] = seed
        key = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        url = self._base.replace("/api/v1", "/compatible-mode/v1") + "/chat/completions"
        data, latency = self._post(url, payload, model=model, kind="chat", upper_tokens=upper, cache_key=key, upper_output=max_tokens)
        usage = data.get("usage") or {}
        choices = data.get("choices") or []
        if not choices:
            raise ApiError(f"{model} returned no choices")
        return ChatResult(choices[0]["message"].get("content") or "", int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0), latency, str(data.get("model") or model))

    def rerank(self, model: str, query: str, documents: Sequence[str], *, top_n: Optional[int] = None) -> RerankResult:
        if len(documents) > RERANK_MAX_DOCUMENTS:
            raise ValueError(f"at most {RERANK_MAX_DOCUMENTS} documents")
        payload = {"model": model, "input": {"query": query, "documents": list(documents)},
                   "parameters": {"top_n": top_n or len(documents)}}
        chars = len(query) * len(documents) + sum(len(d) for d in documents)
        key = hashlib.sha256("\x1f".join([query, *documents]).encode("utf-8")).hexdigest()[:16]
        data, latency = self._post("/services/rerank/text-rerank/text-rerank", payload, model=model, kind="rerank",
                                   upper_tokens=estimate_tokens(chars, len(documents)), cache_key=key)
        results = (data.get("output") or data).get("results") or []
        scores = [(int(r["index"]), float(r["relevance_score"])) for r in results]
        if any(not 0 <= i < len(documents) for i, _ in scores) or len({i for i, _ in scores}) != len(scores):
            raise ApiError(f"{model} returned bad document indexes")
        return RerankResult(scores, int((data.get("usage") or {}).get("total_tokens") or 0), latency)
