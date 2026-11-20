"""DeepSeek chat calls for the reader experiment, with the same ledger discipline as the DashScope client. Standard library only.

The key is read from the environment or ``.env`` and is never logged: errors pass through ``redact``. Every attempt is written to the ledger before the next
one starts, and a request is refused when its conservative estimate could pass the budget. Thinking mode is turned off explicitly, because it is on by default.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

from .alibaba_client import ApiError, BudgetExceeded, OfflineCall, _http, estimate_tokens, redact

BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-flash"
USD_CNY = 7.4                               # a conservative rate; the ledger never under-counts
PRICES_USD_PER_MILLION = {"deepseek-flash": {"input_miss": 0.30, "input_hit": 0.006, "output": 1.20}}   # peak rates, the higher of the two
MAX_ATTEMPTS = 3
TIMEOUT_SECONDS = 60.0


def price_cny(model: str, kind: str) -> float:
    return PRICES_USD_PER_MILLION[model][kind] * USD_CNY


@dataclass
class DeepSeekResult:
    text: str
    input_tokens: int
    cache_hit_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    fingerprint: Optional[str]
    finish_reason: Optional[str]
    cost_cny: float


class DeepSeekClient:
    def __init__(self, env: Mapping[str, str], ledger, *, transport: Callable = _http, offline: bool = False, sleep: Callable[[float], None] = time.sleep) -> None:
        self._key = env.get("DEEPSEEK_API_KEY", "")
        self.ledger = ledger
        self.offline = offline
        self._transport = transport
        self._sleep = sleep
        if not self._key and not offline:
            raise ApiError("DEEPSEEK_API_KEY is not set")

    @staticmethod
    def payload(model: str, messages: Sequence[Mapping[str, str]], *, max_tokens: int, temperature: float, json_mode: bool = False) -> dict:
        body = {"model": model, "messages": list(messages), "max_tokens": max_tokens, "temperature": temperature, "thinking": {"type": "disabled"}, "stream": False}
        if json_mode:                     # JSON object mode; the server does not enforce a schema, the caller validates
            body["response_format"] = {"type": "json_object"}
        return body

    def chat(self, model: str, messages: Sequence[Mapping[str, str]], *, max_tokens: int, temperature: float = 0.0, json_mode: bool = False) -> DeepSeekResult:
        if self.offline:
            raise OfflineCall(f"chat {model}: not cached and running offline")
        if model not in PRICES_USD_PER_MILLION:
            raise ApiError(f"no price recorded for {model}")
        upper_in = estimate_tokens(sum(len(m["content"]) for m in messages), len(messages))
        reserved = upper_in / 1e6 * price_cny(model, "input_miss") + max_tokens / 1e6 * price_cny(model, "output")
        payload = self.payload(model, messages, max_tokens=max_tokens, temperature=temperature, json_mode=json_mode)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        key = hashlib.sha256(body).hexdigest()[:16]
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        last = "no attempt"
        for attempt in range(MAX_ATTEMPTS):
            self.ledger.reserve(reserved)
            entry = {"provider": "deepseek", "model_id": model, "request_type": "chat", "request_id": None, "input_tokens": None, "cost_cny": 0.0, "success": False, "billing_uncertain": False, "cache_key": key, "attempt": attempt}
            started = time.perf_counter()
            try:
                status, raw = self._transport(f"{BASE_URL}/chat/completions", headers, body, TIMEOUT_SECONDS)
            except Exception as exc:                 # a timeout or a dropped connection may have been billed
                entry.update(billing_uncertain=True, cost_cny=reserved, error=type(exc).__name__)
                self.ledger.settle(reserved, entry)
                last = type(exc).__name__
                self._sleep(2 ** attempt)
                continue
            latency = (time.perf_counter() - started) * 1000
            if status == 200:
                data = json.loads(raw)
                usage = data.get("usage") or {}
                prompt = usage.get("prompt_tokens")
                hit = usage.get("prompt_cache_hit_tokens") or 0
                output = usage.get("completion_tokens")
                if prompt is None or output is None:
                    cost, uncertain = reserved, True
                else:
                    cost, uncertain = (prompt - hit) / 1e6 * price_cny(model, "input_miss") + hit / 1e6 * price_cny(model, "input_hit") + output / 1e6 * price_cny(model, "output"), False
                choices = data.get("choices") or []
                entry.update(request_id=str(data.get("id") or "")[:8] or None, input_tokens=prompt, output_tokens=output, cache_hit_tokens=hit, cost_cny=cost, success=bool(choices), billing_uncertain=uncertain, latency_ms=round(latency, 1), resolved_model=data.get("model"), fingerprint=data.get("system_fingerprint"))
                self.ledger.settle(reserved, entry)
                if not choices:
                    raise ApiError(f"{model} returned no choices")
                message = choices[0].get("message") or {}
                return DeepSeekResult(message.get("content") or "", int(prompt or 0), int(hit), int(output or 0), latency, str(data.get("model") or model), data.get("system_fingerprint"), choices[0].get("finish_reason"), cost)
            code = ""
            try:
                code = str((json.loads(raw).get("error") or {}).get("code") or "")
            except (ValueError, AttributeError):
                pass
            entry["error"] = redact(f"HTTP {status} {code}")
            self.ledger.settle(reserved, entry)
            last = entry["error"]
            if status == 429 or status >= 500:
                self._sleep(2 ** attempt)
                continue
            raise ApiError(last)
        raise ApiError(f"gave up after {MAX_ATTEMPTS} attempts: {last}")
