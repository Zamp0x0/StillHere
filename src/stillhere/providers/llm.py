"""One OpenAI-compatible chat client for DeepSeek and custom endpoints, and the generator adapter the persona runtime uses.

* ``ProviderConfig``: provider kind, model, base URL, the NAME of the environment variable holding the key (never the key itself),
  temperature, max output tokens, timeout, bounded retries.
* ``ChatClient.chat``: one request, bounded retries (429 / 5xx / timeouts only, exponential backoff, at most ``max_attempts``), the
  returned model id / fingerprint / finish reason / token usage / latency recorded; an optional JSONL usage log without any text or key.
  Cost is reported only when a price for the RETURNED model is known (``PRICES``); otherwise ``cost`` is ``None`` (never guessed).
* ``ApiGenerator``: the ``persona.models`` generator contract (``__call__(messages, cfg, adapter)`` and ``complete``), so the Fact Reader,
  its grounding / attribution gates and the answer-preservation check run unchanged on an API model. A persona adapter can never be
  used with an API model (the adapter belongs to the local Qwen3-8B base): such a call is refused, not ignored.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

PROVIDERS = ("deepseek_api", "openai_compatible_custom", "huggingface_local", "fake")
API_PROVIDERS = ("deepseek_api", "openai_compatible_custom")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-flash"           # api-docs.deepseek.com (read 2026-11-19): the current Flash id; deepseek-v4-flash is a legacy alias
DEEPSEEK_KEY_ENV = "DEEPSEEK_API_KEY"
# USD per 1M tokens, from the provider's pricing page. DeepSeek bills by the REQUESTED model id (a legacy alias is billed at the Flash
# price); a custom endpoint is looked up by the returned id. A model not listed here gets cost None (never guessed).
PRICES_USD_PER_MILLION: Dict[str, Dict[str, float]] = {
    # api-docs.deepseek.com/quick_start/pricing, read 2026-11-19: PEAK rates (off-peak is half). Reported costs are therefore upper bounds.
    "deepseek-flash": {"input_miss": 0.30, "input_hit": 0.006, "output": 1.20},
}
PRICE_SOURCE = "api-docs.deepseek.com/quick_start/pricing (read 2026-11-19), peak rates; off-peak is half"
_SECRET = re.compile(r"(sk-[A-Za-z0-9_\-]{6,}|Bearer\s+[A-Za-z0-9_\-\.]+)")


class ProviderError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {redact(message)}")
        self.code = code


def redact(text: object) -> str:
    return _SECRET.sub("[redacted]", str(text))


def read_dotenv(path: Optional[Path]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if path and Path(path).is_file():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


@dataclass
class ProviderConfig:
    provider: str = "deepseek_api"
    model: str = DEEPSEEK_DEFAULT_MODEL
    base_url: Optional[str] = None                  # default: the provider's official URL (deepseek_api); required for custom
    api_key_env: str = DEEPSEEK_KEY_ENV              # the NAME of the variable; the value is never stored here
    temperature: float = 0.0
    max_output_tokens: int = 1024
    timeout_s: float = 60.0
    max_attempts: int = 3
    disable_thinking: bool = True                    # DeepSeek: thinking mode is turned off explicitly
    model_dir: Optional[str] = None                  # huggingface_local: a local model directory
    device: str = "cuda:0"                           # huggingface_local
    extra: Dict[str, object] = field(default_factory=dict)

    def validate(self) -> "ProviderConfig":
        if self.provider not in PROVIDERS:
            raise ProviderError("BAD_PROVIDER", f"provider must be one of {PROVIDERS}, not {self.provider!r}")
        if self.provider == "openai_compatible_custom" and not self.base_url:
            raise ProviderError("NO_BASE_URL", "openai_compatible_custom needs base_url (the endpoint you chose)")
        if self.provider in API_PROVIDERS and not self.model:
            raise ProviderError("NO_MODEL", "an API provider needs a model id")
        if not (0 <= float(self.temperature) <= 2):
            raise ProviderError("BAD_TEMPERATURE", "temperature must be within [0, 2]")
        if int(self.max_output_tokens) < 1 or int(self.max_attempts) < 1 or float(self.timeout_s) <= 0:
            raise ProviderError("BAD_LIMITS", "max_output_tokens, max_attempts and timeout_s must be positive")
        if int(self.max_attempts) > 6:
            raise ProviderError("BAD_LIMITS", "at most 6 attempts per request (bounded retry)")
        return self

    @property
    def url(self) -> str:
        base = self.base_url or (DEEPSEEK_BASE_URL if self.provider == "deepseek_api" else "")
        return base.rstrip("/") + "/chat/completions"

    def public(self) -> dict:
        d = asdict(self)
        d["key_set"] = bool(os.environ.get(self.api_key_env)) if self.provider in API_PROVIDERS else None
        return d


@dataclass
class ChatResponse:
    text: str
    finish_reason: Optional[str]
    requested_model: str
    returned_model: Optional[str]
    fingerprint: Optional[str]
    request_id: Optional[str]
    input_tokens: Optional[int]
    cache_hit_tokens: Optional[int]
    output_tokens: Optional[int]
    latency_ms: float
    attempts: int
    cost_usd: Optional[float]
    called_at: str

    def usage(self) -> dict:
        d = asdict(self)
        d.pop("text")
        return d


def cost_usd(model: Optional[str], input_tokens, hit_tokens, output_tokens) -> Optional[float]:
    p = PRICES_USD_PER_MILLION.get(model or "")
    if not p or input_tokens is None or output_tokens is None:
        return None
    hit = hit_tokens or 0
    return round((input_tokens - hit) / 1e6 * p["input_miss"] + hit / 1e6 * p["input_hit"] + output_tokens / 1e6 * p["output"], 8)


def http_post(url: str, headers: Mapping[str, str], body: bytes, timeout: float) -> Tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read() or b""


class UsageLog:
    """Append-only JSONL of request metadata (model ids, tokens, latency, cost, status). Never text, never a key."""

    def __init__(self, path: Optional[Path]):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()

    def write(self, row: Mapping) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


class ChatClient:
    def __init__(self, cfg: ProviderConfig, *, env: Optional[Mapping[str, str]] = None, transport: Callable = http_post,
                 sleep: Callable[[float], None] = time.sleep, usage_log: Optional[UsageLog] = None, role: str = "answer"):
        self.cfg = cfg.validate()
        if cfg.provider not in API_PROVIDERS:
            raise ProviderError("NOT_AN_API_PROVIDER", f"{cfg.provider} is not an API provider")
        source = os.environ if env is None else env
        self._key = source.get(cfg.api_key_env, "")
        if not self._key:
            raise ProviderError("NO_API_KEY", f"set {cfg.api_key_env} in the environment or .env (the key is never stored in a config file)")
        self._transport, self._sleep, self.log, self.role = transport, sleep, usage_log or UsageLog(None), role

    def payload(self, messages: Sequence[Mapping[str, str]], *, temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                top_p: Optional[float] = None, json_mode: bool = False) -> dict:
        body = {"model": self.cfg.model, "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
                "temperature": float(self.cfg.temperature if temperature is None else temperature),
                "max_tokens": int(max_tokens or self.cfg.max_output_tokens), "stream": False}
        if top_p is not None:
            body["top_p"] = float(top_p)
        if self.cfg.provider == "deepseek_api" and self.cfg.disable_thinking:
            body["thinking"] = {"type": "disabled"}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    def chat(self, messages: Sequence[Mapping[str, str]], **kw) -> ChatResponse:
        body = json.dumps(self.payload(messages, **kw), ensure_ascii=False).encode("utf-8")
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        last = "no attempt"
        for attempt in range(1, int(self.cfg.max_attempts) + 1):
            t0 = time.perf_counter()
            at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            row = {"role": self.role, "provider": self.cfg.provider, "requested_model": self.cfg.model, "attempt": attempt, "called_at": at}
            try:
                status, raw = self._transport(self.cfg.url, headers, body, float(self.cfg.timeout_s))
            except Exception as e:                         # timeout / connection reset: may have been billed; retried with backoff
                last = type(e).__name__
                self.log.write({**row, "status": "TRANSPORT_ERROR", "error": last})
                if attempt < self.cfg.max_attempts:
                    self._sleep(min(30.0, 2.0 ** attempt))
                continue
            latency = round((time.perf_counter() - t0) * 1000, 1)
            if status == 200:
                try:
                    data = json.loads(raw)
                except ValueError:
                    last = "BAD_JSON"
                    self.log.write({**row, "status": last, "latency_ms": latency})
                    continue
                usage = data.get("usage") or {}
                choices = data.get("choices") or []
                if not choices:
                    last = "NO_CHOICES"
                    self.log.write({**row, "status": last, "latency_ms": latency})
                    continue
                msg = choices[0].get("message") or {}
                inp, out, hit = usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("prompt_cache_hit_tokens")
                model = data.get("model")
                res = ChatResponse(text=msg.get("content") or "", finish_reason=choices[0].get("finish_reason"), requested_model=self.cfg.model,
                                   returned_model=model, fingerprint=data.get("system_fingerprint"), request_id=str(data.get("id") or "")[:24] or None,
                                   input_tokens=inp, cache_hit_tokens=hit, output_tokens=out, latency_ms=latency, attempts=attempt,
                                   cost_usd=cost_usd(self.cfg.model if self.cfg.provider == "deepseek_api" else model, inp, hit, out), called_at=at)
                self.log.write({**row, "status": "OK", **res.usage()})
                return res
            code = ""
            try:
                code = str((json.loads(raw).get("error") or {}).get("code") or (json.loads(raw).get("error") or {}).get("type") or "")
            except (ValueError, AttributeError):
                pass
            last = f"HTTP {status} {code}".strip()
            self.log.write({**row, "status": "HTTP_ERROR", "http_status": status, "error": redact(last), "latency_ms": latency})
            if status == 429 or status >= 500:
                if attempt < self.cfg.max_attempts:
                    self._sleep(min(60.0, 2.0 ** (attempt + 1)))
                continue
            raise ProviderError("HTTP_ERROR", last)            # 4xx (bad key, bad model, bad request): never retried
        raise ProviderError("GAVE_UP", f"{self.cfg.max_attempts} attempts: {last}")


def check_connection(client: ChatClient) -> dict:
    """``persona provider check``: one tiny request (a few tokens). Returns ids and usage, never the key."""
    r = client.chat([{"role": "user", "content": "回复 OK 两个字母。"}], max_tokens=8, temperature=0.0)
    return {"ok": bool(r.text), "requested_model": r.requested_model, "returned_model": r.returned_model, "fingerprint": r.fingerprint,
            "input_tokens": r.input_tokens, "output_tokens": r.output_tokens, "latency_ms": r.latency_ms, "called_at": r.called_at}


class ApiGenerator:
    """The persona generator contract on an API model. Sampling settings come from the caller's decode config (``do_sample`` false ->
    temperature 0); ``seed`` / ``top_k`` / ``min_p`` have no API equivalent and are recorded as not applied."""

    def __init__(self, client: ChatClient):
        self.client = client
        self.label = f"API:{client.cfg.provider}:{client.cfg.model}"
        self.calls: list = []

    def _resp(self, r: ChatResponse) -> dict:
        self.calls.append(r.usage())
        return {"text": (r.text or "").strip(), "hit_token_limit": r.finish_reason == "length", "new_tokens": r.output_tokens, "prompt_tokens": r.input_tokens,
                "latency_s": round(r.latency_ms / 1000, 3), "peak_vram_gib": None, "returned_model": r.returned_model, "cost_usd": r.cost_usd,
                "provider": self.client.cfg.provider}

    def __call__(self, messages, cfg, adapter: Optional[str]) -> dict:
        if adapter is not None:
            raise ProviderError("ADAPTER_NEEDS_LOCAL_BASE", "a persona adapter only runs on its local base model; turn the adapter off to use an API model")
        sample = bool(cfg.get("do_sample", False))
        temp = float(cfg.get("temperature", 0.0)) if sample else 0.0
        top_p = float(cfg["top_p"]) if sample and cfg.get("top_p") is not None else None
        return self._resp(self.client.chat(messages, temperature=temp, top_p=top_p, max_tokens=int(cfg.get("max_new_tokens", self.client.cfg.max_output_tokens))))

    def complete(self, messages, max_new_tokens: int = 1024) -> dict:
        r = self.client.chat(messages, temperature=0.0, max_tokens=max_new_tokens)
        out = self._resp(r)
        m = re.search(r"\{.*\}", out["text"], re.S)
        return {"text": m.group(0) if m else out["text"], "model": r.returned_model or r.requested_model, "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens, "latency_ms": r.latency_ms, "hit_token_limit": out["hit_token_limit"]}

    def is_oom(self, e) -> bool:
        return False

    def after_error(self):
        pass
