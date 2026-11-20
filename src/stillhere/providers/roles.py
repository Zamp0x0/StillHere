"""Per-role provider configuration. Every role is configured on its own; nothing falls back to another role's model silently.

Precedence (later wins): role defaults < ``.env`` < environment < the profile's ``runtime/settings.json`` ``providers`` block < explicit
overrides (CLI / Studio). Variables per role (ROLE in upper case): ``STILLHERE_<ROLE>_PROVIDER``, ``_MODEL``, ``_BASE_URL``,
``_API_KEY_ENV`` (the NAME of the key variable), ``_TEMPERATURE``, ``_MAX_OUTPUT_TOKENS``, ``_TIMEOUT_S``, ``_MAX_ATTEMPTS``,
``_MODEL_DIR`` (huggingface_local).

Data leaving the machine: an API provider for ``answer`` / ``card_assist`` / ``memory_extraction`` / ``router`` sends private text
(retrieved chat evidence, the persona prompt, the question) to that provider. It is refused unless the user opted in with
``STILLHERE_ALLOW_EXTERNAL_API=1`` (or ``--allow-external-api``); the refusal says exactly what would be sent.
"""
from __future__ import annotations

import json
import os
from dataclasses import fields
from pathlib import Path
from typing import Mapping, Optional

from .llm import API_PROVIDERS, DEEPSEEK_DEFAULT_MODEL, DEEPSEEK_KEY_ENV, ApiGenerator, ChatClient, ProviderConfig, ProviderError, UsageLog, read_dotenv

ROLES = ("answer", "card_assist", "memory_extraction", "router", "judge")
DEFAULTS = {
    "answer": {"provider": "huggingface_local", "temperature": 0.0, "max_output_tokens": 1024},
    "card_assist": {"provider": "deepseek_api", "temperature": 0.3, "max_output_tokens": 2048},
    "memory_extraction": {"provider": "huggingface_local", "temperature": 0.0, "max_output_tokens": 1536},
    "router": {"provider": "deepseek_api", "temperature": 0.0, "max_output_tokens": 256},
    "judge": {"provider": "deepseek_api", "temperature": 0.0, "max_output_tokens": 1024},
}
SENDS = {"answer": "the persona prompt, the retrieved chat evidence and the question",
         "card_assist": "the persona card statistics (and example replies only if you also allow examples)",
         "memory_extraction": "whole chat sessions", "router": "the question text only", "judge": "benchmark questions and answers"}
ALLOW_ENV = "STILLHERE_ALLOW_EXTERNAL_API"
FIELDS = {f.name: f.type for f in fields(ProviderConfig)}
KEYS = ("provider", "model", "base_url", "api_key_env", "temperature", "max_output_tokens", "timeout_s", "max_attempts", "model_dir", "device")
REPO = Path(__file__).resolve().parents[3]


def _var(role: str, key: str) -> str:
    return f"STILLHERE_{role.upper()}_{key.upper()}"


def _coerce(key: str, v):
    if v in (None, ""):
        return None
    if key in ("temperature", "timeout_s"):
        return float(v)
    if key in ("max_output_tokens", "max_attempts"):
        return int(v)
    return str(v)


def resolve(role: str, *, env: Optional[Mapping[str, str]] = None, dotenv: Optional[Path] = None, profile_dir: Optional[Path] = None,
            overrides: Optional[Mapping] = None) -> ProviderConfig:
    if role not in ROLES:
        raise ProviderError("BAD_ROLE", f"role must be one of {ROLES}")
    vals = dict(DEFAULTS[role])
    layers = [read_dotenv(dotenv) if dotenv else (read_dotenv(Path.cwd() / ".env") or read_dotenv(REPO / ".env")), os.environ if env is None else env]
    for layer in layers:
        for k in KEYS:
            v = _coerce(k, layer.get(_var(role, k)))
            if v is not None:
                vals[k] = v
    if profile_dir and (Path(profile_dir) / "runtime" / "settings.json").is_file():
        block = (json.loads((Path(profile_dir) / "runtime" / "settings.json").read_text(encoding="utf-8")).get("providers") or {}).get(role) or {}
        for k in KEYS:
            if k in block and block[k] not in (None, ""):
                vals[k] = _coerce(k, block[k])
    for k, v in (overrides or {}).items():
        if k in KEYS and v not in (None, ""):
            vals[k] = _coerce(k, v)
    if vals["provider"] == "deepseek_api":
        vals.setdefault("model", DEEPSEEK_DEFAULT_MODEL)
        vals.setdefault("api_key_env", DEEPSEEK_KEY_ENV)
    elif vals["provider"] in ("huggingface_local", "fake"):
        vals.setdefault("model", "Qwen/Qwen3-8B@b968826d" if vals["provider"] == "huggingface_local" else "fake")
        vals.setdefault("api_key_env", "")
    elif vals["provider"] == "openai_compatible_custom":
        vals.setdefault("api_key_env", "STILLHERE_CUSTOM_API_KEY")
        vals.setdefault("model", "")
    return ProviderConfig(**{k: v for k, v in vals.items() if k in FIELDS}).validate()


def external_allowed(env: Optional[Mapping[str, str]] = None, explicit: Optional[bool] = None) -> bool:
    if explicit is not None:
        return bool(explicit)
    v = (os.environ if env is None else env).get(ALLOW_ENV, "")
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def notice(role: str, cfg: ProviderConfig) -> str:
    return (f"EXTERNAL_API_NOT_ALLOWED: role '{role}' is set to {cfg.provider} ({cfg.model} at {cfg.url.rsplit('/chat', 1)[0]}). It would send "
            f"{SENDS[role]} to that provider. Nothing is sent unless you opt in: set {ALLOW_ENV}=1 (or pass --allow-external-api), or "
            f"choose a local model (STILLHERE_{role.upper()}_PROVIDER=huggingface_local).")


def api_generator(role: str, cfg: ProviderConfig, *, allow_external: Optional[bool] = None, env: Optional[Mapping[str, str]] = None,
                  usage_log: Optional[Path] = None, transport=None) -> ApiGenerator:
    if cfg.provider not in API_PROVIDERS:
        raise ProviderError("NOT_AN_API_PROVIDER", f"{role} uses {cfg.provider}")
    if not external_allowed(env, allow_external):
        raise ProviderError("EXTERNAL_API_NOT_ALLOWED", notice(role, cfg))
    kw = {"transport": transport} if transport else {}
    source = os.environ if env is None else env
    dot = read_dotenv(Path.cwd() / ".env") or read_dotenv(REPO / ".env")
    merged = {**dot, **dict(source)}
    return ApiGenerator(ChatClient(cfg, env=merged, usage_log=UsageLog(usage_log), role=role, **kw))
