"""One configuration for every persona entry point (CLI, Studio, pipeline, Writer, Reader, runtime). Nothing machine-specific is hard-coded.

Precedence (later wins): defaults < ``.env`` file (current directory, else the repository root) < environment variables < the profile's
``runtime/settings.json`` < explicit CLI / Studio arguments. Secrets are never printed (``public()`` masks them).

Local models: Writer, Reader and chat use the user's LOCAL model (``STILLHERE_BASE_MODEL_DIR``; optional separate writer / reader model
dirs; when they are the same directory one loaded instance serves all three). The optional prompt API is used ONLY to write persona
prompts (``persona prompt --provider openai_compatible``) - never for memory, reading or chat.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Mapping, Optional

REPO = Path(__file__).resolve().parents[3]
BACKENDS = ("auto", "hf-lowvram", "hf", "fake")
PROVIDERS = ("deterministic", "openai_compatible")
ENV = {"base_model_dir": "STILLHERE_BASE_MODEL_DIR", "backend": "STILLHERE_LOCAL_BACKEND", "device": "STILLHERE_DEVICE",
       "writer_model_dir": "STILLHERE_WRITER_MODEL_DIR", "reader_model_dir": "STILLHERE_READER_MODEL_DIR",
       "writer_max_new_tokens": "STILLHERE_WRITER_MAX_NEW_TOKENS", "max_context_tokens": "STILLHERE_MAX_CONTEXT_TOKENS",
       "prompt_provider": "STILLHERE_PROMPT_PROVIDER", "prompt_api_base_url": "STILLHERE_PROMPT_API_BASE_URL", "prompt_api_key": "STILLHERE_PROMPT_API_KEY",
       "prompt_api_model": "STILLHERE_PROMPT_API_MODEL", "prompt_send_examples": "STILLHERE_PROMPT_SEND_EXAMPLES"}
SECRET = {"prompt_api_key"}


@dataclass
class Config:
    base_model_dir: Optional[str] = None          # a local Qwen3-8B (or compatible chat model) directory
    backend: str = "auto"                         # auto: hf-lowvram below 20 GiB VRAM, hf above, error without a GPU
    device: str = "cuda:0"
    writer_model_dir: Optional[str] = None        # default: base_model_dir (shared instance)
    reader_model_dir: Optional[str] = None        # default: base_model_dir (shared instance)
    writer_max_new_tokens: int = 1536
    max_context_tokens: int = 8192
    prompt_provider: str = "deterministic"        # default: no API key, no network
    prompt_api_base_url: Optional[str] = None
    prompt_api_key: Optional[str] = None
    prompt_api_model: Optional[str] = None
    prompt_send_examples: bool = False            # private chat examples never leave the machine unless the user sets this

    def validate(self) -> "Config":
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        if self.prompt_provider not in PROVIDERS:
            raise ValueError(f"prompt provider must be one of {PROVIDERS}")
        if not (self.device == "cpu" or self.device.startswith("cuda")):
            raise ValueError("device must be cpu or cuda[:N]")
        return self

    @property
    def device_index(self) -> int:
        return int(self.device.split(":")[1]) if self.device.startswith("cuda:") else 0

    def writer_dir(self) -> Optional[str]:
        return self.writer_model_dir or self.base_model_dir

    def reader_dir(self) -> Optional[str]:
        return self.reader_model_dir or self.base_model_dir

    def public(self) -> dict:
        d = asdict(self)
        for k in SECRET:
            d[k] = "***set***" if d.get(k) else None
        return d


def _coerce(name: str, value):
    t = {f.name: f.type for f in fields(Config)}[name]
    if value is None or value == "":
        return None
    if "bool" in str(t):
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    if "int" in str(t):
        return int(value)
    return str(value)


def read_dotenv(path: Path) -> dict:
    out = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def load(*, profile_dir: Optional[Path] = None, overrides: Optional[Mapping] = None, env: Optional[Mapping[str, str]] = None, dotenv: Optional[Path] = None) -> Config:
    vals = {}
    dot = read_dotenv(dotenv) if dotenv else (read_dotenv(Path.cwd() / ".env") or read_dotenv(REPO / ".env"))
    environ = os.environ if env is None else env
    for layer in (dot, environ):
        for name, var in ENV.items():
            if layer.get(var) not in (None, ""):
                vals[name] = _coerce(name, layer[var])
    if profile_dir and (Path(profile_dir) / "runtime" / "settings.json").is_file():
        for k, v in json.loads((Path(profile_dir) / "runtime" / "settings.json").read_text(encoding="utf-8")).items():
            if k in ENV and k not in SECRET and v not in (None, ""):
                vals[k] = _coerce(k, v)
    for k, v in (overrides or {}).items():
        if k in ENV and v not in (None, ""):
            vals[k] = _coerce(k, v)
    return Config(**vals).validate()


def resolve_backend(cfg: Config) -> str:
    """'auto' -> hf-lowvram (< 20 GiB) / hf (>= 20 GiB) after a real CUDA probe; a clear error without a GPU (never a silent fake)."""
    if cfg.backend != "auto":
        return cfg.backend
    from . import doctor as DR
    g = DR.gpu_check(nf4=False)
    if not g.get("cuda_available"):
        raise RuntimeError("NO_GPU: no CUDA GPU was found. Prompt, dataset and evidence import work on CPU; chat / Writer / Reader need a local GPU "
                           "(or --backend fake for a labelled dry run). Check with: stillhere persona doctor --target local-cuda")
    return "hf-lowvram" if (g.get("vram_total_gib") or 0) < 20 else "hf"
