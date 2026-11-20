"""Style realization: the runtime step after memory has decided WHAT to say (grounded content) - it decides HOW to say it.

Behind ``STYLE_REALIZATION_V1_ENABLED`` (off: the grounded content is answered in the neutral voice, exactly as without this module).
Every backend receives the SAME UniversalStylePromptContract payload (``stillhere.style.contract.build``); none builds its own prompt:

* ``neutral``  - no model: the grounded statements joined in plain text. Also the fallback of every other backend.
* ``prompt``   - a chat model (an API reader-class model) prompted with the contract messages: Architecture B (prompt-only style).
* ``adapter``  - a local base model + the T2-D2 LoRA adapter, prompted with the same contract messages through the tokenizer's chat template
                 (``enable_thinking=False``): Architecture C. It needs a GPU and the restricted adapter weights, which never enter the public
                 core: the caller attaches a local generator (``scripts/style_gpu_smoke.py`` shows one); without it the backend reports
                 ``BLOCKED`` and the call falls back. It is never auto-loaded.

After generation a deterministic fact check runs on the realized text (``grounding_checks``): every key term of the facts must be kept,
no digit run / Latin name the facts and the conversation do not contain may appear, an unknown must be said as unknown. A failed check
-> the neutral fallback (the facts are never traded for style). The trace records ids, hashes, versions, config, check, fallback,
latency and cost - never the texts.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Mapping, Optional, Sequence

from stillhere.style import contract as sc
from stillhere.style import grounding_checks as gc

FLAG = "STYLE_REALIZATION_V1_ENABLED"
BACKENDS = ("neutral", "prompt", "adapter")
GENERATION = {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 160, "enable_thinking": False}


def enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    return (env if env is not None else os.environ).get(FLAG, "").strip().lower() in ("1", "true", "yes", "on")


class BackendBlocked(RuntimeError):
    """The backend cannot run here (no GPU / no adapter weights / no API access)."""


@dataclass
class RealizationTrace:
    flag_on: bool
    backend: str
    persona_id: str
    contract: Optional[dict]                 # ContractPayload.trace(): contract / schema version, prompt hash, example ids, profile ids, ablation, model
    evidence_ids: List[str]
    generation: dict
    model: Optional[str] = None
    adapter: Optional[str] = None
    claim_check: Optional[dict] = None
    fallback: Optional[str] = None
    latency_ms: float = 0.0
    cost_cny: float = 0.0


@dataclass
class Realization:
    text: str
    styled: bool
    trace: RealizationTrace

    def as_dict(self) -> dict:
        return {"text": self.text, "styled": self.styled, "trace": asdict(self.trace)}


def neutral(grounded: Optional[Mapping]) -> str:
    """The plain answer of the grounded content (no style)."""
    if not grounded or grounded.get("status") == "no_memory" or not (grounded.get("facts") or grounded.get("unknown")):
        return "这个我没有记录，不太清楚。"
    parts = [f["statement"] for f in grounded.get("facts", [])]
    parts += [f"{u}我不太清楚" for u in grounded.get("unknown", [])]
    return "；".join(parts) + "。"


def check(text: str, grounded: Optional[Mapping], conversation: Sequence[Mapping], key_terms: Sequence[str] = ()) -> dict:
    """Deterministic fact check of a realized reply against the grounded content."""
    facts = (grounded or {}).get("facts", [])
    # Simple Context v1: the persona's own earlier (generated) replies are never evidence (same rule as safety.evidence_texts)
    sources = [f.get("statement", "") for f in facts] + [f.get("time") or "" for f in facts] + [m.get("text", "") for m in conversation
                                                                                              if m.get("speaker") != "self" and not m.get("generated")]
    problems = []
    missing = [t for t in key_terms if not gc.key_term_present(t, text)]
    if missing:
        problems.append("dropped_key_terms:" + ",".join(missing[:5]))
    extra = gc.unsupported_specifics(text, *sources)
    if extra:
        problems.append("unsupported_specifics:" + ",".join(extra[:5]))
    if (grounded or {}).get("unknown") and not gc.UNCERTAIN.search(text):
        problems.append("unknown_not_said_as_unknown")
    if (not grounded or grounded.get("status") == "no_memory") and not gc.UNCERTAIN.search(text) and gc.NUMBER.search(text):
        problems.append("specifics_without_memory")
    return {"ok": not problems, "problems": problems}


def realize(persona_id: str, conversation: Sequence[Mapping], grounded: Optional[Mapping], *, history: Sequence[sc.StyleMessage] = (),
            backend: str = "neutral", generate: Optional[Callable[[List[dict], dict], dict]] = None, evidence_ids: Sequence[str] = (), key_terms: Sequence[str] = (),
            env: Optional[Mapping[str, str]] = None, model: Optional[str] = None, adapter: Optional[str] = None, generation: Optional[Mapping] = None,
            **contract_kw) -> Realization:
    """``key_terms``: the words of the facts that the reply must keep (names, places, dates; from the memory records).
    ``generate(messages, config) -> {"text", "cost_cny"?}``: the backend call (API client or local model); never called when the flag is off,
    the backend is ``neutral`` or the backend is blocked."""
    cfg = {**GENERATION, **(generation or {})}
    base = neutral(grounded)
    tr = RealizationTrace(enabled(env), backend, persona_id, None, list(evidence_ids), cfg, model, adapter)
    t0 = time.perf_counter()
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}")
    if not tr.flag_on or backend == "neutral":
        tr.fallback = None if backend == "neutral" else "flag_off"
        return Realization(base, False, tr)
    payload = sc.build(persona_id, conversation, grounded_content=grounded, history=history, **contract_kw)   # ContractError propagates: a contract violation is a bug
    tr.contract = payload.trace()
    try:
        if generate is None:
            raise BackendBlocked(f"{backend}: no generator attached" + (" (needs a GPU and the restricted T2-D2 adapter)" if backend == "adapter" else ""))
        out = generate(payload.messages, cfg)
        text, tr.cost_cny = (out.get("text") or "").strip(), float(out.get("cost_cny") or 0.0)
    except BackendBlocked as exc:
        tr.fallback = f"BLOCKED: {exc}"
        tr.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return Realization(base, False, tr)
    except Exception as exc:  # noqa: BLE001 - any backend failure falls back to the facts
        tr.fallback = f"backend_error:{type(exc).__name__}"
        tr.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return Realization(base, False, tr)
    tr.claim_check = check(text, grounded, conversation, key_terms) if text else {"ok": False, "problems": ["empty"]}
    tr.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    if not tr.claim_check["ok"]:
        tr.fallback = "claim_check_failed"
        return Realization(base, False, tr)
    return Realization(text, True, tr)
