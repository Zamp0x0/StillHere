"""StillHere public Python API: one reply chain for every backend.

    from stillhere import api
    rt = api.demo_runtime(context="FULL_CONTEXT")              # public synthetic persona, in-memory SQLite
    sid = rt.create_session(api.DEMO_PERSONA)
    r = rt.reply_session(api.DEMO_PERSONA, sid, "我下周去哪出差？", "req-1", generate=api.FakeGenerator(), backend="fake", env=api.GUARDS_ON)

The chain (the same for a fake generator, the Qwen3-8B base model and the D4 adapter):
user message -> session store -> L1 window / L2 session history / L3 temporal memory -> grounding + disclosure policy -> renderer
(prompt/2.0 or the pinned D4 renderer) -> generate() -> Loop Guard (optional) -> output guards (grounding, must-not, repetition, length)
-> protected-value check -> stored reply.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Optional

from stillhere.runtime import loop_guard
from stillhere.session.config import D4_COMPATIBLE, DEFAULT, ContextConfig
from stillhere.session.runtime import SessionReply, SessionRuntime

FULL_CONTEXT = DEFAULT
CONTEXTS: Dict[str, ContextConfig] = {"FULL_CONTEXT": FULL_CONTEXT, "D4_COMPATIBLE": D4_COMPATIBLE}
GUARDS_ON = {"GUARD_ENABLED": "1", "GROUNDING_V2_ENABLED": "1"}
Generate = Callable[[List[dict], dict], dict]


def _demo_dataset():
    from stillhere.evaluation import context_dataset_v1 as cd
    return cd


DEMO_PERSONA = "syn.ca"            # = stillhere.evaluation.context_dataset_v1.CA (a public synthetic persona)


def demo_memory():
    """The public synthetic persona's memory (built by the deterministic reference writer from synthetic chats; no model)."""
    from stillhere.evaluation import writer_eval as we
    cd = _demo_dataset()
    return we.run(we.new_world(we.reference_complete(cd.conversations()), "reference", ds=cd, m22=True, m221=True, m222=True), ds=cd).api


def demo_runtime(context: str = "FULL_CONTEXT", *, loop_guard_enabled: Optional[bool] = None, clock=None, api=None) -> SessionRuntime:
    """A SessionRuntime on the synthetic demo memory. ``context``: FULL_CONTEXT (prompt/2.0 renderer, session history search) or
    D4_COMPATIBLE (the pinned D4 renderer, 4 recent turns, 1,536-token budget: the distribution the D4 adapter was trained on)."""
    if context not in CONTEXTS:
        raise ValueError(f"context must be one of {sorted(CONTEXTS)}")
    from datetime import timedelta
    cd = _demo_dataset()
    if clock is None:
        state = {"now": cd.t(9, 10, 20)}

        def clock():
            state["now"] = state["now"] + timedelta(minutes=1)
            return state["now"]
    return SessionRuntime(api or demo_memory(), config=CONTEXTS[context], clock=clock, render_mode="d4" if context == "D4_COMPATIBLE" else "prompt2",
                          loop_guard=loop_guard_enabled)


class FakeGenerator:
    """A deterministic stand-in for a model (no weights, no network): answers from the facts the renderer put into the prompt.
    ``script`` replaces that with fixed texts (one per call), e.g. to show the Loop Guard on a looping reply."""

    def __init__(self, script: Optional[List[str]] = None) -> None:
        self.script, self.calls = list(script or []), 0

    def __call__(self, messages: List[dict], config: Mapping) -> dict:
        self.calls += 1
        if self.script:
            return {"text": self.script[min(self.calls, len(self.script)) - 1], "hit_token_limit": False}
        system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
        head = "【记忆事实】" if "【记忆事实】" in system else "【记忆证据】" if "【记忆证据】" in system else None
        if head is None:                                           # small talk: no memory section in the prompt
            return {"text": "你好呀，在的。", "hit_token_limit": False}
        block = system.split(head, 1)[1].split("【生成要求】", 1)[0]
        facts = []
        for ln in block.splitlines():
            if ln.startswith("- ["):                               # prompt/2.0: "- [当前] 本人..."
                facts.append(ln.split("] ", 1)[-1].strip())
            elif ln.startswith("    ") and ln.strip() and "（不显示）" not in ln:    # D4 renderer: the evidence text under "[F1] ..."
                facts.append(ln.strip())
        asked = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "").splitlines()[-1:] or [""]
        if not facts:
            return {"text": "这个我还真不记得了。" if asked[0].rstrip().endswith(("？", "?")) else "你好呀，在的。", "hit_token_limit": False}
        best = max(facts, key=lambda f: len(set(f) & set(asked[0])))       # the fact that shares the most characters with the question
        best = best.split("（", 1)[0]                                       # drop the renderer's bookkeeping note, e.g. "（9月6日记录为已取消）"
        text = best.replace("本人的", "我的").replace("本人", "我") + "。"
        return {"text": text, "hit_token_limit": False}


D4_ADAPTER = {"checkpoint": 546, "adapter_model_sha256": "78f82d49adaa3e4bf5b12a741bffe0541dff9a7235973e80b5b93f062045e0c9",
              "adapter_config_sha256": "3784ae043d5905608c0f4dcd300d211ca7644da7fe9443c9b6ae96c7f9863905",
              "zip_sha256": "f5dfd72d084d79cf6a5bb6545685a14105046a6f749480236c1a60a4f8733416",
              "base_model": "Qwen/Qwen3-8B", "base_revision": "b968826d9c46dd6066d109eabc6255188de91218"}
D4_ADAPTER_ENV = "STILLHERE_D4_ADAPTER_DIR"


def find_d4_adapter(root: Optional[str] = None, *, verify: bool = True):
    """Locate the D4 checkpoint-546 adapter directory (the one holding ``adapter_model.safetensors``): ``$STILLHERE_D4_ADAPTER_DIR``, else
    ``<root>/private/adapters/d4/extracted/best_adapter`` (root: the repository). With ``verify`` the weights, config, step and base revision
    are checked against the pinned values. Returns the directory (``pathlib.Path``) or raises ``FileNotFoundError`` / ``ValueError``."""
    import hashlib
    import json as _json
    import os
    from pathlib import Path
    cand = os.environ.get(D4_ADAPTER_ENV)
    d = Path(cand) if cand else Path(root or Path(__file__).resolve().parents[2]) / "private" / "adapters" / "d4" / "extracted" / "best_adapter"
    if not (d / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"no D4 adapter at {d} (set {D4_ADAPTER_ENV} or extract t2d4_best_adapter.zip there)")
    if verify:
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        if sha(d / "adapter_model.safetensors") != D4_ADAPTER["adapter_model_sha256"] or sha(d / "adapter_config.json") != D4_ADAPTER["adapter_config_sha256"]:
            raise ValueError("D4 adapter files do not match the pinned checkpoint-546 hashes")
        meta = _json.loads((d / "selection_meta.json").read_text(encoding="utf-8")) if (d / "selection_meta.json").is_file() else {}
        base = _json.loads((d / "BASE_REVISION.json").read_text(encoding="utf-8")) if (d / "BASE_REVISION.json").is_file() else {}
        if meta.get("step") != D4_ADAPTER["checkpoint"] or base.get("revision") != D4_ADAPTER["base_revision"] or base.get("base_model") != D4_ADAPTER["base_model"]:
            raise ValueError("D4 adapter metadata is not checkpoint-546 on Qwen/Qwen3-8B@b968826d…")
    return d


def hf_generator(model_dir: str, adapter_dir: Optional[str] = None, *, device_map: str = "auto") -> Generate:  # pragma: no cover - GPU only
    """OPTIONAL real backend: Qwen3-8B (base) or Qwen3-8B + the D4 LoRA adapter, NF4 4-bit, via transformers / peft / bitsandbytes.
    Not installed by default and never run in the CPU test suite (Windows CPU-only policy). Base and adapter share this function, so
    they go through the identical reply chain."""
    import torch  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model_dir)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, quantization_config=q, device_map=device_map)
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    def generate(messages: List[dict], config: Mapping) -> dict:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, return_tensors="pt").to(model.device)
        mx = int(config.get("max_new_tokens", 160))
        torch.manual_seed(int(config.get("seed", 0)))
        out = model.generate(ids, max_new_tokens=mx, do_sample=float(config.get("temperature", 0)) > 0,
                             temperature=float(config.get("temperature", 1.0)) or None, top_p=float(config.get("top_p", 1.0)),
                             repetition_penalty=float(config.get("repetition_penalty", 1.0)))
        new = out[0][ids.shape[1]:]
        return {"text": tok.decode(new, skip_special_tokens=True).strip(), "hit_token_limit": len(new) >= mx}
    return generate


__all__ = ["CONTEXTS", "D4_COMPATIBLE", "DEMO_PERSONA", "FULL_CONTEXT", "FakeGenerator", "GUARDS_ON", "SessionReply", "SessionRuntime",
           "D4_ADAPTER", "demo_memory", "demo_runtime", "find_d4_adapter", "hf_generator", "loop_guard"]
