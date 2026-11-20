"""Local model loading shared by chat (``runtime``), the Memory Writer (``memory_store``) and the Fact Reader: one loaded base model,
adapters by name, and a greedy ``complete`` for the Writer / Reader. Kept apart from ``runtime`` so ``memory_store`` can load the
user's local model without importing the chat runtime (no import cycle)."""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import evaluation as EV

REPO = Path(__file__).resolve().parents[3]


class LowVramGen:  # pragma: no cover - CUDA only
    """``hf-lowvram``: the same generation contract as ``d5pilot.evaluate.Gen`` (Qwen3 chat template with thinking off, NF4 double-quant
    bf16 compute, stop ids <|im_end|> / <|endoftext|>, per-prompt seed ``t2d4 batch_seed``, adapters attached by name, base = every
    adapter disabled) for an 8 GB laptop GPU: only the input embedding table (1.2 GB bf16, a lookup) stays on the CPU. ONE base model;
    ``complete`` serves the Memory Writer from the same weights (greedy), so no second copy is ever loaded."""
    label = "QWEN3_8B_NF4_LOCAL_LOWVRAM"

    def __init__(self, base_dir: Path, adapters: dict, device_index: int = 0):
        import hashlib as _h
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from . import doctor as DR
        if not torch.cuda.is_available():
            raise RuntimeError("MODEL_LOAD_FAILURE: CUDA is not available")
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(str(base_dir))
        if _h.sha256((self.tok.chat_template or "").encode()).hexdigest() != DR.CHAT_TEMPLATE_SHA256:
            raise RuntimeError("CONFIG_MISMATCH: chat template sha differs from the pinned Qwen3-8B template")
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
                               llm_int8_enable_fp32_cpu_offload=True)
        self.dev = device_index
        dmap = {"model.embed_tokens": "cpu", "model.layers": self.dev, "model.norm": self.dev, "model.rotary_emb": self.dev, "lm_head": self.dev}
        torch.cuda.reset_peak_memory_stats()
        self.base = AutoModelForCausalLM.from_pretrained(str(base_dir), quantization_config=q, device_map=dmap, dtype=torch.bfloat16)
        self.base.eval()
        self.model = self.base
        self.info = {}
        for name, d in adapters.items():
            self.add_adapter(name, Path(d))
        self.load_peak_gib = round(torch.cuda.max_memory_allocated() / 2**30, 3)

    def add_adapter(self, name: str, d: Path) -> dict:
        import hashlib as _h
        from peft import PeftModel
        # PEFT re-dispatches a model whose device map contains "cpu" (and refuses: "offload the whole model to the disk"). The LoRA targets
        # are the GPU decoder layers only; the CPU input embedding keeps its accelerate hook. So the map is hidden while attaching.
        hdm = self.base.__dict__.pop("hf_device_map", None)
        try:
            if not self.info:
                self.model = PeftModel.from_pretrained(self.base, str(d), adapter_name=name)
            else:
                self.model.load_adapter(str(d), adapter_name=name)
        finally:
            if hdm is not None:
                self.base.hf_device_map = hdm
        self.model.eval()
        self.info[name] = {"sha256": _h.sha256((Path(d) / "adapter_model.safetensors").read_bytes()).hexdigest(), "dir": str(d)}
        return self.info[name]

    def render(self, messages) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def _generate(self, messages, kw, adapter: Optional[str], seed_key: Optional[str], seed: int):
        import hashlib as _h
        import json as _j
        import time as _t
        torch = self.torch
        torch.cuda.reset_peak_memory_stats()
        t0 = _t.perf_counter()
        enc = self.tok([self.render(messages)], return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(self.dev) for k, v in enc.items()}
        if seed_key is None:
            seed_key = _h.sha256(_j.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        torch.manual_seed(int(_h.sha256(f"{seed}:{seed_key}".encode()).hexdigest(), 16) % (2 ** 31))     # = t2d4.colab.backend_hf.batch_seed
        with torch.inference_mode():
            if adapter is None and self.info:
                with self.model.disable_adapter():
                    out = self.model.generate(**enc, **kw)
            else:
                if adapter is not None:
                    self.model.set_adapter(adapter)
                out = self.model.generate(**enc, **kw)
        n = enc["input_ids"].shape[1]
        new = [int(t) for t in out[0][n:].tolist()]
        stop = next((i for i, t in enumerate(new) if t in (151645, 151643)), None)
        body = new[:stop] if stop is not None else new
        dt = _t.perf_counter() - t0
        return {"text": self.tok.decode(body, skip_special_tokens=False).strip(), "hit_token_limit": stop is None and len(body) >= kw["max_new_tokens"],
                "new_tokens": len(body), "prompt_tokens": int(n), "latency_s": round(dt, 3), "tokens_per_s": round(len(body) / dt, 2) if dt > 0 else None,
                "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)}

    def __call__(self, messages, cfg, adapter: Optional[str]) -> dict:
        kw = dict(max_new_tokens=int(cfg["max_new_tokens"]), eos_token_id=[151645, 151643], pad_token_id=self.tok.pad_token_id,
                  do_sample=bool(cfg.get("do_sample", True)), repetition_penalty=float(cfg.get("repetition_penalty", 1.0)))
        if kw["do_sample"]:
            kw.update(temperature=float(cfg["temperature"]), top_p=float(cfg["top_p"]), top_k=int(cfg["top_k"]), min_p=float(cfg.get("min_p", 0.0)))
        return self._generate(messages, kw, adapter, None, int(cfg.get("seed", 20261112)))

    def complete(self, messages, max_new_tokens: int = 1024) -> dict:
        """The Memory Writer's model call: greedy, base weights (no persona adapter), JSON object extracted from the answer."""
        import re as _re
        kw = dict(max_new_tokens=max_new_tokens, eos_token_id=[151645, 151643], pad_token_id=self.tok.pad_token_id, do_sample=False)
        r = self._generate(messages, kw, None, "writer", 0)
        m = _re.search(r"\{.*\}", r["text"], _re.S)
        return {"text": m.group(0) if m else r["text"], "model": self.label, "input_tokens": r["prompt_tokens"], "output_tokens": r["new_tokens"],
                "latency_ms": r["latency_s"] * 1000, "peak_vram_gib": r["peak_vram_gib"], "hit_token_limit": r["hit_token_limit"]}

    def is_oom(self, e) -> bool:
        return isinstance(e, self.torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()

    def after_error(self):
        import gc
        gc.collect()
        self.torch.cuda.empty_cache()


def local_complete(gen, max_new_tokens: int = 1536):
    """The Memory Writer's / Fact Reader's model call from an already-loaded local generator (no second model load)."""
    if hasattr(gen, "complete"):
        return lambda messages: gen.complete(messages, max_new_tokens=max_new_tokens)
    greedy = {"do_sample": False, "max_new_tokens": max_new_tokens, "repetition_penalty": 1.0, "seed": 0}

    def complete(messages):
        r = gen(messages, greedy, None)
        m = re.search(r"\{.*\}", r["text"], re.S)
        return {"text": m.group(0) if m else r["text"], "model": gen.label, "output_tokens": r.get("new_tokens")}
    return complete


def make_generator(backend: str, base_dir: Optional[str], adapters: dict, pool: Sequence[str] = (), device_index: int = 0):
    """``fake``: the labelled CPU stand-in (SYNTHETIC / DRY_RUN_ONLY). ``hf``: ``d5pilot.evaluate.Gen`` (GPU-verified D5 generator).
    ``hf-lowvram``: ``LowVramGen`` (same contract; input embedding on CPU, for an 8 GB GPU)."""
    if backend == "fake":
        return EV.FakeGenerator(pool or ("好", "嗯嗯", "我也不知道诶"))
    if backend == "hf-lowvram":
        if not base_dir:
            raise ValueError("--base-dir is required for --backend hf-lowvram")
        return LowVramGen(Path(base_dir), adapters, device_index)
    for p in ("training/persona", "training/d5", "training/d4"):
        if str(REPO / p) not in sys.path:
            sys.path.insert(0, str(REPO / p))
    if not base_dir:
        raise ValueError("--base-dir (a local Qwen3-8B snapshot of the pinned revision) is required for --backend hf")
    from d5pilot.evaluate import Gen
    return Gen(Path(base_dir), {n: Path(d) for n, d in adapters.items()})
