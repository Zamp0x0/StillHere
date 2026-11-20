"""Generation backends (from the A100-proven T2-D3 Mini backend). ONE Qwen3-8B (NF4 double-quant, bf16 compute, exactly as trained) stays
resident; adapters are attached by name (``ckpt-180`` … ``best``) and the Base groups run with every adapter disabled, so Base and Adapter
share the same quantised weights, tokenizer, chat template, prompt and sampling settings.

D4 changes: sampling is the frozen protocol sampling (temperature 0.7, top_p 0.8, top_k 20, min_p 0, repetition_penalty 1.0, no presence
penalty, max_new_tokens 128, enable_thinking false; greedy is never used for a gate metric). Each batch is seeded from (job seed, first job
id), so a rerun with the same batching reproduces it. Special tokens other than the stop tokens are KEPT in the text (so a leaked
``<think>`` or ``<|…|>`` is visible to the metrics), and each row records whether generation stopped at max_new_tokens."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict

REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
REPO = "Qwen/Qwen3-8B"
TEMPLATE_SHA = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
STOP_IDS = (151645, 151643)            # <|im_end|>, <|endoftext|>
SAMPLING = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0}


def batch_seed(seed: int, first_job_id: str) -> int:
    return int(hashlib.sha256(f"{seed}:{first_job_id}".encode()).hexdigest(), 16) % (2 ** 31)


def resolve_base(source: str, cache_dir: str) -> Path:
    """``source``: "hub" (download the pinned revision) or a local directory holding that revision. Returns the snapshot directory."""
    if source != "hub":
        return Path(source)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(REPO, revision=REVISION, cache_dir=cache_dir,
                                  allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json", "tokenizer*"]))


class HFBackend:
    def __init__(self, base_dir: Path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        if not torch.cuda.is_available():
            raise RuntimeError("MODEL_LOAD_FAILURE: CUDA is not available (this backend runs on Colab only)")
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(str(base_dir))
        self.tok.padding_side = "left"
        self.template_sha = hashlib.sha256((self.tok.chat_template or "").encode("utf-8")).hexdigest()
        if self.template_sha != TEMPLATE_SHA:
            raise RuntimeError(f"CONFIG_MISMATCH: chat template sha {self.template_sha[:12]} != {TEMPLATE_SHA[:12]}")
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        kw = dict(device_map={"": 0}, quantization_config=q)
        try:
            self.base = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.bfloat16, **kw)
        except TypeError:
            self.base = AutoModelForCausalLM.from_pretrained(str(base_dir), torch_dtype=torch.bfloat16, **kw)
        self.base.eval()
        self.model = self.base
        self.adapters: Dict[str, dict] = {}
        self.gpu = torch.cuda.get_device_name(0)

    def add_adapter(self, name: str, adapter_dir: Path) -> dict:
        from peft import PeftModel
        info = adapter_info(adapter_dir)
        if name in self.adapters:
            if self.adapters[name]["sha256"] != info["sha256"]:
                raise RuntimeError(f"CONFIG_MISMATCH: adapter name {name} already loaded with another file")
            return self.adapters[name]
        if not self.adapters:
            self.model = PeftModel.from_pretrained(self.base, str(adapter_dir), adapter_name=name)
        else:
            self.model.load_adapter(str(adapter_dir), adapter_name=name)
        self.model.eval()
        self.adapters[name] = info
        return info

    def drop_adapter(self, name: str):
        if name in self.adapters and len(self.adapters) > 1:
            self.model.delete_adapter(name)
            self.adapters.pop(name)
            self.empty_cache()

    def render(self, messages) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def prompt_length(self, messages) -> int:
        return len(self.tok(self.render(messages), add_special_tokens=False)["input_ids"])

    def generate(self, batch, max_new_tokens: int):
        torch = self.torch
        enc = self.tok([self.render(j["messages"]) for j in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(0)
        kw = dict(max_new_tokens=max_new_tokens, eos_token_id=list(STOP_IDS), pad_token_id=self.tok.pad_token_id, **SAMPLING)
        torch.manual_seed(batch_seed(int(batch[0].get("seed") or 0), batch[0]["job_id"]))
        name = batch[0]["model"]
        with torch.inference_mode():
            if name == "base":
                if self.adapters:
                    with self.model.disable_adapter():
                        out = self.model.generate(**enc, **kw)
                else:
                    out = self.model.generate(**enc, **kw)
            else:
                if name not in self.adapters:
                    raise RuntimeError(f"CONFIG_MISMATCH: adapter {name} is not loaded")
                self.model.set_adapter(name)
                out = self.model.generate(**enc, **kw)
        n = enc["input_ids"].shape[1]
        res = []
        for row in out:
            new = [int(t) for t in row[n:].tolist()]
            stop = next((i for i, t in enumerate(new) if t in STOP_IDS), None)
            body = new[:stop] if stop is not None else new
            text = self.tok.decode(body, skip_special_tokens=False).strip()
            res.append({"text": text, "new_tokens": len(body) + (1 if stop is not None else 0), "hit_max": stop is None and len(body) >= max_new_tokens,
                        "raw_special": any(t >= 151643 for t in body)})
        del enc, out
        return res

    def empty_cache(self):
        self.torch.cuda.empty_cache()

    def vram(self) -> dict:
        t = self.torch.cuda
        return {"vram_gib": round(t.memory_allocated() / 2**30, 2), "peak_vram_gib": round(t.max_memory_allocated() / 2**30, 2),
                "vram_total_gib": round(t.get_device_properties(0).total_memory / 2**30, 1)}


class MockBackend:
    """CPU stand-in with the same interface (tests and the CPU release simulation only; every output of a mock run is labelled DRY_RUN).
    Base: a generic reply. Adapter: the item's reference target (so the metrics see grounded replies), except a planted share of leaks,
    copies and loops when ``plant`` is set, so the CPU simulation exercises every failure path of the metrics and the guard."""

    gpu = "mock-cpu"

    def __init__(self, counter=None, plant: bool = True):
        self.counter, self.plant, self.adapters = counter, plant, {}

    def add_adapter(self, name, adapter_dir=None):
        self.adapters[name] = {"sha256": hashlib.sha256(name.encode()).hexdigest()}
        return self.adapters[name]

    def drop_adapter(self, name):
        self.adapters.pop(name, None)

    def prompt_length(self, messages) -> int:
        if self.counter is not None:
            from ..tokens import PROMPT_FMT
            return self.counter.n(PROMPT_FMT.format(system=messages[0]["content"], user=messages[1]["content"]))
        return sum(len(m["content"]) for m in messages) // 2

    def generate(self, batch, max_new_tokens: int):
        out = []
        for j in batch:
            h = int(hashlib.sha256((j["job_id"]).encode()).hexdigest(), 16)
            mock = j.get("mock") or {}
            if j["model"] == "base":
                t = "这个问题可以先看看再说，希望对你有帮助。"
            elif self.plant and mock.get("leak") and h % 9 == 0:
                t = "号码是" + mock["leak"]
            elif self.plant and h % 23 == 0:
                t = "哈哈哈哈哈哈哈哈哈哈哈哈"
            else:
                t = mock.get("reference") or "嗯"
            out.append({"text": t, "new_tokens": max(1, len(t) // 2), "hit_max": False, "raw_special": False})
        return out

    def empty_cache(self):
        pass

    def vram(self) -> dict:
        return {"vram_gib": 0.0, "peak_vram_gib": 0.0, "vram_total_gib": 0.0}


def adapter_info(adapter_dir: Path) -> dict:
    p = Path(adapter_dir)
    h = hashlib.sha256()
    with (p / "adapter_model.safetensors").open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    cfg = json.loads((p / "adapter_config.json").read_text(encoding="utf-8"))
    return {"dir": str(p), "sha256": h.hexdigest(), "r": cfg.get("r"), "alpha": cfg.get("lora_alpha"), "targets": sorted(cfg.get("target_modules") or []),
            "base_model": cfg.get("base_model_name_or_path"), "revision": cfg.get("revision")}
