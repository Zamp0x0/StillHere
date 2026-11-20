"""``persona doctor``: what this machine can do for the persona pipeline. One implementation for the CLI and the Colab notebook.

It measures (never infers from a GPU name alone): torch / CUDA, GPU model, VRAM, bf16, bitsandbytes with a real NF4 quantisation on the
GPU, transformers / peft / LLaMA-Factory versions, the Qwen3-8B base files (pinned tokenizer and chat-template sha256), free disk,
and optionally a data manifest and a config file hash. A PASS means "the formal QLoRA run may START"; that it FITS is decided by the
1-2 step GPU smoke (peak VRAM), not here. Qwen3-8B QLoRA training is refused below ``MIN_TRAIN_VRAM_GIB`` (an RTX 4060 Laptop 8 GB is an
inference / smoke-of-evaluation machine only).
"""
from __future__ import annotations

import hashlib
import importlib
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Optional

TARGETS = ("cpu-only", "local-cuda", "colab-l4", "colab-a100")
BASE_REPO = "Qwen/Qwen3-8B"
BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
CHAT_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
EXPECT = {"transformers": "4.57.1", "peft": "0.18.1", "llamafactory": "0.9.5", "bitsandbytes": "0.48.2"}
PINS_TRAIN = ["llamafactory==0.9.5", "transformers==4.57.1", "peft==0.18.1", "trl==0.24.0", "accelerate==1.11.0", "datasets==4.0.0", "tokenizers==0.22.2",
              "bitsandbytes==0.48.2", "safetensors>=0.4.3", "pyyaml>=6.0", "huggingface_hub>=0.23"]
MIN_TRAIN_VRAM_GIB = 20.0          # Qwen3-8B NF4 QLoRA, cutoff 2048 (D5 at cutoff 8192 peaked at 22.3 GB on A100; smoke decides)
MIN_INFER_VRAM_GIB = 7.0           # Qwen3-8B NF4 weights ~5.7 GB + short-context KV cache (estimate; measured by the evaluation run)
MIN_DISK_GIB = {"cpu-only": 1, "local-cuda": 25, "colab-l4": 40, "colab-a100": 40}


def _ver(mod: str) -> Optional[str]:
    try:
        return importlib.import_module(mod).__version__
    except Exception:  # noqa: BLE001
        return None


def base_check(base_dir) -> dict:
    p = Path(base_dir) if base_dir else None
    if p is None or not p.is_dir():
        return {"present": False}
    tok = p / "tokenizer.json"
    cfg = p / "tokenizer_config.json"
    tsha = hashlib.sha256(tok.read_bytes()).hexdigest() if tok.is_file() else None
    tpl = json.loads(cfg.read_text(encoding="utf-8")).get("chat_template", "") if cfg.is_file() else ""
    csha = hashlib.sha256((tpl or "").encode("utf-8")).hexdigest()
    shards = sorted(x.name for x in p.glob("*.safetensors"))
    return {"present": True, "dir": str(p), "tokenizer_sha256_ok": tsha == TOKENIZER_SHA256, "chat_template_sha256_ok": csha == CHAT_TEMPLATE_SHA256,
            "weight_shards": len(shards), "revision_in_path": BASE_REVISION in str(p)}


def gpu_check(nf4: bool = True) -> dict:
    out = {"torch": _ver("torch"), "cuda_available": False}
    try:
        import torch
    except Exception:  # noqa: BLE001
        return out
    out["torch_cuda"] = torch.version.cuda
    if not torch.cuda.is_available():
        return out
    p = torch.cuda.get_device_properties(0)
    out.update(cuda_available=True, gpu=torch.cuda.get_device_name(0), vram_total_gib=round(p.total_memory / 2**30, 1), bf16=bool(torch.cuda.is_bf16_supported()),
               capability=".".join(map(str, torch.cuda.get_device_capability(0))))
    if nf4:
        try:
            import bitsandbytes.functional as F
            x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
            q, state = F.quantize_4bit(x, quant_type="nf4", compress_statistics=True)
            y = F.dequantize_4bit(q, state, quant_type="nf4")
            out["nf4_roundtrip_ok"] = bool(torch.isfinite(y).all()) and float((y - x).abs().mean()) < 0.2
        except Exception as e:  # noqa: BLE001
            out["nf4_roundtrip_ok"] = False
            out["nf4_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return out


def doctor(target: str, *, base_dir=None, data_dir=None, config_path=None, work_dir=".", allow_other_gpu: bool = False, nf4: bool = True) -> dict:
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    g = gpu_check(nf4=nf4 and target != "cpu-only")
    vers = {m: _ver(m) for m in ("transformers", "peft", "bitsandbytes", "accelerate", "tokenizers", "llamafactory", "yaml")}
    free = round(shutil.disk_usage(Path(work_dir).resolve().anchor or "/").free / 2**30, 1)
    rep = {"target": target, "python": sys.version.split()[0], "platform": platform.platform(), "gpu": g, "versions": vers, "disk_free_gib": free,
           "base": base_check(base_dir), "expected_versions": EXPECT, "base_revision": BASE_REVISION}
    if data_dir:
        d = Path(data_dir)
        man = json.loads((d / "DATA_MANIFEST.json").read_text(encoding="utf-8"))
        bad = [n for n, h in man["files"].items() if not (d / n).is_file() or hashlib.sha256((d / n).read_bytes()).hexdigest() != h]
        rep["dataset"] = {"manifest_sha256": hashlib.sha256((d / "DATA_MANIFEST.json").read_bytes()).hexdigest(), "files": len(man["files"]), "bad_files": bad[:5],
                          "contains_test": man.get("contains_test"), "contains_raw_export": man.get("contains_raw_export")}
    if config_path:
        rep["config_sha256"] = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    checks = {"disk_ok": free >= MIN_DISK_GIB[target]}
    if target == "cpu-only":
        checks["python_ok"] = sys.version_info >= (3, 10)
    else:
        want = {"colab-l4": "L4", "colab-a100": "A100"}.get(target)
        checks.update(cuda=g.get("cuda_available", False), bf16=bool(g.get("bf16")), nf4=bool(g.get("nf4_roundtrip_ok")),
                      versions_ok=all(vers.get(k) == v for k, v in EXPECT.items()))
        if want:
            checks["gpu_model_ok"] = bool(g.get("gpu") and want in g["gpu"]) or allow_other_gpu
        if rep["base"].get("present"):
            checks["base_ok"] = rep["base"]["tokenizer_sha256_ok"] and rep["base"]["chat_template_sha256_ok"]
        if "dataset" in rep:
            checks["dataset_ok"] = not rep["dataset"]["bad_files"] and rep["dataset"]["contains_test"] is False
    vram = g.get("vram_total_gib") or 0
    rep["capabilities"] = {
        "prepare_data_prompt_package": True,
        "qwen3_8b_nf4_inference": bool(g.get("cuda_available")) and vram >= MIN_INFER_VRAM_GIB,
        "qwen3_8b_qlora_training": bool(g.get("cuda_available")) and vram >= MIN_TRAIN_VRAM_GIB and target != "cpu-only",
        "note": "training capability means the formal run may start; the 1-2 step GPU smoke measures whether it fits"}
    rep["checks"] = checks
    blocked = [k for k, v in checks.items() if not v]
    if target in ("local-cuda", "colab-l4", "colab-a100") and not rep["capabilities"]["qwen3_8b_qlora_training"]:
        blocked.append(f"vram {vram} GiB < {MIN_TRAIN_VRAM_GIB} GiB: Qwen3-8B QLoRA training refused here (inference / evaluation only)"
                       if g.get("cuda_available") else "no CUDA GPU")
    rep["blocked"] = blocked
    rep["status"] = "PASS" if not blocked else "BLOCKED"
    return rep
