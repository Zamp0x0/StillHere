"""``persona train`` (local CUDA machine) and the label-audit / dry-run bridges. The trainer itself is ``personatrain.train_persona`` - the same
module the Colab notebook runs - started as a subprocess with the repository's training code on PYTHONPATH.

Local Qwen3-8B QLoRA is refused when ``persona doctor --target local-cuda`` does not report the capability (VRAM < 20 GiB, e.g. an RTX 4060
Laptop 8 GB); ``--dry-run`` (CPU) resolves config, gates, fingerprint and resume plan without importing torch.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional

from . import doctor as DR
from . import store as S

REPO = Path(__file__).resolve().parents[3]


class TrainingRefused(S.StoreError):
    pass


def select_gpu(gpu: Optional[int]) -> dict:
    """Pin ONE physical GPU for this process and its training subprocesses: ``CUDA_VISIBLE_DEVICES=<gpu>``, which every CUDA library then
    sees as ``cuda:0``. It must happen before CUDA is initialised in this process (otherwise the mapping would silently not apply)."""
    if gpu is None:
        return {}
    if int(gpu) < 0:
        raise TrainingRefused("DEVICE", "--gpu must be a GPU index >= 0")
    torch = sys.modules.get("torch")
    if torch is not None and getattr(torch.cuda, "is_initialized", lambda: False)() and os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise TrainingRefused("DEVICE", "CUDA is already initialised in this process; choose --gpu before anything loads a model")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    return {"CUDA_VISIBLE_DEVICES": str(int(gpu)), "in_process_device": "cuda:0", "physical_gpu": int(gpu)}


def env() -> dict:
    return {**os.environ, "PYTHONIOENCODING": "utf-8", "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": os.pathsep.join(str(REPO / x) for x in ("training/persona", "training/d5", "training/d4", "src"))}


def unpack(package_dir: Path, run_dir: Path) -> Path:
    data = run_dir / "data"
    marker = data / ".zip_sha256"
    sums = dict((l.split(None, 1)[1].strip(), l.split(None, 1)[0]) for l in (package_dir / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines() if l.strip())
    import hashlib
    zp = package_dir / "persona_data.zip"
    if hashlib.sha256(zp.read_bytes()).hexdigest() != sums["persona_data.zip"]:
        raise S.StoreError("PACKAGE_CORRUPT", "persona_data.zip does not match SHA256SUMS.txt")
    if not marker.is_file() or marker.read_text().strip() != sums["persona_data.zip"]:
        if data.exists() and any(data.iterdir()):
            raise S.StoreError("RUN_EXISTS", f"{data} holds another package; use a new --run-id")
        with zipfile.ZipFile(zp) as z:
            z.extractall(data)
        marker.write_text(sums["persona_data.zip"])
    return data


def train(root, profile_id: str, *, package_id: str, run_id: Optional[str] = None, base_dir: Optional[str] = None, device: str = "cuda",
          dry_run: bool = False, smoke_only: bool = False, python: Optional[str] = None, gpu: Optional[int] = None) -> dict:
    root = Path(root) if root else S.default_root()
    gpu_map = select_gpu(gpu)
    pdir = S.profile_dir(root, profile_id)
    pkg = pdir / "packages" / package_id
    if not (pkg / "RELEASE_MANIFEST.json").is_file():
        raise S.StoreError("NO_PACKAGE", f"package {package_id} not found (stillhere persona package --target local-cuda)")
    rm = S.read_json(pkg / "RELEASE_MANIFEST.json")
    rid = run_id or rm["run_id_suggested"]
    run = pdir / "runs" / rid
    data = unpack(pkg, run)
    cfg = data / "config" / "training_config.yaml"
    if not dry_run:
        if device != "cuda":
            raise TrainingRefused("DEVICE", "persona train runs on --device cuda (or use the Colab notebook); --dry-run checks everything on CPU")
        rep = DR.doctor("local-cuda", base_dir=base_dir, data_dir=data, config_path=cfg, work_dir=str(run))
        S.write_json(run / "DOCTOR.json", rep)
        if not rep["capabilities"]["qwen3_8b_qlora_training"] and not smoke_only:
            raise TrainingRefused("GPU_INSUFFICIENT", "; ".join(rep["blocked"]) + " -> use the Colab L4 notebook of this package")
        if rep["status"] != "PASS" and not smoke_only:
            raise TrainingRefused("DOCTOR_BLOCKED", "; ".join(rep["blocked"]))
    py = python or sys.executable
    lf = run / "LF_LABELS_RESULT.json"
    local = data / "audit" / "LF_LABELS_RESULT_LOCAL.json"           # the package's CPU audit of this exact train file (the gate re-checks its sha)
    if not lf.is_file() and local.is_file() and S.read_json(local).get("LLAMAFACTORY_LABELS") == "PASS":
        lf = local
    cmds = []
    if not dry_run and not (lf.is_file() and S.read_json(lf).get("LLAMAFACTORY_LABELS") == "PASS"):
        cmds.append([py, "-m", "personatrain.lf_labels", "--data", str(data), "--model-dir", str(base_dir), "--work", str(run / "lf_work"), "--out", str(run),
                     "--splits", "train,dev", "--persona-prompt", str(data / "persona" / "persona_prompt.txt")])
    base = [py, "-m", "personatrain.train_persona", "--data", str(data), "--config", str(cfg)] + (["--base-dir", str(base_dir)] if base_dir else [])
    if dry_run:
        cmds.append(base + ["--output-dir", str(run / "train"), "--lf-result", str(lf), "--dry-run"])
    else:
        cmds.append(base + ["--output-dir", str(run / "smoke"), "--smoke", "--dataset", "persona_smoke", "--max-steps", "2", "--save-steps", "1", "--stop-at-step", "1"])
        cmds.append(base + ["--output-dir", str(run / "smoke"), "--smoke", "--dataset", "persona_smoke", "--max-steps", "2", "--save-steps", "1"])
        if not smoke_only:
            cmds.append(base + ["--output-dir", str(run / "train"), "--lf-result", str(lf)])
    results = []
    for c in cmds:
        r = subprocess.run(c, env=env(), capture_output=True, text=True, encoding="utf-8", errors="replace")
        results.append({"cmd": " ".join(Path(c[0]).name if i == 0 else x for i, x in enumerate(c[:4])), "rc": r.returncode, "tail": (r.stdout + r.stderr)[-1500:]})
        if r.returncode != 0:
            break
    ok = all(x["rc"] == 0 for x in results)
    job = {"schema": "stillhere.persona.training_job/1.0", "package_id": package_id, "run_id": rid, "backend": "local-cuda", "dry_run": dry_run,
           "gpu": gpu_map or None, "dataset_version": rm.get("dataset_version"), "card_id": rm.get("card_id"), "base": rm.get("base"),
           "training_config_sha256": rm.get("training_config_sha256"), "checkpoint_dir": str(run / "train"),
           "status": ("DRY_RUN_OK" if dry_run else "DONE") if ok else "FAILED",
           "resume": "run the same command again: the trainer resumes from the newest checkpoint in checkpoint_dir", "at": S.now_iso()}
    S.write_json(run / "TRAINING_JOB.json", job)
    S.write_json(run / ("TRAIN_DRY_RUN.json" if dry_run else "TRAIN_LOCAL.json"), {"package_id": package_id, "run_id": rid, "steps": results, "gpu": gpu_map or None, "at": S.now_iso()})
    return {"run_id": rid, "run_dir": str(run), "dry_run": dry_run, "steps": results, "ok": ok, "gpu": gpu_map or None, "job": str(run / "TRAINING_JOB.json")}
