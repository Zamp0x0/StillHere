#!/usr/bin/env python3
"""Train the T2-D4 adapter with LLaMA-Factory in process (``run_exp`` + TrainerCallbacks; the Trainer is not changed). From the A100-proven
T2-D3 Mini ``train_mini`` (same resume / fingerprint / pre-flight / ALREADY COMPLETE logic); the review-protocol gate is replaced by the D4 gates.

Order: data manifest (sha256 of every file) -> D4 gates of a formal run (frozen dataset: freeze manifest present, Gate 0 D4_1_DATA_READY = YES,
G0 LLAMAFACTORY_LABELS = PASS) -> resolved config (steps recomputed from the data and compared with the frozen expectation) -> run fingerprint
(resume refused on any difference) -> newest COMPLETE checkpoint -> resume pre-flight -> ``run_exp`` -> records (training_metrics.json,
checkpoint_manifest.json, training_log.json).

Smoke fix (T2-D3 smoke reported no last_train_loss because 3 steps < logging_steps 5): smoke and CPU runs log every step (logging_steps 1),
so the loss is a real per-step value; the end-of-run summary fallback stays for any other short run and is labelled as such.

    python -m t2d4.colab.train_d4 --data <extracted data pkg> --config configs/d4_train.yaml --output-dir <Drive run>/train
    python -m t2d4.colab.train_d4 ... --smoke --max-steps 3 --save-steps 3 --output-dir <Drive run>/smoke      (QLoRA smoke, GPU)
    python -m t2d4.colab.train_d4 ... --cpu-smoke --model <tiny model dir> --max-steps 4 --save-steps 2           (CPU tests only)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import progress as PG

CPU_DROP = ("quantization_bit", "quantization_method", "quantization_type", "double_quantization", "flash_attn")
D4_KEYS = ("d4_expected_train_rows", "d4_expected_steps", "d4_candidate_fractions", "d4_overfit_margin")
TRAIN_FILE = Path("train") / "train.jsonl"
FINGERPRINT_KEYS = ("model_name_or_path", "model_revision", "finetuning_type", "lora_rank", "lora_alpha", "lora_dropout", "lora_target", "quantization_bit",
                    "quantization_type", "double_quantization", "dataset", "eval_dataset", "template", "enable_thinking", "mask_history", "train_on_prompt", "cutoff_len",
                    "packing", "per_device_train_batch_size", "gradient_accumulation_steps", "learning_rate", "lr_scheduler_type", "warmup_ratio",
                    "num_train_epochs", "max_steps", "seed", "optim", "bf16", "eval_steps", "save_steps")
CKPT_FILES = ("adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt", "trainer_state.json")
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def verify_data(data: Path) -> tuple:
    man = json.loads((data / "DATA_MANIFEST.json").read_text(encoding="utf-8"))
    bad = []
    for rel, h in man["files"].items():
        p = data / rel
        if not p.exists():
            bad.append(f"missing {rel}")
        elif sha256(p) != h:
            bad.append(f"sha256 mismatch {rel}")
    return man, bad


def expected_steps(rows: int, cfg: dict) -> int:
    """transformers 4.57.1 ``set_initial_training_values``: ceil(micro-batches / grad-accum) optimizer steps per epoch (a partial last
    accumulation counts as a step), total = ceil(epochs x per-epoch). One GPU."""
    micro = math.ceil(rows / cfg["per_device_train_batch_size"])
    per_epoch = max(math.ceil(micro / cfg["gradient_accumulation_steps"]), 1)
    return int(math.ceil(per_epoch * cfg["num_train_epochs"])) if not cfg.get("max_steps") else int(cfg["max_steps"])


def resolve(config: dict, data: Path, out: Path, *, cpu_smoke: bool, model, max_steps, save_steps) -> dict:
    c = {k: v for k, v in config.items() if k not in D4_KEYS}
    c.update(dataset_dir=str((data / "train").resolve()), output_dir=str(out.resolve()), overwrite_output_dir=False)
    if model:
        c["model_name_or_path"] = model
        c.pop("model_revision", None)
    if max_steps:
        c["max_steps"] = max_steps
    if save_steps:
        c.update(save_steps=save_steps, eval_steps=save_steps)
    if max_steps or save_steps:
        c["logging_steps"] = 1  # short runs (smoke): a real per-step loss, never only the end-of-run summary
    if cpu_smoke:
        for k in CPU_DROP:
            c.pop(k, None)
        c.update(bf16=False, fp16=False, optim="adamw_torch", gradient_checkpointing=False, preprocessing_num_workers=None, use_cpu=True,
                 per_device_train_batch_size=2, gradient_accumulation_steps=2, per_device_eval_batch_size=2, logging_steps=1, plot_loss=False)
    return c


def fingerprint(cfg: dict, data_manifest_sha: str) -> dict:
    keep = {k: cfg.get(k) for k in FINGERPRINT_KEYS}
    keep["data_manifest_sha256"] = data_manifest_sha
    return {"fields": keep, "sha256": hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()}


def checkpoints(out: Path):
    return sorted((p for p in out.glob("checkpoint-*") if p.is_dir() and p.name.split("-")[1].isdigit()), key=lambda p: int(p.name.split("-")[1]))


def missing_files(ck: Path) -> list:
    m = [f for f in CKPT_FILES if not (ck / f).exists()]
    if not list(ck.glob("rng_state*.pth")):
        m.append("rng_state*.pth")
    return m


def preflight(ck: Path) -> dict:
    import torch
    state = json.loads((ck / "trainer_state.json").read_text(encoding="utf-8"))
    sched = torch.load(ck / "scheduler.pt", map_location="cpu", weights_only=False)
    opt = torch.load(ck / "optimizer.pt", map_location="cpu", weights_only=False)
    res = {"checkpoint": ck.name, "trainer_state_global_step": state.get("global_step"),
           "scheduler_last_epoch": sched.get("last_epoch") if isinstance(sched, dict) else None,
           "optimizer_param_groups": len(opt.get("param_groups", [])) if isinstance(opt, dict) else 0,
           "optimizer_state_tensors": len(opt.get("state", {})) if isinstance(opt, dict) else 0}
    res["ok"] = (res["trainer_state_global_step"] == int(ck.name.split("-")[1]) == res["scheduler_last_epoch"] and res["optimizer_param_groups"] > 0
                 and res["optimizer_state_tensors"] > 0)
    return res


def d4_gates(data: Path) -> dict:
    """Formal-run gates: frozen dataset, data Gate 0 YES, real-framework label check PASS (all read from the data package's meta/)."""
    meta = data / "meta"
    rd = lambda n: json.loads((meta / n).read_text(encoding="utf-8")) if (meta / n).exists() else {}
    fz, g0, lab = rd("dataset_freeze_manifest.json"), rd("data_gate0.json"), rd("G0_LABELS_RESULT.json")
    checks = {"dataset_frozen": bool(fz.get("data_version")), "gate0_yes": g0.get("D4_1_DATA_READY") == "YES",
              "g0_labels_pass": lab.get("LLAMAFACTORY_LABELS") == "PASS"}
    failed = [k for k, v in checks.items() if not v]
    return {"checks": checks, "failed": failed, "training_allowed": not failed, "status": "PASS" if not failed else "BLOCKED",
            "data_version": fz.get("data_version")}


def quiet_examples() -> int:
    """LLaMA-Factory prints one decoded training / eval example when it loads a dataset. That text would reach the notebook output and the
    Drive log, so every dataset processor's ``print_data_example`` is replaced by a line with token counts only. Returns how many were patched."""
    import importlib
    import pkgutil
    import llamafactory.data.processor as P
    n = 0

    def withheld(self, example):
        lab = example.get("labels") or []
        print(f"[t2d4] dataset example withheld (private text): {len(example.get('input_ids') or [])} input tokens, "
              f"{sum(1 for x in lab if x != -100)} supervised", flush=True)
    for m in pkgutil.iter_modules(P.__path__):
        mod = importlib.import_module(f"{P.__name__}.{m.name}")
        for obj in vars(mod).values():
            if isinstance(obj, type) and "print_data_example" in vars(obj):
                obj.print_data_example = withheld
                n += 1
    return n


def gpu_info() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return {"gpu": torch.cuda.get_device_name(0), "bf16": torch.cuda.is_bf16_supported(), "vram_gb": round(p.total_memory / 2**30, 1)}
    except Exception:  # noqa: BLE001
        pass
    return {"gpu": "cpu", "bf16": None, "vram_gb": None}


def gpu_mem() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {"gpu_mem_allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 2), "gpu_mem_reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 2),
                    "gpu_mem_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    except Exception:  # noqa: BLE001
        pass
    return {}


def make_callbacks(out: Path, stop_at, resume_ck, ev):
    from transformers import TrainerCallback

    log_path = out / "metrics_log.jsonl"

    def write(row):
        row["at"] = datetime.now(timezone.utc).isoformat()
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    class Recorder(TrainerCallback):
        def __init__(self):
            self.begin_step, self.t_last, self.step_last = None, None, None

        def on_train_begin(self, args, state, control, **kw):
            self.begin_step, self.t_last, self.step_last = state.global_step, time.time(), state.global_step
            write({"event": "train_begin", "global_step": state.global_step, "max_steps": state.max_steps, "num_train_epochs": args.num_train_epochs,
                   "effective_batch": args.per_device_train_batch_size * args.gradient_accumulation_steps * max(args.world_size, 1), **gpu_info(), **gpu_mem()})

        def on_log(self, args, state, control, logs=None, **kw):
            now = time.time()
            steps = state.global_step - (self.step_last or 0)
            row = {"event": "log", "global_step": state.global_step, "epoch": round(state.epoch or 0, 4),
                   **{k: v for k, v in (logs or {}).items() if isinstance(v, (int, float))}, **gpu_mem()}
            if steps > 0 and self.t_last:
                row["seconds_per_step"] = round((now - self.t_last) / steps, 3)
                self.t_last, self.step_last = now, state.global_step
            write(row)

        def on_step_end(self, args, state, control, **kw):
            if stop_at and state.global_step >= stop_at:
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_save(self, args, state, control, **kw):
            ck = out / f"checkpoint-{state.global_step}"
            (out / "latest_resumable.json").write_text(json.dumps({"checkpoint": ck.name, "global_step": state.global_step,
                                                                    "missing": missing_files(ck) if ck.exists() else ["dir"]}, indent=1) + "\n", encoding="utf-8")
            write({"event": "save", "global_step": state.global_step, "checkpoint": ck.name})

    class ProgressEvents(TrainerCallback):
        def __init__(self):
            self.t0 = self.tok0 = None

        def on_init_end(self, args, state, control, **kw):
            ev.emit("phase", phase="READY", note="model and data loaded")

        def on_train_begin(self, args, state, control, **kw):
            ev.emit("train_begin", global_step=state.global_step, max_steps=state.max_steps, epochs=args.num_train_epochs, eval_steps=args.eval_steps,
                    save_steps=args.save_steps, effective_batch=args.per_device_train_batch_size * args.gradient_accumulation_steps, gpu_info=gpu_info(),
                    resumed_from=resume_ck.name if resume_ck else None)
            ev.emit("phase", phase="TRAINING", global_step=state.global_step)

        def on_step_begin(self, args, state, control, **kw):
            self.t0, self.tok0 = time.time(), state.num_input_tokens_seen

        def on_step_end(self, args, state, control, **kw):
            if self.t0 is None:
                return control
            dt = time.time() - self.t0
            tok = (state.num_input_tokens_seen - self.tok0) if self.tok0 is not None and state.num_input_tokens_seen else None
            ev.emit("step", global_step=state.global_step, epoch=round(state.epoch or 0, 4), step_seconds=round(dt, 4), tokens=tok,
                    tokens_seen=state.num_input_tokens_seen or None, gpu_mem=gpu_mem() if state.global_step % 10 == 0 else None)
            self.t0 = None
            if control.should_evaluate:
                ev.emit("phase", phase="VALIDATING", global_step=state.global_step)
            elif control.should_save:
                ev.emit("phase", phase="SAVING", global_step=state.global_step)
            return control

        def on_log(self, args, state, control, logs=None, **kw):
            logs = logs or {}
            if "loss" in logs or "learning_rate" in logs:
                ev.emit("log", global_step=state.global_step, epoch=round(state.epoch or 0, 4), loss=logs.get("loss"), learning_rate=logs.get("learning_rate"),
                        grad_norm=logs.get("grad_norm"), gpu_mem=gpu_mem())

        def on_evaluate(self, args, state, control, metrics=None, **kw):
            for k, v in (metrics or {}).items():
                if k.startswith("eval_") and k.endswith("_loss"):
                    ev.emit("eval", global_step=state.global_step, name=k, loss=v)
            ev.emit("selection", global_step=state.global_step, mean_val_loss=(metrics or {}).get("eval_loss"), probe=None, qualified=None,
                    best_loss=None, best_style=None, early_stop=None, probe_seconds=0.0, early_stopping={})

        def on_save(self, args, state, control, **kw):
            ev.emit("save", global_step=state.global_step, checkpoint=f"checkpoint-{state.global_step}")
            ev.emit("phase", phase="TRAINING", global_step=state.global_step)

        def on_train_end(self, args, state, control, **kw):
            ev.emit("train_end", global_step=state.global_step, final_selection=None, stopped=None)

    return Recorder(), ProgressEvents()


def summary_train_loss(log: list, global_step, meta: dict) -> tuple:
    """Fallback when a run logged no per-step loss (a run shorter than logging_steps, e.g. the 3-step smoke with logging_steps 5): the
    Trainer's own end-of-run ``train_loss`` (mean over the steps of that run). Taken only from the entry of the final step, only for a run that
    was not resumed (after a resume the Trainer divides by the global step, not by the steps of that run), and only if finite and > 0."""
    if meta.get("resumed_from"):
        return None, None
    for x in reversed(log):
        v = x.get("train_loss")
        if x.get("step") == global_step and isinstance(v, (int, float)) and math.isfinite(v) and v > 0:
            return float(v), "trainer_summary.train_loss (mean over the steps of the run; no per-step loss was logged)"
    return None, None


def records(out: Path, meta: dict) -> dict:
    ck = checkpoints(out)
    sf = out / "trainer_state.json" if (out / "trainer_state.json").exists() else (ck[-1] / "trainer_state.json" if ck else None)
    state = json.loads(sf.read_text(encoding="utf-8")) if sf else {}
    log = state.get("log_history", [])
    events = [json.loads(l) for l in (out / "metrics_log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()] if (out / "metrics_log.jsonl").exists() else []
    (out / "training_log.json").write_text(json.dumps({**meta, "log_history": log, "events": events}, indent=1) + "\n", encoding="utf-8")
    tl = [x["loss"] for x in log if "loss" in x]
    last, last_src = (tl[-1], "log_history.loss") if tl else summary_train_loss(log, state.get("global_step"), meta)
    ev_loss = [{"step": x.get("step"), "loss": x["eval_loss"]} for x in log if "eval_loss" in x]
    sps = [e["seconds_per_step"] for e in events if e.get("seconds_per_step")]
    peak = max((e.get("gpu_mem_peak_gb", 0) for e in events), default=None)
    m = {**meta, "global_step": state.get("global_step"), "max_steps": state.get("max_steps"), "epoch": state.get("epoch"),
         "first_train_loss": tl[0] if tl else None, "last_train_loss": last, "last_train_loss_source": last_src, "eval_losses": ev_loss,
         "median_seconds_per_step": sorted(sps)[len(sps) // 2] if sps else None, "gpu_mem_peak_gb": peak,
         "gpus_used": sorted({e.get("gpu") for e in events if e.get("event") == "train_begin" and e.get("gpu")}),
         "completed": bool(state) and state.get("global_step") == state.get("max_steps"),
         "trainer_max_steps_equals_expected": (state.get("max_steps") == meta.get("expected_steps")) if state else None,
         "last_train_loss_finite": isinstance(last, (int, float)) and math.isfinite(last)}
    (out / "training_metrics.json").write_text(json.dumps(m, indent=1) + "\n", encoding="utf-8")
    cm = {**meta, "checkpoints": [{"name": c.name, "global_step": int(c.name.split("-")[1]), "missing": missing_files(c),
                                    "files": {f.name: sha256(f) for f in sorted(c.iterdir()) if f.name in ADAPTER_FILES + ("trainer_state.json",)}} for c in ck]}
    (out / "checkpoint_manifest.json").write_text(json.dumps(cm, indent=1) + "\n", encoding="utf-8")
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--save-steps", type=int)
    ap.add_argument("--stop-at-step", type=int, help="testing only: stop (and save) at this step, as a disconnect would")
    ap.add_argument("--smoke", action="store_true", help="short QLoRA smoke on the real data (no D4 gate requirement, never selected)")
    ap.add_argument("--cpu-smoke", action="store_true")
    ap.add_argument("--records-only", action="store_true")
    ap.add_argument("--progress-dir")
    a = ap.parse_args(argv)
    import yaml

    data, out = Path(a.data), Path(a.output_dir)
    man, bad = verify_data(data)
    if bad:
        print("DATA MANIFEST FAIL:\n  " + "\n  ".join(bad[:10]))
        return 1
    synthetic = man.get("data_origin") == "synthetic_fixture"
    if a.cpu_smoke and not synthetic:
        print("--cpu-smoke is for synthetic fixture packages only")
        return 1
    formal = not (a.smoke or a.cpu_smoke or a.max_steps)
    raw = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    cfg = resolve(raw, data, out, cpu_smoke=a.cpu_smoke, model=a.model, max_steps=a.max_steps, save_steps=a.save_steps)
    rows = sum(1 for l in (data / TRAIN_FILE).read_text(encoding="utf-8").splitlines() if l.strip())
    steps = expected_steps(rows, cfg)
    gate = {"status": "NOT_EVALUATED (not a formal run)"}
    if formal and not synthetic:
        gate = d4_gates(data)
        if not gate["training_allowed"]:
            print(f"TRAINING REFUSED: D4 gates: {gate['failed']}")
            return 3
        if rows != raw.get("d4_expected_train_rows") or steps != raw.get("d4_expected_steps"):
            print(f"CONFIG FAIL: data gives {rows} rows / {steps} steps, config expects {raw.get('d4_expected_train_rows')} / {raw.get('d4_expected_steps')}")
            return 1
    fp = fingerprint(cfg, sha256(data / "DATA_MANIFEST.json"))
    meta = {"data_origin": man.get("data_origin"), "data_version": man.get("version"), "data_manifest_sha256": fp["fields"]["data_manifest_sha256"],
            "config": Path(a.config).name, "run_fingerprint": fp["sha256"], "train_rows": rows, "expected_steps": steps, "smoke": a.smoke,
            "cpu_smoke": a.cpu_smoke, "model_revision": cfg.get("model_revision"), "is_formal_training": bool(formal and not synthetic),
            "d4_gates": gate.get("status"), "learning_rate": cfg.get("learning_rate"), "labels": "AI_REVIEWED data; not HUMAN_VERIFIED"}
    out.mkdir(parents=True, exist_ok=True)
    if a.records_only:
        print(json.dumps(records(out, meta)))
        return 0
    fpp = out / "run_fingerprint.json"
    if fpp.exists():
        old = json.loads(fpp.read_text(encoding="utf-8"))
        if old["sha256"] != fp["sha256"]:
            diff = {k: (old["fields"].get(k), v) for k, v in fp["fields"].items() if old["fields"].get(k) != v}
            print(f"RESUME REFUSED: this output_dir was started with another configuration or data: {diff}. Use a new RUN_NAME.")
            return 1
    else:
        fpp.write_text(json.dumps(fp, indent=1) + "\n", encoding="utf-8")
    ck = checkpoints(out)
    skipped = []
    while ck and missing_files(ck[-1]):
        skipped.append({"checkpoint": ck[-1].name, "missing": missing_files(ck[-1])})
        ck = ck[:-1]
    resume = None
    if ck:
        pf = preflight(ck[-1])
        if not pf["ok"]:
            print(f"RESUME REFUSED: checkpoint state inconsistent: {pf}")
            return 1
        resume = ck[-1]
        cfg["resume_from_checkpoint"] = str(resume)
        meta["resume_preflight"] = pf
        target = int(cfg.get("max_steps") or steps)
        ck_state = json.loads((resume / "trainer_state.json").read_text(encoding="utf-8"))
        if int(resume.name.split("-")[1]) >= target and ck_state.get("max_steps") == target:
            # already finished: a relaunch would resume at max_steps, train one step past it and overwrite the end-of-run records
            mp = out / "training_metrics.json"
            old = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
            meta.pop("resume_preflight", None)
            meta.update({k: old[k] for k in ("resumed_from", "trainer_began_at_step", "skipped_incomplete_checkpoints", "resume_preflight", "hardware") if k in old},
                        already_complete=resume.name, completion_check=pf, records_recomputed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            m = records(out, meta)
            print(f"ALREADY COMPLETE: {resume.name} is at max_steps {target}; nothing trained, records recomputed", flush=True)
            print(json.dumps({k: m.get(k) for k in ("global_step", "max_steps", "first_train_loss", "last_train_loss", "last_train_loss_source", "completed",
                                                    "already_complete")}))
            return 0
    now_gpu = gpu_info()
    if cfg.get("bf16") and now_gpu["gpu"] != "cpu" and now_gpu["bf16"] is False:
        print(f"RESUME REFUSED: the run uses bf16 and {now_gpu['gpu']} has no bf16")
        return 1
    meta.update(resumed_from=resume.name if resume else None, skipped_incomplete_checkpoints=skipped, hardware=now_gpu)
    (out / "resolved_train_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    if a.cpu_smoke:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    ev = PG.EventWriter(Path(a.progress_dir) if a.progress_dir else out / "progress_local", out / "progress")
    ev.emit("phase", phase="STARTING", run_fingerprint=fp["sha256"], resumed_from=resume.name if resume else None, gpu_info=now_gpu)
    print(f"training: resume from {resume.name if resume else 'scratch'}; {rows} rows -> {steps} steps; fingerprint {fp['sha256'][:12]}; {now_gpu['gpu']}", flush=True)
    from llamafactory.train.tuner import run_exp

    meta["example_printers_patched"] = quiet_examples()
    rec, prog = make_callbacks(out, a.stop_at_step, resume, ev)
    ev.emit("phase", phase="LOADING", note="tokenizer, datasets (tokenization) and model")
    try:
        run_exp(args=cfg, callbacks=[rec, prog])
    except BaseException as exc:  # noqa: BLE001 - reported, then re-raised
        ev.emit("error", type=type(exc).__name__, message=str(exc)[:500])
        ev.emit("phase", phase="FAILED")
        raise
    if resume is not None and rec.begin_step != int(resume.name.split("-")[1]):
        meta["resume_step_mismatch"] = {"checkpoint": resume.name, "trainer_began_at": rec.begin_step}
        records(out, meta)
        print(f"RESUME FAILED: trainer began at step {rec.begin_step}, checkpoint is {resume.name}")
        return 1
    meta["trainer_began_at_step"] = rec.begin_step
    m = records(out, meta)
    ev.emit("phase", phase="FINISHED", global_step=m.get("global_step"))
    ev.persist()
    print(json.dumps({k: m.get(k) for k in ("global_step", "max_steps", "first_train_loss", "last_train_loss", "last_train_loss_source", "completed", "resumed_from",
                                            "trainer_began_at_step")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
