"""Train the D5 pilot adapter (fresh LoRA on Qwen3-8B @ b968826d; never initialised from D4) with LLaMA-Factory on one Colab A100 40GB.

Reuses the A100-proven D4 machinery unchanged (``t2d4.colab.train_d4``: data-manifest check, run fingerprint, newest complete checkpoint,
resume pre-flight, callbacks, records). Only the gates and the config keys are D5-specific.

    python -m d5pilot.train_d5 --data <pilot data> --config configs/d5_pilot_train.yaml --output-dir <Drive run>/train
    python -m d5pilot.train_d5 ... --smoke --dataset d5_smoke --max-steps 2 --save-steps 1 --output-dir <Drive run>/smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

D5_KEYS = ("d5_expected_train_rows", "d5_expected_steps", "d5_candidate_steps", "d5_run_id_suffix", "d5_train_file")


def resolve(config: dict, data: Path, out: Path, *, max_steps, save_steps, dataset) -> dict:
    c = {k: v for k, v in config.items() if k not in D5_KEYS}
    c.update(dataset_dir=str((data / "train").resolve()), output_dir=str(out.resolve()), overwrite_output_dir=False)
    if dataset:                                     # smoke: the longest pilot rows, no eval
        c["dataset"] = dataset
        c.pop("eval_dataset", None)
        c.update(eval_strategy="no")
    if max_steps:
        c["max_steps"] = max_steps
    if save_steps:
        c.update(save_steps=save_steps, eval_steps=save_steps)
    if max_steps or save_steps:
        c["logging_steps"] = 1
    return c


def gates(data: Path, lf_result: Path) -> dict:
    man = json.loads((data / "D5_PILOT_DATASET_MANIFEST.json").read_text(encoding="utf-8"))
    lf = json.loads(lf_result.read_text(encoding="utf-8")) if lf_result.is_file() else {}
    checks = {"pilot_manifest": man.get("schema") == "stillhere.d5.pilot_dataset_manifest/1.0",
              "static_gates_pass": all(str(v).startswith(("PASS", "0")) for v in man.get("gates", {}).values()),
              "llamafactory_labels_pass": lf.get("LLAMAFACTORY_LABELS") == "PASS"}
    failed = [k for k, v in checks.items() if not v]
    return {"checks": checks, "failed": failed, "training_allowed": not failed, "status": "PASS" if not failed else "BLOCKED"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--lf-result", help="LF_LABELS_RESULT.json from d5pilot.lf_labels (required for a formal run)")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--save-steps", type=int)
    ap.add_argument("--stop-at-step", type=int, help="stop (and save) at this step, as a disconnect would (resume test)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dataset", help="override the training dataset name (smoke only)")
    ap.add_argument("--progress-dir")
    a = ap.parse_args(argv)
    import yaml
    from t2d4.colab import progress as PG
    from t2d4.colab import train_d4 as T            # the D4 implementation, imported, not modified

    data, out = Path(a.data), Path(a.output_dir)
    man, bad = T.verify_data(data)
    if bad:
        print("DATA MANIFEST FAIL:\n  " + "\n  ".join(bad[:10]))
        return 1
    if man.get("contains_internal_holdout") or man.get("contains_full_pool"):
        print("REFUSED: this data package contains the internal holdout or the full pool")
        return 1
    raw = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    formal = not (a.smoke or a.max_steps)
    cfg = resolve(raw, data, out, max_steps=a.max_steps, save_steps=a.save_steps, dataset=a.dataset if a.smoke else None)
    info = json.loads((data / "train" / "dataset_info.json").read_text(encoding="utf-8"))
    train_file = data / "train" / info[cfg["dataset"]]["file_name"]
    rows = sum(1 for l in train_file.read_text(encoding="utf-8").splitlines() if l.strip())
    steps = T.expected_steps(rows, cfg)
    gate = {"status": "NOT_EVALUATED (smoke)"}
    if formal:
        gate = gates(data, Path(a.lf_result) if a.lf_result else Path("/nonexistent"))
        if not gate["training_allowed"]:
            print(f"TRAINING REFUSED: D5 gates failed: {gate['failed']}")
            return 3
        if rows != raw.get("d5_expected_train_rows") or steps != raw.get("d5_expected_steps"):
            print(f"CONFIG FAIL: data gives {rows} rows / {steps} steps, config expects {raw.get('d5_expected_train_rows')} / {raw.get('d5_expected_steps')}")
            return 1
    fp = T.fingerprint(cfg, T.sha256(data / "DATA_MANIFEST.json"))
    meta = {"model": "D5_PILOT_CANDIDATE", "init": "fresh LoRA on the base (no D4 weights)", "data_version": man.get("version"),
            "data_manifest_sha256": fp["fields"]["data_manifest_sha256"], "config": Path(a.config).name, "config_sha256": T.sha256(Path(a.config)),
            "run_fingerprint": fp["sha256"], "train_rows": rows, "train_file": train_file.name, "expected_steps": steps, "smoke": a.smoke,
            "model_revision": cfg.get("model_revision"), "is_formal_training": formal, "d5_gates": gate.get("status"),
            "candidate_steps": raw.get("d5_candidate_steps"), "learning_rate": cfg.get("learning_rate"), "cutoff_len": cfg.get("cutoff_len")}
    out.mkdir(parents=True, exist_ok=True)
    fpp = out / "run_fingerprint.json"
    if fpp.exists():
        old = json.loads(fpp.read_text(encoding="utf-8"))
        if old["sha256"] != fp["sha256"]:
            diff = {k: (old["fields"].get(k), v) for k, v in fp["fields"].items() if old["fields"].get(k) != v}
            print(f"RESUME REFUSED: this run directory was started with another configuration or data: {diff}. Use a new RUN_ID.")
            return 1
    else:
        fpp.write_text(json.dumps(fp, indent=1) + "\n", encoding="utf-8")
    ck = T.checkpoints(out)
    skipped = []
    while ck and T.missing_files(ck[-1]):
        skipped.append({"checkpoint": ck[-1].name, "missing": T.missing_files(ck[-1])})
        ck = ck[:-1]
    resume = None
    if ck:
        pf = T.preflight(ck[-1])
        if not pf["ok"]:
            print(f"RESUME REFUSED: checkpoint state inconsistent: {pf}")
            return 1
        resume = ck[-1]
        cfg["resume_from_checkpoint"] = str(resume)
        meta["resume_preflight"] = pf
        target = int(cfg.get("max_steps") or steps)
        st = json.loads((resume / "trainer_state.json").read_text(encoding="utf-8"))
        if int(resume.name.split("-")[1]) >= target and st.get("max_steps") == target:
            meta.update(already_complete=resume.name, records_recomputed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            m = T.records(out, meta)
            print(f"ALREADY COMPLETE: {resume.name} is at max_steps {target}; nothing trained")
            print(json.dumps({k: m.get(k) for k in ("global_step", "max_steps", "last_train_loss", "completed")}))
            return 0
    meta.update(resumed_from=resume.name if resume else None, skipped_incomplete_checkpoints=skipped, hardware=T.gpu_info())
    (out / "resolved_train_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    ev = PG.EventWriter(Path(a.progress_dir) if a.progress_dir else out / "progress_local", out / "progress")
    ev.emit("phase", phase="STARTING", run_fingerprint=fp["sha256"], resumed_from=resume.name if resume else None, gpu_info=meta["hardware"])
    print(f"D5 training: resume from {resume.name if resume else 'scratch'}; {rows} rows -> {steps} steps; fingerprint {fp['sha256'][:12]}; "
          f"{meta['hardware']['gpu']}", flush=True)
    from llamafactory.train.tuner import run_exp
    meta["example_printers_patched"] = T.quiet_examples()
    rec, prog = T.make_callbacks(out, a.stop_at_step, resume, ev)
    try:
        run_exp(args=cfg, callbacks=[rec, prog])
    except BaseException as exc:  # noqa: BLE001
        ev.emit("error", type=type(exc).__name__, message=str(exc)[:500])
        ev.emit("phase", phase="FAILED")
        raise
    if resume is not None and rec.begin_step != int(resume.name.split("-")[1]):
        meta["resume_step_mismatch"] = {"checkpoint": resume.name, "trainer_began_at": rec.begin_step}
        T.records(out, meta)
        print(f"RESUME FAILED: trainer began at step {rec.begin_step}, checkpoint is {resume.name}")
        return 1
    meta["trainer_began_at_step"] = rec.begin_step
    m = T.records(out, meta)
    ev.emit("phase", phase="FINISHED", global_step=m.get("global_step"))
    ev.persist()
    print(json.dumps({k: m.get(k) for k in ("global_step", "max_steps", "first_train_loss", "last_train_loss", "eval_losses", "gpu_mem_peak_gb",
                                            "completed", "resumed_from", "trainer_began_at_step")}))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
