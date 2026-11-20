"""Export the selected persona checkpoint as a PERSONA PACKAGE ZIP (adapter + prompt + card + manifests; no base weights), and the run's
results (no weights) as a results ZIP. Both go to ``<run>/exports/`` and are never overwritten.

    python -m personatrain.export adapter --run-dir R --train-dir R/train --checkpoint checkpoint-N --run-id ID --data D [--unqualified]
    python -m personatrain.export results --run-dir R --run-id ID

The adapter may have memorised private training text: it is a PRIVATE derived asset (never published; ``stillhere persona import-adapter``
verifies it on the user's machine).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")
SCHEMA = "stillhere.persona.adapter_manifest/1.0"
SECRET = re.compile(r"(sk-[A-Za-z0-9]{16,}|LLM_API_KEY\s*=\s*\S+|api[_-]?key\"?\s*[:=]\s*\"[A-Za-z0-9]{12,}|hf_[A-Za-z0-9]{20,})")
ROOT = "persona_adapter"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _json(p: Path, default=None):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else default


def adapter(run: Path, train_dir: Path, ck: str, run_id: str, data: Path, unqualified: bool = False) -> Path:
    src = train_dir / ck
    missing = [f for f in ADAPTER_FILES if not (src / f).is_file()]
    if missing:
        raise SystemExit(f"REFUSED: {src} lacks {missing}")
    selection = _json(run / "eval" / "dev" / "CHECKPOINT_SELECTION.json", {"status": "NOT_RUN"})
    chosen = selection.get("selected")
    is_sel = chosen == f"ckpt-{ck.split('-')[1]}" and selection.get("status") == "SELECTED"
    if not is_sel and not unqualified:
        raise SystemExit(f"REFUSED: {ck} is not the pre-registered DEV selection ({selection.get('status')}: {chosen}); --unqualified exports it for diagnosis only")
    pm = _json(data / "PERSONA_TRAINING_MANIFEST.json")
    tag = "candidate" if is_sel else "UNQUALIFIED_diagnostic"
    out = run / "exports" / f"persona_{tag}_{pm['profile_id']}_{run_id}_{ck}.zip"
    if out.exists():
        raise SystemExit(f"REFUSED: {out} exists (never overwritten)")
    out.parent.mkdir(parents=True, exist_ok=True)
    fp = _json(train_dir / "run_fingerprint.json")
    tm = _json(train_dir / "training_metrics.json", {})
    acfg = _json(src / "adapter_config.json")
    files = {f"adapter/{f}": sha(src / f) for f in ADAPTER_FILES}
    prompt = (data / "persona" / "persona_prompt.txt").read_bytes()
    card = (data / "persona" / "persona_card.yaml").read_bytes()
    files["persona_prompt.txt"] = hashlib.sha256(prompt).hexdigest()
    files["persona_card.yaml"] = hashlib.sha256(card).hexdigest()
    score = _json(run / "eval" / "dev" / "SCORE.json", {})
    summary = {"split": "dev", "selection": selection, "groups": score.get("groups"), "paired": score.get("paired"), "status": score.get("status"),
               "note": "DEV only (checkpoint selection); the sealed TEST is evaluated later on the user's machine, once"}
    sb = json.dumps(summary, ensure_ascii=False, indent=1).encode("utf-8")
    files["evaluation_summary.json"] = hashlib.sha256(sb).hexdigest()
    man = {"schema": SCHEMA, "name": "PERSONA_ADAPTER_CANDIDATE" if is_sel else "PERSONA_ADAPTER_UNQUALIFIED_DIAGNOSTIC", "profile_id": pm["profile_id"],
           "dataset_version": pm["dataset_version"], "card_id": pm["card_id"], "persona_prompt_sha256": pm["persona_prompt_sha256"],
           "training_system_prompt_sha256": pm["training_system_prompt_sha256"], "training_system_prompt_version": pm["training_system_prompt_version"],
           "run_id": run_id, "checkpoint": ck, "selection_status": selection.get("status"),
           "base": {"id": "Qwen/Qwen3-8B", "revision": pm["base_revision"], "tokenizer_sha256": pm["tokenizer_sha256"], "chat_template_sha256": pm["chat_template_sha256"]},
           "quantization": "4-bit NF4, double quantization, bf16 compute (train and inference)", "run_fingerprint": fp["sha256"],
           "data_manifest_sha256": fp["fields"].get("data_manifest_sha256"), "code_tree_sha256": pm.get("code_tree_sha256"),
           "adapter_config": {k: acfg.get(k) for k in ("peft_type", "r", "lora_alpha", "lora_dropout", "target_modules", "base_model_name_or_path", "task_type")},
           "training": {k: tm.get(k) for k in ("global_step", "max_steps", "epoch", "first_train_loss", "last_train_loss", "eval_losses", "gpu_mem_peak_gb",
                                                "median_seconds_per_step", "gpus_used", "completed", "resumed_from")},
           "files": files, "usage_scope": "PRIVATE derived asset (may memorise private chat); never publish", "exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    mb = json.dumps(man, ensure_ascii=False, indent=1).encode("utf-8")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ADAPTER_FILES:
            z.write(src / f, f"{ROOT}/adapter/{f}")
        z.writestr(f"{ROOT}/persona_prompt.txt", prompt)
        z.writestr(f"{ROOT}/persona_card.yaml", card)
        z.writestr(f"{ROOT}/evaluation_summary.json", sb)
        z.writestr(f"{ROOT}/persona_manifest.json", mb)
        z.writestr(f"{ROOT}/SHA256SUMS.txt", "".join(f"{h}  {f}\n" for f, h in sorted(files.items())) + f"{hashlib.sha256(mb).hexdigest()}  persona_manifest.json\n")
    (out.parent / (out.name + ".sha256")).write_text(sha(out) + "\n", encoding="utf-8")
    return out


def results(run: Path, run_id: str) -> Path:
    out = run / "exports" / f"persona_results_{run_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(run.rglob("*")):
            rel = p.relative_to(run)
            if not p.is_file() or rel.parts[0] in ("exports", "progress_local") or any(x.startswith("checkpoint-") for x in rel.parts):
                continue
            if p.suffix in (".safetensors", ".pt", ".pth", ".bin") or p.name.endswith(".env"):
                continue
            if p.stat().st_size < 5_000_000 and SECRET.search(p.read_text(encoding="utf-8", errors="ignore")):
                raise SystemExit(f"REFUSED: possible secret in {rel}")
            z.write(p, f"persona_results/{rel.as_posix()}")
            n += 1
    print(f"{n} files -> {out}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["adapter", "results"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--train-dir")
    ap.add_argument("--checkpoint")
    ap.add_argument("--data")
    ap.add_argument("--unqualified", action="store_true")
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    p = adapter(run, Path(a.train_dir) if a.train_dir else run / "train", a.checkpoint, a.run_id, Path(a.data), a.unqualified) if a.cmd == "adapter" else results(run, a.run_id)
    print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
