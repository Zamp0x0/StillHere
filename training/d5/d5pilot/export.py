"""Export the selected D5 pilot checkpoint as an adapter ZIP, and the run's results (no weights) as a results ZIP.

    python -m d5pilot.export adapter --run-dir R --checkpoint checkpoint-125 --run-id ID
    python -m d5pilot.export results --run-dir R --run-id ID
Both write into ``R/exports/`` only; existing files are never overwritten (a new name or RUN_ID is required).
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
SECRET = re.compile(r"(sk-[A-Za-z0-9]{16,}|LLM_API_KEY\s*=\s*\S+|api[_-]?key\"?\s*[:=]\s*\"[A-Za-z0-9]{12,})")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def adapter(run: Path, ck: str, run_id: str, train_dir: Path = None, unqualified: bool = False) -> Path:
    train_dir = Path(train_dir) if train_dir else run / "train"
    src = train_dir / ck
    sel = run / "eval" / "CHECKPOINT_SELECTION.json"
    selection = json.loads(sel.read_text(encoding="utf-8")) if sel.is_file() else {"status": "NOT_RUN"}
    chosen = selection.get("selected")
    if chosen != f"ckpt-{ck.split('-')[1]}" and not unqualified:
        raise SystemExit(f"REFUSED: {ck} is not the pre-registered selection ({chosen}); use --unqualified to export it for diagnosis only")
    tag = "candidate" if chosen == f"ckpt-{ck.split('-')[1]}" else "UNQUALIFIED_diagnostic"
    out = run / "exports" / f"d5_pilot_{tag}_adapter_{run_id}_{ck}.zip"
    if out.exists():
        raise SystemExit(f"REFUSED: {out} exists (never overwritten)")
    out.parent.mkdir(parents=True, exist_ok=True)
    fp = json.loads((train_dir / "run_fingerprint.json").read_text(encoding="utf-8"))
    tm = json.loads((train_dir / "training_metrics.json").read_text(encoding="utf-8")) if (train_dir / "training_metrics.json").is_file() else {}
    man = {"name": "D5_PILOT_CANDIDATE" if tag == "candidate" else "D5_PILOT_UNQUALIFIED_DIAGNOSTIC", "not": "not D5 final, not for release", "run_id": run_id, "checkpoint": ck, "selection": selection,
           "base": {"id": "Qwen/Qwen3-8B", "revision": fp["fields"].get("model_revision")}, "run_fingerprint": fp["sha256"],
           "data_manifest_sha256": fp["fields"].get("data_manifest_sha256"), "training": {k: tm.get(k) for k in (
               "global_step", "max_steps", "epoch", "first_train_loss", "last_train_loss", "eval_losses", "gpu_mem_peak_gb", "median_seconds_per_step")},
           "files": {f: sha(src / f) for f in ADAPTER_FILES}, "usage_scope": "PRIVATE_NONCOMMERCIAL_RESEARCH (CC BY-NC sources); do not publish",
           "exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ADAPTER_FILES:
            z.write(src / f, f"d5_pilot_adapter/{f}")
        z.writestr("d5_pilot_adapter/D5_ADAPTER_MANIFEST.json", json.dumps(man, ensure_ascii=False, indent=1))
        z.writestr("d5_pilot_adapter/SHA256SUMS.txt", "".join(f"{h}  {f}\n" for f, h in man["files"].items()))
    (out.parent / (out.name + ".sha256")).write_text(sha(out) + "\n", encoding="utf-8")
    return out


def results(run: Path, run_id: str) -> Path:
    out = run / "exports" / f"d5_pilot_results_{run_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    skip_dirs = {"exports", "progress_local"}
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(run.rglob("*")):
            rel = p.relative_to(run)
            if not p.is_file() or rel.parts[0] in skip_dirs or any(x.startswith("checkpoint-") for x in rel.parts):
                continue
            if p.suffix in (".safetensors", ".pt", ".pth", ".bin") or p.name.endswith(".env"):
                continue
            if p.stat().st_size < 5_000_000 and SECRET.search(p.read_text(encoding="utf-8", errors="ignore")):
                raise SystemExit(f"REFUSED: possible secret in {rel}")
            z.write(p, f"d5_pilot_results/{rel.as_posix()}")
            n += 1
    print(f"{n} files -> {out}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["adapter", "results"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--train-dir")
    ap.add_argument("--unqualified", action="store_true")
    a = ap.parse_args(argv)
    p = adapter(Path(a.run_dir), a.checkpoint, a.run_id, a.train_dir, a.unqualified) if a.cmd == "adapter" else results(Path(a.run_dir), a.run_id)
    print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
