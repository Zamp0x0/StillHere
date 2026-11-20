"""Results ZIP: every raw record and report of a run (snapshots excluded: they are rebuildable state, not results), with SHA256SUMS."""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

EXCLUDE_DIRS = {"snapshots"}


def build(run_dir: Path, out_zip: Path) -> dict:
    run_dir, out_zip = Path(run_dir), Path(out_zip)
    files = sorted(p for p in run_dir.rglob("*") if p.is_file() and not (set(p.relative_to(run_dir).parts) & EXCLUDE_DIRS)
                   and not p.name.endswith(".partial"))
    sums = []
    tmp = out_zip.with_suffix(".zip.partial")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            rel = p.relative_to(run_dir).as_posix()
            b = p.read_bytes()
            sums.append(f"{hashlib.sha256(b).hexdigest()}  {rel}")
            z.writestr(rel, b)
        z.writestr("SHA256SUMS.txt", "\n".join(sums) + "\n")
    tmp.replace(out_zip)
    return {"zip": str(out_zip), "files": len(files), "bytes": out_zip.stat().st_size, "sha256": hashlib.sha256(out_zip.read_bytes()).hexdigest()}
