"""Crash-safe JSONL handling: tolerant reads, dedupe by job id, atomic writes, merge of a local and a persistent copy."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def read_jsonl(path: Path) -> Tuple[List[dict], int]:
    """Valid rows and the number of broken lines (a half-written last line after a crash is skipped, never fatal)."""
    rows, bad = [], 0
    p = Path(path)
    if not p.exists():
        return rows, 0
    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(r, dict):
                rows.append(r)
            else:
                bad += 1
    return rows, bad


def dedupe(rows: Iterable[dict], key: str = "job_id") -> List[dict]:
    seen, out = set(), []
    for r in rows:
        k = r.get(key)
        if k is None or k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=1) + "\n")


def write_jsonl_atomic(path: Path, rows: Iterable[dict]) -> int:
    rows = list(rows)
    atomic_write_text(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return len(rows)


def merge_outputs(local: Path, persistent: Path, valid_ids: Iterable[str]) -> Dict[str, int]:
    """Union of the local working copy and the persistent (Drive) copy, valid JSON only, known job ids only, first occurrence wins.
    The merged set is written back to BOTH places atomically, so a resumed run never loses a row either side holds."""
    ok = set(valid_ids)
    a, bad_a = read_jsonl(local)
    b, bad_b = read_jsonl(persistent)
    merged = [r for r in dedupe(b + a) if r.get("job_id") in ok]
    unknown = len({r.get("job_id") for r in a + b} - ok - {None})
    write_jsonl_atomic(local, merged)
    if persistent != local:
        write_jsonl_atomic(persistent, merged)
    return {"rows": len(merged), "broken_lines": bad_a + bad_b, "unknown_ids_dropped": unknown}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha256_rows(rows: Iterable[dict]) -> str:
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: r["job_id"]):
        h.update(json.dumps(r, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return h.hexdigest()
