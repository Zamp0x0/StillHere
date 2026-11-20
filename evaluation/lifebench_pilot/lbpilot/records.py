"""Append-only JSONL result stores keyed by a tuple; a finished key is never requested (or paid for) again.

Each line is written and fsync'ed before the next request starts; a torn last line (a crash mid-write) is ignored on reading, so a
resumed run simply repeats that one request. ``final`` decides which records count as finished (e.g. status OK).
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Tuple


class Store:
    def __init__(self, path: Path, key_fields: Tuple[str, ...], final: Callable[[Mapping], bool] = lambda r: r.get("status") == "OK"):
        self.path, self.key_fields, self.final = Path(path), key_fields, final
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def key(self, row: Mapping):
        return tuple(row[k] for k in self.key_fields)

    def rows(self) -> Iterable[dict]:
        if not self.path.is_file():
            return []
        out = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue                      # a torn last line
        return out

    def latest(self) -> Dict[tuple, dict]:
        """The last record per key (a later retry supersedes an earlier failure)."""
        out: Dict[tuple, dict] = {}
        for r in self.rows():
            out[self.key(r)] = r
        return out

    def done(self) -> Dict[tuple, dict]:
        return {k: r for k, r in self.latest().items() if self.final(r)}

    def attempts(self) -> Dict[tuple, int]:
        out: Dict[tuple, int] = {}
        for r in self.rows():
            out[self.key(r)] = out.get(self.key(r), 0) + 1
        return out

    def append(self, row: Mapping) -> None:
        line = json.dumps(dict(row), ensure_ascii=False) + "\n"
        with self.lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())


def write_json_atomic(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)
