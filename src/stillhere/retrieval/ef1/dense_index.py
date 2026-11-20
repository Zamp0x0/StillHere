"""EF1 dense index: one embedding row per raw-message chunk of the persona's own store, aligned with the EF1 evidence units.

Documents are built from the SAME SQLite store EF1 reads (``messages`` table), so a dense hit maps to a unit key without any guessing:
``record_id = message_id`` and the unit is the message's locator. The text is ``<来源> <日期> <说话人>：<内容>`` (date only: a synthetic
clock time would be noise). Matrices, cache and manifest reuse ``retrieval.dense`` (``build_index`` / ``load_index`` refuse another model,
dimension or snapshot). Only use an external embedding service for data that may leave the machine (e.g. the synthetic LifeBench
personas); a private persona's history must not be sent.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from stillhere.memory.labels import SOURCE_LABELS, speaker_label  # noqa: F401  (re-exported)
from stillhere.retrieval.dense import Document, EmbeddingCache, ScopeTable, build_index, chunks, load_index

INDEX_SCHEMA = "stillhere.retrieval.ef1.dense_index/1.0"


class ReadOnlyEmbeddingCache:
    """An existing embedding cache opened with ``mode=ro``: vectors can be reused, the file is never written (an older index's cache)."""

    def __init__(self, path: Path) -> None:
        import sqlite3
        import threading
        self._conn = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True, check_same_thread=False)
        self._lock = threading.Lock()

    def get(self, key: str):
        import numpy as np
        with self._lock:
            row = self._conn.execute("SELECT vector FROM vectors WHERE key = ?", (key,)).fetchone()
        return None if row is None else np.frombuffer(row[0], dtype=np.float32)

    def get_many(self, keys):
        import numpy as np
        out = {}
        with self._lock:
            for i in range(0, len(keys), 500):
                part = list(keys[i:i + 500])
                for k, blob in self._conn.execute(f"SELECT key, vector FROM vectors WHERE key IN ({','.join('?' * len(part))})", part):
                    out[k] = np.frombuffer(blob, dtype=np.float32)
        return out

    def put_many(self, items) -> None:                   # never writes
        return None

    def close(self) -> None:
        self._conn.close()


def message_documents(conn, persona_id: str, names: Mapping[str, str] = {}) -> List[Document]:
    docs = []
    for mid, content, speaker, source_type, ts in conn.execute(
            "SELECT message_id, content, speaker_id, source_type, timestamp_utc FROM messages WHERE target_persona_id = ? ORDER BY message_id",
            (persona_id,)):
        label = SOURCE_LABELS.get(source_type, source_type)
        docs.append(Document(mid, persona_id, "message", f"{label} {ts[:10]} {speaker_label(persona_id, speaker, names)}：{content}"))
    return docs


def unit_map(conn, persona_id: str) -> Dict[str, str]:
    return {mid: (loc or mid) for mid, loc in conn.execute("SELECT message_id, locator FROM messages WHERE target_persona_id = ?", (persona_id,))}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def combined_snapshot_sha(shas: Mapping[str, str]) -> str:
    return hashlib.sha256(json.dumps(dict(sorted(shas.items())), sort_keys=True).encode("utf-8")).hexdigest()


def texts_to_embed(documents: Mapping[str, Sequence[Document]]) -> List[str]:
    return [c for docs in documents.values() for d in docs for c in chunks(d.text)]


SIDECAR = "ef1_rows_meta.json"


def write_sidecar(index_dir: Path, apis: Mapping[str, object]) -> dict:
    """Per index row: the message's time, recorded time and unit (locator). The dense channel's time mask and unit mapping read ONLY this
    file, so a prebuilt index can serve a run whose store is still being filled day by day (no row is visible before its own time)."""
    index_dir = Path(index_dir)
    manifest = json.loads((index_dir / "embedding_index_manifest.json").read_text(encoding="utf-8"))
    meta = {"schema": INDEX_SCHEMA + "#rows", "index_sha256": manifest["index_sha256"], "personas": {}}
    for name in manifest["files"]:
        if not name.endswith(".ids.json"):
            continue
        pid = name[:-len(".ids.json")]
        rows = json.loads((index_dir / name).read_text(encoding="utf-8"))
        conn = apis[pid].store._conn
        info = {mid: (ts, rec, loc or mid) for mid, ts, rec, loc in conn.execute(
            "SELECT message_id, timestamp_utc, recorded_at_utc, locator FROM messages WHERE target_persona_id = ?", (pid,))}
        missing = [r for r in rows if r not in info]
        if missing:
            raise ValueError(f"{pid}: {len(missing)} index rows have no message in this store")
        meta["personas"][pid] = {"timestamp": [info[r][0] for r in rows], "recorded": [info[r][1] for r in rows], "unit": [info[r][2] for r in rows]}
    path = index_dir / SIDECAR
    path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return {"file": SIDECAR, "sha256": file_sha256(path)}


class SidecarScope:
    """Time mask from the sidecar: a row is visible iff its message was said by ``as_of`` and recorded by ``known_at`` (the store's rule
    for messages, ``sqlite_store.search_messages``)."""

    def __init__(self, meta: Mapping) -> None:
        import numpy as np
        from stillhere.retrieval.dense import _clock
        self.ts, self.rec, self.unit = {}, {}, {}
        for pid, m in meta["personas"].items():
            self.ts[pid] = np.array([_clock(x, 0) for x in m["timestamp"]], dtype=np.int64)
            self.rec[pid] = np.array([_clock(x, 0) for x in m["recorded"]], dtype=np.int64)
            self.unit[pid] = m["unit"]

    def mask(self, pid, as_of, known_at, *, include_ended=True):
        from stillhere.retrieval.dense import clock_of
        return (self.ts[pid] <= clock_of(as_of)) & (self.rec[pid] <= clock_of(known_at))


def open_dense(index_dir: Path, *, model: str, dimension: int, caches: Sequence = (), embed_query=None):
    """(DenseChannel, "OK") or (None, reason). The index must match model and dimension and its files their manifest hashes
    (``dense.load_index``); the sidecar must belong to this index."""
    import numpy as np
    from .channels import DenseChannel
    index_dir = Path(index_dir)
    try:
        manifest = json.loads((index_dir / "embedding_index_manifest.json").read_text(encoding="utf-8"))
        index = load_index(index_dir, model=model, dimension=dimension, snapshot_sha256=manifest["snapshot_sha256"])
        meta = json.loads((index_dir / SIDECAR).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, KeyError) as e:
        return None, f"DENSE_UNAVAILABLE: {type(e).__name__}: {e}"
    if meta.get("index_sha256") != manifest["index_sha256"]:
        return None, "DENSE_UNAVAILABLE: sidecar belongs to another index"
    scope = SidecarScope(meta)
    for pid in index.vectors:
        if len(scope.unit.get(pid, [])) != len(index.rows[pid]):
            return None, f"DENSE_UNAVAILABLE: sidecar rows of {pid} do not line up"
        index._scope_rows[pid] = np.arange(len(index.rows[pid]), dtype=np.int64)
    units = {pid: dict(zip(index.rows[pid], scope.unit[pid])) for pid in index.vectors}

    class _Caches:                                        # read the first cache that has the key; write to the first one
        def __init__(self, cs):
            self.cs = list(cs)

        def get(self, key):
            for c in self.cs:
                v = c.get(key)
                if v is not None:
                    return v
            return None

        def put_many(self, items):
            if self.cs:
                self.cs[0].put_many(items)

    return DenseChannel(index, scope, _Caches(caches) if caches else None, model, dimension, embed_query=embed_query, unit_of=units), "OK"
