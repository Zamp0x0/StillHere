"""Dense index: exact search over one NumPy matrix per persona, a derived index that can always be rebuilt.

SQLite stays the authority. ``ScopeTable`` reads each record's times once and answers the same question as the store's
``_believed`` rule, as a mask, so Top-K is chosen only among records this persona may see at that time.
Embedding text holds only what a record states: its own text, its type, its time and speaker. No question or gold.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from stillhere.memory.sqlite_store import _utc

from .alibaba_client import BATCH_LIMITS, DashScopeClient, estimate_tokens

MAX_CHUNK_CHARS = 1200
CHUNK_OVERLAP = 100
NO_LIMIT = np.iinfo(np.int64).max
NO_START = np.iinfo(np.int64).min
MESSAGE_LABELS = {"lifebench_sms": "短信", "lifebench_agent_chat": "AI助手对话"}


_EPOCH = datetime(1970, 1, 1)


def _clock(text: Optional[str], default: int) -> int:
    """Fixed-width UTC text as microseconds from the epoch, so every date of any year keeps its order in int64."""
    return default if text is None else (datetime.fromisoformat(text[:-1]) - _EPOCH) // timedelta(microseconds=1)


def clock_of(when: datetime) -> int:
    return _clock(_utc(when), 0)


@dataclass(frozen=True)
class Document:
    record_id: str
    persona_id: str
    kind: str          # "memory" or "message"
    text: str


def corpus(conn: sqlite3.Connection, persona_id: str, speakers: Mapping[str, Mapping] = {}) -> List[Document]:
    """Everything searchable for a persona, in a fixed order, not a chosen subset."""
    docs = [Document(r[0], persona_id, "memory", r[1])
            for r in conn.execute("SELECT memory_id, statement FROM memories WHERE persona_id = ? ORDER BY memory_id", (persona_id,))]
    for message_id, content, speaker, source_type, timestamp in conn.execute(
            "SELECT message_id, content, speaker_id, source_type, timestamp FROM messages WHERE target_persona_id = ? "
            "ORDER BY message_id", (persona_id,)):
        name = (speakers.get(speaker) or {}).get("name") or speaker
        label = MESSAGE_LABELS.get(source_type, source_type)
        docs.append(Document(message_id, persona_id, "message", f"{label} {timestamp[:16].replace('T', ' ')} {name}：{content}"))
    return docs


def chunks(text: str, *, size: int = MAX_CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Fixed windows with overlap. Short texts stay whole, so most records are one chunk."""
    if len(text) <= size:
        return [text]
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text) - overlap, step)]


class ScopeTable:
    """Per-record times of the snapshot. ``mask`` is the rule of the store's temporal reads."""

    def __init__(self, conn: sqlite3.Connection, persona_ids: Iterable[str]) -> None:
        self.tables: Dict[str, dict] = {}
        for pid in persona_ids:
            rows = [(r[0], _clock(r[1], 0), _clock(r[2], NO_LIMIT), _clock(r[3], NO_START), _clock(r[4], NO_LIMIT)) for r in conn.execute(
                "SELECT memory_id, recorded_at_utc, status_changed_at, valid_from_utc, valid_to_utc FROM memories WHERE persona_id = ?", (pid,))]
            rows += [(r[0], _clock(r[1], 0), NO_LIMIT, _clock(r[2], NO_START), NO_LIMIT) for r in conn.execute(
                "SELECT message_id, recorded_at_utc, timestamp_utc FROM messages WHERE target_persona_id = ?", (pid,))]
            self.tables[pid] = {"row": {r[0]: i for i, r in enumerate(rows)},
                                "recorded": np.array([r[1] for r in rows], dtype=np.int64),
                                "changed": np.array([r[2] for r in rows], dtype=np.int64),
                                "start": np.array([r[3] for r in rows], dtype=np.int64),
                                "end": np.array([r[4] for r in rows], dtype=np.int64)}

    def mask(self, persona_id: str, as_of: datetime, known_at: datetime, *, include_ended: bool) -> np.ndarray:
        t = self.tables[persona_id]
        known, when = clock_of(known_at), clock_of(as_of)
        visible = (t["recorded"] <= known) & (t["changed"] > known) & (t["start"] <= when)
        return visible if include_ended else visible & (t["end"] > when)

    def visible(self, persona_id: str, record_id: str, mask: np.ndarray) -> bool:
        row = self.tables.get(persona_id, {"row": {}})["row"].get(record_id)
        return row is not None and bool(mask[row])


class EmbeddingCache:
    """Vectors by content. Interrupted builds resume from here, and the same text is never paid for twice."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")
        self._lock = threading.Lock()

    @staticmethod
    def key(model: str, dimension: int, text_type: str, text: str) -> str:
        return hashlib.sha256(f"{model}\x1f{dimension}\x1f{text_type}\x1f{text}".encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[np.ndarray]:
        with self._lock:
            row = self._conn.execute("SELECT vector FROM vectors WHERE key = ?", (key,)).fetchone()
        return None if row is None else np.frombuffer(row[0], dtype=np.float32)

    def get_many(self, keys: Sequence[str]) -> Dict[str, np.ndarray]:
        found: Dict[str, np.ndarray] = {}
        with self._lock:
            for start in range(0, len(keys), 500):
                part = keys[start:start + 500]
                marks = ",".join("?" * len(part))
                for key, blob in self._conn.execute(f"SELECT key, vector FROM vectors WHERE key IN ({marks})", part):
                    found[key] = np.frombuffer(blob, dtype=np.float32)
        return found

    def put_many(self, items: Sequence[Tuple[str, np.ndarray]]) -> None:
        with self._lock:
            self._conn.executemany("INSERT OR IGNORE INTO vectors (key, vector) VALUES (?, ?)",
                                   [(k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in items])
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def missing_texts(cache: EmbeddingCache, model: str, dimension: int, texts: Sequence[str]) -> List[str]:
    unique = list(dict.fromkeys(texts))
    keys = {t: EmbeddingCache.key(model, dimension, "document", t) for t in unique}
    have = cache.get_many(list(keys.values()))
    return [t for t in unique if keys[t] not in have]


def embed_documents(client: DashScopeClient, cache: EmbeddingCache, model: str, dimension: int, texts: Sequence[str], *,
                    workers: int = 3, chars_per_minute: int = 700_000) -> dict:
    """Embed what the cache lacks, batch by batch. Stopping at any point loses nothing already paid for."""
    todo = missing_texts(cache, model, dimension, texts)
    size = BATCH_LIMITS[model]
    batches = [todo[i:i + size] for i in range(0, len(todo), size)]
    state = {"tokens": 0, "next": time.monotonic()}
    lock = threading.Lock()

    def work(batch: List[str]) -> None:
        with lock:                                  # stay under the published tokens-per-minute limit
            wait = state["next"] - time.monotonic()
            state["next"] = max(state["next"], time.monotonic()) + sum(len(t) for t in batch) * 60 / chars_per_minute
        if wait > 0:
            time.sleep(wait)
        result = client.embed(model, batch, text_type="document", dimension=dimension)
        cache.put_many([(EmbeddingCache.key(model, dimension, "document", t), normalized(v)) for t, v in zip(batch, result.vectors)])
        with lock:
            state["tokens"] += result.tokens

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, batches))
    return {"texts_total": len(set(texts)), "texts_embedded": len(todo), "requests": len(batches), "tokens": state["tokens"],
            "seconds": round(time.perf_counter() - started, 1)}


def normalized(vector: Sequence[float]) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32)
    length = float(np.linalg.norm(v))
    if not np.isfinite(length) or length == 0.0:
        raise ValueError("an embedding has no usable length")
    return v / length


class DenseIndex:
    """Rows are chunks; several chunks may share a record. Similarity is the dot product of unit vectors (cosine)."""

    def __init__(self, directory: Path, persona_vectors: Dict[str, np.ndarray], persona_rows: Dict[str, List[str]],
                 manifest: dict) -> None:
        self.directory = directory
        self.vectors = persona_vectors
        self.rows = persona_rows
        self.manifest = manifest
        self._scope_rows: Dict[str, np.ndarray] = {}

    def bind(self, scope: ScopeTable) -> None:
        for pid, rows in self.rows.items():
            table = scope.tables[pid]["row"]
            self._scope_rows[pid] = np.array([table[r] for r in rows], dtype=np.int64)

    def search(self, persona_id: str, query: np.ndarray, mask: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """The best ``k`` records among those ``mask`` allows. Filtering comes first, so nothing valid is pushed out."""
        allowed = mask[self._scope_rows[persona_id]]
        scores = self.vectors[persona_id] @ query
        rows = np.flatnonzero(allowed)
        if rows.size == 0:
            return []
        order = rows[np.lexsort((rows, -scores[rows]))]
        out: Dict[str, float] = {}
        records = self.rows[persona_id]
        for row in order:
            record = records[int(row)]
            if record not in out:
                out[record] = float(scores[row])
                if len(out) == k:
                    break
        return list(out.items())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_index(directory: Path, cache: EmbeddingCache, documents: Mapping[str, Sequence[Document]], *, model: str, dimension: int,
                dataset_revision: str, snapshot_sha256: str, config: Mapping) -> DenseIndex:
    """Assemble the matrices from the cache. Every chunk must already be embedded."""
    directory.mkdir(parents=True, exist_ok=True)
    vectors: Dict[str, np.ndarray] = {}
    rows: Dict[str, List[str]] = {}
    files: Dict[str, str] = {}
    content = hashlib.sha256()
    counts = {"records": 0, "chunks": 0}
    for pid, docs in documents.items():
        persona_rows: List[str] = []
        persona_keys: List[str] = []
        for doc in docs:
            for chunk in chunks(doc.text):
                persona_rows.append(doc.record_id)
                persona_keys.append(EmbeddingCache.key(model, dimension, "document", chunk))
                content.update(chunk.encode("utf-8"))
        have = cache.get_many(list(dict.fromkeys(persona_keys)))
        absent = [k for k in persona_keys if k not in have]
        if absent:
            raise ValueError(f"{pid}: {len(absent)} chunks are not embedded")
        matrix = np.stack([have[k] for k in persona_keys]).astype(np.float32)
        np.save(directory / f"{pid}.npy", matrix)
        (directory / f"{pid}.ids.json").write_text(json.dumps(persona_rows), encoding="utf-8")
        files[f"{pid}.npy"] = _sha(directory / f"{pid}.npy")
        files[f"{pid}.ids.json"] = _sha(directory / f"{pid}.ids.json")
        vectors[pid], rows[pid] = matrix, persona_rows
        counts["records"] += len(docs)
        counts["chunks"] += len(persona_rows)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()
    manifest = {"model_id": model, "dimension": dimension, "dataset_revision": dataset_revision, "snapshot_sha256": snapshot_sha256,
                "content_hash": content.hexdigest(), "document_count": counts["records"], "chunk_count": counts["chunks"],
                "embedding_config": dict(config), "files": files, "index_sha256": digest, "similarity": "cosine, unit vectors"}
    (directory / "embedding_index_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return DenseIndex(directory, vectors, rows, manifest)


def load_index(directory: Path, *, model: str, dimension: int, snapshot_sha256: str) -> DenseIndex:
    """Refuse an index built from another model, dimension or snapshot, or one that changed on disk."""
    manifest = json.loads((directory / "embedding_index_manifest.json").read_text(encoding="utf-8"))
    for field, expected in (("model_id", model), ("dimension", dimension), ("snapshot_sha256", snapshot_sha256)):
        if manifest[field] != expected:
            raise ValueError(f"index {field} is {manifest[field]}, expected {expected}")
    for name, digest in manifest["files"].items():
        if _sha(directory / name) != digest:
            raise ValueError(f"{name} changed since the index was built")
    vectors, rows = {}, {}
    for name in manifest["files"]:
        if name.endswith(".npy"):
            pid = name[:-4]
            vectors[pid] = np.load(directory / name)
            rows[pid] = json.loads((directory / f"{pid}.ids.json").read_text(encoding="utf-8"))
            if len(rows[pid]) != vectors[pid].shape[0] or vectors[pid].shape[1] != dimension:
                raise ValueError(f"{pid}: vectors and ids do not line up")
    return DenseIndex(directory, vectors, rows, manifest)
