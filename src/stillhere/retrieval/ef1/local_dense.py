"""Local dense channel for a private persona archive: Qwen3-Embedding-0.6B on the CPU, nothing leaves the machine.

Why windows: in a WeFlow archive one EF1 unit is ONE message and messages are very short (median 4 characters in the real archive),
so a single message carries almost no meaning for an embedding. Each document is therefore a short window of the same conversation
(``before`` messages, the message, ``after`` messages), dated, with the owner-frame speaker labels; ``record_id`` stays the CENTRE
message id, so a dense hit maps to exactly the same unit key EF1's lexical channels use (no guessing).

Build: resumable (vectors go to an ``EmbeddingCache`` keyed by model / dimension / text; a stopped build continues), bounded batch,
bounded threads, one model instance. The finished index reuses ``retrieval.dense.build_index`` + the EF1 sidecar, so ``open_dense``
refuses a different model, dimension, snapshot or changed file.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional

from stillhere.memory.labels import speaker_label
from stillhere.retrieval.dense import Document, EmbeddingCache, build_index, chunks

from . import dense_index as DI

LOCAL_SCHEMA = "stillhere.retrieval.ef1.local_dense/1.0"


def window_documents(conn, persona_id: str, names: Mapping[str, str] = {}, *, before: int = 2, after: int = 2,
                     max_chars: int = 400) -> List[Document]:
    """One document per message: ``<date> <label>：<text>`` lines of its conversation window, oldest first; the centre line is marked
    with ``▶``. Long windows are trimmed symmetrically to ``max_chars`` (the centre line is always kept)."""
    by_conv: Dict[str, list] = {}
    for mid, conv, speaker, content, ts in conn.execute(
            "SELECT message_id, conversation_id, speaker_id, content, timestamp_utc FROM messages WHERE target_persona_id = ? "
            "ORDER BY conversation_id, timestamp_utc, message_id", (persona_id,)):
        by_conv.setdefault(conv or mid, []).append((mid, speaker, content or "", ts))
    docs: List[Document] = []
    for conv, rows in by_conv.items():
        lines = [f"{speaker_label(persona_id, sp, names)}：{c}" for _, sp, c, _ in rows]
        for i, (mid, _, _, ts) in enumerate(rows):
            lo, hi = max(0, i - before), min(len(rows), i + after + 1)
            win = [("▶" if j == i else "") + lines[j] for j in range(lo, hi)]
            while sum(len(x) for x in win) > max_chars and len(win) > 1:
                centre = i - lo
                if centre >= len(win) - 1 - centre and centre > 0:
                    win.pop(0)
                    lo += 1
                else:
                    win.pop()
            docs.append(Document(mid, persona_id, "message", f"{ts[:10]}\n" + "\n".join(win)))
    docs.sort(key=lambda d: d.record_id)
    return docs


def snapshot_sha(conn, persona_id: str) -> str:
    import hashlib
    arch = [[r[0], r[1]] for r in conn.execute("SELECT message_id, content_sha256 FROM messages WHERE target_persona_id = ? ORDER BY message_id",
                                               (persona_id,))]
    return hashlib.sha256(json.dumps(arch).encode("utf-8")).hexdigest()        # = colab_eval.memory_snapshot()["archive_sha256"]


def build_local(api, persona_id: str, index_dir: Path, embedder, *, names: Mapping[str, str] = {}, model: str, dimension: int,
                before: int = 2, after: int = 2, batch_size: int = 16, time_budget_s: Optional[float] = None,
                progress: Optional[Callable[[dict], None]] = None) -> dict:
    """Embed what is missing, then (when nothing is missing) write the index + sidecar. Returns ``status`` COMPLETE or PARTIAL."""
    from stillhere.retrieval.local_embedding import embed_documents_local
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    conn = api.store._conn
    t0 = time.perf_counter()
    docs = window_documents(conn, persona_id, names, before=before, after=after)
    texts = [c for d in docs for c in chunks(d.text)]
    cache = EmbeddingCache(index_dir / "embedding_cache.sqlite")
    try:
        emb = embed_documents_local(embedder, cache, texts, model=model, dimension=dimension, batch_size=batch_size,
                                    time_budget_s=time_budget_s, progress=progress)
        if emb["stopped_by_budget"] or emb["embedded"] < emb["todo"]:
            return {"status": "PARTIAL", "documents": len(docs), "embedding": emb, "seconds": round(time.perf_counter() - t0, 1)}
        snap = snapshot_sha(conn, persona_id)
        idx = build_index(index_dir / "index", cache, {persona_id: docs}, model=model, dimension=dimension, dataset_revision=LOCAL_SCHEMA,
                          snapshot_sha256=snap, config={"document": "conversation window", "before": before, "after": after,
                                                        "max_chars": 400, "query_task": getattr(embedder, "query_task", None)})
        side = DI.write_sidecar(index_dir / "index", {persona_id: api})
    finally:
        cache.close()
    return {"status": "COMPLETE", "documents": len(docs), "chunks": idx.manifest["chunk_count"], "index_sha256": idx.manifest["index_sha256"],
            "snapshot_sha256": snap, "sidecar": side, "embedding": emb, "seconds": round(time.perf_counter() - t0, 1)}


def open_local(index_dir: Path, *, model: str, dimension: int, query_encoder=None, query_task: Optional[str] = None):
    """(DenseChannel, "OK") or (None, reason) for a finished local index. ``query_encoder``: e.g. ``LocalQueryEncoder(get_embedder(dir))``.
    Query vectors go to ``query_cache_<instruction hash>.sqlite``: the channel's cache key holds only the raw question, so a vector made
    with another instruction must never be reused (it was before 2026-11-18: one shared file). Document vectors are never touched."""
    from stillhere.retrieval.local_embedding import QUERY_TASK, task_tag
    index_dir = Path(index_dir)
    task = query_task if query_task is not None else getattr(query_encoder, "query_task", QUERY_TASK)
    cache = EmbeddingCache(index_dir / f"query_cache_{task_tag(task)}.sqlite")
    return DI.open_dense(index_dir / "index", model=model, dimension=dimension, caches=[cache], embed_query=query_encoder)
