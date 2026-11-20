"""The official Mem0 OSS Python SDK (``mem0ai``), used as Mem0 itself works: ``Memory.add(infer=True)`` (LLM fact extraction + embedding
+ vector store) and ``Memory.search``. Nothing in here extracts, embeds, stores or ranks a memory itself.

Rendering and batching follow LifeBench_eval's own Mem0 adapter (``src/adapters/mem0_adapter.py`` at 2bfda547): each turn becomes
``(Conversation date: <Month DD, YYYY>) <speaker>: <text>`` with role ``user``; a date's turns are sent in batches of ``add_batch_size``
(10) so the extractor sees context; the batch metadata carries ``session_date`` / ``session_iso`` / ``session_id`` (+ ``dia_ids``, the
source records of the batch, kept for the provenance audit; metadata is stored, never shown to the extractor).

Configuration (``config``): llm = Mem0's ``deepseek`` provider with ``deepseek_base_url`` = the local counting proxy (``llm_proxy``),
embedder = Mem0's ``huggingface`` provider (sentence-transformers, Qwen/Qwen3-Embedding-0.6B @ the pinned revision), vector store =
Mem0's ``qdrant`` provider in local-path mode on the Colab disk, history db next to it. ``MEM0_TELEMETRY=False`` and ``MEM0_DIR`` are
set before ``mem0`` is imported (no telemetry, no files outside the run directory). ``OPENAI_API_KEY`` must not be set: nothing may
fall back to Mem0's OpenAI defaults (``runtime_audit`` records the classes and URLs actually in use).
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from . import context as CX
from . import protocol as P

SOURCE_LABEL = "mem0"


def render(e: Mapping) -> str:
    """= LifeBench_eval Mem0Adapter._render_message_content with session_date_label (no image fields in this dataset)."""
    label = datetime.fromisoformat(e["date"]).strftime("%B %d, %Y")
    prefix = f"(Conversation date: {label})"
    sp = e.get("speaker") or ""
    return f"{prefix} {sp}: {e['content'].strip()}" if sp else f"{prefix} {e['content'].strip()}"


def batches(day_events: Sequence[Mapping], size: int) -> List[List[Mapping]]:
    return [list(day_events[i:i + size]) for i in range(0, len(day_events), size)]


def batch_metadata(batch: Sequence[Mapping]) -> dict:
    first = batch[0]
    label = datetime.fromisoformat(first["date"]).strftime("%B %d, %Y")
    iso = datetime.fromisoformat(f"{first['date']}T00:00:00").replace(tzinfo=timezone.utc).isoformat()
    return {"session_date": label, "session_iso": iso, "session_id": f"session:{first['date']}", "dia_ids": [e["event_id"] for e in batch]}


def config(work_dir: Path, *, proxy_url: str, api_key: str, collection: str, embedder: Optional[dict] = None, device: str = "cuda") -> dict:
    m = P.MEM0
    emb = embedder or {"provider": "huggingface", "config": {"model": m["embedder"]["model"], "embedding_dims": m["embedder"]["embedding_dims"],
                                                            "model_kwargs": {**m["embedder"]["model_kwargs"], "device": device}}}
    return {"llm": {"provider": "deepseek", "config": {"model": m["llm"]["model"], "api_key": api_key, "deepseek_base_url": proxy_url,
                                                       "temperature": m["llm"]["temperature"], "max_tokens": m["llm"]["max_tokens"]}},
            "embedder": emb,
            "vector_store": {"provider": "qdrant", "config": {"collection_name": collection, "path": str(Path(work_dir) / "qdrant"),
                                                             "embedding_model_dims": m["embedder"]["embedding_dims"], "on_disk": True}},
            "history_db_path": str(Path(work_dir) / "history.db")}


def public_config(cfg: Mapping) -> dict:
    import copy
    c = copy.deepcopy(dict(cfg))
    c["llm"]["config"]["api_key"] = "***"
    return c


class Mem0Memory:
    def __init__(self, work_dir: Path, *, user_id: str, proxy_url: str, api_key: str, collection: str, embedder: Optional[dict] = None,
                 device: str = "cuda", sleep=time.sleep):
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MEM0_TELEMETRY"] = "False"
        os.environ["MEM0_DIR"] = str(work_dir / "mem0_dir")
        if os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is set: unset it so nothing can fall back to Mem0's OpenAI defaults")
        from mem0 import Memory
        self.cfg = config(work_dir, proxy_url=proxy_url, api_key=api_key, collection=collection, embedder=embedder, device=device)
        self.memory = Memory.from_config(self.cfg)
        self.user_id, self.work_dir, self.sleep = user_id, work_dir, sleep
        self.stats = {"add_calls": 0, "add_failed_batches": 0, "add_retries": 0, "add_seconds": 0.0, "events_sent": 0, "memories_returned_by_add": 0,
                      "search_calls": 0, "search_seconds": 0.0}

    def runtime_audit(self) -> dict:
        import importlib.metadata as md
        mem = self.memory
        llm_client = getattr(getattr(mem, "llm", None), "client", None)
        return {"mem0ai_version": md.version("mem0ai"), "llm_class": type(mem.llm).__name__, "llm_model": mem.llm.config.model,
                "llm_base_url": str(getattr(llm_client, "base_url", "")), "embedder_class": type(mem.embedding_model).__name__,
                "embedder_model": mem.embedding_model.config.model, "embedding_dims": mem.embedding_model.config.embedding_dims,
                "vector_store_class": type(mem.vector_store).__name__, "collection": self.cfg["vector_store"]["config"]["collection_name"],
                "qdrant_path": self.cfg["vector_store"]["config"]["path"], "history_db": self.cfg["history_db_path"],
                "telemetry_env": os.environ.get("MEM0_TELEMETRY"), "openai_api_key_set": bool(os.environ.get("OPENAI_API_KEY")),
                "infer": P.MEM0["add"]["infer"], "add_batch_size": P.MEM0["add"]["batch_size"], "search_top_k": P.MEM0["search"]["top_k"]}

    def add_day(self, day_events: Sequence[Mapping]) -> dict:
        """One date: its events in batches, each ``Memory.add(messages, user_id, metadata, infer=True)``; at most ``max_attempts`` per batch."""
        out = {"batches": 0, "failed": 0, "memories": 0}
        for b in batches(day_events, int(P.MEM0["add"]["batch_size"])):
            msgs = [{"role": "user", "content": render(e)} for e in b]
            meta = batch_metadata(b)
            for attempt in range(1, int(P.MEM0["add"]["max_attempts"]) + 1):
                t0 = time.perf_counter()
                try:
                    r = self.memory.add(msgs, user_id=self.user_id, metadata=meta, infer=bool(P.MEM0["add"]["infer"]))
                    self.stats["add_seconds"] += time.perf_counter() - t0
                    n = len((r or {}).get("results") or [])
                    self.stats["memories_returned_by_add"] += n
                    out["memories"] += n
                    break
                except Exception:  # noqa: BLE001 - LLMError / transport: retried with backoff, then recorded as a failed batch
                    self.stats["add_seconds"] += time.perf_counter() - t0
                    if attempt == int(P.MEM0["add"]["max_attempts"]):
                        self.stats["add_failed_batches"] += 1
                        out["failed"] += 1
                    else:
                        self.stats["add_retries"] += 1
                        self.sleep(min(60.0, 2.0 ** attempt))
            self.stats["add_calls"] += 1
            self.stats["events_sent"] += len(b)
            out["batches"] += 1
        return out

    def search(self, query: str) -> dict:
        t0 = time.perf_counter()
        r = self.memory.search(query, top_k=int(P.MEM0["search"]["top_k"]), filters={"user_id": self.user_id})
        ms = round((time.perf_counter() - t0) * 1000, 1)
        self.stats["search_calls"] += 1
        self.stats["search_seconds"] += ms / 1000
        res = (r or {}).get("results") or []
        entries = []
        for rank, x in enumerate(res, 1):
            meta = x.get("metadata") or {}
            iso = meta.get("session_iso")
            day = iso[:10] if iso else None
            ids = [str(x.get("id"))] + [str(i) for i in (meta.get("dia_ids") or [])]
            entries.append(CX.Entry(rank=rank, date=day, source=SOURCE_LABEL, text=x.get("memory") or "", ids=ids, score=x.get("score")))
        return {"entries": entries, "meta": {"retrieval_ms": ms, "results": len(res)}}

    def count(self) -> int:
        r = self.memory.get_all(filters={"user_id": self.user_id}, top_k=1000000)
        return len((r or {}).get("results") or [])

    def close(self) -> None:
        """Release Mem0's files (history db connection, Qdrant local clients) so the live directory can be snapshotted / replaced."""
        mem = self.memory
        for obj in (getattr(mem, "vector_store", None), getattr(mem, "entity_store", None), getattr(mem, "_telemetry_vector_store", None)):
            try:
                obj.client.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            mem.db.connection.close()
        except Exception:  # noqa: BLE001
            pass
        import gc
        gc.collect()


def provenance_ids(entries: Sequence[CX.Entry]) -> List[str]:
    """dia_ids of the ADD batches that produced the returned memories (a batch-level provenance, weaker than StillHere's record ids)."""
    from .timeline import gold_id
    return [gold_id(i.split(":t", 1)[0]) for e in entries for i in e.ids[1:]]
