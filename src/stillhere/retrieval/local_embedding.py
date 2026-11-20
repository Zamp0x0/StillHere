"""Local CPU embedding provider: Qwen/Qwen3-Embedding-0.6B through transformers (no API, no upload of private text).

Follows the model card's transformers usage: left padding, last-token pooling, L2 normalisation; queries carry an instruction
(``Instruct: <task>\\nQuery:<text>``), documents are embedded as they are. The model is loaded lazily, once per process
(``get_embedder``), in float32 on the CPU with a bounded number of torch threads; ``embed_documents_local`` fills the existing
``retrieval.dense.EmbeddingCache`` batch by batch (resumable: the cache is keyed by model / dimension / type / text, so a stopped build
continues where it stopped, and a batch that runs out of memory is retried at half the size).

torch / transformers are imported only when a model is actually loaded, so the rest of StillHere (and its CPU tests) never need them.
"""
from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .dense import EmbeddingCache, missing_texts

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
EXPECTED_DIMENSION = 1024                    # checked against the loaded model, never assumed
CACHE_MODEL_NAME = "local:qwen3-embedding-0.6b@97b0c614"   # the EmbeddingCache / index "model" field (keeps API vectors separate)
QUERY_TASK = "Given a question about a past private chat conversation, retrieve the chat messages that answer it"
MAX_LENGTH = 512                              # documents are short chat windows; longer text is truncated (recorded in stats)


def rss_mb() -> Optional[float]:
    """Resident memory of this process in MB (psutil when available, else the Windows / Linux fallbacks, else None)."""
    try:
        import psutil  # type: ignore
        return round(psutil.Process().memory_info().rss / 2**20, 1)
    except Exception:  # noqa: BLE001
        pass
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except Exception:  # noqa: BLE001
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        h = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            return round(pmc.WorkingSetSize / 2**20, 1)
    except Exception:  # noqa: BLE001
        pass
    return None


@dataclass
class EmbedStats:
    load_seconds: float = 0.0
    rss_before_load_mb: Optional[float] = None
    rss_after_load_mb: Optional[float] = None
    rss_peak_mb: Optional[float] = None
    documents: int = 0
    queries: int = 0
    document_seconds: float = 0.0
    query_seconds: float = 0.0
    truncated: int = 0
    oom_retries: int = 0
    batch_sizes: List[int] = field(default_factory=list)

    def note_rss(self) -> None:
        r = rss_mb()
        if r is not None and (self.rss_peak_mb is None or r > self.rss_peak_mb):
            self.rss_peak_mb = r

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["batch_sizes"] = sorted(set(self.batch_sizes))
        d["docs_per_second"] = round(self.documents / self.document_seconds, 2) if self.document_seconds else None
        d["query_ms_mean"] = round(1000 * self.query_seconds / self.queries, 1) if self.queries else None
        return d


class LocalQwen3Embedder:
    """One loaded Qwen3-Embedding-0.6B. ``embed_documents`` / ``embed_queries`` return float32 unit vectors, shape (n, dimension)."""

    def __init__(self, model_dir: str | Path, *, threads: Optional[int] = None, dtype: str = "float32", max_length: int = MAX_LENGTH,
                 query_task: str = QUERY_TASK, device: str = "cpu") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.stats = EmbedStats(rss_before_load_mb=rss_mb())
        self.model_dir = Path(model_dir)
        if not (self.model_dir / "config.json").is_file():
            raise FileNotFoundError(f"no local {MODEL_ID} at {self.model_dir} (download it once; see research/r2_memory_v2/02_LOCAL_EMBEDDING.md)")
        n = threads or max(1, min(8, (os.cpu_count() or 4) // 2))
        torch.set_num_threads(n)
        self.threads = n
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), padding_side="left")
        td = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
        self.model = AutoModel.from_pretrained(str(self.model_dir), torch_dtype=td, low_cpu_mem_usage=True).eval().to(device)
        self.device = device                                                     # "cpu" locally; "cuda" for a Colab rebuild
        self.dtype = dtype
        self.torch = torch
        self.stats.load_seconds = round(time.perf_counter() - t0, 2)
        self.stats.rss_after_load_mb = rss_mb()
        self.stats.note_rss()
        self.dimension = int(self.model.config.hidden_size)
        if self.dimension != EXPECTED_DIMENSION:
            raise ValueError(f"{MODEL_ID} reports hidden size {self.dimension}, expected {EXPECTED_DIMENSION}")
        self.max_length, self.query_task = max_length, query_task

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        batch = self.tokenizer(list(texts), padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        self.stats.truncated += int((batch["attention_mask"].sum(dim=1) >= self.max_length).sum())
        with torch.inference_mode():
            out = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"]
            if bool((mask[:, -1] == 1).all()):                      # left padding: the last position is every row's last token
                pooled = out[:, -1]
            else:                                                    # defensive: right padding
                idx = mask.sum(dim=1) - 1
                pooled = out[torch.arange(out.shape[0]), idx]
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
        self.stats.note_rss()
        return pooled.cpu().numpy().astype(np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        t0 = time.perf_counter()
        v = self._encode(texts)
        self.stats.documents += len(texts)
        self.stats.document_seconds += time.perf_counter() - t0
        return v

    def query_text(self, text: str) -> str:
        return f"Instruct: {self.query_task}\nQuery:{text}" if self.query_task else text

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        t0 = time.perf_counter()
        v = self._encode([self.query_text(t) for t in texts])
        self.stats.queries += len(texts)
        self.stats.query_seconds += time.perf_counter() - t0
        return v


_SINGLE: Dict[str, LocalQwen3Embedder] = {}


def get_embedder(model_dir: str | Path, **kw) -> LocalQwen3Embedder:
    """One instance per process and model directory (the model is ~1.2 GB in float32 RAM: never load it twice)."""
    key = str(Path(model_dir).resolve())
    if key not in _SINGLE:
        _SINGLE[key] = LocalQwen3Embedder(model_dir, **kw)
    return _SINGLE[key]


def embed_documents_local(embedder, cache: EmbeddingCache, texts: Sequence[str], *, model: str = CACHE_MODEL_NAME,
                          dimension: int = EXPECTED_DIMENSION, batch_size: int = 16, min_batch: int = 1,
                          progress: Optional[Callable[[dict], None]] = None, time_budget_s: Optional[float] = None) -> dict:
    """Embed every text the cache does not have yet, ``batch_size`` at a time (texts sorted by length, so padding stays small). A
    ``MemoryError`` / ``RuntimeError`` mentioning memory halves the batch and retries. Stops cleanly after ``time_budget_s`` (the next
    call resumes). Returns counts and timings; never raises for a stop."""
    todo = sorted(missing_texts(cache, model, dimension, texts), key=len)
    done, t0, bs = 0, time.perf_counter(), batch_size
    stopped = False
    i = 0
    while i < len(todo):
        if time_budget_s is not None and time.perf_counter() - t0 > time_budget_s:
            stopped = True
            break
        part = todo[i:i + bs]
        try:
            vecs = embedder.embed_documents(part)
        except (MemoryError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and "memory" not in str(e).lower():
                raise
            if bs <= min_batch:
                raise
            bs = max(min_batch, bs // 2)
            getattr(embedder, "stats", EmbedStats()).oom_retries += 1
            gc.collect()
            continue
        cache.put_many([(EmbeddingCache.key(model, dimension, "document", t), v) for t, v in zip(part, vecs)])
        getattr(embedder, "stats", EmbedStats()).batch_sizes.append(bs)
        done += len(part)
        i += len(part)
        if progress and (done % (bs * 20) < bs or i >= len(todo)):
            progress({"embedded": done, "todo": len(todo), "seconds": round(time.perf_counter() - t0, 1), "rss_mb": rss_mb(), "batch": bs})
    return {"texts_total": len(set(texts)), "todo": len(todo), "embedded": done, "stopped_by_budget": stopped,
            "seconds": round(time.perf_counter() - t0, 1), "final_batch_size": bs}


def task_tag(task: Optional[str]) -> str:
    """Short hash of a query instruction: query vectors are cached per instruction (the cache key holds only the raw question)."""
    import hashlib
    return hashlib.sha256((task or "").encode("utf-8")).hexdigest()[:10]


class LocalQueryEncoder:
    """``embed_query`` callable for ``ef1.channels.DenseChannel``: one query -> one unit vector (cached by the channel)."""

    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self.query_task = getattr(embedder, "query_task", QUERY_TASK)

    def __call__(self, text: str) -> np.ndarray:
        return self.embedder.embed_queries([text])[0]
