"""The product dense channel of one profile: a local Qwen3-Embedding-0.6B index over the evidence archive (nothing leaves the machine).

    build(root, profile_id, model_dir=None, embedder=None)   resumable; only texts not yet in the cache are embedded
    status(root, profile_id)                                 NOT_BUILT | PARTIAL | READY | STALE (archive changed since the build)
    open_channel(root, profile_id, model_dir=None, encoder=None) -> (DenseChannel | None, reason)

Layout: ``<profile>/indexes/product_dense/`` = ``embedding_cache.sqlite`` (kept across rebuilds) + ``index/`` (the finished index and
its time sidecar) + ``BUILD_REPORT.json``. A STALE index is not deleted: ``build`` moves it to ``index.superseded-<snapshot>`` and
writes a new one (only the new messages are embedded). The model directory: ``model_dir`` argument, else ``STILLHERE_EMBEDDING_MODEL_DIR``,
else the Hugging Face cache snapshot of the pinned revision. torch / transformers are imported only when a model is loaded.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Optional

from . import memory_store as MS
from . import product_config as PC
from . import store as S

DIR = "product_dense"
MODEL_ENV = "STILLHERE_EMBEDDING_MODEL_DIR"


def root_dir(pdir: Path) -> Path:
    return Path(pdir) / "indexes" / DIR


def resolve_model_dir(model_dir: Optional[str] = None) -> Optional[Path]:
    if model_dir:
        return Path(model_dir)
    if os.environ.get(MODEL_ENV):
        return Path(os.environ[MODEL_ENV])
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    snap = hub / "models--Qwen--Qwen3-Embedding-0.6B" / "snapshots" / PC.EMBEDDING["revision"]
    return snap if (snap / "config.json").is_file() else None


def _names(pid: str) -> dict:
    return {pid: "本人", MS.other_id(pid): "对方"}


def _snapshot(pdir: Path, prof: dict) -> str:
    from stillhere.retrieval.ef1 import local_dense as LD
    api = MS.open_api(pdir, prof)
    try:
        return LD.snapshot_sha(api.store._conn, prof["profile_id"])
    finally:
        api.close()


def status(root, profile_id: str) -> dict:
    root = Path(root) if root else S.default_root()
    pdir = S.profile_dir(root, profile_id)
    d = root_dir(pdir)
    man = d / "index" / "embedding_index_manifest.json"
    if MS.status(root, profile_id)["memory_status"] == "NOT_BUILT":
        return {"dense_status": "NO_ARCHIVE", "dir": str(d)}
    if not man.is_file():
        return {"dense_status": "PARTIAL" if (d / "embedding_cache.sqlite").is_file() else "NOT_BUILT", "dir": str(d)}
    m = json.loads(man.read_text(encoding="utf-8"))
    snap = _snapshot(pdir, S.read_json(pdir / "profile.json"))
    return {"dense_status": "READY" if m.get("snapshot_sha256") == snap else "STALE", "dir": str(d), "chunks": m.get("chunk_count"),
            "index_sha256": m.get("index_sha256"), "model": m.get("model")}


def build(root, profile_id: str, *, model_dir: Optional[str] = None, embedder=None, batch_size: int = 16, time_budget_s: Optional[float] = None,
          threads: Optional[int] = None, device: str = "cpu") -> dict:
    from stillhere.retrieval import local_embedding as LE
    from stillhere.retrieval.ef1 import local_dense as LD
    root = Path(root) if root else S.default_root()
    pdir = S.profile_dir(root, profile_id)
    prof = S.read_json(pdir / "profile.json")
    st = status(root, profile_id)
    if st["dense_status"] == "NO_ARCHIVE":
        raise S.StoreError("MEMORY_NOT_BUILT", "build the evidence archive first: stillhere persona memory-build")
    if st["dense_status"] == "READY":
        return {**st, "reused": True}
    d = root_dir(pdir)
    if st["dense_status"] == "STALE":
        old = json.loads((d / "index" / "embedding_index_manifest.json").read_text(encoding="utf-8"))
        shutil.move(str(d / "index"), str(d / f"index.superseded-{str(old.get('snapshot_sha256'))[:12]}"))
    if embedder is None:
        md = resolve_model_dir(model_dir)
        if md is None:
            raise S.StoreError("NO_EMBEDDING_MODEL", f"no local {PC.EMBEDDING['id']} (revision {PC.EMBEDDING['revision'][:8]}): download it once or set {MODEL_ENV}; "
                                                     "without it the product memory uses V2L (lexical) automatically")
        embedder = LE.get_embedder(md, threads=threads, device=device)
    api = MS.open_api(pdir, prof)
    try:
        res = LD.build_local(api, profile_id, d, embedder, names=_names(profile_id), model=PC.EMBEDDING["cache_model"], dimension=PC.EMBEDDING["dimension"],
                             batch_size=batch_size, time_budget_s=time_budget_s)
    finally:
        api.close()
    res.update(model_id=PC.EMBEDDING["id"], model_revision=PC.EMBEDDING["revision"], at=S.now_iso())
    rep = d / "BUILD_REPORT.json"
    prev = S.read_json(rep) if rep.is_file() else {"runs": []}
    prev["runs"].append({k: v for k, v in res.items() if k != "sidecar"})
    prev["last"] = prev["runs"][-1]
    S.write_json(rep, prev)
    return {**status(root, profile_id), "build": {k: res.get(k) for k in ("status", "documents", "chunks", "seconds")}, "reused": False}


def open_channel(root, profile_id: str, *, model_dir: Optional[str] = None, encoder=None):
    """(DenseChannel, "OK") for a READY index, else (None, reason). Never raises: the caller falls back to V2L."""
    try:
        st = status(root, profile_id)
        if st["dense_status"] not in ("READY", "STALE"):
            return None, f"DENSE_UNAVAILABLE: index {st['dense_status']} (stillhere persona memory-build --dense)"
        from stillhere.retrieval.ef1 import local_dense as LD
        if encoder is None:
            from stillhere.retrieval import local_embedding as LE
            md = resolve_model_dir(model_dir)
            if md is None:
                return None, f"DENSE_UNAVAILABLE: no local {PC.EMBEDDING['id']} for query vectors (set {MODEL_ENV})"
            encoder = LE.LocalQueryEncoder(LE.get_embedder(md))
        pdir = S.profile_dir(Path(root) if root else S.default_root(), profile_id)
        ch, why = LD.open_local(root_dir(pdir), model=PC.EMBEDDING["cache_model"], dimension=PC.EMBEDDING["dimension"], query_encoder=encoder)
        if ch is None:
            return None, why
        return ch, "OK" if st["dense_status"] == "READY" else "OK_STALE: messages imported after the dense build are found by the lexical channels only"
    except Exception as e:  # noqa: BLE001 - an unavailable dense channel must never stop chat
        return None, f"DENSE_UNAVAILABLE: {type(e).__name__}: {e}"
