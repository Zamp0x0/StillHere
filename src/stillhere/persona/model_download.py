"""``stillhere persona download-model``: fetch a model at the PINNED revision StillHere was tested with (Hugging Face Hub).

* ``embedding``: Qwen/Qwen3-Embedding-0.6B @ 97b0c614 (about 1.2 GB) for the dense (V2D) memory index; CPU is enough.
* ``base``: Qwen/Qwen3-8B @ b968826d (about 16 GB) for local chat / training on a CUDA GPU.

Without ``--dir`` the files go to the Hugging Face cache (``$HF_HOME/hub``), where StillHere finds the embedding model automatically.
Needs ``pip install -e ".[download]"`` (huggingface_hub). Only these two pinned models are offered: other models are not tested.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from . import final_config as FC
from . import product_config as PC

MODELS = {"embedding": {"repo": PC.EMBEDDING["id"], "revision": PC.EMBEDDING["revision"], "approx_gb": 1.2, "env": "STILLHERE_EMBEDDING_MODEL_DIR"},
          "base": {"repo": FC.FINAL_CONFIG["base"]["id"], "revision": FC.FINAL_CONFIG["base"]["revision"], "approx_gb": 16.4, "env": "STILLHERE_BASE_MODEL_DIR"}}


class DownloadError(RuntimeError):
    def __init__(self, code: str, msg: str):
        super().__init__(f"{code}: {msg}")
        self.code = code


def download(what: str, *, dest: Optional[str] = None, snapshot: Optional[Callable] = None) -> dict:
    if what not in MODELS:
        raise DownloadError("UNKNOWN_MODEL", f"choose one of {sorted(MODELS)}")
    m = MODELS[what]
    if snapshot is None:
        try:
            from huggingface_hub import snapshot_download as snapshot
        except ImportError as e:
            raise DownloadError("NO_HUGGINGFACE_HUB", 'install it first: pip install -e ".[download]"') from e
    kw = {"repo_id": m["repo"], "revision": m["revision"]}
    if dest:
        kw["local_dir"] = str(Path(dest))
    path = Path(snapshot(**kw))
    if not (path / "config.json").is_file():
        raise DownloadError("INCOMPLETE", f"{path} has no config.json; run the command again to resume the download")
    nxt = (f"memory: found automatically (HF cache); or set {m['env']}={path}, then: stillhere persona memory-build --profile <profile-id> --dense"
           if what == "embedding" else f"chat: stillhere persona chat --profile <profile-id> --base-dir \"{path}\"   (or set {m['env']}={path})")
    return {"model": m["repo"], "revision": m["revision"], "path": str(path), "env": m["env"], "next": nxt}
