"""Private persona profile storage, lineage and atomic writes.

Layout (root = ``--root`` > ``STILLHERE_PERSONA_ROOT`` > ``~/.stillhere/personas``; it must not be a tracked part of a git work tree)::

    <root>/<profile_id>/profile.json                 identity binding (private), source fingerprint, versions
    <root>/<profile_id>/datasets/<version_id>/       train / dev / test(.jsonl, sealed) + *_meta, QUALITY_REPORT, manifests
    <root>/<profile_id>/LINEAGE.jsonl                one line per produced artefact (for later revoke / rebuild)

Nothing is overwritten: a dataset version directory is built under a temporary name and renamed into place in one step; an existing
version with a complete manifest is verified and reused (idempotent).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT_ENV = "STILLHERE_PERSONA_ROOT"
PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")


class StoreError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def default_root() -> Path:
    return Path(os.environ.get(ROOT_ENV) or Path.home() / ".stillhere" / "personas")


def _git_toplevel(p: Path) -> Optional[Path]:
    for d in [p, *p.parents]:
        if (d / ".git").exists():
            return d
    return None


def check_private_root(root: Path) -> None:
    """Refuse a root inside a git work tree unless git ignores it (private data must never be trackable)."""
    root = root.resolve()
    probe = root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    top = _git_toplevel(probe)
    if top is None:
        return
    try:
        r = subprocess.run(["git", "-C", str(top), "check-ignore", "-q", str(root) + "/"], capture_output=True, timeout=30)   # "/": a directory, even before it exists
    except (OSError, subprocess.TimeoutExpired):
        raise StoreError("ROOT_NOT_PRIVATE", f"{root} is inside a git work tree and git is unavailable to confirm it is ignored") from None
    if r.returncode != 0:
        raise StoreError("ROOT_NOT_PRIVATE", f"{root} is inside the git work tree {top} and is not ignored; choose a private root")


def profile_dir(root: Path, profile_id: str) -> Path:
    if not PROFILE_ID.match(profile_id or ""):
        raise StoreError("BAD_PROFILE_ID", "profile id: 3-64 chars of a-z 0-9 _ - (starting with a letter or digit)")
    d = (root / profile_id).resolve()
    if d.parent != root.resolve():
        raise StoreError("BAD_PROFILE_ID", "profile id escapes the root")
    return d


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_json(path: Path, obj) -> None:
    atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8"))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_lineage(pdir: Path, entry: dict) -> None:
    p = pdir / "LINEAGE.jsonl"
    line = json.dumps({"at": now_iso(), **entry}, ensure_ascii=False) + "\n"
    with open(p, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


# Local-only by default. Nothing in the pipeline uploads, calls an external API or publishes; a cloud training package is only WRITTEN
# locally, and writing one needs the operator's explicit confirmation that a private chat will leave this machine (``confirm_cloud``).
PRIVACY_DEFAULTS = {"local_only": True, "cloud_upload": False, "external_api": False, "publish": False}
CLOUD_NOTICE = ("This package contains REAL PRIVATE CHAT text (the target person's replies and the other person's messages as context). "
                "Uploading it to Google Drive / Colab sends it to a cloud service. Only upload it if you are entitled to and the people in "
                "the chat would accept that; nothing is uploaded automatically.")


def record_cloud_confirmation(pdir: Path, package_id: str, target: str) -> dict:
    """The operator confirmed (CLI flag or an interactive yes) that this package may be taken to a cloud GPU. One line in LINEAGE; it is
    a statement by the operator about this package only, never a consent record of anyone else."""
    entry = {"artefact": f"packages/{package_id}", "kind": "cloud_transfer_confirmed_by_operator", "target": target}
    append_lineage(pdir, entry)
    return {**entry, "at": now_iso()}
