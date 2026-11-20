"""The D4 renderer ``stillhere.response.grounded_persona/1.0``, reused — never re-implemented.

Source of truth: the T2-D4 workspace (``t2d4_dataset/t2d4/{render,evidence,profile_v1}.py``, t2d4-data-v1.0, commit 4d9c3a6). This package
carries a byte-identical copy (``vendor_d4``) so the Context runtime does not import another thread's worktree at run time. At import the
copy is verified: each file's sha256 (line endings normalised) and the renderer's own ``renderer_sha256()`` must equal the pinned values,
which are the values recorded in the D4.1 dataset freeze manifest. Any difference raises ``RendererMismatch`` and nothing is rendered.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .vendor_d4 import evidence as evidence_mod
from .vendor_d4 import profile_v1 as profile_mod
from .vendor_d4 import render as render_mod

PINNED_FILES = {"evidence.py": "2f3760bbcbb6c06c79a467224884f01f34eaf85e81ade9a8fe4e665229adfccd",
                "profile_v1.py": "82d854fce723b67492497b32400749219cee0a85fec34e898693671a086f774c",
                "render.py": "46edc5350a6f61f82a6106d52253e3ef33ac048bb2dc6d891752fe586661cf02"}
PINNED_RENDERER_SHA256 = "3798c36301246957457be3013cb43451d40d3e0d2cefe9e735ef0682af516128"   # = D4.1 dataset_freeze_manifest.json
RENDERER_VERSION = render_mod.RENDERER_VERSION
D4_SOURCE_COMMIT = "4d9c3a6"


class RendererMismatch(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify(directory: Path = Path(__file__).parent / "vendor_d4") -> dict:
    got = {name: file_sha256(Path(directory) / name) for name in PINNED_FILES}
    bad = {k: v for k, v in got.items() if v != PINNED_FILES[k]}
    if bad:
        raise RendererMismatch(f"D4 renderer files differ from the pinned release: {sorted(bad)}")
    if render_mod.renderer_sha256() != PINNED_RENDERER_SHA256:
        raise RendererMismatch("renderer_sha256() differs from the pinned D4.1 value")
    return {"renderer_version": RENDERER_VERSION, "renderer_sha256": PINNED_RENDERER_SHA256, "files": got, "d4_source_commit": D4_SOURCE_COMMIT}


IDENTITY = verify()                                                # refuses to import a modified copy

Evidence = evidence_mod.Evidence
UnknownSlot = evidence_mod.UnknownSlot
STATUSES = evidence_mod.STATUSES
DISCLOSURE = evidence_mod.DISCLOSURE
SUBJECT_TYPES = evidence_mod.SUBJECT_TYPES
RenderError = render_mod.RenderError
render_messages = render_mod.render_messages
build_profile = profile_mod.build


def renderer_sha256() -> str:
    return render_mod.renderer_sha256()
