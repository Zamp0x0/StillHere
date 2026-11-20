"""The model-visible input of a D4 build. The renderer is always ``grounded_persona/1.0`` (hash 3798c363…f516128, pinned by the Context
runtime); what changes between data versions is the evidence VIEW handed to it:

    raw             t2d4-data-v1.0 (default, so v1.0 rebuilds stay byte-identical): evidence as generated
    disclosure-1.0  t2d4-data-v1.1: MODEL_VISIBLE_REDACTED_VIEW = the runtime's pre-render redaction stillhere.session.disclosure/1.0

Builders call ``render_messages`` from here; scripts select the view once with ``use()`` before building.
"""
from __future__ import annotations

from . import disclosure_view as DV
from . import render as _r10

VIEWS = ("raw", "disclosure-1.0")
DATA_VERSION_OF_VIEW = {"raw": "t2d4-data-v1.0", "disclosure-1.0": "t2d4-data-v1.1"}
_active = {"v": "raw"}
RenderError = _r10.RenderError


def use(view: str) -> None:
    if view not in VIEWS:
        raise ValueError(f"unknown evidence view {view}")
    _active["v"] = view


def view() -> str:
    return _active["v"]


def redacted() -> bool:
    return _active["v"] != "raw"


def render_messages(**kw):
    if not redacted():
        return _r10.render_messages(**kw)
    raw = kw.get("evidence")
    msgs = _r10.render_messages(**{**kw, "evidence": DV.model_visible_view(raw)})
    leaked = DV.raw_values_in(msgs[0]["content"] + "\n" + msgs[1]["content"], raw)
    if leaked:
        raise RenderError(f"raw value of a non-disclosable item reaches the prompt: {leaked}")
    return msgs


def renderer_version() -> str:
    return _r10.RENDERER_VERSION


def renderer_sha256() -> str:
    return _r10.renderer_sha256()


def view_version() -> str:
    return DV.VIEW_VERSION if redacted() else "none (raw evidence)"


def base_length_enabled() -> bool:
    """v1.1 builds carry a pre-rendered BASE_LENGTH prompt on every evaluation item (v1.0 builds did not)."""
    return redacted()


def base_length_messages(**kw):
    return render_messages(**kw, extra_rules=(DV.BASE_LENGTH_RULE,))


def contract() -> dict:
    return DV.contract(renderer_version(), renderer_sha256())
