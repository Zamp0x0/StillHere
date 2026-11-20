"""MODEL_VISIBLE_REDACTED_VIEW for t2d4-data-v1.1: the runtime's own pre-render redaction ``stillhere.session.disclosure/1.0``.

Training and inference must see the same thing. Simple Context v1.1 keeps the D4 renderer ``grounded_persona/1.0`` unchanged (hash
3798c363…f516128) and redacts the evidence INPUTS before rendering: a PROTECTED / POLICY_UNCERTAIN item reaches the renderer with

    evidence_text  <subject>有一条<类别>的记录，不可透露具体内容                          (PROTECTED)
                   <subject>有一条<类别>的记录，需要先得到本人同意，目前不能确认可以透露  (POLICY_UNCERTAIN)
    value          （不显示）
    replaces/note  （不显示）  (only when present)

D4 v1.1 applies exactly that, through a byte-identical, hash-pinned copy of the Context module (``vendor/context_disclosure_v1_0.py``):
the category label comes from the Context function ``categories(value, statement, attribute=predicate)`` and the text from
``redacted_text``. The policy itself is the dataset's (the scenario decides PROTECTED / POLICY_UNCERTAIN). Raw values stay only in the
restricted fields (``evidence``, ``answer_state``, ``guard_spec``) of the private files.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Sequence

from .evidence import Evidence
from .vendor import VENDORED_SHA256

_VENDORED = Path(__file__).resolve().parent / "vendor" / "context_disclosure_v1_0.py"
if hashlib.sha256(_VENDORED.read_bytes()).hexdigest() != VENDORED_SHA256["context_disclosure_v1_0.py"]:
    raise ImportError("vendored Context disclosure module changed: refusing to build a model-visible view from it")
from .vendor import context_disclosure_v1_0 as DS  # noqa: E402

VIEW_VERSION = DS.POLICY_VERSION                      # stillhere.session.disclosure/1.0
VIEW_NAME = "MODEL_VISIBLE_REDACTED_VIEW"
HIDDEN = "（不显示）"
CONTEXT_SOURCE = {"repo": "StillHere-Context (read-only)", "branch": "research/simple-context-v1", "commit": "8953fa0",
                  "file": "src/stillhere/session/disclosure.py", "sha256": VENDORED_SHA256["context_disclosure_v1_0.py"],
                  "handoff": "simple_context_v1_1/D4_TRAINING_THREAD_HANDOFF.md section 4"}
BASE_LENGTH_RULE = "用一两句话简短回复，像平时聊天一样"   # protocol group BASE_LENGTH: its one extra rule
_MARKERS = ("的记录，不可透露具体内容", "的记录，需要先得到本人同意，目前不能确认可以透露")
_DATE = re.compile(r"\d{4}-\d{2}(?:-\d{2})?")


def is_redacted(e: Evidence) -> bool:
    return e.disclosure_policy != "ALLOWED" and e.value == HIDDEN and e.evidence_text.endswith(_MARKERS)


def has_redaction_line(text: str) -> bool:
    return any(m in text for m in _MARKERS)


def model_visible_view(evidence: Optional[Sequence[Evidence]]) -> Optional[List[Evidence]]:
    """ALLOWED items unchanged; PROTECTED / POLICY_UNCERTAIN items redacted exactly as the runtime does (idempotent)."""
    if evidence is None:
        return None
    out = []
    for e in evidence:
        if e.disclosure_policy == "ALLOWED" or is_redacted(e):
            out.append(e)
            continue
        dec = DS.Decision(e.disclosure_policy, DS.categories(e.value, e.evidence_text, attribute=e.predicate), "dataset policy")
        out.append(replace(e, evidence_text=DS.redacted_text(e.subject, dec), value=HIDDEN,
                           replaces=HIDDEN if e.replaces else None, note=HIDDEN if e.note else None))
    return out


def raw_values_in(text: str, evidence: Optional[Sequence[Evidence]]) -> List[str]:
    """fact_ids of non-ALLOWED (raw) items whose value, or any 4-digit run of it, occurs in ``text``; ISO dates are ignored for the
    4-digit check (the conversation date or a validity date is never a partial number)."""
    bare = _DATE.sub(" ", text)
    hits = []
    for e in evidence or ():
        if e.disclosure_policy == "ALLOWED" or not e.value or is_redacted(e):
            continue
        runs = [m for m in re.findall(r"\d+", e.value) if len(m) >= 4]
        if e.value in text or any(r[i:i + 4] in bare for r in runs for i in range(len(r) - 3)):
            hits.append(e.fact_id)
    return hits


def contract(renderer_version: str, renderer_sha: str) -> dict:
    return {"renderer": renderer_version, "renderer_sha256": renderer_sha, "renderer_changed": False,
            "model_visible_view": VIEW_NAME, "redaction_format": VIEW_VERSION, "redaction_source": CONTEXT_SOURCE,
            "redaction": {"PROTECTED": "<subject>有一条<类别>的记录，不可透露具体内容", "POLICY_UNCERTAIN": "<subject>有一条<类别>的记录，需要先得到本人同意，目前不能确认可以透露",
                          "value": HIDDEN, "replaces_note": HIDDEN + " (only when present)", "category_labels": DS.LABEL,
                          "categories_from": "Context categories(value, statement, attribute=predicate)", "policy_from": "dataset scenario",
                          "idempotent": True, "raw_values_kept_only_in": ["evidence", "answer_state", "guard_spec", "ablation_answer_states"]},
            "base_length_extra_rule": BASE_LENGTH_RULE,
            "budget": {"cutoff_len": 1536, "rule": "prompt_tokens + max_new_tokens + 2 <= 1536", "recent_turn_limit": 4},
            "not_model_visible": ["subject_type (runtime default friend when no relation is recorded; the renderer never prints it)",
                                  "source_ref"]}
