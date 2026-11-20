"""The frozen final configuration of the StillHere personalisation research version (2026-11-19), chosen by the registered rules on real
evaluations only. Every component may be an older version: it is the combination the evidence supports, not the newest code.

    base          Qwen3-8B @ b968826d (NF4)                            unchanged
    adapter       OFF by default (r1 / r2 kept for research; R2_RULE = R2_NOT_BETTER)
    persona skill the Skill card (runtime persona prompt) c-<research-card>
    memory        V2 pipeline, LEXICAL retrieval (V2L)                 MEMORY_V2_RULE (HOLDOUT) = ADOPT_V2 / V2L; V4: V2D not shown better
    reader        "p1" (subject rule for mixed 你 + 我 questions, router past-speech cue, speaker note)   V4 rule: V2L_NEW replaces V2L_OLD
    writer        EVIDENCE_ONLY (no structured full build; the CRITICAL wedding check is unresolved on the real model)

``apply(runtime_config)`` returns a copy of a profile's ``runtime_config.json`` with the ``memory_pipeline`` section set; a config without
that section keeps the historical behaviour (V1 pipeline, reader v1), so old configs and every historical evaluation stay reproducible.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Mapping

FINAL_VERSION = "stillhere.persona.final/2026-11-19"
FINAL_MEMORY_PIPELINE = {"pipeline": "v2", "dense": False, "k": 8, "neighbors": [2, 2], "date_mode": "prefer", "strip_qform": False,
                         "reader_version": "p1", "writer_mode": "EVIDENCE_ONLY"}
FINAL_CONFIG = {
    "version": FINAL_VERSION,
    "base": {"id": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218", "quantization": "nf4"},
    "embedding": {"id": "Qwen/Qwen3-Embedding-0.6B", "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", "used_by_default": False},
    "adapter": {"enabled_by_default": False, "available": ["a-<research-profile>-r1-checkpoint-132", "a-<research-profile>-r2-checkpoint-263"],
                "decision": "R2_NOT_BETTER (registered); adapters kept for research, never enabled automatically"},
    "persona_skill": {"card_id": "c-<research-card>", "prompt_sha256": "9baa0e12e923f8e2d49edf3d0b6c1a9fcabfbcddbc6c8ffee5d66ab39c83c63c"},
    "memory_pipeline": FINAL_MEMORY_PIPELINE,
    "decisions": {"MEMORY_V2_RULE": "ADOPT_V2 (V2L), HOLDOUT, 2026-11-18", "V3": "no candidate replaced V2L (K40 not validated, incomplete)",
                  "V4": "V2L_NEW replaces V2L_OLD (judge +5.71 [-1.29, 15.00], n = 34); V2D_NEW not better than V2L_NEW (+3.71 [-3.00, 12.09])",
                  "WRITER": "EVIDENCE_ONLY", "evidence_level": "LOW: AI-verified gold, one judge pass, small n; no human gold"},
}


def fingerprint(cfg: Mapping = FINAL_CONFIG) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def apply(runtime_config: Mapping) -> dict:
    """A copy of ``runtime_config`` with the final memory pipeline; adapter / memory / prompt selection are left as they are."""
    out = copy.deepcopy(dict(runtime_config))
    out["memory_pipeline"] = dict(FINAL_MEMORY_PIPELINE, version=FINAL_VERSION, config_sha256=fingerprint())
    return out
