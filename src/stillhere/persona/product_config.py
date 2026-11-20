"""The PRODUCT memory configuration: V2D hybrid retrieval + Memory P2 by default, V2L lexical as the automatic fallback.

Product versions (a profile keeps the section it was finalized with; nothing is rewritten in place):
  * ``stillhere.persona.product/2026-11-19-v2d-p2`` (``product``, the default since the P2 final evaluation): V2D + P2 (evidence
    requirements + one bounded second pass, ``persona.memory_p2``). Registered rule A-G on the independent fresh-person set
    (LifeBench 于晓雯 100 QA, protocol 3d637b96): ADOPT_P2; answerable 51/83 -> 54/83, +0.036 [-0.048, +0.120] (NOT significant),
    gold-record recall +0.032 [+0.005, +0.063]. Only the retrieval part of P2 is in the product: the evaluation's answer prompt / coverage
    note / calendar note are not part of Reader p1.
  * ``stillhere.persona.product/2026-11-19-v2d`` (``product-v2d``): V2D without P2, the previous product default and the rollback.

This is a separate, versioned configuration. It does NOT change the frozen research version (``persona.final_config``: V2L, chosen by the
registered rules; V4 did not show V2D better than V2L: +3.71 [-3.00, 12.09], n = 34). V2D is chosen for the product because its Gold
recall was higher in every round (V3 +0.07 [0.01, 0.14], V4 +0.06 [0.01, 0.15]) at about the same Reader input, and because it falls
back to V2L whenever the dense channel is unavailable (no embedding model, no index, no torch), so chat never stops working.

Everything else is the research final configuration: Reader p1, attribution / grounding gates, adapter OFF, Persona Skill ON, Writer
EVIDENCE_ONLY.

    memory_pipeline (runtime_config.json)
      pipeline v2, dense true, dense_fallback "v2l", k 8, neighbors [2, 2], date_mode prefer, strip_qform false, reader_version p1,
      writer_mode EVIDENCE_ONLY, embedding {Qwen/Qwen3-Embedding-0.6B @ 97b0c614}, version, config_sha256

Choosing a pipeline at ``persona finalize --memory-pipeline``: ``product`` (V2D + P2, the default for a profile without a section),
``product-v2d`` (V2D without P2: the rollback), ``research-final`` (``final_config``: V2L), ``v1`` (the historical path: no section). A re-finalize without the flag keeps whatever the
profile already has.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Mapping, Optional

from . import final_config as FC

PRODUCT_V2D_VERSION = "stillhere.persona.product/2026-11-19-v2d"
PRODUCT_VERSION = "stillhere.persona.product/2026-11-19-v2d-p2"
EMBEDDING = {"id": "Qwen/Qwen3-Embedding-0.6B", "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", "dimension": 1024,
             "cache_model": "local:qwen3-embedding-0.6b@97b0c614"}
PRODUCT_V2D_MEMORY_PIPELINE = {"pipeline": "v2", "dense": True, "dense_fallback": "v2l", "k": 8, "neighbors": [2, 2], "date_mode": "prefer",
                               "strip_qform": False, "reader_version": "p1", "writer_mode": "EVIDENCE_ONLY", "embedding": EMBEDDING}
P2_VERSION = "stillhere.memory.p2/2026-11-19.2"          # == persona.memory_p2.P2_VERSION (a test keeps them equal)
PRODUCT_MEMORY_PIPELINE = dict(PRODUCT_V2D_MEMORY_PIPELINE, p2=True, p2_version=P2_VERSION)
PRODUCT_V2D_CONFIG = {
    "version": PRODUCT_V2D_VERSION,
    "derived_from": FC.FINAL_VERSION,
    "difference_from_research_final": "retrieval V2D hybrid (lexical EF1 + local Qwen3-Embedding-0.6B dense) instead of V2L; V2L is the fallback",
    "memory_pipeline": PRODUCT_MEMORY_PIPELINE,
    "adapter": {"enabled_by_default": False},
    "persona_skill": "the profile's CURRENT card (one activated card at a time)",
    "evidence": {"V3_recall_V2D_minus_V2L": "+0.07 [0.01, 0.14]", "V4_recall_V2D_minus_V2L": "+0.06 [0.01, 0.15]",
                 "V4_judge_V2D_minus_V2L_NEW": "+3.71 [-3.00, 12.09] (not significant)", "decision": "user product decision, not a registered-rule result"},
}
PRODUCT_V2D_CONFIG["memory_pipeline"] = PRODUCT_V2D_MEMORY_PIPELINE
PRODUCT_CONFIG = {
    "version": PRODUCT_VERSION,
    "derived_from": PRODUCT_V2D_VERSION,
    "difference_from_product_v2d": "Memory P2 retrieval: evidence requirements (date offset, event anchor, items, dates, month, advisory, recent) "
                                   "and one bounded second pass in each requirement's window; no LLM call; the first hop is unchanged",
    "memory_pipeline": PRODUCT_MEMORY_PIPELINE,
    "adapter": {"enabled_by_default": False},
    "persona_skill": "the profile's CURRENT card (one activated card at a time)",
    "evidence": {"protocol_sha256": "3d637b969cd4723f91dc241e6df1f91e69300d0946dff507ff7bca4720137856",
                 "results_zip_sha256": "166425878df7e1692d1ac9aa288aa1b064e1fb1981aded283cfab6a712251be2",
                 "fresh_person": "LifeBench 于晓雯 100 QA (PERSONA_SEEN_QA_UNSEEN)",
                 "answerable": "OLD 51/83 -> NEW 54/83, +0.036 [-0.048, 0.120], W/T/L 8/70/5, McNemar p 0.58 (not significant)",
                 "gold_recall": "+0.032 [0.005, 0.063]", "decision": "ADOPT_P2 by the registered rule A-G (all checks hold)"},
}
CHOICES = ("product", "product-v2d", "research-final", "v1")


def fingerprint(cfg: Mapping = PRODUCT_CONFIG) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def section(choice: str) -> Optional[dict]:
    """The ``memory_pipeline`` section for a choice (``None`` = no section = the historical V1 path)."""
    if choice == "product":
        return dict(copy.deepcopy(PRODUCT_MEMORY_PIPELINE), version=PRODUCT_VERSION, config_sha256=fingerprint())
    if choice == "product-v2d":
        return dict(copy.deepcopy(PRODUCT_V2D_MEMORY_PIPELINE), version=PRODUCT_V2D_VERSION, config_sha256=fingerprint(PRODUCT_V2D_CONFIG))
    if choice == "research-final":
        return FC.apply({})["memory_pipeline"]
    if choice == "v1":
        return None
    raise ValueError(f"memory pipeline must be one of {CHOICES}")


def apply(runtime_config: Mapping, choice: str = "product") -> dict:
    out = copy.deepcopy(dict(runtime_config))
    sec = section(choice)
    if sec is None:
        out.pop("memory_pipeline", None)
    else:
        out["memory_pipeline"] = sec
    return out


def describe(mp: Optional[Mapping]) -> str:
    if not mp:
        return "V1 (historical path)"
    p2 = " + P2" if mp.get("p2") else ""
    if mp.get("dense"):
        return f"V2D hybrid{p2} (fallback {mp.get('dense_fallback', 'v2l').upper()}{p2}), reader {mp.get('reader_version')}"
    return f"V2L lexical{p2}, reader {mp.get('reader_version')}"


def active_pipeline(mp: Optional[Mapping], dense_active: bool) -> str:
    """The memory pipeline that actually runs, as one label (``ACTIVE_MEMORY_PIPELINE``): ``V2D_P2`` | ``V2L_P2_FALLBACK`` (dense configured
    but unavailable) | ``V2D`` | ``V2L_FALLBACK`` | ``V2L`` (lexical by configuration, e.g. research-final) | ``V1`` (no section)."""
    if not mp:
        return "V1"
    p2 = "_P2" if mp.get("p2") else ""
    if mp.get("dense"):
        return f"V2D{p2}" if dense_active else f"V2L{p2}_FALLBACK"
    return f"V2L{p2}"
