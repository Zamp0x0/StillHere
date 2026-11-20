"""TRAIN / EVAL leakage audit. Hard checks (must be 0) and informational overlaps (reported with counts, never hidden).

Hard:
  * persona disjointness between every pair of splits (synthetic personas and real authors);
  * B3 HOLDOUT: no HOLDOUT author and no HOLDOUT source episode in any D4 split; no long HOLDOUT target text in TRAIN;
  * B3 DEV cases are evaluation only: no B3 DEV source episode / long target in TRAIN;
  * isolated-bank entity values used in evaluation evidence never appear in TRAIN evidence;
  * question template families (domain + question frame) of evaluation splits never appear in TRAIN;
  * HOLDOUT-only domains appear only in FROZEN_HOLDOUT;
  * the old 200-item safety probe (HISTORICAL_REGRESSION only) shares no question with any D4 split;
  * real example / profile sources strictly precede the target (example time leakage), and no example equals its target.
"""
from __future__ import annotations

import hashlib
import itertools
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

from .domains import HOLDOUT_ONLY_DOMAINS
from .lexicon import BANKS, SHARED_BANKS

LONG_TARGET_CHARS = 8


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _h(s: str) -> str:
    return hashlib.sha256(_norm(s).encode("utf-8")).hexdigest()


def _isolated_values() -> Set[str]:
    shared = {v for b in SHARED_BANKS for vs in BANKS[b].values() for v in vs}
    return {v for name, pools in BANKS.items() if name not in SHARED_BANKS for vs in pools.values() for v in vs} - shared


def _target(e: Mapping) -> str:
    return e.get("target") if e.get("target") is not None else e.get("reference_target", "")


def _qtemplate(e: Mapping) -> Optional[str]:
    if e["source"] != "SYNTH":
        return None
    return e["question_template"]


def audit(splits: Mapping[str, Sequence[Mapping]], *, b3_holdout: Sequence[Mapping] = (), b3_holdout_author_keys: Iterable[str] = (),
          old_probe_questions: Sequence[str] = ()) -> dict:
    hard: Dict[str, int] = {}
    info: Dict[str, object] = {}
    personas = {k: {e["persona_id"] for e in v} for k, v in splits.items()}
    for a, b in itertools.combinations(sorted(splits), 2):
        hard[f"persona_overlap:{a}~{b}"] = len(personas[a] & personas[b])
    hk = set(b3_holdout_author_keys)
    for k in splits:
        hard[f"b3_holdout_author_in:{k}"] = len(personas[k] & hk)
    hold_src = {r["id"] for r in b3_holdout}
    hold_long = {_h("\n".join(r["target_bubbles"])) for r in b3_holdout if len(_norm("".join(r["target_bubbles"]))) >= LONG_TARGET_CHARS}
    train = splits.get("TRAIN", [])
    train_src = {e.get("source_episode_id") for e in train if e.get("source_episode_id")}
    train_long = {_h(_target(e)) for e in train if len(_norm(_target(e))) >= LONG_TARGET_CHARS}
    for k, v in splits.items():
        hard[f"b3_holdout_source_episode_in:{k}"] = sum(1 for e in v if e.get("source_episode_id") in hold_src)
    hard["b3_holdout_long_target_in:TRAIN"] = len(train_long & hold_long)
    for k, v in splits.items():
        if k == "TRAIN":
            continue
        real = [e for e in v if e["source"] != "SYNTH"]
        hard[f"eval_source_episode_in_TRAIN:{k}"] = sum(1 for e in real if e.get("source_episode_id") in train_src)
        info[f"eval_long_target_in_TRAIN:{k}"] = sum(1 for e in v if len(_norm(_target(e))) >= LONG_TARGET_CHARS and _h(_target(e)) in train_long)
        info[f"eval_short_target_also_in_TRAIN:{k}"] = sum(1 for e in v if len(_norm(_target(e))) < LONG_TARGET_CHARS
                                                           and _norm(_target(e)) in {_norm(_target(t)) for t in train})
    iso = _isolated_values()
    train_vals = {x["value"] for e in train for x in (e.get("evidence") or [])}
    train_q = {_qtemplate(e) for e in train} - {None}
    train_qtext = {_norm(e["conversation"][-1]["text"]) for e in train if e["source"] == "SYNTH"}
    for k, v in splits.items():
        if k == "TRAIN":
            continue
        vals = {x["value"] for e in v for x in (e.get("evidence") or [])}
        hard[f"isolated_entity_overlap:{k}"] = len((vals & iso) & train_vals)
        info[f"generated_value_overlap:{k}"] = len((vals - iso) & train_vals)  # dates, prices, clock times: generated, not isolated
        hard[f"question_template_overlap:{k}"] = len(({_qtemplate(e) for e in v} - {None}) & train_q)
        info[f"synthetic_question_text_overlap:{k}"] = sum(1 for e in v if e["source"] == "SYNTH" and _norm(e["conversation"][-1]["text"]) in train_qtext)
    for k, v in splits.items():
        if k != "FROZEN_HOLDOUT":
            hard[f"holdout_only_domain_in:{k}"] = sum(1 for e in v for x in (e.get("evidence") or []) if x["predicate"] in HOLDOUT_ONLY_DOMAINS)
    oq = {_norm(q) for q in old_probe_questions}
    for k, v in splits.items():
        hard[f"old_safety_probe_question_in:{k}"] = sum(1 for e in v if _norm(e["conversation"][-1]["text"]) in oq)
    # example time leakage (real authors): profile / example sources precede the target; examples never equal the target
    tl = 0
    eq = 0
    for k, v in splits.items():
        for e in v:
            if e["source"] != "SYNTH" and e.get("history_max_key") and e.get("target_key") and str(e["history_max_key"]) > str(e["target_key"]):
                tl += 1
            if any(_norm(x) == _norm(_target(e)) for x in e.get("examples", [])):
                eq += 1
    hard["example_after_target"] = tl
    hard["example_equals_target"] = eq
    # synthetic examples are fact-free: outside EXAMPLE_FACT_TRAP no example names a required or forbidden value of its own episode
    fx = 0
    for v in splits.values():
        for e in v:
            if e["source"] == "SYNTH" and e["category"] != "EXAMPLE_FACT_TRAP":
                key = e["answer_state"]
                vals = [x for x in key.get("required_values", []) + key.get("forbidden_values", []) if len(x) >= 2]
                if any(val in ex for ex in e["examples"] for val in vals):
                    fx += 1
    info["synthetic_examples_naming_episode_values_outside_trap"] = fx
    ok = all(v == 0 for v in hard.values())
    return {"pass": ok, "hard": hard, "informational": info,
            "rule": "every hard count must be 0; informational overlaps are expected (short generic replies, generated dates/prices) and reported"}
