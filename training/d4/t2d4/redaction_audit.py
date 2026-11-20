"""MODEL_VISIBLE_REDACTED_VIEW audit: no PROTECTED / POLICY_UNCERTAIN raw value (or any 4-digit run of it) in any text the model can see.

Model-visible texts of a record: ``system``, ``user``, every ``ablations.<name>.system/user``, ``base_length.system/user``, and for TRAIN the
supervised ``target`` (a target that says a protected value would teach the leak). Raw values come from the restricted fields of the memory
state each prompt was rendered from: the record's ``evidence`` (non-ALLOWED items) and ``answer_state.protected_values`` for every prompt except
WRONG_MEMORY_STATE, which is checked against its sibling state (``ablation_answer_states.WRONG_MEMORY_STATE``). A counterfactual pair can hold
the same value as PROTECTED in one state and ALLOWED in the other; each prompt is judged by its own state.

A cross-record scan also looks for every protected value of the whole build in every model-visible text (informational: a short value such
as a health condition can legitimately be another persona's ALLOWED fact). The report never contains a value, only counts and episode ids.
"""
from __future__ import annotations

import collections
import re
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .evidence import from_json
from .disclosure_view import has_redaction_line, raw_values_in

TARGET_SPLITS = ("TRAIN",)


def model_visible(rec: Mapping) -> Iterable[Tuple[str, str]]:
    for k in ("system", "user"):
        if rec.get(k):
            yield k, rec[k]
    for name, m in (rec.get("ablations") or {}).items():
        for k in ("system", "user"):
            yield f"ablations.{name}.{k}", m[k]
    if rec.get("base_length"):
        for k in ("system", "user"):
            yield f"base_length.{k}", rec["base_length"][k]
    if rec.get("split") in TARGET_SPLITS and rec.get("target"):
        yield "target", rec["target"]


def _restricted(evidence, answer_state) -> Tuple[List, set]:
    ev = [e for e in (from_json(d) for d in (evidence or [])) if e.disclosure_policy != "ALLOWED"]
    return ev, set((answer_state or {}).get("protected_values") or [])


def restricted_for(rec: Mapping, prompt_name: str) -> Tuple[List, set]:
    if prompt_name.startswith("ablations.WRONG_MEMORY_STATE"):
        w = (rec.get("ablation_answer_states") or {}).get("WRONG_MEMORY_STATE") or {}
        return _restricted(w.get("evidence"), w.get("answer_state"))
    return _restricted(rec.get("evidence"), rec.get("answer_state"))


def restricted_evidence(rec: Mapping) -> List:
    out = restricted_for(rec, "system")[0]
    return out + restricted_for(rec, "ablations.WRONG_MEMORY_STATE.system")[0]


def audit(splits: Mapping[str, Sequence[Mapping]]) -> dict:
    per_split: Dict[str, dict] = {}
    failures: List[dict] = []
    all_values = set()
    for split, rows in splits.items():
        c = collections.Counter()
        fields = collections.Counter()
        for rec in rows:
            prompts = list(model_visible(rec))
            states = {n: restricted_for(rec, n) for n, _ in prompts}
            if not any(ev or pv for ev, pv in states.values()):
                continue
            c["records_with_restricted_values"] += 1
            c["restricted_entries"] += len(restricted_for(rec, "system")[0])
            bad = False
            for name, text in prompts:
                ev, pv = states[name]
                all_values |= {e.value for e in ev} | pv
                if raw_values_in(text, ev) or [v for v in pv if v and v in text]:
                    bad = True
                    fields[name.rsplit(".", 1)[0] if name.startswith(("ablations", "base_length")) else name] += 1
            if bad:
                c["records_with_visible_raw_value"] += 1
                failures.append({"episode_id": rec["episode_id"], "split": split})
            if any(has_redaction_line(t) for n, t in prompts):
                c["records_with_redaction_line"] += 1
        per_split[split] = {**dict(c), "visible_by_field": dict(fields)}
    # cross-record (informational): any restricted value of the build inside a prompt rendered from a state where it is not restricted here
    vals = sorted((v for v in all_values if len(v) >= 2), key=len, reverse=True)
    cross = collections.Counter()
    if vals:
        rx = re.compile("|".join(re.escape(v) for v in vals))
        for split, rows in splits.items():
            for rec in rows:
                for name, text in model_visible(rec):
                    if rx.search(text):
                        cross[split] += 1
                        break
    total_visible = sum(v.get("records_with_visible_raw_value", 0) for v in per_split.values())
    return {"view": "MODEL_VISIBLE_REDACTED_VIEW", "per_split": per_split, "distinct_restricted_values": len(all_values),
            "records_with_visible_raw_value": total_visible, "failures": failures[:50],
            "cross_record_value_matches_informational": dict(cross),
            "cross_record_note": "records where some restricted value of the build appears in a prompt: counterfactual siblings that state the same value as ALLOWED, or a value shared by chance; informational only",
            "rule": "raw value or any 4-digit run of it (ISO dates excluded) in system / user / ablations / base_length (and TRAIN target)",
            "pass": total_visible == 0}
