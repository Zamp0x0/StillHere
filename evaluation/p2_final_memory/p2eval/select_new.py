"""The fresh person and its frozen 100 QA (deterministic; never by model results; referenced question ids excluded)."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from lbpilot import select as LS

from . import protocol as P

HERE = Path(__file__).resolve().parent
EXCLUDED = {"于晓雯": HERE / "excluded_qids_fresh_person.json"}


def excluded(sample_id: str) -> set:
    p = EXCLUDED.get(sample_id)
    if p is None:
        raise ValueError(f"no reference audit for {sample_id}: re-scan before using this person")
    return set(json.loads(p.read_text(encoding="utf-8")))


def choose(data: Sequence[Mapping]) -> dict:
    considered = []
    for i, s in enumerate(data):
        if s["sample_id"] not in P.FINAL_TEST_PERSONS or s["sample_id"] in P.NEVER:
            continue
        try:
            ex = excluded(s["sample_id"])
        except ValueError as e:
            considered.append({"position": i, "sample_id": s["sample_id"], "rejected": [str(e)]})
            continue
        pool = [q for q in s["qa"] if q["question_id"] not in ex]
        types = {t for q in pool for t in LS.types_of(q)}
        why = []
        if len(pool) < P.SAMPLE["n"]:
            why.append(f"only {len(pool)} unreferenced QA")
        missing = [t for t in LS.P.OFFICIAL_TYPES if t not in types]
        if missing:
            why.append(f"missing types {missing}")
        considered.append({"position": i, "sample_id": s["sample_id"], "slug": LS.slug(s), "qa": len(s["qa"]), "excluded": len(ex & {q["question_id"] for q in s["qa"]}),
                           "eligible": len(pool), "rejected": why})
        if not why:
            return {"rule": P.NEW_PERSON_RULE, "position": i, "sample_id": s["sample_id"], "slug": LS.slug(s), "considered": considered}
    raise ValueError("NO_ELIGIBLE_PERSON")


def freeze(data: Sequence[Mapping]) -> dict:
    who = choose(data)
    person = data[who["position"]]
    ex = excluded(person["sample_id"])
    pool = [q for q in person["qa"] if q["question_id"] not in ex]
    strata = {}
    for q in pool:
        strata.setdefault(LS.types_of(q)[0], []).append(q)
    alloc = LS._alloc({k: len(v) for k, v in strata.items()}, P.SAMPLE["n"])
    chosen = []
    for k in sorted(strata):
        ranked = sorted(strata[k], key=lambda q: hashlib.sha256(f"{P.SAMPLE['seed']}|{q['question_id']}".encode()).hexdigest())
        chosen.extend(ranked[: alloc[k]])
    chosen.sort(key=lambda q: (q["ask_time"], q["question_id"]))
    tc = Counter(t for q in chosen for t in LS.types_of(q))
    return {"schema": "stillhere.p2_final_memory.frozen_100qa/1.0", "protocol_sha256": P.P2_PROTOCOL_SHA256, "dataset": P.DATASET, "person": who,
            "excluded_referenced_ids": sorted(ex), "question_ids": [q["question_id"] for q in chosen],
            "allocation_by_primary_type": dict(sorted(alloc.items())), "type_counts_multilabel": {t: tc.get(t, 0) for t in LS.P.OFFICIAL_TYPES},
            "ask_time_range": [chosen[0]["ask_time"], chosen[-1]["ask_time"]], "distinct_ask_dates": len({q["ask_time"][:10] for q in chosen}),
            "qa_sha256": hashlib.sha256(json.dumps(chosen, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()}


def selected(data, frozen):
    person = data[frozen["person"]["position"]]
    if person["sample_id"] != frozen["person"]["sample_id"]:
        raise ValueError("FROZEN_PERSON_MISMATCH")
    by = {q["question_id"]: q for q in person["qa"]}
    qs = [by[i] for i in frozen["question_ids"]]
    if hashlib.sha256(json.dumps(qs, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest() != frozen["qa_sha256"]:
        raise ValueError("FROZEN_QA_SHA_MISMATCH")
    if set(frozen["question_ids"]) & set(frozen["excluded_referenced_ids"]):
        raise ValueError("REFERENCED_QA_IN_SAMPLE")
    return person, qs
