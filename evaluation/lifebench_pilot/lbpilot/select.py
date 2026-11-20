"""Person selection and the frozen 100-question sample (deterministic; no model result is ever read here)."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from . import protocol as P

QID = re.compile(r"^lifebench_v\d+_([a-z]+)_qa\d+$")
SESSION_KEY = re.compile(r"^session_(\d+)$")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_dataset(path, *, verify: bool = True) -> List[dict]:
    if verify:
        got = sha256_file(path)
        if got != P.DATASET["sha256"]:
            raise ValueError(f"DATASET_SHA_MISMATCH: {got} != {P.DATASET['sha256']} (revision {P.DATASET['revision'][:8]})")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def slug(sample: Mapping) -> str:
    found = {QID.match(q["question_id"]).group(1) for q in sample["qa"] if QID.match(q.get("question_id", ""))}
    if len(found) != 1:
        raise ValueError(f"cannot derive one person slug for {sample.get('sample_id')!r}")
    return found.pop()


def types_of(q: Mapping) -> List[str]:
    t = q.get("question_type")
    return [str(x) for x in (t if isinstance(t, list) else [t]) if x]


def choose_person(data: Sequence[Mapping]) -> dict:
    tried = []
    for i, s in enumerate(data):
        present = {t for q in s["qa"] for t in types_of(q)}
        reasons = []
        if s["sample_id"] in P.EXCLUDED_PERSONS:
            reasons.append("used by an earlier StillHere LifeBench run")
        if len(s["qa"]) < P.SAMPLE["n"]:
            reasons.append(f"only {len(s['qa'])} QA")
        missing = [t for t in P.OFFICIAL_TYPES if t not in present]
        if missing:
            reasons.append(f"missing types {missing}")
        tried.append({"position": i, "sample_id": s["sample_id"], "slug": slug(s), "qa": len(s["qa"]), "rejected": reasons})
        if not reasons:
            return {"rule": P.PERSON_RULE, "position": i, "sample_id": s["sample_id"], "slug": slug(s), "qa_total": len(s["qa"]), "considered": tried}
    raise ValueError("NO_ELIGIBLE_PERSON")


def _alloc(counts: Mapping[str, int], n: int) -> Dict[str, int]:
    total = sum(counts.values())
    base = {k: max(1, int(n * c / total)) for k, c in counts.items()}
    rem = sorted(counts, key=lambda k: (-(n * counts[k] / total - int(n * counts[k] / total)), k))
    i = 0
    while sum(base.values()) < n:
        k = rem[i % len(rem)]
        if base[k] < counts[k]:
            base[k] += 1
        i += 1
    while sum(base.values()) > n:                  # only when the "at least 1" floor overshoots: take from the largest strata
        k = max(base, key=lambda x: (base[x], x))
        base[k] -= 1
    return base


def sample_questions(person: Mapping) -> dict:
    qa = person["qa"]
    strata: Dict[str, List[dict]] = {}
    for q in qa:
        strata.setdefault(types_of(q)[0], []).append(q)
    alloc = _alloc({k: len(v) for k, v in strata.items()}, P.SAMPLE["n"])
    seed = P.SAMPLE["seed"]
    chosen = []
    for k in sorted(strata):
        ranked = sorted(strata[k], key=lambda q: hashlib.sha256(f"{seed}|{q['question_id']}".encode()).hexdigest())
        chosen.extend(ranked[: alloc[k]])
    chosen.sort(key=lambda q: (q["ask_time"], q["question_id"]))
    ids = [q["question_id"] for q in chosen]
    if len(set(ids)) != len(ids) or len(ids) != P.SAMPLE["n"]:
        raise ValueError("sample size / uniqueness")
    type_counts = Counter(t for q in chosen for t in types_of(q))
    return {"question_ids": ids, "allocation_by_primary_type": dict(sorted(alloc.items())),
            "person_counts_by_primary_type": {k: len(v) for k, v in sorted(strata.items())},
            "type_counts_multilabel": {t: type_counts.get(t, 0) for t in P.OFFICIAL_TYPES},
            "ask_time_range": [chosen[0]["ask_time"], chosen[-1]["ask_time"]], "distinct_ask_dates": len({q["ask_time"][:10] for q in chosen}),
            "qa_sha256": hashlib.sha256(json.dumps(chosen, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
            "questions": [{"question_id": q["question_id"], "ask_time": q["ask_time"], "question_type": types_of(q), "evidence_ids": list(q.get("evidence") or [])}
                          for q in chosen]}


def freeze(dataset_path) -> dict:
    return freeze_data(load_dataset(dataset_path))


def freeze_data(data: Sequence[Mapping]) -> dict:
    who = choose_person(data)
    person = data[who["position"]]
    smp = sample_questions(person)
    return {"schema": "stillhere.lifebench_pilot.frozen_100qa/1.0", "protocol_sha256": P.PROTOCOL_SHA256, "dataset": P.DATASET, "person": who, **smp}


def selected(data: Sequence[Mapping], frozen: Mapping):
    """(person sample, the 100 QA records in frozen order); verifies ids and the QA sha against the frozen manifest."""
    person = data[frozen["person"]["position"]]
    if person["sample_id"] != frozen["person"]["sample_id"]:
        raise ValueError("FROZEN_PERSON_MISMATCH")
    by = {q["question_id"]: q for q in person["qa"]}
    qs = [by[i] for i in frozen["question_ids"]]
    if hashlib.sha256(json.dumps(qs, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest() != frozen["qa_sha256"]:
        raise ValueError("FROZEN_QA_SHA_MISMATCH")
    return person, qs
