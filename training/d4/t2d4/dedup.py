"""Exact and near-duplicate detection (MinHash + LSH over character 3-grams; pure Python, deterministic) and diversity statistics.

Near-duplicate unit = one supervised example: the conversation the model sees plus the evidence values plus the target. Inside a deliberate
contrast group (PERSONA_VARIATION / COUNTERFACTUAL / PARAPHRASE) similarity is the point and is reported, not removed; across groups a pair
with Jaccard >= NEAR_DUP_JACCARD is a near-duplicate and the later episode is dropped from TRAIN.
"""
from __future__ import annotations

import collections
import hashlib
import math
import re
from typing import Dict, Iterable, List, Sequence, Tuple

NUM_PERM = 64
BANDS = 16
NEAR_DUP_JACCARD = 0.9
_MASK = (1 << 61) - 1
_PERM = [(int(hashlib.sha256(f"a{i}".encode()).hexdigest(), 16) % _MASK | 1, int(hashlib.sha256(f"b{i}".encode()).hexdigest(), 16) % _MASK)
         for i in range(NUM_PERM)]


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def shingles(text: str, k: int = 3) -> set:
    t = _norm(text)
    return {t[i:i + k] for i in range(max(1, len(t) - k + 1))} if t else set()


def minhash(sh: Iterable[str]) -> Tuple[int, ...]:
    hs = [int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") for s in sh] or [0]
    return tuple(min((a * h + b) % _MASK for h in hs) for a, b in _PERM)


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def unit_text(e: dict) -> str:
    conv = "".join(t["text"] for t in e["conversation"])
    ev = "|".join(x["value"] for x in (e.get("evidence") or []))
    return conv + "#" + ev + "#" + e.get("target", e.get("reference_target", ""))


def near_duplicates(eps: Sequence[dict], threshold: float = NEAR_DUP_JACCARD) -> Dict[str, object]:
    """Returns pairs (i, j, jaccard, same_group) with jaccard >= threshold, found by LSH and verified exactly."""
    sh = [shingles(unit_text(e)) for e in eps]
    sig = [minhash(s) for s in sh]
    r = NUM_PERM // BANDS
    buckets = collections.defaultdict(list)
    for i, s in enumerate(sig):
        for b in range(BANDS):
            buckets[(b, s[b * r:(b + 1) * r])].append(i)
    cand = set()
    for lst in buckets.values():
        if 1 < len(lst) <= 200:
            for x in range(len(lst)):
                for y in range(x + 1, len(lst)):
                    cand.add((lst[x], lst[y]))
    pairs = []
    for i, j in sorted(cand):
        jac = jaccard(sh[i], sh[j])
        if jac >= threshold:
            pairs.append((i, j, round(jac, 4), eps[i].get("group_id") == eps[j].get("group_id")))
    return {"pairs": pairs, "candidates": len(cand)}


def exact_duplicates(eps: Sequence[dict]) -> Dict[str, int]:
    full = collections.Counter(hashlib.sha256((e["system"] + "\x1f" + e["user"] + "\x1f" + e["target"]).encode()).hexdigest() for e in eps)
    tgt = collections.Counter(_norm(e["target"]) for e in eps)
    io = collections.Counter(hashlib.sha256((e["user"] + "\x1f" + e["target"]).encode()).hexdigest() for e in eps)
    return {"duplicate_rows": sum(c - 1 for c in full.values() if c > 1),
            "duplicate_targets": sum(c - 1 for c in tgt.values() if c > 1),
            "max_target_repeat": max(tgt.values()) if tgt else 0,
            "duplicate_conversation_target_pairs": sum(c - 1 for c in io.values() if c > 1)}


def distinct_n(texts: Sequence[str], n: int) -> float:
    grams, total = set(), 0
    for t in texts:
        t = _norm(t)
        g = [t[i:i + n] for i in range(len(t) - n + 1)]
        grams.update(g)
        total += len(g)
    return round(len(grams) / total, 4) if total else 0.0


def gini(counts: Sequence[int]) -> float:
    xs = sorted(counts)
    n = len(xs)
    if not n or not sum(xs):
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return round((2 * cum) / (n * sum(xs)) - (n + 1) / n, 4)


def entropy(counter: collections.Counter) -> float:
    tot = sum(counter.values())
    return round(-sum(c / tot * math.log2(c / tot) for c in counter.values() if c), 4) if tot else 0.0


def diversity(eps: Sequence[dict], near: Dict[str, object]) -> dict:
    syn = [e for e in eps if e["source"] == "SYNTH"]
    targets = [e["target"] for e in eps]
    per_persona = collections.Counter(e["persona_id"] for e in eps)
    cross = [p for p in near["pairs"] if not p[3]]
    return {
        "template_family_count": len({e["template_family"] for e in syn}),
        "question_template_family_count": len({"|".join(e["template_family"].split("|")[:3]) for e in syn}),
        "unique_semantic_scenario_count": len({e["semantic_key"] for e in syn}),
        "near_duplicate_pairs_total": len(near["pairs"]),
        "near_duplicate_pairs_within_contrast_group": len(near["pairs"]) - len(cross),
        "near_duplicate_pairs_cross_group": len(cross),
        "near_duplicate_rate_cross_group": round(len({p[1] for p in cross}) / max(1, len(eps)), 4),
        "target_unique_ratio": round(len({_norm(t) for t in targets}) / max(1, len(targets)), 4),
        "target_distinct_1": distinct_n(targets, 1), "target_distinct_2": distinct_n(targets, 2), "target_distinct_3": distinct_n(targets, 3),
        "persona_count": len(per_persona), "episodes_per_persona_max": max(per_persona.values()) if per_persona else 0,
        "episodes_per_persona_gini": gini(list(per_persona.values())),
        "source_mix": dict(collections.Counter(e["source"] for e in eps)),
        "category_entropy_bits": entropy(collections.Counter(e["category"] for e in syn)),
        "domain_count": len({e["domain"] for e in syn}),
    }
