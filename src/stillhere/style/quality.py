"""Pilot quality checks: is there a stable, distinguishable style signal that is not just topic?

* ``attribution``: leave-one-out nearest-centroid authorship attribution. With style features (z-scored on the training
  personas only) it measures style signal; with content bigrams (style markers removed) it measures topic signal. A style
  signal that survives on content-controlled replies (every speaker saying the same facts) cannot be topic.
* ``cross_kind``: centroids from one kind of message (natural chat), tested on the other (controlled content): style that
  carries across situations.
* plus lexical diversity, length and bubble distributions, near-duplicates and fact preservation.

Chance level is 1 / number of personas. These are pilot diagnostics on hand-written data, not evidence about any model.
"""
from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Mapping, Sequence

from .features import FEATURE_NAMES, MARKERS, char_grams, content_profile, cosine, jaccard, message_features


# "style_nolex": the style view without any lexicon family (punctuation, symbols, length, splitting only). The lexicon was written by
# the same hand as the pilot personas, so a signal that survives without it is not just the marker list finding its own words.
NOLEX_FEATURES = [k for k in FEATURE_NAMES if k not in MARKERS]


def _zscore(rows: Mapping[str, Dict[str, float]], basis_ids: Sequence[str], names: Sequence[str] = FEATURE_NAMES) -> Dict[str, List[float]]:
    basis = [rows[i] for i in basis_ids]
    mu = {k: statistics.fmean(v[k] for v in basis) for k in names}
    sd = {k: (statistics.pstdev([v[k] for v in basis]) or 1.0) for k in names}
    return {i: [(v[k] - mu[k]) / sd[k] for k in names] for i, v in rows.items()}


def _centroid(vectors: Sequence[Sequence[float]]) -> List[float]:
    return [statistics.fmean(col) for col in zip(*vectors)] if vectors else []


def attribution(messages: Sequence[Mapping], *, view: str, basis_personas: Sequence[str], subset: Callable[[Mapping], bool] = lambda m: True) -> dict:
    """Leave-one-out: each message of ``subset`` against the centroids of every person's other messages of ``subset``."""
    msgs = [m for m in messages if subset(m)]
    persons = sorted({m["persona_id"] for m in msgs})
    if view in ("style", "style_nolex"):
        rows = {m["message_id"]: message_features(m["bubbles"]) for m in messages}
        vec = _zscore(rows, [m["message_id"] for m in messages if m["persona_id"] in set(basis_personas)], FEATURE_NAMES if view == "style" else NOLEX_FEATURES)
        sim = cosine
        rep = lambda m: vec[m["message_id"]]
    else:
        bags = {m["message_id"]: content_profile(m["bubbles"]) for m in messages}
        vocab = sorted({g for b in bags.values() for g in b})
        index = {g: i for i, g in enumerate(vocab)}
        rep_cache = {}
        for mid, bag in bags.items():
            v = [0.0] * len(vocab)
            for g, c in bag.items():
                v[index[g]] = float(c)
            rep_cache[mid] = v
        sim = cosine
        rep = lambda m: rep_cache[m["message_id"]]
    correct, total = 0, 0
    confusion = Counter()
    for m in msgs:
        best, best_s = None, -2.0
        for p in persons:
            others = [rep(x) for x in msgs if x["persona_id"] == p and x["message_id"] != m["message_id"]]
            if not others:
                continue
            s = sim(rep(m), _centroid(others))
            if s > best_s:
                best, best_s = p, s
        total += 1
        correct += best == m["persona_id"]
        if best != m["persona_id"]:
            confusion[(m["persona_id"], best)] += 1
    return {"view": view, "messages": total, "personas": len(persons), "accuracy": round(correct / total, 4) if total else None, "correct": correct,
            "chance": round(1 / len(persons), 4) if persons else None, "top_confusions": [[a, b, n] for (a, b), n in confusion.most_common(5)]}


def cross_kind(messages: Sequence[Mapping], *, basis_personas: Sequence[str], train_kind: str = "natural", test_kind: str = "controlled") -> dict:
    rows = {m["message_id"]: message_features(m["bubbles"]) for m in messages}
    vec = _zscore(rows, [m["message_id"] for m in messages if m["persona_id"] in set(basis_personas)])
    persons = sorted({m["persona_id"] for m in messages})
    cents = {p: _centroid([vec[m["message_id"]] for m in messages if m["persona_id"] == p and m["kind"] == train_kind]) for p in persons}
    test = [m for m in messages if m["kind"] == test_kind]
    correct = sum(max(persons, key=lambda p: cosine(vec[m["message_id"]], cents[p])) == m["persona_id"] for m in test)
    consistency = {}
    for p in persons:
        other = _centroid([vec[m["message_id"]] for m in messages if m["persona_id"] == p and m["kind"] == test_kind])
        consistency[p] = round(cosine(cents[p], other), 3)
    return {"train_kind": train_kind, "test_kind": test_kind, "accuracy": round(correct / len(test), 4) if test else None, "correct": correct, "messages": len(test),
            "chance": round(1 / len(persons), 4), "per_persona_centroid_cosine": consistency}


def distinct_n(texts: Sequence[str], n: int = 2) -> float:
    grams = [t[i:i + n] for t in texts for i in range(len(t) - n + 1)]
    return round(len(set(grams)) / len(grams), 4) if grams else 0.0


def surface(messages: Sequence[Mapping]) -> dict:
    by = defaultdict(list)
    for m in messages:
        by[m["persona_id"]].append(m)
    per = {}
    for p, ms in sorted(by.items()):
        lens = [len("".join(m["bubbles"])) for m in ms]
        per[p] = {"messages": len(ms), "mean_chars": round(statistics.fmean(lens), 1), "sd_chars": round(statistics.pstdev(lens), 1),
                  "bubbles": dict(sorted(Counter(len(m["bubbles"]) for m in ms).items())), "distinct_2": distinct_n(["".join(m["bubbles"]) for m in ms])}
    means = [v["mean_chars"] for v in per.values()]
    return {"per_persona": per, "mean_chars_range": [min(means), max(means)], "corpus_distinct_2": distinct_n(["".join(m["bubbles"]) for m in messages])}


def duplicates(messages: Sequence[Mapping], *, within: float = 0.5, across: float = 0.6) -> dict:
    same, template = [], []
    for i in range(len(messages)):
        for j in range(i + 1, len(messages)):
            a, b = messages[i], messages[j]
            s = jaccard(char_grams("".join(a["bubbles"])), char_grams("".join(b["bubbles"])))
            if a["persona_id"] == b["persona_id"] and s >= within:
                same.append([a["message_id"], b["message_id"], round(s, 3)])
            elif a["persona_id"] != b["persona_id"] and a["item_id"] == b["item_id"] and s >= across:
                template.append([a["message_id"], b["message_id"], round(s, 3)])
    n = len(messages)
    return {"within_persona_near_duplicates": same, "cross_persona_same_item_near_duplicates": template, "duplicate_rate": round(len(same) / n, 4) if n else 0.0}
