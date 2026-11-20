"""Language-agnostic style-vs-topic learnability audit for candidate datasets (T2-D0.1).

The T2-D0 profiler lexicon is Chinese and was co-designed with the pilot, so it cannot be the measuring instrument for new, multilingual
sources. This module uses the standard topic-robust authorship method instead: **text distortion** (Stamatatos 2017). Only the ``k``
most frequent tokens of the *basis* (train-persona) text are kept; every other token is masked. What survives is function words,
particles, punctuation, symbols and length/shape: style. The masked-out tokens themselves form the **topic** view.

* token: a word (lower-cased) for space-delimited text; one character for CJK / kana;
* ``style`` view: token 1-3-grams over the distorted sequence;
* ``topic`` view: bag of the masked (content) tokens;
* ``shape`` view: the T2-D0 lexicon-free features (punctuation, length, splitting, symbols) as a language-agnostic sanity view.

Attribution is nearest centroid by cosine. Leave-one-out excludes the scored message; **leave-group-out** additionally removes every
message of the same group (a conversation, a topic, an item) from **every** centroid, so the score cannot come from a shared
conversation or subject. Chance is 1 / personas. Bootstrap over personas gives an interval.
"""
from __future__ import annotations

import hashlib
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from .features import FEATURE_NAMES, MARKERS, message_features

CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
TOKEN = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]|[A-Za-zÀ-ɏ']+|\d+|[^\sA-Za-z\d぀-ヿ㐀-䶿一-鿿가-힯]")
MASK = "█"
SHAPE_FEATURES = [k for k in FEATURE_NAMES if k not in MARKERS]


def tokens(text: str) -> List[str]:
    return [t.lower() for t in TOKEN.findall(text)]


def keep_set(texts: Sequence[str], k: int) -> set:
    c = Counter(t for x in texts for t in tokens(x))
    return {t for t, _ in c.most_common(k)}


def distort(text: str, keep: set) -> List[str]:
    return [t if t in keep else ("<d>" if t.isdigit() else MASK) for t in tokens(text)]


def style_vec(text: str, keep: set, n_max: int = 3) -> Dict[str, float]:
    seq = ["<s>"] + distort(text, keep) + ["</s>"]
    v: Dict[str, float] = Counter()
    for n in range(1, n_max + 1):
        for i in range(len(seq) - n + 1):
            g = " ".join(seq[i:i + n])
            if g != MASK:
                v[g] += 1.0
    return v


def topic_vec(text: str, keep: set) -> Dict[str, float]:
    return Counter(t for t in tokens(text) if t not in keep and not t.isdigit() and (CJK.match(t) or t.isalpha()))


def _norm(v: Mapping[str, float]) -> float:
    return math.sqrt(sum(x * x for x in v.values()))


def _cos(a: Mapping[str, float], b: Mapping[str, float], nb: float) -> float:
    if not a or not nb:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    na = _norm(a)
    return sum(x * b.get(k, 0.0) for k, x in a.items()) / (na * nb) if na else 0.0


def _tfidf(vecs: Sequence[Dict[str, float]], basis_idx: Sequence[int]) -> List[Dict[str, float]]:
    df = Counter(k for i in basis_idx for k in vecs[i])
    n = max(len(basis_idx), 1)
    idf = {k: math.log((1 + n) / (1 + c)) + 1.0 for k, c in df.items()}
    return [{k: (1 + math.log(x)) * idf.get(k, math.log(1 + n) + 1.0) for k, x in v.items() if x > 0} for v in vecs]


def _shape_vecs(msgs, basis_idx) -> List[Dict[str, float]]:
    rows = [message_features(m["bubbles"] if "bubbles" in m else [m["text"]]) for m in msgs]
    mu = {k: statistics.fmean(rows[i][k] for i in basis_idx) for k in SHAPE_FEATURES}
    sd = {k: (statistics.pstdev([rows[i][k] for i in basis_idx]) or 1.0) for k in SHAPE_FEATURES}
    return [{k: (r[k] - mu[k]) / sd[k] for k in SHAPE_FEATURES} for r in rows]


def vectors(msgs: Sequence[Mapping], view: str, *, basis_personas: Sequence[str], k: int = 300) -> List[Dict[str, float]]:
    basis = set(basis_personas)
    basis_idx = [i for i, m in enumerate(msgs) if m["persona_id"] in basis]
    if view == "shape":
        return _shape_vecs(msgs, basis_idx)
    keep = keep_set([msgs[i]["text"] for i in basis_idx], k)
    raw = [style_vec(m["text"], keep) if view == "style" else topic_vec(m["text"], keep) for m in msgs]
    return _tfidf(raw, basis_idx)


def attribution(msgs: Sequence[Mapping], view: str, *, basis_personas: Sequence[str], group: Optional[str] = None, k: int = 300,
                subset: Callable[[Mapping], bool] = lambda m: True, boot: int = 200, seed: str = "t2d01") -> dict:
    """``msgs``: [{message_id, persona_id, text, <group key>...}]. ``group``: key whose equal values are excluded from every centroid when
    scoring (leave-group-out). Returns accuracy, chance, a persona-bootstrap 95% interval and per-persona accuracy."""
    vec = vectors(msgs, view, basis_personas=basis_personas, k=k)
    persons = sorted({m["persona_id"] for m in msgs})
    sums: Dict[str, Dict[str, float]] = {p: defaultdict(float) for p in persons}
    by_group: Dict[tuple, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for m, v in zip(msgs, vec):
        for kk, x in v.items():
            sums[m["persona_id"]][kk] += x
            if group:
                by_group[(m["persona_id"], m[group])][kk] += x
    gpersons = defaultdict(set)
    if group:
        for m in msgs:
            gpersons[m[group]].add(m["persona_id"])
    sq = {p: sum(x * x for x in c.values()) for p, c in sums.items()}

    def dot(a: Mapping[str, float], b: Mapping[str, float]) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(x * b.get(kk, 0.0) for kk, x in a.items())

    own_n = Counter(m["persona_id"] for m in msgs)
    own_group_n = Counter((m["persona_id"], m[group]) for m in msgs) if group else Counter()
    hits = defaultdict(list)
    not_estimable = 0
    for m, v in zip(msgs, vec):
        if not subset(m):
            continue
        left = own_n[m["persona_id"]] - (own_group_n[(m["persona_id"], m[group])] if group else 1)
        if left < 1:  # nothing of this persona remains outside the excluded group: the question cannot be asked
            not_estimable += 1
            continue
        nv = _norm(v)
        best, best_s = None, -9.0
        for p in persons:
            c = sums[p]
            sub = None
            if group and p in gpersons[m[group]]:
                sub = by_group[(p, m[group])]  # includes v itself when p is the author
            elif p == m["persona_id"]:
                sub = v
            num, den2 = dot(v, c), sq[p]
            if sub is not None:  # cos(v, c - sub) without materialising c - sub
                num -= dot(v, sub)
                den2 += -2.0 * dot(sub, c) + sum(x * x for x in sub.values())
            s = num / (nv * math.sqrt(den2)) if nv and den2 > 1e-12 else -1.0
            if s > best_s:
                best, best_s = p, s
        hits[m["persona_id"]].append(best == m["persona_id"])
    per = {p: sum(h) / len(h) for p, h in hits.items() if h}
    n = sum(len(h) for h in hits.values())
    acc = sum(sum(h) for h in hits.values()) / n if n else None
    rng = random.Random(seed)
    ps = sorted(per)
    bs = []
    for _ in range(boot if ps else 0):
        draw = [rng.choice(ps) for _ in ps]
        bs.append(statistics.fmean(per[p] for p in draw))
    bs.sort()
    ci = [round(bs[int(0.025 * len(bs))], 4), round(bs[int(0.975 * len(bs)) - 1], 4)] if bs else None
    return {"view": view, "group_excluded": group, "messages": n, "personas": len(persons), "accuracy": round(acc, 4) if acc is not None else None,
            "chance": round(1 / len(persons), 4) if persons else None, "ci95_persona_bootstrap": ci, "keep_k": k if view != "shape" else None,
            "not_estimable_messages": not_estimable, "status": "ESTIMATED" if n else "NOT_ESTIMABLE"}


def summary(msgs: Sequence[Mapping]) -> dict:
    by = defaultdict(list)
    for m in msgs:
        by[m["persona_id"]].append(m)
    lens = [len(m["text"]) for m in msgs]
    lens_sorted = sorted(lens)
    return {"personas": len(by), "messages": len(msgs), "messages_per_persona_min": min(map(len, by.values())) if by else 0,
            "chars_mean": round(statistics.fmean(lens), 1) if lens else 0, "chars_p95": lens_sorted[int(0.95 * (len(lens) - 1))] if lens else 0,
            "short_le_10_chars": round(sum(1 for x in lens if x <= 10) / len(lens), 3) if lens else 0}


def template_rate(texts: Sequence[str], n: int = 4, top: int = 20) -> dict:
    """Generator fingerprint: how much of the corpus is covered by its most frequent opening n-grams, and the distinct-2 of tokens."""
    openings = Counter(" ".join(tokens(t)[:n]) for t in texts if len(tokens(t)) >= n)
    total = sum(openings.values())
    toks = [tokens(t) for t in texts]
    bigrams = [tuple(x[i:i + 2]) for x in toks for i in range(len(x) - 1)]
    return {"top_openings_share": round(sum(c for _, c in openings.most_common(top)) / total, 4) if total else None,
            "top_openings": [[g, c] for g, c in openings.most_common(8)], "distinct_2": round(len(set(bigrams)) / len(bigrams), 4) if bigrams else None}


class StyleAttributor:
    """Fitted text-distortion style model for scoring generated text (T2 PoC metrics). Fit only on reference utterances of the persons being
    scored (never on prompts or support shown to the model). ``scores(text)`` -> cosine similarity to every person's centroid."""

    def __init__(self, references: Sequence[Mapping], *, k: int = 50, view: str = "style"):
        assert view in ("style", "topic")
        self.view, self.k = view, k
        self.keep = keep_set([m["text"] for m in references], k)
        raw = [self._raw(m["text"]) for m in references]
        df = Counter(t for v in raw for t in v)
        n = max(len(raw), 1)
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self.default_idf = math.log(1 + n) + 1.0
        cents: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for m, v in zip(references, raw):
            for t, x in self._weight(v).items():
                cents[m["persona_id"]][t] += x
        self.centroids = {p: dict(c) for p, c in cents.items()}
        self.norms = {p: _norm(c) for p, c in self.centroids.items()}
        self.persons = sorted(self.centroids)

    def _raw(self, text: str) -> Dict[str, float]:
        return style_vec(text, self.keep) if self.view == "style" else topic_vec(text, self.keep)

    def _weight(self, v: Mapping[str, float]) -> Dict[str, float]:
        return {t: (1 + math.log(x)) * self.idf.get(t, self.default_idf) for t, x in v.items() if x > 0}

    def scores(self, text: str) -> Dict[str, float]:
        v = self._weight(self._raw(text))
        return {p: _cos(v, self.centroids[p], self.norms[p]) for p in self.persons}

    def predict(self, text: str) -> str:
        s = self.scores(text)
        return max(self.persons, key=lambda p: (s[p], p))


def same_group_pairwise(msgs: Sequence[Mapping], view: str, *, group: str = "group", k: int = 50, max_pairs: int = 4000, seed: str = "t2d03") -> dict:
    """Topic-controlled discrimination. For a message m by author A in group G, pick another author B who also wrote in G. Both centroids
    exclude every message of G, so the two candidates share m's topic exactly; the question is only whose *way of writing* m resembles.
    Accuracy of choosing A; chance = 0.5. Returns NOT_ESTIMABLE when fewer than 100 such pairs exist."""
    persons = sorted({m["persona_id"] for m in msgs})
    vec = vectors(msgs, view, basis_personas=persons, k=k)
    sums: Dict[str, Dict[str, float]] = {p: defaultdict(float) for p in persons}
    by_pg: Dict[tuple, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    in_group = defaultdict(set)
    for m, v in zip(msgs, vec):
        in_group[m[group]].add(m["persona_id"])
        for kk, x in v.items():
            sums[m["persona_id"]][kk] += x
            by_pg[(m["persona_id"], m[group])][kk] += x

    def centroid_minus(p, g):
        c = dict(sums[p])
        for kk, x in by_pg.get((p, g), {}).items():
            c[kk] = c.get(kk, 0.0) - x
        return c

    rng = random.Random(seed)
    trials = []
    for m, v in zip(msgs, vec):
        others = sorted(in_group[m[group]] - {m["persona_id"]})
        if others:
            trials.append((m, v, rng.choice(others)))
    rng.shuffle(trials)
    trials = trials[:max_pairs]
    if len(trials) < 100:
        return {"view": view, "status": "NOT_ESTIMABLE", "pairs": len(trials)}
    hits = []
    for m, v, other in trials:
        ca, cb = centroid_minus(m["persona_id"], m[group]), centroid_minus(other, m[group])
        if not any(abs(x) > 1e-9 for x in ca.values()) or not any(abs(x) > 1e-9 for x in cb.values()):
            continue
        sa, sb = _cos(v, ca, _norm(ca)), _cos(v, cb, _norm(cb))
        hits.append(1.0 if sa > sb else 0.5 if sa == sb else 0.0)
    acc = statistics.fmean(hits) if hits else None
    se_ = math.sqrt(acc * (1 - acc) / len(hits)) if hits else None
    return {"view": view, "status": "ESTIMATED" if hits else "NOT_ESTIMABLE", "pairs": len(hits), "accuracy": round(acc, 4) if acc is not None else None, "chance": 0.5,
            "ci95_normal": [round(acc - 1.96 * se_, 4), round(acc + 1.96 * se_, 4)] if hits else None}


def cooccurrence_sample(msgs: Sequence[Mapping], eligible: Sequence[str], *, author: str = "author", group: str = "group", cap_persons: int = 150,
                        shared_cap: int = 10, other_cap: int = 20, seed: str = "t2d03") -> List[dict]:
    """Authors who wrote in the same groups as other eligible authors, with their shared-group messages and other messages (references).
    Output rows use the attribution keys: message_id, persona_id, text, group."""
    el = set(eligible)
    who = defaultdict(set)
    for m in msgs:
        if m[author] in el:
            who[m[group]].add(m[author])
    shared = {g for g, s in who.items() if len(s) >= 2}
    persons = sorted({a for g in shared for a in who[g]}, key=lambda a: hashlib.sha256(f"{seed}:{a}".encode()).hexdigest())[:cap_persons]
    pset = set(persons)
    by = defaultdict(lambda: ([], []))
    for m in msgs:
        if m[author] in pset:
            by[m[author]][0 if m[group] in shared else 1].append(m)
    out = []
    for a in persons:
        sh, ot = by[a]
        key = lambda m: hashlib.sha256(f"{seed}:{m['id']}".encode()).hexdigest()
        for m in sorted(sh, key=key)[:shared_cap] + sorted(ot, key=key)[:other_cap]:
            out.append({"message_id": m["id"], "persona_id": a, "text": m["text"], "group": m[group]})
    return out
