"""Style Evaluation v2: author-verification metrics for short replies, with length matching and copy control. CPU only, no model.

Why not the old metric: the T2 PoC ``style_attribution_accuracy`` is the TOP-1 hit rate of one reply among all test authors (200 -> chance
0.5%). On short replies (median ~20 chars) almost every reply misses for every system (real references: 4.77%), so 179 of 199 speakers tie
at 0 and the metric cannot rank systems. It is also silent about length.

v2 scores one reply ``t`` for its target author ``a`` against a set of distractor authors ``D`` with a pairwise ranking statistic:

    AUC(t, a, D) = mean_{d in D} [ 1{s(t,a) > s(t,d)} + 0.5 * 1{s(t,a) = s(t,d)} ]

where ``s(t, x)`` is the cosine between ``t`` and author ``x``'s centroid built ONLY from x's held-out REFERENCE replies (never shown to a model,
topic-disjoint from its examples and targets). Chance is 0.5 for any output that does not depend on the author: averaged over target authors,
a fixed text has AUC exactly 0.5 (rank symmetry), so a uniform "forum voice" earns no credit -- only author-specific variation does.

* views: ``char`` (character 1-3 grams, punctuation included, sublinear tf * idf), ``func`` (the text-distortion view of the T2 attributor:
  the 50 most frequent tokens kept, everything else masked, 1-3 grams); diagnostic views ``nopunct`` (char view without punctuation),
  ``punct`` (punctuation / symbol sequence only), ``len`` (length / rhythm only: chars, bubbles, chars per bubble, as a 1-d author profile).
* length matching (``lm``): ``D`` = the half of the other authors whose reference median length is closest to ``a``'s (|log ratio|), so
  "just being short" cannot beat short distractors.
* PRIMARY = STYLE_AUC_V2 = mean(AUC_char_lm, AUC_func_lm), with an output that copies a shown example (longest common substring >= 8 chars)
  scored 0.5 (no credit for copying). Equal weights, fixed before any model output was scored; validated on DEV references only.
"""
from __future__ import annotations

import math
import re
import statistics
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from stillhere.style.learnability import keep_set, style_vec

PUNCT = re.compile(r"[^\w\s]|_", re.U)
WS = re.compile(r"\s+")
COPY_MIN = 8
VERSION = "style-eval-v2-metric-1"
PRIMARY_VIEWS = ("char", "func")


def lcs_len(a: str, b: str) -> int:
    best, prev = 0, [0] * (len(b) + 1)
    for ca in a:
        cur = [0] * (len(b) + 1)
        for j, cb in enumerate(b, 1):
            if ca == cb:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def copies(text: str, shown: Iterable[str], k: int = COPY_MIN) -> bool:
    """A shown example (or one of its " / "-joined bubbles) reproduced: a common substring of >= ``k`` chars, or, for an example shorter
    than ``k`` (>= 4 chars), the whole example inside the output."""
    t = WS.sub("", text)
    for s in shown:
        for part in [s] + s.split(" / "):
            p = WS.sub("", part)
            if len(p) >= 4 and (lcs_len(t, p) >= min(k, len(p))):
                return True
    return False


def _grams(s: str, n_max: int = 3) -> Counter:
    s = "^" + s + "$"
    c = Counter()
    for n in range(1, n_max + 1):
        for i in range(len(s) - n + 1):
            c[s[i:i + n]] += 1
    return c


def raw_vector(text: str, view: str, keep: Optional[set] = None) -> Dict[str, float]:
    t = WS.sub(" ", text.lower()).strip()
    if view == "char":
        return dict(_grams(t.replace(" ", "")))
    if view == "nopunct":
        return dict(_grams(PUNCT.sub("", t).replace(" ", "")))
    if view == "punct":
        p = "".join(PUNCT.findall(t)) or "∅"
        return dict(_grams(p))
    if view == "func":
        return dict(style_vec(text, keep))
    raise ValueError(view)


def length_feats(text: str) -> float:
    return math.log(1 + len(WS.sub("", text)))


class StyleSpace:
    """Author centroids for one view, fitted on REFERENCE replies only."""

    def __init__(self, references: Sequence[Mapping], view: str, *, k: int = 50):
        self.view = view
        texts = [r["text"] for r in references]
        if view == "len":
            by = defaultdict(list)
            for r in references:
                by[r["persona_id"]].append(length_feats(r["text"]))
            self.cent = {p: statistics.fmean(v) for p, v in by.items()}
            self.persons = sorted(self.cent)
            return
        self.keep = keep_set(texts, k) if view == "func" else None
        raw = [raw_vector(t, view, self.keep) for t in texts]
        df = Counter(g for v in raw for g in v)
        n = len(raw)
        self.idf = {g: math.log((1 + n) / (1 + c)) + 1.0 for g, c in df.items()}
        self.default_idf = math.log(1 + n) + 1.0
        cents: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for r, v in zip(references, raw):
            for g, x in self._w(v).items():
                cents[r["persona_id"]][g] += x
        self.cent = {p: dict(c) for p, c in cents.items()}
        self.norm = {p: math.sqrt(sum(x * x for x in c.values())) for p, c in self.cent.items()}
        self.persons = sorted(self.cent)

    def _w(self, v):
        return {g: (1 + math.log(x)) * self.idf.get(g, self.default_idf) for g, x in v.items() if x > 0}

    def sims(self, text: str) -> Dict[str, float]:
        if self.view == "len":
            x = length_feats(text)
            return {p: -abs(x - c) for p, c in self.cent.items()}
        v = self._w(raw_vector(text, self.view, self.keep))
        nv = math.sqrt(sum(x * x for x in v.values()))
        out = {}
        for p in self.persons:
            c = self.cent[p]
            out[p] = (sum(x * c.get(g, 0.0) for g, x in v.items()) / (nv * self.norm[p])) if nv and self.norm[p] else 0.0
        return out


def auc(sims: Mapping[str, float], target: str, distractors: Sequence[str]) -> float:
    st = sims[target]
    vals = [1.0 if st > sims[d] else 0.5 if st == sims[d] else 0.0 for d in distractors if d != target]
    return statistics.fmean(vals) if vals else 0.5


class Scorer:
    """All views over one split's authors. ``length_matched(a)`` -> the closest-length half of the other authors."""

    VIEWS = ("char", "func", "nopunct", "punct", "len")

    def __init__(self, references: Sequence[Mapping]):
        self.refs = list(references)
        self.spaces = {v: StyleSpace(self.refs, v) for v in self.VIEWS}
        self.persons = self.spaces["char"].persons
        med = defaultdict(list)
        for r in self.refs:
            med[r["persona_id"]].append(len(WS.sub("", r["text"])))
        self.median_len = {p: statistics.median(v) for p, v in med.items()}
        self._lm = {}
        for p in self.persons:
            others = sorted((q for q in self.persons if q != p), key=lambda q: (abs(math.log((1 + self.median_len[q]) / (1 + self.median_len[p]))), q))
            self._lm[p] = others[: max(1, len(others) // 2)]

    def length_matched(self, p: str) -> List[str]:
        return self._lm[p]

    def all_others(self, p: str) -> List[str]:
        return [q for q in self.persons if q != p]

    def score(self, text: str, target: str, shown: Sequence[str] = ()) -> dict:
        """Every view's AUC (all distractors and length-matched) + top-1 hit (old-style) + the PRIMARY score."""
        out = {}
        for v, sp in self.spaces.items():
            s = sp.sims(text)
            out[f"auc_{v}"] = auc(s, target, self.all_others(target))
            out[f"auc_{v}_lm"] = auc(s, target, self.length_matched(target))
            if v == "func":
                out["top1_func"] = float(max(self.persons, key=lambda q: (s[q], q)) == target)
        out["copied"] = copies(text, shown) if shown else False
        prim = statistics.fmean(out[f"auc_{v}_lm"] for v in PRIMARY_VIEWS)
        out["style_auc_v2"] = 0.5 if out["copied"] else prim
        out["style_auc_v2_uncapped"] = prim
        out["chars"] = len(WS.sub("", text))
        out["len_log_error"] = abs(math.log((1 + out["chars"]) / (1 + self.median_len[target])))
        return out

    def score_toward(self, text: str, who: str) -> float:
        """PRIMARY (uncapped) AUC of ``text`` toward another author ``who`` (for the swap test)."""
        return statistics.fmean(auc(self.spaces[v].sims(text), who, self.length_matched(who)) for v in PRIMARY_VIEWS)


# ---------------------------------------------------------------- statistics (speaker clusters)

def per_author(rows: Sequence[Mapping], key: str, author: str = "persona_id") -> Dict[str, float]:
    by = defaultdict(list)
    for r in rows:
        if r.get(key) is not None:
            by[r[author]].append(float(r[key]))
    return {a: statistics.fmean(v) for a, v in by.items()}


def cluster_bootstrap(per: Mapping[str, float], seed: str = "sev2", n: int = 5000) -> Optional[List[float]]:
    """95% interval of the mean over AUTHORS (each author's own mean is one draw): the cluster unit, not the reply."""
    import random
    keys = sorted(per)
    if not keys:
        return None
    rng = random.Random(seed)
    vals = sorted(statistics.fmean(per[rng.choice(keys)] for _ in keys) for _ in range(n))
    return [round(vals[int(0.025 * n)], 4), round(vals[int(0.975 * n) - 1], 4)]


def paired(a: Mapping[str, float], b: Mapping[str, float], seed: str = "sev2", n: int = 5000, tie: float = 1e-9) -> dict:
    """Author-paired difference b - a: mean, CI (author bootstrap), win / tie / loss, dz, sign-test p (two-sided)."""
    keys = sorted(set(a) & set(b))
    d = {k: b[k] - a[k] for k in keys}
    if not d:
        return {"authors": 0}
    vals = list(d.values())
    sd = statistics.pstdev(vals)
    w, l = sum(v > tie for v in vals), sum(v < -tie for v in vals)
    return {"authors": len(vals), "mean_diff": round(statistics.fmean(vals), 4), "median_diff": round(statistics.median(vals), 4), "ci": cluster_bootstrap(d, seed, n),
            "win": w, "tie": len(vals) - w - l, "loss": l, "dz": round(statistics.fmean(vals) / sd, 3) if sd else None, "sign_p": sign_p(w, l)}


def sign_p(w: int, l: int) -> Optional[float]:
    n = w + l
    if not n:
        return None
    k = min(w, l)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return round(min(1.0, p), 6)
