"""D4 statistics (D4_EVALUATION_PROTOCOL_V1 STATISTICS), pure Python (runs on Colab and in the Windows CPU importer, no numpy / scipy).

* paired item-level bootstrap clustered by persona, 10,000 resamples, 95% percentile interval, seed fixed (``SEED``);
* McNemar exact (two-sided) for paired binary outcomes (secondary);
* exact Clopper-Pearson 95% interval for counts (leaks);
* minimum detectable paired difference at 80% power (``protocol._mde``)."""
from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .. import protocol as PR

SEED = 20261112
N_BOOT = 10_000


def _pairs(a_rows: Mapping[str, Mapping], b_rows: Mapping[str, Mapping], comps: Sequence[str], persona: Mapping[str, str]):
    """item id -> per-component (a, b) where both are applicable, grouped by persona."""
    cl: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0]))
    for i, ra in a_rows.items():
        rb = b_rows.get(i)
        if rb is None:
            continue
        for c in comps:
            x, y = ra.get(c), rb.get(c)
            if x is None or y is None:
                continue
            s = cl[persona[i]][c]
            s[0] += x
            s[1] += y
            s[2] += 1
    return cl


def _stat(acc: Mapping[str, List[float]]) -> Optional[float]:
    d = [(v[1] - v[0]) / v[2] for v in acc.values() if v[2]]
    return statistics.fmean(d) if d else None


def paired_bootstrap(a_rows: Mapping[str, Mapping], b_rows: Mapping[str, Mapping], comps: Sequence[str], persona: Mapping[str, str],
                     n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """B - A of mean-over-components of the component means (one component = a plain rate). Clusters = personas."""
    cl = _pairs(a_rows, b_rows, comps, persona)
    keys = sorted(cl)
    n_items = {c: sum(cl[p][c][2] for p in keys) for c in comps}
    total = defaultdict(lambda: [0.0, 0.0, 0])
    for p in keys:
        for c, v in cl[p].items():
            t = total[c]
            t[0] += v[0]
            t[1] += v[1]
            t[2] += v[2]
    est = _stat(total)
    if est is None or not keys:
        return {"estimate": None, "ci95": [None, None], "n_items": n_items, "n_clusters": len(keys), "n_boot": 0}
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        acc = defaultdict(lambda: [0.0, 0.0, 0])
        for _k in range(len(keys)):
            for c, v in cl[keys[rng.randrange(len(keys))]].items():
                t = acc[c]
                t[0] += v[0]
                t[1] += v[1]
                t[2] += v[2]
        s = _stat(acc)
        if s is not None:
            vals.append(s)
    vals.sort()
    lo, hi = vals[int(0.025 * (len(vals) - 1))], vals[int(math.ceil(0.975 * (len(vals) - 1)))]
    a_mean = statistics.fmean(v[0] / v[2] for v in total.values() if v[2])
    b_mean = statistics.fmean(v[1] / v[2] for v in total.values() if v[2])
    return {"a": round(a_mean, 6), "b": round(b_mean, 6), "estimate": round(est, 6), "ci95": [round(lo, 6), round(hi, 6)], "n_items": n_items,
            "n_clusters": len(keys), "n_boot": len(vals), "seed": seed}


def cluster_bootstrap_mean(values: Mapping[str, float], n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Mean of per-cluster values (e.g. per-persona style-distance differences) with a percentile interval."""
    keys = sorted(values)
    if not keys:
        return {"estimate": None, "ci95": [None, None], "n_clusters": 0}
    rng = random.Random(seed)
    xs = [values[k] for k in keys]
    vals = sorted(statistics.fmean(xs[rng.randrange(len(xs))] for _ in xs) for _ in range(n_boot))
    return {"estimate": round(statistics.fmean(xs), 6), "ci95": [round(vals[int(0.025 * (n_boot - 1))], 6), round(vals[int(math.ceil(0.975 * (n_boot - 1)))], 6)],
            "n_clusters": len(keys), "n_boot": n_boot, "seed": seed}


def mcnemar_exact(a_rows: Mapping[str, Mapping], b_rows: Mapping[str, Mapping], name: str) -> dict:
    n01 = n10 = 0
    for i, ra in a_rows.items():
        rb = b_rows.get(i)
        if rb is None or ra.get(name) is None or rb.get(name) is None:
            continue
        x, y = bool(ra[name] >= 0.5), bool(rb[name] >= 0.5)
        n01 += (not x) and y
        n10 += x and (not y)
    n = n01 + n10
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p_value": 1.0}
    k = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only": n10, "b_only": n01, "p_value": round(p, 6)}


def _binom_cdf(k: int, n: int, p: float) -> float:
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    return min(1.0, sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq) for i in range(k + 1)))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> List[Optional[float]]:
    if n <= 0:
        return [None, None]

    lower = 0.0 if k == 0 else _bisect(lambda p: 1 - _binom_cdf(k - 1, n, p) < alpha / 2)   # P(X >= k | p) = alpha/2
    upper = 1.0 if k == n else _bisect(lambda p: _binom_cdf(k, n, p) > alpha / 2)           # P(X <= k | p) = alpha/2
    return [round(lower, 6), round(upper, 6)]


def _bisect(go_right) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if go_right(mid):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def mde(n: int) -> float:
    return PR._mde(n)


def power_note(n: int, margin: float) -> dict:
    m = mde(n)
    return {"n": n, "mde_80pct_power": m, "margin": margin, "underpowered": (m > margin) if n else True}
