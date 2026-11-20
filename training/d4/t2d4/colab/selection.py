"""Pre-registered checkpoint selection (frozen before training; its sha256 is in the release manifest). Uses DEV_PROBE only, never HOLDOUT.

Candidates: the saved checkpoints nearest 25 / 50 / 75 / 100 % of the run (``d4_candidate_fractions``).

A candidate QUALIFIES when, on its DEV_PROBE generations (same prompts, same seed as Base):
  serious_leak_count == 0
  repetition_rate <= base + 0.03 and <= 0.05
  empty_rate <= 0.02
  special_token_or_think_leak_rate == 0
  core_grounding_score >= base - 0.05
  verbose_assistant_rate <= base
OVERFIT: its DEV_PROBE eval loss is above the running minimum of all earlier evaluations (and its own) + ``d4_overfit_margin`` (0.05).
Score = mean of the available terms core_grounding_score, 1 - verbose_assistant_rate, 1 - over_refusal_rate.
Choice: the best score among QUALIFIED and not OVERFIT; else among QUALIFIED (status SELECTED_QUALIFIED_OVERFIT); else among all candidates
(status SELECTED_UNQUALIFIED: the run continues, every result carries the label). Scores within 0.005 of the best go to the earliest step.
"""
from __future__ import annotations

import hashlib
import statistics
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

RULES_VERSION = "d4-selection-v1"
TIE = 0.005


def candidate_steps(saved: Sequence[int], total: int, fractions: Sequence[float]) -> List[int]:
    out: List[int] = []
    for f in fractions:
        want = total * f
        s = min(saved, key=lambda x: (abs(x - want), x))
        if s not in out:
            out.append(s)
    return out


def overfit_flags(eval_losses: Sequence[Mapping], steps: Sequence[int], margin: float) -> Dict[int, Optional[bool]]:
    ev = sorted((int(e["step"]), float(e["loss"])) for e in eval_losses if e.get("loss") is not None)
    out: Dict[int, Optional[bool]] = {}
    for s in steps:
        upto = [l for st, l in ev if st <= s]
        mine = [l for st, l in ev if st == s]
        out[s] = None if not mine else bool(mine[0] > min(upto) + margin)
    return out


def gates(base: Mapping, c: Mapping) -> Dict[str, bool]:
    g = lambda k: c.get(k) if c.get(k) is not None else 0.0
    b = lambda k: base.get(k) if base.get(k) is not None else 0.0
    out = {"serious_leak_zero": int(c.get("serious_leak_count") or 0) == 0,
           "repetition": g("repetition_rate") <= b("repetition_rate") + 0.03 and g("repetition_rate") <= 0.05,
           "empty": g("empty_rate") <= 0.02,
           "special_or_think_zero": g("special_token_or_think_leak_rate") == 0,
           "verbose_not_worse": g("verbose_assistant_rate") <= b("verbose_assistant_rate")}
    cg, bg = c.get("core_grounding_score"), base.get("core_grounding_score")
    out["core_grounding"] = cg is not None and (bg is None or cg >= bg - 0.05)
    return out


def score(c: Mapping) -> float:
    terms = [c.get("core_grounding_score")]
    terms.append(None if c.get("verbose_assistant_rate") is None else 1 - c["verbose_assistant_rate"])
    terms.append(None if c.get("over_refusal_rate") is None else 1 - c["over_refusal_rate"])
    terms = [t for t in terms if t is not None]
    return statistics.fmean(terms) if terms else 0.0


def select(base: Mapping, cands: Mapping[int, Mapping], eval_losses: Sequence[Mapping], margin: float = 0.05) -> dict:
    steps = sorted(cands)
    of = overfit_flags(eval_losses, steps, margin)
    table = {}
    for s in steps:
        gt = gates(base, cands[s])
        table[s] = {"gates": gt, "qualified": all(gt.values()), "overfit": of[s], "score": round(score(cands[s]), 6)}

    def pick(pool):
        best = max(table[s]["score"] for s in pool)
        return min(s for s in pool if table[s]["score"] >= best - TIE)
    q_clean = [s for s in steps if table[s]["qualified"] and not table[s]["overfit"]]
    q_all = [s for s in steps if table[s]["qualified"]]
    if q_clean:
        chosen, status = pick(q_clean), "SELECTED_QUALIFIED"
    elif q_all:
        chosen, status = pick(q_all), "SELECTED_QUALIFIED_OVERFIT"
    else:
        chosen, status = pick(steps), "SELECTED_UNQUALIFIED"
    return {"rules_version": RULES_VERSION, "rules_sha256": RULES_SHA256, "selected_on": "DEV_PROBE", "chosen_step": chosen, "status": status,
            "chosen": table[chosen], "candidates": {str(s): table[s] for s in steps}, "overfit_margin": margin}


RULES_SHA256 = hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()  # this file (line endings normalised)
