"""V3 research measurement: the fixed question samples, the spend rule and the retrieval metrics.

Gold reaches this module only after a retrieval has returned. Nothing in ``retrieval`` imports it. Dataset selection reads question ids, types and whether official
evidence can be traced. It never reads an official answer.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from stillhere.retrieval.alibaba_client import BudgetExceeded, Ledger

from . import memory_bench as mb

NEW_WORK_LIMIT_CNY = 8.50
CLOSING_LIMIT_CNY = 9.50
HARD_LIMIT_CNY = 10.00


class ResearchLedger(Ledger):
    """This round's own append-only ledger. It never reads or resets the earlier one, so ``spent`` is this round's cost alone."""

    def __init__(self, path: Path, earlier_cny: float) -> None:
        super().__init__(path, HARD_LIMIT_CNY)
        self.earlier_cny = float(earlier_cny)

    @property
    def cumulative(self) -> float:
        return self.earlier_cny + self.spent

    def require_room(self, estimate: float, *, closing: bool = False) -> None:
        """Called before each batch. New work stops at 8.50, closing work at 9.50, and nothing may be planned past 10.00."""
        limit = CLOSING_LIMIT_CNY if closing else NEW_WORK_LIMIT_CNY
        if self.spent >= limit:
            raise BudgetExceeded(f"spent {self.spent:.4f} CNY: no {'closing' if closing else 'new'} work past {limit:.2f}")
        if self.spent + self.pending + estimate > HARD_LIMIT_CNY:
            raise BudgetExceeded(f"spent {self.spent:.4f} CNY: {estimate:.4f} more could pass {HARD_LIMIT_CNY:.2f}")


def _order(seed: str, qa_id: str) -> str:
    return hashlib.sha256(f"{seed}:{qa_id}".encode("utf-8")).hexdigest()


def sample_questions(questions: Sequence[mb.Question], seed: str, *, scorable: int, unanswerable: int, usable: Set[str]) -> List[str]:
    """A fixed share per persona: types in proportion to what is available, ordered by a seeded hash. ``usable`` are ids that passed the structural checks."""
    by_persona: Dict[str, List[mb.Question]] = defaultdict(list)
    for q in questions:
        if q.qa_id in usable:
            by_persona[q.persona_id].append(q)
    chosen: List[str] = []
    for persona, items in sorted(by_persona.items()):
        scored = [q for q in items if not q.unanswerable]
        strata: Dict[str, List[mb.Question]] = defaultdict(list)
        for q in scored:
            strata[q.primary_type].append(q)
        shares = mb._largest_remainder({k: len(v) for k, v in strata.items()}, scorable)
        for stratum in sorted(strata):
            chosen += [q.qa_id for q in sorted(strata[stratum], key=lambda q: _order(seed, q.qa_id))[:shares.get(stratum, 0)]]
        chosen += [q.qa_id for q in sorted((q for q in items if q.unanswerable), key=lambda q: _order(seed, q.qa_id))[:unanswerable]]
    return sorted(chosen)


# ---------------------------------------------------------------- retrieval metrics

@dataclass(frozen=True)
class GoldView:
    """A question's official evidence, resolved to stored keys, with each record's official daily event."""

    keys: Tuple[str, ...]
    links: Mapping[str, Optional[str]]
    events: Tuple[str, ...]


def covered(keys: Iterable[str], gold: GoldView) -> Set[str]:
    """Gold records present, directly or through their official daily event. A daily event restates its records, so a list holding it does hold the fact."""
    present = set(keys)
    return {g for g in gold.keys if g in present or (gold.links.get(g) and gold.links[g] in present)}


def retrieval_row(gold: GoldView, *, pool: Sequence[str], final: Sequence[str], baseline_pool: Sequence[str] = (), baseline_final: Sequence[str] = ()) -> dict:
    goldset = set(gold.keys)
    top = list(final)[:10]
    strict_pool, strict_top = goldset & set(pool), goldset & set(top)
    event_pool, event_top = covered(pool, gold), covered(top, gold)
    n = len(goldset)
    return {"gold": n, "candidate_coverage": len(strict_pool) / n, "top10_coverage": len(strict_top) / n, "hit10": bool(strict_top), "full10": len(strict_top) == n,
            "event_candidate_coverage": len(event_pool) / n, "event_top10_coverage": len(event_top) / n, "event_hit10": bool(event_top), "event_full10": len(event_top) == n,
            "gold_in_pool": len(strict_pool), "gold_in_top10": len(strict_top), "event_gold_in_pool": len(event_pool), "event_gold_in_top10": len(event_top),
            "new_gold_candidates": len(strict_pool - set(baseline_pool)), "new_gold_top10": len(strict_top - set(baseline_final)), "displaced_gold": len((goldset & set(baseline_final)) - set(top)),
            "new_event_gold_top10": len(event_top - covered(baseline_final, gold)), "displaced_event_gold": len(covered(baseline_final, gold) - event_top)}


def text_grams(text: str) -> Set[str]:
    compact = "".join(text.split())
    return {compact[i:i + 2] for i in range(len(compact) - 1)}


def overlap(a: Set[str], b: Set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def duplicate_rate(texts: Sequence[str], groups: Sequence[Optional[str]], *, threshold: float = 0.6) -> float:
    """Share of a list that repeats an earlier item: the same stored event, or nearly the same words."""
    grams = [text_grams(t) for t in texts]
    seen: Set[str] = set()
    repeats = 0
    for i, group in enumerate(groups):
        if (group and group in seen) or any(overlap(grams[i], grams[j]) >= threshold for j in range(i)):
            repeats += 1
        if group:
            seen.add(group)
    return repeats / len(texts) if texts else 0.0


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    v = [float(x) for x in values if x is not None]
    return round(statistics.fmean(v), 4) if v else None


def summarize(rows: Sequence[Mapping]) -> dict:
    fields = ("candidate_coverage", "top10_coverage", "hit10", "full10", "event_candidate_coverage", "event_top10_coverage", "event_hit10", "event_full10")
    out = {name: mean(r[name] for r in rows) for name in fields}
    out["questions"] = len(rows)
    for name in ("gold", "gold_in_pool", "gold_in_top10", "event_gold_in_pool", "event_gold_in_top10", "new_gold_candidates", "new_gold_top10", "displaced_gold", "new_event_gold_top10", "displaced_event_gold"):
        out[name] = sum(r[name] for r in rows)
    return out


LEGACY_ROUND_TURNS = 3      # D6: the frozen v3-round-1 renderer reads only turns 0-2 of a round (BUILD-M1, docs/M1_D6_REMEDIATION.md)


def round_text(api, scope, mask, persona_id: str, record_id: str, speakers: Mapping[str, str]) -> Optional[str]:
    """FROZEN v3-round-1: the visible turns 0-2 of one chat round, each with its speaker.

    Known defect D6: a round with more than three messages loses the rest, silently. It is kept unchanged only so that every
    frozen V3.x result reproduces byte for byte. The full-round path is ``stillhere.context`` (Context Orchestrator V1), used by
    the reader scripts when CONTEXT_ORCHESTRATOR_V1_ENABLED is on."""
    stem = record_id.rsplit(".", 1)[0]
    turns = []
    for n in range(LEGACY_ROUND_TURNS):
        turn = f"{stem}.{n}"
        message = api.get_message(persona_id, turn) if scope.visible(persona_id, turn, mask) else None
        if message is not None:
            who = "本人" if message.speaker_id == persona_id else speakers.get(message.speaker_id, message.speaker_id)
            turns.append(f"{who}：{message.content}")
    return " / ".join(turns) if turns else None


def manifest_digest(manifest: Mapping) -> str:
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
