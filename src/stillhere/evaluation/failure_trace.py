"""Offline failure audit: where each official evidence record stops on its way to an answer.

This module runs after the pilot and is never imported by retrieval, memory or the answer side. Every status here comes from a stored record, and the rules are
deliberately narrow: a record that could not be seen at query time is not a recall failure, and a missing official record does not by itself mean the question
could not be answered, because the official list is not a minimal sufficient set.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

CAUSES = ("A", "B", "C", "D", "E", "T", "U")
CAUSE_NAMES = {"A": "MEMORY_REPRESENTATION_GAP", "B": "FIRST_HOP_RECALL_GAP", "C": "EVIDENCE_SELECTION_GAP", "D": "TEMPORAL_REPRESENTATION_GAP", "E": "ANSWER_REASONING_GAP", "T": "TEMPORALLY_INELIGIBLE", "U": "UNCERTAIN"}
REVIEW_STATUS = "PROVISIONAL_NOT_HUMAN_VERIFIED"
HUMAN_PENDING = "PENDING"
LEXICAL_BUDGET = 50

# Where one official record stopped, from earliest to latest.
STAGES = ("not_in_raw_file", "not_imported", "ineligible_at_query_time", "no_keyword_match", "below_candidate_budget", "in_candidates_not_top10", "in_top10")


@dataclass(frozen=True)
class GoldStatus:
    key: str
    kind: str
    in_raw: bool
    stored: Optional[str]                 # "memory", "message" or None
    eligible: bool
    lexical_rank: Optional[int]           # among records the scope allows, across facts or messages, without a budget
    dense_rank: Optional[int]
    candidate_rank: Optional[int]         # in the arm's candidate list
    in_top10: bool
    in_context: bool
    locator: Optional[str] = None
    ineligible_reason: Optional[str] = None


def stage(g: GoldStatus) -> str:
    """The last stage this record reached. Eligibility comes before recall: what the time rules hide cannot be a retrieval miss."""
    if not g.in_raw:
        return "not_in_raw_file"
    if g.stored is None:
        return "not_imported"
    if not g.eligible:
        return "ineligible_at_query_time"
    if g.in_top10:
        return "in_top10"
    if g.candidate_rank is not None:
        return "in_candidates_not_top10"
    if g.lexical_rank is None:
        return "no_keyword_match"
    return "below_candidate_budget"


def rule_tags(records: Sequence[GoldStatus], *, sufficient: Optional[str], answer_correct: bool, abstained: bool) -> List[str]:
    """Tags the stored facts alone support. They are a starting point: a person or a closer reading decides the primary cause."""
    if answer_correct:
        return []
    stages = Counter(stage(g) for g in records)
    tags: List[str] = []
    if stages["not_imported"] or stages["not_in_raw_file"]:
        tags.append("A")
    if stages["no_keyword_match"] or stages["below_candidate_budget"]:
        tags.append("B")
    if stages["in_candidates_not_top10"]:
        tags.append("C")
    if stages["ineligible_at_query_time"]:
        tags.append("T")
    if sufficient == "yes":
        tags.append("E")
    return tags or ["U"]


_DAY = re.compile(r"(?<!\d)(\d{1,2})月(\d{1,2})[日号]")
_PART = re.compile(r"(?<!\d)(\d{1,2})月(上旬|中旬|下旬|初|底|末)")
_MONTH = re.compile(r"(?<!\d)(\d{1,2})月(?:份)?")
_PARTS = {"上旬": (1, 10), "初": (1, 10), "中旬": (11, 20), "下旬": (21, 31), "底": (21, 31), "末": (21, 31)}


def _last_day(year: int, month: int) -> int:
    return ((date(year + month // 12, month % 12 + 1, 1)) - timedelta(days=1)).day


def time_window(question: str, year: int) -> Optional[Tuple[date, date]]:
    """The dates a question names, as one span. Audit only: this measures what such a filter could reach, and retrieval does not use it. Relative phrases are left out."""
    spans: List[Tuple[date, date]] = []
    text = question
    for pattern, kind in ((_DAY, "day"), (_PART, "part"), (_MONTH, "month")):
        for m in pattern.finditer(text):
            month = int(m.group(1))
            if not 1 <= month <= 12:
                continue
            if kind == "day":
                day = min(int(m.group(2)), _last_day(year, month))
                spans.append((date(year, month, day), date(year, month, day)))
            elif kind == "part":
                low, high = _PARTS[m.group(2)]
                spans.append((date(year, month, low), date(year, month, min(high, _last_day(year, month)))))
            else:
                spans.append((date(year, month, 1), date(year, month, _last_day(year, month))))
        text = pattern.sub(" ", text)
    return (min(s for s, _ in spans), max(e for _, e in spans)) if spans else None


def validate_label(label: Mapping) -> None:
    """Multiple causes are normal. A primary cause is optional and must be one of them, and nothing may claim a person's review."""
    secondary = list(label.get("secondary_causes") or [])
    primary = label.get("primary_cause")
    for cause in ([primary] if primary else []) + secondary:
        if cause not in CAUSES:
            raise ValueError(f"unknown cause {cause!r}")
    if primary in secondary:
        raise ValueError("a primary cause is not also secondary")
    if label.get("confidence") not in ("high", "medium", "low"):
        raise ValueError("a label states its confidence")
    if label.get("reviewer_type") == "HUMAN":
        raise ValueError("the audit never writes a human judgement")


def count_causes(labels: Iterable[Mapping]) -> dict:
    primary: Counter = Counter()
    anywhere: Counter = Counter()
    multi = 0
    total = 0
    for x in labels:
        total += 1
        tags = set(x.get("secondary_causes") or []) | ({x["primary_cause"]} if x.get("primary_cause") else set())
        anywhere.update(tags)
        multi += len(tags) > 1
        primary[x.get("primary_cause") or "NONE"] += 1
    return {"questions": total, "primary": dict(primary), "any_label": dict(anywhere), "multi_label_questions": multi}


def stage_counts(rows: Iterable[Mapping]) -> dict:
    """By record, and by question using the question's worst and best stage, because one missing record need not sink a question."""
    records: Counter = Counter()
    questions = 0
    with_all_in_top10 = 0
    with_none_in_top10 = 0
    for row in rows:
        stages = [stage(GoldStatus(**g)) for g in row["gold_status"]]
        if not stages:
            continue
        questions += 1
        records.update(stages)
        with_all_in_top10 += all(s == "in_top10" for s in stages)
        with_none_in_top10 += not any(s == "in_top10" for s in stages)
    return {"questions": questions, "official_records": sum(records.values()), "by_stage": {s: records[s] for s in STAGES}, "questions_all_in_top10": with_all_in_top10, "questions_none_in_top10": with_none_in_top10}
