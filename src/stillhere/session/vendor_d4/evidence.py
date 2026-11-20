"""D4 Memory Evidence: the minimal evidence item a runtime must be able to supply, and its mapping onto the Core memory contract.

Core (``stillhere.contracts.memory.MemoryRecord``, SQLite schema v2) stores ``status`` in {active, superseded, retracted}, a validity interval
[valid_from, valid_to), ``supersedes`` / ``superseded_by`` / ``correction_reason`` for corrections, ``conflicts_with`` and ``cardinality``.
The style prompt (Contract v2, ``stillhere.style.prompt/2.0``) shows roles {current, past, corrected, unverified, at_time}.

D4 evidence status is a VIEW of those records at the conversation time (``now``), never a new storage semantic:

    CURRENT     active, valid at now                                   -> prompt role current
    HISTORICAL  active, valid_to <= now (it was true, it no longer is)   -> prompt role past
    CORRECTED   the active replacement of a superseded record; ``replaces`` carries the wrong old value -> prompt role corrected
    CANCELLED   a planned event whose plan was called off: a HISTORICAL plan record plus an event record "cancelled" (Core kind=event)
    CONFLICTED  two or more active single-cardinality records that ``contradicts()`` (same conflict_group)  -> prompt role unverified
    UNKNOWN     nothing recorded for the slot asked (Core: Sufficiency no_record / time_unknown)
    UPCOMING    active, valid_from > now (decided, not yet in effect)    -> D4 addition, Core can express it with valid_from

``disclosure_policy`` has NO Core field today. D4 defines it as an evidence-view attribute the runtime attaches (from a policy table or the
writer), see docs/D4_1_CONTEXT_THREAD_INTERFACE_HANDOFF.md:

    ALLOWED          may be said to this conversation partner
    PROTECTED        known, must not be disclosed (contact details, IDs, third-party private data, revoked consent)
    POLICY_UNCERTAIN known, permission not established: treat as not disclosable, say permission is needed
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

STATUSES = ("CURRENT", "HISTORICAL", "CORRECTED", "CANCELLED", "CONFLICTED", "UPCOMING")
DISCLOSURE = ("ALLOWED", "PROTECTED", "POLICY_UNCERTAIN")
CORE_STATUS_VIEW = {
    "CURRENT": {"core_status": "active", "validity": "valid_from <= now < valid_to", "prompt_v2_role": "current"},
    "HISTORICAL": {"core_status": "active", "validity": "valid_to <= now", "prompt_v2_role": "past"},
    "CORRECTED": {"core_status": "active (supersedes an older record, correction_reason set)", "validity": "valid at now", "prompt_v2_role": "corrected"},
    "CANCELLED": {"core_status": "active plan record with valid_to = cancellation time + event record (kind=event, attribute=<slot>_cancelled)",
                  "validity": "plan no longer holds", "prompt_v2_role": "past (+ cancellation note)"},
    "CONFLICTED": {"core_status": "active; contradicts() true within a conflict_group (cardinality single)", "validity": "overlapping", "prompt_v2_role": "unverified"},
    "UPCOMING": {"core_status": "active", "validity": "valid_from > now", "prompt_v2_role": "at_time (D4: shown as 尚未生效)"},
    "UNKNOWN": {"core_status": "no record (MemoryContextView query sufficiency no_record / time_unknown)", "validity": "-", "prompt_v2_role": "unknown item"},
}
SUBJECT_TYPES = ("self", "friend", "spouse", "family", "colleague", "neighbor")


@dataclass(frozen=True)
class Evidence:
    fact_id: str
    subject: str            # "本人" or a third party label, e.g. "朋友小林"
    subject_type: str       # one of SUBJECT_TYPES
    predicate: str          # slot, e.g. "home_city"
    value: str              # the value that may be said if disclosure allows
    evidence_text: str      # natural-language statement shown to the model
    status: str = "CURRENT"
    valid_from: Optional[str] = None   # ISO date (YYYY-MM-DD) or YYYY-MM
    valid_to: Optional[str] = None
    observed_at: Optional[str] = None
    source_ref: str = "synthetic"
    confidence: str = "high"           # high | medium | low (Contract v2 labels)
    disclosure_policy: str = "ALLOWED"
    conflict_group: Optional[str] = None
    replaces: Optional[str] = None     # CORRECTED: the wrong earlier value (never to be said as current)
    note: Optional[str] = None         # CANCELLED: what was cancelled and when

    def validate(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"{self.fact_id}: status {self.status}")
        if self.disclosure_policy not in DISCLOSURE:
            raise ValueError(f"{self.fact_id}: disclosure {self.disclosure_policy}")
        if self.subject_type not in SUBJECT_TYPES:
            raise ValueError(f"{self.fact_id}: subject_type {self.subject_type}")
        if self.confidence not in ("high", "medium", "low"):
            raise ValueError(f"{self.fact_id}: confidence {self.confidence}")
        if self.status == "CORRECTED" and not self.replaces:
            raise ValueError(f"{self.fact_id}: CORRECTED needs replaces")
        if self.status == "CONFLICTED" and not self.conflict_group:
            raise ValueError(f"{self.fact_id}: CONFLICTED needs conflict_group")
        if self.status == "CANCELLED" and not self.note:
            raise ValueError(f"{self.fact_id}: CANCELLED needs note")
        for name in ("fact_id", "subject", "predicate", "value", "evidence_text"):
            if not str(getattr(self, name) or "").strip():
                raise ValueError(f"{self.fact_id}: empty {name}")
        if self.valid_from and self.valid_to and self.valid_to <= self.valid_from:
            raise ValueError(f"{self.fact_id}: valid_to <= valid_from")

    def to_json(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def from_json(d: dict) -> Evidence:
    return Evidence(**d)


@dataclass(frozen=True)
class UnknownSlot:
    """A slot the question asks about for which nothing is recorded."""
    slot_id: str
    subject: str
    predicate: str
    label: str              # how the prompt names it, e.g. "下周出差住的酒店"

    def to_json(self) -> dict:
        return asdict(self)


def check_conflict_groups(evidence: List[Evidence]) -> None:
    groups = {}
    for e in evidence:
        if e.status == "CONFLICTED":
            groups.setdefault(e.conflict_group, []).append(e)
    for g, es in groups.items():
        if len(es) < 2:
            raise ValueError(f"conflict group {g} has a single member")
        if len({(e.subject, e.predicate) for e in es}) != 1:
            raise ValueError(f"conflict group {g} mixes slots")
        if len({e.value for e in es}) < 2:
            raise ValueError(f"conflict group {g} has no differing values")
