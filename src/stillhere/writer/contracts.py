"""Writer contracts: candidate facts, their verification verdicts and the write operations that may follow.

A candidate is what an extractor (a model) proposes. It is never a memory by itself: only a SUPPORTED candidate, after the
deterministic checks in ``verify``, becomes a ``MemoryRecord`` through ``MemoryAPI``. Everything else is kept in the batch log
for audit and never reaches the memories table.

Predicates come from a closed vocabulary, so the same fact said twice lands in the same slot and a model cannot invent a slot.
Events are subjects of their own (``event:<id>``) with one record per attribute: description, activity, status, planned time,
actual time and part of day. A status change is a new status record; the earlier one is closed at the change, never overwritten.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from stillhere.contracts import Cardinality, MemoryKind

WRITER_VERSION = "writer-v1"

CANDIDATE_TYPES = ("FACT", "EVENT", "PREFERENCE_STATEMENT", "PLAN", "COMMITMENT", "CORRECTION", "RETRACTION", "UNCERTAINTY")
EVENT_STATES = ("PLANNED", "SCHEDULED", "IN_PROGRESS", "COMPLETED", "CANCELLED", "UNKNOWN")
SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED, AMBIGUOUS = "SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "AMBIGUOUS"
ADD, ENRICH, SUPERSEDE, RETRACT, NOOP = "ADD", "ENRICH", "SUPERSEDE", "RETRACT", "NOOP"
CORRECTION, TEMPORAL_CHANGE, CONTRADICTION, DUPLICATE, INDEPENDENT_FACT, UNRESOLVED = (
    "CORRECTION", "TEMPORAL_CHANGE", "CONTRADICTION", "DUPLICATE", "INDEPENDENT_FACT", "UNRESOLVED")
PRECISIONS = ("exact", "day", "month", "year", "unknown")
DAYPARTS = ("早上", "上午", "中午", "下午", "晚上", "夜里")

# predicate -> (memory kind, cardinality, Chinese label). A fact outside this list is not written.
PREDICATES: Dict[str, Tuple[MemoryKind, Cardinality, str]] = {
    "lives_in": (MemoryKind.FACT, Cardinality.SINGLE, "住在"),
    "works_at": (MemoryKind.FACT, Cardinality.SINGLE, "在……工作"),
    "job": (MemoryKind.FACT, Cardinality.SINGLE, "职业是"),
    "school": (MemoryKind.FACT, Cardinality.SINGLE, "就读于"),
    "age": (MemoryKind.FACT, Cardinality.SINGLE, "年龄"),
    "birthday": (MemoryKind.FACT, Cardinality.SINGLE, "生日"),
    "pet": (MemoryKind.FACT, Cardinality.MULTI, "养了"),
    "owns": (MemoryKind.FACT, Cardinality.MULTI, "拥有"),
    "relationship": (MemoryKind.RELATIONSHIP, Cardinality.MULTI, "的关系"),
    "likes": (MemoryKind.PREFERENCE, Cardinality.MULTI, "喜欢"),
    "dislikes": (MemoryKind.PREFERENCE, Cardinality.MULTI, "不喜欢"),
    "routine": (MemoryKind.PREFERENCE, Cardinality.MULTI, "通常的安排"),
    "state": (MemoryKind.FACT, Cardinality.MULTI, "状态"),
    "event": (MemoryKind.EVENT, Cardinality.SINGLE, "事件"),
}
EVENT_ATTRIBUTES = ("event.actor", "event.description", "event.activity", "event.status", "event.planned_time", "event.time", "event.daypart", "event.participant")

# authority levels for everything a reader may see (M2 lifecycle)
RAW_EVIDENCE, EXPLICIT_MEMORY, DERIVED_OBSERVATION, UNVERIFIED_HYPOTHESIS = "RAW_EVIDENCE", "EXPLICIT_MEMORY", "DERIVED_OBSERVATION", "UNVERIFIED_HYPOTHESIS"


@dataclass
class Candidate:
    """One proposed atomic fact. ``source_message_ids`` are real message ids (the extractor's m1.. labels already mapped)."""

    candidate_id: str
    persona_id: str
    type: str
    subject: str                           # "self" or "person:<name>"
    predicate: str
    object: str
    polarity: str = "positive"
    assertion_type: str = "stated"         # "stated" | "inferred"
    certainty: str = "certain"             # "certain" | "uncertain"
    event_key: Optional[str] = None
    event_status: Optional[str] = None
    activity: Optional[str] = None
    daypart: Optional[str] = None
    planned_time: Optional[str] = None     # ISO date or datetime
    event_time: Optional[str] = None
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    temporal_precision: str = "unknown"
    corrects: bool = False
    retracts: bool = False
    refers_to: Optional[str] = None        # an existing memory id or event subject shown to the extractor
    source_message_ids: List[str] = field(default_factory=list)
    source_spans: List[str] = field(default_factory=list)
    extraction_version: str = WRITER_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Verdict:
    candidate: Candidate
    support: str
    reasons: List[str] = field(default_factory=list)
    dropped_fields: List[str] = field(default_factory=list)   # fields removed because their evidence was missing

    @property
    def committable(self) -> bool:
        return self.support in (SUPPORTED, PARTIALLY_SUPPORTED)


@dataclass
class Operation:
    op: str
    relation: str
    memory_id: Optional[str] = None        # the record written (ADD, ENRICH, SUPERSEDE) or acted on (RETRACT)
    target_id: Optional[str] = None        # the earlier record (SUPERSEDE, RETRACT, close of a TEMPORAL_CHANGE)
    candidate_id: Optional[str] = None
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
