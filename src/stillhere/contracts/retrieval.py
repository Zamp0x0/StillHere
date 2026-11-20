"""Memory retrieval with temporal resolution: the query, the per-slot resolution and the result.

``MemoryStore.search`` returns keyword hits. This contract describes what a context builder needs on top of
that: which slots (``subject`` + ``attribute``) a query is about, what each slot holds at ``as_of``, what it
held before, which values were corrected away, which values disagree, and why each record was selected.

Statuses never pick a winner the data does not support:
  * ``SlotStatus.CONFLICT``: active values of a ``single`` slot disagree at ``as_of``; ``value`` stays None
    and every side is returned. Recording time never settles a conflict.
  * ``SlotStatus.UNKNOWN``: nothing is known for the requested time (the slot may still list earlier values).
  * ``RetrievalStatus.AMBIGUOUS_SUBJECT``: the query refers to someone or something ("她") that the query and
    its context do not identify; no records are returned rather than a guess.
  * ``corrected`` records were superseded by a correction: they are former beliefs, not former facts, and
    never appear in ``hits``.

``MemoryContextView`` is what a prompt shows of a result: the selected records grouped by slot, each marked
current, past, unverified (one side of a conflict) or corrected (a mis-record, shown only when asked about
corrections), plus the lookup status. ``InferenceRequest.memory_view`` carries it so the status is never reduced
to plain text before rendering.

Day 7 (Query Understanding V2): a message that asks several things is planned as atomic queries
(``stillhere.contracts.planning``). Its result carries a ``RetrievalBatch`` (each atom's records, sufficiency and
missing evidence) and its view is ``memory_block_v2``: one ``ContextQuery`` per asked query, so "你住哪？我住哪？"
shows the persona's city for one query and "no record" for the other instead of refusing both or mixing them.
A message that asks one thing keeps the Day 6 shapes (no ``batch``, a ``memory_block_v1`` view).

Day 7 Phase 5B (Temporal Query Engine): an atom with a calendar time (scope ``at``) is resolved against that
interval, not against ``as_of``: ``SlotResolution.at_time`` lists the active records whose validity overlaps it
(an SQL interval lookup), and the slot is ``at_time`` (one value for the whole time), ``segmented`` (the value
changed during it), ``partial_time`` (records cover only part of it) or ``unknown`` (a time gap: records exist
for other times only). A time gap is never filled with a neighbouring value, and no record is never read as
"nothing then". ``RetrievalResult.time`` / ``MemoryContextView.time`` carry the interval and
``evidence`` the reason a single query has no records (``time_gap``, ``time_unresolved``, ``life_period``), so
the prompt can say exactly what is missing. All new fields are left out while empty: earlier data reads as before.

This module only defines the shapes. ``stillhere.memory.retrieval.SlotRetriever`` implements
``stillhere.contracts.protocols.MemoryRetriever``; ``stillhere.inference.context`` builds the view.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import ClassVar, Optional, Tuple

from ._base import Contract, ValidationError, require_id, require_text, require_unique
from .memory import Cardinality, MemoryRecord, MemoryStatus, RetrievedMemory
from .planning import (AnswerLevel, DependencyKind, EvidenceState, RetrievalBatch, RetrievalStatus, Sufficiency,
                       TimeExpression, TimeScope)

MAX_LIMIT = 50
RETRIEVER_NAME = "slot_temporal_v0"


class SlotStatus(str, Enum):
    RESOLVED = "resolved"        # one value holds at as_of
    MULTI_VALUE = "multi_value"  # several values coexist (multi-valued or undeclared slot)
    CONFLICT = "conflict"        # active values of a single slot disagree; none is chosen
    HISTORICAL = "historical"    # historical scope: earlier values exist
    UNKNOWN = "unknown"          # nothing known for the requested time
    AT_TIME = "at_time"          # scope "at": one value holds for the whole asked interval
    SEGMENTED = "segmented"      # scope "at": the value changed during the asked interval; each holds for its time
    PARTIAL_TIME = "partial_time"  # scope "at": records cover only part of the asked interval


# Why a query has no records although it was asked properly; shown to the model as its own line.
EVIDENCE_REASONS = ("time_gap", "time_unresolved", "life_period")


class TemporalRole(str, Enum):
    CURRENT = "current"      # active and valid at as_of
    PAST = "past"            # active, validity ended at or before as_of
    CORRECTED = "corrected"  # superseded by a correction (only in SlotResolution.corrected)
    AT_TIME = "at_time"      # active and valid at some instant of the asked interval (scope "at")


@dataclass(frozen=True)
class MemoryQuery(Contract):
    persona_id: str
    text: str
    context: Tuple[str, ...] = ()        # earlier turns, newest last; used only to identify the subject
    as_of: Optional[datetime] = None     # valid-time instant; None means the retriever's clock
    subject: Optional[str] = None        # explicit subject scope, e.g. "persona" or "cat"
    time_scope: Optional[TimeScope] = None  # None: taken from the query wording, else current
    limit: int = 5
    context_roles: Tuple[str, ...] = ()  # speaker of each context turn, "user" or "persona"; empty: read as the user's
    known_at: Optional[datetime] = None  # what the system knew by then; see SlotRetriever.retrieve for the default
    _omit_when_empty: ClassVar[Tuple[str, ...]] = ("context_roles", "known_at")   # absent from data not using them

    def validate(self) -> None:
        require_id("persona_id", self.persona_id)
        if self.time_scope in (TimeScope.AT, TimeScope.UNRESOLVED):
            raise ValidationError(f"time_scope {self.time_scope.value!r} comes from the message's time words; an "
                                  "explicit time_scope is current, historical or all")
        if self.subject is not None:
            require_text("subject", self.subject)
        if self.context_roles and len(self.context_roles) != len(self.context):
            raise ValidationError(f"context_roles has {len(self.context_roles)} entries for {len(self.context)} "
                                  "context turns")
        if any(r not in ("user", "persona") for r in self.context_roles):
            raise ValidationError(f"context_roles must be 'user' or 'persona', got {list(self.context_roles)}")
        if not 1 <= self.limit <= MAX_LIMIT:
            raise ValidationError(f"limit must be within [1, {MAX_LIMIT}], got {self.limit}")


@dataclass(frozen=True)
class MemoryHit(Contract):
    """A selected record, why it was selected and where it sits in time."""

    record: MemoryRecord
    role: TemporalRole
    via: str                     # "slot" (structured slot match) or "keyword" (shared words)
    matched: Tuple[str, ...]     # query words or units that selected it; empty for an added conflict partner
    score: float

    def validate(self) -> None:
        if self.via not in ("slot", "keyword"):
            raise ValidationError(f"via must be 'slot' or 'keyword', got {self.via!r}")
        if self.role is TemporalRole.CORRECTED or self.record.status is not MemoryStatus.ACTIVE:
            raise ValidationError("a hit is an active current or past record")

    @property
    def source_ids(self) -> Tuple[str, ...]:
        return tuple(p.source_id for p in self.record.provenance)


@dataclass(frozen=True)
class SlotResolution(Contract):
    """Everything the store holds for one slot, split by time relative to ``as_of``."""

    subject: str
    attribute: str
    cardinality: Cardinality
    status: SlotStatus
    current: Tuple[MemoryRecord, ...] = ()     # active, valid at as_of
    past: Tuple[MemoryRecord, ...] = ()        # active, validity ended at or before as_of
    corrected: Tuple[MemoryRecord, ...] = ()   # superseded by a correction, in correction order
    conflict_ids: Tuple[str, ...] = ()         # current (or, scope "at", at_time) records that disagree
    value: Optional[str] = None                # only when status is RESOLVED or AT_TIME
    at_time: Tuple[MemoryRecord, ...] = ()     # scope "at": active records valid during the asked interval
    _omit_when_empty: ClassVar[Tuple[str, ...]] = ("at_time",)

    def validate(self) -> None:
        require_text("subject", self.subject)
        require_text("attribute", self.attribute)
        if (self.value is not None) != (self.status in (SlotStatus.RESOLVED, SlotStatus.AT_TIME)):
            raise ValidationError("value is set exactly when the slot is resolved (now or at the asked time)")
        if (self.status is SlotStatus.CONFLICT) != bool(self.conflict_ids):
            raise ValidationError("conflict_ids are listed exactly when the slot is in conflict")
        if not set(self.conflict_ids) <= {r.memory_id for r in self.current + self.at_time}:
            raise ValidationError("conflict_ids must be current (or at_time) records")
        if self.status in (SlotStatus.AT_TIME, SlotStatus.SEGMENTED, SlotStatus.PARTIAL_TIME) and not self.at_time:
            raise ValidationError(f"a {self.status.value} slot lists the records valid at the asked time")
        if any(r.status is not MemoryStatus.ACTIVE for r in self.at_time):
            raise ValidationError("at_time records must be active")
        if any(r.status is not MemoryStatus.SUPERSEDED for r in self.corrected):
            raise ValidationError("corrected records must be superseded")


@dataclass(frozen=True)
class RetrievalResult(Contract):
    status: RetrievalStatus
    query: Optional[MemoryQuery] = None        # None only for an invalid request
    as_of: Optional[datetime] = None           # the instant used
    time_scope: Optional[TimeScope] = None
    time_scope_source: Optional[str] = None    # "explicit", "query" or "default"
    subject: Optional[str] = None
    subject_source: Optional[str] = None       # "explicit", "query", "context" or "slot"
    subject_candidates: Tuple[str, ...] = ()   # for an ambiguous subject: subjects in memory that could fit
    slots: Tuple[SlotResolution, ...] = ()
    hits: Tuple[MemoryHit, ...] = ()           # deterministic order; conflict groups are never split
    truncated: bool = False                    # hits were dropped by the limit
    retriever: str = RETRIEVER_NAME
    errors: Tuple[str, ...] = ()
    batch: Optional[RetrievalBatch] = None     # Day 7 V2: the plan and each atom's outcome, for a message that asks
                                               # several things; absent otherwise (Day 6 format)
    time: Optional[TimeExpression] = None      # Phase 5B: the calendar time every atom asks about, if they share one
    evidence: Optional[str] = None             # Phase 5B: why a one-query message has no records (EVIDENCE_REASONS)
    known_at: Optional[datetime] = None        # the knowledge instant used, when the caller asked for one
    _omit_when_empty: ClassVar[Tuple[str, ...]] = ("batch", "time", "evidence", "known_at")

    def validate(self) -> None:
        require_text("retriever", self.retriever)
        if self.evidence is not None and self.evidence not in EVIDENCE_REASONS:
            raise ValidationError(f"evidence must be one of {EVIDENCE_REASONS}, got {self.evidence!r}")
        if self.evidence is not None and self.hits:
            raise ValidationError("evidence explains a lookup without records")
        require_unique("hit ids", [h.record.memory_id for h in self.hits])
        if self.batch is not None:
            ids = {h.record.memory_id for h in self.hits}
            if any(not set(a.memory_ids) <= ids for a in self.batch.atoms):
                raise ValidationError("an atom lists a record that is not among the hits")
            if ids != {i for a in self.batch.atoms for i in a.memory_ids}:
                raise ValidationError("every hit belongs to an atom of the batch")
        if (self.status is RetrievalStatus.OK) != bool(self.hits):
            raise ValidationError("status 'ok' and a non-empty hit list go together")
        if self.status in (RetrievalStatus.INVALID_REQUEST, RetrievalStatus.ERROR) and not self.errors:
            raise ValidationError(f"status {self.status.value!r} needs an error message")
        if self.status is not RetrievalStatus.INVALID_REQUEST and self.query is None:
            raise ValidationError("only an invalid request may omit the query")
        if self.query is not None:
            for h in self.hits:
                if h.record.persona_id != self.query.persona_id:
                    raise ValidationError(f"hit {h.record.memory_id} belongs to another persona")

    @property
    def memory_ids(self) -> Tuple[str, ...]:
        return tuple(h.record.memory_id for h in self.hits)

    @property
    def has_conflict(self) -> bool:
        return any(s.status is SlotStatus.CONFLICT for s in self.slots)

    def to_memory_context(self) -> Tuple[RetrievedMemory, ...]:
        """The hits as ``InferenceRequest.memory_context`` entries (corrected records are never included)."""
        return tuple(RetrievedMemory(record=h.record, score=h.score, retriever=self.retriever) for h in self.hits)


# ---------------------------------------------------------------- memory context view (what a prompt shows)

MEMORY_VIEW_VERSION = "memory_block_v1"
MEMORY_VIEW_V2 = "memory_block_v2"     # a view that also lists each asked query (a message that asks several things)


class FactRole(str, Enum):
    """How one record is presented to the model. Derived from ``TemporalRole`` and the slot status."""

    CURRENT = "current"        # true at as_of
    PAST = "past"              # was true before as_of, not true now
    UNVERIFIED = "unverified"  # one side of a conflict: true or false is not known, no side is chosen
    CORRECTED = "corrected"    # recorded by mistake and corrected: never a former value
    AT_TIME = "at_time"        # valid during the asked interval (scope "at"), whatever holds now


@dataclass(frozen=True)
class ContextFact(Contract):
    memory_id: str
    role: FactRole
    corrected_by: Optional[str] = None       # CORRECTED only: the record that replaced it
    record: Optional[MemoryRecord] = None    # CORRECTED only: a corrected record is never in memory_context

    def validate(self) -> None:
        require_id("memory_id", self.memory_id)
        corrected = self.role is FactRole.CORRECTED
        if corrected != (self.record is not None) or corrected != (self.corrected_by is not None):
            raise ValidationError("record and corrected_by are given exactly for a corrected fact")
        if self.record is not None:
            if self.record.memory_id != self.memory_id:
                raise ValidationError("a corrected fact's record must have its memory_id")
            if self.record.status is not MemoryStatus.SUPERSEDED:
                raise ValidationError("a corrected fact's record must be superseded")


@dataclass(frozen=True)
class ContextSlot(Contract):
    """One slot as shown to the model: its status and the facts selected for it, as one indivisible group."""

    subject: str
    attribute: str
    status: SlotStatus
    facts: Tuple[ContextFact, ...]

    def validate(self) -> None:
        require_text("subject", self.subject)
        require_text("attribute", self.attribute)
        shown = [f for f in self.facts if f.role is not FactRole.CORRECTED]
        if not shown:
            raise ValidationError(f"slot {self.subject}/{self.attribute} shows no current or past fact")
        unverified = [f for f in self.facts if f.role is FactRole.UNVERIFIED]
        if self.status is SlotStatus.CONFLICT:
            if len(unverified) < 2 or any(f.role is FactRole.CURRENT for f in self.facts):
                raise ValidationError("a conflict slot shows every side as unverified and none as current")
        elif unverified:
            raise ValidationError("unverified facts belong to a conflict slot")


@dataclass(frozen=True)
class ContextQuery(Contract):
    """One asked query as shown to the model (memory_block_v2): whose fact, which attributes, for what time, and
    whether the shown slots answer it. Closed-vocabulary fields only: the user's own words never enter the
    memory block."""

    query_id: str
    subject: Optional[str]
    attributes: Tuple[str, ...]
    time_scope: TimeScope
    sufficiency: Sufficiency
    reason: Optional[str] = None           # why records are missing ("no_record", "unowned:妈妈", ...)
    slots: Tuple[str, ...] = ()            # "subject/attribute" of the shown slots that answer this query
    omitted: bool = False                  # some of its records were left out by the memory budget
    time: Optional[TimeExpression] = None  # Phase 5B: the calendar time it asks about (scope "at"/"unresolved")
    _omit_when_empty: ClassVar[Tuple[str, ...]] = ("time",)

    def validate(self) -> None:
        require_id("query_id", self.query_id)
        if self.subject is not None:
            require_id("subject", self.subject)
        answered = self.sufficiency in (Sufficiency.SUFFICIENT, Sufficiency.CONFLICT, Sufficiency.PARTIAL)
        if self.slots and not answered:
            raise ValidationError(f"{self.query_id}: only an answered query points at slots")
        if answered and not (self.slots or self.omitted):
            raise ValidationError(f"{self.query_id}: an answered query shows its slots or says they were omitted")


@dataclass(frozen=True)
class ContextDependency(Contract):
    kind: DependencyKind
    query_ids: Tuple[str, ...]
    evidence: EvidenceState
    answer: AnswerLevel
    relation: Optional[str] = None


@dataclass(frozen=True)
class MemoryContextView(Contract):
    """The structured form of a request's memory block. ``InferenceRequest.memory_context`` holds the records the
    model may cite; this view says how each one is to be read (current, past, unverified) and carries the lookup
    status, so the meaning survives every step until the prompt text is rendered.

    Only lookups that completed are viewed: ``ok``, ``no_match`` and ``ambiguous_subject``. A failed lookup puts
    no memory in the request at all (fail closed)."""

    status: RetrievalStatus
    as_of: datetime
    time_scope: TimeScope
    slots: Tuple[ContextSlot, ...] = ()
    omitted_slots: int = 0                # whole slots left out by the memory budget
    version: str = MEMORY_VIEW_VERSION
    queries: Tuple[ContextQuery, ...] = ()            # memory_block_v2 only: one entry per asked query
    dependencies: Tuple[ContextDependency, ...] = ()  # memory_block_v2 only
    time: Optional[TimeExpression] = None             # Phase 5B: the calendar time asked (memory_block_v1)
    evidence: Optional[str] = None                    # Phase 5B: why the one query has no records
    _omit_when_empty: ClassVar[Tuple[str, ...]] = ("queries", "dependencies", "time", "evidence")

    def validate(self) -> None:
        if self.evidence is not None and self.evidence not in EVIDENCE_REASONS:
            raise ValidationError(f"evidence must be one of {EVIDENCE_REASONS}, got {self.evidence!r}")
        if self.version not in (MEMORY_VIEW_VERSION, MEMORY_VIEW_V2):
            raise ValidationError(f"unsupported memory view version {self.version!r}")
        if (self.version == MEMORY_VIEW_V2) != bool(self.queries):
            raise ValidationError("a memory_block_v2 view lists its queries; a memory_block_v1 view lists none")
        if self.dependencies and not self.queries:
            raise ValidationError("dependencies need the queries they link")
        if self.queries:
            require_unique("view query ids", [q.query_id for q in self.queries])
            shown = {f"{s.subject}/{s.attribute}" for s in self.slots}
            asked = {k for q in self.queries for k in q.slots}
            if not asked <= shown:
                raise ValidationError(f"queries point at slots the view does not show: {sorted(asked - shown)}")
            if shown - asked:      # a fact no asked query owns would be evidence for the wrong question
                raise ValidationError(f"slots no query asked for: {sorted(shown - asked)}")
            known = {q.query_id for q in self.queries}
            if any(not set(d.query_ids) <= known for d in self.dependencies):
                raise ValidationError("a dependency names a query the view does not list")
        if self.status not in (RetrievalStatus.OK, RetrievalStatus.NO_MATCH, RetrievalStatus.AMBIGUOUS_SUBJECT):
            raise ValidationError(f"a failed lookup ({self.status.value}) is never shown to the model")
        if self.omitted_slots < 0:
            raise ValidationError("omitted_slots must be >= 0")
        if self.status is RetrievalStatus.OK:
            if not self.slots and not self.omitted_slots:
                raise ValidationError("an 'ok' view shows or omits at least one slot")
        elif self.slots or self.omitted_slots:
            raise ValidationError(f"a {self.status.value!r} view carries no slots")
        require_unique("view memory ids", [f.memory_id for s in self.slots for f in s.facts])
        require_unique("view slots", [(s.subject, s.attribute) for s in self.slots])

    @property
    def context_ids(self) -> Tuple[str, ...]:
        """Ids the request's memory_context must hold, in order: every shown fact except corrected ones."""
        return tuple(f.memory_id for s in self.slots for f in s.facts if f.role is not FactRole.CORRECTED)

    @property
    def corrected_ids(self) -> Tuple[str, ...]:
        return tuple(f.memory_id for s in self.slots for f in s.facts if f.role is FactRole.CORRECTED)
