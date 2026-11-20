"""MemoryRecord: a source-linked, time-scoped statement about a persona's world, with correction and conflict links.

Two clocks:
  * ``valid_from`` / ``valid_to``: when the statement is true in the persona's world, half-open [from, to).
    ``None`` means unbounded on that side.
  * ``recorded_at``: when the system learned it. Corrections create a new record; the old one is kept with
    status ``superseded`` so history stays auditable.

Conflicts: ``cardinality`` says whether a slot holds one value at a time (``single``) or several (``multi``).
``contradicts()`` is conservative and only fires for two ``single`` records; ``unknown`` never conflicts.

This module only defines the record. The storage engine (planned: SQLite, transactional) implements
``stillhere.contracts.protocols.MemoryStore``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple

from ._base import SCHEMA_VERSION, Contract, ValidationError, require_id, require_text, require_unique
from .common import DataOrigin, Provenance


class MemoryKind(str, Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    EVENT = "event"
    RELATIONSHIP = "relationship"


class Cardinality(str, Enum):
    """How many values a slot may hold at the same time. Conflicts are only inferred for ``single`` slots."""

    SINGLE = "single"    # one value at a time, e.g. home_city
    MULTI = "multi"      # several values may coexist, e.g. hobby
    UNKNOWN = "unknown"  # not declared; treated conservatively (no automatic conflict)


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"  # replaced by a correction; see superseded_by
    RETRACTED = "retracted"    # withdrawn without replacement


@dataclass(frozen=True)
class MemoryRecord(Contract):
    memory_id: str
    persona_id: str
    kind: MemoryKind
    subject: str        # who or what the statement is about, e.g. "persona", "user", "cat"
    attribute: str      # slot name, e.g. "home_city"; (persona_id, subject, attribute) is the conflict key
    value: str
    statement: str      # natural-language form used for retrieval and prompting
    provenance: Tuple[Provenance, ...]
    origin: DataOrigin  # real / simulated / synthetic
    recorded_at: datetime
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    status: MemoryStatus = MemoryStatus.ACTIVE
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None
    correction_reason: Optional[str] = None
    conflicts_with: Tuple[str, ...] = ()
    confidence: float = 1.0
    cardinality: Cardinality = Cardinality.UNKNOWN
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValidationError(f"unsupported schema_version {self.schema_version!r}")
        require_id("memory_id", self.memory_id)
        require_id("persona_id", self.persona_id)
        for name in ("subject", "attribute", "value", "statement"):
            require_text(name, getattr(self, name))
        if not self.provenance:
            raise ValidationError("a memory must cite at least one provenance entry")
        if self.valid_from and self.valid_to and self.valid_to <= self.valid_from:
            raise ValidationError("valid_to must be later than valid_from")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValidationError("confidence must be within [0, 1]")
        for link in ("supersedes", "superseded_by"):
            other = getattr(self, link)
            if other is not None:
                require_id(link, other)
                if other == self.memory_id:
                    raise ValidationError(f"{link} must not point at the record itself")
        if self.supersedes is not None and not (self.correction_reason or "").strip():
            raise ValidationError("a correction (supersedes set) needs a correction_reason")
        if (self.status is MemoryStatus.SUPERSEDED) != (self.superseded_by is not None):
            raise ValidationError("status 'superseded' and superseded_by must be set together")
        require_unique("conflicts_with", self.conflicts_with)
        if self.memory_id in self.conflicts_with:
            raise ValidationError("a memory cannot conflict with itself")
        for i in self.conflicts_with:
            require_id("conflicts_with[]", i)

    @property
    def slot(self) -> Tuple[str, str, str]:
        return (self.persona_id, self.subject, self.attribute)

    @property
    def is_active(self) -> bool:
        return self.status is MemoryStatus.ACTIVE

    def is_valid_at(self, when: datetime) -> bool:
        """True if the statement holds at ``when`` (ignores status; check ``is_active`` separately)."""
        return (self.valid_from is None or self.valid_from <= when) and (self.valid_to is None or when < self.valid_to)

    def overlaps(self, other: "MemoryRecord") -> bool:
        """True if the two validity intervals share any instant."""
        starts_before_other_ends = other.valid_to is None or self.valid_from is None or self.valid_from < other.valid_to
        other_starts_before_self_ends = self.valid_to is None or other.valid_from is None or other.valid_from < self.valid_to
        return starts_before_other_ends and other_starts_before_self_ends

    def contradicts(self, other: "MemoryRecord") -> bool:
        """True only when both records declare the slot ``single`` and hold different values over overlapping
        validity while both are active. Multi-valued or undeclared slots never yield an automatic conflict."""
        return (self.memory_id != other.memory_id and self.slot == other.slot
                and self.cardinality is Cardinality.SINGLE and other.cardinality is Cardinality.SINGLE
                and self.value != other.value and self.is_active and other.is_active and self.overlaps(other))


@dataclass(frozen=True)
class RetrievedMemory(Contract):
    """A memory returned by retrieval, with its score. Carried into InferenceRequest.memory_context."""

    record: MemoryRecord
    score: float
    retriever: str = "unspecified"  # e.g. "fts5", "slot_lookup"

    def validate(self) -> None:
        require_text("retriever", self.retriever)
