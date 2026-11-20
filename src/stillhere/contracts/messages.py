"""CanonicalMessage: one raw chat message, and the unified shapes ``memory.MemoryAPI`` returns.

A message belongs to one persona's history and is said by one speaker; neither says who it is about:

  * ``target_persona_id``: whose chat history holds the message (the persona the memory is built for).
  * ``speaker_id``: who said it. It may be the persona or the person they talk to.
  * The subject of any fact in the message is not a field here. If B tells A "我明天搬去上海", the message sits in
    A's history with ``speaker_id = B``, and only a later, explicit decision may produce a fact about B. It never
    becomes a fact about A, and ingesting a message creates no memory at all.

Only chat messages have a speaker. Calendar entries and life events are not messages: they stay sources and
memories, so nothing here invents a speaker for them.

Times: ``timestamp`` is when it was said (timezone-aware, offset preserved). ``recorded_at`` is when the system
learned it; when left out the store fills in its clock at ingestion. It is set explicitly only for a replay that
needs the original knowledge timeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import ClassVar, Dict, Optional, Tuple

from ._base import Contract, ValidationError, require_id, require_sha256, require_text
from .common import SourceRef

# A provenance entry cites one message with this locator; the store checks the message and its persona.
MESSAGE_LOCATOR_PREFIX = "message:"


def message_locator(message_id: str) -> str:
    require_id("message_id", message_id)
    return MESSAGE_LOCATOR_PREFIX + message_id


@dataclass(frozen=True)
class CanonicalMessage(Contract):
    conversation_id: str
    message_id: str
    target_persona_id: str
    speaker_id: str
    timestamp: datetime
    content: str
    source_type: str                      # e.g. "wechat_export"; the importer's format name
    source_id: str                        # a registered SourceRef
    locator: Optional[str] = None         # where inside the source, e.g. "line:42"
    recorded_at: Optional[datetime] = None
    _omit_when_empty: ClassVar[Tuple[str, ...]] = ("locator", "recorded_at")

    def validate(self) -> None:
        for name in ("conversation_id", "message_id", "target_persona_id", "speaker_id", "source_type", "source_id"):
            require_id(name, getattr(self, name))
        require_text("content", self.content)
        if self.locator is not None:
            require_text("locator", self.locator)
        if self.recorded_at is not None and self.recorded_at < self.timestamp:
            raise ValidationError("a message cannot be recorded before it was said")


@dataclass(frozen=True)
class IngestResult(Contract):
    added: int
    skipped: int          # already stored with the same content


class HitKind(str, Enum):
    MEMORY = "memory"
    MESSAGE = "message"


@dataclass(frozen=True)
class EvidenceHit(Contract):
    """One retrieval result in the shape every ``MemoryAPI`` search returns, for a memory or a raw message.

    ``source_id`` and ``locator`` are the first provenance entry of a memory (``MemoryAPI.get_evidence`` lists
    them all). ``speaker_id`` of a memory is the speaker of the message it cites, if it cites one; ``subject_id`` is
    set for memories only, because a raw message has no decided subject.
    """

    kind: HitKind
    persona_id: str
    text: str                                # statement of a memory, content of a message
    source_id: str
    score: float
    retriever: str
    memory_id: Optional[str] = None
    message_id: Optional[str] = None
    subject_id: Optional[str] = None
    speaker_id: Optional[str] = None
    conversation_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    recorded_at: Optional[datetime] = None
    locator: Optional[str] = None
    retrieval_origin: Optional[str] = None        # "direct" or "expanded"; empty on the legacy path
    seed_record_key: Optional[str] = None
    relation_type: Optional[str] = None
    path_provenance: Optional[str] = None
    _omit_when_empty: ClassVar[Tuple[str, ...]] = ("retrieval_origin", "seed_record_key", "relation_type", "path_provenance")

    def validate(self) -> None:
        require_id("persona_id", self.persona_id)
        require_text("text", self.text)
        require_id("source_id", self.source_id)
        require_text("retriever", self.retriever)
        if self.kind is HitKind.MEMORY:
            if self.memory_id is None or self.message_id is not None or self.subject_id is None:
                raise ValidationError("a memory hit has a memory_id and a subject_id, and no message_id")
        elif self.message_id is None or self.memory_id is not None or self.subject_id is not None:
            raise ValidationError("a message hit has a message_id, and no memory_id or subject_id")
        if self.kind is HitKind.MESSAGE and self.speaker_id is None:
            raise ValidationError("a message hit has a speaker_id")


@dataclass(frozen=True)
class EvidenceItem(Contract):
    """One record of an evidence pack. An expanded item names the seed and the stored link that reached it."""

    record_key: str
    record_id: str
    persona_id: str
    source_id: str
    text: str
    final_score: float
    retrieval_origin: str = "direct"
    source_locator: Optional[str] = None
    event_time: Optional[datetime] = None
    known_at: Optional[datetime] = None
    seed_record_key: Optional[str] = None
    relation_type: Optional[str] = None
    path_provenance: Optional[str] = None

    def validate(self) -> None:
        if self.retrieval_origin not in ("direct", "expanded"):
            raise ValidationError(f"unknown retrieval_origin {self.retrieval_origin!r}")
        complete = all((self.seed_record_key, self.relation_type, self.path_provenance))
        if (self.retrieval_origin == "expanded") != complete:
            raise ValidationError("expanded evidence has a seed, a relation and a provenance; direct evidence has none")


@dataclass(frozen=True)
class EvidencePack(Contract):
    """A traceable set of records for one question. It is not a summary and merges nothing."""

    persona_id: str
    question: str
    as_of: datetime
    known_at: datetime
    items: Tuple[EvidenceItem, ...] = ()

    def validate(self) -> None:
        require_id("persona_id", self.persona_id)
        if any(i.persona_id != self.persona_id for i in self.items):
            raise ValidationError("an evidence pack holds one persona")


@dataclass(frozen=True)
class Citation(Contract):
    """One provenance entry of a persona's memory, with the raw message it cites, if any."""

    memory_id: str
    locator: Optional[str] = None
    quote: Optional[str] = None
    observed_at: Optional[datetime] = None
    message: Optional[CanonicalMessage] = None


@dataclass(frozen=True)
class EvidenceBundle(Contract):
    """What one persona may see of a source: its definition, their citations and their message count."""

    persona_id: str
    source: SourceRef
    citations: Tuple[Citation, ...] = ()
    message_count: int = 0

    def validate(self) -> None:
        require_id("persona_id", self.persona_id)


@dataclass(frozen=True)
class SnapshotInfo(Contract):
    path: str
    sha256: str
    size_bytes: int
    schema_version: int
    counts: Dict[str, int]

    def validate(self) -> None:
        require_text("path", self.path)
        require_sha256("sha256", self.sha256)
