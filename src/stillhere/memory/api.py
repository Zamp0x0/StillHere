"""MemoryAPI: the public entry to the memory core, a thin layer over ``SQLiteMemoryStore``.

The store stays the only place that holds or queries data. This layer adds three things:

  * **Persona scope.** Every read and write names a persona. A request for another persona's record, message or
    source raises ``PersonaScopeError``; it never returns or changes anything.
  * **The safe time default** (``resolve_times``): ``as_of`` alone means ``known_at = as_of``, so looking at the past
    cannot see what was learned later. Answering about the past with today's knowledge takes an explicit newer
    ``known_at``. With neither, both are the clock's now.
  * **One hit shape** (``EvidenceHit``) for memories and raw messages.

``search`` uses the store's FTS5 path. ``retrieve`` is the older planner and slot path (``SlotRetriever``): it
loads the persona's records on every call, so it is not the path for long histories.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from stillhere.contracts import (MESSAGE_LOCATOR_PREFIX, CanonicalMessage, EvidenceBundle, EvidenceHit, HitKind,
                                 IngestResult, MemoryRecord, Provenance, RetrievalResult, SnapshotInfo, SourceRef)
from stillhere.contracts._base import parse_datetime, require_id

from .retrieval import SlotRetriever
from .sqlite_store import PersonaScopeError, SQLiteMemoryStore, resolve_times

Time = Union[datetime, str, None]


def _time(value: Time) -> Optional[datetime]:
    return parse_datetime(value) if isinstance(value, str) else value


class MemoryAPI:
    """``path`` is the database file (or ``:memory:``). ``read_only=True`` opens an existing database without
    changing it; writes then raise ``ReadOnlyStore``. ``persona_names`` is used only by ``retrieve``."""

    def __init__(self, path: Union[str, "os.PathLike[str]"] = ":memory:", *, read_only: bool = False,
                 public_only: bool = True, clock: Optional[Callable[[], datetime]] = None, fts: Optional[bool] = None,
                 persona_names: Optional[Mapping[str, Sequence[str]]] = None) -> None:
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.store = SQLiteMemoryStore(path, read_only=read_only, public_only=public_only, clock=self._clock, fts=fts)
        self._retriever = SlotRetriever(self.store, persona_names=persona_names, clock=self._clock)
        self._strategies: Dict[str, Callable[..., List[EvidenceHit]]] = {}

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "MemoryAPI":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _times(self, as_of: Time, known_at: Time):
        return resolve_times(_time(as_of), _time(known_at), self._clock)

    # ------------------------------------------------------------------ writes

    def register_source(self, source: SourceRef) -> SourceRef:
        return self.store.register_source(source)

    def ingest_messages(self, messages: Iterable[CanonicalMessage]) -> IngestResult:
        """Store raw messages, all or none; re-sending them adds nothing. No memory is created from them."""
        return self.store.ingest_messages(messages)

    def add_facts(self, persona_id: str, records: Iterable[MemoryRecord], *,
                  reviewer: Optional[str] = None) -> List[MemoryRecord]:
        """Add the persona's records. All are checked for scope before the first write; each is then its own
        transaction, and adding a record again returns the stored one, so a failed batch can simply be re-sent."""
        require_id("persona_id", persona_id)
        records = list(records)
        for r in records:
            if r.persona_id != persona_id:
                raise PersonaScopeError(f"record {r.memory_id} does not belong to persona {persona_id}")
        return [self.store.add(r, reviewer=reviewer) for r in records]

    def import_facts(self, persona_id: str, records: Iterable[MemoryRecord], *,
                     reviewer: Optional[str] = None) -> IngestResult:
        """Add the persona's records in one transaction (``store.import_records``: the same write gate, provenance
        checks and triggers). Records already stored identically are skipped; the same id with other content raises
        ``DuplicateMemoryId`` and the batch rolls back."""
        require_id("persona_id", persona_id)
        records = list(records)
        for r in records:
            if r.persona_id != persona_id:
                raise PersonaScopeError(f"record {r.memory_id} does not belong to persona {persona_id}")
        before = self.store.stats()["counts"]["memories"]
        self.store.import_records(records, reviewer=reviewer)
        added = self.store.stats()["counts"]["memories"] - before
        return IngestResult(added=added, skipped=len(records) - added)

    def correct(self, persona_id: str, memory_id: str, replacement: MemoryRecord, reason: str, *,
                reviewer: Optional[str] = None) -> MemoryRecord:
        require_id("persona_id", persona_id)
        return self.store.correct(memory_id, replacement, reason, reviewer=reviewer, persona_id=persona_id)

    def retract(self, persona_id: str, memory_id: str, reason: str, *, evidence: Optional[Provenance] = None,
                reviewer: Optional[str] = None, at: Time = None) -> MemoryRecord:
        """``at``: when the retraction became known, for a late-arriving source (default: now)."""
        require_id("persona_id", persona_id)
        return self.store.retract(memory_id, reason, evidence=evidence, reviewer=reviewer, persona_id=persona_id, at=_time(at))

    def transaction(self):
        """One atomic batch: every write made inside commits together or not at all (``with api.transaction(): ...``)."""
        return self.store.transaction()

    # ------------------------------------------------------------------ reads

    def get(self, persona_id: str, memory_id: str, *, known_at: Time = None) -> Optional[MemoryRecord]:
        require_id("persona_id", persona_id)
        return self.store.get(memory_id, persona_id=persona_id, known_at=_time(known_at))

    def get_message(self, persona_id: str, message_id: str) -> Optional[CanonicalMessage]:
        require_id("persona_id", persona_id)
        return self.store.get_message(persona_id, message_id)

    def resolve_times(self, as_of: Time = None, known_at: Time = None):
        """The ``(as_of, known_at)`` every read of this API uses for these arguments (see ``resolve_times`` in the store)."""
        return self._times(as_of, known_at)

    def conversation(self, persona_id: str, conversation_id: str, *, as_of: Time = None, known_at: Time = None):
        """All messages of one conversation said by ``as_of`` and known by ``known_at`` (the same safe default as ``search``),
        oldest first, and the number of its messages that time hides: ``(messages, hidden)``. Never truncated."""
        require_id("persona_id", persona_id)
        when, known = self._times(as_of, known_at)
        return self.store.conversation_messages(persona_id, conversation_id, as_of=when, known_at=known)

    def get_source(self, persona_id: str, source_id: str) -> Optional[SourceRef]:
        """The registered source, or None. Sources are shared metadata (a dataset file), not one persona's record."""
        require_id("persona_id", persona_id)
        return self.store.get_source(source_id)

    def history(self, persona_id: str, subject: str, attribute: str, *, known_at: Time = None) -> List[MemoryRecord]:
        require_id("persona_id", persona_id)
        return self.store.history(persona_id, subject, attribute, known_at=_time(known_at))

    def events(self, persona_id: str, memory_id: str):
        require_id("persona_id", persona_id)
        return self.store.events(memory_id, persona_id=persona_id)

    def facts_at(self, persona_id: str, *, as_of: Time = None, known_at: Time = None,
                 subject: Optional[str] = None, attribute: Optional[str] = None) -> List[MemoryRecord]:
        require_id("persona_id", persona_id)
        when, known = self._times(as_of, known_at)
        return self.store.facts_at(persona_id, as_of=when, known_at=known, subject=subject, attribute=attribute)

    def facts_during(self, persona_id: str, *, start: Time, end: Time, known_at: Time = None,
                     subject: Optional[str] = None, attribute: Optional[str] = None) -> List[MemoryRecord]:
        """Records valid during ``[start, end)``. ``known_at`` defaults to ``end``: nothing learned after the interval."""
        require_id("persona_id", persona_id)
        return self.store.facts_during(persona_id, start=_time(start), end=_time(end), subject=subject,
                                       attribute=attribute, known_at=_time(known_at))

    def register_strategy(self, name: str, search: Callable[..., List[EvidenceHit]]) -> None:
        """Add a named retrieval path, called with the persona, the query and resolved times. ``legacy`` cannot be replaced."""
        if name == "legacy":
            raise ValueError("the legacy path cannot be replaced")
        self._strategies[name] = search

    def strategies(self) -> List[str]:
        return ["legacy", *sorted(self._strategies)]

    def search(self, persona_id: str, query: str, *, as_of: Time = None, known_at: Time = None, k: int = 10,
               include_ended: bool = False, strategy: str = "legacy") -> List[EvidenceHit]:
        """``include_ended`` also finds records whose validity ended by ``as_of``, such as events that happened.

        ``strategy`` names a registered path. The default ``legacy`` searches memories only; other paths may also return
        messages, so they are chosen by name and never replace it silently."""
        require_id("persona_id", persona_id)
        when, known = self._times(as_of, known_at)
        if strategy != "legacy":
            if strategy not in self._strategies:
                raise ValueError(f"strategy {strategy!r} is not enabled; available: {self.strategies()}")
            return self._strategies[strategy](persona_id, query, as_of=when, known_at=known, k=k, include_ended=include_ended)
        return [self._memory_hit(h.record, h.score, h.retriever)
                for h in self.store.search(persona_id, query, as_of=when, known_at=known, limit=k,
                                           include_ended=include_ended)]

    def search_messages(self, persona_id: str, query: str, *, as_of: Time = None, known_at: Time = None, k: int = 10,
                        speaker_id: Optional[str] = None) -> List[EvidenceHit]:
        """Messages of the persona's history said by ``as_of`` and known by ``known_at``."""
        require_id("persona_id", persona_id)
        when, known = self._times(as_of, known_at)
        retriever = "messages_fts" if self.store.message_fts_enabled else "message_scan"
        return [EvidenceHit(kind=HitKind.MESSAGE, persona_id=persona_id, text=m.content, source_id=m.source_id,
                            score=float(score), retriever=retriever, message_id=m.message_id,
                            speaker_id=m.speaker_id, conversation_id=m.conversation_id, timestamp=m.timestamp,
                            recorded_at=m.recorded_at, locator=m.locator)
                for m, score in self.store.search_messages(persona_id, query, as_of=when, known_at=known, limit=k,
                                                           speaker_id=speaker_id)]

    def retrieve(self, persona_id: str, text: str, *, as_of: Time = None, known_at: Time = None,
                 **options) -> RetrievalResult:
        """The planner and slot path, under the same time default as ``search``. Not for long histories (see above)."""
        require_id("persona_id", persona_id)
        when, known = self._times(as_of, known_at)
        return self._retriever.retrieve(persona_id, text, as_of=when, known_at=known, **options)

    def get_evidence(self, persona_id: str, source_id: str) -> EvidenceBundle:
        require_id("persona_id", persona_id)
        return self.store.evidence(persona_id, source_id)

    def _memory_hit(self, record: MemoryRecord, score: float, retriever: str) -> EvidenceHit:
        first = record.provenance[0]
        speaker = None
        for p in record.provenance:
            if p.locator is not None and p.locator.startswith(MESSAGE_LOCATOR_PREFIX):
                cited = self.store.get_message(record.persona_id, p.locator[len(MESSAGE_LOCATOR_PREFIX):])
                speaker = None if cited is None else cited.speaker_id
                break
        return EvidenceHit(kind=HitKind.MEMORY, persona_id=record.persona_id, text=record.statement,
                           source_id=first.source_id, score=score, retriever=retriever, memory_id=record.memory_id,
                           subject_id=record.subject, speaker_id=speaker, valid_from=record.valid_from,
                           valid_to=record.valid_to, recorded_at=record.recorded_at, locator=first.locator)

    # ------------------------------------------------------------------ database

    def snapshot(self, dest_path: Union[str, "os.PathLike[str]"]) -> SnapshotInfo:
        return self.store.snapshot(dest_path)

    def stats(self) -> dict:
        return self.store.stats()

    def check_integrity(self) -> List[str]:
        return self.store.check_integrity()
