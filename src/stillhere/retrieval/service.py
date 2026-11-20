"""Connects the pipeline to ``MemoryAPI``. The memory module cannot import this one, so strategies are registered on it.

``enable_strategies`` adds ``fast`` and ``fast_multihop``, which use only the local database and call no API. ``quality`` sends
question text to a third party, so it exists only when the caller passes both a client and ``allow_external=True``. A configured
key alone never enables it.

Scope and relations are read from the database once and rebuilt when it changes, so a retraction takes effect on the next search.
"""
from __future__ import annotations

import dataclasses
import sqlite3
import threading
from datetime import datetime, tzinfo
from typing import Dict, List, Mapping, Optional

from stillhere.contracts import EvidenceHit, EvidenceItem, EvidencePack, HitKind
from stillhere.memory import MemoryAPI

from . import event_graph as eg
from . import profiles
from .alibaba_client import DashScopeClient
from .dense import DenseIndex, Document, EmbeddingCache, ScopeTable, corpus
from .hybrid import Candidate, HybridRetriever, RetrievalConfig, event_groups


def evidence_hit(api: MemoryAPI, c: Candidate, retriever: str = "hybrid") -> EvidenceHit:
    """The stored record behind a candidate, in the shape every search returns, with the path that found it."""
    path = {"retrieval_origin": c.retrieval_origin, "seed_record_key": c.seed_key, "relation_type": c.relation_type, "path_provenance": c.path_provenance}
    record = api.store.get(c.record_id)
    if record is not None:
        return dataclasses.replace(api._memory_hit(record, c.fusion_score, retriever), **path)
    m = api.get_message(c.persona_id, c.record_id)
    return EvidenceHit(kind=HitKind.MESSAGE, persona_id=c.persona_id, text=m.content, source_id=m.source_id, score=c.fusion_score, retriever=retriever,
                       message_id=m.message_id, speaker_id=m.speaker_id, conversation_id=m.conversation_id, timestamp=m.timestamp,
                       recorded_at=m.recorded_at, locator=m.locator, **path)


class RetrievalService:
    def __init__(self, api: MemoryAPI, *, indexes: Optional[Mapping[str, DenseIndex]] = None, client: Optional[DashScopeClient] = None,
                 cache: Optional[EmbeddingCache] = None, calendar: Optional[tzinfo] = None) -> None:
        self.api = api
        self._indexes = indexes
        self._client = client
        self._cache = cache
        self.calendar = calendar           # the persona's timezone for dates in a question; without one, the offset of as_of
        self._lock = threading.RLock()
        self._version = None
        self.retriever: Optional[HybridRetriever] = None

    def _state(self, conn: sqlite3.Connection) -> tuple:
        return (conn.execute("SELECT count(*) FROM memories").fetchone()[0], conn.execute("SELECT count(*) FROM messages").fetchone()[0],
                conn.execute("SELECT max(event_id) FROM memory_events").fetchone()[0])

    def refresh(self) -> None:
        with self._lock:
            conn = self.api.store._conn
            state = self._state(conn)
            if state == self._version:
                return
            personas = sorted({r[0] for r in conn.execute("SELECT persona_id FROM memories UNION SELECT target_persona_id FROM messages")})
            scope = ScopeTable(conn, personas)
            groups: Dict[str, str] = {}
            members = []
            documents: Dict[str, Document] = {}
            for pid in personas:
                groups.update(event_groups(conn, pid))
                members += eg.extract(conn, pid)
                documents.update({d.record_id: d for d in corpus(conn, pid)})
            if self._indexes:
                for index in set(map(id, self._indexes.values())):
                    next(i for i in self._indexes.values() if id(i) == index).bind(scope)
            self.retriever = HybridRetriever(self.api, scope, documents, groups=groups, indexes=self._indexes, client=self._client, cache=self._cache,
                                             graph=eg.EventGraph(members))
            self._version = state

    def run(self, config: RetrievalConfig, persona_id: str, query: str, as_of: datetime, known_at: datetime, k: int, include_ended: bool):
        self.refresh()
        if persona_id not in self.retriever.scope.tables:
            return None
        return self.retriever.retrieve(dataclasses.replace(config, final_k=k, include_ended=include_ended), persona_id, query, as_of, known_at, calendar=self.calendar)

    def strategy(self, config: RetrievalConfig):
        def search(persona_id: str, query: str, *, as_of: datetime, known_at: datetime, k: int, include_ended: bool) -> List[EvidenceHit]:
            result = self.run(config, persona_id, query, as_of, known_at, k, include_ended)
            return [] if result is None else [evidence_hit(self.api, c, config.name) for c in result.final]
        return search

    def explain(self, config: RetrievalConfig, persona_id: str, query: str, *, as_of: datetime, known_at: datetime, k: int = 10, include_ended: bool = True) -> dict:
        """How a result came about: the date read from the question, the date candidates, and the channels behind each final record."""
        result = self.run(config, persona_id, query, as_of, known_at, k, include_ended)
        if result is None:
            return {"temporal_parse": None, "date_candidate_ids": [], "final": []}
        return {"temporal_parse": result.anchor.to_dict() if result.anchor else None, "records_in_window": result.date_window_records, "date_candidate_ids": list(result.channels.get("date_anchor", ())), "final": [{"record_key": c.key, "record_id": c.record_id, "source_channels": list(c.retrieval_channels), "score": c.fusion_score} for c in result.final]}

    def evidence_pack(self, config: RetrievalConfig, persona_id: str, query: str, *, as_of: datetime, known_at: datetime, k: int = 10,
                      include_ended: bool = True) -> EvidencePack:
        """A traceable set of records, one item per record. Nothing is summarised or merged."""
        result = self.run(config, persona_id, query, as_of, known_at, k, include_ended)
        items = []
        for c in result.final if result else []:
            hit = evidence_hit(self.api, c, config.name)
            items.append(EvidenceItem(record_key=c.key, record_id=c.record_id, persona_id=persona_id, source_id=hit.source_id, source_locator=hit.locator,
                                      event_time=hit.timestamp or hit.valid_from, known_at=hit.recorded_at, text=hit.text, retrieval_origin=c.retrieval_origin,
                                      seed_record_key=c.seed_key, relation_type=c.relation_type, path_provenance=c.path_provenance, final_score=c.fusion_score))
        return EvidencePack(persona_id=persona_id, question=query, as_of=as_of, known_at=known_at, items=tuple(items))


def enable_strategies(api: MemoryAPI, *, client: Optional[DashScopeClient] = None, indexes: Optional[Mapping[str, DenseIndex]] = None,
                      cache: Optional[EmbeddingCache] = None, allow_external: bool = False, calendar: Optional[tzinfo] = None) -> RetrievalService:
    service = RetrievalService(api, indexes=indexes, client=client if allow_external else None, cache=cache, calendar=calendar)
    api.register_strategy("fast", service.strategy(profiles.get("fast")))
    api.register_strategy("fast_multihop", service.strategy(profiles.get("fast_multihop")))
    api.register_strategy("fast_time", service.strategy(profiles.get("fast_time")))
    if allow_external and client is not None and indexes:
        api.register_strategy("quality", service.strategy(dataclasses.replace(profiles.get("quality"), parallel=False)))
    return service
