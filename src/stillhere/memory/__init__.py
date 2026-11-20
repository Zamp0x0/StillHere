"""Long-term temporal memory: a local SQLite engine implementing stillhere.contracts.MemoryStore
(source-linked records, validity intervals, conflict tracking, transactional corrections, keyword retrieval).

May import: stillhere.contracts.

- ``sqlite_store``: ``SQLiteMemoryStore`` (standard-library ``sqlite3``): append-only records, provenance
  rows, bitemporal queries, conflict pairs, transactional ``correct``/``retract``, idempotent retries,
  write gate (registered sources, origin rules, review for real data), audit log.
- ``api``: ``MemoryAPI``, the public entry: a thin layer over the store that requires a persona on every call,
  applies the safe ``as_of`` / ``known_at`` default and returns one hit shape for memories and raw messages.
- ``schema``: DDL, ``PRAGMA user_version`` migrations, FTS5 probe.
- ``search``: CJK bigram / word segmentation for FTS5 keyword search and its scan fallback (not semantic).
- ``planner``: ``QueryPlanner`` (Day 7 Query Understanding V2): a message -> ``contracts.QueryPlan`` of atomic
  queries (owner, attribute, time, dialogue focus, comparisons), deterministic and local.
- ``retrieval``: ``SlotRetriever`` (``contracts.MemoryRetriever``): plans the message, then runs read-only slot
  and keyword retrieval with temporal resolution per atom (current / past / corrected values, conflicts,
  ambiguous subjects, match evidence, ``RetrievalBatch`` for messages that ask several things).
- ``semantic_parser``: optional DeepSeek JSON-mode parser the planner may consult for an asked part it cannot
  label; off by default, candidates only.

Not here: retrieval inside the chat loop (``stillhere.inference``), multi-turn correction scenarios, the
review queue in front of the write gate.
"""
from .schema import SCHEMA_VERSION
from .planner import QueryPlanner
from .retrieval import DEFAULT_SLOT_CUES, DEFAULT_SUBJECT_NOUNS, SlotCue, SlotRetriever
from .sqlite_store import (DuplicateMemoryId, DuplicateSourceId, MemoryConflict, MemoryEvent, MemoryNotFound,
                           MemoryStoreError, MessageConflict, PersonaScopeError, ReadOnlyStore, SchemaMismatch,
                           SQLiteMemoryStore, StaleWrite, WriteRejected, resolve_times)
from .api import MemoryAPI

__all__ = [
    "SCHEMA_VERSION", "DEFAULT_SLOT_CUES", "DEFAULT_SUBJECT_NOUNS", "QueryPlanner", "SlotCue", "SlotRetriever", "DuplicateMemoryId", "DuplicateSourceId", "MemoryConflict", "MemoryEvent", "MemoryNotFound",
    "MemoryStoreError", "SQLiteMemoryStore", "StaleWrite", "WriteRejected", "MemoryAPI", "MessageConflict",
    "PersonaScopeError", "ReadOnlyStore", "SchemaMismatch", "resolve_times",
]
