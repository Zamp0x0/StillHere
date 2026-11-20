"""SQLiteMemoryStore: a local, append-only, bitemporal implementation of ``MemoryStore``.

Writes
  Every public write runs in one ``BEGIN IMMEDIATE`` transaction and either fully applies or rolls back.
  Nothing is physically deleted: a correction marks the old record ``superseded`` and inserts the
  replacement; a retraction marks the record ``retracted`` and logs the reason and evidence. Database
  triggers reject DELETE and any change to a record's content.

Write gate (see docs/ARCHITECTURE.md, "Memory write gate")
  * every provenance entry must cite a source registered with ``register_source`` whose origin equals the
    record's origin; a ``simulated`` record must cite only ``simulation`` sources; a ``real`` record may not
    cite a ``simulation`` source;
  * ``public_only=True`` (the default) accepts synthetic records and sources only;
  * a ``real`` record, and any correction or retraction of one, needs ``reviewer=`` (the owner's review);
  * a ``simulated`` record never corrects a ``real`` one, reviewed or not;
  * a record that contradicts an active one is stored with the conflict recorded, never as a replacement.
  Generated text is not trusted by default: it has no source kind of its own, so it can only enter as a
  ``simulated`` record, which cannot overwrite real facts.

Two clocks
  ``valid_from``/``valid_to`` is when a statement holds in the persona's world; ``recorded_at`` is when the
  system learned it, and ``status_changed_at`` when it stopped being believed. A record is believed at T when
  ``recorded_at <= T`` and it had not yet left ``active`` at T. Every read that takes ``known_at`` (``search``,
  ``facts_at``, ``facts_during``, ``history``, ``get``) applies that one rule and returns the record as it stood
  at T: a record corrected or retracted later reads as active, and links to records nobody knew yet are left
  out. Left out, ``known_at`` defaults so that a query cannot see later knowledge (``resolve_times``): ``as_of``
  alone means ``known_at = as_of``, and neither means the clock's now. Answering about the past with today's
  knowledge takes an explicit newer ``known_at``. ``get`` and ``history`` are audit reads: without ``known_at``
  they show everything stored, as it stands now.

Retries
  ``memory_id`` is the stable identity of a write. Re-sending the same record (same id, same content) returns
  the stored record without writing; the same id with different content raises ``DuplicateMemoryId``. An
  ``add`` that repeats an active record's content under a fresh id returns the existing record. A repeated
  ``correct`` returns the replacement already written; a repeated ``retract`` returns the retracted record
  without logging again. This makes retries safe (at-least-once delivery with idempotent effects); it is
  not an exactly-once guarantee for arbitrary callers.

Search is keyword matching over CJK bigrams and words (``stillhere.memory.search``), using FTS5 when the
SQLite build has it and a scan otherwise. It is not semantic search.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sqlite3
import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from stillhere.contracts import (MESSAGE_LOCATOR_PREFIX, CanonicalMessage, Citation, DataOrigin, EvidenceBundle,
                                 IngestResult, MemoryRecord, MemoryStatus, Provenance, PublicBoundaryError,
                                 RetrievedMemory, SnapshotInfo, SourceKind, SourceRef, ValidationError,
                                 assert_public_safe, canonical_json)

from .schema import SCHEMA_VERSION, fts5_available, migrate

LOAD_CACHE_BUCKETS = 2        # distinct ``known_at`` values kept (a benchmark day asks many questions at one time; memory stays bounded)
from .search import (BIGRAM_WEIGHT, UNIGRAM_WEIGHT, IndexTerms, index_terms, match_expression, overlap_score,
                     query_units)

FTS_RETRIEVER = "fts5_bigram"
SCAN_RETRIEVER = "keyword_scan"


class MemoryStoreError(Exception):
    """Base class for store errors. The store is unchanged whenever one is raised from a write."""


class WriteRejected(MemoryStoreError):
    """The write gate refused the record (origin, source, review or link rules)."""


class DuplicateMemoryId(MemoryStoreError):
    """The id is already stored with different content."""


class DuplicateSourceId(MemoryStoreError):
    """The source id is already registered with a different definition."""


class StaleWrite(MemoryStoreError):
    """The target is no longer active (already corrected by someone else, or retracted)."""


class MemoryNotFound(MemoryStoreError, LookupError):
    pass


class PersonaScopeError(MemoryStoreError):
    """The request named a persona that does not own the record, message or source."""


class MessageConflict(MemoryStoreError):
    """The message id is already stored with different content."""


class ReadOnlyStore(MemoryStoreError):
    """A write was attempted on a store opened with ``read_only=True``."""


class SchemaMismatch(MemoryStoreError):
    """A read-only open found a database at another schema version; it cannot be migrated from there."""


@dataclass(frozen=True)
class MemoryEvent:
    """One audit-log entry. ``action`` is add, import, correct, superseded or retract."""

    event_id: int
    memory_id: str
    action: str
    related_id: Optional[str]
    reason: Optional[str]
    source_id: Optional[str]
    locator: Optional[str]
    reviewer: Optional[str]
    recorded_at: datetime


@dataclass(frozen=True)
class MemoryConflict:
    memory_a: MemoryRecord
    memory_b: MemoryRecord
    origin: str          # "detected" (contradicts()) or "declared" (listed in conflicts_with by the writer)
    recorded_at: datetime

    @property
    def is_open(self) -> bool:
        return self.memory_a.is_active and self.memory_b.is_active


_STORE_MANAGED = ("status", "superseded_by", "conflicts_with")


def _digest(data: dict) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def content_sha256(record: MemoryRecord) -> str:
    """Hash of the submitted content, without the fields the store manages (status, links, conflicts)."""
    d = record.to_dict()
    for k in _STORE_MANAGED:
        d.pop(k)
    return _digest(d)


def duplicate_key(record: MemoryRecord) -> str:
    """Hash of what the record says and cites, without its id, its timestamps of recording or its links."""
    d = record.to_dict()
    for k in _STORE_MANAGED + ("memory_id", "recorded_at", "supersedes", "correction_reason"):
        d.pop(k)
    return _digest(d)


def _utc(dt: datetime) -> str:
    """Fixed-width UTC text, so string comparison in SQL orders instants correctly."""
    if not isinstance(dt, datetime):
        raise ValueError(f"expected a timezone-aware datetime, got {type(dt).__name__} {dt!r}")
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"datetime must be timezone-aware, got {dt!r}")
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _opt_utc(dt: Optional[datetime]) -> Optional[str]:
    return None if dt is None else _utc(dt)


def _from_utc(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)


def resolve_times(as_of: Optional[datetime], known_at: Optional[datetime],
                  clock: Callable[[], datetime]) -> Tuple[datetime, datetime]:
    """The (as_of, known_at) a query uses: the default that cannot leak later knowledge.

    Neither given: both are the clock's now. Only ``as_of``: ``known_at = as_of``, so looking at the past never sees
    what was learned later; answering about the past with today's knowledge takes an explicit newer ``known_at``.
    Only ``known_at``: ``as_of = known_at``, what was believed to hold at that time.
    """
    if as_of is None and known_at is None:
        now = clock()
        return now, now
    return (known_at if as_of is None else as_of), (as_of if known_at is None else known_at)


def _believed(known_at: str, *args: list, alias: str = "") -> str:
    """The SQL condition every temporal read shares: the belief at ``known_at`` (fixed-width UTC)."""
    args[0].extend([known_at, known_at])
    return f"{alias}recorded_at_utc <= ? AND ({alias}status_changed_at IS NULL OR {alias}status_changed_at > ?)"


def _message_sha256(m: CanonicalMessage) -> str:
    d = m.to_dict()
    d.pop("recorded_at", None)
    return _digest(d)


def _reviewed(reviewer: Optional[str]) -> bool:
    return isinstance(reviewer, str) and bool(reviewer.strip())


def _require_reason(reason: str) -> str:
    if not isinstance(reason, str) or not reason.strip():
        raise WriteRejected("a non-empty reason is required")
    return reason


def _terms(record: MemoryRecord) -> IndexTerms:
    return index_terms(record.statement, record.value, record.subject, record.attribute)


class SQLiteMemoryStore:
    """``MemoryStore`` backed by one SQLite file (or ``:memory:``), using only the standard library.

    ``fts``: None uses FTS5 when available, True requires it, False searches with the scan fallback.
    ``clock`` supplies "now" (default as_of for search, audit timestamps); inject a fixed one in tests.
    ``read_only=True`` opens an existing database with ``mode=ro``: nothing is migrated, set or rebuilt, and every
    write raises ``ReadOnlyStore``. A database at another schema version raises ``SchemaMismatch``.
    One instance may be shared between threads; several instances may open the same file.
    """

    def __init__(self, path: Union[str, "os.PathLike[str]"] = ":memory:", *, public_only: bool = True,
                 fts: Optional[bool] = None, clock: Optional[Callable[[], datetime]] = None,
                 timeout: float = 5.0, read_only: bool = False) -> None:
        self.path = os.fspath(path)
        self.public_only = public_only
        self.read_only = read_only
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.RLock()
        self._gen = 0                               # bumped by every write (commit or rollback): part of the read-cache key
        self._writing = 0                           # > 0 inside a write: reads then bypass the cache (they may see uncommitted rows)
        self._load_cache: "OrderedDict[Optional[str], Dict[str, MemoryRecord]]" = OrderedDict()
        self._load_cache_gen: Optional[tuple] = None
        if read_only:
            self._conn = self._connect_read_only(timeout)
        else:
            self._conn = sqlite3.connect(self.path, timeout=timeout, isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA foreign_keys = ON")
            if read_only:
                self._open_read_only(fts)
                return
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL")
            available = fts5_available(self._conn)
            if fts and not available:
                raise RuntimeError("FTS5 is not available in this SQLite build")
            migrate(self._conn, fts=available if fts is None else bool(fts))
            self._fts_table = self._has_table("memory_fts")
            self._message_fts_table = self._has_table("messages_fts")
            if self._fts_table and not available:
                raise RuntimeError("this database has an FTS5 index but this SQLite build has no FTS5")
            self.fts_enabled = self._fts_table and fts is not False
            self.message_fts_enabled = self._message_fts_table and fts is not False
            if self._fts_table or self._message_fts_table:
                with self._write() as conn:
                    if self._fts_table and not self._index_complete(conn, "memory_fts", "memories"):
                        self._rebuild_fts(conn)
                    if self._message_fts_table and not self._index_complete(conn, "messages_fts", "messages"):
                        self._rebuild_message_fts(conn)
        except BaseException:
            self._conn.close()
            raise

    def _connect_read_only(self, timeout: float) -> sqlite3.Connection:
        if self.path == ":memory:":
            raise ValueError("read_only needs a database file, not ':memory:'")
        file = Path(self.path).resolve()
        if not file.is_file():
            raise FileNotFoundError(f"no database at {self.path}")
        return sqlite3.connect(file.as_uri() + "?mode=ro", uri=True, timeout=timeout, isolation_level=None,
                               check_same_thread=False)

    def _open_read_only(self, fts: Optional[bool]) -> None:
        """Check the database as it is. A keyword index that is missing rows is not repaired here: search falls
        back to the scan, which is slower and complete, and ``check_integrity`` reports the gap."""
        try:
            version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        except sqlite3.DatabaseError as exc:
            raise SchemaMismatch(f"{self.path} cannot be read as a database: {exc}") from exc
        if version != SCHEMA_VERSION:
            raise SchemaMismatch(f"{self.path} has schema version {version}, this code reads {SCHEMA_VERSION}; "
                                 "open it for writing once to migrate it, or use a matching snapshot")
        self._conn.execute("PRAGMA query_only = ON")
        self._fts_table = self._has_table("memory_fts")
        self._message_fts_table = self._has_table("messages_fts")
        if fts and not self._fts_table:
            raise RuntimeError("this database has no FTS5 index")
        try:
            with self._read() as conn:
                self.fts_enabled = (self._fts_table and fts is not False
                                    and self._index_complete(conn, "memory_fts", "memories"))
                self.message_fts_enabled = (self._message_fts_table and fts is not False
                                            and self._index_complete(conn, "messages_fts", "messages"))
        except sqlite3.OperationalError as exc:
            raise RuntimeError(f"cannot read the FTS5 index in this SQLite build: {exc}") from exc

    def _has_table(self, name: str) -> bool:
        return self._conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                                  (name,)).fetchone() is not None

    @staticmethod
    def _index_complete(conn: sqlite3.Connection, index: str, table: str) -> bool:
        return (conn.execute(f"SELECT count(*) FROM {index}").fetchone()[0]
                == conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])

    # ------------------------------------------------------------------ connection and transactions

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "SQLiteMemoryStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        with self._lock:
            return self._conn.execute("PRAGMA user_version").fetchone()[0]

    @property
    def retriever(self) -> str:
        return FTS_RETRIEVER if self.fts_enabled else SCAN_RETRIEVER

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """One atomic batch of writes: every write inside joins it, and an exception rolls all of them back. Re-entrant."""
        with self._write() as conn:
            self._batch_depth = getattr(self, "_batch_depth", 0) + 1
            try:
                yield conn
            finally:
                self._batch_depth -= 1

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        if self.read_only:
            raise ReadOnlyStore(f"{self.path} is open read-only")
        if getattr(self, "_batch_depth", 0) and self._conn.in_transaction:
            with self._lock:                         # inside transaction(): the batch commits or rolls back
                yield self._conn
            return
        with self._lock:
            self._writing += 1
            self._gen += 1
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    yield self._conn
                    self._conn.execute("COMMIT")   # deferred foreign keys are checked here
                except BaseException:
                    if self._conn.in_transaction:
                        self._conn.execute("ROLLBACK")
                    raise
            finally:
                self._gen += 1
                self._writing -= 1

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        """A read transaction, so multi-statement reads see one snapshot. Inside ``transaction()`` it joins the batch (and sees its writes)."""
        if getattr(self, "_batch_depth", 0) and self._conn.in_transaction:
            with self._lock:
                yield self._conn
            return
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                yield self._conn
            finally:
                self._conn.execute("COMMIT")

    def _now(self) -> str:
        return _utc(self._clock())

    # ------------------------------------------------------------------ read cache (pure function of the database state)

    def read_generation(self) -> Optional[tuple]:
        """A token that changes whenever the stored state may have changed: this store's writes (commit or rollback), any row changed on
        this connection (``total_changes``, also raw ``conn.execute`` writes) and commits by other connections (``PRAGMA data_version``).
        None while a write is open (its reads may see uncommitted rows): callers must not cache then."""
        if self._writing or getattr(self, "_batch_depth", 0):
            return None
        with self._read() as conn:
            return self._token(conn)

    def _token(self, conn: sqlite3.Connection) -> tuple:
        # data_version is only refreshed once the connection holds a read snapshot (WAL): read one row first, inside the transaction
        conn.execute("SELECT 1 FROM memories LIMIT 1").fetchone()
        return (self._gen, conn.total_changes, conn.execute("PRAGMA data_version").fetchone()[0])

    def _cached_load(self, conn: sqlite3.Connection, memory_id: str, known_at: Optional[str]) -> Optional[MemoryRecord]:
        gen = None if (self._writing or getattr(self, "_batch_depth", 0)) else self._token(conn)
        if gen is None:
            return self._load_uncached(conn, memory_id, known_at)
        if gen != self._load_cache_gen:
            self._load_cache.clear()
            self._load_cache_gen = gen
        bucket = self._load_cache.get(known_at)
        if bucket is None:
            bucket = self._load_cache[known_at] = {}
            while len(self._load_cache) > LOAD_CACHE_BUCKETS:
                self._load_cache.popitem(last=False)
        else:
            self._load_cache.move_to_end(known_at)
        rec = bucket.get(memory_id)
        if rec is None:
            rec = self._load_uncached(conn, memory_id, known_at)
            if rec is not None:
                bucket[memory_id] = rec              # records are frozen contracts: sharing one object is safe
        return rec

    # ------------------------------------------------------------------ sources

    def register_source(self, source: SourceRef) -> SourceRef:
        """Register a citable source. Re-registering the same definition is a no-op."""
        if self.public_only and source.origin is not DataOrigin.SYNTHETIC:
            raise WriteRejected(f"source {source.source_id!r}: only synthetic sources in the public store")
        data = canonical_json(source.to_dict())
        with self._write() as conn:
            row = conn.execute("SELECT record_json FROM sources WHERE source_id = ?", (source.source_id,)).fetchone()
            if row is not None:
                if row[0] != data:
                    raise DuplicateSourceId(f"source {source.source_id!r} is registered with another definition")
                return source
            conn.execute("INSERT INTO sources (source_id, kind, origin, description, record_json) VALUES (?,?,?,?,?)",
                         (source.source_id, source.kind.value, source.origin.value, source.description, data))
        return source

    def get_source(self, source_id: str) -> Optional[SourceRef]:
        with self._read() as conn:
            return self._source(conn, source_id)

    def _source(self, conn: sqlite3.Connection, source_id: str) -> Optional[SourceRef]:
        row = conn.execute("SELECT record_json FROM sources WHERE source_id = ?", (source_id,)).fetchone()
        return None if row is None else SourceRef.from_dict(json.loads(row[0]))

    # ------------------------------------------------------------------ reads

    def get(self, memory_id: str, *, persona_id: Optional[str] = None,
            known_at: Optional[datetime] = None) -> Optional[MemoryRecord]:
        """The record, or None. ``persona_id`` refuses another persona's record; ``known_at`` gives it as it stood
        then (None if it was not yet recorded)."""
        t = _opt_utc(known_at)
        with self._read() as conn:
            self._require_owner(conn, memory_id, persona_id)
            row = conn.execute("SELECT recorded_at_utc FROM memories WHERE memory_id = ?", (memory_id,)).fetchone()
            if row is None or (t is not None and row[0] > t):
                return None
            return self._load(conn, memory_id, t)

    def all_records(self, persona_id: str) -> List[MemoryRecord]:
        """Every record of the persona, any status, ordered by recorded_at then id (read-only; for audits such as
        the Phase 5B grounding guard's catalog)."""
        with self._read() as conn:
            rows = conn.execute("SELECT memory_id FROM memories WHERE persona_id = ? ORDER BY recorded_at_utc, memory_id",
                                (persona_id,)).fetchall()
            return [self._load(conn, r[0]) for r in rows]

    def history(self, persona_id: str, subject: str, attribute: str, *,
                known_at: Optional[datetime] = None) -> List[MemoryRecord]:
        """Every record of the slot, superseded and retracted included, ordered by recorded_at then id. With
        ``known_at``: the records known by then, each as it stood then."""
        t = _opt_utc(known_at)
        recorded = "" if t is None else "AND recorded_at_utc <= ? "
        with self._read() as conn:
            rows = conn.execute(
                f"SELECT memory_id FROM memories WHERE persona_id = ? AND subject = ? AND attribute = ? {recorded}"
                "ORDER BY recorded_at_utc, memory_id",
                (persona_id, subject, attribute) + (() if t is None else (t,))).fetchall()
            return [self._load(conn, r[0], t) for r in rows]

    def facts_at(self, persona_id: str, *, as_of: Optional[datetime] = None, known_at: Optional[datetime] = None,
                 subject: Optional[str] = None, attribute: Optional[str] = None) -> List[MemoryRecord]:
        """Records valid at ``as_of`` (any validity if None) as the store believed them at ``known_at``.

        A record counts if it was recorded by then and had not yet been superseded or retracted, and is returned
        as it stood then. ``known_at`` defaults to ``as_of``, else to the clock's now. Ordered by subject,
        attribute, valid_from, recorded_at, id.
        """
        known = _utc(known_at if known_at is not None else as_of if as_of is not None else self._clock())
        args: list = [persona_id]
        sql = ["SELECT memory_id FROM memories WHERE persona_id = ?", "AND " + _believed(known, args)]
        if as_of is not None:
            t = _utc(as_of)
            sql.append("AND (valid_from_utc IS NULL OR valid_from_utc <= ?) AND (valid_to_utc IS NULL OR ? < valid_to_utc)")
            args += [t, t]
        for col, val in (("subject", subject), ("attribute", attribute)):
            if val is not None:
                sql.append(f"AND {col} = ?")
                args.append(val)
        sql.append("ORDER BY subject, attribute, valid_from_utc IS NOT NULL, valid_from_utc, recorded_at_utc, memory_id")
        with self._read() as conn:
            return [self._load(conn, r[0], known) for r in conn.execute(" ".join(sql), args).fetchall()]

    def facts_during(self, persona_id: str, *, start: datetime, end: datetime, subject: Optional[str] = None,
                     attribute: Optional[str] = None, known_at: Optional[datetime] = None) -> List[MemoryRecord]:
        """Records believed at ``known_at`` (default ``end``: nothing learned after the interval), as they stood
        then, valid at some instant of the half-open interval ``[start, end)`` (Phase 5B). A record's validity
        ``[valid_from, valid_to)`` overlaps it when ``valid_from < end`` and ``start < valid_to`` (a missing bound
        is open), compared on the stored fixed-width UTC text, so local calendar times are compared as instants.
        Ordered like ``facts_at``."""
        lo, hi = _utc(start), _utc(end)
        if not lo < hi:
            raise ValueError(f"start {start.isoformat()} must be before end {end.isoformat()}")
        known = _utc(end if known_at is None else known_at)
        args: list = [persona_id]
        sql = ["SELECT memory_id FROM memories WHERE persona_id = ?", "AND " + _believed(known, args),
               "AND (valid_from_utc IS NULL OR valid_from_utc < ?) AND (valid_to_utc IS NULL OR ? < valid_to_utc)"]
        args += [hi, lo]
        for col, val in (("subject", subject), ("attribute", attribute)):
            if val is not None:
                sql.append(f"AND {col} = ?")
                args.append(val)
        sql.append("ORDER BY subject, attribute, valid_from_utc IS NOT NULL, valid_from_utc, recorded_at_utc, memory_id")
        with self._read() as conn:
            return [self._load(conn, r[0], known) for r in conn.execute(" ".join(sql), args).fetchall()]

    def search(self, persona_id: str, query: str, *, as_of: Optional[datetime] = None,
               known_at: Optional[datetime] = None, limit: int = 5,
               include_ended: bool = False) -> List[RetrievedMemory]:
        """Records of the persona valid at ``as_of`` and believed at ``known_at`` (see ``resolve_times``), as they
        stood then, that share keywords with ``query``. The keyword index holds every record of any status, so
        a former belief is found through the same index. ``include_ended`` also keeps records whose validity ended
        by ``as_of``: an event that happened is still something to remember, though it no longer holds.

        Ranked by FTS5 bm25 (bigram hits weigh more than single characters), then newest recorded_at, then
        id. ``score`` is ``-bm25`` (higher is better), or the overlap count in the scan fallback.
        """
        if limit <= 0:
            return []
        units = query_units(query)
        if not units:
            return []
        as_of, known_at = resolve_times(as_of, known_at, self._clock)
        t, known = _utc(as_of), _utc(known_at)
        belief_args: list = []
        valid = (f"m.persona_id = ? AND {_believed(known, belief_args, alias='m.')} "
                 "AND (m.valid_from_utc IS NULL OR m.valid_from_utc <= ?)")
        where: tuple = (persona_id, *belief_args, t)
        if not include_ended:
            valid += " AND (m.valid_to_utc IS NULL OR ? < m.valid_to_utc)"
            where += (t,)
        with self._read() as conn:
            if self.fts_enabled:
                rows = conn.execute(
                    f"SELECT m.memory_id, bm25(memory_fts, 0.0, {BIGRAM_WEIGHT!r}, {UNIGRAM_WEIGHT!r}) AS rank "
                    f"FROM memory_fts JOIN memories AS m ON m.memory_id = memory_fts.memory_id "
                    f"WHERE memory_fts MATCH ? AND {valid} "
                    f"ORDER BY rank, m.recorded_at_utc DESC, m.memory_id LIMIT ?",
                    (match_expression(units), *where, limit)).fetchall()
                hits = [(r[0], -r[1]) for r in rows]
            else:
                rows = conn.execute(f"SELECT m.memory_id, m.recorded_at_utc FROM memories AS m WHERE {valid}",
                                    where).fetchall()
                scored = []
                for r in rows:
                    rec = self._load(conn, r[0])
                    score = overlap_score(units, _terms(rec))
                    if score > 0:
                        scored.append((r[0], score, r[1]))
                scored.sort(key=lambda x: x[0])
                scored.sort(key=lambda x: x[2], reverse=True)
                scored.sort(key=lambda x: x[1], reverse=True)
                hits = [(mid, score) for mid, score, _ in scored[:limit]]
            return [RetrievedMemory(record=self._load(conn, mid, known), score=float(score), retriever=self.retriever)
                    for mid, score in hits]

    def events(self, memory_id: str, *, persona_id: Optional[str] = None) -> List[MemoryEvent]:
        with self._read() as conn:
            self._require_owner(conn, memory_id, persona_id)
            rows = conn.execute("SELECT * FROM memory_events WHERE memory_id = ? ORDER BY event_id", (memory_id,))
            return [MemoryEvent(r["event_id"], r["memory_id"], r["action"], r["related_id"], r["reason"],
                                r["source_id"], r["locator"], r["reviewer"], _from_utc(r["recorded_at"]))
                    for r in rows.fetchall()]

    def conflicts(self, persona_id: str, *, include_resolved: bool = False) -> List[MemoryConflict]:
        """Recorded conflict pairs of the persona, each record with its provenance. By default only open
        conflicts (both records still active); ``include_resolved`` adds pairs where one side was superseded
        or retracted."""
        with self._read() as conn:
            rows = conn.execute(
                "SELECT c.* FROM memory_conflicts AS c JOIN memories AS a ON a.memory_id = c.memory_a "
                "WHERE a.persona_id = ? ORDER BY c.memory_a, c.memory_b", (persona_id,)).fetchall()
            out = [MemoryConflict(self._load(conn, r["memory_a"]), self._load(conn, r["memory_b"]), r["origin"],
                                  _from_utc(r["recorded_at"])) for r in rows]
        return [c for c in out if include_resolved or c.is_open]

    @staticmethod
    def _require_owner(conn: sqlite3.Connection, memory_id: str, persona_id: Optional[str]) -> None:
        if persona_id is None:
            return
        row = conn.execute("SELECT persona_id FROM memories WHERE memory_id = ?", (memory_id,)).fetchone()
        if row is not None and row[0] != persona_id:
            raise PersonaScopeError(f"{memory_id} does not belong to persona {persona_id}")

    def _load(self, conn: sqlite3.Connection, memory_id: str, known_at: Optional[str] = None) -> Optional[MemoryRecord]:
        """The record now, or (``known_at``, fixed-width UTC) as it stood then: a later correction or retraction
        has not happened yet, and conflicts with records recorded later are not known yet. Served from the read cache when the
        database has not changed since (``read_generation``); computed exactly as before otherwise."""
        return self._cached_load(conn, memory_id, known_at)

    def _load_uncached(self, conn: sqlite3.Connection, memory_id: str, known_at: Optional[str] = None) -> Optional[MemoryRecord]:
        row = conn.execute("SELECT * FROM memories WHERE memory_id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        prov = conn.execute("SELECT source_id, locator, quote, observed_at FROM provenance WHERE memory_id = ? "
                            "ORDER BY position", (memory_id,)).fetchall()
        others = conn.execute("SELECT memory_b FROM memory_conflicts WHERE memory_a = ? UNION "
                              "SELECT memory_a FROM memory_conflicts WHERE memory_b = ? ORDER BY 1",
                              (memory_id, memory_id)).fetchall()
        conflicts = [o[0] for o in others]
        status, superseded_by = row["status"], row["superseded_by"]
        if known_at is not None:
            if conflicts:
                marks = ",".join("?" * len(conflicts))
                known = {r[0] for r in conn.execute(f"SELECT memory_id FROM memories WHERE memory_id IN ({marks}) "
                                                    "AND recorded_at_utc <= ?", (*conflicts, known_at))}
                conflicts = [c for c in conflicts if c in known]
            if row["status_changed_at"] is not None and row["status_changed_at"] > known_at:
                status, superseded_by = MemoryStatus.ACTIVE.value, None
        return MemoryRecord.from_dict({
            "memory_id": row["memory_id"], "persona_id": row["persona_id"], "kind": row["kind"],
            "subject": row["subject"], "attribute": row["attribute"], "value": row["value"],
            "statement": row["statement"], "provenance": [dict(p) for p in prov], "origin": row["origin"],
            "recorded_at": row["recorded_at"], "valid_from": row["valid_from"], "valid_to": row["valid_to"],
            "status": status, "supersedes": row["supersedes"], "superseded_by": superseded_by,
            "correction_reason": row["correction_reason"], "conflicts_with": conflicts,
            "confidence": row["confidence"], "cardinality": row["cardinality"],
            "schema_version": row["schema_version"],
        })

    # ------------------------------------------------------------------ write gate

    def _check_record(self, conn: sqlite3.Connection, record: MemoryRecord, reviewer: Optional[str]) -> None:
        if self.public_only:
            try:
                assert_public_safe(record)
            except PublicBoundaryError as exc:
                raise WriteRejected(f"{record.memory_id}: {exc}") from None
        if record.origin is DataOrigin.REAL and not _reviewed(reviewer):
            raise WriteRejected(f"{record.memory_id}: a real record needs the owner's review (reviewer=...)")
        for i, p in enumerate(record.provenance):
            src = self._source(conn, p.source_id)
            if src is None:
                raise WriteRejected(f"{record.memory_id}: provenance[{i}] cites unregistered source {p.source_id!r}")
            if src.origin is not record.origin:
                raise WriteRejected(f"{record.memory_id}: a {record.origin.value} record cannot cite "
                                    f"{src.origin.value} source {src.source_id!r}")
            if record.origin is DataOrigin.SIMULATED and src.kind is not SourceKind.SIMULATION:
                raise WriteRejected(f"{record.memory_id}: a simulated record must come from a simulation source")
            if record.origin is DataOrigin.REAL and src.kind is SourceKind.SIMULATION:
                raise WriteRejected(f"{record.memory_id}: a simulation source cannot back a real record")
            if p.locator is not None and p.locator.startswith(MESSAGE_LOCATOR_PREFIX):
                row = conn.execute("SELECT target_persona_id, source_id FROM messages WHERE message_id = ?",
                                   (p.locator[len(MESSAGE_LOCATOR_PREFIX):],)).fetchone()
                if row is None or row[0] != record.persona_id or row[1] != p.source_id:
                    raise WriteRejected(f"{record.memory_id}: provenance[{i}] cites {p.locator!r}, which is not a "
                                        f"message of persona {record.persona_id} in source {p.source_id!r}")

    def _check_declared(self, conn: sqlite3.Connection, record: MemoryRecord) -> List[str]:
        for other in record.conflicts_with:
            row = conn.execute("SELECT persona_id FROM memories WHERE memory_id = ?", (other,)).fetchone()
            if row is None:
                raise WriteRejected(f"{record.memory_id}: declared conflict with unknown memory {other!r}")
            if row[0] != record.persona_id:
                raise WriteRejected(f"{record.memory_id}: declared conflict with another persona's memory {other!r}")
        return list(record.conflicts_with)

    def _detect(self, conn: sqlite3.Connection, record: MemoryRecord) -> List[str]:
        rows = conn.execute("SELECT memory_id FROM memories WHERE persona_id = ? AND subject = ? AND attribute = ? "
                            "AND status = 'active' AND memory_id <> ? ORDER BY memory_id",
                            (*record.slot, record.memory_id)).fetchall()
        return [r[0] for r in rows if record.contradicts(self._load(conn, r[0]))]

    # ------------------------------------------------------------------ write steps (failure-injection points)

    def _insert_record(self, conn: sqlite3.Connection, record: MemoryRecord,
                       status_changed_at: Optional[str]) -> None:
        conn.execute(
            "INSERT INTO memories (memory_id, persona_id, kind, subject, attribute, value, statement, origin, "
            "cardinality, confidence, recorded_at, recorded_at_utc, valid_from, valid_from_utc, valid_to, "
            "valid_to_utc, status, status_changed_at, supersedes, superseded_by, correction_reason, schema_version, "
            "content_sha256, duplicate_key) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (record.memory_id, record.persona_id, record.kind.value, record.subject, record.attribute, record.value,
             record.statement, record.origin.value, record.cardinality.value, record.confidence,
             record.recorded_at.isoformat(), _utc(record.recorded_at),
             None if record.valid_from is None else record.valid_from.isoformat(), _opt_utc(record.valid_from),
             None if record.valid_to is None else record.valid_to.isoformat(), _opt_utc(record.valid_to),
             record.status.value, status_changed_at, record.supersedes, record.superseded_by,
             record.correction_reason, record.schema_version, content_sha256(record), duplicate_key(record)))

    def _insert_provenance(self, conn: sqlite3.Connection, record: MemoryRecord) -> None:
        conn.executemany(
            "INSERT INTO provenance (memory_id, position, source_id, locator, quote, observed_at) VALUES (?,?,?,?,?,?)",
            [(record.memory_id, i, p.source_id, p.locator, p.quote,
              None if p.observed_at is None else p.observed_at.isoformat()) for i, p in enumerate(record.provenance)])

    def _write_conflicts(self, conn: sqlite3.Connection, memory_id: str, declared: Iterable[str],
                         detected: Iterable[str]) -> None:
        now = self._now()
        for origin, others in (("declared", declared), ("detected", detected)):
            for other in others:
                a, b = sorted((memory_id, other))
                conn.execute("INSERT OR IGNORE INTO memory_conflicts (memory_a, memory_b, origin, recorded_at) "
                             "VALUES (?,?,?,?)", (a, b, origin, now))

    def _index_fts(self, conn: sqlite3.Connection, record: MemoryRecord) -> None:
        if self._fts_table:
            terms = _terms(record)
            conn.execute("INSERT INTO memory_fts (memory_id, bigrams, unigrams) VALUES (?,?,?)",
                         (record.memory_id, terms.bigram_text, terms.unigram_text))

    def _log_event(self, conn: sqlite3.Connection, memory_id: str, action: str, *, related_id: Optional[str] = None,
                   reason: Optional[str] = None, evidence: Optional[Provenance] = None,
                   reviewer: Optional[str] = None) -> None:
        conn.execute("INSERT INTO memory_events (memory_id, action, related_id, reason, source_id, locator, reviewer, "
                     "recorded_at) VALUES (?,?,?,?,?,?,?,?)",
                     (memory_id, action, related_id, reason, evidence.source_id if evidence else None,
                      evidence.locator if evidence else None, reviewer, self._now()))

    def _rebuild_fts(self, conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM memory_fts")
        for r in conn.execute("SELECT memory_id FROM memories ORDER BY rowid").fetchall():
            self._index_fts(conn, self._load(conn, r[0]))

    def _index_message(self, conn: sqlite3.Connection, message_id: str, content: str) -> None:
        if self._message_fts_table:
            terms = index_terms(content)
            conn.execute("INSERT INTO messages_fts (message_id, bigrams, unigrams) VALUES (?,?,?)",
                         (message_id, terms.bigram_text, terms.unigram_text))

    def _rebuild_message_fts(self, conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM messages_fts")
        for r in conn.execute("SELECT message_id, content FROM messages ORDER BY rowid").fetchall():
            self._index_message(conn, r[0], r[1])

    def rebuild_fts(self) -> None:
        """Recreate the derived keyword indexes from their tables (e.g. after changing segmentation)."""
        if not self._fts_table:
            raise RuntimeError("this store has no FTS5 index")
        with self._write() as conn:
            self._rebuild_fts(conn)
            if self._message_fts_table:
                self._rebuild_message_fts(conn)

    # ------------------------------------------------------------------ writes

    def add(self, record: MemoryRecord, *, reviewer: Optional[str] = None) -> MemoryRecord:
        """Insert an active record. Returns it with ``conflicts_with`` holding declared and detected conflicts."""
        if record.status is not MemoryStatus.ACTIVE or record.superseded_by is not None:
            raise WriteRejected("add() takes an active record; import_records() restores history")
        if record.supersedes is not None:
            raise WriteRejected("add() cannot supersede; use correct()")
        sha = content_sha256(record)
        with self._write() as conn:
            row = conn.execute("SELECT content_sha256 FROM memories WHERE memory_id = ?", (record.memory_id,)).fetchone()
            if row is not None:
                stored = self._load(conn, record.memory_id)
                if row[0] != sha or not set(record.conflicts_with) <= set(stored.conflicts_with):
                    raise DuplicateMemoryId(f"{record.memory_id} is already stored with different content")
                return stored                                   # a retry of a write that already happened
            self._check_record(conn, record, reviewer)
            dup = conn.execute("SELECT memory_id FROM memories WHERE persona_id = ? AND duplicate_key = ? AND "
                               "status = 'active' ORDER BY recorded_at_utc, memory_id LIMIT 1",
                               (record.persona_id, duplicate_key(record))).fetchone()
            if dup is not None:
                return self._load(conn, dup[0])                 # same content re-sent under a fresh id
            declared = self._check_declared(conn, record)
            detected = self._detect(conn, record)
            self._insert_record(conn, record, None)
            self._insert_provenance(conn, record)
            self._write_conflicts(conn, record.memory_id, declared, detected)
            self._index_fts(conn, record)
            self._log_event(conn, record.memory_id, "add", reviewer=reviewer)
            return self._load(conn, record.memory_id)

    def correct(self, memory_id: str, replacement: MemoryRecord, reason: str, *, reviewer: Optional[str] = None,
                persona_id: Optional[str] = None) -> MemoryRecord:
        """Atomically mark ``memory_id`` superseded and insert ``replacement`` (store sets ``supersedes`` and
        ``correction_reason``). The replacement must describe the same persona and slot. ``persona_id`` refuses
        another persona's record before anything else, the retry of an earlier correction included."""
        _require_reason(reason)
        if persona_id is not None and replacement.persona_id != persona_id:
            raise PersonaScopeError(f"replacement {replacement.memory_id} does not belong to persona {persona_id}")
        if replacement.status is not MemoryStatus.ACTIVE or replacement.superseded_by is not None:
            raise WriteRejected("a replacement must be active")
        if replacement.supersedes not in (None, memory_id):
            raise WriteRejected(f"replacement already supersedes {replacement.supersedes!r}")
        if replacement.correction_reason not in (None, reason):
            raise WriteRejected("replacement carries a different correction_reason")
        try:
            new = dataclasses.replace(replacement, supersedes=memory_id, correction_reason=reason)
        except ValidationError as exc:
            raise WriteRejected(str(exc)) from None
        sha, dkey = content_sha256(new), duplicate_key(new)
        with self._write() as conn:
            self._require_owner(conn, memory_id, persona_id)
            old_row = conn.execute("SELECT status, superseded_by FROM memories WHERE memory_id = ?",
                                   (memory_id,)).fetchone()
            if old_row is None:
                raise MemoryNotFound(memory_id)
            if old_row["status"] == MemoryStatus.SUPERSEDED.value:
                succ = old_row["superseded_by"]
                srow = conn.execute("SELECT content_sha256, duplicate_key FROM memories WHERE memory_id = ?",
                                    (succ,)).fetchone()
                if (succ == new.memory_id and srow[0] == sha) or (succ != new.memory_id and srow[1] == dkey):
                    return self._load(conn, succ)               # a retry of a correction that already happened
                raise StaleWrite(f"{memory_id} was already corrected by {succ}")
            if old_row["status"] == MemoryStatus.RETRACTED.value:
                raise StaleWrite(f"{memory_id} is retracted")
            if conn.execute("SELECT 1 FROM memories WHERE memory_id = ?", (new.memory_id,)).fetchone():
                raise DuplicateMemoryId(f"{new.memory_id} is already stored")
            old = self._load(conn, memory_id)
            if new.slot != old.slot:
                raise WriteRejected("a correction must keep the persona, subject and attribute")
            if old.origin is DataOrigin.REAL and new.origin is DataOrigin.SIMULATED:
                raise WriteRejected("a simulated record never corrects a real one")
            if old.origin is DataOrigin.REAL and not _reviewed(reviewer):
                raise WriteRejected("correcting a real record needs the owner's review (reviewer=...)")
            self._check_record(conn, new, reviewer)
            if _utc(new.recorded_at) < _utc(old.recorded_at):
                raise WriteRejected("a replacement cannot be recorded before the record it corrects")
            if memory_id in new.conflicts_with:
                raise WriteRejected("a replacement cannot declare a conflict with the record it replaces")
            declared = self._check_declared(conn, new)
            cur = conn.execute("UPDATE memories SET status = 'superseded', superseded_by = ?, status_changed_at = ? "
                               "WHERE memory_id = ? AND status = 'active'",
                               (new.memory_id, _utc(new.recorded_at), memory_id))
            if cur.rowcount != 1:
                raise StaleWrite(f"{memory_id} is no longer active")
            detected = self._detect(conn, new)
            self._insert_record(conn, new, None)
            self._insert_provenance(conn, new)
            self._write_conflicts(conn, new.memory_id, declared, detected)
            self._index_fts(conn, new)
            self._log_event(conn, new.memory_id, "correct", related_id=memory_id, reason=reason, reviewer=reviewer)
            self._log_event(conn, memory_id, "superseded", related_id=new.memory_id, reason=reason, reviewer=reviewer)
            return self._load(conn, new.memory_id)

    def retract(self, memory_id: str, reason: str, *, evidence: Optional[Provenance] = None,
                reviewer: Optional[str] = None, persona_id: Optional[str] = None, at: Optional[datetime] = None) -> MemoryRecord:
        """Mark an active record retracted. The reason, optional evidence and reviewer go to the audit log
        (``events()``): MemoryRecord v0.1 has no retraction fields. Retracting twice returns the record.
        ``persona_id`` refuses another persona's record. ``at`` is when the retraction became known (a late-arriving
        source); it defaults to the clock's now and may not be before the record was recorded or after now."""
        _require_reason(reason)
        with self._write() as conn:
            self._require_owner(conn, memory_id, persona_id)
            old = self._load(conn, memory_id)
            if old is None:
                raise MemoryNotFound(memory_id)
            if old.status is MemoryStatus.RETRACTED:
                return old
            if old.status is MemoryStatus.SUPERSEDED:
                raise StaleWrite(f"{memory_id} is superseded by {old.superseded_by}; retract the current version")
            if old.origin is DataOrigin.REAL and not _reviewed(reviewer):
                raise WriteRejected("retracting a real record needs the owner's review (reviewer=...)")
            if evidence is not None:
                src = self._source(conn, evidence.source_id)
                if src is None:
                    raise WriteRejected(f"retraction cites unregistered source {evidence.source_id!r}")
                if old.origin is DataOrigin.REAL and src.kind is SourceKind.SIMULATION:
                    raise WriteRejected("a simulation source cannot retract a real record")
            when = self._now() if at is None else _utc(at)
            if at is not None and (when < _utc(old.recorded_at) or when > self._now()):
                raise WriteRejected("a retraction time must lie between the record's recording and now")
            cur = conn.execute("UPDATE memories SET status = 'retracted', status_changed_at = ? "
                               "WHERE memory_id = ? AND status = 'active'", (when, memory_id))
            if cur.rowcount != 1:
                raise StaleWrite(f"{memory_id} is no longer active")
            self._log_event(conn, memory_id, "retract", reason=reason, evidence=evidence, reviewer=reviewer)
            return self._load(conn, memory_id)

    def import_records(self, records: Iterable[MemoryRecord], *, reviewer: Optional[str] = None) -> List[MemoryRecord]:
        """Restore records with their history (superseded and retracted ones, correction links, declared
        conflicts) in one transaction. Records already stored identically are skipped. Links must be
        consistent within the batch and the store; an import cannot supersede a record already stored."""
        records = list(records)
        ids = [r.memory_id for r in records]
        if len(set(ids)) != len(ids):
            raise WriteRejected("import batch repeats a memory_id")
        batch: Dict[str, MemoryRecord] = {r.memory_id: r for r in records}
        with self._write() as conn:
            fresh: List[MemoryRecord] = []
            for r in records:
                row = conn.execute("SELECT content_sha256, status, superseded_by FROM memories WHERE memory_id = ?",
                                   (r.memory_id,)).fetchone()
                if row is None:
                    fresh.append(r)
                elif row[0] != content_sha256(r):
                    raise DuplicateMemoryId(f"{r.memory_id} is already stored with different content")
                elif (row[1], row[2]) != (r.status.value, r.superseded_by):
                    raise WriteRejected(f"{r.memory_id}: imported status disagrees with the stored record")
            new_ids = {r.memory_id for r in fresh}

            def lookup(mid: str) -> Optional[MemoryRecord]:
                return batch[mid] if mid in batch else self._load(conn, mid)

            for r in fresh:
                self._check_record(conn, r, reviewer)
                if r.superseded_by is not None:
                    succ = lookup(r.superseded_by)
                    if succ is None or succ.supersedes != r.memory_id or succ.slot != r.slot:
                        raise WriteRejected(f"{r.memory_id}: superseded_by does not point at its correction")
                if r.supersedes is not None:
                    pred = lookup(r.supersedes)
                    if pred is None or pred.superseded_by != r.memory_id:
                        raise WriteRejected(f"{r.memory_id}: supersedes does not point at a record it replaced")
                    if pred.memory_id not in new_ids:
                        raise WriteRejected(f"{r.memory_id}: an import cannot supersede a stored record; use correct()")
                    if pred.origin is DataOrigin.REAL and r.origin is DataOrigin.SIMULATED:
                        raise WriteRejected("a simulated record never corrects a real one")

            depth: Dict[str, int] = {}

            def chain_depth(r: MemoryRecord) -> int:
                if r.memory_id not in depth:
                    depth[r.memory_id] = 0 if r.supersedes not in new_ids else chain_depth(batch[r.supersedes]) + 1
                return depth[r.memory_id]

            position = {mid: i for i, mid in enumerate(ids)}
            for r in sorted(fresh, key=lambda r: (chain_depth(r), position[r.memory_id])):
                if r.status is MemoryStatus.ACTIVE:
                    changed = None
                elif r.status is MemoryStatus.SUPERSEDED:
                    changed = _utc(lookup(r.superseded_by).recorded_at)
                else:
                    changed = self._now()                       # retraction time is not part of the record
                self._insert_record(conn, r, changed)
                self._insert_provenance(conn, r)
                self._index_fts(conn, r)
                self._log_event(conn, r.memory_id, "import", related_id=r.supersedes, reviewer=reviewer)
            for r in fresh:
                declared = self._check_declared(conn, r)
                detected = self._detect(conn, r) if r.is_active else []
                self._write_conflicts(conn, r.memory_id, declared, detected)
            return [self._load(conn, mid) for mid in ids]

    # ------------------------------------------------------------------ raw messages

    def ingest_messages(self, messages: Iterable[CanonicalMessage]) -> IngestResult:
        """Store raw messages in one transaction. A message already stored with the same content is skipped; the
        same id with different content raises ``MessageConflict`` and nothing of the batch is written. Every
        source must be registered. No memory is created: facts come from a later, explicit extraction step."""
        messages = list(messages)
        added = skipped = 0
        with self._write() as conn:
            now = self._clock()
            for m in messages:
                sha = _message_sha256(m)
                row = conn.execute("SELECT content_sha256, recorded_at_utc FROM messages WHERE message_id = ?",
                                   (m.message_id,)).fetchone()
                if row is not None:
                    if row[0] != sha or (m.recorded_at is not None and _utc(m.recorded_at) != row[1]):
                        raise MessageConflict(f"{m.message_id} is already stored with different content")
                    skipped += 1
                    continue
                src = self._source(conn, m.source_id)
                if src is None:
                    raise WriteRejected(f"{m.message_id}: unregistered source {m.source_id!r}")
                if self.public_only and src.origin is not DataOrigin.SYNTHETIC:
                    raise WriteRejected(f"{m.message_id}: only synthetic sources in the public store")
                recorded = now if m.recorded_at is None else m.recorded_at
                if _utc(recorded) < _utc(m.timestamp):
                    raise WriteRejected(f"{m.message_id}: said at {m.timestamp.isoformat()}, after the time it would "
                                        f"be recorded ({recorded.isoformat()}); give recorded_at for a replay")
                conn.execute(
                    "INSERT INTO messages (message_id, conversation_id, target_persona_id, speaker_id, timestamp, "
                    "timestamp_utc, content, source_type, source_id, locator, recorded_at, recorded_at_utc, "
                    "content_sha256) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (m.message_id, m.conversation_id, m.target_persona_id, m.speaker_id, m.timestamp.isoformat(),
                     _utc(m.timestamp), m.content, m.source_type, m.source_id, m.locator, recorded.isoformat(),
                     _utc(recorded), sha))
                self._index_message(conn, m.message_id, m.content)
                added += 1
        return IngestResult(added=added, skipped=skipped)

    @staticmethod
    def _message(row: sqlite3.Row) -> CanonicalMessage:
        return CanonicalMessage(
            conversation_id=row["conversation_id"], message_id=row["message_id"],
            target_persona_id=row["target_persona_id"], speaker_id=row["speaker_id"], timestamp=row["timestamp"],
            content=row["content"], source_type=row["source_type"], source_id=row["source_id"],
            locator=row["locator"], recorded_at=row["recorded_at"])

    def get_message(self, persona_id: str, message_id: str) -> Optional[CanonicalMessage]:
        with self._read() as conn:
            row = conn.execute("SELECT * FROM messages WHERE message_id = ?", (message_id,)).fetchone()
        if row is None:
            return None
        if row["target_persona_id"] != persona_id:
            raise PersonaScopeError(f"{message_id} does not belong to persona {persona_id}")
        return self._message(row)

    def conversation_messages(self, persona_id: str, conversation_id: str, *, as_of: Optional[datetime] = None,
                              known_at: Optional[datetime] = None) -> Tuple[List[CanonicalMessage], int]:
        """Every message of one conversation of the persona that was said by ``as_of`` and known by ``known_at``, in the
        order said (then the order stored), and how many of its messages that time hides. Read only; no limit on count.

        A conversation of another persona raises ``PersonaScopeError``; an unknown one gives no messages."""
        as_of, known_at = resolve_times(as_of, known_at, self._clock)
        with self._read() as conn:
            owners = {r[0] for r in conn.execute("SELECT DISTINCT target_persona_id FROM messages WHERE conversation_id = ?", (conversation_id,))}
            if owners - {persona_id}:
                raise PersonaScopeError(f"{conversation_id} does not belong to persona {persona_id}")
            rows = conn.execute("SELECT * FROM messages WHERE conversation_id = ? AND target_persona_id = ? ORDER BY timestamp_utc, rowid",
                                (conversation_id, persona_id)).fetchall()
        visible = [r for r in rows if r["timestamp_utc"] <= _utc(as_of) and r["recorded_at_utc"] <= _utc(known_at)]
        return [self._message(r) for r in visible], len(rows) - len(visible)

    def search_messages(self, persona_id: str, query: str, *, as_of: Optional[datetime] = None,
                        known_at: Optional[datetime] = None, limit: int = 10,
                        speaker_id: Optional[str] = None, conversation_id: Optional[str] = None) -> List[Tuple[CanonicalMessage, float]]:
        """Messages in the persona's history, said by ``as_of`` and known by ``known_at`` (see ``resolve_times``),
        that share keywords with ``query``, best first: bm25, then newest said, then id."""
        if limit <= 0:
            return []
        units = query_units(query)
        if not units:
            return []
        as_of, known_at = resolve_times(as_of, known_at, self._clock)
        scope = "m.target_persona_id = ? AND m.timestamp_utc <= ? AND m.recorded_at_utc <= ?"
        where: tuple = (persona_id, _utc(as_of), _utc(known_at))
        if speaker_id is not None:
            scope += " AND m.speaker_id = ?"
            where += (speaker_id,)
        if conversation_id is not None:                 # Simple Context v1: one session's history only
            scope += " AND m.conversation_id = ?"
            where += (conversation_id,)
        with self._read() as conn:
            if self.message_fts_enabled:
                rows = conn.execute(
                    f"SELECT m.*, bm25(messages_fts, 0.0, {BIGRAM_WEIGHT!r}, {UNIGRAM_WEIGHT!r}) AS rank "
                    f"FROM messages_fts JOIN messages AS m ON m.message_id = messages_fts.message_id "
                    f"WHERE messages_fts MATCH ? AND {scope} "
                    f"ORDER BY rank, m.timestamp_utc DESC, m.message_id LIMIT ?",
                    (match_expression(units), *where, limit)).fetchall()
                return [(self._message(r), -r["rank"]) for r in rows]
            rows = conn.execute(f"SELECT m.* FROM messages AS m WHERE {scope}", where).fetchall()
        scored = [(r, overlap_score(units, index_terms(r["content"]))) for r in rows]
        scored = sorted((x for x in scored if x[1] > 0), key=lambda x: x[0]["message_id"])
        scored.sort(key=lambda x: x[0]["timestamp_utc"], reverse=True)
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(self._message(r), score) for r, score in scored[:limit]]

    def messages_between(self, persona_id: str, start: datetime, end: datetime, *, as_of: Optional[datetime] = None,
                         known_at: Optional[datetime] = None, limit: int = 500) -> List[CanonicalMessage]:
        """Messages of the persona said in ``[start, end)``, said by ``as_of`` and known by ``known_at`` (the same visibility rule as
        ``search_messages``), oldest first. EF1's time-window channel: a written date or range selects evidence by time, not by words."""
        if limit <= 0:
            return []
        as_of, known_at = resolve_times(as_of, known_at, self._clock)
        with self._read() as conn:
            rows = conn.execute(
                "SELECT m.* FROM messages AS m WHERE m.target_persona_id = ? AND m.timestamp_utc >= ? AND m.timestamp_utc < ? "
                "AND m.timestamp_utc <= ? AND m.recorded_at_utc <= ? ORDER BY m.timestamp_utc, m.message_id LIMIT ?",
                (persona_id, _utc(start), _utc(end), _utc(as_of), _utc(known_at), limit)).fetchall()
            return [self._message(r) for r in rows]

    def messages_by_locator(self, persona_id: str, locators: Sequence[str], *, as_of: Optional[datetime] = None,
                            known_at: Optional[datetime] = None) -> List[CanonicalMessage]:
        """Every visible message of the persona whose source locator is one of ``locators`` (a dialogue or record and all its turns),
        oldest first. Visibility as in ``search_messages``: a later turn of the same dialogue is never returned before it is known."""
        locs = list(dict.fromkeys(l for l in locators if l))
        if not locs:
            return []
        as_of, known_at = resolve_times(as_of, known_at, self._clock)
        out: List[CanonicalMessage] = []
        with self._read() as conn:
            for i in range(0, len(locs), 400):
                part = locs[i:i + 400]
                marks = ",".join("?" * len(part))
                rows = conn.execute(
                    f"SELECT m.* FROM messages AS m WHERE m.target_persona_id = ? AND m.locator IN ({marks}) AND m.timestamp_utc <= ? "
                    f"AND m.recorded_at_utc <= ? ORDER BY m.timestamp_utc, m.message_id", (persona_id, *part, _utc(as_of), _utc(known_at))).fetchall()
                out += [self._message(r) for r in rows]
        return out

    def evidence(self, persona_id: str, source_id: str) -> EvidenceBundle:
        """A source as this persona may see it: the citations of their own memories, each with the message it
        cites, and how many of their messages come from it. A source they neither cite nor have messages from
        raises ``PersonaScopeError``, even when another persona uses it."""
        with self._read() as conn:
            source = self._source(conn, source_id)
            if source is None:
                raise MemoryNotFound(source_id)
            rows = conn.execute(
                "SELECT p.memory_id, p.locator, p.quote, p.observed_at FROM provenance AS p "
                "JOIN memories AS m ON m.memory_id = p.memory_id WHERE p.source_id = ? AND m.persona_id = ? "
                "ORDER BY m.recorded_at_utc, p.memory_id, p.position", (source_id, persona_id)).fetchall()
            n_messages = conn.execute("SELECT count(*) FROM messages WHERE source_id = ? AND target_persona_id = ?",
                                      (source_id, persona_id)).fetchone()[0]
            if not rows and not n_messages:
                raise PersonaScopeError(f"source {source_id} has nothing for persona {persona_id}")
            citations = []
            for r in rows:
                cited = None
                if r["locator"] is not None and r["locator"].startswith(MESSAGE_LOCATOR_PREFIX):
                    row = conn.execute("SELECT * FROM messages WHERE message_id = ? AND target_persona_id = ?",
                                       (r["locator"][len(MESSAGE_LOCATOR_PREFIX):], persona_id)).fetchone()
                    cited = None if row is None else self._message(row)
                citations.append(Citation(memory_id=r["memory_id"], locator=r["locator"], quote=r["quote"],
                                          observed_at=r["observed_at"], message=cited))
        return EvidenceBundle(persona_id=persona_id, source=source, citations=tuple(citations),
                              message_count=n_messages)

    # ------------------------------------------------------------------ snapshot and stats

    _COUNTED = ("sources", "memories", "provenance", "memory_conflicts", "memory_events", "messages")

    def _counts(self, conn: sqlite3.Connection) -> Dict[str, int]:
        return {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in self._COUNTED}

    def stats(self) -> dict:
        with self._read() as conn:
            counts = self._counts(conn)
        return {"schema_version": self.schema_version, "read_only": self.read_only, "fts_enabled": self.fts_enabled,
                "message_fts_enabled": self.message_fts_enabled, "counts": counts}

    def snapshot(self, dest_path: Union[str, "os.PathLike[str]"]) -> SnapshotInfo:
        """Write a transaction-consistent copy of the database to a new file, FTS indexes included, and return its
        sha256. An existing file is never overwritten. The copy leaves WAL mode, so it is one self-contained
        file that opens read-only from anywhere, a read-only folder included."""
        dest = Path(dest_path).resolve()
        if dest.exists():
            raise FileExistsError(f"{dest} already exists")
        tmp = dest.with_name(f"{dest.name}.{os.getpid()}.tmp")
        target = sqlite3.connect(tmp)
        try:
            with self._lock:
                self._conn.backup(target)
            mode = target.execute("PRAGMA journal_mode = DELETE").fetchone()[0]
            if mode != "delete":
                raise RuntimeError(f"the snapshot stayed in journal mode {mode!r}")
            result = [r[0] for r in target.execute("PRAGMA integrity_check").fetchall()]
            if result != ["ok"]:
                raise RuntimeError(f"snapshot failed its integrity check: {result}")
            version = target.execute("PRAGMA user_version").fetchone()[0]
            counts = self._counts(target)
            target.close()
            digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
            size = tmp.stat().st_size
            os.rename(tmp, dest)
        except BaseException:
            target.close()
            tmp.unlink(missing_ok=True)
            raise
        return SnapshotInfo(path=str(dest), sha256=digest, size_bytes=size, schema_version=version, counts=counts)

    # ------------------------------------------------------------------ integrity

    def check_integrity(self) -> List[str]:
        """Problems found by SQLite's own checks and by the store's invariants; empty when consistent."""
        problems: List[str] = []
        with self._read() as conn:
            if self._fts_table:
                # Read the FTS5 table first: FTS5 caches its index structure per connection, and
                # integrity_check run with a cache made stale by another connection's write reports a
                # malformed index although the data is intact (seen on SQLite 3.45 and 3.53).
                n_fts = conn.execute("SELECT count(*) FROM memory_fts").fetchone()[0]
            if self._message_fts_table:
                n_message_fts = conn.execute("SELECT count(*) FROM messages_fts").fetchone()[0]
            res = conn.execute("PRAGMA integrity_check").fetchall()
            if [r[0] for r in res] != ["ok"]:
                problems += [f"integrity_check: {r[0]}" for r in res]
            problems += [f"foreign key: {tuple(r)}" for r in conn.execute("PRAGMA foreign_key_check").fetchall()]
            checks = {
                "memory without provenance": "SELECT memory_id FROM memories m WHERE NOT EXISTS "
                                             "(SELECT 1 FROM provenance p WHERE p.memory_id = m.memory_id)",
                "superseded record whose successor does not point back":
                    "SELECT o.memory_id FROM memories o LEFT JOIN memories n ON n.memory_id = o.superseded_by "
                    "WHERE o.status = 'superseded' AND (n.memory_id IS NULL OR n.supersedes IS NOT o.memory_id)",
                "correction whose predecessor is not superseded by it":
                    "SELECT n.memory_id FROM memories n LEFT JOIN memories o ON o.memory_id = n.supersedes "
                    "WHERE n.supersedes IS NOT NULL AND (o.memory_id IS NULL OR o.superseded_by IS NOT n.memory_id)",
                "conflict across personas":
                    "SELECT c.memory_a FROM memory_conflicts c JOIN memories a ON a.memory_id = c.memory_a "
                    "JOIN memories b ON b.memory_id = c.memory_b WHERE a.persona_id <> b.persona_id",
                "event without memory": "SELECT e.memory_id FROM memory_events e LEFT JOIN memories m "
                                        "ON m.memory_id = e.memory_id WHERE m.memory_id IS NULL",
                "citation of another persona's message":
                    "SELECT p.memory_id FROM provenance p JOIN memories m ON m.memory_id = p.memory_id "
                    f"JOIN messages g ON g.message_id = substr(p.locator, {len(MESSAGE_LOCATOR_PREFIX) + 1}) "
                    f"WHERE p.locator LIKE '{MESSAGE_LOCATOR_PREFIX}%' AND g.target_persona_id <> m.persona_id",
            }
            for label, sql in checks.items():
                problems += [f"{label}: {r[0]}" for r in conn.execute(sql).fetchall()]
            if self._fts_table:
                n_mem = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
                if n_fts != n_mem:
                    problems.append(f"keyword index has {n_fts} rows for {n_mem} memories")
            if self._message_fts_table:
                n_msg = conn.execute("SELECT count(*) FROM messages").fetchone()[0]
                if n_message_fts != n_msg:
                    problems.append(f"message keyword index has {n_message_fts} rows for {n_msg} messages")
        return problems
