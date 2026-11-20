"""SQLite schema for the temporal memory store, and its migration.

Version is kept in ``PRAGMA user_version``. ``migrate()`` brings a database from any older version to
``SCHEMA_VERSION`` step by step inside one transaction, is safe to run on every open, and refuses a database
written by a newer version. Version 1 is the first schema; later versions append a step to ``MIGRATIONS``.
Version 2 adds the raw message layer (``messages``, ``messages_fts``) and leaves every version 1 table as it is.

Tables:
  sources          registered evidence sources (the write gate checks every provenance against them)
  memories         one row per MemoryRecord; append-only (no DELETE, content columns immutable)
  provenance       ordered provenance entries of each memory
  memory_conflicts symmetric conflict pairs, detected or declared
  memory_events    audit log: add, correct, superseded, retract (with reason, evidence, reviewer)
  memory_fts       FTS5 keyword index over pre-segmented text (only when FTS5 is available); a derived
                   index over every record of any status, rebuilt from memories when it is missing rows
  messages         raw chat messages (version 2); append-only, one row per CanonicalMessage. A message is
                   evidence, not a belief: it is never corrected, and no memory is derived from it automatically
  messages_fts     FTS5 keyword index over message content, same segmentation and tokenizer as memory_fts; a
                   derived index, rebuilt from messages when it is missing rows

Datetimes are stored twice: the original ISO-8601 text (offset preserved, for exact round trips) and a
UTC text in fixed width (``*_utc``) used for ordering and interval comparisons.
"""
from __future__ import annotations

import sqlite3
from typing import Callable, Dict

SCHEMA_VERSION = 2

_ORIGINS = "('synthetic','simulated','real')"

V1 = f"""
CREATE TABLE IF NOT EXISTS sources (
    source_id   TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    origin      TEXT NOT NULL CHECK (origin IN {_ORIGINS}),
    description TEXT NOT NULL,
    record_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
    memory_id         TEXT PRIMARY KEY,
    persona_id        TEXT NOT NULL,
    kind              TEXT NOT NULL CHECK (kind IN ('fact','preference','event','relationship')),
    subject           TEXT NOT NULL,
    attribute         TEXT NOT NULL,
    value             TEXT NOT NULL,
    statement         TEXT NOT NULL,
    origin            TEXT NOT NULL CHECK (origin IN {_ORIGINS}),
    cardinality       TEXT NOT NULL CHECK (cardinality IN ('single','multi','unknown')),
    confidence        REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    recorded_at       TEXT NOT NULL,
    recorded_at_utc   TEXT NOT NULL,
    valid_from        TEXT,
    valid_from_utc    TEXT,
    valid_to          TEXT,
    valid_to_utc      TEXT,
    status            TEXT NOT NULL CHECK (status IN ('active','superseded','retracted')),
    status_changed_at TEXT,              -- UTC; when status left 'active'
    supersedes        TEXT REFERENCES memories (memory_id),
    superseded_by     TEXT REFERENCES memories (memory_id) DEFERRABLE INITIALLY DEFERRED,
    correction_reason TEXT,
    schema_version    TEXT NOT NULL,
    content_sha256    TEXT NOT NULL,     -- submitted content without store-managed fields (idempotency)
    duplicate_key     TEXT NOT NULL,     -- content without memory_id / recorded_at (retry with a fresh id)
    CHECK ((status = 'superseded') = (superseded_by IS NOT NULL)),
    CHECK ((status = 'active') = (status_changed_at IS NULL)),
    CHECK (supersedes IS NULL OR (correction_reason IS NOT NULL AND supersedes <> memory_id)),
    CHECK (superseded_by IS NULL OR superseded_by <> memory_id),
    CHECK (valid_from_utc IS NULL OR valid_to_utc IS NULL OR valid_to_utc > valid_from_utc)
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_memories_supersedes ON memories (supersedes) WHERE supersedes IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_memories_slot ON memories (persona_id, subject, attribute, recorded_at_utc);
CREATE INDEX IF NOT EXISTS ix_memories_status ON memories (persona_id, status);
CREATE INDEX IF NOT EXISTS ix_memories_duplicate ON memories (persona_id, duplicate_key);

CREATE TABLE IF NOT EXISTS provenance (
    memory_id   TEXT NOT NULL REFERENCES memories (memory_id),
    position    INTEGER NOT NULL CHECK (position >= 0),
    source_id   TEXT NOT NULL REFERENCES sources (source_id),
    locator     TEXT,
    quote       TEXT,
    observed_at TEXT,
    PRIMARY KEY (memory_id, position)
);
CREATE INDEX IF NOT EXISTS ix_provenance_source ON provenance (source_id);

CREATE TABLE IF NOT EXISTS memory_conflicts (
    memory_a    TEXT NOT NULL REFERENCES memories (memory_id),
    memory_b    TEXT NOT NULL REFERENCES memories (memory_id),
    origin      TEXT NOT NULL CHECK (origin IN ('detected','declared')),
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (memory_a, memory_b),
    CHECK (memory_a < memory_b)
);
CREATE INDEX IF NOT EXISTS ix_conflicts_b ON memory_conflicts (memory_b);

CREATE TABLE IF NOT EXISTS memory_events (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id   TEXT NOT NULL REFERENCES memories (memory_id),
    action      TEXT NOT NULL CHECK (action IN ('add','import','correct','superseded','retract')),
    related_id  TEXT REFERENCES memories (memory_id),
    reason      TEXT,
    source_id   TEXT REFERENCES sources (source_id),
    locator     TEXT,
    reviewer    TEXT,
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_events_memory ON memory_events (memory_id, event_id);

-- Append-only: memories are never deleted, content never changes, status only leaves 'active' once.
CREATE TRIGGER IF NOT EXISTS memories_no_delete BEFORE DELETE ON memories
BEGIN SELECT RAISE(ABORT, 'memories are append-only'); END;
CREATE TRIGGER IF NOT EXISTS memories_immutable BEFORE UPDATE OF
    memory_id, persona_id, kind, subject, attribute, value, statement, origin, cardinality, confidence,
    recorded_at, recorded_at_utc, valid_from, valid_from_utc, valid_to, valid_to_utc, supersedes,
    correction_reason, schema_version, content_sha256, duplicate_key ON memories
BEGIN SELECT RAISE(ABORT, 'memory content is immutable; write a correction instead'); END;
CREATE TRIGGER IF NOT EXISTS memories_status_once BEFORE UPDATE OF status, superseded_by, status_changed_at ON memories
WHEN OLD.status <> 'active'
BEGIN SELECT RAISE(ABORT, 'only an active memory can be superseded or retracted'); END;
CREATE TRIGGER IF NOT EXISTS provenance_no_change BEFORE UPDATE ON provenance
BEGIN SELECT RAISE(ABORT, 'provenance is immutable'); END;
CREATE TRIGGER IF NOT EXISTS provenance_no_delete BEFORE DELETE ON provenance
BEGIN SELECT RAISE(ABORT, 'provenance is immutable'); END;
CREATE TRIGGER IF NOT EXISTS events_no_change BEFORE UPDATE ON memory_events
BEGIN SELECT RAISE(ABORT, 'the audit log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON memory_events
BEGIN SELECT RAISE(ABORT, 'the audit log is append-only'); END;
"""

FTS_V1 = """
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    memory_id UNINDEXED, bigrams, unigrams, tokenize = 'unicode61 remove_diacritics 2'
);
"""

V2 = """
CREATE TABLE IF NOT EXISTS messages (
    message_id        TEXT PRIMARY KEY,
    conversation_id   TEXT NOT NULL,
    target_persona_id TEXT NOT NULL,     -- whose history holds the message; not who it is about
    speaker_id        TEXT NOT NULL,     -- who said it
    timestamp         TEXT NOT NULL,     -- as given, offset preserved
    timestamp_utc     TEXT NOT NULL,
    content           TEXT NOT NULL,
    source_type       TEXT NOT NULL,
    source_id         TEXT NOT NULL REFERENCES sources (source_id),
    locator           TEXT,
    recorded_at       TEXT NOT NULL,     -- when the system learned the message
    recorded_at_utc   TEXT NOT NULL,
    content_sha256    TEXT NOT NULL,     -- submitted content without recorded_at (idempotency)
    CHECK (length(target_persona_id) > 0 AND length(speaker_id) > 0),
    CHECK (recorded_at_utc >= timestamp_utc)
);
CREATE INDEX IF NOT EXISTS ix_messages_persona_time ON messages (target_persona_id, timestamp_utc, message_id);
CREATE INDEX IF NOT EXISTS ix_messages_persona_speaker ON messages (target_persona_id, speaker_id, timestamp_utc);
CREATE INDEX IF NOT EXISTS ix_messages_conversation ON messages (conversation_id, timestamp_utc);
CREATE INDEX IF NOT EXISTS ix_messages_source ON messages (source_id);

CREATE TRIGGER IF NOT EXISTS messages_no_update BEFORE UPDATE ON messages
BEGIN SELECT RAISE(ABORT, 'messages are append-only'); END;
CREATE TRIGGER IF NOT EXISTS messages_no_delete BEFORE DELETE ON messages
BEGIN SELECT RAISE(ABORT, 'messages are append-only'); END;
"""

FTS_V2 = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    message_id UNINDEXED, bigrams, unigrams, tokenize = 'unicode61 remove_diacritics 2'
);
"""


def _migrate_v2(conn: sqlite3.Connection, fts: bool) -> None:
    for stmt in _statements(V2):
        conn.execute(stmt)


MIGRATIONS: Dict[int, Callable[[sqlite3.Connection, bool], None]] = {2: _migrate_v2}


def fts5_available(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE temp._fts5_probe USING fts5(x, tokenize = 'unicode61')")
        conn.execute("DROP TABLE temp._fts5_probe")
        return True
    except sqlite3.OperationalError:
        return False


def _statements(script: str):
    """Split a DDL script into complete statements (triggers contain inner semicolons)."""
    buf = ""
    for line in script.splitlines(keepends=True):
        if not buf and (not line.strip() or line.lstrip().startswith("--")):
            continue
        buf += line
        if sqlite3.complete_statement(buf):
            yield buf.strip()
            buf = ""
    if buf.strip():
        raise ValueError("incomplete SQL statement in schema script")


def migrate(conn: sqlite3.Connection, *, fts: bool) -> int:
    """Create or upgrade the schema in one IMMEDIATE transaction. Returns the resulting version."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        current = conn.execute("PRAGMA user_version").fetchone()[0]
        if current > SCHEMA_VERSION:
            raise RuntimeError(f"database schema version {current} is newer than supported {SCHEMA_VERSION}")
        if current < 1:
            for stmt in _statements(V1):
                conn.execute(stmt)
        if fts:  # the keyword index is optional; create it whenever FTS5 exists
            for stmt in _statements(FTS_V1 + FTS_V2):
                conn.execute(stmt)
        for version in range(max(current, 1) + 1, SCHEMA_VERSION + 1):
            MIGRATIONS[version](conn, fts)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return SCHEMA_VERSION
