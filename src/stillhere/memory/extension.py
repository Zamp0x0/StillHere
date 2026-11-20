"""Additive tables for the Memory Writer and Observations (BUILD-M2). The core schema (``SCHEMA_VERSION``) is not changed.

Why an extension and not a core migration: read-only opening requires the exact core version, so bumping it would make every
frozen snapshot unreadable. These tables are created only on a writable store, carry their own version row and are never needed
to read memories. Nothing here replaces a core table; memories stay the only place facts live.

* ``writer_batches``: one row per committed conversation group: its content and extraction hashes and the operations applied.
  It is the import manifest: a group whose hashes are already here is skipped, so a re-run or a resumed run writes nothing twice.
* ``observations`` / ``observation_links``: versioned derived observations and the memories that support or contradict them.
  A link points at a memory id only, never at an observation, so derivation is acyclic by construction.
"""
from __future__ import annotations

import sqlite3

from .schema import _statements

EXTENSION = "writer_observation"
VERSION = 1

DDL = """
CREATE TABLE IF NOT EXISTS stillhere_extensions (
    name TEXT PRIMARY KEY,
    version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS writer_batches (
    group_id TEXT NOT NULL,
    persona_id TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    extraction_sha256 TEXT NOT NULL,
    writer_version TEXT NOT NULL,
    known_at TEXT NOT NULL,
    committed_at TEXT NOT NULL,
    operations TEXT NOT NULL,
    PRIMARY KEY (group_id, content_sha256, extraction_sha256)
);
CREATE TABLE IF NOT EXISTS observations (
    observation_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    persona_id TEXT NOT NULL,
    observation_type TEXT NOT NULL,
    rule_key TEXT NOT NULL,
    claim TEXT NOT NULL,
    window_start TEXT,
    window_end TEXT,
    valid_from TEXT,
    valid_to TEXT,
    generated_at TEXT NOT NULL,
    generated_at_utc TEXT NOT NULL,
    derivation_method TEXT NOT NULL,
    support_count INTEGER NOT NULL,
    independent_event_count INTEGER NOT NULL,
    heuristic_support REAL NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'superseded', 'invalidated', 'retracted')),
    status_changed_at TEXT,
    status_reason TEXT,
    authority TEXT NOT NULL DEFAULT 'DERIVED_OBSERVATION' CHECK (authority = 'DERIVED_OBSERVATION'),
    PRIMARY KEY (observation_id, version)
);
CREATE INDEX IF NOT EXISTS ix_observations_key ON observations (persona_id, rule_key, version);
CREATE TABLE IF NOT EXISTS observation_links (
    observation_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    memory_id TEXT NOT NULL REFERENCES memories (memory_id),
    role TEXT NOT NULL CHECK (role IN ('support', 'contradict')),
    PRIMARY KEY (observation_id, version, memory_id, role),
    FOREIGN KEY (observation_id, version) REFERENCES observations (observation_id, version)
);
CREATE TRIGGER IF NOT EXISTS observation_links_no_update BEFORE UPDATE ON observation_links
BEGIN SELECT RAISE(ABORT, 'observation links are append-only'); END;
CREATE TRIGGER IF NOT EXISTS writer_batches_no_update BEFORE UPDATE ON writer_batches
BEGIN SELECT RAISE(ABORT, 'writer batches are append-only'); END;
"""


def ensure(store) -> None:
    """Create the extension on a writable store (idempotent)."""
    with store.transaction() as conn:
        for stmt in DDL.strip().split(";\n"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt if stmt.endswith("END") or not stmt.upper().startswith("CREATE TRIGGER") else stmt)
        conn.execute("INSERT OR IGNORE INTO stillhere_extensions (name, version) VALUES (?, ?)", (EXTENSION, VERSION))


def present(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'observations'").fetchone() is not None
