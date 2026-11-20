"""Additive tables for live chat sessions (Simple Context v1). The core schema and its version are NOT changed: the message text itself
is stored in the core ``messages`` table (so the existing bigram FTS index and bitemporal reads cover it); these tables only add what a
live session needs: the session's owner, each message's role / order / request id / generation status, and the chunks of long messages.

Created on a writable store, idempotently (``ensure``); an old database gains them on first use and reads exactly as before.
"""
from __future__ import annotations

EXTENSION = "live_chat_sessions"
VERSION = 1

DDL = """
CREATE TABLE IF NOT EXISTS stillhere_extensions (
    name TEXT PRIMARY KEY,
    version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS live_sessions (
    conversation_id TEXT PRIMARY KEY,
    persona_id      TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('active','archived'))
);
CREATE TABLE IF NOT EXISTS live_messages (
    message_id        TEXT PRIMARY KEY REFERENCES messages (message_id),
    conversation_id   TEXT NOT NULL REFERENCES live_sessions (conversation_id),
    persona_id        TEXT NOT NULL,
    seq               INTEGER NOT NULL,
    role              TEXT NOT NULL CHECK (role IN ('user','assistant')),
    request_id        TEXT NOT NULL,
    source            TEXT NOT NULL,
    generation_status TEXT NOT NULL,
    reply_to          TEXT,
    chunked           INTEGER NOT NULL DEFAULT 0,
    UNIQUE (conversation_id, seq),
    UNIQUE (conversation_id, request_id, role)
);
CREATE INDEX IF NOT EXISTS ix_live_messages_order ON live_messages (conversation_id, seq);
CREATE TABLE IF NOT EXISTS live_message_chunks (
    chunk_id          TEXT PRIMARY KEY,
    parent_message_id TEXT NOT NULL REFERENCES live_messages (message_id),
    conversation_id   TEXT NOT NULL,
    persona_id        TEXT NOT NULL,
    role              TEXT NOT NULL,
    source            TEXT NOT NULL,
    chunk_index       INTEGER NOT NULL,
    chunk_offset      INTEGER NOT NULL,
    heading           TEXT,
    text              TEXT NOT NULL,
    UNIQUE (parent_message_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS ix_live_chunks_conv ON live_message_chunks (conversation_id, parent_message_id, chunk_index)
"""


def ensure(api) -> None:
    with api.transaction() as conn:
        for stmt in DDL.strip().split(";\n"):
            if stmt.strip():
                conn.execute(stmt)
        conn.execute("INSERT OR IGNORE INTO stillhere_extensions (name, version) VALUES (?, ?)", (EXTENSION, VERSION))
