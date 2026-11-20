"""Live chat persistence on the existing SQLite store: a session is a ``conversation_id`` owned by one persona; every user and assistant
message is a row of the core ``messages`` table (text, time, source) plus a ``live_messages`` row (role, order, request id, status).

* idempotent: the same ``request_id`` (per session and role) returns the stored message, never a second one;
* ordered: ``seq`` is assigned inside one IMMEDIATE transaction (SQLite serialises writers; a busy database waits up to the store timeout);
* isolated: every read and write checks that the session belongs to the persona (``SessionScopeError`` otherwise);
* honest: an assistant message exists only for a reply that was produced; a failed generation leaves the user message with status
  FAILED and no assistant row; assistant text is GENERATED (or FALLBACK / NEUTRAL) and is never a source for the Memory Writer
  (its speaker id is never the persona's id).
"""
from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from stillhere.contracts import CanonicalMessage, DataOrigin, SourceKind, SourceRef

import stillhere.session.chunking as chunking
import stillhere.session.schema as sch
from .config import DEFAULT, ContextConfig

LIVE_SOURCE_ID = "stillhere.live_chat"
LIVE_SOURCE_TYPE = "stillhere_live_chat"
USER_SPEAKER = "live:user"
SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
USER_STATUSES = ("RECEIVED", "ANSWERED", "FAILED", "CONTEXT_INSUFFICIENT")
ASSISTANT_STATUSES = ("GENERATED", "FALLBACK", "NEUTRAL")


class SessionError(ValueError):
    pass


class SessionScopeError(SessionError):
    """The session does not exist for this persona (another persona's session, or an unknown id)."""


def assistant_speaker(persona_id: str) -> str:
    return f"live:assistant:{persona_id}"          # never equal to the persona id: the Writer never takes it as the person's own words


@dataclass(frozen=True)
class LiveMessage:
    message_id: str
    conversation_id: str
    persona_id: str
    seq: int
    role: str                    # user | assistant
    content: str
    created_at: str
    source: str                  # live_chat
    generation_status: str
    request_id: str
    reply_to: Optional[str]
    chunked: bool

    @property
    def generated(self) -> bool:
        return self.role == "assistant"


class SessionStore:
    def __init__(self, api, config: ContextConfig = DEFAULT, count=None) -> None:
        from stillhere.context.budget import TokenCounter
        self.api, self.cfg = api, config
        self.count = count or TokenCounter()
        sch.ensure(api)
        if api.store._source(api.store._conn, LIVE_SOURCE_ID) is None:
            origin = DataOrigin.SYNTHETIC if api.store.public_only else DataOrigin.REAL
            api.register_source(SourceRef(source_id=LIVE_SOURCE_ID, kind=SourceKind.CONVERSATION, origin=origin,
                                          description="StillHere live chat sessions (user messages and generated replies)."))

    # ---------------------------------------------------------------- sessions
    def create_session(self, persona_id: str, at: datetime, session_id: Optional[str] = None) -> str:
        sid = session_id or f"sess.{uuid.uuid4().hex}"
        if not SESSION_ID.match(sid):
            raise SessionError(f"invalid session id {sid!r}")
        with self.api.transaction() as conn:
            row = conn.execute("SELECT persona_id FROM live_sessions WHERE conversation_id = ?", (sid,)).fetchone()
            if row is not None:
                if row[0] != persona_id:
                    raise SessionScopeError(f"{sid} belongs to another persona")
                return sid                               # creating an existing session of the same persona continues it
            if conn.execute("SELECT 1 FROM messages WHERE conversation_id = ? LIMIT 1", (sid,)).fetchone():
                raise SessionError(f"{sid} is an imported conversation, not a live session")
            conn.execute("INSERT INTO live_sessions VALUES (?,?,?,?)", (sid, persona_id, at.isoformat(), "active"))
        return sid

    def check(self, persona_id: str, session_id: str) -> None:
        if not isinstance(session_id, str) or not SESSION_ID.match(session_id):
            raise SessionError(f"invalid session id {session_id!r}")
        row = self.api.store._conn.execute("SELECT persona_id, status FROM live_sessions WHERE conversation_id = ?", (session_id,)).fetchone()
        if row is None or row[0] != persona_id:
            raise SessionScopeError(f"no session {session_id!r} for persona {persona_id!r}")
        if row[1] != "active":
            raise SessionError(f"{session_id} is archived")

    def archive(self, persona_id: str, session_id: str) -> None:
        self.check(persona_id, session_id)
        with self.api.transaction() as conn:
            conn.execute("UPDATE live_sessions SET status = 'archived' WHERE conversation_id = ?", (session_id,))

    # ---------------------------------------------------------------- messages
    @staticmethod
    def _mid(session_id: str, request_id: str, role: str) -> str:
        return "lm." + hashlib.sha256(f"{session_id}\x1f{request_id}\x1f{role}".encode("utf-8")).hexdigest()[:24]

    def _existing(self, conn, session_id: str, request_id: str, role: str) -> Optional[LiveMessage]:
        r = conn.execute("SELECT message_id FROM live_messages WHERE conversation_id = ? AND request_id = ? AND role = ?",
                         (session_id, request_id, role)).fetchone()
        return self.get(r[0]) if r else None

    def append(self, persona_id: str, session_id: str, role: str, content: str, at: datetime, request_id: str, *,
               status: str, reply_to: Optional[str] = None) -> LiveMessage:
        """Store one message (all or nothing). The same request id again returns the stored message (an identical retry); the same request
        id with different text is refused."""
        self.check(persona_id, session_id)
        if role not in ("user", "assistant"):
            raise SessionError(f"role {role!r}")
        if status not in (USER_STATUSES if role == "user" else ASSISTANT_STATUSES):
            raise SessionError(f"status {status!r} for a {role} message")
        if not isinstance(content, str) or not content.strip():
            raise SessionError("empty message")
        if not request_id or len(request_id) > 128:
            raise SessionError("a request id is required")
        with self.api.transaction() as conn:
            old = self._existing(conn, session_id, request_id, role)
            if old is not None:
                if old.content != content:
                    raise SessionError(f"request {request_id!r} was already used with a different {role} message")
                return old
            seq = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM live_messages WHERE conversation_id = ?", (session_id,)).fetchone()[0]
            mid = self._mid(session_id, request_id, role)
            speaker = USER_SPEAKER if role == "user" else assistant_speaker(persona_id)
            self.api.ingest_messages([CanonicalMessage(conversation_id=session_id, message_id=mid, target_persona_id=persona_id, speaker_id=speaker,
                                                       timestamp=at, content=content, source_type=LIVE_SOURCE_TYPE, source_id=LIVE_SOURCE_ID,
                                                       locator=f"seq={seq};role={role}", recorded_at=at)])
            long_ = self.count(content) > self.cfg.long_message_tokens
            conn.execute("INSERT INTO live_messages VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (mid, session_id, persona_id, seq, role, request_id, "live_chat", status, reply_to, int(long_)))
            if long_:
                for ch in chunking.split(content, self.count, self.cfg.chunk_tokens):
                    conn.execute("INSERT INTO live_message_chunks VALUES (?,?,?,?,?,?,?,?,?,?)",
                                 (f"{mid}.c{ch.index:03d}", mid, session_id, persona_id, role, "live_chat", ch.index, ch.offset, ch.heading, ch.text))
        return self.get(mid)

    def set_status(self, message_id: str, status: str) -> None:
        if status not in USER_STATUSES:
            raise SessionError(f"status {status!r}")
        with self.api.transaction() as conn:
            conn.execute("UPDATE live_messages SET generation_status = ? WHERE message_id = ? AND role = 'user'", (status, message_id))

    def get(self, message_id: str) -> Optional[LiveMessage]:
        r = self.api.store._conn.execute(
            "SELECT l.*, m.content, m.timestamp FROM live_messages l JOIN messages m ON m.message_id = l.message_id WHERE l.message_id = ?",
            (message_id,)).fetchone()
        return self._row(r) if r else None

    @staticmethod
    def _row(r) -> LiveMessage:
        return LiveMessage(r["message_id"], r["conversation_id"], r["persona_id"], r["seq"], r["role"], r["content"], r["timestamp"], r["source"],
                           r["generation_status"], r["request_id"], r["reply_to"], bool(r["chunked"]))

    def messages(self, persona_id: str, session_id: str) -> List[LiveMessage]:
        """Every message of the session, in order (``seq``)."""
        self.check(persona_id, session_id)
        rows = self.api.store._conn.execute(
            "SELECT l.*, m.content, m.timestamp FROM live_messages l JOIN messages m ON m.message_id = l.message_id "
            "WHERE l.conversation_id = ? AND l.persona_id = ? ORDER BY l.seq", (session_id, persona_id)).fetchall()
        return [self._row(r) for r in rows]

    def reply_for(self, persona_id: str, session_id: str, user_message_id: str) -> Optional[LiveMessage]:
        self.check(persona_id, session_id)
        r = self.api.store._conn.execute("SELECT message_id FROM live_messages WHERE conversation_id = ? AND reply_to = ? AND role = 'assistant'",
                                         (session_id, user_message_id)).fetchone()
        return self.get(r[0]) if r else None

    def chunks(self, message_id: str) -> List[chunking.Chunk]:
        rows = self.api.store._conn.execute("SELECT chunk_index, chunk_offset, heading, text FROM live_message_chunks WHERE parent_message_id = ? "
                                            "ORDER BY chunk_index", (message_id,)).fetchall()
        return [chunking.Chunk(r[0], r[1], r[2], r[3]) for r in rows]
