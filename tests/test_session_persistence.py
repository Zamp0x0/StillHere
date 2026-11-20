"""Simple Context v1 — live message persistence on the existing SQLite store: roles, order, idempotent retries, failure states,
session / persona isolation, migration of an existing database, restore after reopening (CASES 9, 10, 11)."""
import sqlite3

import pytest

from _ctx_world import ON, Boom, Clock, Fake, runtime
from stillhere.evaluation import context_dataset_v1 as cd
from stillhere.memory.api import MemoryAPI
from stillhere.session import SessionError, SessionRuntime, SessionScopeError
from stillhere.session.store import LIVE_SOURCE_TYPE, assistant_speaker


def test_user_and_assistant_messages_are_persisted_with_roles_order_and_status():
    rt, api = runtime()
    s = rt.create_session(cd.CA)
    r = rt.reply_session(cd.CA, s, "你好", "r1", generate=Fake("你好呀"), backend="lora", env={})
    msgs = rt.get_session_messages(cd.CA, s)
    assert [(m.seq, m.role, m.generation_status) for m in msgs] == [(1, "user", "ANSWERED"), (2, "assistant", "GENERATED")]
    assert msgs[1].reply_to == msgs[0].message_id and r.text == "你好呀" and msgs[1].generated
    raw, _ = api.conversation(cd.CA, s, as_of=cd.t(12, 31), known_at=cd.t(12, 31))                                     # the core messages table holds both, with the live source
    assert [m.speaker_id for m in raw] == ["live:user", assistant_speaker(cd.CA)] and {m.source_type for m in raw} == {LIVE_SOURCE_TYPE}
    assert assistant_speaker(cd.CA) != cd.CA                                 # the Writer never takes generated text as the persona's words


def test_retry_with_the_same_request_id_never_duplicates_or_regenerates():
    rt, _ = runtime()
    s = rt.create_session(cd.CA)
    gen = Fake("第一次的回复")
    a = rt.reply_session(cd.CA, s, "在吗", "req-1", generate=gen, backend="lora", env={})
    b = rt.reply_session(cd.CA, s, "在吗", "req-1", generate=gen, backend="lora", env={})     # browser refresh / network retry
    assert b.status == "REPLAYED" and b.text == a.text and len(gen.calls) == 1
    assert len(rt.get_session_messages(cd.CA, s)) == 2
    with pytest.raises(SessionError):
        rt.reply_session(cd.CA, s, "另一句话", "req-1", generate=gen, backend="lora", env=ON)  # same request id, different text: refused


def test_order_is_strict_and_sequential():
    rt, _ = runtime()
    s = rt.create_session(cd.CA)
    for i in range(5):
        rt.send_message(cd.CA, s, f"消息{i}", f"q{i}")
    assert [m.seq for m in rt.get_session_messages(cd.CA, s)] == [1, 2, 3, 4, 5]
    assert [m.content for m in rt.get_session_messages(cd.CA, s)] == [f"消息{i}" for i in range(5)]


def test_model_failure_keeps_the_user_message_and_invents_no_reply():
    rt, _ = runtime()
    s = rt.create_session(cd.CA)
    r = rt.reply_session(cd.CA, s, "你好", "r1", generate=Boom(), backend="lora", env=ON)
    msgs = rt.get_session_messages(cd.CA, s)
    assert r.status == "FAILED" and r.text is None and r.assistant_message is None
    assert [(m.role, m.generation_status) for m in msgs] == [("user", "FAILED")]


def test_invalid_and_foreign_sessions_are_refused():                       # CASE 10 (two personas), invalid ids
    rt, _ = runtime()
    a = rt.create_session(cd.CA, "s-a")
    b = rt.create_session(cd.CB, "s-b")
    rt.send_message(cd.CA, a, "我的会话", "q1")
    with pytest.raises(SessionScopeError):
        rt.get_session_messages(cd.CB, a)
    with pytest.raises(SessionScopeError):
        rt.send_message(cd.CA, b, "写到别人的会话", "q2")
    with pytest.raises(SessionScopeError):
        rt.create_session(cd.CB, "s-a")
    for bad in ("", "../x", "a b", "x" * 200):
        with pytest.raises(SessionError):
            rt.send_message(cd.CA, bad, "hi", "q3")


def test_two_sessions_of_one_persona_are_separate():                       # CASE 9
    rt, _ = runtime()
    a, b = rt.create_session(cd.CA), rt.create_session(cd.CA)
    rt.send_message(cd.CA, a, "A 会话的消息", "q1")
    rt.send_message(cd.CA, b, "B 会话的消息", "q1")                         # the same request id in another session is another message
    assert [m.content for m in rt.get_session_messages(cd.CA, a)] == ["A 会话的消息"]
    assert [m.content for m in rt.get_session_messages(cd.CA, b)] == ["B 会话的消息"]


def test_fresh_database_old_database_and_restore_after_reopen(tmp_path):
    path = tmp_path / "old.sqlite"
    old = MemoryAPI(path, clock=lambda: cd.t(9, 10))                       # a database created before Simple Context v1
    cd.load(old)
    before = [m.message_id for m in old.conversation(cd.CA, "syn.ca.chat.m01")[0]]
    old.store._conn.close()
    api = MemoryAPI(path, clock=lambda: cd.t(9, 10))
    rt = SessionRuntime(api, clock=Clock())                                  # the extension tables are added; imported data reads as before
    assert [m.message_id for m in api.conversation(cd.CA, "syn.ca.chat.m01")[0]] == before
    s = rt.create_session(cd.CA, "persist-1")
    rt.reply_session(cd.CA, s, "记住这句话", "r1", env=ON)
    api.store._conn.close()
    again = SessionRuntime(MemoryAPI(path, clock=lambda: cd.t(9, 10)), clock=Clock())   # a new process: the session is restored
    assert [m.content for m in again.get_session_messages(cd.CA, "persist-1")][0] == "记住这句话"
    tables = {r[0] for r in sqlite3.connect(path).execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"live_sessions", "live_messages", "live_message_chunks", "messages"} <= tables
