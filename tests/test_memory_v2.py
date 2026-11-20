"""BUILD-2: schema v2, raw messages, known_at, persona scope, read-only access, snapshots and MemoryAPI.
All data is synthetic; databases live in tmp dirs or in memory."""
import hashlib
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from conftest import at, make_memory
from memory_support import QIAOMO, QIAOMO_NOW, TEST_SOURCES, fixture_records, fixture_sources
from stillhere.contracts import (CanonicalMessage, Cardinality, DataOrigin, HitKind, MemoryStatus, Provenance,
                                 SourceKind, SourceRef, ValidationError, message_locator)
from stillhere.memory import (SCHEMA_VERSION, MemoryAPI, MemoryNotFound, MessageConflict, PersonaScopeError,
                              ReadOnlyStore, SchemaMismatch, SlotRetriever, SQLiteMemoryStore, WriteRejected,
                              resolve_times, schema as memory_schema)

A, B = "synthetic_a", "synthetic_b"
CST = timezone(timedelta(hours=8))
SEARCH_MODES = pytest.mark.parametrize("fts", [None, False], ids=["fts5", "scan"])


def fact(memory_id, persona=A, *, recorded, value="跑步", statement=None, attribute="hobby", subject="persona",
         valid_from=None, valid_to=None, **kw):
    return make_memory(memory_id=memory_id, persona_id=persona, recorded_at=recorded, value=value, subject=subject,
                       statement=statement or f"她喜欢{value}", attribute=attribute, valid_from=valid_from,
                       valid_to=valid_to, **kw)


def message(message_id="msg_1", *, persona=A, speaker=B, when=None, content="我明天搬去上海", **kw):
    fields = dict(conversation_id="conv_1", message_id=message_id, target_persona_id=persona, speaker_id=speaker,
                  timestamp=when or at(1), content=content, source_type="synthetic_chat", source_id="src_2")
    fields.update(kw)
    return CanonicalMessage(**fields)


def make_api(path=":memory:", *, now=None, **kw):
    clock = kw.pop("clock", lambda: now or at(365))
    api = MemoryAPI(path, clock=clock, **kw)
    for s in TEST_SOURCES:
        api.register_source(s)
    return api


def ids(hits):
    return [h.memory_id for h in hits]


def rows(path, sql):
    conn = sqlite3.connect(path)
    try:
        return [tuple(r) for r in conn.execute(sql)]
    finally:
        conn.close()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- T1 future knowledge

@SEARCH_MODES
def test_a_record_learned_later_stays_invisible_until_then(fts):
    api = make_api(fts=fts)
    api.add_facts(A, [fact("late", recorded=at(100), valid_from=at(1), value="南京", statement="她住在南京")])
    store = api.store
    assert api.search(A, "南京", as_of=at(10)) == []                                      # the safe default
    assert api.search(A, "南京", as_of=at(10), known_at=at(10)) == []
    assert store.search(A, "南京", as_of=at(10), known_at=at(10)) == []
    assert api.facts_at(A, as_of=at(10)) == []
    assert ids(api.search(A, "南京", as_of=at(10), known_at=at(100))) == ["late"]         # today's knowledge, on request
    assert api.search(A, "南京", as_of=at(10), known_at=at(100) - timedelta(microseconds=1)) == []


def test_the_slot_path_cannot_see_later_knowledge():
    api = make_api(persona_names={A: ("小安",)})
    api.add_facts(A, [fact("late", recorded=at(100), valid_from=at(1), value="南京", statement="小安住在南京",
                           attribute="home_city")])
    assert api.retrieve(A, "小安住在哪", as_of=at(10)).hits == ()
    res = SlotRetriever(api.store, persona_names={A: ("小安",)}, clock=lambda: at(365))
    assert res.retrieve(A, "小安住在哪", as_of=at(10), known_at=at(10)).hits == ()
    assert [h.record.memory_id for h in api.retrieve(A, "小安住在哪", as_of=at(10), known_at=at(100)).hits] == ["late"]


def test_time_defaults():
    clock = lambda: at(5)
    assert resolve_times(None, None, clock) == (at(5), at(5))
    assert resolve_times(at(1), None, clock) == (at(1), at(1))
    assert resolve_times(None, at(2), clock) == (at(2), at(2))
    assert resolve_times(at(1), at(2), clock) == (at(1), at(2))


# ---------------------------------------------------------------- T2 / T8 corrections and retractions over time

@SEARCH_MODES
def test_a_corrected_fact_is_found_as_it_was_believed(fts):
    api = make_api(fts=fts)
    api.add_facts(A, [fact("old", recorded=at(1), value="跑步")])
    api.correct(A, "old", fact("new", recorded=at(100), value="游泳"), "记错了")

    old_view = api.search(A, "跑步", known_at=at(50))
    assert ids(old_view) == ["old"]
    assert api.search(A, "游泳", known_at=at(50)) == []
    assert api.search(A, "跑步", known_at=at(120)) == []
    assert ids(api.search(A, "游泳", known_at=at(120))) == ["new"]
    assert api.search(A, "跑步") == []                                                     # current belief
    # the correction takes effect exactly at the replacement's recorded_at: never both, never neither
    assert [r.memory_id for r in api.facts_at(A, known_at=at(100) - timedelta(microseconds=1))] == ["old"]
    assert [r.memory_id for r in api.facts_at(A, known_at=at(100))] == ["new"]
    # the record reads as it stood then, not as it stands now
    [then] = api.facts_at(A, known_at=at(50))
    assert (then.status, then.superseded_by) == (MemoryStatus.ACTIVE, None)
    assert api.get(A, "old").status is MemoryStatus.SUPERSEDED
    assert api.get(A, "old", known_at=at(50)).status is MemoryStatus.ACTIVE
    assert api.get(A, "new", known_at=at(50)) is None
    assert [r.status for r in api.history(A, "persona", "hobby", known_at=at(50))] == [MemoryStatus.ACTIVE]
    assert [r.status for r in api.history(A, "persona", "hobby", known_at=at(120))] == [MemoryStatus.SUPERSEDED,
                                                                                        MemoryStatus.ACTIVE]


@SEARCH_MODES
def test_search_and_facts_at_agree_at_every_moment(fts):
    now = [at(1)]
    api = make_api(fts=fts, clock=lambda: now[0])
    word = dict(statement="她喜欢长笛")
    api.add_facts(A, [fact("v1", recorded=at(1), value="一", valid_from=at(0), valid_to=at(60), **word),
                      fact("w1", recorded=at(5), value="二", attribute="sport", **word),
                      fact("late", recorded=at(70), value="三", attribute="city", valid_from=at(2), **word)])
    api.correct(A, "v1", fact("v2", recorded=at(40), value="四", valid_from=at(0), valid_to=at(60), **word), "改")
    now[0] = at(50)
    api.retract(A, "w1", "撤")
    for known in range(0, 90, 5):
        for as_of in range(0, 90, 15):
            searched = sorted(ids(api.search(A, "长笛", as_of=at(as_of), known_at=at(known))))
            listed = sorted(r.memory_id for r in api.facts_at(A, as_of=at(as_of), known_at=at(known)))
            assert searched == listed, (known, as_of)
    assert sorted(ids(api.search(A, "长笛", as_of=at(30), known_at=at(35)))) == ["v1", "w1"]
    assert sorted(ids(api.search(A, "长笛", as_of=at(30), known_at=at(45)))) == ["v2", "w1"]
    assert sorted(ids(api.search(A, "长笛", as_of=at(30), known_at=at(80)))) == ["late", "v2"]


def test_the_slot_path_reads_the_belief_of_that_time():
    api = make_api(persona_names={A: ("小安",)})
    home = dict(attribute="home_city", cardinality=Cardinality.SINGLE)
    api.add_facts(A, [fact("old", recorded=at(1), value="广州", statement="小安住在广州", **home)])
    api.correct(A, "old", fact("new", recorded=at(100), value="悉尼", statement="小安住在悉尼", **home), "记错了")

    then = api.retrieve(A, "小安住在哪", known_at=at(50))
    assert [h.record.memory_id for h in then.hits] == ["old"] and then.slots[0].corrected == ()
    assert then.slots[0].value == "广州"
    later = api.retrieve(A, "小安住在哪", known_at=at(120))
    assert [h.record.memory_id for h in later.hits] == ["new"]
    assert [r.memory_id for r in later.slots[0].corrected] == ["old"]
    assert later.known_at == at(120)


@SEARCH_MODES
def test_a_retraction_shows_only_after_it_happened(fts):
    now = [at(1)]
    api = make_api(fts=fts, clock=lambda: now[0])
    api.add_facts(A, [fact("m", recorded=at(1), value="攀岩")])
    now[0] = at(100)
    api.retract(A, "m", "并没有这回事")
    now[0] = at(200)

    assert api.search(A, "攀岩") == []
    assert ids(api.search(A, "攀岩", known_at=at(50))) == ["m"]
    assert api.search(A, "攀岩", known_at=at(100)) == []                                    # half-open, like valid time
    assert api.get(A, "m", known_at=at(50)).status is MemoryStatus.ACTIVE
    assert api.get(A, "m", known_at=at(150)).status is MemoryStatus.RETRACTED


def test_a_corrected_then_retracted_chain():
    now = [at(1)]
    api = make_api(clock=lambda: now[0])
    api.add_facts(A, [fact("v1", recorded=at(1), value="跑步")])
    api.correct(A, "v1", fact("v2", recorded=at(100), value="游泳"), "记错了")
    now[0] = at(150)
    api.retract(A, "v2", "也不对")
    now[0] = at(200)
    assert [[r.memory_id for r in api.facts_at(A, known_at=at(d))] for d in (50, 120, 160)] == [["v1"], ["v2"], []]


def test_links_to_records_not_yet_known_stay_hidden():
    api = make_api()
    single = dict(attribute="home_city", cardinality=Cardinality.SINGLE)
    api.add_facts(A, [fact("first", recorded=at(1), value="甲城", **single)])
    api.add_facts(A, [fact("second", recorded=at(50), value="乙城", **single)])
    assert api.get(A, "first").conflicts_with == ("second",)
    assert api.get(A, "first", known_at=at(10)).conflicts_with == ()
    assert [r.conflicts_with for r in api.facts_at(A, known_at=at(10))] == [()]
    assert {r.memory_id for r in api.facts_at(A, known_at=at(60))} == {"first", "second"}


@SEARCH_MODES
def test_low_level_reads_cannot_see_later_knowledge_by_default(fts):
    """The calls of the audit's leak probe: the store and SlotRetriever directly, with as_of and nothing else."""
    store = make_api(fts=fts).store
    store.add(fact("late", recorded=at(100), valid_from=at(0), value="南京", statement="她在南京跑步"))
    retriever = SlotRetriever(store, clock=lambda: at(365))
    asof, known = at(10), at(100)

    assert store.search(A, "南京", as_of=asof) == []
    assert store.facts_at(A, as_of=asof) == []
    assert store.facts_during(A, start=at(5), end=asof) == []
    assert retriever.retrieve(A, "你在南京做了什么", as_of=asof).hits == ()

    assert [h.record.memory_id for h in store.search(A, "南京", as_of=asof, known_at=known)] == ["late"]
    assert [r.memory_id for r in store.facts_at(A, as_of=asof, known_at=known)] == ["late"]
    assert [r.memory_id for r in store.facts_during(A, start=at(5), end=asof, known_at=known)] == ["late"]
    assert [h.record.memory_id for h in retriever.retrieve(A, "你在南京做了什么", as_of=asof, known_at=known).hits] == ["late"]
    # neither time given: the clock's now, for both
    assert [h.record.memory_id for h in store.search(A, "南京")] == ["late"]
    assert [h.record.memory_id for h in retriever.retrieve(A, "你在南京做了什么").hits] == ["late"]


def test_a_timeline_without_as_of_still_stops_at_what_is_known():
    now = [at(50)]
    store = make_api(clock=lambda: now[0]).store
    store.add(fact("past", recorded=at(1), valid_from=at(0), valid_to=at(10)))
    store.add(fact("future", recorded=at(100), attribute="sport"))
    assert [r.memory_id for r in store.facts_at(A)] == ["past"]            # any validity, known by the clock's now
    now[0] = at(120)
    assert {r.memory_id for r in store.facts_at(A)} == {"past", "future"}


def test_facts_during_follows_the_same_rule():
    api = make_api()
    api.add_facts(A, [fact("late", recorded=at(100), valid_from=at(1), valid_to=at(30))])
    assert api.facts_during(A, start=at(5), end=at(10)) == []
    assert [r.memory_id for r in api.facts_during(A, start=at(5), end=at(10), known_at=at(100))] == ["late"]


# ---------------------------------------------------------------- T11 time boundaries

def test_valid_time_is_half_open_and_compared_in_utc():
    api = make_api()
    start = datetime(2026, 3, 15, 0, 0, tzinfo=CST)                   # the same instant as 2026-03-14 16:00 UTC
    end = datetime(2026, 4, 1, 0, 0, tzinfo=CST)
    api.add_facts(A, [fact("m", recorded=at(1), value="北川", statement="她住在北川", valid_from=start, valid_to=end)])
    known = at(200)
    tick = timedelta(microseconds=1)
    assert ids(api.search(A, "北川", as_of=start, known_at=known)) == ["m"]
    assert ids(api.search(A, "北川", as_of="2026-03-14T16:00:00Z", known_at=known)) == ["m"]
    assert api.search(A, "北川", as_of=start - tick, known_at=known) == []
    assert ids(api.search(A, "北川", as_of=end - tick, known_at=known)) == ["m"]
    assert api.search(A, "北川", as_of=end, known_at=known) == []
    with pytest.raises(ValueError):
        api.search(A, "北川", as_of=datetime(2026, 3, 20))             # a naive time is refused, never guessed


# ---------------------------------------------------------------- T3 persona scope

def two_personas():
    api = make_api()
    api.add_facts(A, [fact("mem_a", A, recorded=at(1), value="乐器", statement="她会吹长笛")])
    api.add_facts(B, [fact("mem_b", B, recorded=at(1), value="乐器", statement="他会拉大提琴")])
    return api


def test_a_search_returns_only_that_persona():
    api = two_personas()
    assert ids(api.search(A, "乐器")) == ["mem_a"] and ids(api.search(B, "乐器")) == ["mem_b"]
    assert [r.memory_id for r in api.facts_at(A)] == ["mem_a"]
    assert api.search(A, "大提琴") == []


def test_another_personas_records_cannot_be_read_or_changed():
    api = two_personas()
    before = api.store.get("mem_b")
    with pytest.raises(PersonaScopeError):
        api.get(A, "mem_b")
    with pytest.raises(PersonaScopeError):
        api.retract(A, "mem_b", "不是她的")
    with pytest.raises(PersonaScopeError):
        api.correct(A, "mem_b", fact("mem_x", B, recorded=at(2), value="钢琴"), "不是她的")
    with pytest.raises(PersonaScopeError):                               # A's id, with a replacement owned by B
        api.correct(A, "mem_a", fact("mem_y", B, recorded=at(2), value="钢琴"), "混了人物")
    with pytest.raises(PersonaScopeError):
        api.events(A, "mem_b")
    with pytest.raises(PersonaScopeError):
        api.add_facts(A, [fact("mem_z", B, recorded=at(2), value="钢琴")])
    assert api.store.get("mem_b") == before and api.store.get("mem_x") is None and api.store.get("mem_z") is None
    assert [e.action for e in api.store.events("mem_b")] == ["add"]


def test_the_retry_of_a_correction_is_scoped_too():
    api = two_personas()
    replacement = fact("mem_b2", B, recorded=at(2), value="钢琴")
    api.correct(B, "mem_b", replacement, "改了")
    with pytest.raises(PersonaScopeError):                               # the "already corrected" path returns a record
        api.correct(A, "mem_b", fact("mem_b3", A, recorded=at(2), value="钢琴"), "改了")


def test_messages_and_sources_are_scoped():
    api = two_personas()
    shared = make_memory(memory_id="mem_a3", persona_id=A, provenance=(Provenance(source_id="src_2"),), recorded_at=at(3))
    api.add_facts(A, [shared])
    api.ingest_messages([message("msg_b", persona=B, speaker=A, source_id="src_1", content="秘密是星期三")])
    assert api.search_messages(A, "秘密") == []
    assert [h.message_id for h in api.search_messages(B, "秘密")] == ["msg_b"]
    with pytest.raises(PersonaScopeError):
        api.get_message(A, "msg_b")
    with pytest.raises(PersonaScopeError):
        api.get_evidence(B, "src_2")                                      # only A cites it
    assert [c.memory_id for c in api.get_evidence(A, "src_2").citations] == ["mem_a3"]
    # src_1 backs memories of both personas: each sees their own citations and messages only
    assert [c.memory_id for c in api.get_evidence(A, "src_1").citations] == ["mem_a"]
    b_view = api.get_evidence(B, "src_1")
    assert [c.memory_id for c in b_view.citations] == ["mem_b"] and b_view.message_count == 1
    assert api.get_evidence(A, "src_1").message_count == 0
    with pytest.raises(MemoryNotFound):
        api.get_evidence(A, "src_missing")
    with pytest.raises(ValidationError):
        api.search("", "乐器")


# ---------------------------------------------------------------- T4 migration

def downgrade_to_v1(path):
    """A real v1 file: v2 only added these objects, so removing them leaves exactly what v1 wrote."""
    conn = sqlite3.connect(path)
    conn.execute("DROP TABLE messages")
    conn.execute("DROP TABLE messages_fts")
    conn.execute("PRAGMA user_version = 1")
    conn.close()


V1_TABLES = ("sources", "memories", "provenance", "memory_conflicts", "memory_events")


def v1_database(path):
    now = [at(1)]
    api = make_api(path, clock=lambda: now[0])
    api.add_facts(A, [fact("old", recorded=at(1), value="跑步"), fact("gone", recorded=at(1), value="攀岩")])
    api.correct(A, "old", fact("new", recorded=at(100), value="游泳"), "记错了")
    now[0] = at(110)
    api.retract(A, "gone", "并没有")
    api.close()
    downgrade_to_v1(path)
    return {t: rows(path, f"SELECT * FROM {t} ORDER BY rowid") for t in V1_TABLES}


def test_v1_upgrades_in_place_and_keeps_its_history(tmp_path):
    path = tmp_path / "v1.sqlite3"
    before = v1_database(path)
    assert rows(path, "PRAGMA user_version") == [(1,)]
    assert rows(path, "SELECT name FROM sqlite_master WHERE name = 'messages'") == []

    api = MemoryAPI(path, clock=lambda: at(200))
    assert api.store.schema_version == SCHEMA_VERSION == 2
    assert api.check_integrity() == []
    assert ids(api.search(A, "跑步", known_at=at(50))) == ["old"] and ids(api.search(A, "游泳")) == ["new"]
    assert ids(api.search(A, "攀岩", known_at=at(50))) == ["gone"] and api.search(A, "攀岩") == []
    assert api.ingest_messages([message(when=at(150))]).added == 1
    api.close()
    assert {t: rows(path, f"SELECT * FROM {t} ORDER BY rowid") for t in V1_TABLES} == before

    again = MemoryAPI(path, clock=lambda: at(200))                       # a second open migrates nothing
    assert again.stats()["counts"]["messages"] == 1 and again.store.schema_version == 2
    again.close()


def test_a_failed_migration_leaves_v1_untouched(tmp_path, monkeypatch):
    path = tmp_path / "v1.sqlite3"
    before = v1_database(path)

    def fails(conn, fts):
        conn.execute("CREATE TABLE half_done (x)")
        raise RuntimeError("synthetic failure")

    monkeypatch.setitem(memory_schema.MIGRATIONS, 2, fails)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        SQLiteMemoryStore(path)
    assert rows(path, "PRAGMA user_version") == [(1,)]
    assert rows(path, "SELECT name FROM sqlite_master WHERE name IN ('half_done', 'messages')") == []
    assert {t: rows(path, f"SELECT * FROM {t} ORDER BY rowid") for t in V1_TABLES} == before


# ---------------------------------------------------------------- T5 / T6 raw messages

def test_a_message_keeps_who_holds_it_and_who_said_it():
    api = make_api()
    said = datetime(2026, 2, 1, 21, 30, tzinfo=CST)
    m = message(when=said, locator="line:42")
    assert api.ingest_messages([m]).added == 1
    stored = api.get_message(A, "msg_1")
    assert (stored.target_persona_id, stored.speaker_id, stored.conversation_id) == (A, B, "conv_1")
    assert stored.content == "我明天搬去上海" and (stored.source_id, stored.locator) == ("src_2", "line:42")
    assert stored.timestamp == said and stored.timestamp.utcoffset() == timedelta(hours=8)    # original offset kept
    assert stored.recorded_at == at(365)                                                      # the clock, at ingestion
    assert rows_of(api, "SELECT timestamp_utc FROM messages") == [("2026-02-01T13:30:00.000000Z",)]
    # the fact in it is about the speaker, and ingestion alone decides nothing about anyone
    assert api.stats()["counts"]["memories"] == 0 and api.search(A, "上海") == []
    [hit] = api.search_messages(A, "上海")
    assert (hit.kind, hit.persona_id, hit.speaker_id, hit.subject_id) == (HitKind.MESSAGE, A, B, None)
    assert (hit.message_id, hit.source_id, hit.locator, hit.text) == ("msg_1", "src_2", "line:42", m.content)
    assert hit.retriever == "messages_fts" and hit.score > 0


def rows_of(api, sql):
    return [tuple(r) for r in api.store._conn.execute(sql)]


@pytest.mark.parametrize("bad", [dict(speaker_id=""), dict(target_persona_id=""), dict(content=" "),
                                 dict(timestamp="2026-02-01T10:00:00"), dict(timestamp="not a time"),
                                 dict(recorded_at=at(0))])
def test_a_message_must_be_complete(bad):
    with pytest.raises(ValidationError):
        message(**bad)


def test_ingesting_twice_adds_nothing_and_a_changed_message_is_a_conflict():
    api = make_api()
    batch = [message("msg_1"), message("msg_2", when=at(2), speaker=A, content="那我们周末见")]
    first = api.ingest_messages(batch)
    assert (first.added, first.skipped) == (2, 0)
    second = api.ingest_messages(batch)
    assert (second.added, second.skipped) == (0, 2)
    assert rows_of(api, "SELECT count(*) FROM messages_fts") == [(2,)]

    with pytest.raises(MessageConflict):
        api.ingest_messages([message("msg_3", when=at(3)), message("msg_1", content="改过的内容")])
    assert api.stats()["counts"]["messages"] == 2                                # the whole batch rolled back
    assert api.get_message(A, "msg_1").content == "我明天搬去上海"
    with pytest.raises(MessageConflict):
        api.ingest_messages([message("msg_1", speaker="synthetic_c")])
    assert api.check_integrity() == []


def test_messages_are_append_only_and_need_a_registered_source():
    api = make_api()
    api.ingest_messages([message()])
    for sql in ("UPDATE messages SET content = 'x'", "DELETE FROM messages"):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            api.store._conn.execute(sql)
    with pytest.raises(WriteRejected, match="unregistered"):
        api.ingest_messages([message("msg_2", source_id="src_missing")])
    with pytest.raises(WriteRejected, match="recorded"):
        api.ingest_messages([message("msg_3", when=at(400))])                    # after the clock, without a replay time


@SEARCH_MODES
def test_message_search_respects_time_speaker_and_knowledge(fts):
    api = make_api(fts=fts)
    api.ingest_messages([message("early", when=at(1), content="周末去海边", recorded_at=at(1)),
                         message("late_said", when=at(20), speaker=A, content="周末去海边吗", recorded_at=at(20)),
                         message("late_known", when=at(2), content="海边的周末", recorded_at=at(30))])
    assert {h.message_id for h in api.search_messages(A, "海边")} == {"early", "late_said", "late_known"}
    assert [h.message_id for h in api.search_messages(A, "海边", as_of=at(10))] == ["early"]    # known_at = as_of
    assert {h.message_id for h in api.search_messages(A, "海边", as_of=at(10), known_at=at(30))} == {"early", "late_known"}
    assert [h.message_id for h in api.search_messages(A, "海边", speaker_id=A)] == ["late_said"]
    assert api.search_messages(A, "海边", k=0) == [] and len(api.search_messages(A, "海边", k=1)) == 1


# ---------------------------------------------------------------- T7 evidence

def test_a_fact_traces_back_to_the_message_and_its_speaker():
    api = make_api()
    api.ingest_messages([message("msg_b", persona=A, speaker=B, content="我明天搬去上海", when=at(1))])
    record = fact("mem_b_moves", recorded=at(2), subject="friend_b", attribute="home_city", value="上海",
                  statement="朋友B搬去上海", provenance=(Provenance(source_id="src_2", locator=message_locator("msg_b"),
                                                                 quote="我明天搬去上海"),))
    api.add_facts(A, [record])

    [hit] = api.search(A, "上海")
    assert (hit.kind, hit.memory_id, hit.subject_id, hit.speaker_id) == (HitKind.MEMORY, "mem_b_moves", "friend_b", B)
    assert (hit.source_id, hit.locator, hit.retriever) == ("src_2", "message:msg_b", "fts5_bigram")
    evidence = api.get_evidence(A, "src_2")
    [citation] = evidence.citations
    assert citation.message.speaker_id == B and citation.message.target_persona_id == A
    assert citation.message.content == citation.quote and evidence.message_count == 1


def test_a_fact_cannot_cite_someone_elses_message():
    api = make_api()
    api.ingest_messages([message("msg_b", persona=B, speaker=A)])
    cited = (Provenance(source_id="src_2", locator=message_locator("msg_b")),)
    with pytest.raises(WriteRejected, match="not a message of persona"):
        api.add_facts(A, [make_memory(memory_id="m1", persona_id=A, provenance=cited)])
    with pytest.raises(WriteRejected, match="not a message of persona"):
        api.add_facts(A, [make_memory(memory_id="m2", persona_id=A, provenance=(
            Provenance(source_id="src_2", locator=message_locator("msg_missing")),))])


# ---------------------------------------------------------------- T9 read-only

def built_database(path):
    api = make_api(path, now=at(365))
    api.add_facts(A, [fact("m", recorded=at(1), value="长笛", statement="她会吹长笛")])
    api.ingest_messages([message("msg_1", content="周末去海边")])
    api.close()


def test_read_only_reads_and_refuses_every_write(tmp_path):
    path = tmp_path / "mem.sqlite3"
    built_database(path)
    digest, listing = sha(path), sorted(p.name for p in tmp_path.iterdir())
    api = MemoryAPI(path, read_only=True, clock=lambda: at(365))
    assert ids(api.search(A, "长笛")) == ["m"]
    assert [h.message_id for h in api.search_messages(A, "海边")] == ["msg_1"]
    assert api.check_integrity() == [] and api.stats()["read_only"] is True
    writes = [lambda: api.register_source(SourceRef(source_id="src_9", kind=SourceKind.SCRIPT,
                                                    origin=DataOrigin.SYNTHETIC, description="x")),
              lambda: api.ingest_messages([message("msg_2")]),
              lambda: api.add_facts(A, [fact("m2", recorded=at(2))]),
              lambda: api.correct(A, "m", fact("m3", recorded=at(2)), "x"),
              lambda: api.retract(A, "m", "x"),
              lambda: api.store.rebuild_fts()]
    for write in writes:
        with pytest.raises(ReadOnlyStore):
            write()
    with pytest.raises(sqlite3.OperationalError):                        # the connection itself refuses as well
        api.store._conn.execute("UPDATE messages SET content = 'x'")
    api.close()
    assert sha(path) == digest and rows(path, "SELECT content FROM messages") == [("周末去海边",)]
    assert {p.name for p in tmp_path.iterdir()} - set(listing) <= {"mem.sqlite3-wal", "mem.sqlite3-shm"}


def test_read_only_never_creates_or_migrates(tmp_path):
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(FileNotFoundError):
        MemoryAPI(missing, read_only=True)
    assert not missing.exists()
    with pytest.raises(ValueError):
        MemoryAPI(":memory:", read_only=True)

    old = tmp_path / "v1.sqlite3"
    v1_database(old)
    digest = sha(old)
    with pytest.raises(SchemaMismatch, match="schema version 1"):
        MemoryAPI(old, read_only=True)
    assert sha(old) == digest and rows(old, "PRAGMA user_version") == [(1,)]

    junk = tmp_path / "junk.sqlite3"
    junk.write_bytes(b"this is not a database at all, not even close, really" * 4)
    with pytest.raises(SchemaMismatch):
        MemoryAPI(junk, read_only=True)


def test_read_only_falls_back_when_an_index_is_incomplete(tmp_path):
    path = tmp_path / "mem.sqlite3"
    built_database(path)
    conn = sqlite3.connect(path)
    conn.execute("DELETE FROM messages_fts")
    conn.commit()
    conn.close()
    digest = sha(path)
    api = MemoryAPI(path, read_only=True, clock=lambda: at(365))
    assert not api.store.message_fts_enabled and api.store.fts_enabled
    [hit] = api.search_messages(A, "海边")                                # complete, through the scan
    assert hit.retriever == "message_scan"
    assert any("message keyword index" in p for p in api.check_integrity())
    api.close()
    assert sha(path) == digest


# ---------------------------------------------------------------- T10 snapshot

@pytest.mark.parametrize("source", ["writable", "read_only", "memory"])
def test_a_snapshot_is_one_consistent_self_contained_file(tmp_path, source):
    path = tmp_path / "live" / "mem.sqlite3"
    path.parent.mkdir()
    if source == "memory":
        api = make_api(now=at(365))
        api.add_facts(A, [fact("m", recorded=at(1), value="长笛", statement="她会吹长笛")])
        api.ingest_messages([message("msg_1", content="周末去海边")])
    else:
        built_database(path)
        api = MemoryAPI(path, read_only=source == "read_only", clock=lambda: at(365))
    original = (ids(api.search(A, "长笛")), [h.message_id for h in api.search_messages(A, "海边")])

    dest = tmp_path / "colab" / "snapshot.sqlite3"
    dest.parent.mkdir()
    info = api.snapshot(dest)
    api.close()

    assert info.sha256 == sha(dest) and info.size_bytes == dest.stat().st_size
    assert info.schema_version == SCHEMA_VERSION
    assert (info.counts["memories"], info.counts["messages"], info.counts["sources"]) == (1, 1, 2)
    assert dest.read_bytes()[18:20] == b"\x01\x01"                       # rollback journal, not WAL
    assert [p.name for p in dest.parent.iterdir()] == [dest.name]

    moved = tmp_path / "content"                                         # copied elsewhere, as onto a Colab disk
    moved.mkdir()
    shutil.copy(dest, moved / "s.sqlite3")
    read = MemoryAPI(moved / "s.sqlite3", read_only=True, clock=lambda: at(365))
    assert read.store.fts_enabled and read.store.message_fts_enabled     # the FTS5 indexes came along
    assert (ids(read.search(A, "长笛")), [h.message_id for h in read.search_messages(A, "海边")]) == original
    assert read.check_integrity() == []
    read.close()
    assert [p.name for p in moved.iterdir()] == ["s.sqlite3"]            # reading left no side files
    assert sha(moved / "s.sqlite3") == info.sha256


def test_a_snapshot_never_overwrites(tmp_path):
    path = tmp_path / "mem.sqlite3"
    built_database(path)
    api = MemoryAPI(path, clock=lambda: at(365))
    with pytest.raises(FileExistsError):
        api.snapshot(path)                                               # not even the database itself
    other = tmp_path / "keep.sqlite3"
    other.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        api.snapshot(other)
    assert other.read_bytes() == b"keep me"
    assert not list(tmp_path.glob("*.tmp"))
    api.close()


# ---------------------------------------------------------------- T12 the existing fixture through the new API

@pytest.mark.parametrize("fts", [None, False], ids=["fts5", "scan"])
def test_the_fixture_persona_by_default_sees_only_what_was_known_then(fts):
    """The safe side of the old valid-time tests, which now ask for today's knowledge explicitly."""
    store = SQLiteMemoryStore(clock=lambda: QIAOMO_NOW, fts=fts)
    for s in fixture_sources():
        store.register_source(s)
    store.import_records(fixture_records())
    retriever = SlotRetriever(store, persona_names={QIAOMO: ("乔沫",)}, clock=lambda: QIAOMO_NOW)
    # mem_home_beichuan was recorded on 2 April, holds from 15 March; mem_home_nanyu was recorded on 14 February
    moved = datetime(2026, 3, 20, tzinfo=CST)
    res = retriever.retrieve(QIAOMO, "乔沫住哪", as_of=moved)
    assert res.hits == ()
    assert [r.memory_id for r in res.slots[0].past] == ["mem_home_nanyu"]
    assert store.search(QIAOMO, "北川", as_of=moved) == []


def test_the_fixture_persona_through_the_api():
    api = MemoryAPI(clock=lambda: QIAOMO_NOW, persona_names={QIAOMO: ("乔沫",)})
    for s in fixture_sources():
        api.register_source(s)
    api.store.import_records(fixture_records())
    assert api.stats()["counts"]["memories"] == len(fixture_records())
    assert ids(api.search(QIAOMO, "年糕"))[0] == "mem_cat_name_v2"
    assert "mem_cat_name_v2" not in ids(api.search(QIAOMO, "年糕", known_at=datetime(2026, 5, 2, 22, 11, tzinfo=CST)))
    assert ids(api.search(QIAOMO, "汤圆", known_at=datetime(2026, 5, 2, 22, 13, tzinfo=CST)))[0] == "mem_cat_name_v1"
    res = api.retrieve(QIAOMO, "你现在住哪")
    assert [h.record.memory_id for h in res.hits] == ["mem_home_beichuan"]
    assert api.check_integrity() == []
