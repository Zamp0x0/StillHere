"""SQLiteMemoryStore reliability: idempotent retries, rollback under injected failures, persistence across
restarts, migrations, and basic concurrency. Synthetic data; file databases live in pytest's tmp_path."""
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

import pytest

from conftest import at, make_memory
from memory_support import QIAOMO, fixture_raw, fixture_records, fixture_sources, open_store, qiaomo_store, snapshot
from stillhere.contracts import Cardinality, MemoryRecord, MemoryStatus, Provenance
from stillhere.memory import SCHEMA_VERSION, DuplicateMemoryId, SQLiteMemoryStore, StaleWrite
from stillhere.memory import schema as memory_schema

SINGLE = Cardinality.SINGLE


class Boom(RuntimeError):
    pass


@pytest.fixture
def store():
    s = open_store()
    yield s
    s.close()


def count(store, table="memories"):
    return store._conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def replacement(memory_id="mem_2", value="Beta Town", **kw):
    return make_memory(memory_id=memory_id, value=value, statement=f"The persona lives in {value}.",
                       recorded_at=at(2), cardinality=SINGLE, **kw)


# ---------------------------------------------------------------- 7. idempotency and retries

def test_repeated_add_writes_once(store):
    first = store.add(make_memory())
    assert store.add(make_memory()) == first
    assert count(store) == 1 and [e.action for e in store.events("mem_1")] == ["add"]


def test_same_id_with_different_content_is_refused(store):
    store.add(make_memory())
    snap = snapshot(store)
    with pytest.raises(DuplicateMemoryId):
        store.add(make_memory(value="Changed"))
    with pytest.raises(DuplicateMemoryId):
        store.add(make_memory(conflicts_with=("mem_9",)))        # a retry may not add new claims
    assert snapshot(store) == snap


def test_retry_under_a_fresh_id_returns_the_existing_record(store):
    first = store.add(make_memory(memory_id="req_a", recorded_at=at(0)))
    again = store.add(make_memory(memory_id="req_b", recorded_at=at(1)))
    assert again == first and count(store) == 1
    store.retract("req_a", "withdrawn")
    readded = store.add(make_memory(memory_id="req_c", recorded_at=at(2)))   # a new assertion after retraction
    assert readded.memory_id == "req_c" and count(store) == 2


def test_added_record_can_be_resubmitted_as_returned(store):
    store.add(make_memory(memory_id="a", value="Alpha", cardinality=SINGLE))
    b = store.add(make_memory(memory_id="b", value="Beta", cardinality=SINGLE))
    assert b.conflicts_with == ("a",)
    assert store.add(b) == b                                     # detected conflicts are store-managed


def test_correct_retry_creates_no_new_version(store):
    store.add(make_memory(cardinality=SINGLE))
    first = store.correct("mem_1", replacement(), "moved")
    assert store.correct("mem_1", replacement(), "moved") == first
    assert store.correct("mem_1", replacement(memory_id="mem_2_retry"), "moved") == first   # fresh id, same content
    assert count(store) == 2 and len(store.history("synthetic_p", "persona", "home_city")) == 2
    assert [e.action for e in store.events("mem_1")] == ["add", "superseded"]
    with pytest.raises(StaleWrite):                              # a different correction of a stale version
        store.correct("mem_1", replacement(memory_id="mem_3", value="Gamma Town"), "moved again")


def test_retract_retry_keeps_state(store):
    store.add(make_memory())
    first = store.retract("mem_1", "withdrawn", evidence=Provenance("src_2", "turn:4"))
    assert store.retract("mem_1", "withdrawn") == first
    assert [e.action for e in store.events("mem_1")] == ["add", "retract"]
    assert store.check_integrity() == []


def test_import_is_idempotent_and_detects_changed_content():
    s = qiaomo_store()
    snap = snapshot(s)
    s.import_records(fixture_records())
    assert snapshot(s) == snap
    changed = fixture_raw()
    changed[0]["value"] = "别的城市"
    with pytest.raises(DuplicateMemoryId):
        s.import_records([MemoryRecord.from_dict(changed[0])])
    assert snapshot(s) == snap


def test_concurrent_identical_requests_stay_consistent(tmp_path):
    path = tmp_path / "mem.sqlite3"
    stores = [open_store(path) for _ in range(4)]
    barrier = threading.Barrier(8)

    def submit(i):
        barrier.wait()
        s = stores[i % 4]
        added = s.add(make_memory(cardinality=SINGLE))
        fixed = s.correct("mem_1", replacement(), "moved")
        return added.memory_id, fixed

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(submit, range(8)))
    assert {r[0] for r in results} == {"mem_1"} and len({r[1] for r in results}) == 1
    assert count(stores[0]) == 2 and stores[0].check_integrity() == []
    assert [e.action for e in stores[0].events("mem_1")] == ["add", "superseded"]
    for s in stores:
        s.close()


# ---------------------------------------------------------------- 8. rollback under injected failures

def failing(name, check=None):
    """A replacement for one internal write step that raises (after an optional look at the open transaction)."""
    def boom(conn, *args, **kwargs):
        if check is not None:
            check(conn)
        raise Boom(name)
    return boom


@pytest.mark.parametrize("step", ["_insert_record", "_insert_provenance", "_write_conflicts", "_index_fts", "_log_event"])
def test_failed_correction_rolls_back_completely(store, monkeypatch, step):
    store.add(make_memory(cardinality=SINGLE))
    store.add(make_memory(memory_id="rival", value="Gamma Town", cardinality=SINGLE))   # so the replacement conflicts
    snap = snapshot(store)
    seen = {}

    def old_already_superseded(conn):
        seen["status"] = conn.execute("SELECT status FROM memories WHERE memory_id = 'mem_1'").fetchone()[0]

    monkeypatch.setattr(store, step, failing(step, old_already_superseded))
    with pytest.raises(Boom):
        store.correct("mem_1", replacement(), "moved")
    assert seen["status"] == "superseded"                         # the failure came after the old row changed
    monkeypatch.undo()
    assert snapshot(store) == snap and store.check_integrity() == []
    assert store.get("mem_1").is_active and store.get("mem_1").superseded_by is None
    assert store.get("mem_2") is None
    assert {h.record.memory_id for h in store.search("synthetic_p", "Town")} == {"mem_1", "rival"}
    assert store.correct("mem_1", replacement(), "moved").supersedes == "mem_1"      # not wedged afterwards


@pytest.mark.parametrize("step", ["_insert_provenance", "_write_conflicts", "_index_fts", "_log_event"])
def test_failed_add_rolls_back_completely(store, monkeypatch, step):
    store.add(make_memory(memory_id="a", value="Alpha", cardinality=SINGLE))
    snap = snapshot(store)
    monkeypatch.setattr(store, step, failing(step))
    with pytest.raises(Boom):
        store.add(make_memory(memory_id="b", value="Beta", cardinality=SINGLE))
    monkeypatch.undo()
    assert snapshot(store) == snap and store.get("b") is None
    assert store.get("a").conflicts_with == () and store.conflicts("synthetic_p", include_resolved=True) == []
    assert store.check_integrity() == []


def test_failed_retraction_rolls_back(store, monkeypatch):
    store.add(make_memory())
    snap = snapshot(store)
    monkeypatch.setattr(store, "_log_event", failing("_log_event"))
    with pytest.raises(Boom):
        store.retract("mem_1", "withdrawn")
    monkeypatch.undo()
    assert snapshot(store) == snap and store.get("mem_1").is_active


def test_failed_import_rolls_back(monkeypatch):
    s = open_store(clock=None, sources=fixture_sources())
    calls = {"n": 0}
    original = s._insert_provenance

    def fail_late(conn, record):
        calls["n"] += 1
        if calls["n"] == 6:
            raise Boom("provenance")
        original(conn, record)
    monkeypatch.setattr(s, "_insert_provenance", fail_late)
    with pytest.raises(Boom):
        s.import_records(fixture_records())
    assert count(s) == 0 and count(s, "provenance") == 0 and count(s, "memory_events") == 0


def test_deferred_link_violation_is_rolled_back_at_commit(store):
    store.add(make_memory())
    snap = snapshot(store)
    with pytest.raises(sqlite3.IntegrityError):
        with store._write() as conn:     # a superseded_by link to a row that never gets inserted
            conn.execute("UPDATE memories SET status = 'superseded', superseded_by = 'ghost', status_changed_at = 'x' "
                         "WHERE memory_id = 'mem_1'")
    assert snapshot(store) == snap and not store._conn.in_transaction


def test_concurrent_corrections_race_with_one_winner(tmp_path):
    rounds = 6
    path = tmp_path / "race.sqlite3"
    a, b = open_store(path), open_store(path)
    for i in range(rounds):
        a.add(make_memory(memory_id=f"m{i}", attribute=f"slot{i}", cardinality=SINGLE))
    for i in range(rounds):
        barrier = threading.Barrier(2)
        outcome = {}

        def run(store, tag):
            barrier.wait()
            try:
                outcome[tag] = store.correct(f"m{i}", make_memory(memory_id=f"m{i}_{tag}", attribute=f"slot{i}",
                                                                  value=f"Value {tag}", recorded_at=at(1)), tag)
            except StaleWrite as exc:
                outcome[tag] = exc
        threads = [threading.Thread(target=run, args=(s, tag)) for s, tag in ((a, "x"), (b, "y"))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        winners = [k for k, v in outcome.items() if not isinstance(v, Exception)]
        assert len(winners) == 1 and isinstance(outcome[{"x": "y", "y": "x"}[winners[0]]], StaleWrite)
        chain = a.history("synthetic_p", "persona", f"slot{i}")
        assert [r.memory_id for r in chain] == [f"m{i}", f"m{i}_{winners[0]}"]
        assert chain[0].superseded_by == chain[1].memory_id
    assert a.check_integrity() == [] and b.check_integrity() == []
    a.close()
    b.close()


def test_correct_and_retract_race(tmp_path):
    path = tmp_path / "race2.sqlite3"
    a, b = open_store(path), open_store(path)
    a.add(make_memory(cardinality=SINGLE))
    barrier = threading.Barrier(2)
    outcome = {}

    def do_correct():
        barrier.wait()
        try:
            outcome["correct"] = a.correct("mem_1", replacement(), "moved")
        except StaleWrite as exc:
            outcome["correct"] = exc

    def do_retract():
        barrier.wait()
        try:
            outcome["retract"] = b.retract("mem_1", "withdrawn")
        except StaleWrite as exc:
            outcome["retract"] = exc
    threads = [threading.Thread(target=do_correct), threading.Thread(target=do_retract)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(isinstance(v, StaleWrite) for v in outcome.values()) == 1
    assert a.get("mem_1").status in (MemoryStatus.SUPERSEDED, MemoryStatus.RETRACTED)
    assert a.check_integrity() == []
    a.close()
    b.close()


def test_integrity_check_after_another_connection_wrote(tmp_path):
    path = tmp_path / "two.sqlite3"
    a, b = open_store(path), open_store(path)
    b.add(make_memory())                     # a's cached keyword-index structure is now stale
    assert a.check_integrity() == [] and a.search("synthetic_p", "Alpha")[0].record.memory_id == "mem_1"
    a.add(make_memory(memory_id="mem_2", attribute="job", value="Baker"))
    assert b.check_integrity() == [] and len(b.search("synthetic_p", "Alpha Baker")) == 2
    a.close()
    b.close()


# ---------------------------------------------------------------- 9. persistence and restart

def test_everything_survives_close_and_reopen(tmp_path):
    path = tmp_path / "persist.sqlite3"
    s = qiaomo_store(path)
    s.add(make_memory(memory_id="mem_pref_tea", persona_id=QIAOMO, kind="preference", subject="persona",
                      attribute="likes_drink", value="乌龙茶", statement="乔沫喜欢喝乌龙茶。", cardinality="multi",
                      provenance=(Provenance("src_character_sheet", "section:food"),), recorded_at=at(10)))
    s.correct("mem_job_cafe", make_memory(memory_id="mem_job_cafe_v2", persona_id=QIAOMO, attribute="occupation",
                                          value="半拍咖啡 店长兼烘焙师", statement="乔沫在半拍咖啡当店长，也负责烘焙。",
                                          cardinality="single", provenance=(Provenance("src_chat_2026_04_02", "turn:5"),),
                                          valid_from=datetime(2026, 3, 20, tzinfo=timezone(timedelta(hours=8))),
                                          recorded_at=at(120)), "synthetic chat added detail")
    s.retract("mem_hobby_climbing", "synthetic note retracted", evidence=Provenance("src_note_friend_a"))
    ids = [r[0] for r in s._conn.execute("SELECT memory_id FROM memories ORDER BY memory_id")]
    before = {mid: s.get(mid) for mid in ids}
    before_events = {mid: s.events(mid) for mid in ids}
    before_search = s.search(QIAOMO, "咖啡")
    before_conflicts = s.conflicts(QIAOMO, include_resolved=True)
    s.close()

    for _ in range(2):                                            # schema init is repeatable
        r = SQLiteMemoryStore(path, clock=lambda: s._clock())
        assert r.schema_version == SCHEMA_VERSION
        assert {mid: r.get(mid) for mid in ids} == before
        assert {mid: r.events(mid) for mid in ids} == before_events
        assert r.search(QIAOMO, "咖啡") == before_search
        assert r.conflicts(QIAOMO, include_resolved=True) == before_conflicts
        assert [x.memory_id for x in r.history(QIAOMO, "persona", "occupation")] == \
            ["mem_job_bakery", "mem_job_cafe", "mem_job_cafe_v2"]
        assert r.check_integrity() == []
        assert r._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        r.close()
    for raw in fixture_raw():
        if raw["memory_id"] not in ("mem_job_cafe", "mem_hobby_climbing"):
            assert before[raw["memory_id"]].to_dict() == raw


def test_newer_schema_is_refused(tmp_path):
    path = tmp_path / "future.sqlite3"
    open_store(path).close()
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    conn.close()
    with pytest.raises(RuntimeError, match="newer"):
        SQLiteMemoryStore(path)


def test_migrate_is_idempotent_on_one_connection():
    conn = sqlite3.connect(":memory:", isolation_level=None)
    for _ in range(3):
        assert memory_schema.migrate(conn, fts=False) == SCHEMA_VERSION
    tables = conn.execute("SELECT count(*) FROM sqlite_master").fetchone()[0]
    memory_schema.migrate(conn, fts=False)
    assert conn.execute("SELECT count(*) FROM sqlite_master").fetchone()[0] == tables


@pytest.mark.skipif(not memory_schema.fts5_available(sqlite3.connect(":memory:")), reason="SQLite built without FTS5")
def test_keyword_index_is_rebuilt_when_reopened_with_fts(tmp_path):
    path = tmp_path / "nofts.sqlite3"
    s = qiaomo_store(path, fts=False)
    assert not s.fts_enabled and s.search(QIAOMO, "年糕")[0].retriever == "keyword_scan"
    s.close()
    r = SQLiteMemoryStore(path, clock=lambda: s._clock())
    assert r.fts_enabled and r.check_integrity() == []
    hit = r.search(QIAOMO, "年糕")[0]
    assert (hit.record.memory_id, hit.retriever) == ("mem_cat_name_v2", "fts5_bigram")
    r.close()


# ---------------------------------------------------------------- 12. basic concurrency

def test_one_store_shared_by_threads(store):
    def add(i):
        return store.add(make_memory(memory_id=f"t{i}", attribute=f"slot{i % 5}", value=f"V{i}", recorded_at=at(i)))
    with ThreadPoolExecutor(8) as pool:
        added = list(pool.map(add, range(40)))
    assert len(added) == 40 and count(store) == 40 and store.check_integrity() == []


def test_reader_sees_only_committed_writes(tmp_path, monkeypatch):
    path = tmp_path / "rw.sqlite3"
    writer, reader = open_store(path), open_store(path)
    writer.add(make_memory())
    inside = threading.Event()
    release = threading.Event()
    original = writer._log_event

    def slow_log(conn, *args, **kwargs):
        original(conn, *args, **kwargs)
        inside.set()
        release.wait(5)
    monkeypatch.setattr(writer, "_log_event", slow_log)
    t = threading.Thread(target=writer.retract, args=("mem_1", "withdrawn"))
    t.start()
    assert inside.wait(5)
    assert reader.get("mem_1").is_active                          # the uncommitted retraction is invisible
    release.set()
    t.join()
    assert reader.get("mem_1").status is MemoryStatus.RETRACTED
    writer.close()
    reader.close()
