"""SQLiteMemoryStore: protocol conformance, schema and foreign keys, provenance round trip, temporal queries,
conflict semantics, correct/retract, the write gate and persona isolation. Synthetic data, in-memory DBs."""
import inspect
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from conftest import at, make_memory
from memory_support import QIAOMO, QIAOMO_NOW, fixture_raw, fixture_records, fixture_sources, open_store, qiaomo_store, snapshot
from stillhere.contracts import (Cardinality, DataOrigin, MemoryRecord, MemoryStatus, MemoryStore, Provenance,
                                 RetrievedMemory, SourceKind, SourceRef, ValidationError)
from stillhere.memory import (SCHEMA_VERSION, DuplicateSourceId, MemoryNotFound, SQLiteMemoryStore, StaleWrite,
                              WriteRejected)

SINGLE, MULTI, UNKNOWN = Cardinality.SINGLE, Cardinality.MULTI, Cardinality.UNKNOWN
CST = timezone(timedelta(hours=8))


@pytest.fixture
def store():
    s = open_store()
    yield s
    s.close()


@pytest.fixture
def qiaomo():
    s = qiaomo_store()
    yield s
    s.close()


# ---------------------------------------------------------------- 1. protocol conformance

def test_store_satisfies_the_memory_store_protocol(store):
    assert isinstance(store, MemoryStore)


@pytest.mark.parametrize("name", ["add", "get", "correct", "retract", "history", "search"])
def test_signatures_extend_the_protocol_compatibly(name):
    proto = list(inspect.signature(getattr(MemoryStore, name)).parameters.values())
    impl = list(inspect.signature(getattr(SQLiteMemoryStore, name)).parameters.values())
    assert [(p.name, p.kind) for p in impl[:len(proto)]] == [(p.name, p.kind) for p in proto]
    assert [p.default for p in impl[:len(proto)]] == [p.default for p in proto]
    for extra in impl[len(proto):]:                  # store-only options must not break protocol callers
        assert extra.kind is inspect.Parameter.KEYWORD_ONLY and extra.default is not inspect.Parameter.empty


def test_protocol_methods_return_contract_types(store):
    added = store.add(make_memory())
    assert isinstance(added, MemoryRecord) and store.get("mem_1") == added
    assert store.get("missing") is None
    hits = store.search("synthetic_p", "Alpha Town")
    assert [type(h) for h in hits] == [RetrievedMemory] and hits[0].retriever == store.retriever
    assert store.history("synthetic_p", "persona", "home_city") == [added]
    fixed = store.correct("mem_1", make_memory(memory_id="mem_2", value="Beta", recorded_at=at(1)), "moved")
    assert fixed.supersedes == "mem_1" and fixed.correction_reason == "moved"
    assert store.retract("mem_2", "withdrawn").status is MemoryStatus.RETRACTED


# ---------------------------------------------------------------- 2. schema and foreign keys

def test_schema_version_tables_indexes_and_pragmas(store):
    conn = store._conn
    assert store.schema_version == SCHEMA_VERSION == 2
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','index','trigger')")}
    assert {"sources", "memories", "provenance", "memory_conflicts", "memory_events"} <= names
    assert {"ux_memories_supersedes", "ix_memories_slot", "ix_memories_status", "ix_memories_duplicate",
            "ix_provenance_source", "ix_conflicts_b", "ix_events_memory"} <= names
    assert {"memories_no_delete", "memories_immutable", "memories_status_once"} <= names


def test_foreign_keys_reject_orphans(store):
    store.add(make_memory())
    conn = store._conn
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO provenance (memory_id, position, source_id) VALUES ('ghost', 0, 'src_1')")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO provenance (memory_id, position, source_id) VALUES ('mem_1', 5, 'no_such_src')")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO memory_conflicts VALUES ('ghost', 'mem_1', 'declared', 'x')")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO memory_events (memory_id, action, recorded_at) VALUES ('ghost', 'add', 'x')")
    assert store.check_integrity() == []


def test_records_cannot_be_deleted_or_rewritten(store):
    store.add(make_memory())
    store.correct("mem_1", make_memory(memory_id="mem_2", value="Beta"), "moved")
    conn = store._conn
    for sql in ("DELETE FROM memories WHERE memory_id = 'mem_1'",
                "UPDATE memories SET value = 'Gamma' WHERE memory_id = 'mem_2'",
                "UPDATE memories SET status = 'active', superseded_by = NULL, status_changed_at = NULL "
                "WHERE memory_id = 'mem_1'",                                      # un-supersede
                "DELETE FROM provenance", "UPDATE provenance SET locator = 'x'",
                "DELETE FROM memory_events", "UPDATE memory_events SET reason = 'x'"):
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(sql)
    with pytest.raises(sqlite3.IntegrityError):                                   # no forked correction chain
        store._insert_record(conn, make_memory(memory_id="mem_3", supersedes="mem_1", correction_reason="x"), None)
    with pytest.raises(sqlite3.IntegrityError):                                   # superseded needs a successor
        conn.execute("UPDATE memories SET status = 'superseded' WHERE memory_id = 'mem_2'")
    assert [r.memory_id for r in store.history("synthetic_p", "persona", "home_city")] == ["mem_1", "mem_2"]


def test_integrity_check_reports_a_record_without_provenance(store):
    with pytest.raises(ValidationError):                 # the contract refuses an empty provenance
        make_memory(provenance=())
    with store._write() as conn:                         # bypass the store API on purpose
        store._insert_record(conn, make_memory(memory_id="bare"), None)
    assert "memory without provenance: bare" in store.check_integrity()


# ---------------------------------------------------------------- 3. provenance round trip

def test_fixture_round_trips_json_to_sqlite_to_json(qiaomo):
    for raw in fixture_raw():
        stored = qiaomo.get(raw["memory_id"])
        assert stored.to_dict() == raw                                   # every field, offsets preserved
        assert MemoryRecord.from_json(stored.to_json()) == stored
    assert qiaomo.check_integrity() == []


def test_sources_round_trip_and_reregistration_is_idempotent(qiaomo):
    for src in fixture_sources():
        assert qiaomo.get_source(src.source_id) == src
        assert qiaomo.register_source(src) == src
    changed = SourceRef(source_id="src_character_sheet", kind=SourceKind.SCRIPT, origin=DataOrigin.SYNTHETIC,
                        description="Something else.")
    with pytest.raises(DuplicateSourceId):
        qiaomo.register_source(changed)


def test_multiple_provenance_entries_keep_order_and_detail(store):
    prov = (Provenance(source_id="src_2", locator="turn:7", quote="我搬家啦", observed_at=datetime(2026, 2, 1, 9, tzinfo=CST)),
            Provenance(source_id="src_1", locator="section:home"),
            Provenance(source_id="src_2", locator="turn:9", quote="新家在北边"))
    rec = make_memory(provenance=prov, valid_from=datetime(2026, 2, 1, tzinfo=CST), confidence=0.7)
    assert store.add(rec) == rec
    rows = store._conn.execute("SELECT position, source_id FROM provenance WHERE memory_id = 'mem_1' ORDER BY position")
    assert [tuple(r) for r in rows] == [(0, "src_2"), (1, "src_1"), (2, "src_2")]
    assert store.get("mem_1").provenance[0].observed_at.utcoffset() == timedelta(hours=8)


def test_search_results_carry_provenance(qiaomo):
    hit = qiaomo.search(QIAOMO, "年糕")[0]
    assert hit.record.provenance == (Provenance(source_id="src_chat_2026_05_02", locator="turn:2", quote="是年糕啦 不是汤圆",
                                                observed_at=datetime(2026, 5, 2, 22, 11, tzinfo=CST)),)
    assert qiaomo.get_source(hit.record.provenance[0].source_id).kind is SourceKind.CONVERSATION


# ---------------------------------------------------------------- 4. temporal boundaries

def _home(store, when):
    return [r.record.value for r in store.search(QIAOMO, "住在", as_of=when, known_at=QIAOMO_NOW)]


def test_as_of_respects_the_half_open_boundary(qiaomo):
    move = datetime(2026, 3, 15, tzinfo=CST)
    assert _home(qiaomo, move - timedelta(microseconds=1)) == ["南屿市"]
    assert _home(qiaomo, move) == ["北川市"]                                   # valid_to is exclusive
    assert _home(qiaomo, move.astimezone(timezone.utc)) == ["北川市"]          # same instant in UTC
    assert _home(qiaomo, datetime(2023, 8, 31, tzinfo=CST)) == []             # before anything was valid
    assert _home(qiaomo, None) == ["北川市"]                                   # default: the store clock


def test_facts_at_valid_time_and_gaps(qiaomo):
    def job(when):
        return [r.value for r in qiaomo.facts_at(QIAOMO, as_of=when, known_at=QIAOMO_NOW, subject="persona",
                                                 attribute="occupation")]
    assert job(datetime(2026, 2, 28, tzinfo=CST)) == ["云朵烘焙 甜点师"]
    assert job(datetime(2026, 3, 10, tzinfo=CST)) == []                       # between jobs
    assert job(datetime(2026, 3, 20, tzinfo=CST)) == ["半拍咖啡 店长"]
    timeline = qiaomo.facts_at(QIAOMO, subject="persona", attribute="home_city")
    assert [r.memory_id for r in timeline] == ["mem_home_nanyu", "mem_home_beichuan"]


def test_valid_time_and_recorded_time_are_separate(qiaomo):
    def cat(known_at):
        return [r.value for r in qiaomo.facts_at(QIAOMO, known_at=known_at, subject="cat", attribute="name")]
    assert cat(datetime(2026, 5, 2, 22, 11, tzinfo=CST)) == []                # not yet recorded
    assert cat(datetime(2026, 5, 2, 22, 13, tzinfo=CST)) == ["汤圆"]          # believed before the correction
    assert cat(datetime(2026, 5, 2, 22, 15, tzinfo=CST)) == ["年糕"]
    assert cat(None) == ["年糕"]
    # the move to Beichuan was valid from 15 March but only recorded on 2 April
    before = qiaomo.facts_at(QIAOMO, as_of=datetime(2026, 3, 20, tzinfo=CST), known_at=datetime(2026, 4, 1, tzinfo=CST),
                             attribute="home_city")
    after = qiaomo.facts_at(QIAOMO, as_of=datetime(2026, 3, 20, tzinfo=CST), known_at=datetime(2026, 4, 3, tzinfo=CST),
                            attribute="home_city")
    assert before == [] and [r.value for r in after] == ["北川市"]


def test_default_retrieval_excludes_superseded_and_retracted_but_history_keeps_them(qiaomo):
    assert qiaomo.search(QIAOMO, "汤圆") == []                                # superseded value is not retrieved
    qiaomo.retract("mem_hobby_climbing", "synthetic note: hobby was invented by mistake")
    assert [h.record.memory_id for h in qiaomo.search(QIAOMO, "爱好")] == ["mem_hobby_baking"]
    assert [r.memory_id for r in qiaomo.history(QIAOMO, "cat", "name")] == ["mem_cat_name_v1", "mem_cat_name_v2"]
    assert [r.status for r in qiaomo.history(QIAOMO, "persona", "hobby")] == [MemoryStatus.ACTIVE, MemoryStatus.RETRACTED]


def test_history_order_is_stable(store):
    for mid, day in (("m_c", 2), ("m_a", 2), ("m_b", 1)):
        store.add(make_memory(memory_id=mid, value=mid, recorded_at=at(day)))
    assert [r.memory_id for r in store.history("synthetic_p", "persona", "home_city")] == ["m_b", "m_a", "m_c"]


def test_naive_as_of_is_rejected(qiaomo):
    with pytest.raises(ValueError):
        qiaomo.search(QIAOMO, "北川", as_of=datetime(2026, 3, 20))


# ---------------------------------------------------------------- 5. conflict semantics

def test_single_overlapping_different_values_conflict_and_are_queryable(store):
    a = store.add(make_memory(memory_id="a", value="Alpha", cardinality=SINGLE, provenance=(Provenance("src_1", "l:1"),)))
    b = store.add(make_memory(memory_id="b", value="Beta", valid_from=at(5), cardinality=SINGLE,
                              provenance=(Provenance("src_2", "turn:3"),)))
    assert a.conflicts_with == () and b.conflicts_with == ("a",)
    assert store.get("a").conflicts_with == ("b",)                            # symmetric
    [c] = store.conflicts("synthetic_p")
    assert (c.memory_a.memory_id, c.memory_b.memory_id, c.origin, c.is_open) == ("a", "b", "detected", True)
    assert c.memory_b.provenance[0].source_id == "src_2"                      # conflict keeps its sources
    assert store.get("a").value == "Alpha"                                    # nothing was replaced


@pytest.mark.parametrize("first,second", [
    (dict(cardinality=SINGLE), dict(cardinality=SINGLE, value="Alpha Town")),               # same value
    (dict(cardinality=SINGLE, valid_to=at(30)), dict(cardinality=SINGLE, valid_from=at(30))),  # temporal change
    (dict(cardinality=MULTI), dict(cardinality=MULTI)),                                      # coexisting values
    (dict(cardinality=UNKNOWN), dict(cardinality=UNKNOWN)),                                  # conservative
    (dict(cardinality=SINGLE), dict(cardinality=UNKNOWN)),
    (dict(cardinality=SINGLE), dict(cardinality=SINGLE, attribute="occupation")),           # another slot
])
def test_no_conflict_cases(store, first, second):
    store.add(make_memory(memory_id="x", **first))
    second = {"value": "Beta", **second}
    added = store.add(make_memory(memory_id="y", statement="Another statement.", **second))
    assert added.conflicts_with == () and store.conflicts("synthetic_p", include_resolved=True) == []


def test_multi_values_coexist_in_retrieval(qiaomo):
    assert {h.record.value for h in qiaomo.search(QIAOMO, "爱好")} == {"烘焙", "攀岩"}
    assert qiaomo.get("mem_hobby_baking").conflicts_with == ()


def test_declared_conflicts_are_kept_and_validated(qiaomo, store):
    [c] = qiaomo.conflicts(QIAOMO)
    assert (c.memory_a.memory_id, c.memory_b.memory_id, c.origin) == ("mem_birthday_nov", "mem_birthday_oct", "declared")
    with pytest.raises(WriteRejected, match="unknown memory"):
        store.add(make_memory(conflicts_with=("nope",)))
    store.add(make_memory(memory_id="other", persona_id="synthetic_other"))
    with pytest.raises(WriteRejected, match="another persona"):
        store.add(make_memory(conflicts_with=("other",)))
    declared = store.add(make_memory(memory_id="d", value="Delta", cardinality=UNKNOWN, conflicts_with=("other",),
                                     persona_id="synthetic_other"))
    assert declared.conflicts_with == ("other",)                              # explicit, even for unknown slots


def test_resolving_a_conflict_by_retraction_or_correction(store):
    store.add(make_memory(memory_id="a", value="Alpha", cardinality=SINGLE))
    store.add(make_memory(memory_id="b", value="Beta", cardinality=SINGLE))
    store.add(make_memory(memory_id="c", value="Gamma", cardinality=SINGLE))
    assert len(store.conflicts("synthetic_p")) == 3
    store.retract("c", "duplicate synthetic note")
    store.correct("b", make_memory(memory_id="b2", value="Alpha", cardinality=SINGLE), "matches a after review")
    assert store.conflicts("synthetic_p") == []                               # a and b2 agree
    assert len(store.conflicts("synthetic_p", include_resolved=True)) == 3    # the history of the conflict stays
    assert store.get("b2").conflicts_with == ()


# ---------------------------------------------------------------- 6. correct and retract semantics

def test_correct_supersedes_inserts_and_logs(store):
    store.add(make_memory(cardinality=SINGLE))
    before = store._conn.execute("SELECT count(*) FROM memories").fetchone()[0]
    new = store.correct("mem_1", make_memory(memory_id="mem_2", value="Beta Town", statement="The persona lives in Beta Town.",
                                             recorded_at=at(3), cardinality=SINGLE), "user corrected the town")
    old = store.get("mem_1")
    assert (old.status, old.superseded_by) == (MemoryStatus.SUPERSEDED, "mem_2")
    assert (new.status, new.supersedes, new.correction_reason) == (MemoryStatus.ACTIVE, "mem_1", "user corrected the town")
    assert store._conn.execute("SELECT count(*) FROM memories").fetchone()[0] == before + 1   # nothing deleted
    assert [h.record.memory_id for h in store.search("synthetic_p", "Town")] == ["mem_2"]
    assert [(e.action, e.related_id) for e in store.events("mem_1")] == [("add", None), ("superseded", "mem_2")]
    assert [(e.action, e.related_id, e.reason) for e in store.events("mem_2")] == [("correct", "mem_1", "user corrected the town")]
    assert store.check_integrity() == []


@pytest.mark.parametrize("replacement,err", [
    (dict(attribute="occupation"), WriteRejected),                 # other slot
    (dict(persona_id="synthetic_other"), WriteRejected),           # other persona
    (dict(recorded_at=at(-1)), WriteRejected),                     # recorded before the original
    (dict(supersedes="mem_x", correction_reason="r"), WriteRejected),
    (dict(correction_reason="a different reason"), WriteRejected),
    (dict(provenance=(Provenance("unregistered"),)), WriteRejected),
    (dict(memory_id="mem_1"), WriteRejected),                       # self link
])
def test_invalid_corrections_change_nothing(store, replacement, err):
    store.add(make_memory())
    snap = snapshot(store)
    with pytest.raises(err):
        store.correct("mem_1", make_memory(**{"memory_id": "mem_2", "value": "Beta", **replacement}), "reason")
    assert snapshot(store) == snap


def test_correct_targets_must_exist_and_be_active(store):
    with pytest.raises(MemoryNotFound):
        store.correct("ghost", make_memory(memory_id="mem_2"), "reason")
    store.add(make_memory())
    with pytest.raises(WriteRejected):
        store.correct("mem_1", make_memory(memory_id="mem_2", value="Beta"), "   ")
    store.retract("mem_1", "withdrawn")
    with pytest.raises(StaleWrite):
        store.correct("mem_1", make_memory(memory_id="mem_2", value="Beta"), "too late")


def test_retract_keeps_reason_source_and_reviewer(store):
    store.add(make_memory())
    rec = store.retract("mem_1", "the synthetic chat was misread", evidence=Provenance("src_2", "turn:12"))
    assert rec.status is MemoryStatus.RETRACTED and store.get("mem_1") == rec
    [_, event] = store.events("mem_1")
    assert (event.action, event.reason, event.source_id, event.locator) == ("retract", "the synthetic chat was misread",
                                                                          "src_2", "turn:12")
    assert store.search("synthetic_p", "Alpha") == []
    assert store.history("synthetic_p", "persona", "home_city") == [rec]


def test_retract_rules(store):
    with pytest.raises(MemoryNotFound):
        store.retract("ghost", "reason")
    store.add(make_memory())
    with pytest.raises(WriteRejected):
        store.retract("mem_1", "")
    with pytest.raises(WriteRejected, match="unregistered"):
        store.retract("mem_1", "reason", evidence=Provenance("nope"))
    store.correct("mem_1", make_memory(memory_id="mem_2", value="Beta"), "moved")
    with pytest.raises(StaleWrite, match="current version"):
        store.retract("mem_1", "reason")


# ---------------------------------------------------------------- write gate

REAL_SRC = SourceRef(source_id="src_real", kind=SourceKind.CONVERSATION, origin=DataOrigin.REAL, description="test-only label")
SIM_SRC = SourceRef(source_id="src_sim", kind=SourceKind.SIMULATION, origin=DataOrigin.SIMULATED, description="test-only label")
REAL_SIM_SRC = SourceRef(source_id="src_real_sim", kind=SourceKind.SIMULATION, origin=DataOrigin.REAL,
                         description="mislabelled generated text")


def real(**kw):
    return make_memory(**{"origin": DataOrigin.REAL, "provenance": (Provenance("src_real", "turn:1"),), **kw})


def simulated(**kw):
    return make_memory(**{"origin": DataOrigin.SIMULATED, "provenance": (Provenance("src_sim", "run:1"),), **kw})


@pytest.fixture
def gated():
    """A store with public_only=False, as the private line would use. Only origin labels differ; content is invented."""
    s = open_store(public_only=False, sources=(REAL_SRC, SIM_SRC, REAL_SIM_SRC,
                                               SourceRef(source_id="src_1", kind=SourceKind.SCRIPT,
                                                         origin=DataOrigin.SYNTHETIC, description="syn")))
    yield s
    s.close()


def test_public_store_accepts_only_synthetic(store):
    with pytest.raises(WriteRejected):
        store.register_source(REAL_SRC)
    with pytest.raises(WriteRejected):
        store.add(make_memory(origin=DataOrigin.REAL))
    with pytest.raises(WriteRejected):
        store.add(make_memory(origin=DataOrigin.SIMULATED))
    with pytest.raises(WriteRejected, match="unregistered"):
        store.add(make_memory(provenance=(Provenance("src_unknown"),)))
    assert store.check_integrity() == [] and store.get("mem_1") is None


def test_real_records_need_review(gated):
    with pytest.raises(WriteRejected, match="review"):
        gated.add(real())
    assert gated.add(real(), reviewer="owner").origin is DataOrigin.REAL
    assert [e.reviewer for e in gated.events("mem_1")] == ["owner"]
    with pytest.raises(WriteRejected, match="review"):
        gated.correct("mem_1", real(memory_id="mem_2", value="Beta"), "fix")
    with pytest.raises(WriteRejected, match="review"):
        gated.retract("mem_1", "withdrawn")
    assert gated.correct("mem_1", real(memory_id="mem_2", value="Beta"), "fix", reviewer="owner").supersedes == "mem_1"


def test_simulated_data_never_overwrites_real_facts(gated):
    gated.add(real(cardinality=SINGLE), reviewer="owner")
    with pytest.raises(WriteRejected, match="never corrects"):
        gated.correct("mem_1", simulated(memory_id="mem_2", value="Beta"), "model says so", reviewer="owner")
    added = gated.add(simulated(memory_id="mem_3", value="Gamma", cardinality=SINGLE))   # allowed, but only flagged
    assert added.conflicts_with == ("mem_1",) and gated.get("mem_1").is_active
    with pytest.raises(WriteRejected, match="simulation source cannot retract"):
        gated.retract("mem_1", "model says so", evidence=Provenance("src_sim"), reviewer="owner")


@pytest.mark.parametrize("record,match", [
    (lambda: make_memory(origin=DataOrigin.SIMULATED, provenance=(Provenance("src_1"),)), "cannot cite"),
    (lambda: make_memory(provenance=(Provenance("src_real"),)), "cannot cite"),
    (lambda: real(provenance=(Provenance("src_real_sim"),)), "simulation source cannot back a real record"),
    (lambda: make_memory(origin=DataOrigin.SIMULATED, provenance=(Provenance("src_real"),)), "cannot cite"),
])
def test_origin_must_match_the_source(gated, record, match):
    with pytest.raises(WriteRejected, match=match):
        gated.add(record(), reviewer="owner")
    assert gated.get("mem_1") is None


def test_add_rejects_store_managed_states(store):
    with pytest.raises(WriteRejected):
        store.add(make_memory(status=MemoryStatus.RETRACTED))
    with pytest.raises(WriteRejected):
        store.add(make_memory(supersedes="x", correction_reason="r"))


# ---------------------------------------------------------------- 11. persona isolation

def test_personas_are_isolated(store):
    for pid in ("synthetic_a", "synthetic_b"):
        store.add(make_memory(memory_id=f"{pid}_home", persona_id=pid, cardinality=SINGLE))
    other = store.add(make_memory(memory_id="b_home2", persona_id="synthetic_b", value="Beta", cardinality=SINGLE))
    assert other.conflicts_with == ("synthetic_b_home",)                      # conflicts stay inside one persona
    assert store.get("synthetic_a_home").conflicts_with == ()
    assert [h.record.memory_id for h in store.search("synthetic_a", "Alpha Town")] == ["synthetic_a_home"]
    assert [r.memory_id for r in store.history("synthetic_a", "persona", "home_city")] == ["synthetic_a_home"]
    assert [r.memory_id for r in store.facts_at("synthetic_a")] == ["synthetic_a_home"]
    assert [c.memory_a.persona_id for c in store.conflicts("synthetic_b")] == ["synthetic_b"]
    assert store.conflicts("synthetic_a") == []
    with pytest.raises(WriteRejected):
        store.correct("synthetic_a_home", make_memory(memory_id="x", persona_id="synthetic_b"), "wrong persona")
    assert store.search("synthetic_c", "Alpha Town") == []


def test_same_content_in_two_personas_is_not_deduplicated(store):
    a = store.add(make_memory(memory_id="a", persona_id="synthetic_a"))
    b = store.add(make_memory(memory_id="b", persona_id="synthetic_b"))
    assert (a.memory_id, b.memory_id) == ("a", "b")
