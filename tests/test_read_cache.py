"""Read cache of the SQLite memory store (``_load``) and the ``retrieval_b2._all_records`` memo: results identical to uncached reads, and
never stale after a write, a rollback, a correction, a retraction or another connection's commit; temporal and persona isolation kept."""
import dataclasses
from datetime import timedelta

import pytest

from _ctx_world import world
from stillhere.evaluation import context_dataset_v1 as cd
from stillhere.memory import retrieval_b2 as b2
from stillhere.memory import sqlite_store as SS
from stillhere.memory.api import MemoryAPI


def _uncached(api, persona, known_at):
    return b2._all_records_uncached(api, persona, known_at)


def _ids(rs):
    return [(r.memory_id, r.status, r.superseded_by, tuple(r.conflicts_with)) for r in rs]


def _any_active(api, persona, at):
    return next(r for r in b2._all_records(api, persona, at) if getattr(r.status, "value", r.status) == "active")


def test_cached_reads_equal_uncached_reads():
    api = world().api
    for persona in (cd.CA, cd.CB):
        for at in (cd.t(9, 3), cd.t(9, 10), cd.t(12, 31)):
            a = b2._all_records(api, persona, at)
            b = b2._all_records(api, persona, at)                     # served from the memo
            assert a == b == _uncached(api, persona, at) and a is not b
            assert api.facts_at(persona, as_of=at, known_at=at) == api.facts_at(persona, as_of=at, known_at=at)


def test_a_correction_and_a_retraction_are_seen_immediately():
    api = world().api
    at = cd.t(12, 31)
    before = b2._all_records(api, cd.CA, at)
    r = _any_active(api, cd.CA, at)
    new = dataclasses.replace(r, memory_id=r.memory_id + ".fix", value=(r.value or "") + "（更正）", statement=r.statement + "（更正）",
                              supersedes=None, superseded_by=None, conflicts_with=())
    api.correct(cd.CA, r.memory_id, new, "correction: test")
    after = b2._all_records(api, cd.CA, at)
    assert after != before and after == _uncached(api, cd.CA, at)
    assert any(x.memory_id == new.memory_id for x in after)
    assert getattr(api.get(cd.CA, r.memory_id).status, "value", None) == "superseded"
    other = _any_active(api, cd.CA, at)
    api.retract(cd.CA, other.memory_id, "test retraction")
    assert all(x.memory_id != other.memory_id for x in b2._all_records(api, cd.CA, at)) or \
        getattr(next(x for x in b2._all_records(api, cd.CA, at) if x.memory_id == other.memory_id).status, "value", "") != "active"
    assert b2._all_records(api, cd.CA, at) == _uncached(api, cd.CA, at)


def test_a_rolled_back_write_leaves_no_trace_in_the_cache():
    api = world().api
    at = cd.t(12, 31)
    before = _ids(b2._all_records(api, cd.CA, at))
    r = _any_active(api, cd.CA, at)
    new = dataclasses.replace(r, memory_id=r.memory_id + ".rb", value="回滚", statement="回滚", supersedes=None, superseded_by=None, conflicts_with=())
    with pytest.raises(RuntimeError):
        with api.transaction():
            api.correct(cd.CA, r.memory_id, new, "correction: rolled back")
            inside = b2._all_records(api, cd.CA, at)                    # reads inside the write are never cached
            assert any(x.memory_id == new.memory_id for x in inside)
            raise RuntimeError("abort")
    assert _ids(b2._all_records(api, cd.CA, at)) == before == _ids(_uncached(api, cd.CA, at))
    assert api.get(cd.CA, new.memory_id) is None


def test_known_at_isolation_survives_the_cache():
    api = world().api
    early, late = cd.t(9, 2), cd.t(12, 31)
    a_late = b2._all_records(api, cd.CA, late)
    a_early = b2._all_records(api, cd.CA, early)
    assert all(r.recorded_at <= early for r in a_early) and len(a_early) < len(a_late)
    assert a_early == _uncached(api, cd.CA, early)


def test_persona_isolation_survives_the_cache():
    api = world().api
    at = cd.t(12, 31)
    a = b2._all_records(api, cd.CA, at)
    b = b2._all_records(api, cd.CB, at)
    assert {r.persona_id for r in a} == {cd.CA} and {r.persona_id for r in b} == {cd.CB}
    rid = a[0].memory_id
    api.get(cd.CA, rid)                                                   # now cached
    with pytest.raises(Exception):
        api.get(cd.CB, rid)                                               # the owner check still runs before the cache


def test_another_connection_commit_invalidates(tmp_path):
    db = tmp_path / "m.sqlite"
    src = world().api
    src.snapshot(db)
    a = MemoryAPI(str(db), clock=lambda: cd.t(12, 31, 23))
    at = cd.t(12, 31, 23, 59)                                           # after the outside retraction (stamped 23:00)
    first = b2._all_records(a, cd.CA, at)
    victim = next(r for r in first if getattr(r.status, "value", r.status) == "active")
    other = MemoryAPI(str(db), clock=lambda: cd.t(12, 31, 23))             # a second connection (another process would look the same)
    other.retract(cd.CA, victim.memory_id, "outside retraction")
    other.close()
    again = b2._all_records(a, cd.CA, at)
    assert victim.memory_id not in {r.memory_id for r in again}         # retracted: no longer active / superseded
    assert again == _uncached(a, cd.CA, at)
    a.close()


def test_cached_objects_cannot_be_changed_by_a_caller():
    api = world().api
    at = cd.t(12, 31)
    rs = b2._all_records(api, cd.CA, at)
    rs.clear()                                                            # the caller's list only
    assert b2._all_records(api, cd.CA, at)
    with pytest.raises(dataclasses.FrozenInstanceError):
        b2._all_records(api, cd.CA, at)[0].value = "x"


def test_the_cache_is_bounded():
    api = world().api
    for d in range(1, 20):
        b2._all_records(api, cd.CA, cd.t(9, 1) + timedelta(days=d))
    assert len(b2._ALL_RECORDS_MEMO[api.store]) <= b2.ALL_RECORDS_MEMO_SIZE
    assert len(api.store._load_cache) <= SS.LOAD_CACHE_BUCKETS


def test_concurrent_readers_get_the_uncached_answer():
    import threading
    api = world().api
    ats = [cd.t(12, 31), cd.t(10, 1), cd.t(9, 10)]
    want = {at: _uncached(api, cd.CA, at) for at in ats}
    got, errors = [], []

    def work(at):
        try:
            for _ in range(5):
                got.append((at, b2._all_records(api, cd.CA, at)))
        except Exception as e:                                            # pragma: no cover - reported below
            errors.append(e)
    ts = [threading.Thread(target=work, args=(at,)) for at in ats * 3]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors and all(rs == want[at] for at, rs in got)
