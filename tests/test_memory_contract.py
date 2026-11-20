import pytest

from conftest import at, make_memory
from stillhere.contracts import Cardinality, MemoryRecord, MemoryStatus, ValidationError

SINGLE, MULTI, UNKNOWN = Cardinality.SINGLE, Cardinality.MULTI, Cardinality.UNKNOWN


def test_round_trip(memory):
    assert MemoryRecord.from_dict(memory.to_dict()) == memory
    assert memory.cardinality is UNKNOWN  # default: no automatic conflicts
    assert memory.slot == ("synthetic_p", "persona", "home_city")
    assert memory.is_active


def test_validity_interval_is_half_open():
    m = make_memory(valid_from=at(10), valid_to=at(20))
    assert not m.is_valid_at(at(9))
    assert m.is_valid_at(at(10))
    assert m.is_valid_at(at(19))
    assert not m.is_valid_at(at(20))
    assert make_memory().is_valid_at(at(-1000))  # unbounded


def test_temporal_change_is_not_a_conflict():
    old = make_memory(memory_id="old", value="Alpha", valid_from=at(0), valid_to=at(30), cardinality=SINGLE)
    new = make_memory(memory_id="new", value="Beta", valid_from=at(30), cardinality=SINGLE)
    assert not old.overlaps(new)
    assert not old.contradicts(new)


def test_single_valued_overlapping_different_values_contradict():
    a = make_memory(memory_id="a", value="Alpha", cardinality=SINGLE)
    b = make_memory(memory_id="b", value="Beta", valid_from=at(5), cardinality=SINGLE)
    assert a.contradicts(b) and b.contradicts(a)
    assert not a.contradicts(a)
    assert not a.contradicts(make_memory(memory_id="c", value="Alpha", cardinality=SINGLE))  # same value
    assert not a.contradicts(make_memory(memory_id="d", attribute="job", value="x", cardinality=SINGLE))  # other slot


def test_multi_valued_slots_never_contradict():
    climbing = make_memory(memory_id="h1", attribute="hobby", value="climbing", cardinality=MULTI)
    baking = make_memory(memory_id="h2", attribute="hobby", value="baking", cardinality=MULTI)
    assert not climbing.contradicts(baking) and not baking.contradicts(climbing)


@pytest.mark.parametrize("left, right", [(UNKNOWN, UNKNOWN), (SINGLE, UNKNOWN), (UNKNOWN, SINGLE), (SINGLE, MULTI),
                                         (MULTI, SINGLE)])
def test_undeclared_or_mismatched_cardinality_is_conservative(left, right):
    a = make_memory(memory_id="a", value="Alpha", cardinality=left)
    b = make_memory(memory_id="b", value="Beta", cardinality=right)
    assert not a.contradicts(b)


def test_single_valued_boundary_cases():
    a = make_memory(memory_id="a", value="Alpha", valid_from=at(0), valid_to=at(10), cardinality=SINGLE)
    touching = make_memory(memory_id="b", value="Beta", valid_from=at(10), cardinality=SINGLE)
    one_day_overlap = make_memory(memory_id="c", value="Beta", valid_from=at(9), cardinality=SINGLE)
    open_ended = make_memory(memory_id="d", value="Beta", cardinality=SINGLE)
    retracted = make_memory(memory_id="e", value="Beta", status=MemoryStatus.RETRACTED, cardinality=SINGLE)
    assert not a.contradicts(touching)        # half-open intervals that only touch
    assert a.contradicts(one_day_overlap)
    assert a.contradicts(open_ended)          # unbounded validity overlaps everything
    assert not a.contradicts(retracted)       # only active records conflict


def test_unknown_cardinality_value_is_rejected():
    with pytest.raises(ValidationError, match="cardinality"):
        MemoryRecord.from_dict({**make_memory().to_dict(), "cardinality": "many"})


def test_old_records_without_cardinality_still_load():
    data = make_memory().to_dict()
    del data["cardinality"]
    assert MemoryRecord.from_dict(data).cardinality is UNKNOWN


def test_superseded_records_do_not_contradict():
    old = make_memory(memory_id="old", value="Alpha", status=MemoryStatus.SUPERSEDED, superseded_by="new",
                      cardinality=SINGLE)
    new = make_memory(memory_id="new", value="Beta", supersedes="old", correction_reason="user corrected it",
                      cardinality=SINGLE)
    assert not old.contradicts(new)


@pytest.mark.parametrize("overrides, message", [
    ({"provenance": ()}, "provenance"),
    ({"valid_from": at(5), "valid_to": at(5)}, "valid_to"),
    ({"confidence": 1.5}, "confidence"),
    ({"supersedes": "mem_0"}, "correction_reason"),
    ({"supersedes": "mem_1", "correction_reason": "x"}, "itself"),
    ({"status": MemoryStatus.SUPERSEDED}, "superseded"),
    ({"superseded_by": "mem_2"}, "superseded"),
    ({"conflicts_with": ("mem_1",)}, "itself"),
    ({"conflicts_with": ("a", "a")}, "duplicate"),
    ({"statement": ""}, "statement"),
    ({"recorded_at": "2026-01-01T00:00:00"}, "timezone"),
])
def test_invalid_memories(overrides, message):
    with pytest.raises(ValidationError, match=message):
        make_memory(**overrides)
