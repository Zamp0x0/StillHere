"""Temporal Query Engine (Day 7 Phase 5B, P0-1): time words -> a TimeExpression with a half-open [start, end)
interval in the calendar's timezone -> SQLite interval lookup -> the memory context the model sees.

Every retrieval case runs on the synthetic persona fixture in a real SQLite store (FTS5 and scan), as of
2026-10-14T00:00Z (2026-10-14 08:00 in the +08:00 calendar). Synthetic data only; no model."""
from datetime import datetime, timedelta, timezone

import pytest

from memory_support import QIAOMO, QIAOMO_NOW, qiaomo_store
from stillhere.contracts import (ChatMessage, FactRole, MemoryQuery, RetrievalStatus, Role, SlotStatus, SubjectStatus,
                                 Sufficiency, TemporalRole, TimeScope, ValidationError)
from stillhere.contracts.planning import TimeExpression, TimeKind
from stillhere.data.templates import QWEN3_SITUATION_NOTHINK
from stillhere.inference.context import build_context
from stillhere.memory import QueryPlanner, SlotRetriever
from stillhere.memory.temporal import CALENDAR_TZ, find_times, resolve_time

CST = timezone(timedelta(hours=8))
OK, NO_MATCH = RetrievalStatus.OK, RetrievalStatus.NO_MATCH


def d(y, m, day, h=0, minute=0):
    return datetime(y, m, day, h, minute, tzinfo=CST)


@pytest.fixture(params=[None, False], ids=["fts5", "scan"])
def store(request):
    s = qiaomo_store(fts=request.param)
    yield s
    s.close()


@pytest.fixture
def retriever(store):
    return SlotRetriever(store, persona_names={QIAOMO: ("乔沫",)}, clock=lambda: QIAOMO_NOW)


@pytest.fixture
def ask(retriever):
    return lambda text, **kw: retriever.retrieve(QIAOMO, text, **kw)


def ids(result):
    return [h.record.memory_id for h in result.hits]


def only_slot(result):
    [slot] = result.slots
    return slot


def context_of(result, text):
    return build_context([ChatMessage(role=Role.USER, bubbles=(text,))], QWEN3_SITUATION_NOTHINK, "small_talk", result)


# ---------------------------------------------------------------- the parser: structured intervals, not strings

def test_calendar_timezone_is_fixed_plus_eight():
    assert CALENDAR_TZ.utcoffset(None) == timedelta(hours=8)


@pytest.mark.parametrize("text,kind,start,end,granularity,status", [
    ("2026年3月10号", TimeKind.ABSOLUTE_POINT, d(2026, 3, 10), d(2026, 3, 11), "day", "resolved"),
    ("2026年3月10日", TimeKind.ABSOLUTE_POINT, d(2026, 3, 10), d(2026, 3, 11), "day", "resolved"),
    ("二〇二五年", TimeKind.ABSOLUTE_INTERVAL, d(2025, 1, 1), d(2026, 1, 1), "year", "resolved"),
    ("2025年", TimeKind.ABSOLUTE_INTERVAL, d(2025, 1, 1), d(2026, 1, 1), "year", "resolved"),
    ("2026年3月", TimeKind.ABSOLUTE_INTERVAL, d(2026, 3, 1), d(2026, 4, 1), "month", "resolved"),
    ("3月10号", TimeKind.ABSOLUTE_POINT, d(2026, 3, 10), d(2026, 3, 11), "day", "inferred_year"),
    ("三月十号", TimeKind.ABSOLUTE_POINT, d(2026, 3, 10), d(2026, 3, 11), "day", "inferred_year"),
    ("3月初", TimeKind.ABSOLUTE_INTERVAL, d(2026, 3, 1), d(2026, 3, 11), "part_of_month", "inferred_year"),
    ("3月中旬", TimeKind.ABSOLUTE_INTERVAL, d(2026, 3, 11), d(2026, 3, 21), "part_of_month", "inferred_year"),
    ("3月底", TimeKind.ABSOLUTE_INTERVAL, d(2026, 3, 21), d(2026, 4, 1), "part_of_month", "inferred_year"),
    ("去年", TimeKind.RELATIVE_INTERVAL, d(2025, 1, 1), d(2026, 1, 1), "year", "resolved"),
    ("前年", TimeKind.RELATIVE_INTERVAL, d(2024, 1, 1), d(2025, 1, 1), "year", "resolved"),
    ("今年初", TimeKind.RELATIVE_INTERVAL, d(2026, 1, 1), d(2026, 3, 1), "part_of_year", "resolved"),
    ("今年三月", TimeKind.RELATIVE_INTERVAL, d(2026, 3, 1), d(2026, 4, 1), "month", "resolved"),
    ("今年三月初", TimeKind.RELATIVE_INTERVAL, d(2026, 3, 1), d(2026, 3, 11), "part_of_month", "resolved"),
    ("上个月", TimeKind.RELATIVE_INTERVAL, d(2026, 9, 1), d(2026, 10, 1), "month", "resolved"),
    ("上周", TimeKind.RELATIVE_INTERVAL, d(2026, 10, 5), d(2026, 10, 12), "week", "resolved"),   # Mon..Mon
    ("上上周", TimeKind.RELATIVE_INTERVAL, d(2026, 9, 28), d(2026, 10, 5), "week", "resolved"),
    ("上上个月", TimeKind.RELATIVE_INTERVAL, d(2026, 8, 1), d(2026, 9, 1), "month", "resolved"),   # not 上个月
    ("大前年", TimeKind.RELATIVE_INTERVAL, d(2023, 1, 1), d(2024, 1, 1), "year", "resolved"),       # not 前年
    ("2026-03-10", TimeKind.ABSOLUTE_POINT, d(2026, 3, 10), d(2026, 3, 11), "day", "resolved"),
    ("2026/3/14", TimeKind.ABSOLUTE_POINT, d(2026, 3, 14), d(2026, 3, 15), "day", "resolved"),
    ("2026.3", TimeKind.ABSOLUTE_INTERVAL, d(2026, 3, 1), d(2026, 4, 1), "month", "resolved"),
    ("三年前", TimeKind.RELATIVE_INTERVAL, d(2023, 1, 1), d(2024, 1, 1), "year", "resolved"),      # its whole year
    ("两个月前", TimeKind.RELATIVE_INTERVAL, d(2026, 8, 1), d(2026, 9, 1), "month", "resolved"),
    ("两周前", TimeKind.RELATIVE_INTERVAL, d(2026, 9, 28), d(2026, 10, 5), "week", "resolved"),
    ("三天前", TimeKind.RELATIVE_INTERVAL, d(2026, 10, 11), d(2026, 10, 12), "day", "resolved"),
])
def test_expressions_become_half_open_local_intervals(text, kind, start, end, granularity, status):
    t = resolve_time(text, QIAOMO_NOW)
    assert isinstance(t, TimeExpression)
    assert (t.kind, t.start, t.end, t.granularity, t.resolution_status) == (kind, start, end, granularity, status)
    assert (t.timezone, t.source_text, t.as_of) == ("+08:00", text, QIAOMO_NOW)
    assert t.start.utcoffset() == timedelta(hours=8)          # a local calendar day, not a UTC midnight


def test_this_year_is_cut_at_as_of():
    t = resolve_time("今年", QIAOMO_NOW)
    assert (t.kind, t.start, t.end) == (TimeKind.RELATIVE_INTERVAL, d(2026, 1, 1), QIAOMO_NOW)


@pytest.mark.parametrize("text,kind", [
    ("现在", TimeKind.CURRENT), ("目前", TimeKind.CURRENT), ("以前", TimeKind.HISTORICAL), ("之前", TimeKind.HISTORICAL),
    ("过去", TimeKind.HISTORICAL), ("一直", TimeKind.ALL), ("小时候", TimeKind.LIFE_PERIOD),
])
def test_open_scopes_and_life_periods_have_no_calendar_interval(text, kind):
    t = resolve_time(text, QIAOMO_NOW)
    assert (t.kind, t.start, t.end) == (kind, None, None)


@pytest.mark.parametrize("text,status", [
    ("2026年2月30号", "invalid"), ("2026年13月", "invalid"), ("3月32号", "invalid"), ("2026-02-30", "invalid"),
    ("零年前", "invalid"), ("前阵子", "unresolved"), ("前段时间", "unresolved"),
])
def test_invalid_or_vague_times_are_unresolved_never_current(text, status):
    t = resolve_time(text, QIAOMO_NOW)
    assert (t.kind, t.resolution_status, t.start, t.end) == (TimeKind.UNRESOLVED_TIME, status, None, None)


def test_a_relative_time_without_as_of_is_unresolved():
    t = resolve_time("去年", None)
    assert (t.kind, t.resolution_status) == (TimeKind.UNRESOLVED_TIME, "unresolved")


def test_a_date_after_as_of_is_not_placed():
    t = resolve_time("2026年12月1号", QIAOMO_NOW)
    assert (t.kind, t.resolution_status) == (TimeKind.UNRESOLVED_TIME, "future")


def test_the_missing_year_is_read_in_the_local_calendar():
    """2026-03-14T17:00Z is already 2026-03-15 01:00 in +08:00: 3月15号 is this year's, not last year's."""
    late = datetime(2026, 3, 14, 17, 0, tzinfo=timezone.utc)
    t = resolve_time("3月15号", late)
    assert (t.start, t.end, t.resolution_status) == (d(2026, 3, 15), late, "inferred_year")   # cut at as_of
    early = datetime(2026, 3, 14, 15, 0, tzinfo=timezone.utc)        # 2026-03-14 23:00 local: not yet the 15th
    assert resolve_time("3月15号", early).start == d(2025, 3, 15)


def test_find_times_reports_spans_and_skips_values_after_the_slot():
    text = "你生日是十月还是十一月"
    assert [m.expression.kind for m in find_times(text, QIAOMO_NOW) if m.scope_for(text.find("生日"))] == []
    text = "你今年三月初住在哪个城市"
    [m] = find_times(text, QIAOMO_NOW)
    assert (text[m.start:m.end], m.expression.kind) == ("今年三月初", TimeKind.RELATIVE_INTERVAL)


def test_time_expression_contract_rejects_bad_intervals():
    with pytest.raises(ValidationError):
        TimeExpression(kind=TimeKind.ABSOLUTE_POINT, source_text="x", as_of=QIAOMO_NOW, timezone="+08:00",
                       granularity="day", resolution_status="resolved", start=d(2026, 3, 11), end=d(2026, 3, 10))
    with pytest.raises(ValidationError):
        TimeExpression(kind=TimeKind.UNRESOLVED_TIME, source_text="x", as_of=QIAOMO_NOW, timezone="+08:00",
                       granularity="unknown", resolution_status="invalid", start=d(2026, 3, 10), end=d(2026, 3, 11))
    with pytest.raises(ValidationError):
        MemoryQuery(persona_id=QIAOMO, text="你住哪", time_scope=TimeScope.AT)     # AT comes from the words


# ---------------------------------------------------------------- planner: time spans before owners

def plan(text, **kw):
    q = MemoryQuery(persona_id=QIAOMO, text=text, as_of=QIAOMO_NOW, **kw)
    return QueryPlanner(persona_names={QIAOMO: ("乔沫",)}).plan(q)


def test_t10_owner_is_the_persona_not_unowned():
    [a] = plan("你今年三月初住在哪个城市").atoms
    assert (a.subject_status, a.subject, a.attributes, a.time_scope) == (
        SubjectStatus.RESOLVED, "persona", ("home_city",), TimeScope.AT)
    assert (a.time.start, a.time.end, a.retrievable) == (d(2026, 3, 1), d(2026, 3, 11), True)


def test_t11_date_is_read_not_defaulted_to_current():
    [a] = plan("2026年3月10号那会儿你在哪上班").atoms
    assert (a.subject, a.attributes, a.time_scope, a.time_scope_source) == ("persona", ("occupation",), TimeScope.AT,
                                                                          "query")
    assert (a.time.kind, a.time.start, a.time.end) == (TimeKind.ABSOLUTE_POINT, d(2026, 3, 10), d(2026, 3, 11))


def test_masking_keeps_kinship_owners_unowned():
    [a] = plan("你妈妈去年住哪").atoms
    assert a.subject_status is SubjectStatus.UNOWNED and not a.retrievable


def test_joined_owners_with_a_time_get_one_atom_each():
    p = plan("你和我去年分别住哪")
    assert [(a.subject, a.time_scope) for a in p.atoms] == [("persona", TimeScope.AT), ("user", TimeScope.AT)]


def test_a_dated_atom_is_not_merged_with_an_undated_one():
    p = plan("你住哪？去年呢？")
    assert [(a.subject, a.attributes, a.time_scope) for a in p.atoms] == [
        ("persona", ("home_city",), TimeScope.CURRENT), ("persona", ("home_city",), TimeScope.AT)]


def test_the_next_part_inherits_the_time():
    p = plan("你去年住哪，做什么工作")
    assert [(a.attributes, a.time_scope, a.time.start if a.time else None) for a in p.atoms] == [
        (("home_city",), TimeScope.AT, d(2025, 1, 1)), (("occupation",), TimeScope.AT, d(2025, 1, 1))]


def test_unresolved_time_is_not_retrievable_and_says_why():
    [a] = plan("你2026年2月30号住哪").atoms
    assert (a.time_scope, a.retrievable, a.ambiguity) == (TimeScope.UNRESOLVED, False, "time_unresolved")


def test_plans_without_dates_keep_their_ids():
    """Old messages plan exactly as before: no time field, so the plan id (a hash of the atoms) is unchanged."""
    p = plan("你住哪？我住哪？")
    assert all(a.time is None and "time" not in a.to_dict() for a in p.atoms)


# ---------------------------------------------------------------- SQLite interval lookups

def test_store_interval_lookup_is_half_open(store):
    def during(start, end, attribute):
        return [r.memory_id for r in store.facts_during(QIAOMO, start=start, end=end, attribute=attribute,
                                                        known_at=QIAOMO_NOW)]
    assert during(d(2026, 3, 14), d(2026, 3, 15), "home_city") == ["mem_home_nanyu"]
    assert during(d(2026, 3, 15), d(2026, 3, 16), "home_city") == ["mem_home_beichuan"]       # valid_from inclusive
    assert during(d(2026, 3, 1), d(2026, 3, 2), "occupation") == []                            # valid_to exclusive
    assert during(d(2026, 3, 19), d(2026, 3, 20), "occupation") == []
    assert during(d(2026, 3, 20), d(2026, 3, 21), "occupation") == ["mem_job_cafe"]
    assert during(d(2026, 2, 28), d(2026, 3, 1), "occupation") == ["mem_job_bakery"]
    assert during(d(2026, 1, 1), d(2027, 1, 1), "home_city") == ["mem_home_nanyu", "mem_home_beichuan"]
    assert during(d(2025, 1, 1), d(2026, 1, 1), "name") == ["mem_cat_name_v2"]   # timeless, active only
    with pytest.raises(ValueError):
        store.facts_during(QIAOMO, start=d(2026, 3, 2), end=d(2026, 3, 1))


def test_retrieval_uses_the_sql_interval_lookup(store):
    calls = []

    class Spy:
        facts_at, history = store.facts_at, store.history

        def facts_during(self, *a, **kw):
            calls.append(kw)
            return store.facts_during(*a, **kw)

    res = SlotRetriever(Spy(), persona_names={QIAOMO: ("乔沫",)}, clock=lambda: QIAOMO_NOW).retrieve(
        QIAOMO, "2026年3月14号你住在哪")
    assert ids(res) == ["mem_home_nanyu"]
    assert calls and calls[0]["start"] == d(2026, 3, 14) and calls[0]["end"] == d(2026, 3, 15)


# ---------------------------------------------------------------- the required boundary cases, end to end

@pytest.mark.parametrize("text,expected,value", [
    ("2026年3月14号你住在哪", ["mem_home_nanyu"], "南屿市"),
    ("2026年3月15号你住在哪", ["mem_home_beichuan"], "北川市"),
    ("2026年2月28号你在哪上班", ["mem_job_bakery"], "云朵烘焙 甜点师"),
    ("2026年3月20号你在哪上班", ["mem_job_cafe"], "半拍咖啡 店长"),
    ("你2025年住在哪", ["mem_home_nanyu"], "南屿市"),
    ("你今年三月初住在哪个城市", ["mem_home_nanyu"], "南屿市"),
    ("你去年住在哪个城市", ["mem_home_nanyu"], "南屿市"),                 # t04
    ("你前年在哪上班", ["mem_job_bakery"], "云朵烘焙 甜点师"),
    ("乔沫上个月住哪", ["mem_home_beichuan"], "北川市"),
    ("你上周住在哪", ["mem_home_beichuan"], "北川市"),
    ("你今年初住在哪", ["mem_home_nanyu"], "南屿市"),
    ("2026/3/14你住在哪", ["mem_home_nanyu"], "南屿市"),
    ("2026-03-15你住在哪", ["mem_home_beichuan"], "北川市"),
    ("你上上个月住在哪", ["mem_home_beichuan"], "北川市"),          # 2026-08
    ("你两年前住在哪", ["mem_home_nanyu"], "南屿市"),               # 2024
])
def test_the_value_valid_at_the_asked_time(ask, text, expected, value):
    res = ask(text)
    slot = only_slot(res)
    assert (res.status, res.time_scope, ids(res)) == (OK, TimeScope.AT, expected)
    assert (slot.status, slot.value) == (SlotStatus.AT_TIME, value)
    assert all(h.role is TemporalRole.AT_TIME for h in res.hits)
    built = context_of(res, text)
    assert built.memory_view.time is not None and [m.record.memory_id for m in built.memory_context] == expected
    assert [f.role for s in built.memory_view.slots for f in s.facts] == [FactRole.AT_TIME]
    system = built.prompt.text.split("<|im_end|>")[0]
    assert "time asked: " + res.time.start.isoformat() in system


@pytest.mark.parametrize("text", [
    "2026年3月1号你在哪上班", "2026年3月10号那会儿你在哪上班", "2026年3月19号你在哪上班", "3月10号你在哪上班",
    "2026-03-10你在哪上班",
])
def test_a_time_gap_returns_no_value_and_no_unemployment(ask, text):
    res = ask(text)
    assert (res.status, res.time_scope, ids(res), res.evidence) == (NO_MATCH, TimeScope.AT, [], "time_gap")
    built = context_of(res, text)
    assert built.memory_context == () and built.memory_view.evidence == "time_gap"
    system = built.prompt.text.split("<|im_end|>")[0]
    assert "半拍咖啡" not in system and "云朵烘焙" not in system
    assert "nothing recorded for the time asked" in system and "not being unemployed" in system


def test_a_whole_year_with_a_move_is_segmented_not_picked(ask):
    res = ask("你2026年住在哪")
    slot = only_slot(res)
    assert (slot.status, slot.value, ids(res)) == (SlotStatus.SEGMENTED, None, ["mem_home_nanyu", "mem_home_beichuan"])
    system = context_of(res, "你2026年住在哪").prompt.text.split("<|im_end|>")[0]
    assert "the value changed during the time asked" in system


def test_part_of_a_month_without_a_record_is_partial_time(ask):
    res = ask("你2026年3月在哪上班")              # 云朵 ended 03-01, 半拍 from 03-20: 03-01..03-20 is not recorded
    slot = only_slot(res)
    assert (slot.status, slot.value, ids(res)) == (SlotStatus.PARTIAL_TIME, None, ["mem_job_cafe"])
    system = context_of(res, "你2026年3月在哪上班").prompt.text.split("<|im_end|>")[0]
    assert "cover only part of the time asked" in system


@pytest.mark.parametrize("text", ["你大前年在哪上班", "你三年前在哪上班"])
def test_a_year_ago_read_as_its_whole_year_is_partial_when_records_start_inside_it(ask, text):
    res = ask(text)                               # 2023: 云朵 from 09-01 only, January-August not recorded
    slot = only_slot(res)
    assert (res.time.start, res.time.end) == (d(2023, 1, 1), d(2024, 1, 1))
    assert (slot.status, slot.value, ids(res)) == (SlotStatus.PARTIAL_TIME, None, ["mem_job_bakery"])


def test_always_lived_in_beichuan_shows_the_earlier_city(ask):
    res = ask("你一直住北川吗")
    assert (res.time_scope, ids(res)) == (TimeScope.ALL, ["mem_home_nanyu", "mem_home_beichuan"])


def test_childhood_stays_unknown(ask):
    res = ask("你小时候住在哪里")
    assert (res.status, ids(res), res.evidence) == (NO_MATCH, [], "life_period")
    system = context_of(res, "你小时候住在哪里").prompt.text.split("<|im_end|>")[0]
    assert "北川" not in system and "南屿" not in system and "do not say it did not exist" in system


@pytest.mark.parametrize("text", ["你2026年2月30号住哪", "你前阵子住在哪"])
def test_an_unplaced_time_never_gets_the_current_value(ask, text):
    res = ask(text)
    assert (res.status, ids(res), res.time_scope, res.evidence) == (NO_MATCH, [], TimeScope.UNRESOLVED, "time_unresolved")
    system = context_of(res, text).prompt.text.split("<|im_end|>")[0]
    assert "北川" not in system and "could not be placed on the calendar" in system


def test_a_conflict_stays_a_conflict_at_a_time(ask):
    res = ask("你去年生日是几月")
    slot = only_slot(res)
    assert (slot.status, slot.value) == (SlotStatus.CONFLICT, None)
    assert sorted(ids(res)) == ["mem_birthday_nov", "mem_birthday_oct"]
    built = context_of(res, "你去年生日是几月")
    assert {f.role for s in built.memory_view.slots for f in s.facts} == {FactRole.UNVERIFIED}


def test_a_corrected_name_is_never_the_name_at_an_earlier_time(ask):
    res = ask("你家猫去年叫什么")
    slot = only_slot(res)
    assert (res.time_scope, slot.status, slot.value, ids(res)) == (TimeScope.AT, SlotStatus.AT_TIME, "年糕",
                                                                  ["mem_cat_name_v2"])
    assert [r.memory_id for r in slot.corrected] == ["mem_cat_name_v1"]


def test_several_owners_with_a_time(ask):
    res = ask("你去年住哪？我去年住哪？")
    q1, q2 = res.batch.atoms
    assert (q1.memory_ids, q1.sufficiency) == (("mem_home_nanyu",), Sufficiency.SUFFICIENT)
    assert (q2.subject, q2.memory_ids, q2.sufficiency) == ("user", (), Sufficiency.NO_RECORD)
    built = context_of(res, "你去年住哪？我去年住哪？")
    assert [q.time.start for q in built.memory_view.queries] == [d(2025, 1, 1), d(2025, 1, 1)]
    system = built.prompt.text.split("<|im_end|>")[0]
    assert "query q1: persona/home_city, at 2025-01-01" in system


def test_several_atoms_with_one_time(ask):
    res = ask("你去年住哪，做什么工作")
    assert ids(res) == ["mem_home_nanyu", "mem_job_bakery"]
    assert res.time is not None and res.time.start == d(2025, 1, 1)


def test_a_time_gap_next_to_an_answered_query(ask):
    res = ask("你现在住哪？2026年3月10号你在哪上班？")
    q1, q2 = res.batch.atoms
    assert (q1.memory_ids, q2.memory_ids, q2.missing, q2.sufficiency) == (
        ("mem_home_beichuan",), (), "time_gap", Sufficiency.TIME_UNKNOWN)
    system = context_of(res, "你现在住哪？2026年3月10号你在哪上班？").prompt.text.split("<|im_end|>")[0]
    assert "半拍咖啡" not in system and "do not conclude there was nothing then" in system
