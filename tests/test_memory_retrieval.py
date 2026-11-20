"""SlotRetriever: temporal resolution on the synthetic persona (real fixture ids), retrieval scope, ranking,
limits, ambiguous subjects, safe failures and read-only access. Synthetic data only; no model, no embeddings."""
import inspect
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from conftest import at, make_memory
from memory_support import QIAOMO, QIAOMO_NOW, fixture_records, open_store, qiaomo_store, snapshot
from stillhere.contracts import (BatchStatus, Cardinality, MemoryHit, MemoryQuery, MemoryRetriever, MemoryStatus,
                                 Provenance, RetrievalResult, RetrievalStatus, RetrievedMemory, SlotResolution,
                                 SlotStatus, SubjectStatus, Sufficiency, TemporalRole, TimeScope, ValidationError)
from stillhere.memory import SlotRetriever, SQLiteMemoryStore

CST = timezone(timedelta(hours=8))
MOVE = datetime(2026, 3, 15, tzinfo=CST)          # mem_home_nanyu ends, mem_home_beichuan starts
OK, NO_MATCH, AMBIGUOUS = RetrievalStatus.OK, RetrievalStatus.NO_MATCH, RetrievalStatus.AMBIGUOUS_SUBJECT
ROOT = Path(__file__).resolve().parents[1]


def retriever(store, **kw):
    return SlotRetriever(store, persona_names={QIAOMO: ("乔沫",)}, clock=lambda: QIAOMO_NOW, **kw)


@pytest.fixture(params=[None, False], ids=["fts5", "scan"])
def qiaomo(request):
    s = qiaomo_store(fts=request.param)
    yield s
    s.close()


@pytest.fixture
def ask(qiaomo):
    r = retriever(qiaomo)
    return lambda text, **kw: r.retrieve(QIAOMO, text, **kw)


def ids(result):
    return [h.record.memory_id for h in result.hits]


def only_slot(result) -> SlotResolution:
    [slot] = result.slots
    return slot


# ---------------------------------------------------------------- the six fixture cases

def test_case1_current_residence(ask):
    res = ask("乔沫现在住在哪里？")
    slot = only_slot(res)
    assert res.status is OK and (res.time_scope, res.time_scope_source) == (TimeScope.CURRENT, "query")
    assert (res.subject, res.subject_source) == ("persona", "query")
    assert (slot.subject, slot.attribute, slot.status, slot.value) == ("persona", "home_city", SlotStatus.RESOLVED, "北川市")
    assert ids(res) == ["mem_home_beichuan"]
    hit = res.hits[0]
    assert (hit.role, hit.via, hit.source_ids) == (TemporalRole.CURRENT, "slot", ("src_chat_2026_04_02",))
    assert hit.record.provenance[0].quote is not None and "住" in hit.matched
    assert [r.memory_id for r in slot.past] == ["mem_home_nanyu"]          # history is there, not in the hits


def test_case2_historical_residence(ask):
    res = ask("乔沫以前住哪")
    slot = only_slot(res)
    assert (res.time_scope, slot.status, slot.value) == (TimeScope.HISTORICAL, SlotStatus.HISTORICAL, None)
    assert ids(res) == ["mem_home_nanyu", "mem_home_beichuan"]
    assert [h.role for h in res.hits] == [TemporalRole.PAST, TemporalRole.CURRENT]
    nanyu = res.hits[0].record
    assert (nanyu.value, nanyu.valid_to, nanyu.provenance[0].source_id) == ("南屿市", MOVE, "src_chat_2026_02_14")


def test_case3_cat_name_after_correction(ask):
    res = ask("乔沫的猫现在叫什么")
    slot = only_slot(res)
    assert (res.subject, slot.attribute, slot.status, slot.value) == ("cat", "name", SlotStatus.RESOLVED, "年糕")
    assert ids(res) == ["mem_cat_name_v2"]
    assert res.hits[0].record.supersedes == "mem_cat_name_v1"
    assert [r.memory_id for r in slot.corrected] == ["mem_cat_name_v1"]


def test_case4_old_cat_name_is_a_correction_not_a_former_name(ask):
    res = ask("乔沫的猫以前叫什么")
    slot = only_slot(res)
    assert res.time_scope is TimeScope.HISTORICAL
    assert slot.status is SlotStatus.UNKNOWN and slot.past == ()            # the cat never had another name
    [old] = slot.corrected
    assert (old.memory_id, old.value, old.status, old.superseded_by) == \
        ("mem_cat_name_v1", "汤圆", MemoryStatus.SUPERSEDED, "mem_cat_name_v2")
    assert ids(res) == ["mem_cat_name_v2"]                                  # 汤圆 is never offered as memory
    assert res.hits[0].record.correction_reason
    assert "mem_cat_name_v1" not in [m.record.memory_id for m in res.to_memory_context()]


def test_case5_birthday_conflict_is_never_verified(ask):
    res = ask("乔沫生日是几月")
    slot = only_slot(res)
    assert (slot.status, slot.value) == (SlotStatus.CONFLICT, None) and res.has_conflict
    assert slot.conflict_ids == ("mem_birthday_nov", "mem_birthday_oct")
    assert ids(res) == ["mem_birthday_nov", "mem_birthday_oct"]            # both sides, by id, not by recording time
    assert {h.record.confidence for h in res.hits} == {0.5}
    assert ids(ask("乔沫生日是几月", limit=1)) == ids(res) and not ask("乔沫生日", limit=1).truncated
    one_side = ask("十月")                                                  # a keyword hit brings its partner along
    assert ids(one_side) == ["mem_birthday_oct", "mem_birthday_nov"] and one_side.hits[1].matched == ()
    assert only_slot(one_side).status is SlotStatus.CONFLICT


@pytest.mark.parametrize("text", ["乔沫的血型是什么", "乔沫喜欢什么颜色", "你好呀", "乔沫"])
def test_case6_unknown_is_no_match(ask, qiaomo, text):
    res = ask(text)
    assert res.status is NO_MATCH and res.hits == ()
    if text == "乔沫的血型是什么":
        assert len(qiaomo.search(QIAOMO, text, limit=11)) == 8                        # keyword search alone recalls noise


def test_unknown_slot_and_unknown_subject(ask):
    mine = ask("我住在哪个城市")                                            # the user's city, not the persona's
    assert (mine.status, mine.subject, only_slot(mine).status) == (NO_MATCH, "user", SlotStatus.UNKNOWN)
    assert ask("你叫什么名字").status is NO_MATCH                           # the persona's own name is not a memory
    assert retriever(qiaomo_store()).retrieve("synthetic_nobody", "住哪").status is NO_MATCH


# ---------------------------------------------------------------- time rules

@pytest.mark.parametrize("as_of,expected", [
    (MOVE - timedelta(microseconds=1), "mem_home_nanyu"),
    (MOVE, "mem_home_beichuan"),                                            # valid_to is exclusive
    (MOVE.astimezone(timezone.utc), "mem_home_beichuan"),                   # the same instant in UTC
    ("2026-03-14T16:00:00Z", "mem_home_beichuan"),
    ("2026-03-15T00:00:00+08:00", "mem_home_beichuan"),
    ("2026-03-14T23:59:59.999999+08:00", "mem_home_nanyu"),
])
def test_half_open_boundaries_and_timezones(ask, as_of, expected):
    assert all(ids(ask("乔沫住哪", as_of=as_of, known_at=QIAOMO_NOW)) == [expected] for _ in range(3))


def test_before_anything_and_between_jobs(ask):
    early = ask("乔沫住哪", as_of=datetime(2023, 8, 31, tzinfo=CST))
    assert early.status is NO_MATCH and only_slot(early).status is SlotStatus.UNKNOWN
    gap = ask("乔沫做什么工作", as_of=datetime(2026, 3, 10, tzinfo=CST))
    slot = only_slot(gap)
    assert gap.status is NO_MATCH and slot.status is SlotStatus.UNKNOWN
    assert [r.memory_id for r in slot.past] == ["mem_job_bakery"]          # the earlier job is still visible
    past_job = ask("乔沫以前做什么工作", as_of=datetime(2026, 4, 1, tzinfo=CST), known_at=QIAOMO_NOW)
    assert ids(past_job) == ["mem_job_bakery", "mem_job_cafe"]


def test_historical_question_asked_in_the_past(ask):
    res = ask("乔沫以前住哪", as_of=datetime(2026, 1, 1, tzinfo=CST), known_at=QIAOMO_NOW)
    assert only_slot(res).status is SlotStatus.UNKNOWN and ids(res) == ["mem_home_nanyu"]


def test_valid_time_not_recording_time_decides(ask):
    # mem_home_beichuan was recorded on 2 April but holds from 15 March
    assert ids(ask("乔沫住哪", as_of=datetime(2026, 3, 20, tzinfo=CST), known_at=QIAOMO_NOW)) == ["mem_home_beichuan"]


def test_latest_recorded_does_not_win():
    store = open_store()
    store.add(make_memory(memory_id="m_now", value="Alpha Town", cardinality="single", valid_from=at(0),
                          recorded_at=at(1)))
    # recorded later, but about an earlier period: it must not replace the current value
    store.add(make_memory(memory_id="m_old", value="Beta Town", statement="The persona lived in Beta Town.",
                          cardinality="single", valid_from=at(-400), valid_to=at(0), recorded_at=at(200)))
    res = SlotRetriever(store, clock=lambda: at(300)).retrieve("synthetic_p", "住", subject="persona")
    slot = only_slot(res)
    assert (slot.status, slot.value, ids(res)) == (SlotStatus.RESOLVED, "Alpha Town", ["m_now"])
    assert [r.memory_id for r in slot.past] == ["m_old"]


def test_detected_conflict_without_declaration():
    store = open_store()
    store.add(make_memory(memory_id="m_a", value="Alpha Town", cardinality="single"))
    store.add(make_memory(memory_id="m_b", value="Beta Town", statement="The persona lives in Beta Town.",
                          cardinality="single", recorded_at=at(5)))
    res = SlotRetriever(store, clock=lambda: at(10)).retrieve("synthetic_p", "住哪", subject="persona")
    assert (only_slot(res).status, only_slot(res).value, ids(res)) == (SlotStatus.CONFLICT, None, ["m_a", "m_b"])


def test_multi_valued_and_several_attributes(ask):
    hobbies = ask("乔沫有什么爱好")
    assert (only_slot(hobbies).status, only_slot(hobbies).cardinality) == (SlotStatus.MULTI_VALUE, Cardinality.MULTI)
    assert ids(hobbies) == ["mem_hobby_baking", "mem_hobby_climbing"]
    both = ask("乔沫住哪，做什么工作")
    assert [(s.attribute, s.value) for s in both.slots] == [("home_city", "北川市"), ("occupation", "半拍咖啡 店长")]
    assert ids(both) == ["mem_home_beichuan", "mem_job_cafe"]


@pytest.mark.parametrize("text,scope,source", [
    ("乔沫住哪", TimeScope.CURRENT, "default"),
    ("乔沫目前住哪", TimeScope.CURRENT, "query"),
    ("乔沫之前住哪", TimeScope.HISTORICAL, "query"),
    ("乔沫以前和现在分别住哪", TimeScope.ALL, "query"),
])
def test_time_scope_from_wording(ask, text, scope, source):
    res = ask(text)
    assert (res.time_scope, res.time_scope_source) == (scope, source)


def test_explicit_time_scope_wins(ask):
    res = ask("乔沫现在住哪", time_scope="historical")
    assert (res.time_scope, res.time_scope_source, ids(res)) == \
        (TimeScope.HISTORICAL, "explicit", ["mem_home_nanyu", "mem_home_beichuan"])


# Day 7 regression tests for time words, written after the findings they cover: not an independent evaluation.

@pytest.mark.parametrize("text,attribute", [
    ("你小时候住哪？", "home_city"), ("乔沫童年住在哪里", "home_city"), ("你上大学的时候住哪", "home_city"),
    ("你高中时住哪", "home_city"), ("你小时候喜欢攀岩吗", "hobby"), ("你小时候的猫叫什么", "name"),
])
def test_a_life_period_no_record_places_is_unknown(ask, text, attribute):
    """No record says when the persona was a child or a student, so none answers for that time: 北川市 is not the
    childhood home, and 汤圆 is not the childhood cat."""
    res = ask(text)
    assert (res.status, res.hits, res.time_scope) == (NO_MATCH, (), TimeScope.HISTORICAL)
    assert [(s.attribute, s.status, s.value) for s in res.slots] == [(attribute, SlotStatus.UNKNOWN, None)]
    assert res.to_memory_context() == ()
    assert ask(text, time_scope="current").hits == ()                      # an explicit scope does not place it
    assert ask(text, as_of=datetime(2024, 1, 1, tzinfo=CST)).hits == ()
    # Day 7 V2: the childhood part and the current part are separate atoms. Phase 3.5 refused both (whole-text
    # reading); now the current part is answered and the childhood part still gets nothing.
    mixed = ask("你小时候住哪？你现在住哪？")
    period, now = mixed.batch.atoms
    assert (period.sufficiency, period.memory_ids, period.missing) == (Sufficiency.TIME_UNKNOWN, (), "life_period")
    assert (now.memory_ids, ids(mixed), [h.role for h in mixed.hits]) == (
        ("mem_home_beichuan",), ["mem_home_beichuan"], [TemporalRole.CURRENT])     # no past home for the childhood


@pytest.mark.parametrize("text", ["你一直住北川吗？", "乔沫一直住哪", "你从小就住北川吗", "你始终住在北川吗"])
def test_a_stretch_of_time_brings_what_held_before(ask, text):
    """一直 asks about the whole stretch: the current home alone would let 是的 pass for an answer."""
    res = ask(text)
    slot = only_slot(res)
    assert (res.status, res.time_scope, res.time_scope_source) == (OK, TimeScope.ALL, "query")
    assert (slot.status, slot.value) == (SlotStatus.RESOLVED, "北川市")
    assert ids(res) == ["mem_home_nanyu", "mem_home_beichuan"]
    assert [h.role for h in res.hits] == [TemporalRole.PAST, TemporalRole.CURRENT]


def test_a_stretch_of_time_keeps_corrections_and_the_asked_time(ask):
    cat = ask("你家猫一直叫年糕吗")
    assert (cat.time_scope, ids(cat), only_slot(cat).past) == (TimeScope.ALL, ["mem_cat_name_v2"], ())
    before = ask("你一直住北川吗？", as_of=datetime(2026, 1, 1, tzinfo=CST), known_at=QIAOMO_NOW)
    assert (only_slot(before).value, ids(before)) == ("南屿市", ["mem_home_nanyu"])   # 北川市 not yet valid


@pytest.mark.parametrize("text,expected,value", [
    ("你去年住哪", ["mem_home_nanyu"], "南屿市"),                      # 2025
    ("乔沫上个月住哪", ["mem_home_beichuan"], "北川市"),               # 2026-09
    ("你前年在哪上班", ["mem_job_bakery"], "云朵烘焙 甜点师"),          # 2024
])
def test_a_calendar_past_is_an_interval(ask, text, expected, value):
    """Phase 5B: these words name a calendar interval and only the value valid then is returned. Until Phase 5B
    they read as the open past (both records, scope historical), which left the date arithmetic to the model
    (Phase 5 t04: '我去年住在北川市')."""
    res = ask(text)
    assert (res.time_scope, only_slot(res).status, only_slot(res).value) == (TimeScope.AT, SlotStatus.AT_TIME, value)
    assert ids(res) == expected and all(h.role is TemporalRole.AT_TIME for h in res.hits)


@pytest.mark.parametrize("text,expected", [
    ("你住哪？我小时候住南屿。", ["mem_home_beichuan"]),
    ("你现在住哪？我小时候住南屿。", ["mem_home_beichuan"]),
    ("你家猫叫什么？我儿子一直问。", ["mem_cat_name_v2"]),
])
def test_time_words_in_a_remark_change_nothing(ask, text, expected):
    res = ask(text)
    assert (res.status, res.time_scope, ids(res)) == (OK, TimeScope.CURRENT, expected)
    in_context = ask("你住哪", context=["我小时候住南屿"], context_roles=["user"])
    assert (in_context.time_scope, ids(in_context)) == (TimeScope.CURRENT, ["mem_home_beichuan"])


@pytest.mark.parametrize("text,scope,status,value", [
    ("你家猫咪以前叫什么？", TimeScope.HISTORICAL, SlotStatus.UNKNOWN, None),
    ("你的猫之前叫什么名字", TimeScope.HISTORICAL, SlotStatus.UNKNOWN, None),
    ("你家猫去年叫什么", TimeScope.AT, SlotStatus.AT_TIME, "年糕"),     # Phase 5B: 2025's name is the corrected one
])
def test_a_corrected_name_is_never_a_former_name(ask, text, scope, status, value):
    res = ask(text)
    slot = only_slot(res)
    assert (res.time_scope, slot.status, slot.value, slot.past) == (scope, status, value, ())
    assert [r.memory_id for r in slot.corrected] == ["mem_cat_name_v1"] and ids(res) == ["mem_cat_name_v2"]


@pytest.mark.parametrize("as_of,value,expected", [
    (datetime(2026, 3, 1, tzinfo=CST) - timedelta(microseconds=1), "云朵烘焙 甜点师", ["mem_job_bakery"]),
    (datetime(2026, 3, 1, tzinfo=CST), None, []),                           # valid_to is exclusive
    (datetime(2026, 3, 20, tzinfo=CST) - timedelta(microseconds=1), None, []),
    (datetime(2026, 3, 20, tzinfo=CST), "半拍咖啡 店长", ["mem_job_cafe"]),   # valid_from is inclusive
])
def test_job_boundaries_are_half_open(ask, as_of, value, expected):
    res = ask("乔沫做什么工作", as_of=as_of, known_at=QIAOMO_NOW)
    assert (only_slot(res).value, ids(res)) == (value, expected)


@pytest.mark.parametrize("kwargs,message", [
    (dict(as_of=datetime(2026, 3, 20)), "timezone-aware"),                  # naive
    (dict(as_of="2026-03-15"), "timezone-aware"),                            # a date has no timezone
    (dict(as_of="明天"), "ISO-8601"),
    (dict(as_of="2026-13-01T00:00:00+08:00"), "ISO-8601"),
    (dict(as_of=20260315), "ISO-8601"),
    (dict(time_scope="someday"), "time_scope"),
    (dict(limit=0), "limit"),
    (dict(limit=51), "limit"),
    (dict(limit=True), "integer"),
    (dict(subject=" "), "subject"),
    (dict(context="乔沫"), "list"),
])
def test_invalid_requests_return_a_status_not_an_exception(ask, kwargs, message):
    res = ask("乔沫住哪", **kwargs)
    assert res.status is RetrievalStatus.INVALID_REQUEST and res.hits == () and res.query is None
    assert message in res.errors[0]


def test_invalid_persona_and_text():
    r = retriever(qiaomo_store())
    assert r.retrieve("bad id!", "住哪").status is RetrievalStatus.INVALID_REQUEST
    assert r.retrieve(QIAOMO, None).status is RetrievalStatus.INVALID_REQUEST
    assert r.retrieve(QIAOMO, "").status is NO_MATCH and r.retrieve(QIAOMO, "   ").status is NO_MATCH


# ---------------------------------------------------------------- subjects

def test_ambiguous_pronoun_returns_nothing(ask):
    res = ask("她住哪")
    assert (res.status, res.hits, res.subject_candidates) == (AMBIGUOUS, (), ("persona",))
    assert ask("她").subject_candidates == ("persona",)                     # 她 is not the cat
    assert ask("它叫什么").subject_candidates == ("cat",)                   # 它 is not a person
    assert ask("他们住哪").status is AMBIGUOUS
    assert ask("她的猫叫什么").status is AMBIGUOUS                          # whose cat is not known


def test_context_identifies_the_pronoun(ask):
    res = ask("她住哪", context=["上次说到乔沫搬家了"])
    assert (res.status, res.subject, res.subject_source, ids(res)) == (OK, "persona", "context", ["mem_home_beichuan"])
    cat = ask("它现在叫什么", context=["乔沫说起她的猫", "你家猫最近怎么样"])
    assert (cat.subject, cat.subject_source, ids(cat)) == ("cat", "context", ["mem_cat_name_v2"])
    owned = ask("她的猫叫什么", context=["乔沫最近好吗"])
    assert (owned.subject, ids(owned)) == ("cat", ["mem_cat_name_v2"])
    assert ask("它叫什么", context=["猫和狗都很可爱"]).status is AMBIGUOUS  # the newest turn names two
    assert ask("它叫什么", context=["乔沫好吗"]).status is AMBIGUOUS          # a person is not 它


def test_slot_with_one_owner_and_explicit_subject(ask):
    res = ask("叫什么名字")
    assert (res.subject, res.subject_source, ids(res)) == ("cat", "slot", ["mem_cat_name_v2"])
    assert ask("叫什么", subject="persona").status is NO_MATCH
    scoped = ask("香菜", subject="cat")
    assert scoped.status is NO_MATCH                                        # keyword hits respect the subject


def test_slot_with_several_owners_is_ambiguous():
    store = qiaomo_store()
    store.add(make_memory(memory_id="mem_dog_name", persona_id=QIAOMO, subject="dog", attribute="name", value="豆包",
                          statement="乔沫邻居的狗叫豆包。", cardinality="single",
                          provenance=(Provenance("src_character_sheet", "section:pets"),), recorded_at=at(200)))
    r = retriever(store)
    res = r.retrieve(QIAOMO, "叫什么名字")
    assert (res.status, res.subject_candidates) == (AMBIGUOUS, ("cat", "dog"))
    assert ids(r.retrieve(QIAOMO, "狗叫什么")) == ["mem_dog_name"]


def test_personas_are_isolated():
    store = qiaomo_store()
    store.add(make_memory(memory_id="other_home", persona_id="synthetic_other", value="北川市",
                          statement="另一个人住在北川市。", cardinality="single",
                          provenance=(Provenance("src_character_sheet", "section:other"),)))
    r = SlotRetriever(store, clock=lambda: QIAOMO_NOW)
    assert ids(r.retrieve("synthetic_other", "住哪", subject="persona")) == ["other_home"]
    assert ids(r.retrieve(QIAOMO, "住哪", subject="persona")) == ["mem_home_beichuan"]
    assert ids(r.retrieve(QIAOMO, "北川", subject="persona")) == ["mem_home_beichuan", "mem_job_cafe"]
    assert all(h.record.persona_id == QIAOMO for h in r.retrieve(QIAOMO, "北川市").hits)


# ---------------------------------------------------------------- queries of several parts

def slots(result):
    return [(s.subject, s.attribute, s.status) for s in result.slots]


# ChatSession joins a turn's bubbles with a space; these two are the Day 6 development cases dev_e003 and dev_e004.
CAT_TURN = ("你家那只猫叫什么来着", "上次你说过我又忘了")
BIRTHDAY_TURN = ("你生日到底是十月还是十一月啊", "我想提前准备礼物")


def test_a_remark_bubble_does_not_take_over_the_subject(ask):
    cat = ask(" ".join(CAT_TURN))
    assert (cat.status, cat.subject, cat.subject_source) == (OK, "cat", "query")
    assert slots(cat) == [("cat", "name", SlotStatus.RESOLVED)] and ids(cat) == ["mem_cat_name_v2"]
    assert "mem_cat_name_v1" not in ids(cat)                                # the corrected name stays out
    birthday = ask(" ".join(BIRTHDAY_TURN))
    assert (birthday.status, birthday.subject) == (OK, "persona")
    assert slots(birthday) == [("persona", "birthday_month", SlotStatus.CONFLICT)]
    assert sorted(ids(birthday)) == ["mem_birthday_nov", "mem_birthday_oct"]   # both sides of the conflict
    for turn in (CAT_TURN, BIRTHDAY_TURN):                                  # same as asking the question alone
        joined, alone = ask(" ".join(turn)), ask(turn[0])
        assert (joined.subject, slots(joined), ids(joined)) == (alone.subject, slots(alone), ids(alone))


@pytest.mark.parametrize("text,subject,slot,expected", [
    ("你家猫叫什么？我忘了名字。", "cat", ("cat", "name"), ["mem_cat_name_v2"]),
    ("你生日哪天？我想买生日礼物。", "persona", ("persona", "birthday_month"), ["mem_birthday_nov", "mem_birthday_oct"]),
    ("我问你，你在哪个城市", "persona", ("persona", "home_city"), ["mem_home_beichuan"]),
    ("我好奇，你的爱好是什么", "persona", ("persona", "hobby"), ["mem_hobby_baking", "mem_hobby_climbing"]),
    ("你家那只猫叫什么来着 之前你说过我又忘了", "cat", ("cat", "name"), ["mem_cat_name_v2"]),
    ("你家猫叫什么？我以前也养过猫。", "cat", ("cat", "name"), ["mem_cat_name_v2"]),   # 我 and 以前 are in the remark
    ("你生日哪天？我想买礼物。", "persona", ("persona", "birthday_month"), ["mem_birthday_nov", "mem_birthday_oct"]),
    ("我几乎忘了你住哪里。", "persona", ("persona", "home_city"), ["mem_home_beichuan"]),   # 几乎 asks nothing
    ("我想知道你生日是几月", "persona", ("persona", "birthday_month"), ["mem_birthday_nov", "mem_birthday_oct"]),
    ("你跟我说过你的猫叫什么", "cat", ("cat", "name"), ["mem_cat_name_v2"]),             # 我 owns nothing here
    ("你住哪？我同事想知道。", "persona", ("persona", "home_city"), ["mem_home_beichuan"]),  # a request that asks no fact
    ("你家猫叫什么来着我忘了名字", "cat", ("cat", "name"), ["mem_cat_name_v2"]),           # 我 owns no 名字 here
])
def test_the_question_part_decides_and_a_remark_adds_no_slot_or_time(ask, text, subject, slot, expected):
    res = ask(text)
    assert (res.status, res.subject, res.subject_source) == (OK, subject, "query")
    assert [(s.subject, s.attribute) for s in res.slots] == [slot]          # 名字/生日 in the remark add nothing
    assert res.time_scope is TimeScope.CURRENT                              # 之前 in the remark does not count
    assert sorted(ids(res)) == expected


@pytest.mark.parametrize("text", ["我住在哪个城市", "我生日是几月", "你还记得我生日吗", "你知道我家猫叫什么吗？",
                                  "我家猫叫什么", "我家猫叫什么？我忘了名字"])
def test_questions_about_the_user_stay_with_the_user(ask, text):
    res = ask(text)
    assert (res.status, res.subject, res.hits) == (NO_MATCH, "user", ())


def test_a_part_about_an_unknown_person_fails_closed(ask):
    res = ask("她住哪？我忘了")                                              # 我忘了 asks nothing; 她 is not known
    assert (res.status, res.subject, res.slots, res.hits) == (AMBIGUOUS, None, (), ())


# Day 7 V2: parts about different subjects are separate atoms. Phase 3 refused the whole message
# (ambiguous_subject); now each atom is answered from its own owner's records only, or gets nothing.
@pytest.mark.parametrize("text,answered,other", [
    ("你家猫叫什么？我家猫叫什么？", ("cat", ["mem_cat_name_v2"]), ("user", Sufficiency.NO_RECORD)),
    ("你住哪？我呢？", ("persona", ["mem_home_beichuan"]), ("user", Sufficiency.NO_RECORD)),
    ("告诉我我家猫的名字。你家猫叫什么？", ("cat", ["mem_cat_name_v2"]), ("user", Sufficiency.NO_RECORD)),
])
def test_parts_about_different_subjects_are_answered_separately(ask, text, answered, other):
    res = ask(text)
    assert res.status is OK and res.subject is None and res.batch.status is BatchStatus.PARTIAL_SUCCESS
    good = next(a for a in res.batch.atoms if a.memory_ids)
    bad = next(a for a in res.batch.atoms if not a.memory_ids)
    assert (good.subject, list(good.memory_ids)) == answered and ids(res) == answered[1]
    assert (bad.subject, bad.sufficiency, bad.status) == (other[0], other[1], NO_MATCH)
    assert all(h.record.subject == answered[0] for h in res.hits)          # nothing is the user's


@pytest.mark.parametrize("text,context", [
    ("猫叫什么来着？我忘了", ()), ("生日你是几月？我想准备礼物", ()), ("那它现在叫什么呢？我忘了", ("我家猫叫汤圆",)),
])
def test_unclear_owners_are_not_given_the_personas_memories(ask, text, context):
    """No owner named before the slot (or a pronoun that context without speakers gives to the user): the whole
    text decides as before, and the persona's memories stay out."""
    res = ask(text, context=list(context))
    assert (res.status, res.subject, res.hits) == (NO_MATCH, "user", ())


def test_several_parts_keep_the_single_part_behaviour(ask):
    both = ask("你现在住哪 以前呢")
    assert (both.subject, both.time_scope) == ("persona", TimeScope.ALL)
    assert ids(both) == ["mem_home_nanyu", "mem_home_beichuan"]
    two = ask("你住哪？做什么工作？")
    assert slots(two) == [("persona", "home_city", SlotStatus.RESOLVED), ("persona", "occupation", SlotStatus.RESOLVED)]
    assert (ask("乔沫，你住哪").subject, ids(ask("乔沫，你住哪"))) == ("persona", ["mem_home_beichuan"])
    assert ask("你家猫叫什么？我忘了名字", subject="persona").status is NO_MATCH     # an explicit subject still wins


# ---------------------------------------------------------------- who owns the asked fact

RELATION_QUERIES = ["你妈妈住哪", "你男朋友做什么工作", "你爸爸住哪里", "你妹妹几岁", "你同事住哪", "你朋友叫什么",
                    "你家那位住哪里", "我妈住哪", "乔沫的妈妈住哪", "你妈妈喜欢香菜吗", "你家人住哪",
                    "我问你，你妈妈住哪", "我问你，你男朋友做什么工作", "我问你，你家那位住哪", "你爸爸几月生日",
                    "你堂哥喜欢攀岩吗", "你那个远房表舅不吃什么"]


@pytest.mark.parametrize("text", RELATION_QUERIES)
def test_someone_elses_facts_are_not_answered_from_the_persona(ask, text):
    """A relative, partner, friend or colleague owns the asked fact. With no record for that person the query
    fails closed: no subject, no slot, no hit, never the persona's or the user's own value."""
    res = ask(text)
    assert (res.status, res.subject, res.subject_source, res.slots, res.hits) == (NO_MATCH, None, None, (), ())


@pytest.mark.parametrize("text,expected", [
    ("你住哪", ["mem_home_beichuan"]), ("乔沫住哪", ["mem_home_beichuan"]), ("乔沫，你住哪", ["mem_home_beichuan"]),
    ("你做什么工作", ["mem_job_cafe"]), ("你在哪里上班", ["mem_job_cafe"]), ("你现在在哪个城市", ["mem_home_beichuan"]),
    ("你以前住哪", ["mem_home_nanyu", "mem_home_beichuan"]), ("你家猫叫什么", ["mem_cat_name_v2"]),
    ("你几月生日", ["mem_birthday_nov", "mem_birthday_oct"]),
])
def test_the_personas_own_facts_still_resolve(ask, text, expected):
    res = ask(text)
    assert res.status is OK and res.subject_source == "query" and ids(res) == expected


def test_a_relation_with_its_own_records_is_its_own_subject():
    """A deployment that stores a relative's facts under its own subject names it in subject_nouns; the query
    then reaches that subject only, and the persona and the user keep theirs."""
    store = qiaomo_store()
    store.add(make_memory(memory_id="mem_mother_home", persona_id=QIAOMO, subject="mother", attribute="home_city",
                          value="南屿市", statement="乔沫的妈妈住在南屿市。", cardinality="single",
                          provenance=(Provenance("src_character_sheet", "section:family"),), recorded_at=at(200)))
    r = retriever(store, subject_nouns={"猫": "cat", "妈妈": "mother"})
    for text in ("你妈妈住哪", "乔沫的妈妈住哪"):
        res = r.retrieve(QIAOMO, text)
        assert (res.status, res.subject, ids(res)) == (OK, "mother", ["mem_mother_home"])
    assert (r.retrieve(QIAOMO, "你住哪").subject, ids(r.retrieve(QIAOMO, "你住哪"))) == ("persona", ["mem_home_beichuan"])
    mine = r.retrieve(QIAOMO, "我妈妈住哪")
    assert (mine.status, mine.subject, mine.hits) == (NO_MATCH, "user", ())
    assert r.retrieve(QIAOMO, "你爸爸住哪").status is NO_MATCH                    # no subject for 爸爸: still closed
    both = r.retrieve(QIAOMO, "你住哪，你妈妈住哪")                               # V2: two atoms, two owners
    assert both.batch.status is BatchStatus.COMPLETE_SUCCESS and both.subject is None
    assert [(a.subject, a.memory_ids) for a in both.batch.atoms] == [
        ("persona", ("mem_home_beichuan",)), ("mother", ("mem_mother_home",))]


@pytest.mark.parametrize("text", ["我们家猫叫什么", "我们都住哪里", "我们的生日是哪天", "咱们住哪", "你们家猫叫什么",
                                  "你俩住哪", "咱家猫叫什么"])
def test_shared_owners_are_ambiguous(ask, text):
    res = ask(text)
    assert (res.status, res.subject, res.slots, res.hits) == (AMBIGUOUS, None, (), ())


# Day 7 V2: two named owners joined by a conjunction are one atom each (Phase 3: ambiguous_subject).
@pytest.mark.parametrize("text,persona_ids", [
    ("你和我分别住哪里", ("mem_home_beichuan",)), ("我跟你的生日是几月", ("mem_birthday_nov", "mem_birthday_oct")),
])
def test_joined_owners_are_one_atom_each(ask, text, persona_ids):
    res = ask(text)
    owners = {a.subject: a for a in res.batch.atoms}
    assert set(owners) == {"persona", "user"} and res.batch.status is BatchStatus.PARTIAL_SUCCESS
    assert tuple(sorted(owners["persona"].memory_ids)) == persona_ids
    assert (owners["user"].memory_ids, owners["user"].sufficiency) == ((), Sufficiency.NO_RECORD)


def test_a_plural_word_without_an_asked_fact_changes_nothing(ask):
    assert (ask("我家猫叫什么").status, ask("我家猫叫什么").subject) == (NO_MATCH, "user")
    assert ids(ask("你家猫叫什么")) == ["mem_cat_name_v2"]
    remark = ask("我们组今天整个被裁了 我还在工位上发呆")
    assert remark.hits == () and remark.status is not AMBIGUOUS


# Day 7 V2: parts (or conjoined clauses) with different owners are separate atoms (Phase 3: ambiguous_subject).
# The persona's records answer only the persona's atom; a relative's or the user's atom gets nothing.
@pytest.mark.parametrize("text,answered,other", [
    ("你住哪，你妈妈住哪", ("persona", ["mem_home_beichuan"]), (SubjectStatus.UNOWNED, "unowned:妈妈")),
    ("你生日是几月？你妈妈生日是几月？", ("persona", ["mem_birthday_nov", "mem_birthday_oct"]),
     (SubjectStatus.UNOWNED, "unowned:妈妈")),
    ("你家猫叫什么？我家猫叫什么？", ("cat", ["mem_cat_name_v2"]), (SubjectStatus.RESOLVED, "no_record")),
    ("我想问，你生日和你妈妈生日分别是几月", ("persona", ["mem_birthday_nov", "mem_birthday_oct"]),
     (SubjectStatus.UNOWNED, "unowned:妈妈")),
    ("你生日和你妈妈生日分别是几月", ("persona", ["mem_birthday_nov", "mem_birthday_oct"]),
     (SubjectStatus.UNOWNED, "unowned:妈妈")),
    ("你生日和我生日分别是几月", ("persona", ["mem_birthday_nov", "mem_birthday_oct"]),
     (SubjectStatus.RESOLVED, "no_record")),
    ("乔沫生日和她妈妈生日是几月", ("persona", ["mem_birthday_nov", "mem_birthday_oct"]),
     (SubjectStatus.UNOWNED, "unowned:妈妈")),
])
def test_parts_with_different_owners_are_separate_atoms(ask, text, answered, other):
    res = ask(text)
    assert res.status is OK and res.batch.status is BatchStatus.PARTIAL_SUCCESS
    [good] = [a for a in res.batch.atoms if a.memory_ids]
    [bad] = [a for a in res.batch.atoms if not a.memory_ids]
    assert (good.subject, sorted(good.memory_ids), sorted(ids(res))) == (answered[0], answered[1], answered[1])
    assert (res.batch.plan.atom(bad.query_id).subject_status, bad.missing) == other


@pytest.mark.parametrize("context,roles,expected", [
    (["我家猫叫汤圆"], ["user"], (NO_MATCH, "user", [])),                     # the user's own cat
    (["我家猫叫年糕"], ["persona"], (OK, "cat", ["mem_cat_name_v2"])),        # the persona said 我家猫
    (["你家猫叫什么来着"], ["user"], (OK, "cat", ["mem_cat_name_v2"])),        # the user asked about 你家猫
    (["你家猫叫什么呀"], ["persona"], (NO_MATCH, "user", [])),                 # the persona asked about the user's
    (["我家猫叫汤圆"], [], (NO_MATCH, "user", [])),                            # no speakers: read as the user's
    (["猫很可爱"], ["user"], (AMBIGUOUS, None, [])),                            # nobody owns this cat
    (["她说“我家猫叫年糕”"], ["user"], (AMBIGUOUS, None, [])),                 # a quoted 我 is not the speaker
])
def test_context_owners_follow_who_said_the_turn(ask, context, roles, expected):
    res = ask("那它现在叫什么", context=context, context_roles=roles)
    assert (res.status, res.subject, ids(res)) == expected
    if res.subject is not None:
        assert res.subject_source == "context"


@pytest.mark.parametrize("context,roles,expected", [
    (["上次说到乔沫搬家了"], ["user"], (OK, "persona")),
    (["上次说到乔沫的妈妈"], ["user"], (AMBIGUOUS, None)),                    # 她 may be the mother: unknown owner
    (["我妈妈最近搬家了"], ["persona"], (AMBIGUOUS, None)),
])
def test_a_context_person_is_only_the_persona_when_named_alone(ask, context, roles, expected):
    res = ask("她住哪", context=context, context_roles=roles)
    assert (res.status, res.subject) == expected
    if res.subject is None:
        assert res.hits == ()


def test_context_roles_are_part_of_the_query_contract():
    q = MemoryQuery(persona_id=QIAOMO, text="那它叫什么", context=("我家猫叫年糕",), context_roles=("persona",))
    assert MemoryQuery.from_dict(q.to_dict()) == q
    assert MemoryQuery.from_dict({k: v for k, v in q.to_dict().items() if k != "context_roles"}).context_roles == ()
    for bad in (dict(context=("a", "b"), context_roles=("user",)), dict(context=("a",), context_roles=("system",))):
        with pytest.raises(ValidationError):
            MemoryQuery(persona_id=QIAOMO, text="x", **bad)
    res = retriever(qiaomo_store()).retrieve(QIAOMO, "那它叫什么", context=["我家猫叫年糕"], context_roles=["persona"])
    assert res.query.context_roles == ("persona",)
    bad = retriever(qiaomo_store()).retrieve(QIAOMO, "x", context=["a"], context_roles=["user", "persona"])
    assert (bad.status, bad.hits) == (RetrievalStatus.INVALID_REQUEST, ())


@pytest.mark.parametrize("parts", [("她住哪？", "乔沫的爱好是什么？"), ("乔沫的爱好是什么？", "她住哪？")])
def test_the_same_subject_from_different_sources_is_chosen_the_same_way(ask, parts):
    """One part names the persona through the context (她), the other in the query (乔沫). A part that needs the
    context sends the query back to whole-text reading, so either order gives persona / query, every time."""
    results = [ask("".join(parts), context=["上次说到乔沫搬家了"]) for _ in range(3)]
    assert {(r.subject, r.subject_source) for r in results} == {("persona", "query")}
    assert all(r.to_dict() == results[0].to_dict() for r in results)


def test_subject_choice_does_not_depend_on_hash_order():
    code = ("import sys; sys.path.insert(0, 'tests'); from memory_support import QIAOMO, QIAOMO_NOW, qiaomo_store; "
            "from stillhere.memory import SlotRetriever; "
            "r = SlotRetriever(qiaomo_store(), persona_names={QIAOMO: ('乔沫',)}, clock=lambda: QIAOMO_NOW); "
            "texts = ['你家那只猫叫什么来着 上次你说过我又忘了', '你生日到底是十月还是十一月啊 我想提前准备礼物', "
            "'你家猫叫什么？我家猫叫什么？', '你住哪？做什么工作？', '乔沫，你住哪', '你妈妈住哪', "
            "'我们家猫叫什么', '你家猫叫什么？我以前也养过猫。', '我想知道你生日是几月']; "
            "print([r.retrieve(QIAOMO, t).to_json() for t in texts] + "
            "[r.retrieve(QIAOMO, '她住哪？乔沫的爱好是什么？', context=['上次说到乔沫搬家了']).to_json()] + "
            "[r.retrieve(QIAOMO, '那它叫什么', context=[c], context_roles=[s]).to_json() "
            "for c, s in (('我家猫叫汤圆', 'user'), ('我家猫叫年糕', 'persona'))])")
    outs = {subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True,
                           env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONIOENCODING": "utf-8"}).stdout for seed in ("0", "1", "12345")}
    assert len(outs) == 1


# ---------------------------------------------------------------- ranking, limits, evidence

def test_ordering_is_deterministic(ask):
    first = ask("乔沫以前住哪，做什么工作")
    assert ids(first) == ["mem_home_nanyu", "mem_home_beichuan", "mem_job_bakery", "mem_job_cafe"]
    assert all(ask("乔沫以前住哪，做什么工作").to_dict() == first.to_dict() for _ in range(3))


def test_fts_and_scan_stores_give_the_same_result():
    a, b = (retriever(qiaomo_store(fts=f)) for f in (None, False))
    for text in ("乔沫现在住在哪里", "乔沫以前住哪", "十月", "攀岩", "香菜", "乔沫的血型是什么"):
        assert a.retrieve(QIAOMO, text).to_dict() == b.retrieve(QIAOMO, text).to_dict()


def test_limit_cuts_whole_groups(ask):
    res = ask("乔沫以前住哪，做什么工作", limit=2)
    # Day 7 V2: two atoms; each keeps its first group even above the limit, so no atom loses all its evidence
    # (Phase 3.5 kept only the first group of the whole message: [nanyu, beichuan]).
    assert ids(res) == ["mem_home_nanyu", "mem_home_beichuan", "mem_job_bakery"] and res.truncated
    assert len(res.slots) == 2                                              # the slots stay complete
    assert ids(ask("乔沫以前住哪", limit=1)) == ["mem_home_nanyu"]           # one atom: the Day 6 rule
    assert ids(ask("乔沫有什么爱好", limit=1)) == ["mem_hobby_baking", "mem_hobby_climbing"]


def test_keyword_evidence_and_provenance(ask, qiaomo):
    res = ask("她吃香菜吗", context=["乔沫来了"])
    [hit] = res.hits
    assert (hit.record.memory_id, hit.via) == ("mem_pref_coriander", "keyword") and "香菜" in hit.matched
    assert hit.record.provenance == qiaomo.get("mem_pref_coriander").provenance
    assert only_slot(res).status is SlotStatus.RESOLVED


def test_memory_context_feeds_an_inference_request(ask):
    res = ask("乔沫以前住哪")
    ctx = res.to_memory_context()
    assert all(isinstance(m, RetrievedMemory) and m.retriever == "slot_temporal_v0" for m in ctx)
    assert [m.record.memory_id for m in ctx] == ids(res)


# ---------------------------------------------------------------- safety

def test_retrieval_never_writes():
    store = qiaomo_store()
    before = snapshot(store)
    r = retriever(store)
    for text in ("乔沫现在住在哪里", "乔沫的猫以前叫什么", "乔沫生日", "她住哪", "十月", "乔沫的血型"):
        r.retrieve(QIAOMO, text)
    r.retrieve(QIAOMO, "住哪", as_of="garbage")
    assert snapshot(store) == before
    assert not any(hasattr(r, name) for name in ("add", "correct", "retract", "import_records", "_conn"))


def test_store_failure_is_an_error_status():
    store = qiaomo_store()
    r = retriever(store)
    store.close()
    res = r.retrieve(QIAOMO, "乔沫住哪")
    assert res.status is RetrievalStatus.ERROR and res.hits == () and "ProgrammingError" in res.errors[0]


def test_correction_through_the_store_is_seen_and_history_kept(tmp_path):
    path = tmp_path / "memory.sqlite3"
    store = qiaomo_store(path)
    store.correct("mem_cat_name_v2", make_memory(
        memory_id="mem_cat_name_v3", persona_id=QIAOMO, subject="cat", attribute="name", value="团子",
        statement="乔沫的猫叫团子。", cardinality="single", recorded_at=datetime(2026, 6, 1, tzinfo=CST),
        provenance=(Provenance("src_chat_2026_05_02", "turn:9"),)), "synthetic: the cat was renamed")
    before = retriever(store).retrieve(QIAOMO, "乔沫的猫叫什么").to_dict()
    store.close()
    reopened = SQLiteMemoryStore(path, clock=lambda: QIAOMO_NOW)
    res = retriever(reopened).retrieve(QIAOMO, "乔沫的猫叫什么")
    assert res.to_dict() == before
    slot = only_slot(res)
    assert (slot.value, ids(res)) == ("团子", ["mem_cat_name_v3"])
    assert [r.memory_id for r in slot.corrected] == ["mem_cat_name_v1", "mem_cat_name_v2"]
    reopened.close()


def test_store_search_rejects_a_non_datetime_as_of():
    store = qiaomo_store()
    with pytest.raises(ValueError, match="timezone-aware datetime"):
        store.search(QIAOMO, "北川", as_of="2026-03-20T00:00:00+08:00")


# ---------------------------------------------------------------- contract

def test_retriever_satisfies_the_protocol():
    r = retriever(qiaomo_store())
    assert isinstance(r, MemoryRetriever)
    proto = list(inspect.signature(MemoryRetriever.retrieve).parameters.values())
    impl = list(inspect.signature(SlotRetriever.retrieve).parameters.values())
    assert [(p.name, p.kind, p.default) for p in impl] == [(p.name, p.kind, p.default) for p in proto]


def test_result_round_trips_and_its_invariants(ask):
    res = ask("乔沫生日是几月")
    assert RetrievalResult.from_json(res.to_json()) == res
    record = fixture_records()[0]
    hit = MemoryHit(record=record, role=TemporalRole.CURRENT, via="slot", matched=("住",), score=1.0)
    query = MemoryQuery(persona_id=QIAOMO, text="住")
    with pytest.raises(ValidationError, match="ok"):
        RetrievalResult(status=OK, query=query)
    with pytest.raises(ValidationError, match="ok"):
        RetrievalResult(status=NO_MATCH, query=query, hits=(hit,))
    with pytest.raises(ValidationError, match="error message"):
        RetrievalResult(status=RetrievalStatus.ERROR, query=query)
    with pytest.raises(ValidationError, match="another persona"):
        RetrievalResult(status=OK, query=MemoryQuery(persona_id="synthetic_x", text="住"), hits=(hit,))
    with pytest.raises(ValidationError, match="active"):
        MemoryHit(record=fixture_records()[4], role=TemporalRole.CURRENT, via="slot", matched=(), score=1.0)
    with pytest.raises(ValidationError, match="value"):
        SlotResolution(subject="persona", attribute="home_city", cardinality="single", status="conflict",
                       current=(record,), conflict_ids=(record.memory_id,), value="南屿市")
    with pytest.raises(ValidationError, match="superseded"):
        SlotResolution(subject="persona", attribute="home_city", cardinality="single", status="unknown",
                       corrected=(record,))


# The keys a Day 6 MemoryQuery has; readers of that version reject any other key (from_dict is strict).
DAY6_QUERY_KEYS = {"persona_id", "text", "context", "as_of", "subject", "time_scope", "limit"}


def test_a_query_without_speakers_keeps_the_day6_format(ask):
    plain = MemoryQuery(persona_id=QIAOMO, text="那它叫什么", context=("我家猫叫年糕",))
    assert set(plain.to_dict()) == DAY6_QUERY_KEYS
    assert set(MemoryQuery(persona_id=QIAOMO, text="x", context_roles=[]).to_dict()) == DAY6_QUERY_KEYS
    res = ask("乔沫住哪", context=["乔沫来了"])                              # nested in a result as well
    assert set(res.to_dict()["query"]) == DAY6_QUERY_KEYS
    assert RetrievalResult.from_json(res.to_json()) == res
    assert MemoryQuery.from_dict(plain.to_dict()).context_roles == ()


def test_speakers_survive_a_round_trip_and_are_still_checked():
    q = MemoryQuery(persona_id=QIAOMO, text="那它叫什么", context=("我家猫叫年糕", "嗯"),
                    context_roles=("persona", "user"))
    data = q.to_dict()
    assert set(data) == DAY6_QUERY_KEYS | {"context_roles"} and data["context_roles"] == ["persona", "user"]
    assert MemoryQuery.from_dict(data) == q and MemoryQuery.from_json(q.to_json()) == q
    res = retriever(qiaomo_store()).retrieve(QIAOMO, q.text, context=list(q.context), context_roles=["persona", "user"])
    assert RetrievalResult.from_json(res.to_json()).query.context_roles == ("persona", "user")
    for roles in (["persona"], ["persona", "system"]):
        with pytest.raises(ValidationError):
            MemoryQuery.from_dict(dict(data, context_roles=roles))
