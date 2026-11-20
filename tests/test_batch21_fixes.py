"""Batch 2.1 fixes on CPU: writer M2.2.1 rules, relative weekdays / evaluation clock, M3.1 notes and claim check. No model, no API."""
from datetime import datetime
from types import SimpleNamespace as N

import pytest

from stillhere.evaluation import writer_dataset as wd
from stillhere.evaluation import writer_dataset_v2 as v2
from stillhere.evaluation import writer_dataset_v3 as v3
from stillhere.evaluation import writer_dataset_v4 as v4
from stillhere.evaluation import writer_eval as we
from stillhere.memory import reflection_m31 as m31
from stillhere.memory import relative_days as rd
from stillhere.memory import retrieval_b2 as b2
from stillhere.memory.temporal import CALENDAR_TZ as TZ
from stillhere.writer import m22, m221


def cand(**kw):
    base = dict(subject="self", predicate="likes", object="咖啡", polarity="positive", type="FACT", event_status=None, event_key=None, activity=None,
                planned_time=None, event_time=None, corrects=False, retracts=False, refers_to=None, source_message_ids=["m"], source_spans=["我喜欢咖啡"],
                certainty="certain", persona_id="p")
    base.update(kw)
    return N(**base)


def msgs(text):
    return {"m": N(content=text, speaker_id="p", sender="本人")}


def test_assistant_suggestion_accepted_by_the_persona_is_a_plan():
    ms = {"a": N(content="你喜欢科幻片的话，可以看看《星际穿越》。", speaker_id="p.assistant", sender="AI 助手"), "u": N(content="好，我周末看看。", speaker_id="p", sender="本人")}
    c = cand(predicate="event", object="看《星际穿越》", event_status="PLANNED", source_message_ids=["a", "u"], source_spans=["可以看看《星际穿越》", "我周末看看"])
    assert m221.not_a_fact(c, ms, "p") is None


# ---------------------------------------------------------------- writer M2.2.1
def test_flag_needs_m22():
    assert not m221.enabled({"MEMORY_WRITER_M221_ENABLED": "1"})
    assert m221.enabled({"MEMORY_WRITER_M22_ENABLED": "1", "MEMORY_WRITER_M221_ENABLED": "1"}) and not m221.enabled({})


@pytest.mark.parametrize("text,span,why", [
    ("如果下个月有空，我想去西藏。", "我想去西藏", "R8 hypothetical"),
    ("要是明天不下雨，我就去紫金山跑步。", "我就去紫金山跑步", "R8 hypothetical"),
    ("要不周末去青城山？", "周末去青城山", "R8 suggestion / question"),
    ("医生建议我多睡觉少熬夜。", "多睡觉少熬夜", "R8 advice"),
    ("阿雷说我肯定喜欢猫。", "我肯定喜欢猫", "R8 hearsay about the persona"),
    ("人家都说那家火锅好吃。", "那家火锅好吃", "R8 hearsay about the persona"),
])
def test_not_a_fact(text, span, why):
    c = cand(predicate="event" if "R8 h" in why or "sugg" in why else "likes", source_spans=[span], event_status="PLANNED")
    assert m221.not_a_fact(c, msgs(text)) == why


@pytest.mark.parametrize("text,span,pred,status", [
    ("我喜欢咖啡。", "我喜欢咖啡", "likes", None),
    ("朋友都说我做的沙茶面好吃，我自己也挺喜欢做饭。", "我自己也挺喜欢做饭", "likes", None),       # the hearsay clause does not leak into the next
    ("我打算9月20日去鼓浪屿拍照。", "我打算9月20日去鼓浪屿拍照", "event", "PLANNED"),
    ("下周六原本想去看话剧，后来改成了周日，最后还是没去成。", "下周六原本想去看话剧", "event", "CANCELLED"),
    ("我上周去了香港。", "我上周去了香港", "event", "COMPLETED"),
])
def test_real_facts_are_kept(text, span, pred, status):
    assert m221.not_a_fact(cand(predicate=pred, source_spans=[span], event_status=status), msgs(text)) is None


def _dc(c):
    from dataclasses import make_dataclass
    return make_dataclass("C", [(k, object) for k in vars(c)])(**vars(c))


def test_negated_preference_becomes_dislikes():
    c = m221.prenormalize(_dc(cand(predicate="likes", object="辣", polarity="negative")))
    assert c.predicate == "dislikes" and c.polarity == "positive"


def _norm(c, existing=()):
    from dataclasses import make_dataclass
    D = make_dataclass("C", [(k, object) for k in vars(c)])
    return m221.normalize(D(**vars(c)), msgs(" ".join(c.source_spans)), "p", {"阿雷"}, m22.open_events(existing), existing)


def test_third_party_minimisation_and_generic_values():
    assert _norm(cand(subject="person:阿雷", predicate="likes", object="咖啡"))[0] is None
    assert _norm(cand(subject="person:阿雷", predicate="pet", object="猫"))[0] is None
    assert _norm(cand(subject="person:阿雷", predicate="job", object="设计师", source_spans=["阿雷是设计师"]))[0] is not None
    assert _norm(cand(subject="person:阿雷", predicate="works_at", object="公司（调岗）", source_spans=["公司调岗"]))[0] is None


def test_reference_resolution_and_reschedule_link():
    existing = [("w.1", "event:e1", "event.description", "拔智齿"), ("w.2", "event:e1", "event.status", "SCHEDULED"), ("w.3", "event:e1", "event.actor", "self")]
    c, notes = _norm(cand(predicate="event", object="拔智齿", event_status="PLANNED", planned_time="2026-09-24", refers_to="w.1", source_spans=["拔智齿改到9月24日了"]), existing)
    assert c.refers_to == "event:e1" and any(n.startswith("R5") for n in notes)
    c2, notes2 = _norm(cand(predicate="event", object="拔智齿", event_status="PLANNED", planned_time="2026-09-24", source_spans=["拔智齿改到9月24日了"]), existing)
    assert c2.refers_to == "event:e1"


def test_retraction_links_to_the_event():
    existing = [("w.1", "event:e2", "event.description", "报名社区摄影班"), ("w.2", "event:e2", "event.status", "SCHEDULED"), ("w.9", "persona", "routine", "x")]
    c, notes = _norm(cand(predicate="event", object="报名社区摄影班", retracts=True, refers_to="w.9", source_spans=["摄影班那个其实没报上"]), existing)
    assert c.refers_to == "event:e2" and any(n.startswith("R11") for n in notes)


def test_batch_filter_sub_events_and_routine_details():
    v = lambda **kw: N(candidate=cand(**kw), committable=True)
    vs = [v(predicate="event", object="去香港看演唱会", event_status="CANCELLED", source_message_ids=["m1"]),
          v(predicate="event", object="演唱会延期", event_status="UNKNOWN", source_message_ids=["m1"]),
          v(predicate="event", object="报名社区摄影班", event_status="SCHEDULED", source_message_ids=["m2"]),
          v(predicate="routine", object="每周三上社区摄影班", source_message_ids=["m2"]),
          v(predicate="event", object="游泳", event_status="COMPLETED", source_message_ids=["m3"])]
    drop = dict(m221.batch_filter(vs))
    assert set(drop) == {1, 3}


@pytest.mark.parametrize("ds", [wd, v2, v3, v4])
def test_rules_never_drop_a_gold_fact(ds):
    """The gold extractor (the gold as the model's answer) still scores 100% with M2.2.1 on: no rule removes a true fact."""
    w = we.run(we.new_world(we.reference_complete(ds.conversations()), "reference", ds=ds, m22=True, m221=True), ds=ds)
    o = we.one_to_one(w, ds=ds)
    assert o["precision"] == 1.0 and o["recall"] == 1.0, (o["false_positives"], o["false_negatives"])


# ---------------------------------------------------------------- relative days / evaluation clock
TUE = datetime(2026, 9, 8, 20, tzinfo=TZ)


@pytest.mark.parametrize("q,ask,days", [
    ("我这周六有什么安排？", TUE, ["2026-09-12"]),
    ("上周六干嘛了", TUE, ["2026-09-05"]),
    ("明天有什么事", TUE, ["2026-09-09"]),
    ("后天呢", TUE, ["2026-09-10"]),
    ("昨天干嘛了", TUE, ["2026-09-07"]),
    ("这周末去哪", TUE, ["2026-09-12 ~ 2026-09-13"]),
    ("下个星期一", datetime(2026, 12, 30, 9, tzinfo=TZ), ["2027-01-04"]),          # year boundary
    ("下周日", datetime(2026, 9, 30, 9, tzinfo=TZ), ["2026-10-11", "2026-10-04"]),  # month boundary; said on a Wednesday: both readings
    ("这周天", datetime(2026, 9, 13, 9, tzinfo=TZ), ["2026-09-13"]),               # Sunday is day 7 of its week
])
def test_relative_days(q, ask, days):
    w = rd.resolve(q, ask)
    assert [x.days() for x in (w.readings or [w])] == days


def test_next_week_weekday_is_flagged_ambiguous_early_in_the_week():
    w = rd.resolve("下周六有什么安排", TUE)
    assert w.ambiguous and [r.days() for r in w.readings] == ["2026-09-19", "2026-09-12"]
    assert not rd.resolve("下周一", TUE).ambiguous                       # this week's Monday has passed: no second reading


def test_anchor_is_the_asking_time_only():
    assert rd.resolve("明天", datetime(2025, 1, 1, tzinfo=TZ)).days() == "2025-01-02"


def test_query_windows_end_is_exclusive_and_asking_year_reading():
    ws = b2.query_windows("10月5日有什么安排？", datetime(2026, 9, 10, 20, tzinfo=TZ))
    assert [w[2] for w in ws] == ["parser", "asking-year reading"] and ws[1][0].date().isoformat() == "2026-10-05"
    assert ws[1][1].date().isoformat() == "2026-10-06"                   # [start, end): 10-06 itself is NOT inside


# ---------------------------------------------------------------- M3.1 notes and claim check
@pytest.fixture(scope="module")
def world():
    return we.run(we.new_world(we.reference_complete(v3.conversations()), "reference", ds=v3, m22=True), ds=v3)


def test_no_event_note_shows_the_asking_year_first(world):
    m = m31.reflect(world.api, v3.RC, "10月7日我有什么安排？", datetime(2026, 9, 10, 20, tzinfo=TZ))
    line = next(l for l in m.brief.splitlines() if "所问日期" in l)
    assert line.index("2026-10-07") < line.index("2025-10-07")


def test_claim_check_allows_question_and_asking_date_numbers(world):
    ask = datetime(2026, 11, 7, 20, tzinfo=TZ)
    m = m31.reflect(world.api, v3.RA, "10月18日我有什么安排？", ask)
    assert m31.verify("截至11月7日，10月18日没有安排记录。", False, m, "10月18日我有什么安排？")["ok"]
    assert not m31.verify("10月18日你要去深圳，花了3500元。", False, m, "10月18日我有什么安排？")["ok"]       # a new number is still flagged


def test_event_without_status_says_so():
    from stillhere.memory.reflection_m31 import _event_line
    r = lambda a, v: N(attribute=a, value=v, recorded_at=datetime(2026, 9, 18, tzinfo=TZ), valid_from=None, valid_to=None, memory_id=a, status=N(value="active"),
                       correction_reason=None)
    line = _event_line("event:x", [r("event.description", "带本人去北京玩"), r("event.actor", "person:许晴")], datetime(2026, 10, 1, tzinfo=TZ))
    assert "未记录（没有取消、改期或完成的记录）" in line
