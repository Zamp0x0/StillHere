"""Retrieval B2 and M3.1 reflection on CPU (gold extractor stands in for the model; no API)."""
import pytest

from stillhere.evaluation import writer_dataset as wd
from stillhere.evaluation import writer_dataset_v3 as v3
from stillhere.evaluation import writer_eval as we
from stillhere.memory import reflection_m31 as m31
from stillhere.memory import retrieval_b2 as b2


@pytest.fixture(scope="module")
def world():
    return we.run(we.new_world(we.reference_complete(v3.conversations()), "reference", ds=v3, m22=True), ds=v3)


def text_of(ev):
    return " ".join((r.statement or "") + " " + (r.value or "") for r in ev.records)


def test_flag_off_by_default():
    assert not b2.enabled({}) and b2.enabled({"MEMORY_RETRIEVAL_B2_ENABLED": "1"}) and not m31.enabled({})


def test_alias_cues():
    assert "dislikes" in b2.attributes_for("有什么东西是我不吃的？")
    assert {"job", "works_at"} <= set(b2.attributes_for("我是做什么的？"))
    assert "pet" in b2.attributes_for("我的狗叫什么？")


def test_synonym_recall_beats_legacy(world):
    q, when = "我平时喝什么饮品？", v3.t(9, 10, 20)
    assert "茶" in text_of(b2.retrieve(world.api, v3.RA, q, when))


def test_relation_word_resolves_to_the_person(world):
    ev = b2.retrieve(world.api, v3.RC, "我儿子是做什么工作的？", v3.t(9, 10, 20))
    assert ev.channels.get("subject") and "周斌" in text_of(ev)


def test_date_channel_finds_planned_events(world):
    ev = b2.retrieve(world.api, v3.RC, "我10月5日有什么安排？", v3.t(9, 10, 20))
    assert ev.channels.get("date") and "黄山" in text_of(ev)


def test_known_at_is_the_asking_time(world):
    for when in (v3.t(9, 1, 9), v3.t(9, 10, 20), v3.t(10, 20, 20)):
        ev = b2.retrieve(world.api, v3.RA, "我住在哪里？", when)
        assert all(r.recorded_at <= when for r in ev.records)


def test_events_are_one_unit_and_units_are_unique(world):
    ev = b2.retrieve(world.api, v3.RB, "拔智齿改到哪天了？", v3.t(9, 20, 20))
    keys = [k for k, _ in ev.units]
    assert len(keys) == len(set(keys)) and any(k.startswith("event:") for k in keys) and len(keys) <= 10


def test_persona_isolation(world):
    ev = b2.retrieve(world.api, v3.RA, "我的狗叫什么？", v3.t(9, 10, 20))
    assert all(r.persona_id == v3.RA for r in ev.records) and "年糕" not in text_of(ev)


def test_m31_context_and_brief(world):
    m = m31.reflect(world.api, v3.RA, "我要搬去成都吗？", v3.t(9, 12, 20))
    assert "当事人：阿凯" in m.context and "attribution" in m.notes   # the question is about the persona; 阿凯's plan is flagged as his
    m2 = m31.reflect(world.api, v3.RB, "我的职业和宠物分别是什么？", v3.t(9, 10, 20))
    assert "multi" in m2.notes


def test_m31_claim_check(world):
    m = m31.reflect(world.api, v3.RC, "我血糖怎么样？", v3.t(9, 20, 20))
    ok = m31.verify("根据现有记录无法确定。", True, m, "我血糖怎么样？")
    assert ok["ok"]
    bad = m31.verify("血糖是7.8", False, m, "我血糖怎么样？")
    assert not bad["ok"]


def test_dates_rendered_in_chinese():
    assert m31._zh_date("2026-10-12") == "10月12日（2026-10-12）" and m31._zh_date("2026-10-12 15:00").startswith("10月12日 15:00")


def test_no_regression_on_the_m2_world():
    w = we.run(we.new_world(we.reference_complete(wd.conversations()), "reference"))
    ev = b2.retrieve(w.api, wd.PA, "我妹妹在哪上学？", wd.t(10, 6, 20))
    assert ev.units
