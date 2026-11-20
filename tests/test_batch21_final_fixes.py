"""Batch 2.1 final bounded fixes (CPU; gold extractor, no model, no API): writer M2.2.2 (R13 explicit self-correction outside the span, R14
public-holiday dates), the holiday anchor for questions, the major-in-school wording, and the reader candidate registry (the frozen reader
unchanged)."""
from datetime import datetime, timedelta, timezone

import pytest

from stillhere.evaluation import reader_research as rr
from stillhere.evaluation import writer_dataset_v5 as v5
from stillhere.evaluation import writer_eval as we
from stillhere.memory import relative_days as rd
from stillhere.style import safety as S
from stillhere.writer import m222

TZ = timezone(timedelta(hours=8))


def _lives(world, persona):
    return {r.value: str(r.status).split(".")[-1] for r in world.api.store.all_records(persona) if r.attribute == "lives_in"}


def test_m222_flag_needs_m22_and_is_off_by_default():
    assert not m222.enabled({}) and not m222.enabled({"MEMORY_WRITER_M222_ENABLED": "1"})
    assert m222.enabled({"MEMORY_WRITER_M22_ENABLED": "1", "MEMORY_WRITER_M222_ENABLED": "1"})


def test_r13_explicit_correction_supersedes_and_off_keeps_the_old_behaviour():
    off = we.run(we.new_world(we.reference_complete(v5.conversations()), "reference", ds=v5, m22=True, m221=True), ds=v5)
    on = we.run(we.new_world(we.reference_complete(v5.conversations()), "reference", ds=v5, m22=True, m221=True, m222=True), ds=v5)
    assert _lives(off, v5.UC) == {"南山区": "ACTIVE", "福田区": "ACTIVE"}                 # M2.2.1 exactly as before
    assert _lives(on, v5.UC) == {"南山区": "SUPERSEDED", "福田区": "ACTIVE"}


class _M:
    def __init__(self, mid, speaker, content, ts):
        self.message_id, self.speaker_id, self.content, self.timestamp = mid, speaker, content, ts


class _C:
    def __init__(self, **kw):
        self.corrects, self.planned_time, self.event_time, self.valid_from, self.valid_to = False, None, None, None, None
        self.__dict__.update(kw)


class _V:
    def __init__(self, c, dropped):
        self.candidate, self.dropped_fields, self.reasons, self.support = c, list(dropped), [], "PARTIALLY_SUPPORTED"

    @property
    def committable(self):
        return True


def test_r13_needs_an_explicit_word_in_the_persona_own_message():
    ts = datetime(2026, 9, 10, 20, tzinfo=TZ)
    weak = _V(_C(source_message_ids=["m1"], source_spans=["我住的是福田区"]), ["corrects"])
    assert not m222.restore_explicit_correction(True, weak, [_M("m1", "p", "其实我住的是福田区", ts)], "p")
    other = _V(_C(source_message_ids=["m1"], source_spans=["我住的是福田区"]), ["corrects"])
    assert not m222.restore_explicit_correction(True, other, [_M("m1", "friend", "你说错了，我住的是福田区", ts)], "p")
    ok = _V(_C(source_message_ids=["m1"], source_spans=["我住的是福田区"]), ["corrects"])
    assert m222.restore_explicit_correction(True, ok, [_M("m1", "p", "我之前说错了，我住的是福田区", ts)], "p")
    assert ok.candidate.corrects and ok.support == "SUPPORTED"


@pytest.mark.parametrize("text,value,kept", [
    ("国庆最后一天我去爬了峨眉山。", "2026-10-07", "2026-10-07"),      # one day of the holiday: exact
    ("国庆最后一天我去爬了峨眉山。", "2026-10-06", None),              # not that day
    ("国庆打算回扬州老家住几天。", "2026-10-01", "2026-10"),            # whole holiday: month precision, never a guessed day
    ("国庆打算回扬州老家住几天。", "2026-11-01", None),                # outside the span
    ("下周打算回老家。", "2026-10-01", None),                          # no holiday named
])
def test_r14_holiday_dates(text, value, kept):
    ts = datetime(2026, 9, 22, 19, tzinfo=TZ)
    c = _C(source_message_ids=["m1"], source_spans=[text], planned_time=None)
    v = _V(c, ["planned_time"])
    m222.restore_holiday_dates({"planned_time": value}, v, [_M("m1", "p", text, ts)])
    assert c.planned_time == kept


@pytest.mark.parametrize("q,ask,days", [
    ("国庆最后一天我干什么了？", datetime(2026, 10, 13, 20, tzinfo=TZ), "2026-10-07"),
    ("我国庆打算去哪里？", datetime(2026, 9, 20, tzinfo=TZ), "2026-10-01 ~ 2026-10-07"),
    ("五一第二天呢", datetime(2026, 9, 8, tzinfo=TZ), "2026-05-02"),            # the nearest occurrence (this year's, already past)
    ("元旦去哪", datetime(2026, 12, 20, tzinfo=TZ), "2027-01-01"),
])
def test_holiday_windows(q, ask, days):
    w = rd.resolve(q, ask)
    assert w is not None and w.days() == days


def test_relative_days_take_precedence_and_no_holiday_is_none():
    ask = datetime(2026, 9, 30, 20, tzinfo=TZ)
    assert rd.resolve("明天国庆我去哪", ask).days() == "2026-10-01"
    assert rd.holiday("十一月我去哪", ask) is None and rd.resolve("我住哪", ask) is None


def test_major_in_school_slot_reads_naturally_and_is_deduplicated():
    assert S._second_person("本人在软件工程上学") == "你读的是软件工程"
    assert S._second_person("本人在武汉大学上学") == "你在武汉大学上学"
    text, kind = S.safe_fallback({"facts": [{"statement": "你读的是软件工程专业", "role": "current"}, {"statement": "本人在软件工程上学", "role": "current"}],
                                  "unknown": []})
    assert kind == "SAFE_PARTIAL" and text == "我记得你读的是软件工程专业。"


def test_reader_candidate_is_separate_and_the_frozen_reader_unchanged():
    c1, c2 = rr.CANDIDATES["deepseek_c1_answerability"], rr.CANDIDATES["deepseek_c2_undated_plan"]
    assert c1.digest() == "1062a4454cad497d"                              # the frozen reader's configuration digest (answer cache keys)
    assert c2.system_prompt == c1.system_prompt.replace(rr.ANSWERABILITY_RULES[1], rr.ANSWERABILITY_RULES[1] + rr.UNDATED_PLAN_RULE)
    assert len(rr.CANDIDATES) <= 4


# ---------------------------------------------------------------- R15 multi-valued correction (synthetic records, not the frozen sets)
class _R:
    def __init__(self, value, attribute="pet", mid="w.old"):
        self.value, self.attribute, self.memory_id = value, attribute, mid


def _target(text, new, olds, attribute="pet", corrects=True, speaker="p"):
    ts = datetime(2026, 9, 10, 20, tzinfo=TZ)
    c = _C(corrects=corrects, source_message_ids=["m1"], source_spans=[text])
    return m222.multi_correction_target(c, _R(new, attribute, "w.new"), [_R(v, attribute, f"w.{i}") for i, v in enumerate(olds)], [_M("m1", speaker, text, ts)], "p")


@pytest.mark.parametrize("text,new,olds,attribute,want", [
    ("我之前说错了，我养的是一只猫，不是狗。", "猫", ["狗"], "pet", "狗"),                         # names the wrong value
    ("更正一下：我室友叫小周，不是小邹。", "室友 小周", ["室友 小邹", "同事 阿康"], "relationship", "室友 小邹"),
    ("我说错了，鹦鹉叫球球。", "鹦鹉球球", ["鹦鹉豆豆"], "pet", "鹦鹉豆豆"),                       # explicit word, one earlier value
    ("我记错了，表妹叫思琪。", "表妹 思琪", ["表妹 思思", "同事 阿康"], "relationship", "表妹 思思"),   # same role only
    ("我又养了一只小黑，不是小白那种性格。", "小黑", ["小白"], "pet", None),                        # addition word wins
    ("我以前养小白，现在养的是小黑。", "小黑", ["小白"], "pet", None),                              # change over time
    ("我说错了，养的是小黑。", "小黑", ["小白", "小花"], "pet", None),                              # two candidates: ambiguous
    ("我养的是小黑。", "小黑", ["小白"], "pet", None),                                              # no explicit correction
])
def test_r15_multi_valued_correction_target(text, new, olds, attribute, want):
    got = _target(text, new, olds, attribute)
    assert (got.value if got is not None else None) == want


def test_r15_needs_the_correction_flag_and_the_persona_own_words():
    assert _target("不是狗，是猫。", "猫", ["狗"], corrects=False) is None
    assert _target("你说错了，你养的是猫，不是狗。", "猫", ["狗"], speaker="friend") is None


def test_r15_end_to_end_supersedes_and_keeps_history():
    """On the b21f-6 world (already used once as a confirmation; here only the gold extractor): pet and roommate corrections."""
    from stillhere.evaluation import writer_dataset_v6 as v6
    for flag, pet, room in ((False, {"狗": "ACTIVE", "猫": "ACTIVE"}, {"室友 小邹": "ACTIVE", "室友 小周": "ACTIVE"}),
                            (True, {"狗": "SUPERSEDED", "猫": "ACTIVE"}, {"室友 小邹": "SUPERSEDED", "室友 小周": "ACTIVE"})):
        w = we.run(we.new_world(we.reference_complete(v6.conversations()), "reference", ds=v6, m22=True, m221=True, m222=flag), ds=v6)
        got = lambda p, a: {r.value: str(r.status).split(".")[-1] for r in w.api.store.all_records(p) if r.attribute == a}
        assert got(v6.XC, "pet") == pet
        assert {k: v for k, v in got(v6.XB, "relationship").items() if k.startswith("室友")} == room


def test_m31_brief_names_a_corrected_value_next_to_its_correction():
    """The corrected-away value gets an explicit note (b21g-7 world with the gold extractor; that set's official run is already recorded)."""
    from stillhere.evaluation import writer_dataset_v7 as v7
    from stillhere.memory import reflection_m31 as m31
    w = we.run(we.new_world(we.reference_complete(v7.conversations()), "reference", ds=v7, m22=True, m221=True, m222=True), ds=v7)
    m = m31.reflect(w.api, v7.YC, "我喜欢喝什么茶？", v7.t(9, 12, 20))
    assert "correction" in m.notes and "“绿茶”已被更正为编号" in m.brief and "“红茶”" in m.brief
    plain = m31.reflect(w.api, v7.YA, "我做什么工作？", v7.t(9, 12, 20))
    assert "correction" not in plain.notes
