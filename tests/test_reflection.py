"""M3 Reflection: router rules, bounded retrieval, brief content, known_at, claim check (synthetic, no model)."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from stillhere.memory import reflection as rf

TZ = timezone(timedelta(hours=8))
ASK = datetime(2026, 10, 15, 20, tzinfo=TZ)


@dataclass(frozen=True)
class R:
    memory_id: str
    subject: str
    attribute: str
    value: str
    valid_from: Optional[datetime]
    valid_to: Optional[datetime]
    recorded_at: datetime
    status: str = "active"
    correction_reason: str = ""
    statement: str = ""


def d(m, day):
    return datetime(2026, m, day, 10, tzinfo=TZ)


OLD = R("w.1", "persona", "lives_in", "武侯区", d(9, 1), d(10, 10), d(9, 1))
NEW = R("w.2", "persona", "lives_in", "高新区", d(10, 10), None, d(10, 10))
EV = R("w.3", "event:trip", "event.planned_time", "2026-10-03", d(9, 20), None, d(9, 20))
LATE = R("w.9", "persona", "lives_in", "未来区", d(10, 20), None, d(10, 20))


class FakeAPI:
    def __init__(self):
        self.calls = []

    def facts_during(self, persona, start, end, known_at):
        self.calls.append("during")
        return [r for r in (OLD, NEW, EV, LATE) if r.recorded_at <= known_at and (r.valid_to is None or r.valid_to > start) and (r.valid_from or start) <= end]

    def facts_at(self, persona, as_of, known_at):
        self.calls.append("at")
        return [r for r in (OLD, NEW, EV, LATE) if r.recorded_at <= known_at]

    def history(self, persona, subject, attribute, known_at):
        self.calls.append("history")
        return [r for r in (OLD, NEW, EV, LATE) if r.subject == subject and r.attribute == attribute and r.recorded_at <= known_at]


def test_router():
    assert rf.route("我住在哪个区？", ASK).kind == "SIMPLE_FACT"
    assert rf.route("我还住在武侯区吗？", ASK).kind == "KNOWLEDGE_UPDATE"
    assert rf.route("我现在还住在武侯区吗？", ASK).kind == "TEMPORAL"     # 现在 resolves to a time: the time rule fires first (R2, frozen)
    assert rf.route("我9月6日做了什么？", ASK).kind == "TEMPORAL"
    assert rf.route("我到底在哪家公司？", ASK).kind == "CONFLICT"
    assert rf.route("我弟弟和妹妹分别叫什么", ASK).kind == "MULTI_FACT"
    assert not rf.route("我住在哪个区？", ASK).complex


def test_simple_question_keeps_the_fast_path():
    api = FakeAPI()
    r = rf.reflect(api, "p", "我住在哪个区？", ASK, [NEW])
    assert r.extra_retrievals == 0 and r.brief == "" and api.calls == []


def test_bounded_and_history_in_the_brief():
    api = FakeAPI()
    r = rf.reflect(api, "p", "我现在还住在武侯区吗？", ASK, [NEW])
    assert r.extra_retrievals <= rf.MAX_EXTRA_RETRIEVALS
    assert "武侯区" in r.brief and "高新区" in r.brief and "当前" in r.brief and "过去" in r.brief


def test_known_at_hides_later_records():
    r = rf.reflect(FakeAPI(), "p", "我现在还住在哪？", ASK, [NEW])
    assert "未来区" not in r.brief and "w.9" not in r.evidence_ids


def test_dated_question_finds_the_event_by_its_planned_date():
    r = rf.reflect(FakeAPI(), "p", "我10月3日有什么安排？", ASK, [])
    assert r.route.kind == "TEMPORAL" and "2026-10-03" in r.brief and r.extra_retrievals == 2


def test_never_more_than_two_extra_retrievals():
    for q in ("我9月6日做了什么？", "我到底住哪？", "我现在还住在武侯区吗？", "我弟弟和妹妹分别叫什么"):
        assert rf.reflect(FakeAPI(), "p", q, ASK, [OLD, NEW]).extra_retrievals <= 2


def test_verify():
    assert rf.verify("你住在高新区。", False, "高新区（当前）")["ok"]
    assert "unsupported_numbers:7" in rf.verify("你住在7号楼。", False, "高新区")["problems"][0]
    assert rf.verify("你住在高新区。", False, "")["problems"] == ["answer_without_evidence"]
    assert rf.verify("无法确定", True, "")["ok"]
    empty = rf.Reflection(rf.route("我8月1日做了什么", ASK), brief="- 所问时间（2026-08-01 ~ ?）内没有相关记录：不要用其他时间的值回答。")
    assert "answered_a_time_with_no_record" in rf.verify("你去健身了。", False, "x", empty)["problems"]
    assert rf.verify("记录里没有那天的信息。", False, "x", empty)["ok"]


def test_reflection_never_writes():
    api = FakeAPI()
    rf.reflect(api, "p", "我到底住哪？", ASK, [OLD, NEW])
    assert set(api.calls) <= {"during", "at", "history"}
