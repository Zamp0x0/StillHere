"""BUILD-3: the generic chat importer and the LifeBench mapping, on a small synthetic fixture in the dataset's shape."""
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from stillhere.contracts import DataOrigin, MemoryKind, SourceKind, SourceRef
from stillhere.importers import GenericConversationImporter, LifeBenchImporter, ensure_selection, parse_persona, stable_id
from stillhere.importers.lifebench import scan_persona, select_personas
from stillhere.memory import MemoryAPI, MessageConflict, SchemaMismatch
from stillhere.memory.sqlite_store import DuplicateMemoryId

CST = timezone(timedelta(hours=8))
NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)
A, B = "lifebench_alpha", "lifebench_beta"
QA_SECRET = "只在标准答案里出现的句子"


def write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def persona_files(root: Path, name: str, *, display="甲某", city="甲市", event_word="长跑", shift=0) -> Path:
    d = root / name
    write(d / "persona.json", {
        "name": display, "birth": "1990-01-01", "gender": "女", "education": "本科", "job": "工程师", "occupation": "甲公司",
        "belief": "", "family": "已婚", "age": 35, "home_address": {"province": "省", "city": city, "district": "区",
                                                                  "street_name": "路", "street_number": "1号"},
        "workplace": {"province": "省", "city": city, "district": "区", "street_name": "街", "street_number": "2号"},
        "birth_place": {"province": "省", "city": city, "district": "区"}, "hobbies": ["读书", "爬山"],
        "favorite_foods": ["面条"], "aim": ["考取证书"], "relation": [[{"name": "乙某", "relation": "朋友"}]]})
    event = {"type": "Personal Life", "participant": [{"name": display, "relation": "自己"}, {"name": "乙某", "relation": "朋友"}],
             "location": "甲市公园"}
    write(d / "daily_event.json", [
        {"event_id": "1", "name": f"{event_word}训练", "date": ["2025-01-01 06:00:00至2025-01-01 07:00:00"],
         "description": f"早晨在公园{event_word}五公里", "atomic_id": ["7-1", "9-9"], **event},
        {"event_id": "2", "name": "夜间加班", "date": ["2025-12-31 22:10:00至2025-12-31 00:10:00"],
         "description": "写完年终报告", "atomic_id": [], **event},
        {"event_id": "3", "name": "通宵看书", "date": ["2025-06-01 23:00:00至2025-06-01 26:15:00"],
         "description": "读完一本小说", "atomic_id": [], **event},
        {"event_id": "4", "name": "坏记录", "date": ["不是时间"], "description": "无法解析", "atomic_id": [], **event},
        {"event_id": "5", "name": f"{event_word}训练", "date": ["2025-01-01 06:00:00至2025-01-01 07:00:00"],
         "description": f"早晨在公园{event_word}五公里", "atomic_id": [], **event}])
    write(d / "event_tree.json", [{"event_id": 7, "name": "全年训练", "date": ["2025-01-01至2025-03-01"], "type": "Health",
                                   "subevent": [{"event_id": "7-1", "name": "第一次", "date": ["2025-01-01"]},
                                                {"event_id": "7-2", "name": "第二次", "date": ["2025-02-01"]}]}])
    write(d / "daily_draft.json", {"2025-01": [{"date": "2025-01-01", "events": [{"name": "大纲"}]}]})
    phone = d / "phone_data"
    write(phone / "contact.json", [{"name": "乙某", "relation": "朋友", "gender": "男", "nickname": "小乙",
                                    "phoneNumber": "13800000001", "phone_id": 1}])
    write(phone / "sms.json", [
        {"type": "sms", "message_content": "明天公园见", "contactName": "乙某", "phoneNumber": "13800000001",
         "datetime": "2025-01-01 20:00:00", "message_type": "发送", "daily_event_id": "1", "event_id": ["7-1"], "phone_id": 0},
        {"type": "sms", "message_content": "我下个月搬去海边", "contactName": "乙某", "phoneNumber": None,
         "datetime": "2025-01-01 20:05:00", "message_type": "接收", "daily_event_id": "1", "event_id": [], "phone_id": 1},
        {"type": "sms", "message_content": "您的验证码", "contactName": "银行", "phoneNumber": "95588",
         "datetime": "2025-01-02 09:00:00", "message_type": "接收", "daily_event_id": "1", "event_id": [], "phone_id": 2},
        {"type": "sms", "message_content": "无法判断是谁", "contactName": "", "phoneNumber": None,
         "datetime": "2025-01-02 10:00:00", "message_type": "接收", "daily_event_id": "1", "event_id": [], "phone_id": 3},
        {"type": "sms", "message_content": "类型不明", "contactName": "乙某", "phoneNumber": None,
         "datetime": "2025-01-02 11:00:00", "message_type": "未知", "daily_event_id": "1", "event_id": [], "phone_id": 4}])
    write(phone / "agent_chat.json", [
        {"date": "2025-01-01", "type": "agent_chat", "daily_event_id": "1", "event_id": [], "phone_id": 0, "conversation": {
            "turn 1": {"user": {"action": "topic query", "content": "帮我做训练计划"},
                       "assistant": {"action": "solution proposal", "content": "建议每周三次"}}}},
        {"date": "2025-01-03", "type": "agent_chat", "daily_event_id": "1", "event_id": [], "phone_id": 1, "conversation": {
            "turn 1": {"user": {"action": "x", "content": "不在同一天"}, "assistant": {"action": "y", "content": ""},
                       "turn 2": {"user": {"action": "x", "content": "嵌套的一轮"}, "assistant": {"action": "y", "content": "好"}}}}}])
    write(phone / "calendar.json", [{"type": "calendar", "title": "体检", "description": "去医院", "start_time": "2025-03-01 09:00:00",
                                     "end_time": "2025-03-01 10:00:00", "datetime": "2025-02-01 08:00:00", "daily_event_id": "1",
                                     "event_id": ["7-2"], "phone_id": 0, "summarized_info": ""}])
    write(phone / "note.json", [{"type": "note", "title": "", "content": "周末买菜", "datetime": "2025-02-02 12:00:00",
                                 "daily_event_id": "1", "event_id": [], "phone_id": 0, "summarized_info": ""}])
    write(phone / "photo.json", [{"type": "photo", "caption": "公园合影", "title": "IMG_1", "datetime": "2025-01-01 07:10:00",
                                  "location": {"city": city, "district": "区", "poi": "公园"}, "faceRecognition": "乙某",
                                  "imageTag": ["合影"], "ocrText": "", "daily_event_id": "1", "event_id": [], "phone_id": 0}])
    write(phone / "push.json", [{"type": "push", "title": "天气", "content": "明日有雨", "datetime": "2025-01-05 07:00:00",
                                 "source": "天气", "daily_event_id": "1", "event_id": [], "phone_id": 0}])
    write(phone / "call.json", [{"type": "call", "phoneNumber": "13800000001", "contactName": "乙某",
                                 "datetime": "2025-01-06 10:00:00", "datetime_end": "2025-01-06 10:02:00", "direction": 1,
                                 "call_result": "接通", "daily_event_id": "1", "event_id": [], "phone_id": 0}])
    write(phone / "fitness_health.json", [
        {"日期": "2025-01-01", "城市": city, "summarized_info": "全天步数两万", "event_id": [], "phone_id": 0},
        {"日期": "2025-01-02", "城市": city, "summarized_info": "", "event_id": [], "phone_id": 1}])
    write(d / "QA_all" / "QA.json", [{"question": "问题", "answer": QA_SECRET, "evidence": [{"x": QA_SECRET}],
                                      "score_points": [], "required_events_id": [], "ask_time": "2025-06-20",
                                      "question_type": ["Single_hop", "Temporal"]}])
    return d


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "data"
    return {"root": root, A: persona_files(root, "alpha"),
            B: persona_files(root, "beta", display="丙某", city="丙市", event_word="游泳")}


@pytest.fixture
def api(dataset):
    memory = MemoryAPI(clock=lambda: NOW)
    importer = LifeBenchImporter(memory, "rev1")
    importer.import_persona(dataset[A])
    importer.import_persona(dataset[B])
    yield memory
    memory.close()


def all_ids(api, sql):
    return [r[0] for r in api.store._conn.execute(sql)]


# ---------------------------------------------------------------- T1 / T2 conversion, persona and speaker mapping

def test_sms_speakers_follow_the_message_direction(dataset):
    parsed = parse_persona(dataset[A], "rev1")
    by_id = {m.message_id: m for m in parsed.messages}
    sent, received = by_id["lb.alpha.sms.0"], by_id["lb.alpha.sms.1"]
    assert (sent.target_persona_id, sent.speaker_id) == (A, A)
    assert (received.target_persona_id, received.speaker_id) == (A, "lb.alpha.contact.1")   # matched by name
    assert parsed.speakers[received.speaker_id]["name"] == "乙某"
    assert sent.conversation_id == received.conversation_id
    outside = by_id["lb.alpha.sms.2"]                                     # not in contacts: stable, not the contact
    assert outside.speaker_id.startswith("lb.alpha.ext.") and parsed.speakers[outside.speaker_id]["name"] == "银行"
    assert "lb.alpha.sms.3" not in by_id and "lb.alpha.sms.4" not in by_id  # no speaker is invented
    assert parsed.report.skipped["sms_speaker_unresolved"] == 2


def test_a_friends_first_person_stays_the_friends(api):
    """"我下个月搬去海边" comes from B's friend: a message, no fact, and no statement about the persona."""
    [hit] = api.search_messages(A, "海边")
    assert hit.speaker_id == "lb.alpha.contact.1" and hit.subject_id is None
    assert not [s for s in all_ids(api, "SELECT statement FROM memories") if "海边" in s]


def test_agent_chat_roles_time_and_nested_turns(dataset):
    parsed = parse_persona(dataset[A], "rev1")
    chat = [m for m in parsed.messages if m.source_type == "lifebench_agent_chat"]
    assert [(m.message_id.rsplit(".", 2)[-2:], m.speaker_id) for m in chat[:2]] == [
        (["0", "0"], A), (["0", "1"], "lb.alpha.assistant")]
    assert chat[0].timestamp == datetime(2025, 1, 1, 7, 0, tzinfo=CST)     # end of the linked event, an assumption
    assert chat[1].timestamp == chat[0].timestamp + timedelta(seconds=1)
    assert "time=assumed_daily_event_end" in chat[0].locator and "date=2025-01-01" in chat[0].locator
    other_day = [m for m in chat if ".agent_chat.1." in m.message_id]
    assert other_day[0].timestamp == datetime(2025, 1, 3, 23, 59, tzinfo=CST)
    assert [m.content for m in other_day] == ["不在同一天", "嵌套的一轮", "好"]   # the empty one is skipped and counted
    assert parsed.report.skipped["agent_chat_empty_message"] == 1


# ---------------------------------------------------------------- T3 structured events

def test_events_are_memories_with_their_own_interval(api):
    record = api.get(A, "lb.alpha.daily_event.1")
    assert record.kind is MemoryKind.EVENT and record.subject == "persona"
    assert record.valid_from == datetime(2025, 1, 1, 6, 0, tzinfo=CST)
    assert record.valid_to == datetime(2025, 1, 1, 7, 0, tzinfo=CST)       # never an open-ended state
    assert record.recorded_at == record.valid_to                           # assumed: known when it ended
    assert "2025-01-01 06:00:00至2025-01-01 07:00:00" in record.statement and "乙某" in record.statement


def test_dirty_times_are_repaired_and_counted(dataset):
    parsed = parse_persona(dataset[A], "rev1")
    facts = {f.memory_id: f for f in parsed.facts}
    midnight = facts["lb.alpha.daily_event.2"]
    assert midnight.valid_to == datetime(2026, 1, 1, 0, 10, tzinfo=CST)
    assert "至2025-12-31 00:10:00" in midnight.statement                    # the original text stays
    assert facts["lb.alpha.daily_event.3"].valid_to == datetime(2025, 6, 2, 2, 15, tzinfo=CST)
    notes = parsed.report.notes
    assert (notes["repaired_range_past_midnight"], notes["repaired_hour_past_24"]) == (1, 1)
    assert parsed.report.error_count == 1 and "daily_event.json 4" in parsed.report.errors[0]
    assert parsed.report.duplicate_count == 1 and "lb.alpha.daily_event.5" not in facts


def test_every_record_is_accounted_for(dataset):
    report = parse_persona(dataset[A], "rev1").report
    assert report.source_only["daily_draft_days"] == 1
    assert report.source_only["persona_other_fields"] > 0
    assert report.notes["atomic_id_not_in_event_tree"] == 1
    assert report.notes["empty_title"] == 1
    assert report.skipped["fitness_without_summary"] == 1
    assert report.raw_by_file["sms.json"] == 5 and report.raw_by_file["daily_event.json"] == 5


# ---------------------------------------------------------------- T4 provenance

def test_a_record_traces_back_to_the_file_revision_and_row(api, dataset):
    record = api.get(A, "lb.alpha.calendar.0")
    [own, daily, tree] = record.provenance
    assert (own.source_id, own.locator) == ("lb.alpha.calendar", "phone_id=0")
    assert (daily.source_id, daily.locator) == ("lb.alpha.daily_event", "event_id=1")
    assert (tree.source_id, tree.locator) == ("lb.alpha.event_tree", "node=7-2")
    evidence = api.get_evidence(A, "lb.alpha.calendar")
    assert evidence.source.locator == "hf://datasets/C1754955896/Lifebenchv2.0@rev1/data/alpha/phone_data/calendar.json"
    assert evidence.source.origin is DataOrigin.SYNTHETIC and len(evidence.source.sha256) == 64
    # an event tree node is a link, not a second copy of the event
    assert not [m for m in all_ids(api, "SELECT memory_id FROM memories") if "tree" in m]
    assert api.get(A, "lb.alpha.daily_event.1").provenance[1].locator == "node=7-1"


def test_qa_never_reaches_the_store(api):
    for table, column in (("memories", "statement"), ("messages", "content"), ("provenance", "quote"), ("sources", "record_json")):
        assert not [v for v in all_ids(api, f"SELECT {column} FROM {table}") if v and QA_SECRET in v]
    assert not [s for s in all_ids(api, "SELECT source_id FROM sources") if "QA" in s or "qa" in s.lower().split(".")]


def test_the_scan_sees_question_types_and_nothing_else(dataset):
    stats = scan_persona(dataset[A])
    assert (stats["qa_count"], stats["qa_types"]) == (1, {"Single_hop": 1, "Temporal": 1})
    assert QA_SECRET not in json.dumps(stats, ensure_ascii=False)


# ---------------------------------------------------------------- T5 / T6 idempotency and conflicts

def test_importing_again_adds_nothing(dataset):
    api = MemoryAPI(clock=lambda: NOW)
    importer = LifeBenchImporter(api, "rev1")
    first = importer.import_persona(dataset[A]).report
    assert first.added_messages > 0 and first.added_facts > 0
    before = api.stats()["counts"]
    again = importer.import_persona(dataset[A]).report
    assert (again.added_messages, again.added_facts) == (0, 0)
    assert api.stats()["counts"] == before and api.check_integrity() == []


def test_changed_content_under_a_stored_id_is_an_error(dataset):
    api = MemoryAPI(clock=lambda: NOW)
    LifeBenchImporter(api, "rev1").import_persona(dataset[A])
    path = dataset[A] / "phone_data" / "sms.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows[0]["message_content"] = "改过的内容"
    write(path, rows)
    with pytest.raises(Exception, match="another definition"):            # the source's file hash changed first
        LifeBenchImporter(api, "rev1").import_persona(dataset[A])

    fresh = MemoryAPI(clock=lambda: NOW)
    original = parse_persona(dataset[A], "rev1")
    for s in original.sources:
        fresh.register_source(s)
    fresh.ingest_messages(original.messages)
    fresh.import_facts(A, original.facts)
    changed = original.messages[0].__class__.from_dict({**original.messages[0].to_dict(), "content": "别的话"})
    with pytest.raises(MessageConflict):
        fresh.ingest_messages([changed])
    fact = original.facts[0].__class__.from_dict({**original.facts[0].to_dict(), "statement": "别的说法"})
    with pytest.raises(DuplicateMemoryId):
        fresh.import_facts(A, [fact])


def test_generic_chat_ids_are_stable_and_conflicts_are_reported(tmp_path):
    api = MemoryAPI(clock=lambda: NOW)
    source = SourceRef(source_id="chat_export", kind=SourceKind.CONVERSATION, origin=DataOrigin.SYNTHETIC, description="合成聊天")
    rows = [{"target_persona_id": "synthetic_a", "conversation_id": "c1", "speaker_id": "synthetic_b",
             "timestamp": "2025-01-01T10:00:00+08:00", "content": "我明天搬去上海"},
            {"target_persona_id": "synthetic_a", "conversation_id": "c1", "speaker_id": "synthetic_a",
             "timestamp": "2025-01-01T10:01:00", "content": "没有时区"},
            {"target_persona_id": "synthetic_a", "conversation_id": "c1", "speaker_id": "synthetic_a",
             "timestamp": "2025-01-01T10:02:00Z", "content": "好的", "message_id": "m_fixed"}]
    file = tmp_path / "chat.jsonl"
    file.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    importer = GenericConversationImporter(api, source)

    first = importer.import_file(file)
    assert (first.added_messages, first.error_count, first.fact_count) == (2, 1, 0)
    assert "no timezone" in first.errors[0]
    assert (importer.import_file(file).added_messages, api.stats()["counts"]["memories"]) == (0, 0)
    [hit] = api.search_messages("synthetic_a", "上海")
    assert hit.speaker_id == "synthetic_b" and hit.subject_id is None     # the speaker is not made the subject
    assert hit.message_id.startswith("msg.")                              # generated, deterministic

    rows[2]["content"] = "内容变了"
    with pytest.raises(MessageConflict):
        importer.import_records(rows, origin="chat.jsonl")
    assert GenericConversationImporter(api, source, default_tz=CST).import_records(rows[1:2], origin="b").added_messages == 1
    assert stable_id("x", 1, "a") == stable_id("x", 1, "a") != stable_id("x", 1, "b")


# ---------------------------------------------------------------- T7 time

def test_nothing_is_visible_before_it_was_known(api):
    during = datetime(2025, 1, 1, 6, 30, tzinfo=CST)
    after = datetime(2025, 1, 1, 7, 0, tzinfo=CST)
    word = dict(include_ended=True)
    assert api.search(A, "长跑", as_of=during, **word) == []                # happening, not known until it ended
    assert api.search(A, "长跑", as_of=datetime(2024, 12, 31, tzinfo=CST), known_at=NOW, **word) == []   # not yet happened
    assert [h.memory_id for h in api.search(A, "长跑", as_of=after, **word)] == ["lb.alpha.daily_event.1"]
    assert api.search(A, "长跑", as_of=after) == []                         # ended: it no longer holds at as_of
    # a calendar entry is known from its own creation time, before the appointment
    created = datetime(2025, 2, 1, 8, 0, tzinfo=CST)
    assert api.search(A, "体检", as_of=created - timedelta(seconds=1), known_at=created, **word) == []
    assert [h.memory_id for h in api.search(A, "体检", as_of=datetime(2025, 3, 1, 9, 30, tzinfo=CST), known_at=created)] == [
        "lb.alpha.calendar.0"]
    assert api.search_messages(A, "公园见", as_of=datetime(2025, 1, 1, 19, 59, tzinfo=CST)) == []


def test_times_keep_the_dataset_rule_not_a_hidden_shift(dataset):
    parsed = parse_persona(dataset[A], "rev1")
    sent = next(m for m in parsed.messages if m.message_id == "lb.alpha.sms.0")
    assert sent.timestamp.isoformat() == "2025-01-01T20:00:00+08:00"
    assert sent.timestamp.astimezone(timezone.utc).hour == 12


# ---------------------------------------------------------------- T8 isolation, T9 snapshot, T10 unsupported records

def test_two_personas_stay_apart(api):
    assert api.search(A, "游泳", include_ended=True) == [] and api.search(B, "长跑", include_ended=True) == []
    assert all(h.persona_id == A for h in api.search(A, "训练", include_ended=True))
    assert all(h.persona_id == B for h in api.search_messages(B, "公园"))
    assert set(all_ids(api, "SELECT DISTINCT persona_id FROM memories")) == {A, B}
    assert "lb.beta.contact.1" not in {h.speaker_id for h in api.search_messages(A, "海边")}


def test_only_chat_has_a_speaker(api):
    types = set(all_ids(api, "SELECT DISTINCT source_type FROM messages"))
    assert types == {"lifebench_sms", "lifebench_agent_chat"}
    hits = api.search(A, "体检", include_ended=True) + api.search(A, "合影", include_ended=True)
    assert hits and all(h.speaker_id is None for h in hits)


def test_a_snapshot_of_an_import_reads_back(dataset, tmp_path):
    db = tmp_path / "db" / "memory.sqlite"
    db.parent.mkdir()
    with MemoryAPI(db, clock=lambda: NOW) as api:
        LifeBenchImporter(api, "rev1").import_persona(dataset[A])
        info = api.snapshot(tmp_path / "snapshot.sqlite")
    with MemoryAPI(info.path, read_only=True, clock=lambda: NOW) as read:
        assert read.check_integrity() == []
        assert [h.memory_id for h in read.search(A, "长跑", include_ended=True)] == ["lb.alpha.daily_event.1"]
        assert read.search_messages(A, "公园见")[0].speaker_id == A


def test_import_facts_is_scoped_and_atomic(dataset):
    parsed = parse_persona(dataset[A], "rev1")
    api = MemoryAPI(clock=lambda: NOW)
    for s in parsed.sources:
        api.register_source(s)
    with pytest.raises(Exception, match="does not belong"):
        api.import_facts(B, parsed.facts[:2])
    bad = parsed.facts[1].__class__.from_dict({**parsed.facts[1].to_dict(), "memory_id": parsed.facts[0].memory_id})
    with pytest.raises(Exception):
        api.import_facts(A, [parsed.facts[0], bad])
    assert api.stats()["counts"]["memories"] == 0
    result = api.import_facts(A, parsed.facts)
    assert (result.added, result.skipped) == (len(parsed.facts), 0)
    assert api.import_facts(A, parsed.facts).added == 0


def test_include_ended_changes_nothing_else():
    """The default still means "holds at as_of"."""
    from conftest import at, make_memory
    api = MemoryAPI(clock=lambda: at(365))
    api.register_source(SourceRef(source_id="src_1", kind=SourceKind.SCRIPT, origin=DataOrigin.SYNTHETIC, description="x"))
    api.add_facts("synthetic_p", [make_memory(valid_from=at(1), valid_to=at(10), recorded_at=at(10))])
    assert api.search("synthetic_p", "Alpha") == []
    assert [h.memory_id for h in api.search("synthetic_p", "Alpha", include_ended=True)] == ["mem_1"]
    assert api.search("synthetic_p", "Alpha", as_of=at(0), known_at=at(20), include_ended=True) == []  # not started


# ---------------------------------------------------------------- selection

def stats(name, city, **kw):
    base = {"persona_id": name, "home_city": city, "first_date": "2025-01-01", "last_date": "2025-12-31", "daily_events": 1,
            "sms": 1, "agent_chat_records": 1, "calendar": 1, "note": 1, "qa_count": 5, "raw_records": 100, "change_qa_count": 1,
            "multi_day_chains": 1, "tree_link_rate": 0.5, "event_link_rate": 1.0, "event_type_counts": {"a": 1}}
    return {**base, **kw}


def test_selection_is_by_rule_and_needs_a_full_year():
    rows = [stats("p3", "x", raw_records=300), stats("p2", "x", raw_records=200, event_type_counts={"b": 1}),
            stats("p1", "y", raw_records=150, event_type_counts={"b": 1}), stats("p0", "y", last_date="2025-06-01", raw_records=999)]
    result = select_personas(rows)
    assert result["selected"] == ["p3", "p1"]                              # p0 is short; p2 shares p3's city
    assert select_personas(list(reversed(rows)))["selected"] == result["selected"]


def test_ties_break_by_persona_id():
    rows = [stats("b", "y"), stats("a", "x")]
    assert select_personas(rows)["selected"] == ["a", "b"]


def test_the_selection_is_frozen(dataset, tmp_path):
    manifest = tmp_path / "out" / "selection_manifest.json"
    first = ensure_selection(dataset["root"], manifest, "rev1")
    assert first["personas_scanned"] == 2 and len(first["selected"]) == 2
    edited = json.loads(manifest.read_text(encoding="utf-8"))
    edited["selected"] = list(reversed(edited["selected"]))
    manifest.write_text(json.dumps(edited), encoding="utf-8")
    assert ensure_selection(dataset["root"], manifest, "rev1")["selected"] == edited["selected"]     # reused as it is
    with pytest.raises(ValueError, match="reselect"):
        ensure_selection(dataset["root"], manifest, "rev2")
    write(dataset[A] / "phone_data" / "sms.json", [])
    with pytest.raises(ValueError, match="reselect"):
        ensure_selection(dataset["root"], manifest, "rev1")
