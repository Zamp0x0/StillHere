"""BUILD-4: the benchmark's scoring and bookkeeping, on the synthetic LifeBench fixture of test_importers."""
import inspect
import json
from datetime import datetime, timedelta, timezone

import pytest

from stillhere.contracts import EvidenceHit, HitKind
from stillhere.evaluation import memory_bench as mb
from stillhere.importers import LifeBenchImporter
from stillhere.memory import MemoryAPI
from test_importers import A, B, CST, NOW, persona_files, write


def rows(n, *, kinds=("Single_hop",), evidence=None, required=(), ask="2025-12-31"):
    return [{"question": f"问题{i}", "answer": "答案", "ask_time": ask, "question_type": list(kinds), "score_points": [],
             "evidence": evidence or [], "required_events_id": list(required)} for i in range(n)]


@pytest.fixture
def world(tmp_path):
    root = tmp_path / "data"
    alpha, beta = persona_files(root, "alpha"), persona_files(root, "beta", display="丙某", city="丙市", event_word="游泳")
    write(alpha / "QA_all" / "QA.json", [
        {"question": "长跑训练和体检", "answer": "答案不进入评测器", "ask_time": "2025-12-31", "question_type": ["Multi_hop", "Temporal"],
         "score_points": [], "required_events_id": ["1"], "evidence": [
             {"type": "calendar", "phone_id": 0, "daily_event_id": "1"}, {"type": "calendar", "phone_id": 0, "daily_event_id": "1"},
             {"type": "sms", "phone_id": 0, "daily_event_id": "1"}, {"type": "agent_chat", "phone_id": 0, "daily_event_id": "1"}]},
        {"question": "不存在的问题", "answer": "无法回答", "ask_time": "2025-12-31", "question_type": ["Unanswerable"],
         "score_points": [], "required_events_id": [], "evidence": []}])
    api = MemoryAPI(clock=lambda: NOW)
    importer = LifeBenchImporter(api, "rev1")
    importer.import_persona(alpha)
    importer.import_persona(beta)
    yield {"root": root, "api": api, "questions": mb.load_questions(root, "alpha")}
    api.close()


def test_the_loader_reads_official_fields_and_no_answer(world):
    q, un = world["questions"]
    assert q.qa_id == "alpha.qa.0000" and q.persona_id == A and q.primary_type == "Temporal"
    assert q.gold == (f"{A}:calendar:0", f"{A}:sms:0", f"{A}:agent_chat:0")     # repeats in the list count once
    assert q.gold_events == (f"{A}:daily_event:1",)
    assert un.unanswerable and un.gold == ()
    assert "答案" not in json.dumps([str(x) for x in world["questions"]], ensure_ascii=False)


def test_the_split_is_stratified_fixed_and_blind_to_gold():
    make = lambda persona, offset: [
        mb.Question(f"{persona}.qa.{i:04d}", mb.PERSONA_PREFIX + persona, f"q{i}", "2025-12-31", (kind,), kind, (), (), ())
        for i, kind in enumerate(["Single_hop"] * 120 + ["Multi_hop"] * 60 + ["Unanswerable"] * 20)]
    questions = make("a", 0) + make("b", 0)
    dev, test, quotas = mb.make_split(questions, 7)
    assert (len(dev), len(test), len(set(dev) & set(test))) == (20, 80, 0)
    assert sum(1 for i in dev if i.startswith("a.")) == 10 and sum(1 for i in test if i.startswith("a.")) == 40
    assert quotas["lifebench_a"]["chosen"] == {"Multi_hop": 15, "Single_hop": 30, "Unanswerable": 5}
    rare = [mb.Question(f"a.qa.{900 + i}", "lifebench_a", "q", "2025-12-31", ("Causal",), "Causal", (), (), ()) for i in range(3)]
    dev2, test2, _ = mb.make_split(make("a", 0) + rare, 7)
    assert not any(i in dev2 for i in test2) and any(i.startswith("a.qa.9") for i in test2)   # a rare type stays in test
    assert mb.make_split(list(reversed(questions)), 7)[:2] == (dev, test)
    assert mb.make_split(questions, 8)[0] != dev
    assert "gold" not in inspect.signature(mb._order).parameters


def test_a_frozen_split_refuses_changes(world):
    by_id = {q.qa_id: q for q in world["questions"]}
    manifest = mb.seal({"dataset_revision": "r1", "question_sha256": {i: q.question_sha256 for i, q in by_id.items()}})
    mb.check_split(manifest, "r1", by_id)
    with pytest.raises(ValueError, match="revision"):
        mb.check_split(manifest, "r2", by_id)
    with pytest.raises(ValueError, match="edited"):
        mb.check_split({**manifest, "dataset_revision": "r2"}, "r2", by_id)
    changed = dict(by_id)
    first = next(iter(by_id))
    changed[first] = mb.Question(first, A, "另一个问题", "2025-12-31", (), "Other", (), (), ())
    with pytest.raises(ValueError, match="differs"):
        mb.check_split(manifest, "r1", changed)


def hit(key_id, *, kind=HitKind.MEMORY, persona=A, source="lb.alpha.calendar", locator="phone_id=0", **kw):
    fields = dict(memory_id=key_id, subject_id="persona") if kind is HitKind.MEMORY else dict(message_id=key_id, speaker_id=A)
    return EvidenceHit(kind=kind, persona_id=persona, text="文本", source_id=source, score=1.0, retriever="t", locator=locator,
                       **fields, **kw)


def test_keys_are_fine_grained_not_per_file():
    assert mb.evidence_key(hit("lb.alpha.calendar.579")) == f"{A}:calendar:579"
    assert mb.evidence_key(hit("lb.alpha.daily_event.12")) == f"{A}:daily_event:12"
    assert mb.evidence_key(hit("lb.alpha.sms.3", kind=HitKind.MESSAGE)) == f"{A}:sms:3"
    two = [hit("lb.alpha.agent_chat.8.0", kind=HitKind.MESSAGE), hit("lb.alpha.agent_chat.8.5", kind=HitKind.MESSAGE)]
    assert {mb.evidence_key(h) for h in two} == {f"{A}:agent_chat:8"}          # one conversation
    assert len(mb._unique(two, "messages")) == 1                                # and it takes one place in the top K
    # the same source file, another record: not a hit
    scored = mb.score_list([f"{A}:calendar:999"], [f"{A}:calendar:579"], {}, [], [5])
    assert scored.hit[5] is False


def test_hit_coverage_and_full_recall_are_different_things():
    gold = ["g1", "g2", "g3"]
    links = {"g1": "e1", "g2": "e2", "g3": None}
    keys = ["x1", "g1", "e2", "x2", "x3", "g2", "g3"]
    s = mb.score_list(keys, gold, links, ["e9"], [5, 10])
    assert s.hit == {5: True, 10: True}
    assert s.coverage[5] == pytest.approx(1 / 3) and s.coverage[10] == 1.0
    assert s.full == {5: False, 10: True}
    assert s.event_coverage[5] == pytest.approx(2 / 3)                         # g2 through its official daily event
    assert mb.score_list(["e9"], gold, links, ["e9"], [5]).event_hit[5] is True
    assert mb.score_list(["e9"], gold, links, ["e9"], [5]).hit[5] is False      # strict never counts events
    assert mb.score_list(["x"], [], {}, [], [5]).hit[5] is False


def test_fusion_is_a_fixed_global_ranking(world):
    def item(key, channel):
        return mb.Retrieved(key, channel, 1.0, hit("lb.alpha.note.0"))
    facts = [item("a", "facts"), item("b", "facts"), item("c", "facts")]
    messages = [item("m", "messages"), item("b", "messages")]
    assert [r.key for r in mb.fuse(facts, messages, 60)] == ["b", "a", "m", "c"]
    assert [r.key for r in mb.fuse(facts, messages, 60)] == [r.key for r in mb.fuse(facts, messages, 60)]


def test_times_come_from_ask_time_and_nothing_later_is_visible(world):
    api = world["api"]
    q = mb.Question("x", A, "长跑训练", "2025-01-01", (), "Other", (), (), ())
    when, status = mb.query_times(q.ask_time)
    assert when == datetime(2025, 1, 1, 23, 59, 59, 999999, tzinfo=CST) and status == "ask_date_end_of_day"
    config = dict(mb.DEFAULT_CONFIG)
    assert config["include_ended"] is True
    found = mb.search_facts(api, A, q.question, when, config)
    assert f"{A}:daily_event:1" in [r.key for r in found]                       # ended the same morning
    assert mb.temporal_violations(found, when) == []
    early = datetime(2024, 12, 31, tzinfo=CST)
    assert mb.search_facts(api, A, q.question, early, config) == []
    assert mb.search_messages(api, A, "明天公园见", early, config) == []
    assert mb.temporal_violations(found, early)                                 # the check itself sees a violation
    # gold never reaches the search functions
    for function in (mb.search_facts, mb.search_messages):
        assert list(inspect.signature(function).parameters) == ["api", "persona_id", "question", "when", "config"]


def test_a_run_scores_isolates_and_classifies(world):
    api = world["api"]
    gold_map = mb.GoldMap(api, world["root"])
    predictions = mb.run_benchmark(api, world["questions"], gold_map, dict(mb.DEFAULT_CONFIG))
    p, un = predictions
    assert p.status == "SCORABLE" and un.status == "UNANSWERABLE_OFFICIAL" and un.scores == {}
    assert (p.persona_violations, p.temporal_violations, p.locator_problems) == (0, 0, {})
    assert p.as_of == p.known_at and p.include_ended is True
    assert all(not r["key"].startswith(B) for rows_ in p.retrieved.values() for r in rows_)
    summary = mb.summarize(predictions, mb.DEFAULT_CONFIG)
    assert summary["questions"] == 2 and summary["channels"]["combined"]["scorable_questions"] == 1
    assert summary["question_status"] == {"SCORABLE": 1, "UNANSWERABLE_OFFICIAL": 1}
    assert summary["unanswerable"]["count"] == 1
    assert summary["persona_isolation"]["violations"] == 0


def test_gold_that_cannot_be_mapped_stays_visible(world):
    api = world["api"]
    gold_map = mb.GoldMap(api, world["root"])
    assert gold_map.resolve(f"{A}:calendar:0") == (f"{A}:calendar:0", "MAPPED")
    assert gold_map.resolve(f"{A}:agent_chat:0")[1] == "MAPPED"
    assert gold_map.resolve(f"{A}:calendar:777") == (None, "NOT_IN_RAW_FILE")
    assert gold_map.resolve(f"{A}:sms:3") == (None, "NOT_IN_DATABASE")          # speaker unresolved, not imported
    q = mb.Question("y", A, "问题", "2025-12-31", ("Single_hop",), "Single_hop", (f"{A}:calendar:777",), (), ())
    [p] = mb.run_benchmark(api, [q], gold_map, dict(mb.DEFAULT_CONFIG))
    assert p.status == "GOLD_UNMAPPED" and p.scores == {} and p.gold_status == {"NOT_IN_RAW_FILE": 1}
    summary = mb.summarize([p], mb.DEFAULT_CONFIG)
    assert summary["questions"] == 1 and summary["mapping"]["questions_scorable"] == 0   # counted, not dropped


def test_an_identical_record_the_import_dropped_maps_to_the_kept_one(tmp_path):
    root = tmp_path / "data"
    d = persona_files(root, "alpha")
    path = d / "phone_data" / "note.json"
    rows_ = json.loads(path.read_text(encoding="utf-8"))
    rows_[0]["title"] = "买菜"
    write(path, rows_ + [{**rows_[0], "phone_id": 5}])
    api = MemoryAPI(clock=lambda: NOW)
    LifeBenchImporter(api, "rev1").import_persona(d)
    assert mb.GoldMap(api, root).resolve(f"{A}:note:5") == (f"{A}:note:0", "ALIASED_IDENTICAL_RECORD")


def test_source_attribution_checks_the_record_not_the_file():
    key = f"{A}:calendar:5"
    item = lambda **kw: mb.Retrieved(key, "facts", 1.0, hit("lb.alpha.calendar.5", **kw))
    assert mb.locator_error(item(locator="phone_id=5")) is None
    assert mb.locator_error(item(locator="phone_id=6")) == "WRONG_LOCATOR"     # the right file, another record
    assert mb.locator_error(item(locator=None)) == "MISSING_LOCATOR"
    assert mb.locator_error(item(locator="phone_id=5", source="lb.alpha.note")) == "INVALID_SOURCE"


def test_latency_statistics():
    stats = mb.latency_stats([float(i) for i in range(1, 101)])
    assert (stats["samples"], stats["p50_ms"], stats["p95_ms"], stats["max_ms"]) == (100, 50.5, 95.0, 100.0)
    assert mb.latency_stats([]) == {"samples": 0}
