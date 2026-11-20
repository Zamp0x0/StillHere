"""BUILD-4.1.6: the date parser, the date channel and the fast_time strategy. No network, no model, no real data."""
import ast
import dataclasses
import json
import socket
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from stillhere.memory import MemoryAPI
from stillhere.retrieval import profiles
from stillhere.retrieval import temporal_anchor as ta
from stillhere.retrieval.date_channel import date_candidates, records_in_window
from stillhere.retrieval.dense import ScopeTable, corpus
from stillhere.retrieval.hybrid import HybridRetriever, event_groups, fuse_rrf
from stillhere.retrieval.service import RetrievalService, enable_strategies
from conftest import at, make_memory
from stillhere.contracts import CanonicalMessage, DataOrigin, Provenance, SourceKind, SourceRef

ROOT = Path(__file__).resolve().parents[1]
CST = timezone(timedelta(hours=8))
ASK = datetime(2025, 12, 31, 23, 59, 59, tzinfo=CST)
CASES = json.loads((ROOT / "tests/temporal_parser_cases.json").read_text(encoding="utf-8"))
A, B = "lifebench_alpha", "lifebench_beta"
DAY = datetime(2025, 7, 30, 12, 0, tzinfo=CST)
NOW = datetime(2026, 1, 1, tzinfo=CST)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["query"][:18])
def test_parser_cases(case):
    asked = datetime.fromisoformat(case["asked"]) if case.get("asked") else ASK
    got = ta.parse_anchor(case["query"], asked)
    assert got.parse_status == case["status"]
    if case["status"] == ta.OK:
        assert got.anchor_type == case["type"]
        assert got.start.strftime("%Y-%m-%d") == case["start"] and got.end.strftime("%Y-%m-%d") == case["end"]


def test_a_window_is_half_open_in_the_questions_calendar():
    got = ta.parse_anchor("2025年7月30日发生了什么", ASK)
    assert got.start == datetime(2025, 7, 30, tzinfo=CST) and got.end == datetime(2025, 7, 31, tzinfo=CST) and got.timezone == "+08:00" and got.source_span == ("2025年7月30日",)


def test_the_calendar_decides_which_day_yesterday_is():
    asked = datetime(2025, 7, 30, 20, 0, tzinfo=timezone.utc)                     # already 7-31 in Shanghai
    assert ta.parse_anchor("昨天吃了什么", asked).start.date().isoformat() == "2025-07-29"
    assert ta.parse_anchor("昨天吃了什么", asked, CST).start.date().isoformat() == "2025-07-30"


def test_without_an_asking_moment_nothing_is_guessed():
    assert ta.parse_anchor("7月30日", None).parse_status == ta.ABSTAIN


def memory(mid, persona, statement, start, end=None, recorded=None):
    return make_memory(memory_id=mid, persona_id=persona, statement=statement, value="v", attribute=f"a_{mid}", valid_from=start, valid_to=end, recorded_at=recorded or start, provenance=(Provenance(source_id="src", locator=f"id={mid}"),))


def message(mid, persona, text, when):
    return CanonicalMessage(conversation_id="c", message_id=mid, target_persona_id=persona, speaker_id="friend", timestamp=when, content=text, source_type="synthetic_chat", source_id="src", recorded_at=when, locator=f"line={mid}")


@pytest.fixture
def world():
    api = MemoryAPI(clock=lambda: NOW)
    api.register_source(SourceRef(source_id="src", kind=SourceKind.SCRIPT, origin=DataOrigin.SYNTHETIC, description="synthetic"))
    h = lambda hour: DAY.replace(hour=hour)
    api.add_facts(A, [memory("run", A, "晚上去河边跑步八公里", h(19), h(20)), memory("eat", A, "早餐吃了粥", h(7), h(8)), memory("note", A, "笔记：今天很累", h(21)),
                      memory("next", A, "第二天去开会", h(19) + timedelta(days=1), h(20) + timedelta(days=1)), memory("late", A, "这天晚上跑步的补记", h(19), h(20), recorded=datetime(2025, 9, 1, tzinfo=CST)),
                      memory("over", A, "连续出差到外地", h(19) - timedelta(days=2), h(19) + timedelta(days=2)), memory("old", A, "以前常常跑步，也喜欢长跑，每周都跑步", datetime(2025, 1, 5, tzinfo=CST), datetime(2025, 1, 6, tzinfo=CST))])
    api.add_facts(B, [memory("other", B, "晚上去河边跑步", h(19), h(20))])
    api.ingest_messages([message("m_a", A, "刚跑完步，好累", h(20)), message("m_b", B, "刚跑完步", h(20))])
    yield api
    api.close()


def search(api, query, when=NOW, strategy="fast_time", **kw):
    return api.search(A, query, as_of=when, known_at=when, include_ended=True, strategy=strategy, **kw)


def ids(hits):
    return [h.memory_id or h.message_id for h in hits]


@pytest.fixture
def no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))


def test_the_strategy_works_through_memory_api_without_network(world, no_network):
    enable_strategies(world)
    hits = search(world, "2025年7月30日晚上跑步的情况")
    assert hits and "run" in ids(hits)[:2] and hits[0].retriever == "fast_time" and next(h for h in hits if h.memory_id == "run").locator == "id=run"
    assert "fast_time" in world.strategies() and world.strategies()[0] == "legacy"


def test_frozen_profiles_keep_their_identity():
    assert profiles.get("fast").digest() == "23b6479228da7d5b" and profiles.get("fast_multihop").digest() == "1903dcd5e65a5af4" and profiles.get("quality").digest() == "45ba0bd3228205d1"
    assert profiles.get("fast").time_anchor is None and profiles.get("fast_time").time_anchor.date_k == 20
    frozen = dataclasses.replace(profiles.get("fast_time"), name="fast", time_anchor=None)
    assert frozen.digest() == profiles.get("fast").digest()


def test_the_default_path_is_still_legacy(world):
    enable_strategies(world)
    assert world.search(A, "跑步", as_of=NOW, include_ended=True) == world.search(A, "跑步", as_of=NOW, include_ended=True, strategy="legacy")


@pytest.mark.parametrize("query", ["我喜欢什么运动", "最近的跑步情况", "7月份跑了多少", "活动前一天做了什么", "年初的跑步", "2025年2月30日跑步"])
def test_without_a_usable_date_the_result_is_fast_exactly(world, query):
    service = RetrievalService(world)
    fast = service.run(profiles.get("fast"), A, query, NOW, NOW, 10, True)
    timed = service.run(profiles.get("fast_time"), A, query, NOW, NOW, 10, True)
    assert [(c.key, c.fusion_score, c.retrieval_channels) for c in fast.pool] == [(c.key, c.fusion_score, c.retrieval_channels) for c in timed.pool]
    assert "date_anchor" not in timed.channels


def test_dated_records_of_the_window_are_added_and_fast_candidates_stay(world):
    service = RetrievalService(world)
    fast = service.run(profiles.get("fast"), A, "2025年7月30日的跑步", NOW, NOW, 10, True)
    timed = service.run(profiles.get("fast_time"), A, "2025年7月30日的跑步", NOW, NOW, 10, True)
    pool = [c.record_id for c in timed.pool]
    assert {"run", "eat", "note", "over", "m_a"} <= set(pool) and set(c.record_id for c in fast.pool) <= set(pool)
    assert "old" in pool and "next" not in timed.channels["date_anchor"]        # another date stays through FAST, and is not a date candidate
    assert not any(k.endswith(":next") for k in timed.channels["date_anchor"])


def test_the_query_topic_orders_a_day_with_many_events(world):
    service = RetrievalService(world)
    timed = service.run(profiles.get("fast_time"), A, "2025年7月30日晚上跑步", NOW, NOW, 10, True)
    assert timed.channels["date_anchor"][0].endswith("run") or timed.channels["date_anchor"][0].endswith("late")


def test_visibility_rules_come_before_the_date(world):
    enable_strategies(world)
    assert "late" not in ids(search(world, "2025年7月30日跑步", when=datetime(2025, 8, 1, tzinfo=CST)))         # known_at: not recorded yet
    assert "late" in ids(search(world, "2025年7月30日跑步", when=datetime(2025, 10, 1, tzinfo=CST)))
    assert search(world, "2025年7月30日跑步", when=datetime(2025, 7, 29, tzinfo=CST)) == [] or "run" not in ids(search(world, "2025年7月30日跑步", when=datetime(2025, 7, 29, tzinfo=CST)))   # as_of: not yet
    ended = world.search(A, "2025年7月30日跑步", as_of=NOW, known_at=NOW, include_ended=False, strategy="fast_time")
    assert "run" not in ids(ended) and "run" in ids(search(world, "2025年7月30日跑步"))


def test_as_of_and_known_at_are_separate(world):
    enable_strategies(world)
    hits = world.search(A, "2025年7月30日跑步", as_of=datetime(2025, 8, 1, tzinfo=CST), known_at=datetime(2025, 10, 1, tzinfo=CST), include_ended=True, strategy="fast_time")
    assert "late" in ids(hits)


def test_a_retracted_record_is_not_a_date_candidate(world):
    enable_strategies(world)
    assert "eat" in ids(search(world, "2025年7月30日早餐"))
    world.retract(A, "eat", "记错了")
    assert "eat" not in ids(search(world, "2025年7月30日早餐", when=NOW + timedelta(days=1)))


def test_personas_stay_apart_on_the_same_day(world):
    enable_strategies(world)
    hits = search(world, "2025年7月30日晚上跑步")
    assert all(h.persona_id == A for h in hits) and "other" not in ids(hits) and "m_b" not in ids(hits)


def test_facts_and_messages_both_come_through_with_their_locators(world):
    enable_strategies(world)
    hits = search(world, "2025年7月30日跑完步")
    assert {"run", "m_a"} <= set(ids(hits))
    assert next(h for h in hits if h.message_id == "m_a").locator == "line=m_a"


def test_a_record_found_twice_appears_once_and_adds_up():
    assert fuse_rrf([["a", "b"], ["b", "c"]], 60)["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_several_sources_of_one_day_stay_separate(world):
    enable_strategies(world)
    found = ids(search(world, "2025年7月30日跑步"))
    assert len(found) == len(set(found)) and "run" in found and "m_a" in found


def test_interval_overlap_and_undated_records():
    scope_api = MemoryAPI(clock=lambda: NOW)
    scope_api.register_source(SourceRef(source_id="src", kind=SourceKind.SCRIPT, origin=DataOrigin.SYNTHETIC, description="synthetic"))
    scope_api.add_facts(A, [memory("across", A, "跨越多日", datetime(2025, 7, 29, tzinfo=CST), datetime(2025, 8, 2, tzinfo=CST)), memory("before", A, "open start before", datetime(2025, 7, 1, tzinfo=CST)), memory("edge", A, "ends at midnight", datetime(2025, 7, 29, 22, tzinfo=CST), datetime(2025, 7, 30, tzinfo=CST)), make_memory(memory_id="none", persona_id=A, statement="没有时间", value="v", attribute="a_none", valid_from=None, valid_to=None, recorded_at=datetime(2025, 7, 1, tzinfo=CST), provenance=(Provenance(source_id="src", locator="x"),))])
    scope = ScopeTable(scope_api.store._conn, [A])
    mask = scope.mask(A, NOW, NOW, include_ended=True)
    anchor = ta.parse_anchor("2025年7月30日", ASK)
    rows = records_in_window(scope, A, mask, anchor)
    names = [list(scope.tables[A]["row"])[r] for r in rows.tolist()]
    assert names == ["across"]
    scope_api.close()


def test_the_budget_is_a_ceiling(world):
    scope = ScopeTable(world.store._conn, [A])
    documents = {d.record_id: d for d in corpus(world.store._conn, A)}
    mask = scope.mask(A, NOW, NOW, include_ended=True)
    found, total = date_candidates(scope, documents, A, mask, ta.parse_anchor("2025年7月30日跑步", ASK), 2)
    assert len(found) == 2 and total >= 5


def test_a_persona_timezone_reaches_the_channel(world):
    tokyo = enable_strategies(world, calendar=timezone(timedelta(hours=9)))
    asked = datetime(2025, 7, 30, 16, 0, tzinfo=timezone.utc)                      # 7-31 01:00 in Tokyo
    assert tokyo.explain(profiles.get("fast_time"), A, "昨天跑步", as_of=asked, known_at=asked)["temporal_parse"]["start"].startswith("2025-07-30")


def test_explain_reports_the_path_without_any_official_field(world):
    service = enable_strategies(world)
    out = service.explain(profiles.get("fast_time"), A, "2025年7月30日跑步", as_of=NOW, known_at=NOW)
    assert out["temporal_parse"]["anchor_type"] == "FULL_DATE" and out["date_candidate_ids"] and any("date_anchor" in r["source_channels"] for r in out["final"])
    assert "gold" not in json.dumps(out) and "official" not in json.dumps(out)


def test_runtime_reads_no_official_field():
    for name in ("temporal_anchor.py", "date_channel.py"):
        tree = ast.parse((ROOT / "src/stillhere/retrieval" / name).read_text(encoding="utf-8"))
        names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        text = (ROOT / "src/stillhere/retrieval" / name).read_text(encoding="utf-8")
        assert not names & {"gold", "qa_id", "score_points", "required_events_id", "official_answer", "event_id"} and "lifebench" not in text.lower() and "phone_id" not in text


def test_quality_still_needs_permission(world):
    enable_strategies(world)
    assert "quality" not in world.strategies()


def test_the_answer_test_is_skipped_without_room_on_the_gpu():
    sys.path.insert(0, str(ROOT / "scripts"))
    import evaluate_time_anchor as et
    assert et.may_answer(None) is False and et.may_answer(et.MIN_FREE_MIB - 1) is False and et.may_answer(et.MIN_FREE_MIB) is True


def test_answer_questions_are_chosen_without_results():
    sys.path.insert(0, str(ROOT / "scripts"))
    import evaluate_time_anchor as et
    rows = [{"qa_id": f"q{i}", "group": g} for i, g in enumerate("AAAABBBCCC")]
    first = et.answer_questions(rows, a=2, b=1, c=1)
    assert first == et.answer_questions(list(reversed(rows)), a=2, b=1, c=1) and len(first) == 4
    assert et.group_of(ta.OK) == "A" and et.group_of(ta.NO_ANCHOR) == "B" and et.group_of(ta.UNSUPPORTED) == "C" and et.group_of(ta.ABSTAIN) == "C"
