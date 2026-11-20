"""BUILD-4.1.1: the diagnostic tools, on the synthetic LifeBench fixture. The real data is not read here."""
import inspect
import json
import subprocess
from pathlib import Path

import pytest

from stillhere.evaluation import memory_bench as mb
from stillhere.evaluation import retrieval_diagnostics as rd
from stillhere.importers import LifeBenchImporter
from stillhere.memory import MemoryAPI
from test_importers import A, B, NOW, persona_files

ROOT = Path(__file__).resolve().parents[1]
WHEN = mb.query_times("2025-12-31")[0]


@pytest.fixture
def world(tmp_path):
    root = tmp_path / "data"
    api = MemoryAPI(clock=lambda: NOW)
    importer = LifeBenchImporter(api, "rev1")
    importer.import_persona(persona_files(root, "alpha"))
    importer.import_persona(persona_files(root, "beta", display="丙某", city="丙市", event_word="游泳"))
    yield api
    api.close()


def test_candidate_hit_and_coverage():
    assert rd.hit_coverage(["a", "b"], ["b", "c", "d", "e"]) == (True, 0.25)
    assert rd.hit_coverage(["a"], ["b"]) == (False, 0.0)
    assert rd.bucket(10) == "1-10" and rd.bucket(11) == "11-20" and rd.bucket(600) == "501+" and rd.bucket(None) == "no_match"
    assert rd.rank_of(["a", "b"], "b") == 2 and rd.rank_of(["a"], "z") is None


def test_the_extension_set_never_touches_the_frozen_questions():
    make = lambda p: [mb.Question(f"{p}.qa.{i:04d}", mb.PERSONA_PREFIX + p, "q", "2025-12-31", (t,), t, (), (), ())
                      for i, t in enumerate(["Single_hop"] * 70 + ["Temporal"] * 30)]
    questions = make("a") + make("b")
    dev, test, _ = mb.make_split(questions, 7)
    rest = [q for q in questions if q.qa_id not in set(dev) | set(test)]
    _, ext, _ = mb.make_split(rest, 20261106, dev=0, test=20)
    assert len(ext) == 40 and not set(ext) & (set(dev) | set(test))
    assert mb.make_split(rest, 20261106, dev=0, test=20)[1] == ext


def test_rankings_take_no_gold_and_match_the_api(world):
    for function in (rd.deep_ranking, rd.fact_sql, rd.message_sql, rd.timed_facts, rd.timed_messages):
        assert not [p for p in inspect.signature(function).parameters if "gold" in p or "event" in p]
    q = mb.Question("x", A, "长跑训练公园", "2025-12-31", (), "Other", (), (), ())
    conn = world.store._conn
    deep = {c: rd.deep_ranking(conn, A, q.question, WHEN, c) for c in ("facts", "messages")}
    assert deep["facts"].keys and deep["messages"].keys
    rd.check_same_ranking(world, q, WHEN, deep)
    with pytest.raises(AssertionError):
        rd.check_same_ranking(world, q, WHEN, {**deep, "facts": rd.Ranking(list(reversed(deep["facts"].keys)), 0)})


def test_event_groups_never_join_two_personas(world):
    conn = world.store._conn
    a, b = rd.event_groups(conn, A), rd.event_groups(conn, B)
    assert a[f"{A}:calendar:0"] == f"{A}:event:1" and b[f"{B}:calendar:0"] == f"{B}:event:1"
    assert not set(a.values()) & set(b.values())


def test_one_event_seen_through_several_sources_is_recognised(world):
    groups = rd.event_groups(world.store._conn, A)
    same = [f"{A}:daily_event:1", f"{A}:calendar:0", f"{A}:sms:0", f"{A}:agent_chat:0"]
    assert {groups[k] for k in same} == {f"{A}:event:1"}                     # facts and messages meet here
    assert rd.duplicate_slots(same, groups) == 3
    assert rd.duplicate_slots([f"{A}:contact:1", f"{A}:fitness:0"], groups) == 0   # no group never counts as a repeat
    capped = rd.simulate_cap(same + [f"{A}:contact:1"], groups, 10)
    assert capped == [f"{A}:daily_event:1", f"{A}:contact:1"]
    grouped = rd.simulate_group_fusion([same[0], same[1]], [same[2]], groups, 60)
    assert grouped == [[same[0], same[2], same[1]]]                          # every member keeps its own id


def test_stage_timing_returns_what_the_api_returns(world):
    config = {"pool": 10, "include_ended": True}
    for question in ("长跑训练", "公园"):
        items, stages = rd.timed_facts(world, A, question, WHEN, 10)
        assert [r.key for r in items] == [r.key for r in mb.search_facts(world, A, question, WHEN, config)]
        assert {"sql_first_row", "row_fetch", "load_records", "total"} <= set(stages)
        items, _ = rd.timed_messages(world, A, question, WHEN, 10)
        assert [r.key for r in items] == [r.key for r in mb.search_messages(world, A, question, WHEN, config)]


def test_a_baseline_manifest_notices_any_change(tmp_path):
    (tmp_path / "test_results.json").write_text("{}", encoding="utf-8")
    manifest = {"files": {"test_results.json": rd.sha256_file(tmp_path / "test_results.json")}}
    manifest["manifest_sha256"] = rd.manifest_digest(manifest)
    assert rd.verify_baseline(manifest, tmp_path) == []
    (tmp_path / "test_results.json").write_text('{"hit": 1}', encoding="utf-8")
    assert rd.verify_baseline(manifest, tmp_path) == ["test_results.json changed"]
    assert rd.verify_baseline({**manifest, "extra": 1}, tmp_path)[0] == "the manifest was edited"
    (tmp_path / "test_results.json").unlink()
    assert "test_results.json is missing" in rd.verify_baseline(manifest, tmp_path)


def test_no_api_key_can_reach_git():
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    values = [l.split("=", 1) for l in lines if l and not l.startswith("#")]
    assert {name for name, _ in values} >= {"DASHSCOPE_API_KEY", "STILLHERE_EMBEDDING_MODEL", "STILLHERE_RERANK_MODEL",
                                            "STILLHERE_DASHSCOPE_REGION", "STILLHERE_API_BUDGET_CNY"}
    assert all(value == "" for _, value in values)
    ignored = subprocess.run(["git", "-C", str(ROOT), "check-ignore", ".env", ".env.local"], capture_output=True, text=True)
    assert ignored.stdout.split() == [".env", ".env.local"]
    assert subprocess.run(["git", "-C", str(ROOT), "check-ignore", ".env.example"], capture_output=True).returncode == 1
