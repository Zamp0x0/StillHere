"""BUILD-4.1.3: the relation index, one bounded hop, evidence packs and MemoryAPI strategies. No network in any test."""
import ast
import dataclasses
import json
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from stillhere.contracts import EvidenceItem, ValidationError
from stillhere.importers import LifeBenchImporter
from stillhere.memory import MemoryAPI
from stillhere.retrieval import event_graph as eg
from stillhere.retrieval import profiles
from stillhere.retrieval.alibaba_client import DashScopeClient, Ledger
from stillhere.retrieval.dense import ScopeTable, corpus
from stillhere.retrieval.expansion import ExpansionConfig, expand
from stillhere.retrieval.hybrid import HybridRetriever, event_groups
from stillhere.retrieval.service import RetrievalService, enable_strategies
from test_importers import A, B, CST, NOW, persona_files

ROOT = Path(__file__).resolve().parents[1]
LATE = datetime(2026, 1, 2, tzinfo=CST)
HOP = dataclasses.replace(profiles.get("fast"), name="hop", expansion=ExpansionConfig(("same_event", "tree_root"), seeds=10, per_seed=5, global_budget=50))


@pytest.fixture
def world(tmp_path):
    root = tmp_path / "data"
    api = MemoryAPI(clock=lambda: NOW)
    importer = LifeBenchImporter(api, "rev1")
    importer.import_persona(persona_files(root, "alpha"))
    importer.import_persona(persona_files(root, "beta", display="丙某", city="丙市", event_word="游泳"))
    yield api
    api.close()


def graph_of(api, *personas):
    return eg.EventGraph(m for p in personas for m in eg.extract(api.store._conn, p))


def retriever(api, graph=None):
    conn = api.store._conn
    groups = {**event_groups(conn, A), **event_groups(conn, B)}
    return HybridRetriever(api, ScopeTable(conn, [A, B]), {d.record_id: d for p in (A, B) for d in corpus(conn, p)}, groups=groups, graph=graph)


def keys(result):
    return [c.key for c in result.final]


def test_the_index_rebuilds_byte_for_byte_and_keeps_its_provenance(world, tmp_path):
    conn = world.store._conn
    one = eg.build_index(tmp_path / "a", conn, [A, B], dataset_revision="r", snapshot_sha256="s", schema_version=2)
    two = eg.build_index(tmp_path / "b", conn, [A, B], dataset_revision="r", snapshot_sha256="s", schema_version=2)
    assert one["index_sha256"] == two["index_sha256"] and one["relation_type_counts"]["same_event"]["groups"] > 0
    assert one["edge_count"] > 0 and set(one["relation_type_counts"]) == set(eg.RELATIONS)
    graph, manifest = eg.load_graph(tmp_path / "a", snapshot_sha256="s")
    assert manifest["index_sha256"] == one["index_sha256"]
    with pytest.raises(ValueError):
        eg.load_graph(tmp_path / "a", snapshot_sha256="other")
    members = graph.groups[f"{A}:same_event:1"]
    assert all(m.provenance for m in members) and {m.key for m in members} >= {f"{A}:calendar:0", f"{A}:sms:0", f"{A}:daily_event:1"}


def test_no_relation_is_built_from_qa_data():
    for name in ("event_graph.py", "expansion.py", "service.py"):
        text = (ROOT / "src/stillhere/retrieval" / name).read_text(encoding="utf-8")
        tree = ast.parse(text)
        names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
        assert not names & {"gold", "gold_events", "required_events_id", "score_points", "answer", "evidence_keys"}, name
        assert "QA_all" not in text and "stillhere.evaluation" not in text


def test_same_event_joins_sources_and_keeps_their_ids(world):
    graph = graph_of(world, A)
    group = graph.groups[graph.groups_of(f"{A}:calendar:0", "same_event")[0]]
    keys_ = {m.key for m in group}
    assert {f"{A}:daily_event:1", f"{A}:calendar:0", f"{A}:sms:1"} <= keys_           # different sources stay different records


def test_records_of_different_events_are_not_joined(world):
    graph = graph_of(world, A)
    first = graph.groups_of(f"{A}:daily_event:1", "same_event")
    second = graph.groups_of(f"{A}:daily_event:2", "same_event")
    assert first and second and not set(first) & set(second)


def test_personas_never_share_a_group(world):
    graph = graph_of(world, A, B)
    assert all(len({m.key.split(":")[0] for m in members}) == 1 for members in graph.groups.values())
    result = retriever(world, graph).retrieve(HOP, A, "长跑训练", LATE, LATE)
    assert all(c.persona_id == A and c.key.startswith(A) for c in result.pool)


def test_a_legal_seed_never_exposes_a_neighbour_known_later(world):
    """The calendar entry of event 7-2 is created on 1 February; the seed event is known on 1 January."""
    r = retriever(world, graph_of(world, A))
    when = datetime(2025, 1, 1, 12, 0, tzinfo=CST)
    result = r.retrieve(dataclasses.replace(HOP, expansion=ExpansionConfig(("same_event",), seeds=10, per_seed=50, global_budget=500)), A, "长跑训练", when, when)
    assert f"{A}:daily_event:1" in [c.key for c in result.pool]
    assert f"{A}:calendar:0" not in [c.key for c in result.pool]
    later = datetime(2025, 3, 2, tzinfo=CST)
    found = r.retrieve(dataclasses.replace(HOP, expansion=ExpansionConfig(("same_event",), seeds=10, per_seed=50, global_budget=500)), A, "长跑训练", later, later)
    assert f"{A}:calendar:0" in [c.key for c in found.pool]


def test_include_ended_decides_for_neighbours_too(world):
    r = retriever(world, graph_of(world, A))
    open_only = dataclasses.replace(HOP, include_ended=False)
    assert all(world.store.get(c.record_id) is None or c.retrieval_origin == "direct" or c.key not in (f"{A}:daily_event:1",)
               for c in r.retrieve(open_only, A, "周末买菜", LATE, LATE).pool)
    assert f"{A}:daily_event:1" not in [c.key for c in r.retrieve(open_only, A, "长跑训练", LATE, LATE).pool]


def test_a_retracted_record_is_not_reached_afterwards_but_was_before(tmp_path):
    root = tmp_path / "data"
    now = [NOW]
    api = MemoryAPI(clock=lambda: now[0])
    LifeBenchImporter(api, "rev1").import_persona(persona_files(root, "alpha"))
    api.retract(A, "lb.alpha.calendar.0", "没有这回事")
    service = RetrievalService(api)
    expansion = dataclasses.replace(HOP, expansion=ExpansionConfig(("same_event",), seeds=10, per_seed=50, global_budget=500))
    after = NOW + timedelta(days=1)
    result = service.run(expansion, A, "长跑训练", after, after, 50, True)
    assert f"{A}:calendar:0" not in [c.key for c in result.pool]
    before = datetime(2025, 6, 1, tzinfo=CST)
    assert f"{A}:calendar:0" in [c.key for c in service.run(expansion, A, "长跑训练", before, before, 50, True).pool]


def test_budgets_bound_the_hop_and_it_never_expands_twice(world):
    graph = graph_of(world, A)
    scope = ScopeTable(world.store._conn, [A])
    mask = scope.mask(A, LATE, LATE, include_ended=True)
    seed = [(f"{A}:daily_event:1", 0.5)]
    wide = expand(graph, scope, mask, A, seed, {seed[0][0]}, {}, ExpansionConfig(("same_event",), per_seed=100, global_budget=100))
    assert len(wide) > 2
    assert len(expand(graph, scope, mask, A, seed, {seed[0][0]}, {}, ExpansionConfig(("same_event",), per_seed=2, global_budget=100))) == 2
    assert len(expand(graph, scope, mask, A, seed, {seed[0][0]}, {}, ExpansionConfig(("same_event",), per_seed=100, global_budget=1))) == 1
    assert all(e.seed_key == seed[0][0] for e in wide)                        # every path starts at a first-hop seed
    assert all(e.score <= 0.5 for e in wide)                                  # never above the seed that led to it
    assert expand(graph, scope, mask, A, seed, {seed[0][0]}, {}, ExpansionConfig(("same_event",), max_group=1)) == []


def test_the_same_evidence_appears_once_and_different_sources_stay_apart(world):
    result = retriever(world, graph_of(world, A)).retrieve(HOP, A, "长跑训练 公园", LATE, LATE)
    pool = [c.key for c in result.pool]
    assert len(pool) == len(set(pool))
    assert len({c.key.split(":")[1] for c in result.pool}) > 1


def test_no_relation_falls_back_to_the_first_hop(world):
    r = retriever(world, eg.EventGraph([]))
    assert keys(r.retrieve(HOP, A, "长跑训练", LATE, LATE)) == keys(retriever(world).retrieve(profiles.get("fast"), A, "长跑训练", LATE, LATE))


def test_a_reserved_place_goes_to_expanded_evidence_and_ranking_stays_stable(world):
    graph = graph_of(world, A, B)
    reserve = dataclasses.replace(HOP, final_k=4, expansion=ExpansionConfig(("same_event",), seeds=10, per_seed=5, global_budget=50, reserve=2))
    result = retriever(world, graph).retrieve(reserve, A, "长跑训练", LATE, LATE)
    assert [c.retrieval_origin for c in result.final].count("expanded") <= 2
    again = retriever(world, graph).retrieve(reserve, A, "长跑训练", LATE, LATE)
    assert keys(result) == keys(again)


# ---------------------------------------------------------------- evidence packs

def test_an_evidence_pack_keeps_each_record_and_its_path(world):
    service = RetrievalService(world)
    config = dataclasses.replace(HOP, expansion=ExpansionConfig(("same_event",), seeds=10, per_seed=5, global_budget=50, reserve=3))
    pack = service.evidence_pack(config, A, "长跑训练", as_of=LATE, known_at=LATE)
    expanded = [i for i in pack.items if i.retrieval_origin == "expanded"]
    assert expanded and all(i.seed_record_key and i.relation_type == "same_event" and i.path_provenance and i.source_locator for i in expanded)
    assert all(i.seed_record_key is None for i in pack.items if i.retrieval_origin == "direct")
    assert len({i.record_key for i in pack.items}) == len(pack.items)
    with pytest.raises(ValidationError):
        EvidenceItem(record_key="k", record_id="r", persona_id=A, source_id="s", text="t", final_score=0.1, retrieval_origin="expanded")


# ---------------------------------------------------------------- MemoryAPI

@pytest.fixture
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a test tried to open a network connection")
    monkeypatch.setattr(socket.socket, "connect", refuse)


def test_the_legacy_path_is_unchanged_and_stays_the_default(world):
    before = world.search(A, "长跑", include_ended=True)
    enable_strategies(world)
    assert world.search(A, "长跑", include_ended=True) == before
    assert world.search(A, "长跑", include_ended=True, strategy="legacy") == before
    assert all(h.kind.value == "memory" for h in before) and all(h.retrieval_origin is None for h in before)
    assert "retrieval_origin" not in before[0].to_dict()
    with pytest.raises(ValueError):
        world.register_strategy("legacy", lambda *a, **k: [])


def test_fast_needs_no_key_and_no_network(world, no_network):
    enable_strategies(world)
    hits = world.search(A, "长跑训练", as_of=LATE, include_ended=True, strategy="fast")
    assert hits and all(h.persona_id == A and h.retriever == "fast" and h.retrieval_origin == "direct" and h.locator for h in hits)
    assert world.search(A, "长跑训练", as_of=LATE, include_ended=True, strategy="fast_multihop", k=5)
    assert world.search("lifebench_nobody", "长跑", as_of=LATE, strategy="fast") == []


def test_quality_stays_off_without_explicit_permission(world, tmp_path, no_network):
    sent = []
    client = DashScopeClient({"DASHSCOPE_API_KEY": "sk-testvalue123456"}, Ledger(tmp_path / "l.jsonl", 1.0), transport=lambda *a: sent.append(a))
    enable_strategies(world, client=client)                                   # a configured client alone permits nothing
    assert "quality" not in world.strategies()
    with pytest.raises(ValueError, match="not enabled"):
        world.search(A, "长跑", strategy="quality")
    assert sent == []


def test_quality_exists_only_with_a_client_indexes_and_permission(world, tmp_path, no_network):
    client = DashScopeClient({"DASHSCOPE_API_KEY": "sk-testvalue123456"}, Ledger(tmp_path / "l.jsonl", 1.0), transport=lambda *a: None)
    enable_strategies(world, client=client, indexes={A: object()}, allow_external=True)
    assert world.strategies() == ["legacy", "fast", "fast_multihop", "fast_time", "quality"]


def test_strategies_keep_time_and_persona_rules(world, no_network):
    enable_strategies(world)
    early = datetime(2025, 1, 1, 6, 30, tzinfo=CST)
    assert not [h for h in world.search(A, "长跑训练", as_of=early, include_ended=True, strategy="fast") if h.memory_id == "lb.alpha.daily_event.1"]
    ended = world.search(A, "长跑训练", as_of=LATE, strategy="fast")
    assert "lb.alpha.daily_event.1" not in [h.memory_id for h in ended]       # include_ended keeps its meaning
    assert "lb.alpha.daily_event.1" in [h.memory_id for h in world.search(A, "长跑训练", as_of=LATE, include_ended=True, strategy="fast")]
    assert all(h.persona_id == B for h in world.search(B, "游泳训练", as_of=LATE, include_ended=True, strategy="fast"))


def test_a_new_write_reaches_the_next_search(world):
    service = enable_strategies(world)
    world.search(A, "长跑训练", as_of=LATE, include_ended=True, strategy="fast")
    first = service.retriever
    world.search(A, "长跑训练", as_of=LATE, include_ended=True, strategy="fast")
    assert service.retriever is first                                        # nothing changed, nothing rebuilt
    world.retract(A, "lb.alpha.note.0", "撤回")
    world.search(A, "长跑训练", as_of=LATE, include_ended=True, strategy="fast")
    assert service.retriever is not first


def test_results_repeat_with_the_frozen_configuration(world):
    enable_strategies(world)
    run = lambda: [(h.memory_id, h.message_id) for h in world.search(A, "长跑训练 公园", as_of=LATE, include_ended=True, strategy="fast_multihop")]
    assert run() == run()
    frozen = profiles.get("fast_multihop")
    assert frozen.event_weight == profiles.get("fast").event_weight == 0.1 and profiles.get("quality").event_weight == 0.5
    assert frozen.expansion == ExpansionConfig(("same_event",), seeds=20, per_seed=5, global_budget=50, reserve=3)
