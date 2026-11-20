"""BUILD-4.1.2: hybrid retrieval with a fake API. No test here makes a paid call or reads a real key."""
import ast
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from conftest import at, make_memory
from stillhere.contracts import CanonicalMessage, DataOrigin, Provenance, SourceKind, SourceRef
from stillhere.memory import MemoryAPI
from stillhere.retrieval import alibaba_client as ac
from stillhere.retrieval import dense as dn
from stillhere.retrieval.hybrid import (HybridRetriever, RetrievalConfig, apply_event_signal, evidence_key, fuse_best_rank, fuse_rrf,
                                        lexical_optimized)
from stillhere.retrieval.alibaba_client import ApiError, BudgetExceeded, DashScopeClient, Ledger, OfflineCall

ROOT = Path(__file__).resolve().parents[1]
A, B = "lifebench_alpha", "lifebench_beta"
NOW = at(400)
FLASH = "qwen3.7-text-embedding-flash"


def vector_for(text, dim=8):
    """Deterministic fake embedding: similar words share dimensions."""
    v = np.zeros(dim)
    for word, axis in (("跑步", 0), ("游泳", 1), ("体检", 2), ("南京", 3)):
        if word in text:
            v[axis] = 1.0
    v[7] = 0.01
    return (v / np.linalg.norm(v)).tolist()


class FakeServer:
    def __init__(self):
        self.calls = []
        self.fail = None
        self.always = None

    def __call__(self, url, headers, body, timeout):
        payload = json.loads(body)
        self.calls.append((url, payload))
        if self.always is not None:
            return self.always
        if self.fail:
            result, self.fail = self.fail, None
            if isinstance(result, Exception):
                raise result
            return result
        if "rerank" in url:
            docs = payload["input"]["documents"]
            order = sorted(range(len(docs)), key=lambda i: ("体检" not in docs[i], i))
            return 200, json.dumps({"output": {"results": [{"index": i, "relevance_score": 1.0 / (r + 1)} for r, i in enumerate(order)]},
                                    "usage": {"total_tokens": 100}, "request_id": "abcdefgh1234"}).encode()
        texts = payload["input"]["texts"]
        return 200, json.dumps({"output": {"embeddings": [{"text_index": i, "embedding": vector_for(t)} for i, t in enumerate(texts)]},
                                "usage": {"total_tokens": 10 * len(texts)}, "request_id": "abcdefgh1234"}).encode()


def client(tmp_path, server=None, *, budget=10.0, key="sk-testvalue123456", **kw):
    return DashScopeClient({"DASHSCOPE_API_KEY": key}, Ledger(tmp_path / "ledger.jsonl", budget), transport=server or FakeServer(), sleep=lambda s: None, **kw)


def message(mid, persona, day, content, **kw):
    return CanonicalMessage(conversation_id="c", message_id=mid, target_persona_id=persona, speaker_id="friend", timestamp=at(day), content=content,
                            source_type="synthetic_chat", source_id="src", recorded_at=at(day), **kw)


def fact(mid, persona, recorded, statement, **kw):
    return make_memory(memory_id=mid, persona_id=persona, recorded_at=recorded, statement=statement, value="v", attribute=f"a_{mid}",
                       provenance=(Provenance(source_id="src", locator="x=1"),), **kw)


class Index:
    """A small world with two personas, indexed with fake vectors."""

    def __init__(self, tmp_path):
        self.api = api = MemoryAPI(clock=lambda: NOW)
        api.register_source(SourceRef(source_id="src", kind=SourceKind.SCRIPT, origin=DataOrigin.SYNTHETIC, description="synthetic"))
        api.add_facts(A, [fact("early", A, at(1), "她在南京跑步", valid_from=at(0), valid_to=at(2)),
                          fact("late", A, at(100), "她后来在南京跑步", valid_from=at(0), valid_to=at(2)),
                          fact("checkup", A, at(1), "她去体检")])
        api.add_facts(B, [fact("other", B, at(1), "他在南京跑步")])
        api.ingest_messages([message("m_a", A, 3, "周末去游泳"), message("m_b", B, 3, "周末去游泳吧")])
        self.scope = dn.ScopeTable(api.store._conn, [A, B])
        self.cache = dn.EmbeddingCache(tmp_path / "cache.sqlite")
        self.docs = {p: dn.corpus(api.store._conn, p) for p in (A, B)}
        self.client = client(tmp_path)
        dn.embed_documents(self.client, self.cache, FLASH, 8, [d.text for ds in self.docs.values() for d in ds], workers=1, chars_per_minute=10 ** 9)
        self.dense = dn.build_index(tmp_path / "index", self.cache, self.docs, model=FLASH, dimension=8, dataset_revision="r", snapshot_sha256="s", config={})
        self.dense.bind(self.scope)
        documents = {d.record_id: d for ds in self.docs.values() for d in ds}
        self.retriever = HybridRetriever(api, self.scope, documents, indexes={A: self.dense, B: self.dense}, client=self.client, cache=self.cache)

    def run(self, question, persona=A, when=at(200), known=None, **kw):
        c = RetrievalConfig("t", dimension=8, **kw)
        return self.retriever.retrieve(c, persona, question, when, known or when)


@pytest.fixture
def world(tmp_path):
    return Index(tmp_path)


def records(result):
    return [c.record_id for c in result.final]


# ---------------------------------------------------------------- the API client

def test_the_key_never_appears_in_errors_or_the_ledger(tmp_path):
    server = FakeServer()
    server.fail = (401, json.dumps({"code": "InvalidApiKey", "message": "sk-testvalue123456 is bad"}).encode())
    c = client(tmp_path, server)
    with pytest.raises(ApiError) as caught:
        c.embed(FLASH, ["x"], text_type="query", dimension=8)
    text = str(caught.value) + (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")
    assert "sk-testvalue" not in text and "401" in text
    assert ac.redact("Authorization: Bearer abc.def sk-abcdefgh") == "Authorization: [redacted] [redacted]"
    (tmp_path / "local.env").write_text("# comment\nDASHSCOPE_API_KEY=fromfile\nSTILLHERE_API_BUDGET_CNY=\"10\"\n", encoding="utf-8")
    env = ac.read_env(tmp_path / "local.env")
    assert env["DASHSCOPE_API_KEY"] == "fromfile" and env["STILLHERE_API_BUDGET_CNY"] == "10"
    with pytest.raises(ApiError, match="not set"):
        DashScopeClient({}, Ledger(tmp_path / "other.jsonl", 1.0))


def test_the_budget_stops_a_call_before_it_is_sent(tmp_path):
    server = FakeServer()
    c = client(tmp_path, server, budget=0.0000001)
    with pytest.raises(BudgetExceeded):
        c.embed(FLASH, ["很长的文本" * 100], text_type="document", dimension=8)
    assert server.calls == []
    assert Ledger(tmp_path / "x.jsonl", 500).budget == 10.0                   # the ceiling of this build


def test_the_ledger_survives_a_restart_and_counts_uncertain_billing(tmp_path):
    server = FakeServer()
    c = client(tmp_path, server)
    c.embed(FLASH, ["甲"], text_type="query", dimension=8)
    spent = c.ledger.spent
    assert spent == pytest.approx(10 / 1e6 * 0.125)
    server.fail = TimeoutError("slow")
    c.embed(FLASH, ["乙"], text_type="query", dimension=8)                     # retried and answered; the timeout still counts
    assert c.ledger.summary()["billing_uncertain"] == 1
    again = Ledger(tmp_path / "ledger.jsonl", 10.0)
    assert again.spent == pytest.approx(c.ledger.spent) and again.spent > 2 * spent
    entry = json.loads((tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert entry["request_id"] == "abcdefgh" and entry["model_id"] == FLASH


def test_retries_are_finite_and_offline_never_calls(tmp_path):
    server = FakeServer()
    server.always = (429, b"{}")
    with pytest.raises(ApiError, match="gave up"):
        client(tmp_path, server).embed(FLASH, ["x"], text_type="query", dimension=8)
    assert len(server.calls) == ac.MAX_ATTEMPTS
    other = FakeServer()
    with pytest.raises(OfflineCall):
        client(tmp_path, other, offline=True).embed(FLASH, ["x"], text_type="query", dimension=8)
    assert other.calls == []


# ---------------------------------------------------------------- the dense index

def test_vectors_stay_aligned_with_their_records(world):
    for pid, docs in world.docs.items():
        rows = world.dense.rows[pid]
        assert sorted(rows) == sorted(d.record_id for d in docs) and world.dense.vectors[pid].shape == (len(rows), 8)
        for row, record in enumerate(rows):
            text = next(d.text for d in docs if d.record_id == record)
            assert np.allclose(world.dense.vectors[pid][row], vector_for(text))


def test_the_cache_saves_paying_twice_and_chunks_are_deterministic(world):
    before = len(world.client._transport.calls)
    texts = [d.text for ds in world.docs.values() for d in ds]
    assert dn.embed_documents(world.client, world.cache, FLASH, 8, texts, workers=1)["requests"] == 0
    assert len(world.client._transport.calls) == before
    long = "字" * 2500
    parts = dn.chunks(long)
    assert parts == dn.chunks(long) and len(parts) == 3 and all(len(p) <= dn.MAX_CHUNK_CHARS for p in parts)
    assert dn.chunks("短") == ["短"]


def test_an_index_of_another_model_or_snapshot_is_refused(world, tmp_path):
    for kw in (dict(model="other", dimension=8, snapshot_sha256="s"), dict(model=FLASH, dimension=4, snapshot_sha256="s"),
               dict(model=FLASH, dimension=8, snapshot_sha256="other")):
        with pytest.raises(ValueError):
            dn.load_index(tmp_path / "index", **kw)
    assert dn.load_index(tmp_path / "index", model=FLASH, dimension=8, snapshot_sha256="s").manifest["document_count"] == 6
    with (tmp_path / "index" / f"{A}.ids.json").open("a") as f:
        f.write(" ")
    with pytest.raises(ValueError, match="changed"):
        dn.load_index(tmp_path / "index", model=FLASH, dimension=8, snapshot_sha256="s")


# ---------------------------------------------------------------- scope

def test_dense_top_k_is_chosen_inside_the_persona(world):
    result = world.run("南京跑步", lexical=None, dense_k=5)
    assert records(result) and all(c.persona_id == A for c in result.final)
    assert "other" not in records(result)
    assert all(not k.startswith(B) for k in result.channels["dense"])


def test_scope_masks_match_the_stores_rule(world):
    conn_times = [(at(0), at(0)), (at(1), at(1)), (at(5), at(5)), (at(1), at(200)), (at(200), at(200))]
    for as_of, known in conn_times:
        for persona in (A,):
            mask = world.scope.mask(persona, as_of, known, include_ended=True)
            table = world.scope.tables[persona]["row"]
            memories = {r for r, i in table.items() if mask[i] and world.api.store.get(r)}
            expected = {r.memory_id for r in world.api.store.facts_at(persona, known_at=known) if r.valid_from is None or r.valid_from <= as_of}
            assert memories == expected, (as_of, known)


def test_nothing_known_later_is_visible_on_any_path(world):
    when = at(10)
    for mode in (dict(lexical=None), dict(lexical="optimized", dense=False, embedding_model=None), dict(lexical="v1", dense=False, embedding_model=None), dict()):
        result = world.run("南京跑步", when=when, **mode)
        assert "late" not in records(result), mode
    assert "late" in records(world.run("南京跑步", when=when, known=at(100)))


def test_a_corrected_or_retracted_fact_is_seen_as_it_was_believed():
    now = [at(1)]
    api = MemoryAPI(clock=lambda: now[0])
    api.register_source(SourceRef(source_id="src", kind=SourceKind.SCRIPT, origin=DataOrigin.SYNTHETIC, description="synthetic"))
    api.add_facts(A, [fact("old", A, at(1), "她喜欢跑步"), fact("gone", A, at(1), "她喜欢攀岩")])
    old = api.get(A, "old")
    replacement = make_memory(memory_id="new", persona_id=A, recorded_at=at(100), statement="她喜欢游泳", value="v", attribute=old.attribute,
                              provenance=old.provenance)
    api.store.correct("old", replacement, "记错了")
    now[0] = at(150)
    api.retract(A, "gone", "没有这回事")
    scope = dn.ScopeTable(api.store._conn, [A])
    def seen(day):
        mask = scope.mask(A, at(day), at(day), include_ended=True)
        return {r for r in ("old", "new", "gone") if scope.visible(A, r, mask)}
    assert seen(50) == {"old", "gone"} and seen(120) == {"new", "gone"} and seen(200) == {"new"}
    assert not scope.visible(B, "old", scope.mask(A, at(50), at(50), include_ended=True))


def test_ended_events_need_include_ended_and_never_skip_known_at(world):
    assert "early" in records(world.run("南京跑步", when=at(50)))
    assert "early" not in records(world.run("南京跑步", when=at(50), include_ended=False))
    assert "late" not in records(world.run("南京跑步", when=at(50), include_ended=True))


# ---------------------------------------------------------------- ranking

def test_rrf_adds_up_only_for_the_same_evidence():
    scores = fuse_rrf([["a", "b"], ["b", "c"]], 60)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61) and max(scores, key=scores.get) == "b"
    assert len(scores) == 3
    assert fuse_best_rank([["a", "b"], ["c"]])["a"] > fuse_best_rank([["a", "b"], ["c"]])["c"]
    assert evidence_key(A, "lb.alpha.agent_chat.7.0") == evidence_key(A, "lb.alpha.agent_chat.7.3") == f"{A}:agent_chat:7"
    assert evidence_key(A, "lb.alpha.calendar.7") != evidence_key(B, "lb.alpha.calendar.7")


def test_the_final_list_is_global_not_half_and_half(world):
    result = world.run("南京跑步", final_k=3)
    assert len(result.final) <= 3
    assert result.final[0].record_id in ("early", "late", "checkup", "m_a")


def test_the_event_signal_only_moves_records(world):
    scores = {"a": 1.0, "b": 0.9, "c": 0.8}
    groups = {"b": "e", "c": "e"}
    boosted = apply_event_signal(scores, groups, 0.5)
    assert set(boosted) == set(scores) and boosted["b"] > 0.9 and boosted["a"] == 1.0
    assert apply_event_signal(scores, groups, -0.5)["b"] < 0.9
    assert apply_event_signal(scores, groups, 0.0) == scores


def test_rerank_results_map_back_and_a_failed_call_keeps_the_fused_order(world):
    plain = world.run("南京跑步")
    ranked = world.run("南京跑步", rerank_model="qwen3-rerank", rerank_k=5)
    assert ranked.final[0].record_id == "checkup" and ranked.final[0].rerank_score == 1.0
    assert sorted(records(ranked)) == sorted(records(plain))
    world.retriever.rerank_cache.clear()
    world.client._transport.fail = (400, b'{"code":"BadRequest"}')
    fallback = world.run("南京跑步", rerank_model="qwen3-rerank", rerank_k=5)
    assert fallback.rerank_failed and records(fallback) == records(plain)


def test_the_optimized_lexical_path_falls_back_to_single_characters(world):
    mask = world.scope.mask(A, at(200), at(200), include_ended=True)
    found = lexical_optimized(world.api, world.scope, A, "体", mask, 5, "memory_fts", "memory_id")
    assert [r for r, _ in found] == ["checkup"]
    assert [r for r, _ in lexical_optimized(world.api, world.scope, A, "南京", mask, 1, "memory_fts", "memory_id")] in (["early"], ["late"])


def test_v1_stays_available_and_unchanged(world):
    result = world.run("南京跑步", lexical="v1", dense=False, embedding_model=None, final_k=10)
    hits = world.api.search(A, "南京跑步", as_of=at(200), known_at=at(200), k=50, include_ended=True)
    assert set(records(result)) >= {h.memory_id for h in hits}


def test_profiles_are_the_same_pipeline():
    from stillhere.retrieval import profiles
    quality, fast = profiles.get("quality"), profiles.get("fast")
    assert isinstance(quality, RetrievalConfig) and isinstance(fast, RetrievalConfig)
    assert quality.digest() != "" and fast.mode in ("lexical", "dense", "hybrid", "hybrid_rerank")
    assert quality.expansion is None and fast.expansion is None and fast.digest() == "23b6479228da7d5b"
    with pytest.raises(KeyError):
        profiles.get("other")


def test_no_gold_can_reach_the_runtime():
    for path in (ROOT / "src/stillhere/retrieval").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
        assert not names & {"gold", "gold_events", "gold_links", "required_events_id", "score_points", "answer"}, path
        text = path.read_text(encoding="utf-8")
        assert "QA_all" not in text and "stillhere.evaluation" not in text
