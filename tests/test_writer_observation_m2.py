"""BUILD-M2: Memory Writer and Observations. Deterministic: the gold of the synthetic dataset stands in for the model (reference
extractor), and fake calls test the schema, repair, budget and retry guards. The real DeepSeek extraction is measured by
scripts/run_writer_m2.py, not here."""
import copy
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from stillhere.context import ContextOrchestrator, ContextRequest
from stillhere.contracts import MemoryStatus
from stillhere.evaluation import writer_dataset as wd
from stillhere.evaluation import writer_eval as we
from stillhere.memory import MemoryAPI
from stillhere.memory.schema import SCHEMA_VERSION
from stillhere.memory.sqlite_store import PersonaScopeError, WriteRejected
from stillhere.observation.context import for_context
from stillhere.observation.engine import ObservationEngine, dependency_audit
from stillhere.writer import contracts as wc
from stillhere.writer import dates as dt
from stillhere.writer import extract as ex
from stillhere.writer.pipeline import Writer, pending_index_ids
from stillhere.writer.verify import verify

ROOT = Path(__file__).resolve().parents[1]
TZ = wd.TZ


@pytest.fixture(scope="module")
def world():
    return we.run(we.new_world(we.reference_complete(wd.conversations()), "reference"))


@pytest.fixture(scope="module")
def scores(world):
    return we.score(world), we.score_observations(world)


def gm(label, mid, speaker, text, at, sender="本人", recipient="AI 助手", persona=wd.PA):
    return ex.GroupMessage(label, mid, speaker, sender, recipient, at, at, text, wd.CHAT)


def cand(**kw):
    base = dict(candidate_id="c1", persona_id=wd.PA, type="FACT", subject="self", predicate="lives_in", object="杭州", source_message_ids=["m.0"], source_spans=["我住在杭州"])
    base.update(kw)
    return wc.Candidate(**base)


GROUP = [gm("m1", "m.0", wd.PA, "我住在杭州，不太喜欢咖啡。", wd.t(10, 1, 9)), gm("m2", "m.1", wd.BOT[wd.PA], "你喜欢科幻片吧。", wd.t(10, 1, 9, 2), "AI 助手", "本人")]


# ---------------------------------------------------------------- 1 schema, 32-33 guards

def test_candidate_schema_validation():
    labels = {"m1": "a", "m2": "b"}
    good = {"candidates": [{"type": "FACT", "subject": "self", "predicate": "lives_in", "object": "杭州", "source_message_ids": ["m1"], "source_spans": ["杭州"]}]}
    assert len(ex.validate(good, labels)[0]) == 1
    for bad in ({"type": "FACT", "subject": "self", "predicate": "invented", "object": "x", "source_message_ids": ["m1"], "source_spans": ["x"]},
                {"type": "FACT", "subject": "someone", "predicate": "job", "object": "x", "source_message_ids": ["m1"], "source_spans": ["x"]},
                {"type": "FACT", "subject": "self", "predicate": "job", "object": "x", "source_message_ids": ["m9"], "source_spans": ["x"]},
                {"type": "EVENT", "subject": "self", "predicate": "event", "object": "x", "source_message_ids": ["m1"], "source_spans": ["x"]},
                {"type": "FACT", "subject": "self", "predicate": "job", "object": "x", "event_status": "DONE", "source_message_ids": ["m1"], "source_spans": ["x"]}):
        valid, problems = ex.validate({"candidates": [bad]}, labels)
        assert not valid and problems
    assert ex.validate([], labels)[1]


def test_invalid_output_is_repaired_once_then_given_up(tmp_path):
    calls = []

    def fake(messages):
        calls.append(len(messages))
        return {"text": "not json", "input_tokens": 1, "output_tokens": 1, "cost_cny": 0.0, "latency_ms": 1.0, "model": "fake"}
    cands, rec = ex.extract(fake, "fake", wd.PA, "林晓", GROUP, [], ex.ExtractionCache(tmp_path / "c.jsonl"))
    assert cands == [] and len(calls) == 2 and calls[1] == 4 and rec["schema_ok"] is False      # one repair turn, no third call
    again, _ = ex.extract(fake, "fake", wd.PA, "林晓", GROUP, [], ex.ExtractionCache(tmp_path / "c.jsonl"))
    assert len(calls) == 2                                     # cached: never paid twice


def test_budget_guard_refuses_new_work(tmp_path):
    from stillhere.evaluation import reader_research as rr
    from stillhere.retrieval.alibaba_client import BudgetExceeded
    lg = rr.ProviderLedger(tmp_path / "l.jsonl", 0.1, "deepseek")
    with pytest.raises(BudgetExceeded):
        lg.require_room(0.2)
    lg.require_room(0.05)


def test_retry_guard_in_the_client(monkeypatch):
    from stillhere.evaluation import reader_research as rr
    from stillhere.retrieval import deepseek_client as dc
    calls = []

    def transport(url, headers, body, timeout):
        calls.append(1)
        return 500, b'{"error": {"code": "server"}}'
    import tempfile
    lg = rr.ProviderLedger(Path(tempfile.mkdtemp()) / "l.jsonl", 1.0, "deepseek")
    c = dc.DeepSeekClient({"DEEPSEEK_API_KEY": "x" * 20}, lg, transport=transport, sleep=lambda s: None)
    with pytest.raises(dc.ApiError):
        c.chat("deepseek-flash", [{"role": "user", "content": "hi"}], max_tokens=10, json_mode=True)
    assert len(calls) == dc.MAX_ATTEMPTS


def test_json_mode_is_opt_in_and_old_requests_are_byte_identical():
    from stillhere.retrieval.deepseek_client import DeepSeekClient
    old = DeepSeekClient.payload("deepseek-flash", [{"role": "user", "content": "x"}], max_tokens=5, temperature=0.0)
    assert "response_format" not in old
    assert DeepSeekClient.payload("deepseek-flash", [], max_tokens=5, temperature=0.0, json_mode=True)["response_format"] == {"type": "json_object"}


def test_secret_redaction():
    from stillhere.retrieval.alibaba_client import redact
    fake = "sk-" + "a" * 24                                   # built at run time, so no key-shaped literal sits in the repository
    assert fake not in redact(f"key {fake} failed")


# ---------------------------------------------------------------- 3, 6, 7, 20 verification

def test_span_must_be_verbatim():
    assert verify(cand(source_spans=["我住在上海"]), GROUP, wd.PA, []).support == wc.UNSUPPORTED


def test_owner_facts_must_be_said_by_the_owner():
    v = verify(cand(predicate="likes", object="科幻片", source_message_ids=["m.1"], source_spans=["你喜欢科幻片吧"]), GROUP, wd.PA, [])
    assert v.support == wc.UNSUPPORTED and "did not say" in v.reasons[0]


def test_negation_is_preserved_and_normalised():
    v = verify(cand(predicate="likes", object="咖啡", source_spans=["不太喜欢咖啡"]), GROUP, wd.PA, [])
    assert v.committable and v.candidate.predicate == "dislikes"
    g = [gm("m1", "m.0", wd.PA, "我没有养狗。", wd.t(10, 1))]
    assert verify(cand(predicate="pet", object="狗", polarity="negative", source_spans=["我没有养狗"]), g, wd.PA, []).support == wc.AMBIGUOUS


def test_inferred_uncertain_and_unknown_entities_are_never_written():
    assert verify(cand(assertion_type="inferred"), GROUP, wd.PA, []).support == wc.UNSUPPORTED
    g = [gm("m1", "m.0", wd.PA, "我可能住在杭州。", wd.t(10, 1))]
    assert verify(cand(source_spans=["我可能住在杭州"]), g, wd.PA, []).support == wc.AMBIGUOUS
    assert verify(cand(subject="person:老张"), GROUP, wd.PA, []).support == wc.UNSUPPORTED
    assert verify(cand(object="北京"), GROUP, wd.PA, []).support == wc.UNSUPPORTED
    g2 = [gm("m1", "m.0", wd.PA, "她在上海读大学。", wd.t(10, 1))]
    assert verify(cand(subject="self", object="上海", source_spans=["她在上海读大学"]), g2, wd.PA, []).support == wc.UNSUPPORTED   # lives_in without living words


def test_unsupported_fields_are_dropped_not_guessed():
    g = [gm("m1", "m.0", wd.PA, "我打算去日本旅行。", wd.t(11, 2))]
    v = verify(cand(type="PLAN", predicate="event", object="去日本旅行", event_key="japan", event_status="COMPLETED", planned_time="2026-11-10", source_spans=["我打算去日本旅行"]), g, wd.PA, [])
    assert v.support == wc.PARTIALLY_SUPPORTED and {"event_status", "planned_time"} <= set(v.dropped_fields)


def test_plan_is_not_completed():
    g = [gm("m1", "m.0", wd.PA, "明天我打算去爬山了。", wd.t(10, 14))]
    v = verify(cand(predicate="event", object="爬山", event_key="hike", event_status="COMPLETED", source_spans=["明天我打算去爬山了"]), g, wd.PA, [])
    assert v.candidate.event_status is None


def test_dates_are_checked_against_the_words():
    said = datetime(2026, 11, 2, 10, tzinfo=TZ)                 # a Monday
    assert dt.supported_date("2026-11-03", "明天去", said) and dt.supported_date("2026-11-04", "这周三", said)
    assert dt.supported_date("2026-11-11", "下周三", said) and dt.supported_date("2026-11-10", "11月10日", said)
    assert dt.supported_date("2026-12", "下个月", said) and not dt.supported_date("2026-11-05", "明天去", said)
    assert dt.dayparts_in("今晚读书") == {"晚上"}


# ---------------------------------------------------------------- 2, 4, 5 provenance, persona, attribution

def test_every_writer_memory_cites_raw_messages_and_raw_stays(world):
    conn = world.api.store._conn
    assert conn.execute("SELECT COUNT(*) FROM memories m WHERE m.memory_id LIKE 'w.%' AND NOT EXISTS (SELECT 1 FROM provenance p WHERE p.memory_id = m.memory_id AND p.locator LIKE 'message:%')").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == wd.manifest()["messages"]


def test_store_refuses_a_citation_of_another_personas_message(world):
    rec = world.api.store.get(next(r[0] for r in world.api.store._conn.execute("SELECT memory_id FROM memories WHERE persona_id = ? AND memory_id LIKE 'w.%'", (wd.PB,))))
    forged = copy.deepcopy(rec)
    from dataclasses import replace
    forged = replace(rec, memory_id="w.forged", persona_id=wd.PA, conflicts_with=(), supersedes=None, superseded_by=None, status=MemoryStatus.ACTIVE)
    with pytest.raises(WriteRejected):
        world.api.add_facts(wd.PA, [forged])


def test_no_fact_crosses_personas(scores):
    s, _ = scores
    assert s["PERSONA_LEAKAGE"] == 0 and s["ENTITY_ATTRIBUTION_ACCURACY"] == 1.0


def test_group_carries_sender_and_recipient(world):
    msgs = world.writer.group(wd.PA, wd.group_id(next(c for c in wd.conversations() if c.cid == "a10")))
    first = msgs[0]
    assert (first.sender, first.recipient) == ("李雷", "本人")


def test_other_peoples_plans_keep_their_actor(world):
    conn = world.api.store._conn
    actors = {v for (v,) in conn.execute("SELECT value FROM memories WHERE persona_id = ? AND attribute = 'event.actor'", (wd.PA,))}
    assert "person:李雷" in actors and "self" in actors


# ---------------------------------------------------------------- 9-12, 16, 17, 19 lifecycle

def test_reference_pipeline_meets_every_target(scores):
    s, o = scores
    assert s["FACT_PRECISION"] == 1.0 and s["FACT_RECALL"] == 1.0 and s["SOURCE_BINDING_ACCURACY"] == 1.0 and s["EVENT_STATE_ACCURACY"] == 1.0
    assert s["DUPLICATE_RATE"] == 0 and s["FALSE_MERGE_RATE"] == 0 and s["MISSED_CORRECTION_RATE"] == 0 and s["RETRACTION_ERROR_RATE"] == 0
    assert s["UNSUPPORTED_FACTS_COMMITTED"] == 0 and s["TEMPORAL_LEAKAGE"] == 0 and s["SILENT_SOURCE_LOSS"] == 0 and not s["FORBIDDEN_FACTS_COMMITTED"]


@pytest.mark.parametrize("name", [c.name for c in wd.checkpoints()])
def test_bitemporal_checkpoint(scores, name):
    s, _ = scores
    cp = next(c for c in s["CHECKPOINT_DETAIL"] if c["name"] == name)
    assert cp["ok"], cp


def test_plan_reschedule_cancel_history_is_kept(world):
    conn = world.api.store._conn
    statuses = [v for (v,) in conn.execute("SELECT value FROM memories WHERE persona_id = ? AND attribute = 'event.status' AND subject IN "
                                           "(SELECT subject FROM memories WHERE attribute = 'event.description' AND value LIKE '%日本%') ORDER BY recorded_at_utc", (wd.PA,))]
    assert statuses[0] == "PLANNED" and statuses[-1] == "CANCELLED" and "COMPLETED" not in statuses


def test_correction_supersedes_and_keeps_history(world):
    rows = world.api.store._conn.execute("SELECT value, status FROM memories WHERE persona_id = ? AND subject = 'person:小雨' AND attribute = 'school' ORDER BY recorded_at_utc", (wd.PA,)).fetchall()
    assert [r[1] for r in rows] == ["superseded", "active"] and "南京" in rows[-1][0]


def test_temporal_change_closes_the_old_value(world):
    rows = world.api.store._conn.execute("SELECT value, status, valid_to FROM memories WHERE persona_id = ? AND subject = 'persona' AND attribute = 'lives_in' AND status = 'active' ORDER BY valid_from_utc", (wd.PA,)).fetchall()
    assert "滨江" in rows[0][0] and rows[0][2] is not None and "西湖" in rows[-1][0] and rows[-1][2] is None


def test_retraction_removes_the_whole_event(world):
    rows = world.api.store._conn.execute("SELECT status FROM memories WHERE persona_id = ? AND subject IN (SELECT subject FROM memories WHERE value = '陶艺课')", (wd.PA,)).fetchall()
    assert rows and all(r[0] == "retracted" for r in rows)


def test_contradiction_keeps_both_and_records_the_conflict(tmp_path):
    w = we.new_world(we.reference_complete(wd.conversations()), "reference", observations=False)
    c1 = wd.C("x1", wd.PA, wd.t(10, 1, 9), [("u", "我在一家医院工作。")], [wd.Fact("x1", "self", "works_at", "一家医院", (0,), ("在一家医院工作",))])
    c2 = wd.C("x2", wd.PA, wd.t(10, 2, 9), [("u", "我在一家银行工作。")], [wd.Fact("x2", "self", "works_at", "一家银行", (0,), ("在一家银行工作",))])
    for c in (c1, c2):
        w.api.ingest_messages(wd.raw_messages(c))
    w.writer.complete = we.reference_complete([c1, c2])
    we.run(w, [c1, c2])
    rows = w.api.store._conn.execute("SELECT memory_id, status FROM memories WHERE persona_id = ? AND attribute = 'works_at'", (wd.PA,)).fetchall()
    assert len(rows) == 2 and all(r[1] == "active" for r in rows)
    assert w.api.store._conn.execute("SELECT COUNT(*) FROM memory_conflicts").fetchone()[0] == 1


def test_near_duplicates_are_restatements(world):
    rows = world.api.store._conn.execute("SELECT value FROM memories WHERE persona_id = ? AND attribute = 'pet' AND status = 'active'", (wd.PA,)).fetchall()
    assert len(rows) == 1


# ---------------------------------------------------------------- 13-15 idempotency and atomicity

def test_rerun_writes_nothing(world):
    before = world.api.stats()["counts"]
    n_batches = world.api.store._conn.execute("SELECT COUNT(*) FROM writer_batches").fetchone()[0]
    for conv in wd.order():
        known = max(m.recorded_at or m.at for m in conv.messages)
        r = world.writer.write_group(conv.persona, wd.group_id(conv), until=known)
        assert r.skipped
    assert world.api.stats()["counts"] == before and world.api.store._conn.execute("SELECT COUNT(*) FROM writer_batches").fetchone()[0] == n_batches


def test_failed_group_rolls_back_entirely():
    w = we.new_world(we.reference_complete(wd.conversations()), "reference", observations=False)
    conv = next(c for c in wd.conversations() if c.cid == "a01")
    w.clock.now = conv.messages[-1].at + timedelta(seconds=1)
    before = w.api.stats()["counts"]
    with pytest.raises(RuntimeError):
        w.writer.write_group(conv.persona, wd.group_id(conv), fail_after=2)
    assert w.api.stats()["counts"] == before
    assert w.api.store._conn.execute("SELECT COUNT(*) FROM writer_batches").fetchone()[0] == 0
    r = w.writer.write_group(conv.persona, wd.group_id(conv))        # resumes cleanly
    assert not r.skipped and w.api.stats()["counts"]["memories"] > before["memories"]


def test_store_transaction_is_atomic():
    api = MemoryAPI(":memory:")
    wd.load(api)
    with pytest.raises(ZeroDivisionError):
        with api.transaction():
            api.ingest_messages([wd.raw_messages(wd.conversations()[0])[0].__class__(**{**wd.raw_messages(wd.conversations()[0])[0].__dict__, "message_id": "extra.0"})])
            assert api.store._conn.execute("SELECT COUNT(*) FROM messages WHERE message_id = 'extra.0'").fetchone()[0] == 1
            1 / 0
    assert api.store._conn.execute("SELECT COUNT(*) FROM messages WHERE message_id = 'extra.0'").fetchone()[0] == 0


def test_retract_at_respects_the_record_and_now():
    w = we.run(we.new_world(we.reference_complete(wd.conversations()), "reference", observations=False), [c for c in wd.order() if c.cid == "a01"])
    mid = w.api.store._conn.execute("SELECT memory_id FROM memories WHERE attribute = 'job'").fetchone()[0]
    with pytest.raises(WriteRejected):
        w.api.retract(wd.PA, mid, "test", at=wd.t(9, 1))           # before it was recorded
    with pytest.raises(WriteRejected):
        w.api.retract(wd.PA, mid, "test", at=wd.t(12, 1))          # after now (the clock is at the group's knowledge time)
    w.clock.now = wd.t(10, 2)
    w.api.retract(wd.PA, mid, "test", at=wd.t(10, 1, 9, 5))
    assert w.api.get(wd.PA, mid, known_at=wd.t(10, 1, 9, 4)).is_active and not w.api.get(wd.PA, mid).is_active


# ---------------------------------------------------------------- 21-28 observations

@pytest.mark.parametrize("name", [x.name for x in wd.observation_expectations()])
def test_observation_expectation(scores, name):
    _, o = scores
    row = next(r for r in o["expectations"] if r["name"] == name)
    assert row["ok"], row


def test_observation_versions_and_lifecycle(world):
    actions = [c["action"] for c in world.observation_changes]
    assert {"CREATE", "UPDATE", "INVALIDATE", "RETRACT"} <= set(actions)
    conn = world.api.store._conn
    evening = conn.execute("SELECT version, status, valid_to FROM observations WHERE rule_key = 'HABIT:读书:晚上' ORDER BY version").fetchall()
    assert evening[-1][1] == "invalidated" and evening[-1][2] is not None and all(v[1] == "superseded" for v in evening[:-1])


def test_observations_are_derived_and_acyclic(world, scores):
    _, o = scores
    audit = dependency_audit(world.api)
    assert audit["cycles"] == 0 and audit["links_to_missing_memory"] == 0 and o["UNSUPPORTED_OBSERVATIONS_PROMOTED"] == 0
    conn = world.api.store._conn
    assert {r[0] for r in conn.execute("SELECT DISTINCT authority FROM observations")} == {"DERIVED_OBSERVATION"}
    obs_ids = {r[0] for r in conn.execute("SELECT observation_id FROM observations")}
    assert not obs_ids & {r[0] for r in conn.execute("SELECT memory_id FROM memories")}        # an observation is never a memory


def test_observation_never_reinforces_itself(world):
    with pytest.raises(Exception):
        world.api.store._conn.execute("INSERT INTO observation_links (observation_id, version, memory_id, role) VALUES ('obs.x', 1, 'obs.y', 'support')")
        world.api.store._conn.commit()
    world.api.store._conn.rollback()


def test_observation_engine_needs_its_flag():
    with pytest.raises(RuntimeError):
        ObservationEngine(MemoryAPI(":memory:"), env={})
    with pytest.raises(RuntimeError):
        Writer(MemoryAPI(":memory:"), complete=lambda m: None, model="x", persona_names={}, env={})


def test_no_observation_from_two_events_or_one_statement(scores):
    _, o = scores
    assert o["OVERGENERALIZATION_RATE"] == 0.0


# ---------------------------------------------------------------- 29-31, 18 integration

def test_observations_reach_the_orchestrator_as_derived_only(world):
    known = wd.t(11, 8)
    hits, items = for_context(world.api, world.engine, wd.PA, known, known)
    r = ContextOrchestrator(world.api, speakers=wd.SPEAKERS).assemble(ContextRequest(persona_id=wd.PA, question="我有什么习惯？", as_of=known, known_at=known, hits=tuple(hits), derived=tuple(items), allow_derived=True))
    derived = [p for p in r.provenance_map.values() if p["section"] == "derived"]
    assert r.status == "OK" and derived and all(p["authority"] == "derived_below_raw" for p in derived)
    assert "【推导内容】（不是原始记录）" in r.assembled_context
    off = ContextOrchestrator(world.api, speakers=wd.SPEAKERS).assemble(ContextRequest(persona_id=wd.PA, question="我有什么习惯？", as_of=known, known_at=known, hits=tuple(hits), derived=tuple(items)))
    assert not [p for p in off.provenance_map.values() if p["section"] == "derived"]      # default: derived stays out


def test_writer_memories_are_found_by_legacy_search(world):
    hits = world.api.search(wd.PA, "西湖区", as_of=wd.t(11, 20), k=5)
    assert any((h.memory_id or "").startswith("w.") for h in hits)
    assert world.api.strategies() == ["legacy"]


def test_dense_index_gap_is_explicit(world):
    pending = pending_index_ids(world.api, wd.PA, [])
    assert pending and pending_index_ids(world.api, wd.PA, pending) == []


def test_core_schema_version_is_unchanged():
    assert SCHEMA_VERSION == 2


def test_m1_boundary_semantics_untouched():
    from stillhere.context import boundary as bd
    asked = datetime(2026, 11, 20, 12, tzinfo=TZ)
    b = bd.parse_boundary("11月5日之前", asked)
    assert (b.operator, b.upper) == (bd.BEFORE, datetime(2026, 11, 5, tzinfo=TZ))


# ---------------------------------------------------------------- 35-36 privacy and snapshots

def test_privacy_guard_on_m2_files():
    files = [str(p.relative_to(ROOT)) for d in ("writer", "observation") for p in (ROOT / "src" / "stillhere" / d).glob("*.py")]
    files += ["src/stillhere/memory/extension.py", "src/stillhere/evaluation/writer_dataset.py", "src/stillhere/evaluation/writer_eval.py", "scripts/run_writer_m2.py", "scripts/validate_m1_reader.py"]
    r = subprocess.run([sys.executable, "scripts/privacy_guard.py", *files], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_dataset_is_synthetic_and_large_enough():
    m = wd.manifest()
    assert m["origin"] == "synthetic" and 100 <= m["messages"] <= 300 and len(m["personas"]) == 3
    need = {"plan", "plan_change", "plan_cancel", "plan_to_done", "correction", "retraction", "negation", "uncertainty", "late_arriving", "sender_recipient", "entity",
            "same_event_repeated", "habit", "habit_change", "dependency_invalidation", "persona_isolation", "multi_fact", "temporal_change", "trend", "preference_repeat", "relationship"}
    assert need <= set(m["tags"])


def test_frozen_snapshots_did_not_move():
    def rev(name):
        r = subprocess.run(["git", "rev-parse", f"{name}^{{commit}}"], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    if rev("HEAD") is None:
        pytest.skip("not a git checkout")
    assert rev("memory-reader-deepseek-v3.3-audited-snapshot") == "82e5c241526c941c99ed8ea003b2a26f0b7afe52"
    assert rev("memory-retrieval-v3.2-research-snapshot") == "7bfaaf7d10924c80be96e44f1947cad907a2e00c"
    assert rev("memory-retrieval-v2-baseline") == "984bc260833dff9ec5978c1f51b342e3331184ed"
    assert rev("memory-retrieval-v1") == "61a26c008b35f162948f1388a03a7866382c4d32"
