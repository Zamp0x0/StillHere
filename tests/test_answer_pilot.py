"""BUILD-4.1.4: the answer pilot with a fake model, a fake judge and a fake API. No test here makes a paid call."""
import ast
import dataclasses
import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest

from stillhere.evaluation import answer_analysis as an
from stillhere.evaluation import answer_judge as aj
from stillhere.evaluation import answer_pilot as ap
from stillhere.retrieval import profiles
from stillhere.retrieval.alibaba_client import BudgetExceeded, OfflineCall
from stillhere.retrieval.hybrid_keys import evidence_key
from stillhere.retrieval.service import enable_strategies
from test_hybrid_retrieval import A, B, FLASH, FakeServer, Index, at, client

ROOT = Path(__file__).resolve().parents[1]
WHEN = at(200)
JUDGE = "qwen3.7-flash-2026-07-15"
TEMPLATE = "{question}|{ask_time}|{question_types}|{reference_answer}|{score_points}|{generated_answer}"
PROMPTS = aj.JudgePrompts("评分员", TEMPLATE, "x")


class Chat(FakeServer):
    def __call__(self, url, headers, body, timeout):
        if "chat/completions" not in url:
            return super().__call__(url, headers, body, timeout)
        self.calls.append((url, json.loads(body)))
        return 200, json.dumps({"id": "chatcmpl-1", "model": JUDGE, "choices": [{"message": {"content": '{"label": "CORRECT"}'}}], "usage": {"prompt_tokens": 1000, "completion_tokens": 500}}).encode()


class FakeProvider:
    model_id = "fake-model"
    config = {"do_sample": False}

    def __init__(self, text='{"answer": "去跑步了", "abstain": false, "citations": ["E01"]}'):
        self.text = text
        self.calls = []

    def generate(self, messages, max_new_tokens):
        self.calls.append(messages)
        return {"text": self.text, "input_tokens": 10, "output_tokens": 5, "latency_ms": 1.0}


@pytest.fixture
def world(tmp_path):
    w = Index(tmp_path)
    service = enable_strategies(w.api, client=w.client, indexes={A: w.dense, B: w.dense}, cache=w.cache, allow_external=True)
    w.api.register_strategy("quality", service.strategy(dataclasses.replace(profiles.get("quality"), parallel=False, dimension=8)))
    return w


def evidence(key="p:sms:1", text="内容", **kw):
    fields = dict(key=key, record_id="r", persona_id="p", source_id="lb.p.sms", source_type="sms", text=text, score=0.1, locator="phone_id=1", event_time="2025-01-02T03:04:00+08:00", known_at=None)
    return ap.Evidence(**{**fields, **kw})


def row(qa, arm, label="CORRECT", **kw):
    fields = dict(qa_id=qa, arm_id=arm, persona_id="p", primary_type="Single_hop", unanswerable=False, label=label, votes={"correct": 3, "wrong": 0}, abstain=False, keys=[], gold=[], coverage=0.0, full=False, grounding=None, sufficiency=None,
                  citation_check={"cited": 0, "valid": 0, "invalid": []})
    return {**fields, **kw}


@pytest.mark.parametrize("strategy", list(ap.ARMS.values()))
def test_each_arm_returns_legal_evidence_with_its_source(world, strategy):
    hits = world.api.search(A, "南京跑步", as_of=WHEN, known_at=WHEN, k=10, include_ended=True, strategy=strategy)
    assert hits and all(h.persona_id == A and h.source_id == "src" and h.recorded_at is not None and (h.memory_id or h.message_id) for h in hits)
    assert all(h.locator == "x=1" for h in hits if h.memory_id)            # messages of this fixture have no locator to carry
    items = [ap.to_evidence(h, evidence_key(A, h.memory_id or h.message_id), {}) for h in hits]
    assert ap.build_context(items).evidence_ids and all(e.known_at and e.source_id == "src" and e.locator == h.locator for e, h in zip(items, hits))


@pytest.mark.parametrize("strategy", list(ap.ARMS.values()))
def test_time_and_persona_rules_hold_in_every_arm(world, strategy):
    found = lambda known, ended=True: {h.memory_id for h in world.api.search(A, "南京跑步", as_of=known, known_at=known, include_ended=ended, strategy=strategy)}
    assert "late" in found(at(200)) and "late" not in found(at(50))       # known only on day 100
    assert "other" not in found(at(200))                                  # a persona of its own
    early = world.api.search(A, "南京跑步", as_of=at(0), known_at=at(0), include_ended=True, strategy=strategy)
    assert "early" not in {h.memory_id for h in early}                    # include_ended does not open a later known_at


def test_quality_is_absent_without_permission_and_the_arms_use_one_interface(tmp_path):
    w = Index(tmp_path)
    enable_strategies(w.api, client=w.client, indexes={A: w.dense})
    assert "quality" not in w.api.strategies()
    with pytest.raises(ValueError):
        w.api.search(A, "跑步", strategy="quality")
    enable_strategies(w.api, client=w.client, indexes={A: w.dense}, allow_external=True)
    assert "quality" in w.api.strategies()


def test_the_rendered_item_keeps_only_real_fields():
    item = ap.render_item("E01", evidence(event_time=None, locator=None))
    assert item == "[E01] | 来源：sms\n内容：内容" and "时间" not in item
    full = ap.render_item("E02", evidence(speaker="本人"))
    assert "时间：2025-01-02 03:04" in full and "发言人：本人" in full and "定位：phone_id=1" in full


def test_a_record_is_shown_once_and_separate_records_stay_separate():
    c = ap.build_context([evidence("a"), evidence("a"), evidence("b")])
    assert c.evidence_ids == ("a", "b") and c.id_map == {"E01": "a", "E02": "b"} and c.text.count("内容：内容") == 2


def test_the_same_ceiling_and_cut_apply_to_every_list():
    long = [evidence(f"k{i}", "字" * 500) for i in range(12)]
    c = ap.build_context(long, max_tokens=1500, top_k=10)
    assert c.truncated and len(c.evidence_ids) == 2 and c.dropped_keys == tuple(f"k{i}" for i in range(2, 10))
    assert c.estimated_tokens <= 1500 and all(f"k{i}" in c.id_map.values() for i in (0, 1))
    assert not ap.build_context(long[:3], max_tokens=4000).truncated
    a, b = ap.build_context(long, max_tokens=1500), ap.build_context(long, max_tokens=1500)
    assert a.sha256 == b.sha256
    assert ap.build_context([]).text == "（没有找到相关记忆）"


def test_the_prompt_is_stable_and_names_no_arm():
    assert ap.prompt_sha256() == ap.prompt_sha256()
    text = json.dumps(ap.build_messages("问题", "2025-01-01", "证据"), ensure_ascii=False)
    assert not any(word in text for word in ("legacy", "fast", "quality", "multihop", "Multi_hop", "score_points", "参考答案"))


def test_generation_code_cannot_reach_official_answers():
    source = (ROOT / "src/stillhere/evaluation/answer_pilot.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {getattr(n, "module", None) for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert "answer_judge" not in imports and "memory_bench" not in imports and "answer_analysis" not in imports
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)} | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert not names & {"gold", "gold_events", "score_points", "required_events_id", "official", "reference_answer", "primary_type"}
    assert "question" in names           # the search did read the code


@pytest.mark.parametrize("raw,answer,abstain,citations,valid", [
    ('{"answer": "奶茶", "abstain": false, "citations": ["E01"]}', "奶茶", False, ("E01",), True),
    ('好的：\n```json\n{"answer": "根据现有记忆，无法确定。", "abstain": true, "citations": []}\n```', "根据现有记忆，无法确定。", True, (), True),
    ("就是奶茶", "就是奶茶", False, (), False),
    ("无法确定。", "无法确定。", True, (), False),
    ("", "", False, (), False),
    ('{"answer": 3}', '{"answer": 3}', False, (), False)])
def test_model_output_is_read_without_losing_it(raw, answer, abstain, citations, valid):
    p = ap.parse_answer(raw)
    assert (p.answer, p.abstain, p.citations, p.valid_json) == (answer, abstain, citations, valid)


def test_citations_are_checked_against_what_was_shown():
    assert ap.citation_check(["E01", "E09"], {"E01": "k"}) == {"cited": 2, "invalid": ["E09"], "valid": 1}


def generate(provider, results, arm="A1", text="证据"):
    context = ap.build_context([evidence(text=text)])
    return ap.generate_one(provider, results, qa_id="q1", persona_id="p", question="问", ask_time="2025-01-01", arm_id=arm, context=context, run_id="t")


def test_a_saved_answer_is_never_produced_twice_and_survives_a_restart(tmp_path):
    path, provider = tmp_path / "g.jsonl", FakeProvider()
    first = generate(provider, ap.ResultFile(path))
    again = generate(provider, ap.ResultFile(path))
    assert len(provider.calls) == 1 and first == again and first["citation_check"]["valid"] == 1
    for field in ("qa_id", "persona_id", "question", "ask_time", "arm_id", "evidence_ids", "evidence_hash", "prompt_hash", "answer_model", "decoding_config", "raw_model_output", "parsed_answer", "abstain", "citations", "input_tokens", "output_tokens", "latency_ms", "cost_cny"):
        assert field in first


def test_any_change_of_model_prompt_or_evidence_misses_the_cache(tmp_path):
    results, provider = ap.ResultFile(tmp_path / "g.jsonl"), FakeProvider()
    generate(provider, results)
    generate(provider, results, text="别的证据")
    generate(provider, results, arm="A2")
    other = FakeProvider()
    other.model_id = "other"
    generate(other, results)
    assert len(provider.calls) == 3 and len(other.calls) == 1
    base = dict(model_id="m", model_config={"t": 0}, prompt_hash="p", evidence_hash="e", qa_id="q", arm_id="A1")
    assert len({ap.cache_key(**base), ap.cache_key(**{**base, "prompt_hash": "x"}), ap.cache_key(**base, experiment_version="2")}) == 3


def test_every_question_has_a_fixed_order_and_no_arm_is_always_first():
    orders = [ap.arm_order(f"q{i}") for i in range(40)]
    assert orders == [ap.arm_order(f"q{i}") for i in range(40)]
    assert {o[0] for o in orders} == set(ap.ARMS)


OFFICIAL = aj.Official("q1", "奶茶", ({"description": "喝奶茶", "score": 10},), ("Single_hop",))


def replying(*labels):
    calls = []

    def complete(messages, max_tokens):
        calls.append(messages)
        label = labels[min(len(calls) - 1, len(labels) - 1)]
        return {"text": label if label == "garbage" else json.dumps({"label": label, "point_hits": [{"hit": label == "CORRECT"}], "reasoning": "r"}), "input_tokens": 5, "output_tokens": 5, "latency_ms": 1.0}
    return complete, calls


def test_the_judge_takes_a_majority_of_three_as_the_official_rule_does(tmp_path):
    complete, calls = replying("CORRECT", "WRONG", "CORRECT")
    r = aj.judge_answer(complete, ap.ResultFile(tmp_path / "j.jsonl"), PROMPTS, model=JUDGE, question="问", ask_time="d", official=OFFICIAL, answer="奶茶")
    assert r["label"] == "CORRECT" and r["votes"] == {"correct": 2, "wrong": 1} and r["weighted_score"] == 1.0 and len(calls) == 3


def test_the_judge_sees_no_arm_and_shares_a_verdict_for_identical_words(tmp_path):
    assert "arm" not in " ".join(aj.judge_answer.__code__.co_varnames)
    complete, calls = replying("CORRECT")
    results = ap.ResultFile(tmp_path / "j.jsonl")
    for _ in ("A1", "A2"):
        aj.judge_answer(complete, results, PROMPTS, model=JUDGE, question="问", ask_time="d", official=OFFICIAL, answer="奶茶")
    assert len(calls) == 3
    assert not any(word in json.dumps(calls, ensure_ascii=False) for word in ("A1", "A2", "fast", "legacy", "quality"))


def test_an_unusable_vote_is_missing_rather_than_a_silent_wrong(tmp_path):
    complete, _ = replying("garbage")
    r = aj.judge_answer(complete, ap.ResultFile(tmp_path / "j.jsonl"), PROMPTS, model=JUDGE, question="问", ask_time="d", official=OFFICIAL, answer="奶茶")
    assert r["label"] == "MISSING" and r["failed_votes"] == 3


def test_official_prompts_are_read_without_running_the_file(tmp_path):
    path = tmp_path / "llm_judge.py"
    path.write_text('raise SystemExit("ran")\nSYSTEM_PROMPT = "甲"\nUSER_PROMPT_TEMPLATE = """乙{question}"""\n', encoding="utf-8")
    p = aj.read_prompts(path)
    assert (p.system, p.user_template) == ("甲", "乙{question}") and len(p.source_sha256) == 64
    assert aj.render_score_points([{"description": "x", "score": 10}]) == "    1. x"


def test_a_chat_call_is_priced_by_input_and_output_and_joins_the_old_ledger(tmp_path):
    server = Chat()
    c = client(tmp_path, server)
    c.ledger.settle(0, {"model_id": FLASH, "request_type": "embedding:query", "input_tokens": 1, "cost_cny": 1.0, "success": True, "billing_uncertain": False})
    r = c.chat(JUDGE, [{"role": "user", "content": "你好"}], max_tokens=100)
    assert r.text == '{"label": "CORRECT"}' and server.calls[0][1]["enable_thinking"] is False and "compatible-mode" in server.calls[0][0]
    assert c.ledger.spent == pytest.approx(1.0 + 1000 / 1e6 * 0.2 + 500 / 1e6 * 0.8)
    assert client(tmp_path, server).ledger.spent == pytest.approx(c.ledger.spent)       # a new run does not start from zero
    assert c.ledger.summary()["output_tokens"] == 500


def test_a_chat_call_stops_at_the_budget_before_it_is_sent(tmp_path):
    server = Chat()
    c = client(tmp_path, server, budget=0.00001)
    with pytest.raises(BudgetExceeded):
        c.chat(JUDGE, [{"role": "user", "content": "你好"}], max_tokens=1000)
    assert server.calls == []


def test_offline_never_calls(tmp_path):
    server = Chat()
    c = client(tmp_path, server, offline=True)
    with pytest.raises(OfflineCall):
        c.chat(JUDGE, [{"role": "user", "content": "你好"}], max_tokens=10)
    assert server.calls == []


def test_the_script_sends_nothing_without_explicit_permission():
    run = lambda *a: subprocess.run([sys.executable, str(ROOT / "scripts/evaluate_answer_pilot.py"), *a], capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    judged = run("--score-only")
    assert judged.returncode != 0 and "--allow-api" in judged.stderr
    assert run("--score-only", "--allow-api", "--offline").returncode != 0
    assert run("--dry-run", "--run").returncode != 0


def test_paired_figures_use_only_questions_both_arms_finished():
    a = {"q1": True, "q2": True, "q3": False, "q4": None, "q5": True}
    b = {"q1": True, "q2": False, "q3": False, "q4": True}
    r = aj.paired_compare(a, b, rounds=200)
    assert r["questions"] == 3 and (r["wins"], r["losses"], r["both_correct"], r["both_wrong"]) == (1, 0, 1, 1)
    assert r["difference"] == pytest.approx(1 / 3, abs=1e-3) and r["wrong_to_correct"] == ["q2"]
    assert aj.paired_compare({}, {})["questions"] == 0


def test_mcnemar_is_exact():
    assert aj.exact_mcnemar(0, 0) == 1.0
    assert aj.exact_mcnemar(0, 6) == pytest.approx(2 / 64)
    assert aj.exact_mcnemar(5, 5) == 1.0
    assert aj.holm({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}


def test_an_unfinished_arm_is_reported_and_never_counts_as_correct():
    rows = [row("q1", "A1"), row("q2", "A1", "MISSING"), row("q1", "A2", "WRONG")]
    accuracy = an.accuracy(rows)
    assert accuracy["A1"]["missing"] == 1 and accuracy["A1"]["correct"] == 1 and accuracy["A1"]["total"] == 2
    assert an.paired(rows)["MULTIHOP_MINUS_FAST"]["questions"] == 1


def test_unanswerable_questions_are_counted_on_their_own():
    rows = [row("u1", "A1", "CORRECT", unanswerable=True, abstain=True), row("u2", "A1", "WRONG", unanswerable=True), row("q1", "A1", "WRONG", abstain=True, sufficiency={"sufficient": "yes"}), row("q2", "A1", "WRONG", abstain=True, sufficiency={"sufficient": "no"})]
    r = an.abstention(rows)["A1"]
    assert (r["unanswerable"], r["correct_abstention"], r["incorrect_answer"], r["unnecessary_abstention"], r["abstained_without_enough_evidence"]) == (2, 1, 1, 1, 1)
    assert an.accuracy(rows)["A1"]["total"] == 2


def test_failure_categories_follow_their_definitions():
    rows = [row("q1", "A1", "WRONG", sufficiency={"sufficient": "yes"}), row("q2", "A1", "WRONG", sufficiency={"sufficient": "no"}, grounding={"unsupported_personal_fact": True, "temporal_misuse": True}),
            row("q3", "A1", "CORRECT", votes={"correct": 2, "wrong": 1}), row("q4", "A1", "WRONG")]
    r = an.attribution(rows)["arms"]["A1"]
    assert r["questions"]["F5_reasoning_failed"] == ["q1"] and r["questions"]["F1_retrieval_missing"] == ["q2"] and r["questions"]["F6_unsupported_inference"] == ["q2"]
    assert r["questions"]["F7_temporal_error"] == ["q2"] and r["questions"]["F9_judge_uncertain"] == ["q3"] and r["wrong_without_sufficiency_verdict"] == 1


def test_swaps_separate_added_evidence_from_displaced_evidence():
    rows = [row("q1", "A1", "WRONG", keys=["x"], gold=["g"]), row("q1", "A2", "CORRECT", keys=["g"], gold=["g"], coverage=1.0, full=True),
            row("q2", "A1", "CORRECT", keys=["g"], gold=["g"], coverage=1.0, full=True), row("q2", "A2", "WRONG", keys=["y"], gold=["g"]),
            row("q3", "A1", "WRONG", keys=[], gold=["g"]), row("q3", "A2", "WRONG", keys=["g"], gold=["g"], coverage=1.0)]
    s = an.swap_analysis(rows, "A2", "A1")
    assert (s["wrong_to_correct"], s["correct_to_wrong"], s["improved_with_added_gold"], s["worsened_with_displaced_gold"], s["full_recall_gained"], s["full_gained_and_improved"]) == (1, 1, 1, 1, 1, 1)
    p = an.attribution(rows)["pairs"]["MULTIHOP_MINUS_FAST"]["questions"]
    assert p["F2_evidence_displacement"] == ["q2"] and p["F3_evidence_added_answer_improved"] == ["q1"] and p["F4_evidence_added_answer_unchanged"] == ["q3"]
