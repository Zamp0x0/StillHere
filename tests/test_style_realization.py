"""Style realization: flag, backends, contract use, claim check, fallback, trace (CPU only, synthetic)."""
import pytest

from stillhere.style import contract as sc
from stillhere.style import prompt_builder as pb
from stillhere.style import realization as sr

ON = {sr.FLAG: "1"}
P = "syn.style.a"
HIST = [sc.StyleMessage(f"m{i}", P, t, topic=f"topic{i % 4}") for i, t in enumerate(
    ["哈哈行啊", "周末去爬山不", "我刚到家，累死了", "这个我得想想", "好嘞，明天见", "你吃了没", "下雨了带伞", "今天加班到九点", "那家火锅还行", "改天约"])]
CONV = [{"speaker": "other", "text": "你搬去哪了来着？"}]
G = pb.grounded([{"statement": "现在住在高新区", "role": "current", "time": "2026-10-10"}])


def gen(text, cost=0.001):
    calls = []

    def f(messages, cfg):
        calls.append((messages, cfg))
        return {"text": text, "cost_cny": cost}
    f.calls = calls
    return f


def test_flag_off_is_neutral_and_never_calls_the_model():
    g = gen("搬到高新区啦")
    r = sr.realize(P, CONV, G, history=HIST, backend="prompt", generate=g, env={})
    assert not r.styled and r.text == "现在住在高新区。" and r.trace.fallback == "flag_off" and not g.calls


def test_prompt_backend_uses_the_contract_messages():
    g = gen("搬到高新区啦，离公司近多了")
    r = sr.realize(P, CONV, G, history=HIST, backend="prompt", generate=g, env=ON, key_terms=["高新区"], evidence_ids=["w.1"], model="api-model")
    assert r.styled and r.text.startswith("搬到高新区")
    want = sc.build(P, CONV, grounded_content=G, history=HIST)
    assert g.calls[0][0] == want.messages
    t = r.trace
    assert t.contract["prompt_hash"] == want.prompt_hash and t.contract["contract_version"] == sc.CONTRACT_VERSION
    assert t.contract["selected_example_ids"] and t.evidence_ids == ["w.1"] and t.claim_check["ok"] and t.cost_cny == 0.001
    assert "messages" not in t.contract and t.generation["enable_thinking"] is False


def test_dropped_fact_falls_back_to_the_facts():
    r = sr.realize(P, CONV, G, history=HIST, backend="prompt", generate=gen("搬家了哈哈"), env=ON, key_terms=["高新区"])
    assert not r.styled and r.text == "现在住在高新区。" and r.trace.fallback == "claim_check_failed"
    assert any(p.startswith("dropped_key_terms") for p in r.trace.claim_check["problems"])


def test_invented_number_falls_back():
    r = sr.realize(P, CONV, G, history=HIST, backend="prompt", generate=gen("高新区，3号线旁边"), env=ON, key_terms=["高新区"])
    assert not r.styled and "unsupported_specifics:3" in r.trace.claim_check["problems"][0]


def test_unknown_must_stay_unknown():
    g = pb.grounded(unknown=["新家的门牌号"])
    bad = sr.realize(P, CONV, g, history=HIST, backend="prompt", generate=gen("就在那边嘛"), env=ON)
    ok = sr.realize(P, CONV, g, history=HIST, backend="prompt", generate=gen("门牌号我不太清楚哎"), env=ON)
    assert not bad.styled and "unknown_not_said_as_unknown" in bad.trace.claim_check["problems"] and ok.styled


def test_adapter_backend_is_blocked_without_weights():
    r = sr.realize(P, CONV, G, history=HIST, backend="adapter", env=ON, adapter="t2d2_r16_run1")
    assert not r.styled and r.trace.fallback.startswith("BLOCKED") and r.trace.contract["prompt_hash"]


def test_backend_error_falls_back():
    def boom(messages, cfg):
        raise TimeoutError("x")
    r = sr.realize(P, CONV, G, history=HIST, backend="prompt", generate=boom, env=ON)
    assert not r.styled and r.trace.fallback == "backend_error:TimeoutError"


def test_persona_isolation_is_enforced():
    foreign = HIST + [sc.StyleMessage("x1", "syn.style.b", "别人的话")]
    with pytest.raises(sc.ContractError):
        sr.realize(P, CONV, G, history=foreign, backend="prompt", generate=gen("高新区"), env=ON)


def test_no_memory_reply():
    r = sr.realize(P, CONV, pb.no_memory(), history=HIST, backend="neutral", env=ON)
    assert r.text == "这个我没有记录，不太清楚。" and not r.styled
    bad = sr.realize(P, CONV, pb.no_memory(), history=HIST, backend="prompt", generate=gen("在18楼"), env=ON)
    assert not bad.styled


def test_unknown_backend():
    with pytest.raises(ValueError):
        sr.realize(P, CONV, G, backend="gpt", env=ON)
