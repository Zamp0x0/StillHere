"""Batch 2.1 post-GPU fixes on CPU (fake generators; no model): grounding v2 (unknown / partial / answered), the unknown gate, the internal-leak
gate, the safe fallback (never raw evidence), the content-aware length plan, must-not time semantics, and the runtime chain end to end."""
from datetime import datetime

import pytest

from stillhere.evaluation import batch2_style_history as sh
from stillhere.evaluation import writer_dataset_v5 as v5
from stillhere.evaluation import writer_eval as we
from stillhere.memory import reflection_m31 as m31
from stillhere.runtime import pipeline as RP
from stillhere.style import contract as sc
from stillhere.style import safety as S

ON = {"GUARD_ENABLED": "1", "LENGTH_POLICY_ENABLED": "1"}
RAW_EVENT = ("事件：实习面试 / 面试；当事人：本人；当前状态：已完成；状态记录：计划中（已更正（原记录有误）；记录于 2026-09-10） → 已完成（当前；"
             "记录于 2026-09-17）；发生时间：9月17日（2026-09-17）")


class Fake:
    def __init__(self, *texts):
        self.texts, self.calls = list(texts), []

    def __call__(self, messages, config):
        self.calls.append((messages, dict(config)))
        return {"text": self.texts[min(len(self.calls) - 1, len(self.texts) - 1)]}


@pytest.fixture(scope="module")
def world():
    return we.run(we.new_world(we.reference_complete(v5.conversations()), "reference", ds=v5, m22=True, m221=True), ds=v5)


def g2(world, persona, q, ask, answer=None, abstained=True):
    m = m31.reflect(world.api, persona, q, ask)
    return RP.grounded_from_evidence_v2(m, ask, q, answer, abstained)


# ---------------------------------------------------------------- grounding v2
def test_complete_unknown_gets_no_unrelated_facts(world):
    g = g2(world, v5.UA, "我养猫吗？", v5.t(9, 25, 20))
    assert g["answer_status"] == "unknown" and g["facts"] == [] and g["unknown"]
    g3 = g2(world, v5.UC, "苗苗喜欢上海吗？", v5.t(9, 20, 20))                     # about someone else: the persona's own likes are not related
    assert not any("篮球" in f["statement"] for f in g3["facts"]) and g3["unknown"]


def test_same_kind_preference_is_partial_and_still_admits_unknown(world):
    g = g2(world, v5.UC, "我喜欢游泳吗？", v5.t(9, 20, 20))
    assert g["answer_status"] == "partial" and g["unknown"] and not any("游泳" in f["statement"] for f in g["facts"])


def test_partial_unknown_keeps_the_related_fact(world):
    g = g2(world, v5.UC, "我在哪家公司？", v5.t(9, 15, 20))
    assert g["answer_status"] == "partial" and any("支付" in f["statement"] for f in g["facts"]) and g["unknown"]


def test_answered_question_keeps_the_answer_first(world):
    g = g2(world, v5.UC, "我是做什么的？", v5.t(9, 15, 20), answer="你是写后端的。", abstained=False)
    assert g["answer_status"] == "answered" and g["facts"][0]["statement"] == "你是写后端的。"


def test_corrected_value_stays_forbidden():
    from stillhere.evaluation import writer_dataset_v3 as v3
    w3 = we.run(we.new_world(we.reference_complete(v3.conversations()), "reference", ds=v3, m22=True), ds=v3)
    ask = v3.t(9, 25, 20)
    m = m31.reflect(w3.api, v3.RC, "我退休前做什么工作？", ask)
    g = RP.grounded_from_evidence_v2(m, ask, "我退休前做什么工作？", "你退休前是出纳。", False)
    assert any(f.get("role") == "corrected" and "会计" in f.get("replaces", "") for f in g["facts"])
    assert S.must_not("我退休前是会计", g) and not S.must_not("之前说成会计了，其实是出纳", g)


def test_unresolved_conflict_becomes_unverified(world):
    # b21p world: the correction message's gold span lacks the correction wording, so memory holds 南山区 and 福田区 as a conflict
    g = g2(world, v5.UC, "我现在住哪个区？", v5.t(9, 15, 20))
    roles = {f["statement"]: f["role"] for f in g["facts"]}
    assert roles.get("本人住在南山区") == "unverified" and roles.get("本人住在福田区") == "unverified"
    text, kind = S.safe_fallback({**g, "unknown": []})
    assert kind == "CONFLICT" and "南山" not in text


def test_no_internal_structure_in_grounded_statements(world):
    for q, ask in (("骑行赛还比吗？", v5.t(9, 25, 20)), ("杭州去了吗？", v5.t(10, 9, 20))):
        persona = v5.UA if "骑行" in q else v5.UB
        g = g2(world, persona, q, ask, abstained=False)
        assert g["facts"] and not any(S.LEAK.search(f["statement"]) for f in g["facts"]) and all("source" not in f for f in g["facts"])


def test_generic_words_are_not_relevance():
    assert not RP.relevant("本人不喜欢下棋", "宁宁喜欢画画吗？") and RP.relevant("本人的“实习面试”已经完成", "我在哪家公司实习？")


# ---------------------------------------------------------------- guard gates
def test_unknown_gate_blocks_an_answer_that_ignores_the_unknown(world):
    g = g2(world, v5.UC, "我喜欢游泳吗？", v5.t(9, 20, 20))
    assert any(c.startswith("unknown_not_admitted") for c in S.claims("喜欢呀，周末常去", g, []))
    assert not S.claims("这个你没跟我说过，不太清楚", g, [])


def test_leak_gate_and_fallback_never_returns_raw_evidence():
    g = {"status": "facts", "facts": [{"statement": RAW_EVENT, "role": "current"}], "unknown": ["在哪家公司实习"], "answer_status": "partial"}
    r = S.safe_generate(Fake("在XX科技实习", "在XX科技实习"), [], {}, grounded=g, conversation=[{"speaker": "other", "text": "我在哪家公司实习？"}], env=ON)
    assert r.status == "FALLBACK" and not S.LEAK.search(r.text) and "XX" not in r.text and "{" not in r.text
    assert r.fallback_kind in ("NATURAL_UNKNOWN", "MINIMAL") and len(r.attempts) == 2          # raw line dropped, bounded attempts
    r2 = S.safe_generate(Fake(RAW_EVENT), [], {}, grounded={"status": "facts", "facts": [{"statement": "本人在一家银行工作", "role": "current"}], "unknown": []}, env=ON,
                         max_retries=0)
    assert r2.attempts[0].checks.get("internal_leak") and r2.status == "FALLBACK"


@pytest.mark.parametrize("grounded,kind,must_contain,must_not_contain", [
    ({"status": "facts", "facts": [{"statement": "本人的“面试”已经完成（9月17日）", "role": "current"}], "unknown": ["在哪家公司实习"], "answer_status": "partial"},
     "SAFE_PARTIAL", ["面试", "没有记录"], ["本人", "“"]),
    ({"status": "facts", "facts": [], "unknown": ["喜欢游泳"], "answer_status": "unknown"}, "NATURAL_UNKNOWN", ["没有记录"], []),
    ({"status": "facts", "facts": [{"statement": "本人可能在银行工作", "role": "unverified"}], "unknown": []}, "CONFLICT", ["两种说法"], ["银行"]),
    ({"status": "facts", "facts": [{"statement": "本人在软件工程上学", "role": "corrected", "replaces": "本人在武汉大学上学"}], "unknown": []},
     "SAFE_PARTIAL", ["软件工程"], ["武汉大学"]),
    ({"status": "no_memory", "facts": [], "unknown": []}, "MINIMAL", [], []),
])
def test_fallback_kinds_are_natural_and_safe(grounded, kind, must_contain, must_not_contain):
    text, k = S.safe_fallback(grounded)
    assert k == kind and not S.LEAK.search(text)
    assert all(x in text for x in must_contain) and not any(x in text for x in must_not_contain)


def test_fallback_is_checked_like_a_reply():
    g = {"status": "facts", "facts": [{"statement": "本人在一家银行工作，工号8831", "role": "current"}], "unknown": [], "answer_status": "facts"}
    text, k = S.safe_fallback(g, must_not_terms=["8831"])
    assert k == "MINIMAL" and "8831" not in text                       # the fallback would state a forbidden value: minimal reply instead


def test_retry_then_accept_and_no_more_than_one_retry(world):
    g = g2(world, v5.UC, "我喜欢游泳吗？", v5.t(9, 20, 20))
    fake = Fake("喜欢呀", "这个你没跟我说过，不清楚")
    r = S.safe_generate(fake, [], {}, grounded=g, env=ON)
    assert r.status == "RETRIED_ACCEPTED" and len(fake.calls) == 2
    fake2 = Fake("喜欢呀")
    r2 = S.safe_generate(fake2, [], {}, grounded=g, env=ON, max_retries=1)
    assert r2.status == "FALLBACK" and len(fake2.calls) == 2


# ---------------------------------------------------------------- length plan
def test_same_plan_for_base_and_lora_and_persona_differences():
    short = S.pre_generation_plan(sh.history("syn.ua"))
    long_ = S.pre_generation_plan(sh.history("syn.ub"))
    bubbles = S.pre_generation_plan(sh.history("syn.uc"))
    assert short.target_chars <= bubbles.target_chars < long_.target_chars
    assert short.policy.soft_high < bubbles.policy.soft_high < long_.policy.soft_high and short.policy.hard_cap <= bubbles.policy.hard_cap < long_.policy.hard_cap
    assert short == S.pre_generation_plan(sh.history("syn.ua"))         # a pure function of the history: identical for base and LoRA


def test_content_aware_target_keeps_the_answer_sayable():
    hist = sh.history("syn.ua")
    plain = S.pre_generation_plan(hist, grounded={"status": "facts", "facts": [{"statement": "x", "role": "current"}]})
    need = S.pre_generation_plan(hist, grounded={"status": "facts", "facts": [{"statement": "x", "role": "current"}]}, content_chars=27)
    assert need.target_chars == 30 > plain.target_chars and need.max_new_tokens >= 30 // 1.3
    capped = S.pre_generation_plan(hist, grounded={"status": "facts", "facts": [{"statement": "x", "role": "current"}]}, content_chars=300)
    assert capped.target_chars == S.CONTENT_TARGET_MAX


# ---------------------------------------------------------------- runtime chain
def test_runtime_chain_v2_unknown_ends_natural(world):
    hist = [sc.StyleMessage(f"h{i}", v5.UC, t) for i, t in enumerate(sh.history("syn.uc"))]
    fake = Fake("喜欢游泳，每周都去", "喜欢游泳")
    tr = RP.reply(world.api, v5.UC, [{"speaker": "other", "text": "我喜欢游泳吗？"}], v5.t(9, 20, 20), history=hist, generate=fake, backend="lora",
                  answerer=lambda q, t, ev: {"answer": "根据现有记忆，无法确定。", "abstain": True}, env={**ON, "GROUNDING_V2_ENABLED": "1"})
    assert tr.grounded["answer_status"] in ("unknown", "partial") and tr.status == "FALLBACK" and "没有记录" in tr.text and not S.LEAK.search(tr.text)
    assert "喜欢游泳" not in tr.text
    prompt = fake.calls[0][0][0]["content"]
    assert "event:" not in prompt and "来源：w." not in prompt                     # no internal ids reach the model


def test_runtime_chain_v1_unchanged_by_default(world):
    hist = [sc.StyleMessage(f"h{i}", v5.UC, t) for i, t in enumerate(sh.history("syn.uc"))]
    tr = RP.reply(world.api, v5.UC, [{"speaker": "other", "text": "我喜欢游泳吗？"}], v5.t(9, 20, 20), history=hist, generate=Fake("好"), backend="lora", env={})
    assert "answer_status" not in tr.grounded and tr.status == "PASS_THROUGH"
