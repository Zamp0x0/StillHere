"""Batch 2 end to end on CPU: synthetic conversations -> M2.2 Writer (the gold stands in for the extraction model) -> bitemporal SQLite ->
B2 retrieval -> M3.1 reflection -> grounded content -> style prompt contract -> fake generator (base / LoRA stand-ins) -> safety guards.
No model is loaded and no API is called."""
from datetime import datetime

import pytest

from stillhere.evaluation import writer_dataset_v3 as v3
from stillhere.evaluation import writer_eval as we
from stillhere.memory import reflection_m31 as m31
from stillhere.memory import retrieval_b2 as b2
from stillhere.runtime import RUNTIME_PIPELINE_VERSION, grounded_from_evidence, reply
from stillhere.style import contract as sc
from stillhere.style import prompt_builder as pb
from stillhere.style import safety as S

ON = {"GUARD_ENABLED": "1"}


@pytest.fixture(scope="module")
def world():
    return we.run(we.new_world(we.reference_complete(v3.conversations()), "reference", ds=v3, m22=True), ds=v3)


def ask(q, when):
    return [{"speaker": "other", "text": q}], when


class Fake:
    def __init__(self, *texts):
        self.texts, self.calls = list(texts), []

    def __call__(self, messages, config):
        self.calls.append((messages, dict(config)))
        return {"text": self.texts[min(len(self.calls) - 1, len(self.texts) - 1)]}


HISTORY = [sc.StyleMessage(f"h{i}", v3.RA, t, topic=f"t{i % 4}") for i, t in enumerate(
    ["嗯嗯好的", "周末再说吧", "我最近挺忙的", "哈哈可以啊", "那就这么定了", "我看看情况", "行，没问题", "改天约", "最近在练羽毛球", "下班了"])]


def test_flags_default_off(world):
    assert not b2.enabled({}) and not m31.enabled({}) and not any(S.flags({}).values())


def test_neutral_chain_runs_without_a_model(world):
    conv, when = ask("我平时喜欢什么运动？", v3.t(9, 10, 20))
    tr = reply(world.api, v3.RA, conv, when)
    assert tr.version == RUNTIME_PIPELINE_VERSION and tr.status == "NEUTRAL" and "羽毛球" in tr.text
    assert tr.evidence_ids and tr.contract is None


def test_synonym_question_is_grounded(world):
    m = m31.reflect(world.api, v3.RA, "我平时喝什么饮品？", v3.t(9, 10, 20))
    g = grounded_from_evidence(m, v3.t(9, 10, 20))
    assert any("茶" in f["statement"] for f in g["facts"])


def test_no_leak_of_later_records(world):
    early = v3.t(9, 1, 9)
    m = m31.reflect(world.api, v3.RA, "我现在住在哪里？", early)
    assert all(r.recorded_at <= early for r in m.evidence.records)


def test_other_people_are_labelled(world):
    m = m31.reflect(world.api, v3.RA, "我要搬去成都吗？", v3.t(9, 12, 20))
    assert "阿凯" in m.context


def test_style_contract_and_guard_with_fake_lora(world):
    conv, when = ask("我平时喜欢什么运动？", v3.t(9, 10, 20))
    fake = Fake("羽毛球呀，羽毛球呀，羽毛球呀，羽毛球呀")
    tr = reply(world.api, v3.RA, conv, when, history=HISTORY, generate=fake, backend="lora", env={**ON, "LENGTH_POLICY_ENABLED": "1"})
    assert tr.contract["prompt_schema_version"] == pb.PROMPT_SCHEMA_VERSION == "stillhere.style.prompt/2.0"
    sys_prompt = fake.calls[0][0][0]["content"]
    assert "回复长度：按这个人平时说话的长度" in sys_prompt                     # the length rule went in BEFORE generation (contract constraints)
    assert tr.status == "REPAIRED" and tr.text.startswith("羽毛球") and "羽毛球呀，羽毛球呀" not in tr.text
    assert tr.safety["length_policy"]["source"] == "author_history"
    assert fake.calls[0][1]["max_new_tokens"] <= 160


def test_base_and_lora_share_the_interface(world):
    conv, when = ask("我住在哪个区？", v3.t(9, 10, 20))
    outs = [reply(world.api, v3.RA, conv, when, history=HISTORY, generate=Fake("雁塔区那边"), backend=b, env=ON) for b in ("base", "lora")]
    assert [o.status for o in outs] == ["ACCEPTED", "ACCEPTED"] and outs[0].contract["prompt_hash"] == outs[1].contract["prompt_hash"]


def test_unsupported_claim_triggers_retry_then_fallback(world):
    conv, when = ask("我住在哪个区？", v3.t(9, 10, 20))
    fake = Fake("我住在8号楼302", "我住在8号楼302")
    tr = reply(world.api, v3.RA, conv, when, history=HISTORY, generate=fake, backend="lora", env=ON, max_retries=1)
    assert tr.status == "FALLBACK" and len(fake.calls) == 2 and "302" not in tr.text


def test_guard_off_keeps_old_behaviour(world):
    conv, when = ask("我住在哪个区？", v3.t(9, 10, 20))
    tr = reply(world.api, v3.RA, conv, when, history=HISTORY, generate=Fake("我住在8号楼302"), backend="lora", env={})
    assert tr.status == "PASS_THROUGH" and tr.text == "我住在8号楼302"


def test_unknown_question(world):
    conv, when = ask("我打算去西藏拍星空吗？", v3.t(10, 20, 20))
    tr = reply(world.api, v3.RC, conv, when)
    assert tr.status == "NEUTRAL"


def test_reader_answer_is_checked_before_it_is_used(world):
    conv, when = ask("我住在哪个区？", v3.t(9, 10, 20))
    good = reply(world.api, v3.RA, conv, when, answerer=lambda q, t, ev: {"answer": "你住在西安雁塔区。", "abstain": False})
    assert good.reader["claim_check"]["ok"] and good.grounded["facts"][0]["source"].startswith("reader")
    bad = reply(world.api, v3.RA, conv, when, answerer=lambda q, t, ev: {"answer": "你住在8号楼302。", "abstain": False})
    assert not bad.reader["claim_check"]["ok"] and not any(f.get("source", "").startswith("reader") for f in bad.grounded["facts"])


def test_corrected_values_become_must_not(world):
    # 周默's job was corrected (会计 -> 出纳): the old value travels as ``replaces`` and is a forbidden CURRENT claim
    when = v3.t(9, 25, 20)
    g = grounded_from_evidence(m31.reflect(world.api, v3.RC, "我退休前做什么工作？", when), when)
    assert any(f["role"] == "corrected" and "会计" in f.get("replaces", "") for f in g["facts"])
    assert S.must_not("我退休前是会计", g) and not S.must_not("我之前说是会计，其实是出纳", g)


def test_a_move_is_past_not_corrected(world):
    # 方远 moved (雁塔 -> 曲江): the old address is a past fact, never a forbidden value
    when = v3.t(10, 15, 20)
    g = grounded_from_evidence(m31.reflect(world.api, v3.RA, "我现在住在哪里？", when), when)
    assert any(f["role"] == "past" and "雁塔" in f["statement"] for f in g["facts"]) and S.forbidden_terms(g) == []


def test_trace_has_no_foreign_persona_text(world):
    conv, when = ask("我住在哪？", v3.t(9, 10, 20))
    tr = reply(world.api, v3.RB, conv, when)
    blob = str(tr.as_dict())
    assert "方远" not in blob and "周默" not in blob
