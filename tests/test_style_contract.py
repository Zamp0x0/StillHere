"""UniversalStylePromptContract v2 (synthetic fixtures): one builder, stable hash, isolation, leak guard, ablations, budget."""
import pytest

from stillhere.style import contract as sc
from stillhere.style import prompt_builder as pb

P = "syn.style.a"
HIST = [sc.StyleMessage(f"m{i}", P, t, topic=f"topic{i % 4}") for i, t in enumerate(
    ["哈哈行啊", "周末去爬山不", "我刚到家，累死了", "这个我得想想", "好嘞，明天见", "你吃了没", "下雨了带伞", "今天加班到九点", "那家火锅还行", "改天约"])]
CONV = [{"speaker": "other", "text": "周末有空吗？"}]
G = pb.grounded([{"statement": "周六要去体检", "role": "current", "time": "2026-09-26"}])


def test_same_input_same_prompt_and_hash():
    a = sc.build(P, CONV, grounded_content=G, history=HIST)
    b = sc.build(P, CONV, grounded_content=G, history=HIST)
    assert a.messages == b.messages and a.prompt_hash == b.prompt_hash == sc.prompt_hash(a.messages)
    assert a.contract_version == "stillhere.style.contract/2.0" and a.prompt_schema_version == "stillhere.style.prompt/2.0"
    assert len(a.selected_example_ids) == sc.TRAINING_K_EXAMPLES and not a.ablation


def test_messages_come_from_the_shared_builder():
    a = sc.build(P, CONV, grounded_content=G, history=HIST)
    sys_text = a.messages[0]["content"]
    for section in ("【说话方式】", "【说话方式示例】", "【记忆事实】", "【生成要求】"):
        assert section in sys_text
    assert "周六要去体检" in sys_text and a.messages[0]["role"] == "system"


def test_facts_and_style_are_separate():
    a = sc.build(P, CONV, grounded_content=G, history=HIST)
    b = sc.build(P, CONV, grounded_content=pb.no_memory(), history=HIST)
    assert a.selected_example_ids == b.selected_example_ids and a.prompt_hash != b.prompt_hash


def test_persona_isolation():
    with pytest.raises(sc.ContractError):
        sc.build(P, CONV, history=HIST + [sc.StyleMessage("x", "syn.style.b", "别人")])


def test_answer_leak_guard():
    a = sc.build(P, CONV, history=HIST, reference_reply="周末去爬山不")
    assert "m1" not in a.selected_example_ids and "周末去爬山不" not in a.messages[0]["content"].split("【记忆事实】")[0].split("【说话方式示例】")[-1]


def test_ablations_are_marked():
    assert sc.build(P, CONV, history=HIST, use_examples=False).ablation == ["no_examples"]
    assert sc.build(P, CONV, history=HIST, use_profile=False).ablation == ["no_profile"]
    assert "fewer_examples:2" in sc.build(P, CONV, history=HIST[:2]).ablation


def test_token_budget():
    with pytest.raises(sc.ContractError):
        sc.build(P, CONV, history=HIST, count_tokens=lambda m: 5000)
    assert sc.build(P, CONV, history=HIST, count_tokens=lambda m: 900).prompt_tokens == 900


def test_trace_has_no_texts():
    t = sc.build(P, CONV, grounded_content=G, history=HIST).trace()
    assert "messages" not in t and t["model"]["base_model"] == "Qwen/Qwen3-8B" and t["prompt_hash"]
    assert "周六要去体检" not in str(t)


def test_training_episode_reproduction_on_a_synthetic_episode():
    ep = {"current_context": [{"text": "周末有空吗？"}], "support_examples": [{"bubbles": ["哈哈行啊"]}, {"bubbles": ["改天约", "好嘞"]}],
          "grounded": G, "style_profile": None}
    msgs = sc.from_training_episode(ep)
    assert msgs == pb.build_style_conditioned_messages([{"speaker": "other", "text": "周末有空吗？"}], grounded_content=G, style_profile=None,
                                                       style_examples=["哈哈行啊", "改天约 / 好嘞"])
