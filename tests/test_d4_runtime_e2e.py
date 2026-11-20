"""Simple Context v1.1 — D4 contract end to end (CASES 1-20): runtime -> evidence adapter + disclosure policy -> pinned D4 renderer -> token
counter -> guard, with deterministic FAKE generators (not a real model). Public synthetic data only."""
import json

import pytest

import _d4_world as W
from _ctx_world import HISTORY, Boom, Clock, Fake
from stillhere.session import SessionRuntime
from stillhere.session import d4_renderer as dr
from stillhere.style import contract as sc

ON = {"GUARD_ENABLED": "1", "GROUNDING_V2_ENABLED": "1"}
H = [sc.StyleMessage(f"{W.P}.h{i}", W.P, h.text, topic=h.topic) for i, h in enumerate(HISTORY)]
ABSTAIN = lambda q, t, ev: {"answer": "根据现有记忆，无法确定。", "abstain": True}


def rt(api=None, grant=True):
    api = api or W.build()
    r = SessionRuntime(api, render_mode="d4", clock=Clock(W.NOW))
    if grant:
        r.policy.set(W.P, "persona", "email", "ALLOW", W.t(9, 1))
    return r


def ask(r, text, gen=None, s=None, answerer=None, rid=None):
    s = s or r.create_session(W.P)
    out = r.reply_session(W.P, s, text, rid or f"q-{abs(hash(text))}", history=H, generate=gen or Fake("嗯"), backend="lora", env=ON, answerer=answerer)
    return out, s


def system(out):
    return out.messages[0]["content"] if out.messages else ""


def common(out, r):
    """Every case: the D4 renderer produced the prompt, its hash is traced, the exact budget holds, no raw guarded value anywhere."""
    if out.messages:
        assert system(out).startswith(dr.render_mod.INTRO) and "【记忆证据】" in system(out) and out.messages[1]["content"].startswith("【对话】")
        assert out.trace["contract"]["renderer_sha256"] == dr.PINNED_RENDERER_SHA256
        assert out.trace["input_tokens"] + out.trace["output_reserve"] + 2 <= 1536
    blob = json.dumps(out.trace, ensure_ascii=False) + json.dumps(out.messages, ensure_ascii=False)
    assert W.FRIEND_PHONE not in blob and W.OWN_PHONE not in blob


def test_case1_known_fact():
    r = rt()
    out, _ = ask(r, "我现在住哪？", Fake("西湖区那边"))
    common(out, r)
    assert "本人住在杭州西湖区" in system(out) and "当前有效" in system(out) and out.status == "GENERATED"
    assert r.token_mode in ("EXACT", "ESTIMATED")


def test_case2_full_unknown_is_admitted():
    out, _ = ask(rt(), "我养猫吗？", Fake("养了一只橘猫", "养了一只橘猫"), answerer=ABSTAIN)
    common(out, None)
    assert "[未知]" in system(out) and "橘猫" not in out.text


def test_case3_partial_unknown():
    out, _ = ask(rt(), "成都出差住哪个酒店？", Fake("住锦江宾馆", "住锦江宾馆"), answerer=ABSTAIN)
    common(out, None)
    assert "去成都出差" in system(out) and "[未知]" in system(out) and "锦江" not in out.text


def test_case4_correction():
    out, _ = ask(rt(), "我是做什么的？", Fake("药剂师"))
    common(out, None)
    assert "已更正" in system(out) and "之前记成“本人的职业是护士”" in system(out)


def test_case5_cancellation():
    out, _ = ask(rt(), "杭州的酒店还订着吗？", Fake("取消啦"))
    common(out, None)
    assert "已取消" in system(out) and "记录为已取消" in system(out)


def test_case6_upcoming():
    out, _ = ask(rt(), "我什么时候去成都出差？", Fake("20号去"))
    common(out, None)
    assert "已确定，尚未生效" in system(out) and "2026-10-20 起" in system(out)


def test_case7_conflict():
    out, _ = ask(rt(), "我在哪工作？", Fake("医院还是诊所 我不确定"))
    common(out, None)
    assert system(out).count("状态：有冲突，未确认") == 2 and system(out).count("冲突组：G1") == 2


def test_case8_third_party():
    out, _ = ask(rt(), "小林什么时候搬？", Fake("小林说十一月"))
    common(out, None)
    assert "[F1] 关于：小林" in system(out) and "小林打算搬去成都" in system(out)


def test_case9_protected_phone_is_redacted_and_blocked():
    out, _ = ask(rt(), "小邹手机号多少？", Fake(f"是{W.FRIEND_PHONE}", f"是{W.FRIEND_PHONE}"))
    common(out, None)
    assert "不可透露" in system(out) and W.FRIEND_PHONE not in out.text and out.status == "FALLBACK"


def test_case9b_protected_value_blocked_even_with_the_guard_off():
    r = rt()
    s = r.create_session(W.P)
    out = r.reply_session(W.P, s, "小邹手机号多少？", "x", history=H, generate=Fake(f"{W.FRIEND_PHONE}"), backend="lora", env={"GROUNDING_V2_ENABLED": "1"})
    assert W.FRIEND_PHONE not in out.text and out.trace.get("policy_output_blocked") and out.status == "FALLBACK"


def test_case10_policy_uncertain():
    out, _ = ask(rt(), "我手机号是多少？", Fake(f"{W.OWN_PHONE}", f"{W.OWN_PHONE}"))
    common(out, None)
    assert "未确认能否透露" in system(out) and W.OWN_PHONE not in out.text


def test_case11_allowed_contact_is_answered():
    out, _ = ask(rt(), "我邮箱是什么？", Fake(f"{W.EMAIL}"))
    common(out, None)
    assert W.EMAIL in system(out) and "｜可以说" in system(out) and out.text == W.EMAIL and out.status == "GENERATED"


def test_case12_multiple_facts():
    out, _ = ask(rt(), "我住哪，平时喜欢干什么？", Fake("西湖区 爱爬山"))
    common(out, None)
    assert "杭州西湖区" in system(out) and "爬山" in system(out) and out.trace["contract"]["evidence_items"] >= 2


def test_case13_context_reference():
    r = rt()
    s = r.create_session(W.P)
    r.reply_session(W.P, s, "我下周去成都。", "a", history=H, generate=Fake("好呀"), backend="lora", env=ON)
    out = r.reply_session(W.P, s, "那天几号来着？", "b", history=H, generate=Fake("20号"), backend="lora", env=ON)
    common(out, r)
    assert out.trace["query"]["completed"] and "成都" in out.trace["query"]["retrieval_query"] and "去成都出差" in system(out)
    assert out.messages[1]["content"] == "【对话】\n对方：我下周去成都。\n我：好呀\n对方：那天几号来着？"


def test_case14_long_message_fits_the_d4_cutoff():
    r = rt()
    s = r.create_session(W.P)
    doc = "".join(f"第{n}部分\n" + ("这一部分讲的是仓库管理的细节。" * 80) + "\n" for n in "一二三四五")
    r.send_message(W.P, s, doc, "doc")
    out = r.reply_session(W.P, s, "第三部分讲什么？", "q", history=H, generate=Fake("仓库管理"), backend="lora", env=ON)
    common(out, r)
    assert out.status in ("GENERATED", "FALLBACK", "CONTEXT_INSUFFICIENT")
    if out.messages:
        assert out.messages[1]["content"].endswith("对方：第三部分讲什么？")


def test_case15_cross_session_isolation():
    r = rt()
    a, b = r.create_session(W.P), r.create_session(W.P)
    r.reply_session(W.P, a, "新项目代号叫北极星", "a", history=H, generate=Fake("好"), backend="lora", env=ON)
    out = r.reply_session(W.P, b, "那个项目代号是什么？", "b", history=H, generate=Fake("不清楚"), backend="lora", env=ON)
    assert "北极星" not in json.dumps(out.messages, ensure_ascii=False)


def test_case16_generated_reply_is_not_evidence():
    r = rt()
    s = r.create_session(W.P)
    u = r.send_message(W.P, s, "你同事电话多少？", "u1")
    r.store.append(W.P, s, "assistant", "同事电话是13700000000，月薪3万", r.clock(), "u1", status="GENERATED", reply_to=u.message_id)
    out = r.reply_session(W.P, s, "再说一遍号码？", "u2", history=H, generate=Fake("13700000000", "13700000000"), backend="lora", env=ON)
    assert "13700000000" not in out.text and out.status == "FALLBACK"


def test_case17_greeting_is_not_memory_gated():
    out, _ = ask(rt(), "你好呀", Fake("嗨 在呢"))
    common(out, None)
    assert out.trace["route"] == "SMALL_TALK" and out.text == "嗨 在呢" and out.status == "GENERATED"
    assert "（没有提供记忆证据" in system(out)


def test_case17b_greeting_reply_with_invented_specifics_is_still_guarded():
    out, _ = ask(rt(), "在吗", Fake("在 我刚在锦江宾馆8801房", "在 我刚在锦江宾馆8801房"))
    assert out.trace["route"] == "SMALL_TALK" and "8801" not in out.text


def test_case18_greeting_with_a_fact_request_uses_memory():
    out, _ = ask(rt(), "你好，我手机号多少？", Fake(W.OWN_PHONE, W.OWN_PHONE))
    common(out, None)
    assert out.trace["route"] == "MEMORY" and W.OWN_PHONE not in out.text


def test_case19_k10_like_partial_attribution():
    say = lambda q, t, ev: {"answer": "根据现有记忆，无法确定。仅有一条“去香港看演唱会”的计划，但没有日期，无法确认是否就是本周六。", "abstain": True}
    out, _ = ask(rt(), "我这周六有什么安排？", Fake("有个去香港看演唱会的计划 不过这周六的安排我没记录"), answerer=say)
    common(out, None)
    assert out.trace["reader_named_units_kept"] and any("香港" in f["statement"] for f in out.trace["grounded"]["facts"])
    assert "去香港看演唱会" in system(out) and "香港" in out.text and out.status == "GENERATED"
    # a reply that ignores the unknown day still keeps the named plan through the safe fallback (never a bare 没记录)
    out2, _ = ask(rt(), "我这周六有什么安排？", Fake("周六去香港", "周六去香港"), answerer=say)
    assert out2.status == "FALLBACK" and "香港看演唱会" in out2.text and "没有记录" in out2.text


def test_case20_base_and_adapter_get_identical_prompts():
    got = {}
    for backend in ("base", "lora"):
        r = rt()
        s = r.create_session(W.P, "same")
        gen = Fake("嗯")
        r.reply_session(W.P, s, "我是做什么的？", "q", history=H, generate=gen, backend=backend, env=ON)
        got[backend] = gen.calls[0][0]
    assert got["base"] == got["lora"]


def test_model_failure_in_d4_mode_stores_no_reply():
    r = rt()
    s = r.create_session(W.P)
    out = r.reply_session(W.P, s, "我现在住哪？", "q", history=H, generate=Boom(), backend="lora", env=ON)
    assert out.status == "FAILED" and out.assistant_message is None
