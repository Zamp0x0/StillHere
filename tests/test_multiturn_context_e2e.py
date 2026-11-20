"""Simple Context v1 — end to end on CPU with fake generators (NOT a real model): session isolation (CASE 5), recent turns in the prompt,
Base / Adapter receive identical messages, a one-message session gives exactly the single-turn runtime prompt, the frozen contract version,
and the D4 record (PROPOSED)."""
from datetime import datetime

from _ctx_world import HISTORY, V2, Fake, runtime
from stillhere.evaluation import context_dataset_v1 as cd
from stillhere.runtime import pipeline as rp
from stillhere.session import d4


def test_recent_three_turns_enter_the_prompt_in_order():
    rt, _ = runtime()
    s = rt.create_session(cd.CA)
    for i, (u, a) in enumerate([("我下周去深圳。", "好呀"), ("要待三天。", "那挺久的"), ("住福田那边。", "方便")]):
        rt.reply_session(cd.CA, s, u, f"r{i}", generate=Fake(a), backend="lora", env={})
    gen = Fake("订好了")
    r = rt.reply_session(cd.CA, s, "酒店定了没？", "r9", history=HISTORY, generate=gen, backend="lora", env=V2)
    assert r.messages[1]["content"] == "\n".join(["对方：我下周去深圳。", "我：好呀", "对方：要待三天。", "我：那挺久的", "对方：住福田那边。", "我：方便",
                                                  "对方：酒店定了没？"])


def test_case5_sessions_do_not_leak_into_each_other():
    rt, _ = runtime()
    a, b = rt.create_session(cd.CA), rt.create_session(cd.CA)
    rt.reply_session(cd.CA, a, "我们新项目代号叫北极星，先别告诉别人。", "r1", generate=Fake("好的"), backend="lora", env={})
    gen = Fake("不太清楚")
    r = rt.reply_session(cd.CA, b, "之前说的那个项目代号是什么？", "r1", history=HISTORY, generate=gen, backend="lora", env=V2)
    joined = "\n".join(m["content"] for m in r.messages)
    assert "北极星" not in joined and r.trace["history"]["status"] in ("NOT_FOUND", "NOT_NEEDED")


def test_base_and_adapter_receive_identical_messages():
    msgs = {}
    for backend in ("base", "lora"):
        rt, _ = runtime()
        s = rt.create_session(cd.CA, "same-id")
        rt.reply_session(cd.CA, s, "我下周去深圳。", "r1", generate=Fake("好"), backend=backend, env={})
        gen = Fake("订好了")
        rt.reply_session(cd.CA, s, "酒店定了没？", "r2", history=HISTORY, generate=gen, backend=backend, env=V2)
        msgs[backend] = gen.calls[0][0]
    assert msgs["base"] == msgs["lora"]


def test_one_message_session_equals_the_single_turn_runtime():
    rt, api = runtime()
    s = rt.create_session(cd.CA)
    gen = Fake("订好了")
    r = rt.reply_session(cd.CA, s, "深圳出差的酒店订好了吗？", "r1", history=HISTORY, generate=gen, backend="lora", env=V2)
    old = Fake("订好了")
    rp.reply(api, cd.CA, [{"speaker": "other", "text": "深圳出差的酒店订好了吗？"}], datetime.fromisoformat(r.trace["ask_time"]), history=HISTORY,
             generate=old, backend="lora", env=V2)
    assert gen.calls[0][0] == old.calls[0][0]
    assert r.trace["contract"]["prompt_schema_version"] == "stillhere.style.prompt/2.0"


def test_d4_record_is_proposed_and_hides_runtime_ids():
    rt, _ = runtime()
    s = rt.create_session(cd.CA, "sess-d4")
    rt.reply_session(cd.CA, s, "我下周去深圳。", "r1", generate=Fake("好"), backend="lora", env={})
    r = rt.reply_session(cd.CA, s, "酒店定了没？", "r2", history=HISTORY, generate=Fake("订好了"), backend="lora", env=V2)
    rec = d4.record_from_reply(cd.CA, s, r)
    assert rec["schema_version"] == "stillhere.response.grounded_persona/1.0" and rec["status"] == "PROPOSED"
    vis = d4.model_visible(rec)
    blob = repr(vis)
    assert "sess-d4" not in blob and "lm." not in blob and cd.CA not in blob and "prompt_hash" not in blob
    assert vis["current_user_message"] == "酒店定了没？" and [c["role"] for c in vis["recent_conversation"]] == ["user", "assistant"]
    assert vis["memory_evidence"]["facts"] and set(d4.D4_REQUIRED_FIELDS) <= set(vis)


def test_d4_compatible_preset_matches_the_d4_training_distribution():
    from stillhere.session.config import D4_COMPATIBLE
    assert D4_COMPATIBLE.recent_turn_limit == 4 and D4_COMPATIBLE.total_input_token_budget == 1536 and D4_COMPATIBLE.history_search_top_k == 0
    assert d4.STATUS == "PROPOSED" and any("disclosure_policy" in p for p in d4.PENDING_ALIGNMENT)
