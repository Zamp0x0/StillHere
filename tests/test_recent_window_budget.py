"""Simple Context v1 — L1 recent window and the total input budget (CASES 7, 14): whole turns, at most 8, token budget, oldest unrelated
turns go first, the newest turn / current message / memory facts / system rules are never cut, deterministic, token mode reported."""
from dataclasses import replace

from _ctx_world import HISTORY, ON, V2, Fake, runtime
from stillhere.evaluation import context_dataset_v1 as cd
from stillhere.session import DEFAULT
from stillhere.session import window as wn


def _chat(rt, s, pairs):
    for i, (u, a) in enumerate(pairs):
        um = rt.send_message(cd.CA, s, u, f"u{i}")
        if a:
            rt.store.append(cd.CA, s, "assistant", a, rt.clock(), f"u{i}", status="GENERATED", reply_to=um.message_id)


def test_turns_are_whole_and_at_most_eight():
    rt, _ = runtime()
    s = rt.create_session(cd.CA)
    _chat(rt, s, [(f"第{i}个问题", f"第{i}个回答") for i in range(12)])
    cur = rt.send_message(cd.CA, s, "现在的问题", "cur")
    w = wn.build(rt.store, cd.CA, s, "现在的问题", cur.message_id, rt.count)
    assert len(w.turns) == 8 and all([m.role for m in t] == ["user", "assistant"] for t in w.turns)
    assert w.turns[-1][0].text == "第11个问题" and cur.message_id not in w.message_ids and w.older_exists


def test_over_budget_drops_oldest_unrelated_turns_first_and_keeps_the_newest():
    rt, _ = runtime(config=replace(DEFAULT, recent_context_token_budget=35))
    s = rt.create_session(cd.CA)
    _chat(rt, s, [("我们聊聊深圳的酒店吧", "好"), ("今天天气不错", "是啊"), ("晚饭吃了面条", "不错"), ("最近工作很忙", "辛苦")])
    cur = rt.send_message(cd.CA, s, "深圳的酒店怎么样", "cur")
    w = wn.build(rt.store, cd.CA, s, "深圳的酒店怎么样", cur.message_id, rt.count, rt.cfg)
    kept = [t[0].text for t in w.turns]
    assert "我们聊聊深圳的酒店吧" in kept and kept[-1] == "最近工作很忙" and "今天天气不错" not in kept   # related old turn kept, unrelated dropped
    assert w.tokens <= 35 and "RECENT_TRIMMED" in w.status and w.dropped_turns
    again = wn.build(rt.store, cd.CA, s, "深圳的酒店怎么样", cur.message_id, rt.count, rt.cfg)
    assert again.message_ids == w.message_ids and again.tokens == w.tokens                          # deterministic


def test_estimated_and_exact_token_modes_are_reported():
    rt, _ = runtime()
    assert rt.token_mode == "ESTIMATED" and rt.count("一二三四五六七八九十") == 11                         # chars x 1.05, rounded up
    rt2, _ = runtime(count=lambda t: len(t.encode("utf-8")), exact_tokens=True)
    assert rt2.token_mode == "EXACT" and rt2.count("一二") == 6


def test_current_message_memory_and_rules_are_never_cut():                 # CASE 14
    rt, _ = runtime(config=replace(DEFAULT, total_input_token_budget=900))
    s = rt.create_session(cd.CA)
    _chat(rt, s, [("随便聊聊" * 30, "嗯" * 50) for _ in range(6)])
    gen = Fake("订好了")
    r = rt.reply_session(cd.CA, s, "深圳出差的酒店订好了吗？", "q", history=HISTORY, generate=gen, backend="lora", env=V2)
    sysmsg, user = r.messages[0]["content"], r.messages[1]["content"]
    assert user.endswith("对方：深圳出差的酒店订好了吗？")                                                     # the current message, whole, last
    assert "深圳出差" in sysmsg and "【记忆事实】" in sysmsg                                                 # memory facts kept
    assert r.trace["input_tokens"] <= 900 - r.trace["output_reserve"]
    order = [d.split(" ")[0] + " " + d.split(" ")[1] for d in r.trace["budget_decisions"]]
    assert any(d.startswith("style examples") for d in r.trace["budget_decisions"])                        # examples give way first
    first_turn_drop = next((i for i, d in enumerate(r.trace["budget_decisions"]) if "recent turn" in d), None)
    last_example = max(i for i, d in enumerate(r.trace["budget_decisions"]) if d.startswith("style examples"))
    assert first_turn_drop is None or first_turn_drop > last_example and order


def test_fixed_parts_alone_over_budget_gives_context_insufficient_without_generation():
    rt, _ = runtime(config=replace(DEFAULT, total_input_token_budget=200))
    s = rt.create_session(cd.CA)
    gen = Fake("随便说点")
    r = rt.reply_session(cd.CA, s, "深圳出差的酒店订好了吗？", "q", history=HISTORY, generate=gen, backend="lora", env=ON)
    assert r.status == "CONTEXT_INSUFFICIENT" and gen.calls == [] and r.assistant_message.generation_status == "FALLBACK"
    assert r.user_message.generation_status == "CONTEXT_INSUFFICIENT"
