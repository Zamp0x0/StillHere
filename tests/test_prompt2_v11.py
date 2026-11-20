"""Simple Context v1.1 in the default (prompt/2.0) mode: greetings are no longer replaced by "没有记录"; protected values are redacted before
the frozen style prompt is built and blocked on output; the earlier v1 behaviour of a normal question is unchanged."""
import json

import _d4_world as W
from _ctx_world import HISTORY, Clock, Fake
from stillhere.session import SessionRuntime
from stillhere.style import contract as sc

ON = {"GUARD_ENABLED": "1", "GROUNDING_V2_ENABLED": "1"}
H = [sc.StyleMessage(f"{W.P}.h{i}", W.P, h.text, topic=h.topic) for i, h in enumerate(HISTORY)]


def test_greeting_with_the_guard_on_is_answered_naturally():
    r = SessionRuntime(W.build(), clock=Clock(W.NOW))
    s = r.create_session(W.P)
    out = r.reply_session(W.P, s, "在吗", "a", history=H, generate=Fake("在呢"), backend="lora", env=ON)
    assert out.status == "GENERATED" and out.text == "在呢" and out.trace["route"] == "SMALL_TALK"


def test_protected_value_never_reaches_the_prompt2_prompt():
    r = SessionRuntime(W.build(), clock=Clock(W.NOW))
    s = r.create_session(W.P)
    out = r.reply_session(W.P, s, "小邹手机号多少？", "a", history=H, generate=Fake(W.FRIEND_PHONE, W.FRIEND_PHONE), backend="lora", env=ON)
    blob = json.dumps(out.messages, ensure_ascii=False) + json.dumps(out.trace, ensure_ascii=False)
    assert W.FRIEND_PHONE not in blob and W.FRIEND_PHONE not in out.text and "不可透露" in out.messages[0]["content"]
