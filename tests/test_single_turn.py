"""prepare_single_turn + finish_single_turn == the first message of a new session (reply_session), without storing anything.
Used by benchmarks that must freeze the prompt at retrieval time (LifeBench: SEARCH at the question's date, ANSWER later)."""
import json

import pytest

from _ctx_world import ON, Clock, Fake, world
from stillhere.evaluation import context_dataset_v1 as cd
from stillhere.session import SessionRuntime
from stillhere.session.config import D4_COMPATIBLE
from stillhere.session.runtime import PreparedReply

QUESTIONS = ["我下周去哪出差？", "杭州的酒店还订着吗？", "你好", "我之前跟你说过我住哪吗？", "小林什么时候搬去成都？", "我养猫吗？"]


def _rt(api, mode):
    cfg = D4_COMPATIBLE if mode == "d4" else None
    return SessionRuntime(api, config=cfg, clock=Clock(cd.t(9, 10, 20)), render_mode=mode, loop_guard=False)


@pytest.mark.parametrize("mode", ["d4", "prompt2"])
@pytest.mark.parametrize("q", QUESTIONS)
def test_single_turn_equals_a_fresh_session(mode, q):
    api = world().api
    a = _rt(api, mode)
    s = a.create_session(cd.CA)
    gen_a = Fake("我打算去深圳出差。")
    r1 = a.reply_session(cd.CA, s, q, "r1", generate=gen_a, backend="lora", env=ON)
    b = _rt(world().api, mode)
    at = __import__("datetime").datetime.fromisoformat(r1.trace["ask_time"])
    prep = b.prepare_single_turn(cd.CA, q, at, env=ON)
    gen_b = Fake("我打算去深圳出差。")
    r2 = b.finish_single_turn(PreparedReply.from_json(json.loads(json.dumps(prep.to_json(), ensure_ascii=False))), gen_b, backend="lora", env=ON)
    assert gen_a.calls[0][0] == gen_b.calls[0][0]                      # the model receives the same messages
    assert (r1.status, r1.text) == (r2.status, r2.text)
    assert r2.user_message is None and r2.assistant_message is None
    assert r1.trace["grounded"] == r2.trace["grounded"]


def test_single_turn_writes_nothing():
    api = world().api
    rt = _rt(api, "d4")                    # creating the runtime registers its live-session source once
    before = api.stats()
    prep = rt.prepare_single_turn(cd.CA, "我下周去哪出差？", cd.t(9, 10, 21), env=ON)
    rt.finish_single_turn(prep, Fake("好"), backend="lora", env=ON)
    assert api.stats() == before
