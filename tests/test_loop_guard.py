"""Loop Guard v1 (stillhere.runtime.loop_guard) — synthetic cases only: what it trims, what it must never touch, and where it sits in the
reply chain (LLM text -> Loop Guard -> the existing output guards)."""
import pytest

from _ctx_world import ON, Fake, runtime
from stillhere.evaluation import context_dataset_v1 as cd
from stillhere.runtime import loop_guard as lg

LOOP = "我觉得可以试试，" + "先试试，" * 6 + "然后再说。"


# ---------------------------------------------------------------- trims (HIGH confidence, first copy kept)
@pytest.mark.parametrize("text, expected", [
    ("然后又重启，又失败，" * 5, "然后又重启，又失败"),
    ("我上面和你说过了。" * 4, "我上面和你说过了。"),
    (LOOP, "我觉得可以试试，先试试，然后再说。"),
    ("毕业典礼毕业典礼毕业典礼毕业典礼", "毕业典礼"),
    ("你看" + "哈" * 40, "你看" + "哈" * 5),
    ("论坛说，你升级了什么，然后我就说，我升级了什么，" * 3 + "论坛说，你升", "论坛说，你升级了什么，然后我就说，我升级了什么"),
])
def test_consecutive_loops_are_collapsed_to_their_first_copy(text, expected):
    r = lg.guard(text)
    assert r.status == "TRIMMED" and r.loop_guard_applied and r.loop_detected and r.confidence == "HIGH"
    assert r.final_response == expected and r.removed_char_count == len(text) - len(expected) and r.raw_response == text
    it = iter(text)
    assert r.final_response and all(ch in it for ch in r.final_response)       # only deletions: the result is a subsequence of the raw text


# ---------------------------------------------------------------- never touched
@pytest.mark.parametrize("text", [
    "好好好", "哈哈哈哈哈哈", "不不不，不是这样", "真的真的，我没骗你", "嗯嗯嗯嗯嗯嗯嗯嗯嗯",
    "我2026年10月1日去杭州，10月3日回武汉，10月5日再去上海。",
    "要带：护照、充电器、充电器线、耳机、外套。", "1. 买菜\n2. 买菜单上的东西\n3. 买水果",
    "chrome 打不开，chrome 的设置也进不去，重装 chrome 也没用。",
    "他说“好的好的好的”，然后就走了。", "我大概记得是周三，不太确定，可能是周四。",
    "小林下个月搬去成都，小邹也说要去成都，我还没定去不去成都。",
    "转账 300 元，又转了 300 元，一共 600 元。", "我好像跟你说过，我好像也跟小林说过，我好像记不清了。",
    "-------------\n分割线下面是正文", "怎么样   是不是装上就可以了        就是开机能选的那种", "╮(╯▽╰)╭╮(╯▽╰)╭╮(╯▽╰)╭",
    "到了\n你在哪\n我去找你", "我那边还没定。\n等我问问小林。\n明天告诉你。",                          # multi-bubble
    "这个我不方便说，你直接问本人吧。", "我之前在医院上班，后来更正了，是在诊所。", "那个酒店已经取消了，不去杭州了。",
    "我记得是去深圳出差，具体哪天还没定，酒店倒是订好了。",                                         # partial unknown
    "小林说她下个月搬去成都，是她搬，不是我。",                                                      # third party
    "这事说来话长。" + "我们一开始是在论坛上认识的，后来一起做了个项目，项目做完以后他去了北京，我留在杭州，" * 1
    + "现在每年还会见一两次，聊聊各自的近况，有时候也会一起出去玩几天。",                           # long reply
    "", "嗯",
])
def test_ordinary_repetition_and_structure_are_left_alone(text):
    r = lg.guard(text)
    assert r.final_response == text and not r.loop_guard_applied and r.removed_char_count == 0


def test_escalating_fragments_are_flagged_not_trimmed():
    text = "我想，我想去，我想去成都，我想去成都看看"
    r = lg.guard(text)
    assert r.status == "DETECTED_NOT_TRIMMED" and r.confidence == "LOW" and r.final_response == text and not r.loop_guard_applied


def test_disabled_returns_the_text_unchanged():
    r = lg.guard(LOOP, enabled_=False)
    assert r.status == "DISABLED" and r.final_response == LOOP and not r.loop_detected


def test_flag_defaults_off_and_reads_the_environment():
    assert lg.enabled({}) is False
    assert lg.enabled({"LOOP_GUARD_ENABLED": "1"}) is True and lg.enabled({"LOOP_GUARD_ENABLED": "off"}) is False


def test_trace_keeps_no_text():
    t = lg.guard(LOOP).trace()
    assert "raw_response" not in t and "final_response" not in t and "先试试" not in repr(t) and t["pattern_chars"] > 0


def test_a_loop_of_facts_keeps_every_fact_once():
    text = "我住在杭州西湖区，手机尾号是3344。" * 4
    r = lg.guard(text)
    assert r.loop_guard_applied and "杭州西湖区" in r.final_response and "3344" in r.final_response
    assert r.final_response.count("杭州西湖区") == 1


# ---------------------------------------------------------------- the reply chain
def test_runtime_trims_before_the_output_guards_and_records_the_trace():
    rt, _ = runtime(loop_guard=True)
    s = rt.create_session(cd.CA)
    r = rt.reply_session(cd.CA, s, "在吗", "r1", generate=Fake("在的，" * 6 + "怎么了？"), backend="lora", env=ON)
    assert r.text == "在的，怎么了？"
    calls = r.trace["loop_guard"]["calls"]
    assert r.trace["loop_guard"]["enabled"] and calls and calls[0]["status"] == "TRIMMED" and "在的" not in repr(r.trace["loop_guard"])


def test_runtime_default_follows_the_flag_and_is_off():
    rt, _ = runtime()
    s = rt.create_session(cd.CA)
    raw = "好的，" * 2 + "明天见。"
    r = rt.reply_session(cd.CA, s, "在吗", "r1", generate=Fake(raw), backend="lora", env={})
    assert r.trace["loop_guard"]["enabled"] is False and r.trace["loop_guard"]["calls"][0]["status"] == "DISABLED"


def test_base_and_adapter_share_the_same_chain():
    out = {}
    for backend in ("base", "lora"):
        rt, _ = runtime(loop_guard=True)
        s = rt.create_session(cd.CA)
        out[backend] = rt.reply_session(cd.CA, s, "在吗", "r1", generate=Fake("在的，" * 6), backend=backend, env=ON).text
    assert out["base"] == out["lora"] == "在的"
