"""Simple Context v1.1 — small-talk routing, the k10 reader-named-units rule, exact Qwen3 token counting, and the pinned D4 renderer."""
import hashlib
import shutil
from pathlib import Path

import pytest

import _d4_world as W
from stillhere.runtime import pipeline as rp
from stillhere.session import chat_route as cr
from stillhere.session import d4_renderer as dr
from stillhere.session import grounding_compat as gc
from stillhere.session import tokens as tk

V2 = {"GROUNDING_V2_ENABLED": "1"}
D4_SRC = Path(__file__).resolve().parents[1] / "training" / "d4" / "t2d4"     # the D4 toolkit shipped in this repository


# ---------------------------------------------------------------- chat routing
@pytest.mark.parametrize("text", ["你好", "你好呀", "在吗", "在吗在吗？", "早上好", "哈哈", "哈哈哈哈", "晚安~", "👍", "😂😂", "谢谢！", "ok", "嗯嗯"])
def test_pure_small_talk_is_routed(text):
    assert cr.is_small_talk(text)


@pytest.mark.parametrize("text", ["你手机号呢", "你住哪里？", "你还记得我吗？", "你以前在哪里工作？", "你朋友手机号是什么？", "你刚才说的第二个方案是什么？",
                                  "昨天是不是取消了？", "你好，我手机号多少？", "在吗？酒店定了没", "哈哈你住哪", "好的，那几点出发？", ""])
def test_fact_requests_are_never_small_talk(text):
    assert not cr.is_small_talk(text)


# ---------------------------------------------------------------- k10: an abstention that names evidence
def _ground(answerer):
    api = W.build()
    m, reader, _, g, _ = rp.ground(api, W.P, "我这周六有什么安排？", W.NOW, env=V2, answerer=answerer)
    text = reader.pop("text", None)
    return m, reader, g, gc.keep_reader_named_units(m, dict(g), reader, text, W.NOW)


def test_k10_named_undated_plan_is_kept_as_partial():
    say = lambda q, t, ev: {"answer": "根据现有记忆，无法确定。仅有一条“去香港看演唱会”的计划，但没有日期，无法确认是否就是本周六。", "abstain": True}
    m, reader, before, after = _ground(say)
    assert not any("香港" in f["statement"] for f in before.get("facts", []))       # what grounding v2 alone keeps (the k10 loss)
    kept = [f for f in after["facts"] if "香港" in f["statement"]]
    assert kept and kept[0]["statement"].startswith("本人打算去香港看演唱会") and "没有记录具体日期" in kept[0]["statement"]
    assert after["answer_status"] == "partial" and after["unknown"]                  # the asked day stays unknown


def test_k10_rule_ignores_names_that_are_not_evidence_and_plain_abstentions():
    invented = lambda q, t, ev: {"answer": "根据现有记忆，无法确定。可能是去上海看话剧吧。", "abstain": True}
    *_, after = _ground(invented)
    assert not any("话剧" in f["statement"] or "上海" in f["statement"] for f in after.get("facts", []))
    plain = lambda q, t, ev: {"answer": "根据现有记忆，无法确定。", "abstain": True}
    m, reader, before, after2 = _ground(plain)
    assert after2 == before


def test_k10_third_party_unit_keeps_its_owner():
    api = W.build()
    say = lambda q, t, ev: {"answer": "根据现有记忆，无法确定。记录里搬去成都的是小林，不是你。", "abstain": True}
    m, reader, _, g, _ = rp.ground(api, W.P, "我要搬家吗？", W.NOW, env=V2, answerer=say)
    after = gc.keep_reader_named_units(m, dict(g), reader, reader.pop("text", None), W.NOW)
    moved = [f["statement"] for f in after["facts"] if "搬去成都" in f["statement"]]
    assert moved and all(s.startswith("小林") for s in moved)


# ---------------------------------------------------------------- tokens
def test_exact_tokenizer_is_pinned_and_reproducible():
    try:
        c = tk.ExactTokenCounter()
    except tk.TokenizerUnavailable as e:
        pytest.skip(f"EXACT tokenizer unavailable here: {e}")            # reported as CONDITIONAL in the gate, never as PASS
    assert c.mode == "EXACT" and c.sha256 == tk.TOKENIZER_SHA256
    msgs = [{"role": "system", "content": "你好"}, {"role": "user", "content": "【对话】\n对方：在吗"}]
    n = c.chat(msgs)
    assert n == c.chat(msgs) and n > c("你好") + c("【对话】\n对方：在吗")       # the chat layout's special tokens are counted
    assert c("<|im_end|>") == 1 and c.suffix() == 2


def test_estimated_counter_is_labelled_and_conservative():
    e = tk.EstimatedTokenCounter()
    assert e.mode == "ESTIMATED" and e("一二三四五六七八九十") == 11


def test_d4_layout_equals_the_d4_training_layout():
    src = D4_SRC / "tokens.py"
    if not src.exists():
        pytest.fail("training/d4/t2d4/tokens.py is missing")
    text = src.read_text(encoding="utf-8")
    assert repr(tk.PROMPT_FMT)[1:-1] in text and tk.TOKENIZER_SHA256 in text


# ---------------------------------------------------------------- renderer pin
def test_vendored_renderer_is_byte_identical_to_the_d4_source():
    info = dr.verify()
    assert info["renderer_sha256"] == dr.PINNED_RENDERER_SHA256 == dr.renderer_sha256()
    if not D4_SRC.exists():
        pytest.fail("training/d4/t2d4 is missing")
    for name in dr.PINNED_FILES:
        assert dr.file_sha256(D4_SRC / name) == dr.PINNED_FILES[name]


def test_a_modified_renderer_copy_is_refused(tmp_path):
    vend = Path(dr.__file__).parent / "vendor_d4"
    for f in vend.glob("*.py"):
        shutil.copy(f, tmp_path / f.name)
    (tmp_path / "render.py").write_text((tmp_path / "render.py").read_text(encoding="utf-8").replace("不要猜", "可以猜"), encoding="utf-8")
    with pytest.raises(dr.RendererMismatch):
        dr.verify(tmp_path)


def test_one_disclosure_policy_for_runtime_and_d4_toolkit():
    """training/d4 vendors the runtime disclosure policy: both must be the same bytes (one policy, one contract)."""
    runtime = Path(__file__).resolve().parents[1] / "src" / "stillhere" / "session" / "disclosure.py"
    vendored = D4_SRC / "vendor" / "context_disclosure_v1_0.py"
    norm = lambda p: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert norm(runtime) == norm(vendored)
