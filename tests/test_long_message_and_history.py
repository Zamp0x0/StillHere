"""Simple Context v1 — long messages (CASES 8, 12) and L2 session history search (CASES 7, 13, 17): deterministic chunks with offsets, the
referenced section and its neighbours shown under an explicit marker, honest AMBIGUOUS / NOT_FOUND, same-session only, Chinese search with
the FTS5 bigram index and the pure-Python fallback, GENERATED labels."""
from _ctx_world import HISTORY, ON, V2, Fake, runtime
from stillhere.evaluation import context_dataset_v1 as cd
from stillhere.session import history as hs
from stillhere.session import runtime as srt
from stillhere.session import window as wn

SECTIONS = ["第一部分 背景", "第二部分 现状", "第三部分 问题", "第四部分 方案", "第五部分 预算", "第六部分 风险"]
BODY = {"第三部分 问题": "仓库的温控系统经常失灵，夜班没人巡检，冷链货物损耗偏高。"}


def long_doc():
    parts = []
    for h in SECTIONS:
        body = BODY.get(h, f"这一部分是关于{h[-2:]}的说明。") + ("这里是填充内容，用来让文档足够长。" * 60)
        parts.append(f"{h}\n{body}\n")
    return "".join(parts)


def test_second_turn_long_message_is_chunked_stored_and_the_referenced_part_shown():   # CASES 8, 12
    rt, _ = runtime()
    s = rt.create_session(cd.CA)
    doc = long_doc()
    assert rt.count(doc) > 5000
    first = rt.send_message(cd.CA, s, doc, "doc")
    chunks = rt.store.chunks(first.message_id)
    assert first.chunked and len(chunks) >= 6 and all(doc[c.offset:c.offset + len(c.text)] == c.text for c in chunks)   # offsets index the original
    gen = Fake("温控和巡检")
    r = rt.reply_session(cd.CA, s, "第三部分有什么问题？", "q", history=HISTORY, generate=gen, backend="lora", env=V2)
    user = r.messages[1]["content"]
    assert "温控系统经常失灵" in user and "〔长消息，共" in user and "其余已保存但未显示" in user          # the part, under a marker
    assert "第六部分 风险" not in user and r.trace["input_tokens"] <= rt.cfg.total_input_token_budget         # not the whole document
    assert "LONG_MESSAGE_EXCERPT" in r.trace["window"]["status"]
    assert rt.get_session_messages(cd.CA, s)[0].content == doc                                               # stored whole, never truncated


def test_a_section_that_does_not_exist_is_ambiguous_not_guessed():
    rt, _ = runtime()
    s = rt.create_session(cd.CA)
    rt.send_message(cd.CA, s, long_doc(), "doc")
    gen = Fake("第九部分是关于人员的")
    r = rt.reply_session(cd.CA, s, "第九部分说了什么？", "q", history=HISTORY, generate=gen, backend="lora", env=ON)
    assert "AMBIGUOUS_CONTEXT" in r.trace["window"]["status"] and srt.LONG_UNKNOWN in r.trace["grounded"]["unknown"]
    assert r.status in ("FALLBACK", "GENERATED") and "人员" not in (r.text or "")                         # the unknown gate rejects a guess


def _long_session(rt, s, n=30):
    rt.send_message(cd.CA, s, "我第一个想法是在钱塘江边开一家自行车租赁店", "u0")
    rt.store.append(cd.CA, s, "assistant", "自行车租赁店听起来不错", rt.clock(), "u0", status="GENERATED")
    for i in range(1, n):
        um = rt.send_message(cd.CA, s, f"闲聊第{i}句：今天吃了什么", f"u{i}")
        rt.store.append(cd.CA, s, "assistant", f"随便聊聊{i}", rt.clock(), f"u{i}", status="GENERATED", reply_to=um.message_id)


def test_turn_30_finds_turn_1_in_the_same_session_only():                 # CASE 7
    rt, _ = runtime()
    s, other = rt.create_session(cd.CA), rt.create_session(cd.CA)
    rt.send_message(cd.CA, other, "另一个会话里提到的自行车计划", "x")
    _long_session(rt, s)
    gen = Fake("自行车租赁店")
    r = rt.reply_session(cd.CA, s, "还记得我之前说的自行车那个想法吗？", "q", history=HISTORY, generate=gen, backend="lora", env=V2)
    h = r.trace["history"]
    assert h["status"] == "FOUND" and 1 <= len(h["excerpts"]) <= 2 and h["engine"] in ("fts5_bigram", "python_bigram")
    top = h["excerpts"][0]
    assert top["seq"] == 1 and top["role"] == "user" and not top["generated"] and top["source"] == "live_chat"
    user = r.messages[1]["content"]
    assert "〔较早的对话，第1条〕我第一个想法是在钱塘江边开一家自行车租赁店" in user and "另一个会话" not in user


def test_python_fallback_finds_the_same_two_character_words():           # CASE 13
    rt, _ = runtime()
    s = rt.create_session(cd.CA)
    _long_session(rt, s)
    w = wn.build(rt.store, cd.CA, s, "之前说的自行车", "none", rt.count)
    args = (rt.store, cd.CA, s, "之前说的自行车", "之前说的自行车", w.message_ids, "", True)
    fts = hs.search(*args, at=cd.t(12, 31))
    py = hs.search(*args, force_python=True, at=cd.t(12, 31))
    assert rt.api.store.message_fts_enabled and fts.engine == "fts5_bigram" and py.engine == "python_bigram"
    assert fts.excerpts[0].message_id == py.excerpts[0].message_id and fts.excerpts[0].seq == 1


def test_assistant_excerpts_are_labelled_generated():
    rt, _ = runtime()
    s = rt.create_session(cd.CA)
    _long_session(rt, s)
    w = wn.build(rt.store, cd.CA, s, "你之前说的租赁店", "none", rt.count)
    res = hs.search(rt.store, cd.CA, s, "你之前说的租赁店", "租赁店", w.message_ids, "", True, at=cd.t(12, 31))
    gen = [e for e in res.excerpts if e.role == "assistant"]
    assert gen and all(e.generated for e in gen)


def test_not_found_history_is_admitted_not_invented():                    # CASE 17
    rt, _ = runtime()
    s = rt.create_session(cd.CA)
    _long_session(rt, s, 12)
    gen = Fake("你之前说要去冰岛看极光", "你之前说要去冰岛看极光")
    r = rt.reply_session(cd.CA, s, "我之前说的潜水计划是什么来着？", "q", history=HISTORY, generate=gen, backend="lora", env=ON)
    assert r.trace["history"]["status"] == "NOT_FOUND" and srt.HISTORY_UNKNOWN in r.trace["grounded"]["unknown"]
    assert r.status == "FALLBACK" and "冰岛" not in r.text and ("没有记录" in r.text or "不太清楚" in r.text)
