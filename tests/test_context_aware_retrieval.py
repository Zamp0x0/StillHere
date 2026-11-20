"""Simple Context v1 — follow-up questions reach the right memory (CASES 1, 2, 3, 4, 18). The query builder only adds earlier USER words
to what is retrieved and asked; the memory reader / grounding stay the judges of facts; the original question is kept unchanged."""
from _ctx_world import HISTORY, ON, V2, Fake, runtime
from stillhere.evaluation import context_dataset_v1 as cd
from stillhere.runtime import pipeline as rp
from stillhere.session import query as qb

ABSTAIN = staticmethod(lambda q, t, ev: {"answer": "根据现有记忆，无法确定。", "abstain": True})


def facts(r):
    return [f["statement"] for f in r.trace["grounded"]["facts"]]


def test_case1_hotel_follow_up_is_about_the_shenzhen_trip():
    rt, api = runtime()
    s = rt.create_session(cd.CA)
    rt.reply_session(cd.CA, s, "我下周去深圳。", "r1", env=V2)
    r = rt.reply_session(cd.CA, s, "酒店定了没？", "r2", env=V2)
    q = r.trace["query"]
    assert q["completed"] and q["original_question"] == "酒店定了没？" and "深圳" in q["retrieval_query"] and "深圳" in q["reader_question"]
    assert facts(r)[0].startswith("本人打算去深圳出差") and any("深圳出差的酒店" in f for f in facts(r))
    *_, bare, _ = rp.ground(api, cd.CA, "酒店定了没？", cd.t(9, 10, 21), env=V2)                    # without the context: the wrong hotel first
    assert "杭州" in bare["facts"][0]["statement"]


def test_query_builder_never_borrows_assistant_words():
    class M:
        def __init__(self, role, text, mid):
            self.role, self.text, self.message_id = role, text, mid
    cq = qb.complete("他什么时候搬？", [M("user", "小林下个月搬成都。", "u1"), M("assistant", "他说要搬去重庆的南岸区", "a1")])
    assert cq.completed and "小林" in cq.retrieval_query and "重庆" not in cq.retrieval_query and cq.borrowed_message_ids == ["u1"]
    assert not qb.complete("我在哪家公司上班？", [M("user", "你好", "u1")]).completed                       # a self-contained question is not changed


def test_case2_third_party_stays_third_party():
    rt, _ = runtime()
    s = rt.create_session(cd.CA)
    rt.reply_session(cd.CA, s, "小林下个月搬成都。", "r1", env=V2)
    r = rt.reply_session(cd.CA, s, "他什么时候搬？", "r2", env=V2)
    move = [f for f in facts(r) if "成都" in f]
    assert move and all(f.startswith("小林") for f in move) and not any(f.startswith("本人") and "成都" in f for f in facts(r))


def test_case3_current_residence_is_the_corrected_one():
    rt, _ = runtime()
    s = rt.create_session(cd.CA)
    rt.reply_session(cd.CA, s, "我以前住武汉，现在搬广州。", "r1", env=V2)
    r = rt.reply_session(cd.CA, s, "现在住哪里？", "r2", env=V2)
    cur = [f for f in r.trace["grounded"]["facts"] if f["role"] == "current"]
    assert any("广州" in f["statement"] for f in cur) and not any("武汉" in f["statement"] for f in cur)


def test_case4_cancelled_stays_cancelled_and_unknown_payment_is_not_invented():
    rt, _ = runtime()
    s = rt.create_session(cd.CA)
    rt.reply_session(cd.CA, s, "酒店已经取消了。", "r1", env=V2)
    gen = Fake("还要付500块违约金", "还要付500块违约金")
    r = rt.reply_session(cd.CA, s, "那还需要付款吗？", "r2", history=HISTORY, generate=gen, backend="lora", env=ON, answerer=ABSTAIN.__func__)
    assert any("取消" in f for f in facts(r)) and r.trace["grounded"]["unknown"]
    assert "500" not in r.text and r.status == "FALLBACK"


def test_case18_partial_facts_keep_the_known_part_and_say_the_unknown():
    rt, _ = runtime()
    s = rt.create_session(cd.CA)
    gen = Fake("住在香格里拉", "住在香格里拉")
    r = rt.reply_session(cd.CA, s, "深圳出差住哪家酒店？", "r1", history=HISTORY, generate=gen, backend="lora", env=ON, answerer=ABSTAIN.__func__)
    g = r.trace["grounded"]
    assert g["answer_status"] == "partial" and g["unknown"] and any("酒店" in f["statement"] for f in g["facts"])
    assert "香格里拉" not in r.text and "没有记录" in r.text and ("酒店" in r.text or "深圳" in r.text)
