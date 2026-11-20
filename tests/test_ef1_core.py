"""EF1 (Evidence-First Retrieval v1): query preprocessing, event-time understanding, question form, retrieval channels, fusion, fact pack,
reader interface. All data is synthetic; no network, no model."""
import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from stillhere.contracts import CanonicalMessage, DataOrigin, MemoryKind, MemoryRecord, Provenance, SourceKind, SourceRef
from stillhere.memory import MemoryAPI
from stillhere.retrieval.ef1.pipeline import EF1Config, retrieve
from stillhere.retrieval.ef1 import shape as sh
from stillhere.retrieval.ef1.query import prepare, strip_ask_prefix
from stillhere.retrieval.ef1.temporal import add_months, cn_number, find_mentions, resolve

ROOT = Path(__file__).resolve().parents[1]
TZ = timezone(timedelta(hours=8))
A, B = "p.alpha", "p.beta"


def ask(d: str) -> datetime:
    return datetime.fromisoformat(f"{d}T23:59:59").replace(tzinfo=TZ)


def windows(text, asked="2025-12-31"):
    t = resolve(text, ask(asked))
    return [(w.start.isoformat(), w.end.isoformat(), w.role) for w in t.windows], t


# ---------------------------------------------------------------- query preprocessing
@pytest.mark.parametrize("q,body,d", [
    ("（提问时间：2025-12-15）我3月10日之后34天做了什么？", "我3月10日之后34天做了什么？", date(2025, 12, 15)),
    ("(提问时间:2025-12-15) 我做了什么", "我做了什么", date(2025, 12, 15)),
    ("（提问日期：2025年3月5日）那天下雨了吗", "那天下雨了吗", date(2025, 3, 5)),
    ("（当前时间：2025/07/01 09:30）最近怎么样", "最近怎么样", date(2025, 7, 1)),
    ("（提问时间：２０２５－０４－１５）全角日期", "全角日期", date(2025, 4, 15)),
    ("我3月10日去了哪？", "我3月10日去了哪？", None),
])
def test_ask_prefix_is_separated_from_the_body(q, body, d):
    b, got = strip_ask_prefix(q)
    assert b == body and got == d


def test_body_keeps_its_own_dates_and_punctuation():
    p = prepare("（提问时间：2025-12-15）我在3月10日，和妈妈去了哪？", ask("2025-12-15"))
    assert p.body == "我在3月10日，和妈妈去了哪？" and "3月10日" in p.body and "，" in p.body
    assert p.prefix_date == date(2025, 12, 15) and not p.warnings


def test_prefix_and_ask_time_mismatch_is_a_warning_and_ask_time_wins():
    p = prepare("（提问时间：2025-12-15）问题", ask("2025-12-16"))
    assert p.warnings and "ASK_PREFIX_MISMATCH" in p.warnings[0] and p.ask_date == date(2025, 12, 16)


def test_prefix_date_never_becomes_an_event_window():
    p = prepare("（提问时间：2025-12-15）我一共收到过多少出版预付款？", ask("2025-12-15"))
    w, _ = windows(p.body, "2025-12-15")
    assert w == []                                               # the v3 defect: the ask date became the event window


# ---------------------------------------------------------------- temporal
def test_explicit_day_and_year_inference():
    w, _ = windows("我3月8号下午做了什么", "2025-06-15")
    assert w == [("2025-03-08", "2025-03-09", "mention")]
    w, _ = windows("12月5日我做了什么", "2025-03-15")             # a month-day after the ask date is last year's
    assert w[0][0] == "2024-12-05"
    w, t = windows("我计划12月5日去体检，可以吗", "2025-03-15")     # a plan may be in the future
    assert w[0][0] == "2025-12-05" and t.plan
    w, _ = windows("去年12月30日我做了什么", "2025-03-15")
    assert w[0][0] == "2024-12-30"
    w, _ = windows("2024年2月29日我在哪", "2025-03-15")             # leap day with an explicit year
    assert w[0][:2] == ("2024-02-29", "2024-03-01")


def test_iso_and_english_dates():
    assert windows("2025-03-10 发生了什么")[0][0][:2] == ("2025-03-10", "2025-03-11")
    assert windows("what did I do on March 10, 2025")[0][0][:2] == ("2025-03-10", "2025-03-11")
    assert windows("on 10 Mar 2025")[0][0][:2] == ("2025-03-10", "2025-03-11")


def test_ranges_months_quarters_and_years():
    assert windows("2025年5月10日到12日，我的情绪")[0] == [("2025-05-10", "2025-05-13", "range")]
    assert windows("3月1日到3月20日之间")[0][0][:2] == ("2025-03-01", "2025-03-21")
    assert windows("2024年12月30日到2025年1月2日")[0][0][:2] == ("2024-12-30", "2025-01-03")      # across the year
    assert windows("12月30日至1月2日我在哪", "2025-01-05")[0][0][:2] == ("2024-12-30", "2025-01-03")
    assert windows("回顾2025年10月")[0] == [("2025-10-01", "2025-11-01", "range")]
    assert windows("10月上半月和下半月重心变化")[0] == [("2025-10-01", "2025-10-16", "range"), ("2025-10-16", "2025-11-01", "range")]
    assert windows("2月份我学了什么", "2024-06-01")[0][0][:2] == ("2024-02-01", "2024-03-01")      # leap February
    assert windows("第一季度我看了几次电影")[0][0][:2] == ("2025-01-01", "2025-04-01")
    assert windows("我今年做了几件事")[0][0][:2] == ("2025-01-01", "2026-01-01")
    assert windows("上个月我去了哪", "2025-03-15")[0][0][:2] == ("2025-02-01", "2025-03-01")


def test_offsets_are_computed_deterministically():
    w, t = windows("（提问时间：2025-12-15）我3月10日之后34天做了什么".split("）", 1)[1])
    assert ("2025-04-13", "2025-04-14", "target") in w and ("2025-03-10", "2025-03-11", "anchor") in w
    assert any("2025年4月13日" in n for n in t.notes)
    w, _ = windows("3月10日项目启动会议后，34天后我在社区做了什么")
    assert ("2025-04-13", "2025-04-14", "target") in w
    w, t = windows("5月31日报告展示后的第167天，我做了什么运动")
    assert ("2025-11-14", "2025-11-15", "target") in w and ("2025-11-13", "2025-11-14", "target_alt") in w   # both readings of an ordinal
    assert [x for x in t.windows if x.role == "target_alt"][0].weight < 1.0
    w, _ = windows("12月24日绩效面谈前3天我在做什么")
    assert ("2025-12-21", "2025-12-22", "target") in w
    w, _ = windows("1月5日两周后我去了哪")
    assert ("2025-01-19", "2025-01-20", "target") in w
    w, _ = windows("1月31日一个月后我在哪")
    assert ("2025-02-28", "2025-03-01", "target") in w                                   # month arithmetic clips to the month end
    w, _ = windows("2024年2月28日后的第二天", "2025-01-01")                              # "后的第二天" = the next day
    assert ("2024-02-29", "2024-03-01", "target") in w
    w, _ = windows("2023年2月28日后的第二天", "2025-01-01")
    assert ("2023-03-01", "2023-03-02", "target") in w


def test_intervals_and_event_anchors():
    _, t = windows("3月1日和3月20日之间间隔多少天")
    assert t.intervals == [(date(2025, 3, 1), date(2025, 3, 20), 19)] and any("19 天" in n and "20 天" in n for n in t.notes)
    _, t = windows("投票站开放前5天下午我用什么交通方式回家")
    assert t.lookups and t.lookups[0].days == -5 and "投票站开放" in t.lookups[0].phrase
    w, t = windows("我最近一次接种疫苗的原因是什么")
    assert w == [] and not t.lookups


def test_chinese_numbers_and_month_arithmetic():
    assert [cn_number(x) for x in ("34", "三十四", "十", "一百零五", "两", "十二")] == [34, 34, 10, 105, 2, 12]
    assert add_months(date(2024, 1, 31), 1) == date(2024, 2, 29) and add_months(date(2025, 12, 15), 1) == date(2026, 1, 15)


def test_find_mentions_reports_explicit_years():
    ms = find_mentions("2025年3月1日和4月2日", date(2025, 12, 31))
    assert [m.explicit_year for m in ms] == [True, False]


# ---------------------------------------------------------------- question form
def test_shape_flags():
    s = sh.detect("我今年做了几件事，能帮我按时间顺序排一下吗？\n1. 去北京出差\n2. 参加婚礼\n3. 买车")
    assert s.has(sh.ENUMERATED_LIST) and s.has(sh.ORDERING) and s.items == ["去北京出差", "参加婚礼", "买车"]
    assert sh.subqueries("x\n1. 去北京出差\n2. 参加婚礼", sh.detect("x\n1. 去北京出差\n2. 参加婚礼")) == ["去北京出差", "参加婚礼"]
    assert sh.detect("我计划在4月25日晚上参加一个家庭聚会，这个安排怎么样？").has(sh.PLAN_CHECK)
    assert sh.detect("11月我一共跑了几次步").has(sh.COUNT)
    assert sh.detect("回顾12月，我健身训练通常安排在周几？有什么固定规律吗？").has(sh.RANGE_SUMMARY)
    assert sh.detect("我跑了5公里，用了多久").primary in (sh.INTERVAL, sh.SINGLE)
    assert sh.detect("我在西青区有自己的房子吗").primary in (sh.SINGLE, sh.STATE_NOW)
    assert sh.list_items("跑了5公里，3. 是错的") == []


# ---------------------------------------------------------------- a small synthetic world (two personas)
def _msg(pid, mid, day, content, loc=None, src="lifebench_note", speaker=None):
    t = datetime.fromisoformat(f"{day}T12:00:00").replace(tzinfo=TZ)
    return CanonicalMessage(conversation_id=f"{pid}.c", message_id=mid, target_persona_id=pid, speaker_id=speaker or pid, timestamp=t,
                            content=content, source_type=src, source_id=f"{pid}.src", locator=loc or f"{day}_{mid}", recorded_at=t)


def _fact(pid, mid, day, statement, msg_id, attribute="note", value="v", **kw):
    t = datetime.fromisoformat(f"{day}T12:00:01").replace(tzinfo=TZ)
    return MemoryRecord(memory_id=mid, persona_id=pid, kind=MemoryKind.FACT, subject="persona", attribute=attribute, value=value, statement=statement,
                        provenance=(Provenance(source_id=f"{pid}.src", locator=f"message:{msg_id}"),), origin=DataOrigin.SYNTHETIC, recorded_at=t, **kw)


@pytest.fixture
def world():
    api = MemoryAPI(clock=lambda: ask("2026-01-01"))
    for pid in (A, B):
        api.register_source(SourceRef(source_id=f"{pid}.src", kind=SourceKind.SIMULATION, origin=DataOrigin.SYNTHETIC, description="synthetic"))
    msgs = [_msg(A, "a1", "2025-03-10", "项目启动会议，分工完成", src="lifebench_calendar"),
            _msg(A, "a2", "2025-04-13", "在社区卫生服务中心就诊，医生诊断为轻度腰肌劳损"),
            _msg(A, "a3", "2025-12-15", "今天翠湖晨跑5公里"),
            _msg(A, "a4", "2025-07-06", "收到出版社预付款3000元", src="lifebench_sms"),
            _msg(A, "a5", "2025-07-08", "第二笔出版预付款8000元到账", src="lifebench_sms"),
            _msg(A, "a6", "2026-02-01", "第三笔出版预付款5000元到账（未来）", src="lifebench_sms"),     # after every ask time used below
            _msg(A, "a7a", "2025-05-01", "我想学水彩", loc="2025-05-01_agent_chat9", src="lifebench_agent_chat"),
            _msg(A, "a7b", "2025-05-01", "可以先报一个录播课", loc="2025-05-01_agent_chat9", src="lifebench_agent_chat", speaker=f"{A}.assistant"),
            _msg(B, "b1", "2025-07-06", "收到出版社预付款9999元", src="lifebench_sms")]
    for i in range(1, 13):
        msgs.append(_msg(A, f"r{i}", f"2025-08-{i:02d}", f"晨跑记录第{i}次，跑了{i}公里", src="lifebench_fitness_health"))
    api.ingest_messages(msgs)
    api.add_facts(A, [_fact(A, "m4", "2025-07-06", "本人收到出版预付款3000元", "a4"), _fact(A, "m5", "2025-07-08", "本人收到出版预付款8000元", "a5")])
    return api


def _keys(cs):
    return [c.key for c in cs.top]


def test_prefix_no_longer_pulls_ask_day_records(world):
    cs = retrieve(world, A, "（提问时间：2025-12-15）我一共收到过多少出版预付款？", ask("2025-12-15"))
    keys = _keys(cs)
    assert keys[:2] and {k for k in keys[:3]} >= {"2025-07-06_a4", "2025-07-08_a5"}
    assert not any(w.start == date(2025, 12, 15) for w in cs.intent.windows)
    assert "time_window" not in cs.timings_ms                       # no event date in the body: no window channel


def test_no_future_message_or_memory_in_any_channel(world):
    cs = retrieve(world, A, "出版预付款到账", ask("2025-12-31"))
    assert "2026-02-01_a6" not in [c.key for c in cs.candidates]
    for r in cs.channel_results:
        assert "2026-02-01_a6" not in r.keys
    early = retrieve(world, A, "出版预付款", ask("2025-07-07"))      # the second payment is not known yet
    assert "2025-07-08_a5" not in [c.key for c in early.candidates] and "2025-07-06_a4" in [c.key for c in early.candidates]
    assert all("m5" not in c.memories for c in early.candidates)


def test_no_cross_persona_leakage(world):
    cs = retrieve(world, A, "出版社预付款", ask("2025-12-31"))
    assert all(not k.startswith("b") and "b1" not in k for k in [c.key for c in cs.candidates])
    other = retrieve(world, B, "出版社预付款", ask("2025-12-31"))
    assert [c.key for c in other.candidates] == ["2025-07-06_b1"]


def test_offset_question_reaches_the_target_day(world):
    cs = retrieve(world, A, "3月10日我参加了项目启动会议及分工，34天后我在社区卫生服务中心做了什么？", ask("2025-12-31"))
    assert "2025-04-13_a2" in _keys(cs)[:3]
    assert any(w.role == "target" and w.start == date(2025, 4, 13) for w in cs.intent.windows)


def test_top_k_is_the_configured_value_and_expands_for_counting(world):
    cfg = EF1Config(top_k=3, expanded_top_k=7, adaptive_k=True)
    single = retrieve(world, A, "晨跑记录", ask("2025-12-31"), cfg)
    assert single.k == 3 and len(single.top) == 3
    count = retrieve(world, A, "8月我一共晨跑了几次", ask("2025-12-31"), cfg)
    assert count.k in (7, cfg.range_top_k) and len(count.top) >= 7
    assert EF1Config().top_k == 40 and EF1Config().candidate_k >= 40


def test_single_channel_and_fused_orders_are_available_for_fair_comparison(world):
    from stillhere.retrieval.ef1.fusion import fuse
    cs = retrieve(world, A, "出版预付款", ask("2025-12-31"))
    msg = [c.key for c in fuse(cs.channel_results, strategy="msg_bm25")]
    assert msg == next(r for r in cs.channel_results if r.name == "msg_bm25").keys
    rrf = [c.key for c in fuse(cs.channel_results, strategy="rrf", weights={"msg_bm25": 1.0})]
    assert rrf == msg                                                 # one channel through RRF keeps its order


# ---------------------------------------------------------------- fact pack
def test_pack_keeps_every_top_k_unit_with_source_and_time_order(world):
    from stillhere.context import fact_pack as fp
    cs = retrieve(world, A, "8月我一共晨跑了几次，分别跑了多少公里", ask("2025-12-31"), EF1Config(top_k=40))
    pack = fp.build(world, cs)
    runs = [i for i in pack.items if "晨跑记录" in i.text]
    assert len(runs) == 12                                            # all twelve independent events, no 5-item cap
    times = [i.event_time for i in pack.items]
    assert times == sorted(times)                                     # chronological
    assert all(i.source_label and i.unit_key and i.message_ids for i in pack.items if i.kind == "record")
    rendered = fp.render_evidence(pack)
    assert rendered.count("[E") == len(pack.items) and not pack.dropped
    assert pack.tokens["total_input"] > 0 and pack.output_tokens == 512


def test_pack_attaches_memory_status_and_keeps_dialogue_turns(world):
    from stillhere.context import fact_pack as fp
    cs = retrieve(world, A, "出版预付款", ask("2025-12-31"))
    pack = fp.build(world, cs)
    item = next(i for i in pack.items if i.unit_key == "2025-07-06_a4")
    assert item.memories and item.memories[0].status == "当前" and "3000" in item.text
    cs = retrieve(world, A, "学水彩 录播课", ask("2025-12-31"))
    pack = fp.build(world, cs)
    chat = next(i for i in pack.items if i.unit_key == "2025-05-01_agent_chat9")
    assert "我想学水彩" in chat.text and "录播课" in chat.text and len(chat.message_ids) == 2   # every turn of the dialogue


def test_pack_dedup_removes_exact_copies_only():
    from stillhere.context import fact_pack as fp
    api = MemoryAPI(clock=lambda: ask("2026-01-01"))
    api.register_source(SourceRef(source_id=f"{A}.src", kind=SourceKind.SIMULATION, origin=DataOrigin.SYNTHETIC, description="synthetic"))
    api.ingest_messages([_msg(A, "x1", "2025-03-01", "去游泳"), _msg(A, "x2", "2025-03-01", "去游泳"), _msg(A, "x3", "2025-03-02", "去游泳")])
    pack = fp.build(api, retrieve(api, A, "去游泳", ask("2025-12-31")))
    assert len(pack.items) == 2 and len(pack.dedup) == 1 and pack.dedup[0]["reason"] == "same text, same day"


def test_pack_overflow_drops_lowest_ranked_unprotected_items_and_records_them(world):
    from stillhere.context import fact_pack as fp
    cs = retrieve(world, A, "8月晨跑记录", ask("2025-12-31"), EF1Config(top_k=40))
    pack = fp.build(world, cs, context_window=900)
    assert pack.dropped and all(d["reason"].startswith("context_window") for d in pack.dropped)
    assert pack.tokens["fits"] and pack.tokens["total_input"] <= 900 - pack.output_tokens - 64
    assert "没有列出" in fp.render_notes(pack)                          # the reader is told that records were left out
    dropped_keys = {d["unit_key"] for d in pack.dropped}
    assert not (dropped_keys & {i.unit_key for i in pack.items})
    # "8月" makes every run record a range-window (protected) item: when nothing else is left, protected items go too - never an overflow
    assert any(d["reason"] == "context_window_protected" for d in pack.dropped)
    unprot = [d for d in pack.dropped if d["reason"] == "context_window"]
    prot = [d for d in pack.dropped if d["reason"] == "context_window_protected"]
    assert not unprot or pack.dropped.index(unprot[-1]) < pack.dropped.index(prot[0])    # unprotected first


def test_owner_audience_shows_own_records_and_third_party_audience_redacts(world):
    from stillhere.context import fact_pack as fp
    from stillhere.session import disclosure as ds
    cs = retrieve(world, A, "腰肌劳损 就诊", ask("2025-12-31"))
    own = fp.build(world, cs)
    assert any("腰肌劳损" in i.text for i in own.items)
    from stillhere.reader import fact as rf
    other = fp.build(world, cs, audience="third_party", redactor=rf.third_party_redactor(ds.PolicyResolver(world), A))
    assert not any("腰肌劳损" in i.text for i in other.items) and any(i.redacted for i in other.items)
    with pytest.raises(ValueError):
        fp.build(world, cs, audience="third_party")                    # another audience without a disclosure policy is refused


def test_pack_json_round_trip(world):
    from stillhere.context import fact_pack as fp
    pack = fp.build(world, retrieve(world, A, "出版预付款", ask("2025-12-31")))
    again = fp.FactPack.from_json(json.loads(json.dumps(pack.to_json(), ensure_ascii=False)))
    assert fp.render_evidence(again) == fp.render_evidence(pack) and again.source_map() == pack.source_map()


# ---------------------------------------------------------------- reader interface
def test_fact_prompt_states_the_owner_frame_and_carries_every_item(world):
    from stillhere.context import fact_pack as fp
    from stillhere.reader import fact as rf
    pack = fp.build(world, retrieve(world, A, "8月我一共晨跑了几次", ask("2025-12-31")))
    msgs = rf.fact_messages(pack)
    assert "问题中的“我”指用户本人" in msgs[0]["content"]
    assert all(f"[{i.eid}]" in msgs[1]["content"] for i in pack.items)
    assert "（提问时间：2025-12-31）" in msgs[1]["content"]


def test_b_and_c_differ_only_in_the_adapter(world):
    from stillhere.context import fact_pack as fp
    from stillhere.reader import fact as rf
    b, c = rf.READER_CONFIGS["B_base_fact"], rf.READER_CONFIGS["C_d4_fact"]
    assert {k: v for k, v in b.items() if k != "adapter_enabled"} == {k: v for k, v in c.items() if k != "adapter_enabled"}
    assert b["adapter_enabled"] is False and c["adapter_enabled"] is True
    pack = fp.build(world, retrieve(world, A, "出版预付款", ask("2025-12-31")))
    g = rf.generation_for("FACT", pack)
    assert g["max_new_tokens"] == 512 and g["enable_thinking"] is False and g["seed"] == rf.FACT_GENERATION["seed"]
    # Qwen3 non-thinking sampling with a mild repetition penalty (greedy looped on 16K-31K prompts in the Phase 3 smoke); never the
    # legacy D4 chain's 128-token budget
    assert g["do_sample"] is True and g["top_k"] == 20 and g["repetition_penalty"] > 1.0 and g["max_new_tokens"] != 128
    long_pack = fp.build(world, retrieve(world, A, "按时间顺序排一下：\n1. 收到预付款\n2. 就诊", ask("2025-12-31")))
    assert rf.generation_for("FACT", long_pack)["max_new_tokens"] == 1024


def test_d4_prompts_render_all_items_and_the_style_stage(world):
    from stillhere.context import fact_pack as fp
    from stillhere.reader import fact as rf
    pack = fp.build(world, retrieve(world, A, "8月我一共晨跑了几次", ask("2025-12-31")))
    m = rf.d4_persona_messages(pack)
    assert m[0]["content"].count("[F") == len(pack.items) > 5
    s = rf.d4_style_messages("你8月一共晨跑了12次。", pack, [pack.items[0].eid])
    assert "你8月一共晨跑了12次" in s[0]["content"]


def test_parse_citations_support_and_fact_preservation(world):
    from stillhere.context import fact_pack as fp
    from stillhere.reader import fact as rf
    pack = fp.build(world, retrieve(world, A, "出版预付款", ask("2025-12-31")))
    eid = next(i.eid for i in pack.items if "8000" in i.text)
    r = rf.parse('```json\n{"needed":"金额","evidence":["%s","E99"],"quotes":["第二笔出版预付款8000元到账"],"answerable":true,"answer":"一共11000元：3000元和8000元","abstain":false}\n```' % eid)
    assert r.ok and r.evidence[0] == eid
    cit = rf.citation_check(r, pack)
    assert cit["unknown_ids"] == ["E99"] and cit["quotes_not_in_evidence"] == 0
    sup = rf.support_check(r.answer, pack)
    assert sup["unsupported_numbers"] == ["11000"]                   # a sum is not literally in the evidence: reported, not rejected
    fin = rf.finalize(r, pack)
    assert fin["answer"] == r.answer and fin["action"] == "kept"      # "annotate": the guard never turns an answer into a refusal
    bad = rf.parse('{"answer": "是林小满", "abstain": false, "evidence": []}')
    assert rf.finalize(bad, pack, guard_action="abstain_unsupported")["action"] == "kept"      # no number / quote: nothing to flag
    num = rf.parse('{"answer": "一共4200元", "abstain": false, "evidence": []}')
    assert rf.finalize(num, pack, guard_action="abstain_unsupported")["action"] == "abstained_unsupported"
    free = rf.parse("我记不清了")
    assert not free.ok and free.abstain
    assert rf.fact_preserved("3月10日后34天是4月13日", "哦，是4月13日，3月10日开始算34天")["preserved"]
    assert rf.fact_preserved("8000元", "八千块")["missing_numbers"] == ["8000"]


def test_support_scope_is_the_whole_pack_not_the_first_five(world):
    from stillhere.context import fact_pack as fp
    from stillhere.reader import fact as rf
    pack = fp.build(world, retrieve(world, A, "8月晨跑记录 跑了多少公里", ask("2025-12-31")))
    last = pack.items[-1]
    n = re.findall(r"\d+", last.text)[-1]
    assert pack.items.index(last) >= 5 and rf.support_check(f"跑了{n}公里", pack)["supported"]


# ---------------------------------------------------------------- legacy compatibility and isolation
def test_legacy_defaults_are_unchanged():
    import inspect
    from stillhere.memory import reflection_m31 as m31
    from stillhere.memory import retrieval_b2 as b2
    from stillhere.session import d4_evidence as d4e
    from stillhere.session.config import D4_COMPATIBLE
    assert inspect.signature(d4e.build).parameters["max_items"].default == 5
    assert inspect.signature(b2.retrieve).parameters["k"].default == 10 and inspect.signature(m31.reflect).parameters["k"].default == 10
    assert D4_COMPATIBLE.total_input_token_budget == 1536


def test_ef1_modules_never_read_benchmark_fields():
    """Code only (names, attributes, string constants other than docstrings): no benchmark field is ever read by EF1."""
    import ast
    files = list((ROOT / "src" / "stillhere" / "retrieval" / "ef1").glob("*.py")) + [ROOT / "src" / "stillhere" / "context" / "fact_pack.py",
                                                                                      ROOT / "src" / "stillhere" / "reader" / "fact.py"]
    forbidden = ("question_type", "score_points", "gold", "category", "qa_id", "question_id", "reference_answer")
    for f in files:
        tree = ast.parse(f.read_text(encoding="utf-8"))
        docs = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
                and n.body and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant)}
        words = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Name):
                words.append(n.id)
            elif isinstance(n, ast.Attribute):
                words.append(n.attr)
            elif isinstance(n, ast.arg):
                words.append(n.arg)
            elif isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs:
                words.append(n.value)
        for w in words:
            assert not any(x in w.lower() for x in forbidden), (f.name, w[:60])


def test_dense_channel_masks_by_time_and_maps_rows_to_units(tmp_path):
    from stillhere.retrieval.dense import EmbeddingCache, build_index, Document
    from stillhere.retrieval.ef1 import dense_index as di
    api = MemoryAPI(clock=lambda: ask("2026-01-01"))
    api.register_source(SourceRef(source_id=f"{A}.src", kind=SourceKind.SIMULATION, origin=DataOrigin.SYNTHETIC, description="synthetic"))
    api.ingest_messages([_msg(A, "d1", "2025-03-01", "游泳"), _msg(A, "d2", "2025-09-01", "游泳"), _msg(A, "d3", "2025-05-01", "跑步", loc="2025-05-01_call3")])
    docs = {A: di.message_documents(api.store._conn, A, {A: "甲"})}
    cache = EmbeddingCache(tmp_path / "c.sqlite")
    vec = lambda t: np.array([1.0, 0.0] if "游泳" in t else [0.0, 1.0], dtype=np.float32)
    cache.put_many([(EmbeddingCache.key("m", 2, "document", d.text), vec(d.text)) for d in docs[A]])
    build_index(tmp_path / "idx", cache, docs, model="m", dimension=2, dataset_revision="r", snapshot_sha256="s", config={})
    di.write_sidecar(tmp_path / "idx", {A: api})
    cache.put_many([(EmbeddingCache.key("m", 2, "query", "游泳"), np.array([1.0, 0.0], dtype=np.float32))])
    ch, status = di.open_dense(tmp_path / "idx", model="m", dimension=2, caches=[cache])
    assert status == "OK"
    res = ch(A, "游泳", ask("2025-06-01"), 5)
    assert res.keys[0] == "2025-03-01_d1" and "2025-09-01_d2" not in res.keys         # a later message is never visible
    assert "2025-05-01_call3" in res.keys                                              # row -> unit (locator)
    assert ch(A, "无缓存的问题", ask("2025-06-01"), 5).keys == [] and ch.stats["skipped"] == 1   # offline: skipped, not invented
    assert di.open_dense(tmp_path / "missing", model="m", dimension=2)[0] is None
    with pytest.raises(ValueError):
        di.write_sidecar(tmp_path / "idx", {A: MemoryAPI()})                           # a store without those rows is refused


# ---------------------------------------------------------------- time visibility edge cases (H2)
def test_a_fact_written_later_about_an_earlier_time_is_invisible_before_it_was_written(world):
    from stillhere.context import fact_pack as fp
    late = _fact(A, "m_late", "2025-11-01", "本人4月份收到过一笔稿费2000元", "a4", valid_from=ask("2025-04-01"))
    world.add_facts(A, [late])
    before = retrieve(world, A, "稿费", ask("2025-06-01"))
    assert all("m_late" not in c.memories for c in before.candidates)
    assert all(m.memory_id != "m_late" for i in fp.build(world, before).items for m in i.memories)
    after = retrieve(world, A, "稿费", ask("2025-12-01"))
    assert any("m_late" in c.memories for c in after.candidates)


def test_corrected_facts_read_as_they_stood_at_each_ask_time():
    from stillhere.context import fact_pack as fp
    now = {"t": ask("2025-03-01")}
    api = MemoryAPI(clock=lambda: now["t"])
    api.register_source(SourceRef(source_id=f"{A}.src", kind=SourceKind.SIMULATION, origin=DataOrigin.SYNTHETIC, description="synthetic"))
    api.ingest_messages([_msg(A, "h1", "2025-03-01", "我的车牌号是津A12345"), _msg(A, "h2", "2025-06-01", "之前记错了，车牌号是津A54321")])
    api.add_facts(A, [_fact(A, "plate1", "2025-03-01", "本人车牌号津A12345", "h1", attribute="plate", value="津A12345")])
    now["t"] = ask("2025-06-01")
    api.correct(A, "plate1", _fact(A, "plate2", "2025-06-01", "本人车牌号津A54321", "h2", attribute="plate", value="津A54321"), "correction from message h2")
    early = fp.build(api, retrieve(api, A, "车牌号", ask("2025-04-01")))
    st = {m.memory_id: m.status for i in early.items for m in i.memories}
    assert st.get("plate1") == "当前" and "plate2" not in st                       # before the correction: the old value was the belief
    late = fp.build(api, retrieve(api, A, "车牌号", ask("2025-12-31")))
    st = {m.memory_id: m.status for i in late.items for m in i.memories}
    assert st.get("plate2") == "当前" and st.get("plate1", "已更正").startswith("已更正")
    assert any("津A12345" in i.text for i in late.items)                         # the original message stays (it is what was said)


def test_same_day_records_are_visible_and_the_next_day_is_not():
    api = MemoryAPI(clock=lambda: ask("2026-01-01"))
    api.register_source(SourceRef(source_id=f"{A}.src", kind=SourceKind.SIMULATION, origin=DataOrigin.SYNTHETIC, description="synthetic"))
    nxt = _msg(A, "s2", "2025-05-11", "开会讨论预算")
    nxt = CanonicalMessage(**{**nxt.__dict__, "timestamp": ask("2025-05-10") + timedelta(seconds=2), "recorded_at": ask("2025-05-10") + timedelta(seconds=2)})
    api.ingest_messages([_msg(A, "s1", "2025-05-10", "开会讨论预算"), nxt])
    keys = [c.key for c in retrieve(api, A, "开会讨论预算", ask("2025-05-10")).candidates]
    assert keys == ["2025-05-10_s1"]
    tw = api.store.messages_between(A, ask("2025-05-09"), ask("2025-05-12"), as_of=ask("2025-05-10"), known_at=ask("2025-05-10"))
    assert [m.message_id for m in tw] == ["s1"]
    assert [m.message_id for m in api.store.messages_by_locator(A, ["2025-05-11_s2"], as_of=ask("2025-05-10"), known_at=ask("2025-05-10"))] == []


def test_exact_token_count_with_the_pinned_tokenizer_when_available(world):
    import os
    from stillhere.context import fact_pack as fp
    from stillhere.reader import fact as rf
    tok = os.environ.get("STILLHERE_QWEN3_TOKENIZER", "")
    if not tok or not Path(tok).is_file():
        pytest.skip("pinned Qwen3 tokenizer not configured (STILLHERE_QWEN3_TOKENIZER)")
    from stillhere.session.tokens import ExactTokenCounter
    c = ExactTokenCounter(tok)
    pack = fp.build(world, retrieve(world, A, "8月晨跑记录", ask("2025-12-31")), counter=c, system_tokens=c(rf.FACT_SYSTEM))
    assert pack.token_mode == "EXACT" and pack.tokens["system"] == c(rf.FACT_SYSTEM) > 100
    assert abs(c.chat(rf.fact_messages(pack)) - pack.tokens["total_input"]) < 64
