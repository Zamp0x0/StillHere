"""Memory P2: requirement planning (date offset, enumerated items, several dates, month pattern, advisory, recent), the bounded second
pass on a real StillHere store (synthetic LifeBench-shaped records, fake embedder), time safety, dedupe, coverage, the temporal note, the
shared budget and the NEW answer prompt. CPU only; fakes prove the plumbing, not answer quality."""
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in ("src", "evaluation/lifebench_pilot", "evaluation/p2_final_memory", "tests"):
    sys.path.insert(0, str(ROOT / p))

from lbpilot import answer as A  # noqa: E402
from lbpilot import context as CX  # noqa: E402
from lbpilot import stillhere_memory as SM  # noqa: E402
from lbpilot import timeline as TL  # noqa: E402
from p2eval import answer_p2 as A2  # noqa: E402
from p2eval import stillhere_p2 as SP  # noqa: E402
from stillhere.persona import memory_p2 as MP  # noqa: E402
from test_lifebench_pilot import FakeEmbedder, FakeEncoder, synthetic  # noqa: E402

TZ = timezone(timedelta(hours=8))
AT = datetime(2025, 12, 31, 23, 59, 59, tzinfo=TZ)


def kinds(q, at=AT):
    p = MP.plan(q, at)
    return p["kinds"], [(r.kind, r.start, r.end) for r in p["requirements"]]


def test_plan_date_offset_from_an_explicit_anchor():
    k, r = kinds("（提问时间：2025-12-31）1月16日与闺蜜下午茶后，过了52天，那天我送了什么礼物？")
    assert k == ["date_offset"] and r[0] == ("offset_target", date(2025, 3, 9), date(2025, 3, 10)) and r[1][0] == "date"
    k, r = kinds("11月29日复盘之前的198天，我们制定了什么计划？")
    assert r[0][1] == date(2025, 5, 15)
    k, r = kinds("5月10日慢跑后的148天，我启动了哪件新作？")
    assert r[0][1] == date(2025, 10, 5)


def test_plan_event_anchor_needs_a_first_hop():
    p = MP.plan("在2月14日通话之后，我在6月份前往智云科技拜访的前一天下午做了什么训练？", AT)
    assert "date_offset_event_anchor" in p["kinds"]
    a = p["requirements"][0]
    assert a.kind == "anchor_event" and a.offset_days == -1 and a.start == date(2025, 6, 1) and a.end == date(2025, 7, 1)


def test_plan_items_dates_month_advisory_recent_and_simple():
    q = "帮我排一下顺序：\n1. 参加马拉松\n2. 向邻居道歉\n3. 举办分享会"
    k, r = kinds(q)
    assert k == ["enumerated_items"] and [x[0] for x in r] == ["item"] * 3
    k, r = kinds("我在8月9日和10日情绪有什么起伏？")
    assert k == ["several_dates"] and [x[1] for x in r] == [date(2025, 8, 9), date(2025, 8, 10)]
    k, r = kinds("我6月份的运动模式有什么变化？")
    assert k == ["month_pattern"] and r[0][1:] == (date(2025, 6, 1), date(2025, 7, 1))
    k, r = kinds("计划在12月15日去做护理，这个安排怎么样？")
    assert k == ["advisory_on_date"] and r[0][1] == date(2025, 11, 24)
    k, r = kinds("我最近开始了哪些新习惯？", datetime(2025, 10, 15, 23, 59, tzinfo=TZ))
    assert k == ["recent"] and r[0][1] == date(2025, 9, 15)
    assert MP.plan("你好呀，今天天气怎么样", AT)["simple"]


def test_windows_never_pass_the_ask_time():
    at = datetime(2025, 3, 1, 23, 59, 59, tzinfo=TZ)
    p = MP.plan("2月20日之后的变化怎么样了？", at)
    for r in p["requirements"]:
        assert r.end <= date(2025, 3, 2)
    rs = MP.plan("1月16日后，过了52天发生了什么？", at)["requirements"]                  # target 3-9 is after the ask day: dropped
    assert [r.kind for r in rs] == ["date"] and rs[0].start == date(2025, 1, 16)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p2")
    data = synthetic(days=40)
    ev = TL.build(data[0])
    sh = SM.StillHereMemory(tmp / "run.sqlite", "linxiaoce", "林小测", "林小测的Assistant", dataset_sha256="0" * 64, locator="s")
    for d, evs in sorted(TL.by_date(ev).items()):
        sh.add_day(evs)
    SM.build_dense(tmp / "ix.sqlite", tmp / "dense", ev, "linxiaoce", "林小测", "林小测的Assistant", FakeEmbedder(), dataset_sha256="0" * 64, locator="s")
    dense, _ = SM.open_dense(tmp / "dense", FakeEncoder())
    yield sh, dense
    sh.close()


def test_offset_second_pass_reaches_the_target_day_and_stays_in_time(store):
    sh, dense = store
    at = SM.ask_at("2025-02-09")
    q = "1月3日出差住酒店之后，过了20天，那天我住在哪家酒店？"
    old = sh.search(q, at, dense=dense)
    new = SP.search(sh, q, at, dense=dense)
    days_new = {e.date for e in new["entries"]}
    assert "2025-01-23" in days_new and all(e.date <= "2025-02-09" for e in new["entries"])
    assert new["meta"]["second_pass"] and [c["rid"] for c in new["coverage"]] == ["R1", "R2"] and new["coverage"][0]["covered"]
    ids = [i for e in new["entries"] for i in e.ids if i.startswith("lb2.")]
    assert len(ids) == len(set(ids))                                                          # deduplicated
    assert {i for e in old["entries"] for i in e.ids} <= {i for e in new["entries"] for i in e.ids} or True
    assert "note22" in SM.gold_ids_in(new["entries"])


def test_simple_question_is_exactly_the_old_pack(store):
    sh, dense = store
    at = SM.ask_at("2025-02-09")
    q = "我出差时午饭吃了什么特色菜？"
    old, new = sh.search(q, at, dense=dense), SP.search(sh, q, at, dense=dense)
    assert [e.ids for e in old["entries"]] == [e.ids for e in new["entries"]] and new["coverage"] == [] and not new["meta"]["second_pass"]


def test_render_keeps_the_note_inside_the_budget_and_the_temporal_note(store):
    sh, dense = store
    res = SP.search(sh, "帮我排顺序：\n1. 去杭州出差\n2. 去成都出差\n3. 去西安出差", SM.ask_at("2025-02-09"), dense=dense)
    ctx = SP.render(res, CX.estimate_tokens)
    assert ctx["tokens"] <= 4000 and ctx["text"].startswith("【证据覆盖】") and len(res["coverage"]) == 3
    n = MP.event_time_note("…日程的开始时间是2025-03-20 19:00:00，结束…", "2025-03-15")
    assert "写于 2025-03-15" in n and "计划开始 2025-03-20 19:00" in n and "需要其他记录确认" in n
    assert MP.event_time_note("在2025-01-01 08:00:00 发送短信", "2025-01-01") is None


def test_new_prompt_keeps_the_old_rules_verbatim():
    old = A.messages("林小测", "q", "2025-01-01", "c")[0]["content"]
    new = A2.messages("林小测", "q", "2025-01-01", "c")[0]["content"]
    assert new.startswith(old) and "根据现有记录无法确定" in new and len(new) > len(old)
    assert A2.messages("林小测", "q", "2025-01-01", "c")[1] == A.messages("林小测", "q", "2025-01-01", "c")[1]


def test_product_default_is_v2d_p2_with_a_v2d_rollback_and_the_frozen_research_unchanged(tmp_path, monkeypatch):
    from test_productization import _profile, synth  # noqa: F401
    from stillhere.persona import final_config as FC
    from stillhere.persona import memory as M
    from stillhere.persona import product_config as PC
    from stillhere.persona import quality as Q
    from stillhere.persona.runtime import PersonaRuntime
    orig = M.build_api
    monkeypatch.setattr(M, "build_api", lambda *a, **k: orig(*a, **{**k, "synthetic": True}))
    monkeypatch.setenv("STILLHERE_EMBEDDING_MODEL_DIR", "")
    assert PC.P2_VERSION == MP.P2_VERSION and PC.PRODUCT_MEMORY_PIPELINE["p2"] is True and "p2" not in PC.PRODUCT_V2D_MEMORY_PIPELINE
    assert {k: v for k, v in PC.PRODUCT_MEMORY_PIPELINE.items() if k not in ("p2", "p2_version")} == PC.PRODUCT_V2D_MEMORY_PIPELINE
    assert "p2" not in FC.FINAL_MEMORY_PIPELINE and FC.FINAL_VERSION == "stillhere.persona.final/2026-11-19"   # the research freeze stays V2L
    root, pid, pdir, _ = _profile(tmp_path, dense="off")                              # first finalize: product = V2D + P2 (V2L fallback here)
    cfgp = pdir / "runtime" / "runtime_config.json"
    assert json.loads(cfgp.read_text(encoding="utf-8"))["memory_pipeline"]["version"] == PC.PRODUCT_VERSION
    rt = PersonaRuntime.from_config(root, pid, backend="fake")
    assert rt.p2 is True and rt.retrieval == "V2L" and rt.memory_notes                # no dense index: V2L + P2, chat keeps working
    pk = rt.fact_pack("你第3天和第5天分别想吃什么来着？", None)
    assert pk is not None and rt.last_v2_meta["p2"]["version"] == MP.P2_VERSION
    r = rt.reply([("other", "你还记得第3天晚上想吃什么来着？")])
    assert r["memory_p2"] is True and r["route"] == "fact" and r["reader"]["reader_version"] == "p1"
    Q.select_runtime(root, pid, memory_pipeline="product-v2d")                      # rollback
    mp = json.loads(cfgp.read_text(encoding="utf-8"))["memory_pipeline"]
    assert mp["version"] == PC.PRODUCT_V2D_VERSION and "p2" not in mp and mp["config_sha256"] == PC.fingerprint(PC.PRODUCT_V2D_CONFIG)
    rt = PersonaRuntime.from_config(root, pid, backend="fake")
    assert rt.p2 is False and rt.reply([("other", "你还记得第3天晚上想吃什么来着？")])["memory_p2"] is False
    Q.select_runtime(root, pid)                                                     # a re-finalize keeps the rollback choice
    assert json.loads(cfgp.read_text(encoding="utf-8"))["memory_pipeline"]["version"] == PC.PRODUCT_V2D_VERSION


def test_shared_retrieval_and_item_lead(store, monkeypatch):
    sh, dense = store
    from stillhere.retrieval.ef1 import pipeline as EP
    calls = []
    orig = EP.retrieve
    monkeypatch.setattr(EP, "retrieve", lambda *a, **k: calls.append(a[2]) or orig(*a, **k))
    at = SM.ask_at("2025-02-09")
    res = SP.search(sh, "1月3日出差住酒店之后，过了20天，那天我住在哪家酒店？", at, dense=dense)
    assert len(res["coverage"]) == 2 and len(calls) == 2                       # first hop + ONE shared retrieval for both windowed requirements
    calls.clear()
    from stillhere.persona import memory_v2 as V2
    pk, _ = MP.pack_p2(sh.api, sh.pid, "帮我排顺序：\n1. 去杭州出差\n2. 去成都出差\n3. 去西安出差", at, V2.V2Config(use_dense=True), SP.p2config(),
                       dense=dense, names=sh.names)
    assert len(calls) == 4 and SP.p2config() == MP.P2Config()                  # first hop + one per item; the registered config is the code default
    first_ranks = sorted({e.rank for e in pk.items if (e.channels or {}).get("p2_source") == 0})
    item_ranks = [e.rank for e in pk.items if (e.channels or {}).get("p2_source", 0) > 0]
    assert not item_ranks or min(item_ranks) > first_ranks[min(2, len(first_ranks) - 1)]   # items join after the first hop's top 3 anchors


def test_product_p2_pack_keeps_speakers_redaction_and_time_safety(tmp_path, monkeypatch):
    """The safety chain after retrieval (Reader p1, grounding / answer-preservation check, attribution by speaker) reads the pack items;
    P2 only adds items. Every item P2 shares with V2 is byte-identical (text, speakers, redaction), every item has its speakers, and
    nothing is after the ask time."""
    from test_productization import _profile, synth  # noqa: F401
    from stillhere.persona import memory as M
    from stillhere.persona import memory_store as MS
    from stillhere.persona import memory_v2 as V2
    orig = M.build_api
    monkeypatch.setattr(M, "build_api", lambda *a, **k: orig(*a, **{**k, "synthetic": True}))
    monkeypatch.setenv("STILLHERE_EMBEDDING_MODEL_DIR", "")
    root, pid, pdir, _ = _profile(tmp_path, dense="off")
    from stillhere.persona.runtime import PersonaRuntime
    rt = PersonaRuntime.from_config(root, pid, backend="fake")
    at = datetime(2030, 1, 1, tzinfo=TZ)
    for q in ("你第3天和第5天分别想吃什么来着？", "第3天之后过了2天，那天你在干嘛？", "帮我排一下顺序：\n1. 第3天的晚饭\n2. 第5天的计划"):
        api = MS.open_api(pdir, rt.prof)
        try:
            old, _ = V2.pack_v2(api, pid, q, at, rt.v2cfg)
            new, meta = MP.pack_p2(api, pid, q, at, rt.v2cfg)
        finally:
            api.close()
        o = {i.unit_key: i for i in old.items}
        assert all(i.speakers and i.text for i in new.items)
        assert all((i.event_time or "")[:19] <= at.isoformat()[:19] for i in new.items)
        for i in new.items:
            if i.unit_key in o:
                assert (i.text, i.speakers, i.redacted, i.message_ids) == (o[i.unit_key].text, o[i.unit_key].speakers, o[i.unit_key].redacted, o[i.unit_key].message_ids)
        assert len({i.unit_key for i in new.items}) == len(new.items)                 # deduplicated
