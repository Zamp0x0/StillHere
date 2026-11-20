"""LifeBench 100-QA pilot engineering chain on CPU: selection, timeline, time visibility, the shared evidence format / budget, StillHere
V2L / V2D on the canonical events (fake embedder), the counting proxy, answer / judge resume, the official judge prompt and majority,
completeness gates, snapshots / restore, metrics and export. Synthetic LifeBench-shaped data, a FAKE Mem0 stand-in and fake API
transports: this proves the plumbing only, never a benchmark score. The real Mem0 SDK path is tested in test_lifebench_pilot_mem0.py."""
import hashlib
import json
import sys
import threading
import zipfile
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evaluation" / "lifebench_pilot"))

from lbpilot import answer as A  # noqa: E402
from lbpilot import context as CX  # noqa: E402
from lbpilot import export as EX  # noqa: E402
from lbpilot import judge as J  # noqa: E402
from lbpilot import llm_proxy as LP  # noqa: E402
from lbpilot import mem0_memory as MM  # noqa: E402
from lbpilot import metrics as MT  # noqa: E402
from lbpilot import protocol as P  # noqa: E402
from lbpilot import runner as R  # noqa: E402
from lbpilot import select as SEL  # noqa: E402
from lbpilot import stillhere_memory as SM  # noqa: E402
from lbpilot import timeline as TL  # noqa: E402
from stillhere.providers import llm as PLM  # noqa: E402

OFFICIAL = ROOT / "private" / "external" / "LifeBench_eval"
REAL_DATA = ROOT / "private" / "external" / "lifebench_v2.0" / "lifebench_locomo_conversation_format_v2.0_3380QA.json"
CITIES = ["杭州", "成都", "西安", "青岛", "厦门", "南京", "苏州", "长沙"]
TYPES = list(P.OFFICIAL_TYPES)


# ------------------------------------------------------------------ synthetic LifeBench-shaped data (invented person and records)
def synthetic(days=40, qa_n=130):
    name, asst = "林小测", "林小测的Assistant"
    conv = {"speaker_a": name, "speaker_b": asst}
    d0 = datetime(2025, 1, 1)
    for k in range(days):
        d = (d0 + timedelta(days=k)).strftime("%Y-%m-%d")
        city = CITIES[k % len(CITIES)]
        conv[f"session_{k + 1}_date_time"] = d
        conv[f"session_{k + 1}"] = [
            {"speaker": name, "dia_id": f"{d}_note{k}", "text": f"{name}在{d} 09:00:00记录了一个笔记：今天去{city}出差，住在{city}第{k}号酒店。"},
            {"speaker": name, "dia_id": f"{d}_sms{k}", "text": f"在{d} 12:00:00，有一条{name} 发送 王同事的短信：午饭吃了{city}特色菜{k}号。"},
            {"speaker": name, "dia_id": f"{d}_agent_chat{k}", "text": f"我在{city}有点累，推荐个放松方法？"},
            {"speaker": asst, "dia_id": f"{d}_agent_chat{k}", "text": "可以试试散步二十分钟。"},
        ]
    qa = []
    for i in range(qa_n):
        k = i % days
        d = (d0 + timedelta(days=k)).strftime("%Y-%m-%d")
        ask = (d0 + timedelta(days=min(days - 1, k + 1 + i % 3))).strftime("%Y-%m-%d")
        t = TYPES[i % len(TYPES)]
        qa.append({"question": f"（提问时间：{ask}）我{d}那天出差住在哪家酒店？", "answer": f"{CITIES[k % len(CITIES)]}第{k}号酒店。",
                   "evidence": [f"note{k}"] if t != "Unanswerable" else [], "category": [0], "question_type": [t], "question_id": f"lifebench_v5_linxiaoce_qa{i}",
                   "ask_time": ask, "score_points": [{"description": f"指出第{k}号酒店", "score": 10}]})
    return [{"sample_id": name, "conversation": conv, "qa": qa}]


class FakeEmbedder:
    query_task = "fake"

    def _vec(self, t):
        v = np.zeros(1024, dtype=np.float32)
        for i in range(len(t) - 1):
            v[int(hashlib.sha256(t[i:i + 2].encode()).hexdigest()[:6], 16) % 1024] += 1
        n = np.linalg.norm(v)
        return v / n if n else v + 1 / 32

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])


class FakeEncoder:
    query_task = "fake"

    def __call__(self, text):
        return FakeEmbedder()._vec(text)


class FakeMem0:
    """NOT Mem0: a labelled stand-in with the same interface, used only to test the runner's plumbing on CPU."""
    label = "FAKE_MEM0_FOR_TESTS_ONLY"

    def __init__(self):
        self.mem, self.stats = [], {"add_calls": 0}

    def add_day(self, evs):
        for b in MM.batches(evs, 10):
            meta = MM.batch_metadata(b)
            self.mem.extend({"text": e["content"], "date": e["date"], "ids": meta["dia_ids"]} for e in b)
            self.stats["add_calls"] += 1
        return {"batches": len(MM.batches(evs, 10)), "failed": 0, "memories": len(evs)}

    def search(self, q):
        sc = sorted(self.mem, key=lambda m: -len(MT.bigrams(m["text"]) & MT.bigrams(q)))[:40]
        return {"entries": [CX.Entry(rank=i + 1, date=m["date"], source="mem0", text=m["text"], ids=["uuid"] + m["ids"]) for i, m in enumerate(sc)],
                "meta": {"retrieval_ms": 1.0}}


def resp(text, model="deepseek-flash"):
    return PLM.ChatResponse(text=text, finish_reason="stop", requested_model="deepseek-flash", returned_model=model, fingerprint=None, request_id="r",
                            input_tokens=100, cache_hit_tokens=0, output_tokens=10, latency_ms=5.0, attempts=1, cost_usd=0.00004, called_at="t")


def fake_answer_chat(messages):
    u = messages[-1]["content"]
    import re
    m = re.search(r"[一-鿿]{2}第\d+号酒店", u)
    return resp(m.group(0) if m else "根据现有记录无法确定")


def fake_judge_chat(messages):
    u = messages[-1]["content"]
    gen = u.split("待评的生成答案：", 1)[1].split("\n", 1)[0].strip()
    ref = u.split("参考答案：", 1)[1].split("\n", 1)[0].strip()
    ok = gen.rstrip("。") in ref or (("无法" in ref or "没有" in ref) and "无法" in gen)
    return resp(json.dumps({"reasoning": "fake", "point_hits": [{"description": "x", "hit": ok}], "label": "CORRECT" if ok else "WRONG"}, ensure_ascii=False))


# ------------------------------------------------------------------ selection / timeline
@pytest.mark.skipif(not REAL_DATA.is_file(), reason="official dataset not present (private/external, git-ignored)")
def test_frozen_selection_on_the_official_file_is_reproducible():
    f = SEL.freeze(REAL_DATA)
    assert f["person"]["sample_id"] == "于晓薇" and f["person"]["position"] == 0 and len(f["question_ids"]) == 100
    assert f["qa_sha256"] == "62a588685b46658e244027a95815fed34cf1de435967537294b798d529ae1318"
    assert all(v > 0 for v in f["type_counts_multilabel"].values())                       # all 9 official types present
    data = SEL.load_dataset(REAL_DATA)
    person, qs = SEL.selected(data, f)
    ev = TL.build(person, until=f["ask_time_range"][1])
    assert TL.sha(ev) == "f6ed6d428dc4b40d6bee4e3c996fae1c734ec3be750e268a4c9bed28d3dccbc3" and len(ev) == 7097
    seen = {}
    for e in ev:
        seen.setdefault(e["gold_id"], []).append(e["date"])
    for q in qs:                                                                          # every gold record is visible at its ask time
        for g in q["evidence"]:
            assert any(d <= q["ask_time"][:10] for d in seen[g])


def test_person_rule_excludes_earlier_persons_and_needs_all_types():
    data = synthetic()
    excluded = json.loads(json.dumps(data[0]))
    excluded["sample_id"] = "孙雨薇"
    small = json.loads(json.dumps(data[0]))
    small["qa"] = small["qa"][:50]
    small["sample_id"] = "小样本"
    who = SEL.choose_person([excluded, small, data[0]])
    assert who["position"] == 2 and who["considered"][0]["rejected"] and who["considered"][1]["rejected"]
    f = SEL.freeze_data(data)
    assert len(f["question_ids"]) == 100 and f == SEL.freeze_data(synthetic())            # deterministic
    assert sum(f["allocation_by_primary_type"].values()) == 100


def test_timeline_unique_ids_shared_dialog_ids_and_ask_prefix():
    ev = TL.build(synthetic()[0])
    ids = [e["event_id"] for e in ev]
    assert len(ids) == len(set(ids)) and any(i.endswith(":t1") for i in ids)                    # agent_chat turns share a dia_id
    assert all(e["gold_id"] == e["dia_id"].split("_", 1)[1] for e in ev)
    assert TL.search_query("（提问时间：2025-03-01）我去哪了？") == "我去哪了？"
    assert all("answer" not in e and "evidence" not in e for e in ev)                   # no gold on an event
    v = TL.visible(ev, "2025-01-03")
    assert v and max(e["date"] for e in v) == "2025-01-03"


# ------------------------------------------------------------------ shared context
def test_one_context_format_and_a_ranking_prefix_budget():
    es = [CX.Entry(rank=i, date=f"2025-01-{10 - i:02d}", source="短信", text="内容" * 30, ids=[f"x{i}"]) for i in range(1, 8)]
    r = CX.render(es, budget=200)
    assert r["tokens"] <= 200 and r["entries_kept"] < 7 and r["dropped_ranks"] == list(range(r["entries_kept"] + 1, 8))
    lines = [l for l in r["text"].splitlines() if l.startswith("[")]
    assert lines == sorted(lines, key=lambda l: l.split("]")[1])                        # oldest first
    assert CX.render([], budget=200)["text"] == CX.EMPTY


# ------------------------------------------------------------------ StillHere on the canonical timeline
def test_stillhere_v2l_v2d_never_show_a_later_event(tmp_path):
    data = synthetic()
    person = data[0]
    ev = TL.build(person)
    days = TL.by_date(ev)
    sh = SM.StillHereMemory(tmp_path / "run.sqlite", "linxiaoce", "林小测", "林小测的Assistant", dataset_sha256="0" * 64, locator="synthetic")
    idx = SM.build_dense(tmp_path / "index.sqlite", tmp_path / "dense", ev, "linxiaoce", "林小测", "林小测的Assistant", FakeEmbedder(),
                         dataset_sha256="0" * 64, locator="synthetic")
    assert idx["status"] == "COMPLETE"
    dense, why = SM.open_dense(tmp_path / "dense", FakeEncoder())
    assert dense is not None, why
    for d in sorted(days)[:10]:
        sh.add_day(days[d])
    at = SM.ask_at("2025-01-10")
    assert sh.future_messages(at) == 0
    for dn in (None, dense):
        r = sh.search("我2025-01-05那天出差住在哪家酒店？", at, dense=dn)
        assert r["entries"] and all(e.date <= "2025-01-10" for e in r["entries"])
        assert r["meta"]["dense"] is (dn is not None)
        assert "note4" in SM.gold_ids_in(r["entries"])                                    # the gold record of 2025-01-05 is found
    # the dense index holds the whole timeline, but its sidecar masks every later row
    r = sh.search("我2025-01-30那天出差住在哪家酒店？", at, dense=dense)
    assert all(e.date <= "2025-01-10" for e in r["entries"])
    sh.close()


# ------------------------------------------------------------------ counting proxy (Mem0's LLM route)
class _Upstream(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Upstream.seen.append({"body": body, "auth": self.headers.get("Authorization")})
        raw = json.dumps({"model": body["model"], "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 50, "completion_tokens": 5}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def test_counting_proxy_enforces_the_model_disables_thinking_and_counts(tmp_path):
    up = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    px = LP.CountingProxy(expected_model="deepseek-flash", upstream=f"http://127.0.0.1:{up.server_address[1]}", log_path=tmp_path / "p.jsonl").start()
    try:
        c = PLM.ChatClient(PLM.ProviderConfig(provider="openai_compatible_custom", base_url=px.url, model="deepseek-flash", api_key_env="K"), env={"K": "k1"})
        assert c.chat([{"role": "user", "content": "x"}]).input_tokens == 50
        assert _Upstream.seen[-1]["body"]["thinking"] == {"type": "disabled"} and _Upstream.seen[-1]["auth"] == "Bearer k1"
        bad = PLM.ChatClient(PLM.ProviderConfig(provider="openai_compatible_custom", base_url=px.url, model="gpt-4o-mini", api_key_env="K"), env={"K": "k1"})
        with pytest.raises(PLM.ProviderError, match="HTTP 400"):
            bad.chat([{"role": "user", "content": "x"}])
        s = px.summary()
        assert s["calls"] == 1 and s["model_violations"] == 1 and s["input_tokens"] == 50 and len(_Upstream.seen) == 1
        assert "k1" not in (tmp_path / "p.jsonl").read_text(encoding="utf-8")
    finally:
        px.stop()
        up.shutdown()


# ------------------------------------------------------------------ the whole chain, resumed
@pytest.mark.skipif(not (OFFICIAL / "src" / "evaluators" / "llm_judge.py").is_file(), reason="official LifeBench_eval checkout not present")
def test_end_to_end_chain_with_resume_completeness_metrics_and_export(tmp_path):
    data = synthetic()
    ds = tmp_path / "data.json"
    ds.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    frozen = SEL.freeze_data(data)
    run = R.Run(tmp_path / "drive_run", tmp_path / "scratch", log=lambda s: None)
    inp = run.load_inputs(ds, frozen, None, verify_dataset=False)
    live = run.live()

    def systems():
        sh = SM.StillHereMemory(live / "stillhere" / "run.sqlite", inp["slug"], inp["name"], inp["assistant"], dataset_sha256="0" * 64, locator="s")
        return sh
    SM.build_dense(tmp_path / "scratch" / "index.sqlite", tmp_path / "scratch" / "dense", inp["events"], inp["slug"], inp["name"], inp["assistant"],
                   FakeEmbedder(), dataset_sha256="0" * 64, locator="s")
    dense, _ = SM.open_dense(tmp_path / "scratch" / "dense", FakeEncoder())
    run.restore_if_needed()
    sh, mem = systems(), FakeMem0()
    # a crash after a few dates: only part of the timeline is added
    days = sorted(TL.by_date(inp["events"]))
    partial = dict(inp)
    run.inputs = {**inp, "last": days[12]}
    run.chronological(stillhere=sh, mem0=mem, dense=dense, count=CX.estimate_tokens)
    sh.close()
    n_ctx = len(run.contexts.done())
    assert run.state["added_through"] == days[12] and run.state["snapshots"]
    # a fresh runtime: the live dir is gone, restore from the newest snapshot and continue
    import shutil
    shutil.rmtree(live)
    run2 = R.Run(tmp_path / "drive_run", tmp_path / "scratch", log=lambda s: None)
    run2.load_inputs(ds, frozen, None, verify_dataset=False)
    assert run2.restore_if_needed() == days[12]
    sh2 = systems()
    assert sh2.latest_time() and sh2.latest_time()[:10] == days[12]
    mem2 = FakeMem0()
    for d in days[:13]:
        mem2.add_day(TL.by_date(inp["events"])[d])                                    # (a real Mem0 is restored from the same snapshot)
    st = run2.chronological(stillhere=sh2, mem0=mem2, dense=dense, count=CX.estimate_tokens)
    assert st["status"] == R.PASS and len(run2.contexts.done()) == 400 and len(run2.contexts.done()) >= n_ctx
    rows = list(run2.contexts.done().values())
    assert all(r["future_messages_in_store"] == 0 for r in rows if r["group"].startswith(("M1", "M2")))
    assert all(r["tokens"] <= P.EVIDENCE["token_budget"] for r in rows)
    m0 = [r for r in rows if r["group"] == "M0_NO_MEMORY"]
    assert all(r["text"] == CX.EMPTY for r in m0)
    # answers: one shared function; a failure is retried; nothing finished is requested twice
    calls = {"n": 0}

    def flaky(messages):
        calls["n"] += 1
        if calls["n"] == 3:
            raise PLM.ProviderError("HTTP_ERROR", "HTTP 503")
        return fake_answer_chat(messages)
    a = run2.run_answers(flaky)
    assert a["status"] == R.PASS and calls["n"] == 401
    before = calls["n"]
    run2.run_answers(flaky)
    assert calls["n"] == before                                                         # resumed: zero new requests
    systems_prompts = {json.dumps(A.messages(inp["name"], "q", "2025-01-01", "c")[0], ensure_ascii=False)}
    assert len(systems_prompts) == 1
    official = J.Official(OFFICIAL)
    assert official.prompt_sha256 == J.OFFICIAL_PROMPT_SHA256
    jc = {"n": 0}

    def judge_chat(messages):
        jc["n"] += 1
        if jc["n"] % 97 == 0:
            return resp("这不是 JSON")                                                 # unparseable: retried within the job
        return fake_judge_chat(messages)
    j = run2.run_judge(official, judge_chat)
    assert j["status"] == R.PASS and all(v == 300 for v in j["per_group"].values())
    jn = jc["n"]
    run2.run_judge(official, judge_chat)
    assert jc["n"] == jn
    cal = run2.calibrate(official, fake_judge_chat)
    assert cal["of"] == 4
    met = run2.compute_metrics(official)
    m = json.loads((tmp_path / "drive_run" / "METRICS.json").read_text(encoding="utf-8"))
    assert m["formal"] and set(m["groups"]) == set(P.GROUP_ORDER)
    g1 = m["groups"]["M1_STILLHERE_V2L"]
    assert g1["micro_accuracy"] > m["groups"]["M0_NO_MEMORY"]["micro_accuracy"]          # the fake chain is wired: memory helps
    assert g1["macro_types_present"] == 9 and g1["evidence"]["strict_id_recall_mean"] is not None
    assert m["groups"]["M3_MEM0_OSS"]["evidence"]["batch_provenance_recall_mean"] is not None
    assert m["pairs"]["M2_STILLHERE_V2D__vs__M3_MEM0_OSS"]["decidable"]
    p = m["pairs"]["M1_STILLHERE_V2L__vs__M0_NO_MEMORY"]
    assert p["wins"] + p["ties"] + p["losses"] == 100 and p["ci95"][0] <= p["diff"] <= p["ci95"][1]
    # recomputation from the records alone gives the same numbers
    again = R.compute(inp["qs"], inp["events"], run2.contexts.done(), run2.answers.done(), run2.judgments.done(), official)
    assert again["groups"] == m["groups"] and again["pairs"] == m["pairs"]
    z = EX.build(tmp_path / "drive_run", tmp_path / "results.zip")
    names = zipfile.ZipFile(z["zip"]).namelist()
    assert "METRICS.json" in names and "SHA256SUMS.txt" in names and not any(n.startswith("snapshots/") for n in names)
    sh2.close()


def test_an_incomplete_group_is_never_decidable(tmp_path):
    rows = [{"question_id": f"q{i}", "types": ["Single_hop"], "unanswerable": False, "is_correct": i % 2 == 0, "judged": True, "answered": True,
             "abstained": False, "weighted_score": None, "unanimous": True, "evidence_tokens": 10, "entries_kept": 1, "entries_dropped": 0,
             "retrieval_ms": 1, "answer_latency_ms": 1} for i in range(10)]
    g = MT.group_metrics(rows)
    assert g["micro_accuracy"] == 0.5 and g["macro_types_na"] == [t for t in P.OFFICIAL_TYPES if t != "Single_hop"]
    p = MT.paired(rows, rows)
    assert p["diff"] == 0 and not p["significant"] and p["ties"] == 10


def test_official_majority_and_weighted_score():
    if not (OFFICIAL / "src" / "evaluators" / "llm_judge.py").is_file():
        pytest.skip("official checkout not present")
    o = J.Official(OFFICIAL)
    qa = {"score_points": [{"description": "a", "score": 3}, {"description": "b", "score": 7}]}
    recs = [{"repeat": 0, "is_correct": True, "point_hits": [{"hit": True}, {"hit": False}]},
            {"repeat": 1, "is_correct": False, "point_hits": [{"hit": True}, {"hit": True}]},
            {"repeat": 2, "is_correct": True, "point_hits": [{"hit": False}, {"hit": True}]}]
    v = J.aggregate(recs, qa, o)
    assert v["is_correct"] and v["correct_votes"] == 2 and [h["hit"] for h in v["point_hits"]] == [True, True] and v["weighted_score"] == 1.0
    assert not v["unanimous"]


def test_mem0_rendering_matches_the_official_adapter_format():
    e = {"date": "2025-03-05", "speaker": "林小测", "content": " 今天跑步5公里 ", "event_id": "2025-03-05_note1"}
    assert MM.render(e) == "(Conversation date: March 05, 2025) 林小测: 今天跑步5公里"
    m = MM.batch_metadata([e])
    assert m["session_date"] == "March 05, 2025" and m["session_iso"].startswith("2025-03-05") and m["dia_ids"] == ["2025-03-05_note1"]
    cfg = MM.config(Path("w"), proxy_url="http://127.0.0.1:1", api_key="k", collection="c")
    assert cfg["llm"]["provider"] == "deepseek" and cfg["llm"]["config"]["model"] == "deepseek-flash"
    assert cfg["embedder"]["provider"] == "huggingface" and cfg["embedder"]["config"]["model"] == "Qwen/Qwen3-Embedding-0.6B"
    assert cfg["vector_store"]["provider"] == "qdrant" and cfg["vector_store"]["config"]["embedding_model_dims"] == 1024
    assert MM.public_config(cfg)["llm"]["config"]["api_key"] == "***"


def test_a_stop_in_the_middle_of_a_day_restores_the_last_snapshot(tmp_path):
    data = synthetic(days=10)
    ds = tmp_path / "d.json"
    ds.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    run = R.Run(tmp_path / "drive", tmp_path / "scratch", log=lambda s: None)
    inp = run.load_inputs(ds, SEL.freeze_data(data), None, verify_dataset=False)
    run.restore_if_needed()
    sh = SM.StillHereMemory(run.live() / "stillhere" / "run.sqlite", inp["slug"], inp["name"], inp["assistant"], dataset_sha256="0" * 64, locator="s")
    days = sorted(TL.by_date(inp["events"]))
    run.inputs = {**inp, "last": days[3]}
    run.chronological(stillhere=sh, mem0=FakeMem0(), dense=None, count=CX.estimate_tokens)
    sh.close()
    snap = max(s["date"] for s in run.state["snapshots"])
    (run.live() / "DAY_IN_PROGRESS").write_text(days[4], encoding="utf-8")        # stopped while adding the next day
    assert run.restore_if_needed() == snap and not (run.live() / "DAY_IN_PROGRESS").exists()


def test_proxy_answers_504_before_the_caller_times_out_and_survives_a_closed_client(tmp_path):
    import socket
    import time as _t

    class Slow(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            _t.sleep(1.5)
            raw = json.dumps({"model": "deepseek-flash", "choices": [{"message": {"content": "{}"}}], "usage": {"prompt_tokens": 9, "completion_tokens": 1}}).encode()
            try:
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            except OSError:
                pass
    up = ThreadingHTTPServer(("127.0.0.1", 0), Slow)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{up.server_address[1]}"
    px = LP.CountingProxy(expected_model="deepseek-flash", upstream=base, log_path=tmp_path / "p.jsonl", deadline_s=0.5).start()
    try:
        c = PLM.ChatClient(PLM.ProviderConfig(provider="openai_compatible_custom", base_url=px.url, model="deepseek-flash", api_key_env="K", max_attempts=1),
                           env={"K": "k"})
        with pytest.raises(PLM.ProviderError, match="504"):
            c.chat([{"role": "user", "content": "x"}])
        assert px.summary()["deadline_504"] == 1
        px.deadline = 10.0                                                  # a caller that hangs up before the (slow) answer
        body = json.dumps({"model": "deepseek-flash", "messages": [{"role": "user", "content": "x"}]}).encode()
        s = socket.create_connection(("127.0.0.1", px.port))
        s.sendall(b"POST /chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        _t.sleep(0.3)                                                       # the proxy has read the request and waits upstream
        s.close()
        _t.sleep(2.5)
        assert px.summary()["calls"] == 2                                     # counted (it may have been billed), no traceback, proxy alive
        px.deadline = 10.0
        assert PLM.ChatClient(PLM.ProviderConfig(provider="openai_compatible_custom", base_url=px.url, model="deepseek-flash", api_key_env="K"),
                              env={"K": "k"}).chat([{"role": "user", "content": "x"}]).input_tokens == 9
    finally:
        px.stop()
        up.shutdown()
