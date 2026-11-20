"""The REAL Mem0 OSS SDK (mem0ai) through the pilot's wrapper, offline: Memory.from_config / Memory.add(infer=True) / Memory.search run for
real (Qdrant local, Mem0's own extraction pipeline and history db); only the two remote services are replaced by an in-process fake
OpenAI-compatible server (chat completions for Mem0's DeepSeek LLM provider, via the pilot's counting proxy; embeddings for Mem0's
OpenAI embedder provider). It shows the integration path and the proxy accounting, not memory quality. Skipped when mem0ai is not
installed (run it with the dedicated environment, e.g. <data-dir>\\envs\\mem0_cpu)."""
import hashlib
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evaluation" / "lifebench_pilot"))

mem0 = pytest.importorskip("mem0")

from lbpilot import llm_proxy as LP  # noqa: E402
from lbpilot import mem0_memory as MM  # noqa: E402


def _vec(t):
    v = [0.0] * 1024
    for i in range(len(t) - 1):
        v[int(hashlib.sha256(t[i:i + 2].encode()).hexdigest()[:6], 16) % 1024] += 1.0
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


class Fake(BaseHTTPRequestHandler):
    chats = []

    def log_message(self, *a):
        pass

    def _send(self, obj):
        raw = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/embeddings"):
            inp = body["input"] if isinstance(body["input"], list) else [body["input"]]
            return self._send({"object": "list", "model": body["model"], "data": [{"object": "embedding", "index": i, "embedding": _vec(t)} for i, t in enumerate(inp)],
                               "usage": {"prompt_tokens": 1, "total_tokens": 1}})
        Fake.chats.append(body)
        user = body["messages"][-1]["content"]
        facts = [{"text": f"林小测{m.strip()}"} for m in re.findall(r"林小测: ([^\n\"]+)", user)][:5]
        return self._send({"id": "x", "object": "chat.completion", "model": body["model"],
                           "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps({"memory": facts}, ensure_ascii=False)}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}})


@pytest.fixture
def servers(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    up = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{up.server_address[1]}"
    px = LP.CountingProxy(expected_model="deepseek-flash", upstream=base, log_path=tmp_path / "proxy.jsonl").start()
    yield base, px
    px.stop()
    up.shutdown()


def test_real_mem0_add_search_through_the_counting_proxy(tmp_path, servers):
    base, px = servers
    emb = {"provider": "openai", "config": {"model": "fake-embedding", "api_key": "fake", "openai_base_url": base + "/v1", "embedding_dims": 1024}}
    m = MM.Mem0Memory(tmp_path / "mem0", user_id="lb2.linxiaoce", proxy_url=px.url, api_key="fake", collection="lb_test", embedder=emb, device="cpu",
                      sleep=lambda s: None)
    audit = m.runtime_audit()
    assert audit["mem0ai_version"] == "2.2.1" and audit["llm_class"] == "DeepSeekLLM" and audit["llm_model"] == "deepseek-flash"
    assert audit["llm_base_url"].startswith(px.url) and audit["vector_store_class"] == "Qdrant" and not audit["openai_api_key_set"]
    assert audit["telemetry_env"] == "False"
    day = [{"event_id": f"2025-03-0{i}_note{i}", "date": f"2025-03-0{i}", "speaker": "林小测", "content": c}
           for i, c in enumerate(["去杭州出差", "在杭州吃了西湖醋鱼", "回到上海开会"], 1)]
    for d in day:
        r = m.add_day([d])
        assert r["batches"] == 1 and r["failed"] == 0
    assert len(Fake.chats) == 3 and all(c["model"] == "deepseek-flash" and c["thinking"] == {"type": "disabled"} for c in Fake.chats)
    assert all(c.get("response_format") == {"type": "json_object"} for c in Fake.chats)            # Mem0's own extraction call
    assert m.count() >= 3
    res = m.search("我在杭州吃了什么？")
    assert res["entries"] and res["entries"][0].source == "mem0"
    top = res["entries"][0]
    assert top.date and top.date.startswith("2025-03-0") and len(top.ids) >= 2                     # memory id + its batch dia_ids
    assert MM.provenance_ids(res["entries"])
    s = px.summary()
    assert s["calls"] == 3 and s["model_violations"] == 0 and s["input_tokens"] == 360 and s["output_tokens"] == 90
    m.close()


def test_runner_snapshot_restore_with_the_real_mem0_store(tmp_path, servers):
    """Real Qdrant / history db files go through the runner's snapshot and come back after the live directory is lost."""
    import shutil
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_lifebench_pilot import FakeEmbedder, FakeEncoder, synthetic   # noqa: E402
    from lbpilot import context as CX
    from lbpilot import runner as R
    from lbpilot import select as SEL
    from lbpilot import stillhere_memory as SM
    from lbpilot import timeline as TL
    base, px = servers
    emb = {"provider": "openai", "config": {"model": "fake-embedding", "api_key": "fake", "openai_base_url": base + "/v1", "embedding_dims": 1024}}
    data = synthetic(days=12, qa_n=130)
    ds = tmp_path / "d.json"
    ds.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    frozen = SEL.freeze_data(data)
    run = R.Run(tmp_path / "drive", tmp_path / "scratch", log=lambda s: None)
    inp = run.load_inputs(ds, frozen, None, verify_dataset=False)
    SM.build_dense(tmp_path / "scratch" / "ix.sqlite", tmp_path / "scratch" / "dense", inp["events"], inp["slug"], inp["name"], inp["assistant"],
                   FakeEmbedder(), dataset_sha256="0" * 64, locator="s")
    dense, _ = SM.open_dense(tmp_path / "scratch" / "dense", FakeEncoder())

    def open_all():
        live = run.live()
        sh = SM.StillHereMemory(live / "stillhere" / "run.sqlite", inp["slug"], inp["name"], inp["assistant"], dataset_sha256="0" * 64, locator="s")
        mm = MM.Mem0Memory(live / "mem0", user_id="lb2.x", proxy_url=px.url, api_key="fake", collection="lb_x", embedder=emb, device="cpu", sleep=lambda s: None)
        return sh, mm
    run.restore_if_needed()
    sh, mm = open_all()
    days = sorted(TL.by_date(inp["events"]))
    full_last = inp["last"]
    run.inputs = {**inp, "last": days[5]}
    run.chronological(stillhere=sh, mem0=mm, dense=dense, count=CX.estimate_tokens)
    n_before = mm.count()
    sh.close()
    mm.close()
    shutil.rmtree(run.live())                                                       # a lost Colab runtime: the live databases are gone
    assert run.restore_if_needed() == days[5]
    run.inputs = {**inp, "last": full_last}
    sh, mm = open_all()
    assert mm.count() == n_before > 0
    st = run.chronological(stillhere=sh, mem0=mm, dense=dense, count=CX.estimate_tokens)
    assert st["status"] == R.PASS and mm.count() > n_before
    rows = run.contexts.done()
    assert len(rows) == 400 and all(r["status"] == "OK" for r in rows.values())
    sh.close()
    mm.close()
