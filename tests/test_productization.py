"""Productization (2026-11-19): per-role providers (DeepSeek / custom OpenAI-compatible / local), the API generator under the unchanged
reader chain, the versioned V2D product memory with its V2L fallback, finalize defaults, the quickstart flow, card management, GPU
selection, the remote PACKAGE_ONLY target and the optional LLM router. Synthetic chats and fake transports / embedders only: no network,
no model, no GPU."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import persona_fixture as F  # noqa: E402
from stillhere.persona import card as K  # noqa: E402
from stillhere.persona import card_admin as CA  # noqa: E402
from stillhere.persona import dense_memory as DM  # noqa: E402
from stillhere.persona import llm_router as LR  # noqa: E402
from stillhere.persona import memory as M  # noqa: E402
from stillhere.persona import memory_store as MS  # noqa: E402
from stillhere.persona import package as P  # noqa: E402
from stillhere.persona import pipeline as PL  # noqa: E402
from stillhere.persona import product_config as PC  # noqa: E402
from stillhere.persona import quality as Q  # noqa: E402
from stillhere.persona import store as S  # noqa: E402
from stillhere.persona import training as TR  # noqa: E402
from stillhere.persona.runtime import PersonaRuntime  # noqa: E402
from stillhere.providers import llm as PLM  # noqa: E402
from stillhere.providers import roles as PR  # noqa: E402

KEY = "fake-key-for-tests-only"


# ------------------------------------------------------------------ fakes
class Transport:
    """Fake OpenAI-compatible server: a queue of (status, body) answers; records every request."""

    def __init__(self, answers):
        self.answers, self.requests = list(answers), []

    def __call__(self, url, headers, body, timeout):
        self.requests.append({"url": url, "headers": dict(headers), "body": json.loads(body), "timeout": timeout})
        st, obj = self.answers.pop(0)
        if isinstance(obj, Exception):
            raise obj
        return st, json.dumps(obj).encode()


def ok(text, model="deepseek-flash", finish="stop"):
    return 200, {"id": "req-1", "model": model, "system_fingerprint": "fp_x", "choices": [{"message": {"content": text}, "finish_reason": finish}],
                 "usage": {"prompt_tokens": 100, "completion_tokens": 20, "prompt_cache_hit_tokens": 40}}


class FakeEmbedder:
    """Deterministic unit vectors from character bigrams: similar texts get similar vectors (enough to exercise the dense channel)."""
    query_task = "fake"

    def _vec(self, t):
        v = np.zeros(1024, dtype=np.float32)
        for i in range(len(t) - 1):
            v[int(hashlib.sha256(t[i:i + 2].encode()).hexdigest()[:6], 16) % 1024] += 1
        n = np.linalg.norm(v)
        return v / n if n else v + 1 / 32

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_queries(self, texts):
        return self.embed_documents(texts)


class FakeEncoder:
    query_task = "fake"

    def __call__(self, text):
        return FakeEmbedder()._vec(text)


@pytest.fixture
def synth(monkeypatch):
    orig = M.build_api
    monkeypatch.setattr(M, "build_api", lambda *a, **k: orig(*a, **{**k, "synthetic": True}))
    monkeypatch.delenv("STILLHERE_ALLOW_EXTERNAL_API", raising=False)
    monkeypatch.setenv("STILLHERE_EMBEDDING_MODEL_DIR", "")


def _profile(tmp_path, **kw):
    chat = F.standard(days=60).write(tmp_path / "c.json")
    root = tmp_path / "r"
    res = PL.run(chat, speaker="contact", root=root, target="cpu-only", **kw)
    return root, res["profile_id"], S.profile_dir(root, res["profile_id"]), res


# ------------------------------------------------------------------ providers
def test_deepseek_client_payload_retry_usage_and_no_secret_in_the_log(tmp_path):
    t = Transport([(429, {"error": {"code": "rate_limit"}}), (503, {}), ok("好的")])
    log = PLM.UsageLog(tmp_path / "u.jsonl")
    c = PLM.ChatClient(PLM.ProviderConfig(), env={"DEEPSEEK_API_KEY": KEY}, transport=t, sleep=lambda s: None, usage_log=log)
    r = c.chat([{"role": "user", "content": "私人内容不进日志"}])
    body = t.requests[-1]["body"]
    assert body["model"] == "deepseek-flash" and body["thinking"] == {"type": "disabled"} and body["temperature"] == 0.0 and body["stream"] is False
    assert t.requests[0]["url"] == "https://api.deepseek.com/chat/completions" and r.attempts == 3
    assert r.returned_model == "deepseek-flash" and r.input_tokens == 100 and r.cache_hit_tokens == 40
    assert r.cost_usd == pytest.approx(60 / 1e6 * 0.30 + 40 / 1e6 * 0.006 + 20 / 1e6 * 1.20)
    text = (tmp_path / "u.jsonl").read_text(encoding="utf-8")
    assert KEY not in text and "私人内容" not in text and text.count("\n") == 3


def test_client_never_retries_a_4xx_and_gives_up_after_the_bound():
    t = Transport([(401, {"error": {"code": "invalid_api_key"}})])
    c = PLM.ChatClient(PLM.ProviderConfig(), env={"DEEPSEEK_API_KEY": KEY}, transport=t, sleep=lambda s: None)
    with pytest.raises(PLM.ProviderError, match="HTTP 401"):
        c.chat([{"role": "user", "content": "x"}])
    assert len(t.requests) == 1
    t = Transport([(500, {})] * 3)
    c = PLM.ChatClient(PLM.ProviderConfig(max_attempts=3), env={"DEEPSEEK_API_KEY": KEY}, transport=t, sleep=lambda s: None)
    with pytest.raises(PLM.ProviderError, match="GAVE_UP"):
        c.chat([{"role": "user", "content": "x"}])
    assert len(t.requests) == 3
    with pytest.raises(PLM.ProviderError, match="NO_API_KEY"):
        PLM.ChatClient(PLM.ProviderConfig(), env={})
    with pytest.raises(PLM.ProviderError, match="BAD_LIMITS"):
        PLM.ProviderConfig(max_attempts=50).validate()


def test_roles_are_configured_separately_and_api_use_needs_the_opt_in(tmp_path):
    env = {"STILLHERE_ANSWER_PROVIDER": "openai_compatible_custom", "STILLHERE_ANSWER_BASE_URL": "http://127.0.0.1:9/v1", "STILLHERE_ANSWER_MODEL": "m-x",
           "STILLHERE_JUDGE_MODEL": "deepseek-flash"}
    a = PR.resolve("answer", env=env, dotenv=tmp_path / "none.env")
    j = PR.resolve("judge", env=env, dotenv=tmp_path / "none.env")
    w = PR.resolve("memory_extraction", env=env, dotenv=tmp_path / "none.env")
    assert (a.provider, a.model, a.url) == ("openai_compatible_custom", "m-x", "http://127.0.0.1:9/v1/chat/completions")
    assert (j.provider, j.model) == ("deepseek_api", "deepseek-flash") and w.provider == "huggingface_local"
    assert PR.resolve("answer", env=env, dotenv=tmp_path / "none.env", overrides={"model": "m-y"}).model == "m-y"
    with pytest.raises(PLM.ProviderError, match="EXTERNAL_API_NOT_ALLOWED") as e:
        PR.api_generator("answer", a, env={"STILLHERE_CUSTOM_API_KEY": KEY})
    assert "retrieved chat evidence" in str(e.value)
    g = PR.api_generator("answer", a, env={"STILLHERE_CUSTOM_API_KEY": KEY, "STILLHERE_ALLOW_EXTERNAL_API": "1"}, transport=Transport([ok("嗯", model="m-x")]))
    assert g.label == "API:openai_compatible_custom:m-x" and "thinking" not in g.client.payload([{"role": "user", "content": "x"}])


def test_api_generator_contract_and_adapter_refusal():
    t = Transport([ok("就那样吧", finish="length"), ok('前言 {"answer": "广州"} 后记')])
    g = PLM.ApiGenerator(PLM.ChatClient(PLM.ProviderConfig(), env={"DEEPSEEK_API_KEY": KEY}, transport=t))
    r = g([{"role": "user", "content": "x"}], {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "max_new_tokens": 96}, None)
    assert r["text"] == "就那样吧" and r["hit_token_limit"] and r["prompt_tokens"] == 100 and r["new_tokens"] == 20
    assert t.requests[0]["body"]["temperature"] == 0.7 and t.requests[0]["body"]["top_p"] == 0.8 and t.requests[0]["body"]["max_tokens"] == 96
    assert g.complete([{"role": "user", "content": "x"}])["text"] == '{"answer": "广州"}' and t.requests[1]["body"]["temperature"] == 0.0
    with pytest.raises(PLM.ProviderError, match="ADAPTER_NEEDS_LOCAL_BASE"):
        g([{"role": "user", "content": "x"}], {"do_sample": False}, "a-1")


def test_runtime_on_an_api_answer_provider_keeps_the_reader_chain(tmp_path, synth, monkeypatch):
    root, pid, pdir, _ = _profile(tmp_path, dense="off")
    t = Transport([ok("{}")] * 6)
    made = {}

    def fake_api(role, cfg, **kw):
        made.update(role=role, allow=kw.get("allow_external"))
        return PLM.ApiGenerator(PLM.ChatClient(cfg, env={"DEEPSEEK_API_KEY": KEY}, transport=t))
    monkeypatch.setattr(PR, "api_generator", fake_api)
    rt = PersonaRuntime.from_config(root, pid, answer={"provider": "deepseek_api"}, allow_external_api=True)
    assert rt.backend == "api" and rt.gen.label == "API:deepseek_api:deepseek-flash" and made == {"role": "answer", "allow": True}
    r = rt.reply([("other", "你还记得第3天晚上想吃什么来着？")])
    assert r["route"] == "fact" and r["reader"] is not None and r["reader"]["reader_version"] == "p1"
    assert r["reader"]["fail"] in ("READER_ABSTAINED", "GROUNDEDNESS_FAIL", "SUBJECT_RESOLUTION_FAIL", None)
    assert len(t.requests) == 2                                                    # Fact Reader + persona, both through the API
    assert all("thinking" in q["body"] for q in t.requests)
    with pytest.raises(Exception, match="ADAPTER_NEEDS_LOCAL_BASE"):           # an adapter never runs on an API model
        cfgp = pdir / "runtime" / "runtime_config.json"
        cfg = json.loads(cfgp.read_text(encoding="utf-8"))
        monkeypatch.setattr("stillhere.persona.adapters.resolve", lambda pdir, a: {"manifest": {"card_id": K.current(pdir)["card_id"]}, "dir": "x"})
        PersonaRuntime(root, pid, adapters=["a-x"], answer={"provider": "deepseek_api"}, allow_external_api=True, card_id=cfg["persona_prompt"]["card_id"])


# ------------------------------------------------------------------ product memory configuration
def test_first_finalize_is_the_product_v2d_and_a_re_finalize_keeps_the_choice(tmp_path, synth):
    root, pid, pdir, res = _profile(tmp_path, dense="off")
    cfgp = pdir / "runtime" / "runtime_config.json"
    mp = json.loads(cfgp.read_text(encoding="utf-8"))["memory_pipeline"]
    assert mp["version"] == PC.PRODUCT_VERSION and mp["dense"] is True and mp["dense_fallback"] == "v2l" and mp["reader_version"] == "p1"
    assert mp["writer_mode"] == "EVIDENCE_ONLY" and mp["config_sha256"] == PC.fingerprint()
    cfg = json.loads(cfgp.read_text(encoding="utf-8"))
    assert cfg["adapter"]["enabled"] is False
    Q.select_runtime(root, pid)                                                     # re-finalize: kept
    assert json.loads(cfgp.read_text(encoding="utf-8"))["memory_pipeline"]["version"] == PC.PRODUCT_VERSION
    Q.select_runtime(root, pid, override_memory=True)                               # a CLI override of something else: still kept
    assert json.loads(cfgp.read_text(encoding="utf-8"))["memory_pipeline"]["dense"] is True
    Q.select_runtime(root, pid, memory_pipeline="research-final")
    assert json.loads(cfgp.read_text(encoding="utf-8"))["memory_pipeline"]["dense"] is False
    Q.select_runtime(root, pid)
    assert json.loads(cfgp.read_text(encoding="utf-8"))["memory_pipeline"]["version"] == "stillhere.persona.final/2026-11-19"
    Q.select_runtime(root, pid, memory_pipeline="v1")
    assert "memory_pipeline" not in json.loads(cfgp.read_text(encoding="utf-8"))
    Q.select_runtime(root, pid)                                                     # an existing V1 config is NOT upgraded silently
    assert "memory_pipeline" not in json.loads(cfgp.read_text(encoding="utf-8"))


def test_product_memory_without_a_dense_index_falls_back_to_v2l(tmp_path, synth):
    root, pid, pdir, res = _profile(tmp_path, dense="auto")                        # no embedding model here: dense SKIPPED
    assert res["dense"]["status"] == "SKIPPED" and res["overall"] in ("READY", "PARTIAL_READY")
    rt = PersonaRuntime.from_config(root, pid, backend="fake")
    assert rt.retrieval == "V2L" and rt.v2cfg.use_dense is False and rt.memory_notes and rt.active is None
    r = rt.reply([("other", "你还记得第3天晚上想吃什么来着？")])
    assert r["retrieval"] == "V2L" and r["memory_notes"]


def test_quickstart_flow_builds_the_dense_index_and_runs_v2d(tmp_path, synth):
    root, pid, pdir, res = _profile(tmp_path, dense="on", embedder=FakeEmbedder())
    assert res["dense"]["status"] == "PASS" and res["dense"]["dense_status"] == "READY"
    rt = PersonaRuntime.from_config(root, pid, backend="fake", dense_encoder=FakeEncoder())
    assert rt.retrieval == "V2D" and rt.v2cfg.use_dense and rt.memory_notes == []
    pk = rt.fact_pack("本人第3天晚上想吃什么？", None)
    assert pk.items and rt.last_v2_meta["dense"] is True
    again = DM.build(root, pid, embedder=FakeEmbedder())                           # idempotent
    assert again["reused"] and again["dense_status"] == "READY"
    rep = S.read_json(DM.root_dir(pdir) / "BUILD_REPORT.json")
    assert rep["last"]["model_revision"] == PC.EMBEDDING["revision"]


def test_dense_index_goes_stale_after_a_new_import_and_is_rebuilt_without_deleting(tmp_path, synth):
    root, pid, pdir, _ = _profile(tmp_path, dense="on", embedder=FakeEmbedder())
    newer = F.standard(days=70).write(tmp_path / "newer.json")
    prof = S.read_json(pdir / "profile.json")
    MS.build_evidence(root, pid, source=newer)
    assert DM.status(root, pid)["dense_status"] == "STALE"
    ch, why = DM.open_channel(root, pid, encoder=FakeEncoder())
    assert ch is not None and why.startswith("OK_STALE")
    DM.build(root, pid, embedder=FakeEmbedder())
    assert DM.status(root, pid)["dense_status"] == "READY" and list(DM.root_dir(pdir).glob("index.superseded-*"))
    assert prof == S.read_json(pdir / "profile.json")


# ------------------------------------------------------------------ cards
def test_card_sources_activation_rollback_and_llm_assist(tmp_path, synth):
    root, pid, pdir, _ = _profile(tmp_path, dense="off")
    first = K.current(pdir)["card_id"]
    assert [v["source"] for v in CA.list_versions(pdir)] == ["AUTO"]
    K.review(pdir, approve=True)
    manual = K.current(pdir)["card_id"]

    class Gen:
        def __call__(self, messages, cfg, adapter):
            assert "examples" not in messages[1]["content"] or json.loads(messages[1]["content"]).get("examples") is None
            rules = ["这是一个模仿此人说话方式的 AI，不是本人。", "回复简短，常常分成几条短消息发出，先回应对方最新的一句话。",
                     "语气随意自然，可以用常用的口头表达，但不要机械地重复同一句口头禅。", "对方情绪低落时先接住情绪，再说别的事情。",
                     "事实只能来自提供的记忆记录；没有依据的时候，自然地说不知道或者记不清，绝不编造经历、人物、地点和时间。",
                     "不要使用书面化的长段落，不要像客服一样说话，也不要主动讲大道理。", "对方问到不确定的事情时，可以反问一句确认。"]
            return {"text": chr(10).join(rules), "returned_model": "fake-m"}
    r = CA.assist(root, pid, generator=Gen())
    assert r["source"] == "LLM_ASSISTED" and r["activated"] is False and K.current(pdir)["card_id"] == manual
    srcs = {v["card_id"]: v["source"] for v in CA.list_versions(pdir)}
    assert srcs[first] == "AUTO" and srcs[manual] == "MANUAL" and srcs[r["card_id"]] == "LLM_ASSISTED"
    a = CA.activate(pdir, r["card_id"])
    assert a["previous"] == manual and K.current(pdir)["card_id"] == r["card_id"] and a["next"]       # runtime_config must be refreshed
    CA.activate(pdir, first)                                                        # rollback: nothing deleted
    assert K.current(pdir)["card_id"] == first and len(CA.list_versions(pdir)) == 3
    assert sum(v["current"] for v in CA.list_versions(pdir)) == 1
    assert CA.preview(pdir)["prompt"] == K.current(pdir)["prompt"]


# ------------------------------------------------------------------ training
def test_gpu_selection_maps_to_cuda_visible_devices(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert TR.select_gpu(None) == {}
    m = TR.select_gpu(1)
    import os
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1" and m["in_process_device"] == "cuda:0" and TR.env()["CUDA_VISIBLE_DEVICES"] == "1"
    with pytest.raises(TR.TrainingRefused):
        TR.select_gpu(-1)


def test_remote_gpu_target_is_package_only_and_needs_the_cloud_confirmation(tmp_path, synth):
    root, pid, pdir, _ = _profile(tmp_path, dense="off")
    with pytest.raises(P.PackageError, match="CLOUD_NOT_CONFIRMED"):
        P.build(root, pid, target="remote-gpu")
    r = P.build(root, pid, target="remote-gpu", confirm_cloud=True)
    rm = S.read_json(Path(r["package_dir"]) / "RELEASE_MANIFEST.json")
    assert rm["support_status"].startswith("PACKAGE_ONLY") and (Path(r["package_dir"]) / "README_REMOTE_GPU_CN.md").is_file()
    assert "CUDA_VISIBLE_DEVICES" in (Path(r["package_dir"]) / "README_REMOTE_GPU_CN.md").read_text(encoding="utf-8")


def test_fake_training_job_dry_run_writes_the_job_record(tmp_path, synth):
    root, pid, pdir, _ = _profile(tmp_path, dense="off")
    pkg = P.build(root, pid, target="local-cuda")
    r = TR.train(root, pid, package_id=pkg["package_id"], dry_run=True)
    job = S.read_json(Path(r["job"]))
    assert job["schema"] == "stillhere.persona.training_job/1.0" and job["status"] in ("DRY_RUN_OK", "FAILED") and job["dry_run"]
    assert job["dataset_version"] and job["card_id"] and job["base"]["revision"]


# ------------------------------------------------------------------ LLM router
def test_llm_router_schema_fallbacks_and_success():
    def gen_of(text):
        return lambda m, c, a: {"text": text}
    good = '{"intent": "fact", "time": {"kind": "relative", "start": "2025-03-01", "end": "2025-03-07"}, "confidence": 0.9}'
    r = LR.route("上周我们去哪吃的饭来着？", None, gen_of(good))
    assert r["source"] == "llm" and r["route"] == "fact" and r["llm"]["time"]["kind"] == "relative"
    for bad in ("不是 JSON", '{"intent": "answer", "time": {"kind": "none", "start": null, "end": null}, "confidence": 1}',
                '{"intent": "fact", "time": {"kind": "none", "start": "2025-01-01", "end": null}, "confidence": 1}',
                '{"intent": "fact", "time": {"kind": "absolute", "start": "2025-13-01", "end": null}, "confidence": 1}',
                '{"intent": "fact", "time": {"kind": "none", "start": null, "end": null}, "confidence": 1, "answer": "广州"}'):
        r = LR.route("你好呀", None, gen_of(bad))
        assert r["source"] == "rules" and r["llm"]["status"] == "INVALID"
    low = '{"intent": "fact", "time": {"kind": "none", "start": null, "end": null}, "confidence": 0.2}'
    assert LR.route("你好呀", None, gen_of(low))["llm"]["status"] == "LOW_CONFIDENCE"

    def boom(m, c, a):
        raise RuntimeError("down")
    assert LR.route("你好呀", None, boom)["llm"]["status"] == "ERROR"
