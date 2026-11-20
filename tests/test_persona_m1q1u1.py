"""BUILD M1 (WeFlow -> evidence archive -> existing Memory Writer -> SQLite -> EF1), Q1 (quality controller, prompt-first refinement,
runtime selection, quick blind review), U1 (one-import pipeline, runtime, studio, CLI). Synthetic chats only. The memory tests use a
REFERENCE extractor (a test double for the model, including planted wrong candidates) - never presented as semantic extraction."""
import json
import os
import subprocess
import sys
import threading
import urllib.request
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in ("src", "training/persona", "training/d5", "training/d4"):
    sys.path.insert(0, str(ROOT / p))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import persona_fixture as F  # noqa: E402
import persona_memory_fixture as MF  # noqa: E402
from stillhere.persona import card as K  # noqa: E402
from stillhere.persona import evaluation as EV  # noqa: E402
from stillhere.persona import memory as M  # noqa: E402
from stillhere.persona import memory_store as MS  # noqa: E402
from stillhere.persona import package as P  # noqa: E402
from stillhere.persona import pipeline as PL  # noqa: E402
from stillhere.persona import quality as Q  # noqa: E402
from stillhere.persona import service as svc  # noqa: E402
from stillhere.persona import store as S  # noqa: E402
from stillhere.persona.runtime import PersonaRuntime  # noqa: E402


def mem_rows(root, pid):
    pdir = S.profile_dir(root, pid)
    api = MS.open_api(pdir, S.read_json(pdir / "profile.json"))
    try:
        return [tuple(r) for r in api.store._conn.execute("SELECT subject, attribute, value, status, valid_to FROM memories ORDER BY rowid").fetchall()]
    finally:
        api.close()


@pytest.fixture()
def mem(tmp_path):
    path = MF.chat().write(tmp_path / "c.json")
    root = tmp_path / "r"
    pid = svc.init_profile(path, speaker="contact", root=root)["profile_id"]
    return {"path": path, "root": root, "pid": pid, "tmp": tmp_path}


# ======================================================================= M1
def test_evidence_archive_roles_kinds_sessions_and_pending_status(mem):
    r = MS.build_evidence(mem["root"], mem["pid"])
    assert r["added"] == len(MF.SCRIPT) + 1 and r["by_kind"] == {"text": len(MF.SCRIPT), "media": 1} and r["archive_sessions"] == 6
    st = MS.status(mem["root"], mem["pid"])
    assert st["memory_status"] == "EVIDENCE_READY" and st["semantic_extraction"] == MS.PENDING and st["memories_active"] == 0
    pdir = S.profile_dir(mem["root"], mem["pid"])
    api = MS.open_api(pdir, S.read_json(pdir / "profile.json"))
    rows = api.store._conn.execute("SELECT speaker_id, content, locator FROM messages").fetchall()
    api.close()
    by = {c: s for s, c, _ in rows}
    assert by["对，我之前在广州上班"] == mem["pid"] and by["我在澳洲读书呢，最近好忙"] == MS.other_id(mem["pid"])     # roles kept
    assert any(c == "[图片]" and "kind=media" in loc for _, c, loc in rows)                                          # media: a typed event, not text
    res = MS.build_semantic(mem["root"], mem["pid"])                                                                    # no writer: evidence only, honest
    assert res["semantic_extraction"] == MS.PENDING and res["writer"] == "none"
    with pytest.raises(MS.MemoryError_) as e:
        MS.build_semantic(mem["root"], mem["pid"], writer="deepseek")                                                  # external API never by default
    assert e.value.code == "EXTERNAL_API_NOT_ALLOWED"


def test_writer_attribution_temporal_change_plans_and_wrong_candidates(mem):
    MS.build_evidence(mem["root"], mem["pid"])
    with pytest.raises(MS.MemoryError_) as e:
        MS.build_semantic(mem["root"], mem["pid"], complete=MF.reference_complete)                                    # the store's owner-review gate
    assert e.value.code == "OWNER_REVIEW_REQUIRED"
    r = MS.build_semantic(mem["root"], mem["pid"], complete=MF.reference_complete, reviewer="tester")
    assert r["run"]["errors"] == [] and r["memories_active"] > 0 and r["memory_status"] == "READY"
    rows = mem_rows(mem["root"], mem["pid"])
    self_rows = [x for x in rows if x[0] == "persona"]
    works = {(v, s, vt is None) for _, a, v, s, vt in self_rows if a == "works_at"}
    assert ("深圳", "active", True) in works and ("广州", "active", False) in works                                  # 广州 closed, 深圳 current
    assert not any("澳洲" in str(x) for x in self_rows)                                                                # the other person's school
    assert not any(v == "去香港" and a == "event.status" for _, a, v, _, _ in rows)                                     # a question / maybe: no trip
    assert not any(x[1] == "event.actor" and x[2] == "self" and any(y[0] == x[0] and y[2] == "去香港" for y in rows) for x in rows)
    st = {v for s_, a, v, status, vt in rows if a == "event.status" and status == "active" and vt is None}
    assert "CANCELLED" in st                                                                                            # plan -> cancelled
    assert any(x[1] == "event.actor" and x[2] == "person:小雨" for x in rows) and ("persona", "relationship", "妹妹 小雨", "active", None) in rows
    assert ("persona", "dislikes", "咖啡", "active", None) in rows                                                      # negation kept
    q = MS.query(mem["root"], mem["pid"], "现在在哪里上班", ask_time=datetime(2025, 2, 1, tzinfo=MS.TZ))
    mems = [m for i in q["items"] for m in i["memories"]]
    assert any("深圳" in m for m in mems) and all(i["message_ids"] or i["memories"] for i in q["items"])              # EF1 finds it, with provenance


def test_idempotent_reimport_incremental_update_and_edit_conflict(mem):
    MS.build_evidence(mem["root"], mem["pid"])
    MS.build_semantic(mem["root"], mem["pid"], complete=MF.reference_complete, reviewer="tester")
    before = mem_rows(mem["root"], mem["pid"])
    again = MS.build_evidence(mem["root"], mem["pid"])
    r2 = MS.build_semantic(mem["root"], mem["pid"], complete=MF.reference_complete, reviewer="tester")
    assert again["added"] == 0 and again["unchanged"] == len(MF.SCRIPT) + 1 and r2["run"]["sessions_processed"] == 0 and mem_rows(mem["root"], mem["pid"]) == before
    newer = MF.chat(extra=[(30, 0, F.CONTACT, "我搬到杭州了"), (30, 60, F.SELF, "哇")]).write(mem["tmp"] / "c2.json")
    inc = MS.build_evidence(mem["root"], mem["pid"], source=newer)
    r3 = MS.build_semantic(mem["root"], mem["pid"], complete=MF.reference_complete, reviewer="tester")
    assert inc["added"] == 2 and inc["unchanged"] == len(MF.SCRIPT) + 1 and inc["edited_conflicts_kept_original"] == 0 and r3["run"]["sessions_processed"] == 1 and r3["run"]["sessions_already_done"] >= 5
    after = mem_rows(mem["root"], mem["pid"])
    assert after[:len(before)] == before and any(x[2] == "杭州" for x in after)                                         # history kept, only the new part
    raw = json.loads(newer.read_text(encoding="utf-8"))
    raw["messages"][1]["content"] = "改过的内容"                                                                        # same id, other content
    (mem["tmp"] / "c3.json").write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    ed = MS.build_evidence(mem["root"], mem["pid"], source=mem["tmp"] / "c3.json")
    assert ed["edited_conflicts_kept_original"] == 1 and ed["added"] == 0
    raw["messages"] = raw["messages"][2:]
    (mem["tmp"] / "c4.json").write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    assert MS.build_evidence(mem["root"], mem["pid"], source=mem["tmp"] / "c4.json")["absent_from_this_export_kept"] == 2
    manifests = sorted((S.profile_dir(mem["root"], mem["pid"]) / "memory" / "manifests").glob("IMPORT_*.json"))
    assert len(manifests) == 5


def test_date_conditioned_retrieval_and_audit_layers(mem):
    MS.build_evidence(mem["root"], mem["pid"])
    MS.build_semantic(mem["root"], mem["pid"], complete=MF.reference_complete, reviewer="tester")
    q = MS.query(mem["root"], mem["pid"], "2025年1月5日聊了什么", ask_time=datetime(2025, 3, 1, tzinfo=MS.TZ))
    on_day = [i for i in q["items"] if (i["time"] or "").startswith("2025-01-05")]
    assert len(on_day) >= 2 and len(on_day) >= len(q["items"]) // 2                                                     # the date window dominates
    a = MS.audit(mem["root"], mem["pid"], probes=10)
    L = a["layers"]
    assert L["imported_messages"] == len(MF.SCRIPT) + 1 and L["searchable_text_evidence"] == len(MF.SCRIPT)
    assert L["extracted_candidates"] >= L["verified_and_committed_operations"] > 0 and L["persisted_active_memories"] > 0
    assert a["retrieval_probes"]["exact_message_recall_at_8"].split("/")[0] == a["retrieval_probes"]["exact_message_recall_at_8"].split("/")[1]
    assert Path(a["path"]).is_file() and "广州" not in json.dumps(a, ensure_ascii=False)                                 # counts only


def test_production_and_evaluation_memory_views_never_mix(tmp_path, monkeypatch):
    orig = M.build_api
    monkeypatch.setattr(M, "build_api", lambda *a, **k: orig(*a, **{**k, "synthetic": True}))
    chat = F.standard(days=80).write(tmp_path / "c.json")
    root = tmp_path / "r"
    pid = svc.init_profile(chat, speaker="contact", root=root)["profile_id"]
    prep = svc.prepare_all(root, pid)
    ddir = Path(prep["dataset_dir"])
    MS.build_evidence(root, pid)
    cut = M.cutoff_from_train(ddir)
    prof = S.read_json(S.profile_dir(root, pid) / "profile.json")
    api, meta = M.build_api(prof["source"]["path"], persona_id=pid, target_username=prof["target"]["username"], cutoff=cut)
    eval_n = meta["messages"]
    api.close()
    prod = MS.status(root, pid)
    assert prod["archive_messages"] > eval_n and prod["last"] > cut.isoformat()                                         # production holds the later period
    P.dev_eval_set(S.profile_dir(root, pid), ddir, prof, K.current(S.profile_dir(root, pid)))
    items = EV.read_eval_set(next((S.profile_dir(root, pid) / "evals" / "sets").glob("*_dev.jsonl")))
    import re
    for it in items:                                                                                                    # D / E use the train-cutoff view only
        for d in re.findall(r"\] (\d{4}-\d{2}-\d{2})", it["memory"]):
            assert d <= cut.date().isoformat()
    rt_e = PersonaRuntime(root, pid, backend="fake", memory="eval")
    rt_p = PersonaRuntime(root, pid, backend="fake", memory="production")
    assert rt_e.memory_view == "eval" and rt_p.memory_view == "production" and rt_e.api is not None and rt_p.api is None


# ======================================================================= Q1
def _world(tmp_path, monkeypatch):
    orig = M.build_api
    monkeypatch.setattr(M, "build_api", lambda *a, **k: orig(*a, **{**k, "synthetic": True}))
    chat = F.standard(days=80).write(tmp_path / "c.json")
    root = tmp_path / "r"
    pid = svc.init_profile(chat, speaker="contact", root=root)["profile_id"]
    prep = svc.prepare_all(root, pid)
    pdir = S.profile_dir(root, pid)
    ddir = Path(prep["dataset_dir"])
    cur = K.current(pdir)
    es = P.dev_eval_set(pdir, ddir, S.read_json(pdir / "profile.json"), cur)
    return root, pid, pdir, ddir, cur, es


def _answers(run, group, texts, gen="QWEN3_8B_NF4_D5_REAL_GPU"):
    (run / group).mkdir(parents=True, exist_ok=True)
    with open(run / group / "answers.jsonl", "w", encoding="utf-8") as f:
        for sid, t in texts.items():
            f.write(json.dumps({"sample_id": sid, "group": group, "status": "OK", "text": t, "generator": gen}, ensure_ascii=False) + "\n")


def _scored_run(pdir, ddir, cur, es, name, groups_texts):
    run = pdir / "evals" / "runs" / name
    items = EV.read_eval_set(es)
    for g, fn in groups_texts.items():
        _answers(run, g, {it["sample_id"]: fn(i, it) for i, it in enumerate(items)})
    head = S.read_json(es.with_suffix(".meta.json"))
    S.write_json(run / "RUN_FINGERPRINT.json", {"fields": {"eval_set_sha256": head["file_sha256"]}, "sha256": "x"})
    EV.score_run(run, items, list(groups_texts), EV.Reference(ddir, cur["card"]))
    return run, items


def test_quality_controller_diagnoses_and_proposes_generic_candidates(tmp_path, monkeypatch):
    root, pid, pdir, ddir, cur, es = _world(tmp_path, monkeypatch)
    long = lambda i, it: f"这个问题我觉得我们可以先慢慢看看再说{i}\n然后再决定\n别着急\n好吗"                                   # base: long, split
    short = lambda i, it: ["嗯", "好"][i % 2]                                                                           # adapter: collapsed
    run, items = _scored_run(pdir, ddir, cur, es, "r_syn", {"B": long, "C": short})
    d = Q.diagnose(root, pid, run_dir=run)
    types = {g: {f["type"] for f in v} for g, v in d["failures"].items()}
    assert {"too_long", "over_split"} <= types["B"] and "generic_collapse" in types["C"]
    assert "too_short" not in types["C"]                   # V2: 1-2 char replies are inside this person's own length range (authentic short style)
    assert all("adapter" in f["likely_source"] for f in d["failures"]["C"]) and d["test_read"] is False
    cands = d["sections"]["prompt_improvement_candidates"]
    assert 1 <= len(cands) <= Q.MAX_CANDIDATES and all(set(c["options"]) <= {"length", "respond_first", "anti_generic", "bubble_cap"} for c in cands)
    assert cands[0]["options"]["bubble_cap"] == max(2, Q.ref_stats(items)["bubbles_p95"])                               # from the person's own data
    assert d["sections"]["recommended_runtime"]["adapter"] is False and d["retrain_recommendation"]["recommend_retrain"] is False
    for k in ("style_fidelity", "reply_appropriateness", "naturalness", "reply_length", "characteristic_expressions", "generic_short_reply_rate", "collapse_risk",
              "repetition_risk", "memorisation_risk", "unsupported_fact_risk", "privacy_leakage_risk", "memory_consistency", "base_vs_adapter",
              "prompt_improvement_candidates", "recommended_runtime"):
        assert k in d["sections"]
    assert d["sections"]["naturalness"].startswith("NOT_VERIFIED")


def test_refinement_versions_prompts_keeps_original_on_dry_run_and_adopts_only_a_real_fix(tmp_path, monkeypatch):
    root, pid, pdir, ddir, cur, es = _world(tmp_path, monkeypatch)
    refs = {it["sample_id"]: it["reference"] for it in EV.read_eval_set(es)}
    run, items = _scored_run(pdir, ddir, cur, es, "r_syn", {"B": lambda i, it: refs[it["sample_id"]] + "呀", "C": lambda i, it: ["嗯", "好"][i % 2]})
    plan = Q.refine_prepare(root, pid)
    assert plan["candidates"] and K.current(pdir)["card_id"] == cur["card_id"]                                          # candidates are not adopted yet
    for c in plan["candidates"]:
        set_items = EV.read_eval_set(Path(c["eval_set"]))
        assert all(it["messages"]["B"] == it["messages"]["C"] for it in set_items) and "先回应对方最新一句话" in set_items[0]["messages"]["B"][0]["content"]
        assert K.load(pdir, c["card_id"])["manifest"]["parent_card"] == cur["card_id"]
    Q.refine_generate(root, pid, plan["refine_id"], backend="fake", log=lambda s: None)
    sel = Q.refine_select(root, pid, plan["refine_id"])
    assert sel["status"] == "DRY_RUN_ONLY" and sel["selected"] is None and K.current(pdir)["card_id"] == cur["card_id"]
    c0 = plan["candidates"][0]
    crun = Path(c0["eval_set"]).parent / "run"
    import shutil
    shutil.rmtree(crun)
    _answers(crun, "B", {it["sample_id"]: refs[it["sample_id"]] + "呀" for it in items})
    cn = "一二三四五六七八九十"
    _answers(crun, "C", {it["sample_id"]: f"收到啦{cn[i % 10]}{cn[(i // 10) % 10]}号" for i, it in enumerate(items)})      # the fix: varied, not generic, normal length
    (crun / "GENERATION_STATUS.json").write_text("{}", encoding="utf-8")
    sel = Q.refine_select(root, pid, plan["refine_id"])
    assert sel["status"] == "ADOPTED" and sel["selected"] == c0["name"] and K.current(pdir)["card_id"] == c0["card_id"]
    assert set(sel["candidates"][c0["name"]]["fixed"]) >= {"generic_collapse"} and not sel["candidates"][c0["name"]]["new"]


def test_runtime_selection_follows_the_verdict_and_memory_and_allows_override(tmp_path, monkeypatch):
    root, pid, pdir, ddir, cur, es = _world(tmp_path, monkeypatch)
    run, items = _scored_run(pdir, ddir, cur, es, "r_syn", {"B": lambda i, it: "好的呀那就这样吧", "C": lambda i, it: ["嗯", "好"][i % 2]})
    Q.diagnose(root, pid, run_dir=run)
    c = Q.select_runtime(root, pid)
    assert c["adapter"]["enabled"] is False and c["memory"]["enabled"] is False and "NOT_BETTER" in c["selection_reason"][0]
    MS.build_evidence(root, pid)
    MS.audit(root, pid, probes=10)
    Q.diagnose(root, pid, run_dir=run)
    c = Q.select_runtime(root, pid)
    assert c["memory"]["enabled"] is True and c["memory"]["view"] == "production_full_history" and c["memory"]["semantic_extraction"] == MS.PENDING
    c = Q.select_runtime(root, pid, override_memory=False)
    assert c["memory"]["enabled"] is False and c["user_override"]["memory"] is False
    rt = PersonaRuntime.from_config(root, pid, backend="fake")
    assert rt.active is None and rt.memory_view is None and rt.reply([("other", "在吗")])["mode"] == "B"


def test_quick_blind_review_is_small_diverse_and_never_overwrites(tmp_path, monkeypatch):
    root, pid, pdir, ddir, cur, es = _world(tmp_path, monkeypatch)
    run, items = _scored_run(pdir, ddir, cur, es, "r_syn", {"B": lambda i, it: f"左边第{i}句", "C": lambda i, it: "好" if i % 3 else f"右边{i}"})
    r = Q.make_blind_review(run, items, ["B", "C"], n=8)
    assert 5 <= r["rows"] <= 8 and Path(r["html"]).is_file() and r["label"].startswith("EXPLORATORY")
    html = Path(r["html"]).read_text(encoding="utf-8")
    assert "__DATA__" not in html and '"B"' not in html.split("const DATA=")[1][:20]
    with pytest.raises(EV.EvalError) as e:
        Q.make_blind_review(run, items, ["B", "C"], n=8)
    assert e.value.code == "BLIND_EXISTS"


# ======================================================================= U1
def test_one_import_pipeline_partial_ready_resumable_and_chains_independent(tmp_path, monkeypatch):
    orig = M.build_api
    monkeypatch.setattr(M, "build_api", lambda *a, **k: orig(*a, **{**k, "synthetic": True}))
    chat = F.standard(days=80).write(tmp_path / "c.json")
    root = tmp_path / "r"
    r = PL.run(chat, speaker="contact", root=root, target="colab-l4")
    st = {k: v["status"] for k, v in r["stages"].items()}
    assert [st[c] for c in ("01", "02", "03", "05", "06", "07", "08", "10", "11", "16")] == ["PASS"] * 10
    assert st["09"] == "PENDING" and st["12"] == "BLOCKED" and st["13"] == "PENDING" and st["14"] == "PENDING" and r["overall"] == "PARTIAL_READY"
    cfg = S.read_json(Path(r["runtime_config"]))
    assert cfg["memory"]["enabled"] and not cfg["adapter"]["enabled"]
    r2 = PL.run(chat, speaker="contact", root=root, target="cpu-only")
    assert r2["stages"]["12"]["status"] == "PASS" and r2["profile_id"] == r["profile_id"]
    monkeypatch.setattr(MS, "build_evidence", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))
    r3 = PL.run(chat, speaker="contact", root=tmp_path / "r3", target="cpu-only")
    st3 = {k: v["status"] for k, v in r3["stages"].items()}
    assert st3["08"] == "FAILED" and st3["07"] == "PASS" and st3["12"] == "PASS" and st3["16"] == "PASS" and r3["overall"] == "PARTIAL_READY"
    assert (S.profile_dir(tmp_path / "r3", r3["profile_id"]) / "logs" / "pipeline.log").is_file()


def test_runtime_guards_loop_and_redacts_private_values(tmp_path, monkeypatch):
    orig = M.build_api
    monkeypatch.setattr(M, "build_api", lambda *a, **k: orig(*a, **{**k, "synthetic": True}))
    chat = F.standard(days=80).write(tmp_path / "c.json")
    root = tmp_path / "r"
    pid = PL.run(chat, speaker="contact", root=root, target="cpu-only")["profile_id"]
    rt = PersonaRuntime.from_config(root, pid, backend="fake")
    class Loop(EV.FakeGenerator):
        def __call__(self, messages, cfg, adapter):
            return {"text": "好的呀，好的呀，好的呀，好的呀，我的电话是13812345678", "hit_token_limit": False}
    rt.gen = Loop()
    out = rt.reply([("other", "今天怎么样")])
    assert out["mode"] == "D" and "13812345678" not in out["text"] and out["redacted"] and out["loop_guard"]["status"] != "CLEAN"
    assert out["synthetic"] and out["evidence_chars"] > 0                                                              # production memory reached the prompt


def test_persona_studio_serves_locally_and_calls_the_services(tmp_path, monkeypatch):
    from stillhere.persona.studio import serve
    orig = M.build_api
    monkeypatch.setattr(M, "build_api", lambda *a, **k: orig(*a, **{**k, "synthetic": True}))
    chat = F.standard(days=80).write(tmp_path / "c.json")
    root = tmp_path / "r"
    pid = PL.run(chat, speaker="contact", root=root, target="cpu-only")["profile_id"]
    httpd, studio = serve(root, 0, "fake")
    port = httpd.server_address[1]
    assert httpd.server_address[0] == "127.0.0.1"
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    def call(path, body=None, host=None):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **({"Host": host} if host else {})})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
    try:
        code, page = call("/")
        assert code == 200 and b"Persona Studio" in page
        assert json.loads(call("/api/profiles")[1])["profiles"][0]["profile_id"] == pid
        assert json.loads(call(f"/api/state?profile={pid}")[1])["stages"]["17"]["status"] == "PARTIAL_READY"
        assert json.loads(call(f"/api/card?profile={pid}")[1])["card_id"]
        assert json.loads(call(f"/api/memory?profile={pid}")[1])["status"]["memory_status"] == "EVIDENCE_READY"
        first = next(m["content"] for m in json.loads(chat.read_text(encoding="utf-8"))["messages"] if len(m["content"]) >= 6)
        q = json.loads(call("/api/memory/query", {"profile": pid, "q": first[:12]})[1])
        assert q["items"]
        c = json.loads(call("/api/chat", {"profile": pid, "history": [{"role": "other", "text": "在吗"}]})[1])
        assert c["synthetic"] and c["mode"] == "D" and c["text"]
        sp = json.loads(call("/api/speakers", {"input": str(chat)})[1])
        assert len(sp["speakers"]) == 2
        assert call("/api/profiles", host="evil.example")[0] == 403                                                     # DNS-rebinding guard
    finally:
        httpd.shutdown()


def cli(*args):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, "-m", "stillhere.cli", "persona", *args], capture_output=True, text=True, encoding="utf-8", env=env, timeout=600)


def test_cli_unified_pipeline_memory_commands_finalize_and_chat(tmp_path):
    chat = F.standard(days=80).write(tmp_path / "c.json")
    root = tmp_path / "r"
    r = cli("pipeline", "--input", str(chat), "--speaker", "contact", "--root", str(root), "--non-interactive", "--json")
    out = json.loads(r.stdout[r.stdout.index("{"):])
    assert r.returncode == 0 and out["overall"] == "PARTIAL_READY" and out["stages"]["12"]["status"] == "BLOCKED"
    pid = out["profile_id"]
    m = json.loads(cli("memory-build", "--profile", pid, "--root", str(root), "--json").stdout)
    assert m["evidence"]["added"] == 0 and m["memory"]["semantic_extraction"] == MS.PENDING
    assert cli("memory-build", "--profile", pid, "--root", str(root), "--writer", "deepseek").returncode == 3            # never external by default
    a = json.loads(cli("memory-audit", "--profile", pid, "--root", str(root), "--json").stdout)
    assert a["status"] == "EVIDENCE_READY" and a["layers"]["persisted_active_memories"] == 0
    first = next(m["content"] for m in json.loads(chat.read_text(encoding="utf-8"))["messages"] if len(m["content"]) >= 6)
    assert json.loads(cli("memory-query", "--profile", pid, "--root", str(root), "--q", first[:12], "--json").stdout)["items"]
    f = json.loads(cli("finalize", "--profile", pid, "--root", str(root), "--json").stdout)
    assert f["quality_report"].startswith("NOT_EVALUATED") and f["memory"]["enabled"] and not f["adapter"]["enabled"]
    c = cli("chat", "--profile", pid, "--root", str(root), "--backend", "fake", "--message", "在吗")
    assert c.returncode == 0 and "memory production" in c.stdout and "SYNTHETIC" in c.stdout
    assert cli("refine", "--profile", pid, "--root", str(root)).returncode == 6                                         # no DEV evaluation yet: honest error


def test_pipeline_evaluation_and_refinement_stages_with_an_adapter_and_a_dev_run(tmp_path, monkeypatch):
    root, pid, pdir, ddir, cur, es = _world(tmp_path, monkeypatch)
    _scored_run(pdir, ddir, cur, es, "r_syn", {"B": lambda i, it: f"这个我们改天再慢慢说吧{i}", "C": lambda i, it: ["嗯", "好"][i % 2]})
    from stillhere.persona import adapters as AD
    monkeypatch.setattr(AD, "installed", lambda pdir: [{"adapter_id": "a-x", "name": "PERSONA_ADAPTER_CANDIDATE", "run_id": "r", "checkpoint": "checkpoint-1",
                                                        "card_id": cur["card_id"], "adapter_sha256": "0" * 64}])
    chat = Path(S.read_json(pdir / "profile.json")["source"]["path"])
    r = PL.run(chat, speaker="contact", root=root, target="cpu-only")
    st = r["stages"]
    assert st["14"]["status"] == "PASS" and st["14"]["quality_status"] == "REAL_MODEL" and st["14"]["verdicts"]["B_vs_C"] == "NOT_BETTER"
    assert st["15"]["status"] == "PENDING" and st["15"]["result"] == "PENDING_GPU_EVALUATION" and st["15"]["candidates"]
    assert st["16"]["status"] == "PASS" and st["16"]["adapter"] is False and r["overall"] == "PARTIAL_READY"


def test_review_only_preview_of_real_sessions_commits_nothing(mem):
    MS.build_evidence(mem["root"], mem["pid"])
    pdir = S.profile_dir(mem["root"], mem["pid"])
    api = MS.open_api(pdir, S.read_json(pdir / "profile.json"))
    convs = [r[0] for r in api.store._conn.execute("SELECT DISTINCT conversation_id FROM messages").fetchall()][:3]
    api.close()
    out = mem["tmp"] / "candidates.json"
    r = MS.preview_sessions(mem["root"], mem["pid"], convs, complete=MF.reference_complete, db_path=mem["tmp"] / "smoke.sqlite", out_json=out)
    assert r["sessions"] == 3 and r["candidates"] == r["verified"] + r["ambiguous"] + r["rejected"] and r["committed"] == 0
    assert r["rejected"] >= 1 and "a fact about the owner that the owner did not say" in r["rejection_reasons"]           # the planted wrong attribution
    assert MS.status(mem["root"], mem["pid"])["memories_active"] == 0                                                       # production untouched
    assert all(c["status"] in ("AUTO_VERIFIED_PENDING_HUMAN_REVIEW", "AMBIGUOUS", "UNSUPPORTED") for c in json.loads(out.read_text(encoding="utf-8"))["candidates"])
