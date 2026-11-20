"""Final V4 evaluation (eval_v4 + final_package_v4) on the synthetic persona: OLD / NEW reader groups, the completeness gate (the V3 7 / 35
case), the registered rule, the local-disk dense index, the optional Writer diagnostics and the export. CPU only, no real model
(ModelDouble), no network (mock transport). Not evidence of model quality."""
import hashlib
import json
import sqlite3
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_persona_colab_final import ENV, REPO_ONLY, ModelDouble, mock_transport, world  # noqa: E402,F401
from test_persona_memory_v2 import FakeEmbedder  # noqa: E402
from stillhere.persona import colab_eval as CE  # noqa: E402
from stillhere.persona import eval_v4 as EV4  # noqa: E402
from stillhere.persona import final_package_v4 as FP4  # noqa: E402


def test_v4_notebook_is_valid_and_keeps_sqlite_on_local_disk():
    import ast
    nb = FP4.notebook()
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for src in code:
        ast.parse(src)
    j = "\n".join(code)
    assert "F:\\" not in j and "sk-" not in j and FP4.ZIP_NAME in j and "Gen(BASE, {})" in j and "v4-run-" in j
    assert "local_dir=DENSE_LOCAL" in j and "/content/stillhere_v4_dense" in j and "EV4.stage_memory" in j
    assert nb["metadata"]["colab"]["gpuType"] == "A100"


def test_groups_old_and_new_reader():
    s = EV4.MEMORY_SPECS_V4
    assert list(s) == ["V2L_OLD", "V2L_NEW", "V2D_NEW"] and EV4.CORE == tuple(s)
    assert s["V2L_OLD"]["reader_version"] == "v1" and s["V2L_NEW"]["reader_version"] == s["V2D_NEW"]["reader_version"] == "p1"
    assert not s["V2L_OLD"]["dense"] and not s["V2L_NEW"]["dense"] and s["V2D_NEW"]["dense"] and all(x["adapter"] is None for x in s.values())
    rv = EV4.reader_versions()
    assert rv["v1"]["subject_rules_sha256"] != rv["p1"]["subject_rules_sha256"] and rv["v1"]["fact_system_sha256"] == rv["p1"]["fact_system_sha256"]


def _g(u=0.1, e=3):
    return {"unsupported_claim_rate": u, "attribution_temporal_errors": e}


def test_an_incomplete_core_group_blocks_every_decision():
    comp = {"V2L_OLD": {"complete": True}, "V2L_NEW": {"complete": True}, "V2D_NEW": {"complete": False}}   # the V3 7 / 35 shape
    paired = {"V2L_NEW-V2L_OLD": {"n": 35, "mean_diff": 9.0, "ci95": [3, 15]}}
    d = EV4.decision({"V2L_OLD": _g(), "V2L_NEW": _g(), "V2D_NEW": _g()}, paired, comp)
    assert d["verdict"] == "INCOMPLETE" and d["incomplete_groups"] == ["V2D_NEW"] and "NO_DECISION" in d["recommendation"]


def test_rule_needs_every_condition():
    comp = {g: {"complete": True} for g in EV4.CORE}
    groups = {"V2L_OLD": _g(), "V2L_NEW": _g(), "V2D_NEW": _g()}
    paired = {"V2L_NEW-V2L_OLD": {"n": 35, "mean_diff": 5.0, "ci95": [-1.0, 11]}, "V2D_NEW-V2L_NEW": {"n": 35, "mean_diff": 1.0, "ci95": [-4, 6]}}
    assert EV4.decision(groups, paired, comp)["recommendation"] == "V2L_NEW"            # dense not shown better -> the simpler V2L
    paired["V2D_NEW-V2L_NEW"] = {"n": 35, "mean_diff": 4.0, "ci95": [0.5, 8]}
    assert EV4.decision(groups, paired, comp)["recommendation"] == "V2D_NEW"
    groups["V2L_NEW"] = _g(e=5)                                                          # more attribution errors blocks the new reader
    assert EV4.decision(groups, paired, comp)["recommendation"] == "V2L_OLD"
    groups["V2L_NEW"] = _g()
    paired["V2L_NEW-V2L_OLD"]["ci95"] = [-2.5, 11]
    assert EV4.decision(groups, paired, comp)["recommendation"] == "V2L_OLD"


def _gold(world, tmp_path):
    from stillhere.persona import store as S
    pdir = S.profile_dir(world["root"], world["pid"])
    con = sqlite3.connect(str(pdir / "memory" / "memory.sqlite"))
    ids = [r[0] for r in con.execute("SELECT message_id FROM messages WHERE target_persona_id = ? ORDER BY timestamp_utc LIMIT 3", (world["pid"],))]
    con.close()
    ask = "2030-01-01T00:00:00+08:00"
    rows = [{"question_id": f"V4-T{i}", "question": f"我那次说的第{i}件事是什么？", "answer": "答案", "answerability": "ANSWERABLE", "evidence_message_ids": [m],
             "category": "SINGLE_HOP", "subject_id": "joint", "relevant_timestamps": [], "ask_time": ask, "gold_source": "AI_VERIFIED"} for i, m in enumerate(ids)]
    rows.append({"question_id": "V4-TU", "question": "我们一起去玩滑翔伞是在哪儿？", "answer": "无记录", "answerability": "UNANSWERABLE", "evidence_message_ids": [],
                 "category": "UNANSWERABLE", "subject_id": "joint", "relevant_timestamps": [], "ask_time": ask, "gold_source": "AI_VERIFIED"})
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    g = tmp_path / "AI_VERIFIED_V4_GOLD_v1.jsonl"
    g.write_bytes(body)
    m = tmp_path / "AI_VERIFIED_V4_GOLD_v1_MANIFEST.json"
    m.write_text(json.dumps({"name": "AI_VERIFIED_V4_GOLD_v1", "sha256": hashlib.sha256(body).hexdigest(), "gold_source": "AI_VERIFIED", "is_human_gold": False}),
                 encoding="utf-8")
    return g, m, len(rows)


@REPO_ONLY
def test_final_round_flow_on_cpu(world, tmp_path):
    g, m, n = _gold(world, tmp_path)
    p4 = FP4.build(world["root"], world["pid"], skill_card=world["old"], gold_jsonl=g, gold_manifest=m, out_dir=tmp_path / "pkg4")
    man = p4["manifest"]
    assert man["adapters"] == {} and man["gold"]["gold_source"] == "AI_VERIFIED" and man["plan"]["memory"] == 3 * n and man["core_groups"] == list(EV4.CORE)
    assert (tmp_path / "pkg4" / "V4_EVAL_PROTOCOL.json").is_file()
    sums = dict((l.split("  ")[1], l.split("  ")[0]) for l in (tmp_path / "pkg4" / "SHA256SUMS.txt").read_text().splitlines())
    assert all(hashlib.sha256((tmp_path / "pkg4" / k).read_bytes()).hexdigest() == v for k, v in sums.items()) and len(sums) == 5
    with zipfile.ZipFile(p4["zip"]) as z:
        names = z.namelist()
        assert not any("/adapters/" in x for x in names) and "code/tests/persona_memory_fixture.py" in names
        z.extractall(tmp_path / "colab4")
    with pytest.raises(FP4.PackageV4Error):
        FP4.build(world["root"], world["pid"], skill_card=world["old"], gold_jsonl=g, gold_manifest=m, out_dir=tmp_path / "pkg4")
    run = CE.Run(work_root=tmp_path / "colab4" / "data" / "personas", profile_id=world["pid"], run_dir=tmp_path / "drive" / "v4", manifest=dict(man, code_dir=str(ROOT)),
                 cfg=EV4.defaults({"judge_concurrency": 2, "judge_per_second": 0}), gen=ModelDouble(), log=lambda s: None)
    fz = tmp_path / "colab4" / "frozen"
    EV4.seed_frozen(run, gold_jsonl=fz / g.name, gold_manifest=fz / "AI_VERIFIED_NEW_GOLD_MANIFEST.json", eval_rows_jsonl=fz / "NEW_GOLD_EVAL_ROWS.jsonl")
    # the dense index built on "local disk" with a persisted Drive snapshot
    from stillhere.memory.api import MemoryAPI
    from stillhere.retrieval import local_embedding as LE
    from stillhere.retrieval.ef1 import local_dense as LD
    local = tmp_path / "local_dense"
    api = MemoryAPI(str(run.pdir / "memory" / "memory.sqlite"), public_only=False, persona_names={run.profile_id: ["TA"]})
    LD.build_local(api, run.profile_id, local, FakeEmbedder(), model=LE.CACHE_MODEL_NAME, dimension=16)
    api.close()
    dense, why = LD.open_local(local, model=LE.CACHE_MODEL_NAME, dimension=16, query_encoder=LE.LocalQueryEncoder(FakeEmbedder()))
    assert dense is not None, why
    # the V3 failure shape: one group's rows fail on the first pass -> retried on the next pass, nothing generated twice
    real = EV4.EV2.stage_memory
    calls = {"n": 0}

    def flaky(run_, **kw):
        calls["n"] += 1
        out = real(run_, **kw)
        if calls["n"] == 1:                                            # corrupt V2D_NEW after the first pass: rows become ERROR
            p = Path(run_.run_dir) / "memory_qa" / "V2D_NEW" / "answers.jsonl"
            rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
            p.write_text("".join(json.dumps(dict(r, status="ERROR", error="OperationalError: disk I/O error")) + "\n" for r in rows), encoding="utf-8")
        return out
    EV4.EV2.stage_memory = flaky
    try:
        comp = EV4.stage_memory(run, dense=dense)
    finally:
        EV4.EV2.stage_memory = real
    assert calls["n"] == 2 and all(v["complete"] for v in comp.values())
    oks = [json.loads(l) for l in (Path(run.run_dir) / "memory_qa" / "V2L_OLD" / "answers.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(oks) == n and {r["reader_version"] for r in oks} == {"v1"}           # V2L_OLD generated once, with the OLD reader
    new = [json.loads(l) for l in (Path(run.run_dir) / "memory_qa" / "V2L_NEW" / "answers.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert {r["reader_version"] for r in new} == {"p1"}
    EV4.stage_judge(run, ENV, transport=mock_transport())
    rep = EV4.analysis(run)
    assert rep["decision"]["verdict"] == "COMPLETE" and rep["decision"]["recommendation"] in ("V2L_OLD", "V2L_NEW", "V2D_NEW")
    assert {"valid_answer_preservation", "attribution_false_rejection_with_gold", "third_party_attribution_errors"} <= set(rep["groups"]["V2L_NEW"])
    out = EV4.stage_export(run, {"gpu": "cpu-test"})
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert "reports/ANALYSIS_V4.json" in names and "reports/V4_COMPLETENESS.json" in names and not any(x.endswith(".sqlite") for x in names)
        assert json.loads(z.read("RUN_MANIFEST_V4.json"))["complete"] is True


@REPO_ONLY
def test_an_incomplete_run_is_reported_and_never_adopted(world, tmp_path):
    g, m, n = _gold(world, tmp_path)
    p4 = FP4.build(world["root"], world["pid"], skill_card=world["old"], gold_jsonl=g, gold_manifest=m, out_dir=tmp_path / "pkg4")
    with zipfile.ZipFile(p4["zip"]) as z:
        z.extractall(tmp_path / "colab4")
    run = CE.Run(work_root=tmp_path / "colab4" / "data" / "personas", profile_id=world["pid"], run_dir=tmp_path / "drive" / "v4", manifest=dict(p4["manifest"], code_dir=str(ROOT)),
                 cfg=EV4.defaults({"judge_concurrency": 2, "judge_per_second": 0}), gen=ModelDouble(), log=lambda s: None)
    fz = tmp_path / "colab4" / "frozen"
    EV4.seed_frozen(run, gold_jsonl=fz / g.name, gold_manifest=fz / "AI_VERIFIED_NEW_GOLD_MANIFEST.json", eval_rows_jsonl=fz / "NEW_GOLD_EVAL_ROWS.jsonl")
    comp = EV4.stage_memory(run, dense=None)                        # no dense channel: V2D_NEW is skipped -> INCOMPLETE
    assert comp["V2D_NEW"]["complete"] is False and comp["V2L_OLD"]["complete"] and comp["V2L_NEW"]["complete"]
    st = json.loads((Path(run.run_dir) / "STAGE_STATUS.json").read_text(encoding="utf-8"))
    assert st["V4 STAGE 13 memory completeness"]["status"] == "INCOMPLETE"
    EV4.stage_judge(run, ENV, transport=mock_transport())
    rep = EV4.analysis(run)
    assert rep["decision"]["verdict"] == "INCOMPLETE" and "NO_DECISION" in rep["decision"]["recommendation"]
