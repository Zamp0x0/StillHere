"""Second-round evaluation (eval_v2 + final_package_v2) on the synthetic persona: frozen inputs, two adapters, memory V1 / V2 groups with a
FAKE dense channel, the Writer gate, style + general capability groups, the mock DeepSeek judge, the pre-registered decisions and the export.
CPU only, no real model (ModelDouble), no network (mock transport). Not evidence of model quality."""
import json
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
from stillhere.persona import eval_v2 as EV2  # noqa: E402
from stillhere.persona import evaluation as EV  # noqa: E402
from stillhere.persona import final_package as FP  # noqa: E402
from stillhere.persona import final_package_v2 as FP2  # noqa: E402
from stillhere.persona import memory_store as MS  # noqa: E402


def test_v2_notebook_cells_are_valid_python_and_generic():
    import ast
    nb = FP2.notebook()
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for src in code:
        ast.parse(src)
    joined = "\n".join(code)
    assert "F:\\" not in joined and "sk-" not in joined and FP2.ZIP_NAME in joined and "stage_writer_gate" in joined
    assert nb["metadata"]["colab"]["gpuType"] == "A100"


def test_decision_rules_need_every_condition():
    sp = {"R2S-R1S": {"n": 10, "mean_diff": 3, "ci95": [1, 5]}, "R2S-B1": {"n": 10, "mean_diff": 2, "ci95": [0.5, 4]}}
    v2 = {"groups": {"R2S": {"checks": {"no_conditional_collapse": True}}}}
    gen = {"accuracy": {"GR1": 0.9, "GR2": 0.88}}
    mem = {"paired": {"V2D_R2-V2D": {"n": 50, "mean_diff": 1, "ci95": [-1.5, 3]}},
           "groups": {"V2D": {"unsupported_claim_rate": 0.1}, "V2D_R2": {"unsupported_claim_rate": 0.11}}}
    d = EV2.r2_decision(sp, v2, gen, mem)
    assert d["verdict"] == "R2_BETTER" and d["recommend_enable"]
    v2c = {"groups": {"R2S": {"checks": {"no_conditional_collapse": False}}}}
    assert EV2.r2_decision(sp, v2c, gen, mem)["verdict"] == "R2_NOT_BETTER"                      # collapse alone blocks
    assert EV2.r2_decision(sp, v2, {"accuracy": {"GR1": 0.9, "GR2": 0.8}}, mem)["verdict"] == "R2_NOT_BETTER"
    hold = {"paired": {"V2D-V2L": {"n": 40, "mean_diff": 1, "ci95": [-1, 3]}, "V2L-M1": {"n": 40, "mean_diff": 4, "ci95": [-1, 9]}},
            "groups": {"M1": {"grounded_answer_accuracy": 0.2, "unsupported_claim_rate": 0.14}, "V2L": {"grounded_answer_accuracy": 0.3, "unsupported_claim_rate": 0.12}}}
    m = EV2.memory_v2_decision(hold)
    assert m["chosen_v2_group"] == "V2L" and m["verdict"] == "ADOPT_V2"                          # V2D not clearly better -> the simpler V2L
    assert EV2.memory_v2_decision({})["verdict"] == "NOT_EVALUATED"


@REPO_ONLY
def test_second_round_flow_on_cpu(world, tmp_path):
    from stillhere.persona import adapters as AD
    from stillhere.retrieval.ef1 import local_dense as LD
    from stillhere.retrieval import local_embedding as LE
    from personatrain import export as EX
    import test_persona_p2p3 as T
    # a second adapter of the same card stands in for r2 (synthetic; CPU only)
    run_dir, data = T.make_run(world, tmp_path / "r2run")
    r2 = AD.import_adapter(world["root"], world["pid"], EX.adapter(run_dir, run_dir / "train", "checkpoint-6", "run-r2", data))["adapter_id"]
    assert r2 != world["aid"]
    # frozen FAST inputs from a (small) FAST run directory
    pk = FP.build(world["root"], world["pid"], old_card=world["old"], new_card=world["new"], adapter_id=world["aid"], out_dir=tmp_path / "pkg")
    colab = tmp_path / "colab"
    with zipfile.ZipFile(pk["zip"]) as z:
        z.extractall(colab)
    gen = ModelDouble()
    small = {"PERSONA_SELF_EXPERIENCE": 1, "COUNTERPART_EXPERIENCE": 1, "THIRD_PARTY": 1, "DATE_SUMMARY": 1, "TEMPORAL_CHANGE": 1, "NEGATION_PLAN_CANCEL": 1,
             "CROSS_MESSAGE": 1, "UNANSWERABLE": 1}
    fcfg = CE.defaults({"style_n": 6, "qa_targets": small, "writer_audit_n": 3, "judge_concurrency": 2, "judge_per_second": 0})
    fast = CE.Run(work_root=colab / "data" / "personas", profile_id=world["pid"], run_dir=tmp_path / "drive" / "fast", manifest=pk["manifest"], cfg=fcfg,
                  gen=gen, log=lambda s: None)
    CE.bind(fast)
    CE.stage_07_memory(fast)
    CE.stage_08_09_reports(fast)
    CE.stage_10_qa(fast)
    CE.stage_11_writer_sample(fast)
    CE.dev_items(fast)
    # the v2 package needs a FAST results ZIP: a minimal one with the frozen QA is enough for the builder
    fz = tmp_path / "fast.zip"
    with zipfile.ZipFile(fz, "w") as z:
        z.writestr("MEMORY_QA_100.jsonl", (fast.run_dir / "qa" / "REAL_HISTORY_QA_200.jsonl").read_bytes())
    ids = [json.loads(l)["question_id"] for l in (fast.run_dir / "qa" / "REAL_HISTORY_QA_200.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    fm = tmp_path / "freeze.json"
    fm.write_text(json.dumps({"fast_qa_split": {"dev": ids[::2], "holdout": ids[1::2], "salt": "t", "rule": "t"}}), encoding="utf-8")
    p2 = FP2.build(world["root"], world["pid"], skill_card=world["old"], r1_adapter=world["aid"], r2_adapter=r2, fast_results_zip=fz, freeze_manifest=fm,
                   out_dir=tmp_path / "pkg2", expect_qa_sha256=None)
    man = p2["manifest"]
    assert set(man["adapters"]) == {"r1", "r2"} and not man["contains"]["test_split"] and man["frozen"]["split"] == {"dev": len(ids[::2]), "holdout": len(ids[1::2])}
    with zipfile.ZipFile(p2["zip"]) as z:
        names = z.namelist()
        assert not any("/datasets/" in n and n.rsplit("/", 1)[-1].startswith("test") for n in names)
        assert f"code/{EV2.GENERAL_FILE}" in names and "frozen/MEMORY_V2_SPLIT.json" in names
        z.extractall(tmp_path / "colab2")
    with pytest.raises(FP2.PackageV2Error):
        FP2.build(world["root"], world["pid"], skill_card=world["old"], r1_adapter=world["aid"], r2_adapter=r2, fast_results_zip=fz, freeze_manifest=fm,
                  out_dir=tmp_path / "pkg2", expect_qa_sha256=None)                               # never overwritten
    # the second round itself
    man = dict(man, code_dir=str(ROOT))
    work = tmp_path / "colab2" / "data" / "personas"
    run = CE.Run(work_root=work, profile_id=world["pid"], run_dir=tmp_path / "drive" / "v2", manifest=man,
                 cfg=EV2.defaults({"style_n": 6, "writer_audit_n": 3, "judge_concurrency": 2, "judge_per_second": 0}), gen=gen, log=lambda s: None)
    seeded = EV2.seed_frozen(run, fast_run_dir=fast.run_dir, split={"dev": ids[::2], "holdout": ids[1::2]})
    assert seeded["questions"] == len(ids)
    assert EV2.seed_frozen(run, fast_run_dir=fast.run_dir, split={"dev": ids[::2], "holdout": ids[1::2]})["questions"] == len(ids)   # idempotent
    from stillhere.memory.api import MemoryAPI
    api = MemoryAPI(str(run.pdir / "memory" / "memory.sqlite"), public_only=False, persona_names={run.profile_id: ["TA"]})
    LD.build_local(api, run.profile_id, Path(run.run_dir) / "dense_index", FakeEmbedder(), model=LE.CACHE_MODEL_NAME, dimension=16)
    api.close()
    dense, why = LD.open_local(Path(run.run_dir) / "dense_index", model=LE.CACHE_MODEL_NAME, dimension=16, query_encoder=LE.LocalQueryEncoder(FakeEmbedder()))
    assert dense is not None, why
    mem = EV2.stage_memory(run, dense=dense)
    assert set(mem) == set(EV2.MEMORY_SPECS) and all(v == len(ids) for v in mem.values())
    v2rows = CE._jsonl(run.run_dir / "memory_qa" / "V2D" / "answers.jsonl")
    assert all(r["pipeline"] == "v2" for r in v2rows) and any((r.get("v2_meta") or {}).get("dense") for r in v2rows if r["route"] == "fact")
    gate = EV2.stage_writer_gate(run, synthetic_dir=tmp_path / "syn", env=ENV, transport=mock_transport())
    assert gate["decision"]["decision"] in ("FULL_BUILD", "EVIDENCE_ONLY") and gate["full_build"] != "RUN"
    con_mem = MS.status(run.work_root, run.profile_id)
    assert con_mem.get("memories_active") in (0, None)                                    # allow_full_build False: the store is untouched
    EV2.stage_style(run)
    assert all(len(EV.done_rows(run.run_dir / "style" / g / "answers.jsonl")) == 6 for g in EV2.STYLE_SPECS)
    gr = EV2.stage_general(run)
    assert set(gr["accuracy"]) == set(EV2.GENERAL_SPECS) and gr["n"] == 40
    EV2.stage_judge(run, ENV, transport=mock_transport())
    rep = EV2.analysis(run)
    assert rep["decisions"]["r2"]["verdict"] in ("R2_BETTER", "R2_NOT_BETTER") and rep["decisions"]["memory_v2"]["verdict"] in ("ADOPT_V2", "KEEP_V1")
    assert set(rep["memory"]) == {"ALL", "DEV", "HOLDOUT"} and "V2L-M1" in rep["memory"]["HOLDOUT"]["paired"]
    assert "V2D-V2D_K40" in rep["memory"]["ALL"]["paired"] and set(rep["memory_cost"]) == set(EV2.MEMORY_SPECS)
    assert all(v["input_tokens_mean"] is not None for v in rep["memory_cost"].values())
    assert set(rep["failure_classes"]["by_group"]) == set(EV2.MEMORY_SPECS)
    assert set(rep["failure_classes"]["by_group"]["M0"]) <= {"OK", "N"}                 # no-memory failures are never called retrieval failures
    probe = json.loads((run.run_dir / "reports" / "WRITER_THIRD_PERSON_PROBE.json").read_text(encoding="utf-8"))
    assert set(probe) >= {"third_person_not_self", "violations", "memories", "lines"} and probe["violations"] == (0 if probe["third_person_not_self"] else probe["violations"])
    if not probe["third_person_not_self"]:                                   # a failed probe is a CRITICAL gate finding
        assert any(f["check"] == "third_person_not_self" and f["severity"] == "CRITICAL" for f in gate["decision"]["findings"])
    out = EV2.stage_export(run, {"gpu": "cpu-dry-run"})
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
    assert {"reports/ANALYSIS_V2.json", "RUN_MANIFEST_V2.json", "SHA256SUMS.txt", "memory_qa/V2D/answers.jsonl"} <= names
    assert not any(n.endswith(".sqlite") for n in names)
