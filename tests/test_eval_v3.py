"""Third-round evaluation (eval_v3 + final_package_v3) on the synthetic persona: an AI_VERIFIED gold set, the five frozen memory groups
with a FAKE dense channel, the mock DeepSeek judge, the registered rule and the export. CPU only, no real model (ModelDouble), no network
(mock transport). Not evidence of model quality."""
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
from stillhere.persona import eval_v2 as EV2  # noqa: E402
from stillhere.persona import eval_v3 as EV3  # noqa: E402
from stillhere.persona import final_package_v3 as FP3  # noqa: E402
from stillhere.persona import memory_v2 as V2  # noqa: E402


def test_v3_notebook_cells_are_valid_python_and_load_no_adapter():
    import ast
    nb = FP3.notebook()
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for src in code:
        ast.parse(src)
    joined = "\n".join(code)
    assert "F:\\" not in joined and "sk-" not in joined and FP3.ZIP_NAME in joined and "Gen(BASE, {})" in joined
    assert "EV3.stage_memory" in joined and "stage_writer_gate" not in joined and "stage_style" not in joined
    assert nb["metadata"]["colab"]["gpuType"] == "L4" and "v3-run-" in joined


def test_groups_are_the_frozen_candidates():
    s = EV3.MEMORY_SPECS_V3
    assert list(s) == ["M1", "V2L", "V2D", "BEST_STABLE_V2", "BEST_EVIDENCE_FOCUSED"]
    for g in ("M1", "V2L", "V2D"):
        assert s[g] == EV2.MEMORY_SPECS[g]
    fc = V2.FUTURE_CANDIDATES
    assert s["BEST_STABLE_V2"]["dense"] is fc["BEST_STABLE_V2"]["use_dense"] and s["BEST_STABLE_V2"]["strip_qform"] is True
    assert s["BEST_EVIDENCE_FOCUSED"]["k"] == fc["BEST_EVIDENCE_FOCUSED"]["k"] == 40 and s["BEST_EVIDENCE_FOCUSED"]["neighbors"] == (0, 0)
    assert all(sp["adapter"] is None for sp in s.values())
    assert "strip_qform" not in s["BEST_EVIDENCE_FOCUSED"] and "strip_qform" not in s["V2L"]


def test_rule_needs_every_condition():
    g = lambda u, e: {"unsupported_claim_rate": u, "attribution_temporal_errors": e}           # noqa: E731
    groups = {"V2L": g(0.1, 3), "BEST_STABLE_V2": g(0.1, 3), "BEST_EVIDENCE_FOCUSED": g(0.2, 3)}
    paired = {"BEST_STABLE_V2-V2L": {"n": 36, "mean_diff": 2.0, "ci95": [-1.5, 5]}, "BEST_EVIDENCE_FOCUSED-V2L": {"n": 36, "mean_diff": 4.0, "ci95": [1, 7]}}
    d = EV3.decision(groups, paired)
    assert d["per_candidate"]["BEST_STABLE_V2"]["verdict"] == "REPLACES_V2L"
    assert d["per_candidate"]["BEST_EVIDENCE_FOCUSED"]["verdict"] == "V2L_STAYS"                 # more unsupported claims blocks
    assert d["recommendation"] == "BEST_STABLE_V2" and "AI_VERIFIED" in d["gold"]
    paired["BEST_STABLE_V2-V2L"]["ci95"] = [-2.5, 5]
    assert EV3.decision(groups, paired)["recommendation"] == "V2L"
    assert EV3.decision({}, {})["per_candidate"]["BEST_STABLE_V2"]["verdict"] == "NOT_EVALUATED"


def test_an_incomplete_group_gets_no_verdict():
    # the first real V3 run: 28 / 35 BEST_EVIDENCE_FOCUSED generations failed (disk I/O error); the 7 left were chat / date routed
    g = lambda n: {"n": n, "unsupported_claim_rate": 0.0, "attribution_temporal_errors": 0}       # noqa: E731
    groups = {"V2L": g(35), "BEST_STABLE_V2": g(35), "BEST_EVIDENCE_FOCUSED": g(7)}
    paired = {"BEST_STABLE_V2-V2L": {"n": 35, "mean_diff": -0.4, "ci95": [-1.9, 1.2]}, "BEST_EVIDENCE_FOCUSED-V2L": {"n": 7, "mean_diff": -1.1, "ci95": [-3.4, 0]}}
    d = EV3.decision(groups, paired, 35)
    assert d["per_candidate"]["BEST_EVIDENCE_FOCUSED"]["verdict"] == "INCOMPLETE" and d["per_candidate"]["BEST_STABLE_V2"]["verdict"] == "V2L_STAYS"
    assert d["recommendation"] == "V2L_PENDING_BEST_EVIDENCE_FOCUSED"


def test_eval_rows_keep_the_gold_and_mark_it_ai_verified():
    gold = [{"question_id": "NG-1", "question": "q", "answer": "a", "answerability": "ANSWERABLE", "evidence_message_ids": ["m1"], "category": "MULTI_HOP",
             "subject_id": "third_party:对方的妈妈", "relevant_timestamps": ["2025-05-01T10:00+08:00"], "ask_time": "2026-04-05T00:00:00+08:00"},
            {"question_id": "NG-2", "question": "q2", "answer": "无记录", "answerability": "UNANSWERABLE", "evidence_message_ids": [], "category": "UNANSWERABLE",
             "subject_id": "joint", "relevant_timestamps": [], "ask_time": "2026-04-05T00:00:00+08:00"}]
    rows = EV3.to_eval_rows(gold, {"m1": {"time_local": "2025-05-01T10:00+08:00", "speaker": "对方", "text": "t"}})
    assert rows[0]["subject"] == "THIRD_PARTY" and rows[0]["reference_facts"] == ["2025-05-01 10:00 对方：t"] and rows[0]["verification_status"] == "AI_VERIFIED"
    assert rows[1]["subject"] == "JOINT" and rows[1]["unanswerable_basis"] == "无记录" and rows[1]["evidence_message_ids"] == []
    assert rows[0]["time_scope"] == {"ask_time": "2026-04-05T00:00:00+08:00", "evidence_dates": ["2025-05-01"]}


def _gold(world, tmp_path):
    from stillhere.persona import store as S
    pdir = S.profile_dir(world["root"], world["pid"])
    con = sqlite3.connect(str(pdir / "memory" / "memory.sqlite"))
    ids = [r[0] for r in con.execute("SELECT message_id FROM messages WHERE target_persona_id = ? ORDER BY timestamp_utc LIMIT 3", (world["pid"],))]
    last = con.execute("SELECT MAX(timestamp_utc) FROM messages").fetchone()[0]
    con.close()
    ask = "2030-01-01T00:00:00+08:00"
    assert last < "2030"
    rows = [{"question_id": f"NG-T{i}", "question": f"你还记得那件事吗{i}？", "answer": "答案", "answerability": "ANSWERABLE", "evidence_message_ids": [m],
             "category": "SINGLE_HOP", "subject_id": "self", "relevant_timestamps": [], "ask_time": ask, "gold_source": "AI_VERIFIED"} for i, m in enumerate(ids)]
    rows.append({"question_id": "NG-TU", "question": "我们一起去玩滑翔伞是在哪儿？", "answer": "无记录", "answerability": "UNANSWERABLE", "evidence_message_ids": [],
                 "category": "UNANSWERABLE", "subject_id": "joint", "relevant_timestamps": [], "ask_time": ask, "gold_source": "AI_VERIFIED"})
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    g = tmp_path / "AI_VERIFIED_NEW_GOLD_v1.jsonl"
    g.write_bytes(body)
    m = tmp_path / "AI_VERIFIED_NEW_GOLD_v1_MANIFEST.json"
    m.write_text(json.dumps({"name": "AI_VERIFIED_NEW_GOLD_v1", "sha256": hashlib.sha256(body).hexdigest(), "gold_source": "AI_VERIFIED", "is_human_gold": False}),
                 encoding="utf-8")
    return g, m, len(rows)


def test_package_refuses_a_gold_file_that_changed(world, tmp_path):
    g, m, _ = _gold(world, tmp_path)
    g.write_bytes(g.read_bytes() + b"\n")
    with pytest.raises(FP3.PackageV3Error, match="AI_VERIFIED"):
        FP3.build(world["root"], world["pid"], skill_card=world["old"], gold_jsonl=g, gold_manifest=m, out_dir=tmp_path / "pkg3")
    m.write_text(json.dumps({"name": "x", "sha256": hashlib.sha256(g.read_bytes()).hexdigest(), "gold_source": "HUMAN_VERIFIED", "is_human_gold": True}),
                 encoding="utf-8")
    with pytest.raises(FP3.PackageV3Error):
        FP3.build(world["root"], world["pid"], skill_card=world["old"], gold_jsonl=g, gold_manifest=m, out_dir=tmp_path / "pkg3")


@REPO_ONLY
def test_third_round_flow_on_cpu(world, tmp_path):
    from stillhere.memory.api import MemoryAPI
    from stillhere.retrieval import local_embedding as LE
    from stillhere.retrieval.ef1 import local_dense as LD
    g, m, n = _gold(world, tmp_path)
    p3 = FP3.build(world["root"], world["pid"], skill_card=world["old"], gold_jsonl=g, gold_manifest=m, out_dir=tmp_path / "pkg3")
    man = p3["manifest"]
    assert man["adapters"] == {} and man["gold"]["gold_source"] == "AI_VERIFIED" and man["gold"]["is_human_gold"] is False and man["plan"]["memory"] == 5 * n
    with zipfile.ZipFile(p3["zip"]) as z:
        names = z.namelist()
        assert not any("/adapters/" in x for x in names) and not any("/datasets/" in x and x.rsplit("/", 1)[-1].startswith("test") for x in names)
        assert {"frozen/AI_VERIFIED_NEW_GOLD_v1.jsonl", "frozen/NEW_GOLD_EVAL_ROWS.jsonl"} <= set(names)
        z.extractall(tmp_path / "colab3")
    with pytest.raises(FP3.PackageV3Error):
        FP3.build(world["root"], world["pid"], skill_card=world["old"], gold_jsonl=g, gold_manifest=m, out_dir=tmp_path / "pkg3")   # never overwritten
    run = CE.Run(work_root=tmp_path / "colab3" / "data" / "personas", profile_id=world["pid"], run_dir=tmp_path / "drive" / "v3", manifest=dict(man, code_dir=str(ROOT)),
                 cfg=EV3.defaults({"judge_concurrency": 2, "judge_per_second": 0}), gen=ModelDouble(), log=lambda s: None)
    fz = tmp_path / "colab3" / "frozen"
    seeded = EV3.seed_frozen(run, gold_jsonl=fz / "AI_VERIFIED_NEW_GOLD_v1.jsonl", gold_manifest=fz / "AI_VERIFIED_NEW_GOLD_MANIFEST.json",
                             eval_rows_jsonl=fz / "NEW_GOLD_EVAL_ROWS.jsonl")
    assert seeded["questions"] == n
    assert EV3.seed_frozen(run, gold_jsonl=fz / "AI_VERIFIED_NEW_GOLD_v1.jsonl", gold_manifest=fz / "AI_VERIFIED_NEW_GOLD_MANIFEST.json",
                           eval_rows_jsonl=fz / "NEW_GOLD_EVAL_ROWS.jsonl")["questions"] == n                  # idempotent
    api = MemoryAPI(str(run.pdir / "memory" / "memory.sqlite"), public_only=False, persona_names={run.profile_id: ["TA"]})
    LD.build_local(api, run.profile_id, Path(run.run_dir) / "dense_index", FakeEmbedder(), model=LE.CACHE_MODEL_NAME, dimension=16)
    api.close()
    dense, why = LD.open_local(Path(run.run_dir) / "dense_index", model=LE.CACHE_MODEL_NAME, dimension=16, query_encoder=LE.LocalQueryEncoder(FakeEmbedder()))
    assert dense is not None, why
    mem = EV3.stage_memory(run, dense=dense)
    assert set(mem) == set(EV3.MEMORY_SPECS_V3) and all(v == n for v in mem.values())
    EV3.stage_judge(run, ENV, transport=mock_transport())
    rep = EV3.analysis(run)
    assert rep["gold_source"] == "AI_VERIFIED" and rep["is_human_gold"] is False and set(rep["groups"]) == set(EV3.MEMORY_SPECS_V3)
    assert "BEST_STABLE_V2-V2L" in rep["paired"] and rep["decision"]["recommendation"]
    for gm in rep["groups"].values():
        assert {"answer_accuracy", "gold_evidence_recall", "false_abstention_rate", "unsupported_claim_rate", "attribution_temporal_errors",
                "input_tokens_reader", "reader_latency_p50_s", "by_category"} <= set(gm)
    out = EV3.stage_export(run, {"gpu": "cpu-test"})
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert "reports/ANALYSIS_V3.json" in names and not any(x.endswith(".sqlite") for x in names)
        assert sum(x.endswith("answers.jsonl") for x in names) == 5
