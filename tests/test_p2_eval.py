"""P2 evaluation runner on CPU: both groups share one store, contexts frozen per group, prompts differ only as registered, resume with
no repeated requests, judge x 3 majority, metrics recomputed from records, the adoption rule (incl. NO_DECISION when incomplete), the
budget guard, protocol / fingerprint checks. Synthetic data + fake API; plumbing only."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in ("src", "evaluation/lifebench_pilot", "evaluation/p2_final_memory", "tests"):
    sys.path.insert(0, str(ROOT / p))

from lbpilot import context as CX  # noqa: E402
from lbpilot import judge as J  # noqa: E402
from lbpilot import select as LS  # noqa: E402
from lbpilot import stillhere_memory as SM  # noqa: E402
from p2eval import metrics_p2 as M  # noqa: E402
from p2eval import protocol as P  # noqa: E402
from p2eval import runner as R  # noqa: E402
from test_lifebench_pilot import FakeEmbedder, FakeEncoder, fake_answer_chat, fake_judge_chat, resp, synthetic  # noqa: E402

OFFICIAL = ROOT / "private" / "external" / "LifeBench_eval"


def test_protocol_is_frozen_and_the_old_path_is_fingerprinted():
    assert len(P.P2_PROTOCOL_SHA256) == 64 and P.P2_PROTOCOL_SHA256 == P.sha(P.protocol())
    assert "evaluation/lifebench_pilot/lbpilot/stillhere_memory.py" in P.OLD_FINGERPRINT_FILES
    assert "src/stillhere/persona/memory_v2.py" in P.OLD_FINGERPRINT_FILES
    assert P.ANSWER["model"] == "deepseek-flash" and P.JUDGE["repeats"] == 3 and P.EVIDENCE["token_budget"] == 4000
    import dataclasses
    from stillhere.persona import memory_p2 as MP
    cfg = {k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(MP.P2Config()).items()}
    assert P.P2["config"] == cfg and P.P2["version"] == MP.P2_VERSION                 # the registered P2 is the code's P2


def test_adoption_rule():
    def g(**kw):
        base = {"complete": True, "answerable": {"correct": 45}, "unsupported": {"count": 10}, "unanswerable": {"correct": 17}, "future_leaks": 0,
                "multi_hop": {"correct": 12}, "temporal": {"correct": 2}, "false_abstention": {"count": 28}, "micro": {"correct": 62},
                "evidence_tokens_median": 2500, "retrieval_ms_p50": 400}
        for k, v in kw.items():
            base[k] = v
        return base
    old = g()
    assert M.adoption(old, g(answerable={"correct": 50}, multi_hop={"correct": 16}, micro={"correct": 67}), {"significant": False})["decision"] == "ADOPT_P2"
    assert M.adoption(old, g(answerable={"correct": 44}), {})["decision"] == "KEEP_OLD"
    assert M.adoption(old, g(unanswerable={"correct": 15}, multi_hop={"correct": 20}, micro={"correct": 70}, answerable={"correct": 52}), {})["decision"] == "KEEP_OLD"
    assert M.adoption(old, g(complete=False), {})["decision"] == "NO_DECISION_INCOMPLETE"
    assert M.adoption(old, g(retrieval_ms_p50=5000, multi_hop={"correct": 16}, micro={"correct": 66}, answerable={"correct": 49}), {})["decision"] == "KEEP_OLD"


@pytest.mark.skipif(not (OFFICIAL / "src" / "evaluators" / "llm_judge.py").is_file(), reason="official checkout not present")
def test_runner_end_to_end_with_resume_and_budget(tmp_path, monkeypatch):
    data = synthetic(days=40)
    fn = LS.freeze_data(data)
    person, qs = LS.selected(data, fn)
    run = R.Run(tmp_path / "drive", tmp_path / "scratch", log=lambda s: None)
    inN = run.set_inputs("NEWP", person, qs, "linxiaoce")
    inK = run.set_inputs("KNOWN", person, qs, "linxiaoce")
    dense = {}
    for exp, inp in (("NEWP", inN), ("KNOWN", inK)):
        SM.build_dense(tmp_path / f"ix_{exp}.sqlite", tmp_path / f"dense_{exp}", inp["events"], inp["slug"], inp["name"], inp["assistant"], FakeEmbedder(),
                       dataset_sha256="0" * 64, locator="s")
        dense[exp], _ = SM.open_dense(tmp_path / f"dense_{exp}", FakeEncoder())
        run.restore_if_needed(exp)
        st = SM.StillHereMemory(run.live(exp) / "run.sqlite", inp["slug"], inp["name"], inp["assistant"], dataset_sha256="0" * 64, locator="s")
        r = run.chronological(exp, stillhere=st, dense=dense[exp], count=CX.estimate_tokens)
        st.close()
        assert r["status"] == R.PASS
    ctx = run.contexts.done()
    assert len(ctx) == 300 and all(c["future_messages_in_store"] == 0 and c["tokens"] <= 4000 for c in ctx.values())
    newk = [c for k, c in ctx.items() if k[1] == "NEW_P2_V2D"]
    assert any(c["text"].startswith("【证据覆盖】") for c in newk)
    prompts = set()

    def chat(msgs):
        prompts.add(msgs[0]["content"][-30:])
        return fake_answer_chat(msgs)
    a = run.run_answers(chat)
    assert a["status"] == R.PASS and len(prompts) == 2                            # exactly two registered prompts
    n = len(run.answers.rows())
    run.run_answers(chat)
    assert len(run.answers.rows()) == n                                            # resumed: nothing requested again
    official = J.Official(OFFICIAL)
    j = run.run_judge(official, fake_judge_chat)
    assert j["status"] == R.PASS and all(v == 300 for v in j["per_group"].values())
    known_old = {"answers": {("M2_STILLHERE_V2D", q["question_id"]): {"status": "OK", "abstained": False} for q in qs},
                 "contexts": {("M2_STILLHERE_V2D", q["question_id"]): {"kept": [], "tokens": 1000} for q in qs},
                 "judgments": {("M2_STILLHERE_V2D", q["question_id"], r): {"status": "OK", "is_correct": False} for q in qs for r in range(3)}}
    out = M.compute(run.inputs, run.answers.done(), run.judgments.done(), run.contexts.done(), known_old=known_old)
    np_ = out["new_person"]
    assert np_["OLD_V2D"]["complete"] and np_["NEW_P2_V2D"]["complete"] and np_["adoption"]["decision"] in ("ADOPT_P2", "KEEP_OLD")
    assert out["known_set"]["NEW_P2_V2D"]["judged"] == 100 and out["known_set"]["answerable_pair"]["n"] > 0
    monkeypatch.setattr(P, "COST_BUDGET_USD", 0.0)
    with pytest.raises(R.BudgetExceeded):
        run.guard()
