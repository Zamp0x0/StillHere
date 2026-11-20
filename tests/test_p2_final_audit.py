"""The P2 final audit's own statistics (independent of p2eval.metrics_p2): exact McNemar, paired bootstrap, percentiles, the paired
comparison and the per-question majority / abstention fields, on synthetic records."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("src", "evaluation/lifebench_pilot", "evaluation/p2_final_memory"):
    sys.path.insert(0, str(ROOT / p))
spec = importlib.util.spec_from_file_location("p2_final_audit", ROOT / "evaluation" / "p2_final_memory" / "final_audit.py")
FA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FA)


def test_statistics():
    assert FA.mcnemar_exact(0, 0) == 1.0 and abs(FA.mcnemar_exact(8, 5) - 0.5811) < 1e-3 and abs(FA.mcnemar_exact(15, 2) - 0.0023) < 1e-3
    assert FA.pct([1, 2, 3, 4], 0.5) == 2.5 and FA.pct([], 0.5) is None
    lo, hi = FA.boot([1] * 10 + [0] * 80 + [-1] * 3)
    assert lo <= 7 / 93 <= hi


def test_per_question_majority_abstention_and_pairing():
    qs = [{"question_id": f"q{i}", "question_type": ["Single_hop"] + (["Unanswerable"] if i == 3 else []), "evidence": ["d1"]} for i in range(4)]

    def recs(group, votes, answers):
        ctx = {(group, f"q{i}"): {"kept": [{"ids": ["2025-01-01_d1:t1"]}], "tokens": 100, "retrieval_ms": 5.0, "future_messages_in_store": 0,
                                 "ask_time": "2025-01-02", "text": "", "meta": {}} for i in range(4)}
        ans = {(group, f"q{i}"): {"status": "OK", "answer": answers[i], "abstained": str("无法确定" in answers[i])} for i in range(4)}
        jud = {(group, f"q{i}", r): {"status": "OK", "is_correct": r < votes[i]} for i in range(4) for r in range(3)}
        return FA.per_question(qs, ctx, ans, jud, lambda q: (group, q))
    old = recs("O", [3, 1, 0, 3], ["a", "b", "根据现有记录无法确定", "根据现有记录无法确定"])
    new = recs("N", [3, 2, 2, 3], ["a", "b", "c", "根据现有记录无法确定"])
    assert [r["correct"] for r in old] == [True, False, False, True] and [r["correct"] for r in new] == [True, True, True, True]
    assert old[2]["abstained"] and old[2]["abstain_recomputed"] and old[0]["recall"] == 1.0
    s = FA.summarise(old)
    assert s["false_abstention"] == {"count": 1, "n": 3} and s["correct_abstention"] == {"count": 1, "n": 1} and s["unsupported"]["count"] == 1
    p, flips = FA.pair(new, old, True)
    assert (p["n"], p["wins"], p["losses"]) == (3, 2, 0) and flips["win"] == ["q1", "q2"]
