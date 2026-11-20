"""Answer pilot, analysis: from one row per (question, arm) to accuracy, paired comparisons and failure attribution.

Each row is a plain dict built by the script from saved files. Fields: qa_id, arm_id, persona_id, primary_type, unanswerable, label (CORRECT, WRONG or MISSING),
votes, abstain, keys (records in the context), gold (official keys, empty for an unanswerable question), coverage, full, grounding and sufficiency (None when not judged).
Failure categories overlap on purpose: one wrong answer can belong to several.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .answer_judge import paired_compare, point_biserial

PAIRS = {"FAST_MINUS_LEGACY": ("A1", "A0"), "MULTIHOP_MINUS_FAST": ("A2", "A1"), "QUALITY_MINUS_FAST": ("A3", "A1"), "QUALITY_MINUS_MULTIHOP": ("A3", "A2")}


def by_arm(rows: Iterable[Mapping]) -> Dict[str, Dict[str, Mapping]]:
    out: Dict[str, Dict[str, Mapping]] = {}
    for r in rows:
        out.setdefault(r["arm_id"], {})[r["qa_id"]] = r
    return out


def _outcome(r: Mapping) -> Optional[bool]:
    return None if r["label"] == "MISSING" else r["label"] == "CORRECT"


def _rate(rows: Sequence[Mapping]) -> dict:
    correct = sum(r["label"] == "CORRECT" for r in rows)
    wrong = sum(r["label"] == "WRONG" for r in rows)
    missing = sum(r["label"] == "MISSING" for r in rows)
    return {"total": len(rows), "correct": correct, "wrong": wrong, "missing": missing, "accuracy": round(correct / len(rows), 4) if rows else None}


def accuracy(rows: Sequence[Mapping]) -> dict:
    out = {}
    for arm, group in sorted(by_arm(rows).items()):
        scorable = [r for r in group.values() if not r["unanswerable"]]
        types: Dict[str, List[Mapping]] = {}
        people: Dict[str, List[Mapping]] = {}
        for r in scorable:
            types.setdefault(r["primary_type"], []).append(r)
            people.setdefault(r["persona_id"], []).append(r)
        weights = [r["weighted_score"] for r in scorable if r.get("weighted_score") is not None]
        out[arm] = {**_rate(scorable), "all_questions": _rate(list(group.values())), "mean_weighted_score": round(sum(weights) / len(weights), 4) if weights else None, "unparsable_output": sum(not r.get("valid_json", True) for r in group.values()),
                    "by_type": {k: _rate(v) for k, v in sorted(types.items())}, "by_persona": {k: _rate(v) for k, v in sorted(people.items())}}
    return out


def paired(rows: Sequence[Mapping]) -> dict:
    """On answerable questions, using only questions both arms finished."""
    arms = by_arm(rows)
    outcomes = {a: {q: _outcome(r) for q, r in group.items() if not r["unanswerable"]} for a, group in arms.items()}
    return {name: paired_compare(outcomes[x], outcomes[y]) for name, (x, y) in PAIRS.items() if x in outcomes and y in outcomes}


def abstention(rows: Sequence[Mapping]) -> dict:
    out = {}
    for arm, group in sorted(by_arm(rows).items()):
        no_answer = [r for r in group.values() if r["unanswerable"]]
        answerable = [r for r in group.values() if not r["unanswerable"]]
        refused = [r for r in answerable if r["abstain"]]
        out[arm] = {"unanswerable": len(no_answer), "correct_abstention": sum(r["label"] == "CORRECT" for r in no_answer), "incorrect_answer": sum(r["label"] == "WRONG" for r in no_answer), "missing": sum(r["label"] == "MISSING" for r in no_answer),
                    "said_it_could_not_tell": sum(r["abstain"] for r in no_answer), "answerable_abstained": len(refused), "unnecessary_abstention": sum((r.get("sufficiency") or {}).get("sufficient") == "yes" for r in refused),
                    "abstained_without_enough_evidence": sum((r.get("sufficiency") or {}).get("sufficient") in ("no", "partial") for r in refused)}
    return out


def grounding(rows: Sequence[Mapping]) -> dict:
    out = {}
    for arm, group in sorted(by_arm(rows).items()):
        items = list(group.values())
        cited = sum(r["citation_check"]["cited"] for r in items)
        valid = sum(r["citation_check"]["valid"] for r in items)
        answered = [r for r in items if not r["abstain"]]
        judged = [r for r in answered if r.get("grounding")]
        no_citation = sum(1 for r in answered if not r["citation_check"]["cited"])
        share = lambda test: round(sum(1 for r in judged if test(r["grounding"])) / len(judged), 4) if judged else None
        out[arm] = {"citations": cited, "valid_citations": valid, "citation_validity": round(valid / cited, 4) if cited else None, "answers": len(answered), "answers_without_citation": no_citation, "judged": len(judged),
                    "supported_answer_rate": share(lambda g: g.get("support") == "full"), "partly_supported_rate": share(lambda g: g.get("support") == "partial"), "unsupported_inference_rate": share(lambda g: g.get("unsupported_personal_fact") is True or g.get("unsupported_causal") is True),
                    "temporal_answer_errors": sum(1 for r in judged if r["grounding"].get("temporal_misuse") is True)}
    return out


def _difference(new: Mapping, old: Mapping) -> dict:
    added = [k for k in new["keys"] if k not in old["keys"]]
    lost = [k for k in old["keys"] if k not in new["keys"]]
    gold = set(new["gold"])
    return {"added": added, "displaced": lost, "added_gold": [k for k in added if k in gold], "displaced_gold": [k for k in lost if k in gold]}


def attribution(rows: Sequence[Mapping]) -> dict:
    """F1, F5 to F9 per arm, and F2 to F4 for each comparison. Links between evidence and outcome are associations from one run, not causes."""
    arms = by_arm(rows)
    per_arm = {}
    for arm, group in sorted(arms.items()):
        cats: Dict[str, List[str]] = {k: [] for k in ("F1_retrieval_missing", "F5_reasoning_failed", "F6_unsupported_inference", "F7_temporal_error", "F8_abstention_error", "F9_judge_uncertain")}
        unknown = 0
        for q, r in group.items():
            wrong = r["label"] == "WRONG"
            enough = (r.get("sufficiency") or {}).get("sufficient")
            g = r.get("grounding") or {}
            if wrong and not r["unanswerable"]:
                if enough == "yes":
                    cats["F5_reasoning_failed"].append(q)
                elif enough in ("no", "partial"):
                    cats["F1_retrieval_missing"].append(q)
                else:
                    unknown += 1
            if g.get("unsupported_personal_fact") is True or g.get("unsupported_causal") is True:
                cats["F6_unsupported_inference"].append(q)
            if g.get("temporal_misuse") is True:
                cats["F7_temporal_error"].append(q)
            if (r["unanswerable"] and wrong) or (not r["unanswerable"] and r["abstain"] and enough == "yes"):
                cats["F8_abstention_error"].append(q)
            if r["label"] == "MISSING" or (r.get("votes") and min(r["votes"].values()) > 0):
                cats["F9_judge_uncertain"].append(q)
        per_arm[arm] = {"counts": {k: len(v) for k, v in cats.items()}, "questions": cats, "wrong_without_sufficiency_verdict": unknown}
    pairs = {}
    for name, (x, y) in PAIRS.items():
        if x not in arms or y not in arms:
            continue
        cats = {"F2_evidence_displacement": [], "F3_evidence_added_answer_improved": [], "F4_evidence_added_answer_unchanged": []}
        for q in sorted(set(arms[x]) & set(arms[y])):
            new, old = arms[x][q], arms[y][q]
            if new["unanswerable"] or "MISSING" in (new["label"], old["label"]):
                continue
            d = _difference(new, old)
            if d["displaced_gold"] and old["label"] == "CORRECT" and new["label"] == "WRONG":
                cats["F2_evidence_displacement"].append(q)
            if d["added_gold"]:
                changed = old["label"] == "WRONG" and new["label"] == "CORRECT"
                cats["F3_evidence_added_answer_improved" if changed else "F4_evidence_added_answer_unchanged"].append(q)
        pairs[name] = {"counts": {k: len(v) for k, v in cats.items()}, "questions": cats}
    return {"arms": per_arm, "pairs": pairs}


def swap_analysis(rows: Sequence[Mapping], new_arm: str, old_arm: str) -> dict:
    """What one arm changed relative to another, question by question, on answerable questions."""
    arms = by_arm(rows)
    table = []
    for q in sorted(set(arms.get(new_arm, {})) & set(arms.get(old_arm, {}))):
        new, old = arms[new_arm][q], arms[old_arm][q]
        if new["unanswerable"]:
            continue
        d = _difference(new, old)
        table.append({"qa_id": q, "primary_type": new["primary_type"], "before": old["label"], "after": new["label"], "added": len(d["added"]), "displaced": len(d["displaced"]), "added_gold": d["added_gold"], "displaced_gold": d["displaced_gold"],
                      "coverage_before": old["coverage"], "coverage_after": new["coverage"], "full_before": old["full"], "full_after": new["full"]})
    improved = [t for t in table if t["before"] == "WRONG" and t["after"] == "CORRECT"]
    worsened = [t for t in table if t["before"] == "CORRECT" and t["after"] == "WRONG"]
    types: Dict[str, Counter] = {}
    for t in table:
        c = types.setdefault(t["primary_type"], Counter())
        c["improved" if t in improved else "worsened" if t in worsened else "unchanged"] += 1
    full_gained = [t for t in table if t["full_after"] and not t["full_before"]]
    return {"questions": len(table), "evidence_added": sum(t["added"] for t in table), "evidence_displaced": sum(t["displaced"] for t in table), "gold_added": sum(len(t["added_gold"]) for t in table), "gold_displaced": sum(len(t["displaced_gold"]) for t in table),
            "wrong_to_correct": len(improved), "correct_to_wrong": len(worsened), "improved_with_added_gold": sum(bool(t["added_gold"]) for t in improved), "improved_with_added_any": sum(t["added"] > 0 for t in improved), "worsened_with_displaced_gold": sum(bool(t["displaced_gold"]) for t in worsened),
            "worsened_with_displaced_any": sum(t["displaced"] > 0 for t in worsened), "full_recall_gained": len(full_gained), "full_gained_and_improved": sum(t in improved for t in full_gained), "by_type": {k: dict(v) for k, v in sorted(types.items())}, "changed": improved + worsened}


def coverage_relation(rows: Sequence[Mapping]) -> dict:
    """Within each arm, and between arms: does more strict evidence go with correct answers? Questions differ in difficulty, so this is association only."""
    usable = [r for r in rows if not r["unanswerable"] and r["label"] != "MISSING"]
    out: Dict[str, dict] = {"pooled_questions": len(usable)}
    arms = by_arm(usable)
    for arm, group in sorted(arms.items()):
        out[arm] = {"coverage_vs_correct": point_biserial([r["coverage"] for r in group.values()], [r["label"] == "CORRECT" for r in group.values()])}
    bands = {"0": lambda c: c == 0, "(0,0.5)": lambda c: 0 < c < 0.5, "[0.5,1)": lambda c: 0.5 <= c < 1, "1": lambda c: c == 1}
    out["accuracy_by_coverage_band"] = {}
    for name, test in bands.items():
        inside = [r for r in usable if test(r["coverage"])]
        out["accuracy_by_coverage_band"][name] = {"answers": len(inside), "accuracy": round(sum(r["label"] == "CORRECT" for r in inside) / len(inside), 4) if inside else None}
    changes = []
    for a, b in (("A1", "A0"), ("A2", "A1"), ("A3", "A1")):
        if a in arms and b in arms:
            for q in set(arms[a]) & set(arms[b]):
                changes.append((arms[a][q]["coverage"] - arms[b][q]["coverage"], int(arms[a][q]["label"] == "CORRECT") - int(arms[b][q]["label"] == "CORRECT")))
    moved = [c for c in changes if c[0] != 0]
    out["same_question_coverage_up"] = Counter("improved" if d > 0 else "worsened" if d < 0 else "unchanged" for c, d in moved if c > 0)
    out["same_question_coverage_down"] = Counter("improved" if d > 0 else "worsened" if d < 0 else "unchanged" for c, d in moved if c < 0)
    return out
