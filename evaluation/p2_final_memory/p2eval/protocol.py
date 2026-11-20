"""The frozen P2 evaluation protocol (registered before any NEW answer exists; ``P2_PROTOCOL_SHA256`` covers every field)."""
from __future__ import annotations

import hashlib
import json

from lbpilot import protocol as LP

P2_EVAL_VERSION = "stillhere.p2_final_memory_eval/1.0"
DATASET = LP.DATASET
OFFICIAL_EVAL = LP.OFFICIAL_EVAL

# Every LifeBench person has had some question ids referenced by earlier StillHere research (research/lifebench_improvement/phase1:
# person_exposure.json; independently re-scanned 2026-11-19). The earlier phase-1 plan reserved 于晓雯 / 宋雅静 / 尹浩 / 雷明轩 as
# FINAL_TEST ("run once on the final version") and none of them has been run since. Rule: the first FINAL_TEST person in file order
# with >= 100 UNREFERENCED questions covering the 9 types; referenced question ids are excluded from the sample.
NEW_PERSON_RULE = ("first person in file order among the phase-1 FINAL_TEST persons (于晓雯, 宋雅静, 尹浩, 雷明轩) with >= 100 questions never "
                   "referenced by earlier StillHere research; referenced ids excluded; never 于晓薇 / 叶明轩 / 孙雨薇")
FINAL_TEST_PERSONS = ("于晓雯", "宋雅静", "尹浩", "雷明轩")
NEVER = ("于晓薇", "叶明轩", "孙雨薇")
VALIDITY = "PERSONA_SEEN_QA_UNSEEN (no LifeBench person is fully unseen by earlier StillHere research)"
SAMPLE = {"n": 100, "seed": "stillhere-p2-final-2026-11-19", "stratum": "question_type[0]",
          "allocation": "proportional, largest remainder, >= 1 per stratum", "order_within_stratum": "sha256(seed|question_id)"}

GROUPS_NEW_PERSON = {"OLD_V2D": "lbpilot V2D path as run in the pilot (unchanged files, fingerprinted) + answer prompt lbpilot-answer-v1",
                     "NEW_P2_V2D": "same V2D first hop + P2 requirements / second pass / temporal view / coverage note + answer prompt lbpilot-answer-p2"}
GROUPS_KNOWN_SET = {"NEW_P2_V2D": "re-run on the 于晓薇 pilot 100 QA; compared with the STORED pilot M2_STILLHERE_V2D records (KNOWN_SET_REGRESSION)"}
ANSWER = dict(LP.ANSWER, prompt_version_old="lbpilot-answer-v1", prompt_version_new="lbpilot-answer-p2")
JUDGE = LP.JUDGE
EVIDENCE = LP.EVIDENCE                  # 4,000 tokens for both groups (the NEW coverage note counts inside the budget)
STILLHERE = LP.STILLHERE
P2 = {"version": "stillhere.memory.p2/2026-11-19.2", "config": {"max_requirements": 6, "sub_k": 3, "window_k": 5, "neighbors": [1, 1], "pool": 200,
                                                                "recent_days": 30, "advisory_days": 21, "span_days": 7, "few_days": 7,
                                                                "item_pool": 40, "item_lead": 3},
      "llm_calls_in_retrieval": 0}
PRIMARY = {"metric": "answerable grounded accuracy (majority-of-3 CORRECT on non-Unanswerable questions)", "comparison": "NEW_P2_V2D - OLD_V2D",
           "set": "the fresh person's 100 QA"}
SECONDARY = ["micro accuracy", "multi-hop accuracy", "temporal accuracy", "knowledge-update accuracy", "false abstention on answerable",
             "unsupported claims (wrong and not abstained)", "unanswerable accuracy (safety: abstention kept)", "strict gold-record recall in the context",
             "evidence tokens", "retrieval latency", "API cost"]
ADOPTION_RULE = {
    "A": "both groups complete (100 contexts, 100 answers, 300 judge records each)",
    "B": "NEW answerable accuracy >= OLD answerable accuracy (point estimate)",
    "C": "NEW unsupported claims <= OLD unsupported claims + 2",
    "D": "NEW unanswerable accuracy >= OLD unanswerable accuracy - 1 question (no loss of abstention safety)",
    "E": "0 temporal leakage in both groups",
    "F": "at least one of: multi-hop accuracy +, temporal accuracy +, false abstention - ; and micro accuracy not lower",
    "G": "NEW median evidence tokens <= 4,000 and NEW retrieval p50 <= 3 x OLD",
    "decision": "ADOPT_P2 iff A-G all hold; otherwise KEEP_OLD. 'Significant' only if the paired bootstrap 95% CI of the primary difference excludes 0",
}
BOOTSTRAP = {"resamples": 10000, "seed": 20261119, "unit": "question (paired)"}
COMPLETENESS = {"answers_per_group": 100, "judge_records_per_group": 300}
OLD_FINGERPRINT_FILES = (
    "src/stillhere/persona/memory_v2.py", "src/stillhere/persona/memory_store.py", "src/stillhere/persona/product_config.py",
    "src/stillhere/persona/final_config.py", "src/stillhere/persona/routing.py",
    "src/stillhere/context/fact_pack.py", "src/stillhere/memory/api.py", "src/stillhere/memory/sqlite_store.py", "src/stillhere/memory/labels.py",
    "src/stillhere/retrieval/ef1/pipeline.py", "src/stillhere/retrieval/ef1/channels.py", "src/stillhere/retrieval/ef1/fusion.py",
    "src/stillhere/retrieval/ef1/query.py", "src/stillhere/retrieval/ef1/temporal.py", "src/stillhere/retrieval/ef1/dense_index.py",
    "src/stillhere/retrieval/ef1/local_dense.py", "src/stillhere/retrieval/ef1/shape.py", "src/stillhere/retrieval/dense.py",
    "src/stillhere/retrieval/local_embedding.py",
    "evaluation/lifebench_pilot/lbpilot/stillhere_memory.py", "evaluation/lifebench_pilot/lbpilot/context.py", "evaluation/lifebench_pilot/lbpilot/answer.py",
    "evaluation/lifebench_pilot/lbpilot/judge.py", "evaluation/lifebench_pilot/lbpilot/timeline.py", "evaluation/lifebench_pilot/lbpilot/select.py",
    "evaluation/lifebench_pilot/lbpilot/protocol.py", "evaluation/lifebench_pilot/lbpilot/records.py", "evaluation/lifebench_pilot/lbpilot/metrics.py")
OLD_SOURCE_COMMIT = "a95cf6a"          # the final-audit commit; these files are byte-identical to the pilot's run package (06f63ac)
PILOT_RESULTS_ZIP_SHA256 = "8de071057960968b9f800ec82a7125a8b10868f71ea688be1ee5403ee11b1a2d"
COST_BUDGET_USD = 6.0                  # hard stop for the run (answers + judge, peak-price estimate)


def protocol() -> dict:
    return {"version": P2_EVAL_VERSION, "dataset": DATASET, "official_eval": OFFICIAL_EVAL, "new_person_rule": NEW_PERSON_RULE,
            "final_test_persons": list(FINAL_TEST_PERSONS), "never": list(NEVER), "validity": VALIDITY, "sample": SAMPLE,
            "groups_new_person": GROUPS_NEW_PERSON, "groups_known_set": GROUPS_KNOWN_SET, "answer": ANSWER, "judge": JUDGE, "evidence": EVIDENCE,
            "stillhere": STILLHERE, "p2": P2, "primary": PRIMARY, "secondary": SECONDARY, "adoption_rule": ADOPTION_RULE, "bootstrap": BOOTSTRAP,
            "completeness": COMPLETENESS, "old_fingerprint_files": list(OLD_FINGERPRINT_FILES), "old_source_commit": OLD_SOURCE_COMMIT,
            "pilot_results_zip_sha256": PILOT_RESULTS_ZIP_SHA256, "cost_budget_usd": COST_BUDGET_USD}


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


P2_PROTOCOL_SHA256 = sha(protocol())
