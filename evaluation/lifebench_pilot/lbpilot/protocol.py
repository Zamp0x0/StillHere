"""The frozen protocol of the pilot (registered before any answer is generated; ``PROTOCOL_SHA256`` covers every field below).

Scope: ONE_PERSONA_100QA_PILOT. Not the official full LifeBench 2.0 ranking; scores are not comparable to the official leaderboard
(different answer prompt, one persona, 100 questions, our own judge client).
"""
from __future__ import annotations

import hashlib
import json

PILOT_VERSION = "stillhere.lifebench_pilot/1.0"
SCOPE = "ONE_PERSONA_100QA_PILOT"

DATASET = {
    "repo": "C1754955896/Lifebenchv2.0", "revision": "3d06052d5ef2d94d9042bd2845787bfd37ba925d",
    "file": "locomo_format/lifebench_locomo_conversation_format_v2.0_3380QA.json",
    "sha256": "67eb8678fe36953de18c0aa97908259599023827de543c17a4b20c42b08ca4ee", "bytes": 27088496, "licence": "Apache-2.0",
}
DATASET["url"] = f"https://huggingface.co/datasets/{DATASET['repo']}/resolve/{DATASET['revision']}/{DATASET['file']}"

OFFICIAL_EVAL = {
    "repo": "https://github.com/1754955896/LifeBench_eval.git", "commit": "2bfda54723c3b35feac16f12ede2bfd73ddd2be9",
    "judge_module": "src/evaluators/llm_judge.py", "licence": "none stated (not redistributed: cloned at the pinned commit in Colab)",
    "time_rule": "date order: a date's events are ADDed before that date's questions are searched (src/pipeline/runner.py)",
}

# Person: an objective rule fixed before selection. 叶明轩 and 孙雨薇 were used by earlier StillHere LifeBench runs (2P release, EF1
# DEV200), so they are excluded to keep this pilot free of any earlier tuning on the same person.
PERSON_RULE = ("first sample in file order with >= 100 QA, all 9 official question types present, and not used by an earlier StillHere "
               "LifeBench run")
EXCLUDED_PERSONS = ("叶明轩", "孙雨薇")
OFFICIAL_TYPES = ("Single_hop", "Multi_hop", "Temporal", "Pattern_recognition(Non-declarative)", "Knowledge_update", "Causal", "Conflict",
                  "Hidden_info", "Unanswerable")
SAMPLE = {"n": 100, "seed": "stillhere-lifebench-pilot-2026-11-19", "stratum": "question_type[0] (the first listed type)",
          "allocation": "proportional to the person's QA per stratum, largest remainder, at least 1 per stratum",
          "order_within_stratum": "sha256(seed|question_id) ascending", "never_by_model_result": True}

GROUPS = {
    "M0_NO_MEMORY": {"memory": "none", "adapter": "OFF"},
    "M1_STILLHERE_V2L": {"memory": "StillHere V2 lexical (EF1 FTS5 bigram BM25 + time window, k = 8, evidence window +-2, date prefer)", "adapter": "OFF"},
    "M2_STILLHERE_V2D": {"memory": "StillHere V2 hybrid (V2L + local Qwen3-Embedding-0.6B dense, weighted RRF)", "adapter": "OFF"},
    "M3_MEM0_OSS": {"memory": "official Mem0 OSS (mem0ai, Memory.add infer=True / Memory.search)", "adapter": "N/A"},
}
GROUP_ORDER = tuple(GROUPS)
PAIRS = (("M1_STILLHERE_V2L", "M0_NO_MEMORY"), ("M2_STILLHERE_V2D", "M0_NO_MEMORY"), ("M3_MEM0_OSS", "M0_NO_MEMORY"),
         ("M2_STILLHERE_V2D", "M1_STILLHERE_V2L"), ("M2_STILLHERE_V2D", "M3_MEM0_OSS"), ("M1_STILLHERE_V2L", "M3_MEM0_OSS"))
PRIMARY_PAIR = ("M2_STILLHERE_V2D", "M3_MEM0_OSS")

STILLHERE = {"config": "stillhere.persona.product/2026-11-19-v2d memory_pipeline (k 8, neighbors [2, 2], date_mode prefer, strip_qform false)",
             "embedding": {"id": "Qwen/Qwen3-Embedding-0.6B", "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", "dimension": 1024,
                           "query_instruction": "unchanged StillHere instruction (retrieval.local_embedding.QUERY_TASK)"},
             "always_retrieve": "every question goes through the fact retrieval (pack_v2); the chat router is recorded, not applied",
             "reader": "NOT used: every group answers with the one shared Answer function"}
MEM0 = {"package": "mem0ai", "version": "2.2.1", "install": "pip install mem0ai==2.2.1 qdrant-client sentence-transformers (no [nlp] extra)",
        "llm": {"provider": "deepseek", "model": "deepseek-flash", "temperature": 0.1, "max_tokens": 32768,
                "route": "deepseek_base_url -> local counting proxy -> https://api.deepseek.com (model enforced, thinking disabled)"},
        "embedder": {"provider": "huggingface", "model": "Qwen/Qwen3-Embedding-0.6B", "embedding_dims": 1024,
                     "model_kwargs": {"revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"}},
        "vector_store": {"provider": "qdrant", "mode": "local path on /content (never on Drive)", "on_disk": True},
        "add": {"batch_size": 10, "role": "user", "infer": True, "rendering": "LifeBench_eval Mem0Adapter._render_message_content (date prefix + speaker)",
                "metadata": "session_date, session_iso, session_id, dia_ids", "max_attempts": 5},
        "search": {"top_k": 40, "threshold": "mem0 default (0.1)", "rerank": False, "filters": "user_id = the persona"},
        "telemetry": "MEM0_TELEMETRY=False"}
SEARCH_QUERY = "the official question with its leading （提问时间：…） prefix removed; the same text for every memory system; ask time passed structurally"
EVIDENCE = {"token_budget": 4000, "tokenizer": "Qwen3-Embedding-0.6B tokenizer (Qwen2 BPE) on Colab; char estimate in CPU tests",
            "truncation": "drop the lowest-ranked entries first until the rendered context fits; render the kept entries oldest first",
            "format": "[n] YYYY-MM-DD｜source｜text (one line block per entry, same for every system)"}
ANSWER = {"provider": "deepseek_api", "base_url": "https://api.deepseek.com", "model": "deepseek-flash", "temperature": 0.0, "max_tokens": 1024,
          "thinking": "disabled", "prompt_version": "lbpilot-answer-v1", "repeats": 1, "max_attempts": 4, "concurrency": 4}
JUDGE = {"provider": "deepseek_api", "model": "deepseek-flash", "temperature": 0.0, "thinking": "disabled", "repeats": 3,
         "prompt": "official SYSTEM_PROMPT + USER_PROMPT_TEMPLATE of LifeBench_eval@2bfda547 src/evaluators/llm_judge.py (verified by sha256)",
         "aggregation": "official: CORRECT iff correct_votes * 2 >= runs; point hits by the same majority; weighted score = hit weight / total",
         "parse": "official LLMJudge._parse_robust; an unparseable reply is retried (at most 3 attempts) and else recorded PARSE_FAIL",
         "concurrency": 6, "same_family_bias": "answer and judge are the same model family (DeepSeek Flash): reported as a limitation"}
COMPLETENESS = {"answers_per_group": 100, "judge_records_per_group": 300, "rule": "a group with fewer is INCOMPLETE: no pairwise conclusion"}
BOOTSTRAP = {"resamples": 10000, "seed": 20261119, "unit": "question (paired)"}


def protocol() -> dict:
    return {"version": PILOT_VERSION, "scope": SCOPE, "dataset": DATASET, "official_eval": OFFICIAL_EVAL, "person_rule": PERSON_RULE,
            "excluded_persons": list(EXCLUDED_PERSONS), "official_types": list(OFFICIAL_TYPES), "sample": SAMPLE, "groups": GROUPS,
            "pairs": [list(p) for p in PAIRS], "primary_pair": list(PRIMARY_PAIR), "stillhere": STILLHERE, "mem0": MEM0, "search_query": SEARCH_QUERY,
            "evidence": EVIDENCE, "answer": ANSWER, "judge": JUDGE, "completeness": COMPLETENESS, "bootstrap": BOOTSTRAP}


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


PROTOCOL_SHA256 = sha(protocol())
