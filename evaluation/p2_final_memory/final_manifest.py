#!/usr/bin/env python3
"""Write ``FINAL_CONFIG_MANIFEST.json`` for the StillHere_Final closure from the REAL state: git HEAD / tree of the committed source, the
product / research configuration fingerprints, the P2 evaluation records and the test summary passed in. Run it AFTER the source commit
(the manifest names that commit; it is then committed on top, so there is no self-reference).

    python evaluation/p2_final_memory/final_manifest.py --tests "<summary>" --out research/p2_final_memory/final_audit/FINAL_CONFIG_MANIFEST.json
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "evaluation" / "lifebench_pilot"), str(ROOT / "evaluation" / "p2_final_memory")]

from p2eval import protocol as P  # noqa: E402
from stillhere.persona import final_config as FC  # noqa: E402
from stillhere.persona import memory_p2 as MP  # noqa: E402
from stillhere.persona import product_config as PC  # noqa: E402
from stillhere.providers import llm as PLM  # noqa: E402


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", required=True, help="the full CPU regression summary line, as run on this HEAD")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dirty = [l for l in git("status", "--porcelain").splitlines() if not l.endswith("research/lifebench_pilot.zip")]
    if dirty:
        raise SystemExit(f"refusing: the working tree has uncommitted changes, the manifest would not describe HEAD: {dirty[:5]}")
    audit = json.loads((ROOT / "research/p2_final_memory/final_audit/P2_FINAL_AUDIT_METRICS.json").read_text(encoding="utf-8"))
    np_ = audit["metrics"]["new_person"]
    prov = dataclasses.asdict(PLM.ProviderConfig())
    prov.pop("extra", None)
    man = {
        "schema": "stillhere.final_closure_manifest/1.0",
        "release": "StillHere Latest Product / Research Release (StillHere_Final closure, 2026-11-19)",
        "source_commit": git("rev-parse", "HEAD"), "source_tree_sha": git("rev-parse", "HEAD^{tree}"), "branch": git("branch", "--show-current"),
        "note_on_commit": "the manifest describes this commit; it is added in the NEXT commit (and tagged locally), never by editing this one",
        "product": {"version": PC.PRODUCT_VERSION, "memory_config_sha256": PC.fingerprint(), "memory_pipeline": PC.PRODUCT_MEMORY_PIPELINE,
                    "rollback": {"choice": "product-v2d", "version": PC.PRODUCT_V2D_VERSION, "config_sha256": PC.fingerprint(PC.PRODUCT_V2D_CONFIG)},
                    "memory": "V2D + P2 (dense unavailable -> V2L + P2)"},
        "p2": {"version": MP.P2_VERSION, "config": dataclasses.asdict(MP.P2Config()), "module_sha256": sha(ROOT / "src/stillhere/persona/memory_p2.py"),
               "decision": audit["decision"]["registered_decision"], "primary_significant": audit["decision"]["primary_significant"],
               "rule_checks": audit["decision"]["rule"],
               "primary": np_["answerable_pair"], "gold_recall_pair": np_["recall_pair"],
               "product_scope": "P2-A retrieval only; the evaluation's calendar note, coverage note and p2 answer prompt are not in Reader p1"},
        "reader": {"version": PC.PRODUCT_MEMORY_PIPELINE["reader_version"], "gates": ["attribution", "grounding", "answer preservation"]},
        "writer_mode": PC.PRODUCT_MEMORY_PIPELINE["writer_mode"],
        "research_freeze_unchanged": {"version": FC.FINAL_VERSION, "config_sha256": FC.fingerprint(), "memory": "V2L", "source_commit": "855448c",
                                      "freeze_manifest_commit": "abf19fb"},
        "base": FC.FINAL_CONFIG["base"], "embedding": PC.EMBEDDING,
        "adapter_registry": FC.FINAL_CONFIG["adapter"], "persona_skill": {"default": "ON (the profile's current card)", "card_sources": ["AUTO", "MANUAL", "LLM_ASSISTED"],
                                                                           "research_card": FC.FINAL_CONFIG["persona_skill"]},
        "provider_default": prov,
        "evaluation": {"dataset": P.DATASET, "protocol_sha256": P.P2_PROTOCOL_SHA256,
                       "input_zip_sha256": audit["checks"]["input_package"]["zip_sha256"], "input_package_commit": audit["checks"]["input_package"]["git_commit"],
                       "results_zip_sha256": audit["checks"]["zip_sha256"], "integrity": "PASS",
                       "fresh_person": audit["checks"]["new_person"], "fresh_qa_sha256": audit["checks"]["frozen_new_qa_sha256"]["manifest"],
                       "known_qa_sha256": audit["checks"]["frozen_known_qa_sha256"]["manifest"],
                       "pilot_lifebench_mem0": "于晓薇 100 QA: M0 18, V2L 63, V2D 63, Mem0 OSS 62 (previous round, unchanged)"},
        "logo": {"path": "docs/logo.png", "sha256": sha(ROOT / "docs/logo.png")},
        "tests": a.tests,
        "known_limitations": [
            "P2 primary improvement not statistically significant (n = 83 answerable, CI crosses 0)",
            "one fresh LifeBench person; answer and judge are the same model family (DeepSeek Flash)",
            "known-set gains are optimistic (P2 designed on those errors): regression only",
            "wrong attribution / wrong temporal claims UNAVAILABLE (no such judge labels)",
            "the product's P2 QA gain is not separately validated (benchmark bypasses Reader p1 and gates)",
            "unsupported answers +2 on the fresh person (within the registered tolerance)",
            "no human gold; one real persona for the personalisation research",
            "PUBLIC_PUSH_BLOCKED_BY_PRIVATE_GIT_HISTORY (commit bcc1720 holds real private DEV questions)"],
        "push": False,
    }
    Path(a.out).write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: man[k] for k in ("source_commit", "source_tree_sha")} | {"memory_config_sha256": man["product"]["memory_config_sha256"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
