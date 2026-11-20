"""D4 evaluation jobs (pre-registered, built deterministically from the frozen evaluation files; no prompt is rendered here).

Every prompt is the pre-rendered one stored in the frozen data (``system`` / ``user``, ``base_length``, ``ablations``), so Base and Adapter see
byte-identical inputs. A logical job is (set, episode, variant, model, seed); a GENERATION key is (model, seed, prompt sha256), so a prompt is
generated once per model and seed and fanned out to every job that needs it.

Plan (frozen in GEN_PLAN; the notebook never changes it):

    PROBE     DEV_PROBE, all items, FULL, base + every candidate checkpoint, seed 20261112          (checkpoint selection only)
    DEV       DEV_EVAL, all items: FULL (base, best), BASE_LENGTH (base);
              synthetic items: NO_MEMORY / NO_PROFILE / NO_EXAMPLES / SWAPPED_PERSONA / WRONG_MEMORY_STATE (base, best);
              STABILITY subset (first 120 synthetic DEV_EVAL items by episode_id): FULL with seeds 20261113 and 20261114 (base, best)
    SAFETY    SAFETY_EVAL, FULL (base, best)
    HOLDOUT   FROZEN_HOLDOUT synthetic, FULL (base, best), run once, after selection
    B3 HOLDOUT (real, 1,540 cases) is not in any package: DEFERRED (needs its own upload authorisation)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

GEN = {"model_repo": "Qwen/Qwen3-8B", "model_revision": "b968826d9c46dd6066d109eabc6255188de91218", "quantization": "nf4 double-quant, bf16 compute",
       "enable_thinking": False, "do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0,
       "repetition_penalty": 1.0, "max_new_tokens": 128, "stop": ["<|im_end|>", "<|endoftext|>"], "primary_seed": 20261112,
       "stability_seeds": [20261113, 20261114], "greedy": "diagnostic only, never used here"}
MAX_PROMPT_TOKENS = 1536 - 128 - 2
PRIMARY = GEN["primary_seed"]
ABLATIONS = ("NO_MEMORY", "NO_PROFILE", "NO_EXAMPLES", "SWAPPED_PERSONA", "WRONG_MEMORY_STATE")
STABILITY_N = 120
# package layout: PROBE / DEV are full records in the PRIVATE data package; SAFETY / HOLDOUT prompts are in the PUBLIC safety package
# (model-visible fields only, no raw restricted value) and their answer keys in the PRIVATE data package (keys/). load_set joins the two and
# refuses any prompt that differs (HASH_MISMATCH), so the model sees exactly the published prompts.
SET_FILES = {"PROBE": ("eval", "dev_probe.jsonl"), "DEV": ("eval", "dev_eval.jsonl"), "SAFETY": ("keys", "safety_eval.jsonl"),
             "HOLDOUT": ("keys", "frozen_holdout_synthetic.jsonl")}
PUBLIC_PROMPTS = {"SAFETY": "safety_eval_prompts.jsonl", "HOLDOUT": "frozen_holdout_synthetic_prompts.jsonl"}
PROMPT_FIELDS = ("system", "user", "base_length", "ablations")
PUBLIC_FIELDS = ("episode_id", "split", "persona_id", "source", "category", "family", "domain", "subject_type") + PROMPT_FIELDS
GEN_PLAN = {
    "PROBE": "DEV_PROBE all items FULL; base + candidate checkpoints; primary seed",
    "DEV": "DEV_EVAL FULL (base, best) + BASE_LENGTH (base) + 5 ablations on synthetic items (base, best) + stability seeds on 120 items",
    "SAFETY": "SAFETY_EVAL FULL (base, best)",
    "HOLDOUT": "FROZEN_HOLDOUT synthetic FULL (base, best), run once after selection",
    "B3_HOLDOUT": "DEFERRED (real data, not packaged)",
}
GROUP_OF = {("base", "FULL"): "A", ("model", "FULL"): "B", ("base", "BASE_LENGTH"): "BASE_LENGTH"}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(p: Path) -> List[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def prompt_sha(rec: dict) -> str:
    return sha(json.dumps({k: rec.get(k) for k in PROMPT_FIELDS}, ensure_ascii=False, sort_keys=True))


def public_prompt_record(rec: dict) -> dict:
    """The model-visible part of an eval record (what the public safety package holds)."""
    return {k: rec.get(k) for k in PUBLIC_FIELDS if rec.get(k) is not None}


def load_set(data_dir: Path, safety_dir: Path, name: str) -> List[dict]:
    sub, f = SET_FILES[name]
    rows = read_jsonl(Path(data_dir) / sub / f)
    if name in PUBLIC_PROMPTS:
        pub = {r["episode_id"]: r for r in read_jsonl(Path(safety_dir) / PUBLIC_PROMPTS[name])}
        bad = [r["episode_id"] for r in rows if r["episode_id"] not in pub or prompt_sha(pub[r["episode_id"]]) != prompt_sha(r)]
        if bad or len(pub) != len(rows):
            raise RuntimeError(f"HASH_MISMATCH: {name}: {len(bad)} answer-key records differ from the public prompts ({len(pub)} vs {len(rows)})")
    return rows


def prompt_of(item: dict, variant: str) -> Optional[dict]:
    if variant == "FULL":
        return {"system": item["system"], "user": item["user"]}
    if variant == "BASE_LENGTH":
        return item.get("base_length")
    return (item.get("ablations") or {}).get(variant)


def group_of(model: str, variant: str) -> str:
    if variant == "FULL":
        return "A" if model == "base" else "B"
    if variant == "BASE_LENGTH":
        return "BASE_LENGTH"
    return f"{'A' if model == 'base' else 'B'}:{variant}"


def job(set_name: str, item: dict, variant: str, model: str, seed: int, mock: bool = False) -> Optional[dict]:
    p = prompt_of(item, variant)
    if p is None:
        return None
    ps = sha(p["system"] + "\x1f" + p["user"])
    j = {"job_id": f"{set_name}|{item['episode_id']}|{variant}|{model}|{seed}", "set": set_name, "episode_id": item["episode_id"],
         "persona_id": item["persona_id"], "variant": variant, "model": model, "seed": seed, "group": group_of(model, variant),
         "messages": [{"role": "system", "content": p["system"]}, {"role": "user", "content": p["user"]}], "prompt_sha256": ps,
         "gen_key": f"{model}:{seed}:{ps}"}
    if mock:
        j["mock"] = {"reference": item.get("reference_target"), "leak": ((item.get("guard_spec") or {}).get("must_not_terms") or [None])[0]}
    return j


def stability_ids(dev_items: Sequence[dict]) -> List[str]:
    return sorted(e["episode_id"] for e in dev_items if e["source"] == "SYNTH")[:STABILITY_N]


def plan(set_name: str, items: Sequence[dict], models: Sequence[str], mock: bool = False) -> List[dict]:
    """Logical jobs of one set for the given model slots ("base", "best", "ckpt-N")."""
    out: List[dict] = []
    stab = set(stability_ids(items)) if set_name == "DEV" else set()
    for e in sorted(items, key=lambda x: x["episode_id"]):
        for m in models:
            out.append(job(set_name, e, "FULL", m, PRIMARY, mock))
            if set_name != "DEV":
                continue
            if m == "base":
                out.append(job(set_name, e, "BASE_LENGTH", m, PRIMARY, mock))
            if e["source"] == "SYNTH":
                out += [job(set_name, e, v, m, PRIMARY, mock) for v in ABLATIONS]
            if e["episode_id"] in stab:
                out += [job(set_name, e, "FULL", m, s, mock) for s in GEN["stability_seeds"]]
    return [j for j in out if j is not None]


def unique_generation_jobs(jobs: Iterable[dict]) -> List[dict]:
    seen: Dict[str, dict] = {}
    for j in jobs:
        if j["gen_key"] not in seen:
            seen[j["gen_key"]] = {**j, "job_id": j["gen_key"]}
    return list(seen.values())


def fan_out(jobs: Iterable[dict], gen_rows: Iterable[dict]) -> List[dict]:
    by = {r["job_id"]: r for r in gen_rows}
    out = []
    for j in jobs:
        r = by.get(j["gen_key"])
        if r is None:
            continue
        out.append({k: j[k] for k in ("job_id", "set", "episode_id", "persona_id", "variant", "model", "seed", "group", "prompt_sha256", "gen_key")}
                   | {"output": r["output"], "new_tokens": r.get("new_tokens"), "hit_max": r.get("hit_max"), "raw_special": r.get("raw_special")})
    return out


def counts(data_dir: Path, safety_dir: Path, n_candidates: int = 4) -> dict:
    """Generation counts of the plan (for the README and the CPU validation)."""
    res = {}
    for s in ("PROBE", "DEV", "SAFETY", "HOLDOUT"):
        items = load_set(data_dir, safety_dir, s)
        models = ["base"] + [f"ckpt-{i}" for i in range(n_candidates)] if s == "PROBE" else ["base", "best"]
        js = plan(s, items, models)
        res[s] = {"items": len(items), "logical_jobs": len(js), "unique_generations": len(unique_generation_jobs(js)),
                  "base_generations": len(unique_generation_jobs(j for j in js if j["model"] == "base"))}
    res["total_unique_generations"] = sum(v["unique_generations"] for v in res.values())
    return res
