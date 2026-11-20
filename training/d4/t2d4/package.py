"""T2-D4 Colab release builder (CPU): three ZIPs + release manifest from a FROZEN dataset directory and this code tree.

    t2d4_code.zip            t2d4/ (incl. colab/, vendor/), scripts/g0_llamafactory_labels.py, configs/d4_train.yaml, requirements-colab.txt
    t2d4_data_PRIVATE.zip    train/ (LLaMA-Factory files + train_meta for the Colab G0), eval/ (DEV_PROBE, DEV_EVAL full records), keys/
                             (SAFETY / HOLDOUT answer keys), meta/ (freeze manifest, protocol, ledger, gates, audits, G0 result).
                             PRIVATE_CLOUD_UPLOAD_AUTHORIZATION_REQUIRED: real IRC / V2EX chat text inside.
    t2d4_public_safety.zip   SAFETY_EVAL and FROZEN_HOLDOUT (synthetic) model-visible prompts only: no answer key, no raw restricted value.

ZIPs are deterministic (sorted entries, fixed timestamps), so the same inputs give the same sha256. B3 real HOLDOUT is never packaged."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

from .colab import evaljobs as EJ
from .credentials import CREDENTIAL
from .colab import metrics as M

RELEASE_VERSION = "t2d4-colab-release-v1"
DATA_VERSION = "t2d4-data-v1.1"
DRIVE_SUBDIR = "t2d4_formal_v1"
DEFAULT_RUN_NAME = "d4_run1"
PRIVATE_LABEL = "PRIVATE_CLOUD_UPLOAD_AUTHORIZATION_REQUIRED"
ZIP_TIME = (2026, 1, 1, 0, 0, 0)
META_FILES = ("dataset_manifest.json", "dataset_freeze_manifest.json", "evaluation_protocol_v1.json", "token_ledger.json", "data_gate0.json",
              "model_visible_contract.json", "redaction_audit.json", "tokenizer_audit.json", "dataset_stats.json", "leakage_audit.json",
              "grounded_response_schema_v1.json", "profile_schema_v1.json")
CODE_DIRS = ("t2d4",)
CODE_FILES = ("scripts/g0_llamafactory_labels.py", "configs/d4_train.yaml", "requirements-colab.txt")
SECRET = CREDENTIAL   # t2d4/credentials.py: the same pattern the dataset builder uses to keep credential-shaped text out of training
WINPATH = re.compile(r"\b[A-Za-z]:\\\\|\b[A-Za-z]:\\(?:RAG|Users)\b", re.I)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rd(p: Path) -> List[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def wjsonl(p: Path, rows: Iterable[Mapping]) -> int:
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def wjson(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def tree_manifest(root: Path, name: str, extra: Mapping) -> dict:
    files = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name != name}
    man = {**extra, "files": files}
    wjson(root / name, man)
    return man


def det_zip(src: Path, zp: Path) -> dict:
    tmp = Path(str(zp) + ".partial")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                info = zipfile.ZipInfo(p.relative_to(src).as_posix(), ZIP_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                z.writestr(info, p.read_bytes())
    tmp.replace(zp)
    with zipfile.ZipFile(zp) as z:
        assert z.testzip() is None
        n = len(z.infolist())
    return {"sha256": sha(zp), "bytes": zp.stat().st_size, "entries": n}


# ---------------------------------------------------------------- data
def probe_lf_rows(items: Iterable[Mapping]) -> List[dict]:
    """DEV_PROBE as LLaMA-Factory sharegpt rows (same layout as train.jsonl): the eval-loss set (records it, never selects with it)."""
    return [{"id": e["episode_id"], "system": e["system"], "conversations": [{"from": "human", "value": e["user"]},
                                                                             {"from": "gpt", "value": e["reference_target"]}]}
            for e in sorted(items, key=lambda x: x["episode_id"])]


def build_data(dataset: Path, g0_result: Path, stage: Path, origin: str) -> dict:
    d = Path(dataset)
    t = stage / "train"
    t.mkdir(parents=True)
    for f in ("train.jsonl", "train_meta.jsonl", "dataset_manifest.json"):
        shutil.copy2(d / f, t / f)
    info = json.loads((d / "dataset_info.json").read_text(encoding="utf-8"))
    info["d4_probe"] = {**info["d4_train"], "file_name": "dev_probe_lf.jsonl"}
    wjson(t / "dataset_info.json", info)
    probe = rd(d / "dev_probe.jsonl")
    wjsonl(t / "dev_probe_lf.jsonl", probe_lf_rows(probe))
    for sub, f in (("eval", "dev_probe.jsonl"), ("eval", "dev_eval.jsonl"), ("keys", "safety_eval.jsonl"), ("keys", "frozen_holdout_synthetic.jsonl")):
        (stage / sub).mkdir(exist_ok=True)
        shutil.copy2(d / f, stage / sub / f)
    (stage / "meta").mkdir()
    for f in META_FILES:
        if (d / f).exists():
            shutil.copy2(d / f, stage / "meta" / f)
    if (d / "SHA256SUMS.txt").exists():
        shutil.copy2(d / "SHA256SUMS.txt", stage / "meta" / "dataset_SHA256SUMS.txt")
    shutil.copy2(g0_result, stage / "meta" / "G0_LABELS_RESULT.json")
    fz = stage / "meta" / "dataset_freeze_manifest.json"
    man = json.loads((d / "dataset_manifest.json").read_text(encoding="utf-8"))
    return tree_manifest(stage, "DATA_MANIFEST.json", {
        "version": man["data_version"], "data_origin": origin, "label": PRIVATE_LABEL, "contains_b3_holdout": False,
        "freeze_manifest_sha256": sha(fz) if fz.exists() else None, "train_rows": sum(1 for _ in open(t / "train.jsonl", encoding="utf-8")),
        "privacy": "real IRC / V2EX chat text (TRAIN, DEV_PROBE, DEV_EVAL real style cases) and raw synthetic restricted values (answer keys); "
                   "never public Git; cloud upload only with the owner's explicit authorisation"})


# ---------------------------------------------------------------- public safety
def build_safety(dataset: Path, stage: Path) -> dict:
    d = Path(dataset)
    stage.mkdir(parents=True)
    counts, leaks = {}, []
    for name, f in (("SAFETY", "safety_eval.jsonl"), ("HOLDOUT", "frozen_holdout_synthetic.jsonl")):
        rows = rd(d / f)
        if any(r.get("source") != "SYNTH" for r in rows):
            raise RuntimeError(f"POLICY_VIOLATION: {f} holds non-synthetic rows; only synthetic prompts may be public")
        pub = [EJ.public_prompt_record(r) for r in rows]
        for r, p in zip(rows, pub):
            if public_leaks(r, p):
                leaks.append(r["episode_id"])
        counts[name] = wjsonl(stage / EJ.PUBLIC_PROMPTS[name], pub)
    personas = [r for r in rd(d / "personas_synthetic.jsonl") if r.get("persona_kind") == "SYNTHETIC_PERSONA" or str(r.get("persona_id", "")).startswith("SYN-")]
    counts["PERSONAS"] = wjsonl(stage / "personas_synthetic.jsonl", personas)
    shutil.copy2(d / "evaluation_protocol_v1.json", stage / "evaluation_protocol_v1.json")
    all_vals = sorted({v for f in ("safety_eval.jsonl", "frozen_holdout_synthetic.jsonl") for r in rd(d / f) for v in M.restricted_values(r) if len(v) >= 4})
    ptxt = (stage / "personas_synthetic.jsonl").read_text(encoding="utf-8")
    if any(v in ptxt for v in all_vals):
        leaks.append("personas_synthetic.jsonl")
    if leaks:
        raise RuntimeError(f"PRIVACY_FAIL: {len(leaks)} public records contain a restricted value (first ids {leaks[:3]})")
    (stage / "README_SAFETY.md").write_text(
        "# T2-D4 public safety package\n\nSynthetic only (invented personas, invented values): the model-visible prompts of SAFETY_EVAL "
        "and FROZEN_HOLDOUT synthetic, the synthetic personas (style history and profile only) and the frozen D4_EVALUATION_PROTOCOL_V1. Prompts are rendered with grounded_persona/1.0 over the MODEL_VISIBLE_REDACTED_VIEW (stillhere.session.disclosure/1.0): "
        "PROTECTED / POLICY_UNCERTAIN values never appear. Answer keys are in the private data package.\n", encoding="utf-8", newline="\n")
    return tree_manifest(stage, "SAFETY_MANIFEST.json", {"version": DATA_VERSION, "data_origin": "synthetic", "counts": counts,
                                                         "restricted_value_scan": {"records_with_value": 0, "rule": "raw value or any 4-digit run"}})


def public_leaks(rec: Mapping, pub: Mapping) -> List[str]:
    """Prompt fields of a public record that show a restricted value of the memory state they were rendered from (raw value or any 4-digit
    run; the WRONG_MEMORY_STATE ablation is judged by its sibling state, as in redaction_audit). Non-prompt public fields are checked against
    both states."""
    from .redaction_audit import model_visible, restricted_for
    bad = []
    for name, text in model_visible({k: v for k, v in pub.items()}):
        ev, pv = restricted_for(rec, name)
        if M.leak(text, sorted({e.value for e in ev} | pv)):
            bad.append(name)
    meta = json.dumps({k: v for k, v in pub.items() if k not in EJ.PROMPT_FIELDS}, ensure_ascii=False)
    both = set(M.restricted_values(rec))
    w = (rec.get("ablation_answer_states") or {}).get("WRONG_MEMORY_STATE") or {}
    both |= set(M.restricted_values(rec, w.get("answer_state") or {}, w.get("evidence") or []))
    if both and M.leak(meta, sorted(both)):
        bad.append("metadata")
    return bad


# ---------------------------------------------------------------- code
def build_code(repo: Path, stage: Path) -> dict:
    stage.mkdir(parents=True)
    for d in CODE_DIRS:
        for p in sorted((repo / d).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".json", ".md", ".txt", ".yaml"):
                q = stage / p.relative_to(repo)
                q.parent.mkdir(parents=True, exist_ok=True)
                q.write_bytes(p.read_bytes())
    for f in CODE_FILES:
        q = stage / f
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_bytes((repo / f).read_bytes())
    bad = scan_text_tree(stage)
    if bad:
        raise RuntimeError(f"SECRET_OR_PATH_FOUND in code package: {bad[:5]}")
    return tree_manifest(stage, "CODE_MANIFEST.json", {"version": RELEASE_VERSION})


def scan_text_tree(root: Path) -> List[str]:
    bad = []
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and p.suffix in (".py", ".json", ".md", ".txt", ".yaml", ".ipynb"):
            t = p.read_text(encoding="utf-8", errors="replace")
            if SECRET.search(t):
                bad.append(f"secret-like token in {p.name}")
            if WINPATH.search(t):
                bad.append(f"Windows path in {p.name}")
    return bad
