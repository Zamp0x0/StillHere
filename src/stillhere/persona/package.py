"""``persona package``: a self-contained training package for Colab (L4 / A100) or a local CUDA machine. Built locally; nothing is uploaded.

    <profile>/packages/<package_id>/
        persona_data.zip            train / dev / smoke + dataset_info, persona card + prompt, dev evaluation set (A-E messages), training
                                    config, PERSONA_TRAINING_MANIFEST, DATA_MANIFEST (sha256 of every file), local label audit (if run)
        persona_code.zip            the code needed on the GPU (src/stillhere, training/{d4/t2d4, d5/d5pilot, persona}, pyproject.toml)
        StillHere_Persona_L4.ipynb  the notebook with this package id filled in (A100 variant for colab-a100)
        RELEASE_MANIFEST.json, SHA256SUMS.txt, README_COLAB_CN.md

NEVER in the package: the test split (it stays sealed on this machine), the raw WeFlow export, any other conversation, credentials.
ZIPs are deterministic (fixed timestamps, sorted entries), so the same inputs give the same package id; an existing package is verified
and reused, never overwritten. The code ZIP is taken from the WORKING TREE (tracked + untracked-not-ignored files of the shipped paths;
uncommitted work is recorded as ``git_dirty`` with a tree sha256), because this pipeline never commits on the user's behalf.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import card as K
from . import dataset as D
from . import doctor as DR
from . import evaluation as EV
from . import memory as M
from . import notebook as NB
from . import store as S

ROOT = Path(__file__).resolve().parents[3]
SHIPPED = ("pyproject.toml", "src/stillhere", "training/d4/t2d4", "training/d5/d5pilot", "training/persona")
FORBIDDEN_IN_CODE = ("/private/", ".env", ".sqlite", ".safetensors", "__pycache__", ".jsonl", ".pyc", "personas/")
TARGETS = ("cpu-only", "local-cuda", "colab-l4", "colab-a100", "remote-gpu")
CLOUD = ("colab-l4", "colab-a100", "remote-gpu")
SUPPORT = {"cpu-only": "DATA_AND_PROMPT_ONLY", "local-cuda": "AUTOMATED (persona train, >= 20 GiB VRAM)", "colab-l4": "AUTOMATED (notebook Run All)",
           "colab-a100": "AUTOMATED (notebook Run All)", "remote-gpu": "PACKAGE_ONLY (manual: copy, run the documented commands, bring the adapter ZIP back)"}
CONFIG_TEMPLATE = ROOT / "training" / "persona" / "configs" / "persona_qlora.yaml"
CONFIG_TEMPLATE_R2 = ROOT / "training" / "persona" / "configs" / "persona_qlora_r2_a100.yaml"
TRAIN_SYSTEM_MODES = ("training", "persona_prompt")      # system message of every training row: the fixed SFT prompt (r1) or the persona prompt
SMOKE_ROWS = 16
ZIP_DATE = (2020, 1, 1, 0, 0, 0)
PACKAGE_SCHEMA = "stillhere.persona.release/1.0"


class PackageError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def det_zip(entries: List[Tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(entries):
            zi = zipfile.ZipInfo(name, date_time=ZIP_DATE)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, data)
    return buf.getvalue()


def _walk_shipped(root: Path) -> List[str]:
    """The SHIPPED files without git (a source ZIP download, not a clone): every file, minus caches and compiled files."""
    out = []
    for s in SHIPPED:
        p = root / s
        if p.is_file():
            out.append(s)
        elif p.is_dir():
            out += [str(f.relative_to(root)).replace("\\", "/") for f in p.rglob("*")
                    if f.is_file() and "__pycache__" not in f.parts and f.suffix not in (".pyc", ".pyo")]
    return out


def code_files(root: Path = ROOT) -> List[str]:
    try:
        r = subprocess.run(["git", "-C", str(root), "ls-files", "-co", "--exclude-standard", "--", *SHIPPED], capture_output=True, text=True, check=True)
        listed = r.stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):           # not a git checkout (or no git at all): walk the source tree instead
        listed = _walk_shipped(root)
    files = sorted({f for f in listed if f and (root / f).is_file()})
    bad = [f for f in files if any(x in "/" + f for x in FORBIDDEN_IN_CODE)]
    return [f for f in files if f not in bad]


def code_zip(root: Path = ROOT) -> Tuple[bytes, dict]:
    files = code_files(root)
    entries = [(f, (root / f).read_bytes()) for f in files]
    tree = hashlib.sha256("".join(f"{f}\0{hashlib.sha256(b).hexdigest()}\n" for f, b in entries).encode()).hexdigest()
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", *SHIPPED], capture_output=True, text=True).stdout.strip())
    return det_zip(entries), {"files": len(entries), "code_tree_sha256": tree, "git_head": head, "git_dirty": dirty}


def plan(rows: int, template: dict) -> dict:
    """Steps, save / eval cadence and the pre-registered candidates.

    ``persona_candidate_policy``: ``half_full`` (default, r1: 50 % and 100 % of the run) or ``every_save`` (r2: six evenly spaced saved
    checkpoints, so an earlier, less collapsed checkpoint can be chosen by the behavioural selection rule)."""
    ga, mb = template["gradient_accumulation_steps"], template["per_device_train_batch_size"]
    per_epoch = max(math.ceil(math.ceil(rows / mb) / ga), 1)                 # = t2d4.colab.train_d4.expected_steps (transformers 4.57.1)
    steps = int(math.ceil(per_epoch * template["num_train_epochs"]))
    if template.get("persona_candidate_policy", "half_full") == "every_save":
        save = max(1, math.ceil(steps / 6))
        cands = sorted({*range(save, steps + 1, save), steps})
        return {"rows": rows, "steps": steps, "save_steps": save, "eval_steps": save, "candidate_steps": cands,
                "checkpoints_expected": cands, "candidate_policy": "every_save"}
    half = math.ceil(steps / 2)
    divisors = [s for s in range(10, 41) if half % s == 0]
    save = max(divisors) if divisors else (half if half < 10 else 25)        # a save step lands exactly on 50 % whenever possible
    cand_half = min(steps, half if half % save == 0 else math.ceil(half / save) * save)
    return {"rows": rows, "steps": steps, "save_steps": save, "eval_steps": save, "candidate_steps": sorted({cand_half, steps}),
            "checkpoints_expected": sorted({*range(save, steps + 1, save), steps})}


def resolved_config(rows: int, template_path: Optional[Path] = None) -> Tuple[bytes, dict]:
    import yaml
    t = yaml.safe_load(Path(template_path or CONFIG_TEMPLATE).read_text(encoding="utf-8"))
    p = plan(rows, t)
    t.update(save_steps=p["save_steps"], eval_steps=p["eval_steps"], persona_expected_train_rows=rows, persona_expected_steps=p["steps"],
             persona_candidate_steps=p["candidate_steps"])
    head = ("# StillHere persona QLoRA pilot config, resolved by stillhere persona package from training/persona/configs/persona_qlora.yaml\n"
            f"# rows {rows} -> {p['steps']} optimizer steps (micro-batch {t['per_device_train_batch_size']} x grad-accum {t['gradient_accumulation_steps']}, "
            f"1 epoch); checkpoints + dev loss every {p['save_steps']} steps; candidates {p['candidate_steps']} ({p.get('candidate_policy', '50 % / 100 %')})\n")
    return (head + yaml.safe_dump(t, sort_keys=False, allow_unicode=True)).encode("utf-8"), p


def lf_config_check(cfg_bytes: bytes) -> dict:
    """Parse the resolved config with LLaMA-Factory's own argument parser (unknown / invalid keys raise). CPU, no model load."""
    try:
        import yaml
        from llamafactory.hparams.parser import _parse_train_args
        import llamafactory
    except Exception as e:  # noqa: BLE001
        return {"status": "NOT_RUN", "reason": f"llamafactory not importable here ({type(e).__name__})"}
    c = {k: v for k, v in yaml.safe_load(cfg_bytes).items() if not k.startswith("persona_")}
    c.update(dataset_dir=".", output_dir=str(Path(os.environ.get("TEMP", "/tmp")) / "persona_parse_only"))

    def parse(args):
        try:
            _parse_train_args(args)
            return None
        except SystemExit as e:
            return f"SystemExit {e}"
        except Exception as e:  # noqa: BLE001
            return f"{type(e).__name__}: {str(e)[:300]}"
    err = parse(dict(c))
    if err is None:
        return {"status": "PASS", "framework_version": llamafactory.__version__, "keys": len(c)}
    if "bf16" in err or "gpu" in err.lower() or "cuda" in err.lower():
        # every key and value type is accepted; only the hardware checks of TrainingArguments fail on a CPU machine
        err2 = parse({**c, "bf16": False, "use_cpu": True})
        if err2 is None:
            return {"status": "PASS_KEYS_CPU", "framework_version": llamafactory.__version__, "keys": len(c),
                    "note": "all keys parsed by LLaMA-Factory 0.9.5; bf16 / NF4 / paged optimizer need a GPU: checked by notebook STAGE 06 (doctor) and STAGE 08 (smoke)",
                    "cpu_error": err}
        return {"status": "FAIL", "error": err2}
    return {"status": "FAIL", "error": err}


def dataset_info() -> dict:
    tags = {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"}
    cols = {"messages": "conversations", "system": "system"}
    return {n: {"file_name": f, "formatting": "sharegpt", "columns": cols, "tags": tags}
            for n, f in (("persona_train", "train.jsonl"), ("persona_dev", "dev.jsonl"), ("persona_smoke", "smoke.jsonl"))}


def smoke(train_bytes: bytes) -> bytes:
    rows = [json.loads(l) for l in train_bytes.decode("utf-8").splitlines() if l.strip()]
    longest = sorted(rows, key=lambda r: (-len(r["system"] + r["conversations"][0]["value"] + r["conversations"][1]["value"]), r["id"]))[:SMOKE_ROWS]
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in longest).encode("utf-8")


def latest_dataset(pdir: Path) -> Path:
    best = None
    for d in sorted((pdir / "datasets").glob("*")):
        if d.is_dir() and not d.name.startswith(".") and (d / "DATASET_MANIFEST.json").is_file():
            m = S.read_json(d / "DATASET_MANIFEST.json")
            if best is None or m["created_at"] > best[0]:
                best = (m["created_at"], d)
    if best is None:
        raise PackageError("NO_DATASET", "no dataset yet: run stillhere persona prepare first")
    return best[1]


def dev_eval_set(pdir: Path, ddir: Path, prof: dict, cur: dict, *, with_memory: bool = True) -> Path:
    """The DEV evaluation set (messages of A-E) for this dataset + card; cached under <profile>/evals/sets/."""
    out = pdir / "evals" / "sets" / f"{ddir.name}_{cur['card_id']}_dev.jsonl"
    if out.is_file() and out.with_suffix(".meta.json").is_file():
        EV.read_eval_set(out)
        return out
    items = EV.eval_items(ddir, "dev")
    mem_meta = None
    if with_memory:
        api, mem_meta = M.build_api(prof["source"]["path"], persona_id=prof["profile_id"], target_username=prof["target"]["username"],
                                    cutoff=M.cutoff_from_train(ddir))
        blocks = M.blocks_for(items, api, prof["profile_id"])
        api.close()
        for it in items:
            it["memory"] = blocks[it["sample_id"]]
    groups = ["A", "B", "C", "D", "E"] if with_memory else ["A", "B", "C"]
    EV.write_eval_set(out, items, persona_prompt=cur["prompt"], groups=groups,
                      meta={"split": "dev", "dataset_version": ddir.name, "card_id": cur["card_id"], "memory": mem_meta})
    return out


def run_label_audit(data_dir: Path, out_dir: Path, base_dir: Optional[str], python: Optional[str] = None) -> dict:
    """The real LLaMA-Factory label audit on CPU (``personatrain.lf_labels``, all train + dev rows). Needs llamafactory and a local
    Qwen3-8B snapshot whose tokenizer / chat template match the pinned sha256 (no weights are loaded)."""
    bc = DR.base_check(base_dir)
    if not bc.get("present") or not (bc["tokenizer_sha256_ok"] and bc["chat_template_sha256_ok"]):
        return {"LLAMAFACTORY_LABELS": "PENDING", "reason": "no local Qwen3-8B tokenizer with the pinned sha256 (--base-dir); the notebook runs it on Colab"}
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": os.pathsep.join(str(ROOT / x) for x in ("training/persona", "training/d5", "training/d4", "src"))}
    work = out_dir / "lf_work"
    cmd = [python or sys.executable, "-m", "personatrain.lf_labels", "--data", str(data_dir), "--model-dir", str(base_dir), "--work", str(work),
           "--out", str(out_dir), "--cutoff", "2048", "--splits", "train,dev", "--persona-prompt", str(data_dir / "persona" / "persona_prompt.txt")]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, encoding="utf-8", errors="replace")
    res_p = out_dir / "LF_LABELS_RESULT.json"
    if not res_p.is_file():
        return {"LLAMAFACTORY_LABELS": "PENDING", "reason": "label audit did not run here: " + (r.stderr or r.stdout)[-400:]}
    import shutil
    shutil.rmtree(work, ignore_errors=True)
    return S.read_json(res_p)


def with_system(rows_b: bytes, system: str) -> bytes:
    """The same rows with ``system`` replaced (r2 prompt-consistent training); ids, order and conversations are untouched."""
    out = []
    for line in rows_b.decode("utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            r["system"] = system
            out.append(json.dumps(r, ensure_ascii=False) + "\n")
    return "".join(out).encode("utf-8")


def build(root, profile_id: str, *, target: str, confirm_cloud: bool = False, base_dir: Optional[str] = None, run_id: Optional[str] = None,
          with_memory: bool = True, python: Optional[str] = None, config_template: Optional[str] = None) -> dict:
    """``config_template``: default r1 template; the r2 template (``persona_qlora_r2_a100.yaml``) also sets ``persona_train_system``,
    ``persona_selection_rule``, ``persona_candidate_policy`` and ``persona_run_label``, applied here and recorded in the manifests."""
    if target not in TARGETS:
        raise PackageError("BAD_TARGET", f"target must be one of {TARGETS}")
    try:
        import yaml
    except ImportError:
        raise PackageError("NEEDS_PYYAML", "building a training package needs PyYAML: pip install pyyaml (or pip install -e .)") from None
    tpl_path = Path(config_template) if config_template else CONFIG_TEMPLATE
    tpl = yaml.safe_load(tpl_path.read_text(encoding="utf-8"))
    train_system = tpl.get("persona_train_system", "training")
    if train_system not in TRAIN_SYSTEM_MODES:
        raise PackageError("BAD_CONFIG", f"persona_train_system must be one of {TRAIN_SYSTEM_MODES}")
    selection_rule = tpl.get("persona_selection_rule", "r1")
    if selection_rule not in EV.SELECTION_RULES:
        raise PackageError("BAD_CONFIG", f"persona_selection_rule must be one of {sorted(EV.SELECTION_RULES)}")
    root = Path(root) if root else S.default_root()
    S.check_private_root(root)
    pdir = S.profile_dir(root, profile_id)
    prof = S.read_json(pdir / "profile.json")
    ddir = latest_dataset(pdir)
    dman = S.read_json(ddir / "DATASET_MANIFEST.json")
    q = S.read_json(ddir / "QUALITY_REPORT.json")
    cur = K.current(pdir)
    if cur is None or cur["manifest"]["dataset_version"] != ddir.name:
        K.build_auto(pdir, ddir, prof)
        cur = K.current(pdir)
    if target in CLOUD and not confirm_cloud:
        raise PackageError("CLOUD_NOT_CONFIRMED", S.CLOUD_NOTICE + " Re-run with --confirm-cloud (or answer yes) to build it.")
    for n, h in dman["files"].items():
        if D.sha256_bytes((ddir / n).read_bytes()) != h:
            raise PackageError("DATASET_CORRUPT", f"{n} does not match the dataset manifest")
    train_b, dev_b = (ddir / "train.jsonl").read_bytes(), (ddir / "dev.jsonl").read_bytes()
    persona_prompt_text = (Path(cur["dir"]) / "persona_prompt.txt").read_text(encoding="utf-8")
    system_text = persona_prompt_text if train_system == "persona_prompt" else D.SYSTEM_PROMPT
    if train_system == "persona_prompt":
        train_b, dev_b = with_system(train_b, system_text), with_system(dev_b, system_text)
    rows = sum(1 for l in train_b.decode("utf-8").splitlines() if l.strip())
    cfg_b, pl = resolved_config(rows, tpl_path)
    evs = dev_eval_set(pdir, ddir, prof, cur, with_memory=with_memory)
    files: Dict[str, bytes] = {
        "train/train.jsonl": train_b, "train/dev.jsonl": dev_b, "train/smoke.jsonl": smoke(train_b),
        "train/dataset_info.json": json.dumps(dataset_info(), ensure_ascii=False, indent=1).encode("utf-8"),
        "persona/persona_card.yaml": (Path(cur["dir"]) / "persona_card.yaml").read_bytes(),
        "persona/persona_prompt.txt": (Path(cur["dir"]) / "persona_prompt.txt").read_bytes(),
        "persona/persona_card_manifest.json": (Path(cur["dir"]) / "persona_card_manifest.json").read_bytes(),
        "eval/dev_eval.jsonl": evs.read_bytes(), "eval/dev_eval.meta.json": evs.with_suffix(".meta.json").read_bytes(),
        "config/training_config.yaml": cfg_b,
    }
    code_b, code_meta = code_zip()
    tm = {"schema": "stillhere.persona.training_manifest/1.0", "profile_id": profile_id, "dataset_version": ddir.name, "card_id": cur["card_id"],
          "card_review_status": cur["manifest"]["review_status"], "persona_prompt_sha256": cur["manifest"]["prompt_sha256"], "persona_prompt_version": K.PROMPT_VERSION,
          "training_system_prompt_version": D.SYSTEM_PROMPT_VERSION if train_system == "training" else "persona_prompt:" + cur["card_id"],
          "training_system_prompt_sha256": EV.sha(system_text), "training_system_mode": train_system, "selection_rule": selection_rule,
          "candidate_policy": pl.get("candidate_policy", "half_full"), "config_template": tpl_path.name,
          "neutral_prompt_sha256": EV.sha(EV.NEUTRAL_PROMPT), "train_rows": rows, "dev_rows": q["dataset"]["splits"]["dev"]["samples"], "smoke_rows": SMOKE_ROWS,
          "expected_steps": pl["steps"], "save_steps": pl["save_steps"], "candidate_steps": pl["candidate_steps"], "dataset_gates_pass": q["gates_pass"],
          "supervised_tokens": {k: v["supervised_tokens"] for k, v in q["dataset"]["splits"].items() if k != "test"},
          "base_revision": DR.BASE_REVISION, "tokenizer_sha256": DR.TOKENIZER_SHA256, "chat_template_sha256": DR.CHAT_TEMPLATE_SHA256,
          "template": {"name": "qwen3", "enable_thinking": False, "train_on_prompt": False, "mask_history": True, "cutoff_len": 2048,
                       "eos": "<|im_end|>", "training_vs_inference": ("same chat template and user-turn transcript; system = training prompt (train) vs persona prompt (inference)"
                                                 if train_system == "training" else
                                                 "same chat template, transcript AND system prompt (the persona prompt) in training and inference")},
          "decode": EV.DECODE, "code_tree_sha256": code_meta["code_tree_sha256"], "target": target,
          "privacy": "PRIVATE: real chat; no test split, no raw export; not for publication"}
    files["PERSONA_TRAINING_MANIFEST.json"] = json.dumps(tm, ensure_ascii=False, indent=1).encode("utf-8")
    staging = pdir / "packages" / ".staging"
    import shutil
    shutil.rmtree(staging, ignore_errors=True)
    for n, b in files.items():
        S.atomic_write(staging / n, b)
    dm = {"version": f"{profile_id}/{ddir.name}/{cur['card_id']}", "files": {n: hashlib.sha256(b).hexdigest() for n, b in sorted(files.items())},
          "contains_test": False, "contains_raw_export": False, "contains_internal_holdout": False, "contains_full_pool": False}
    S.atomic_write(staging / "DATA_MANIFEST.json", json.dumps(dm, ensure_ascii=False, indent=1).encode("utf-8"))
    audit_dir = pdir / "audits" / ddir.name / cur["card_id"]
    lf_p = audit_dir / "LF_LABELS_RESULT.json"
    if lf_p.is_file() and S.read_json(lf_p).get("splits", {}).get("train", {}).get("file_sha256") == dm["files"]["train/train.jsonl"]:
        lf = S.read_json(lf_p)
    else:
        audit_dir.mkdir(parents=True, exist_ok=True)
        lf = run_label_audit(staging, audit_dir, base_dir, python)
    cfg_check = lf_config_check(cfg_b)
    data_entries = [(n, b) for n, b in files.items()] + [("DATA_MANIFEST.json", (staging / "DATA_MANIFEST.json").read_bytes())]
    if lf.get("LLAMAFACTORY_LABELS") in ("PASS", "FAIL"):
        data_entries.append(("audit/LF_LABELS_RESULT_LOCAL.json", json.dumps(lf, ensure_ascii=False, indent=1).encode("utf-8")))
    data_b = det_zip(data_entries)
    nb_target = target if target in ("colab-l4", "colab-a100") else "colab-l4"
    rid = run_id or f"{profile_id}-{tpl.get('persona_run_label', 'r1')}"
    nb_name = "StillHere_Persona_A100.ipynb" if target == "colab-a100" else "StillHere_Persona_L4.ipynb"
    pid_hash = hashlib.sha256((hashlib.sha256(data_b).hexdigest() + code_meta["code_tree_sha256"] + target + rid).encode()).hexdigest()[:12]
    package_id = f"pk-{pid_hash}"
    nb_b = NB.dumps(NB.build(package_id, rid, nb_target))
    out = pdir / "packages" / package_id
    shutil.rmtree(staging, ignore_errors=True)
    if out.exists():
        sums = dict((l.split(None, 1)[1].strip(), l.split(None, 1)[0]) for l in (out / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines() if l.strip())
        if all(hashlib.sha256((out / n).read_bytes()).hexdigest() == h for n, h in sums.items()):
            rm = S.read_json(out / "RELEASE_MANIFEST.json")
            return {"package_id": package_id, "package_dir": str(out), "reused": True, "target": target, "notebook": str(out / nb_name),
                    "label_audit": rm["local_label_audit"], "plan": rm["plan"], "lf_config_check": rm["lf_config_check"]}
        raise PackageError("PACKAGE_CORRUPT", f"{out} exists but does not match its SHA256SUMS; it is not overwritten")
    rm = {"schema": PACKAGE_SCHEMA, "package_id": package_id, "profile_id": profile_id, "target": target, "run_id_suggested": rid,
          "dataset_version": ddir.name, "card_id": cur["card_id"], "card_review_status": cur["manifest"]["review_status"],
          "persona_prompt_sha256": tm["persona_prompt_sha256"], "training_system_prompt_sha256": tm["training_system_prompt_sha256"],
          "data_manifest_sha256": hashlib.sha256((dict(data_entries))["DATA_MANIFEST.json"]).hexdigest(), "data_zip_files": sorted(n for n, _ in data_entries),
          "training_config_sha256": hashlib.sha256(cfg_b).hexdigest(), "plan": pl, "lf_config_check": cfg_check,
          "training_system_mode": train_system, "selection_rule": selection_rule, "config_template": tpl_path.name,
          "local_label_audit": {"status": lf.get("LLAMAFACTORY_LABELS"), "reason": lf.get("reason"),
                                "splits": {k: {x: v.get(x) for x in ("rows", "status", "failures_total", "negative_control_caught", "max_input_ids")} for k, v in lf.get("splits", {}).items()}},
          "base": {"id": DR.BASE_REPO, "revision": DR.BASE_REVISION, "tokenizer_sha256": DR.TOKENIZER_SHA256, "chat_template_sha256": DR.CHAT_TEMPLATE_SHA256},
          "pins_train": DR.PINS_TRAIN, "expect_versions": DR.EXPECT, **code_meta, "created_at": S.now_iso(),
          "contains": {"test": False, "raw_export": False, "other_conversations": False}, "privacy_notice": S.CLOUD_NOTICE,
          "cloud_transfer_confirmed_by_operator": bool(confirm_cloud) if target in CLOUD else None, "support_status": SUPPORT[target]}
    tmp = pdir / "packages" / f".tmp-{package_id}"
    shutil.rmtree(tmp, ignore_errors=True)
    S.atomic_write(tmp / "persona_data.zip", data_b)
    S.atomic_write(tmp / "persona_code.zip", code_b)
    S.atomic_write(tmp / nb_name, nb_b)
    S.write_json(tmp / "RELEASE_MANIFEST.json", rm)
    S.atomic_write(tmp / "README_COLAB_CN.md", readme(rm, nb_name, {n: len(b) for n, b in (("persona_data.zip", data_b), ("persona_code.zip", code_b), (nb_name, nb_b))}).encode("utf-8"))
    sums = "".join(f"{hashlib.sha256((tmp / n).read_bytes()).hexdigest()}  {n}\n" for n in ("persona_code.zip", "persona_data.zip", "RELEASE_MANIFEST.json", nb_name))
    S.atomic_write(tmp / "SHA256SUMS.txt", sums.encode("utf-8"))
    if target == "remote-gpu":
        S.atomic_write(tmp / "README_REMOTE_GPU_CN.md", remote_readme(rm).encode("utf-8"))
    os.replace(tmp, out)
    S.append_lineage(pdir, {"artefact": f"packages/{package_id}", "kind": "training_package", "target": target, "dataset_version": ddir.name, "card_id": cur["card_id"]})
    if target in CLOUD:
        S.record_cloud_confirmation(pdir, package_id, target)
    return {"package_id": package_id, "package_dir": str(out), "reused": False, "target": target, "notebook": str(out / nb_name),
            "label_audit": rm["local_label_audit"], "plan": pl, "lf_config_check": cfg_check}


def readme(rm: dict, nb_name: str, sizes: dict) -> str:
    rows = "\n".join(f"| `{n}` | {s / 2**20:.2f} MB |" for n, s in sizes.items())
    return f"""# Persona 训练包 `{rm['package_id']}`（{rm['target']}）

**含真实私聊**（目标人物的回复与对方的消息上下文）。不含 Test 集、不含原始 WeFlow JSON、不含其他对话。只有你确认可以放到云端时才上传。

## 1. 上传到 Google Drive（只需一次）

在 Drive 建立 `MyDrive/StillHere/persona/{rm['package_id']}/input/`，把下面 4 个文件放进去；`{nb_name}` 放在 `MyDrive/StillHere/persona/{rm['package_id']}/`。

| 文件 | 大小 |
|---|---|
{rows}
| `RELEASE_MANIFEST.json`、`SHA256SUMS.txt` | 小 |

## 2. 运行

1. 在 Drive 里双击打开 `{nb_name}`（用 Google Colaboratory 打开）。
2. Runtime → Change runtime type → **{'A100' if rm['target'] == 'colab-a100' else 'L4'} GPU**。
3. STAGE 00：确认 `PACKAGE_ID = "{rm['package_id']}"`、`RUN_ID = "{rm['run_id_suggested']}"`；读完提示后把 `CONFIRM_PRIVATE_DATA_ON_CLOUD` 改成 `True`。
4. Runtime → Run all。顺序：校验 → 依赖 → Base → GPU Preflight → Loss Mask（全部行）→ 2 步 Smoke（含续训）→ 正式训练（{rm['plan']['steps']} 步，每 {rm['plan']['save_steps']} 步保存）→ Dev 选检查点（候选 {rm['plan']['candidate_steps']}）→ 导出。
5. 断线：重新连接同一种 GPU，从 STAGE 00 再运行；会从最新完整检查点续训，已完成的不会重训。

## 3. 拿回结果

下载 `MyDrive/StillHere/persona/{rm['package_id']}/runs/<RUN_ID>/exports/` 里的 `persona_candidate_*.zip`（人物 Adapter 包）和 `persona_results_*.zip`（不含权重），然后在本地：

```
stillhere persona import-adapter --profile {rm['profile_id']} --file <persona_candidate_*.zip>
stillhere persona evaluate --profile {rm['profile_id']} --split dev --groups A,B,C,D,E --backend hf --adapter <导入后显示的 id>
```
"""



def remote_readme(rm: dict) -> str:
    """PACKAGE_ONLY: the commands for a GPU machine you manage yourself (>= 20 GiB VRAM; Linux + CUDA). Nothing here is automated."""
    return f"""# 远程 GPU 训练（PACKAGE_ONLY：手动，StillHere 不会自动登录或上传）

包：`{rm['package_id']}`，Run：`{rm['run_id_suggested']}`，Base：`{rm['base']['id']}@{rm['base']['revision'][:8]}`（NF4 QLoRA，配置见 `config/training_config.yaml`）。

这个包含有**真实私人聊天**（不含 Test、不含原始导出）。只拷到你自己控制的机器上。

1. 把 `persona_code.zip`、`persona_data.zip`、`SHA256SUMS.txt` 拷到远程机器（例如 `scp`），核对：`sha256sum -c SHA256SUMS.txt`。
2. 解压：`unzip persona_code.zip -d code && unzip persona_data.zip -d data`。
3. 环境：Python 3.11 / 3.12，按 `RELEASE_MANIFEST.json` 的 `pins_train` 安装 torch / transformers / peft / bitsandbytes / llamafactory。
4. 选 GPU：`export CUDA_VISIBLE_DEVICES=<编号>`（进程内看到的就是 `cuda:0`）。
5. 冒烟：`PYTHONPATH=code/training/persona:code/training/d5:code/training/d4:code/src python -m personatrain.train_persona --data data --config data/config/training_config.yaml --base-dir <Qwen3-8B 目录> --output-dir run/smoke --smoke --dataset persona_smoke --max-steps 2 --save-steps 1`
6. 正式：同一条命令去掉 `--smoke --dataset persona_smoke --max-steps 2 --save-steps 1`，改成 `--output-dir run/train`；中断后用同一条命令续跑（从最近 checkpoint 恢复）。
7. 结束后把导出的 `persona_candidate_*.zip` 拷回本机：`stillhere persona import-adapter --profile {rm['profile_id']} --file <zip>`（校验 Profile / Base / Card / Dataset / SHA）。
"""


def build_refine(root, profile_id: str, refine_id: str, *, adapter_id: str, confirm_cloud: bool = False, target: str = "colab-l4") -> dict:
    """Q1 on Colab: the candidate DEV sets + the installed adapter + code + a no-training notebook. Same layout / checks as a training
    package (STAGE 00-05 are shared), so nothing new has to be trusted; no test, no raw export."""
    from . import adapters as AD
    if target in CLOUD and not confirm_cloud:
        raise PackageError("CLOUD_NOT_CONFIRMED", S.CLOUD_NOTICE + " Re-run with --confirm-cloud (or answer yes) to build it.")
    root = Path(root) if root else S.default_root()
    pdir = S.profile_dir(root, profile_id)
    rdir = pdir / "evals" / "refine" / refine_id
    plan = S.read_json(rdir / "REFINE_PLAN.json")
    ad = AD.resolve(pdir, adapter_id)
    ddir = latest_dataset(pdir)
    files = {"refine/REFINE_PLAN.json": (rdir / "REFINE_PLAN.json").read_bytes(), "train/train.jsonl": (ddir / "train.jsonl").read_bytes(),
             "adapter/adapter_model.safetensors": (Path(ad["dir"]) / "adapter_model.safetensors").read_bytes(),
             "adapter/adapter_config.json": (Path(ad["dir"]) / "adapter_config.json").read_bytes()}
    groups = None
    for c in plan["candidates"]:
        es = Path(c["eval_set"])
        files[f"refine/{c['name']}/dev_eval.jsonl"] = es.read_bytes()
        files[f"refine/{c['name']}/dev_eval.meta.json"] = es.with_suffix(".meta.json").read_bytes()
        groups = S.read_json(es.with_suffix(".meta.json"))["groups"]
    if not groups:
        raise PackageError("NO_CANDIDATES", "the refinement plan has no candidate prompts (nothing to evaluate)")
    dm = {"version": f"{profile_id}/refine/{refine_id}", "files": {n: hashlib.sha256(b).hexdigest() for n, b in sorted(files.items())},
          "contains_test": False, "contains_raw_export": False}
    entries = sorted(files.items()) + [("DATA_MANIFEST.json", json.dumps(dm, indent=1).encode("utf-8"))]
    data_b = det_zip(entries)
    code_b, code_meta = code_zip()
    package_id = "pr-" + hashlib.sha256((hashlib.sha256(data_b).hexdigest() + code_meta["code_tree_sha256"] + target).encode()).hexdigest()[:12]
    out = pdir / "packages" / package_id
    nb_name = "StillHere_Persona_Refine_L4.ipynb"
    if out.exists():
        return {"package_id": package_id, "package_dir": str(out), "reused": True, "notebook": str(out / nb_name)}
    rm = {"schema": PACKAGE_SCHEMA, "package_id": package_id, "kind": "prompt_refinement_eval", "profile_id": profile_id, "refine_id": refine_id,
          "adapter_id": adapter_id, "refine_groups": groups, "data_zip_files": sorted(n for n, _ in entries),
          "base": {"id": DR.BASE_REPO, "revision": DR.BASE_REVISION, "tokenizer_sha256": DR.TOKENIZER_SHA256, "chat_template_sha256": DR.CHAT_TEMPLATE_SHA256},
          "pins_train": DR.PINS_TRAIN, "expect_versions": DR.EXPECT, **code_meta, "contains": {"test": False, "raw_export": False}, "privacy_notice": S.CLOUD_NOTICE,
          "cloud_transfer_confirmed_by_operator": bool(confirm_cloud) if target in CLOUD else None, "created_at": S.now_iso()}
    import shutil
    tmp = pdir / "packages" / f".tmp-{package_id}"
    shutil.rmtree(tmp, ignore_errors=True)
    S.atomic_write(tmp / "persona_data.zip", data_b)
    S.atomic_write(tmp / "persona_code.zip", code_b)
    S.atomic_write(tmp / nb_name, NB.dumps(NB.build_refine(package_id, refine_id, target)))
    S.write_json(tmp / "RELEASE_MANIFEST.json", rm)
    sums = "".join(f"{hashlib.sha256((tmp / n).read_bytes()).hexdigest()}  {n}\n" for n in ("persona_code.zip", "persona_data.zip", "RELEASE_MANIFEST.json", nb_name))
    S.atomic_write(tmp / "SHA256SUMS.txt", sums.encode("utf-8"))
    os.replace(tmp, out)
    S.append_lineage(pdir, {"artefact": f"packages/{package_id}", "kind": "refine_package", "refine_id": refine_id})
    if target in CLOUD:
        S.record_cloud_confirmation(pdir, package_id, target)
    return {"package_id": package_id, "package_dir": str(out), "reused": False, "notebook": str(out / nb_name)}


def import_refine_results(root, profile_id: str, zip_path) -> dict:
    """Unpack a Colab refine results ZIP into ``evals/refine/<rid>/<candidate>/run`` (only the candidates of a known plan)."""
    root = Path(root) if root else S.default_root()
    pdir = S.profile_dir(root, profile_id)
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.startswith("refine/") and not n.endswith("/")]
        if any(".." in n.split("/") or n.startswith("/") for n in names):
            raise PackageError("ZIP_UNSAFE", "unsafe member path")
        cand = {n.split("/")[1] for n in names}
        plans = sorted((pdir / "evals" / "refine").glob("*/REFINE_PLAN.json"))
        match = [p for p in plans if cand and {c["name"] for c in S.read_json(p)["candidates"]} >= cand]
        if not match:
            raise PackageError("REFINE_UNKNOWN", "these results do not belong to a refinement plan of this profile")
        rdir = max(match, key=lambda p: p.stat().st_mtime).parent
        for n in names:
            dest = rdir / "/".join(n.split("/")[1:])
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(z.read(n))
    return {"refine_id": rdir.name, "candidates": sorted(cand)}
