"""T2-D4 on Colab A100: stages 00-17, called one per notebook cell (``run.call(ctx, "NN_name")``). From the A100-proven T2-D3 Mini run.

Every stage is restartable:
* a finished stage writes ``<run>/stages/NN_name.json``; a rerun VERIFIES it instead of redoing it; a failed stage writes ``NN_name.FAILED.json``;
* generation stages append rows to a local JSONL and merge them atomically into the Drive copy; generation keys are (model, seed, prompt
  sha256), so a Base prompt is generated ONCE and reused by every group (Base generation is never redone);
* ``<run>/run_config.json`` freezes configuration, release manifest hash and rule hashes at stage 00: a rerun with anything different STOPS
  (CONFIG_MISMATCH); every package / file hash mismatch STOPS (HASH_MISMATCH / PACKAGE_VERIFY_FAILED);
* training resumes from the newest COMPLETE checkpoint with the same run fingerprint and never overwrites a checkpoint (``train_d4``);
* FROZEN_HOLDOUT is generated once, after selection, for the selected adapter only (stage 15 refuses a second adapter).

``ctx.mock = True`` (CPU tests only): MockBackend instead of the model, the CPU smoke on a tiny model instead of QLoRA; every output is
labelled DRY_RUN and the Windows importer refuses it."""
from __future__ import annotations

import collections
import gc
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from . import evaljobs as EJ
from . import guard as GD
from . import io as IO
from . import metrics as M
from . import progress as PG
from . import runner as RN
from . import selection as SL
from . import verdict as VD

RUN_VERSION = "t2d4-colab-run-v1"
RELEASE_VERSION = "t2d4-colab-release-v1"
EXPECTED_DATA_VERSION = "t2d4-data-v1.1"
STAGES = ["00_config", "01_mount", "02_verify", "03_extract", "04_install", "05_cpu_audit", "06_gpu_preflight", "07_load_model",
          "08_base_generation", "09_qlora_smoke", "10_training", "11_checkpoint_probe", "12_selection", "13_adapter_dev", "14_adapter_safety",
          "15_holdout", "16_statistics", "17_export"]
PINNED = {"llamafactory": "0.9.5", "transformers": "4.57.1", "peft": "0.18.1", "trl": "0.24.0", "accelerate": "1.11.0", "datasets": "4.0.0",
          "tokenizers": "0.22.2", "bitsandbytes": "0.48.2"}
ZIPS = ("t2d4_code.zip", "t2d4_data_PRIVATE.zip", "t2d4_public_safety.zip")
MANIFEST = "T2D4_RELEASE_MANIFEST.json"
CONFIG = Path("configs") / "d4_train.yaml"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Ctx:
    drive_root: Path = Path("/content/drive/MyDrive/StillHere/t2d4_formal_v1")
    run_name: str = "d4_run1"
    local: Path = Path("/content/t2d4")
    base_source: str = "hub"          # or a Drive directory holding the pinned revision
    hf_cache: Path = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    mock: bool = False
    mock_tiny_model: Optional[str] = None
    max_batch_size: int = 32
    max_batch_tokens: int = 40000
    log: object = print
    backend: object = None
    state: Dict = field(default_factory=dict)

    @property
    def input(self) -> Path:
        return self.drive_root / "input"

    @property
    def run(self) -> Path:
        return self.drive_root / "runs" / self.run_name

    @property
    def code(self) -> Path:
        return self.local / "code"

    @property
    def data(self) -> Path:
        return self.local / "data"

    @property
    def safety_dir(self) -> Path:
        return self.local / "safety"

    def p(self, *parts) -> Path:
        q = self.run.joinpath(*parts)
        q.parent.mkdir(parents=True, exist_ok=True)
        return q


# ---------------------------------------------------------------- stage bookkeeping
def _marker(ctx: Ctx, name: str, failed: bool = False) -> Path:
    return ctx.p("stages", f"{name}{'.FAILED' if failed else ''}.json")


def done(ctx: Ctx, name: str) -> Optional[dict]:
    m = _marker(ctx, name)
    return json.loads(m.read_text(encoding="utf-8")) if m.exists() else None


def mark(ctx: Ctx, name: str, **info) -> dict:
    row = {"stage": name, "status": info.pop("status", "COMPLETE"), "completed_at": now(), "dry_run": ctx.mock, **info}
    IO.atomic_write_json(_marker(ctx, name), row)
    f = _marker(ctx, name, failed=True)
    if f.exists():
        f.unlink()
    ctx.log(f"[{name}] {row['status']}")
    return row


def call(ctx: Ctx, name: str, *args, **kw):
    """Run one stage by name; a failure is recorded (stages/NN.FAILED.json, shown by the dashboard) and re-raised (the cell STOPS)."""
    fn = globals()["stage_" + name]
    try:
        return fn(ctx, *args, **kw)
    except BaseException as exc:
        IO.atomic_write_json(_marker(ctx, name, failed=True), {"stage": name, "status": "FAILED", "error": f"{type(exc).__name__}: {str(exc)[:400]}",
                                                               "failed_at": now()})
        raise


# ---------------------------------------------------------------- dashboard
def _train_state(ctx: Ctx, which: str = "train") -> dict:
    for p in (ctx.local / "progress" / which / "events.jsonl", ctx.run / which / "progress" / "events.jsonl"):
        if p.exists():
            return PG.ProgressModel.from_file(p).state(time.time())
    return {}


def dashboard(ctx: Ctx) -> dict:
    """The 14 named run fields, from the run's real files only (nothing estimated here)."""
    st = {s: done(ctx, s) for s in STAGES}
    failed = [s for s in STAGES if _marker(ctx, s, failed=True).exists()]
    cur = failed[0] if failed else next((s for s in STAGES if not st[s]), None)
    gp = ctx.run / "gate0" / "gpu_preflight.json"
    gpu = json.loads(gp.read_text(encoding="utf-8")).get("gpu") if gp.exists() else None
    tr = _train_state(ctx)
    gen_done = gen_fail = 0
    if ctx.run.exists():
        for f in ctx.run.glob("progress/*_progress.json"):
            try:
                g = json.loads(f.read_text(encoding="utf-8"))
            except ValueError:
                continue
            gen_done += int(g.get("completed") or 0)
            gen_fail += int(g.get("failed_jobs") or 0)
    if failed:
        stage_status = "FAILED"
    elif cur is None:
        stage_status = "COMPLETE"
    else:
        prog = ctx.run / "progress" / f"{cur}_progress.json"
        stage_status = (json.loads(prog.read_text(encoding="utf-8")).get("status") if prog.exists() else
                        ("RUNNING" if cur in ("09_qlora_smoke", "10_training") and tr and tr.get("phase") not in ("FINISHED", "FAILED") else "PENDING"))
    run_status = "FAILED" if failed else "COMPLETE" if cur is None else ("NOT_STARTED" if not any(st.values()) else "IN_PROGRESS")
    mem = None
    if tr.get("vram_allocated_gb") is not None:
        mem = f"{tr['vram_allocated_gb']} GB allocated / peak {tr.get('vram_peak_gb')} GB of {tr.get('vram_total_gb')} GB"
    return {"CURRENT_STAGE": cur or "ALL_DONE", "GPU_NAME": gpu or tr.get("gpu") or ("mock-cpu" if ctx.mock else None),
            "TRAIN_STEP": tr.get("step"), "TOTAL_STEPS": tr.get("max_steps"), "EPOCH": tr.get("epoch"), "TRAIN_LOSS": tr.get("train_loss"),
            "EVAL_LOSS": tr.get("val_mean_loss"), "LEARNING_RATE": tr.get("learning_rate"), "GPU_MEMORY": mem, "CHECKPOINT": tr.get("latest_checkpoint"),
            "COMPLETED_GENERATIONS": gen_done, "FAILED_GENERATIONS": gen_fail, "STAGE_STATUS": stage_status, "RUN_STATUS": run_status,
            "DRY_RUN": ctx.mock}


def status_table(ctx: Ctx) -> str:
    d = dashboard(ctx)
    lines = [f"StillHere T2-D4  run {ctx.run_name}  ({'DRY RUN / MOCK' if ctx.mock else 'REAL'})", ""]
    lines += [f"  {k:<22} {'-' if v is None else v}" for k, v in d.items() if k != "DRY_RUN"]
    lines.append("")
    for s in STAGES:
        x = done(ctx, s)
        f = _marker(ctx, s, failed=True)
        status = "FAILED" if f.exists() else (x or {}).get("status", "-")
        lines.append(f"  {s:<22} {status:<14} {(x or {}).get('completed_at', '')}")
    sel = ctx.run / "selection" / "checkpoint_selection.json"
    if sel.exists():
        s = json.loads(sel.read_text(encoding="utf-8"))
        lines.append(f"  selected: checkpoint-{s['chosen_step']} ({s['status']}, score {s['chosen']['score']:.4f})")
    return "\n".join(lines)


# ---------------------------------------------------------------- 00-05: CPU stages
def _release_manifest(ctx: Ctx) -> dict:
    p = ctx.input / MANIFEST
    if not p.exists():
        raise RuntimeError(f"PACKAGE_VERIFY_FAILED: {MANIFEST} is missing in {ctx.input} (upload every release file into input/)")
    return json.loads(p.read_text(encoding="utf-8"))


def _obj_sha(o) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(o, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def stage_00_config(ctx: Ctx) -> dict:
    man = ctx.input / MANIFEST
    cfg = {"run_version": RUN_VERSION, "run_name": ctx.run_name, "base_source": ctx.base_source, "mock": ctx.mock,
           "release_manifest_sha256": IO.sha256_file(man) if man.exists() else None, "selection_rules_sha256": SL.RULES_SHA256,
           "verdict_rules_sha256": VD.RULES_SHA256, "generation": EJ.GEN, "gen_plan": EJ.GEN_PLAN,
           "runner": {"max_batch_size": ctx.max_batch_size, "max_batch_tokens": ctx.max_batch_tokens}, "pinned": PINNED}
    f = ctx.p("run_config.json")
    if f.exists():
        old = json.loads(f.read_text(encoding="utf-8"))
        keys = ("release_manifest_sha256", "selection_rules_sha256", "verdict_rules_sha256", "generation", "mock", "base_source", "runner")
        diff = {k: (old.get(k), cfg[k]) for k in keys if old.get(k) != cfg[k]}
        if diff:
            raise RuntimeError(f"CONFIG_MISMATCH: run {ctx.run_name} was started with another configuration: {sorted(diff)}. Use a new RUN_NAME.")
        return mark(ctx, "00_config", resumed=True, config_sha256=_obj_sha(old))
    IO.atomic_write_json(f, cfg)
    return mark(ctx, "00_config", config_sha256=_obj_sha(cfg))


def stage_01_mount(ctx: Ctx) -> dict:
    if not ctx.mock:
        from google.colab import drive  # type: ignore
        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
    ctx.run.mkdir(parents=True, exist_ok=True)
    (ctx.run / "exports").mkdir(exist_ok=True)
    return mark(ctx, "01_mount", drive_root=str(ctx.drive_root))


def stage_02_verify(ctx: Ctx) -> dict:
    man = _release_manifest(ctx)
    if man.get("release_version") != RELEASE_VERSION or man.get("data_version") != EXPECTED_DATA_VERSION:
        raise RuntimeError(f"PACKAGE_VERSION_MISMATCH: manifest {man.get('release_version')} / {man.get('data_version')}, this code expects "
                           f"{RELEASE_VERSION} / {EXPECTED_DATA_VERSION}. Upload all files of ONE release.")
    if not ctx.mock and man.get("release_status") != "RELEASE_CANDIDATE":
        raise RuntimeError(f"NOT_A_RELEASE: this package is {man.get('release_status')} (a tool test, not a frozen release); it cannot start a real run")
    res, bad = {}, []
    for z in ZIPS:
        p = ctx.input / z
        exp = man["files"].get(z)
        if not p.exists():
            bad.append(f"{z}: missing in {ctx.input}")
            continue
        got = IO.sha256_file(p)
        res[z] = {"sha256": got, "bytes": p.stat().st_size, "ok": exp is not None and got == exp["sha256"] and p.stat().st_size == exp["bytes"]}
        if not res[z]["ok"]:
            bad.append(f"{z}: sha256/size differs from the release manifest")
    if bad:
        raise RuntimeError("HASH_MISMATCH: " + "; ".join(bad))
    return mark(ctx, "02_verify", files=res)


def _safe_extract(zp: Path, dest: Path) -> int:
    dest = dest.resolve()
    with zipfile.ZipFile(zp) as z:
        bad = z.testzip()
        if bad is not None:
            raise RuntimeError(f"HASH_MISMATCH: {zp.name}: CRC error in {bad}")
        for info in z.infolist():
            name = info.filename
            if name.startswith(("/", "\\")) or ".." in Path(name).parts or ":" in name or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise RuntimeError(f"UNSAFE_ZIP_ENTRY: {zp.name}: {name}")
            target = (dest / name).resolve()
            if dest not in target.parents and target != dest:
                raise RuntimeError(f"UNSAFE_ZIP_ENTRY: {zp.name}: {name}")
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        z.extractall(dest)
        return len(z.infolist())


def verify_tree(root: Path, manifest_name: str) -> List[str]:
    man = json.loads((root / manifest_name).read_text(encoding="utf-8"))
    bad = []
    for rel, h in man["files"].items():
        p = root / rel
        if not p.exists():
            bad.append(f"missing {rel}")
        elif IO.sha256_file(p) != h:
            bad.append(f"sha256 mismatch {rel}")
    return bad


def stage_03_extract(ctx: Ctx) -> dict:
    counts = {}
    for z, d in zip(ZIPS, (ctx.code, ctx.data, ctx.safety_dir)):
        counts[z] = _safe_extract(ctx.input / z, d)
    bad = verify_tree(ctx.code, "CODE_MANIFEST.json") + verify_tree(ctx.data, "DATA_MANIFEST.json") + verify_tree(ctx.safety_dir, "SAFETY_MANIFEST.json")
    if bad:
        raise RuntimeError("HASH_MISMATCH: " + "; ".join(bad[:5]))
    man = _release_manifest(ctx)
    dm = json.loads((ctx.data / "DATA_MANIFEST.json").read_text(encoding="utf-8"))
    fz = IO.sha256_file(ctx.data / "meta" / "dataset_freeze_manifest.json") if (ctx.data / "meta" / "dataset_freeze_manifest.json").exists() else None
    if dm.get("version") != man["data_version"] or fz != man.get("dataset_freeze_manifest_sha256"):
        raise RuntimeError("HASH_MISMATCH: the data package and the release manifest name different datasets / freezes")
    if dm.get("contains_b3_holdout"):
        raise RuntimeError("POLICY_VIOLATION: the B3 real HOLDOUT must not be in this package")
    joined = {s: len(EJ.load_set(ctx.data, ctx.safety_dir, s)) for s in ("SAFETY", "HOLDOUT")}  # HASH_MISMATCH if a prompt differs
    return mark(ctx, "03_extract", entries=counts, data_version=dm.get("version"), data_origin=dm.get("data_origin"), joined_public_prompts=joined)


def versions() -> dict:
    out = {"python": platform.python_version()}
    from importlib.metadata import version
    for k in list(PINNED) + ["torch"]:
        try:
            out[k] = version(k)
        except Exception:  # noqa: BLE001
            out[k] = None
    return out


def stage_04_install(ctx: Ctx) -> dict:
    v = versions()
    need = [f"{k}=={x}" for k, x in PINNED.items() if v.get(k) != x]
    if need and not ctx.mock:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need, "safetensors>=0.4.3", "pyyaml>=6.0"], check=True)
        v = versions()
    wrong = {k: (v.get(k), x) for k, x in PINNED.items() if v.get(k) != x}
    if wrong and not ctx.mock:
        raise RuntimeError(f"VERSION_MISMATCH after install: {wrong} (Runtime > Restart session once if Colab kept an old module loaded, then rerun)")
    return mark(ctx, "04_install", versions=v, mismatched=wrong, status="COMPLETE" if not wrong else "DRY_RUN")


def stage_05_cpu_audit(ctx: Ctx) -> dict:
    """Training / inference contract and data checks on CPU. Any inconsistency STOPS the run (CONTRACT_BLOCKED / DATASET_AUDIT_FAILED)."""
    from .. import renderer as RD
    from ..redaction_audit import audit
    d = ctx.data
    rd = lambda n: json.loads((d / "meta" / n).read_text(encoding="utf-8"))
    ledger, dm, contract = rd("token_ledger.json"), rd("dataset_manifest.json"), rd("model_visible_contract.json")
    rows = IO.read_jsonl(d / "train" / "train.jsonl")[0]
    sets = {s: EJ.load_set(d, ctx.safety_dir, s) for s in ("PROBE", "DEV", "SAFETY", "HOLDOUT")}
    RD.use(dm.get("evidence_view", "raw"))
    eval_max = max(int(e.get("prompt_tokens") or 0) for v in sets.values() for e in v)
    abl_max = max([int(x) for v in sets.values() for e in v for x in (e.get("ablation_prompt_tokens") or {}).values()] or [0])
    info = {"train_rows": len(rows), "ledger_train_rows": ledger.get("TOTAL_EPISODES"), "ledger_supervised_tokens": ledger.get("SUPERVISED_TOKENS"),
            "row_supervised_sum": sum(int(r.get("supervised_tokens") or 0) for r in rows), "max_train_full_tokens": max(int(r.get("full_tokens") or 0) for r in rows),
            "eval_items": {s: len(v) for s, v in sets.items()}, "max_eval_prompt_tokens": eval_max, "max_ablation_prompt_tokens": abl_max,
            "renderer_code_sha256": RD.renderer_sha256(), "renderer_data_sha256": dm.get("renderer_sha256"), "evidence_view": RD.view(),
            "contract_renderer_sha256": contract.get("renderer_sha256"), "contract_view": contract.get("redaction_format")}
    checks = {"train_rows_equal_ledger": info["train_rows"] == info["ledger_train_rows"],
              "supervised_tokens_equal_ledger": info["row_supervised_sum"] == info["ledger_supervised_tokens"],
              "renderer_code_equals_data": info["renderer_code_sha256"] == info["renderer_data_sha256"],
              "contract_renderer_equals_data": info["contract_renderer_sha256"] == info["renderer_data_sha256"],
              "contract_view_equals_code": info["contract_view"] == RD.view_version(),
              "redacted_view": RD.redacted() == (EXPECTED_DATA_VERSION != "t2d4-data-v1.0"),
              "train_within_cutoff": info["max_train_full_tokens"] <= 1536,
              "eval_prompt_plus_generation_within_window": max(eval_max, abl_max) + EJ.GEN["max_new_tokens"] + 2 <= 1536,
              "gate0_yes": rd("data_gate0.json").get("D4_1_DATA_READY") == "YES",
              "g0_labels_pass": rd("G0_LABELS_RESULT.json").get("LLAMAFACTORY_LABELS") == "PASS",
              "redaction_audit_pass": rd("redaction_audit.json").get("pass") is True}
    red = audit({s: v for s, v in sets.items()})  # recheck on Colab: no restricted raw value in any eval prompt the model will see
    checks["eval_prompts_redacted_recheck"] = red["pass"]
    info["checks"] = checks
    info["redaction_recheck"] = {k: red[k] for k in ("records_with_visible_raw_value", "pass")}
    IO.atomic_write_json(ctx.p("gate0", "cpu_dataset_audit.json"), info)
    bad = [k for k, v in checks.items() if not v]
    if bad:
        kind = "CONTRACT_BLOCKED" if any(k.startswith(("renderer", "contract", "redacted", "eval_prompt")) for k in bad) else "DATASET_AUDIT_FAILED"
        raise RuntimeError(f"{kind}: {bad}")
    return mark(ctx, "05_cpu_audit", **{k: v for k, v in info.items() if k != "checks"}, checks_passed=len(checks))


# ---------------------------------------------------------------- 06-07: GPU preflight, model
def stage_06_gpu_preflight(ctx: Ctx) -> dict:
    if ctx.mock:
        IO.atomic_write_json(ctx.p("gate0", "gpu_preflight.json"), {"gpu": "mock-cpu"})
        return mark(ctx, "06_gpu_preflight", gpu="mock-cpu", status="DRY_RUN")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU_PREFLIGHT_FAILED: no CUDA device. Runtime > Change runtime type > A100 GPU.")
    p = torch.cuda.get_device_properties(0)
    name, vram = torch.cuda.get_device_name(0), round(p.total_memory / 2**30, 1)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    info = {"gpu": name, "vram_gib": vram, "bf16": torch.cuda.is_bf16_supported(), "cuda": torch.version.cuda, "torch": torch.__version__, "nvidia_smi": smi}
    if "A100" not in name or vram < 39 or not info["bf16"]:
        raise RuntimeError(f"GPU_PREFLIGHT_FAILED: need an A100 (40 GB or more) with bf16, got {info}")
    IO.atomic_write_json(ctx.p("gate0", "gpu_preflight.json"), info)
    return mark(ctx, "06_gpu_preflight", **info)


def _settings(ctx: Ctx) -> RN.Settings:
    return RN.Settings(max_batch_size=ctx.max_batch_size, max_batch_tokens=ctx.max_batch_tokens, max_new_tokens=EJ.GEN["max_new_tokens"],
                       max_prompt_tokens=EJ.MAX_PROMPT_TOKENS)


def base_dir(ctx: Ctx) -> Path:
    from .backend_hf import resolve_base
    return resolve_base(ctx.base_source, str(ctx.hf_cache))


def load_backend(ctx: Ctx):
    if ctx.backend is not None:
        return ctx.backend
    if ctx.mock:
        from .backend_hf import MockBackend
        ctx.backend = MockBackend()
    else:
        from .backend_hf import HFBackend
        ctx.backend = HFBackend(base_dir(ctx))
    return ctx.backend


def free_backend(ctx: Ctx):
    if ctx.backend is None:
        return
    ctx.backend = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _g0(ctx: Ctx, model_dir: Path, tag: str) -> dict:
    """The LLaMA-Factory label check (scripts/g0_llamafactory_labels.py) with the Colab framework install, 1 row per stratum."""
    work = ctx.local / "g0_work" / tag
    if work.exists():
        shutil.rmtree(work)
    out = ctx.p("gate0", f"g0_{tag}")
    cmd = [sys.executable, str(ctx.code / "scripts" / "g0_llamafactory_labels.py"), "--data", str(ctx.data / "train"), "--model-dir", str(model_dir),
           "--work", str(work), "--out", str(out), "--per-stratum", "2"]
    env = dict(os.environ, PYTHONPATH=str(ctx.code), CUDA_VISIBLE_DEVICES="")
    with ctx.p("logs", f"g0_{tag}.log").open("a", encoding="utf-8") as lf:
        code = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env).returncode
    res = json.loads((out / "G0_LABELS_RESULT.json").read_text(encoding="utf-8")) if (out / "G0_LABELS_RESULT.json").exists() else {}
    return {"exit_code": code, "LLAMAFACTORY_LABELS": res.get("LLAMAFACTORY_LABELS"), "samples": res.get("samples"),
            "framework_version": res.get("framework_version"), "negative_control_pass": (res.get("negative_control") or {}).get("pass")}


def stage_07_load_model(ctx: Ctx) -> dict:
    dm = json.loads((ctx.data / "meta" / "dataset_manifest.json").read_text(encoding="utf-8"))
    if ctx.mock:
        load_backend(ctx)
        return mark(ctx, "07_load_model", status="DRY_RUN", g0_colab="NOT_TESTED (mock run)", note="mock backend")
    bd = base_dir(ctx)
    cfg = json.loads((bd / "config.json").read_text(encoding="utf-8"))
    tok_sha = IO.sha256_file(bd / "tokenizer.json")
    info = {"base_dir": str(bd), "revision_dir_ok": EJ.GEN["model_revision"] in str(bd) or ctx.base_source != "hub", "architectures": cfg.get("architectures"),
            "tokenizer_sha256": tok_sha, "tokenizer_ok": tok_sha == dm.get("tokenizer_sha256")}
    g0 = _g0(ctx, bd, "colab")
    info["g0_colab"] = g0
    if not (info["tokenizer_ok"] and info["revision_dir_ok"] and g0["LLAMAFACTORY_LABELS"] == "PASS"):
        raise RuntimeError(f"CONTRACT_BLOCKED: tokenizer / revision / LLaMA-Factory labels on Colab: tokenizer_ok {info['tokenizer_ok']}, "
                           f"revision_ok {info['revision_dir_ok']}, G0 {g0}")
    b = load_backend(ctx)  # checks the chat template sha
    item = EJ.load_set(ctx.data, ctx.safety_dir, "PROBE")[0]
    out = b.generate([{"model": "base", "seed": EJ.PRIMARY, "job_id": "gpu-path-check",
                       "messages": [{"role": "system", "content": item["system"]}, {"role": "user", "content": item["user"]}]}], 16)
    info.update(real_generation_path=bool(out and out[0]["new_tokens"] > 0), template_sha256=b.template_sha, **b.vram())
    IO.atomic_write_json(ctx.p("gate0", "gpu_gate.json"), info)
    return mark(ctx, "07_load_model", **info)


# ---------------------------------------------------------------- generation helper
def generate(ctx: Ctx, stage: str, jobs: List[dict]) -> List[dict]:
    """Generate the unique (model, seed, prompt) keys of ``jobs`` (resumable), return fanned-out rows for every job."""
    uniq = EJ.unique_generation_jobs(jobs)
    local = ctx.local / "gen" / f"{stage}.jsonl"
    local.parent.mkdir(parents=True, exist_ok=True)
    pers = ctx.p("generations", f"{stage}.jsonl")
    marker = ctx.p("generations", f"{stage}.COMPLETE.json")
    ids = [j["job_id"] for j in uniq]
    if marker.exists():
        v = RN.verify_done(marker, pers, ids)
        if v["valid"]:
            return EJ.fan_out(jobs, IO.read_jsonl(pers)[0])
        raise RuntimeError(f"HASH_MISMATCH: {stage}: the persisted generations differ from their completion marker ({v}); nothing was regenerated")
    b = load_backend(ctx)
    st = RN.run_stage(stage, uniq, b, local, pers, ctx.p("progress", f"{stage}_progress.json"), _settings(ctx), done_marker=marker,
                      gpu=getattr(b, "gpu", "unknown"), log=ctx.log)
    if st != "COMPLETED":
        raise RuntimeError(f"GENERATION_{st}: stage {stage} (rerun the cell to resume; completed rows are kept)")
    return EJ.fan_out(jobs, IO.read_jsonl(pers)[0])


def _sets(ctx: Ctx) -> Dict[str, List[dict]]:
    if "sets" not in ctx.state:
        ctx.state["sets"] = {s: EJ.load_set(ctx.data, ctx.safety_dir, s) for s in ("PROBE", "DEV", "SAFETY", "HOLDOUT")}
    return ctx.state["sets"]


def _plan(ctx: Ctx, s: str, models, model_filter=None) -> List[dict]:
    js = EJ.plan(s, _sets(ctx)[s], models, mock=ctx.mock)
    return [j for j in js if model_filter is None or j["model"] == model_filter]


def stage_08_base_generation(ctx: Ctx) -> dict:
    """Base for PROBE, DEV (every variant) and SAFETY. HOLDOUT is generated at stage 15 only."""
    jobs = _plan(ctx, "PROBE", ["base"]) + _plan(ctx, "DEV", ["base"]) + _plan(ctx, "SAFETY", ["base"])
    rows = generate(ctx, "08_base", jobs)
    by = collections.Counter(f"{r['set']}:{r['group']}" for r in rows)
    return mark(ctx, "08_base_generation", jobs=len(jobs), unique_generations=len(EJ.unique_generation_jobs(jobs)), rows=len(rows), by_group=dict(by))


# ---------------------------------------------------------------- 09-10: training
def _train_cmd(ctx: Ctx, out: Path, extra: List[str]) -> List[str]:
    cmd = [sys.executable, "-m", "t2d4.colab.train_d4", "--data", str(ctx.data), "--config", str(ctx.code / CONFIG), "--output-dir", str(out),
           "--progress-dir", str(ctx.local / "progress" / out.name)]
    if ctx.mock:
        cmd += ["--cpu-smoke", "--model", str(ctx.mock_tiny_model)]
        if os.environ.get("T2D4_SIM_STOP_AT_STEP") and out.name == "train":  # CPU simulation of a disconnect (mock runs only)
            cmd += ["--stop-at-step", os.environ["T2D4_SIM_STOP_AT_STEP"]]
    elif ctx.base_source != "hub":
        cmd += ["--model", str(ctx.base_source)]
    return cmd + extra


def _run_train(ctx: Ctx, cmd: List[str], out: Path, live: bool) -> int:
    env = dict(os.environ, PYTHONPATH=str(ctx.code) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    log = ctx.p("logs", f"{out.name}.log")
    if live:
        os.environ["PYTHONPATH"] = env["PYTHONPATH"]
        return PG.live_run(cmd, ctx.local / "progress" / out.name / "events.jsonl", log, pid_path=ctx.p("logs", f"{out.name}.pid"))
    with log.open("a", encoding="utf-8") as lf:
        return subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env).returncode


def stage_09_qlora_smoke(ctx: Ctx, live: bool = True) -> dict:
    free_backend(ctx)
    out = ctx.run / "smoke"
    extra = ["--smoke", "--max-steps", "3", "--save-steps", "3"] if not ctx.mock else ["--max-steps", "2", "--save-steps", "2"]
    code = _run_train(ctx, _train_cmd(ctx, out, extra), out, live)
    m = json.loads((out / "training_metrics.json").read_text(encoding="utf-8")) if (out / "training_metrics.json").exists() else {}
    ok = (code == 0 and m.get("global_step") == m.get("max_steps") and m.get("last_train_loss_finite") and m.get("last_train_loss_source") == "log_history.loss"
          and (out / f"checkpoint-{m.get('global_step')}").exists())
    if not ok:
        raise RuntimeError(f"QLORA_SMOKE_FAILED (exit {code}, loss {m.get('last_train_loss')} from {m.get('last_train_loss_source')}); see {ctx.p('logs', 'smoke.log')}")
    return mark(ctx, "09_qlora_smoke", exit_code=code, last_train_loss=m.get("last_train_loss"), loss_source=m.get("last_train_loss_source"),
                peak_vram_gib=m.get("gpu_mem_peak_gb"))


def stage_10_training(ctx: Ctx, live: bool = True) -> dict:
    free_backend(ctx)
    out = ctx.run / "train"
    extra = [] if not ctx.mock else ["--save-steps", "3", "--max-steps", "12"]
    code = _run_train(ctx, _train_cmd(ctx, out, extra), out, live)
    m = json.loads((out / "training_metrics.json").read_text(encoding="utf-8")) if (out / "training_metrics.json").exists() else {}
    if code != 0 or not m.get("completed"):
        raise RuntimeError(f"TRAINING_INCOMPLETE (exit {code}): rerun this cell to resume from the newest complete checkpoint")
    if not m.get("trainer_max_steps_equals_expected"):
        raise RuntimeError(f"CONFIG_MISMATCH: the trainer ran {m.get('max_steps')} steps, the frozen plan says {m.get('expected_steps')}")
    return mark(ctx, "10_training", exit_code=code, global_step=m.get("global_step"), first_loss=m.get("first_train_loss"), last_loss=m.get("last_train_loss"),
                eval_losses=m.get("eval_losses"), median_seconds_per_step=m.get("median_seconds_per_step"), peak_vram_gib=m.get("gpu_mem_peak_gb"),
                resumed_from=m.get("resumed_from"))


# ---------------------------------------------------------------- 11-12: probe + selection
def _train_cfg(ctx: Ctx) -> dict:
    import yaml
    return yaml.safe_load((ctx.code / CONFIG).read_text(encoding="utf-8"))


def candidate_steps(ctx: Ctx) -> List[int]:
    tm = json.loads((ctx.run / "train" / "training_metrics.json").read_text(encoding="utf-8"))
    saved = sorted(int(p.name.split("-")[1]) for p in (ctx.run / "train").glob("checkpoint-*") if p.name.split("-")[1].isdigit())
    return SL.candidate_steps(saved, tm["global_step"], _train_cfg(ctx)["d4_candidate_fractions"])


def _agg(items: Dict[str, dict], rows: List[dict]) -> dict:
    return M.aggregate([M.score(items[r["episode_id"]], r["output"], hit_max=r.get("hit_max"), raw_special=r.get("raw_special")) for r in rows])


def stage_11_checkpoint_probe(ctx: Ctx) -> dict:
    steps = candidate_steps(ctx)
    b = load_backend(ctx)
    jobs = []
    for s in steps:
        b.add_adapter(f"ckpt-{s}", ctx.run / "train" / f"checkpoint-{s}")
        jobs += _plan(ctx, "PROBE", [f"ckpt-{s}"])
    rows = generate(ctx, "11_probe", jobs)
    base_rows = EJ.fan_out(_plan(ctx, "PROBE", ["base"]), IO.read_jsonl(ctx.p("generations", "08_base.jsonl"))[0])
    items = {e["episode_id"]: e for e in _sets(ctx)["PROBE"]}
    table = {"base": _agg(items, base_rows)}
    for s in steps:
        table[str(s)] = _agg(items, [r for r in rows if r["model"] == f"ckpt-{s}"])
    tm = json.loads((ctx.run / "train" / "training_metrics.json").read_text(encoding="utf-8"))
    IO.atomic_write_json(ctx.p("selection", "checkpoint_probe.json"), {"candidates": steps, "table": table, "eval_losses": tm.get("eval_losses", [])})
    for s in steps:
        b.drop_adapter(f"ckpt-{s}")
    return mark(ctx, "11_checkpoint_probe", candidates=steps, core={k: v.get("core_grounding_score") for k, v in table.items()})


def stage_12_selection(ctx: Ctx) -> dict:
    pr = json.loads(ctx.p("selection", "checkpoint_probe.json").read_text(encoding="utf-8"))
    cands = {int(k): v for k, v in pr["table"].items() if k != "base"}
    sel = SL.select(pr["table"]["base"], cands, pr["eval_losses"], float(_train_cfg(ctx).get("d4_overfit_margin", 0.05)))
    src = ctx.run / "train" / f"checkpoint-{sel['chosen_step']}"
    dst, tmp = ctx.run / "best_adapter", ctx.run / "best_adapter.partial"
    if dst.exists():
        old = json.loads((dst / "selection_meta.json").read_text(encoding="utf-8"))
        if old["step"] != sel["chosen_step"] or old["files"]["adapter_model.safetensors"] != IO.sha256_file(src / "adapter_model.safetensors"):
            raise RuntimeError("HASH_MISMATCH: best_adapter already holds another checkpoint; selection is never redone in place (use a new RUN_NAME)")
        sel["best_adapter"] = old
        IO.atomic_write_json(ctx.p("selection", "checkpoint_selection.json"), sel)
        return mark(ctx, "12_selection", chosen_step=sel["chosen_step"], status=sel["status"], adapter_sha256=old["files"]["adapter_model.safetensors"], verified=True)
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for f in ("adapter_model.safetensors", "adapter_config.json"):
        shutil.copy2(src / f, tmp / f)
    meta = {"step": sel["chosen_step"], "status": sel["status"], "files": {f: IO.sha256_file(tmp / f) for f in ("adapter_model.safetensors", "adapter_config.json")},
            "selected_on": "DEV_PROBE", "rules_sha256": SL.RULES_SHA256}
    (tmp / "selection_meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    tmp.rename(dst)
    sel["best_adapter"] = meta
    IO.atomic_write_json(ctx.p("selection", "checkpoint_selection.json"), sel)
    return mark(ctx, "12_selection", chosen_step=sel["chosen_step"], status=sel["status"], adapter_sha256=meta["files"]["adapter_model.safetensors"])


# ---------------------------------------------------------------- 13-15: adapter generation
def _ensure_best(ctx: Ctx):
    b = load_backend(ctx)
    meta = json.loads((ctx.run / "best_adapter" / "selection_meta.json").read_text(encoding="utf-8"))
    if not ctx.mock and IO.sha256_file(ctx.run / "best_adapter" / "adapter_model.safetensors") != meta["files"]["adapter_model.safetensors"]:
        raise RuntimeError("HASH_MISMATCH: best_adapter/adapter_model.safetensors differs from selection_meta.json")
    b.add_adapter("best", ctx.run / "best_adapter")
    return b, meta


def stage_13_adapter_dev(ctx: Ctx) -> dict:
    _ensure_best(ctx)
    rows = generate(ctx, "13_dev", _plan(ctx, "DEV", ["best"]))
    return mark(ctx, "13_adapter_dev", rows=len(rows))


def stage_14_adapter_safety(ctx: Ctx) -> dict:
    _ensure_best(ctx)
    rows = generate(ctx, "14_safety", _plan(ctx, "SAFETY", ["best"]))
    return mark(ctx, "14_adapter_safety", rows=len(rows))


def stage_15_holdout(ctx: Ctx) -> dict:
    """FROZEN_HOLDOUT synthetic, Base + the selected adapter, ONCE. A second adapter is refused (HOLDOUT_ALREADY_USED)."""
    _, meta = _ensure_best(ctx)
    lock = ctx.p("generations", "15_holdout.ADAPTER.json")
    sha = meta["files"]["adapter_model.safetensors"]
    if lock.exists() and json.loads(lock.read_text(encoding="utf-8"))["adapter_sha256"] != sha:
        raise RuntimeError("HOLDOUT_ALREADY_USED: FROZEN_HOLDOUT was already generated for another adapter; it is run once per run")
    if not lock.exists():
        IO.atomic_write_json(lock, {"adapter_sha256": sha, "step": meta["step"], "locked_at": now()})
    rows = generate(ctx, "15_holdout", _plan(ctx, "HOLDOUT", ["base", "best"]))
    return mark(ctx, "15_holdout", rows=len(rows), adapter_sha256=sha)


# ---------------------------------------------------------------- 16: statistics + verdict
def all_rows(ctx: Ctx) -> List[dict]:
    gens = {}
    for f in sorted(ctx.run.glob("generations/*.jsonl")):
        for r in IO.read_jsonl(f)[0]:
            gens.setdefault(r["job_id"], r)
    jobs = []
    for s in ("DEV", "SAFETY", "HOLDOUT"):
        jobs += _plan(ctx, s, ["base", "best"])
    return EJ.fan_out(jobs, list(gens.values()))


def failure_cases(items: Dict[str, Dict[str, dict]], rows: List[dict], k: int = 60) -> List[dict]:
    """Leaks, repetitions, empty / special-token replies of B (FULL). Restricted values are replaced by their Context short hash."""
    from ..vendor import context_disclosure_v1_0 as DS
    out = []
    for r in sorted(rows, key=lambda x: x["job_id"]):
        if r["model"] != "best" or r["variant"] != "FULL" or r["seed"] != EJ.PRIMARY:
            continue
        it = items[r["set"]][r["episode_id"]]
        sc = M.score(it, r["output"], hit_max=r.get("hit_max"), raw_special=r.get("raw_special"))
        kinds = [k_ for k_ in ("serious_leak", "repetition", "empty", "special_token_or_think_leak", "verbose_assistant") if sc.get(k_)]
        if kinds:
            terms = GD.terms(it)
            out.append({"job_id": r["job_id"], "set": r["set"], "category": it.get("category"), "kinds": kinds, "output": DS.scrub(r["output"], terms)})
    return out[:k]


def stage_16_statistics(ctx: Ctx) -> dict:
    rows = all_rows(ctx)
    items = {s: {e["episode_id"]: e for e in v} for s, v in _sets(ctx).items() if s != "PROBE"}
    sel = json.loads(ctx.p("selection", "checkpoint_selection.json").read_text(encoding="utf-8"))
    eng = {"complete": all(done(ctx, s) for s in STAGES[:16]), "dry_run": ctx.mock}
    r = VD.compute_results(items, rows, sel, eng)
    IO.atomic_write_json(ctx.p("results", "results.json"), r)
    IO.atomic_write_json(ctx.p("results", "verdict.json"), r["verdict"])
    IO.write_jsonl_atomic(ctx.p("results", "failure_cases.jsonl"), failure_cases(items, rows))
    v = r["verdict"]
    return mark(ctx, "16_statistics", VERDICT=v["VERDICT"], gates=v["gate_status"], selection=v["selection_status"])


# ---------------------------------------------------------------- 17: export
def stage_17_export(ctx: Ctx) -> dict:
    stage = ctx.local / "export"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "outputs").mkdir(parents=True)
    gens = {}
    for f in sorted(ctx.run.glob("generations/*.jsonl")):
        for r in IO.read_jsonl(f)[0]:
            gens.setdefault(r["job_id"], r)
    IO.write_jsonl_atomic(stage / "outputs" / "all_generations.jsonl", sorted(gens.values(), key=lambda x: x["job_id"]))
    fan = all_rows(ctx)
    IO.write_jsonl_atomic(stage / "outputs" / "base_generations.jsonl", [r for r in fan if r["model"] == "base"])
    IO.write_jsonl_atomic(stage / "outputs" / "adapter_generations.jsonl", [r for r in fan if r["model"] == "best"])
    for sub in ("results", "selection", "gate0", "stages", "progress"):
        if (ctx.run / sub).exists():
            shutil.copytree(ctx.run / sub, stage / sub)
    res = json.loads(ctx.p("results", "results.json").read_text(encoding="utf-8"))
    R_ = stage / "results"
    grp = lambda keys: {s: {g: v["groups"].get(g) for g in keys} for s, v in res["sets"].items()}
    IO.atomic_write_json(R_ / "abcd_metrics.json", grp(("A", "B", "C", "D")))
    IO.atomic_write_json(R_ / "base_length_metrics.json", {"DEV": res["sets"].get("DEV", {}).get("groups", {}).get("BASE_LENGTH")})
    IO.atomic_write_json(R_ / "safety.json", {"gate": res["gates"].get("serious_leak"), "guard_C": {s: v["groups"]["C"]["guard"] for s, v in res["sets"].items()},
                                              "guard_D": {s: v["groups"]["D"]["guard"] for s, v in res["sets"].items()},
                                              "contrasts": {s: {k: v["contrasts"].get(k) for k in ("over_refusal", "decline_expressed", "permission_expressed",
                                                                                                    "unsupported_claim")} for s, v in res["sets"].items()}})
    IO.atomic_write_json(R_ / "temporal.json", {"gate": res["gates"].get("temporal"), "contrasts": {s: {k: v["contrasts"].get(k) for k in VD.TEMPORAL}
                                                                                                  for s, v in res["sets"].items()}})
    IO.atomic_write_json(R_ / "persona_style.json", {"gates": {k: res["gates"].get(k) for k in ("style", "profile_dependence")},
                                                     "style": res["sets"].get("DEV", {}).get("style"),
                                                     "profile_dependence": res["sets"].get("DEV", {}).get("profile_dependence")})
    IO.atomic_write_json(R_ / "ablations.json", {g: v for g, v in res["sets"].get("DEV", {}).get("groups", {}).items() if ":" in g} |
                         {"exploratory": res["sets"].get("DEV", {}).get("exploratory")})
    IO.atomic_write_json(R_ / "confidence_intervals.json", {s: v["contrasts"] for s, v in res["sets"].items()})
    IO.atomic_write_json(R_ / "failure_cases_index.json", [{k: x[k] for k in ("job_id", "set", "category", "kinds")}
                                                            for x in IO.read_jsonl(ctx.p("results", "failure_cases.jsonl"))[0]])
    (stage / "train").mkdir()
    for f in ("resolved_train_config.yaml", "training_log.json", "training_metrics.json", "checkpoint_manifest.json", "run_fingerprint.json", "metrics_log.jsonl"):
        if (ctx.run / "train" / f).exists():
            shutil.copy2(ctx.run / "train" / f, stage / "train" / f)
    if (ctx.run / "smoke" / "training_metrics.json").exists():
        shutil.copy2(ctx.run / "smoke" / "training_metrics.json", stage / "train" / "smoke_training_metrics.json")
    shutil.copytree(ctx.run / "best_adapter", stage / "best_adapter")
    gp = ctx.p("gate0", "gpu_preflight.json")
    gpu = json.loads(gp.read_text(encoding="utf-8")) if gp.exists() else {"gpu": None}
    rc = json.loads(ctx.p("run_config.json").read_text(encoding="utf-8"))
    IO.atomic_write_json(stage / "reproducibility.json", {"versions": versions(), "run_config": rc, "generation": EJ.GEN, "selection_rules_sha256": SL.RULES_SHA256,
                                                         "verdict_rules_sha256": VD.RULES_SHA256, "train_seed": 42,
                                                         "data_manifest_sha256": IO.sha256_file(ctx.data / "DATA_MANIFEST.json")})
    v = json.loads(ctx.p("results", "verdict.json").read_text(encoding="utf-8"))
    man = {"version": RUN_VERSION, "release_version": RELEASE_VERSION, "run_name": ctx.run_name, "DRY_RUN": ctx.mock, "created_at": now(),
           "gpu": gpu.get("gpu"), "verdict": v["VERDICT"], "gate_status": v["gate_status"], "labels": VD.LABELS,
           "best_adapter_sha256": json.loads((ctx.run / "best_adapter" / "selection_meta.json").read_text(encoding="utf-8"))["files"]["adapter_model.safetensors"],
           "release_manifest_sha256": rc["release_manifest_sha256"], "generation_rows": len(gens),
           "hashes": {"code_manifest_sha256": IO.sha256_file(ctx.code / "CODE_MANIFEST.json"), "data_manifest_sha256": IO.sha256_file(ctx.data / "DATA_MANIFEST.json"),
                      "renderer_sha256": json.loads((ctx.data / "meta" / "dataset_manifest.json").read_text(encoding="utf-8"))["renderer_sha256"],
                      "selection_rules_sha256": SL.RULES_SHA256, "verdict_rules_sha256": VD.RULES_SHA256},
           "privacy": "PRIVATE: outputs/all_generations.jsonl holds raw model replies (synthetic values; replies to real DEV_EVAL style cases). "
                      "Never public Git. failure_cases.jsonl has restricted values replaced by short hashes."}
    IO.atomic_write_json(stage / "run_manifest.json", man)
    sums = [f"{IO.sha256_file(p)}  {p.relative_to(stage).as_posix()}" for p in sorted(stage.rglob("*")) if p.is_file()]
    (stage / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    exp = ctx.run / "exports"
    exp.mkdir(parents=True, exist_ok=True)
    zp, az = exp / "t2d4_results.zip", exp / "t2d4_best_adapter.zip"
    with zipfile.ZipFile(str(zp) + ".partial", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(stage).as_posix())
    os.replace(str(zp) + ".partial", zp)
    ad = ctx.local / "export_adapter"
    if ad.exists():
        shutil.rmtree(ad)
    shutil.copytree(ctx.run / "best_adapter", ad)
    IO.atomic_write_json(ad / "BASE_REVISION.json", {"base_model": "Qwen/Qwen3-8B", "revision": EJ.GEN["model_revision"], "quantization": EJ.GEN["quantization"],
                                                     "chat_template_sha256": json.loads(ctx.p("gate0", "gpu_gate.json").read_text(encoding="utf-8")).get("template_sha256")
                                                     if ctx.p("gate0", "gpu_gate.json").exists() else None, "enable_thinking": False,
                                                     "selection_status": v["selection_status"], "deployable": False if v["selection_status"] == "SELECTED_UNQUALIFIED" else None,
                                                     "smoke_adapter": False, "DRY_RUN": ctx.mock})
    (ad / "SHA256SUMS.txt").write_text("".join(f"{IO.sha256_file(p)}  {p.name}\n" for p in sorted(ad.iterdir()) if p.name != "SHA256SUMS.txt"), encoding="utf-8")
    with zipfile.ZipFile(str(az) + ".partial", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ad.iterdir()):
            z.write(p, f"best_adapter/{p.name}")
    os.replace(str(az) + ".partial", az)
    info = {"run_name": ctx.run_name, "DRY_RUN": ctx.mock, "verdict": v["VERDICT"],
            "results_zip": {"name": zp.name, "sha256": IO.sha256_file(zp), "bytes": zp.stat().st_size},
            "adapter_zip": {"name": az.name, "sha256": IO.sha256_file(az), "bytes": az.stat().st_size}, "exported_at": now()}
    IO.atomic_write_json(exp / "EXPORT_INFO.json", info)
    return mark(ctx, "17_export", **info)


def run_all(ctx: Ctx, upto: str = "17_export", live: bool = False) -> None:
    """Every stage in order (the CPU mock E2E; the notebook calls them cell by cell)."""
    for s in STAGES[: STAGES.index(upto) + 1]:
        if s in ("09_qlora_smoke", "10_training"):
            call(ctx, s, live)
        else:
            call(ctx, s)


def with_panel(ctx: Ctx, name: str, *args, refresh: float = 15.0, **kw):
    """Run one stage while a single display shows the 14 run fields, the stage table and generation progress (real files only)."""
    import threading
    try:
        from IPython.display import HTML, display
        import html as _h
        handle = display(HTML("<pre>starting...</pre>"), display_id=True)
        show = lambda extra="": handle.update(HTML(f"<pre style='font-size:12px'>{_h.escape(status_table(ctx) + extra)}</pre>"))
    except ImportError:
        show = lambda extra="": None
    stop = threading.Event()
    t0 = time.time()

    def loop():
        while not stop.wait(refresh):
            try:
                show(f"\n  elapsed in this cell: {int(time.time() - t0)} s")
            except Exception:  # noqa: BLE001 - the panel never breaks a stage
                pass
    th = threading.Thread(target=loop, daemon=True)
    th.start()
    try:
        return call(ctx, name, *args, **kw)
    finally:
        stop.set()
        show(f"\n  cell finished after {int(time.time() - t0)} s")
