"""Cell sources of ``StillHere_Persona_L4.ipynb`` (generated; edit here, not in the .ipynb). Generic: no private path, token, account or
chat text. ``stillhere persona package`` writes a copy with that package's id / target filled into STAGE 00; the repository copy
(``notebooks/StillHere_Persona_L4.ipynb``) is the same generator with placeholders.

    python -m stillhere.persona.notebook            # (re)write notebooks/StillHere_Persona_L4.ipynb
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

INTRO = """# StillHere · Persona QLoRA（Colab L4 / A100）

把 `stillhere persona package` 生成的训练包上传到 Google Drive，然后在这里按顺序运行。每个 STAGE 打印 `PASS / FAIL / BLOCKED / SKIP`，状态写入 Drive。

| 阶段 | 做什么 |
|---|---|
| 00–04 | 配置、环境、导入训练包、SHA256 校验、安装固定版本依赖 |
| 05–06 | 校验 Qwen3-8B（固定 revision、tokenizer / chat template sha）、GPU Preflight（实测 NF4、显存、bf16、版本） |
| 07 | 用真实 LLaMA-Factory 逐 token 检查 train + dev **全部行**的 Loss Mask（不通过就不训练） |
| 08 | 2 步 GPU Smoke（第 1 步后模拟断线再续训），记录显存峰值 |
| 09–10 | 正式 QLoRA 训练（1 epoch）；检查点状态与续训 |
| 11 | Dev 上按预注册规则选检查点（Base+Persona Prompt 对照；loss 不参与选择） |
| 12–14 | 导出人物 Adapter 包、Run Manifest、结果 ZIP |

- **断线后**：重新连接同一种 GPU → 从 STAGE 00 重新运行。训练从最新**完整**检查点续训；配置或数据不同会被拒绝；已经训练完的不会再训。
- 训练包里有**真实私聊**（目标人物的回复，以及对方的消息作为上下文）。只有你确认可以放到云端时，才在 STAGE 00 设 `CONFIRM_PRIVATE_DATA_ON_CLOUD = True`。
- Test 集不在训练包里；Test 评测在你本地电脑上、配置锁定后进行一次。
"""

STAGE00 = r'''# STAGE 00 · 配置（只改这里）
DRIVE_DIR = "/content/drive/MyDrive/StillHere/persona"   # 你在 Drive 上放训练包的目录（可改）
PACKAGE_ID = "__PACKAGE_ID__"                             # 训练包 ID（= Drive 上 DRIVE_DIR 下的文件夹名）
RUN_ID = "__RUN_ID__"                                     # 一次训练一个 ID；改任何配置都要换 ID
GPU_TARGET = "__GPU_TARGET__"                             # "colab-l4"（默认）或 "colab-a100"
CONFIRM_PRIVATE_DATA_ON_CLOUD = False                     # 训练包含真实私聊：你确认可以在云端处理后改成 True
ALLOW_OTHER_GPU = False                                   # True：允许非目标 GPU（结果会标明）
RUN_DEV_ABLATION = False                                  # True：选完检查点后在 Dev 上再跑 A / D / E（额外约 1 小时）
EXPORT_UNQUALIFIED = False                                # True：没有检查点通过预注册规则时仍导出最后一个（标为 UNQUALIFIED，只用于诊断）

import os, sys, json, hashlib, subprocess, shutil, time, zipfile
from pathlib import Path
assert GPU_TARGET in ("colab-l4", "colab-a100"), GPU_TARGET
assert CONFIRM_PRIVATE_DATA_ON_CLOUD, "训练包含真实私聊。确认可以在云端处理后，把 CONFIRM_PRIVATE_DATA_ON_CLOUD 改成 True 再运行。"
STATUS = {}
def stage(name, status, **info):
    STATUS[name] = {"status": status, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), **info}
    try:
        (RUN / "STAGE_STATUS.json").write_text(json.dumps(STATUS, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass
    print(f"\n===== {name}: {status} =====", json.dumps(info, ensure_ascii=False)[:900] if info else "")
print("STAGE 00: PASS (config)", PACKAGE_ID, RUN_ID, GPU_TARGET)
'''

STAGE01 = r'''# STAGE 01 · Environment Check（Colab、挂载 Drive、目录）
from google.colab import drive
drive.mount("/content/drive")
ROOT = Path(DRIVE_DIR) / PACKAGE_ID
IN, RUN, CKPT = ROOT / "input", ROOT / "runs" / RUN_ID, ROOT / "checkpoints" / RUN_ID
EXPORTS = RUN / "exports"
LOCAL = Path("/content/persona")
CODE, DATA, HF_HOME = LOCAL / "code", LOCAL / "data", LOCAL / "hf"
for p in (RUN, CKPT, EXPORTS, LOCAL, HF_HOME):
    p.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_HOME)
prev = RUN / "STAGE_STATUS.json"
if prev.is_file():
    STATUS.update(json.loads(prev.read_text(encoding="utf-8")))
    print("已有阶段状态（续跑）：", {k: v["status"] for k, v in STATUS.items()})
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def sh(cmd, log=None):
    print("$", cmd)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
           "PYTHONPATH": os.pathsep.join(str(CODE / x) for x in ("training/persona", "training/d5", "training/d4", "src"))}
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    buf = []
    for line in p.stdout:
        print(line, end=""); buf.append(line)
        if log and len(buf) >= 20:
            with open(log, "a", encoding="utf-8") as f: f.writelines(buf)
            buf.clear()
    rc = p.wait()
    if log and buf:
        with open(log, "a", encoding="utf-8") as f: f.writelines(buf)
    return rc
stage("STAGE 01 environment", "PASS", root=str(ROOT), run=str(RUN), checkpoints=str(CKPT), python=sys.version.split()[0])
'''

STAGE02 = r'''# STAGE 02 · Dataset Import（在 Drive 上找到训练包的 4 个文件）
need = ["persona_code.zip", "persona_data.zip", "RELEASE_MANIFEST.json", "SHA256SUMS.txt"]
missing = [n for n in need if not (IN / n).is_file()]
for n in need:
    if (IN / n).is_file(): print("found", n, f"{(IN / n).stat().st_size / 2**20:.2f} MB")
stage("STAGE 02 import", "FAIL" if missing else "PASS", missing=missing, input=str(IN))
assert not missing, f"把训练包的文件放到 {IN}"
'''

STAGE03 = r'''# STAGE 03 · SHA256 Verification（文件、ZIP CRC、训练包内容：无 Test、无原始导出）
sums = dict((l.split(None, 1)[1].strip(), l.split(None, 1)[0]) for l in (IN / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines() if l.strip())
problems = [f"SHA256 MISMATCH {n}" for n in ("persona_code.zip", "persona_data.zip", "RELEASE_MANIFEST.json") if sha256(IN / n) != sums.get(n)]
RM = json.loads((IN / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
for n in ("persona_code.zip", "persona_data.zip"):
    with zipfile.ZipFile(IN / n) as z:
        if z.testzip() is not None: problems.append(f"ZIP CRC {n}")
        names = [x for x in z.namelist() if not x.endswith("/")]
        if n == "persona_data.zip":
            if sorted(names) != sorted(RM["data_zip_files"]): problems.append("data ZIP content differs from RELEASE_MANIFEST")
            if any(x.split("/")[-1].startswith("test") or x.endswith((".json",)) and "weflow" in x.lower() for x in names): problems.append("data ZIP contains test / raw export")
if RM.get("package_id") != PACKAGE_ID: problems.append(f"RELEASE_MANIFEST is package {RM.get('package_id')}, STAGE 00 says {PACKAGE_ID}")
stage("STAGE 03 sha256", "FAIL" if problems else "PASS", problems=problems, package=RM.get("package_id"), dataset_version=RM.get("dataset_version"),
      card_id=RM.get("card_id"), persona_prompt_sha=RM.get("persona_prompt_sha256", "")[:16])
assert not problems, problems
'''

STAGE04 = r'''# STAGE 04 · Install Compatible Dependencies（固定版本）+ 解压代码与数据
for zname, target in (("persona_code.zip", CODE), ("persona_data.zip", DATA)):
    marker = target / ".zip_sha256"
    if not marker.is_file() or marker.read_text().strip() != sums[zname]:
        shutil.rmtree(target, ignore_errors=True)
        with zipfile.ZipFile(IN / zname) as z: z.extractall(target)
        marker.write_text(sums[zname])
        print("extracted", zname)
        if zname == "persona_code.zip" and any(m.split(".")[0] in ("stillhere", "personatrain", "d5pilot", "t2d4") for m in sys.modules):
            stage("STAGE 04 dependencies", "BLOCKED", reason="新代码已解压但旧代码还在内存：Runtime → Restart session，然后从 STAGE 00 重新运行")
            raise SystemExit("restart the session")
rc = sh("pip -q install " + " ".join(f'"{p}"' for p in RM["pins_train"]))
sys.path[:0] = [str(CODE / x) for x in ("training/persona", "training/d5", "training/d4", "src")]
import importlib
vers = {m: importlib.import_module(m).__version__ for m in ("transformers", "peft", "bitsandbytes", "accelerate", "tokenizers", "torch", "llamafactory")}
ok = rc == 0 and all(vers.get(k) == v for k, v in RM["expect_versions"].items())
stage("STAGE 04 dependencies", "PASS" if ok else "FAIL", versions=vers)
assert ok, "依赖版本不符：如果刚安装完，请 Runtime → Restart session，然后从 STAGE 00 重新运行"
'''

STAGE05 = r'''# STAGE 05 · Qwen3-8B Base Verification（固定 revision；tokenizer 与 chat template sha256）
from huggingface_hub import snapshot_download
from stillhere.persona import doctor as DR
BASE = Path(snapshot_download(RM["base"]["id"], revision=RM["base"]["revision"], cache_dir=str(HF_HOME / "hub"),
                              allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json", "tokenizer*"]))
bc = DR.base_check(BASE)
ok = bc["tokenizer_sha256_ok"] and bc["chat_template_sha256_ok"] and bc["revision_in_path"] and bc["weight_shards"] >= 1
stage("STAGE 05 base", "PASS" if ok else "FAIL", **bc)
assert ok
'''

STAGE06 = r'''# STAGE 06 · GPU Preflight（torch、CUDA、GPU 型号、显存、bf16、bitsandbytes NF4 实测、版本、磁盘、数据与配置 hash）
CFG_PATH = DATA / "config" / "training_config.yaml"
rep = DR.doctor(GPU_TARGET, base_dir=BASE, data_dir=DATA, config_path=CFG_PATH, work_dir="/content", allow_other_gpu=ALLOW_OTHER_GPU)
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout[:1200])
cfg_ok = rep.get("config_sha256") == RM["training_config_sha256"]
stage("STAGE 06 gpu preflight", "PASS" if rep["status"] == "PASS" and cfg_ok else "BLOCKED", blocked=rep["blocked"] + ([] if cfg_ok else ["config sha differs"]),
      gpu=rep["gpu"].get("gpu"), vram_gib=rep["gpu"].get("vram_total_gib"), bf16=rep["gpu"].get("bf16"), nf4=rep["gpu"].get("nf4_roundtrip_ok"),
      disk_free_gib=rep["disk_free_gib"], versions=rep["versions"])
assert rep["status"] == "PASS" and cfg_ok, "GPU / 环境不满足：换成 L4（或 A100）运行时，或按提示处理"
'''

STAGE07 = r'''# STAGE 07 · Loss Mask Verification（真实 LLaMA-Factory 预处理；train + dev 全部行；推理模板一致性；负对照）
LF_RESULT = RUN / "LF_LABELS_RESULT.json"
if not LF_RESULT.is_file() or json.loads(LF_RESULT.read_text()).get("LLAMAFACTORY_LABELS") != "PASS":
    work = LOCAL / "lf_work"
    shutil.rmtree(work, ignore_errors=True)
    rc = sh(f"python -m personatrain.lf_labels --data {DATA} --model-dir {BASE} --work {work} --out {RUN} --cutoff 2048 --splits train,dev "
            f"--persona-prompt {DATA / 'persona' / 'persona_prompt.txt'}", log=RUN / "lf_labels.log")
lf = json.loads(LF_RESULT.read_text(encoding="utf-8"))
stage("STAGE 07 loss mask", lf["LLAMAFACTORY_LABELS"], **{s: {k: v[k] for k in ("rows", "status", "failures_total", "negative_control_caught", "max_input_ids")}
                                                         for s, v in lf["splits"].items()})
assert lf["LLAMAFACTORY_LABELS"] == "PASS", "Loss Mask 审计没有通过：禁止训练"
'''

STAGE08 = r'''# STAGE 08 · Short GPU Smoke（最长的 16 条训练样本，2 步；第 1 步后模拟断线再续训；记录显存峰值）
SMOKE = RUN / "smoke"
base_cmd = f"python -m personatrain.train_persona --data {DATA} --config {CFG_PATH} --output-dir {SMOKE} --smoke --dataset persona_smoke --max-steps 2 --save-steps 1"
m = json.loads((SMOKE / "training_metrics.json").read_text()) if (SMOKE / "training_metrics.json").is_file() else {}
if not m.get("completed"):
    rc1 = sh(base_cmd + " --stop-at-step 1", log=RUN / "smoke.log")
    rc2 = sh(base_cmd, log=RUN / "smoke.log")
    m = json.loads((SMOKE / "training_metrics.json").read_text()) if (SMOKE / "training_metrics.json").is_file() else {}
log = (RUN / "smoke.log").read_text(encoding="utf-8", errors="ignore") if (RUN / "smoke.log").is_file() else ""
oom = "out of memory" in log.lower() or "OutOfMemoryError" in log
resumed = m.get("resumed_from") == "checkpoint-1" and m.get("trainer_began_at_step") == 1
info = {k: m.get(k) for k in ("global_step", "first_train_loss", "last_train_loss", "gpu_mem_peak_gb", "median_seconds_per_step", "resumed_from")}
if oom:
    stage("STAGE 08 smoke", "FAIL", reason="OOM。不要改训练标签、不要换 Base；可改的只有资源参数（例如换 A100 运行时，或在本地重新打包更小的 cutoff 并换 RUN_ID）", **info)
    raise SystemExit("OOM")
ok = bool(m.get("completed")) and resumed and m.get("last_train_loss_finite")
stage("STAGE 08 smoke", "PASS" if ok else "FAIL", resume_verified=resumed, **info)
assert ok
'''

STAGE09 = r'''# STAGE 09 · QLoRA Training（1 epoch；按 save_steps 保存检查点 + dev loss；断线后重跑本 STAGE 自动续训；已完成不会再训）
t0 = time.time()
rc = sh(f"python -m personatrain.train_persona --data {DATA} --config {CFG_PATH} --output-dir {CKPT} --lf-result {LF_RESULT} "
        f"--progress-dir {LOCAL / 'progress'}", log=RUN / "train.log")
m = json.loads((CKPT / "training_metrics.json").read_text()) if (CKPT / "training_metrics.json").is_file() else {}
info = {k: m.get(k) for k in ("global_step", "max_steps", "epoch", "first_train_loss", "last_train_loss", "eval_losses", "gpu_mem_peak_gb",
                              "median_seconds_per_step", "resumed_from", "run_fingerprint")}
if (CKPT / "training_metrics.json").is_file(): shutil.copy2(CKPT / "training_metrics.json", RUN / "training_metrics.json")
stage("STAGE 09 train", "PASS" if rc == 0 and m.get("completed") else "FAIL", this_session_seconds=round(time.time() - t0), **info)
assert rc == 0 and m.get("completed"), "训练未完成：断线 / 中断后从 STAGE 00 重新运行，会从最新完整检查点续训"
'''

STAGE10 = r'''# STAGE 10 · Checkpoint Save / Resume（检查点完整性；续训信息）
from t2d4.colab import train_d4 as T
cks = T.checkpoints(CKPT)
table = [{"checkpoint": c.name, "complete": not T.missing_files(c), "missing": T.missing_files(c)} for c in cks]
cands = [f"checkpoint-{s}" for s in json.loads((DATA / "PERSONA_TRAINING_MANIFEST.json").read_text())["candidate_steps"]]
ok = all(any(t["checkpoint"] == c and t["complete"] for t in table) for c in cands)
stage("STAGE 10 checkpoints", "PASS" if ok else "FAIL", checkpoints=table, candidates=cands,
      resume="断线后从 STAGE 00 重新运行：train_persona 只从最新的完整检查点续训；指纹不同会拒绝")
assert ok, "候选检查点不完整"
'''

STAGE11 = r'''# STAGE 11 · Dev Evaluation + Checkpoint Selection（B = Base + Persona Prompt；候选 = 同一 Prompt + 检查点 Adapter；预注册规则）
EVAL_SET = DATA / "eval" / "dev_eval.jsonl"
DEV = RUN / "eval" / "dev"
cand = [f"ckpt-{c.split('-')[1]}" for c in cands]
ad = " ".join(f"--adapter {g}={CKPT / ('checkpoint-' + g.split('-')[1])}" for g in cand)
rc = sh(f"python -m personatrain.evaluate generate --eval-set {EVAL_SET} --base-dir {BASE} {ad} --groups B,{','.join(cand)} --candidates {','.join(cand)} "
        f"--run-dir {DEV}", log=RUN / "dev_eval.log")
RULE = json.loads((DATA / "PERSONA_TRAINING_MANIFEST.json").read_text()).get("selection_rule", "r1")   # r1 / r2, fixed in the package
rc |= sh(f"python -m personatrain.evaluate select --eval-set {EVAL_SET} --run-dir {DEV} --train-dir {DATA / 'train'} --card {DATA / 'persona' / 'persona_card.yaml'} "
         f"--candidates {','.join(cand)} --rule {RULE}")
SEL = json.loads((DEV / "CHECKPOINT_SELECTION.json").read_text(encoding="utf-8"))
if RUN_DEV_ABLATION and SEL["status"] == "SELECTED":
    best = CKPT / ("checkpoint-" + SEL["selected"].split("-")[1])
    ABL = RUN / "eval" / "dev_ablation"
    rc |= sh(f"python -m personatrain.evaluate generate --eval-set {EVAL_SET} --base-dir {BASE} --adapter C={best} --adapter E={best} --groups A,B,C,D,E "
             f"--run-dir {ABL}", log=RUN / "dev_ablation.log")
    rc |= sh(f"python -m personatrain.evaluate score --eval-set {EVAL_SET} --run-dir {ABL} --train-dir {DATA / 'train'} --card {DATA / 'persona' / 'persona_card.yaml'} --groups A,B,C,D,E")
stage("STAGE 11 dev selection", "PASS" if rc == 0 and SEL["status"] == "SELECTED" else "FAIL", selection=SEL["status"], selected=SEL["selected"],
      candidates=SEL["candidates"], base=SEL["base"])
'''

STAGE12 = r'''# STAGE 12 · Adapter Export（人物包：adapter + persona_prompt + persona_card + manifest + dev 评测摘要；不含 Base 权重；不覆盖）
def adapter_zips():                                       # only adapter packages (never the results ZIP)
    return sorted(list(EXPORTS.glob("persona_candidate_*.zip")) + list(EXPORTS.glob("persona_UNQUALIFIED_diagnostic_*.zip")))
existing = adapter_zips()
if existing:
    print("已导出：", [p.name for p in existing]); rc = 0
elif SEL["status"] == "SELECTED":
    rc = sh(f"python -m personatrain.export adapter --run-dir {RUN} --train-dir {CKPT} --checkpoint checkpoint-{SEL['selected'].split('-')[1]} --run-id {RUN_ID} --data {DATA}")
elif EXPORT_UNQUALIFIED:
    rc = sh(f"python -m personatrain.export adapter --run-dir {RUN} --train-dir {CKPT} --checkpoint {cands[-1]} --run-id {RUN_ID} --data {DATA} --unqualified")
else:
    rc = 1
    print("没有检查点通过预注册规则（NO_QUALIFIED_CHECKPOINT）：不导出。请导出结果（STAGE 14）并如实报告。")
zips = adapter_zips()
stage("STAGE 12 export adapter", "PASS" if rc == 0 and zips else "FAIL", adapters=[f"{p.name} ({p.stat().st_size / 2**20:.1f} MB)" for p in zips])
'''

STAGE13 = r'''# STAGE 13 · Release Manifest（训练 Run 的完整记录：代码 / 数据 / Base / Prompt / GPU / Loss / 步数 / 时长 / 显存 / 检查点 / 导出 sha）
tm = json.loads((CKPT / "training_metrics.json").read_text(encoding="utf-8"))
ev = [json.loads(l) for l in (CKPT / "metrics_log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()] if (CKPT / "metrics_log.jsonl").is_file() else []
lrs = [e.get("learning_rate") for e in ev if e.get("learning_rate") is not None]
from datetime import datetime as _dt
times = [_dt.fromisoformat(e["at"]) for e in ev if e.get("at")]
run_manifest = {"schema": "stillhere.persona.run_manifest/1.0", "run_id": RUN_ID, "package_id": PACKAGE_ID, "profile_id": RM["profile_id"],
                "code_tree_sha256": RM["code_tree_sha256"], "git_head": RM["git_head"], "git_dirty": RM["git_dirty"], "dataset_version": RM["dataset_version"],
                "data_manifest_sha256": RM["data_manifest_sha256"], "base": RM["base"], "training_system_prompt_sha256": RM["training_system_prompt_sha256"],
                "persona_prompt_sha256": RM["persona_prompt_sha256"], "training_config_sha256": RM["training_config_sha256"], "gpu": tm.get("hardware") or tm.get("gpus_used"),
                "global_step": tm.get("global_step"), "max_steps": tm.get("max_steps"), "first_train_loss": tm.get("first_train_loss"), "last_train_loss": tm.get("last_train_loss"),
                "eval_losses": tm.get("eval_losses"), "learning_rate_first_last": [lrs[0], lrs[-1]] if lrs else None, "peak_vram_gb": tm.get("gpu_mem_peak_gb"),
                "train_wall_seconds_logged": round((max(times) - min(times)).total_seconds()) if len(times) > 1 else None,
                "checkpoints": [c.name for c in T.checkpoints(CKPT)], "checkpoint_dir": str(CKPT), "selection": SEL,
                "exports": {p.name: sha256(p) for p in adapter_zips()}, "lf_labels": lf["LLAMAFACTORY_LABELS"],
                "stage_status": {k: v["status"] for k, v in STATUS.items()}, "verified_on": "REAL_GPU"}
(RUN / "RUN_MANIFEST.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=1), encoding="utf-8")
stage("STAGE 13 run manifest", "PASS", path=str(RUN / "RUN_MANIFEST.json"), exports=list(run_manifest["exports"]))
'''

STAGE14 = r'''# STAGE 14 · Return Artifacts（结果 ZIP：不含权重 / 检查点 / 密钥；Adapter 包在 exports/）
rc = sh(f"python -m personatrain.export results --run-dir {RUN} --run-id {RUN_ID}")
res = sorted(EXPORTS.glob("persona_results_*.zip"))
stage("STAGE 14 return artifacts", "PASS" if rc == 0 and res else "FAIL", results=res[-1].name if res else None,
      adapter=[p.name for p in adapter_zips()])
print("下载到本地：", EXPORTS)
print("本次 Run 的实际路径：")
print("  Adapter 包（下载）   :", [str(p) for p in adapter_zips()])
print("  结果包（下载）       :", [str(p) for p in res])
print("  训练检查点（保留在 Drive，不必下载）:", CKPT)
print("  阶段状态 / 日志      :", RUN / "STAGE_STATUS.json", "|", RUN / "train.log")
print("然后在本地运行：stillhere persona import-adapter --profile", RM["profile_id"], "--file <下载的 persona_candidate_*.zip>")
print(json.dumps({k: v["status"] for k, v in STATUS.items()}, ensure_ascii=False, indent=1))
'''

PROGRESS = r'''# （随时可运行）进度
import json, pathlib
r = pathlib.Path(DRIVE_DIR) / PACKAGE_ID / "runs" / RUN_ID
print(json.dumps({k: v["status"] for k, v in json.loads((r / "STAGE_STATUS.json").read_text()).items()}, ensure_ascii=False, indent=1) if (r / "STAGE_STATUS.json").is_file() else "no status yet")
ev = pathlib.Path(DRIVE_DIR) / PACKAGE_ID / "checkpoints" / RUN_ID / "progress" / "events.jsonl"
if ev.is_file():
    rows = [json.loads(l) for l in ev.read_text().splitlines() if l.strip()]
    steps = [x for x in rows if x.get("event") == "step"]
    logs = [x for x in rows if x.get("event") == "log" and x.get("loss") is not None]
    print("last step:", steps[-1].get("global_step") if steps else None, "| last loss:", logs[-1].get("loss") if logs else None)
'''

CELLS = [("markdown", INTRO), ("code", STAGE00), ("code", STAGE01), ("code", STAGE02), ("code", STAGE03), ("code", STAGE04), ("code", STAGE05),
         ("code", STAGE06), ("code", STAGE07), ("code", STAGE08), ("code", STAGE09), ("code", STAGE10), ("code", STAGE11), ("code", STAGE12),
         ("code", STAGE13), ("code", STAGE14), ("code", PROGRESS)]
REPO_NOTEBOOK = Path(__file__).resolve().parents[3] / "notebooks" / "StillHere_Persona_L4.ipynb"


def build(package_id: str = "<PACKAGE_ID>", run_id: str = "<RUN_ID>", target: str = "colab-l4") -> dict:
    cells = []
    for kind, src in CELLS:
        src = src.replace("__PACKAGE_ID__", package_id).replace("__RUN_ID__", run_id).replace("__GPU_TARGET__", target)
        lines = src.strip("\n").splitlines(keepends=True)
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines})
    gpu = "A100" if target == "colab-a100" else "L4"
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": gpu, "provenance": []},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def dumps(nb: dict) -> bytes:
    return (json.dumps(nb, ensure_ascii=False, indent=1) + "\n").encode("utf-8")


def write_repo_notebook() -> Path:
    REPO_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    REPO_NOTEBOOK.write_bytes(dumps(build()))
    return REPO_NOTEBOOK


if __name__ == "__main__":
    print(write_repo_notebook())
    sys.exit(0)



# ------------------------------------------------------------------ Q1 prompt-refinement notebook (reuses STAGE 00-05; no training)
REFINE_INTRO = """# StillHere · Persona Prompt 优化评测（Colab L4，不训练）

只在 **Dev** 上为每个候选 Prompt 生成回复：同一个 Qwen3-8B NF4、同一个已训练的 Adapter、同一套解码参数；每个候选的 B（Base）和 C（Adapter）用**同一个** Prompt。
Test 不在包里。跑完下载结果 ZIP，在本地运行 `stillhere persona refine --profile <id> --import <zip>`，系统按预注册规则决定是否采用新 Prompt。
"""

REFINE_GEN = r'''# STAGE R1 · 候选 Prompt 的 Dev 生成
ADIR = DATA / "adapter"
plan = json.loads((DATA / "refine" / "REFINE_PLAN.json").read_text(encoding="utf-8"))
res = {}
for c in plan["candidates"]:
    es = DATA / "refine" / c["name"] / "dev_eval.jsonl"
    groups = RM["refine_groups"]
    ad = " ".join(f"--adapter {g}={ADIR}" for g in groups if g in ("C", "E"))
    res[c["name"]] = sh(f"python -m personatrain.evaluate generate --eval-set {es} --base-dir {BASE} {ad} --groups {','.join(groups)} "
                        f"--run-dir {RUN / 'refine' / c['name'] / 'run'}", log=RUN / "refine.log")
stage("STAGE R1 refine generation", "PASS" if all(v == 0 for v in res.values()) else "FAIL", **res)
'''

REFINE_EXPORT = r'''# STAGE R2 · 导出结果（只含生成的回复与状态，不含权重）
out = EXPORTS / f"persona_refine_results_{RUN_ID}.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted((RUN / "refine").rglob("*")):
        if p.is_file():
            z.write(p, f"refine/{p.relative_to(RUN / 'refine').as_posix()}")
    z.write(RUN / "STAGE_STATUS.json", "STAGE_STATUS.json")
stage("STAGE R2 export", "PASS", results=out.name)
print("下载", out, "然后在本地：stillhere persona refine --profile", RM["profile_id"], "--import <zip>")
'''


def build_refine(package_id: str, run_id: str, target: str = "colab-l4") -> dict:
    cells = [("markdown", REFINE_INTRO), ("code", STAGE00), ("code", STAGE01), ("code", STAGE02), ("code", STAGE03), ("code", STAGE04), ("code", STAGE05),
             ("code", REFINE_GEN), ("code", REFINE_EXPORT)]
    nb = build(package_id, run_id, target)
    out = []
    for kind, src in cells:
        src = src.replace("__PACKAGE_ID__", package_id).replace("__RUN_ID__", run_id).replace("__GPU_TARGET__", target)
        lines = src.strip("\n").splitlines(keepends=True)
        out.append({"cell_type": kind, "metadata": {}, "source": lines} if kind == "markdown" else
                   {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines})
    nb["cells"] = out
    return nb
