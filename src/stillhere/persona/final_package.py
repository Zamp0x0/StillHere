"""Build the ONE Colab L4 input package of the final evaluation (outside git):

    <profile>/packages/final_colab_l4/
        StillHere_Final_L4_Input.zip     code (committed tree) + the profile subset the run needs (no test split, no raw export)
        StillHere_Final_L4.ipynb         one notebook, Run All
        PACKAGE_MANIFEST.json            what is inside, every pin, the expected counts
        SHA256SUMS.txt
        README_COLAB_CN.md

The profile subset: profile.json, the old and new card versions (CURRENT = old: the local selection is not changed), the dataset's
train / dev splits and manifests (test.jsonl / test_meta.jsonl are NOT copied), the r1 adapter, and the evidence archive
(memory.sqlite, which must hold no semantic memory yet: the Colab run builds it from scratch). Qwen3-8B is downloaded in Colab at
the pinned revision (tokenizer / chat-template sha256 checked), not shipped.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import colab_eval as CE
from . import doctor as DR
from . import package as P
from . import store as S

OUT_DIR = "final_colab_l4"
ZIP_NAME = "StillHere_Final_L4_Input.zip"
NB_NAME = "StillHere_Final_L4.ipynb"
SCHEMA = "stillhere.persona.final_colab_package/1.0"
TESTS = ("tests/conftest.py", "tests/persona_fixture.py", "tests/persona_memory_fixture.py", "tests/test_persona_final_quality.py",
         "tests/test_persona_colab_final.py")
PINS_EVAL = [p for p in DR.PINS_TRAIN if not p.startswith(("llamafactory", "trl", "datasets"))]
EXPECT_EVAL = {k: v for k, v in DR.EXPECT.items() if k != "llamafactory"}
DATASET_FILES = ("DATASET_MANIFEST.json", "SPLIT_MANIFEST.json", "QUALITY_REPORT.json", "cleaning_config.json", "dataset_info.json",
                 "train.jsonl", "train_meta.jsonl", "dev.jsonl", "dev_meta.jsonl")


class FinalPackageError(ValueError):
    def __init__(self, code: str, msg: str):
        super().__init__(f"{code}: {msg}")
        self.code = code


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build(root, profile_id: str, *, old_card: str, new_card: str, adapter_id: str, out_dir: Optional[Path] = None) -> dict:
    from . import adapters as AD
    from . import card as K
    root = Path(root) if root else S.default_root()
    pdir = S.profile_dir(root, profile_id)
    for cid in (old_card, new_card):
        K.load(pdir, cid)                                                     # manifest + prompt sha verified
    ad = AD.resolve(pdir, adapter_id)
    ddir = P.latest_dataset(pdir)
    db = pdir / "memory" / "memory.sqlite"
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        n_msgs = con.execute("SELECT COUNT(*) FROM messages WHERE target_persona_id = ?", (profile_id,)).fetchone()[0]
        n_mem = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        n_text = con.execute("SELECT COUNT(*) FROM messages WHERE target_persona_id = ? AND locator NOT LIKE '%kind=media%'", (profile_id,)).fetchone()[0]
        if not con.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]:
            raise FinalPackageError("NO_EVIDENCE_INDEX", "the evidence archive has no full-text index; FAST memory would have nothing to retrieve")
    finally:
        con.close()
    if n_mem:
        raise FinalPackageError("MEMORY_NOT_EMPTY", f"the archive already holds {n_mem} semantic memories; the final run builds memory from scratch")
    entries = []
    pre = f"data/personas/{profile_id}/"
    entries.append((pre + "profile.json", (pdir / "profile.json").read_bytes()))
    for cid in (old_card, new_card):
        for f in sorted((pdir / "cards" / cid).iterdir()):
            entries.append((pre + f"cards/{cid}/{f.name}", f.read_bytes()))
    entries.append((pre + "cards/CURRENT.json", json.dumps({"card_id": old_card, "at": "package"}).encode("utf-8")))
    for n in DATASET_FILES:
        if (ddir / n).is_file():
            entries.append((pre + f"datasets/{ddir.name}/{n}", (ddir / n).read_bytes()))
    adir = pdir / "adapters" / adapter_id
    for f in sorted(adir.rglob("*")):
        if f.is_file():
            entries.append((pre + f"adapters/{adapter_id}/{f.relative_to(adir).as_posix()}", f.read_bytes()))
    entries.append((pre + "memory/memory.sqlite", db.read_bytes()))
    names = [n for n, _ in entries]
    if any("/test" in n.split("datasets/")[-1] for n in names if "/datasets/" in n):
        raise FinalPackageError("TEST_IN_PACKAGE", "the sealed test split may not be packaged")
    code_b, code_meta = P.code_zip()
    import zipfile
    import io
    with zipfile.ZipFile(io.BytesIO(code_b)) as z:
        code_entries = [(f"code/{n}", z.read(n)) for n in z.namelist()]
    for t in TESTS:
        tp = P.ROOT / t
        if tp.is_file():
            code_entries.append((f"code/{t}", tp.read_bytes()))
    inner = {"schema": SCHEMA, "files": {n: _sha(b) for n, b in sorted(entries + code_entries)}}
    zip_b = P.det_zip(entries + code_entries + [("INNER_MANIFEST.json", json.dumps(inner, indent=1, sort_keys=True).encode("utf-8"))])
    zsha = _sha(zip_b)
    old, new = K.load(pdir, old_card), K.load(pdir, new_card)
    adapter_sha = _sha((Path(ad["dir"]) / "adapter_model.safetensors").read_bytes())
    man = {"schema": SCHEMA, "package_id": "fl4-" + zsha[:12], "profile_id": profile_id, "input_zip": ZIP_NAME, "input_zip_sha256": zsha,
           "input_zip_bytes": len(zip_b), "code_tree_sha256": code_meta["code_tree_sha256"], "git_commit": code_meta["git_head"], "git_dirty": code_meta["git_dirty"],
           "base": {"id": DR.BASE_REPO, "revision": DR.BASE_REVISION, "tokenizer_sha256": DR.TOKENIZER_SHA256, "chat_template_sha256": DR.CHAT_TEMPLATE_SHA256},
           "pins_eval": PINS_EVAL, "expect_versions": EXPECT_EVAL, "dataset_version": ddir.name,
           "adapter": {"adapter_id": adapter_id, "sha256": adapter_sha, "note": "r1 NOT_BETTER (unchanged); evaluated as C1 / C2 / E only"},
           "prompts": {"old": {"card_id": old_card, "prompt_sha256": old["manifest"]["prompt_sha256"]},
                       "new": {"card_id": new_card, "prompt_sha256": new["manifest"]["prompt_sha256"], "author": new["card"].get("prompt_author")}},
           "counts": {"archive_messages": n_msgs, "text_messages": n_text, "dev_items": sum(1 for l in (ddir / "dev.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()),
                      "default_mode": CE.DEFAULT_MODE, "memory_groups": list(CE.MEMORY_GROUPS), "style_groups": list(CE.STYLE_GROUPS),
                      "modes": {m: {"plan": CE.plan_units(CE.defaults({"eval_mode": m})), "style_items": CE.MODES[m]["style_n"] or CE.FULL_DEV_ITEMS,
                                    "qa": sum(CE.MODES[m]["qa_targets"].values()), "writer_audit_sessions": CE.MODES[m]["writer_audit_n"],
                                    "memory_build": CE.MODES[m]["memory_build"], "memory_mode": CE.MODES[m]["memory_mode"],
                                    "case_studies": f"{CE.MODES[m]['case_min']}-{CE.MODES[m]['case_max']}", "results_zip": CE.MODES[m]["results_zip"]} for m in CE.MODES}},
           "run_salt": "fl4-" + zsha[:16], "tests_dir": "code/tests", "contains": {"test_split": False, "raw_export": False, "base_weights": False, "api_keys": False},
           "privacy_notice": S.CLOUD_NOTICE, "created_at": datetime.now(timezone.utc).isoformat()}
    out = Path(out_dir) if out_dir else pdir / "packages" / OUT_DIR
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    (tmp / ZIP_NAME).write_bytes(zip_b)
    (tmp / NB_NAME).write_bytes(notebook_bytes())
    (tmp / "PACKAGE_MANIFEST.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    (tmp / "README_COLAB_CN.md").write_text(README_CN, encoding="utf-8", newline="\n")
    sums = "".join(f"{_sha((tmp / n).read_bytes())}  {n}\n" for n in (ZIP_NAME, "PACKAGE_MANIFEST.json", NB_NAME, "README_COLAB_CN.md"))
    (tmp / "SHA256SUMS.txt").write_text(sums, encoding="utf-8", newline="\n")
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    S.append_lineage(pdir, {"artefact": f"packages/{OUT_DIR}", "kind": "final_colab_l4_package", "zip_sha256": zsha})
    return {"package_dir": str(out), "zip": str(out / ZIP_NAME), "zip_mb": round(len(zip_b) / 2**20, 1), "manifest": man}


# ---------------------------------------------------------------- notebook
INTRO = """# StillHere · 最终 Colab L4 评测（FAST 默认 / FULL 可选：Style + Memory 消融 + DeepSeek Judge）

**EVAL_MODE（STAGE 00）：**
- `"FAST"`（默认）：100 条 Dev × 4 组、100 道真实历史题 × 3 组（M0 / M1 / M2）、20 个 Writer 审计会话、720 个 DeepSeek 评分单元。记忆使用**完整证据档案**（20,298 条消息，EF1 + Fact Reader），**不做**全量结构化语义记忆抽取。
- `"FULL"`：268 条 × 4、200 题 × 3、100 个会话、1,772 个单元，外加全量语义记忆抽取。**只有手动改成 FULL 才会运行。**


**用法：** 把 `StillHere_Final_L4_Input.zip`、`PACKAGE_MANIFEST.json`、`SHA256SUMS.txt` 放进 Google Drive 的 `DRIVE_DIR`；在 Colab Secrets 添加 `DEEPSEEK_API_KEY`
（打开 Notebook 访问权限）；运行时选 **L4 GPU**；然后 **Runtime → Run all**。

- 一个 Notebook、一个输入 ZIP、一次 Run All。不训练、不改 Adapter、不读取 Test。
- 断线后重新 Run All：同一个 `RUN_NAME` 会自动续跑（记忆库、题集、生成、评分都从 Drive 上的检查点继续，已完成的不会重做）。
- 安装依赖后如果提示 Restart，请 Runtime → Restart session，再 Run all。
- 结果：`DRIVE_DIR/runs/<RUN_NAME>/export/`，FAST 为 `StillHere_FAST_L4_Results.zip`，FULL 为 `StillHere_Final_L4_Results.zip`。
- FAST 和 FULL 使用不同的 RUN_NAME、不同的检查点和不同的指纹，互不续用。
"""

CONFIG = r'''# STAGE 00 · Configuration
EVAL_MODE = "FAST"                  # "FAST"（默认，快速正式评测）或 "FULL"（完整评测，耗时很长；只有手动改这里才会运行）
DRIVE_DIR = "/content/drive/MyDrive/StillHere/final_colab_l4"   # 放 3 个输入文件的 Drive 目录
RUN_NAME = ""                       # 留空：按模式 + 输入 ZIP 的 sha256 自动命名（断线重跑会续上同一次实验）
JUDGE_MODEL = ""                    # 留空：项目默认的 DeepSeek 模型；其他模型需同时设置 STILLHERE_JUDGE_PRICE_USD_IN / _OUT
JUDGE_BUDGET_CNY = 45               # DeepSeek 评分的费用上限（人民币）；FAST 约 720 个单元，实际费用见结果里的 JUDGE_USAGE_REPORT
JUDGE_CONCURRENCY = 4
ACCEPT_AUTO_VERIFIED_BY = "owner (final Colab run)"   # 记忆写入时记录为 auto_verified;accepted_by:<这个名字>（不是人工审核）
CONFIRM_PRIVATE_DATA_ON_CLOUD = True                  # 输入包含真实私聊；你已决定在 Colab 处理
import os, sys, json, hashlib, subprocess, shutil, time, zipfile
from pathlib import Path
assert CONFIRM_PRIVATE_DATA_ON_CLOUD, "输入包含真实私聊：确认可以在 Colab 处理后改为 True"
EVAL_MODE = EVAL_MODE.strip().upper()
assert EVAL_MODE in ("FAST", "FULL"), "EVAL_MODE 只能是 FAST 或 FULL"
try:
    from google.colab import userdata
    os.environ["DEEPSEEK_API_KEY"] = userdata.get("DEEPSEEK_API_KEY")
except Exception as e:
    raise SystemExit("没有读到 Colab Secrets 里的 DEEPSEEK_API_KEY：左侧钥匙图标 → 添加 DEEPSEEK_API_KEY 并打开 Notebook 访问权限，然后重新 Run all。") from e
assert os.environ.get("DEEPSEEK_API_KEY"), "DEEPSEEK_API_KEY 为空"
if JUDGE_MODEL: os.environ["STILLHERE_JUDGE_MODEL"] = JUDGE_MODEL
os.environ["STILLHERE_JUDGE_BUDGET_CNY"] = str(JUDGE_BUDGET_CNY)
print("STAGE 00: PASS (config; the key is read from Secrets and never printed)")
'''

PREFLIGHT = r'''# STAGE 01 · GPU / Runtime Preflight
smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
print(smi)
assert smi, "没有 GPU：Runtime → Change runtime type → L4 GPU"
vram = float(smi.split(",")[-1].strip().split()[0]) / 1024
assert ("L4" in smi or "A100" in smi or "H100" in smi) and vram >= 20, f"需要 L4 24GB（或更大）：当前 {smi}"
free = shutil.disk_usage("/content").free / 2**30
assert free >= 40, f"磁盘剩余 {free:.0f} GiB，至少需要 40 GiB"
print(f"STAGE 01: PASS gpu={smi} disk_free={free:.0f}GiB")
'''

MOUNT = r'''# STAGE 02 · Mount Drive & locate the input
from google.colab import drive
drive.mount("/content/drive")
IN = Path(DRIVE_DIR)
need = ["StillHere_Final_L4_Input.zip", "PACKAGE_MANIFEST.json", "SHA256SUMS.txt"]
missing = [n for n in need if not (IN / n).is_file()]
assert not missing, f"把这些文件放到 {IN}：{missing}"
MAN = json.loads((IN / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
RUN_NAME = RUN_NAME or (EVAL_MODE.lower() + "-run-" + MAN["input_zip_sha256"][:10])   # FAST / FULL never share a run directory
RUN_DIR = IN / "runs" / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)
LOCAL = Path("/content/stillhere_final")
print("STAGE 02: PASS", RUN_DIR)
'''

VERIFY = r'''# STAGE 03 · Manifest / SHA256 verification
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
sums = dict((l.split(None, 1)[1].strip(), l.split(None, 1)[0]) for l in (IN / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines() if l.strip())
zsha = sha256(IN / "StillHere_Final_L4_Input.zip")
problems = []
if zsha != sums.get("StillHere_Final_L4_Input.zip") or zsha != MAN["input_zip_sha256"]: problems.append("input ZIP sha256 mismatch")
if sha256(IN / "PACKAGE_MANIFEST.json") != sums.get("PACKAGE_MANIFEST.json"): problems.append("PACKAGE_MANIFEST sha256 mismatch")
with zipfile.ZipFile(IN / "StillHere_Final_L4_Input.zip") as z:
    if z.testzip() is not None: problems.append("ZIP CRC error")
    inner = json.loads(z.read("INNER_MANIFEST.json"))
    for n, h in inner["files"].items():
        if hashlib.sha256(z.read(n)).hexdigest() != h: problems.append(f"inner sha256 {n}"); break
    if any("/datasets/" in n and n.rsplit("/", 1)[-1].startswith("test") for n in z.namelist()): problems.append("test split inside the package")
assert not problems, problems
print("STAGE 03: PASS", MAN["package_id"], "files:", len(inner["files"]))
'''

INSTALL = r'''# STAGE 04 · Extract + install pinned dependencies
marker = LOCAL / ".zip_sha256"
if not marker.is_file() or marker.read_text().strip() != zsha:
    shutil.rmtree(LOCAL, ignore_errors=True)
    LOCAL.mkdir(parents=True)
    with zipfile.ZipFile(IN / "StillHere_Final_L4_Input.zip") as z: z.extractall(LOCAL)
    marker.write_text(zsha)
    print("extracted")
CODE = LOCAL / "code"
sys.path[:0] = [str(CODE / x) for x in ("training/persona", "training/d5", "training/d4", "src", "tests")]
rc = subprocess.run("pip -q install " + " ".join(f'"{p}"' for p in MAN["pins_eval"]), shell=True).returncode
import importlib
vers = {m: importlib.import_module(m).__version__ for m in ("transformers", "peft", "bitsandbytes", "accelerate", "torch")}
ok = rc == 0 and all(vers.get(k) == v for k, v in MAN["expect_versions"].items())
print(vers)
assert ok, "依赖版本不符：如果刚安装完，请 Runtime → Restart session，然后重新 Run all（会自动续跑）"
print("STAGE 04: PASS")
'''

LOAD = r'''# STAGE 05 · Qwen3-8B base (pinned revision) + r1 adapter — ONE model for Writer / Reader / chat
from huggingface_hub import snapshot_download
from stillhere.persona import doctor as DR, colab_eval as CE, store as S
HF_HOME = LOCAL / "hf"; os.environ["HF_HOME"] = str(HF_HOME)
BASE = Path(snapshot_download(MAN["base"]["id"], revision=MAN["base"]["revision"], cache_dir=str(HF_HOME / "hub"),
                              allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json", "tokenizer*"]))
bc = DR.base_check(BASE)
assert bc["tokenizer_sha256_ok"] and bc["chat_template_sha256_ok"] and bc["revision_in_path"], bc
WORK = LOCAL / "data" / "personas"
PID = MAN["profile_id"]
ADIR = S.profile_dir(WORK, PID) / "adapters" / MAN["adapter"]["adapter_id"] / "adapter"
assert sha256(ADIR / "adapter_model.safetensors") == MAN["adapter"]["sha256"], "adapter sha256 mismatch"
from d5pilot.evaluate import Gen
GEN = Gen(BASE, {MAN["adapter"]["adapter_id"]: ADIR})
RUN = CE.Run(work_root=WORK, profile_id=PID, run_dir=RUN_DIR, manifest=MAN, cfg=CE.defaults({"eval_mode": EVAL_MODE,
             "accept_auto_verified_by": ACCEPT_AUTO_VERIFIED_BY, "judge_concurrency": JUDGE_CONCURRENCY}), gen=GEN)
print("mode:", EVAL_MODE, "| plan:", CE.plan_units(RUN.cfg), "| fingerprint:", CE.bind(RUN))
import torch
ENV = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "versions": vers, "base": MAN["base"], "base_check": bc,
       "python": sys.version.split()[0]}
RUN.status("STAGE 05 model", "PASS", gpu=ENV["gpu"], vram_gib=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
'''

STAGES = [
    ("# STAGE 06 · Engineering checks (CPU tests) + synthetic gold on the real model\n"
     "with CE.timer(RUN, \"STAGE 06 checks\"):\n"
     "    CE.stage_06_checks(RUN, synthetic_dir=LOCAL / \"synthetic\", pytest_cmd=[sys.executable, \"-m\", \"pytest\", \"-q\", \"-p\", \"no:cacheprovider\",\n"
     "                       str(CODE / \"tests\" / \"test_persona_colab_final.py\"), str(CODE / \"tests\" / \"test_persona_final_quality.py\")])\n"),
    ("# STAGE 07 · Memory. FAST: the complete evidence archive (no semantic build) + the real Writer on the 20 audit sessions (isolated copy).\n"
     "#            FULL: the full semantic Memory Writer over every session (checkpoint every 20 sessions). Re-run resumes.\n"
     "with CE.timer(RUN, \"STAGE 07 memory\"):\n    CE.stage_07_memory(RUN)\n"),
    "# STAGE 08–09 · Temporal report, SQLite + EF1 report, memory snapshot\nwith CE.timer(RUN, \"STAGE 08-09 reports\"):\n    CE.stage_08_09_reports(RUN)\n",
    "# STAGE 10 · Generate & freeze the real-history QA (FAST 100 / FULL 200; evidence first; machine-constructed, evidence-verified)\nwith CE.timer(RUN, \"STAGE 10 QA\"):\n    CE.stage_10_qa(RUN)\n",
    "# STAGE 11 · Writer audit sessions (FAST 20 / FULL 100; stratified, fixed seed)\nwith CE.timer(RUN, \"STAGE 11 writer sample\"):\n    CE.stage_11_writer_sample(RUN)\n",
    "# STAGE 12 · Style DEV generation: (FAST 100 / FULL 268) x B1 / B2 / C1 / C2\nwith CE.timer(RUN, \"STAGE 12 style generation\"):\n    CE.stage_12_style(RUN)\n",
    "# STAGE 14 · DeepSeek Style Judge (calibration first, then the blind units) + pre-registered prompt selection\nwith CE.timer(RUN, \"STAGE 14 style judge\"):\n    CE.stage_14_style_judge(RUN, os.environ)\n",
    "# STAGE 13 · Memory ablation generation: QA x M0 (no memory) / M1 (memory) / M2 (memory + r1); selected prompt; one snapshot\nwith CE.timer(RUN, \"STAGE 13 memory generation\"):\n    CE.stage_13_memory_qa(RUN)\n",
    "# STAGE 15 · DeepSeek Memory Judge (blind units)\nwith CE.timer(RUN, \"STAGE 15 memory judge\"):\n    CE.stage_15_memory_judge(RUN, os.environ)\n",
    "# STAGE 16 · DeepSeek Writer Judge (the audit sessions)\nwith CE.timer(RUN, \"STAGE 16 writer judge\"):\n    CE.stage_16_writer_judge(RUN, os.environ)\n",
    "# STAGE 17–18 · Statistics: ablation (M1-M0, M2-M1, B2-B1, C1-B1, C2-B2), case studies, academic results, recommendation\nwith CE.timer(RUN, \"STAGE 17-18 analysis\"):\n    REPORT = CE.stage_17_18_analysis(RUN)\n",
    "# STAGE 19 · Export the results ZIP\nwith CE.timer(RUN, \"STAGE 19 export\"):\n    RESULTS = CE.stage_19_export(RUN, ENV)\nprint(RESULTS)\n",
    ("# STAGE 20 · Final summary (+ per-stage timings)\nSUMMARY = CE.export_names(EVAL_MODE).get(\"FINAL_SUMMARY_CN.md\", \"FINAL_SUMMARY_CN.md\")\n"
     "print((RUN_DIR / \"export\" / \"files\" / SUMMARY).read_text(encoding=\"utf-8\"))\nprint(json.loads((RUN_DIR / \"STAGE_TIMINGS.json\").read_text(encoding=\"utf-8\")))\n"),
]


def notebook() -> dict:
    cells = [("markdown", INTRO), ("code", CONFIG), ("code", PREFLIGHT), ("code", MOUNT), ("code", VERIFY), ("code", INSTALL), ("code", LOAD)] + [("code", s) for s in STAGES]
    return {"cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(True), **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in cells],
            "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4", "provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}


def notebook_bytes() -> bytes:
    return (json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n").encode("utf-8")


REPO_NOTEBOOK = P.ROOT / "notebooks" / NB_NAME


def write_repo_notebook() -> Path:
    REPO_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    REPO_NOTEBOOK.write_bytes(notebook_bytes())
    return REPO_NOTEBOOK


README_CN = """# StillHere 最终 Colab L4 评测：使用说明

## 1. 上传（只需一次）
在 Google Drive 新建目录 `MyDrive/StillHere/final_colab_l4/`，放入：
- `StillHere_Final_L4_Input.zip`（完整输入：全部证据档案、Dev、两个 Prompt、r1 Adapter、代码；不含 Test）
- `PACKAGE_MANIFEST.json`
- `SHA256SUMS.txt`

## 2. DeepSeek Key
Colab 左侧钥匙图标（Secrets）→ 添加 `DEEPSEEK_API_KEY`，打开 “Notebook access”。Key 不会写进 Notebook、日志、报告或结果 ZIP。

## 3. 选择模式并运行
打开 `StillHere_Final_L4.ipynb`，STAGE 00 里的 `EVAL_MODE` 默认是 `"FAST"`，不用改。
Runtime → Change runtime type → **L4 GPU** → Run all。第一次安装依赖后如果提示版本不符：Restart session → Run all。

| 模式 | 规模 | 记忆 |
|---|---|---|
| FAST（默认） | Style 100 条 × 4 = 400；Memory 100 题 × 3 = 300；Writer 20 个会话；DeepSeek 720 个单元 | 完整证据档案检索（EVIDENCE_MEMORY），不做全量语义抽取 |
| FULL（手动） | Style 268 × 4；Memory 200 × 3；Writer 100 个会话；DeepSeek 1,772 个单元 | 全量结构化语义记忆抽取 |

## 4. 断线
重新打开同一个 Notebook → Run all，同一个模式会自动续跑：已冻结的题集、已完成的生成和评分不会重做，已付费的评分不会再次发送。
FAST 与 FULL 的运行目录（`runs/fast-run-…` 与 `runs/full-run-…`）、检查点和指纹完全分开。

## 5. 结果
FAST：`MyDrive/StillHere/final_colab_l4/runs/fast-run-<…>/export/StillHere_FAST_L4_Results.zip`，下载到本地后：

    python -m stillhere.cli persona import-colab-results --profile <research-profile> --file <下载的 ZIP>

FAST 结果只导入报告，**不改动本地记忆库**（它没有建立结构化语义记忆）。

## 6. 时间
每个阶段的耗时会实时打印，并写入 `runs/<RUN_NAME>/STAGE_TIMINGS.json`。运行速度没有在 L4 上实测过，所以这里不承诺具体时长；FAST 的工作量明显小于 FULL（不做全量 Writer 抽取）。
"""
