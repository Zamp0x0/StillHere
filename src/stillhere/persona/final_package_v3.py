"""Input package of the third-round Colab evaluation (``persona.eval_v3``): the Reader on the frozen AI_VERIFIED_NEW_GOLD set.

Contents (private; never in git): profile, the Skill card, the dataset TRAIN / DEV metadata files (never the sealed test), the evidence
archive (memory.sqlite, no structured memories), the frozen AI-verified gold + its manifest + the evaluation rows, code + tests. NO adapter
(every group runs the base model; the Reader never uses an adapter), no base or embedding weights (downloaded on Colab, pinned revisions).
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import sqlite3
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from . import doctor as DR
from . import eval_v3 as EV3
from . import final_package as FP
from . import final_package_v2 as FP2
from . import package as P
from . import store as S

OUT_DIR = "eval_v3_colab"
ZIP_NAME = "StillHere_V3_Input.zip"
NB_NAME = "StillHere_V3_Eval.ipynb"
SCHEMA = "stillhere.persona.eval_v3_package/1.0"
DRIVE_DIR = "/content/drive/MyDrive/StillHere/eval_v3"
TZ = timezone(timedelta(hours=8))


class PackageV3Error(ValueError):
    pass


def eval_rows(gold_jsonl: Path, db: Path, profile_id: str) -> bytes:
    gold = [json.loads(l) for l in Path(gold_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    ids = sorted({m for g in gold for m in g.get("evidence_message_ids") or []})
    con = sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True)
    try:
        got = {r[0]: r for r in con.execute(f"SELECT message_id, timestamp_utc, speaker_id, content FROM messages WHERE message_id IN ({','.join('?' * len(ids))})",
                                            ids)} if ids else {}
    finally:
        con.close()
    if set(ids) - set(got):
        raise PackageV3Error(f"{len(set(ids) - set(got))} gold evidence ids are not in the archive")
    text = {m: {"time_local": datetime.fromisoformat(r[1].replace("Z", "+00:00")).astimezone(TZ).isoformat(timespec="minutes"),
                "speaker": "本人" if r[2] == profile_id else "对方", "text": r[3]} for m, r in got.items()}
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in EV3.to_eval_rows(gold, text)).encode("utf-8")


def build(root, profile_id: str, *, skill_card: str, gold_jsonl: Path, gold_manifest: Path, out_dir: Optional[Path] = None) -> dict:
    from . import card as K
    from stillhere.retrieval import local_embedding as LE
    root = Path(root) if root else S.default_root()
    pdir = S.profile_dir(root, profile_id)
    card = K.load(pdir, skill_card)
    gman = json.loads(Path(gold_manifest).read_text(encoding="utf-8"))
    gold_b = Path(gold_jsonl).read_bytes()
    if hashlib.sha256(gold_b).hexdigest() != gman["sha256"] or gman.get("gold_source") != "AI_VERIFIED" or gman.get("is_human_gold") is not False:
        raise PackageV3Error("the gold file is not the frozen AI_VERIFIED_NEW_GOLD of its manifest")
    db = pdir / "memory" / "memory.sqlite"
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        n_msgs = con.execute("SELECT COUNT(*) FROM messages WHERE target_persona_id = ?", (profile_id,)).fetchone()[0]
        n_mem = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    finally:
        con.close()
    if n_mem:
        raise PackageV3Error(f"the archive holds {n_mem} structured memories; this evaluation reads the evidence archive only")
    rows_b = eval_rows(gold_jsonl, db, profile_id)
    ddir = P.latest_dataset(pdir)
    pre = f"data/personas/{profile_id}/"
    entries = [(pre + "profile.json", (pdir / "profile.json").read_bytes())]
    for f in sorted((pdir / "cards" / skill_card).iterdir()):
        entries.append((pre + f"cards/{skill_card}/{f.name}", f.read_bytes()))
    entries.append((pre + "cards/CURRENT.json", json.dumps({"card_id": skill_card, "at": "package"}).encode("utf-8")))
    for n in FP.DATASET_FILES:
        if (ddir / n).is_file():
            entries.append((pre + f"datasets/{ddir.name}/{n}", (ddir / n).read_bytes()))
    entries.append((pre + "memory/memory.sqlite", db.read_bytes()))
    gname = Path(gold_jsonl).name
    entries += [(f"frozen/{gname}", gold_b), ("frozen/AI_VERIFIED_NEW_GOLD_MANIFEST.json", Path(gold_manifest).read_bytes()),
                ("frozen/NEW_GOLD_EVAL_ROWS.jsonl", rows_b)]
    if any("/datasets/" in n and n.rsplit("/", 1)[-1].startswith("test") for n, _ in entries):
        raise PackageV3Error("the sealed test split may not be packaged")
    if any("/adapters/" in n for n, _ in entries):
        raise PackageV3Error("no adapter belongs in this package")
    code_b, code_meta = P.code_zip()
    with zipfile.ZipFile(io.BytesIO(code_b)) as z:
        code_entries = [(f"code/{n}", z.read(n)) for n in z.namelist()]
    for t in FP.TESTS + ("tests/test_persona_memory_v2.py", "tests/test_p0_retrieval.py", "tests/test_eval_v3.py"):
        tp = P.ROOT / t
        if tp.is_file():
            code_entries.append((f"code/{t}", tp.read_bytes()))
    inner = {"schema": SCHEMA, "files": {n: FP._sha(b) for n, b in sorted(entries + code_entries)}}
    zip_b = P.det_zip(entries + code_entries + [("INNER_MANIFEST.json", json.dumps(inner, indent=1, sort_keys=True).encode("utf-8"))])
    zsha = FP._sha(zip_b)
    n_q = len([l for l in gold_b.decode("utf-8").splitlines() if l.strip()])
    man = {"schema": SCHEMA, "package_id": "v3-" + zsha[:12], "profile_id": profile_id, "input_zip": ZIP_NAME, "input_zip_sha256": zsha, "input_zip_bytes": len(zip_b),
           "code_tree_sha256": code_meta["code_tree_sha256"], "git_commit": code_meta["git_head"], "git_dirty": code_meta["git_dirty"],
           "base": {"id": DR.BASE_REPO, "revision": DR.BASE_REVISION, "tokenizer_sha256": DR.TOKENIZER_SHA256, "chat_template_sha256": DR.CHAT_TEMPLATE_SHA256},
           "embedding": {"id": LE.MODEL_ID, "revision": LE.MODEL_REVISION, "dimension": LE.EXPECTED_DIMENSION},
           "pins_eval": FP.PINS_EVAL, "expect_versions": FP.EXPECT_EVAL, "dataset_version": ddir.name, "adapters": {},
           "prompts": {"old": {"card_id": skill_card, "prompt_sha256": card["manifest"]["prompt_sha256"]}},
           "gold": {"name": gman["name"], "file": gname, "sha256": gman["sha256"], "n": n_q, "gold_source": "AI_VERIFIED", "is_human_gold": False,
                    "eval_rows_sha256": FP._sha(rows_b)},
           "groups": {g: {k: (list(v) if isinstance(v, tuple) else v) for k, v in sp.items()} for g, sp in EV3.MEMORY_SPECS_V3.items()},
           "plan": {"memory": len(EV3.MEMORY_SPECS_V3) * n_q}, "rule": EV3.RULE_V3, "counts": {"archive_messages": n_msgs},
           "run_salt": "v3-" + zsha[:16], "tests_dir": "code/tests", "code_dir": "code",
           "contains": {"test_split": False, "raw_export": False, "base_weights": False, "adapters": False, "api_keys": False, "structured_memories": False},
           "privacy_notice": S.CLOUD_NOTICE, "created_at": datetime.now(timezone.utc).isoformat()}
    out = Path(out_dir) if out_dir else pdir / "packages" / OUT_DIR
    if out.exists():
        raise PackageV3Error(f"{out} exists: packages are never overwritten (move it away or pass another out_dir)")
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    (tmp / ZIP_NAME).write_bytes(zip_b)
    (tmp / NB_NAME).write_bytes(notebook_bytes())
    (tmp / "PACKAGE_MANIFEST.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    (tmp / "README_COLAB_CN.md").write_text(README_CN, encoding="utf-8", newline="\n")
    sums = "".join(f"{FP._sha((tmp / n).read_bytes())}  {n}\n" for n in (ZIP_NAME, "PACKAGE_MANIFEST.json", NB_NAME, "README_COLAB_CN.md"))
    (tmp / "SHA256SUMS.txt").write_text(sums, encoding="utf-8", newline="\n")
    tmp.rename(out)
    S.append_lineage(pdir, {"artefact": f"packages/{out.name}", "kind": "eval_v3_package", "zip_sha256": zsha})
    return {"package_dir": str(out), "zip": str(out / ZIP_NAME), "zip_mb": round(len(zip_b) / 2**20, 1), "manifest": man}


# ---------------------------------------------------------------- notebook
INTRO = """# StillHere · 第三轮 Colab 评测（新独立集 AI_VERIFIED_NEW_GOLD · Reader 对照）

一次 Run all。每个阶段打印 PASS / INCOMPLETE / SKIP 并写入 Drive；断线后重新 Run all 会续跑（已完成的生成和已付费的评分不会重做）。

**注意：** 本轮的标准答案由 AI 双阶段审核（AI_VERIFIED），不是人工 Gold。结果是工程对照证据；不要在同一批题上继续调参。

| 阶段 | 内容 |
|---|---|
| 00–05 | 配置、GPU、挂载、SHA256、依赖、Qwen3-8B Base（固定 revision，不加载 Adapter）、Qwen3-Embedding-0.6B |
| 06 | 冻结输入：AI_VERIFIED_NEW_GOLD（sha 与 Manifest 一致才继续） |
| 07b | 本地 dense 索引（在 Colab GPU 上重建；断点续建） |
| 13 | Memory：M1 / V2L / V2D / BEST_STABLE_V2 / BEST_EVIDENCE_FOCUSED |
| 14–15 | DeepSeek Judge（与 FAST 相同的 rubric）+ 标定 |
| 17 / 19 | 统计（答案准确率、Gold 证据召回、误拒答、无依据断言、归属 / 时间、token、延迟、分类、配对差）+ 预注册规则、导出结果 ZIP |
"""

CONFIG = r'''# STAGE 00 · Configuration（只改这里）
DRIVE_DIR = "%s"   # 放 3 个输入文件的 Drive 目录
RUN_NAME = ""                       # 留空：按输入 ZIP 的 sha256 自动命名（断线重跑会续上同一次实验）
JUDGE_BUDGET_CNY = 20               # DeepSeek 评分费用上限（人民币）
JUDGE_CONCURRENCY = 4
CONFIRM_PRIVATE_DATA_ON_CLOUD = True
import os, sys, json, hashlib, shutil, subprocess, zipfile, time
from pathlib import Path
assert CONFIRM_PRIVATE_DATA_ON_CLOUD
try:
    from google.colab import userdata
    os.environ.setdefault("DEEPSEEK_API_KEY", userdata.get("DEEPSEEK_API_KEY") or "")
except Exception:
    pass
assert os.environ.get("DEEPSEEK_API_KEY"), "在 Colab Secrets 里添加 DEEPSEEK_API_KEY 并打开 Notebook access"
os.environ["STILLHERE_JUDGE_BUDGET_CNY"] = str(JUDGE_BUDGET_CNY)
print("STAGE 00: PASS")
''' % DRIVE_DIR

MOUNT = FP.MOUNT.replace('"StillHere_Final_L4_Input.zip"', f'"{ZIP_NAME}"').replace(
    'RUN_NAME = RUN_NAME or (EVAL_MODE.lower() + "-run-" + MAN["input_zip_sha256"][:10])   # FAST / FULL never share a run directory',
    'RUN_NAME = RUN_NAME or ("v3-run-" + MAN["input_zip_sha256"][:10])').replace('Path("/content/stillhere_final")', 'Path("/content/stillhere_v3")')
VERIFY = FP.VERIFY.replace("StillHere_Final_L4_Input.zip", ZIP_NAME)
INSTALL = FP.INSTALL.replace("StillHere_Final_L4_Input.zip", ZIP_NAME)

LOAD = r'''# STAGE 05 · Qwen3-8B base (pinned, no adapter) + Qwen3-Embedding-0.6B (pinned)
from huggingface_hub import snapshot_download
from stillhere.persona import doctor as DR, colab_eval as CE, eval_v3 as EV3, eval_v2 as EV2, store as S
HF_HOME = LOCAL / "hf"; os.environ["HF_HOME"] = str(HF_HOME)
BASE = Path(snapshot_download(MAN["base"]["id"], revision=MAN["base"]["revision"], cache_dir=str(HF_HOME / "hub"),
                              allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json", "tokenizer*"]))
bc = DR.base_check(BASE)
assert bc["tokenizer_sha256_ok"] and bc["chat_template_sha256_ok"] and bc["revision_in_path"], bc
EMB = Path(snapshot_download(MAN["embedding"]["id"], revision=MAN["embedding"]["revision"], cache_dir=str(HF_HOME / "hub"),
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json", "tokenizer*"]))
WORK = LOCAL / "data" / "personas"
PID = MAN["profile_id"]
from d5pilot.evaluate import Gen
GEN = Gen(BASE, {})
MAN["code_dir"] = str(CODE)
RUN = CE.Run(work_root=WORK, profile_id=PID, run_dir=RUN_DIR, manifest=MAN, cfg=EV3.defaults({"judge_concurrency": JUDGE_CONCURRENCY}), gen=GEN)
import torch
ENV = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "versions": vers, "base": MAN["base"], "base_check": bc,
       "embedding": MAN["embedding"], "python": sys.version.split()[0]}
RUN.status("V3 STAGE 05 model", "PASS", gpu=ENV["gpu"], adapters=0)
'''

STAGES = [
    "# STAGE 06 · Frozen AI_VERIFIED_NEW_GOLD (sha must match the manifest)\nwith CE.timer(RUN, \"V3 STAGE 06 frozen gold\"):\n"
    "    EV3.seed_frozen(RUN, gold_jsonl=LOCAL / \"frozen\" / MAN[\"gold\"][\"file\"], gold_manifest=LOCAL / \"frozen\" / \"AI_VERIFIED_NEW_GOLD_MANIFEST.json\",\n"
    "                    eval_rows_jsonl=LOCAL / \"frozen\" / \"NEW_GOLD_EVAL_ROWS.jsonl\")\n",
    "# STAGE 07b · Local dense index (Qwen3-Embedding-0.6B on the GPU; resumable)\nwith CE.timer(RUN, \"V3 STAGE 07b dense\"):\n"
    "    EV2.stage_dense(RUN, EMB)\nDENSE, WHY = EV2.dense_channel(RUN, EMB)\nprint(\"dense:\", WHY)\n",
    "# STAGE 13 · Memory groups: M1 / V2L / V2D / BEST_STABLE_V2 / BEST_EVIDENCE_FOCUSED\nwith CE.timer(RUN, \"V3 STAGE 13 memory\"):\n"
    "    EV3.stage_memory(RUN, dense=DENSE)\n",
    "# STAGE 14-15 · DeepSeek judge: calibration + memory (same rubric as FAST)\nwith CE.timer(RUN, \"V3 STAGE 14-15 judge\"):\n"
    "    EV3.stage_judge(RUN, os.environ)\n",
    "# STAGE 17 · Analysis + registered rule\nwith CE.timer(RUN, \"V3 STAGE 17 analysis\"):\n    REPORT = EV3.analysis(RUN)\n"
    "print(json.dumps(REPORT[\"decision\"], ensure_ascii=False, indent=1))\n",
    "# STAGE 19 · Export\nwith CE.timer(RUN, \"V3 STAGE 19 export\"):\n    OUT = EV3.stage_export(RUN, ENV)\n"
    "print(\"下载这个文件：\", OUT)\nprint(\"阶段状态：\", RUN_DIR / \"STAGE_STATUS.json\")\n",
]


def notebook() -> dict:
    cells = [("markdown", INTRO), ("code", CONFIG), ("code", FP.PREFLIGHT), ("code", MOUNT), ("code", VERIFY), ("code", INSTALL), ("code", LOAD)] + \
        [("code", s) for s in STAGES]
    return {"cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(True), **({"outputs": [], "execution_count": None} if t == "code" else {})}
                      for t, s in cells],
            "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4", "provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}


def notebook_bytes() -> bytes:
    return (json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n").encode("utf-8")


README_CN = """# StillHere 第三轮 Colab 评测（新独立集 Reader 对照）：使用说明

## 1. 上传（只需一次）
在 Google Drive 新建 `MyDrive/StillHere/eval_v3/`，放入同一个包目录里的 3 个文件：
- `StillHere_V3_Input.zip`
- `PACKAGE_MANIFEST.json`
- `SHA256SUMS.txt`

`StillHere_V3_Eval.ipynb` 放在任意 Drive 位置（建议同一目录），用 Google Colaboratory 打开。

## 2. DeepSeek Key
Colab 左侧钥匙图标（Secrets）→ 添加 `DEEPSEEK_API_KEY`，打开 Notebook access。

## 3. 运行
Runtime → Change runtime type → **L4 GPU**（A100 更快）→ Run all。依赖装完提示版本不符：Runtime → Restart session → Run all（会续跑）。

## 4. 结果
`MyDrive/StillHere/eval_v3/runs/v3-run-<输入 ZIP sha 前 10 位>/export/StillHere_V3_Results.zip`（STAGE 19 会打印实际路径）。
下载到本地：`<data-dir>\\personas\\<research-profile>\\downloads\\StillHere_V3_Results.zip`。

本轮标准答案是 AI_VERIFIED（AI 双阶段审核），不是人工 Gold。
"""
