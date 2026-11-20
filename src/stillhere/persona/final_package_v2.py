"""``stillhere persona final-package-v2``: the ONE input package of the second-round Colab evaluation (``persona.eval_v2``).

Contents (private; never in git): profile, the Skill card (the runtime persona prompt), r1 AND r2 adapters, the dataset TRAIN / DEV files
(never the sealed test), the evidence archive (memory.sqlite, no structured memories), the frozen FAST results ZIP (frozen QA, style
sample, Writer audit sessions), the Memory V2 DEV / HOLDOUT split, code + tests + the public general-knowledge questions. The embedding
model (Qwen3-Embedding-0.6B, pinned revision) and the base model are downloaded on Colab from Hugging Face (public weights).
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import sqlite3
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import colab_eval as CE
from . import doctor as DR
from . import eval_v2 as EV2
from . import final_package as FP
from . import package as P
from . import store as S

OUT_DIR = "eval_v2_colab"
ZIP_NAME = "StillHere_V2_Input.zip"
NB_NAME = "StillHere_V2_Eval.ipynb"
SCHEMA = "stillhere.persona.eval_v2_package/1.0"
FAST_QA_SHA256 = "01af8c541ae35d11e27d334d123959e95b2d118cb3b6fb6995f8b7f30ed575fb"


class PackageV2Error(ValueError):
    pass


def build(root, profile_id: str, *, skill_card: str, r1_adapter: str, r2_adapter: str, fast_results_zip: Path, freeze_manifest: Path,
          out_dir: Optional[Path] = None, expect_qa_sha256: Optional[str] = FAST_QA_SHA256) -> dict:
    from . import adapters as AD
    from . import card as K
    from stillhere.retrieval import local_embedding as LE
    root = Path(root) if root else S.default_root()
    pdir = S.profile_dir(root, profile_id)
    K.load(pdir, skill_card)
    ids = {"r1": r1_adapter, "r2": r2_adapter}
    ads = {k: AD.resolve(pdir, v) for k, v in ids.items()}
    for k, a in ads.items():
        if a["manifest"]["card_id"] != skill_card:
            raise PackageV2Error(f"{k} adapter belongs to card {a['manifest']['card_id']}, not the Skill card {skill_card}")
    ddir = P.latest_dataset(pdir)
    db = pdir / "memory" / "memory.sqlite"
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        n_msgs = con.execute("SELECT COUNT(*) FROM messages WHERE target_persona_id = ?", (profile_id,)).fetchone()[0]
        n_mem = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    finally:
        con.close()
    if n_mem:
        raise PackageV2Error(f"the archive holds {n_mem} structured memories; the second round starts from the evidence archive only")
    with zipfile.ZipFile(fast_results_zip) as z:
        qa = z.read(next(n for n in z.namelist() if n.endswith("MEMORY_QA_100.jsonl")))
    if expect_qa_sha256 and hashlib.sha256(qa).hexdigest() != expect_qa_sha256:
        raise PackageV2Error("the FAST results ZIP does not hold the frozen QA set")
    split = json.loads(Path(freeze_manifest).read_text(encoding="utf-8"))["fast_qa_split"]
    pre = f"data/personas/{profile_id}/"
    entries = [(pre + "profile.json", (pdir / "profile.json").read_bytes())]
    for f in sorted((pdir / "cards" / skill_card).iterdir()):
        entries.append((pre + f"cards/{skill_card}/{f.name}", f.read_bytes()))
    entries.append((pre + "cards/CURRENT.json", json.dumps({"card_id": skill_card, "at": "package"}).encode("utf-8")))
    for n in FP.DATASET_FILES:
        if (ddir / n).is_file():
            entries.append((pre + f"datasets/{ddir.name}/{n}", (ddir / n).read_bytes()))
    for aid in ids.values():
        base = pdir / "adapters" / aid
        for f in sorted(base.rglob("*")):
            if f.is_file():
                entries.append((pre + f"adapters/{aid}/{f.relative_to(base).as_posix()}", f.read_bytes()))
    entries.append((pre + "memory/memory.sqlite", db.read_bytes()))
    entries.append(("frozen/StillHere_FAST_L4_Results.zip", Path(fast_results_zip).read_bytes()))
    entries.append(("frozen/MEMORY_V2_SPLIT.json", json.dumps({"dev": split["dev"], "holdout": split["holdout"], "salt": split["salt"], "rule": split["rule"]},
                                                              indent=1).encode("utf-8")))
    if any("/datasets/" in n and n.rsplit("/", 1)[-1].startswith("test") for n, _ in entries):
        raise PackageV2Error("the sealed test split may not be packaged")
    code_b, code_meta = P.code_zip()
    with zipfile.ZipFile(io.BytesIO(code_b)) as z:
        code_entries = [(f"code/{n}", z.read(n)) for n in z.namelist()]
    for t in FP.TESTS + ("tests/test_persona_memory_v2.py", "tests/test_writer_gate.py", EV2.GENERAL_FILE):
        tp = P.ROOT / t
        if tp.is_file():
            code_entries.append((f"code/{t}", tp.read_bytes()))
    inner = {"schema": SCHEMA, "files": {n: FP._sha(b) for n, b in sorted(entries + code_entries)}}
    zip_b = P.det_zip(entries + code_entries + [("INNER_MANIFEST.json", json.dumps(inner, indent=1, sort_keys=True).encode("utf-8"))])
    zsha = FP._sha(zip_b)
    card = K.load(pdir, skill_card)
    plan = {"style": len(EV2.STYLE_SPECS) * 100, "general": len(EV2.GENERAL_SPECS) * 40, "memory": len(EV2.MEMORY_SPECS) * 100, "writer_audit_sessions": 20}
    man = {"schema": SCHEMA, "package_id": "v2-" + zsha[:12], "profile_id": profile_id, "input_zip": ZIP_NAME, "input_zip_sha256": zsha, "input_zip_bytes": len(zip_b),
           "code_tree_sha256": code_meta["code_tree_sha256"], "git_commit": code_meta["git_head"], "git_dirty": code_meta["git_dirty"],
           "base": {"id": DR.BASE_REPO, "revision": DR.BASE_REVISION, "tokenizer_sha256": DR.TOKENIZER_SHA256, "chat_template_sha256": DR.CHAT_TEMPLATE_SHA256},
           "embedding": {"id": LE.MODEL_ID, "revision": LE.MODEL_REVISION, "dimension": LE.EXPECTED_DIMENSION},
           "pins_eval": FP.PINS_EVAL, "expect_versions": FP.EXPECT_EVAL, "dataset_version": ddir.name,
           "adapters": {k: {"adapter_id": ids[k], "sha256": FP._sha((Path(a["dir"]) / "adapter_model.safetensors").read_bytes()),
                            "card_id": a["manifest"]["card_id"]} for k, a in ads.items()},
           "adapter": {"adapter_id": ids["r1"], "sha256": FP._sha((Path(ads["r1"]["dir"]) / "adapter_model.safetensors").read_bytes())},
           "prompts": {"old": {"card_id": skill_card, "prompt_sha256": card["manifest"]["prompt_sha256"]},
                       "new": {"card_id": skill_card, "prompt_sha256": card["manifest"]["prompt_sha256"], "note": "single Skill card in the second round"}},
           "frozen": {"fast_results_zip_sha256": FP._sha(Path(fast_results_zip).read_bytes()), "qa_sha256": hashlib.sha256(qa).hexdigest(),
                      "split": {"dev": len(split["dev"]), "holdout": len(split["holdout"])}},
           "counts": {"archive_messages": n_msgs, "plan": plan}, "rules": {"r2": EV2.R2_RULE, "memory_v2": EV2.MEMORY_V2_RULE, "unseen": EV2.UNSEEN_RULE},
           "run_salt": "v2-" + zsha[:16], "tests_dir": "code/tests", "code_dir": "code",
           "contains": {"test_split": False, "raw_export": False, "base_weights": False, "api_keys": False, "structured_memories": False},
           "privacy_notice": S.CLOUD_NOTICE, "created_at": datetime.now(timezone.utc).isoformat()}
    out = Path(out_dir) if out_dir else pdir / "packages" / OUT_DIR
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    (tmp / ZIP_NAME).write_bytes(zip_b)
    (tmp / NB_NAME).write_bytes(notebook_bytes())
    (tmp / "PACKAGE_MANIFEST.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    (tmp / "README_COLAB_CN.md").write_text(README_CN, encoding="utf-8", newline="\n")
    sums = "".join(f"{FP._sha((tmp / n).read_bytes())}  {n}\n" for n in (ZIP_NAME, "PACKAGE_MANIFEST.json", NB_NAME, "README_COLAB_CN.md"))
    (tmp / "SHA256SUMS.txt").write_text(sums, encoding="utf-8", newline="\n")
    if out.exists():
        raise PackageV2Error(f"{out} exists: packages are never overwritten (move it away or pass another --out-dir)")
    tmp.rename(out)
    S.append_lineage(pdir, {"artefact": f"packages/{out.name}", "kind": "eval_v2_package", "zip_sha256": zsha})
    return {"package_dir": str(out), "zip": str(out / ZIP_NAME), "zip_mb": round(len(zip_b) / 2**20, 1), "manifest": man}


# ---------------------------------------------------------------- notebook
INTRO = """# StillHere · 第二轮 Colab 评测（r1 vs r2 · Memory V1 vs V2 · Writer Gate）

一次 Run all。每个阶段打印 PASS / INCOMPLETE / SKIP，并写入 Drive；断线后重新 Run all 会续跑（已完成的生成和已付费的评分不会重做）。

| 阶段 | 内容 |
|---|---|
| 00–05 | 配置、GPU、挂载、SHA256、依赖、Base + r1 + r2、Embedding 模型（固定 revision） |
| 06 | 冻结输入：FAST 的 100 道真实历史题（sha 01af8c54…）、100 条 DEV 风格样本、20 个 Writer 审计会话、DEV / HOLDOUT 划分 |
| 07b | 本地 dense 索引（Qwen3-Embedding-0.6B，在 Colab GPU 上重建；断点续建） |
| 13 | Memory：M0 / M1（V1）/ V2L / V2D / V2D_R2 |
| 07 | Writer：合成金标准 + 20 个会话的小样本 + Writer Judge → Full Build Gate（默认不做全量构建） |
| 12 / 12b | Style：A / B1 / R1T / R1S / R2S / R1S_T10 / R2S_T10；通用知识保留：GA / GB1 / GR1 / GR2 |
| 14–15 | DeepSeek Judge（与 FAST 相同的 rubric 1.0）+ 标定 |
| 17 / 19 | 统计、预注册决策（R2_RULE、MEMORY_V2_RULE、Writer Gate）、导出结果 ZIP |
"""

CONFIG = r'''# STAGE 00 · Configuration（只改这里）
DRIVE_DIR = "/content/drive/MyDrive/StillHere/eval_v2"   # 放 3 个输入文件的 Drive 目录
RUN_NAME = ""                       # 留空：按输入 ZIP 的 sha256 自动命名（断线重跑会续上同一次实验）
JUDGE_BUDGET_CNY = 45               # DeepSeek 评分费用上限（人民币）
JUDGE_CONCURRENCY = 4
ALLOW_FULL_BUILD = False            # True：Writer Gate 判 FULL_BUILD 时才做全量结构化记忆构建（耗时很长，默认关闭）
ACCEPT_AUTO_VERIFIED_BY = "owner (second Colab run)"
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
'''

MOUNT = FP.MOUNT.replace('"StillHere_Final_L4_Input.zip"', f'"{ZIP_NAME}"').replace(
    'RUN_NAME = RUN_NAME or (EVAL_MODE.lower() + "-run-" + MAN["input_zip_sha256"][:10])   # FAST / FULL never share a run directory',
    'RUN_NAME = RUN_NAME or ("v2-run-" + MAN["input_zip_sha256"][:10])').replace('Path("/content/stillhere_final")', 'Path("/content/stillhere_v2")')
VERIFY = FP.VERIFY.replace("StillHere_Final_L4_Input.zip", ZIP_NAME)
INSTALL = FP.INSTALL.replace("StillHere_Final_L4_Input.zip", ZIP_NAME)

LOAD = r'''# STAGE 05 · Qwen3-8B base (pinned) + r1 + r2 adapters (ONE model) + Qwen3-Embedding-0.6B (pinned)
from huggingface_hub import snapshot_download
from stillhere.persona import doctor as DR, colab_eval as CE, eval_v2 as EV2, store as S
HF_HOME = LOCAL / "hf"; os.environ["HF_HOME"] = str(HF_HOME)
BASE = Path(snapshot_download(MAN["base"]["id"], revision=MAN["base"]["revision"], cache_dir=str(HF_HOME / "hub"),
                              allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json", "tokenizer*"]))
bc = DR.base_check(BASE)
assert bc["tokenizer_sha256_ok"] and bc["chat_template_sha256_ok"] and bc["revision_in_path"], bc
EMB = Path(snapshot_download(MAN["embedding"]["id"], revision=MAN["embedding"]["revision"], cache_dir=str(HF_HOME / "hub"),
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json", "tokenizer*"]))
WORK = LOCAL / "data" / "personas"
PID = MAN["profile_id"]
ADS = {}
for k, a in MAN["adapters"].items():
    d = S.profile_dir(WORK, PID) / "adapters" / a["adapter_id"] / "adapter"
    assert sha256(d / "adapter_model.safetensors") == a["sha256"], f"{k} adapter sha256 mismatch"
    ADS[a["adapter_id"]] = d
from d5pilot.evaluate import Gen
GEN = Gen(BASE, ADS)
MAN["code_dir"] = str(CODE)
RUN = CE.Run(work_root=WORK, profile_id=PID, run_dir=RUN_DIR, manifest=MAN, cfg=EV2.defaults({"allow_full_build": ALLOW_FULL_BUILD,
             "accept_auto_verified_by": ACCEPT_AUTO_VERIFIED_BY, "judge_concurrency": JUDGE_CONCURRENCY}), gen=GEN)
import torch
ENV = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "versions": vers, "base": MAN["base"], "base_check": bc,
       "embedding": MAN["embedding"], "python": sys.version.split()[0]}
RUN.status("V2 STAGE 05 model", "PASS", gpu=ENV["gpu"], adapters=list(MAN["adapters"]))
'''

STAGES = [
    "# STAGE 06 · Frozen inputs (FAST QA / style sample / writer sessions / DEV-HOLDOUT split)\n"
    "with CE.timer(RUN, \"V2 STAGE 06 frozen inputs\"):\n"
    "    EV2.seed_frozen(RUN, fast_zip=LOCAL / \"frozen\" / \"StillHere_FAST_L4_Results.zip\",\n"
    "                    split=json.loads((LOCAL / \"frozen\" / \"MEMORY_V2_SPLIT.json\").read_text()), expect_qa_sha256=MAN[\"frozen\"][\"qa_sha256\"])\n",
    "# STAGE 07b · Local dense index (Qwen3-Embedding-0.6B on the GPU; resumable)\nwith CE.timer(RUN, \"V2 STAGE 07b dense\"):\n"
    "    EV2.stage_dense(RUN, EMB)\nDENSE, WHY = EV2.dense_channel(RUN, EMB)\nprint(\"dense:\", WHY)\n",
    "# STAGE 13 · Memory groups on the evidence archive: M0 / M1 / V2L / V2D / V2D_R2\nwith CE.timer(RUN, \"V2 STAGE 13 memory\"):\n"
    "    EV2.stage_memory(RUN, dense=DENSE)\n",
    "# STAGE 07 · Writer: synthetic gold + 20 frozen sessions (isolated copy) + Writer judge -> Full Build Gate (+ optional full build)\n"
    "with CE.timer(RUN, \"V2 STAGE 07 writer gate\"):\n"
    "    GATE = EV2.stage_writer_gate(RUN, synthetic_dir=LOCAL / \"synthetic\", env=os.environ)\n"
    "if isinstance(GATE.get(\"full_build\"), dict) and GATE[\"full_build\"].get(\"built\"):\n"
    "    EV2.stage_memory(RUN, dense=DENSE, groups=[EV2.STRUCTURED_GROUP])\n",
    "# STAGE 12 · Style DEV generation (7 groups x 100)\nwith CE.timer(RUN, \"V2 STAGE 12 style\"):\n    EV2.stage_style(RUN)\n",
    "# STAGE 12b · Base capability retention (40 public questions x 4 groups, keyword scoring)\nwith CE.timer(RUN, \"V2 STAGE 12b general\"):\n"
    "    EV2.stage_general(RUN)\n",
    "# STAGE 14-15 · DeepSeek judge: calibration + style + memory (same rubric as FAST)\nwith CE.timer(RUN, \"V2 STAGE 14-15 judge\"):\n"
    "    EV2.stage_judge(RUN, os.environ)\n",
    "# STAGE 17 · Analysis + pre-registered decisions\nwith CE.timer(RUN, \"V2 STAGE 17 analysis\"):\n    REPORT = EV2.analysis(RUN)\n"
    "print(json.dumps(REPORT[\"decisions\"], ensure_ascii=False, indent=1))\n",
    "# STAGE 19 · Export\nwith CE.timer(RUN, \"V2 STAGE 19 export\"):\n    OUT = EV2.stage_export(RUN, ENV)\n"
    "print(\"下载这个文件：\", OUT)\nprint(\"阶段状态：\", RUN_DIR / \"STAGE_STATUS.json\")\n",
]


def notebook() -> dict:
    cells = [("markdown", INTRO), ("code", CONFIG), ("code", FP.PREFLIGHT), ("code", MOUNT), ("code", VERIFY), ("code", INSTALL), ("code", LOAD)] + \
        [("code", s) for s in STAGES]
    return {"cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(True), **({"outputs": [], "execution_count": None} if t == "code" else {})}
                      for t, s in cells],
            "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}


def notebook_bytes() -> bytes:
    return (json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n").encode("utf-8")


README_CN = """# StillHere 第二轮 Colab 评测：使用说明

## 1. 上传（只需一次）
在 Google Drive 新建 `MyDrive/StillHere/eval_v2/`，放入同一个包目录里的 3 个文件：
- `StillHere_V2_Input.zip`
- `PACKAGE_MANIFEST.json`
- `SHA256SUMS.txt`

`StillHere_V2_Eval.ipynb` 放在任意 Drive 位置，用 Google Colaboratory 打开。

## 2. DeepSeek Key
Colab 左侧钥匙图标（Secrets）→ 添加 `DEEPSEEK_API_KEY`，打开 Notebook access。

## 3. 运行
Runtime → Change runtime type → **A100 GPU**（L4 也可以，更慢）→ Run all。依赖装完提示版本不符：Restart session → Run all（会续跑）。

## 4. 结果
`MyDrive/StillHere/eval_v2/runs/v2-run-<输入 ZIP sha 前 10 位>/export/StillHere_V2_Results.zip`（STAGE 19 会打印实际路径）。
下载到本地：`<data-dir>\\personas\\<research-profile>\\downloads\\`。
"""
