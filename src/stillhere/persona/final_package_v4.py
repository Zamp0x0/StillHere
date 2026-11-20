"""Input package of the FINAL (fourth-round) Colab evaluation (``persona.eval_v4``): OLD vs NEW Reader on the frozen V4 AI-verified set.

Contents (private; never in git): profile, the Skill card, the dataset TRAIN / DEV metadata files (never the sealed test), the evidence
archive (memory.sqlite, no structured memories), the frozen V4 gold + its manifest + the evaluation rows, code + tests (incl. the public
synthetic memory fixture for the optional Writer diagnostics). NO adapter, no base or embedding weights (downloaded on Colab, pinned).
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

from . import doctor as DR
from . import eval_v4 as EV4
from . import final_package as FP
from . import final_package_v3 as FP3
from . import package as P
from . import store as S

OUT_DIR = "eval_v4_colab"
ZIP_NAME = "StillHere_V4_Input.zip"
NB_NAME = "StillHere_V4_Final_Reader_Eval.ipynb"
SCHEMA = "stillhere.persona.eval_v4_package/1.0"
DRIVE_DIR = "/content/drive/MyDrive/StillHere/eval_v4"
EXTRA_TESTS = ("tests/test_persona_memory_v2.py", "tests/test_p0_retrieval.py", "tests/test_eval_v3.py", "tests/test_eval_v4.py",
               "tests/test_p1_reader_writer.py", "tests/persona_memory_fixture.py", "tests/persona_fixture.py")


class PackageV4Error(ValueError):
    pass


def build(root, profile_id: str, *, skill_card: str, gold_jsonl: Path, gold_manifest: Path, out_dir: Optional[Path] = None) -> dict:
    from . import card as K
    from stillhere.retrieval import local_embedding as LE
    root = Path(root) if root else S.default_root()
    pdir = S.profile_dir(root, profile_id)
    card = K.load(pdir, skill_card)
    gman = json.loads(Path(gold_manifest).read_text(encoding="utf-8"))
    gold_b = Path(gold_jsonl).read_bytes()
    if hashlib.sha256(gold_b).hexdigest() != gman["sha256"] or gman.get("gold_source") != "AI_VERIFIED" or gman.get("is_human_gold") is not False:
        raise PackageV4Error("the gold file is not the frozen AI_VERIFIED set of its manifest")
    db = pdir / "memory" / "memory.sqlite"
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        n_msgs = con.execute("SELECT COUNT(*) FROM messages WHERE target_persona_id = ?", (profile_id,)).fetchone()[0]
        n_mem = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    finally:
        con.close()
    if n_mem:
        raise PackageV4Error(f"the archive holds {n_mem} structured memories; this evaluation reads the evidence archive only")
    rows_b = FP3.eval_rows(gold_jsonl, db, profile_id)
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
        raise PackageV4Error("the sealed test split may not be packaged")
    if any("/adapters/" in n for n, _ in entries):
        raise PackageV4Error("no adapter belongs in this package")
    code_b, code_meta = P.code_zip()
    with zipfile.ZipFile(io.BytesIO(code_b)) as z:
        code_entries = [(f"code/{n}", z.read(n)) for n in z.namelist()]
    have = {n for n, _ in code_entries}
    for t in FP.TESTS + EXTRA_TESTS:
        tp = P.ROOT / t
        if tp.is_file() and f"code/{t}" not in have:
            code_entries.append((f"code/{t}", tp.read_bytes()))
            have.add(f"code/{t}")
    inner = {"schema": SCHEMA, "files": {n: FP._sha(b) for n, b in sorted(entries + code_entries)}}
    zip_b = P.det_zip(entries + code_entries + [("INNER_MANIFEST.json", json.dumps(inner, indent=1, sort_keys=True).encode("utf-8"))])
    zsha = FP._sha(zip_b)
    n_q = len([l for l in gold_b.decode("utf-8").splitlines() if l.strip()])
    man = {"schema": SCHEMA, "package_id": "v4-" + zsha[:12], "profile_id": profile_id, "input_zip": ZIP_NAME, "input_zip_sha256": zsha, "input_zip_bytes": len(zip_b),
           "code_tree_sha256": code_meta["code_tree_sha256"], "git_commit": code_meta["git_head"], "git_dirty": code_meta["git_dirty"],
           "base": {"id": DR.BASE_REPO, "revision": DR.BASE_REVISION, "tokenizer_sha256": DR.TOKENIZER_SHA256, "chat_template_sha256": DR.CHAT_TEMPLATE_SHA256},
           "embedding": {"id": LE.MODEL_ID, "revision": LE.MODEL_REVISION, "dimension": LE.EXPECTED_DIMENSION},
           "pins_eval": FP.PINS_EVAL, "expect_versions": FP.EXPECT_EVAL, "dataset_version": ddir.name, "adapters": {},
           "prompts": {"old": {"card_id": skill_card, "prompt_sha256": card["manifest"]["prompt_sha256"]}},
           "gold": {"name": gman["name"], "file": gname, "sha256": gman["sha256"], "n": n_q, "gold_source": "AI_VERIFIED", "is_human_gold": False,
                    "eval_rows_sha256": FP._sha(rows_b)},
           "groups": {g: dict(sp) for g, sp in EV4.MEMORY_SPECS_V4.items()}, "core_groups": list(EV4.CORE), "reader_versions": EV4.reader_versions(),
           "plan": {"memory": len(EV4.MEMORY_SPECS_V4) * n_q}, "rule": EV4.RULE_V4, "counts": {"archive_messages": n_msgs},
           "run_salt": "v4-" + zsha[:16], "tests_dir": "code/tests", "code_dir": "code",
           "contains": {"test_split": False, "raw_export": False, "base_weights": False, "adapters": False, "api_keys": False, "structured_memories": False},
           "privacy_notice": S.CLOUD_NOTICE, "created_at": datetime.now(timezone.utc).isoformat()}
    out = Path(out_dir) if out_dir else pdir / "packages" / OUT_DIR
    if out.exists():
        raise PackageV4Error(f"{out} exists: packages are never overwritten (move it away or pass another out_dir)")
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    (tmp / ZIP_NAME).write_bytes(zip_b)
    (tmp / NB_NAME).write_bytes(notebook_bytes())
    (tmp / "PACKAGE_MANIFEST.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    (tmp / "README_COLAB_CN.md").write_text(README_CN, encoding="utf-8", newline="\n")
    (tmp / "V4_EVAL_PROTOCOL.json").write_text(json.dumps(protocol(man), ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    names = (ZIP_NAME, "PACKAGE_MANIFEST.json", NB_NAME, "README_COLAB_CN.md", "V4_EVAL_PROTOCOL.json")
    (tmp / "SHA256SUMS.txt").write_text("".join(f"{FP._sha((tmp / n).read_bytes())}  {n}\n" for n in names), encoding="utf-8", newline="\n")
    tmp.rename(out)
    S.append_lineage(pdir, {"artefact": f"packages/{out.name}", "kind": "eval_v4_package", "zip_sha256": zsha})
    return {"package_dir": str(out), "zip": str(out / ZIP_NAME), "zip_mb": round(len(zip_b) / 2**20, 1), "manifest": man}


def protocol(man: dict) -> dict:
    """The frozen V4 protocol (next to the ZIP and inside PACKAGE_MANIFEST)."""
    return {"schema": "stillhere.persona.v4_protocol/1.0", "gold": man["gold"], "groups": man["groups"], "core_groups": man["core_groups"],
            "reader_versions": man["reader_versions"], "rule": man["rule"], "base": man["base"], "embedding": man["embedding"],
            "same_for_all_groups": ["frozen questions and gold", "base model revision", "decoding (runtime defaults)", "judge model and rubric 1.0",
                                    "ask time of the gold set", "evidence archive", "Reader prompt FACT_SYSTEM (sha in reader_versions)"],
            "metrics": ["judge mean", "answerable / grounded accuracy", "unanswerable correct abstention", "false abstention", "valid answer preservation",
                        "unsupported claims", "attribution / temporal errors", "third-party attribution errors", "attribution false rejections with gold",
                        "gold evidence recall", "input / output tokens", "latency", "judge failures / retries"],
            "results_contract": {"zip": "export/StillHere_V4_Results.zip", "contains": list(EV4.EXPORT) + ["memory_qa/<group>/answers.jsonl", "RUN_MANIFEST_V4.json",
                                                                                                    "SHA256SUMS.txt"], "never_contains": ["sqlite", "API keys"]},
            "note": "AI_VERIFIED gold (two AI review passes), not human gold; the result is engineering evidence"}


# ---------------------------------------------------------------- notebook
INTRO = """# StillHere · 最终 Colab 评测 V4（旧 Reader vs 新 Reader · 新独立集 AI_VERIFIED V4）

一次 Run all。每个阶段打印 PASS / INCOMPLETE / FAIL / SKIP 并写入 Drive；断线后重新 Run all 会续跑（已完成的生成和已付费的评分不会重做）。

**注意：** 标准答案由 AI 双阶段审核（AI_VERIFIED），不是人工 Gold。任何组不完整时，分析会报 INCOMPLETE，不给出采用结论。

| 阶段 | 内容 |
|---|---|
| 00–05 | 配置、GPU、挂载、SHA256、依赖、Qwen3-8B Base（固定 revision，不加载 Adapter）、Qwen3-Embedding-0.6B |
| 06 | 冻结输入：V4 AI_VERIFIED Gold（sha 与 Manifest 一致才继续） |
| 07b | dense 索引：在 /content 本地盘上建立和查询（Drive 只保存快照，供续跑） |
| 13 | Memory：V2L_OLD / V2L_NEW / V2D_NEW（最多 3 轮补跑；仍不完整则标 INCOMPLETE） |
| 07w | （可选）合成聊天的 Writer 诊断：婚礼问题的完整记录；失败不影响其他阶段 |
| 14–15 | DeepSeek Judge（与 FAST 相同的 rubric）+ 标定 |
| 17 / 19 | 统计、完整性门槛、预注册规则、导出结果 ZIP |
"""

CONFIG = r'''# STAGE 00 · Configuration（只改这里）
DRIVE_DIR = "%s"   # 放 3 个输入文件的 Drive 目录
RUN_NAME = ""                       # 留空：按输入 ZIP 的 sha256 自动命名（断线重跑会续上同一次实验）
JUDGE_BUDGET_CNY = 20               # DeepSeek 评分费用上限（人民币）
JUDGE_CONCURRENCY = 4
RUN_WRITER_DIAGNOSTICS = True       # 可选：合成聊天的 Writer 诊断（几分钟；失败不影响 Reader 评测）
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
    'RUN_NAME = RUN_NAME or ("v4-run-" + MAN["input_zip_sha256"][:10])').replace('Path("/content/stillhere_final")', 'Path("/content/stillhere_v4")')
VERIFY = FP.VERIFY.replace("StillHere_Final_L4_Input.zip", ZIP_NAME)
INSTALL = FP.INSTALL.replace("StillHere_Final_L4_Input.zip", ZIP_NAME)
LOAD = (FP3.LOAD.replace("eval_v3 as EV3, eval_v2 as EV2", "eval_v3 as EV3, eval_v2 as EV2, eval_v4 as EV4")
        .replace('cfg=EV3.defaults({"judge_concurrency": JUDGE_CONCURRENCY})', 'cfg=EV4.defaults({"judge_concurrency": JUDGE_CONCURRENCY})')
        .replace('RUN.status("V3 STAGE 05 model"', 'RUN.status("V4 STAGE 05 model"'))

STAGES = [
    "# STAGE 06 · Frozen V4 AI-verified gold (sha must match the manifest)\nwith CE.timer(RUN, \"V4 STAGE 06 frozen gold\"):\n"
    "    EV4.seed_frozen(RUN, gold_jsonl=LOCAL / \"frozen\" / MAN[\"gold\"][\"file\"], gold_manifest=LOCAL / \"frozen\" / \"AI_VERIFIED_NEW_GOLD_MANIFEST.json\",\n"
    "                    eval_rows_jsonl=LOCAL / \"frozen\" / \"NEW_GOLD_EVAL_ROWS.jsonl\")\n",
    "# STAGE 07b · Dense index on LOCAL disk (/content); Drive keeps a snapshot for resume\nDENSE_LOCAL = Path(\"/content/stillhere_v4_dense\")\n"
    "with CE.timer(RUN, \"V4 STAGE 07b dense\"):\n    EV2.stage_dense(RUN, EMB, local_dir=DENSE_LOCAL)\nDENSE, WHY = EV2.dense_channel(RUN, EMB, local_dir=DENSE_LOCAL)\n"
    "print(\"dense:\", WHY)\n",
    "# STAGE 13 · Memory groups V2L_OLD / V2L_NEW / V2D_NEW (retries only what is missing; INCOMPLETE is never reported as PASS)\n"
    "with CE.timer(RUN, \"V4 STAGE 13 memory\"):\n    COMP = EV4.stage_memory(RUN, dense=DENSE)\n"
    "for g, v in COMP.items():\n    print(g, f\"{v['ok']}/{v['of']}\", \"OK\" if v[\"complete\"] else f\"INCOMPLETE {v['errors']}\")\n"
    "if not all(v[\"complete\"] for v in COMP.values()):\n    print(\"有组不完整：可以先继续评分；之后重新 Run all 会只补缺失的题。分析不会给出采用结论。\")\n",
    "# STAGE 07w · (optional) Writer diagnostics on the public synthetic chat\nif RUN_WRITER_DIAGNOSTICS:\n"
    "    with CE.timer(RUN, \"V4 STAGE 07w writer\"):\n        EV4.stage_writer_diagnostics(RUN, LOCAL / \"synthetic\")\n"
    "else:\n    RUN.status(\"V4 STAGE 07w writer diagnostics (optional)\", \"SKIP\")\n",
    "# STAGE 14-15 · DeepSeek judge: calibration + memory (same rubric as FAST)\nwith CE.timer(RUN, \"V4 STAGE 14-15 judge\"):\n    EV4.stage_judge(RUN, os.environ)\n",
    "# STAGE 17 · Analysis + completeness gate + registered rule\nwith CE.timer(RUN, \"V4 STAGE 17 analysis\"):\n    REPORT = EV4.analysis(RUN)\n"
    "print(json.dumps(REPORT[\"decision\"], ensure_ascii=False, indent=1)[:3000])\n",
    "# STAGE 19 · Export\nwith CE.timer(RUN, \"V4 STAGE 19 export\"):\n    OUT = EV4.stage_export(RUN, ENV)\n"
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


README_CN = """# StillHere 最终 Colab 评测 V4：使用说明

## 1. 上传（只需一次）
在 Google Drive 新建 `MyDrive/StillHere/eval_v4/`，放入同一个包目录里的 3 个文件：
- `StillHere_V4_Input.zip`
- `PACKAGE_MANIFEST.json`
- `SHA256SUMS.txt`

`StillHere_V4_Final_Reader_Eval.ipynb` 也放在 `MyDrive/StillHere/eval_v4/`，在 Drive 里双击，用 Google Colaboratory 打开。

## 2. DeepSeek Key
Colab 左侧钥匙图标（Secrets）→ 添加 `DEEPSEEK_API_KEY`，打开 Notebook access。

## 3. 运行
Runtime → Change runtime type → **A100 GPU**（拿不到可选 L4，更慢）→ Run all。依赖装完提示版本不符：Runtime → Restart session → Run all（会续跑）。
STAGE 00 默认值不用改。

## 4. 断线
重新连接 → Run all。已完成的生成和评分不会重做；dense 索引会从 Drive 快照恢复到本地盘。

## 5. 结果
`MyDrive/StillHere/eval_v4/runs/v4-run-<输入 ZIP sha 前 10 位>/export/StillHere_V4_Results.zip`（STAGE 19 会打印实际路径）。
下载到本地：`<data-dir>\\personas\\<research-profile>\\downloads\\StillHere_V4_Results.zip`。

本轮标准答案是 AI_VERIFIED（AI 双阶段审核），不是人工 Gold。
"""
