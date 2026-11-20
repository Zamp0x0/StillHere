#!/usr/bin/env python3
"""Build the Colab L4 package of the P2 final memory evaluation from ONE commit (CPU only; no network; no API call).

    python evaluation/p2_final_memory/build_release.py --dataset <official json> --pilot-raw <extracted pilot Results ZIP> [--commit HEAD]

Writes releases/p2_final_memory/: StillHere_P2_Final_Memory_Eval_Input.zip, StillHere_P2_Final_Memory_Eval_Colab.ipynb,
P2_MANIFEST.json, SHA256SUMS.txt, P2_COLAB_RUN_GUIDE_CN.md. Refuses when the OLD-path files differ from the pilot's code.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "evaluation" / "lifebench_pilot"), str(ROOT / "evaluation" / "p2_final_memory")]

from lbpilot import select as LS  # noqa: E402
from lbpilot import timeline as TL  # noqa: E402
from p2eval import protocol as P  # noqa: E402
from p2eval import select_new as SN  # noqa: E402

sys.path.insert(0, str(ROOT / "evaluation" / "lifebench_pilot"))
import build_release as PILOT_BUILD  # noqa: E402  (the pilot builder: deterministic zip, smoke fixture)

REL = ROOT / "releases" / "p2_final_memory"
ZIP_NAME = "StillHere_P2_Final_Memory_Eval_Input.zip"
NB_NAME = "StillHere_P2_Final_Memory_Eval_Colab.ipynb"
DRIVE = "/content/drive/MyDrive/StillHere/p2_final_memory"
PATHS = ("src/stillhere", "evaluation/lifebench_pilot/lbpilot", "evaluation/p2_final_memory/p2eval", "pyproject.toml")
REQUIREMENTS = "huggingface_hub\naiohttp\ntqdm\n"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def git(*a) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *a], check=True, capture_output=True, text=True).stdout.strip()


def blob(commit: str, path: str) -> bytes:
    return subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{path}"], check=True, capture_output=True).stdout


def code_entries(commit: str):
    raw = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=zip", commit, *PATHS], check=True, capture_output=True).stdout
    out = []
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for n in sorted(z.namelist()):
            if n.endswith("/") or "__pycache__" in n:
                continue
            if any(f in n for f in PILOT_BUILD.FORBIDDEN):
                raise SystemExit(f"REFUSED: {n}")
            out.append((f"code/{n}", z.read(n)))
    return out


def cell(src, kind="code"):
    return PILOT_BUILD.cell(src, kind)


def notebook(zip_sha: str, m: dict) -> dict:
    cells = [cell(f"""# StillHere P2 · 最终 Memory 评测（Colab L4，Run All）

- **决策集**：新人物 {m['new_person']['person']}（100 题，PERSONA_SEEN_QA_UNSEEN），OLD_V2D vs NEW_P2_V2D，主指标：可回答题 Grounded 准确率。
- **回归集**：于晓薇 100 题（已用于根因分析：KNOWN_SET_REGRESSION），只跑 NEW，与 Pilot 保存的 OLD 对比。
- 回答 `deepseek-flash`；官方评分提示词 × 3；证据上限 4,000 token；不跑 Mem0、不跑无记忆组、不加载 Adapter。
- 断线后再 Run All（RUN_ID 不变）：已完成的检索、回答、评分不会重复请求。""", "markdown"),
             cell(f'''# 00 · 设置
DRIVE_ROOT = "{DRIVE}"
RUN_ID = "p2_final_v1"
ZIP_SHA256 = "{zip_sha}"
SMOKE_FIRST = True
import os, subprocess, sys, json, time
print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True, capture_output=True, text=True).stdout)'''),
             cell('''# 01 · Google Drive
from google.colab import drive
drive.mount("/content/drive")
from pathlib import Path
BASE = Path(DRIVE_ROOT); RUN_DIR = BASE / "runs" / RUN_ID; SCRATCH = Path("/content/p2") / RUN_ID
for p in (BASE / "input", RUN_DIR, SCRATCH, BASE / "exports"): p.mkdir(parents=True, exist_ok=True)'''),
             cell(f'''# 02 · 输入 ZIP 校验与解包
import hashlib, zipfile, shutil
ZIP = BASE / "input" / "{ZIP_NAME}"
assert ZIP.is_file(), f"把 {ZIP_NAME} 上传到 {{BASE / 'input'}}"
assert hashlib.sha256(ZIP.read_bytes()).hexdigest() == ZIP_SHA256, "ZIP sha256 不一致"
PKG = Path("/content/p2/pkg"); shutil.rmtree(PKG, ignore_errors=True); zipfile.ZipFile(ZIP).extractall(PKG)
sums = dict(l.split("  ", 1)[::-1] for l in (PKG / "SHA256SUMS.txt").read_text().splitlines() if l.strip())
assert not [n for n, s in sums.items() if hashlib.sha256((PKG / n).read_bytes()).hexdigest() != s]
MAN = json.loads((PKG / "P2_MANIFEST.json").read_text(encoding="utf-8"))
sys.path[:0] = [str(PKG / "code" / "src"), str(PKG / "code" / "evaluation" / "lifebench_pilot"), str(PKG / "code" / "evaluation" / "p2_final_memory")]
print("package OK, commit", MAN["git_commit"][:12])'''),
             cell('''# 03 · 依赖 + 环境 + OLD 指纹
print(subprocess.run(f"pip install -q -r {PKG / 'requirements-p2.txt'}", shell=True, capture_output=True, text=True).stdout[-800:])
os.environ.pop("OPENAI_API_KEY", None)
from p2eval import colab_p2 as C, protocol as P
assert P.P2_PROTOCOL_SHA256 == MAN["protocol_sha256"], "protocol changed"
print(C.verify_old_fingerprint(PKG / "code", MAN)); print(json.dumps(C.environment(RUN_DIR)["packages"], indent=1))'''),
             cell('''# 04 · DeepSeek 预检（Colab Secrets: DEEPSEEK_API_KEY）
KEY = C.read_key(Path("/content/drive/MyDrive/StillHere/_private/secrets/lifebench.env"))
ANS = C.chat_client(KEY, RUN_DIR, "answer", max_tokens=1024); JUD = C.chat_client(KEY, RUN_DIR, "judge", max_tokens=1024)
print(C.preflight(ANS))'''),
             cell('''# 05 · 官方数据集 + 官方评分代码
DATA = C.dataset(Path("/content/p2/data"), drive_copy=BASE / "input" / "lifebench_locomo_conversation_format_v2.0_3380QA.json")
EVAL = C.official_eval(Path("/content/p2/LifeBench_eval"))
from lbpilot import judge as J
OFFICIAL = J.Official(EVAL); assert OFFICIAL.prompt_sha256 == J.OFFICIAL_PROMPT_SHA256'''),
             cell('''# 06–07 · 冻结题集（新人物 + 已见集）与时间线；Pilot 保存的 OLD 记录
from lbpilot import select as LS, timeline as TL
from p2eval import select_new as SN, runner as R
data = LS.load_dataset(DATA)
FN = json.loads((PKG / "FROZEN_NEW_100QA.json").read_text(encoding="utf-8")); FK = json.loads((PKG / "FROZEN_KNOWN_100QA.json").read_text(encoding="utf-8"))
assert SN.freeze(data)["qa_sha256"] == FN["qa_sha256"] and LS.freeze_data(data)["qa_sha256"] == FK["qa_sha256"]
run = R.Run(RUN_DIR, SCRATCH)
pn, qn = SN.selected(data, FN); pk, qk = LS.selected(data, FK)
inN = run.set_inputs("NEWP", pn, qn, FN["person"]["slug"]); inK = run.set_inputs("KNOWN", pk, qk, FK["person"]["slug"])
assert TL.sha(inN["events"]) == MAN["timelines"]["NEWP"] and TL.sha(inK["events"]) == MAN["timelines"]["KNOWN"]
KNOWN_OLD = C.load_known_old(PKG)
run.stage("inputs", "PASS", new_person=FN["person"]["sample_id"], new_events=len(inN["events"]), known_events=len(inK["events"]))'''),
             cell('''# 08 · Qwen3-Embedding-0.6B（固定 revision，L4）+ 冒烟
EMB_DIR = C.embedding_model()
from stillhere.retrieval import local_embedding as LE
EMBEDDER = LE.get_embedder(EMB_DIR, device="cuda", dtype="float32"); COUNT = C.token_counter(EMB_DIR)
if SMOKE_FIRST and not (RUN_DIR / "smoke" / "SMOKE_REPORT.json").is_file():
    s = C.smoke(RUN_DIR, SCRATCH, fixture=PKG / "smoke" / "synthetic_lifebench.json", key=KEY, embedder=EMBEDDER, official=OFFICIAL, count=COUNT)
    print(s["status"]); assert s["status"] == "PASS", "smoke FAILED"'''),
             cell('''# 09 · 稠密索引（每个人物整条时间线一次；检索时按提问时间屏蔽）
from lbpilot import stillhere_memory as SM
DENSE = {}
for exp in ("NEWP", "KNOWN"):
    C.dense_index(run, exp, EMBEDDER, scratch=SCRATCH)
    DENSE[exp], why = SM.open_dense(SCRATCH / f"dense_{exp}", LE.LocalQueryEncoder(EMBEDDER)); assert DENSE[exp] is not None, why'''),
             cell('''# 10 · 按日期 ADD → 当天检索并冻结上下文（新人物：OLD + NEW；已见集：NEW）
for exp, inp in (("NEWP", inN), ("KNOWN", inK)):
    print(exp, "live at", run.restore_if_needed(exp))
    store = SM.StillHereMemory(run.live(exp) / "run.sqlite", inp["slug"], inp["name"], inp["assistant"], dataset_sha256=P.DATASET["sha256"], locator=P.DATASET["url"])
    print(run.chronological(exp, stillhere=store, dense=DENSE[exp], count=COUNT)); store.close()'''),
             cell('''# 11 · 回答（OLD 用 lbpilot-answer-v1，NEW 用 lbpilot-answer-p2；已完成的不再请求；预算上限见协议）
print(run.run_answers(ANS.chat))'''),
             cell('''# 12 · 评分器校准 + 官方评分 × 3
if "judge_calibration" not in run.state["stages"]: print(C.calibrate(run, OFFICIAL, JUD.chat))
print(run.run_judge(OFFICIAL, JUD.chat))'''),
             cell('''# 13 · 指标、配对统计、预注册采用规则
out = C.metrics(run, KNOWN_OLD); txt = C.report(out); (RUN_DIR / "REPORT.md").write_text(txt, encoding="utf-8"); print(txt)'''),
             cell('''# 14 · 导出 Results ZIP
from lbpilot import export as EX
print(json.dumps(EX.build(RUN_DIR, BASE / "exports" / f"StillHere_P2_Final_Memory_Results_{RUN_ID}.zip"), indent=1))'''),
             ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4", "provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def guide(m: dict) -> str:
    return f"""# Colab 运行指南：StillHere P2 最终 Memory 评测

版本 `{P.P2_EVAL_VERSION}`，协议 sha `{P.P2_PROTOCOL_SHA256[:16]}…`，代码 commit `{m['git_commit']}`。

| 本机文件（Windows） | Google Drive |
|---|---|
| `{REL / ZIP_NAME}` | `{DRIVE}/input/{ZIP_NAME}` |
| `{REL / NB_NAME}` | `{DRIVE}/{NB_NAME}` |
| （可选）官方数据 JSON | `{DRIVE}/input/lifebench_locomo_conversation_format_v2.0_3380QA.json`（否则从 HF 固定 revision 下载并校验） |

ZIP SHA256：`{m['zip_sha256']}`。

1. Drive 里双击 Notebook，用 Colab 打开；“代码执行程序 → 更改运行时类型 → GPU → **L4**”。
2. Secrets（钥匙图标）：`DEEPSEEK_API_KEY`，打开 Notebook 访问权限（沿用上次的即可）。
3. “全部运行”。先跑合成数据冒烟（几分钱），失败会停下。
4. 断线：重新连接，再“全部运行”，**不要改 RUN_ID**；从 Drive 快照续跑，已完成的检索 / 回答 / 评分不会重复付费。

请求量：新人物 200 次回答 + 600 次评分；已见集 100 次回答 + 300 次评分；冒烟与校准约 12 次。没有 Mem0 ADD。
费用上限估计（高峰价）：约 ${m['plan']['cost_upper_usd']}；协议的硬上限 ${P.COST_BUDGET_USD}（超过会停下，不会继续花钱）。时间约 40–70 分钟（两次稠密索引各约 5 分钟）。

结果：`{DRIVE}/exports/StillHere_P2_Final_Memory_Results_p2_final_v1.zip` → 下载到 `<data-dir>\\downloads\\p2_final_memory\\` → 交给 Claude 审计
（`evaluation/p2_final_memory/p2eval/metrics_p2.compute` 只用原始记录重算；采用决定只看新人物，已见集只做回归）。
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pilot-raw", required=True)
    ap.add_argument("--commit", default="HEAD")
    a = ap.parse_args()
    commit = git("rev-parse", a.commit)
    if git("status", "--porcelain", "--", *PATHS):
        raise SystemExit("REFUSED: uncommitted changes in the packaged paths; commit first")
    old_files = {}
    for f in P.OLD_FINGERPRINT_FILES:
        b_new, b_old = blob(commit, f), blob(P.OLD_SOURCE_COMMIT, f)
        if b_new != b_old:
            raise SystemExit(f"REFUSED: OLD-path file {f} differs from {P.OLD_SOURCE_COMMIT}")
        old_files[f] = sha(b_new)
    data = LS.load_dataset(a.dataset)
    fn = SN.freeze(data)
    fk = LS.freeze_data(data)
    pn, qn = SN.selected(data, fn)
    pk, qk = LS.selected(data, fk)
    tn, tk = TL.build(pn, until=fn["ask_time_range"][1]), TL.build(pk, until=fk["ask_time_range"][1])
    raw = Path(a.pilot_raw)
    if hashlib.sha256((raw / "SHA256SUMS.txt").read_bytes()).hexdigest() and not (raw / "STATE.json").is_file():
        raise SystemExit("pilot raw results missing")

    def m2(name, key="group"):
        rows = [json.loads(l) for l in open(raw / name, encoding="utf-8") if l.strip()]
        return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows if r[key] == "M2_STILLHERE_V2D").encode("utf-8")
    files = {"FROZEN_NEW_100QA.json": json.dumps(fn, ensure_ascii=False, indent=1).encode("utf-8"),
             "FROZEN_KNOWN_100QA.json": json.dumps(fk, ensure_ascii=False, indent=1).encode("utf-8"),
             "P2_PROTOCOL.json": json.dumps(P.protocol(), ensure_ascii=False, indent=1).encode("utf-8"),
             "requirements-p2.txt": REQUIREMENTS.encode(), "smoke/synthetic_lifebench.json": PILOT_BUILD.smoke_fixture(),
             "old_baseline/answers_M2.jsonl": m2("answers.jsonl"), "old_baseline/contexts_M2.jsonl": m2("contexts.jsonl"), "old_baseline/judge_M2.jsonl": m2("judge.jsonl")}
    cost = (300 * (4500 * 0.30 + 250 * 1.2) + 900 * (2500 * 0.30 + 400 * 1.2)) / 1e6
    manifest = {"schema": "stillhere.p2_final_memory.manifest/1.0", "version": P.P2_EVAL_VERSION, "protocol_sha256": P.P2_PROTOCOL_SHA256, "git_commit": commit,
                "old_fingerprint": {"source_commit": P.OLD_SOURCE_COMMIT, "files": old_files, "tree_sha256": sha("".join(f"{k}:{v}\n" for k, v in sorted(old_files.items())).encode())},
                "new_p2": {"module": "src/stillhere/persona/memory_p2.py", "sha256": sha(blob(commit, "src/stillhere/persona/memory_p2.py")),
                           "answer_prompt": "evaluation/p2_final_memory/p2eval/answer_p2.py", "answer_prompt_sha256": sha(blob(commit, "evaluation/p2_final_memory/p2eval/answer_p2.py"))},
                "dataset": P.DATASET, "new_person": {"person": fn["person"]["sample_id"], "qa_sha256": fn["qa_sha256"], "type_counts": fn["type_counts_multilabel"],
                                                     "excluded_referenced": len(fn["excluded_referenced_ids"]), "validity": P.VALIDITY},
                "known_set": {"person": fk["person"]["sample_id"], "qa_sha256": fk["qa_sha256"], "label": "KNOWN_SET_REGRESSION",
                              "pilot_results_zip_sha256": P.PILOT_RESULTS_ZIP_SHA256},
                "timelines": {"NEWP": TL.sha(tn), "KNOWN": TL.sha(tk)}, "events": {"NEWP": len(tn), "KNOWN": len(tk)},
                "plan": {"answers": 300, "judge_calls": 900, "cost_upper_usd": round(cost * 1.5, 2)}, "requirements": REQUIREMENTS.split(),
                "privacy": "public code, public synthetic LifeBench ids / records and the pilot's own synthetic-data outputs; no private chat, card, adapter, key or private gold"}
    entries = code_entries(commit)
    files["P2_MANIFEST.json"] = json.dumps(manifest, ensure_ascii=False, indent=1).encode("utf-8")
    allf = entries + sorted(files.items())
    sums = "".join(f"{sha(b)}  {n}\n" for n, b in allf)
    zb = PILOT_BUILD.det_zip(allf + [("SHA256SUMS.txt", sums.encode())])
    REL.mkdir(parents=True, exist_ok=True)
    (REL / ZIP_NAME).write_bytes(zb)
    manifest["zip_sha256"] = sha(zb)
    (REL / NB_NAME).write_bytes(json.dumps(notebook(sha(zb), manifest), ensure_ascii=False, indent=1).encode("utf-8"))
    (REL / "P2_MANIFEST.json").write_bytes(json.dumps(manifest, ensure_ascii=False, indent=1).encode("utf-8"))
    (REL / "P2_COLAB_RUN_GUIDE_CN.md").write_bytes(guide(manifest).encode("utf-8"))
    outer = "".join(f"{sha((REL / n).read_bytes())}  {n}\n" for n in (ZIP_NAME, NB_NAME, "P2_MANIFEST.json", "P2_COLAB_RUN_GUIDE_CN.md"))
    (REL / "SHA256SUMS.txt").write_bytes(outer.encode("utf-8"))
    print(json.dumps({"release": str(REL), "zip_sha256": manifest["zip_sha256"], "zip_bytes": len(zb), "commit": commit, "plan": manifest["plan"],
                      "new_person": manifest["new_person"], "timelines": manifest["timelines"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
