#!/usr/bin/env python3
"""Build the Colab L4 input package of the LifeBench 100-QA pilot from ONE commit (CPU only; no network; no API call).

    python evaluation/lifebench_pilot/build_release.py --dataset <lifebench_locomo_conversation_format_v2.0_3380QA.json> [--commit HEAD]

Writes ``releases/lifebench_pilot_100qa/``:
    StillHere_LifeBench_Mem0_100QA_Input.zip   code (git archive of the commit: src/stillhere + evaluation/lifebench_pilot), FROZEN_100QA.json,
                                               PROTOCOL.json, BENCHMARK_MANIFEST.json, requirements-pilot.txt, smoke fixture, RUN_GUIDE, SHA256SUMS
    StillHere_LifeBench_100QA_Mem0_Colab.ipynb the notebook (expects the ZIP's sha256)
    BENCHMARK_MANIFEST.json, SHA256SUMS.txt, COLAB_100QA_RUN_GUIDE_CN.md
The official dataset and LifeBench_eval are NOT inside: the notebook downloads them at the pinned revision / commit and checks them.
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evaluation" / "lifebench_pilot"))

from lbpilot import protocol as P  # noqa: E402
from lbpilot import select as SEL  # noqa: E402
from lbpilot import timeline as TL  # noqa: E402

REL = ROOT / "releases" / "lifebench_pilot_100qa"
ZIP_NAME = "StillHere_LifeBench_Mem0_100QA_Input.zip"
NB_NAME = "StillHere_LifeBench_100QA_Mem0_Colab.ipynb"
DRIVE = "/content/drive/MyDrive/StillHere/lifebench_pilot_100qa"
PATHS = ("src/stillhere", "evaluation/lifebench_pilot/lbpilot", "pyproject.toml")
FORBIDDEN = ("private/", ".env", "secrets", ".sqlite", ".safetensors", "adapter_model")
REQUIREMENTS = "mem0ai==2.2.1\nqdrant-client==1.20.0\nsentence-transformers==6.3.0\naiohttp\ntqdm\nhuggingface_hub\n"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def git(*a) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *a], check=True, capture_output=True, text=True).stdout.strip()


def code_entries(commit: str):
    raw = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=zip", commit, *PATHS], check=True, capture_output=True).stdout
    out = []
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for n in sorted(z.namelist()):
            if n.endswith("/"):
                continue
            if any(f in n for f in FORBIDDEN) or "__pycache__" in n:
                raise SystemExit(f"REFUSED: {n} would be packaged")
            out.append((f"code/{n}", z.read(n)))
    return out


def det_zip(entries) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in entries:
            zi = zipfile.ZipInfo(n, date_time=(2026, 11, 19, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, b)
    return buf.getvalue()


def smoke_fixture() -> bytes:
    """Invented, tiny, LifeBench-shaped: 3 days of a fictional person (no dataset content)."""
    name, asst = "林小测", "林小测的Assistant"
    conv = {"speaker_a": name, "speaker_b": asst}
    rows = [("2025-03-01", [("note1", name, f"{name}在2025-03-01 09:00:00记录了一个笔记：下周去杭州出差。"),
                            ("agent_chat1", name, "我下周要去杭州出差，帮我列个行李清单？"), ("agent_chat1", asst, "带上身份证、电脑、充电器和雨伞。")]),
            ("2025-03-02", [("sms1", name, f"在2025-03-02 12:00:00，有一条{name} 发送 王同事的短信：周三的杭州会议改到下午三点。")]),
            ("2025-03-03", [("calendar1", name, f"{name}在2025-03-03 08:00:00时间设置了一个主题为杭州客户会议的日程，开始时间是2025-03-05 15:00:00。")])]
    for i, (d, evs) in enumerate(rows, 1):
        conv[f"session_{i}_date_time"] = d
        conv[f"session_{i}"] = [{"speaker": s, "dia_id": f"{d}_{k}", "text": t} for k, s, t in evs]
    qa = [{"question": "（提问时间：2025-03-03）我下周去哪个城市出差？", "answer": "杭州。", "evidence": ["note1"], "category": [0], "question_type": ["Single_hop"],
           "question_id": "smoke_qa0", "ask_time": "2025-03-03", "score_points": [{"description": "指出城市是杭州", "score": 10}]},
          {"question": "（提问时间：2025-03-03）杭州的会议改到几点了？", "answer": "下午三点。", "evidence": ["sms1", "calendar1"], "category": [0],
           "question_type": ["Multi_hop"], "question_id": "smoke_qa1", "ask_time": "2025-03-03", "score_points": [{"description": "指出下午三点", "score": 10}]}]
    return json.dumps([{"sample_id": name, "conversation": conv, "qa": qa}], ensure_ascii=False, indent=1).encode("utf-8")


def cell(src: str, kind: str = "code") -> dict:
    lines = src.strip("\n").splitlines(keepends=True)
    c = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def notebook(zip_sha: str, manifest: dict) -> dict:
    cells = [cell(f"""# StillHere × LifeBench 2.0 · 1 人物 100 题 Pilot（Colab L4）

**Run All 即可。** 断线后重新连接、再 Run All：每一步都会从 Google Drive 上的记录继续，已完成的检索、回答和评分不会再请求或再付费。

- 人物：{manifest['frozen']['person']}（{manifest['frozen']['questions']} 题，冻结 sha `{manifest['frozen']['qa_sha256'][:12]}`）
- 四组：M0 无记忆 · M1 StillHere V2L · M2 StillHere V2D · M3 官方 Mem0 OSS {P.MEM0['version']}
- 回答模型：{P.ANSWER['model']}（四组相同）；评分：官方 LifeBench_eval 评分提示词 × {P.JUDGE['repeats']} 次，多数票
- 这是 `{P.SCOPE}`，不是官方完整排行榜。""", "markdown"),
             cell(f'''# 00 · 设置（只需确认）
DRIVE_ROOT = "{DRIVE}"
RUN_ID = "formal_v1"            # 一个 RUN_ID = 一次正式运行；不要中途修改
SMOKE_FIRST = True              # 先用合成数据跑几分钱的端到端检查（真实 Mem0 + 真实 API）
ZIP_SHA256 = "{zip_sha}"
import os, subprocess, sys, json, time
print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True, capture_output=True, text=True).stdout)'''),
             cell('''# 01 · 挂载 Google Drive
from google.colab import drive
drive.mount("/content/drive")
from pathlib import Path
BASE = Path(DRIVE_ROOT); RUN_DIR = BASE / "runs" / RUN_ID; SCRATCH = Path("/content/pilot") / RUN_ID
for p in (BASE / "input", RUN_DIR, SCRATCH, BASE / "exports"): p.mkdir(parents=True, exist_ok=True)
print(RUN_DIR)'''),
             cell(f'''# 02 · 输入 ZIP 的 SHA256 与解包
import hashlib, zipfile, shutil
ZIP = BASE / "input" / "{ZIP_NAME}"
assert ZIP.is_file(), f"把 {ZIP_NAME} 上传到 {{BASE / 'input'}}"
h = hashlib.sha256(ZIP.read_bytes()).hexdigest()
assert h == ZIP_SHA256, f"ZIP sha256 {{h}} != {{ZIP_SHA256}}（上传的不是这次构建的 ZIP）"
PKG = Path("/content/pilot/pkg"); shutil.rmtree(PKG, ignore_errors=True)
zipfile.ZipFile(ZIP).extractall(PKG)
sums = dict(l.split("  ", 1)[::-1] for l in (PKG / "SHA256SUMS.txt").read_text().splitlines() if l.strip())
bad = [n for n, s in sums.items() if hashlib.sha256((PKG / n).read_bytes()).hexdigest() != s]
assert not bad, bad
MAN = json.loads((PKG / "BENCHMARK_MANIFEST.json").read_text(encoding="utf-8"))
sys.path[:0] = [str(PKG / "code" / "src"), str(PKG / "code" / "evaluation" / "lifebench_pilot")]
print("package OK:", len(sums), "files; commit", MAN["git_commit"][:12])'''),
             cell('''# 03 · 安装固定版本依赖（mem0ai / qdrant-client / sentence-transformers），记录实际版本
print(subprocess.run(f"pip install -q -r {PKG / 'requirements-pilot.txt'}", shell=True, capture_output=True, text=True).stdout[-1500:])
os.environ.pop("OPENAI_API_KEY", None); os.environ["MEM0_TELEMETRY"] = "False"
from lbpilot import colab as C, protocol as P
assert P.PROTOCOL_SHA256 == MAN["protocol_sha256"], "protocol changed"
print(json.dumps(C.environment(RUN_DIR)["packages"], indent=1))'''),
             cell('''# 04 · DeepSeek API 预检（Key 来自 Colab Secrets 的 DEEPSEEK_API_KEY；不会打印）
KEY = C.read_key(Path("/content/drive/MyDrive/StillHere/_private/secrets/lifebench.env"))
ANS = C.chat_client(KEY, RUN_DIR, "answer", max_tokens=P.ANSWER["max_tokens"])
JUD = C.chat_client(KEY, RUN_DIR, "judge", max_tokens=1024)
pf = C.preflight(ANS); print(pf)
(RUN_DIR / "PREFLIGHT.json").write_text(json.dumps(pf, indent=1))'''),
             cell('''# 05 · 官方数据集（HF 固定 revision，校验 sha256）与官方 LifeBench_eval（固定 commit）
DATA = C.dataset(Path("/content/pilot/data"), drive_copy=BASE / "input" / "lifebench_locomo_conversation_format_v2.0_3380QA.json")
EVAL = C.official_eval(Path("/content/pilot/LifeBench_eval"))
from lbpilot import judge as J
OFFICIAL = J.Official(EVAL); assert OFFICIAL.prompt_sha256 == J.OFFICIAL_PROMPT_SHA256
print(DATA, OFFICIAL.prompt_sha256[:12])'''),
             cell('''# 06–07 · 冻结的 100 题与规范化时间线（与 Manifest 比对）
from lbpilot import runner as R
FROZEN = json.loads((PKG / "FROZEN_100QA.json").read_text(encoding="utf-8"))
run = R.Run(RUN_DIR, SCRATCH)
inp = run.load_inputs(DATA, FROZEN, MAN["timeline_sha256"])
print(inp["name"], len(inp["qs"]), "questions,", len(inp["events"]), "events up to", inp["last"])'''),
             cell('''# 08 · 本地 Qwen3-Embedding-0.6B（固定 revision，L4 GPU）
EMB_DIR = C.embedding_model()
from stillhere.retrieval import local_embedding as LE
EMBEDDER = LE.get_embedder(EMB_DIR, device="cuda", dtype="float32")
COUNT = C.token_counter(EMB_DIR)
run.stage("08_embedding", "PASS", model=str(EMB_DIR), revision=P.STILLHERE["embedding"]["revision"], token_counter=COUNT.mode)'''),
             cell('''# 08b · 冒烟检查（合成数据、几分钱）：真实 Mem0 SDK + 真实 DeepSeek + 真实嵌入
if SMOKE_FIRST and not (RUN_DIR / "smoke" / "SMOKE_REPORT.json").is_file():
    s = C.smoke(RUN_DIR, SCRATCH, fixture=PKG / "smoke" / "synthetic_lifebench.json", key=KEY, embedder=EMBEDDER, emb_dir=EMB_DIR, official=OFFICIAL, count=COUNT)
    print(json.dumps({k: s[k] for k in ("status", "mem0_memories", "proxy", "dense")}, ensure_ascii=False, indent=1))
    assert s["status"] == "PASS", "smoke FAILED: see smoke/SMOKE_REPORT.json before spending on the formal run"'''),
             cell('''# 09 · StillHere V2D 稠密索引（整条时间线建一次；检索时按提问时间屏蔽之后的行）
from lbpilot import stillhere_memory as SM
C.dense_index(run, EMBEDDER, scratch=SCRATCH)
DENSE, why = SM.open_dense(SCRATCH / "dense", LE.LocalQueryEncoder(EMBEDDER)); assert DENSE is not None, why'''),
             cell('''# 10 · 恢复断点（如有）→ 打开 StillHere 运行库与官方 Mem0 OSS
from lbpilot import llm_proxy as LP, mem0_memory as MM
print("live state at:", run.restore_if_needed())
PROXY = LP.CountingProxy(expected_model=P.MEM0["llm"]["model"], log_path=RUN_DIR / "logs" / "mem0_llm_proxy.jsonl").start()
LIVE = run.live()
STILL = SM.StillHereMemory(LIVE / "stillhere" / "run.sqlite", inp["slug"], inp["name"], inp["assistant"], dataset_sha256=P.DATASET["sha256"], locator=P.DATASET["url"])
EMB_CFG = {"provider": "huggingface", "config": {"model": str(EMB_DIR), "embedding_dims": 1024, "model_kwargs": {"device": "cuda"}}}
MEM0 = MM.Mem0Memory(LIVE / "mem0", user_id=f"lb2.{inp['slug']}", proxy_url=PROXY.url, api_key=KEY, collection=f"lb_{inp['slug']}_{RUN_ID}", embedder=EMB_CFG)
AUDIT = {**MEM0.runtime_audit(), "config": MM.public_config(MEM0.cfg)}
(RUN_DIR / "MEM0_RUNTIME_AUDIT.json").write_text(json.dumps(AUDIT, ensure_ascii=False, indent=1))
assert AUDIT["llm_class"] == "DeepSeekLLM" and AUDIT["embedder_class"] == "HuggingFaceEmbedding" and not AUDIT["openai_api_key_set"]
run.stage("10_mem0_init", "PASS", **{k: AUDIT[k] for k in ("mem0ai_version", "llm_model", "embedder_model", "vector_store_class")})'''),
             cell('''# 11–16 · 按日期顺序：当天事件 ADD 进 StillHere 和 Mem0 → 当天提问的题在四组检索并冻结上下文（最耗时：约 1–2 小时）
st = run.chronological(stillhere=STILL, mem0=MEM0, dense=DENSE, count=COUNT)
run.state["mem0_proxy"] = PROXY.summary(); run.state["mem0_stats"] = MEM0.stats; run.state["mem0_memories_final"] = MEM0.count(); run.save()
print(st["status"], st.get("contexts"), "contexts;", PROXY.summary())'''),
             cell('''# 13–16 · 统一回答（DeepSeek Flash，四组同一提示词；已完成的不会再请求）
print(run.run_answers(ANS.chat))'''),
             cell('''# 17 · 评分器校准（合成锚点，记录，不作为门槛）
if "17_judge_calibration" not in run.state["stages"]:
    print(run.calibrate(OFFICIAL, JUD.chat))'''),
             cell('''# 18 · 官方评分提示词 × 3 次（每次独立原始记录；已完成的不会再请求）
print(run.run_judge(OFFICIAL, JUD.chat))'''),
             cell('''# 19 · 指标与配对比较（只用原始记录计算）
m = run.compute_metrics(OFFICIAL, extra={"mem0_proxy": PROXY.summary(), "mem0_stats": MEM0.stats})
print(C.report(RUN_DIR))'''),
             cell(f'''# 20 · 导出 Results ZIP（Drive：exports/）
from lbpilot import export as EX
out = EX.build(RUN_DIR, BASE / "exports" / f"StillHere_LifeBench_100QA_Results_{{RUN_ID}}.zip")
print(json.dumps(out, indent=1))
print("下载这个 ZIP，交给下一轮审计。状态：", json.dumps({{k: v["status"] for k, v in run.state["stages"].items()}}, indent=1))'''),
             ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4", "provenance": []},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def guide(m: dict) -> str:
    return f"""# Colab 运行指南：LifeBench 2.0 · 1 人物 100 题 Pilot

版本 `{P.PILOT_VERSION}`，协议 sha `{P.PROTOCOL_SHA256[:16]}…`，代码 commit `{m['git_commit']}`。

## 1. 上传（只需一次）

| 本机文件（Windows） | Google Drive 位置 |
|---|---|
| `{REL / ZIP_NAME}` | `{DRIVE}/input/{ZIP_NAME}` |
| `{REL / NB_NAME}` | `{DRIVE}/{NB_NAME}`（或直接在 Colab 里“上传笔记本”） |
| （可选）官方数据 JSON | `{DRIVE}/input/lifebench_locomo_conversation_format_v2.0_3380QA.json`；不放的话 Notebook 会从 Hugging Face 固定 revision 下载并校验 sha256 |

ZIP 的 SHA256：`{m['zip_sha256']}`（Notebook 第 02 步会核对）。

## 2. 运行

1. 在 Drive 里双击 Notebook，用 Google Colab 打开。
2. 菜单“代码执行程序 → 更改运行时类型”：选 **GPU → L4**。
3. 左侧钥匙图标（Secrets）：新建 `DEEPSEEK_API_KEY`，填你的 DeepSeek Key，并打开“笔记本访问权限”。（也可用 `/content/drive/MyDrive/StillHere/_private/secrets/lifebench.env` 里的 `DEEPSEEK_API_KEY=` 一行。）
4. “代码执行程序 → 全部运行”（Run All）。第一次会请求挂载 Drive，按提示允许。
5. 先跑合成数据冒烟（几分钱）；冒烟失败会停下，不会花正式预算。

预计：Mem0 ADD 约 {m['plan']['mem0_add_batches']} 次 LLM 抽取（全年 {m['plan']['events']} 条事件，每批 10 条）；回答 {m['plan']['answers']} 次；评分 {m['plan']['judge_calls']} 次。
费用上限估计（DeepSeek 高峰价）：约 ${m['plan']['cost_upper_usd']}。时间约 1.5–3 小时（主要是 Mem0 ADD）。

## 3. 断线 / 中断

重新连接后**再点一次 Run All**，不要改 `RUN_ID`：
- 已冻结的检索上下文、回答、评分都在 `{DRIVE}/runs/<RUN_ID>/`，不会重复请求；
- Mem0 / StillHere 的数据库在 `/content`（本地盘），断线会丢；Notebook 从 `runs/<RUN_ID>/snapshots/` 里最新的快照恢复，再从快照日期继续 ADD；
- 稠密索引缓存在 `runs/<RUN_ID>/cache/dense.tar`。

## 4. 结果

- Results ZIP：`{DRIVE}/exports/StillHere_LifeBench_100QA_Results_<RUN_ID>.zip`
- 下载到 Windows：`<data-dir>\\downloads\\lifebench_pilot_100qa\\`（新建这个文件夹即可）
- 下一轮：把 ZIP 的路径告诉 Claude，用 `evaluation/lifebench_pilot` 的审计（`lbpilot.runner.compute` 可以只用原始记录重算全部指标）。

不完整的组（回答不足 100 / 评分不足 300 / 检索出错）会标 `INCOMPLETE`，它参与的配对比较 `decidable = false`，不能下结论。
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--commit", default="HEAD")
    a = ap.parse_args()
    commit = git("rev-parse", a.commit)
    if git("status", "--porcelain", "--", *PATHS):
        raise SystemExit("REFUSED: uncommitted changes in the packaged paths; commit first")
    frozen = SEL.freeze(a.dataset)
    data = SEL.load_dataset(a.dataset)
    person, qs = SEL.selected(data, frozen)
    events = TL.build(person, until=frozen["ask_time_range"][1])
    tsha = TL.sha(events)
    days = TL.by_date(events)
    batches = sum((len(v) + 9) // 10 for v in days.values())
    # upper estimate with peak prices and no cache hits: Mem0 ~9k in / 400 out per add (its extraction prompt alone was 7.9k tokens in the
    # 2026-11-19 live check); answer ~4.5k in / 200 out; judge ~2.5k in / 400 out
    cost = (batches * (9000 * 0.30 + 400 * 1.2) + 400 * (4500 * 0.30 + 200 * 1.2) + 1200 * (2500 * 0.30 + 400 * 1.2)) / 1e6
    plan = {"events": len(events), "mem0_add_batches": batches, "answers": 400, "judge_calls": 1200, "cost_upper_usd": round(cost * 1.5, 2)}
    entries = code_entries(commit)
    files = {"FROZEN_100QA.json": json.dumps(frozen, ensure_ascii=False, indent=1).encode("utf-8"),
             "PROTOCOL.json": json.dumps(P.protocol(), ensure_ascii=False, indent=1).encode("utf-8"),
             "requirements-pilot.txt": REQUIREMENTS.encode(), "smoke/synthetic_lifebench.json": smoke_fixture()}
    manifest = {"schema": "stillhere.lifebench_pilot.benchmark_manifest/1.0", "pilot_version": P.PILOT_VERSION, "scope": P.SCOPE,
                "protocol_sha256": P.PROTOCOL_SHA256, "git_commit": commit, "code_tree_sha256": sha(b"".join(sha(b).encode() + n.encode() for n, b in entries)),
                "dataset": P.DATASET, "official_eval": P.OFFICIAL_EVAL, "mem0": {"package": "mem0ai", "version": P.MEM0["version"]},
                "frozen": {"person": frozen["person"]["sample_id"], "slug": frozen["person"]["slug"], "questions": len(frozen["question_ids"]),
                           "qa_sha256": frozen["qa_sha256"], "type_counts": frozen["type_counts_multilabel"], "ask_time_range": frozen["ask_time_range"]},
                "timeline_sha256": tsha, "timeline": TL.stats(events), "plan": plan, "requirements": REQUIREMENTS.split(),
                "privacy": "public code + public synthetic benchmark ids only; no private chat, persona card, adapter, key or gold of the persona project"}
    files["BENCHMARK_MANIFEST.json"] = json.dumps(manifest, ensure_ascii=False, indent=1).encode("utf-8")
    all_entries = entries + sorted(files.items())
    sums = "".join(f"{sha(b)}  {n}\n" for n, b in all_entries)
    zb = det_zip(all_entries + [("SHA256SUMS.txt", sums.encode())])
    REL.mkdir(parents=True, exist_ok=True)
    (REL / ZIP_NAME).write_bytes(zb)
    manifest["zip_sha256"] = sha(zb)
    nb = json.dumps(notebook(sha(zb), manifest), ensure_ascii=False, indent=1).encode("utf-8")
    (REL / NB_NAME).write_bytes(nb)
    (REL / "BENCHMARK_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    (REL / "COLAB_100QA_RUN_GUIDE_CN.md").write_text(guide(manifest), encoding="utf-8")
    outer = "".join(f"{sha((REL / n).read_bytes())}  {n}\n" for n in (ZIP_NAME, NB_NAME, "BENCHMARK_MANIFEST.json", "COLAB_100QA_RUN_GUIDE_CN.md"))
    (REL / "SHA256SUMS.txt").write_text(outer, encoding="utf-8")
    print(json.dumps({"release": str(REL), "zip_sha256": manifest["zip_sha256"], "zip_bytes": len(zb), "commit": commit, "plan": plan,
                      "timeline_sha256": tsha, "qa_sha256": frozen["qa_sha256"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
