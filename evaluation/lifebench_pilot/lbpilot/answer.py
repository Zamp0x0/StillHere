"""The ONE answer function every group uses: same model, prompt, generation settings and parsing; only the evidence context differs.

The prompt is generic (no StillHere- or Mem0-specific instruction): answer from the memory records as of the ask time; say it cannot be
determined when the records have no basis; reasonable inference from the records is allowed (pattern / causal / hidden-info questions
need it). The official question text (with its ask-time prefix) is shown unchanged.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Mapping, Sequence

from . import protocol as P
from .records import Store

PROMPT_VERSION = P.ANSWER["prompt_version"]
SYSTEM = ("你是一个基于个人记忆记录回答问题的助手。记录来自用户{name}的手机数据（短信、通话、日程、笔记、照片、推送、健康数据、与 AI 助手的对话等），"
          "或由记忆系统从这些数据整理出的记忆。请只根据给出的记忆记录回答问题，以提问时间为准理解“现在”“上个月”等相对时间；"
          "可以根据记录做合理的归纳和推断。如果记录中没有足以回答的依据，就直接回答“根据现有记录无法确定”，不要编造。"
          "回答用中文，简洁直接，先给出结论。")
USER = "提问时间：{ask_time}\n\n{context}\n\n问题：{question}"
ABSTAIN = ("无法确定", "没有相关记录", "没有找到", "没有提到", "未提及", "无法回答", "不清楚", "没有记录", "无从得知", "无法得知")


def messages(name: str, question: str, ask_time: str, context: str):
    return [{"role": "system", "content": SYSTEM.format(name=name)},
            {"role": "user", "content": USER.format(ask_time=ask_time[:10], context=context, question=question)}]


def is_abstention(text: str) -> bool:
    return any(x in (text or "") for x in ABSTAIN)


def run(items: Sequence[Mapping], *, chat: Callable, store: Store, name: str, log=print, concurrency: int = P.ANSWER["concurrency"],
        max_rounds: int = 2) -> dict:
    """``items``: [{group, question_id, question, ask_time, context}]; ``chat(messages) -> ChatResponse``. Finished keys are skipped."""
    def one(it):
        t0 = time.perf_counter()
        base = {"group": it["group"], "question_id": it["question_id"], "prompt_version": PROMPT_VERSION, "context_sha256": it.get("context_sha256")}
        try:
            r = chat(messages(name, it["question"], it["ask_time"], it["context"]))
            return {**base, "status": "OK" if (r.text or "").strip() else "EMPTY", "answer": (r.text or "").strip(), "abstained": is_abstention(r.text),
                    **r.usage(), "wall_ms": round((time.perf_counter() - t0) * 1000, 1)}
        except Exception as e:  # noqa: BLE001 - recorded; retried in the next round / a resumed run
            return {**base, "status": "ERROR", "error": f"{type(e).__name__}: {str(e)[:200]}", "wall_ms": round((time.perf_counter() - t0) * 1000, 1)}
    for rnd in range(max_rounds):
        done = store.done()
        todo = [it for it in items if (it["group"], it["question_id"]) not in done]
        if not todo:
            break
        log(f"answer round {rnd + 1}: {len(todo)} requests ({len(done)} done)")
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            for f in as_completed([ex.submit(one, it) for it in todo]):
                store.append(f.result())
    done = store.done()
    return {"requested": len(items), "done": sum(1 for it in items if (it["group"], it["question_id"]) in done)}
