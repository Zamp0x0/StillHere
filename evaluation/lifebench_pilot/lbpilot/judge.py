"""Judge x 3 with the OFFICIAL LifeBench_eval prompt and aggregation, one raw record per call.

The prompt strings, the score-point renderer, the weighted score and the parser are imported from the official checkout (pinned
commit; ``verify_official`` checks the prompt sha256 against ``OFFICIAL_PROMPT_SHA256``). The call itself goes through our client so that
every raw reply, its parse status and its usage are stored, and a finished (question, group, repeat) is never requested again.
Aggregation = official: CORRECT iff correct_votes * 2 >= runs; each score point hit by the same majority.
"""
from __future__ import annotations

import hashlib
import importlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence

from . import protocol as P
from .records import Store

OFFICIAL_PROMPT_SHA256 = "15240b2e8f2af1f56134f9700fccd251599f3b136dee28e909d40cd31b3930bb"   # Official.prompt_sha256 at LifeBench_eval@2bfda547


class Official:
    def __init__(self, eval_dir: Path):
        eval_dir = Path(eval_dir)
        if str(eval_dir) not in sys.path:
            sys.path.insert(0, str(eval_dir))
        self.mod = importlib.import_module("src.evaluators.llm_judge")
        self.system = self.mod.SYSTEM_PROMPT
        self.template = self.mod.USER_PROMPT_TEMPLATE

    @property
    def prompt_sha256(self) -> str:
        return hashlib.sha256((self.system + "\n\x00\n" + self.template).encode("utf-8")).hexdigest()

    def messages(self, qa: Mapping, generated: str):
        types = self.mod._normalize_types(qa.get("question_type"))
        user = self.template.format(question=qa["question"], ask_time=qa.get("ask_time") or "（未提供）",
                                    question_types=", ".join(types) if types else "（未标注）", reference_answer=qa["answer"],
                                    score_points=self.mod._render_score_points(qa.get("score_points") or []), generated_answer=generated)
        return [{"role": "system", "content": self.system}, {"role": "user", "content": user}]

    def parse(self, content: str):
        obj = self.mod.LLMJudge._parse_robust(None, content or "")
        if not isinstance(obj, dict):
            return None
        label = str(obj.get("label", "")).strip().upper()
        if label not in ("CORRECT", "WRONG"):
            return None
        hits = obj.get("point_hits", [])
        return {"is_correct": label == "CORRECT", "point_hits": hits if isinstance(hits, list) else [], "reasoning": str(obj.get("reasoning", ""))}

    def weighted(self, score_points, hits):
        return self.mod._weighted_score(score_points or [], hits)


def run(jobs: Sequence[Mapping], *, official: Official, chat: Callable, store: Store, log=print, concurrency: int = P.JUDGE["concurrency"],
        max_parse_attempts: int = 3, max_rounds: int = 2) -> dict:
    """``jobs``: [{group, question_id, repeat, qa, answer}]. One call per job; an unparseable reply is retried (bounded) within the job."""
    def one(j):
        base = {"group": j["group"], "question_id": j["question_id"], "repeat": j["repeat"], "judge_prompt_sha256": official.prompt_sha256}
        raw, last = None, None
        for attempt in range(1, max_parse_attempts + 1):
            t0 = time.perf_counter()
            try:
                r = chat(official.messages(j["qa"], j["answer"]))
            except Exception as e:  # noqa: BLE001
                last = {**base, "status": "ERROR", "error": f"{type(e).__name__}: {str(e)[:200]}", "parse_attempts": attempt}
                continue
            raw = r.text
            v = official.parse(raw)
            if v is not None:
                return {**base, "status": "OK", "parse_attempts": attempt, "raw": raw, **v, **r.usage(), "wall_ms": round((time.perf_counter() - t0) * 1000, 1)}
            last = {**base, "status": "PARSE_FAIL", "parse_attempts": attempt, "raw": raw, **r.usage()}
        return last
    for rnd in range(max_rounds):
        done = store.done()
        todo = [j for j in jobs if (j["group"], j["question_id"], j["repeat"]) not in done]
        if not todo:
            break
        log(f"judge round {rnd + 1}: {len(todo)} requests ({len(done)} done)")
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            for f in as_completed([ex.submit(one, j) for j in todo]):
                store.append(f.result())
    done = store.done()
    return {"requested": len(jobs), "done": sum(1 for j in jobs if (j["group"], j["question_id"], j["repeat"]) in done)}


def aggregate(records: Sequence[Mapping], qa: Mapping, official: Official, runs: int = P.JUDGE["repeats"]) -> dict:
    """Official majority over the ``runs`` repeat records of one (question, group)."""
    rs = sorted(records, key=lambda r: r["repeat"])
    votes = sum(int(bool(r.get("is_correct"))) for r in rs)
    sp = qa.get("score_points") or []
    hits: List[dict] = []
    for i, p in enumerate(sp):
        n = 0
        for r in rs:
            h = (r.get("point_hits") or [])
            if i < len(h):
                x = h[i]
                n += int(x.get("hit", False) if isinstance(x, dict) else bool(x))
        hits.append({"description": (p or {}).get("description", ""), "hit": n * 2 >= runs})
    return {"is_correct": votes * 2 >= runs, "correct_votes": votes, "runs": len(rs), "point_hits": hits, "weighted_score": official.weighted(sp, hits),
            "unanimous": votes in (0, len(rs))}


def calibration_items() -> List[dict]:
    """Synthetic anchors (invented, not from the dataset): the judge must accept a correct answer, reject a wrong one, accept a proper
    abstention on an unanswerable question and reject an invented answer to it."""
    q = {"question": "（提问时间：2025-05-01）我上个月去哪个城市出差了？", "ask_time": "2025-05-01", "question_type": ["Single_hop"],
         "answer": "杭州。", "score_points": [{"description": "指出城市是杭州", "score": 10}]}
    ua = {"question": "（提问时间：2025-05-01）我去年在哪家医院做的手术？", "ask_time": "2025-05-01", "question_type": ["Unanswerable"],
          "answer": "记录中没有相关信息，无法回答。", "score_points": [{"description": "指出记录中没有相关信息", "score": 10}]}
    return [{"id": "CAL_correct", "qa": q, "answer": "你上个月去杭州出差了。", "expect": True},
            {"id": "CAL_wrong", "qa": q, "answer": "你上个月去了成都出差。", "expect": False},
            {"id": "CAL_abstain", "qa": ua, "answer": "根据现有记录无法确定。", "expect": True},
            {"id": "CAL_invented", "qa": ua, "answer": "你去年在市第一人民医院做的手术。", "expect": False}]
