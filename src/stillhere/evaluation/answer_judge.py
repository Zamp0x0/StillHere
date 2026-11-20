"""Answer pilot, scoring side: the official judge protocol, independent checks, and paired statistics.

Official answers are read only here, after generation, and flow only to the judge. A judge prompt holds the question, the reference
and one answer. Which arm wrote the answer is not in the prompt, the cache key or the order of the calls.

The official prompts are not copied here. They are read from a pinned copy of ``LifeBench_eval``'s ``llm_judge.py``, without running it.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .answer_pilot import ResultFile, sha256_text

NUM_RUNS = 3
JUDGE_MAX_TOKENS = 800
CHECK_MAX_TOKENS = 300

Complete = Callable[[Sequence[Mapping[str, str]], int], dict]     # (messages, max tokens) -> text, input_tokens, output_tokens, latency_ms


@dataclass(frozen=True)
class Official:
    """What the judge may read of one question. Generation never receives this."""

    qa_id: str
    answer: str
    score_points: Tuple[Mapping, ...]
    question_types: Tuple[str, ...]


def load_official(data_root: Path, directory: str) -> Dict[str, Official]:
    rows = json.loads((Path(data_root) / directory / "QA_all" / "QA.json").read_text(encoding="utf-8"))
    return {f"{directory}.qa.{i:04d}": Official(f"{directory}.qa.{i:04d}", str(r.get("answer") or ""), tuple(r.get("score_points") or ()), tuple(r["question_type"]))
            for i, r in enumerate(rows)}


@dataclass(frozen=True)
class JudgePrompts:
    system: str
    user_template: str
    source_sha256: str


def read_prompts(path: Path) -> JudgePrompts:
    """The two string constants of the official file, by parsing it. The file is data here and is never imported or run."""
    raw = Path(path).read_bytes()
    found: Dict[str, str] = {}
    for node in ast.parse(raw.decode("utf-8")).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("SYSTEM_PROMPT", "USER_PROMPT_TEMPLATE"):
            found[node.targets[0].id] = ast.literal_eval(node.value)
    return JudgePrompts(found["SYSTEM_PROMPT"], found["USER_PROMPT_TEMPLATE"], hashlib.sha256(raw).hexdigest())


def render_score_points(points: Sequence[Mapping]) -> str:
    """As the official judge renders them: numbered, weights hidden."""
    if not points:
        return "（本题无评分点清单，只判整体 label）"
    return "\n".join(f"    {i}. {(p or {}).get('description', '')}" for i, p in enumerate(points, 1))


def judge_messages(p: JudgePrompts, *, question: str, ask_time: str, official: Official, answer: str) -> List[Dict[str, str]]:
    user = p.user_template.format(question=question, ask_time=ask_time or "（未提供）", question_types=", ".join(official.question_types) or "（未标注）",
                                  reference_answer=official.answer, score_points=render_score_points(official.score_points), generated_answer=answer)
    return [{"role": "system", "content": p.system}, {"role": "user", "content": user}]


def parse_verdict(content: str) -> Optional[dict]:
    """The official parser's order: fenced JSON, raw JSON, then the label by regex. None when no label can be read."""
    objects = []
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content or "", re.DOTALL)
    if m:
        objects.append(m.group(1))
    start, end = (content or "").find("{"), (content or "").rfind("}")
    if start != -1 and end > start:
        objects.append(content[start:end + 1])
    for text in objects:
        try:
            obj = json.loads(text)
        except ValueError:
            continue
        label = str(obj.get("label", "")).strip().upper()
        if label in ("CORRECT", "WRONG"):
            hits = obj.get("point_hits")
            return {"label": label, "point_hits": hits if isinstance(hits, list) else [], "reasoning": str(obj.get("reasoning", ""))}
    m = re.search(r'"?label"?\s*:\s*"?(CORRECT|WRONG)"?', content or "", re.IGNORECASE)
    if not m:
        return None
    hits = [v.lower() == "true" for v in re.findall(r'"?hit"?\s*:\s*(true|false)', content, re.IGNORECASE)]
    return {"label": m.group(1).upper(), "point_hits": [{"hit": h} for h in hits], "reasoning": "(regex-recovered)"}


def weighted_score(points: Sequence[Mapping], hits: Sequence) -> Optional[float]:
    total = sum(float((p or {}).get("score", 0) or 0) for p in points)
    if not points or total <= 0 or len(hits) != len(points):
        return None
    got = sum(float((p or {}).get("score", 0) or 0) for p, h in zip(points, hits) if (h.get("hit") if isinstance(h, dict) else bool(h)))
    return got / total


def majority(verdicts: Sequence[Optional[dict]], points: Sequence[Mapping]) -> dict:
    """Official rule: correct when at least half the votes say so. A failed vote is not a WRONG vote here: it is reported as missing."""
    usable = [v for v in verdicts if v]
    if len(usable) < len(verdicts):
        return {"label": "MISSING", "votes": {"correct": sum(v["label"] == "CORRECT" for v in usable), "wrong": sum(v["label"] == "WRONG" for v in usable)}, "failed_votes": len(verdicts) - len(usable), "weighted_score": None}
    correct = sum(v["label"] == "CORRECT" for v in usable)
    n = len(usable)
    merged = []
    for i, _ in enumerate(points):
        votes = 0
        for v in usable:
            hits = v["point_hits"]
            if i < len(hits):
                votes += int(hits[i].get("hit", False) if isinstance(hits[i], dict) else bool(hits[i]))
        merged.append({"hit": votes * 2 >= n})
    is_correct = correct * 2 >= n
    return {"label": "CORRECT" if is_correct else "WRONG", "votes": {"correct": correct, "wrong": n - correct}, "failed_votes": 0, "weighted_score": weighted_score(points, merged),
            "reasoning": next(v["reasoning"] for v in usable if (v["label"] == "CORRECT") == is_correct)}


def judge_answer(complete: Complete, results: ResultFile, prompts: JudgePrompts, *, model: str, question: str, ask_time: str, official: Official, answer: str, runs: int = NUM_RUNS, retries: int = 3) -> dict:
    """``runs`` votes. An identical input has the same key, so two arms that wrote the same words share one paid verdict."""
    messages = judge_messages(prompts, question=question, ask_time=ask_time, official=official, answer=answer)
    verdicts: List[Optional[dict]] = []
    latency = []
    for run in range(runs):
        key = sha256_text(json.dumps({"kind": "official", "model": model, "run": run, "messages": messages}, ensure_ascii=False, sort_keys=True))
        row = results.get(key)
        if row is None:
            reply, parsed = {}, None
            for _ in range(retries):
                reply = complete(messages, JUDGE_MAX_TOKENS)
                parsed = parse_verdict(reply["text"])
                if parsed:
                    break
            row = {"cache_key": key, "kind": "official", "model": model, "run": run, "qa_id": official.qa_id, "verdict": parsed, "raw": reply.get("text"), "input_tokens": reply.get("input_tokens"),
                   "output_tokens": reply.get("output_tokens"), "latency_ms": reply.get("latency_ms")}
            results.add(row)
        verdicts.append(row["verdict"])
        latency.append(row.get("latency_ms") or 0.0)
    return {**majority(verdicts, official.score_points), "judge_latency_ms": latency, "input_sha256": sha256_text(answer)}


# ---------------------------------------------------------------- checks of this project, not the official protocol

GROUNDING_SYSTEM = """你是证据核查员。只根据给出的证据判断回答中的陈述是否有依据，不评价回答是否正确。输出一个 JSON 对象，不要输出其他内容。"""
GROUNDING_USER = """提问时间：{ask_time}
问题：{question}

证据：
{context}

待核查的回答：{answer}

逐项判断并输出：
{{"support": "full 或 partial 或 none", "unsupported_personal_fact": true 或 false, "unsupported_causal": true 或 false, "temporal_misuse": true 或 false, "note": "一句话说明"}}
support：回答中的陈述是否都能在证据中找到依据。
unsupported_personal_fact：回答是否含有证据里没有的个人经历或事实。
unsupported_causal：回答是否给出证据没有表明的因果关系。
temporal_misuse：回答是否把计划当成已发生，把过期事实当成当前事实，或说错了证据里的时间先后。"""

SUFFICIENCY_SYSTEM = """你是评测分析员。只判断给出的证据是否足以推出参考答案，不要使用其他知识。输出一个 JSON 对象，不要输出其他内容。"""
SUFFICIENCY_USER = """问题：{question}

参考答案：{reference}

证据：
{context}

输出：{{"sufficient": "yes 或 partial 或 no", "note": "一句话说明"}}
yes：只读这些证据就能得出参考答案的核心内容。partial：只能得出一部分。no：得不出来。"""


def parse_json(text: str) -> Optional[dict]:
    found = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not found:
        return None
    try:
        data = json.loads(found.group(0))
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def check(complete: Complete, results: ResultFile, *, kind: str, model: str, system: str, user: str, qa_id: str, retries: int = 2) -> Optional[dict]:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    key = sha256_text(json.dumps({"kind": kind, "model": model, "messages": messages}, ensure_ascii=False, sort_keys=True))
    row = results.get(key)
    if row is None:
        reply, parsed = {}, None
        for _ in range(retries):
            reply = complete(messages, CHECK_MAX_TOKENS)
            parsed = parse_json(reply["text"])
            if parsed:
                break
        row = {"cache_key": key, "kind": kind, "model": model, "qa_id": qa_id, "verdict": parsed, "raw": reply.get("text"), "input_tokens": reply.get("input_tokens"), "output_tokens": reply.get("output_tokens"),
               "latency_ms": reply.get("latency_ms")}
        results.add(row)
    return row["verdict"]


# ---------------------------------------------------------------- statistics

def paired_rows(a: Mapping[str, Optional[bool]], b: Mapping[str, Optional[bool]]) -> List[Tuple[str, bool, bool]]:
    """Questions both arms finished. Everything below uses this shared denominator."""
    return [(q, a[q], b[q]) for q in sorted(a) if q in b and a[q] is not None and b[q] is not None]


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_compare(a: Mapping[str, Optional[bool]], b: Mapping[str, Optional[bool]], *, rounds: int = 5000, seed: int = 20261107) -> dict:
    """a minus b, per question."""
    rows = paired_rows(a, b)
    n = len(rows)
    if n == 0:
        return {"questions": 0}
    wins = sum(x and not y for _, x, y in rows)
    losses = sum(y and not x for _, x, y in rows)
    both_right = sum(x and y for _, x, y in rows)
    both_wrong = sum((not x) and (not y) for _, x, y in rows)
    diffs = [int(x) - int(y) for _, x, y in rows]
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(rounds))
    return {"questions": n, "difference": round(statistics.fmean(diffs), 4), "ci95": [round(means[int(0.025 * rounds)], 4), round(means[int(0.975 * rounds)], 4)], "wins": wins, "losses": losses,
            "both_correct": both_right, "both_wrong": both_wrong, "mcnemar_exact_p": round(exact_mcnemar(wins, losses), 4), "wrong_to_correct": [q for q, x, y in rows if x and not y],
            "correct_to_wrong": [q for q, x, y in rows if y and not x]}


def holm(pvalues: Mapping[str, float]) -> Dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda x: x[1])
    adjusted: Dict[str, float] = {}
    worst = 0.0
    for i, (name, p) in enumerate(ordered):
        worst = max(worst, min(1.0, p * (len(ordered) - i)))
        adjusted[name] = round(worst, 4)
    return adjusted


def percentile_stats(values: Sequence[float]) -> dict:
    v = sorted(x for x in values if x is not None)
    if not v:
        return {"samples": 0}
    at = lambda p: v[min(len(v) - 1, int(math.ceil(p * len(v))) - 1)]
    return {"samples": len(v), "p50_ms": round(at(0.5), 1), "p95_ms": round(at(0.95), 1), "mean_ms": round(statistics.fmean(v), 1), "max_ms": round(v[-1], 1)}


def point_biserial(xs: Sequence[float], ys: Sequence[bool]) -> Optional[float]:
    """Pearson correlation of a score with a binary outcome; None when either side does not vary."""
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return round(statistics.correlation(list(xs), [float(y) for y in ys]), 4)
