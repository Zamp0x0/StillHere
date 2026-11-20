"""Evidence-first answering: one generation that names what is needed, quotes what was found, and only then answers.

It takes the same frozen context as the baseline and returns the fields ``answer_pilot`` already records. Everything is general: nothing here knows a dataset, a
question id or an official answer. The structured fields before the answer are short on purpose; they are not a free chain of thought.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .answer_pilot import Context, ResultFile, build_messages, cache_key, citation_check, sha256_text

EXPERIMENT_VERSION = "evidence-first-1"
MAX_NEW_TOKENS = 400

SYSTEM_PROMPT = """你是用户的个人记忆助手。用户会就自己的过去提问，你只能依据下面提供的记忆证据回答。

请按顺序填写一个 JSON 对象，只输出这个对象：
{"needed": "问题要查的具体事实，一句话", "quotes": [{"id": "E01", "text": "证据原文片段"}], "answerable": true, "answer": "自然语言答案", "abstain": false, "citations": ["E01"]}

填写规则：
1. needed：写清问题到底要什么，比如一个名字、一个数字、一段原话、一个顺序或一个原因。
2. quotes：把相关内容从证据里逐字摘出来，最多 3 段，每段不超过 30 字。没有相关内容就写空数组。
3. answerable：quotes 里的内容能回答问题就写 true，否则写 false。证据的说法和问题不一样，但意思相同，也算能回答。
4. answer：能回答就直接回答，不要因为证据措辞不同而拒答。只知道一部分时，回答已知部分，并说明缺什么。
5. 不能回答时，answerable 写 false，abstain 写 true，answer 写“根据现有记忆，无法确定。”，citations 留空数组。
6. 问题要的具体细节（名字、品牌、金额、日期、原话）在证据里没有时，就是不能回答。不要用类别、口味、用途或相近的事实代替，也不要根据常识猜测。
7. 问题问某一天发生了什么时，只用那一天的证据，别的日子发生的事不能算进去。
8. 排序或比较时间时，先看每条证据的时间，再按时间先后排列。
9. 证据有冲突时，用提问时间之前最新的实际发生的事实。计划不等于已发生，过期事实不算当前事实。
10. 不能只因为两件事先后发生，就说它们有因果关系。
11. citations 只能填写证据中出现过的编号。"""


@dataclass(frozen=True)
class Reading:
    answer: str
    abstain: bool
    citations: Tuple[str, ...]
    needed: Optional[str]
    quotes: Tuple[Tuple[str, str], ...]
    answerable: Optional[bool]
    valid_json: bool


_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_ANSWER = re.compile(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"')


def prompt_sha256() -> str:
    return sha256_text(SYSTEM_PROMPT + build_messages("{q}", "{t}", "{c}")[1]["content"])


def messages_for(question: str, ask_time: str, context_text: str) -> List[Dict[str, str]]:
    """The same user message as the baseline; only the instructions differ."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, build_messages(question, ask_time, context_text)[1]]


def parse_reading(raw: str) -> Reading:
    """A reply that cut off or broke its JSON stays visible: the answer is taken if it can be read, and it is never counted as valid."""
    found = _OBJECT.search(raw or "")
    data = None
    if found:
        try:
            data = json.loads(found.group(0))
        except ValueError:
            data = None
    if isinstance(data, dict) and isinstance(data.get("answer"), str):
        quotes = tuple((str(q.get("id", "")), str(q.get("text", ""))) for q in data.get("quotes") or [] if isinstance(q, dict))
        answerable = data.get("answerable") if isinstance(data.get("answerable"), bool) else None
        citations = tuple(str(c) for c in data.get("citations") or []) if isinstance(data.get("citations"), list) else ()
        return Reading(data["answer"].strip(), data.get("abstain") is True or answerable is False, citations, str(data.get("needed") or "") or None, quotes, answerable, True)
    partial = _ANSWER.search(raw or "")
    if partial:
        text = partial.group(1)
        return Reading(text, "无法确定" in text, (), None, (), None, False)
    return Reading((raw or "").strip(), False, (), None, (), None, False)


def quote_check(quotes: Sequence[Tuple[str, str]], evidence_text: Mapping[str, str]) -> dict:
    """Whether each quote really stands in the evidence it names. A plain string test, so it needs no judge."""
    exact = sum(1 for eid, text in quotes if text and text in evidence_text.get(eid, ""))
    return {"quotes": len(quotes), "exact": exact, "unknown_id": sum(1 for eid, _ in quotes if eid not in evidence_text)}


def generate_candidate(provider, results: ResultFile, *, qa_id: str, persona_id: str, question: str, ask_time: str, context: Context, evidence_text: Mapping[str, str], run_id: str) -> dict:
    prompt_hash = prompt_sha256()
    key = cache_key(model_id=provider.model_id, model_config=provider.config, prompt_hash=prompt_hash, evidence_hash=context.sha256, qa_id=qa_id, arm_id="EVIDENCE_FIRST", experiment_version=EXPERIMENT_VERSION)
    found = results.get(key)
    if found:
        return found
    reply = provider.generate(messages_for(question, ask_time, context.text), MAX_NEW_TOKENS)
    r = parse_reading(reply["text"])
    row = {"cache_key": key, "experiment_version": EXPERIMENT_VERSION, "run_id": run_id, "qa_id": qa_id, "persona_id": persona_id, "question": question, "ask_time": ask_time, "arm_id": "EVIDENCE_FIRST", "evidence_ids": list(context.evidence_ids), "evidence_map": context.id_map, "evidence_hash": context.sha256, "prompt_hash": prompt_hash, "answer_model": provider.model_id, "decoding_config": dict(provider.config), "raw_model_output": reply["text"], "parsed_answer": r.answer, "abstain": r.abstain, "citations": list(r.citations),
           "needed": r.needed, "quotes": [list(q) for q in r.quotes], "answerable": r.answerable, "valid_json": r.valid_json, "citation_check": citation_check(r.citations, context.id_map), "quote_check": quote_check(r.quotes, evidence_text), "input_tokens": reply["input_tokens"], "output_tokens": reply["output_tokens"], "latency_ms": reply["latency_ms"], "peak_vram_mb": reply.get("peak_vram_mb"), "cost_cny": reply.get("cost_cny", 0.0)}
    results.add(row)
    return row


def change_tags(base_label: str, cand_label: str, *, base_abstain: bool, cand_abstain: bool, level: Optional[str], unanswerable: bool, valid_json: bool, grounding: Optional[Mapping]) -> List[str]:
    """Rule-based tags for a changed answer. They start a reading; they do not replace it."""
    g = grounding or {}
    tags: List[str] = []
    if not valid_json:
        tags.append("U9_FORMAT_OR_PARSING_FAILURE")
    if base_label != "CORRECT" and cand_label == "CORRECT":
        tags.append("U2_UNNECESSARY_ABSTENTION_RESOLVED" if base_abstain else "U1_CORRECT_EVIDENCE_UTILIZED")
    if base_label == "CORRECT" and cand_label != "CORRECT":
        tags.append("U4_HALLUCINATION_INTRODUCED" if unanswerable else "U5_CORRECT_ANSWER_LOST")
    if cand_label != "CORRECT" and not cand_abstain and level == "S1":
        tags.append("U3_PARTIAL_EVIDENCE_OVERCLAIMED")
    if cand_label != "CORRECT" and not cand_abstain and (g.get("unsupported_personal_fact") is True) and "U4_HALLUCINATION_INTRODUCED" not in tags:
        tags.append("U4_HALLUCINATION_INTRODUCED")
    if cand_label != "CORRECT" and g.get("temporal_misuse") is True:
        tags.append("U6_TEMPORAL_REASONING_ERROR")
    if cand_label != "CORRECT" and cand_abstain and level == "S2":
        tags.append("U8_EVIDENCE_PRESENT_BUT_IGNORED")
    return tags or ["U10_UNKNOWN"]
