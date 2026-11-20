"""Answer pilot, generation side: retrieved evidence becomes one fixed context, and one fixed model answers from it.

This module never sees an official answer, evidence list or question type. ``Question`` carries the question text and time here,
and the scoring side lives in ``answer_judge``, which this module does not import. Every arm is rendered, truncated and answered
the same way. The arm's name reaches neither the prompt nor the cache of a judge.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Set, Tuple

from stillhere.contracts import EvidenceHit, HitKind

EXPERIMENT_VERSION = "answer-pilot-1"
ARMS = {"A0": "legacy", "A1": "fast", "A2": "fast_multihop", "A3": "quality"}
TOP_K = 10
MAX_CONTEXT_TOKENS = 4000                  # the same ceiling for every arm, not the same actual length
MAX_NEW_TOKENS = 400
ORDER_SEED = 20261107
ABSTAIN_TEXT = "根据现有记忆，无法确定。"

SYSTEM_PROMPT = """你是用户的个人记忆助手。用户会就自己的过去提问，你只能依据下面提供的记忆证据回答。

规则：
1. 优先使用提供的证据，不使用证据之外的私人事实。
2. 不编造个人经历，也不根据常识猜测用户的历史经历。
3. 留意时间：只回答到提问时间为止的情况，不把已过期的事实当成当前事实。
4. 不把计划或打算当成已经发生的事。
5. 不能只因为两件事先后发生，就说它们有因果关系。
6. 证据不足以回答时，明确表示无法确定，不要勉强作答。
7. 回答简洁、准确、完整，直接回答问题。

只输出一个 JSON 对象，不要输出其他内容，格式：
{"answer": "自然语言答案", "abstain": false, "citations": ["E01", "E02"]}
citations 只能填写证据中出现过的编号。证据不足时输出：
{"answer": "根据现有记忆，无法确定。", "abstain": true, "citations": []}"""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    """Deterministic and independent of any model, so the ceiling does not depend on which provider runs: one per character, a safe upper bound for Chinese."""
    return len(text)


@dataclass(frozen=True)
class Evidence:
    """One record of a retrieved list, in the fields that exist in the data. Anything unknown stays None."""

    key: str
    record_id: str
    persona_id: str
    source_id: str
    source_type: str
    text: str
    score: float
    locator: Optional[str]
    event_time: Optional[str]
    known_at: Optional[str]
    speaker: Optional[str] = None
    retrieval_origin: Optional[str] = None
    seed_record_key: Optional[str] = None
    relation_type: Optional[str] = None
    path_provenance: Optional[str] = None


def to_evidence(hit: EvidenceHit, key: str, speakers: Mapping[str, str]) -> Evidence:
    record = hit.memory_id or hit.message_id
    when = hit.timestamp if hit.kind is HitKind.MESSAGE else hit.valid_from
    speaker = None
    if hit.speaker_id:
        speaker = "本人" if hit.speaker_id == hit.persona_id else speakers.get(hit.speaker_id, hit.speaker_id)
    return Evidence(key, record, hit.persona_id, hit.source_id, hit.source_id.rsplit(".", 1)[-1], hit.text, hit.score, hit.locator, when.isoformat() if when else None,
                    hit.recorded_at.isoformat() if hit.recorded_at else None, speaker, hit.retrieval_origin, hit.seed_record_key, hit.relation_type, hit.path_provenance)


def render_item(evidence_id: str, e: Evidence) -> str:
    head = [f"[{evidence_id}]", f"来源：{e.source_type}"]
    if e.event_time:
        head.append(f"时间：{e.event_time[:16].replace('T', ' ')}")
    if e.speaker:
        head.append(f"发言人：{e.speaker}")
    if e.locator:
        head.append(f"定位：{e.locator}")
    return " | ".join(head) + f"\n内容：{e.text}"


@dataclass(frozen=True)
class Context:
    text: str
    evidence_ids: Tuple[str, ...]           # the record keys rendered, in order; E01 is the first
    id_map: Dict[str, str]
    dropped_keys: Tuple[str, ...]
    estimated_tokens: int
    truncated: bool
    sha256: str


def build_context(items: Sequence[Evidence], *, max_tokens: int = MAX_CONTEXT_TOKENS, top_k: int = TOP_K) -> Context:
    """Rank order is kept. A record that would pass the ceiling is left out whole, with the rest after it: no text is cut mid-record."""
    seen: Set[str] = set()
    unique = [e for e in items if not (e.key in seen or seen.add(e.key))][:top_k]
    parts: List[str] = []
    kept: List[str] = []
    used = 0
    dropped: List[str] = []
    for e in unique:
        block = render_item(f"E{len(parts) + 1:02d}", e)
        if dropped or used + estimate_tokens(block) > max_tokens:
            dropped.append(e.key)
            continue
        parts.append(block)
        kept.append(e.key)
        used += estimate_tokens(block)
    text = "\n\n".join(parts) if parts else "（没有找到相关记忆）"
    return Context(text, tuple(kept), {f"E{i:02d}": k for i, k in enumerate(kept, 1)}, tuple(dropped), used, bool(dropped), sha256_text(text))


def build_messages(question: str, ask_time: str, context_text: str) -> List[Dict[str, str]]:
    user = f"提问时间：{ask_time}\n\n记忆证据：\n{context_text}\n\n问题：{question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def prompt_sha256() -> str:
    return sha256_text(SYSTEM_PROMPT + build_messages("{q}", "{t}", "{c}")[1]["content"])


@dataclass(frozen=True)
class Parsed:
    answer: str
    abstain: bool
    citations: Tuple[str, ...]
    valid_json: bool


_JSON = re.compile(r"\{.*\}", re.DOTALL)
_ABSTAINS = ("无法确定", "无法判断", "没有相关记忆", "未提及")


def parse_answer(raw: str) -> Parsed:
    """The model's JSON, or its plain text when there is none: an unparsable reply stays an answer and is counted as such."""
    found = _JSON.search(raw or "")
    if found:
        try:
            data = json.loads(found.group(0))
        except ValueError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("answer"), str):
            citations = data.get("citations")
            cited = tuple(str(c) for c in citations) if isinstance(citations, list) else ()
            return Parsed(data["answer"].strip(), data.get("abstain") is True, cited, True)
    text = (raw or "").strip()
    return Parsed(text, any(w in text for w in _ABSTAINS) and len(text) < 40, (), False)


def citation_check(citations: Sequence[str], context: Mapping[str, str]) -> dict:
    """Whether each cited ID was shown to the model. Nothing here says the citation supports the claim."""
    invalid = [c for c in citations if c not in context]
    return {"cited": len(citations), "invalid": invalid, "valid": len(citations) - len(invalid)}


class AnswerProvider(Protocol):
    model_id: str
    config: Mapping

    def generate(self, messages: Sequence[Mapping[str, str]], max_new_tokens: int) -> dict:
        """Returns text, input_tokens, output_tokens, latency_ms and cost_cny."""


def cache_key(*, model_id: str, model_config: Mapping, prompt_hash: str, evidence_hash: str, qa_id: str, arm_id: str, experiment_version: str = EXPERIMENT_VERSION) -> str:
    payload = {"model_id": model_id, "model_config": model_config, "prompt_hash": prompt_hash, "evidence_hash": evidence_hash, "qa_id": qa_id, "arm_id": arm_id,
               "experiment_version": experiment_version}
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


class ResultFile:
    """Append-only JSONL keyed by ``cache_key``. A stored row is never paid for or produced twice."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.rows: Dict[str, dict] = {}
        self._lock = threading.Lock()
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.rows[row["cache_key"]] = row

    def get(self, key: str) -> Optional[dict]:
        return self.rows.get(key)

    def add(self, row: Mapping) -> None:
        with self._lock:
            self.rows[row["cache_key"]] = dict(row)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def arm_order(qa_id: str, seed: int = ORDER_SEED) -> List[str]:
    """A fixed shuffle per question, so that no arm always runs first."""
    return sorted(ARMS, key=lambda a: sha256_text(f"{seed}:{qa_id}:{a}"))


def generate_one(provider: AnswerProvider, results: ResultFile, *, qa_id: str, persona_id: str, question: str, ask_time: str, arm_id: str, context: Context,
                 run_id: str) -> dict:
    prompt_hash = prompt_sha256()
    key = cache_key(model_id=provider.model_id, model_config=provider.config, prompt_hash=prompt_hash, evidence_hash=context.sha256, qa_id=qa_id, arm_id=arm_id)
    found = results.get(key)
    if found:
        return found
    reply = provider.generate(build_messages(question, ask_time, context.text), MAX_NEW_TOKENS)
    parsed = parse_answer(reply["text"])
    row = {"cache_key": key, "experiment_version": EXPERIMENT_VERSION, "run_id": run_id, "qa_id": qa_id, "persona_id": persona_id, "question": question, "ask_time": ask_time, "arm_id": arm_id,
           "evidence_ids": list(context.evidence_ids), "evidence_map": context.id_map, "evidence_hash": context.sha256, "prompt_hash": prompt_hash, "answer_model": provider.model_id,
           "decoding_config": dict(provider.config), "raw_model_output": reply["text"], "parsed_answer": parsed.answer, "abstain": parsed.abstain, "citations": list(parsed.citations),
           "valid_json": parsed.valid_json, "citation_check": citation_check(parsed.citations, context.id_map), "input_tokens": reply["input_tokens"], "output_tokens": reply["output_tokens"],
           "latency_ms": reply["latency_ms"], "cost_cny": reply.get("cost_cny", 0.0)}
    results.add(row)
    return row
