"""V3.2 research: choosing the final ten from a pool that is already found. An optional experiment around the frozen EXP-B pool, which it never changes.

Two choosers, used alone or together:

  * question-aware rerank: the first ``k`` of the fused pool go to a chat model, which only scores the ids it was shown. Nothing it writes becomes evidence.
  * provenance: once a stored event's summary is chosen, a raw record of that same event is skipped. The link comes from the records themselves.

No QA field reaches this module. Any failure of the model falls back to the pool's own order, so the frozen retriever is never blocked.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .alibaba_client import ApiError, BudgetExceeded, OfflineCall
from .hybrid import Candidate, Retrieval
from .multifact import MultiFactConfig, MultiFactRetriever

RERANK_MODEL = "qwen3.7-flash-2026-07-15"
RERANK_MAX_CANDIDATES = 30
RERANK_MAX_TOKENS = 500
ITEM_CHARS = 300
SCORES = (0, 1, 2, 3)
RERANK_SYSTEM = """你为一个个人记忆问答系统挑选证据。下面是用户的问题和若干条候选记录。你只给记录打分，不回答问题，不改写记录，也不补充记录里没有的内容。

分数：
3：直接包含回答问题所需的事实。
2：包含回答所需的一部分，例如问题提到的人或事、关键时间、多件事中的一件、变化前或变化后的状态、互相矛盾的说法之一。
1：只是主题相关，对回答没有实际帮助。
0：无关。

留意时间：问题问的是某个时间点或时间段时，日期不符的记录最多 1 分。计划和实际发生是不同的事。

只输出 JSON，必须给每个编号打分：{"scores": {"C01": 3, "C02": 0}}"""


KEEP_MAX = 12
RERANK_SYSTEM_COMPACT = """你为一个个人记忆问答系统挑选证据。下面是用户的问题和若干条候选记录。你只挑选记录，不回答问题，不改写记录，也不补充记录里没有的内容。

挑出对回答有实际帮助的记录：直接包含所需事实的，或包含所需的一部分，例如问题提到的人或事、关键时间、多件事中的一件、变化前或变化后的状态、互相矛盾的说法之一。只是主题相关的不要选。

留意时间：问题问的是某个时间点或时间段时，日期不符的记录不要选。计划和实际发生是不同的事。

按有用程度从高到低排列，最多 12 个编号，没有有用的记录就输出空列表。只输出 JSON：{"keep": ["C03", "C01"]}"""


RERANK_SYSTEM_LIST = RERANK_SYSTEM.rsplit("\n\n", 1)[0] + """

按编号顺序给每条记录一个分数，只输出 JSON 列表，长度等于记录条数。例如有三条记录：{"s": [3, 0, 2]}"""


@dataclass(frozen=True)
class SelectionConfig:
    name: str
    pool: MultiFactConfig                 # the frozen pool this chooser works on
    rerank_k: int = 0                     # 0 is off; at most RERANK_MAX_CANDIDATES
    provenance: bool = False
    final_k: int = 10
    compact: bool = False                 # revision 1: the model lists the useful ids, best first, instead of scoring every one
    score_list: bool = False              # revision 2: every id scored as in V32-1, written as a bare list in id order

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode("utf-8")).hexdigest()[:16]


def rerank_messages(question: str, ask_time: str, items: Sequence[Tuple[str, str]], *, compact: bool = False, score_list: bool = False) -> List[Dict[str, str]]:
    """``items`` are (id, text). Every text is cut to the same length for scoring only; the reader still gets the whole record."""
    body = "\n".join(f"[{cid}] {text[:ITEM_CHARS]}" for cid, text in items)
    system = RERANK_SYSTEM_COMPACT if compact else RERANK_SYSTEM_LIST if score_list else RERANK_SYSTEM
    return [{"role": "system", "content": system}, {"role": "user", "content": f"提问时间：{ask_time}\n\n问题：{question}\n\n候选记录：\n{body}"}]


def parse_scores(raw: str, ids: Sequence[str]) -> Optional[Dict[str, int]]:
    """Scores for the ids shown, or None. An unknown, repeated or out-of-range id, or fewer than half the ids, makes the whole reply unusable."""
    found = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not found:
        return None
    try:
        data = json.loads(found.group(0))
    except ValueError:
        return None
    scores = data.get("scores") if isinstance(data, dict) else None
    if not isinstance(scores, dict):
        return None
    allowed = set(ids)
    out: Dict[str, int] = {}
    for cid, value in scores.items():
        if cid not in allowed or isinstance(value, bool) or not isinstance(value, (int, float)) or value not in SCORES:
            return None
        out[cid] = int(value)
    if len(out) * 2 < len(ids):
        return None
    return {cid: out.get(cid, 0) for cid in ids}


def parse_keep(raw: str, ids: Sequence[str]) -> Optional[Dict[str, int]]:
    """The listed ids as scores, best first, or None. An unknown or repeated id, or more than ``KEEP_MAX``, makes the reply unusable. An empty list is valid."""
    found = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not found:
        return None
    try:
        data = json.loads(found.group(0))
    except ValueError:
        return None
    keep = data.get("keep") if isinstance(data, dict) else None
    if not isinstance(keep, list) or len(keep) > KEEP_MAX or len(set(map(str, keep))) != len(keep) or any(not isinstance(k, str) or k not in set(ids) for k in keep):
        return None
    return {cid: (len(keep) - keep.index(cid) if cid in keep else 0) for cid in ids}


def parse_score_list(raw: str, ids: Sequence[str]) -> Optional[Dict[str, int]]:
    """One score per id, in id order. Any other length or value makes the reply unusable."""
    found = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not found:
        return None
    try:
        data = json.loads(found.group(0))
    except ValueError:
        return None
    values = data.get("s") if isinstance(data, dict) else None
    if not isinstance(values, list) or len(values) != len(ids) or any(isinstance(v, bool) or not isinstance(v, int) or v not in SCORES for v in values):
        return None
    return dict(zip(ids, values))


class RerankCache:
    """Keyed by model, prompt, question, time and the exact candidate texts in order. A stored request is never paid for twice."""

    def __init__(self, stored: Optional[Dict[str, Mapping[str, int]]] = None, on_add: Optional[Callable[[str, Mapping[str, int], dict], None]] = None) -> None:
        self.stored = dict(stored or {})
        self.on_add = on_add
        self._lock = threading.Lock()

    @staticmethod
    def key(messages: Sequence[Mapping[str, str]], model: str = RERANK_MODEL) -> str:
        return hashlib.sha256(json.dumps({"model": model, "messages": list(messages)}, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Dict[str, int]]:
        with self._lock:
            found = self.stored.get(key)
        return None if found is None else dict(found)

    def put(self, key: str, scores: Mapping[str, int], usage: dict) -> None:
        with self._lock:
            self.stored[key] = dict(scores)
        if self.on_add:
            self.on_add(key, dict(scores), usage)


def rerank(client, cache: RerankCache, question: str, ask_time: str, candidates: Sequence[Candidate], texts: Mapping[str, str], timings: Dict[str, float], *, attempts: int = 2, compact: bool = False, score_list: bool = False) -> Tuple[List[Candidate], str]:
    """(the candidates reordered by score then by their own order, how it went). On failure the order is unchanged."""
    ids = [f"C{n:02d}" for n in range(1, len(candidates) + 1)]
    messages = rerank_messages(question, ask_time, [(cid, texts.get(c.key, "")) for cid, c in zip(ids, candidates)], compact=compact, score_list=score_list)
    key = RerankCache.key(messages)
    scores = cache.get(key)
    status = "cache"
    timings["rerank_api"] = 0.0
    if scores is None:
        if client is None:
            return list(candidates), "fallback_no_client"
        status = "fallback_invalid"
        for _ in range(attempts):
            try:
                reply = client.chat(RERANK_MODEL, messages, max_tokens=120 if compact or score_list else RERANK_MAX_TOKENS, temperature=0)
            except (BudgetExceeded, OfflineCall):
                raise
            except (ApiError, OSError, ValueError):
                status = "fallback_api_error"
                continue
            timings["rerank_api"] = timings.get("rerank_api", 0.0) + reply.latency_ms
            if reply.model != RERANK_MODEL:
                status = "fallback_model_changed"
                break
            scores = (parse_keep if compact else parse_score_list if score_list else parse_scores)(reply.text, ids)
            if scores is not None:
                cache.put(key, scores, {"input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens, "latency_ms": reply.latency_ms})
                status = "api"
                break
        if scores is None:
            return list(candidates), status
    order = sorted(range(len(candidates)), key=lambda n: (-scores[ids[n]], n))
    return [candidates[n] for n in order], status


def skip_raw_after_summary(ranked: Sequence[Candidate], groups: Mapping[str, str], k: int) -> List[Candidate]:
    """In order, but a raw record whose own stored event already has its summary chosen gives its place to the next record. Fewer than ``k`` only if the pool is short."""
    chosen: List[Candidate] = []
    summaries = set()
    skipped: List[Candidate] = []
    for c in ranked:
        group = groups.get(c.key)
        if c.record_type != "daily_event" and group and group in summaries:
            skipped.append(c)
            continue
        chosen.append(c)
        if c.record_type == "daily_event" and group:
            summaries.add(group)
        if len(chosen) == k:
            return chosen
    return (chosen + skipped)[:k]


class SelectionRetriever:
    """The frozen EXP-B pool, then a new choice of the final ten. With nothing enabled, the EXP-B result is returned as it is."""

    def __init__(self, base: MultiFactRetriever, *, client=None, cache: Optional[RerankCache] = None) -> None:
        self.base = base
        self.client = client
        self.cache = cache or RerankCache()

    def retrieve(self, c: SelectionConfig, persona_id: str, question: str, as_of: datetime, known_at: datetime, *, ask_time: str, calendar=None) -> Tuple[Retrieval, str]:
        if c.rerank_k > RERANK_MAX_CANDIDATES:
            raise ValueError(f"at most {RERANK_MAX_CANDIDATES} candidates go to the reranker")
        first = self.base.retrieve(c.pool, persona_id, question, as_of, known_at, calendar=calendar)
        return self.choose(c, first, question, ask_time)

    def choose(self, c: SelectionConfig, first: Retrieval, question: str, ask_time: str) -> Tuple[Retrieval, str]:
        """The new final ten from a pool already retrieved. Safe to call from several threads."""
        if c.rerank_k > RERANK_MAX_CANDIDATES:
            raise ValueError(f"at most {RERANK_MAX_CANDIDATES} candidates go to the reranker")
        if not (c.rerank_k or c.provenance):
            return first, "unchanged"
        started = time.perf_counter()
        timings = dict(first.timings_ms)
        ranked = list(first.pool)
        status = "cpu"
        if c.rerank_k:
            head, tail = ranked[:c.rerank_k], ranked[c.rerank_k:]
            documents = self.base.retriever.documents
            texts = {x.key: documents[x.record_id].text for x in head}
            head, status = rerank(self.client, self.cache, question, ask_time, head, texts, timings, compact=c.compact, score_list=c.score_list)
            ranked = head + tail
        final = skip_raw_after_summary(ranked, self.base.retriever.groups, c.final_k) if c.provenance else ranked[:c.final_k]
        timings["selection"] = (time.perf_counter() - started) * 1000 - timings.get("rerank_api", 0.0)
        timings["total"] = first.timings_ms.get("total", 0.0) + (time.perf_counter() - started) * 1000
        return replace(first, final=final, timings_ms=timings), status


def candidates(pool: MultiFactConfig) -> Dict[str, SelectionConfig]:
    """Registered before any of them ran (docs/V3_2_RESEARCH_PROTOCOL.md, amendment 1)."""
    return {"v32_r2_rerank30": SelectionConfig("v32_r2_rerank30", pool, rerank_k=30),
            "v32_r1_provenance": SelectionConfig("v32_r1_provenance", pool, provenance=True),
            "v32_r4_rerank_provenance": SelectionConfig("v32_r4_rerank_provenance", pool, rerank_k=30, provenance=True)}


def revisions(pool: MultiFactConfig) -> Dict[str, SelectionConfig]:
    """Revision 1, registered after V32-1 and V32-2 ran (amendment 2): the same thirty, a short reply, for latency."""
    return {"v32_r2b_rerank30_compact": SelectionConfig("v32_r2b_rerank30_compact", pool, rerank_k=30, compact=True),
            # revision 2, registered after revision 1 ran (amendment 3)
            "v32_r2c_rerank30_scorelist": SelectionConfig("v32_r2c_rerank30_scorelist", pool, rerank_k=30, score_list=True)}
