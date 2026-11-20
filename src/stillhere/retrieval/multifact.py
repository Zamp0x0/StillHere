"""V3 research: bounded multi-fact retrieval. An optional experiment around the frozen pipeline, which it calls and never changes.

question -> frozen first pass -> optional sub-queries, month windows and stored links -> one bounded pool -> optional coverage-aware selection -> final evidence

Every extra path passes the same scope mask. A record that may not be seen at query time is never considered. The inputs are the question, the persona, the times
and the records: no QA field reaches this module. Nothing here is registered on ``MemoryAPI`` by default.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from stillhere.memory.search import normalize, query_units

from .date_channel import records_in_window
from .dense import ScopeTable
from .event_graph import EventGraph
from .expansion import ExpansionConfig, expand
from .hybrid import Candidate, HybridRetriever, Retrieval, RetrievalConfig, fuse_rrf, lexical_optimized
from .hybrid_keys import evidence_key, record_type_of
from .profiles import QUALITY
from .temporal_anchor import OK, TemporalAnchor, parse_anchor

MAX_SUBQUERIES = 3
MAX_POOL = 300
SUBQUERY_MODEL = "qwen3.7-flash-2026-07-15"
SUBQUERY_MAX_TOKENS = 200
SUBQUERY_SYSTEM = """你把用户关于自己过去的问题，改写成用于搜索个人记录的短查询。

规则：
1. 只根据问题本身改写，不要回答问题，不要编造问题里没有的人名、地点或事实。
2. 问题涉及多个事件、多个时间点或多类事实时，每条查询只针对其中一个。
3. 问题要求归纳或判断时，写出需要查找的具体记录类型，例如相关的安排、记录或变化。
4. 每条查询 4 到 20 个字，保留问题里的人名、地点和日期，去掉"请问""帮我"这类词。
5. 最多 3 条，不要重复原问题。

只输出 JSON：{"queries": ["查询1", "查询2"]}"""


@dataclass(frozen=True)
class MultiQueryConfig:
    max_subqueries: int = MAX_SUBQUERIES
    original_weight: float = 2.0          # the original question counts this many times in the fusion
    lexical_k: int = 30                   # per channel, per sub-query
    dense_k: int = 30


@dataclass(frozen=True)
class WindowConfig:
    """Search inside a span of time the question names, by month or part of a month."""

    lexical_k: int = 20
    dense_k: int = 20


@dataclass(frozen=True)
class SelectionConfig:
    """Greedy choice of the final list: relevance first, less credit for repeating an event, a day or the same words."""

    head: int = 60                        # how far down the ranked pool the choice looks
    same_event_penalty: float = 0.5
    near_text_penalty: float = 0.7
    near_text_threshold: float = 0.6
    facet_bonus: float = 0.25             # credit for a sub-query with nothing in the list yet


@dataclass(frozen=True)
class MultiFactConfig:
    name: str
    base: RetrievalConfig
    multi_query: Optional[MultiQueryConfig] = None
    window: Optional[WindowConfig] = None
    links: Optional[ExpansionConfig] = None
    selection: Optional[SelectionConfig] = None
    max_pool: int = MAX_POOL
    span_reserve: int = 0                 # revision 2: final places kept for records found inside a span the question names
    link_channel: int = 0                 # revision 1: rank linked neighbours by their own similarity to the question, as one more fused list of this length

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- sub-queries

def parse_subqueries(raw: str, question: str, limit: int = MAX_SUBQUERIES) -> Tuple[str, ...]:
    """Valid, different, bounded queries from a model reply. A broken reply gives none: the original question always runs."""
    found = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not found:
        return ()
    try:
        data = json.loads(found.group(0))
    except ValueError:
        return ()
    items = data.get("queries") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return ()
    base = normalize(question)
    seen = {base}
    out: List[str] = []
    for item in items:
        if isinstance(item, str):
            text = item.strip()
            key = normalize(text)
            if 2 <= len(text) <= 60 and key not in seen:
                seen.add(key)
                out.append(text)
    return tuple(out[:limit])


class SubqueryCache:
    """In memory, keyed by model, prompt and question. A stored question is never paid for twice."""

    def __init__(self, stored: Optional[Dict[str, Sequence[str]]] = None, on_add: Optional[Callable[[str, Sequence[str], dict], None]] = None) -> None:
        self.stored = dict(stored or {})
        self.on_add = on_add
        self._lock = threading.Lock()

    @staticmethod
    def key(question: str, *, model: str = SUBQUERY_MODEL, system: str = SUBQUERY_SYSTEM) -> str:
        return hashlib.sha256("\x1f".join((model, system, question)).encode("utf-8")).hexdigest()

    def get(self, question: str) -> Optional[Tuple[str, ...]]:
        with self._lock:
            found = self.stored.get(self.key(question))
        return None if found is None else tuple(found)

    def put(self, question: str, queries: Sequence[str], usage: dict) -> None:
        key = self.key(question)
        with self._lock:
            self.stored[key] = list(queries)
        if self.on_add:
            self.on_add(key, list(queries), usage)


def subqueries(client, cache: SubqueryCache, question: str, limit: int, timings: Dict[str, float]) -> Tuple[str, ...]:
    found = cache.get(question)
    if found is not None:
        timings["subquery_api"] = 0.0
        return found[:limit]
    messages = [{"role": "system", "content": SUBQUERY_SYSTEM}, {"role": "user", "content": question}]
    reply = client.chat(SUBQUERY_MODEL, messages, max_tokens=SUBQUERY_MAX_TOKENS, temperature=0)
    timings["subquery_api"] = reply.latency_ms
    queries = parse_subqueries(reply.text, question, MAX_SUBQUERIES)
    cache.put(question, queries, {"input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens, "latency_ms": reply.latency_ms})
    return queries[:limit]


# ---------------------------------------------------------------- time spans

_CN = {c: i for i, c in enumerate("一二三四五六七八九", 1)} | {"十": 10, "十一": 11, "十二": 12}
_MONTH = "(?:1[0-2]|[1-9]|十一|十二|[一二三四五六七八九十])"
_SPAN = re.compile(f"(?:(?P<year>20[0-9]{{2}})年)?(?P<month>{_MONTH})月(?:份)?(?P<part>上旬|中旬|下旬|月初|月底|月末|初|底|末|上半月|下半月)?(?![0-9一二三四五六七八九十日号])")
_PARTS = {"上旬": (1, 11), "初": (1, 11), "月初": (1, 11), "中旬": (11, 21), "下旬": (21, 32), "底": (21, 32), "月底": (21, 32), "末": (21, 32), "月末": (21, 32), "上半月": (1, 16), "下半月": (16, 32)}


def _month_number(text: str) -> int:
    return int(text) if text.isdigit() else _CN[text]


def month_spans(question: str, asked: datetime, tz: Optional[tzinfo] = None) -> List[Tuple[datetime, datetime]]:
    """Spans named by month, at most three, none after the question. Without a year, the latest such month that has begun by the question."""
    local = asked.astimezone(tz) if tz else asked
    offset = timezone(local.utcoffset())
    text = normalize(question)
    out: List[Tuple[datetime, datetime]] = []
    for found in _SPAN.finditer(text):
        month = _month_number(found.group("month"))
        year = int(found.group("year")) if found.group("year") else (local.year if month <= local.month else local.year - 1)
        low, high = _PARTS.get(found.group("part") or "", (1, 32))
        start = datetime(year, month, low, tzinfo=offset)
        end = datetime(year + month // 12, month % 12 + 1, 1, tzinfo=offset) if high == 32 else datetime(year, month, high, tzinfo=offset)
        if start <= local and (start, end) not in out:
            out.append((start, min(end, local + timedelta(microseconds=1))))
    return out[:3]


def question_spans(question: str, asked: datetime, tz: Optional[tzinfo] = None) -> List[Tuple[datetime, datetime]]:
    """Exact dates through the frozen parser, then spans of a month. At most three."""
    anchor = parse_anchor(question, asked, tz)
    spans = [(anchor.start, anchor.end)] if anchor.usable else []
    return (spans + [s for s in month_spans(question, asked, tz) if s not in spans])[:3]


def window_mask(scope: ScopeTable, persona_id: str, mask: np.ndarray, start: datetime, end: datetime) -> np.ndarray:
    """The scope mask narrowed to records dated inside the span. It can only remove records."""
    rows = records_in_window(scope, persona_id, mask, TemporalAnchor(OK, "SPAN", start, end))
    narrowed = np.zeros_like(mask)
    narrowed[rows] = True
    return narrowed & mask


# ---------------------------------------------------------------- selection

def _grams(text: str) -> Set[str]:
    compact = "".join(text.split())
    return {compact[i:i + 2] for i in range(len(compact) - 1)}


def _similar(a: Set[str], b: Set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def select_covering(ranked: Sequence[Candidate], texts: Mapping[str, str], config: SelectionConfig, k: int, facets: Optional[Mapping[str, Set[int]]] = None) -> List[Candidate]:
    """``ranked`` is best first. A record's relevance is its rank, so the top record always stays and nothing far down can pass on variety alone. Ties go to rank."""
    head = list(ranked[:config.head])
    grams = {c.key: _grams(texts.get(c.key, "")) for c in head}
    relevance = {c.key: 1.0 / (1 + rank / 10) for rank, c in enumerate(head)}
    chosen: List[Candidate] = []
    events: Set[str] = set()
    seen_facets: Set[int] = set()
    left = list(head)
    while left and len(chosen) < k:
        best, best_score = None, float("-inf")
        for c in left:
            score = relevance[c.key]
            if c.event_group_id and c.event_group_id in events:
                score *= config.same_event_penalty
            if any(_similar(grams[c.key], grams[o.key]) >= config.near_text_threshold for o in chosen):
                score *= config.near_text_penalty
            if facets and (facets.get(c.key, set()) - seen_facets):
                score += config.facet_bonus
            if score > best_score:
                best, best_score = c, score
        chosen.append(best)
        left.remove(best)
        if best.event_group_id:
            events.add(best.event_group_id)
        if facets:
            seen_facets |= facets.get(best.key, set())
    return chosen


def reserve_spans(ranked: Sequence[Candidate], origin: Mapping[str, str], k: int, reserve: int) -> List[Candidate]:
    """The best ``k - reserve`` in fused order, then up to ``reserve`` of the best records only a span search found, then the rest in order. Nothing is reordered within a group."""
    head = list(ranked[:k - reserve])
    taken = {c.key for c in head}
    extra = [c for c in ranked if c.key not in taken and origin.get(c.key, "").startswith("span")][:reserve]
    taken |= {c.key for c in extra}
    return (head + extra + [c for c in ranked if c.key not in taken])[:k]


# ---------------------------------------------------------------- pipeline

class MultiFactRetriever:
    """Runs the frozen retriever for the question, then adds the optional paths. With nothing enabled, the result is the frozen one, untouched."""

    def __init__(self, retriever: HybridRetriever, *, graph: Optional[EventGraph] = None, client=None, cache: Optional[SubqueryCache] = None) -> None:
        self.retriever = retriever
        self.graph = graph
        self.client = client
        self.cache = cache or SubqueryCache()
        self._rows: Dict[str, Dict[str, List[int]]] = {}

    def similarity(self, persona_id: str, vector: np.ndarray, record_ids: Sequence[str]) -> Dict[str, float]:
        """Cosine of each record's best chunk to the question vector."""
        index = self.retriever.indexes[persona_id]
        if persona_id not in self._rows:
            rows: Dict[str, List[int]] = {}
            for n, record in enumerate(index.rows[persona_id]):
                rows.setdefault(record, []).append(n)
            self._rows[persona_id] = rows
        rows = self._rows[persona_id]
        matrix = index.vectors[persona_id]
        return {r: float(max(matrix[i] @ vector for i in rows[r])) for r in record_ids if r in rows}

    def _search(self, base: RetrievalConfig, persona_id: str, query: str, mask: np.ndarray, lexical_k: int, dense_k: int, timings: Dict[str, float]) -> Tuple[List[str], List[str], List[str]]:
        """Ranked stored records for one query inside one mask: memories, messages, and the dense list."""
        r = self.retriever
        facts = lexical_optimized(r.api, r.scope, persona_id, query, mask, lexical_k, "memory_fts", "memory_id")
        messages = lexical_optimized(r.api, r.scope, persona_id, query, mask, lexical_k, "messages_fts", "message_id")
        vector = r.query_vector(base.embedding_model, base.dimension, query, timings)
        dense = r.indexes[persona_id].search(persona_id, vector, mask, dense_k * 2)
        return [x for x, _ in facts], [x for x, _ in messages], [x for x, _ in dense]

    def retrieve(self, c: MultiFactConfig, persona_id: str, question: str, as_of: datetime, known_at: datetime, *, calendar: Optional[tzinfo] = None) -> Retrieval:
        started = time.perf_counter()
        base = replace(c.base, parallel=False)
        first = self.retriever.retrieve(base, persona_id, question, as_of, known_at, calendar=calendar)
        if not (c.multi_query or c.window or c.links or c.selection):
            return first
        timings = dict(first.timings_ms)
        scope = self.retriever.scope
        mask = scope.mask(persona_id, as_of, known_at, include_ended=base.include_ended)
        lists: List[Sequence[str]] = []
        ranked_first = [x.key for x in first.pool]
        weight = c.multi_query.original_weight if c.multi_query else 1.0
        lists += [ranked_first] * max(int(weight), 1)
        records: Dict[str, str] = {x.key: x.record_id for x in first.pool}
        channels: Dict[str, List[str]] = {x: list(v) for x, v in first.channels.items()}
        facets: Dict[str, Set[int]] = {}
        origin: Dict[str, str] = {}

        def add(ranked: Sequence[str], tag: str) -> None:
            keys: List[str] = []
            for record in ranked:
                key = evidence_key(persona_id, record)
                if key not in keys:
                    keys.append(key)
                    records.setdefault(key, record)
                    origin.setdefault(key, tag)
            if keys:
                lists.append(keys)
                channels[tag] = keys

        if c.multi_query and self.client is not None:
            queries = subqueries(self.client, self.cache, question, c.multi_query.max_subqueries, timings)
            begin = time.perf_counter()
            for number, query in enumerate(queries):
                f, m, d = self._search(base, persona_id, query, mask, c.multi_query.lexical_k, c.multi_query.dense_k, timings)
                for tag, ranked in (("f", f), ("m", m), ("d", d[:c.multi_query.dense_k])):
                    add(ranked, f"sub{number}_{tag}")
                    for key in channels.get(f"sub{number}_{tag}", [])[:10]:
                        facets.setdefault(key, set()).add(number)
            timings["subquery_search"] = (time.perf_counter() - begin) * 1000
        if c.links and c.link_channel and self.graph is not None:
            begin = time.perf_counter()
            seeds = [(x.key, x.fusion_score) for x in first.pool[:c.links.seeds]]
            reached = expand(self.graph, scope, mask, persona_id, seeds, {x.key for x in first.pool}, {}, c.links)
            vector = self.retriever.query_vector(base.embedding_model, base.dimension, question, timings)
            close = self.similarity(persona_id, vector, [e.record_id for e in reached])
            ranked = sorted((e for e in reached if e.record_id in close), key=lambda e: (-close[e.record_id], e.key))[:c.link_channel]
            add([e.record_id for e in ranked], "link_ranked")
            timings["link_channel"] = (time.perf_counter() - begin) * 1000
        if c.window:
            begin = time.perf_counter()
            for number, (start, end) in enumerate(question_spans(question, as_of, calendar)):
                narrowed = window_mask(scope, persona_id, mask, start, end)
                if narrowed.any():
                    f, m, d = self._search(base, persona_id, question, narrowed, c.window.lexical_k, c.window.dense_k, timings)
                    for tag, ranked in (("f", f), ("m", m), ("d", d[:c.window.dense_k])):
                        add(ranked, f"span{number}_{tag}")
            timings["span_search"] = (time.perf_counter() - begin) * 1000
        begin = time.perf_counter()
        scores = fuse_rrf(lists, base.rrf_k)
        ordered = sorted(scores, key=lambda k: (-scores[k], k))[:c.max_pool]
        by_key = {x.key: x for x in first.pool}
        pool: List[Candidate] = []
        for key in ordered:
            found = by_key.get(key)
            if found is not None:
                pool.append(replace(found, fusion_score=scores[key]))
            else:
                record = records[key]
                pool.append(Candidate(key, record, record_type_of(record), persona_id, event_group_id=self.retriever.groups.get(key), retrieval_channels=(origin.get(key, "extra"),), fusion_score=scores[key]))
        timings["v3_fusion"] = (time.perf_counter() - begin) * 1000
        if c.links and not c.link_channel and self.graph is not None:
            begin = time.perf_counter()
            seeds = [(x.key, x.fusion_score) for x in pool[:c.links.seeds]]
            reached = expand(self.graph, scope, mask, persona_id, seeds, {x.key for x in pool}, {}, c.links)
            extra = [Candidate(e.key, e.record_id, record_type_of(e.record_id), persona_id, event_group_id=self.retriever.groups.get(e.key), retrieval_channels=("link",), fusion_score=e.score, retrieval_origin="expanded", seed_key=e.seed_key, relation_type=e.relation, path_provenance=e.provenance) for e in reached]
            pool = sorted(pool + extra, key=lambda x: (-x.fusion_score, x.key))[:c.max_pool]
            timings["link_expansion"] = (time.perf_counter() - begin) * 1000
        begin = time.perf_counter()
        visible = [x for x in pool if scope.visible(persona_id, x.record_id, mask)]
        removed = len(pool) - len(visible)
        if c.selection:
            texts = {x.key: self.retriever.documents[x.record_id].text for x in visible[:c.selection.head]}
            final = select_covering(visible, texts, c.selection, base.final_k, facets if c.multi_query else None)
        elif c.span_reserve and any(t.startswith("span") for t in channels):
            final = reserve_spans(visible, origin, base.final_k, c.span_reserve)
        else:
            final = visible[:base.final_k]
        timings["v3_selection"] = (time.perf_counter() - begin) * 1000
        timings["total"] = (time.perf_counter() - started) * 1000
        return Retrieval(final, visible, channels, timings, first.rerank_failed, first.removed_by_validation + removed, first.anchor, first.date_window_records)


# Registered before any candidate ran. A new entry is a new experiment and counts against the limit of six in docs/V3_RESEARCH_PROTOCOL.md.
BASELINE = MultiFactConfig("v3_baseline_quality", QUALITY)
CANDIDATES: Dict[str, MultiFactConfig] = {
    "exp_a_multiquery": MultiFactConfig("exp_a_multiquery", QUALITY, multi_query=MultiQueryConfig()),
    "exp_b_time_links": MultiFactConfig("exp_b_time_links", QUALITY, window=WindowConfig(), links=ExpansionConfig(("same_event", "tree_root"), seeds=20, per_seed=5, global_budget=50)),
    "exp_c_coverage": MultiFactConfig("exp_c_coverage", QUALITY, selection=SelectionConfig()),
}

# Revision round 1, registered after the first three ran on D0 and D1 (docs/V3_RESEARCH_PROTOCOL.md, amendment 1).
CANDIDATES["exp_b2_link_channel"] = replace(CANDIDATES["exp_b_time_links"], name="exp_b2_link_channel", link_channel=30)

# Revision round 2, registered after revision 1 ran on D0 and D1 (amendment 1).
CANDIDATES["exp_b3_span_reserve"] = replace(CANDIDATES["exp_b_time_links"], name="exp_b3_span_reserve", span_reserve=4)
