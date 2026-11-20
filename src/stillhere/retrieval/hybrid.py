"""The configurable retrieval pipeline. SQLite stays the only store; this module reads it and ranks.

question -> persona and time scope -> lexical and dense candidates -> union by evidence -> fusion
         -> optional rerank -> temporal validation -> final evidence

A configuration chooses the parts, so every profile runs the same code. The inputs are the question, the persona, the
times and the records themselves: nothing here can see a QA file. Event groups come from each record's own links.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, tzinfo
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from stillhere.memory import MemoryAPI
from stillhere.memory.search import BIGRAM_WEIGHT, UNIGRAM_WEIGHT, QueryUnits, match_expression, query_units

from .alibaba_client import ApiError, BudgetExceeded, DashScopeClient, OfflineCall
from .date_channel import TimeAnchorConfig, date_candidates
from .dense import DenseIndex, Document, EmbeddingCache, ScopeTable, normalized
from .event_graph import EventGraph
from .expansion import ExpansionConfig, expand
from .hybrid_keys import evidence_key, record_type_of
from .temporal_anchor import TemporalAnchor, parse_anchor


@dataclass(frozen=True)
class RetrievalConfig:
    name: str
    lexical: Optional[str] = "optimized"        # "v1", "optimized" or None
    dense: bool = True
    lexical_k: int = 50                         # per lexical channel
    dense_k: int = 50
    fusion: str = "rrf"                         # "rrf", "best_rank" or "score"
    rrf_k: int = 60
    rerank_model: Optional[str] = None
    rerank_k: int = 50
    event_weight: float = 0.0                   # soft signal on fused scores; 0 is off, and no record is ever dropped
    final_k: int = 10
    include_ended: bool = True
    embedding_model: Optional[str] = "qwen3.7-text-embedding-flash"
    dimension: int = 1024
    parallel: bool = False
    expansion: Optional[ExpansionConfig] = None
    time_anchor: Optional[TimeAnchorConfig] = None

    @property
    def mode(self) -> str:
        if not self.dense:
            return "lexical"
        if not self.lexical:
            return "dense"
        return "hybrid_rerank" if self.rerank_model else "hybrid"

    def digest(self) -> str:
        values = asdict(self)
        if self.expansion is None:      # a configuration without a hop keeps the identity it had before hops existed
            del values["expansion"]
        if self.time_anchor is None:    # likewise for the date channel
            del values["time_anchor"]
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()[:16]


@dataclass
class Candidate:
    key: str
    record_id: str
    record_type: str
    persona_id: str
    source_locator: Optional[str] = None
    event_group_id: Optional[str] = None
    retrieval_channels: Tuple[str, ...] = ()
    lexical_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    fusion_score: float = 0.0
    rerank_score: Optional[float] = None
    time_eligibility: bool = True
    retrieval_origin: str = "direct"            # "direct" or "expanded"
    seed_key: Optional[str] = None
    relation_type: Optional[str] = None
    path_provenance: Optional[str] = None


@dataclass
class Retrieval:
    final: List[Candidate]
    pool: List[Candidate]
    channels: Dict[str, List[str]]
    timings_ms: Dict[str, float] = field(default_factory=dict)
    rerank_failed: bool = False
    removed_by_validation: int = 0
    anchor: Optional[TemporalAnchor] = None
    date_window_records: int = 0


_MESSAGE_GROUP = re.compile(r"daily_event:event_id=([^;]+)")


def event_groups(conn: sqlite3.Connection, persona_id: str) -> Dict[str, str]:
    """Evidence key -> event group, from provenance and locators already stored with each record."""
    groups: Dict[str, str] = {}
    directory = persona_id.split("_", 1)[-1]
    for memory_id, locator in conn.execute(
            "SELECT p.memory_id, p.locator FROM provenance AS p JOIN memories AS m ON m.memory_id = p.memory_id "
            "WHERE m.persona_id = ? AND p.source_id = ?", (persona_id, f"lb.{directory}.daily_event")):
        groups.setdefault(evidence_key(persona_id, memory_id), f"{persona_id}:event:{locator.split('=', 1)[1]}")
    for message_id, locator in conn.execute("SELECT message_id, locator FROM messages WHERE target_persona_id = ?", (persona_id,)):
        found = _MESSAGE_GROUP.search(locator or "")
        if found:
            groups.setdefault(evidence_key(persona_id, message_id), f"{persona_id}:event:{found.group(1)}")
    return groups


# ---------------------------------------------------------------- fusion

def fuse_rrf(lists: Sequence[Sequence[str]], k: int) -> Dict[str, float]:
    """A record found by several retrievers adds up; each list gives a record one rank."""
    scores: Dict[str, float] = {}
    for ranked in lists:
        for rank, key in enumerate(ranked, 1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return scores


def fuse_best_rank(lists: Sequence[Sequence[str]]) -> Dict[str, float]:
    """The plain union: the best rank a record has anywhere. Later lists lose ties."""
    scores: Dict[str, float] = {}
    width = len(lists) + 1
    for position, ranked in enumerate(lists):
        for rank, key in enumerate(ranked, 1):
            scores[key] = max(scores.get(key, float("-inf")), -(rank + position / width))
    return scores


def fuse_score(score_lists: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    """Min-max within each list, then the mean over all lists; a record absent from a list adds zero there."""
    total: Dict[str, float] = {}
    for scored in score_lists:
        if not scored:
            continue
        low, high = min(scored.values()), max(scored.values())
        span = (high - low) or 1.0
        for key, value in scored.items():
            total[key] = total.get(key, 0.0) + (value - low) / span
    return {k: v / max(len(score_lists), 1) for k, v in total.items()}


def apply_event_signal(scores: Mapping[str, float], groups: Mapping[str, str], weight: float) -> Dict[str, float]:
    """Raise (weight > 0) or lower (weight < 0) a record by how many others of its event are in the pool, up to three."""
    if not weight:
        return dict(scores)
    sizes: Dict[str, int] = {}
    for key in scores:
        group = groups.get(key)
        if group:
            sizes[group] = sizes.get(group, 0) + 1
    out = {}
    for key, score in scores.items():
        group = groups.get(key)
        extra = min(sizes.get(group, 1) - 1, 3) if group else 0
        out[key] = score + abs(score) * weight * extra
    return out


# ---------------------------------------------------------------- lexical

def bigram_units(query: str) -> Tuple[QueryUnits, QueryUnits]:
    """(the query's word pairs and whole words, its single characters). Single characters are only a fallback."""
    units = query_units(query)
    return QueryUnits(units.bigrams, ()), QueryUnits((), units.unigrams)


def _match(conn, table: str, id_column: str, units: QueryUnits) -> List[Tuple[str, float]]:
    return conn.execute(f"SELECT {id_column}, bm25({table}, 0.0, {BIGRAM_WEIGHT!r}, {UNIGRAM_WEIGHT!r}) FROM {table} "
                        f"WHERE {table} MATCH ?", (match_expression(units),)).fetchall()


def lexical_optimized(api: MemoryAPI, scope: ScopeTable, persona_id: str, query: str, mask: np.ndarray, k: int,
                      table: str, id_column: str, matched: Optional[Dict[str, float]] = None) -> List[Tuple[str, float]]:
    """Rank inside the FTS index and filter by the scope mask, with no join to the record tables.

    Word pairs first. Single characters run only when pairs alone find fewer than ``k`` records, so a short query or
    a one-character name still finds something. ``matched`` collects the score of every allowed record, cut or not."""
    pairs, singles = bigram_units(query)
    table_rows = scope.tables[persona_id]["row"]
    found: Dict[str, float] = {}
    with api.store._read() as conn:
        for units in (pairs, singles):
            if not units or len(found) >= k:
                continue
            rows = []
            for record_id, rank in _match(conn, table, id_column, units):
                row = table_rows.get(record_id)
                if row is not None and mask[row] and record_id not in found:
                    rows.append((rank, record_id))
            rows.sort()
            if matched is not None:
                top = -rows[0][0] if rows else 1.0
                for rank, record_id in rows:
                    matched.setdefault(record_id, -rank / top if top > 0 else 0.0)
            for rank, record_id in rows[:k - len(found)]:
                found[record_id] = -rank
    return list(found.items())


# ---------------------------------------------------------------- pipeline

class HybridRetriever:
    def __init__(self, api: MemoryAPI, scope: ScopeTable, documents: Mapping[str, Document], *, groups: Mapping[str, str] = {},
                 indexes: Optional[Mapping[str, DenseIndex]] = None, client: Optional[DashScopeClient] = None,
                 cache: Optional[EmbeddingCache] = None, rerank_cache: Optional[dict] = None, query_cache: bool = True,
                 graph: Optional[EventGraph] = None) -> None:
        self.api = api
        self.scope = scope
        self.documents = documents
        self.groups = groups
        self.indexes = indexes or {}
        self.client = client
        self.cache = cache
        self.rerank_cache = rerank_cache if rerank_cache is not None else {}
        self.query_cache = query_cache
        self.graph = graph
        self._cache_lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=2)   # one for the retriever's life: its threads keep their connections

    def query_vector(self, model: str, dimension: int, question: str, timings: Dict[str, float]) -> np.ndarray:
        key = EmbeddingCache.key(model, dimension, "query", question)
        if self.query_cache and self.cache is not None:
            found = self.cache.get(key)
            if found is not None:
                timings["query_embedding_api"] = 0.0
                return found
        result = self.client.embed(model, [question], text_type="query", dimension=dimension)
        timings["query_embedding_api"] = result.latency_ms
        vector = normalized(result.vectors[0])
        if self.cache is not None:
            self.cache.put_many([(key, vector)])
        return vector

    def _lexical(self, c: RetrievalConfig, persona_id: str, question: str, mask, as_of, known_at,
                 matched: Optional[Dict[str, float]] = None) -> Tuple[List[str], List[str]]:
        if c.lexical == "v1":
            facts = self.api.search(persona_id, question, as_of=as_of, known_at=known_at, k=c.lexical_k, include_ended=c.include_ended)
            messages = self.api.search_messages(persona_id, question, as_of=as_of, known_at=known_at, k=c.lexical_k)
            return [h.memory_id for h in facts], [h.message_id for h in messages]
        facts = lexical_optimized(self.api, self.scope, persona_id, question, mask, c.lexical_k, "memory_fts", "memory_id", matched)
        messages = lexical_optimized(self.api, self.scope, persona_id, question, mask, c.lexical_k, "messages_fts", "message_id", matched)
        return [r for r, _ in facts], [r for r, _ in messages]

    def retrieve(self, c: RetrievalConfig, persona_id: str, question: str, as_of: datetime, known_at: datetime, *, calendar: Optional[tzinfo] = None) -> Retrieval:
        """``calendar`` is the persona's timezone for reading dates in the question. Without one, the offset of ``as_of`` is used."""
        t: Dict[str, float] = {}
        mark = time.perf_counter
        started = mark()
        mask = self.scope.mask(persona_id, as_of, known_at, include_ended=c.include_ended)
        t["temporal_scope"] = (mark() - started) * 1000
        lexical_facts: List[str] = []
        lexical_messages: List[str] = []
        vector = None
        matched: Dict[str, float] = {}

        def run_lexical():
            begin = mark()
            result = self._lexical(c, persona_id, question, mask, as_of, known_at, matched if c.expansion else None)
            t["lexical"] = (mark() - begin) * 1000
            return result

        def run_embedding():
            return self.query_vector(c.embedding_model, c.dimension, question, t)

        if c.parallel and c.lexical and c.dense:
            lexical_future, vector_future = self._pool.submit(run_lexical), self._pool.submit(run_embedding)
            lexical_facts, lexical_messages = lexical_future.result()
            vector = vector_future.result()
        else:
            if c.lexical:
                lexical_facts, lexical_messages = run_lexical()
            if c.dense:
                vector = run_embedding()
        dense_hits: List[Tuple[str, float]] = []
        if c.dense:
            begin = mark()
            dense_hits = self.indexes[persona_id].search(persona_id, vector, mask, c.dense_k * 2)
            t["dense"] = (mark() - begin) * 1000

        begin = mark()
        channels = {"lexical_facts": _unique(persona_id, lexical_facts), "lexical_messages": _unique(persona_id, lexical_messages),
                    "dense": _unique(persona_id, [r for r, _ in dense_hits])[:c.dense_k]}
        best_record: Dict[str, str] = {}
        anchor: Optional[TemporalAnchor] = None
        dated: List[Tuple[str, int]] = []
        window_records = 0
        if c.time_anchor:
            t["union"] = (mark() - begin) * 1000
            begin = mark()
            anchor = parse_anchor(question, as_of, calendar)
            t["anchor_parse"] = (mark() - begin) * 1000
            begin = mark()
            if anchor.usable:
                dated, window_records = date_candidates(self.scope, self.documents, persona_id, mask, anchor, c.time_anchor.date_k)
                if dated:
                    channels["date_anchor"] = _unique(persona_id, [r for r, _ in dated])
            t["date_channel"] = (mark() - begin) * 1000
            begin = mark()
        for ranked in (dense_hits, [(r, 0.0) for r in lexical_facts], [(r, 0.0) for r in lexical_messages], dated):
            for record, _ in ranked:
                best_record.setdefault(evidence_key(persona_id, record), record)
        dense_scores: Dict[str, float] = {}
        for record, score in dense_hits:
            dense_scores.setdefault(evidence_key(persona_id, record), score)
        t["union"] = (mark() - begin) * 1000

        begin = mark()
        lists = [v for v in channels.values() if v]
        if c.fusion == "rrf":
            scores = fuse_rrf(lists, c.rrf_k)
        elif c.fusion == "best_rank":
            scores = fuse_best_rank(lists)
        elif c.fusion == "score":
            scores = fuse_score([{k: 1.0 / r for r, k in enumerate(v, 1)} for k_, v in channels.items() if v and k_ != "dense"]
                                + [{k: dense_scores[k] for k in channels["dense"]}])
        else:
            raise ValueError(f"unknown fusion {c.fusion}")
        scores = apply_event_signal(scores, self.groups, c.event_weight)
        ordered = sorted(scores, key=lambda k: (-scores[k], k))
        t["fusion"] = (mark() - begin) * 1000

        pool = [self._candidate(persona_id, k, best_record[k], channels, scores[k], dense_scores.get(k)) for k in ordered]
        expanded: List[Candidate] = []
        if c.expansion and self.graph is not None:
            begin = mark()
            seeds = [(x.key, x.fusion_score) for x in pool[:c.expansion.seeds]]
            reached = expand(self.graph, self.scope, mask, persona_id, seeds, {x.key for x in pool}, matched, c.expansion)
            t["graph_lookup"] = (mark() - begin) * 1000
            begin = mark()
            expanded = [Candidate(e.key, e.record_id, record_type_of(e.record_id), persona_id, event_group_id=self.groups.get(e.key), retrieval_channels=("expansion",),
                                  fusion_score=e.score, retrieval_origin="expanded", seed_key=e.seed_key, relation_type=e.relation, path_provenance=e.provenance) for e in reached]
            pool = _with_reserved(sorted(pool + expanded, key=lambda x: (-x.fusion_score, x.key)), expanded, c.final_k, c.expansion.reserve)
            t["expansion_rank"] = (mark() - begin) * 1000
        final, failed = pool, False
        t["rerank_api"] = 0.0
        if c.rerank_model:
            final, failed = self._rerank(c, question, pool, t)
        begin = mark()
        valid = []
        removed = 0
        for candidate in final[:c.final_k + 20]:
            candidate.time_eligibility = self.scope.visible(persona_id, candidate.record_id, mask)
            if candidate.time_eligibility:
                valid.append(candidate)
            else:
                removed += 1
        t["validation"] = (mark() - begin) * 1000
        t["total"] = (mark() - started) * 1000
        return Retrieval(valid[:c.final_k], pool, channels, t, failed, removed, anchor, window_records)

    def _candidate(self, persona_id: str, key: str, record: str, channels: Mapping[str, Sequence[str]], score: float,
                   dense_score: Optional[float]) -> Candidate:
        in_lexical = [n for n in ("lexical_facts", "lexical_messages") if key in channels[n]]
        lexical_rank = min((channels[n].index(key) + 1 for n in in_lexical), default=None)
        dense_rank = channels["dense"].index(key) + 1 if key in channels["dense"] else None
        names = tuple(in_lexical) + (("dense",) if dense_rank else ()) + (("date_anchor",) if key in channels.get("date_anchor", ()) else ())
        return Candidate(key, record, record_type_of(record), persona_id, event_group_id=self.groups.get(key), retrieval_channels=names,
                         lexical_rank=lexical_rank, dense_rank=dense_rank, dense_score=dense_score, fusion_score=score)

    def _rerank(self, c: RetrievalConfig, question: str, pool: List[Candidate], t: Dict[str, float]) -> Tuple[List[Candidate], bool]:
        """Reorder the first ``rerank_k``. If the call fails the fused order stands; a spent budget is never hidden."""
        head, tail = pool[:c.rerank_k], pool[c.rerank_k:]
        texts = [self.documents[x.record_id].text for x in head]
        key = hashlib.sha256("\x1f".join([c.rerank_model, question, *[x.key for x in head]]).encode("utf-8")).hexdigest()
        with self._cache_lock:
            cached = self.rerank_cache.get(key)
        if cached is None:
            try:
                result = self.client.rerank(c.rerank_model, question, texts)
            except (BudgetExceeded, OfflineCall):
                raise
            except (ApiError, OSError, ValueError):
                return pool, True
            t["rerank_api"] = result.latency_ms
            cached = result.scores
            with self._cache_lock:
                self.rerank_cache[key] = cached
        ranked = []
        for index, score in cached:
            head[index].rerank_score = score
            ranked.append(head[index])
        seen = {x.key for x in ranked}
        return ranked + [x for x in head if x.key not in seen] + tail, False


def _with_reserved(ranked: List[Candidate], expanded: List[Candidate], final_k: int, reserve: int) -> List[Candidate]:
    """Keep the last ``reserve`` of the final places for the best expanded evidence, when there is any."""
    if not reserve or not expanded:
        return ranked
    kept = [x for x in ranked if x.retrieval_origin == "direct"][:final_k - reserve]
    held = set(id(x) for x in kept)
    extra = [x for x in ranked if x.retrieval_origin == "expanded"][:reserve]
    held |= set(id(x) for x in extra)
    return kept + extra + [x for x in ranked if id(x) not in held]


def _unique(persona_id: str, records: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(evidence_key(persona_id, r) for r in records))
