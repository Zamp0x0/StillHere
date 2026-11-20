"""EF1 retrieval channels. Each returns a ranked list of evidence-unit keys; every read uses ``as_of = known_at = ask time``.

Evidence unit: one source record of the persona's history, i.e. the messages sharing a source locator (one SMS, one calendar entry, one AI
dialogue with its turns). A message without a locator is its own unit (``message_id``). A memory record joins the units of the messages its
provenance cites; a memory with no message provenance is a unit of its own (``mem:<id>``).

Channels:
  msg_bm25    the store's FTS5 bm25 over raw messages (``search_messages``)                     - details the Writer did not keep
  mem_bm25    the store's FTS5 bm25 over memory statements (``MemoryAPI.search``)                - normalised facts
  mem_struct  B2's subject / date / alias channels on the BODY, capped (``retrieval_b2.retrieve``) - entity / slot / event state
  time_window messages said inside the question's event windows (``messages_between``), ranked by word overlap - time-selected evidence
  dense       cosine over a message embedding index masked by the store's time rule (``retrieval.dense``) - paraphrase recall
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timezone, timedelta
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from stillhere.contracts import MESSAGE_LOCATOR_PREFIX
from stillhere.memory import retrieval_b2 as b2

from .temporal import Window

CHANNELS = ("msg_bm25", "mem_bm25", "mem_struct", "time_window", "dense")


@dataclass
class ChannelResult:
    name: str
    keys: List[str] = field(default_factory=list)            # unit keys, best first (deduplicated)
    scores: Dict[str, float] = field(default_factory=dict)
    memories: Dict[str, List[str]] = field(default_factory=dict)   # unit key -> memory ids this channel found for it
    seconds: float = 0.0
    note: str = ""


def unit_of_message(m) -> str:
    return m.locator or m.message_id


def message_ids_of(record) -> List[str]:
    return [p.locator[len(MESSAGE_LOCATOR_PREFIX):] for p in (record.provenance or ()) if p.locator and p.locator.startswith(MESSAGE_LOCATOR_PREFIX)]


class UnitIndex:
    """message_id -> unit key for one persona (one cheap read of id + locator; reused for the whole day of questions)."""

    def __init__(self, api, persona_id: str) -> None:
        conn = api.store._conn
        self.unit = {mid: (loc or mid) for mid, loc in conn.execute(
            "SELECT message_id, locator FROM messages WHERE target_persona_id = ?", (persona_id,))}

    def units_of_memory(self, record) -> List[str]:
        out = [self.unit[mid] for mid in message_ids_of(record) if mid in self.unit]
        return list(dict.fromkeys(out)) or [f"mem:{record.memory_id}"]


def _add(res: ChannelResult, key: str, score: float, memory_id: Optional[str] = None) -> None:
    if key not in res.scores:
        res.keys.append(key)
        res.scores[key] = score
    if memory_id:
        res.memories.setdefault(key, [])
        if memory_id not in res.memories[key]:
            res.memories[key].append(memory_id)


def msg_bm25(api, persona_id: str, text: str, at: datetime, k: int, name: str = "msg_bm25") -> ChannelResult:
    t0 = time.perf_counter()
    res = ChannelResult(name)
    for h in api.search_messages(persona_id, text, as_of=at, known_at=at, k=k * 3):      # several turns of one dialogue may hit
        key = h.locator or h.message_id
        _add(res, key, float(h.score))
        if len(res.keys) >= k:
            break
    res.seconds = time.perf_counter() - t0
    return res


def mem_bm25(api, persona_id: str, text: str, at: datetime, k: int, units: UnitIndex) -> ChannelResult:
    t0 = time.perf_counter()
    res = ChannelResult("mem_bm25")
    for h in api.search(persona_id, text, as_of=at, known_at=at, k=k, include_ended=True):
        if not h.memory_id:
            continue
        r = api.get(persona_id, h.memory_id, known_at=at)
        if r is None:
            continue
        for key in units.units_of_memory(r):
            _add(res, key, float(h.score), r.memory_id)
    res.seconds = time.perf_counter() - t0
    return res


def mem_struct(api, persona_id: str, body: str, at: datetime, k: int, units: UnitIndex) -> ChannelResult:
    """Structured memory on the body: records of a person the question names (or of the relation word it uses: "我妹妹" -> her name),
    then the persona's own records of the slots the question cues (B2's alias table: ``做什么工作`` -> job / works_at ...), newest first,
    capped at ``k`` records. Selected by indexed SQL (subject, attribute, recorded time) and only the selected rows are read with
    ``known_at`` - never the whole table per ask time (the cost Recovery v3 removed). The records become candidates and features."""
    t0 = time.perf_counter()
    res = ChannelResult("mem_struct")
    conn = api.store._conn
    known = at.astimezone(timezone.utc).isoformat()
    names = {}
    rows = conn.execute("SELECT DISTINCT subject, '', '' FROM memories WHERE persona_id = ? AND subject >= 'person:' AND subject < 'person;' "
                        "AND recorded_at_utc <= ?", (persona_id, known)).fetchall()
    rows += conn.execute("SELECT subject, attribute, value FROM memories WHERE persona_id = ? AND subject = 'persona' AND attribute = 'relationship' "
                         "AND recorded_at_utc <= ? AND status != 'retracted'", (persona_id, known)).fetchall()
    for subj, attr, value in rows:
        if subj.startswith("person:"):
            names[subj.split(":", 1)[1]] = subj
        if attr == "relationship" and value and len(value.split()) >= 2:
            rel, name = value.split()[0], value.split()[-1]
            if rel and rel in body:
                names[name] = f"person:{name}"
    subjects = sorted({s for n, s in names.items() if n and n in body} | {f"person:{n}" for n, _s in names.items() if n and n in body})
    attrs = b2.attributes_for(body)
    picked: List[str] = []
    if subjects:
        marks = ",".join("?" * len(subjects))
        picked += [r[0] for r in conn.execute(
            f"SELECT memory_id FROM memories WHERE persona_id = ? AND subject IN ({marks}) AND recorded_at_utc <= ? "
            f"ORDER BY recorded_at_utc DESC, memory_id LIMIT ?", (persona_id, *subjects, known, k))]
    named = [n for n in names if n and n in body]
    for n in named:                                         # the stated relation itself ("本人的导师 吴明远"): indexed (persona, subject, attribute)
        if len(picked) >= k:
            break
        picked += [r[0] for r in conn.execute(
            "SELECT memory_id FROM memories WHERE persona_id = ? AND subject = 'persona' AND attribute = 'relationship' AND recorded_at_utc <= ? "
            "AND value LIKE ? ORDER BY recorded_at_utc DESC, memory_id LIMIT ?", (persona_id, known, f"%{n}", k - len(picked)))]
    slot_attrs = [a for a in attrs if a != "event"]
    if slot_attrs and len(picked) < k:
        marks = ",".join("?" * len(slot_attrs))
        picked += [r[0] for r in conn.execute(
            f"SELECT memory_id FROM memories WHERE persona_id = ? AND subject = 'persona' AND attribute IN ({marks}) AND recorded_at_utc <= ? "
            f"ORDER BY recorded_at_utc DESC, memory_id LIMIT ?", (persona_id, *slot_attrs, known, k - len(picked)))]
    rank = 0
    for mid in dict.fromkeys(picked):
        r = api.get(persona_id, mid, known_at=at)
        if r is None or r.recorded_at > at or getattr(r.status, "value", r.status) == "retracted":
            continue
        rank += 1
        for key in units.units_of_memory(r):
            _add(res, key, 1.0 / rank, r.memory_id)
    res.note = f"subjects={len(subjects)} attrs={','.join(slot_attrs) or '-'} records={rank}"
    res.seconds = time.perf_counter() - t0
    return res


_BIGRAM = re.compile(r"[一-鿿A-Za-z0-9]")
_STOP = set("的了是在和与及或吗呢吧啊呀么嘛哦这那哪谁什怎几个有没不也就都很还又把被给我你他她它们")


def _bigrams(text: str) -> set:
    s = "".join(ch for ch in (text or "") if _BIGRAM.match(ch))
    return {s[i:i + 2] for i in range(len(s) - 1) if not (s[i] in _STOP and s[i + 1] in _STOP)}


def time_window(api, persona_id: str, windows: Sequence[Window], body: str, at: datetime, k: int, tz=timezone(timedelta(hours=8)),
                per_window_limit: int = 2000) -> ChannelResult:
    """Messages said inside each window (clipped to the ask time by the store), ranked by window weight, then word overlap with the
    question, then time. Every unit of a short window is kept; a long range keeps its ``k`` best."""
    t0 = time.perf_counter()
    res = ChannelResult("time_window")
    q = _bigrams(body)
    scored: Dict[str, float] = {}
    for w in windows:
        start = datetime.combine(w.start, dtime.min, tzinfo=tz)
        end = datetime.combine(w.end, dtime.min, tzinfo=tz)
        for m in api.store.messages_between(persona_id, start, end, as_of=at, known_at=at, limit=per_window_limit):
            key = unit_of_message(m)
            overlap = len(q & _bigrams(m.content)) / max(1, len(q))
            s = w.weight * (1.0 + overlap)
            if s > scored.get(key, 0.0):
                scored[key] = s
    for key, s in sorted(scored.items(), key=lambda x: (-x[1], x[0]))[:k]:
        _add(res, key, s)
    res.note = f"windows={len(windows)} units_in_windows={len(scored)}"
    res.seconds = time.perf_counter() - t0
    return res


class DenseChannel:
    """Cosine search over a message embedding index (``retrieval.dense.DenseIndex`` + ``ScopeTable`` time mask). The query vector comes
    from the embedding cache, else from ``embed_query`` (an API call; may be None for an offline run - the channel is then skipped)."""

    def __init__(self, index, scope, cache, model: str, dimension: int, embed_query: Optional[Callable[[str], Optional[object]]] = None,
                 unit_of: Optional[Mapping[str, Mapping[str, str]]] = None) -> None:
        self.index, self.scope, self.cache, self.model, self.dimension = index, scope, cache, model, dimension
        self.embed_query = embed_query
        self.unit_of = unit_of or {}
        self.stats = {"cache_hits": 0, "api_calls": 0, "skipped": 0}

    def query_vector(self, text: str):
        from stillhere.retrieval.dense import EmbeddingCache
        key = EmbeddingCache.key(self.model, self.dimension, "query", text)
        v = self.cache.get(key) if self.cache is not None else None
        if v is not None:
            self.stats["cache_hits"] += 1
            return v
        if self.embed_query is None:
            return None
        v = self.embed_query(text)
        if v is not None:
            self.stats["api_calls"] += 1
            if self.cache is not None:
                self.cache.put_many([(key, v)])
        return v

    def __call__(self, persona_id: str, text: str, at: datetime, k: int) -> ChannelResult:
        t0 = time.perf_counter()
        res = ChannelResult("dense")
        if persona_id not in getattr(self.index, "vectors", {}):
            res.note = "no index for persona"
            return res
        v = self.query_vector(text)
        if v is None:
            self.stats["skipped"] += 1
            res.note = "no query vector (offline): channel skipped"
            return res
        mask = self.scope.mask(persona_id, at, at, include_ended=True)
        units = self.unit_of.get(persona_id, {})
        for rid, s in self.index.search(persona_id, v, mask, k * 3):
            _add(res, units.get(rid, rid), s)
            if len(res.keys) >= k:
                break
        res.seconds = time.perf_counter() - t0
        return res
