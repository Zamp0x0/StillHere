"""Offline diagnosis of the V1 retriever: how deep gold sits, what repeats in the top K, where the time goes.

Nothing here changes retrieval. Gold is read only after a ranked list exists, to measure it; the search functions never
see it. ``deep_ranking`` repeats the store's own SQL to rank every match, because ``search`` loads each record it
returns; ``check_same_ranking`` asserts the two orders agree. The simulations (``simulate_*``) and the oracle ceiling
are estimates for planning, measured on non-test questions only, and are not a retriever.
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from stillhere.memory import MemoryAPI
from stillhere.memory.search import BIGRAM_WEIGHT, UNIGRAM_WEIGHT, index_terms, match_expression, normalize, query_units
from stillhere.memory.sqlite_store import _utc

from . import memory_bench as mb

CANDIDATE_KS = (10, 20, 50, 100)
RANK_BUCKETS = ((1, 10, "1-10"), (11, 20, "11-20"), (21, 50, "21-50"), (51, 100, "51-100"), (101, 500, "101-500"),
                (501, 10 ** 9, "501+"))


# ---------------------------------------------------------------- frozen files

def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_digest(manifest: Mapping) -> str:
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def verify_baseline(manifest: Mapping, directory: Path) -> List[str]:
    """Problems found comparing a baseline manifest with the files; empty when everything is byte-identical."""
    problems = []
    if manifest_digest(manifest) != manifest.get("manifest_sha256"):
        problems.append("the manifest was edited")
    for name, digest in manifest["files"].items():
        path = Path(directory) / name
        if not path.is_file():
            problems.append(f"{name} is missing")
        elif sha256_file(path) != digest:
            problems.append(f"{name} changed")
    return problems


# ---------------------------------------------------------------- deep ranking

@dataclass
class Ranking:
    """Every match of a query, best first, one entry per evidence key."""

    keys: List[str]
    matched_rows: int


def key_of_stored(persona_id: str, stored: str) -> str:
    parts = stored.split(".")
    return mb.key_of(persona_id, parts[2], parts[3])


def _unique_keys(persona_id: str, rows: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(key_of_stored(persona_id, r) for r in rows))


def fact_sql(*, temporal: bool = True, select: Optional[str] = None) -> str:
    """The FTS path of ``SQLiteMemoryStore.search`` with ``include_ended=True``, without its LIMIT."""
    time_part = ("AND m.recorded_at_utc <= ? AND (m.status_changed_at IS NULL OR m.status_changed_at > ?) "
                 "AND (m.valid_from_utc IS NULL OR m.valid_from_utc <= ?) ") if temporal else ""
    return (f"SELECT {select or f'm.memory_id, bm25(memory_fts, 0.0, {BIGRAM_WEIGHT!r}, {UNIGRAM_WEIGHT!r}) AS rank'} "
            f"FROM memory_fts JOIN memories AS m ON m.memory_id = memory_fts.memory_id "
            f"WHERE memory_fts MATCH ? AND m.persona_id = ? {time_part}"
            + ("" if select else "ORDER BY rank, m.recorded_at_utc DESC, m.memory_id"))


def message_sql(*, temporal: bool = True, select: Optional[str] = None) -> str:
    """The FTS path of ``SQLiteMemoryStore.search_messages``, without its LIMIT."""
    time_part = "AND m.timestamp_utc <= ? AND m.recorded_at_utc <= ? " if temporal else ""
    return (f"SELECT {select or f'm.message_id, bm25(messages_fts, 0.0, {BIGRAM_WEIGHT!r}, {UNIGRAM_WEIGHT!r}) AS rank'} "
            f"FROM messages_fts JOIN messages AS m ON m.message_id = messages_fts.message_id "
            f"WHERE messages_fts MATCH ? AND m.target_persona_id = ? {time_part}"
            + ("" if select else "ORDER BY rank, m.timestamp_utc DESC, m.message_id"))


def deep_ranking(conn, persona_id: str, question: str, when: Optional[datetime], channel: str) -> Ranking:
    """All matches. ``when=None`` drops the time filter, to see what the filter removes."""
    units = query_units(question)
    if not units:
        return Ranking([], 0)
    temporal = when is not None
    args: tuple = (match_expression(units), persona_id)
    if channel == "facts":
        sql = fact_sql(temporal=temporal)
        args += (_utc(when), _utc(when), _utc(when)) if temporal else ()
    else:
        sql = message_sql(temporal=temporal)
        args += (_utc(when), _utc(when)) if temporal else ()
    rows = [r[0] for r in conn.execute(sql, args).fetchall()]
    return Ranking(_unique_keys(persona_id, rows), len(rows))


def check_same_ranking(api: MemoryAPI, q: mb.Question, when: datetime, deep: Mapping[str, Ranking], k: int = 100) -> None:
    """The deep ranking must agree with what the real API returns, or this diagnosis measures the wrong thing."""
    config = {"pool": k, "include_ended": True}
    real = {"facts": mb.search_facts(api, q.persona_id, q.question, when, config),
            "messages": mb.search_messages(api, q.persona_id, q.question, when, config)}
    for channel, items in real.items():
        keys = [r.key for r in items]
        if deep[channel].keys[:len(keys)] != keys:
            raise AssertionError(f"{q.qa_id}: deep {channel} ranking differs from the API")


# ---------------------------------------------------------------- candidate recall

def hit_coverage(candidates: Sequence[str], gold: Sequence[str]) -> Tuple[bool, float]:
    found = sum(1 for g in gold if g in set(candidates))
    return found > 0, found / len(gold) if gold else 0.0


def rank_of(keys: Sequence[str], key: str) -> Optional[int]:
    try:
        return keys.index(key) + 1
    except ValueError:
        return None


def bucket(rank: Optional[int]) -> str:
    if rank is None:
        return "no_match"
    return next(name for low, high, name in RANK_BUCKETS if low <= rank <= high)


def channel_of(key: str) -> str:
    return "messages" if key.split(":")[1] in ("sms", "agent_chat") else "facts"


@dataclass
class QuestionDiagnosis:
    qa_id: str
    primary_type: str
    persona_id: str
    source_set: str
    question_length: int
    gold_count: int
    candidates: Dict[str, Dict[int, Tuple[bool, float]]] = field(default_factory=dict)
    gold: List[dict] = field(default_factory=list)
    union_sizes: Dict[int, int] = field(default_factory=dict)
    final_top10: Tuple[bool, float] = (False, 0.0)
    oracle: Dict[int, float] = field(default_factory=dict)
    simulated: Dict[str, Tuple[bool, float]] = field(default_factory=dict)
    pack: Tuple[int, float] = (0, 0.0)


def word_overlap(question: str, text: str) -> str:
    units, terms = query_units(question), index_terms(text)
    if set(units.bigrams) & set(terms.bigrams):
        return "bigram"
    return "unigram_only" if set(units.unigrams) & set(terms.unigrams) else "none"


# ---------------------------------------------------------------- event groups

_MESSAGE_GROUP = re.compile(r"daily_event:event_id=([^;]+)")


def group_id(persona_id: str, event_id: object) -> str:
    return f"{persona_id}:event:{event_id}"


def event_groups(conn, persona_id: str) -> Dict[str, str]:
    """Evidence key -> event group, from the store's own provenance and locators. No QA field is read."""
    groups: Dict[str, str] = {}
    prefix = "lb." + persona_id[len(mb.PERSONA_PREFIX):] + "."
    for memory_id, locator in conn.execute(
            "SELECT p.memory_id, p.locator FROM provenance AS p JOIN memories AS m ON m.memory_id = p.memory_id "
            "WHERE m.persona_id = ? AND p.source_id = ?", (persona_id, prefix + "daily_event")):
        groups.setdefault(key_of_stored(persona_id, memory_id), group_id(persona_id, locator.split("=", 1)[1]))
    for message_id, locator in conn.execute("SELECT message_id, locator FROM messages WHERE target_persona_id = ?", (persona_id,)):
        found = _MESSAGE_GROUP.search(locator or "")
        if found:
            groups.setdefault(key_of_stored(persona_id, message_id), group_id(persona_id, found.group(1)))
    return groups


def duplicate_slots(keys: Sequence[str], groups: Mapping[str, str]) -> int:
    """Slots beyond the first of each event group. A record with no group never repeats."""
    seen: set = set()
    repeats = 0
    for key in keys:
        group = groups.get(key)
        if group is None:
            continue
        repeats += group in seen
        seen.add(group)
    return repeats


def simulate_cap(ranked: Sequence[str], groups: Mapping[str, str], k: int, per_group: int = 1) -> List[str]:
    """The ranked list with at most ``per_group`` records of each event group."""
    used: Counter = Counter()
    out = []
    for key in ranked:
        group = groups.get(key, key)
        if used[group] < per_group:
            used[group] += 1
            out.append(key)
        if len(out) == k:
            break
    return out


def simulate_group_fusion(facts: Sequence[str], messages: Sequence[str], groups: Mapping[str, str], rrf_k: int) -> List[List[str]]:
    """RRF over event groups, so that the two channels can agree. Returns groups best first, members in rank order."""
    scores: Dict[str, float] = defaultdict(float)
    members: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for channel in (facts, messages):
        seen: set = set()
        for rank, key in enumerate(channel, 1):
            group = groups.get(key, key)
            members[group].append((rank, key))
            if group not in seen:
                seen.add(group)
                scores[group] += 1.0 / (rrf_k + rank)
    ordered = sorted(scores, key=lambda g: (-scores[g], g))
    return [[key for _, key in sorted(members[g])] for g in ordered]


def diagnose_question(api: MemoryAPI, conn, q: mb.Question, source_set: str, gold_map: mb.GoldMap,
                      groups: Mapping[str, str]) -> Optional[QuestionDiagnosis]:
    """Rank first, then look at gold. Returns None for a question with no gold (it is counted by the caller)."""
    when, _ = mb.query_times(q.ask_time)
    deep = {c: deep_ranking(conn, q.persona_id, q.question, when, c) for c in ("facts", "messages")}
    check_same_ranking(api, q, when, deep)
    gold = list(dict.fromkeys(k for k, _ in (gold_map.resolve(key) for key in q.gold) if k))
    if not gold:
        return None
    open_time = {c: deep_ranking(conn, q.persona_id, q.question, None, c) for c in ("facts", "messages")}
    d = QuestionDiagnosis(q.qa_id, q.primary_type, q.persona_id, source_set, len(q.question), len(gold))
    for k in CANDIDATE_KS:
        f, m = deep["facts"].keys[:k], deep["messages"].keys[:k]
        union = list(dict.fromkeys(f + m))
        d.union_sizes[k] = len(union)
        d.candidates.setdefault("facts", {})[k] = hit_coverage(f, gold)
        d.candidates.setdefault("messages", {})[k] = hit_coverage(m, gold)
        d.candidates.setdefault("union", {})[k] = hit_coverage(union, gold)
        d.oracle[k] = min(sum(1 for g in gold if g in set(union)), 10) / len(gold)
    def item(key: str, channel: str):
        return mb.Retrieved(key, channel, 0.0, None)
    fused = [r.key for r in mb.fuse([item(k, "facts") for k in deep["facts"].keys[:50]],
                                    [item(k, "messages") for k in deep["messages"].keys[:50]], 60)]
    d.final_top10 = hit_coverage(fused[:10], gold)
    for key in gold:
        channel = channel_of(key)
        rank = rank_of(deep[channel].keys, key)
        d.gold.append({"key": key, "channel": channel, "rank": rank, "bucket": bucket(rank),
                       "in_final_top10": key in fused[:10], "time_excluded": rank is None and key in open_time[channel].keys,
                       "overlap": word_overlap(q.question, _text(api, key)),
                       "group": groups.get(key), "group_top10": _crowded(key, deep, groups)})
    f100, m100 = deep["facts"].keys[:100], deep["messages"].keys[:100]
    d.simulated["cap_1_per_event"] = hit_coverage(simulate_cap(_fused(f100, m100), groups, 10), gold)
    d.simulated["cap_2_per_event"] = hit_coverage(simulate_cap(_fused(f100, m100), groups, 10, per_group=2), gold)
    grouped = simulate_group_fusion(f100, m100, groups, 60)
    d.simulated["group_rrf_best_member"] = hit_coverage([g[0] for g in grouped[:10]], gold)
    pack = [key for g in grouped[:5] for key in g]
    d.pack = (len(pack), hit_coverage(pack, gold)[1])
    return d


def _fused(facts: Sequence[str], messages: Sequence[str]) -> List[str]:
    def item(key: str, channel: str):
        return mb.Retrieved(key, channel, 0.0, None)
    return [r.key for r in mb.fuse([item(k, "facts") for k in facts], [item(k, "messages") for k in messages], 60)]


def _crowded(key: str, deep: Mapping[str, Ranking], groups: Mapping[str, str]) -> bool:
    """A sibling of this record, from the same event, is in the channel top 10 while it is not."""
    group = groups.get(key)
    channel = channel_of(key)
    top = deep[channel].keys[:10]
    return group is not None and key not in top and any(groups.get(k) == group for k in top)


def _text(api: MemoryAPI, key: str) -> str:
    persona_id, kind, record_id = key.split(":", 2)
    directory = persona_id[len(mb.PERSONA_PREFIX):]
    if kind == "sms":
        return api.get_message(persona_id, f"lb.{directory}.sms.{record_id}").content
    if kind == "agent_chat":
        return " ".join(m.content for m in (api.get_message(persona_id, f"lb.{directory}.agent_chat.{record_id}.{n}")
                                            for n in range(40)) if m)
    return api.get(persona_id, f"lb.{directory}.{kind}.{record_id}").statement


def summarize_candidates(rows: Sequence[QuestionDiagnosis]) -> dict:
    def table(items: Sequence[QuestionDiagnosis]) -> dict:
        out: dict = {"questions": len(items)}
        for name in ("facts", "messages", "union"):
            for k in CANDIDATE_KS:
                out[f"{name}_hit@{k}"] = round(statistics.fmean(d.candidates[name][k][0] for d in items), 4)
                out[f"{name}_coverage@{k}"] = round(statistics.fmean(d.candidates[name][k][1] for d in items), 4)
        for k in CANDIDATE_KS:
            out[f"union_size@{k}"] = round(statistics.fmean(d.union_sizes[k] for d in items), 1)
            out[f"oracle_top10_from_union@{k}"] = round(statistics.fmean(d.oracle[k] for d in items), 4)
        out["final_v1_hit@10"] = round(statistics.fmean(d.final_top10[0] for d in items), 4)
        out["final_v1_coverage@10"] = round(statistics.fmean(d.final_top10[1] for d in items), 4)
        return out

    gold = [g for d in rows for g in d.gold]
    missed = [g for g in gold if not g["in_final_top10"]]
    groups: Dict[str, Dict[str, List[QuestionDiagnosis]]] = {"by_primary_type": defaultdict(list), "by_set": defaultdict(list),
                                                             "by_persona": defaultdict(list), "by_question_length": defaultdict(list)}
    for d in rows:
        groups["by_primary_type"][d.primary_type].append(d)
        groups["by_set"][d.source_set].append(d)
        groups["by_persona"][d.persona_id].append(d)
        groups["by_question_length"]["<=30" if d.question_length <= 30 else "31-60" if d.question_length <= 60 else ">60"].append(d)
    channel_buckets = {c: dict(Counter(g["bucket"] for g in gold if g["channel"] == c)) for c in ("facts", "messages")}
    return {
        "questions_scored": len(rows), "gold_records": len(gold), "all": table(rows),
        "breakdown": {name: {k: table(v) for k, v in sorted(g.items())} for name, g in groups.items()},
        "gold_rank_buckets": dict(Counter(g["bucket"] for g in gold)), "gold_rank_buckets_by_channel": channel_buckets,
        "missed_from_final_top10": len(missed), "missed_rank_buckets": dict(Counter(g["bucket"] for g in missed)),
        "missed_in_top100_of_channel": sum(1 for g in missed if g["rank"] is not None and g["rank"] <= 100),
        "missed_below_top100": sum(1 for g in missed if g["rank"] is not None and g["rank"] > 100),
        "missed_no_fts_match": sum(1 for g in missed if g["rank"] is None),
        "time_excluded_gold": sum(1 for g in gold if g["time_excluded"]),
        "gold_overlap": dict(Counter(g["overlap"] for g in gold)),
        "missed_overlap": dict(Counter(g["overlap"] for g in missed)),
        "median_gold_rank": statistics.median([g["rank"] for g in gold if g["rank"]]) if any(g["rank"] for g in gold) else None,
    }


def summarize_events(rows: Sequence[QuestionDiagnosis]) -> dict:
    n = len(rows)
    gold = [g for d in rows for g in d.gold]
    per_question = [len({g["group"] for g in d.gold if g["group"]}) for d in rows]
    return {
        "simulated_top10": {name: {"hit": round(statistics.fmean(d.simulated[name][0] for d in rows), 4),
                                   "coverage": round(statistics.fmean(d.simulated[name][1] for d in rows), 4)}
                            for name in rows[0].simulated},
        "v1_top10": {"hit": round(statistics.fmean(d.final_top10[0] for d in rows), 4),
                     "coverage": round(statistics.fmean(d.final_top10[1] for d in rows), 4)},
        "pack_top5_groups": {"mean_records": round(statistics.fmean(d.pack[0] for d in rows), 1),
                             "coverage": round(statistics.fmean(d.pack[1] for d in rows), 4)},
        "gold_with_group": round(sum(1 for g in gold if g["group"]) / len(gold), 4),
        "gold_records_per_event_group": round(sum(1 for g in gold if g["group"]) / max(sum(per_question), 1), 3),
        "mean_gold_groups_per_question": round(statistics.fmean(per_question), 2),
        "gold_crowded_out_by_sibling": sum(1 for g in gold if g["group_top10"]), "questions": n,
    }


# ---------------------------------------------------------------- timing

def timed_facts(api: MemoryAPI, persona_id: str, question: str, when: datetime, k: int) -> Tuple[List[mb.Retrieved], Dict[str, float]]:
    """``api.search`` with include_ended, in stages. It runs the store's statements and helpers, in the same order."""
    store = api.store
    t: Dict[str, float] = {}
    mark = time.perf_counter
    t0 = mark()
    normalize(question)
    t1 = mark()
    units = query_units(question)
    t2 = mark()
    expression = match_expression(units)
    utc = _utc(when)
    t3 = mark()
    with store._read() as conn:
        cursor = conn.execute(fact_sql() + " LIMIT ?", (expression, persona_id, utc, utc, utc, k))
        first = cursor.fetchone()
        t4 = mark()
        rows = ([first] + cursor.fetchall()) if first else []
        t5 = mark()
        records = [(store._load(conn, r[0], utc), -r[1]) for r in rows]
        t6 = mark()
    hits = [api._memory_hit(record, float(score), store.retriever) for record, score in records]
    t7 = mark()
    items = mb._unique(hits, "facts")
    t8 = mark()
    t.update(normalize=(t1 - t0), tokenize=(t2 - t1), build_query=(t3 - t2), sql_first_row=(t4 - t3), row_fetch=(t5 - t4),
             load_records=(t6 - t5), to_hits=(t7 - t6), evidence_keys=(t8 - t7), total=(t8 - t0))
    return items, {name: value * 1000 for name, value in t.items()}


def timed_messages(api: MemoryAPI, persona_id: str, question: str, when: datetime, k: int) -> Tuple[List[mb.Retrieved], Dict[str, float]]:
    from stillhere.contracts import EvidenceHit, HitKind
    store = api.store
    mark = time.perf_counter
    t0 = mark()
    normalize(question)
    t1 = mark()
    units = query_units(question)
    t2 = mark()
    expression = match_expression(units)
    utc = _utc(when)
    t3 = mark()
    select = f"m.*, bm25(messages_fts, 0.0, {BIGRAM_WEIGHT!r}, {UNIGRAM_WEIGHT!r}) AS rank"
    sql = message_sql(select=select) + " ORDER BY rank, m.timestamp_utc DESC, m.message_id LIMIT ?"
    with store._read() as conn:
        cursor = conn.execute(sql, (expression, persona_id, utc, utc, k))
        first = cursor.fetchone()
        t4 = mark()
        rows = ([first] + cursor.fetchall()) if first else []
        t5 = mark()
        messages = [(store._message(r), -r["rank"]) for r in rows]
        t6 = mark()
    hits = [EvidenceHit(kind=HitKind.MESSAGE, persona_id=persona_id, text=m.content, source_id=m.source_id, score=float(s),
                        retriever="messages_fts", message_id=m.message_id, speaker_id=m.speaker_id,
                        conversation_id=m.conversation_id, timestamp=m.timestamp, recorded_at=m.recorded_at, locator=m.locator)
            for m, s in messages]
    t7 = mark()
    items = mb._unique(hits, "messages")
    t8 = mark()
    t = dict(normalize=(t1 - t0), tokenize=(t2 - t1), build_query=(t3 - t2), sql_first_row=(t4 - t3), row_fetch=(t5 - t4),
             load_records=(t6 - t5), to_hits=(t7 - t6), evidence_keys=(t8 - t7), total=(t8 - t0))
    return items, {name: value * 1000 for name, value in t.items()}


def stage_stats(rows: Sequence[Mapping[str, float]]) -> dict:
    return {name: mb.latency_stats([r[name] for r in rows]) for name in rows[0]} if rows else {}


def measure_stages(api: MemoryAPI, questions: Sequence[mb.Question], *, ks: Sequence[int] = CANDIDATE_KS, rounds: int = 2) -> dict:
    conn = api.store._conn
    out: dict = {}
    for q in questions[:5]:                       # warm the page cache; not counted
        when, _ = mb.query_times(q.ask_time)
        timed_facts(api, q.persona_id, q.question, when, 50)
        timed_messages(api, q.persona_id, q.question, when, 50)
    for k in ks:
        facts_t, messages_t, fusion_t, totals = [], [], [], []
        config = {"pool": k, "include_ended": True}
        started = time.perf_counter()
        for _ in range(rounds):
            for q in questions:
                when, _ = mb.query_times(q.ask_time)
                f, ft = timed_facts(api, q.persona_id, q.question, when, k)
                m, mt = timed_messages(api, q.persona_id, q.question, when, k)
                t0 = time.perf_counter()
                mb.fuse(f, m, 60)
                fusion = (time.perf_counter() - t0) * 1000
                facts_t.append(ft)
                messages_t.append(mt)
                fusion_t.append({"fusion": fusion})
                totals.append({"total": ft["total"] + mt["total"] + fusion})
        elapsed = time.perf_counter() - started
        api_times = []
        for q in questions:
            when, _ = mb.query_times(q.ask_time)
            t0 = time.perf_counter()
            mb.search_facts(api, q.persona_id, q.question, when, config)
            mb.search_messages(api, q.persona_id, q.question, when, config)
            api_times.append((time.perf_counter() - t0) * 1000)
        out[str(k)] = {"facts": stage_stats(facts_t), "messages": stage_stats(messages_t), "fusion": stage_stats(fusion_t)["fusion"],
                       "total": stage_stats(totals)["total"], "api_both_channels_check": mb.latency_stats(api_times),
                       "queries_per_second": round(len(totals) / elapsed, 2), "samples": len(totals)}
    return out


def query_shape(api: MemoryAPI, questions: Sequence[mb.Question]) -> dict:
    """What each query asks SQLite to do: its OR terms, matched rows, and the cost of the time filter."""
    conn = api.store._conn
    rows = []
    for q in questions:
        when, _ = mb.query_times(q.ask_time)
        units = query_units(q.question)
        expression = match_expression(units)
        utc = _utc(when)
        t0 = time.perf_counter()
        match_only = conn.execute("SELECT count(*) FROM memory_fts WHERE memory_fts MATCH ?", (expression,)).fetchone()[0]
        t1 = time.perf_counter()
        persona = conn.execute(fact_sql(temporal=False, select="count(*)"), (expression, q.persona_id)).fetchone()[0]
        t2 = time.perf_counter()
        eligible = conn.execute(fact_sql(select="count(*)"), (expression, q.persona_id, utc, utc, utc)).fetchone()[0]
        t3 = time.perf_counter()
        ranked = conn.execute(fact_sql() + " LIMIT 10", (expression, q.persona_id, utc, utc, utc)).fetchall()
        t4 = time.perf_counter()
        rows.append({"length": len(q.question), "bigrams": len(units.bigrams), "unigrams": len(units.unigrams),
                     "matched_all_personas": match_only, "matched_persona": persona, "eligible": eligible,
                     "match_ms": (t1 - t0) * 1000, "persona_join_ms": (t2 - t1) * 1000, "with_time_filter_ms": (t3 - t2) * 1000,
                     "ranked_limit10_ms": (t4 - t3) * 1000, "rows": len(ranked)})
    numeric = [k for k in rows[0] if k != "rows"]
    return {"samples": len(rows), "median": {k: round(statistics.median(r[k] for r in rows), 2) for k in numeric},
            "p95": {k: round(sorted(r[k] for r in rows)[int(0.95 * (len(rows) - 1))], 2) for k in numeric}}


def query_plans(conn) -> dict:
    expression = '"测试"'
    utc = "2025-12-31T00:00:00.000000Z"
    facts = conn.execute("EXPLAIN QUERY PLAN " + fact_sql() + " LIMIT 10", (expression, "p", utc, utc, utc)).fetchall()
    messages = conn.execute("EXPLAIN QUERY PLAN " + message_sql(select="m.message_id, 0 AS rank") + " ORDER BY rank LIMIT 10",
                            (expression, "p", utc, utc)).fetchall()
    return {"facts": [r[3] for r in facts], "messages": [r[3] for r in messages]}
