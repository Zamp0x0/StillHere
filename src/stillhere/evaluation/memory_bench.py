"""LifeBench memory retrieval benchmark: fixed questions, fine-grained evidence, temporal and persona checks.

Gold comes from two official fields, and neither reaches the retriever:

  * ``evidence``: full copies of phone records. Each is identified by ``type`` and ``phone_id``, so it maps to one
    record, not one file: ``<persona>:<type>:<phone_id>``. An ``agent_chat`` record is a whole conversation, so all its
    messages share one key.
  * ``required_events_id``: ids of ``daily_event`` records, ``<persona>:daily_event:<id>``.

Strict scoring counts only ``evidence`` keys, and is the main result. Event-aware scoring also accepts the official
daily event of a gold record: the same event in its other representation, linked by id and never by text.

The retriever sees only the persona, the question text and the times. ``as_of`` and ``known_at`` are both the end of
``ask_time``, a date; time words inside the question are not interpreted.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from stillhere.contracts import EvidenceHit, HitKind, canonical_json
from stillhere.memory import MemoryAPI
from stillhere.memory.search import index_terms, query_units

SCORER_VERSION = "1"
DATASET_TZ = timezone(timedelta(hours=8))
PERSONA_PREFIX = "lifebench_"
DEV_PER_PERSONA = 10
TEST_PER_PERSONA = 40
TOP_KS = (5, 10)
# A question with several types is counted once, under the rarest kind of difficulty first.
PRIMARY_ORDER = ("Unanswerable", "Knowledge_update", "Temporal", "Conflict", "Causal", "Hidden_info",
                 "Pattern_recognition(Non-declarative)", "Multi_hop", "Single_hop")
DEFAULT_CONFIG = {"pool": 50, "top_k": list(TOP_KS), "rrf_k": 60, "include_ended": True,
                  "time_policy": "as_of = known_at = end of ask_time (+08:00); question text is not parsed for time",
                  "query": "the question text, verbatim", "fusion": "reciprocal rank fusion, ties by evidence key"}
_SOURCE_STEM = {"fitness": "fitness_health"}
# The fields the importer compares to drop an identical record: a gold record it dropped equals the kept one.
_TWIN_FIELDS = {"calendar": ("title", "start_time", "end_time", "description"), "note": ("title", "datetime", "content"),
                "photo": ("title", "datetime", "caption"), "push": ("title", "datetime", "content"),
                "call": ("contactName", "datetime", "datetime_end")}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def key_of(persona_id: str, kind: str, record_id: object) -> str:
    return f"{persona_id}:{kind}:{record_id}"


# ---------------------------------------------------------------- questions

@dataclass(frozen=True)
class Question:
    """One official question. The answer and score points are not loaded at all."""

    qa_id: str
    persona_id: str
    question: str
    ask_time: str
    types: Tuple[str, ...]
    primary_type: str
    gold: Tuple[str, ...]            # evidence keys in official order, without repeats
    gold_events: Tuple[str, ...]     # daily_event keys from required_events_id
    gold_links: Tuple[Tuple[str, Optional[str]], ...]   # (evidence key, its official daily_event key)

    @property
    def unanswerable(self) -> bool:
        return "Unanswerable" in self.types

    @property
    def question_sha256(self) -> str:
        return hashlib.sha256(self.question.encode("utf-8")).hexdigest()


def primary_type(types: Sequence[str]) -> str:
    return next((t for t in PRIMARY_ORDER if t in types), "Other")


def load_questions(data_root: Path, directory: str) -> List[Question]:
    """The official order gives the id: there is no id in the data."""
    persona_id = PERSONA_PREFIX + directory
    out = []
    for index, row in enumerate(_load(data_root / directory / "QA_all" / "QA.json")):
        links: Dict[str, Optional[str]] = {}
        for item in row["evidence"]:
            key = key_of(persona_id, item["type"], item["phone_id"])
            event = item.get("daily_event_id")
            links.setdefault(key, key_of(persona_id, "daily_event", event) if event else None)
        events = dict.fromkeys(key_of(persona_id, "daily_event", e) for e in row["required_events_id"])
        out.append(Question(qa_id=f"{directory}.qa.{index:04d}", persona_id=persona_id, question=row["question"],
                            ask_time=row["ask_time"], types=tuple(row["question_type"]),
                            primary_type=primary_type(row["question_type"]), gold=tuple(links), gold_events=tuple(events),
                            gold_links=tuple(links.items())))
    return out


def query_times(ask_time: str) -> Tuple[datetime, str]:
    """``ask_time`` is a date, so the question may come at any time of it: the end of that day."""
    day = datetime.strptime(ask_time, "%Y-%m-%d").replace(tzinfo=DATASET_TZ)
    return day + timedelta(days=1) - timedelta(microseconds=1), "ask_date_end_of_day"


# ---------------------------------------------------------------- fixed split

def _order(seed: int, qa_id: str) -> str:
    return hashlib.sha256(f"{seed}:{qa_id}".encode("utf-8")).hexdigest()


def _largest_remainder(weights: Mapping[str, int], total: int, *, least: int = 1) -> Dict[str, int]:
    """Whole shares of ``total`` proportional to ``weights``. With ``least``, a non-empty group keeps that many when
    there are enough places, taken from the largest share."""
    groups = sorted(k for k, v in weights.items() if v)
    whole = sum(weights[k] for k in groups)
    exact = {k: total * weights[k] / whole for k in groups}
    least = least if len(groups) * least <= total else 0
    share = {k: max(least, math.floor(exact[k])) for k in groups}
    while sum(share.values()) > total:
        largest = max(groups, key=lambda g: (share[g], g))
        share[largest] -= 1
    for k in sorted(groups, key=lambda g: (share[g] - exact[g], g)):
        if sum(share.values()) >= total:
            break
        share[k] += 1
    return share


def make_split(questions: Sequence[Question], seed: int, *, dev: int = DEV_PER_PERSONA,
               test: int = TEST_PER_PERSONA) -> Tuple[List[str], List[str], Dict[str, dict]]:
    """Stratified by persona and primary type, ordered by a seeded hash inside each stratum. Difficulty, the answer and
    whether the evidence maps play no part. Returns (dev ids, test ids, the quota of each persona)."""
    by_persona: Dict[str, List[Question]] = defaultdict(list)
    for q in questions:
        by_persona[q.persona_id].append(q)
    dev_ids: List[str] = []
    test_ids: List[str] = []
    quotas = {}
    for persona, items in sorted(by_persona.items()):
        strata: Dict[str, List[Question]] = defaultdict(list)
        for q in items:
            strata[q.primary_type].append(q)
        sizes = {k: len(v) for k, v in strata.items()}
        chosen = _largest_remainder(sizes, dev + test)
        # test is the measurement, so it keeps every type: each stratum leaves at least one question there
        room = {k: n - 1 for k, n in chosen.items()}
        dev_share = _largest_remainder(room, dev, least=0)
        for stratum in sorted(strata):
            ordered = sorted(strata[stratum], key=lambda q: _order(seed, q.qa_id))[:chosen[stratum]]
            cut = dev_share.get(stratum, 0)
            dev_ids += [q.qa_id for q in ordered[:cut]]
            test_ids += [q.qa_id for q in ordered[cut:]]
        quotas[persona] = {"chosen": chosen, "dev": dev_share, "available": sizes}
    return sorted(dev_ids), sorted(test_ids), quotas


def seal(manifest: dict) -> dict:
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return {**body, "manifest_sha256": hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()}


def check_split(manifest: dict, revision: str, questions: Mapping[str, Question]) -> None:
    """A split is used as it was frozen, or not at all: there is no silent re-sampling."""
    if seal(manifest)["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("the split manifest was edited after it was frozen")
    if manifest["dataset_revision"] != revision:
        raise ValueError(f"the split is for revision {manifest['dataset_revision']}, the data is {revision}")
    for qa_id, digest in manifest["question_sha256"].items():
        if qa_id not in questions or questions[qa_id].question_sha256 != digest:
            raise ValueError(f"{qa_id} differs from the frozen question")


# ---------------------------------------------------------------- gold mapping

class GoldMap:
    """Whether a gold key names a stored record, and which one."""

    def __init__(self, api: MemoryAPI, data_root: Path) -> None:
        self.api = api
        self.root = data_root
        self._raw: Dict[Tuple[str, str], List[dict]] = {}

    def _rows(self, persona_id: str, kind: str) -> List[dict]:
        key = (persona_id, kind)
        if key not in self._raw:
            self._raw[key] = _load(self.root / persona_id[len(PERSONA_PREFIX):] / "phone_data" / f"{kind}.json")
        return self._raw[key]

    def _stored(self, persona_id: str, kind: str, record_id: str) -> bool:
        directory = persona_id[len(PERSONA_PREFIX):]
        if kind == "sms":
            return self.api.get_message(persona_id, f"lb.{directory}.sms.{record_id}") is not None
        if kind == "agent_chat":
            return any(self.api.get_message(persona_id, f"lb.{directory}.agent_chat.{record_id}.{n}") is not None
                       for n in range(3))
        return self.api.get(persona_id, f"lb.{directory}.{kind}.{record_id}") is not None

    def resolve(self, key: str) -> Tuple[Optional[str], str]:
        """(the key to score against, status). A record the import dropped as identical maps to the one it kept."""
        persona_id, kind, record_id = key.split(":", 2)
        if self._stored(persona_id, kind, record_id):
            return key, "MAPPED"
        fields = _TWIN_FIELDS.get(kind)
        if fields is None:
            return None, "NOT_IN_DATABASE"
        rows = self._rows(persona_id, kind)
        row = next((r for r in rows if str(r["phone_id"]) == record_id), None)
        if row is None:
            return None, "NOT_IN_RAW_FILE"
        for other in rows:
            if str(other["phone_id"]) != record_id and all(other.get(f) == row.get(f) for f in fields) \
                    and self._stored(persona_id, kind, str(other["phone_id"])):
                return key_of(persona_id, kind, other["phone_id"]), "ALIASED_IDENTICAL_RECORD"
        return None, "NOT_IN_DATABASE"


# ---------------------------------------------------------------- retrieval

@dataclass
class Retrieved:
    key: str
    channel: str
    score: float
    hit: EvidenceHit


def evidence_key(hit: EvidenceHit) -> str:
    """The same key format as gold, from the record's own stored id. All messages of a chat share one key."""
    stored = hit.memory_id if hit.kind is HitKind.MEMORY else hit.message_id
    parts = stored.split(".")
    if len(parts) < 4 or parts[0] != "lb":
        raise ValueError(f"not a LifeBench record id: {stored}")
    return key_of(hit.persona_id, parts[2], parts[3])


def _unique(hits: Iterable[EvidenceHit], channel: str) -> List[Retrieved]:
    seen: Dict[str, Retrieved] = {}
    for hit in hits:
        key = evidence_key(hit)
        seen.setdefault(key, Retrieved(key, channel, hit.score, hit))
    return list(seen.values())


def search_facts(api: MemoryAPI, persona_id: str, question: str, when: datetime, config: Mapping) -> List[Retrieved]:
    return _unique(api.search(persona_id, question, as_of=when, known_at=when, k=config["pool"],
                              include_ended=config["include_ended"]), "facts")


def search_messages(api: MemoryAPI, persona_id: str, question: str, when: datetime, config: Mapping) -> List[Retrieved]:
    return _unique(api.search_messages(persona_id, question, as_of=when, known_at=when, k=config["pool"]), "messages")


def fuse(facts: Sequence[Retrieved], messages: Sequence[Retrieved], rrf_k: int) -> List[Retrieved]:
    """Reciprocal rank fusion: scores of the two channels are not comparable, their ranks are."""
    scored: Dict[str, Tuple[float, Retrieved]] = {}
    for channel in (facts, messages):
        for rank, item in enumerate(channel, 1):
            score = scored[item.key][0] if item.key in scored else 0.0
            scored[item.key] = (score + 1.0 / (rrf_k + rank), scored[item.key][1] if item.key in scored else item)
    ordered = sorted(scored.items(), key=lambda kv: (-kv[1][0], kv[0]))
    return [Retrieved(key, item.channel, score, item.hit) for key, (score, item) in ordered]


# ---------------------------------------------------------------- scoring

@dataclass
class Scored:
    """Scores of one ranked list for one question at each K, on ``gold`` keys already resolved."""

    hit: Dict[int, bool]
    coverage: Dict[int, float]
    full: Dict[int, bool]
    event_hit: Dict[int, bool]
    event_coverage: Dict[int, float]


def score_list(keys: Sequence[str], gold: Sequence[str], links: Mapping[str, Optional[str]], events: Sequence[str],
               top_ks: Sequence[int]) -> Scored:
    """``gold`` are the official evidence keys. Every one counts: the data lists no alternatives, so nothing is optional.

    Strict: a retrieved key equals a gold key. Event-aware: a gold record also counts through its official daily
    event, and a required event counts for the hit. ``keys`` must already be unique."""
    scored = Scored({}, {}, {}, {}, {})
    for k in top_ks:
        top = set(keys[:k])
        found = [g for g in gold if g in top]
        linked = [g for g in gold if g in top or links.get(g) in top]
        scored.hit[k] = bool(found)
        scored.coverage[k] = len(found) / len(gold) if gold else 0.0
        scored.full[k] = bool(gold) and len(found) == len(gold)
        scored.event_hit[k] = bool(linked) or bool(top & set(events))
        scored.event_coverage[k] = len(linked) / len(gold) if gold else 0.0
    return scored


def temporal_violations(items: Sequence[Retrieved], when: datetime) -> List[str]:
    """Anything returned must be known by then and have started by then. An ended event is allowed."""
    bad = []
    for item in items:
        h = item.hit
        started = h.timestamp if h.timestamp is not None else h.valid_from
        if (h.recorded_at is not None and h.recorded_at > when) or (started is not None and started > when):
            bad.append(item.key)
    return bad


def locator_error(item: Retrieved) -> Optional[str]:
    """The hit's source and locator must name the same record as its key."""
    h = item.hit
    _, kind, record_id = item.key.split(":", 2)
    directory = h.persona_id[len(PERSONA_PREFIX):]
    if h.locator is None:
        return "MISSING_LOCATOR"
    if h.source_id != f"lb.{directory}.{_SOURCE_STEM.get(kind, kind)}":
        return "INVALID_SOURCE"
    expected = {"daily_event": f"event_id={record_id}", "persona": f"field={record_id}"}.get(kind, f"phone_id={record_id}")
    return None if h.locator.split(";")[0] == expected else "WRONG_LOCATOR"


def _shares_words(question: str, text: str) -> bool:
    units = query_units(question)
    return bool(set(units.bigrams) & set(index_terms(text).bigrams))


def classify(api: MemoryAPI, q: Question, gold: Sequence[str], keys: Sequence[str], when: datetime, top: int) -> dict:
    """Why the gold records missing from the top ``top`` are missing. Runs after retrieval, for the report only."""
    shown = set(keys[:top])
    missing = [g for g in gold if g not in shown]
    reasons: Counter = Counter()
    for key in missing:
        persona_id, kind, record_id = key.split(":", 2)
        directory = persona_id[len(PERSONA_PREFIX):]
        if kind in ("sms", "agent_chat"):
            message = next((m for m in (api.get_message(persona_id, f"lb.{directory}.{kind}.{record_id}" +
                                                        ("" if kind == "sms" else f".{n}")) for n in range(3)) if m), None)
            text, known, started = message.content, message.recorded_at, message.timestamp
        else:
            record = api.get(persona_id, f"lb.{directory}.{kind}.{record_id}")
            text, known, started = record.statement, record.recorded_at, record.valid_from
        if known > when or (started is not None and started > when):
            reasons["TIME_FILTERED"] += 1
        elif key in keys:
            reasons["LOW_RANK"] += 1                 # in the pool, below the cut
        elif _shares_words(q.question, text):
            reasons["LOW_RANK"] += 1                 # matches, ranked below the pool
        else:
            reasons["KEYWORD_MISS"] += 1
    if not missing:
        label = "COMPLETE"
    elif len(shown & set(gold)):
        label = "MULTIHOP_INCOMPLETE"
    else:
        label = reasons.most_common(1)[0][0]
    return {"class": label, "missing_reasons": dict(reasons), "gold_count": len(gold), "missing_count": len(missing)}


def latency_stats(values: Sequence[float]) -> dict:
    if not values:
        return {"samples": 0}
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    return {"samples": len(ordered), "p50_ms": round(statistics.median(ordered), 2), "p95_ms": round(p95, 2),
            "mean_ms": round(statistics.fmean(ordered), 2), "max_ms": round(ordered[-1], 2)}


# ---------------------------------------------------------------- one run

CHANNELS = ("facts", "messages", "combined")


@dataclass
class Prediction:
    qa_id: str
    persona_id: str
    types: Tuple[str, ...]
    primary_type: str
    status: str
    as_of: str
    known_at: str
    include_ended: bool
    time_resolution_status: str
    gold_count: int
    gold_status: Dict[str, int]
    scores: Dict[str, dict] = field(default_factory=dict)
    retrieved: Dict[str, List[dict]] = field(default_factory=dict)
    latency_ms: Dict[str, float] = field(default_factory=dict)
    failure: Optional[dict] = None
    temporal_violations: int = 0
    persona_violations: int = 0
    locator_problems: Dict[str, int] = field(default_factory=dict)


def run_question(api: MemoryAPI, q: Question, gold_map: GoldMap, config: Mapping) -> Prediction:
    when, status = query_times(q.ask_time)
    top_ks = config["top_k"]
    pool_top = max(top_ks)

    started = time.perf_counter()
    facts = search_facts(api, q.persona_id, q.question, when, config)
    facts_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    messages = search_messages(api, q.persona_id, q.question, when, config)
    messages_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()                 # end to end: both channels again, then the merge
    combined = fuse(search_facts(api, q.persona_id, q.question, when, config),
                    search_messages(api, q.persona_id, q.question, when, config), config["rrf_k"])
    combined_ms = (time.perf_counter() - started) * 1000

    resolved = [gold_map.resolve(key) for key in q.gold]
    gold_status = Counter(s for _, s in resolved)
    gold = list(dict.fromkeys(k for k, _ in resolved if k))
    links = {new: dict(q.gold_links).get(old) for old, (new, _) in zip(q.gold, resolved) if new}
    if q.unanswerable:
        state = "UNANSWERABLE_OFFICIAL"
    elif not q.gold:
        state = "GOLD_MISSING"
    elif not gold:
        state = "GOLD_UNMAPPED"
    elif len(gold) < len(q.gold):
        state = "SCORABLE_PARTIAL_GOLD"
    else:
        state = "SCORABLE"
    p = Prediction(q.qa_id, q.persona_id, q.types, q.primary_type, state, when.isoformat(), when.isoformat(),
                   config["include_ended"], status, len(q.gold), dict(gold_status),
                   latency_ms={"facts": round(facts_ms, 3), "messages": round(messages_ms, 3),
                               "combined": round(combined_ms, 3)})
    for name, ranked in (("facts", facts), ("messages", messages), ("combined", combined)):
        top = ranked[:pool_top]
        p.temporal_violations += len(temporal_violations(ranked, when))
        p.persona_violations += sum(1 for r in ranked if r.hit.persona_id != q.persona_id
                                    or not r.key.startswith(q.persona_id + ":"))
        errors = Counter(e for e in (locator_error(r) for r in top) if e)
        for e, n in errors.items():
            p.locator_problems[e] = p.locator_problems.get(e, 0) + n
        p.retrieved[name] = [{"rank": i, "key": r.key, "channel": r.channel, "retriever": r.hit.retriever,
                              "score": round(r.score, 6), "source_id": r.hit.source_id, "locator": r.hit.locator,
                              "speaker_id": r.hit.speaker_id, "text": r.hit.text[:80]}
                             for i, r in enumerate(top, 1)]
        if gold:
            p.scores[name] = asdict(score_list([r.key for r in ranked], gold, links, q.gold_events, top_ks))
    if gold:
        p.failure = classify(api, q, gold, [r.key for r in combined], when, pool_top)
    return p


# ---------------------------------------------------------------- aggregation

def _mean(values: Sequence[float]) -> Optional[float]:
    return round(statistics.fmean(values), 4) if values else None


def _channel_metrics(rows: Sequence[Prediction], channel: str, top_ks: Sequence[int]) -> dict:
    out: dict = {"scorable_questions": len(rows)}
    for k in top_ks:
        s = [p.scores[channel] for p in rows]
        small = [p.scores[channel] for p in rows if p.gold_count <= k]
        out[f"hit@{k}"] = _mean([float(x["hit"][k]) for x in s])
        out[f"coverage@{k}"] = _mean([x["coverage"][k] for x in s])
        out[f"full_recall@{k}"] = _mean([float(x["full"][k]) for x in s])
        out[f"full_recall@{k}_gold_le_k"] = _mean([float(x["full"][k]) for x in small])
        out[f"questions_gold_le_{k}"] = len(small)
        out[f"event_hit@{k}"] = _mean([float(x["event_hit"][k]) for x in s])
        out[f"event_coverage@{k}"] = _mean([x["event_coverage"][k] for x in s])
    return out


def summarize(predictions: Sequence[Prediction], config: Mapping) -> dict:
    top_ks = config["top_k"]
    scorable = [p for p in predictions if p.scores]
    channels = {c: _channel_metrics(scorable, c, top_ks) for c in CHANNELS}
    groups: Dict[str, Dict[str, List[Prediction]]] = {"by_primary_type": defaultdict(list), "by_persona": defaultdict(list),
                                                      "by_gold_size": defaultdict(list)}
    for p in scorable:
        groups["by_primary_type"][p.primary_type].append(p)
        groups["by_persona"][p.persona_id].append(p)
        size = "1" if p.gold_count == 1 else "2-5" if p.gold_count <= 5 else "6-10" if p.gold_count <= 10 else "11+"
        groups["by_gold_size"][size].append(p)
    breakdown = {name: {k: {"questions": len(v), "combined": _channel_metrics(v, "combined", top_ks)}
                        for k, v in sorted(group.items())} for name, group in groups.items()}
    statuses = Counter(p.status for p in predictions)
    gold_status: Counter = Counter()
    for p in predictions:
        gold_status.update(p.gold_status)
    gold_total = sum(gold_status.values())
    locators: Counter = Counter()
    for p in predictions:
        locators.update(p.locator_problems)
    shown = sum(len(p.retrieved[c]) for p in predictions for c in CHANNELS)
    failures = Counter(p.failure["class"] for p in scorable)
    reasons: Counter = Counter()
    for p in scorable:
        reasons.update(p.failure["missing_reasons"])
    unanswerable = [p for p in predictions if p.status == "UNANSWERABLE_OFFICIAL"]
    return {
        "questions": len(predictions), "question_status": dict(sorted(statuses.items())),
        "mapping": {"questions_total": len(predictions), "questions_with_official_gold": sum(1 for p in predictions if p.gold_count),
                    "questions_scorable": len(scorable), "gold_records": gold_total, "gold_status": dict(sorted(gold_status.items())),
                    "gold_mapped_rate": round((gold_total - sum(v for k, v in gold_status.items() if k.startswith("NOT_")))
                                              / gold_total, 4) if gold_total else None},
        "channels": channels, "breakdown": breakdown,
        "temporal": {"eligibility_violations": sum(p.temporal_violations for p in predictions),
                     "questions_checked": len(predictions)},
        "persona_isolation": {"violations": sum(p.persona_violations for p in predictions)},
        "source_attribution": {"retrieved_checked": shown, "problems": dict(sorted(locators.items())),
                               "correct_locator_rate": round(1 - sum(locators.values()) / shown, 4) if shown else None},
        "failures": {"by_class_combined_top10": dict(failures.most_common()), "missing_gold_reasons": dict(reasons.most_common())},
        "unanswerable": {"count": len(unanswerable), "questions_with_results":
                         sum(1 for p in unanswerable if p.retrieved["combined"]),
                         "note": "the retriever has no abstention: it returns its best matches for these too"},
        "latency": {c: latency_stats([p.latency_ms[c] for p in predictions]) for c in CHANNELS},
    }


def run_benchmark(api: MemoryAPI, questions: Sequence[Question], gold_map: GoldMap, config: Mapping) -> List[Prediction]:
    return [run_question(api, q, gold_map, config) for q in questions]


def measure_latency(api: MemoryAPI, questions: Sequence[Question], config: Mapping, *, rounds: int) -> dict:
    """Steady-state timing. No gold is involved. The first questions warm the page cache and are not counted."""
    for q in questions[:10]:
        when, _ = query_times(q.ask_time)
        search_facts(api, q.persona_id, q.question, when, config)
        search_messages(api, q.persona_id, q.question, when, config)
    values: Dict[str, List[float]] = {c: [] for c in CHANNELS}
    for _ in range(rounds):
        for q in questions:
            when, _ = query_times(q.ask_time)
            started = time.perf_counter()
            facts = search_facts(api, q.persona_id, q.question, when, config)
            middle = time.perf_counter()
            messages = search_messages(api, q.persona_id, q.question, when, config)
            end = time.perf_counter()
            values["facts"].append((middle - started) * 1000)
            values["messages"].append((end - middle) * 1000)
            started = time.perf_counter()
            fuse(search_facts(api, q.persona_id, q.question, when, config),
                 search_messages(api, q.persona_id, q.question, when, config), config["rrf_k"])
            values["combined"].append((time.perf_counter() - started) * 1000)
    return {c: latency_stats(v) for c, v in values.items()}
