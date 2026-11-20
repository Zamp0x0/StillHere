"""Replay and scoring for the M2 Writer and Observations against the independent gold of ``writer_dataset``.

``replay`` feeds conversations to the Writer in the order a live system learns them, with the store's clock at each group's
knowledge time, and refreshes observations after every group. ``reference_complete`` answers the extraction step from the gold
(a deterministic stand-in for the model, used to test everything after extraction); a real run passes the DeepSeek call instead.
Scores compare what the store holds with the gold specification, never with the Writer's own output.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from stillhere.memory import MemoryAPI
from stillhere.observation.engine import ObservationEngine, affected_keys, dependency_audit
from stillhere.writer.extract import ExtractionCache
from stillhere.writer.pipeline import Writer

from . import writer_dataset as wd

FLAGS = {"MEMORY_WRITER_V1_ENABLED": "1", "OBSERVATION_V1_ENABLED": "1"}


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _labels(conv: wd.Conv) -> Dict[int, str]:
    return {i: f"m{i + 1}" for i in range(len(conv.messages))}


def reference_complete(convs_all: Sequence[wd.Conv]) -> Callable:
    """The gold as an extractor's answer, in the same JSON the model gives. It knows nothing the gold does not say.
    The labels are read from the request itself, so they are the ones the model would see."""
    def complete(messages):
        user = messages[1]["content"]
        lines = {}
        for line in user.splitlines():
            if line.startswith("[m") and "：" in line:
                lines[line.split("：", 1)[1]] = line[1:line.index("]")]
        convs = [c for c in convs_all if all(m.text in lines for m in c.messages)]
        conv = max(convs, key=lambda c: max(m.at for m in c.messages))     # an SMS thread holds earlier conversations: answer for the newest
        lab = {i: lines[m.text] for i, m in enumerate(conv.messages)}
        cands = []
        for f in conv.facts:
            cands.append({"type": f.type if f.predicate != "event" else ("PLAN" if f.status in ("PLANNED", "SCHEDULED") else "EVENT"), "subject": f.subject, "predicate": f.predicate,
                          "object": f.object, "polarity": "positive", "assertion_type": "stated", "certainty": "certain", "event_key": f.event_key, "activity": f.activity,
                          "event_status": f.status or ("UNKNOWN" if f.predicate == "event" else None), "planned_time": f.planned, "event_time": f.event_time, "daypart": f.daypart,
                          "valid_from": None, "valid_to": None, "temporal_precision": "day" if (f.planned or f.event_time) else "unknown", "corrects": f.corrects, "retracts": f.retracts,
                          "refers_to": None, "source_message_ids": [lab[i] for i in f.src], "source_spans": list(f.spans)})
        return {"text": json.dumps({"candidates": cands}, ensure_ascii=False), "input_tokens": 0, "output_tokens": 0, "cost_cny": 0.0, "latency_ms": 0.0, "model": "reference"}
    return complete


@dataclass
class Replay:
    api: MemoryAPI
    clock: Clock
    writer: Writer
    engine: Optional[ObservationEngine]
    batches: List[dict] = field(default_factory=list)
    observation_changes: List[dict] = field(default_factory=list)
    latency_ms: List[float] = field(default_factory=list)


def new_world(complete: Callable, model: str, cache: Optional[ExtractionCache] = None, observations: bool = True, prompt_version: str = "writer-extract-v1", ds=wd,
              m22: bool = False, m221: bool = False, m222: bool = False) -> Replay:
    """``ds``: the dataset module (default the M2 dataset; ``writer_dataset_v2`` for the M2.1 independent world)."""
    clock = Clock(ds.t(8, 30) if ds is not wd else wd.t(9, 30))
    api = MemoryAPI(":memory:", clock=clock)
    ds.load(api)
    env = {**FLAGS, "MEMORY_WRITER_M22_ENABLED": "1"} if (m22 or m221 or m222) else FLAGS
    if m221:
        env = {**env, "MEMORY_WRITER_M221_ENABLED": "1"}
    if m222:                                    # Batch 2.1 final: + R13 (explicit self-correction) / R14 (holiday dates), after verification
        env = {**env, "MEMORY_WRITER_M222_ENABLED": "1"}
    writer = Writer(api, complete=complete, model=model, persona_names=ds.NAMES, speakers=ds.SPEAKERS, cache=cache, env=env, prompt_version=prompt_version)
    engine = ObservationEngine(api, env=FLAGS) if observations else None
    return Replay(api, clock, writer, engine)


def run(world: Replay, convs: Optional[Sequence[wd.Conv]] = None, ds=wd) -> Replay:
    for conv in convs or ds.order():
        known = max(m.recorded_at or m.at for m in conv.messages)
        world.clock.now = known + timedelta(seconds=1)
        started = time.perf_counter()
        r = world.writer.write_group(conv.persona, ds.group_id(conv), until=known)
        world.latency_ms.append((time.perf_counter() - started) * 1000)
        world.batches.append({"cid": conv.cid, "persona": conv.persona, "skipped": r.skipped, "operations": [o.to_dict() for o in r.operations],
                              "verdicts": [{"candidate": v.candidate.to_dict(), "support": v.support, "reasons": v.reasons, "dropped": v.dropped_fields} for v in r.verdicts],
                              "schema_ok": (r.extraction or {}).get("schema_ok"), "attempts": len((r.extraction or {}).get("attempts", []))})
        if world.engine is not None:
            touched = [o.memory_id for o in r.operations if o.memory_id] + [o.target_id for o in r.operations if o.target_id]
            for ch in world.engine.refresh(conv.persona, known, keys=affected_keys(world.api, conv.persona, touched)):
                world.observation_changes.append({"cid": conv.cid, "known_at": known.isoformat(), **ch.as_dict()})
    return world


# ---------------------------------------------------------------- matching against the gold

def norm(text: str) -> str:
    return re.sub(r"[\s，,。.、:：/《》“”\"'！!？?的]+", "", text or "")


def value_match(got: str, gold: wd.Fact) -> bool:
    g = norm(got)
    for v in (gold.object, *gold.alts):
        n = norm(v)
        if n and g and (n in g or g in n):
            return True
    return False


def subject_of(f: wd.Fact) -> str:
    return "persona" if f.subject == "self" else f.subject


def attribute_of(f: wd.Fact) -> str:
    if f.predicate == "state":
        return "state:" + f.object.split(":", 1)[0]
    return f.predicate


@dataclass
class Unit:
    kind: str                   # "fact" | "event"
    persona: str
    subject: str
    attribute: str
    value: str
    memory_ids: List[str]
    messages: List[str]
    fields: Dict[str, str] = field(default_factory=dict)       # event: current attribute -> value
    status_history: List[str] = field(default_factory=list)
    history: Dict[str, str] = field(default_factory=dict)      # event: attribute -> value of any status (a retracted event keeps its identity)


def committed_units(api: MemoryAPI) -> List[Unit]:
    """What the Writer put in memory, as units comparable to the gold: one per fact record (closed copies of a temporal change
    excluded, they restate an earlier record) and one per event subject."""
    conn = api.store._conn
    rows = conn.execute("SELECT memory_id, persona_id, subject, attribute, value, status, valid_to, supersedes, correction_reason FROM memories WHERE memory_id LIKE 'w.%' ORDER BY recorded_at_utc, memory_id").fetchall()
    msgs = lambda mid: [r[0][len("message:"):] for r in conn.execute("SELECT locator FROM provenance WHERE memory_id = ? AND locator LIKE 'message:%'", (mid,))]
    closed = {r[0] for r in rows if (r[8] or "").startswith("temporal change")}
    units, events = [], {}
    for mid, persona, subject, attribute, value, status, valid_to, supersedes, reason in rows:
        if mid in closed:
            continue
        if subject.startswith("event:"):
            u = events.setdefault(subject, Unit("event", persona, subject, "event", "", [], []))
            u.memory_ids.append(mid)
            u.messages += msgs(mid)
            if status == "active" and valid_to is None:
                u.fields[attribute] = value
            u.history.setdefault(attribute, value)
            if attribute == "event.status":
                u.status_history.append(value)
            if attribute in ("event.description", "event.activity") and not u.value:
                u.value = value
        else:
            units.append(Unit("fact", persona, subject, attribute, value, [mid], msgs(mid)))
    for u in events.values():
        u.value = " ".join(filter(None, (u.fields.get("event.description"), u.fields.get("event.activity")))) or u.value
    return units + list(events.values())


def gold_units(convs: Sequence[wd.Conv]):
    """(fact units, event units). A fact unit is a fact that should create a record; an event unit is one event (its key) with the
    fields its last mention leaves, unless it was retracted."""
    facts, events = [], {}
    for c in convs:
        for f in c.facts:
            mids = {wd.message_id(c, i) for i in f.src}
            if f.predicate == "event":
                e = events.setdefault((c.persona, f.event_key), {"persona": c.persona, "key": f.event_key, "facts": [], "messages": set(), "retracted": False, "subject": subject_of(f)})
                e["facts"].append(f)
                if f.op != "NOOP":          # a duplicate mention writes nothing (pipeline: NOOP / DUPLICATE), as on the fact side
                    e["messages"] |= mids
                e["retracted"] = e["retracted"] or f.retracts
            elif f.op != "NOOP" and not f.retracts:
                facts.append((c, f, mids))
    for e in events.values():
        last = {}
        for f in e["facts"]:
            for name in ("status", "planned", "event_time", "daypart", "activity"):
                if getattr(f, name):
                    last[name] = getattr(f, name)
        e["final"] = last
        e["first"] = e["facts"][0]
    return facts, list(events.values())


def _event_matches(u: Unit, e: dict) -> bool:
    if u.persona != e["persona"]:
        return False
    names = [norm(x) for f in e["facts"] for x in (f.object, *f.alts, f.activity or "") if norm(x)]
    text = norm(u.value) + norm(u.fields.get("event.activity", ""))
    if not any(n in text or (text and text in n) for n in names):
        return False
    gold_day = e["final"].get("event_time") or e["first"].event_time
    got_day = u.fields.get("event.time") or u.history.get("event.time") or ""
    if gold_day and e["final"].get("status") == "COMPLETED" and got_day and got_day[:10] != gold_day[:10]:
        return False
    return True


def score(world: Replay, convs: Optional[Sequence[wd.Conv]] = None, ds=wd) -> dict:
    api = world.api
    convs = list(convs or ds.conversations())
    units = committed_units(api)
    gfacts, gevents = gold_units(convs)
    live_events = [e for e in gevents if not e["retracted"]]
    # facts
    fact_units = [u for u in units if u.kind == "fact"]
    event_units = [u for u in units if u.kind == "event"]
    matched_units, matched_gold, attr_ok, attr_total, bind_ok, bind_total = set(), set(), 0, 0, 0, 0
    for gi, (c, f, mids) in enumerate(gfacts):
        hit = [k for k, u in enumerate(fact_units) if u.persona == c.persona and u.attribute == attribute_of(f) and value_match(u.value.split(":")[-1] if f.predicate == "state" else u.value, f if f.predicate != "state" else wd.Fact(f.key, f.subject, f.predicate, f.object.split(":", 1)[1], f.src, f.spans, tuple(a.split(":", 1)[-1] for a in f.alts)))]
        right_subject = [k for k in hit if fact_units[k].subject == subject_of(f)]
        if hit:
            attr_total += 1
            attr_ok += bool(right_subject)
        if right_subject:
            matched_gold.add(gi)
            for k in right_subject:
                matched_units.add(("f", k))
            k = right_subject[0]
            bind_total += 1
            bind_ok += mids <= set(fact_units[k].messages)
    ev_matched_gold, ev_matched_units, false_merge, state_ok, state_total, time_ok, time_total = set(), {}, 0, 0, 0, 0, 0
    for k, u in enumerate(event_units):
        hits = [gi for gi, e in enumerate(gevents) if _event_matches(u, e)]
        if len(hits) > 1:
            dated = [gi for gi in hits if (gevents[gi]["final"].get("event_time") or gevents[gi]["final"].get("planned") or "")[:10] == (u.fields.get("event.time") or u.fields.get("event.planned_time") or u.history.get("event.time") or u.history.get("event.planned_time") or "")[:10]]
            hits = dated or hits
        if len(hits) > 1:
            false_merge += 1
        if hits:
            ev_matched_units[k] = hits[0]
            ev_matched_gold.add(hits[0])
    for gi, e in enumerate(gevents):
        mine = [event_units[k] for k, g in ev_matched_units.items() if g == gi]
        if e["retracted"]:
            continue
        state_total += 1
        current = [u.fields.get("event.status") for u in mine if u.fields.get("event.status")]
        state_ok += bool(current) and current[-1] == e["final"].get("status")
        for name, attr in (("planned", "event.planned_time"), ("event_time", "event.time")):
            if e["final"].get(name):
                time_total += 1
                time_ok += any((u.fields.get(attr) or "")[:len(e["final"][name])] == e["final"][name] for u in mine)
        if mine:
            attr_total += 1
            attr_ok += mine[-1].fields.get("event.actor", "self") == e["first"].subject
            bind_total += 1
            bind_ok += e["messages"] <= set().union(*[set(u.messages) for u in mine])
    committed = len(fact_units) + len(event_units)
    correct_units = len(matched_units) + sum(1 for k, g in ev_matched_units.items() if not gevents[g]["retracted"])
    retracted_units = sum(1 for k, g in ev_matched_units.items() if gevents[g]["retracted"])
    precision_den = committed - retracted_units
    gold_total = len(gfacts) + len(live_events)
    recall_num = len(matched_gold) + sum(1 for gi in ev_matched_gold if not gevents[gi]["retracted"])
    # duplicates: several committed units for one gold unit
    per_gold = {}
    for (kind, k) in matched_units:
        u = fact_units[k]
        per_gold.setdefault((u.persona, u.subject, u.attribute, norm(u.value)), 0)
        per_gold[(u.persona, u.subject, u.attribute, norm(u.value))] += 1
    ev_count = {}
    for k, g in ev_matched_units.items():
        ev_count[g] = ev_count.get(g, 0) + 1
    duplicates = sum(n - 1 for n in per_gold.values() if n > 1) + sum(n - 1 for n in ev_count.values() if n > 1)
    # corrections, retractions, forbidden
    corrections = [(c, f) for c in convs for f in c.facts if f.corrects]
    missed_corr = 0
    for c, f in corrections:
        old = next(g for cc in convs for g in cc.facts if g.key == f.refers)
        still = [u for u in fact_units if u.persona == c.persona and u.attribute == attribute_of(old) and value_match(u.value, old) and api.store.get(u.memory_ids[0]).is_active]
        missed_corr += bool(still)
    retractions = [e for e in gevents if e["retracted"]]
    retraction_errors = 0
    for e in retractions:
        mine = [event_units[k] for k, g in ev_matched_units.items() if gevents[g] is e]
        if any(u.fields for u in mine):
            retraction_errors += 1
    retracted_wrongly = 0
    for (kind, k) in matched_units:
        u = fact_units[k]
        if api.store.get(u.memory_ids[0]).status.value == "retracted":
            retracted_wrongly += 1
    forbidden_hits = []
    for c in convs:
        for pred, frag in c.forbidden:
            target, _, cond = frag.partition(":")
            for u in units:
                if u.persona != c.persona:
                    continue
                if pred == "event" and u.kind == "event" and target in u.value and (not cond or (cond == "self" and u.fields.get("event.actor") == "self") or (cond == "COMPLETED" and u.fields.get("event.status") == "COMPLETED")):
                    forbidden_hits.append((c.cid, pred, frag))
                elif u.kind == "fact" and u.attribute.startswith(pred) and target in u.value:
                    forbidden_hits.append((c.cid, pred, frag))
    # checkpoints (bitemporal)
    cps = []
    for cp in ds.checkpoints():
        gold = next(f for c in convs for f in c.facts if f.key == cp.fact)
        if gold.predicate == "event":
            e = next(x for x in gevents if x["key"] == gold.event_key and x["persona"] == cp.persona)
            subjects = {event_units[k].subject for k, g in ev_matched_units.items() if gevents[g] is e}
        else:
            subjects = {subject_of(gold)}
        values = []
        for subject in subjects:
            for (mid,) in api.store._conn.execute("SELECT memory_id FROM memories WHERE persona_id = ? AND subject = ? AND attribute = ?", (cp.persona, subject, cp.attribute)):
                r = api.get(cp.persona, mid, known_at=cp.known_at)
                if r is not None and r.is_active and r.is_valid_at(cp.as_of):
                    values.append(r.value)
        ok = (not values) if cp.expect is None else any(cp.expect in v for v in values)
        cps.append({"name": cp.name, "expect": cp.expect, "got": values, "ok": ok})
    # safety
    conn = api.store._conn
    persona_leaks = conn.execute("SELECT COUNT(*) FROM provenance p JOIN memories m ON m.memory_id = p.memory_id JOIN messages g ON 'message:' || g.message_id = p.locator WHERE g.target_persona_id <> m.persona_id").fetchone()[0]
    temporal_leaks = conn.execute("SELECT COUNT(*) FROM provenance p JOIN memories m ON m.memory_id = p.memory_id JOIN messages g ON 'message:' || g.message_id = p.locator WHERE g.recorded_at_utc > m.recorded_at_utc").fetchone()[0]
    unsupported_committed = sum(1 for b in world.batches for v in b["verdicts"] if v["support"] in ("UNSUPPORTED", "AMBIGUOUS") and any(o["candidate_id"] == v["candidate"]["candidate_id"] and o["op"] in ("ADD", "ENRICH", "SUPERSEDE", "RETRACT") for o in b["operations"]))
    silent_source_loss = conn.execute("SELECT COUNT(*) FROM memories m WHERE m.memory_id LIKE 'w.%' AND NOT EXISTS (SELECT 1 FROM provenance p WHERE p.memory_id = m.memory_id AND p.locator LIKE 'message:%')").fetchone()[0]
    p = correct_units / precision_den if precision_den else 0.0
    r = recall_num / gold_total if gold_total else 0.0
    verdicts = [v for b in world.batches for v in b["verdicts"]]
    return {
        "TOTAL_RAW_MESSAGES": sum(len(c.messages) for c in convs), "TOTAL_GOLD_FACTS": gold_total, "TOTAL_EXTRACTED_FACTS": committed,
        "CANDIDATES": len(verdicts), "CANDIDATES_BY_SUPPORT": {s: sum(v["support"] == s for v in verdicts) for s in ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "AMBIGUOUS")},
        "FACT_PRECISION": round(p, 4), "FACT_RECALL": round(r, 4), "FACT_F1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
        "PRECISION_COUNTS": [correct_units, precision_den], "RECALL_COUNTS": [recall_num, gold_total],
        "SOURCE_BINDING_ACCURACY": round(bind_ok / bind_total, 4) if bind_total else None, "SOURCE_BINDING_COUNTS": [bind_ok, bind_total],
        "ENTITY_ATTRIBUTION_ACCURACY": round(attr_ok / attr_total, 4) if attr_total else None, "ENTITY_ATTRIBUTION_COUNTS": [attr_ok, attr_total],
        "TEMPORAL_NORMALIZATION_ACCURACY": round(time_ok / time_total, 4) if time_total else None, "TEMPORAL_COUNTS": [time_ok, time_total],
        "EVENT_STATE_ACCURACY": round(state_ok / state_total, 4) if state_total else None, "EVENT_STATE_COUNTS": [state_ok, state_total],
        "DUPLICATE_RATE": round(duplicates / committed, 4) if committed else 0.0, "FALSE_MERGE_RATE": round(false_merge / len(event_units), 4) if event_units else 0.0,
        "MISSED_CORRECTION_RATE": round(missed_corr / len(corrections), 4) if corrections else 0.0, "RETRACTION_ERROR_RATE": round((retraction_errors + retracted_wrongly) / max(len(retractions), 1), 4),
        "UNSUPPORTED_FACT_RATE": round(1 - p, 4) if precision_den else 0.0, "FORBIDDEN_FACTS_COMMITTED": forbidden_hits,
        "BITEMPORAL_CHECKPOINTS": [sum(x["ok"] for x in cps), len(cps)], "CHECKPOINT_DETAIL": cps,
        "PERSONA_LEAKAGE": persona_leaks, "TEMPORAL_LEAKAGE": temporal_leaks, "UNSUPPORTED_FACTS_COMMITTED": unsupported_committed, "SILENT_SOURCE_LOSS": silent_source_loss,
        "SCHEMA_FAILURES": sum(1 for b in world.batches if b["schema_ok"] is False), "REPAIRS": sum(1 for b in world.batches if b["attempts"] > 1),
        "WRITER_LATENCY_MS_P50": sorted(world.latency_ms)[len(world.latency_ms) // 2] if world.latency_ms else None,
    }


def score_observations(world: Replay) -> dict:
    eng, api = world.engine, world.api
    rows = []
    for x in wd.observation_expectations():
        cur = {k: v for k, v in eng.current(x.persona, x.known_at).items() if k.startswith(x.rule)}
        got = [v for v in cur.values()]
        status = got[0]["status"] if got else None
        if x.expect is None:
            ok = not any(v["status"] == "active" for v in got)
        else:
            ok = status == x.expect and (x.support is None or got[0]["independent_event_count"] == x.support)
        rows.append({"name": x.name, "rule": x.rule, "known_at": x.known_at.isoformat(), "expect": x.expect, "expect_support": x.support, "got": [{k: v[k] for k in ("rule_key", "status", "independent_event_count", "version", "claim")} for v in got], "ok": ok})
    conn = api.store._conn
    versions = conn.execute("SELECT observation_id, version, persona_id, generated_at FROM observations").fetchall()
    unsupported = 0
    for oid, ver, persona, gen in versions:
        when = datetime.fromisoformat(gen)
        for (mid,) in conn.execute("SELECT memory_id FROM observation_links WHERE observation_id = ? AND version = ? AND role = 'support'", (oid, ver)):
            r = api.get(persona, mid, known_at=when)
            if r is None or not r.is_active:
                unsupported += 1
                break
    neg = [r for r in rows if r["expect"] is None]
    inv = [r for r in rows if r["expect"] in ("invalidated", "retracted")]
    counted = [r for r in rows if r["expect_support"] is not None]
    dedup = [r for r in rows if r["name"] in ("evening_reading_habit_formed", "run_once_is_not_a_habit")]
    audit = dependency_audit(api)
    return {"expectations": rows, "OBSERVATIONS_CREATED": len({v[0] for v in versions}), "OBSERVATION_VERSIONS": len(versions),
            "OBSERVATION_SUPPORT_RATE": round(1 - unsupported / len(versions), 4) if versions else None, "UNSUPPORTED_OBSERVATIONS_PROMOTED": unsupported,
            "OBSERVATION_INVALIDATION_ACCURACY": round(sum(r["ok"] for r in inv) / len(inv), 4) if inv else None, "INVALIDATION_COUNTS": [sum(r["ok"] for r in inv), len(inv)],
            "OBSERVATION_CONTRADICTION_HANDLING": next(r["ok"] for r in rows if r["name"] == "evening_reading_invalidated_by_routine"),
            "DEPENDENCY_INVALIDATION_RATE": next(r["ok"] for r in rows if r["name"] == "swim_retracted_after_support_retracted"),
            "INDEPENDENT_EVIDENCE_DEDUP_RATE": round(sum(r["ok"] for r in dedup) / len(dedup), 4), "SUPPORT_COUNT_ACCURACY": [sum(r["ok"] for r in counted), len(counted)],
            "OVERGENERALIZATION_RATE": round(sum(not r["ok"] for r in neg) / len(neg), 4) if neg else 0.0, "EXPECTATIONS_PASSED": [sum(r["ok"] for r in rows), len(rows)],
            "DEPENDENCY_AUDIT": audit, "DERIVATION_CYCLE_COUNT": audit["cycles"], "changes": world.observation_changes}


def one_to_one(world: Replay, convs: Optional[Sequence[wd.Conv]] = None, ds=wd) -> dict:
    """M2.2 metric definition (the M2.1 gold audit's): every gold unit is matched by at most ONE committed unit and every committed unit
    matches at most one gold unit, so TP + FP = committed live units and TP + FN = gold units (precision and recall share one TP).
    A duplicate committed unit is a FP. Retracted gold events and the units that match them are excluded from both sides (as in ``score``).
    FPs are split into own-persona errors and units about other people (subject ``person:*``), reported, never re-scored."""
    convs = list(convs or ds.conversations())
    units = committed_units(world.api)
    gfacts, gevents = gold_units(convs)
    fact_units = [u for u in units if u.kind == "fact"]
    event_units = [u for u in units if u.kind == "event"]
    chosen, tp, fn = set(), 0, []
    for c, f, mids in gfacts:
        g = f if f.predicate != "state" else wd.Fact(f.key, f.subject, f.predicate, f.object.split(":", 1)[1], f.src, f.spans, tuple(a.split(":", 1)[-1] for a in f.alts))
        hit = [k for k, u in enumerate(fact_units) if ("f", k) not in chosen and u.persona == c.persona and u.attribute == attribute_of(f) and u.subject == subject_of(f)
               and value_match(u.value.split(":")[-1] if f.predicate == "state" else u.value, g)]
        if hit:
            chosen.add(("f", hit[0]))
            tp += 1
        else:
            fn.append(("fact", c.cid, f.key))
    ev_hits = {}
    for k, u in enumerate(event_units):
        hits = [gi for gi, e in enumerate(gevents) if _event_matches(u, e)]
        if hits:
            ev_hits.setdefault(hits[0], []).append(k)
    excluded = 0
    for gi, e in enumerate(gevents):
        ks = ev_hits.get(gi, [])
        if e["retracted"]:
            excluded += len(ks)
            chosen.update(("e", k) for k in ks)
            continue
        if ks:
            chosen.add(("e", ks[0]))
            tp += 1
        else:
            fn.append(("event", e["persona"], e["key"]))
    fps = [u for k, u in enumerate(fact_units) if ("f", k) not in chosen] + [u for k, u in enumerate(event_units) if ("e", k) not in chosen]
    live = len(fact_units) + len(event_units) - excluded
    gold = len(gfacts) + sum(1 for e in gevents if not e["retracted"])
    other = [u for u in fps if (u.subject.startswith("person:") or (u.kind == "event" and (u.fields.get("event.actor") or "self") != "self"))]
    forbidden_self = 0
    for c in convs:
        for pred, frag in c.forbidden:
            target = frag.partition(":")[0]
            for u in units:
                if u.persona != c.persona:
                    continue
                own = u.subject == "persona" if u.kind == "fact" else (u.fields.get("event.actor") or "self") == "self"
                if own and target in u.value and (u.kind == "event" if pred == "event" else u.attribute.startswith(pred)):
                    forbidden_self += 1
    p, r = (tp / live if live else 0.0), tp / gold if gold else 0.0
    return {"TP": tp, "FP": len(fps), "FN": len(fn), "committed_live": live, "gold": gold, "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0, "tp_plus_fp_equals_committed": tp + len(fps) == live, "tp_plus_fn_equals_gold": tp + len(fn) == gold,
            "fp_about_other_people": len(other), "fp_own": len(fps) - len(other), "forbidden_committed_as_persona": forbidden_self,
            "false_negatives": [list(x) for x in fn], "false_positives": [{"persona": u.persona, "subject": u.subject, "attribute": u.attribute, "value": u.value} for u in fps]}
