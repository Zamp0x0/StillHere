"""Reversible observations: higher-level patterns derived from several independent, verified memories. Deterministic rules; no model.

Rules (thresholds in ``Params``, all configurable):

=======================  ===================================================================================  =================
type                     forms when                                                                           key
=======================  ===================================================================================  =================
REPEATED_BEHAVIOR        >= 3 completed events of one activity on distinct days inside a 30-day window          activity
HABIT                    as REPEATED_BEHAVIOR, and >= 3 of them in the same part of the day                     activity, daypart
PREFERENCE_PATTERN       one current likes / dislikes said on >= 2 distinct days (re-statements counted)        polarity, object
STABLE_RELATIONSHIP      one current relationship said on >= 2 distinct days                                    value
STATE_TREND              >= 3 values of one state slot inside 45 days, strictly better or strictly worse          aspect
=======================  ===================================================================================  =================

Independence: events count once per event and once per calendar day, so five mentions of one evening's reading are one.

Lifecycle, per (persona, rule key), each change a new version and nothing deleted:

* CREATE: the rule holds and no observation is active.
* UPDATE: it still holds with a different support set; the old version becomes ``superseded``.
* INVALIDATE: a contradiction ends it (a later routine or the opposite preference): ``valid_to`` is the contradiction's time and
  the history stays readable for the window it held.
* RETRACT: it no longer holds because a memory it rested on was retracted or corrected: it should not have existed in that form.

Authority: an observation is always DERIVED_OBSERVATION, below the memories it cites. Its links point at memory ids only (never
at another observation), memories cite raw messages only, and the Writer reads raw messages only: derivation cannot loop and an
observation can never be counted as evidence for itself or written back as a fact.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from stillhere.contracts import MemoryRecord
from stillhere.memory import extension as ext
from stillhere.memory.sqlite_store import _utc
from stillhere.memory.temporal import CALENDAR_TZ

FLAG = "OBSERVATION_V1_ENABLED"
DERIVATION = "rules-v1"
ORDINAL = (("差", 0), ("不好", 0), ("糟", 0), ("一般", 1), ("还行", 1), ("好", 2), ("不错", 2))
DAYPART_WORDS = ("早上", "上午", "中午", "下午", "晚上", "夜里")


def enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    return (env if env is not None else os.environ).get(FLAG, "").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Params:
    min_events: int = 3
    window_days: int = 30
    min_statements: int = 2
    trend_points: int = 3
    trend_days: int = 45


@dataclass
class Desired:
    rule_key: str
    type: str
    claim: str
    support: List[str]
    contradict: List[str]
    independent: int
    window: Tuple[Optional[datetime], Optional[datetime]]
    invalid_from: Optional[datetime] = None       # set when a contradiction ends it


@dataclass
class Change:
    action: str
    observation_id: str
    version: int
    rule_key: str
    reason: str = ""

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _day(t: datetime):
    return t.astimezone(CALENDAR_TZ).date()


def _ordinal(value: str) -> Optional[int]:
    for word, n in ORDINAL:
        if word in value:
            return n
    return None


def _norm(activity: str) -> str:
    a = (activity or "").strip()
    return {"阅读": "读书", "看书": "读书"}.get(a, a)


class ObservationEngine:
    def __init__(self, api, *, params: Params = Params(), env: Optional[Mapping[str, str]] = None) -> None:
        if not enabled(env):
            raise RuntimeError(f"{FLAG} is off: observations are a research path")
        self.api, self.params = api, params
        ext.ensure(api.store)

    # ------------------------------------------------------------------ reading memories at a knowledge time

    def _believed(self, persona: str, known_at: datetime) -> List[MemoryRecord]:
        rows = self.api.store._conn.execute("SELECT memory_id FROM memories WHERE persona_id = ? AND recorded_at_utc <= ? ORDER BY recorded_at_utc, memory_id",
                                            (persona, _utc(known_at))).fetchall()
        out = []
        for (mid,) in rows:
            r = self.api.get(persona, mid, known_at=known_at)
            if r is not None and r.is_active:
                out.append(r)
        return out

    def _restatements(self, persona: str, known_at: datetime) -> Dict[str, List[datetime]]:
        """memory id -> times it was said again (the Writer logs these instead of writing a duplicate)."""
        out: Dict[str, List[datetime]] = {}
        for ops, known in self.api.store._conn.execute("SELECT operations, known_at FROM writer_batches WHERE persona_id = ?", (persona,)):
            if datetime.fromisoformat(known) > known_at:
                continue
            for op in json.loads(ops):
                if op["op"] == "NOOP" and op["relation"] == "DUPLICATE" and op.get("detail", "").startswith("{"):
                    out.setdefault(op["memory_id"], []).append(datetime.fromisoformat(json.loads(op["detail"])["said"]))
        return out

    # ------------------------------------------------------------------ rules

    def desired(self, persona: str, known_at: datetime) -> Dict[str, Desired]:
        p = self.params
        records = self._believed(persona, known_at)
        events: Dict[str, Dict[str, MemoryRecord]] = {}
        for r in records:
            if r.subject.startswith("event:") and r.valid_to is None:
                events.setdefault(r.subject, {})[r.attribute] = r
        done = []
        for subject, attrs in events.items():
            st = attrs.get("event.status")
            if st is None or st.value != "COMPLETED" or "event.activity" not in attrs:
                continue
            when = attrs["event.time"].value if "event.time" in attrs else None
            day = datetime.fromisoformat(when).date() if when and len(when) >= 10 else _day(st.valid_from)
            done.append((subject, _norm(attrs["event.activity"].value), attrs.get("event.daypart").value if "event.daypart" in attrs else None, day, st.memory_id))
        out: Dict[str, Desired] = {}
        routines = [r for r in records if r.attribute == "routine"]
        for activity in sorted({d[1] for d in done}):
            mine = sorted([d for d in done if d[1] == activity], key=lambda d: d[3])
            last = mine[-1][3]
            window = [d for d in mine if (last - d[3]).days < p.window_days]
            by_day = {}
            for d in window:
                by_day.setdefault(d[3], d)
            if len(by_day) >= p.min_events:
                days = sorted(by_day)
                out[f"REPEATED_BEHAVIOR:{activity}"] = Desired(f"REPEATED_BEHAVIOR:{activity}", "REPEATED_BEHAVIOR", f"{days[0]}至{days[-1]}之间至少{len(by_day)}天{activity}",
                                                               [by_day[x][4] for x in days], [], len(by_day), (datetime(days[0].year, days[0].month, days[0].day, tzinfo=CALENDAR_TZ), datetime(days[-1].year, days[-1].month, days[-1].day, tzinfo=CALENDAR_TZ)))
            for part in DAYPART_WORDS:
                same = {}
                for d in window:
                    if d[2] == part:
                        same.setdefault(d[3], d)
                if len(same) < p.min_events:
                    continue
                days = sorted(same)
                start = datetime(days[0].year, days[0].month, days[0].day, tzinfo=CALENDAR_TZ)
                end = datetime(days[-1].year, days[-1].month, days[-1].day, tzinfo=CALENDAR_TZ)
                against = [r for r in routines if activity[:1] in r.value and any(w in r.value for w in DAYPART_WORDS if w != part) and r.valid_from > end]
                key = f"HABIT:{activity}:{part}"
                out[key] = Desired(key, "HABIT", f"{days[0]}至{days[-1]}期间习惯在{part}{activity}（{len(same)}天）", [same[x][4] for x in days], [r.memory_id for r in against], len(same),
                                   (start, end), min((r.valid_from for r in against), default=None))
        restated = self._restatements(persona, known_at)
        for r in records:
            if r.attribute in ("likes", "dislikes", "relationship") and r.valid_to is None:
                days = {_day(r.valid_from)} | {_day(t) for t in restated.get(r.memory_id, [])}
                if len(days) < p.min_statements:
                    continue
                if r.attribute == "relationship":
                    key, typ, claim = f"STABLE_RELATIONSHIP:{r.value}", "STABLE_RELATIONSHIP", f"在{len(days)}个不同日子提到{r.value}，关系稳定"
                    against = []
                else:
                    key, typ = f"PREFERENCE_PATTERN:{r.attribute}:{r.value}", "PREFERENCE_PATTERN"
                    claim = f"在{len(days)}个不同日子表示{'喜欢' if r.attribute == 'likes' else '不喜欢'}{r.value}"
                    opposite = "dislikes" if r.attribute == "likes" else "likes"
                    against = [o for o in records if o.attribute == opposite and o.value == r.value and o.valid_from > r.valid_from]
                ds = sorted(days)
                out[key] = Desired(key, typ, claim, [r.memory_id], [o.memory_id for o in against], len(days),
                                   (datetime(ds[0].year, ds[0].month, ds[0].day, tzinfo=CALENDAR_TZ), datetime(ds[-1].year, ds[-1].month, ds[-1].day, tzinfo=CALENDAR_TZ)),
                                   min((o.valid_from for o in against), default=None))
        series: Dict[str, List[MemoryRecord]] = {}
        for r in records:
            if r.attribute.startswith("state:") and _ordinal(r.value) is not None:
                series.setdefault(r.attribute[len("state:"):], []).append(r)
        for aspect, points in series.items():
            points = sorted(points, key=lambda r: r.valid_from)
            recent = [x for x in points if (points[-1].valid_from - x.valid_from).days <= p.trend_days][-p.trend_points:]
            if len(recent) < p.trend_points:
                continue
            values = [_ordinal(x.value) for x in recent]
            if all(a < b for a, b in zip(values, values[1:])) or all(a > b for a, b in zip(values, values[1:])):
                direction = "好转" if values[-1] > values[0] else "变差"
                key = f"STATE_TREND:{aspect}"
                out[key] = Desired(key, "STATE_TREND", f"{aspect}在{_day(recent[0].valid_from)}至{_day(recent[-1].valid_from)}期间{direction}（{'→'.join(x.value for x in recent)}）",
                                   [x.memory_id for x in recent], [], len(recent), (recent[0].valid_from, recent[-1].valid_from))
        return out

    # ------------------------------------------------------------------ stored observations

    def current(self, persona: str, known_at: datetime, rule_key: Optional[str] = None) -> Dict[str, dict]:
        """rule key -> the version believed at ``known_at`` (any status), the latest one generated by then."""
        conn = self.api.store._conn
        q = "SELECT * FROM observations WHERE persona_id = ? AND generated_at_utc <= ?" + (" AND rule_key = ?" if rule_key else "") + " ORDER BY generated_at_utc, version"
        rows = conn.execute(q, (persona, _utc(known_at), *( [rule_key] if rule_key else []))).fetchall()
        out: Dict[str, dict] = {}
        for row in rows:
            d = dict(row)
            if d["status_changed_at"] and d["status_changed_at"] > _utc(known_at):
                d["status"] = "active"
            d["support"] = [r[0] for r in conn.execute("SELECT memory_id FROM observation_links WHERE observation_id = ? AND version = ? AND role = 'support' ORDER BY memory_id", (d["observation_id"], d["version"]))]
            d["contradict"] = [r[0] for r in conn.execute("SELECT memory_id FROM observation_links WHERE observation_id = ? AND version = ? AND role = 'contradict' ORDER BY memory_id", (d["observation_id"], d["version"]))]
            out[d["rule_key"]] = d
        return out

    def _insert(self, conn, persona: str, oid: str, version: int, d: Desired, known_at: datetime, status: str = "active", valid_to: Optional[datetime] = None, reason: str = "") -> None:
        conn.execute("INSERT INTO observations (observation_id, version, persona_id, observation_type, rule_key, claim, window_start, window_end, valid_from, valid_to, generated_at, generated_at_utc, "
                     "derivation_method, support_count, independent_event_count, heuristic_support, status, status_changed_at, status_reason) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (oid, version, persona, d.type, d.rule_key, d.claim, d.window[0].isoformat() if d.window[0] else None, d.window[1].isoformat() if d.window[1] else None,
                      d.window[0].isoformat() if d.window[0] else None, valid_to.isoformat() if valid_to else None, known_at.isoformat(), _utc(known_at), DERIVATION, len(d.support), d.independent,
                      round(min(1.0, d.independent / (self.params.min_events * 2)), 4), status, _utc(known_at) if status != "active" else None, reason or None))
        conn.executemany("INSERT INTO observation_links (observation_id, version, memory_id, role) VALUES (?,?,?,?)",
                         [(oid, version, m, "support") for m in d.support] + [(oid, version, m, "contradict") for m in d.contradict])

    def _set_status(self, conn, row: dict, status: str, known_at: datetime, reason: str, valid_to: Optional[datetime] = None) -> None:
        conn.execute("UPDATE observations SET status = ?, status_changed_at = ?, status_reason = ?, valid_to = COALESCE(?, valid_to) WHERE observation_id = ? AND version = ? AND status = 'active'",
                     (status, _utc(known_at), reason, valid_to.isoformat() if valid_to else None, row["observation_id"], row["version"]))

    # ------------------------------------------------------------------ lifecycle

    def refresh(self, persona: str, known_at: datetime, *, keys: Optional[Iterable[str]] = None) -> List[Change]:
        """Bring the persona's observations in line with its memories as believed at ``known_at``. ``keys`` (rule-key prefixes)
        limits the work to what a write touched; None re-checks every key."""
        want = self.desired(persona, known_at)
        have = self.current(persona, known_at)
        prefixes = None if keys is None else tuple(keys)
        believed = {r.memory_id for r in self._believed(persona, known_at)}
        changes: List[Change] = []
        conn = self.api.store._conn
        with self.api.transaction():
            for key in sorted(set(want) | set(have)):
                if prefixes is not None and not key.startswith(prefixes) and not (key in have and set(have[key]["support"]) - believed):
                    continue
                d, cur = want.get(key), have.get(key)
                active = cur is not None and cur["status"] == "active"
                if d is not None and d.invalid_from is not None:
                    if active:
                        self._set_status(conn, cur, "invalidated", known_at, f"contradicted by {','.join(d.contradict)}", d.invalid_from)
                        conn.executemany("INSERT OR IGNORE INTO observation_links (observation_id, version, memory_id, role) VALUES (?,?,?,?)",
                                         [(cur["observation_id"], cur["version"], m, "contradict") for m in d.contradict])
                        changes.append(Change("INVALIDATE", cur["observation_id"], cur["version"], key, "contradiction"))
                    continue
                if d is not None and not active:
                    generation = conn.execute("SELECT COUNT(DISTINCT observation_id) FROM observations WHERE persona_id = ? AND rule_key = ?", (persona, key)).fetchone()[0]
                    if cur is not None and cur["status"] == "invalidated" and d.window[0] and cur["window_end"] and d.window[0] <= datetime.fromisoformat(cur["window_end"]):
                        continue                          # the same evidence that was contradicted does not revive it
                    oid = "obs." + hashlib.sha256(f"{persona}|{key}|{generation}".encode()).hexdigest()[:16]
                    self._insert(conn, persona, oid, 1, d, known_at)
                    changes.append(Change("CREATE", oid, 1, key))
                elif d is not None and active:
                    if sorted(d.support) != sorted(cur["support"]) or d.claim != cur["claim"]:
                        self._set_status(conn, cur, "superseded", known_at, "updated")
                        self._insert(conn, persona, cur["observation_id"], cur["version"] + 1, d, known_at)
                        changes.append(Change("UPDATE", cur["observation_id"], cur["version"] + 1, key))
                elif d is None and active:
                    lost = set(cur["support"]) - believed
                    if lost:
                        self._set_status(conn, cur, "retracted", known_at, f"support no longer current: {','.join(sorted(lost))}")
                        changes.append(Change("RETRACT", cur["observation_id"], cur["version"], key, "support_retracted"))
                    else:
                        self._set_status(conn, cur, "invalidated", known_at, "rule no longer holds")
                        changes.append(Change("INVALIDATE", cur["observation_id"], cur["version"], key, "rule_no_longer_holds"))
        return changes


def affected_keys(api, persona: str, memory_ids: Sequence[str]) -> List[str]:
    """Rule-key prefixes a set of written or retracted memories can change: the incremental scope of a refresh."""
    out: Set[str] = set()
    for mid in memory_ids:
        r = api.store.get(mid, persona_id=persona)
        if r is None:
            continue
        if r.subject.startswith("event:"):
            out.update({"REPEATED_BEHAVIOR:", "HABIT:"})
        elif r.attribute in ("likes", "dislikes"):
            out.add(f"PREFERENCE_PATTERN:")
        elif r.attribute == "relationship":
            out.add("STABLE_RELATIONSHIP:")
        elif r.attribute.startswith("state:"):
            out.add(f"STATE_TREND:{r.attribute[len('state:'):]}")
        elif r.attribute == "routine":
            out.add("HABIT:")
    return sorted(out)


def dependency_audit(api) -> dict:
    """The acyclicity guarantee, checked: every link names a memory, no memory cites an observation, every Writer memory cites messages."""
    conn = api.store._conn
    links = conn.execute("SELECT DISTINCT l.memory_id FROM observation_links l").fetchall()
    missing = [m for (m,) in links if conn.execute("SELECT 1 FROM memories WHERE memory_id = ?", (m,)).fetchone() is None]
    observation_ids = {r[0] for r in conn.execute("SELECT DISTINCT observation_id FROM observations")}
    cites_observation = conn.execute("SELECT COUNT(*) FROM provenance WHERE locator LIKE 'observation:%' OR source_id IN (SELECT observation_id FROM observations)").fetchone()[0]
    link_to_observation = sum(1 for (m,) in links if m in observation_ids)
    writer_without_message = conn.execute("SELECT COUNT(*) FROM memories m WHERE m.memory_id LIKE 'w.%' AND NOT EXISTS (SELECT 1 FROM provenance p WHERE p.memory_id = m.memory_id AND p.locator LIKE 'message:%')").fetchone()[0]
    return {"links": len(links), "links_to_missing_memory": len(missing), "links_to_observation": link_to_observation, "memories_citing_observations": cites_observation,
            "writer_memories_without_message_source": writer_without_message, "cycles": link_to_observation + cites_observation}
