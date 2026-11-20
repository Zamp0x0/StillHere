"""The Memory Writer: raw conversation group -> candidates -> verification -> resolution -> one atomic commit.

    group (MemoryAPI.conversation: every message, with sender and recipient)
      -> extract (injected model call, JSON mode, local schema check, one repair at most, cached)
      -> verify (deterministic, against the raw messages)
      -> resolve against what the persona's memories held at the group's knowledge time
      -> commit through MemoryAPI inside one transaction, with the batch row (the import manifest)

Logical operations and how each maps onto the existing store (no new memory table, no overwrite):

* ADD / ENRICH: ``add`` of a new record (ENRICH when it adds an attribute to an event already known).
* SUPERSEDE (CORRECTION): ``correct(old, new)``: the old record is superseded at the new record's knowledge time.
* TEMPORAL_CHANGE: the old record is closed (``correct`` to a copy whose ``valid_to`` is the change) and the new value is added,
  so "where did I live in 2024" still finds the old value, and an event's earlier status stays readable for its time.
* RETRACT: ``retract(at=known time, evidence=the message)``.
* CONTRADICTION: the new value is added; the store records the conflict on a single-valued slot. Nothing is overwritten.
* DUPLICATE / NOOP: nothing is written; the candidate is logged.

The knowledge time of a group is the latest ``recorded_at`` of its messages, so a late-arriving message changes nothing that
an earlier ``known_at`` can see. A group whose content and extraction hashes are already in ``writer_batches`` is skipped.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from stillhere.context.attribution import round_direction, sms_counterpart, sms_direction, name_of, Direction, UNKNOWN
from stillhere.contracts import Cardinality, MemoryRecord, MemoryStatus, Provenance, message_locator
from stillhere.memory import extension as ext
from stillhere.memory.sqlite_store import _utc

from . import m22, m221, m222
from .contracts import (ADD, CONTRADICTION, CORRECTION, DUPLICATE, ENRICH, INDEPENDENT_FACT, NOOP, PREDICATES, RETRACT, SUPERSEDE, TEMPORAL_CHANGE,
                        UNRESOLVED, UNSUPPORTED, WRITER_VERSION, Candidate, Operation, Verdict)
from .dates import parse_iso
from .extract import ExtractionCache, GroupMessage, extract, prompt_sha256
from .verify import verify

FLAG = "MEMORY_WRITER_V1_ENABLED"
CHANGE_CUES = ("搬到", "搬去", "现在", "改成", "换成", "以后", "已经不", "不再", "辞职", "换了", "后来", "改到", "改为", "推迟", "提前", "升职", "毕业", "从今天起", "开始")
STATUS_ZH = {"PLANNED": "计划中", "SCHEDULED": "已预约", "IN_PROGRESS": "进行中", "COMPLETED": "已完成", "CANCELLED": "已取消", "UNKNOWN": "不明"}
TEMPLATES = {"lives_in": "{s}住在{v}", "works_at": "{s}在{v}工作", "job": "{s}的职业是{v}", "school": "{s}在{v}上学", "age": "{s}的年龄：{v}", "birthday": "{s}的生日是{v}",
             "pet": "{s}养了{v}", "owns": "{s}有{v}", "relationship": "{s}的{v}", "likes": "{s}喜欢{v}", "dislikes": "{s}不喜欢{v}", "routine": "{s}通常{v}", "state": "{s}的状态：{v}"}
EVENT_TEMPLATES = {"event.actor": "事件当事人：{v}", "event.description": "事件：{v}", "event.activity": "事件活动：{v}", "event.status": "事件状态：{v}", "event.planned_time": "事件计划时间：{v}",
                   "event.time": "事件发生时间：{v}", "event.daypart": "事件时段：{v}"}


def enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    return (env if env is not None else os.environ).get(FLAG, "").strip().lower() in ("1", "true", "yes", "on")


def sha(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


@dataclass
class BatchResult:
    group_id: str
    persona_id: str
    skipped: bool
    operations: List[Operation] = field(default_factory=list)
    verdicts: List[Verdict] = field(default_factory=list)
    extraction: Optional[dict] = None
    known_at: Optional[datetime] = None
    error: Optional[str] = None


class Writer:
    def __init__(self, api, *, complete: Callable, model: str, persona_names: Mapping[str, str], speakers: Optional[Mapping[str, str]] = None,
                 cache: Optional[ExtractionCache] = None, env: Optional[Mapping[str, str]] = None, round_types=("synthetic_chat_round", "lifebench_agent_chat"),
                 prompt_version: str = "writer-extract-v1", reviewer: Optional[str] = None, strict_self_attribution: bool = False,
                 pre_verify: Optional[Callable] = None, post_verify: Optional[Callable] = None) -> None:
        if not enabled(env):
            raise RuntimeError(f"{FLAG} is off: the Memory Writer is a research path")
        self.api, self.complete, self.model = api, complete, model
        self.persona_names, self.speakers = dict(persona_names), dict(speakers or {})
        self.cache = cache or ExtractionCache(None)
        self.round_types = round_types
        self.prompt_version = prompt_version
        self.reviewer = reviewer             # the owner's review for REAL records (store write gate); None = unchanged default behaviour
        # optional, default off (unchanged behaviour): strict two-person attribution and caller hooks around verification
        self.strict_self, self.pre_verify, self.post_verify = strict_self_attribution, pre_verify, post_verify
        self.env = env
        ext.ensure(api.store)

    # ------------------------------------------------------------------ inputs

    def group(self, persona_id: str, conversation_id: str, *, until: Optional[datetime] = None) -> List[GroupMessage]:
        """The group as known at ``until`` (default: everything stored): every message, ordered, with both ends."""
        far = until or datetime(9999, 1, 1, tzinfo=timezone.utc)
        raw, _ = self.api.conversation(persona_id, conversation_id, as_of=far, known_at=far)
        participants = [m.speaker_id for m in raw]
        out = []
        for i, m in enumerate(raw, 1):
            if m.source_type in self.round_types:
                d = round_direction(m.speaker_id, participants, persona_id, self.speakers)
            else:
                who = name_of(m.speaker_id, persona_id, self.speakers)
                d = sms_direction(m.locator, who, sms_counterpart(m.conversation_id, persona_id, self.speakers)) or Direction(who, UNKNOWN)
            out.append(GroupMessage(f"m{i}", m.message_id, m.speaker_id, d.sender, d.recipient, m.timestamp, m.recorded_at or m.timestamp, m.content, m.source_id))
        return out

    def _slot(self, persona_id: str, subject: str, attribute: str, known_at: datetime) -> List[MemoryRecord]:
        """Records of a slot believed at ``known_at`` and still valid (no closed validity)."""
        conn = self.api.store._conn
        rows = conn.execute("SELECT memory_id FROM memories WHERE persona_id = ? AND subject = ? AND attribute = ? ORDER BY recorded_at_utc, memory_id",
                            (persona_id, subject, attribute)).fetchall()
        out = []
        for (mid,) in rows:
            r = self.api.get(persona_id, mid, known_at=known_at)
            if r is not None and r.is_active and r.valid_to is None:
                out.append(r)
        return out

    def existing(self, persona_id: str, known_at: datetime, limit: int = 60) -> List[Tuple[str, str, str, str]]:
        conn = self.api.store._conn
        rows = conn.execute("SELECT memory_id FROM memories WHERE persona_id = ? AND recorded_at_utc <= ? ORDER BY recorded_at_utc DESC, memory_id LIMIT ?",
                            (persona_id, _utc(known_at), limit * 3)).fetchall()
        out = []
        for (mid,) in rows:
            r = self.api.get(persona_id, mid, known_at=known_at)
            if r is not None and r.is_active and r.valid_to is None:
                out.append((r.memory_id, r.subject, r.attribute, r.value))
        return list(reversed(out[:limit]))

    # ------------------------------------------------------------------ records

    def _provenance(self, c: Candidate, msgs: Mapping[str, GroupMessage]) -> Tuple[Provenance, ...]:
        out = []
        for mid in c.source_message_ids:
            m = msgs[mid]
            quote = next((s for s in c.source_spans if s.replace(" ", "") in m.content.replace(" ", "")), None)
            out.append(Provenance(source_id=m.source_id, locator=message_locator(m.message_id), quote=(quote or "")[:200] or None, observed_at=m.timestamp))
        return tuple(out)

    def _record(self, c: Candidate, subject: str, attribute: str, value: str, statement: str, msgs, cardinality: Cardinality, kind, valid_from, valid_to=None) -> MemoryRecord:
        cited = [msgs[m] for m in c.source_message_ids]
        recorded = max(m.recorded_at for m in cited)
        origin = self.api.get_source(c.persona_id, cited[0].source_id).origin
        vf = valid_from or min(m.timestamp for m in cited)
        mid = "w." + sha(c.persona_id, subject, attribute, value, vf.isoformat(), ",".join(sorted(c.source_message_ids)))[:20]
        return MemoryRecord(memory_id=mid, persona_id=c.persona_id, kind=kind, subject=subject, attribute=attribute, value=value, statement=statement, provenance=self._provenance(c, msgs),
                            origin=origin, recorded_at=recorded, valid_from=vf, valid_to=valid_to, cardinality=cardinality, confidence=1.0)

    @staticmethod
    def _time(value: Optional[str], tz) -> Optional[datetime]:
        got, precision = parse_iso(value) if value else (None, None)
        if got is None or precision == "month":
            return None
        if precision == "exact":
            return datetime.fromisoformat(value).replace(tzinfo=datetime.fromisoformat(value).tzinfo or tz)
        return datetime(got.year, got.month, got.day, tzinfo=tz)

    def event_subject(self, c: Candidate, said: datetime) -> str:
        """A plan keeps one identity through reschedules and its end (``event:<persona, key>``); a one-off occurrence with no plan
        behind it is keyed by its day as well, so five evenings of reading stay five events. A valid ``refers_to`` wins."""
        if (c.refers_to or "").startswith("event:"):
            return c.refers_to
        key = (c.event_key or c.object).strip().lower()
        plan = "event:" + sha(c.persona_id, c.subject, key)[:16]
        if c.event_status in ("PLANNED", "SCHEDULED", "IN_PROGRESS", "CANCELLED") or self.api.store._conn.execute("SELECT 1 FROM memories WHERE persona_id = ? AND subject = ?", (c.persona_id, plan)).fetchone():
            return plan
        day = c.event_time or c.planned_time or said.astimezone(said.tzinfo).date().isoformat()
        return "event:" + sha(c.persona_id, c.subject, key, day[:10])[:16]

    def records_for(self, c: Candidate, msgs: Mapping[str, GroupMessage]) -> List[MemoryRecord]:
        kind, cardinality, _ = PREDICATES[c.predicate]
        tz = msgs[c.source_message_ids[0]].timestamp.tzinfo
        said = min(msgs[m].timestamp for m in c.source_message_ids)
        if c.predicate != "event":
            subject = "persona" if c.subject == "self" else c.subject
            label = "本人" if c.subject == "self" else c.subject.split(":", 1)[1]
            attribute, value = c.predicate, c.object
            if c.predicate == "state" and ":" in c.object.replace("：", ":"):
                aspect, value = [x.strip() for x in c.object.replace("：", ":").split(":", 1)]
                attribute = f"state:{aspect}"
                cardinality = Cardinality.SINGLE
            vf = self._time(c.valid_from, tz) or said
            vt = self._time(c.valid_to, tz)
            if vt is not None and vt <= vf:
                vt = None
            return [self._record(c, subject, attribute, value, TEMPLATES[c.predicate].format(s=label, v=c.object), msgs, cardinality, kind, vf, vt)]
        subject = self.event_subject(c, said)
        actor = "本人" if c.subject == "self" else c.subject.split(":", 1)[1]
        fields = [("event.actor", c.subject), ("event.description", c.object), ("event.activity", c.activity), ("event.status", c.event_status), ("event.planned_time", c.planned_time),
                  ("event.time", c.event_time), ("event.daypart", c.daypart)]
        out = []
        for attribute, value in fields:
            if not value:
                continue
            shown = STATUS_ZH.get(value, value) if attribute == "event.status" else actor if attribute == "event.actor" else value
            out.append(self._record(c, subject, attribute, value, EVENT_TEMPLATES[attribute].format(v=shown), msgs, Cardinality.SINGLE, kind, said))
        return out

    # ------------------------------------------------------------------ decisions

    def decide(self, v: Verdict, msgs: Mapping[str, GroupMessage], known_at: datetime, existing_ids) -> List[Operation]:
        c = v.candidate
        spans = " ".join(c.source_spans)
        ops: List[Operation] = []
        if c.retracts:
            targets = [c.refers_to] if c.refers_to and not c.refers_to.startswith("event:") else []
            pointed = self.api.store.get(targets[0], persona_id=c.persona_id) if targets else None
            if pointed is not None and pointed.subject.startswith("event:"):
                targets = []                                    # a pointer to one attribute of an event: the event is retracted whole
            if not targets and c.predicate == "event":         # an event is retracted whole: every current attribute of it
                subject = pointed.subject if pointed is not None and pointed.subject.startswith("event:") else c.refers_to if (c.refers_to or "").startswith("event:") else self.event_subject(c, min(msgs[m].timestamp for m in c.source_message_ids))
                conn = self.api.store._conn
                targets = [mid for (mid,) in conn.execute("SELECT memory_id FROM memories WHERE persona_id = ? AND subject = ? ORDER BY memory_id", (c.persona_id, subject))
                           if (r := self.api.get(c.persona_id, mid, known_at=known_at)) is not None and r.is_active]
            elif not targets:
                rec = self.records_for(c, msgs)[0]
                targets = [r.memory_id for r in self._slot(c.persona_id, rec.subject, rec.attribute, known_at) if r.value == rec.value or rec.value in r.value or r.value in rec.value]
            if not targets:
                return [Operation(NOOP, UNRESOLVED, candidate_id=c.candidate_id, detail="retraction with nothing to retract")]
            for t in targets:
                first = msgs[c.source_message_ids[0]]
                self.api.retract(c.persona_id, t, f"retracted by {first.message_id}", evidence=Provenance(source_id=first.source_id, locator=message_locator(first.message_id)), at=known_at, reviewer=self.reviewer)
                ops.append(Operation(RETRACT, CORRECTION, memory_id=t, target_id=t, candidate_id=c.candidate_id))
            return ops
        for rec in self.records_for(c, msgs):
            current = self._slot(c.persona_id, rec.subject, rec.attribute, known_at)
            same = [r for r in current if r.value == rec.value]
            if not same and rec.cardinality is Cardinality.MULTI:     # 团子 and 叫团子的橘猫: one value inside the other is the same thing said again
                n = lambda v: "".join(ch for ch in v if ch.isalnum())
                same = [r for r in current if n(r.value) and n(rec.value) and (n(r.value) in n(rec.value) or n(rec.value) in n(r.value))]
            if same:                                # said again: nothing is written; the re-statement is logged with its source and time
                said = min(msgs[m].timestamp for m in c.source_message_ids)
                ops.append(Operation(NOOP, DUPLICATE, memory_id=same[0].memory_id, candidate_id=c.candidate_id,
                                     detail=json.dumps({"reaffirmed_by": list(c.source_message_ids), "said": said.isoformat()}, ensure_ascii=False)))
                continue
            if self.api.store._conn.execute("SELECT 1 FROM memories WHERE memory_id = ?", (rec.memory_id,)).fetchone():
                ops.append(Operation(NOOP, DUPLICATE, memory_id=rec.memory_id, candidate_id=c.candidate_id, detail="already written"))
                continue
            different = [r for r in current if r.value != rec.value]
            single = rec.cardinality is Cardinality.SINGLE
            time_varying = rec.attribute.startswith(("event.", "state:"))      # an event's fields and a state change over time by nature
            # M2.2.2 R15: an explicit correction of a multi-valued slot replaces the one earlier value it names (see m222)
            m_old = (m222.multi_correction_target(c, rec, different, list(msgs.values()), c.persona_id)
                     if (m222.enabled(self.env) and different and c.corrects and not single and not time_varying) else None)
            if m_old is not None or (different and c.corrects and single and not time_varying):
                old = m_old if m_old is not None else different[-1]
                if not c.valid_from:                       # a correction fixes the value for the time the old one claimed
                    rec = replace(rec, valid_from=old.valid_from, valid_to=old.valid_to if not c.valid_to else rec.valid_to,
                                  memory_id="w." + sha(rec.memory_id, "corrects", old.memory_id)[:20])
                self.api.correct(c.persona_id, old.memory_id, rec, f"correction from {c.source_message_ids[0]}", reviewer=self.reviewer)
                ops.append(Operation(SUPERSEDE, CORRECTION, memory_id=rec.memory_id, target_id=old.memory_id, candidate_id=c.candidate_id))
            elif different and single and (time_varying or c.corrects or any(w in spans for w in CHANGE_CUES)):
                old = different[-1]
                change = rec.valid_from
                if old.valid_from is not None and change > old.valid_from and not c.corrects:
                    closed = replace(old, memory_id="w." + sha(old.memory_id, "closed", change.isoformat())[:20], valid_to=change, recorded_at=max(rec.recorded_at, old.recorded_at),
                                     supersedes=None, superseded_by=None, correction_reason=None, conflicts_with=())
                    self.api.correct(c.persona_id, old.memory_id, closed, f"temporal change from {c.source_message_ids[0]}", reviewer=self.reviewer)
                    self.api.add_facts(c.persona_id, [rec], reviewer=self.reviewer)
                    ops.append(Operation(SUPERSEDE, TEMPORAL_CHANGE, memory_id=rec.memory_id, target_id=old.memory_id, candidate_id=c.candidate_id, detail=f"closed as {closed.memory_id}"))
                else:
                    self.api.correct(c.persona_id, old.memory_id, rec, f"{'correction' if c.corrects else 'state change'} from {c.source_message_ids[0]}", reviewer=self.reviewer)
                    ops.append(Operation(SUPERSEDE, CORRECTION if c.corrects else TEMPORAL_CHANGE, memory_id=rec.memory_id, target_id=old.memory_id, candidate_id=c.candidate_id))
            elif different and single:
                self.api.add_facts(c.persona_id, [rec], reviewer=self.reviewer)
                ops.append(Operation(ADD, CONTRADICTION, memory_id=rec.memory_id, target_id=different[-1].memory_id, candidate_id=c.candidate_id, detail="kept both; the store records the conflict"))
            else:
                self.api.add_facts(c.persona_id, [rec], reviewer=self.reviewer)
                known_subject = rec.subject.startswith("event:") and any(e[1] == rec.subject for e in existing_ids)
                ops.append(Operation(ENRICH if known_subject else ADD, INDEPENDENT_FACT, memory_id=rec.memory_id, candidate_id=c.candidate_id))
        return ops

    # ------------------------------------------------------------------ one group

    def write_group(self, persona_id: str, conversation_id: str, *, until: Optional[datetime] = None, fail_after: Optional[int] = None,
                    window: Optional[Tuple[int, int]] = None, group_key: Optional[str] = None) -> BatchResult:
        """``until``: the knowledge time of this write; the group is read as known then (a replay never sees its own future).
        ``window`` (start, end): only messages[start:end] of the group (long-session chunking; labels stay those of the whole group);
        ``group_key``: the ``writer_batches`` key of that chunk (default: the conversation id). Defaults leave the behaviour unchanged."""
        msgs = self.group(persona_id, conversation_id, until=until)
        if window is not None:
            msgs = msgs[window[0]:window[1]]
        batch_key = group_key or conversation_id
        if not msgs:
            return BatchResult(conversation_id, persona_id, True, error="empty group")
        known_at = max(m.recorded_at for m in msgs)
        content = sha(*[f"{m.message_id}:{sha(m.content)}" for m in msgs])
        existing = self.existing(persona_id, known_at)
        candidates, record = extract(self.complete, self.model, persona_id, self.persona_names.get(persona_id, persona_id), msgs, existing, self.cache, self.prompt_version)
        extraction = sha(json.dumps(record["candidates"], ensure_ascii=False, sort_keys=True), prompt_sha256(self.prompt_version))
        conn = self.api.store._conn
        if conn.execute("SELECT 1 FROM writer_batches WHERE group_id = ? AND content_sha256 = ? AND extraction_sha256 = ?", (batch_key, content, extraction)).fetchone():
            return BatchResult(conversation_id, persona_id, True, extraction=record, known_at=known_at)
        by_id = {m.message_id: m for m in msgs}
        ids = [e[0] for e in existing] + sorted({e[1] for e in existing if e[1].startswith("event:")})
        if m221.enabled(self.env):
            candidates = [m221.prenormalize(c) for c in candidates]
        # before verification (it may drop fields): M2.2.2 R13 / R14
        flags = [(bool(c.corrects), {n: getattr(c, n) for n in m222.DATE_FIELDS}) for c in candidates]
        if self.pre_verify is not None:
            candidates = [self.pre_verify(c, msgs, existing) for c in candidates]
        verdicts = [verify(c, msgs, persona_id, ids, strict_self=self.strict_self) for c in candidates]
        if self.post_verify is not None:
            verdicts = [self.post_verify(v, msgs) for v in verdicts]
        if m222.enabled(self.env):
            for (f, dates), v in zip(flags, verdicts):
                m222.restore_explicit_correction(f, v, msgs, persona_id)
                m222.restore_holiday_dates(dates, v, msgs)
        if m221.enabled(self.env):
            people = m22.known_people(existing, self.speakers.values(), candidates)
            events = m22.open_events(existing)
            for v in verdicts:
                if not v.committable:
                    continue
                c2, notes = m221.normalize(v.candidate, by_id, persona_id, people, events, existing)
                v.reasons += notes
                if c2 is None:
                    v.support = UNSUPPORTED
                else:
                    v.candidate = c2
            for i, why in m221.batch_filter(verdicts):
                verdicts[i].reasons.append(why)
                verdicts[i].support = UNSUPPORTED
        elif m22.enabled(self.env):
            people = m22.known_people(existing, self.speakers.values(), candidates)
            events = m22.open_events(existing)
            for v in verdicts:
                if not v.committable:
                    continue
                c2, notes = m22.normalize(v.candidate, by_id, persona_id, people, events)
                v.reasons += notes
                if c2 is None:
                    v.support = UNSUPPORTED
                else:
                    v.candidate = c2
        ops: List[Operation] = []
        with self.api.transaction():
            for n, v in enumerate(verdicts):
                if fail_after is not None and n >= fail_after:
                    raise RuntimeError("injected failure")      # tests: the whole group must roll back
                if v.committable:
                    ops += self.decide(v, by_id, known_at, existing)
                else:
                    ops.append(Operation(NOOP, UNRESOLVED, candidate_id=v.candidate.candidate_id, detail=f"{v.support}: {'; '.join(v.reasons)}"))
            conn.execute("INSERT INTO writer_batches (group_id, persona_id, content_sha256, extraction_sha256, writer_version, known_at, committed_at, operations) VALUES (?,?,?,?,?,?,?,?)",
                         (batch_key, persona_id, content, extraction, WRITER_VERSION, known_at.isoformat(), self.api.store._now(), json.dumps([o.to_dict() for o in ops], ensure_ascii=False)))
        return BatchResult(conversation_id, persona_id, False, ops, verdicts, record, known_at)


def pending_index_ids(api, persona_id: str, indexed_ids) -> List[str]:
    """Active memories of a persona that a dense index does not hold yet: the index is derived and is rebuilt from these, never trusted to be current."""
    rows = api.store._conn.execute("SELECT memory_id FROM memories WHERE persona_id = ? AND status = 'active' ORDER BY memory_id", (persona_id,)).fetchall()
    have = set(indexed_ids)
    return [r[0] for r in rows if r[0] not in have]
