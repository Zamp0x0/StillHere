"""Evidence units: what one ranked hit stands for once it is shown to a reader.

* A message of a **round** source (an AI chat round: one conversation id per round) becomes the whole round, every message the
  time allows, read through ``MemoryAPI.conversation`` with no count limit. This replaces the frozen ``v3_bench.round_text``,
  which read only the first three turns (D6). Hits on several messages of one round merge into one unit.
* Any other message (an SMS) is one unit with its own sender and recipient.
* A memory hit is its record as believed at ``known_at``; a record retracted or superseded by then is excluded, never shown as current.

Every check goes through ``MemoryAPI``, so persona scope and the ``as_of`` / ``known_at`` rule are the store's own. Nothing here
is written back.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from stillhere.contracts import EvidenceHit, HitKind, MemoryStatus
from stillhere.memory.sqlite_store import PersonaScopeError

import stillhere.context.attribution as at

ROUND_SOURCE_TYPES = frozenset({"lifebench_agent_chat", "synthetic_chat_round"})
TIME_KIND_LABELS = {"said": "发送时间", "occurred": "发生时间", "scheduled": "计划时间", "valid_from": "生效时间"}


@dataclass(frozen=True)
class UnitMessage:
    message_id: str
    position: int                 # 1-based place in the round as shown
    speaker_id: str
    sender: str
    recipient: str
    timestamp: datetime
    recorded_at: Optional[datetime]
    content: str
    source_id: str
    source_type: str
    locator: Optional[str]
    persona_id: str


@dataclass
class Unit:
    key: str
    kind: str                                   # "round" | "message" | "memory"
    persona_id: str
    rank: int
    source_type: str
    source_ids: Tuple[str, ...]
    hit_keys: List[str] = field(default_factory=list)
    hit_message_ids: List[str] = field(default_factory=list)
    conversation_id: Optional[str] = None
    messages: List[UnitMessage] = field(default_factory=list)
    hidden_later: int = 0                       # messages of the round said or learned after as_of / known_at
    order_anomaly: bool = False
    memory_id: Optional[str] = None
    statement: Optional[str] = None
    recorded_at: Optional[datetime] = None
    event_time: Optional[datetime] = None
    time_kind: str = "said"
    day_precision: bool = False                 # the event time is known only to the day
    locator: Optional[str] = None

    @property
    def message_ids(self) -> List[str]:
        return [m.message_id for m in self.messages]


def time_kind_of(source_id: str, kind: str) -> str:
    if kind != "memory":
        return "said"
    name = source_id.lower()
    if "calendar" in name:
        return "scheduled"
    if "daily_event" in name or "event_tree" in name or ".event" in name:
        return "occurred"
    return "valid_from"


def _position(message_id: str) -> Optional[int]:
    tail = message_id.rsplit(".", 1)[-1]
    return int(tail) if tail.isdigit() else None


def _day_precision(locator: Optional[str]) -> bool:
    how = at.locator_field(locator, "time")
    return bool(how) and how.startswith("assumed")


def _messages(raw, persona_id: str, speakers: Mapping[str, str], round_kind: bool) -> List[UnitMessage]:
    participants = [m.speaker_id for m in raw]
    out = []
    for i, m in enumerate(raw, 1):
        if round_kind:
            d = at.round_direction(m.speaker_id, participants, persona_id, speakers)
        else:
            counterpart = at.sms_counterpart(m.conversation_id, persona_id, speakers)
            d = at.sms_direction(m.locator, at.name_of(m.speaker_id, persona_id, speakers), counterpart) or at.Direction(at.name_of(m.speaker_id, persona_id, speakers), at.UNKNOWN)
        out.append(UnitMessage(m.message_id, i, m.speaker_id, d.sender, d.recipient, m.timestamp, m.recorded_at, m.content, m.source_id, m.source_type, m.locator, m.target_persona_id))
    return out


def expand(api, persona_id: str, hits: Sequence[EvidenceHit], keys: Sequence[str], as_of: datetime, known_at: datetime,
           speakers: Mapping[str, str], round_types=ROUND_SOURCE_TYPES) -> Tuple[List[Unit], Dict[str, str], List[dict]]:
    """(units in rank order, excluded hit key -> reason, trace). Exclusions are explicit; nothing is dropped silently."""
    units: List[Unit] = []
    by_key: Dict[str, Unit] = {}
    excluded: Dict[str, str] = {}
    trace: List[dict] = []
    for rank, (hit, key) in enumerate(zip(hits, keys)):
        if hit.persona_id != persona_id:
            excluded[key] = "persona_scope"
            continue
        if hit.kind is HitKind.MESSAGE:
            try:
                message = api.get_message(persona_id, hit.message_id)
            except PersonaScopeError:
                excluded[key] = "persona_scope"
                continue
            if message is None:
                excluded[key] = "missing_record"
                continue
            if message.timestamp > as_of or (message.recorded_at or message.timestamp) > known_at:
                excluded[key] = "temporal_scope"
                continue
            if message.source_type in round_types:
                unit_key = f"round:{message.conversation_id}"
                if unit_key in by_key:
                    by_key[unit_key].hit_keys.append(key)
                    by_key[unit_key].hit_message_ids.append(message.message_id)
                    continue
                raw, hidden = api.conversation(persona_id, message.conversation_id, as_of=as_of, known_at=known_at)
                stored = [_position(m.message_id) for m in raw]
                anomaly = all(p is not None for p in stored) and stored != sorted(stored)
                msgs = _messages(raw, persona_id, speakers, True)
                unit = Unit(key, "round", persona_id, rank, message.source_type, tuple(dict.fromkeys(m.source_id for m in raw)), [key], [message.message_id],
                            message.conversation_id, msgs, hidden, anomaly, event_time=raw[0].timestamp if raw else message.timestamp,
                            day_precision=_day_precision(message.locator), locator=message.locator)
                trace.append({"step": "expand_round", "key": key, "conversation_id": message.conversation_id, "messages": len(msgs), "hidden_later": hidden, "order_anomaly": anomaly})
            else:
                unit_key = f"message:{message.message_id}"
                if unit_key in by_key:
                    by_key[unit_key].hit_keys.append(key)
                    continue
                msgs = _messages([message], persona_id, speakers, False)
                unit = Unit(key, "message", persona_id, rank, message.source_type, (message.source_id,), [key], [message.message_id], message.conversation_id, msgs,
                            event_time=message.timestamp, day_precision=_day_precision(message.locator), locator=message.locator)
            by_key[unit_key] = unit
            units.append(unit)
        else:
            record = api.get(persona_id, hit.memory_id, known_at=known_at)
            if record is None:
                excluded[key] = "temporal_scope"            # not yet recorded at known_at
                continue
            if record.status is not MemoryStatus.ACTIVE:
                excluded[key] = f"not_current:{record.status.value}"
                continue
            if record.recorded_at > known_at:
                excluded[key] = "temporal_scope"
                continue
            unit_key = f"memory:{record.memory_id}"
            if unit_key in by_key:
                by_key[unit_key].hit_keys.append(key)
                continue
            sources = tuple(dict.fromkeys(p.source_id for p in record.provenance))
            unit = Unit(key, "memory", persona_id, rank, sources[0].rsplit(".", 1)[-1], sources, [key], memory_id=record.memory_id, statement=record.statement,
                        recorded_at=record.recorded_at, event_time=record.valid_from, time_kind=time_kind_of(sources[0], "memory"), locator=record.provenance[0].locator)
            by_key[unit_key] = unit
            units.append(unit)
    return units, excluded, trace
