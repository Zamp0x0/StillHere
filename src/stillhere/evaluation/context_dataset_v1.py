"""Public synthetic memory world for the Simple Context v1 multi-turn tests (``ctx-dataset-1``): one invented persona and a contact.
Long-term memory (L3) is written from these conversations by the gold extractor + writer rules; the live-chat turns of the tests are
separate (sessions) and never written to memory. No real person, place-of-residence or message is described."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import List

from stillhere.contracts import CanonicalMessage, DataOrigin, SourceKind, SourceRef

from .writer_dataset import C, Checkpoint, Conv, Fact, ObsExpect  # noqa: F401

VERSION = "ctx-dataset-1"
TZ = timezone(timedelta(hours=8))
CA, CB = "syn.ca", "syn.cb"
NAMES = {CA: "何舟", CB: "沈禾"}
BOT = {p: f"{p}.assistant" for p in NAMES}
LIN = f"{CA}.contact.lin"
CONTACTS = {LIN: "小林"}
SPEAKERS = {**{b: "AI 助手" for b in BOT.values()}, **CONTACTS}
CHAT, SMS = "ctx1.src.chat", "ctx1.src.sms"


def t(month: int, day: int, hour: int = 10, minute: int = 0) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=TZ)


def E(key, subject, obj, src, spans, alts, ekey, status, activity, planned=None, event_time=None, daypart=None, **kw):
    return Fact(key, subject, "event", obj, src, spans, alts, "EVENT", ekey, status, activity, planned, event_time, daypart, **kw)


def conversations() -> List[Conv]:
    return [
        C("m01", CA, t(9, 1, 21), [("u", "我下周要去深圳出差。"), ("a", "一路顺利。")],
          [E("ca.sz", "self", "去深圳出差", (0,), ("我下周要去深圳出差",), ("深圳",), "sz_trip", "PLANNED", "出差")], tags=("plan",)),
        C("m02", CA, t(9, 3, 21), [("u", "深圳出差的酒店订好了，住福田。"), ("a", "好的。")],
          [E("ca.sz_hotel", "self", "订深圳出差的酒店", (0,), ("深圳出差的酒店订好了",), ("酒店",), "sz_hotel", "SCHEDULED", "订酒店")], tags=("schedule",)),
        C("m03", CA, t(9, 4, 20), [(LIN, "我下个月搬去成都。"), ("u", "到了说一声。")],
          [E("ca.lin_move", "person:小林", "搬去成都", (0,), ("我下个月搬去成都",), ("成都",), "lin_move", "PLANNED", "搬家", "2026-10")],
          forbidden=(("lives_in", "成都"),), tags=("sender_recipient",), kind="sms"),
        C("m04", CA, t(9, 5, 21), [("u", "我以前住武汉，现在搬到广州了。"), ("a", "广州挺好。")],
          [Fact("ca.home", "self", "lives_in", "广州", (0,), ("现在搬到广州了",), ())], tags=("change_over_time",)),
        C("m05", CA, t(9, 6, 21), [("u", "国庆去杭州的酒店我已经取消了。"), ("a", "好的。")],
          [E("ca.hz_hotel", "self", "订杭州的酒店", (0,), ("去杭州的酒店我已经取消了",), ("杭州", "酒店"), "hz_hotel", "CANCELLED", "订酒店")], tags=("cancel",)),
        C("m06", CB, t(9, 2, 21), [("u", "我在一家书店上班。"), ("a", "好的。")],
          [Fact("cb.work", "self", "works_at", "一家书店", (0,), ("我在一家书店上班",), ("书店",))], tags=("simple",)),
    ]


def checkpoints() -> List[Checkpoint]:
    return []


def observation_expectations() -> List[ObsExpect]:
    return []


def message_id(conv: Conv, i: int) -> str:
    return f"{conv.persona}.{conv.kind}.{conv.cid}.{i}"


def group_id(conv: Conv) -> str:
    if conv.kind == "chat":
        return f"{conv.persona}.{conv.kind}.{conv.cid}"
    contact = next(m.who for m in conv.messages if m.who not in ("u", "a"))
    return f"{conv.persona}.sms.{contact.rsplit('.', 1)[-1]}"


def raw_messages(conv: Conv) -> List[CanonicalMessage]:
    out = []
    for i, m in enumerate(conv.messages):
        speaker = conv.persona if m.who == "u" else BOT[conv.persona] if m.who == "a" else m.who
        loc = f"pos={i}" + ((";type=" + ("发送" if m.who == "u" else "接收")) if conv.kind == "sms" else "")
        out.append(CanonicalMessage(conversation_id=group_id(conv), message_id=message_id(conv, i), target_persona_id=conv.persona, speaker_id=speaker, timestamp=m.at,
                                    content=m.text, source_type="synthetic_chat_round" if conv.kind == "chat" else "synthetic_sms", source_id=CHAT if conv.kind == "chat" else SMS,
                                    locator=loc, recorded_at=m.recorded_at or m.at))
    return out


def sources() -> List[SourceRef]:
    return [SourceRef(source_id=CHAT, kind=SourceKind.CONVERSATION, origin=DataOrigin.SYNTHETIC, description="Simple Context v1 synthetic dataset: chats."),
            SourceRef(source_id=SMS, kind=SourceKind.CONVERSATION, origin=DataOrigin.SYNTHETIC, description="Simple Context v1 synthetic dataset: SMS.")]


def load(api) -> None:
    for s in sources():
        api.register_source(s)
    for conv in conversations():
        api.ingest_messages(raw_messages(conv))


def order() -> List[Conv]:
    return sorted(conversations(), key=lambda c: (max(m.recorded_at or m.at for m in c.messages), c.cid))


def manifest() -> dict:
    cs = conversations()
    body = {"version": VERSION, "conversations": [asdict(c) for c in cs], "checkpoints": [asdict(x) for x in checkpoints()]}
    digest = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return {"version": VERSION, "dataset_sha256": digest, "personas": NAMES, "conversations": len(cs), "messages": sum(len(c.messages) for c in cs),
            "gold_facts": sum(len(c.facts) for c in cs), "forbidden": sum(len(c.forbidden) for c in cs), "origin": "synthetic"}
