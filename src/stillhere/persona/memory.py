"""SQLite temporal memory for the D / E ablation groups and for persona chat, built from the TRAIN period of the export only.

The store holds every text message of both people up to the last train target (``train_meta`` max time), never a dev or test message.
Retrieval is the existing EF1 path (``retrieval.ef1.pipeline.retrieve``, BM25 + structured + time channels, no dense / external API)
and the evidence is rendered by ``context.fact_pack`` exactly as the fact reader does (``render_evidence``). The evidence block of each
evaluation item is computed ONCE and stored in the evaluation set, so D and E see byte-identical memory.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional

from . import cleaning as C
from . import dataset as D
from . import identity as I
from . import weflow as W

TZ = timezone(timedelta(hours=8))
MEMORY_VERSION = "stillhere.persona.memory/1.0"
TOP_K = 8
MAX_EVIDENCE_CHARS = 1200
REDACT = ("credential", "id_number", "bank_card", "phone", "email")


def cutoff_from_train(dataset_dir) -> datetime:
    times = [datetime.fromisoformat(m["time"]) for m in D.load_meta(dataset_dir, "train", "persona_prompt")]
    return max(times)


def build_api(source_path, *, persona_id: str, target_username: str, cutoff: datetime, db_path: str = ":memory:", synthetic: bool = False):
    """A MemoryAPI with the train-period text messages of both people (privacy values redacted like the dataset contexts). A real export
    is a PRIVATE store (``public_only=False``, in memory by default, never written into the repository); ``synthetic`` is for test fixtures."""
    from stillhere.contracts import CanonicalMessage, DataOrigin, SourceKind, SourceRef
    from stillhere.memory.api import MemoryAPI
    exp = W.load(source_path)
    api = MemoryAPI(db_path, public_only=synthetic)
    sid = "weflow-" + exp.sha256[:12]
    api.register_source(SourceRef(source_id=sid, kind=SourceKind.CONVERSATION, origin=DataOrigin.SYNTHETIC if synthetic else DataOrigin.REAL,
                                  description="WeFlow chat (train period)", sha256=exp.sha256))
    msgs, n = [], 0
    for m in exp.messages:
        if I.is_system(m) or m.type not in C.TEXT_TYPES or not (m.content or "").strip():
            continue
        ts = datetime.fromtimestamp(m.create_time, TZ)
        if ts > cutoff:
            continue
        speaker = "me" if m.sender == target_username else "other"
        msgs.append(CanonicalMessage(conversation_id="c-main", message_id="m-" + hashlib.sha256(m.platform_id.encode()).hexdigest()[:16],
                                     target_persona_id=persona_id, speaker_id=speaker, timestamp=ts, content=C.redact(m.content, REDACT),
                                     source_type="weflow", source_id=sid, locator=f"msg:{n}"))
        n += 1
    api.ingest_messages(msgs)
    return api, {"messages": len(msgs), "cutoff": cutoff.isoformat(), "source_sha256": exp.sha256, "version": MEMORY_VERSION}


def pack(api, persona_id: str, query: str, ask_time: datetime):
    """The FactPack of the EVALUATION (train-cutoff) view, for the Fact Reader."""
    from stillhere.context import fact_pack as FP
    from stillhere.retrieval.ef1.pipeline import EF1Config, retrieve
    cs = retrieve(api, persona_id, query, ask_time, EF1Config(top_k=TOP_K, adaptive_k=False, use_dense=False))
    return FP.build(api, cs, names={"me": "本人", "other": "对方"})


def evidence(api, persona_id: str, query: str, ask_time: datetime) -> str:
    from stillhere.context import fact_pack as FP
    from stillhere.retrieval.ef1.pipeline import EF1Config, retrieve
    cs = retrieve(api, persona_id, query, ask_time, EF1Config(top_k=TOP_K, adaptive_k=False, use_dense=False))
    pack = FP.build(api, cs, names={"me": "我", "other": "对方"})
    text = FP.render_evidence(pack)
    return text if len(text) <= MAX_EVIDENCE_CHARS else text[:MAX_EVIDENCE_CHARS] + "\n（其余记录因长度限制省略）"


def blocks_for(items, api, persona_id: str) -> Dict[str, str]:
    """``items``: evaluation items with ``query`` (the other person's latest line) and ``time`` (iso). One evidence block per item."""
    return {it["sample_id"]: evidence(api, persona_id, it["query"] or "最近", datetime.fromisoformat(it["time"])) for it in items}
