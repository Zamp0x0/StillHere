"""StillHere V2L / V2D over the canonical timeline: the PRODUCT memory pipeline (``persona.product_config``), evidence only, no Writer.

* ``StillHereMemory.add_day(events)``: the day's events become canonical messages of ONE conversation per day (the same unit the
  evidence windows of the persona archive use), times = the day at 12:00 (+8) plus the position in seconds (order kept); recorded at
  the same time. Added strictly in date order by the runner.
* ``search(query, at)``: ``memory_v2.pack_v2`` with the product V2Config (k 8, neighbours (2, 2), date mode prefer); V2D adds the dense
  channel. ``at`` = the ask date 23:59:59 +08:00, so the ask day's events are visible (official date order) and nothing later; every
  pack is checked: no item may be later than ``at`` (``FutureEvidence`` otherwise).
* Dense index (V2D): built ONCE over a separate store holding the whole timeline (``build_dense``); its row-time sidecar masks every row
  that is not visible at the ask time, so the prebuilt index never shows a later event (``ef1.dense_index.SidecarScope``).
"""
from __future__ import annotations

import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

from . import context as CX

TZ = timezone(timedelta(hours=8))
SOURCES = ("agent_chat", "sms", "call", "calendar", "note", "photo", "push", "fitness_health", "unknown")


class FutureEvidence(AssertionError):
    pass


def ask_at(ask_time: str) -> datetime:
    return datetime.fromisoformat(f"{ask_time[:10]}T23:59:59").replace(tzinfo=TZ)


def event_dt(e: Mapping) -> datetime:
    return datetime.fromisoformat(f"{e['date']}T12:00:00").replace(tzinfo=TZ) + timedelta(seconds=int(e["order"]))


class StillHereMemory:
    def __init__(self, db_path: Path, slug: str, person_name: str, assistant_name: str, *, dataset_sha256: str, locator: str):
        from stillhere.contracts import SourceRef
        from stillhere.contracts.common import DataOrigin, SourceKind
        from stillhere.memory.api import MemoryAPI
        self.pid = f"lb2.{slug}"
        self.person, self.assistant = person_name, assistant_name
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path)
        self.api = MemoryAPI(str(db_path), persona_names={self.pid: [person_name]})
        for src in SOURCES:
            sid = f"{self.pid}.src.{src}"
            if self.api.get_source(self.pid, sid) is None:
                self.api.register_source(SourceRef(source_id=sid, kind=SourceKind.SIMULATION, origin=DataOrigin.SYNTHETIC,
                                                   description=f"LifeBench 2.0 (synthetic) {src} records of {person_name}", locator=locator,
                                                   sha256=dataset_sha256))
        self.names = {self.pid: person_name, f"{self.pid}.assistant": "AI助手"}

    def speaker_id(self, speaker: str) -> str:
        if speaker == self.person:
            return self.pid
        if speaker == self.assistant:
            return f"{self.pid}.assistant"
        import hashlib
        return f"{self.pid}.ext.{hashlib.sha256(speaker.encode('utf-8')).hexdigest()[:10]}"

    def messages(self, events: Sequence[Mapping]):
        from stillhere.contracts import CanonicalMessage
        out = []
        for e in events:
            at = event_dt(e)
            src = e["source_type"] if e["source_type"] in SOURCES else "unknown"
            out.append(CanonicalMessage(conversation_id=f"{self.pid}.day.{e['date']}", message_id=f"{self.pid}.{e['event_id']}", target_persona_id=self.pid,
                                        speaker_id=self.speaker_id(e["speaker"]), timestamp=at, content=e["content"], source_type=f"lifebench_{src}",
                                        source_id=f"{self.pid}.src.{src}", locator=e["dia_id"], recorded_at=at))
        return out

    def add_day(self, events: Sequence[Mapping]) -> dict:
        if not events:
            return {"added": 0}
        r = self.api.ingest_messages(self.messages(events))
        return {"added": r.added, "messages": len(events)}

    def message_count(self) -> int:
        return int(self.api.store._conn.execute("SELECT count(*) FROM messages WHERE target_persona_id = ?", (self.pid,)).fetchone()[0])

    def latest_time(self) -> Optional[str]:
        row = self.api.store._conn.execute("SELECT max(timestamp_utc) FROM messages WHERE target_persona_id = ?", (self.pid,)).fetchone()
        return row[0] if row else None

    def future_messages(self, at: datetime) -> int:
        at_utc = at.astimezone(timezone.utc).isoformat()
        return int(self.api.store._conn.execute("SELECT count(*) FROM messages WHERE target_persona_id = ? AND timestamp_utc > ?",
                                                (self.pid, at_utc)).fetchone()[0])

    def search(self, query: str, at: datetime, *, dense=None) -> dict:
        """-> {"entries": [Entry], "meta": {...}}. ``dense`` given = V2D, else V2L."""
        from stillhere.persona import memory_v2 as V2
        from stillhere.persona import product_config as PC
        mp = PC.PRODUCT_MEMORY_PIPELINE
        cfg = V2.V2Config(use_dense=dense is not None, k=int(mp["k"]), neighbors=tuple(mp["neighbors"]), date_mode=mp["date_mode"],
                          strip_qform=bool(mp["strip_qform"]))
        import time
        t0 = time.perf_counter()
        pk, meta = V2.pack_v2(self.api, self.pid, query, at, cfg, dense=dense, names=self.names)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        limit = at.astimezone(TZ).isoformat()[:19]
        late = [i.eid for i in pk.items if (i.event_time or "")[:19] > limit]
        if late:
            raise FutureEvidence(f"items after the ask time: {late}")
        entries = []
        for i in pk.items:
            day = (i.event_time or "")[:10] or None
            ids = []
            if i.unit_key:
                ids.append(i.unit_key)
            entries.append(CX.Entry(rank=int(i.rank), date=day, source=i.source_label, text=i.text, ids=ids + [m for m in (i.message_ids or []) if m not in ids],
                                    score=float(i.score) if i.score is not None else None))
        route = V2.route_v2(query, at, "p1")
        return {"entries": entries, "meta": {"retrieval_ms": ms, "items": len(pk.items), "dense": bool(meta.get("dense")), "ef1_weights": meta.get("ef1_weights"),
                                             "date_notes": meta.get("date_notes"), "neighbors_added": meta.get("neighbors_added"),
                                             "router_would_route": route["route"], "router_why": route["why"]}}

    def snapshot(self, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".partial")
        if tmp.exists():
            tmp.unlink()
        self.api.snapshot(tmp)
        shutil.move(str(tmp), str(dest))
        return dest

    def close(self) -> None:
        self.api.close()


def build_dense(index_store: Path, index_dir: Path, events: Sequence[Mapping], slug: str, person: str, assistant: str, embedder, *,
                dataset_sha256: str, locator: str, batch_size: int = 32, progress=None) -> dict:
    """The V2D index over the WHOLE timeline (a separate store; never the run store). Resumable through the embedding cache."""
    from stillhere.persona import product_config as PC
    from stillhere.retrieval.ef1 import local_dense as LD
    m = StillHereMemory(index_store, slug, person, assistant, dataset_sha256=dataset_sha256, locator=locator)
    try:
        if m.message_count() == 0:
            from .timeline import by_date
            for _, evs in sorted(by_date(events).items()):
                m.add_day(evs)
        man = Path(index_dir) / "index" / "embedding_index_manifest.json"
        if man.is_file():
            return {"status": "COMPLETE", "reused": True, "messages": m.message_count()}
        res = LD.build_local(m.api, m.pid, Path(index_dir), embedder, names=m.names, model=PC.EMBEDDING["cache_model"], dimension=PC.EMBEDDING["dimension"],
                             batch_size=batch_size, progress=progress)
        res["messages"] = m.message_count()
        return res
    finally:
        m.close()


def open_dense(index_dir: Path, encoder):
    from stillhere.persona import product_config as PC
    from stillhere.retrieval.ef1 import local_dense as LD
    return LD.open_local(Path(index_dir), model=PC.EMBEDDING["cache_model"], dimension=PC.EMBEDDING["dimension"], query_encoder=encoder)


def gold_ids_in(entries: Iterable[CX.Entry]) -> List[str]:
    """dia_ids (date prefix removed) present in the entries: the strict StillHere evidence-ID recall."""
    from .timeline import gold_id
    out = []
    for e in entries:
        for i in e.ids:
            if "." not in i:
                out.append(gold_id(i))
    return out
