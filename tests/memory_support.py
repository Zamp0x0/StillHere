"""Helpers for the SQLite memory store tests. All data is synthetic; databases are in-memory or in tmp dirs."""
import json
from datetime import datetime, timezone
from pathlib import Path

from conftest import at
from stillhere.contracts import DataOrigin, MemoryRecord, SourceKind, SourceRef
from stillhere.memory import SQLiteMemoryStore

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "synthetic" / "persona_qiaomo"
QIAOMO = "synthetic_qiaomo"
QIAOMO_NOW = datetime(2026, 10, 14, tzinfo=timezone.utc)   # fixed "now" for the fixture persona
NOW = at(365)                                              # fixed "now" for make_memory() records

TEST_SOURCES = (
    SourceRef(source_id="src_1", kind=SourceKind.SCRIPT, origin=DataOrigin.SYNTHETIC, description="Synthetic script one."),
    SourceRef(source_id="src_2", kind=SourceKind.CONVERSATION, origin=DataOrigin.SYNTHETIC,
              description="Synthetic chat two."),
)


def fixture_raw():
    return json.loads((EXAMPLE / "memories.json").read_text(encoding="utf-8"))


def fixture_records():
    return [MemoryRecord.from_dict(r) for r in fixture_raw()]


def fixture_sources():
    persona = json.loads((EXAMPLE / "persona.json").read_text(encoding="utf-8"))
    return [SourceRef.from_dict(s) for s in persona["sources"]]


def open_store(path=":memory:", *, clock=None, sources=TEST_SOURCES, **kwargs) -> SQLiteMemoryStore:
    store = SQLiteMemoryStore(path, clock=clock or (lambda: NOW), **kwargs)
    for s in sources:
        store.register_source(s)
    return store


def qiaomo_store(path=":memory:", **kwargs) -> SQLiteMemoryStore:
    store = open_store(path, clock=lambda: QIAOMO_NOW, sources=fixture_sources(), **kwargs)
    store.import_records(fixture_records())
    return store


TABLES = ("sources", "memories", "provenance", "memory_conflicts", "memory_events")


def snapshot(store: SQLiteMemoryStore) -> dict:
    """Every row of every table (and the keyword index), for before/after comparisons."""
    conn = store._conn
    snap = {t: [tuple(r) for r in conn.execute(f"SELECT * FROM {t} ORDER BY rowid")] for t in TABLES}
    if store._fts_table:
        snap["memory_fts"] = [tuple(r) for r in conn.execute("SELECT memory_id, bigrams, unigrams FROM memory_fts "
                                                             "ORDER BY rowid")]
    return snap
