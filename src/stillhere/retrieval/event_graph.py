"""Lightweight relation index: which records a stored link joins, and one bounded hop along those links.

The index is derived from the snapshot alone and can be rebuilt. It holds groups, not pairs: a record belongs to a group
when its own stored fields say so, and every member of a group is a neighbour of every other. No QA field is read here.

  same_event    the same ``daily_event_id``
  tree_root     event tree nodes under one root (``173-1``, ``173-5``): an explicit chain across events
  conversation  one SMS thread with one contact (an agent chat is already one evidence item)
  entity        a known contact's exact full name, or a participant or speaker that is one
  same_day      the same calendar date: weak, and proof of nothing but the date

Explicit change or correction links are not listed because this corpus has none (no ``supersedes``, no conflict pairs).
A shared source file is not a relation either: it would join every record of a file.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .hybrid_keys import evidence_key

BUILDER_VERSION = "1"
RELATIONS = ("same_event", "tree_root", "conversation", "entity", "same_day")
_EVENT = re.compile(r"daily_event:event_id=([^;]+)")
_NODE = re.compile(r"event_tree:node=([^;]+)")
_CONTACT = re.compile(r"^联系人(.+?)（")
_DAY_FIELD = {"memory": "valid_from_utc", "message": "timestamp_utc"}


@dataclass(frozen=True)
class Member:
    relation: str
    group: str
    key: str
    record_id: str
    provenance: str
    day: Optional[str]


def _rows(conn: sqlite3.Connection, persona_id: str) -> Tuple[list, list, list]:
    memories = conn.execute("SELECT memory_id, statement, valid_from FROM memories WHERE persona_id = ? ORDER BY memory_id", (persona_id,)).fetchall()
    messages = conn.execute("SELECT message_id, content, timestamp, locator, conversation_id, source_type, speaker_id FROM messages "
                            "WHERE target_persona_id = ? ORDER BY message_id", (persona_id,)).fetchall()
    provenance = conn.execute("SELECT p.memory_id, p.source_id, p.locator FROM provenance AS p JOIN memories AS m ON m.memory_id = p.memory_id "
                              "WHERE m.persona_id = ? ORDER BY p.memory_id, p.position", (persona_id,)).fetchall()
    return memories, messages, provenance


def contact_names(memories: Sequence, persona_id: str) -> Dict[str, str]:
    """Exact full names from the persona's own contact records, by the record id that is also a speaker id."""
    names = {}
    for memory_id, statement, _ in memories:
        if ".contact." in memory_id:
            found = _CONTACT.match(statement)
            if found and len(found.group(1)) >= 2:
                names[memory_id] = found.group(1)
    return names


def extract(conn: sqlite3.Connection, persona_id: str) -> List[Member]:
    memories, messages, provenance = _rows(conn, persona_id)
    names = contact_names(memories, persona_id)
    known = sorted(set(names.values()))
    days = {r[0]: (r[2] or "")[:10] or None for r in memories}
    out: List[Member] = []

    def add(relation: str, group: str, record_id: str, why: str, day: Optional[str]) -> None:
        out.append(Member(relation, f"{persona_id}:{relation}:{group}", evidence_key(persona_id, record_id), record_id, why, day))

    for memory_id, source_id, locator in provenance:
        kind = source_id.rsplit(".", 1)[-1]
        if kind == "daily_event" and locator:
            add("same_event", locator.split("=", 1)[1], memory_id, f"{source_id} {locator}", days[memory_id])
        elif kind == "event_tree" and locator:
            add("tree_root", locator.split("=", 1)[1].split("-")[0], memory_id, f"{source_id} {locator}", days[memory_id])
    for message_id, content, timestamp, locator, conversation, source_type, speaker in messages:
        day = timestamp[:10]
        found = _EVENT.search(locator or "")
        if found:
            add("same_event", found.group(1), message_id, locator, day)
        for node in _NODE.findall(locator or ""):
            add("tree_root", node.split("-")[0], message_id, locator, day)
        if source_type.endswith("sms"):
            add("conversation", conversation, message_id, f"conversation {conversation}", day)
        if speaker in names:
            add("entity", names[speaker], message_id, f"speaker {speaker}", day)
        for name in known:
            if name in content:
                add("entity", name, message_id, f"text names {name}", day)
    for memory_id, statement, _ in memories:
        for name in known:
            if name in statement and not memory_id.endswith(".contact." + memory_id.rsplit(".", 1)[-1]):
                add("entity", name, memory_id, f"text names {name}", days[memory_id])
    for record_id, day in list(days.items()) + [(m[0], m[2][:10]) for m in messages]:
        if day:
            add("same_day", day, record_id, f"date {day}", day)
    return sorted(set(out), key=lambda m: (m.relation, m.group, m.key, m.record_id, m.provenance))


class EventGraph:
    """Group membership in memory. ``group_size_limit`` leaves out groups so large that belonging says nothing."""

    def __init__(self, members: Iterable[Member]) -> None:
        self.groups: Dict[str, List[Member]] = defaultdict(list)
        self.of_key: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for m in members:
            self.groups[m.group].append(m)
            groups = self.of_key[(m.key, m.relation)]
            if m.group not in groups:
                groups.append(m.group)

    def groups_of(self, key: str, relation: str) -> List[str]:
        return self.of_key.get((key, relation), [])

    def counts(self) -> dict:
        out = {}
        for relation in RELATIONS:
            sizes = [len({m.key for m in g}) for name, g in self.groups.items() if g[0].relation == relation]
            out[relation] = {"groups": len(sizes), "memberships": sum(len(g) for g in self.groups.values() if g[0].relation == relation),
                             "implied_pairs": sum(n * (n - 1) // 2 for n in sizes), "largest_group": max(sizes, default=0),
                             "groups_with_two_or_more": sum(n >= 2 for n in sizes)}
        return out


def build_index(path: Path, conn: sqlite3.Connection, personas: Sequence[str], *, dataset_revision: str, snapshot_sha256: str, schema_version: int) -> dict:
    """Write the derived index beside nothing else: the snapshot is only read."""
    path.mkdir(parents=True, exist_ok=True)
    database = path / "event_graph.sqlite"
    database.unlink(missing_ok=True)
    out = sqlite3.connect(database)
    out.execute("CREATE TABLE memberships (persona_id TEXT NOT NULL, relation_type TEXT NOT NULL, group_key TEXT NOT NULL, record_key TEXT NOT NULL, "
                "record_id TEXT NOT NULL, provenance TEXT NOT NULL, day TEXT, build_version TEXT NOT NULL)")
    digest = hashlib.sha256()
    members = []
    for pid in personas:
        rows = extract(conn, pid)
        members += rows
        out.executemany("INSERT INTO memberships VALUES (?,?,?,?,?,?,?,?)", [(pid, m.relation, m.group, m.key, m.record_id, m.provenance, m.day, BUILDER_VERSION) for m in rows])
        for m in rows:
            digest.update("\x1f".join((m.relation, m.group, m.key, m.record_id, m.provenance, m.day or "")).encode("utf-8"))
    out.execute("CREATE INDEX ix_persona_key ON memberships (persona_id, record_key, relation_type)")
    out.execute("CREATE INDEX ix_group ON memberships (group_key)")
    out.commit()
    out.close()
    graph = EventGraph(members)
    manifest = {"dataset_revision": dataset_revision, "snapshot_sha256": snapshot_sha256, "schema_version": schema_version, "graph_builder_version": BUILDER_VERSION,
                "record_count": len({m.record_id for m in members}), "membership_count": len(members), "relation_type_counts": graph.counts(),
                "edge_count": sum(v["implied_pairs"] for v in graph.counts().values()), "edge_count_note": "pairs implied by groups; groups are stored, not pairs",
                "not_available": {"explicit_change": "no supersedes and no conflict pairs in this corpus", "shared_source_file": "a file joins all its records, which says nothing"},
                "index_sha256": digest.hexdigest()}
    (path / "index_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


def load_graph(path: Path, *, snapshot_sha256: str, personas: Optional[Sequence[str]] = None) -> Tuple[EventGraph, dict]:
    manifest = json.loads((path / "index_manifest.json").read_text(encoding="utf-8"))
    if manifest["snapshot_sha256"] != snapshot_sha256 or manifest["graph_builder_version"] != BUILDER_VERSION:
        raise ValueError("the relation index was built from another snapshot or builder")
    conn = sqlite3.connect(f"file:{(path / 'event_graph.sqlite').as_posix()}?mode=ro", uri=True)
    rows = conn.execute("SELECT relation_type, group_key, record_key, record_id, provenance, day FROM memberships").fetchall()
    conn.close()
    return EventGraph(Member(*r) for r in rows), manifest
