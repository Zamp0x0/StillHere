"""LifeBench 2.0 (Chinese ``data/``): metadata scan, frozen selection of two personas, and the mapping to contracts.

What each file becomes (docs/LIFEBENCH_IMPORT.md has the reasons):

  ``agent_chat``, ``sms``     raw messages. Nothing else is a message: a call has no words, a note has no speaker.
  ``daily_event``             one EVENT memory per event, the main structured source.
  ``calendar``, ``note``, ``photo``, ``push``, ``call``, ``fitness_health``    EVENT memories with no speaker.
  ``contact``                 one RELATIONSHIP memory per contact.
  ``persona``                 a few profile FACT memories; the rest stays in the source.
  ``event_tree``              links only: a daily event cites the tree nodes it belongs to.
  ``daily_draft``             a registered source only: a day-by-day outline of the same events.
  ``QA_all``                  never read here, except question types and counts for the selection scan.

Times in the dataset have no offset. Every persona lives in UTC+8, so they are read as +08:00 (``DATASET_TZ``); the
original text stays in the statement or locator. The dataset gives no knowledge time, so ``recorded_at`` follows
``TIME_POLICY``: an engineering assumption, recorded in the import manifest, not an official field.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from stillhere.contracts import (CanonicalMessage, DataOrigin, MemoryKind, MemoryRecord, Provenance, SourceKind,
                                 SourceRef, canonical_json)
from stillhere.memory import MemoryAPI

from .base import ImportReport, batches, parse_naive, stable_id, utc_offset

DATASET = "C1754955896/Lifebenchv2.0"
DATASET_TZ = utc_offset(8)
BATCH_SIZE = 2000
PERSONA_PREFIX = "lifebench_"
PERSONA_COUNT = 2
SELECTION_RULE_VERSION = 1
BASELINE = datetime(2025, 1, 1, tzinfo=DATASET_TZ)     # the first day of the dataset

TIME_POLICY = {
    "timezone": "+08:00: the dataset has no offsets; all ten personas live in UTC+8 (engineering assumption)",
    "daily_event": "assumed_known_at = end of the event; valid time is the event's own interval",
    "calendar": "recorded_at = the entry's `datetime` field; valid time is start_time..end_time",
    "note_photo_push": "recorded_at = the record's `datetime`; valid_from = the same, open-ended (the record exists from then)",
    "call": "recorded_at = call end; valid time is the call",
    "fitness_health": "assumed_known_at = end of that day; valid time is that day",
    "sms": "timestamp = `datetime`; recorded_at = timestamp",
    "agent_chat": "the record has only a date in most cases: timestamp = end of the linked daily event on that date, "
                  "else 23:59:00 of the date, plus one second per message for order; recorded_at = timestamp",
    "persona_contact": "assumed_known_at = 2025-01-01T00:00:00+08:00 (dataset start); no valid time given",
}

_FILES = ("persona.json", "daily_event.json", "event_tree.json", "daily_draft.json")
_PHONE = ("agent_chat", "calendar", "call", "contact", "fitness_health", "note", "photo", "push", "sms")
_CHANGE_TYPES = ("Knowledge_update", "Temporal", "Conflict")
_PROFILE_FIELDS = ("name", "birth", "gender", "education", "job", "occupation", "belief", "family")
_RANGE = "至"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def persona_id(directory: str) -> str:
    return PERSONA_PREFIX + directory


def _address(a: Mapping) -> str:
    return "".join(str(a.get(k) or "") for k in ("province", "city", "district", "street_name", "street_number"))


def _digits(phone: Optional[str]) -> str:
    return re.sub(r"\D", "", phone or "")[-11:]


def _event_time(text: str, report: ImportReport) -> datetime:
    """A wall-clock time. Some events run past midnight on the same date (``26:15:00``, ``24:00:00``): that hour is
    read as the next day, and counted. The original text stays in the statement."""
    match = re.fullmatch(r"(\d{4}-\d{2}-\d{2}) (\d{2})(:\d{2}:\d{2})", text.strip())
    if match and int(match.group(2)) >= 24:
        report.note("repaired_hour_past_24")
        hour = int(match.group(2)) - 24
        return parse_naive(f"{match.group(1)} {hour:02d}{match.group(3)}", DATASET_TZ) + timedelta(days=1)
    return parse_naive(text, DATASET_TZ)


def _ranges(dates: Sequence[str], report: ImportReport) -> Tuple[datetime, datetime]:
    """The whole span of an event's date ranges. An event listed in several parts keeps its text in the statement.
    A range that ends earlier on the same date than it starts crossed midnight (``22:10`` to ``00:10``): its end moves
    to the next day, and is counted. Any other reversed range stays reversed, and is an error in ``_interval``."""
    starts, ends = [], []
    for text in dates:
        first, last = text.split(_RANGE)
        start, end = _event_time(first, report), _event_time(last, report)
        if end < start and end.date() == start.date():
            report.note("repaired_range_past_midnight")
            end += timedelta(days=1)
        starts.append(start)
        ends.append(end)
    return min(starts), max(ends)


def _interval(start: datetime, end: datetime, report: ImportReport) -> datetime:
    """``valid_to`` of a half-open interval: an instant gets one second, a reversed interval is an error."""
    if end < start:
        raise ValueError(f"interval ends before it starts: {start.isoformat()} .. {end.isoformat()}")
    if end == start:
        report.note("instant_given_one_second")
        return start + timedelta(seconds=1)
    return end


# ---------------------------------------------------------------- metadata scan and selection

def _tree_nodes(nodes: Iterable[Mapping]) -> Iterable[Tuple[Mapping, int]]:
    stack = [(n, 0) for n in nodes]
    while stack:
        node, depth = stack.pop()
        yield node, depth
        stack.extend((s, depth + 1) for s in node.get("subevent") or [])


def _spans_days(node: Mapping) -> bool:
    return len(node.get("date") or []) > 1 or any(_RANGE in d and d.split(_RANGE)[0][:10] != d.split(_RANGE)[1][:10]
                                                  for d in node.get("date") or [])


def _qa_metadata(path: Path) -> dict:
    """Question types and counts only. Questions, answers, evidence and score points are not looked at."""
    types: Counter = Counter()
    rows = _load(path)
    for row in rows:
        types.update(row["question_type"])
        del row
    return {"qa_count": len(rows), "qa_types": dict(sorted(types.items()))}


def scan_persona(directory: Path) -> dict:
    """The selection statistics of one persona: data shape only, nothing read from what the data says."""
    persona = _load(directory / "persona.json")
    events = _load(directory / "daily_event.json")
    tree = _load(directory / "event_tree.json")
    phone = {name: _load(directory / "phone_data" / f"{name}.json") for name in _PHONE}
    nodes = list(_tree_nodes(tree))
    node_ids = {str(n["event_id"]) for n, _ in nodes}
    dates = {d[:10] for e in events for d in e["date"]}
    dates |= {str(r.get("date") or r.get("datetime") or r.get("start_time") or r.get("日期"))[:10]
              for name, rows in phone.items() if name != "contact" for r in rows}
    linked = [r for name, rows in phone.items() if name not in ("contact", "fitness_health") for r in rows]
    atoms = [str(a) for e in events for a in e["atomic_id"]]
    duplicates = 0
    for rows in phone.values():
        seen = Counter(canonical_json({k: v for k, v in r.items() if k != "phone_id"}) for r in rows)
        duplicates += sum(n - 1 for n in seen.values())
    seen_events = Counter(canonical_json({k: e[k] for k in ("name", "date", "description")}) for e in events)
    duplicates += sum(n - 1 for n in seen_events.values())
    qa = _qa_metadata(directory / "QA_all" / "QA.json")
    other = sum(len(phone[n]) for n in ("call", "photo", "push", "fitness_health", "contact"))
    event_dates = sorted(d[:10] for e in events for d in e["date"])
    return {
        "persona_id": persona_id(directory.name), "directory": directory.name, "display_name": persona["name"],
        "home_city": persona["home_address"]["city"], "gender": persona["gender"],
        "first_date": event_dates[0], "last_date": event_dates[-1], "dates_with_records": len(dates - {"None"}),
        "raw_records": len(events) + sum(len(r) for r in phone.values()), "daily_events": len(events),
        "agent_chat_records": len(phone["agent_chat"]), "sms": len(phone["sms"]), "calendar": len(phone["calendar"]),
        "note": len(phone["note"]), "other_phone": other,
        "event_type_counts": dict(sorted(Counter(e["type"] for e in events).items())),
        "event_link_rate": round(sum(1 for r in linked if r.get("daily_event_id")) / max(len(linked), 1), 4),
        "tree_link_rate": round(sum(1 for a in atoms if a in node_ids) / max(len(atoms), 1), 4),
        "source_id_rate": round(sum(1 for rows in phone.values() for r in rows if r.get("phone_id") is not None)
                                / max(sum(len(r) for r in phone.values()), 1), 4),
        "duplicate_records": duplicates, "tree_nodes": len(nodes),
        "multi_day_chains": sum(1 for n, _ in nodes if _spans_days(n)), **qa,
        "change_qa_count": sum(qa["qa_types"].get(t, 0) for t in _CHANGE_TYPES),
    }


def _type_shares(stats: Mapping) -> Dict[str, float]:
    total = sum(stats["event_type_counts"].values())
    return {k: v / total for k, v in stats["event_type_counts"].items()}


def _distance(a: Mapping, b: Mapping) -> float:
    x, y = _type_shares(a), _type_shares(b)
    return sum(abs(x.get(k, 0.0) - y.get(k, 0.0)) for k in set(x) | set(y))


_SCORE_FIELDS = ("raw_records", "qa_count", "change_qa_count", "multi_day_chains", "tree_link_rate", "event_link_rate")


def select_personas(stats: Sequence[Mapping]) -> dict:
    """Two personas by a fixed rule. Eligible: every source file has records and the history spans the whole year.
    Score: the sum of each persona's rank over ``_SCORE_FIELDS``; ties break by persona id. The first is the best
    score; the second is the best score in another city whose event-type mix is at least as far from the first as
    the median persona's. Only data shape is used: no answer, job or hobby."""
    eligible = [s for s in stats if s["first_date"] <= "2025-01-02" and s["last_date"] >= "2025-12-30"
                and min(s["daily_events"], s["sms"], s["agent_chat_records"], s["calendar"], s["note"], s["qa_count"]) > 0]
    if len(eligible) < PERSONA_COUNT:
        raise ValueError(f"only {len(eligible)} personas are eligible")
    scores = {s["persona_id"]: 0 for s in eligible}
    for name in _SCORE_FIELDS:
        ordered = sorted({s[name] for s in eligible})
        for s in eligible:
            scores[s["persona_id"]] += ordered.index(s[name])
    ranked = sorted(eligible, key=lambda s: (-scores[s["persona_id"]], s["persona_id"]))
    first = ranked[0]
    distances = {s["persona_id"]: round(_distance(first, s), 6) for s in ranked[1:]}
    values = sorted(distances.values())
    median = values[len(values) // 2]
    second = next((s for s in ranked[1:] if s["home_city"] != first["home_city"]
                   and distances[s["persona_id"]] >= median), ranked[1])
    return {"selected": [first["persona_id"], second["persona_id"]], "scores": scores, "distances_from_first": distances,
            "median_distance": median, "score_fields": list(_SCORE_FIELDS), "rule_version": SELECTION_RULE_VERSION}


def _dataset_fingerprint(data_root: Path) -> str:
    """The hash of every file the selection and the import read, so a changed dataset is noticed."""
    digest = hashlib.sha256()
    for path in sorted(p for p in data_root.rglob("*.json")):
        digest.update(path.relative_to(data_root).as_posix().encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def ensure_selection(data_root: Path, manifest_path: Path, revision: str, *, reselect: bool = False) -> dict:
    """The frozen selection. An existing manifest is reused; one from another revision or dataset state is an error
    unless ``reselect`` asks to replace it, so a re-run can never change the personas silently."""
    fingerprint = _dataset_fingerprint(data_root)
    if manifest_path.exists() and not reselect:
        manifest = _load(manifest_path)
        if manifest["revision"] != revision or manifest["data_fingerprint"] != fingerprint:
            raise ValueError(f"{manifest_path} was made for revision {manifest['revision']}, the data here is "
                             f"{revision}: pass the matching data, or --reselect to choose again")
        return manifest
    stats = [scan_persona(p) for p in sorted(data_root.iterdir()) if p.is_dir()]
    manifest = {"dataset": DATASET, "revision": revision, "data_fingerprint": fingerprint,
                "personas_scanned": len(stats), **select_personas(stats), "stats": stats}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------- mapping one persona to contracts

@dataclass
class ParsedPersona:
    persona_id: str
    display_name: str
    sources: List[SourceRef]
    messages: List[CanonicalMessage]
    facts: List[MemoryRecord]
    speakers: Dict[str, dict]
    report: ImportReport = field(default_factory=ImportReport)


class _Mapper:
    def __init__(self, directory: Path, revision: str) -> None:
        self.dir = directory
        self.revision = revision
        self.pid = persona_id(directory.name)
        self.prefix = f"lb.{directory.name}"
        self.report = ImportReport()
        self.sources: Dict[str, SourceRef] = {}
        self.messages: List[CanonicalMessage] = []
        self.facts: List[MemoryRecord] = []
        self.speakers: Dict[str, dict] = {}
        self.seen: set = set()
        self.events: Dict[str, Tuple[datetime, datetime]] = {}
        self.contacts_by_name: Dict[str, str] = {}
        self.contacts_by_phone: Dict[str, str] = {}
        self.assistant = f"{self.prefix}.assistant"
        self.tree_nodes: set = set()

    def source(self, rel: str, kind: SourceKind) -> str:
        path = self.dir / rel
        stem = rel.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        source_id = f"{self.prefix}.{stem}"
        self.sources[stem] = SourceRef(
            source_id=source_id, kind=kind, origin=DataOrigin.SYNTHETIC, sha256=_sha256(path),
            description=f"LifeBench 2.0 中文 {self.dir.name}/{rel}",
            locator=f"hf://datasets/{DATASET}@{self.revision}/data/{self.dir.name}/{rel}")
        return source_id

    def fresh(self, *key: object) -> bool:
        """False for a record identical to one already taken, which is counted and kept once."""
        if key in self.seen:
            self.report.duplicate_count += 1
            return False
        self.seen.add(key)
        return True

    def each(self, file: str, rows: Iterable[Mapping], handle: Callable[[Mapping], None]) -> None:
        """Run one record through its mapping; a record that cannot be read is listed, the others go on."""
        for row in rows:
            self.report.read(file)
            try:
                handle(row)
            except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
                label = row.get("phone_id", row.get("event_id", "?")) if isinstance(row, Mapping) else "?"
                self.report.error(f"{self.pid} {file} {label}: {type(exc).__name__}: {exc}")

    def fact(self, memory_id: str, kind: MemoryKind, attribute: str, value: str, statement: str,
             provenance: Sequence[Provenance], recorded_at: datetime, *, subject: str = "persona",
             valid_from: Optional[datetime] = None, valid_to: Optional[datetime] = None) -> None:
        self.facts.append(MemoryRecord(memory_id=memory_id, persona_id=self.pid, kind=kind, subject=subject,
                                       attribute=attribute, value=value, statement=statement, provenance=provenance,
                                       origin=DataOrigin.SYNTHETIC, recorded_at=recorded_at, valid_from=valid_from,
                                       valid_to=valid_to))

    def daily_link(self, row: Mapping) -> List[Provenance]:
        """Provenance back to the daily event and tree nodes a phone record is linked to."""
        links = []
        event = str(row.get("daily_event_id") or "")
        if event in self.events:
            links.append(Provenance(source_id=self.sources["daily_event"].source_id, locator=f"event_id={event}"))
        for node in row.get("event_id") or []:
            if str(node) in self.tree_nodes:
                links.append(Provenance(source_id=self.sources["event_tree"].source_id, locator=f"node={node}"))
        return links

    # -------------------------------------------------------------- persona and contacts

    def persona(self) -> str:
        data = _load(self.dir / "persona.json")
        source = self.source("persona.json", SourceKind.DOCUMENT)
        self.speakers[self.pid] = {"name": data["name"], "relation": "目标人物本人"}
        self.speakers[self.assistant] = {"name": "AI 助手", "relation": "agent_chat 对话方"}
        rows = [(k, str(data[k])) for k in _PROFILE_FIELDS if data.get(k)]
        rows += [(k, _address(data[k])) for k in ("home_address", "workplace", "birth_place") if data.get(k)]
        rows += [(k, "、".join(data[k])) for k in ("hobbies", "favorite_foods", "aim") if data.get(k)]
        self.report.read("persona.json", len(rows))
        labels = {"name": "姓名", "birth": "出生日期", "gender": "性别", "education": "教育背景", "job": "工作",
                  "occupation": "单位或职业", "belief": "信仰", "family": "家庭情况", "home_address": "家庭住址",
                  "workplace": "工作地点", "birth_place": "出生地", "hobbies": "爱好", "favorite_foods": "喜欢的食物",
                  "aim": "目标"}
        for key, value in rows:
            self.fact(f"{self.prefix}.persona.{key}", MemoryKind.FACT, f"profile:{key}", value,
                      f"{data['name']}的{labels[key]}：{value}",
                      [Provenance(source_id=source, locator=f"field={key}")], BASELINE)
        self.report.keep_source_only("persona_other_fields", len(data) - len(rows))
        return data["name"]

    def contacts(self) -> None:
        source = self.source("phone_data/contact.json", SourceKind.DOCUMENT)

        def handle(c: Mapping) -> None:
            speaker = f"{self.prefix}.contact.{c['phone_id']}"
            self.contacts_by_name.setdefault(c["name"], speaker)
            if _digits(c.get("phoneNumber")):
                self.contacts_by_phone.setdefault(_digits(c["phoneNumber"]), speaker)
            self.speakers[speaker] = {"name": c["name"], "relation": c.get("relation")}
            nickname = f"，昵称{c['nickname']}" if c.get("nickname") else ""
            phone = f"，电话{c['phoneNumber']}" if c.get("phoneNumber") else ""
            self.fact(f"{self.prefix}.contact.{c['phone_id']}", MemoryKind.RELATIONSHIP, "contact:relation",
                      str(c.get("relation") or "联系人"), f"联系人{c['name']}（{c.get('relation')}）{nickname}{phone}",
                      [Provenance(source_id=source, locator=f"phone_id={c['phone_id']}")], BASELINE,
                      subject=speaker)

        self.each("contact.json", _load(self.dir / "phone_data" / "contact.json"), handle)

    # -------------------------------------------------------------- events

    def daily_events(self) -> None:
        source = self.source("daily_event.json", SourceKind.DOCUMENT)
        tree = self.source("event_tree.json", SourceKind.DOCUMENT)
        self.tree_nodes = {str(n["event_id"]) for n, _ in _tree_nodes(_load(self.dir / "event_tree.json"))}
        self.source("daily_draft.json", SourceKind.DOCUMENT)
        draft = _load(self.dir / "daily_draft.json")
        self.report.keep_source_only("daily_draft_days", sum(len(v) for v in draft.values()))

        def handle(e: Mapping) -> None:
            event_id = str(e["event_id"])
            start, end = _ranges(e["date"], self.report)
            valid_to = _interval(start, end, self.report)
            self.events[event_id] = (start, valid_to)
            if not self.fresh("daily_event", e["name"], tuple(e["date"]), e["description"]):
                return
            others = [p["name"] for p in e.get("participant") or [] if p.get("relation") != "自己"]
            with_whom = f"；参与者：{'、'.join(others)}" if others else ""
            statement = (f"{e['name']}：{e['description']}（时间：{'；'.join(e['date'])}；地点：{e.get('location')}"
                         f"{with_whom}）")
            provenance = [Provenance(source_id=source, locator=f"event_id={event_id}", quote=e["name"],
                                     observed_at=start)]
            atoms = [str(a) for a in e.get("atomic_id") or []]
            provenance += [Provenance(source_id=tree, locator=f"node={a}") for a in atoms if a in self.tree_nodes]
            missing = sum(1 for a in atoms if a not in self.tree_nodes)
            if missing:
                self.report.note("atomic_id_not_in_event_tree", missing)
            self.fact(f"{self.prefix}.daily_event.{event_id}", MemoryKind.EVENT, f"event:{event_id}", e["name"],
                      statement, provenance, valid_to, valid_from=start, valid_to=valid_to)

        self.each("daily_event.json", _load(self.dir / "daily_event.json"), handle)
        linked = {str(p.locator)[5:] for f in self.facts for p in f.provenance if p.source_id == tree}
        self.report.keep_source_only("event_tree_nodes_without_memory", len(self.tree_nodes - linked))

    def phone_records(self) -> None:
        phone = self.dir / "phone_data"

        def rows(name: str) -> List[Mapping]:
            self.source(f"phone_data/{name}.json", SourceKind.DOCUMENT)
            return _load(phone / f"{name}.json")

        def own(name: str, row: Mapping, quote: Optional[str] = None, when: Optional[datetime] = None) -> List[Provenance]:
            return [Provenance(source_id=self.sources[name].source_id, locator=f"phone_id={row['phone_id']}", quote=quote,
                               observed_at=when), *self.daily_link(row)]

        def label(r: Mapping) -> str:
            """The record's title; a few have none, and then its type and id stand in as the value."""
            if r.get("title"):
                return r["title"]
            self.report.note("empty_title")
            return f"{r['type']}:{r['phone_id']}"

        def calendar(r: Mapping) -> None:
            start, end = parse_naive(r["start_time"], DATASET_TZ), parse_naive(r["end_time"], DATASET_TZ)
            if not self.fresh("calendar", r["title"], r["start_time"], r["end_time"], r.get("description")):
                return
            created = parse_naive(r["datetime"], DATASET_TZ)
            self.fact(f"{self.prefix}.calendar.{r['phone_id']}", MemoryKind.EVENT, f"calendar:{r['phone_id']}", label(r),
                      f"日程 {r['title']}：{r.get('description') or ''}（时间：{r['start_time']} 至 {r['end_time']}）",
                      own("calendar", r, r["title"], start), created, valid_from=start,
                      valid_to=_interval(start, end, self.report))

        def note(r: Mapping) -> None:
            if not self.fresh("note", r["title"], r["datetime"], r["content"]):
                return
            when = parse_naive(r["datetime"], DATASET_TZ)
            self.fact(f"{self.prefix}.note.{r['phone_id']}", MemoryKind.EVENT, f"note:{r['phone_id']}", label(r),
                      f"笔记 {r['title']}：{r['content']}（记录时间：{r['datetime']}）", own("note", r, r["title"], when), when,
                      valid_from=when)

        def photo(r: Mapping) -> None:
            if not self.fresh("photo", r["title"], r["datetime"], r.get("caption")):
                return
            when = parse_naive(r["datetime"], DATASET_TZ)
            place = r.get("location") or {}
            where = "".join(str(place.get(k) or "") for k in ("city", "district", "poi"))
            faces = f"；人物：{r['faceRecognition']}" if r.get("faceRecognition") else ""
            tags = f"；标签：{'、'.join(r['imageTag'])}" if r.get("imageTag") else ""
            ocr = f"；图中文字：{r['ocrText']}" if r.get("ocrText") else ""
            self.fact(f"{self.prefix}.photo.{r['phone_id']}", MemoryKind.EVENT, f"photo:{r['phone_id']}", label(r),
                      f"照片：{r['caption']}（拍摄时间：{r['datetime']}；地点：{where}{faces}{tags}{ocr}）",
                      own("photo", r, r["title"], when), when, valid_from=when)

        def push(r: Mapping) -> None:
            if not self.fresh("push", r["title"], r["datetime"], r["content"]):
                return
            when = parse_naive(r["datetime"], DATASET_TZ)
            self.fact(f"{self.prefix}.push.{r['phone_id']}", MemoryKind.EVENT, f"push:{r['phone_id']}", label(r),
                      f"推送 {r['title']}：{r['content']}（来源：{r.get('source')}；时间：{r['datetime']}）",
                      own("push", r, r["title"], when), when, valid_from=when)

        def call(r: Mapping) -> None:
            if not self.fresh("call", r.get("contactName"), r["datetime"], r["datetime_end"]):
                return
            start, end = parse_naive(r["datetime"], DATASET_TZ), parse_naive(r["datetime_end"], DATASET_TZ)
            valid_to = _interval(start, end, self.report)
            # `direction` is not documented, so the statement does not say who called whom
            self.fact(f"{self.prefix}.call.{r['phone_id']}", MemoryKind.EVENT, f"call:{r['phone_id']}",
                      str(r.get("contactName") or r.get("phoneNumber") or "未知联系人"),
                      f"与{r.get('contactName') or r.get('phoneNumber') or '未知联系人'}的通话记录（{r.get('call_result')}；"
                      f"时间：{r['datetime']} 至 {r['datetime_end']}）",
                      [Provenance(source_id=self.sources["call"].source_id,
                                  locator=f"phone_id={r['phone_id']};direction={r.get('direction')}", observed_at=start),
                       *self.daily_link(r)], valid_to, valid_from=start, valid_to=valid_to)

        def fitness(r: Mapping) -> None:
            text = r.get("summarized_info")
            if not text:
                self.report.skip("fitness_without_summary")
                return
            start = parse_naive(r["日期"], DATASET_TZ, "%Y-%m-%d")
            end = start + timedelta(days=1)
            self.fact(f"{self.prefix}.fitness.{r['phone_id']}", MemoryKind.EVENT, f"fitness:{r['phone_id']}", r["日期"],
                      f"健康记录 {r['日期']}（{r.get('城市')}）：{text}", own("fitness_health", r, r["日期"], start), end,
                      valid_from=start, valid_to=end)

        for name, handle in (("calendar", calendar), ("note", note), ("photo", photo), ("push", push), ("call", call),
                             ("fitness_health", fitness)):
            self.each(f"{name}.json", rows(name), handle)

    # -------------------------------------------------------------- messages

    def _speaker_for(self, name: Optional[str], phone: Optional[str]) -> Optional[str]:
        if name and name in self.contacts_by_name:
            return self.contacts_by_name[name]
        digits = _digits(phone)
        if digits and digits in self.contacts_by_phone:
            return self.contacts_by_phone[digits]
        if not name and not digits:
            return None
        speaker = stable_id(f"{self.prefix}.ext", name or "", "" if name else digits)
        self.speakers.setdefault(speaker, {"name": name or digits, "relation": None})
        return speaker

    def sms(self) -> None:
        source = self.source("phone_data/sms.json", SourceKind.CONVERSATION)

        def handle(r: Mapping) -> None:
            other = self._speaker_for(r.get("contactName"), r.get("phoneNumber"))
            kind = r["message_type"]
            if kind == "发送":
                speaker = self.pid
            elif kind == "接收":
                speaker = other
            else:
                speaker = None
            if speaker is None:                      # never invent who said it
                self.report.skip("sms_speaker_unresolved")
                return
            when = parse_naive(r["datetime"], DATASET_TZ)
            conversation = f"{self.prefix}.sms.{(other or 'unknown').rsplit('.', 1)[-1]}"
            if not self.fresh("sms", conversation, speaker, r["datetime"], r["message_content"]):
                return
            links = ";".join(f"{p.source_id.rsplit('.', 1)[-1]}:{p.locator}" for p in self.daily_link(r))
            self.messages.append(CanonicalMessage(
                conversation_id=conversation, message_id=f"{self.prefix}.sms.{r['phone_id']}", target_persona_id=self.pid,
                speaker_id=speaker, timestamp=when, content=r["message_content"], source_type="lifebench_sms",
                source_id=source, locator=f"phone_id={r['phone_id']};type={kind};{links}".rstrip(";"), recorded_at=when))

        self.each("sms.json", _load(self.dir / "phone_data" / "sms.json"), handle)

    def agent_chat(self) -> None:
        source = self.source("phone_data/agent_chat.json", SourceKind.CONVERSATION)

        def base_time(r: Mapping) -> Tuple[datetime, str]:
            text = r["date"]
            if len(text) > 10:
                return parse_naive(text, DATASET_TZ), "given"
            linked = self.events.get(str(r.get("daily_event_id") or ""))
            if linked is not None and linked[1].strftime("%Y-%m-%d") == text:
                return linked[1], "assumed_daily_event_end"
            return parse_naive(text + " 23:59:00", DATASET_TZ), "assumed_end_of_date"

        def turns(conversation: Mapping) -> Iterable[Tuple[str, str, Mapping]]:
            for name, turn in conversation.items():
                for role in ("user", "assistant"):
                    if role in turn:
                        yield name, role, turn[role]
                nested = {k: v for k, v in turn.items() if k not in ("user", "assistant")}
                if nested:
                    yield from turns(nested)

        def handle(r: Mapping) -> None:
            base, how = base_time(r)
            self.report.note(f"agent_chat_time_{how}")
            conversation = f"{self.prefix}.agent_chat.{r['phone_id']}"
            links = ";".join(f"{p.source_id.rsplit('.', 1)[-1]}:{p.locator}" for p in self.daily_link(r))
            for position, (name, role, message) in enumerate(turns(r["conversation"])):
                content = message.get("content")
                if not content or not str(content).strip():
                    self.report.skip("agent_chat_empty_message")
                    continue
                self.messages.append(CanonicalMessage(
                    conversation_id=conversation, message_id=f"{self.prefix}.agent_chat.{r['phone_id']}.{position}",
                    target_persona_id=self.pid, speaker_id=self.pid if role == "user" else self.assistant,
                    timestamp=base + timedelta(seconds=position), content=content, source_type="lifebench_agent_chat",
                    source_id=source, recorded_at=base + timedelta(seconds=position),
                    locator=f"phone_id={r['phone_id']};{name};{role};date={r['date']};time={how};{links}".rstrip(";")))

        self.each("agent_chat.json", _load(self.dir / "phone_data" / "agent_chat.json"), handle)


def parse_persona(directory: Path, revision: str) -> ParsedPersona:
    """Everything one persona's files map to, in memory. Writes nothing."""
    m = _Mapper(directory, revision)
    name = m.persona()
    m.contacts()
    m.daily_events()
    m.phone_records()
    m.sms()
    m.agent_chat()
    m.report.source_count = len(m.sources)
    m.report.message_count = len(m.messages)
    m.report.fact_count = len(m.facts)
    return ParsedPersona(m.pid, name, list(m.sources.values()), m.messages, m.facts, m.speakers, m.report)


class LifeBenchImporter:
    def __init__(self, api: MemoryAPI, revision: str) -> None:
        self.api = api
        self.revision = revision

    def import_persona(self, directory: Path) -> ParsedPersona:
        """Write one persona through ``MemoryAPI``, batch by batch. Each batch is one transaction; a re-run adds
        nothing, and changed content under a stored id stops the import with an error."""
        started = time.perf_counter()
        parsed = parse_persona(directory, self.revision)
        for source in parsed.sources:
            self.api.register_source(source)
        for batch in batches(parsed.messages, BATCH_SIZE):
            parsed.report.added_messages += self.api.ingest_messages(batch).added
        for batch in batches(parsed.facts, BATCH_SIZE):
            parsed.report.added_facts += self.api.import_facts(parsed.persona_id, batch).added
        parsed.report.elapsed_seconds = time.perf_counter() - started
        return parsed
