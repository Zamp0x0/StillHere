"""What every importer shares: stable ids, time normalisation and the import report.

An importer turns a dataset into contracts and hands them to ``memory.MemoryAPI``. It never opens the database.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Dict, Iterable, List, Optional, Sequence, TypeVar

T = TypeVar("T")

MAX_LISTED_ERRORS = 200


def stable_id(prefix: str, *parts: object) -> str:
    """A deterministic id: the same inputs give the same id on every run, which makes re-imports idempotent."""
    digest = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}.{digest}"


def parse_naive(text: str, tz: tzinfo, fmt: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    """A wall-clock string without an offset, read in the timezone the dataset's rule declares."""
    return datetime.strptime(text.strip(), fmt).replace(tzinfo=tz)


def parse_aware(text: str, default_tz: Optional[tzinfo] = None) -> datetime:
    """An ISO-8601 time. Without an offset it needs ``default_tz``; it is never guessed."""
    value = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    if value.tzinfo is None or value.utcoffset() is None:
        if default_tz is None:
            raise ValueError(f"timestamp {text!r} has no timezone and no dataset rule gives one")
        value = value.replace(tzinfo=default_tz)
    return value


def utc_offset(hours: int) -> timezone:
    return timezone(timedelta(hours=hours))


def batches(items: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


@dataclass
class ImportReport:
    """Counts of one import. ``added_*`` are rows this run wrote; a repeated import adds zero. ``raw_count`` and the
    counters below it account for every record read, so nothing disappears without a reason."""

    source_count: int = 0
    message_count: int = 0        # messages prepared for the store
    fact_count: int = 0           # memory records prepared for the store
    added_messages: int = 0
    added_facts: int = 0
    duplicate_count: int = 0      # identical records in the input, kept once
    skipped_count: int = 0
    error_count: int = 0
    raw_count: int = 0
    elapsed_seconds: float = 0.0
    raw_by_file: Dict[str, int] = field(default_factory=dict)
    skipped: Dict[str, int] = field(default_factory=dict)
    source_only: Dict[str, int] = field(default_factory=dict)   # kept as a source, not imported as records
    notes: Dict[str, int] = field(default_factory=dict)         # repairs and assumptions applied to imported records
    errors: List[str] = field(default_factory=list)

    def read(self, file: str, n: int = 1) -> None:
        self.raw_by_file[file] = self.raw_by_file.get(file, 0) + n
        self.raw_count += n

    def skip(self, reason: str, n: int = 1) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + n
        self.skipped_count += n

    def keep_source_only(self, kind: str, n: int) -> None:
        self.source_only[kind] = self.source_only.get(kind, 0) + n

    def note(self, kind: str, n: int = 1) -> None:
        self.notes[kind] = self.notes.get(kind, 0) + n

    def error(self, message: str) -> None:
        self.error_count += 1
        if len(self.errors) < MAX_LISTED_ERRORS:
            self.errors.append(message)

    def merge(self, other: "ImportReport") -> None:
        for name in ("source_count", "message_count", "fact_count", "added_messages", "added_facts", "duplicate_count",
                     "skipped_count", "error_count", "raw_count"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.elapsed_seconds += other.elapsed_seconds
        for mine, theirs in ((self.raw_by_file, other.raw_by_file), (self.skipped, other.skipped),
                             (self.source_only, other.source_only), (self.notes, other.notes)):
            merged = Counter(mine)
            merged.update(theirs)
            mine.clear()
            mine.update(merged)
        self.errors = (self.errors + other.errors)[:MAX_LISTED_ERRORS]

    def to_dict(self) -> dict:
        return asdict(self)
