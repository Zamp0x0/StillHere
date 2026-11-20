"""GenericConversationImporter: raw chat history from standardised JSON or JSONL.

One record is an object with ``target_persona_id``, ``conversation_id``, ``speaker_id``, ``timestamp``, ``content``
and optionally ``message_id``, ``source_id`` and ``locator``.

  * ``target_persona_id`` is the persona whose history this is, ``speaker_id`` who said it. Who a statement is about
    is not decided here: this importer stores messages and creates no memory.
  * A record without ``message_id`` gets a deterministic one from the source, its path, the conversation, its
    position and its content, so importing the same file again adds nothing.
  * A timestamp needs an offset, or a ``default_tz`` the caller declares for the whole dataset.
  * A record that cannot be read is counted and listed in the report. The same message id with other content is not
    a bad record but a changed source: ``MessageConflict`` stops the import and the batch rolls back.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import tzinfo
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Union

from stillhere.contracts import CanonicalMessage, SourceRef, ValidationError
from stillhere.memory import MemoryAPI

from .base import ImportReport, batches, parse_aware, stable_id

BATCH_SIZE = 2000


class GenericConversationImporter:
    def __init__(self, api: MemoryAPI, source: SourceRef, *, source_type: str = "generic_chat",
                 default_tz: Optional[tzinfo] = None) -> None:
        self.api = api
        self.source = source
        self.source_type = source_type
        self.default_tz = default_tz

    def import_file(self, path: Union[str, "os.PathLike[str]"]) -> ImportReport:
        file = Path(path)
        text = file.read_text(encoding="utf-8")
        if file.suffix == ".jsonl":
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            records = json.loads(text)
        return self.import_records(records, origin=file.name)

    def import_records(self, records: Iterable[Mapping], *, origin: str = "records") -> ImportReport:
        started = time.perf_counter()
        report = ImportReport()
        self.api.register_source(self.source)
        report.source_count = 1
        messages: List[CanonicalMessage] = []
        for position, record in enumerate(records):
            report.read(origin)
            try:
                messages.append(self._message(record, origin, position))
            except (KeyError, TypeError, ValueError, AttributeError) as exc:   # ValidationError is a ValueError
                report.error(f"{origin}[{position}]: {type(exc).__name__}: {exc}")
        report.message_count = len(messages)
        for batch in batches(messages, BATCH_SIZE):
            result = self.api.ingest_messages(batch)
            report.added_messages += result.added
        report.duplicate_count = len(messages) - report.added_messages
        report.elapsed_seconds = time.perf_counter() - started
        return report

    def _message(self, r: Mapping, origin: str, position: int) -> CanonicalMessage:
        content = r["content"]
        source_id = r.get("source_id") or self.source.source_id
        if source_id != self.source.source_id:
            raise ValidationError(f"record cites source {source_id!r}, this import is {self.source.source_id!r}")
        message_id = r.get("message_id") or stable_id(
            "msg", self.source.sha256 or self.source.source_id, origin, r["conversation_id"], position,
            hashlib.sha256(str(content).encode("utf-8")).hexdigest())
        timestamp = parse_aware(r["timestamp"], self.default_tz)
        return CanonicalMessage(conversation_id=r["conversation_id"], message_id=message_id,
                                target_persona_id=r["target_persona_id"], speaker_id=r["speaker_id"],
                                timestamp=timestamp, content=content, source_type=self.source_type,
                                source_id=source_id, locator=r.get("locator") or f"{origin}#{position}",
                                recorded_at=r.get("recorded_at"))
