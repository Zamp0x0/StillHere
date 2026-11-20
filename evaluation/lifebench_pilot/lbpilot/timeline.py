"""Canonical event adapter: one LifeBench person (locomo conversation format) -> one ordered list of events every system reads.

The locomo-format file holds 365 daily sessions; each record is one phone-data item rendered as text by the dataset authors, with
``dia_id`` = ``<date>_<source><n>`` (sources found in v2.0: agent_chat, calendar, call, fitness_health, note, photo, push, sms). The
event fields:

    event_id        dia_id, or dia_id:t<k> for the k-th turn of an agent_chat conversation (they share one dia_id); unique
    gold_id         dia_id without the date prefix (= the QA ``evidence`` ids)
    source_type     agent_chat | sms | call | ...         date        the session date (YYYY-MM-DD)
    event_time      the time written in the record when it has one on that date, else null (kept for provenance only)
    order           position within the day                speaker / actor   the record's speaker (the person or the assistant)
    subject         the person (every record is about the person's own life)                       content     the record text
    provenance      dataset repo@revision, session key     visible_from  the date: an event is visible to a question asked on or
                                                                         after that date (official date order: ADD, then SEARCH)

StillHere and Mem0 receive exactly these contents, dates and order. No gold answer or evidence list is ever attached to an event.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Dict, List, Mapping, Sequence

from . import protocol as P

SESSION_KEY = re.compile(r"^session_(\d+)$")
TIME_IN_TEXT = re.compile(r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}(?::\d{2})?)")
ASK_PREFIX = re.compile(r"^\s*[（(]\s*提问时间[:：]\s*\d{4}-\d{2}-\d{2}\s*[)）]\s*")


def source_of(dia_id: str) -> str:
    return (dia_id.split("_", 1)[-1] if "_" in dia_id else dia_id).rstrip("0123456789") or "unknown"


def gold_id(dia_id: str) -> str:
    return dia_id.split("_", 1)[-1] if re.match(r"^\d{4}-\d{2}-\d{2}_", dia_id) else dia_id


def search_query(question: str) -> str:
    return ASK_PREFIX.sub("", question or "").strip()


def build(person: Mapping, *, until: str = "9999-12-31") -> List[dict]:
    conv = person["conversation"]
    keys = sorted((k for k in conv if SESSION_KEY.match(k)), key=lambda k: int(SESSION_KEY.match(k).group(1)))
    out: List[dict] = []
    seen: Dict[str, int] = {}
    for k in keys:
        date = str(conv.get(f"{k}_date_time", ""))[:10]
        if not date or date > until:
            continue
        for i, m in enumerate(conv[k]):
            text = (m.get("text") or "").strip()
            dia = m.get("dia_id") or f"{date}_pos{i}"
            if not text:
                continue
            turn = seen.get(dia, -1) + 1                     # an agent_chat conversation shares one dia_id across its turns
            seen[dia] = turn
            eid = dia if turn == 0 else f"{dia}:t{turn}"
            t = next((tm for d, tm in TIME_IN_TEXT.findall(text) if d == date), None)
            out.append({"event_id": eid, "dia_id": dia, "turn": turn, "gold_id": gold_id(dia), "source_type": source_of(dia), "date": date, "order": i,
                        "event_time": f"{date} {t}" if t else None, "speaker": m.get("speaker") or "", "subject": person["sample_id"],
                        "content": text, "provenance": f"hf:{P.DATASET['repo']}@{P.DATASET['revision'][:8]}#{k}", "visible_from": date})
    return out


def sha(events: Sequence[Mapping]) -> str:
    return hashlib.sha256("\n".join(json.dumps(e, sort_keys=True, ensure_ascii=False) for e in events).encode("utf-8")).hexdigest()


def by_date(events: Sequence[Mapping]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for e in events:
        out.setdefault(e["date"], []).append(e)
    return out


def visible(events: Sequence[Mapping], ask_date: str) -> List[dict]:
    return [e for e in events if e["visible_from"] <= ask_date]


def stats(events: Sequence[Mapping]) -> dict:
    from collections import Counter
    return {"events": len(events), "dates": len({e["date"] for e in events}), "range": [events[0]["date"], events[-1]["date"]] if events else None,
            "by_source": dict(sorted(Counter(e["source_type"] for e in events).items())),
            "speakers": dict(sorted(Counter(e["speaker"] for e in events).items())), "with_time_in_text": sum(1 for e in events if e["event_time"])}
