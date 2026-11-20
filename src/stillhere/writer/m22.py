"""M2.2 attribution, privacy-boundary and event-identity normalisation (deterministic, after verification, before the write decision).

Behind ``MEMORY_WRITER_M22_ENABLED`` (off: the M2.1 writer is unchanged). Every rule acts only on evidence the candidate already cites:

R1 relationship direction: "relationship" is always the persona's relation to someone. A candidate ``subject=person:X, predicate=relationship,
   object="<relation> X"`` is the same statement seen from the wrong side: the subject becomes ``self``.
R2 speaker attribution: when EVERY cited message was written by someone other than the persona, a ``self`` candidate is a misattribution:
   an event becomes the speaker's own event (``person:<speaker>``); any other fact is dropped (the persona said nothing about itself);
   a candidate citing only the assistant is dropped.
R3 third-party boundary: facts about a person who is neither a contact of the persona nor someone the persona has a stated relationship with
   (in memory or in the same batch) are not stored at all: a colleague's address or a stranger's preference is not the persona's memory.
R4 event identity: an event update (completed / cancelled / rescheduled) with no explicit link to an earlier event is linked to the persona's
   open event (planned / scheduled, same actor; or, for a completed mention, the completed event of the same day: a repeated mention) whose description is the same activity: one contains the other (>= 2 chars) or their character
   bigrams overlap with Jaccard >= 0.5. One link at most; ties -> no link (better two events than a wrong merge).
"""
from __future__ import annotations

import os
import re
from dataclasses import replace
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

FLAG = "MEMORY_WRITER_M22_ENABLED"
UPDATE_STATUSES = ("COMPLETED", "CANCELLED", "IN_PROGRESS")


def enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    return (env if env is not None else os.environ).get(FLAG, "").strip().lower() in ("1", "true", "yes", "on")


def _bigrams(s: str) -> set:
    s = re.sub(r"\W", "", s or "")
    return {s[i:i + 2] for i in range(len(s) - 1)} or ({s} if s else set())


def same_activity(a: str, b: str) -> bool:
    a, b = re.sub(r"\W", "", a or ""), re.sub(r"\W", "", b or "")
    if not a or not b:
        return False
    if len(min(a, b, key=len)) >= 2 and (a in b or b in a):
        return True
    x, y = _bigrams(a), _bigrams(b)
    return bool(x and y) and len(x & y) / len(x | y) >= 0.5


# kin words a relationship object may start with ("妹妹小雨", "妹妹 小雨", "妹妹"): both the kin word and the name are the stated person
KIN_WORDS = ("妹妹", "姐姐", "哥哥", "弟弟", "爸爸", "妈妈", "父亲", "母亲", "爷爷", "奶奶", "外公", "外婆", "姥姥", "姥爷", "老爸", "老妈",
             "叔叔", "阿姨", "舅舅", "姑姑", "表哥", "表姐", "表弟", "表妹", "堂哥", "堂姐", "堂弟", "堂妹", "儿子", "女儿", "老公", "老婆",
             "男朋友", "女朋友", "对象", "同事", "室友", "同学", "朋友", "闺蜜", "老板", "领导", "老师", "妹", "姐", "哥", "弟", "爸", "妈")


def relation_names(value: str) -> set:
    """Names a relationship object states: the last token (the old rule) plus, when it starts with a kin word, the kin word and the rest
    ("妹妹小雨" -> {妹妹小雨, 妹妹, 小雨}). P2 (2026-11-19): the old rule kept only "妹妹小雨" (no space), so the correctly attributed
    ``person:小雨`` wedding was rejected by R3, and ``person:妹妹`` (the follow-up prompt's own example) was always rejected."""
    out = set()
    v = (value or "").strip()
    if not v:
        return out
    out.add(v.split()[-1])
    flat = re.sub(r"\s+", "", v)
    for k in sorted(KIN_WORDS, key=len, reverse=True):
        if flat.startswith(k):
            out.add(k)
            if flat[len(k):]:
                out.add(flat[len(k):])
            break
    return out


def stated_in_message(name: str, cited) -> bool:
    """The cited message itself states the relation: '我(的)<kin><name>' ("我妹妹小雨下个月结婚") or '我(的)<kin>' for name = the kin word."""
    if not name:
        return False
    kin = "|".join(sorted(KIN_WORDS, key=len, reverse=True))
    rx = re.compile(rf"我的?(?:{kin})\s*{re.escape(name)}") if name not in KIN_WORDS else re.compile(rf"我的?{re.escape(name)}")
    return any(rx.search(getattr(m, "content", "") or "") for m in cited)


def known_people(existing: Sequence[Tuple[str, str, str, str]], contacts: Iterable[str], batch_candidates: Sequence) -> set:
    people = {c for c in contacts if c}
    for _mid, subject, attribute, value in existing:
        if attribute == "relationship" and value:
            people |= relation_names(value)
    for c in batch_candidates:
        if c.predicate == "relationship" and c.object:
            people |= relation_names(c.object)
    return people


def open_events(existing: Sequence[Tuple[str, str, str, str]]) -> List[dict]:
    ev = {}
    for _mid, subject, attribute, value in existing:
        if subject.startswith("event:"):
            ev.setdefault(subject, {})[attribute] = value
    return [{"subject": s, **f} for s, f in ev.items() if f.get("event.status") in ("PLANNED", "SCHEDULED", "COMPLETED")]


def normalize(c, msgs: Mapping[str, object], persona_id: str, people: set, events: Sequence[dict]) -> Tuple[Optional[object], List[str]]:
    """Returns (candidate or None, rule notes). None = do not write."""
    notes: List[str] = []
    cited = [msgs[m] for m in c.source_message_ids if m in msgs]
    # R1
    if c.predicate == "relationship" and c.subject.startswith("person:"):
        name = c.subject.split(":", 1)[1]
        if name and name in (c.object or ""):
            c = replace(c, subject="self")
            notes.append("R1 relationship subject set to the persona")
    # R2
    if c.subject == "self" and cited and all(m.speaker_id != persona_id for m in cited):
        speakers = {m.speaker_id for m in cited}
        if all(s.endswith(".assistant") for s in speakers):
            return None, notes + ["R2 cites only the assistant"]
        if c.predicate == "event" and len({m.sender for m in cited}) == 1:
            own = [e for e in events if (e.get("event.actor") or "self") == "self" and (same_activity(c.object, e.get("event.description", ""))
                                                                                         or same_activity(c.event_key or "", e.get("event.description", "")))]
            if len(own) == 1:      # a contact reporting on the persona's own open event (e.g. "tonight's team dinner is cancelled")
                c = replace(c, refers_to=c.refers_to if (c.refers_to or "").startswith("event:") else own[0]["subject"])
                notes.append(f"R2 contact update of the persona's open event {own[0]['subject']}")
            else:
                who = cited[0].sender
                c = replace(c, subject=f"person:{who}")
                notes.append(f"R2 event re-attributed to its speaker {who}")
        else:
            return None, notes + ["R2 every cited message is someone else's: not a fact about the persona"]
    # R3
    if c.subject.startswith("person:"):
        name = c.subject.split(":", 1)[1]
        if name not in people and not stated_in_message(name, cited):
            return None, notes + [f"R3 third-party boundary: {name} is not a contact or a stated relation"]
    # R4
    if c.predicate == "event" and not (c.refers_to or "").startswith("event:") and (c.event_status in UPDATE_STATUSES or (c.event_status == "SCHEDULED" and c.planned_time)):
        actor = c.subject
        day = (c.event_time or "")[:10]

        def eligible(e):
            if e.get("event.status") != "COMPLETED":
                return True
            # an already completed event: only the same occurrence (a repeated mention), i.e. a completed candidate on the same day
            return c.event_status == "COMPLETED" and bool(day) and (e.get("event.time") or "")[:10] == day
        hits = [e for e in events if eligible(e) and (e.get("event.actor") or "self") == actor
                and (same_activity(c.object, e.get("event.description", "")) or same_activity(c.event_key or "", e.get("event.description", "")))]
        if len(hits) == 1:
            c = replace(c, refers_to=hits[0]["subject"])
            notes.append(f"R4 linked to open event {hits[0]['subject']}")
        elif len(hits) > 1:
            notes.append("R4 several open events match: not linked")
    return c, notes
