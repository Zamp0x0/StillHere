"""Core memory -> D4 Evidence View (``stillhere.session.d4_evidence/1.0``), computed at ``query_time`` from the Core record fields only.

Facts (one record per unit), active records only:
    valid_to <= now                       -> HISTORICAL
    valid_from > now                      -> UPCOMING
    supersedes + correction_reason "correction from …" (written by the Writer for an explicit correction; "state change from …" /
    "temporal change from …" are changes over time)  -> CORRECTED, ``replaces`` = the superseded record's statement
    another active single-valued record of the same slot, valid now, different value -> CONFLICTED (one conflict group per slot)
    otherwise                             -> CURRENT
    superseded / retracted records are never items (a corrected old value appears only as ``replaces``).
Events (the records of one event subject), latest status valid at now:
    CANCELLED -> CANCELLED (note: when the cancellation was recorded); COMPLETED -> CURRENT (a past event that happened);
    PLANNED / SCHEDULED with a planned date after now -> UPCOMING; otherwise CURRENT.
Unknown items of the grounding become D4 ``UnknownSlot``s. Nothing is inferred from words like "取消" / "更正" in free text.

Every item passes the disclosure policy (``disclosure.PolicyResolver``) and is redacted before it exists: a non-ALLOWED item carries no raw
value in any model-visible field. Memory ids stay in the runtime map (``id_map``), never in an item; ``fact_id`` is a display number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Sequence

from stillhere.memory import reflection as r2
from stillhere.memory import retrieval_b2 as b2
from stillhere.runtime import pipeline as rp

import stillhere.session.d4_renderer as dr
import stillhere.session.disclosure as ds

EVIDENCE_SCHEMA_VERSION = "stillhere.session.d4_evidence/1.0"
RELATION_TYPES = (("spouse", ("老公", "老婆", "丈夫", "妻子", "爱人", "先生", "太太")),
                  ("family", ("爸", "妈", "父亲", "母亲", "哥", "姐", "弟", "妹", "儿子", "女儿", "爷爷", "奶奶", "外公", "外婆", "表", "堂", "舅", "姑", "姨", "叔")),
                  ("colleague", ("同事", "领导", "老板", "上司", "下属")), ("neighbor", ("邻居",)),
                  ("friend", ("朋友", "室友", "同学", "闺蜜", "兄弟", "发小", "网友")))
CONF = ((0.8, "high"), (0.5, "medium"), (0.0, "low"))


@dataclass
class EvidenceView:
    evidence: List                         # d4 Evidence items (validated, redacted)
    unknown: List                          # d4 UnknownSlot items
    id_map: Dict[str, List[str]]           # fact_id -> memory ids (runtime only)
    decisions: Dict[str, dict]             # fact_id -> policy trace (categories, reason, hashed terms)
    guard_terms: List[str] = field(default_factory=list)   # RESTRICTED raw values for the output guard; never rendered or logged
    notes: List[str] = field(default_factory=list)

    def trace(self) -> dict:
        return {"schema": EVIDENCE_SCHEMA_VERSION, "statuses": {e.fact_id: e.status for e in self.evidence},
                "policies": {e.fact_id: e.disclosure_policy for e in self.evidence}, "unknown": [u.slot_id for u in self.unknown],
                "decisions": self.decisions, "notes": self.notes}


def _d(x) -> Optional[str]:
    if x is None:
        return None
    return (x.date() if isinstance(x, datetime) else x).isoformat()


def _date_of(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10]) if len(value) >= 10 else date(int(value[:4]), int(value[5:7]), 1)
    except ValueError:
        return None


def _confidence(c: float) -> str:
    return next(lab for cut, lab in CONF if c >= cut)


def _relations(api, persona_id: str, now: datetime) -> Dict[str, str]:
    out = {}
    for r in b2._all_records(api, persona_id, now):
        if r.attribute == "relationship" and r2._status(r, now) == "当前":
            parts = (r.value or "").split()
            if len(parts) >= 2:
                out[parts[-1]] = parts[0]
    return out


def subject_of(subject: str, relations: Dict[str, str]):
    """(label shown, subject_type, defaulted?) — '本人' / self, or the person's name with the known relation prefixed."""
    if subject in ("persona", "self", "本人") or not subject:
        return "本人", "self", False
    name = subject.split(":", 1)[-1]
    rel = relations.get(name, "")
    for st, words in RELATION_TYPES:
        if rel and any(w in rel for w in words):
            return (rel + name if rel not in name else name), st, False
    return name, "friend", True                    # D4 has no generic "other" type: recorded as defaulted (PENDING_ALIGNMENT)


def _is_correction(r) -> bool:
    return bool(r.supersedes) and (r.correction_reason or "").startswith("correction")


def _fact_status(api, persona_id: str, r, now: datetime, live_slots) -> tuple:
    if r.valid_to is not None and r.valid_to <= now:
        return "HISTORICAL", None
    if r.valid_from is not None and r.valid_from > now:
        return "UPCOMING", None
    if _is_correction(r):
        old = api.store.get(r.supersedes, persona_id=persona_id)
        return "CORRECTED", (old.statement if old is not None else None)
    others = [x for x in live_slots.get((r.subject, r.attribute), []) if x.memory_id != r.memory_id and x.value != r.value]
    if others and getattr(r.cardinality, "value", str(r.cardinality)) == "single":
        return "CONFLICTED", None
    return "CURRENT", None


def build(api, persona_id: str, m, grounded: dict, now: datetime, resolver: ds.PolicyResolver, *, max_items: int = 5) -> EvidenceView:
    """``m``: the M3.1 result (its ranked evidence units); ``grounded``: the grounding (for unknown items)."""
    by_id = {r.memory_id: r for r in m.evidence.records}
    relations = _relations(api, persona_id, now)
    live_slots: Dict[tuple, list] = {}
    for r in b2._all_records(api, persona_id, now):
        if r2._status(r, now) == "当前" and not r.subject.startswith("event:"):
            live_slots.setdefault((r.subject, r.attribute), []).append(r)
    items, id_map, decisions, guard, notes = [], {}, {}, [], []
    conflict_groups: Dict[tuple, str] = {}
    for key, ids in m.evidence.units:
        if len(items) >= max_items:
            break
        rs = [by_id[i] for i in ids if i in by_id]
        if not rs:
            continue
        fid = f"F{len(items) + 1}"
        if key.startswith("event:"):
            spec = _event(rs, now, relations)
            if spec is None:
                continue
        else:
            r = rs[0]
            if getattr(r.status, "value", str(r.status)) != "active":
                continue                                            # superseded / retracted: only via ``replaces`` of its correction
            status, replaces = _fact_status(api, persona_id, r, now, live_slots)
            label, stype, defaulted = subject_of(r.subject, relations)
            if defaulted:
                notes.append(f"{fid}: subject_type defaulted to friend (no relation recorded)")
            group = None
            if status == "CONFLICTED":
                group = conflict_groups.setdefault((r.subject, r.attribute), f"G{len(conflict_groups) + 1}")
            spec = dict(subject=label, subject_type=stype, predicate=r.attribute, value=r.value, evidence_text=_third(r.statement, label),
                        status=status, valid_from=_d(r.valid_from), valid_to=_d(r.valid_to), observed_at=_d(r.recorded_at),
                        source_ref=(r.provenance[0].source_id if r.provenance else "memory"), confidence=_confidence(r.confidence),
                        conflict_group=group, replaces=_third(replaces, label) if replaces else None, note=None, raw_subject=r.subject)
            if status == "CORRECTED" and not spec["replaces"]:
                spec["status"], spec["replaces"] = "CURRENT", None
                notes.append(f"{fid}: superseded record not readable; shown as CURRENT")
        dec = resolver.decide(persona_id, spec.pop("raw_subject"), spec["predicate"], spec["value"],
                              " ".join(x for x in (spec["evidence_text"], spec.get("replaces") or "", spec.get("note") or "") if x), now)
        spec["disclosure_policy"] = dec.policy
        if dec.policy != ds.ALLOWED:
            spec["evidence_text"] = ds.redacted_text(spec["subject"], dec)
            spec["value"] = "（不显示）"
            if spec.get("replaces"):
                spec["replaces"] = "（不显示）"
            if spec.get("note"):
                spec["note"] = "（不显示）"
            guard += dec.guard_terms
        e = dr.Evidence(fact_id=fid, **spec)
        e.validate()
        items.append(e)
        id_map[fid] = list(ids)
        decisions[fid] = dec.trace()
    # a raw guarded value must not survive in any other (ALLOWED) item either
    for i, e in enumerate(items):
        if e.disclosure_policy == ds.ALLOWED and any(t and t in (e.evidence_text + (e.replaces or "") + (e.note or "")) for t in guard):
            items[i] = dr.Evidence(**{**e.to_json(), "evidence_text": f"{e.subject}有一条相关记录，其中含不可透露的内容", "value": "（不显示）",
                                      "replaces": "（不显示）" if e.replaces else None, "note": "（不显示）" if e.note else None,
                                      "disclosure_policy": ds.PROTECTED})
            notes.append(f"{e.fact_id}: contained a guarded value; protected")
    # a conflict is shown whole: the other conflicting records of a group join the view even past ``max_items``
    for (subj, attr_), gid in conflict_groups.items():
        shown = {mid for f, mids in id_map.items() for mid in mids}
        for other in live_slots.get((subj, attr_), []):
            if other.memory_id in shown:
                continue
            fid = f"F{len(items) + 1}"
            label, stype, _ = subject_of(other.subject, relations)
            dec = resolver.decide(persona_id, other.subject, other.attribute, other.value, other.statement, now)
            text, value = (_third(other.statement, label), other.value) if dec.policy == ds.ALLOWED else (ds.redacted_text(label, dec), "（不显示）")
            guard += dec.guard_terms
            e = dr.Evidence(fact_id=fid, subject=label, subject_type=stype, predicate=other.attribute, value=value, evidence_text=text,
                            status="CONFLICTED", valid_from=_d(other.valid_from), valid_to=_d(other.valid_to), observed_at=_d(other.recorded_at),
                            source_ref=(other.provenance[0].source_id if other.provenance else "memory"), confidence=_confidence(other.confidence),
                            disclosure_policy=dec.policy, conflict_group=gid)
            e.validate()
            items.append(e)
            id_map[fid] = [other.memory_id]
            decisions[fid] = dec.trace()
    unknown = [dr.UnknownSlot(f"U{i + 1}", "本人", "asked", u) for i, u in enumerate(grounded.get("unknown") or [])]
    if any(x.status == "CONFLICTED" for x in items):
        dr.evidence_mod.check_conflict_groups(items)
    return EvidenceView(items, unknown, id_map, decisions, sorted(set(guard), key=lambda s: (-len(s), s)), notes)


def _third(statement: Optional[str], label: str) -> Optional[str]:
    """Core statements of other people start with the bare name; keep them as they are (the label carries the relation)."""
    if not statement:
        return statement
    return statement.strip().rstrip("。")


def _event(rs, now: datetime, relations) -> Optional[dict]:
    live = [r for r in rs if r2._status(r, now) == "当前"] or rs
    attr = lambda a: next((r for r in sorted(live, key=lambda r: (r.recorded_at, r.memory_id), reverse=True) if r.attribute == a), None)
    st, planned, done = attr("event.status"), attr("event.planned_time"), attr("event.time")
    actor = attr("event.actor")
    who = actor.value if actor and actor.value.startswith("person:") else "persona"
    label, stype, _ = subject_of(who, relations)
    text = rp.natural_event_statement(rs[0].subject, rs, now)          # "本人…" / "<name>…": the Core wording, owner kept
    status = (st.value if st else None) or "UNKNOWN"
    base = dict(subject=label, subject_type=stype, predicate="event", value=text, evidence_text=text, valid_from=None, valid_to=None,
                observed_at=_d(max(r.recorded_at for r in rs)), source_ref=(rs[0].provenance[0].source_id if rs[0].provenance else "memory"),
                confidence=_confidence(min(r.confidence for r in rs)), conflict_group=None, replaces=None, note=None,
                raw_subject=who)
    when = (planned.value if planned else None)
    if status == "CANCELLED":
        base.update(status="CANCELLED", note=f"{st.recorded_at.month}月{st.recorded_at.day}日记录为已取消")
    elif status == "COMPLETED":
        base.update(status="CURRENT", valid_from=(done.value[:10] if done else None))
    elif status in ("PLANNED", "SCHEDULED") and _date_of(when) and _date_of(when) > now.date():
        base.update(status="UPCOMING", valid_from=when[:10] if len(when) >= 10 else when[:7])
    else:
        base.update(status="CURRENT")
    return base
