"""Runtime compatibility rule for a reader ABSTENTION that still names evidence (``stillhere.session.grounding_compat/1.0``).

Found on Batch 2.1 k10 ("我这周六有什么安排？"): the frozen reader abstains and explains "仅有一条‘去香港看演唱会’的计划，但没有日期，无法确认";
grounding v2 treats every abstention as unknown and keeps only units that share words with the question, so the real, related plan is
dropped and the reply says "没记录". The reader is NOT changed. The rule, applied after grounding:

* only when the reader abstained (``abstain`` true and the claim check passed) and the answer status is unknown / partial;
* a unit is kept only if its own Core text (event description, or the record value / statement) appears VERBATIM in the reader's
  explanation (>= 3 characters), i.e. the reader named that exact evidence; nothing from the reader's wording is used;
* the kept fact is the Core statement (owner kept: "本人…" or "<name>…"), marked with the uncertainty the reader stated being kept as the
  unknown item; an undated event is shown with "（没有记录具体日期）";
* answer status becomes partial; the unknown item stays, so the reply must still say the asked part is not known.
A third party's unit stays that person's (Core owner), so attribution answers ("那是小梅的计划") survive without being moved to "本人".
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from stillhere.runtime import pipeline as rp

COMPAT_VERSION = "stillhere.session.grounding_compat/1.0"
MIN_QUOTE = 3


def _unit_texts(key: str, rs) -> List[str]:
    if key.startswith("event:"):
        return [r.value for r in rs if r.attribute in ("event.description", "event.activity") and r.value]
    return [x for r in rs for x in (r.value, r.statement) if x]


def keep_reader_named_units(m, grounded: dict, reader: Optional[dict], reader_text: Optional[str], ask_time: datetime) -> dict:
    if not reader or not reader.get("abstain") or not reader.get("claim_check", {}).get("ok") or not reader_text:
        return grounded
    if grounded.get("answer_status") not in ("unknown", "partial"):
        return grounded
    by_id = {r.memory_id: r for r in m.evidence.records}
    shown = " ".join(f.get("statement", "") for f in grounded.get("facts", []))
    kept = []
    for key, ids in m.evidence.units:
        rs = [by_id[i] for i in ids if i in by_id]
        named = [t for t in _unit_texts(key, rs) if len(t) >= MIN_QUOTE and t in reader_text]
        if not rs or not named:
            continue
        if key.startswith("event:"):
            st = rp.natural_event_statement(key, rs, ask_time)
            if not any(r.attribute in ("event.planned_time", "event.time") for r in rs):
                st += "（没有记录具体日期）"
        else:
            r = rs[0]
            if getattr(r.status, "value", str(r.status)) != "active":
                continue
            st = r.statement
        if st and st not in shown:
            kept.append({"statement": st, "role": "current", "source": "reader-named evidence", "kept_by": COMPAT_VERSION})
    if not kept:
        return grounded
    out = dict(grounded)
    out["facts"] = list(grounded.get("facts", [])) + kept
    out["answer_status"] = "partial"
    out["status"] = "facts"
    if not out.get("unknown"):
        out["unknown"] = [rp.question_core(grounded.get("question", "")) or "所问的具体安排"]
    return out
