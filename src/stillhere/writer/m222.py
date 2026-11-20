"""Batch 2.1 final writer rules (M2.2.2), behind ``MEMORY_WRITER_M222_ENABLED`` (needs M2.2 on; off = the writer exactly as before). They act
after verification only, so they stack on M2.2 (conf1 / conf2 worlds) or M2.2.1 (conf3) without changing either.

R13 explicit self-correction in the same message: the verifier drops a ``corrects`` flag whose cited span has no correction word. When the
    persona's OWN message that holds the span says so explicitly ("我之前说错了，我住的是福田区，不是南山区"), the flag is restored, so the
    correction supersedes the old value instead of leaving two current values (found on b21p-dataset-5.1, question p15). Only the explicit
    words count (说错 / 记错 / 搞错 / 弄错 / 更正 / 纠正); the weak cues (其实 / 不是 / 应该是) still have to be in the span itself.
R14 public-holiday dates: the verifier drops a date it cannot find in the text ("国庆最后一天我去爬了峨眉山" -> event_time 2026-10-07 dropped,
    found on b21-confirm-3 r07). A dropped day-precision date is restored when it lies inside the holiday window the cited message names
    (``relative_days.holiday``, anchored on the message time; usual spans of 国庆 / 五一 / 元旦 only, with 第N天 / 最后一天 resolved exactly).
    A whole-holiday phrase ("国庆打算回扬州老家") names no day: the date is kept at MONTH precision ("2026-10"), the weaker claim the text
    supports (like SCHEDULED -> PLANNED), never a guessed day; a date outside the span is never kept.
"""
from __future__ import annotations

import os
import re
from typing import Mapping, Optional, Sequence

from ..memory import relative_days
from . import m22
from .dates import parse_iso

FLAG = "MEMORY_WRITER_M222_ENABLED"
EXPLICIT_CORRECTION = ("说错", "记错", "搞错", "弄错", "更正", "纠正")


def enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    e = env if env is not None else os.environ
    return m22.enabled(e) and e.get(FLAG, "").strip().lower() in ("1", "true", "yes", "on")


def _compact(t: str) -> str:
    return re.sub(r"\s+", "", t)


def restore_explicit_correction(original_corrects: bool, verdict, msgs: Sequence, persona_id: str) -> bool:
    """R13. True when the flag was restored. ``original_corrects`` is the candidate's flag before verification."""
    c = verdict.candidate
    if not original_corrects or c.corrects or "corrects" not in verdict.dropped_fields or not verdict.committable:
        return False
    own = [m for m in msgs if m.message_id in c.source_message_ids and m.speaker_id == persona_id
           and any(_compact(s) and _compact(s) in _compact(m.content) for s in c.source_spans)]
    if not any(w in m.content for m in own for w in EXPLICIT_CORRECTION):
        return False
    c.corrects = True
    verdict.dropped_fields = [f for f in verdict.dropped_fields if f != "corrects"]
    verdict.reasons.append("R13 explicit self-correction in the same message: corrects kept")
    if not verdict.dropped_fields and verdict.support == "PARTIALLY_SUPPORTED":
        verdict.support = "SUPPORTED"
    return True


DATE_FIELDS = ("planned_time", "event_time", "valid_from", "valid_to")


def restore_holiday_dates(original: Mapping[str, Optional[str]], verdict, msgs: Sequence) -> list:
    """R14. The names of the restored fields. ``original`` holds the candidate's date fields before verification."""
    c = verdict.candidate
    if not verdict.committable:
        return []
    cited = [m for m in msgs if m.message_id in c.source_message_ids]
    done = []
    for name in DATE_FIELDS:
        value = original.get(name)
        if not value or getattr(c, name) or name not in verdict.dropped_fields:
            continue
        got, precision = parse_iso(value)
        if got is None or precision not in ("day", "exact"):
            continue
        for m in cited:
            w = relative_days.holiday(m.content, m.timestamp)
            if w is not None and w.start.date() <= got < w.end.date():
                exact = (w.end - w.start).days == 1                     # 国庆最后一天 / 五一第二天: one day
                setattr(c, name, value if exact else f"{got.year:04d}-{got.month:02d}")
                done.append(name)
                break
    if done:
        verdict.dropped_fields = [f for f in verdict.dropped_fields if f not in done]
        verdict.reasons.append(f"R14 holiday date kept: {', '.join(done)}")
        if not verdict.dropped_fields and verdict.support == "PARTIALLY_SUPPORTED":
            verdict.support = "SUPPORTED"
    return done


# R15 explicit correction of a MULTI-valued slot (pet / relationship / likes / dislikes / routine ...; no field is hard-coded): the writer
# superseded only single-valued slots, so "我养的是一只猫，不是狗" kept both 狗 and 猫 current (found on b21f-confirm-6 f08 / f15). A candidate
# flagged as a correction replaces ONE earlier value of the same slot when the persona's own cited message names that value as wrong
# ("不是Y" / "不叫Y"), or says so explicitly (说错 / 记错 / 更正 ...) and exactly one earlier value fits (for a relationship: the same role,
# 表妹 -> 表妹). Never for an addition ("又养了", "还有个", "也喜欢") or a change over time ("以前…现在…"): those keep both values (or the
# normal temporal handling). The old record is superseded (history and provenance kept), never deleted.
ADDITION_CUES = ("又养", "又有", "又多", "还有个", "还有一", "还养", "另外", "也喜欢", "也爱", "也养", "再养", "多了一")
CHANGE_OVER_TIME_CUES = ("以前", "曾经", "过去", "后来", "如今", "现在养的", "现在的", "已经不")
_NEGATED = re.compile(r"(?:不是|不叫)([^，。,；;！!？?\s]{1,12})")


def _tokens(value: str):
    return [p for p in re.split(r"[\s:：/、，,]+", value or "") if p]


def multi_correction_target(c, rec, different: Sequence, msgs: Sequence, persona_id: str):
    """R15: the earlier record an explicit multi-valued correction replaces, or None (then the writer adds the value as before)."""
    if not c.corrects or not different:
        return None
    own = [m for m in msgs if m.message_id in c.source_message_ids and m.speaker_id == persona_id
           and any(_compact(s) and _compact(s) in _compact(m.content) for s in c.source_spans)]
    if not own:
        return None
    text = " ".join(m.content for m in own)
    if any(w in text for w in ADDITION_CUES) or any(w in text for w in CHANGE_OVER_TIME_CUES):
        return None
    negated = [g for g in _NEGATED.findall(text)]
    named = [r for r in different if any(tok in g or (len(g) >= 1 and g in _compact(r.value)) for g in negated for tok in _tokens(r.value) if tok)]
    if len(named) == 1:
        return named[0]
    if named:
        return None                                   # several earlier values named: ambiguous, keep both
    if not any(w in text for w in EXPLICIT_CORRECTION):
        return None
    role = _tokens(rec.value)[0] if rec.attribute == "relationship" and len(_tokens(rec.value)) > 1 else None
    pool = [r for r in different if role is None or (_tokens(r.value) and _tokens(r.value)[0] == role)]
    return pool[0] if len(pool) == 1 else None
