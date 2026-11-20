"""Validators. Every check works from the evidence, the answer key and the text; none reads the realiser's frames, so a wrong frame or a
realiser bug is caught, not trusted.

* grounding: required values present (answer completeness), forbidden / protected values absent (including partial digit runs), values
  allowed only with a marker (past, cancelled, uncertain, future) carry it in the same clause, every named entity and number in the target is
  supported by allowed evidence or the conversation, required expressions (unknown, decline, permission ...) present, third-party facts not
  claimed as the speaker's own;
* temporal: evidence statuses agree with the dates and the conversation date;
* disclosure: protected values never in the target, the conversation or the examples;
* persona: no copied example, bounded catchphrases / openers, punctuation and bubbling consistent with the persona;
* generation hygiene: non-empty, no special tokens, no loop, bounded length.
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .evidence import Evidence
from .lexicon import all_entity_values
from .persona_synth import ADDONS, LATIN, QUESTION_BACK, REGISTER_OPENERS

ENTITIES = sorted(all_entity_values(), key=len, reverse=True)
MARKERS = {
    "UNKNOWN": r"还没|没定|不知道|不清楚|没记|忘了|记不清|没订|没有|不了解|没听|没问|没约上|排号|没出|没着落|没存|不太清楚|还不知道|没印象|没聊过|不太了解",
    "UNCERTAIN": r"好像|记不清|记不太清|说不准|不确定|还是|拿不准|可能|两个说法|还没确认|或者",
    "CANCELLED": r"取消|退了|退掉|不去了|不搬了",
    "PAST": r"以前|之前|原来|早|不在|不做|不干|不教|不开|不了|换|搬|改|上一家|停|旧",
    "FUTURE": r"才|要|下|准备|打算|月底|明年|过完年|还在",
    "DECLINE": r"不方便|不能|不好|就不|先不|不给|隐私|私事|别问|问.{0,5}吧|直接问|直接联系|自己给|不太想|不随便",
    "PERMISSION": r"先问|问下|问问|同意|问.{0,3}自己|愿不愿意",
}
CLAUSE_SPLIT = re.compile(r"[，,。！？!?；;～~\n]| (?=[^\sA-Za-z])")
LOOP = re.compile(r"(.{4,}?)\1{2,}", re.S)
SHORT_LOOP = re.compile(r"(.{1,3}?)\1{5,}", re.S)
SPECIAL = re.compile(r"<\|[^|<>\s]{1,64}\|>|</?think>")
DIGITS = re.compile(r"\d+[Xx]?")


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _clauses(text: str) -> List[str]:
    return [c for c in CLAUSE_SPLIT.split(text) if c and c.strip()]


def _digits_of(values: Iterable[str]) -> set:
    out = set()
    for v in values:
        out.update(DIGITS.findall(v))
    return out


def _value_in(value: str, text: str) -> bool:
    t = _norm(text)
    v = _norm(value)
    if v in t:
        return True
    ds = DIGITS.findall(v)
    # number-like values ("1280块" said as "1280元" / "1280"); not names that merely contain a digit ("拓途S3")
    if ds and re.fullmatch(r"[块元号月日点时分栋室万千百岁年个楼单]*", re.sub(r"\d+[Xx]?", "", v)):
        td = DIGITS.findall(t)
        if any(td[i:i + len(ds)] == ds for i in range(len(td) - len(ds) + 1)):  # same runs, same order ("5月10号" != "10月5号")
            return True
    return False


def entities_in(text: str) -> List[str]:
    t = _norm(text)
    found, taken = [], [False] * len(t)
    for e in ENTITIES:
        start = 0
        while True:
            i = t.find(e, start)
            if i < 0:
                break
            if not any(taken[i:i + len(e)]):
                found.append(e)
                for k in range(i, i + len(e)):
                    taken[k] = True
            start = i + 1
    return found


def grounding(target: str, evidence: Sequence[Evidence], key: Mapping, conversation: Sequence[Mapping]) -> List[str]:
    errs: List[str] = []
    t = _norm(target)
    conv_text = "".join(x["text"] for x in conversation)
    for v in key.get("required_values", []):
        if not _value_in(v, target):
            errs.append(f"missing_required:{v}")
    for v in key.get("forbidden_values", []):
        if _value_in(v, target):
            errs.append(f"forbidden_value:{v}")
        for run in DIGITS.findall(v):
            if len(run) >= 4:
                for i in range(len(run) - 3):
                    if run[i:i + 4] in t:
                        errs.append(f"protected_digits:{v}")
                        break
    for v, mk in key.get("marker_values", {}).items():
        if _value_in(v, target):
            cl = [c for c in _clauses(target) if _value_in(v, c)] or [target]
            if not any(re.search(MARKERS[mk], c) for c in cl) and not re.search(MARKERS[mk], target):
                errs.append(f"value_without_marker:{mk}:{v}")
    for v in key.get("historical_values", []):
        if v in key.get("marker_values", {}) or v in key.get("forbidden_values", []):
            continue
        if _value_in(v, target):
            cl = [c for c in _clauses(target) if _value_in(v, c)]
            if not any(re.search(MARKERS["PAST"] + "|" + MARKERS["CANCELLED"], c) for c in cl):
                errs.append(f"historical_as_current:{v}")
    for m in key.get("must_express", []):
        if m in ("THIRD_PARTY",):
            continue
        if not re.search(MARKERS[m], target):
            errs.append(f"missing_expression:{m}")
    for v in key.get("self_claim_forbidden_values", []):
        for c in _clauses(target):
            if _value_in(v, c) and not any(n in c for n in key.get("third_party_names", [])) and re.search(r"我", c):
                errs.append(f"third_party_claimed_as_self:{v}")
    for n in key.get("third_party_names", []):
        if key.get("category") in ("THIRD_PARTY", "MULTI_SUBJECT") and n not in t:
            errs.append(f"missing_third_party_name:{n}")
    # unsupported specifics: entities and numbers must come from disclosable evidence values, subjects or the conversation
    support = set(key.get("allowed_values", [])) | set(key.get("required_values", [])) | set(key.get("marker_values", {})) \
        | set(key.get("historical_values", [])) | set(key.get("third_party_names", []))
    allowed_text = "".join(e.evidence_text for e in evidence if e.disclosure_policy == "ALLOWED")
    support_text = _norm("".join(support) + conv_text + "".join(e.subject for e in evidence) + allowed_text)
    for ent in entities_in(target):
        if ent not in support_text and not any(len(s_) >= 2 and _norm(s_) in ent for s_ in support):
            errs.append(f"unsupported_entity:{ent}")
    sup_digits = _digits_of(list(support) + [conv_text])
    for d in DIGITS.findall(t):
        if d not in sup_digits:
            errs.append(f"unsupported_number:{d}")
    return errs


def temporal(evidence: Sequence[Evidence], now: str) -> List[str]:
    errs = []
    n = dt.date.fromisoformat(now)

    def d(x):
        return dt.date.fromisoformat(x) if x and len(x) == 10 else None
    for e in evidence:
        vf, vt = d(e.valid_from), d(e.valid_to)
        if e.status == "HISTORICAL" and not (vt and vt <= n):
            errs.append(f"historical_not_ended:{e.fact_id}")
        if e.status == "UPCOMING" and not (vf and vf > n):
            errs.append(f"upcoming_not_future:{e.fact_id}")
        if e.status in ("CURRENT", "CORRECTED") and vf and vf > n:
            errs.append(f"current_in_future:{e.fact_id}")
        if e.status in ("CURRENT", "CORRECTED") and vt and vt <= n:
            errs.append(f"current_already_ended:{e.fact_id}")
        if e.status == "CANCELLED" and vt and vt > n:
            errs.append(f"cancel_in_future:{e.fact_id}")
        if e.observed_at and d(e.observed_at) and d(e.observed_at) > n:
            errs.append(f"observed_after_now:{e.fact_id}")
    return errs


def disclosure(target: str, evidence: Sequence[Evidence], conversation: Sequence[Mapping], examples: Sequence[str]) -> List[str]:
    errs = []
    blob_c = _norm("".join(x["text"] for x in conversation))
    blob_x = _norm("".join(examples))
    for e in evidence:
        if e.disclosure_policy == "ALLOWED":
            continue
        v = _norm(e.value)
        if v in _norm(target):
            errs.append(f"protected_in_target:{e.fact_id}")
        if v in blob_c:
            errs.append(f"protected_in_conversation:{e.fact_id}")
        if v in blob_x:
            errs.append(f"protected_in_examples:{e.fact_id}")
    return errs


def conversation_consistency(conversation: Sequence[Mapping], evidence: Sequence[Evidence]) -> List[str]:
    errs = []
    if not conversation or conversation[-1]["speaker"] != "partner":
        errs.append("last_turn_not_partner")
    vals = _norm("".join(e.value + e.evidence_text for e in evidence if e.disclosure_policy == "ALLOWED"))
    for t in conversation:
        if t["speaker"] == "self":
            for ent in entities_in(t["text"]):
                if ent not in vals:
                    errs.append(f"self_turn_unsupported_entity:{ent}")
            for dg in DIGITS.findall(t["text"]):
                if dg not in vals:
                    errs.append(f"self_turn_unsupported_number:{dg}")
    return errs


def lcs_ratio(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    best = 0
    for ca in a:
        cur = [0] * (len(b) + 1)
        for j, cb in enumerate(b, 1):
            if ca == cb:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best / len(a)


_FILLERS = sorted({x for v in list(ADDONS.values()) + list(QUESTION_BACK.values()) + list(REGISTER_OPENERS.values()) for x in v} | set(LATIN), key=len, reverse=True)


def persona(target: str, examples: Sequence[str], style: Optional[Mapping] = None) -> List[str]:
    errs = []
    for x in examples:
        if _norm(x) == _norm(target):
            errs.append("copies_example")
        elif len(_norm(target)) >= 6 and lcs_ratio(target, x) >= 0.6:
            errs.append("near_copy_example")
    rest, n_fill = target, 0
    for f in _FILLERS:  # longest first, non-overlapping ("反正就这样" counts once, not also as "就这样")
        if f in rest:
            n_fill += rest.count(f)
            rest = rest.replace(f, "\x00")
    if n_fill > 3:
        errs.append(f"filler_overload:{n_fill}")
    if style:
        if style.get("multi_bubble", 0) == 0 and "\n" in target.strip():
            errs.append("bubbles_inconsistent")
        if style.get("end") == "none":
            for b in target.split("\n"):
                if b and b[-1] in "。！":
                    errs.append("end_punct_inconsistent")
                    break
    return errs


def hygiene(target: str, max_chars: int = 160) -> List[str]:
    errs = []
    if not target or not target.strip():
        errs.append("empty_target")
    if SPECIAL.search(target or ""):
        errs.append("special_token_in_target")
    if LOOP.search(target or "") or SHORT_LOOP.search(target or ""):
        errs.append("loop_in_target")
    if len(_norm(target)) > max_chars:
        errs.append("target_too_long")
    if target != target.strip():
        errs.append("untrimmed_target")
    if re.search(r"\n\s*\n", target or ""):
        errs.append("empty_bubble")
    return errs
