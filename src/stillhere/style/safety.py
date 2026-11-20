"""Style runtime safety: a model-agnostic guard layer around ONE generation call (base model or LoRA adapter, local or remote).

The adapter's weights are never touched; the guards only read the generated text, the grounded content (the facts the reply may use, from
memory) and the persona's own earlier messages. Every decision is deterministic and recorded in an audit trace.

Flags (all OFF by default: with every flag off the generator is called once and its text is returned unchanged, exactly as before):

    GUARD_ENABLED               guard master switch
    REPETITION_GUARD_ENABLED    degenerate-loop detection (emphasis is not a loop)                          (guard sub-flag)
    MUST_NOT_GUARD_ENABLED      evidence-derived forbidden content (corrected old values, explicit must-not)  (guard sub-flag)
    CLAIM_GUARD_ENABLED         unsupported specifics (gate) / dropped key facts, unknown not said (advisory)  (guard sub-flag)
    FALLBACK_ENABLED            when every attempt fails: the plain grounded answer instead of a failed text   (guard sub-flag)
    LENGTH_POLICY_ENABLED       PRE-GENERATION persona length policy (Batch 2.1; independent of the guard): a length rule from the
                                author's OWN history passed through the contract's existing ``constraints`` input + a per-author
                                ``max_new_tokens`` cap with a content floor (``pre_generation_plan``)
    POSTHOC_TRIM_ENABLED        post-hoc trim of an over-long reply at the policy's hard cap (Batch 2.1: separated; analysis only)

With ``GUARD_ENABLED`` on, a guard sub-flag that is unset counts as ON; set it to 0 to switch one guard off. Retries are bounded
(``max_retries``, default 1) and each retry changes only sampling (seed, a repetition-penalty step), never the prompt contract
(``stillhere.style.prompt/2.0`` is untouched). Runtime fields are versioned separately as ``RUNTIME_SCHEMA_VERSION``.
"""
from __future__ import annotations

import math
import os
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from . import grounding_checks as gc

RUNTIME_SCHEMA_VERSION = "stillhere.style.runtime/1.2"      # 1.1 (Batch 2.1): pre-generation length plan, post-hoc trim separated;
                                                           # 1.2 (post-GPU): content-aware target, unknown gate, leak gate, safe fallback
GUARD_SUBFLAGS = ("REPETITION_GUARD_ENABLED", "MUST_NOT_GUARD_ENABLED", "CLAIM_GUARD_ENABLED", "FALLBACK_ENABLED")
INDEPENDENT = ("LENGTH_POLICY_ENABLED", "POSTHOC_TRIM_ENABLED")
FLAGS = ("GUARD_ENABLED",) + GUARD_SUBFLAGS + INDEPENDENT
_ON = ("1", "true", "yes", "on")
_OFF = ("0", "false", "no", "off")


def flags(env: Optional[Mapping[str, str]] = None) -> Dict[str, bool]:
    e = env if env is not None else os.environ
    master = e.get("GUARD_ENABLED", "").strip().lower() in _ON
    out = {"GUARD_ENABLED": master}
    for f in GUARD_SUBFLAGS:
        v = e.get(f, "").strip().lower()
        out[f] = master and (v not in _OFF)
    for f in INDEPENDENT:
        out[f] = e.get(f, "").strip().lower() in _ON
    return out


# ----------------------------------------------------------------------------------------------------------------------------------- repetition
LOOP = re.compile(r"(.{4,}?)\1{2,}", re.S)           # Batch 1 detector: a span of >= 4 chars repeated >= 3 times in a row
SHORT_LOOP = re.compile(r"(.{1,3}?)\1{5,}", re.S)    # a 1-3 char unit repeated >= 6 times
EMPHASIS_UNIT = re.compile(r"^[哈呵嘿嘻啊呀哇哦噢嗯唔呜啦嘛对好是行冲顶赞牛6!！?？。.~～、，,…—\-+]+$")
EMPHASIS_MAX = 24                                    # an emphasis run longer than this many characters is a loop anyway


@dataclass
class RepetitionVerdict:
    degenerate: bool
    kind: Optional[str] = None
    span: str = ""
    repeats: int = 0
    emphasis_allowed: List[str] = field(default_factory=list)


def repetition(text: str, *, hit_token_limit: bool = False) -> RepetitionVerdict:
    """Degenerate loop vs emphasis.

    * a span of >= 4 characters repeated >= 3 times in a row is a loop, unless the span is made only of laughter / interjection / punctuation
      characters and the whole run is short (``哈哈哈哈哈哈``, ``对对对对``, ``！！！！``: emphasis);
    * a 1-3 character unit repeated >= 6 times is a loop under the same emphasis exemption (``好好好好好好好`` is emphasis, ``的的的的的的`` is not);
    * three or more lines where at most half are distinct is a loop (the model restarting the same line);
    * a reply that hit the token limit and ends inside a repeated span is a loop even with two repeats.
    """
    t = re.sub(r"\s+", "", text or "")
    v = RepetitionVerdict(False)
    # the last copy of a clause loop often lacks its trailing separator ("X，X，X"): also test with one separator appended
    variants = [t] + ([t + s for s in "，。、；"] if t and t[-1] not in "，,。、；;！!？?~～" else [])
    for pat, kind in ((LOOP, "span_loop"), (SHORT_LOOP, "short_unit_loop")):
        for vi, m in ((vi, m) for vi, t_ in enumerate(variants) for m in pat.finditer(t_)):
            unit, run = m.group(1), m.group(0)
            if vi and len(run) < 16:     # a short three-fold echo with no final separator ("我也是，我也是，我也是") is emphasis, as in Batch 1
                continue
            if EMPHASIS_UNIT.match(unit) and len(run) <= EMPHASIS_MAX:
                v.emphasis_allowed.append(run)
                continue
            return RepetitionVerdict(True, kind, unit, len(run) // max(1, len(unit)), v.emphasis_allowed)
    lines = [x.strip() for x in (text or "").splitlines() if x.strip()]
    if len(lines) >= 3 and len(set(lines)) <= len(lines) // 2:
        dup = max(set(lines), key=lines.count)
        return RepetitionVerdict(True, "line_loop", dup, lines.count(dup), v.emphasis_allowed)
    if hit_token_limit and len(t) >= 16:
        tail = t[-max(8, len(t) // 3):]
        m = re.search(r"(.{3,}?)\1+$", tail)
        if m and not EMPHASIS_UNIT.match(m.group(1)):
            return RepetitionVerdict(True, "truncated_loop", m.group(1), len(m.group(0)) // len(m.group(1)), v.emphasis_allowed)
    return v


def trim_repetition(text: str, verdict: RepetitionVerdict) -> str:
    """Repair: keep the text up to (and including) the FIRST occurrence of the repeated unit; '' when nothing sensible is left."""
    if not verdict.degenerate or not verdict.span:
        return text
    if verdict.kind == "line_loop":
        seen, out = set(), []
        for line in text.splitlines():
            s = line.strip()
            if s and s in seen:
                continue
            seen.add(s)
            out.append(line)
        return "\n".join(out).strip()
    i = text.find(verdict.span)
    if i < 0:   # whitespace differs: work on the stripped text
        flat = re.sub(r"\s+", "", text)
        j = flat.find(verdict.span)
        return flat[: j + len(verdict.span)].strip() if j >= 0 else ""
    return text[: i + len(verdict.span)].rstrip("，,、 ").strip()


# ------------------------------------------------------------------------------------------------------------------------------------ must-not
PAST_MARK = re.compile(r"以前|之前|原来|原先|曾经|过去|当时|那时|后来|改成|改到|改了|换成|已经不|不再|记错|搞错|弄错|更正|纠正|其实|不是")
SPECIFIC = re.compile(r"\d+(?:\.\d+)?(?:年|月|日|号|点|分|元|块|岁|次|个|天|周|小时|公里|km|%)?|[A-Za-z][A-Za-z0-9+#.\-]{1,}")


def _changed_part(old: str, new: str) -> str:
    """The part of ``old`` that the correction changed: ``old`` minus the prefix and suffix it shares with ``new``
    ("本人的职业是退休的会计" vs "...退休的出纳" -> "会计")."""
    o, n = gc.norm(old), gc.norm(new)
    i = 0
    while i < min(len(o), len(n)) and o[i] == n[i]:
        i += 1
    j = 0
    while j < min(len(o), len(n)) - i and o[len(o) - 1 - j] == n[len(n) - 1 - j]:
        j += 1
    return o[i:len(o) - j].strip("，,。的了是 ")


def forbidden_terms(grounded: Optional[Mapping], explicit: Sequence[str] = ()) -> List[dict]:
    """The terms this reply must not assert, derived from the evidence for THIS reply (not a global blacklist):

    * explicit ``must_not`` terms supplied with the grounded content (e.g. guesses for an unknown item);
    * the specifics (numbers / dates / Latin names) of a corrected fact's old value (``replaces``) that the corrected statement does not
      contain: saying them as current is stating a value known to be wrong. Mentioning them WITH a correction / past marker is allowed;
    A PAST fact (something that happened, e.g. "上周搬家花了380元") is still true and is never forbidden: only a value known to be WRONG
    (the corrected one) is. Time-awareness is the marker rule: an old value said as old ("之前记成45元") is allowed, said as current is not.
    """
    out: List[dict] = []
    facts = (grounded or {}).get("facts", [])
    current = " ".join(f.get("statement", "") for f in facts if f.get("role") in ("current", "corrected", "at_time"))
    olds = gc.norm(" ".join(f.get("replaces", "") for f in facts if f.get("role") == "corrected"))
    for t in explicit:
        if gc.norm(t):
            # an explicit term that is a corrected old value may be mentioned as the corrected one (as the Batch 1 scorer allows)
            out.append({"term": t, "why": "explicit", "allow_with_marker": gc.norm(t) in olds})
    for f in facts:
        if f.get("role") == "corrected" and f.get("replaces"):
            for m in SPECIFIC.finditer(f["replaces"]):
                if m.group(0) not in f.get("statement", "") and m.group(0) not in current:
                    out.append({"term": m.group(0), "why": "corrected_old_value", "allow_with_marker": True})
            diff = _changed_part(f["replaces"], f.get("statement", ""))
            if diff and len(diff) >= 2 and diff not in current and not SPECIFIC.fullmatch(diff):
                out.append({"term": diff, "why": "corrected_old_value", "allow_with_marker": True})
    seen, uniq = set(), []
    for d in out:
        if d["term"] not in seen:
            seen.add(d["term"])
            uniq.append(d)
    return uniq


def _contains(hay: str, term: str) -> bool:
    """Substring match, except that a term starting / ending with a digit must not be part of a longer number ("2日" is not in "12日")."""
    pat = re.escape(term)
    if term[0].isdigit():
        pat = r"(?<![\d.])" + pat
    if term[-1].isdigit():
        pat = pat + r"(?![\d.])"
    return re.search(pat, hay) is not None


def must_not(text: str, grounded: Optional[Mapping], explicit: Sequence[str] = ()) -> List[str]:
    """Violations: forbidden terms asserted in the reply. A term that is allowed with a marker passes when a past / correction marker is in
    the same clause."""
    viol = []
    r = gc.norm(text)
    for d in forbidden_terms(grounded, explicit):
        term = gc.norm(d["term"])
        if not term or not _contains(r, term):
            continue
        if d["allow_with_marker"]:
            clauses = [c for c in re.split(r"[。！？!?；;\n]", text) if term in gc.norm(c)]
            if clauses and all(PAST_MARK.search(c) for c in clauses):
                continue
        viol.append(f"{d['why']}:{d['term']}")
    return viol


# ---------------------------------------------------------------------------------------------------------------------------------- claim guard
ADVISORY = ("dropped_key_terms", "unknown_not_said_as_unknown")
# Batch 2.1 post-GPU: when grounding v2 says the question is NOT answered by memory (answer_status unknown / partial), saying so is a GATE
UNKNOWN_SAID = re.compile(gc.UNCERTAIN.pattern + r"|没有?(提|说|聊)过|没有?记录|没记|不太记得|记不太清|没听你说|你没说过|不知道")
# internal memory structure must never reach a user-visible reply (it came from the M3.1 evidence line format and from record ids)
LEAK = re.compile(r"当事人[：:]|状态记录|记录于|事件[：:]|event:|\bw\.[0-9a-f]{6,}|（当前）|原记录有误|关于[：:]本人|有效 \d{4}|reader\(|source|provenance|\{\s*\"")


def evidence_texts(conversation: Sequence[Mapping]) -> List[str]:
    """Conversation text that may SUPPORT a specific in a reply: the other side's messages only. The persona's own earlier replies
    (``speaker == "self"``) and anything marked ``generated`` are model output, never evidence (GENERATED_OUTPUT_NOT_FACT_EVIDENCE: a number,
    name or price the model invented in one turn must not pass the gate in the next)."""
    return [m.get("text", "") for m in conversation if m.get("speaker") != "self" and not m.get("generated")]


def claims(text: str, grounded: Optional[Mapping], conversation: Sequence[Mapping], key_terms: Sequence[str] = ()) -> List[str]:
    """Unsupported specifics (a GATE: a number / name the facts and the conversation do not contain), plus two ADVISORY findings that are
    reported but never fail a reply on their own: dropped key facts (a style reply may legitimately answer only the part asked) and unknown
    items not said as unknown (the reply may simply not touch the unknown item; a guessed value is caught by the gate / must-not)."""
    facts = (grounded or {}).get("facts", [])
    sources = [f.get("statement", "") for f in facts] + [f.get("time") or "" for f in facts] + evidence_texts(conversation)
    problems = []
    extra = gc.unsupported_specifics(text, *sources)
    if extra:
        problems.append("unsupported_specifics:" + ",".join(extra[:5]))
    shown = gc.norm(" ".join(f.get("statement", "") for f in facts))
    missing = [t for t in key_terms if gc.norm(t) and gc.norm(t) in shown and not gc.key_term_present(t, text)]
    if missing:
        problems.append("dropped_key_terms:" + ",".join(missing[:5]))
    if (grounded or {}).get("answer_status") in ("unknown", "partial"):
        if not UNKNOWN_SAID.search(text):
            problems.append("unknown_not_admitted")             # gate (grounding v2): memory does not hold the answer
    elif (grounded or {}).get("unknown") and not gc.UNCERTAIN.search(text):
        problems.append("unknown_not_said_as_unknown")
    return problems


# ---------------------------------------------------------------------------------------------------------------------------- length policy
@dataclass
class LengthPolicy:
    source: str                 # "author_history" | "fallback"
    n_history: int
    median_chars: Optional[float]
    soft_low: int
    soft_high: int
    hard_cap: int
    max_new_tokens: int

    def within(self, text: str) -> bool:
        n = len(gc.norm(text))
        return self.soft_low <= n <= self.soft_high


FALLBACK_POLICY = dict(soft_low=2, soft_high=120, hard_cap=240, max_new_tokens=160)


def length_policy(history_texts: Sequence[str], *, min_history: int = 8, chars_per_token: float = 1.3, default_max_new_tokens: int = 160,
                  reference_reply: Optional[str] = None) -> LengthPolicy:
    """From the author's OWN earlier messages (never the reply being predicted: ``reference_reply`` is removed if present).

    soft range = [p10, p90] of the author's lengths (widened by 25%); hard cap = max(2 x p90, 24) characters; max_new_tokens = enough for the
    hard cap, never above the default. Fewer than ``min_history`` messages -> the conservative fallback (the old default budget).
    """
    ref = gc.norm(reference_reply or "")
    lens = sorted(len(gc.norm(t)) for t in history_texts if gc.norm(t) and gc.norm(t) != ref)
    if len(lens) < min_history:
        return LengthPolicy("fallback", len(lens), None, **FALLBACK_POLICY)
    q = lambda p: lens[min(len(lens) - 1, max(0, int(round(p * (len(lens) - 1)))))]
    lo, hi = q(0.10), q(0.90)
    soft_low, soft_high = max(1, int(lo * 0.75)), int(math.ceil(hi * 1.25)) + 2
    hard = max(24, 2 * hi)
    mnt = min(default_max_new_tokens, max(16, int(math.ceil(hard / chars_per_token)) + 8))
    return LengthPolicy("author_history", len(lens), float(statistics.median(lens)), soft_low, soft_high, hard, mnt)


LENGTH_POLICY_RULE = "回复长度：按这个人平时说话的长度，大约{n}个字以内；要说的事实必须说完整，不要为了变短删掉事实。"
CONTENT_FLOOR_CHARS = 15        # a reply that must carry memory facts is never asked to be shorter than this
CONTENT_FLOOR_TOKENS = 48       # ... nor capped below this many new tokens


@dataclass
class LengthPlan:
    """The PRE-GENERATION length condition of one reply (decided before generation, from the author's own history only)."""
    source: str                     # author_history | fallback
    target_chars: Optional[int]     # the length named in the rule (None: no rule, conservative default)
    constraint: Optional[str]       # the extra generation rule (contract ``constraints`` input); None = prompt unchanged
    max_new_tokens: int             # the generation cap (an upper bound: it does not make the model shorter by itself)
    content_floor_applied: bool
    policy: LengthPolicy
    version: str = RUNTIME_SCHEMA_VERSION


CONTENT_TARGET_MAX = 40       # a content-driven target never exceeds this many characters


def pre_generation_plan(history_texts: Sequence[str], *, grounded: Optional[Mapping] = None, default_max_new_tokens: int = 128,
                        reference_reply: Optional[str] = None, content_chars: Optional[int] = None) -> LengthPlan:
    """target = the author's median length rounded up to a multiple of 5 (at least 5); cap = the policy's token budget (2 x p90 chars).
    When the reply has memory facts to carry, both get a content floor (CONTENT_FLOOR_CHARS / _TOKENS): length never wins over facts.
    Too little history (< 8 messages) -> no rule and the default budget (conservative fallback)."""
    pol = length_policy(history_texts, default_max_new_tokens=default_max_new_tokens, reference_reply=reference_reply)
    has_facts = bool(grounded) and grounded.get("status") != "no_memory" and bool(grounded.get("facts"))
    if pol.source != "author_history":
        return LengthPlan("fallback", None, None, default_max_new_tokens, False, pol)
    n = max(5, int(math.ceil((pol.median_chars or 5) / 5.0) * 5))
    cap = pol.max_new_tokens
    floor = False
    if has_facts and (n < CONTENT_FLOOR_CHARS or cap < CONTENT_FLOOR_TOKENS):
        n, cap, floor = max(n, CONTENT_FLOOR_CHARS), max(cap, CONTENT_FLOOR_TOKENS), True
    if content_chars:      # post-GPU: the length of what must be said (the checked answer), not a fixed floor, sets the minimum target
        need = min(CONTENT_TARGET_MAX, max(CONTENT_FLOOR_CHARS, int(math.ceil(content_chars / 5.0) * 5)))
        if need > n:
            n, floor = need, True
            cap = max(cap, int(math.ceil(need / 1.3)) + 16)
    cap = min(cap, default_max_new_tokens)
    return LengthPlan("author_history", n, LENGTH_POLICY_RULE.format(n=n), cap, floor, pol)


def trim_to(text: str, cap: int) -> str:
    """Over the hard cap: cut at the last sentence / clause end before the cap (never mid-word when a boundary exists)."""
    if len(gc.norm(text)) <= cap:
        return text
    cut, count = len(text), 0
    for i, ch in enumerate(text):
        if not ch.isspace():
            count += 1
        if count >= cap:
            cut = i + 1
            break
    head = text[:cut]
    m = max(head.rfind(c) for c in "。！？!?\n；;…~～")
    if m < cap // 3:
        m = max(head.rfind(c) for c in "，,、 ")
    return (head[: m + 1] if m >= cap // 3 else head).strip()


# ------------------------------------------------------------------------------------------------------------------------------- the generator
@dataclass
class Attempt:
    index: int
    text: str
    config: dict
    checks: Dict[str, object]
    action: str                 # accept | repaired | retry | fail
    latency_ms: float


@dataclass
class SafeResult:
    text: str
    status: str                 # PASS_THROUGH | ACCEPTED | REPAIRED | RETRIED_ACCEPTED | FALLBACK | FAILED_NO_FALLBACK
    flags: Dict[str, bool]
    backend: str
    attempts: List[Attempt] = field(default_factory=list)
    length_policy: Optional[dict] = None
    runtime_schema: str = RUNTIME_SCHEMA_VERSION
    fallback_kind: Optional[str] = None          # SAFE_PARTIAL | NATURAL_UNKNOWN | CONFLICT | MINIMAL (Batch 2.1 post-GPU)

    def audit(self) -> dict:
        """The trace a run records (texts of failed attempts included: they are the persona's generated text, store with the run's privacy)."""
        return asdict(self)


def check_text(text: str, f: Mapping[str, bool], *, grounded=None, conversation=(), key_terms=(), must_not_terms=(), hit_token_limit=False,
               policy: Optional[LengthPolicy] = None) -> Dict[str, object]:
    out: Dict[str, object] = {}
    if not (text or "").strip():
        out["empty"] = True
    if LEAK.search(text or ""):
        out["internal_leak"] = LEAK.search(text).group(0)
    if f.get("REPETITION_GUARD_ENABLED"):
        v = repetition(text, hit_token_limit=hit_token_limit)
        out["repetition"] = asdict(v)
    if f.get("MUST_NOT_GUARD_ENABLED"):
        out["must_not"] = must_not(text, grounded, must_not_terms)
    if f.get("CLAIM_GUARD_ENABLED") and grounded is not None:
        out["claims"] = claims(text, grounded, conversation, key_terms)
    if f.get("LENGTH_POLICY_ENABLED") and policy is not None:
        out["length_within_soft_range"] = policy.within(text)       # reported, never a failure by itself (soft constraint)
    return out


def failed(checks: Mapping[str, object]) -> List[str]:
    why = []
    if checks.get("empty"):
        why.append("empty")
    if checks.get("internal_leak"):
        why.append("internal_leak")
    rep = checks.get("repetition")
    if rep and rep.get("degenerate"):
        why.append("repetition:" + str(rep.get("kind")))
    if checks.get("must_not"):
        why.append("must_not")
    gate = [c for c in (checks.get("claims") or []) if not c.startswith(ADVISORY)]
    if gate:
        why.append("claims")
    return why


def safe_generate(generate: Callable[[List[dict], dict], dict], messages: List[dict], config: Mapping, *, backend: str = "unknown",
                  grounded: Optional[Mapping] = None, conversation: Sequence[Mapping] = (), key_terms: Sequence[str] = (),
                  must_not_terms: Sequence[str] = (), history_texts: Sequence[str] = (), fallback_text: Optional[str] = None,
                  max_retries: int = 1, env: Optional[Mapping[str, str]] = None, length_plan: Optional[LengthPlan] = None) -> SafeResult:
    """``generate(messages, config) -> {"text", "hit_token_limit"?}``: the backend call; base and LoRA use this same interface.

    ``length_plan`` (from ``pre_generation_plan``; the caller has already put its rule into the contract's ``constraints``): its token cap is
    applied to every attempt when LENGTH_POLICY_ENABLED. With POSTHOC_TRIM_ENABLED (default off) an over-long reply is trimmed at the hard cap.
    Flow: generate -> checks -> accept | repair (repetition trim, re-checked) | retry (bounded, sampling change only) -> fallback.
    """
    f = flags(env)
    cfg = dict(config)
    if f["LENGTH_POLICY_ENABLED"] and length_plan is None:
        length_plan = pre_generation_plan(history_texts, grounded=grounded, default_max_new_tokens=int(cfg.get("max_new_tokens", 160)))
    if f["LENGTH_POLICY_ENABLED"]:
        cfg["max_new_tokens"] = min(int(cfg.get("max_new_tokens", 160)), length_plan.max_new_tokens)
    policy = length_plan.policy if (f["LENGTH_POLICY_ENABLED"] and length_plan is not None) else None
    if not f["GUARD_ENABLED"]:
        t0 = time.perf_counter()
        out = generate(messages, dict(cfg))
        text = out.get("text") or ""
        if f["POSTHOC_TRIM_ENABLED"] and policy is not None:
            text = trim_to(text, policy.hard_cap)
        a = Attempt(0, text, dict(cfg), {}, "accept", round((time.perf_counter() - t0) * 1000, 1))
        return SafeResult(text, "PASS_THROUGH", f, backend, [a], length_policy=asdict(length_plan) if policy else None)
    res = SafeResult("", "FAILED_NO_FALLBACK", f, backend, length_policy=asdict(length_plan) if policy else None)
    kw = dict(grounded=grounded, conversation=conversation, key_terms=key_terms, must_not_terms=must_not_terms, policy=policy)
    for i in range(max(0, max_retries) + 1):
        c = dict(cfg)
        if i:
            c["seed"] = int(c.get("seed", 0)) + 1000 * i
            c["repetition_penalty"] = round(float(c.get("repetition_penalty", 1.0)) + 0.1 * i, 2)
        t0 = time.perf_counter()
        out = generate(messages, c)
        text = (out.get("text") or "").strip()
        hit = bool(out.get("hit_token_limit"))
        if f["POSTHOC_TRIM_ENABLED"] and policy is not None:
            text = trim_to(text, policy.hard_cap)
        ch = check_text(text, f, hit_token_limit=hit, **kw)
        why = failed(ch)
        lat = round((time.perf_counter() - t0) * 1000, 1)
        if not why:
            res.attempts.append(Attempt(i, text, c, ch, "accept", lat))
            res.text, res.status = text, ("ACCEPTED" if i == 0 else "RETRIED_ACCEPTED")
            return res
        # repair: a degenerate loop is cut back to its first occurrence; the repaired text must pass every check again
        rep = ch.get("repetition") or {}
        if rep.get("degenerate") and set(why) <= {"repetition:" + str(rep.get("kind"))}:
            fixed = trim_repetition(text, RepetitionVerdict(**rep))
            ch2 = check_text(fixed, f, **kw)
            if fixed and len(gc.norm(fixed)) >= 2 and not failed(ch2):
                res.attempts.append(Attempt(i, text, c, ch, "repaired", lat))
                res.text, res.status = fixed, "REPAIRED"
                return res
        res.attempts.append(Attempt(i, text, c, ch, "retry" if i < max_retries else "fail", lat))
    if f["FALLBACK_ENABLED"]:
        if grounded is not None:
            fb, kind = safe_fallback(grounded, conversation, f, must_not_terms)
        else:
            fb, kind = (fallback_text, "CALLER") if fallback_text and not LEAK.search(fallback_text) else (MINIMAL_REPLY, "MINIMAL")
        if fb:
            res.text, res.status, res.fallback_kind = fb, "FALLBACK", kind
            return res
    res.text = res.attempts[-1].text if res.attempts else ""
    res.status = "FAILED_NO_FALLBACK"
    return res


MINIMAL_REPLY = "这个我记不太清了。"


_REL = re.compile(r"^本人的(\S{1,4}) (\S{1,8})$")                      # "本人的弟弟 顾林"
_FACT_TEMPLATES = ((re.compile(r"^本人的职业是(.+)$"), "你是{0}"), (re.compile(r"^本人住在(.+)$"), "你住在{0}"), (re.compile(r"^本人在(.+)工作$"), "你在{0}工作"),
                   (re.compile(r"^本人在(.*(?:大学|学院|学校|中学|小学|高中|初中|幼儿园|大专|技校))上学$"), "你在{0}上学"),
                   (re.compile(r"^本人在(.+)上学$"), "你读的是{0}"),          # a major stored in the school slot (c43): "你在软件工程上学" reads wrong (re.compile(r"^本人喜欢(.+)$"), "你喜欢{0}"), (re.compile(r"^本人不喜欢(.+)$"), "你不喜欢{0}"),
                   (re.compile(r"^本人养了(.+)$"), "你养了{0}"), (re.compile(r"^本人通常(.+)$"), "你平时{0}"))


def _second_person(statement: str) -> str:
    """A stored third-person statement as a plain second-person clause ("本人的弟弟 顾林" -> "你的弟弟叫顾林")."""
    s_ = re.sub(r"[“”]", "", statement.strip()).rstrip("。；;，, ")
    m_ = _REL.match(s_)
    if m_:
        return f"你的{m_.group(1)}叫{m_.group(2)}"
    for pat, tpl in _FACT_TEMPLATES:
        m_ = pat.match(s_)
        if m_:
            return tpl.format(m_.group(1))
    s_ = re.sub(r"^本人的", "你的", s_)
    s_ = re.sub(r"^本人", "你", s_)
    return s_.replace("本人", "你")


def safe_fallback(grounded: Mapping, conversation: Sequence[Mapping] = (), f: Optional[Mapping[str, bool]] = None,
                  must_not_terms: Sequence[str] = ()) -> tuple:
    """The user-visible reply when every attempt failed (Batch 2.1 post-GPU fix: never the raw evidence).

    SAFE_PARTIAL: up to 2 verifiable facts (current / corrected / at_time; never a past or corrected-away value as current) in plain
    second-person words, plus "X 我这边没有记录" for unknown items; NATURAL_UNKNOWN: nothing known; CONFLICT: only unverified (conflicting)
    facts; MINIMAL: anything else, or a candidate that fails the safety checks itself (internal leak, must-not, unsupported specifics)."""
    facts = grounded.get("facts", []) if grounded else []
    unknown = grounded.get("unknown", []) if grounded else []
    usable, seen = [], set()
    for x in facts:
        if x.get("role") not in ("current", "corrected", "at_time"):
            continue
        st = _second_person(x.get("statement", ""))
        key = re.sub(r"专业|的|是|。", "", st)
        if st and not any(key in k or k in key for k in seen) and not LEAK.search(st):     # "你读的是软件工程专业" = "你读的是软件工程"
            seen.add(key)
            usable.append(st)
    usable = usable[:2]
    if usable:
        text = "我记得" + "，还有".join(usable) + "。" + ("不过你问的这个，我这边没有记录。" if unknown else "")
        kind = "SAFE_PARTIAL"
    elif unknown:
        text, kind = "这个我这边没有记录，不太清楚。", "NATURAL_UNKNOWN"
    elif any(x.get("role") == "unverified" for x in facts):
        text, kind = "这件事有两种说法，我不确定哪个是对的。", "CONFLICT"
    else:
        text, kind = MINIMAL_REPLY, "MINIMAL"
    # the fallback passes the same safety checks as a model reply (it must not bypass the guard)
    if LEAK.search(text) or must_not(text, grounded, must_not_terms) or any(not c.startswith(ADVISORY) and not c.startswith("unknown_not_admitted")
                                                                         for c in claims(text, grounded, conversation)):
        return MINIMAL_REPLY, "MINIMAL"
    return text, kind


def _grounded_plain(grounded: Optional[Mapping]) -> str:
    """The realization layer's neutral answer of the grounded content (one definition of the plain answer)."""
    from .realization import neutral
    return neutral(grounded)
