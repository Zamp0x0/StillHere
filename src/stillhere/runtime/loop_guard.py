"""Loop Guard v1 (``stillhere.runtime.loop_guard/1.0``): a deterministic post-generation check for degenerate phrase loops.

Why: the D4 adapter (checkpoint-546) repeats itself on real chat (6.39 % of 1,470 frozen real-chat DEV replies, official evaluator), mostly
clause loops that run to ``max_new_tokens``. This is an OUTPUT-LAYER mitigation: it never changes the model, it cannot teach EOS, and every
number it produces is reported as post-processing, never as a model result.

What it removes (only with HIGH confidence, only consecutive copies, always keeping the first copy):

* clause loops: a block of 1-4 clauses (split at ， 。 ！ ？ ； , . ! ? ; and newlines) whose normalised text (whitespace and end
  punctuation removed) repeats >= 3 times in a row and carries >= 4 content characters -> the extra copies go; text after the loop stays;
  a trailing partial copy (the generation stopped mid-copy) goes too;
* span loops inside a clause: a span of >= 4 characters repeated >= 3 times in a row (the official LOOP shape) -> one copy, unless the span
  is pure expressive repetition ("哈哈");
* short-unit runs: a 1-3 character unit repeated >= 8 times in a row -> 1 copy; EXPRESSIVE units (哈 呵 嘿 嘻 啊 哦 好 不 对 是 嗯 啦 呀 真的 …)
  are left alone unless the run is a degenerate tail of >= 20 characters, which is shortened to 5 units.

What it never does: delete words that merely occur several times ("chrome … chrome … chrome"), rewrite or reorder text, touch
non-consecutive similar sentences, lists, dates, names or numbers outside a removed duplicate copy (a removed copy is, by construction,
identical to the kept one). Escalating fragments ("A，AB，ABC") are DETECTED (LOW) and left as they are. Disabled -> the text is returned
unchanged with ``loop_guard_applied`` False.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import asdict, dataclass
from typing import List, Mapping, Optional, Tuple

VERSION = "stillhere.runtime.loop_guard/1.0"
FLAG = "LOOP_GUARD_ENABLED"
CLAUSE = re.compile(r"[^，。！？；,.!?;\n]+[，。！？；,.!?;\n]*|[，。！？；,.!?;\n]+")
END_PUNCT = "，。！？；,.!?;、…~～ \t\r\n"
SPAN_LOOP = re.compile(r"(.{4,}?)\1{2,}", re.S)
SHORT_RUN = re.compile(r"(.{1,3}?)\1{7,}", re.S)
EXPRESSIVE_RUN = re.compile(r"(.{1,3}?)\1{5,}", re.S)
EXPRESSIVE_CHARS = set("哈呵嘿嘻啊哦噢喔好不对是嗯啦呀嘛呢哇诶唉吼6hH!！?？.。~～")
EXPRESSIVE_UNITS = {"真的", "好的", "对对", "是是", "不不", "哈哈", "嘿嘿", "嘻嘻", "呵呵", "啊啊", "好好"}


@dataclass
class LoopResult:
    raw_response: str
    final_response: str
    loop_detected: bool
    loop_guard_applied: bool
    repeat_pattern: Optional[str]
    repeat_count: int
    removed_char_count: int
    confidence: str            # HIGH | LOW | NONE
    reason: str
    status: str                # CLEAN | TRIMMED | DETECTED_NOT_TRIMMED | DISABLED
    version: str = VERSION

    def trace(self) -> dict:
        """What a run trace may keep: statistics and a short hash of the pattern, never the texts."""
        d = asdict(self)
        d.pop("raw_response")
        d.pop("final_response")
        d["repeat_pattern"] = hashlib.sha256(self.repeat_pattern.encode("utf-8")).hexdigest()[:12] if self.repeat_pattern else None
        d["pattern_chars"] = len(self.repeat_pattern or "")
        return d


def enabled(env: Optional[Mapping[str, str]] = None, default: bool = False) -> bool:
    e = env if env is not None else os.environ
    v = e.get(FLAG)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s).strip(END_PUNCT)


def _content_len(s: str) -> int:
    return sum(1 for ch in _norm(s) if ch not in END_PUNCT)


def _has_words(unit: str) -> bool:
    """Letters, digits or CJK. Runs of spaces, dashes or other symbols (layout, ASCII art) are never loops."""
    return any(ch.isalnum() for ch in unit)


def _expressive(unit: str) -> bool:
    u = _norm(unit) or unit
    return bool(u) and (all(ch in EXPRESSIVE_CHARS for ch in u) or u in EXPRESSIVE_UNITS)


def _clause_loop(text: str) -> Optional[Tuple[str, str, int, int]]:
    """(new_text, pattern, copies, removed) for the earliest clause-block loop, else None."""
    units = CLAUSE.findall(text)
    if "".join(units) != text:                       # segmentation must be lossless, or we do not touch the text
        return None
    norm = [_norm(u) for u in units]
    n = len(units)
    for i in range(n):
        for k in (1, 2, 3, 4):
            block = norm[i:i + k]
            if len(block) < k or not all(block) or sum(len(b) for b in block) < 4:
                continue
            if all(_expressive(units[j]) for j in range(i, i + k)) or not _has_words("".join(block)):
                continue
            copies, j = 1, i + k
            while norm[j:j + k] == block:
                copies += 1
                j += k
            if copies >= 3:
                # a block made of a shorter repeated clause ("在的，在的，") is that clause repeated: keep ONE copy of the shortest period
                p = next(q for q in range(1, k + 1) if k % q == 0 and block == block[:q] * (k // q))
                tail = units[j:]
                tail_norm = "".join(norm[j:])
                block_text = "".join(block)
                # a generation cut mid-copy leaves a prefix of the block: it is part of the loop
                if tail and tail_norm and block_text.startswith(tail_norm):
                    tail = []
                kept = "".join(units[:i + p]) + "".join(tail)
                if not tail:
                    kept = kept.rstrip(" \t\r\n")
                return kept, "".join(units[i:i + p]), copies * (k // p), len(text) - len(kept)
    return None


def _span_loop(text: str) -> Optional[Tuple[str, str, int, int]]:
    for m in SPAN_LOOP.finditer(text):
        span = m.group(1)
        if _expressive(span) or len(set(_norm(span))) < 2 or not _has_words(span):
            continue
        copies = len(m.group(0)) // len(span)
        kept = text[:m.start()] + span + text[m.end():]
        return kept, span, copies, len(text) - len(kept)
    return None


def _short_runs(text: str) -> Optional[Tuple[str, str, int, int]]:
    for m in SHORT_RUN.finditer(text):
        unit = m.group(1)
        if _expressive(unit) or not _has_words(unit):
            continue
        copies = len(m.group(0)) // len(unit)
        kept = text[:m.start()] + unit + text[m.end():]
        return kept, unit, copies, len(text) - len(kept)
    for m in EXPRESSIVE_RUN.finditer(text):         # expressive: only a degenerate tail of >= 20 characters, shortened to 5 units
        unit = m.group(1)
        if _expressive(unit) and len(m.group(0)) >= 20:
            copies = len(m.group(0)) // len(unit)
            kept = text[:m.start()] + unit * 5 + text[m.end():]
            return kept, unit, copies, len(text) - len(kept)
    return None


def _escalating(text: str) -> bool:
    """A, AB, ABC: each clause extends the previous one (>= 3 steps). Detected only."""
    norm = [x for x in (_norm(u) for u in CLAUSE.findall(text)) if x]
    run = 1
    for a, b in zip(norm, norm[1:]):
        run = run + 1 if (len(a) >= 2 and b.startswith(a) and len(b) > len(a)) else 1
        if run >= 3:
            return True
    return False


def guard(text: Optional[str], *, enabled_: bool = True, hit_max: Optional[bool] = None, max_passes: int = 6) -> LoopResult:
    raw = text or ""
    if not enabled_:
        return LoopResult(raw, raw, False, False, None, 0, 0, "NONE", "disabled", "DISABLED")
    cur, first_pattern, total_copies, steps = raw, None, 0, []
    for _ in range(max_passes):                      # several loops in one reply are removed one by one, earliest first
        hit = _clause_loop(cur) or _span_loop(cur) or _short_runs(cur)
        if hit is None:
            break
        new, pattern, copies, _removed = hit
        if new == cur:
            break
        steps.append(pattern)
        first_pattern = first_pattern or pattern
        total_copies = max(total_copies, copies)
        cur = new
    removed = len(raw) - len(cur)
    if steps:
        if not cur.strip():                          # never return an empty reply in place of a non-empty one
            return LoopResult(raw, raw, True, False, first_pattern, total_copies, 0, "LOW", "trim would empty the reply", "DETECTED_NOT_TRIMMED")
        cur = cur.rstrip("，,；;、 \t\r\n") or cur              # a dangling clause separator after the cut (never adds text)
        removed = len(raw) - len(cur)
        reason = f"{len(steps)} consecutive loop(s) collapsed to their first copy" + (" (reply hit max_new_tokens)" if hit_max else "")
        return LoopResult(raw, cur, True, True, first_pattern, total_copies, removed, "HIGH", reason, "TRIMMED")
    if _escalating(raw):
        return LoopResult(raw, raw, True, False, None, 0, 0, "LOW", "escalating fragments: detected, not trimmed", "DETECTED_NOT_TRIMMED")
    return LoopResult(raw, raw, False, False, None, 0, 0, "NONE", "no consecutive loop", "CLEAN")


def wrap_generate(generate, *, enabled_: bool = True, records: Optional[List[dict]] = None):
    """A ``generate(messages, config)`` that applies the guard to the model text BEFORE every downstream check (safety guard, grounding,
    must-not, disclosure) sees it. The raw text and the guard trace go to ``records`` (one per call: the guard retries see their own)."""
    def _gen(messages, config):
        out = dict(generate(messages, config))
        res = guard(out.get("text", ""), enabled_=enabled_, hit_max=out.get("hit_token_limit"))
        if records is not None:
            records.append(res.trace())
        out["text"] = res.final_response
        out["loop_guard"] = res.trace()
        return out
    return _gen
