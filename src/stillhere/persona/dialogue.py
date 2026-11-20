"""Dialogue reconstruction: events -> sessions -> turns -> (context, target) samples.

* session: a gap > ``session_gap_minutes`` between consecutive events starts a new session; nothing crosses a session;
* turn: consecutive events of one person in one session, each within ``turn_merge_minutes`` of the previous one;
* sample: a TARGET turn with at least one text / caption bubble, preceded in its session by at least ``min_other_turns_in_context`` turns
  of the other person. Context = the previous turns (latest ``context_turns``, the oldest whole turns dropped first until the context fits
  ``max_context_tokens``), BOTH people, never the target or anything after it. The target is never truncated.
* the target text is only the target person's own new words: text bubbles and caption stickers; media of the target turn and a quoted
  message are context information (a quote is rendered in the context as ``（我引用了：…）`` / ``（对方引用了：…）``).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import DIALOGUE_VERSION
from .cleaning import CleaningConfig, Event, privacy_hits, redact

LABEL = {"target": "我", "other": "对方"}


@dataclass
class Turn:
    role: str
    events: List[Event]
    session_id: str

    @property
    def start(self) -> int:
        return self.events[0].time

    @property
    def end(self) -> int:
        return self.events[-1].time

    @property
    def message_ids(self) -> List[str]:
        return [i for e in self.events for i in e.message_ids]

    def target_bubbles(self, sticker_ok: bool) -> List[str]:
        return [b for e in self.events if e.kind == "text" or (e.kind == "sticker" and sticker_ok) for b in e.bubbles]

    def media_only(self) -> bool:
        return all(e.kind == "media" for e in self.events)


@dataclass
class Sample:
    sample_id: str
    session_id: str
    session_start: int
    time: int
    context_lines: List[str]
    target_bubbles: List[str]
    context_message_ids: List[str]
    target_message_ids: List[str]
    flags: List[str] = field(default_factory=list)
    context_tokens: int = 0

    @property
    def target_text(self) -> str:
        return "\n".join(self.target_bubbles)


def sessions(evs: Sequence[Event], gap_minutes: int) -> List[List[Event]]:
    out: List[List[Event]] = []
    for e in evs:
        if not out or e.time - out[-1][-1].time > gap_minutes * 60:
            out.append([])
        out[-1].append(e)
    return out


def session_id(sess: Sequence[Event]) -> str:
    return "s_" + hashlib.sha256(("|".join(sess[0].message_ids) + "|" + "|".join(sess[-1].message_ids)).encode()).hexdigest()[:12]


def turns(sess: Sequence[Event], merge_minutes: int) -> List[Turn]:
    sid = session_id(sess)
    out: List[Turn] = []
    for e in sess:
        if out and out[-1].role == e.role and e.time - out[-1].end <= merge_minutes * 60:
            out[-1].events.append(e)
        else:
            out.append(Turn(e.role, [e], sid))
    return out


def render_turn(t: Turn, quoted: Callable[[str], Optional[str]], redact_cats: Sequence[str]) -> str:
    parts = []
    for e in t.events:
        txt = e.render()
        if e.quote_of:
            q = quoted(e.quote_of)
            if q:
                txt = f"（{LABEL[t.role]}引用了：{q}）{txt}"
        parts.append(redact(txt, redact_cats) if redact_cats else txt)
    return f"{LABEL[t.role]}：" + " / ".join(parts)


def build(evs: Sequence[Event], cfg: CleaningConfig, *, count: Callable[[str], int],
          quoted_text: Callable[[str], Optional[str]]) -> Tuple[List[Sample], Dict[str, int], dict]:
    stats: Dict[str, int] = {}

    def bump(k, n=1):
        stats[k] = stats.get(k, 0) + n
    redact_cats = cfg.exclude_target_privacy if cfg.redact_context_privacy else ()
    sess = sessions(evs, cfg.session_gap_minutes)
    shape = {"sessions": len(sess), "turns": 0, "target_turns": 0, "other_turns": 0, "merged_turns": 0, "merged_long_gap": 0}
    out: List[Sample] = []
    for s in sess:
        ts = turns(s, cfg.turn_merge_minutes)
        shape["turns"] += len(ts)
        for t in ts:
            shape["target_turns" if t.role == "target" else "other_turns"] += 1
            if len(t.events) > 1:
                shape["merged_turns"] += 1
                if any(b.time - a.time > 120 for a, b in zip(t.events, t.events[1:])):
                    shape["merged_long_gap"] += 1
        for i, t in enumerate(ts):
            if t.role != "target":
                continue
            bubbles = t.target_bubbles(cfg.sticker_caption_as_target)
            if not bubbles:
                bump("excluded_target_media_only" if t.media_only() else "excluded_target_no_text")
                continue
            ctx = ts[max(0, i - cfg.context_turns):i]
            if sum(1 for c in ctx if c.role == "other") < cfg.min_other_turns_in_context:
                bump("excluded_no_context")
                continue
            hits = {c for b in bubbles for c, _ in privacy_hits(b)}
            if hits & set(cfg.exclude_target_privacy):
                bump("excluded_target_privacy")
                continue
            flags = sorted(f"privacy:{c}" for c in hits & set(cfg.flag_target_privacy))
            if any(e.kind == "media" for e in t.events):
                flags.append("target_turn_has_media")
            if ctx and ctx[-1].role == "other" and ctx[-1].media_only():
                flags.append("media_dependent")
                if cfg.exclude_media_dependent:
                    bump("excluded_media_dependent")
                    continue
            if any(e.quote_of for e in t.events):
                flags.append("target_quotes")
            lines = [render_turn(c, quoted_text, redact_cats) for c in ctx]
            ctx_ids = {i for c in ctx for i in c.message_ids}
            for e in t.events:                              # the target quotes a message the window does not show: show WHAT it quotes
                if e.quote_of and e.quote_of not in ctx_ids:
                    qt = quoted_text(e.quote_of)
                    if qt:
                        lines.append(f"（我接下来引用了：{redact(qt, redact_cats) if redact_cats else qt}）")
                        flags.append("target_quote_shown_in_context")
            n = count("\n".join(lines))
            while len(lines) > 1 and n > cfg.max_context_tokens:
                lines.pop(0)
                ctx = ctx[1:]
                n = count("\n".join(lines))
            if n > cfg.max_context_tokens or not any(c.role == "other" for c in ctx):
                bump("excluded_context_too_long")
                continue
            tids = t.message_ids
            out.append(Sample(sample_id="ps_" + hashlib.sha256("|".join(tids).encode()).hexdigest()[:16], session_id=t.session_id,
                              session_start=s[0].time, time=t.start, context_lines=lines, target_bubbles=bubbles,
                              context_message_ids=[i for c in ctx for i in c.message_ids], target_message_ids=tids, flags=flags, context_tokens=n))
    shape["version"] = DIALOGUE_VERSION
    return out, stats, shape
