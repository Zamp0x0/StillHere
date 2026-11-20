"""Deterministic, configurable cleaning: WeFlow messages -> dialogue events. Never rewrites what a person typed.

Event kinds:

* ``text``     a typed text message (or the reply part of a 引用消息): bubbles = its non-empty lines, verbatim (stripped of surrounding
               whitespace only). WeChat emoji codes, dialect, punctuation and very short replies are kept as they are;
* ``sticker``  a sticker WITH ``emojiCaption``: the known words only, rendered ``[表情：<caption>]`` (may be a target bubble when
               ``sticker_caption_as_target``); a sticker without a caption is a ``media`` event;
* ``media``    image / video / voice without transcript / call / transfer / link / mini program / location / file / card / chat record /
               other / XML: an invisible event in the context (``[图片]`` ...), never target text;
* system messages and revoke notices are dropped (counted).

A text message whose whole content is a media placeholder (``[图片]``) is not a person's words: it becomes a ``media`` event.
Consecutive identical media events of one person within ``media_collapse_seconds`` collapse into one event with a count.

Privacy (pattern hits from ``stillhere.session.disclosure.PATTERNS`` plus credential shapes): a TARGET with an excluded category is
dropped as a sample (the text is never rewritten); a flagged category is kept and counted; CONTEXT values of excluded categories are
replaced by ``（已隐藏）`` when ``redact_context_privacy``. Ordinary first-person experiences, dates and places are NOT filtered: the
targets are a real person's own words, not model answers that need grounding.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from stillhere.session.disclosure import PATTERNS

from . import CLEANING_VERSION
from .identity import is_system
from .weflow import Message

MEDIA_LABEL = {"图片消息": "[图片]", "视频消息": "[视频]", "语音消息": "[语音]", "通话消息": "[通话]", "转账消息": "[转账]", "链接消息": "[链接]",
               "小程序消息": "[小程序]", "位置消息": "[位置]", "文件消息": "[文件]", "名片消息": "[名片]", "聊天记录": "[聊天记录]", "其他消息": "[其他消息]",
               "动画表情": "[表情]"}
TEXT_TYPES = ("文本消息", "引用消息")
PLACEHOLDER = re.compile(r"^\[(图片|视频|语音|语音消息[^\]]*|语音通话|视频通话|通话|表情包?|动画表情|转账|转账收款|链接|小程序|位置|文件|名片|聊天记录|其他消息)\]$")
CREDENTIAL = re.compile(r"(sk-[A-Za-z0-9]{16,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|(?<![A-Za-z0-9])[A-Fa-f0-9]{32,}(?![A-Za-z0-9])|AKIA[0-9A-Z]{16})")
_ID_FULL = re.compile(r"\d{17}[0-9Xx]")


@dataclass(frozen=True)
class CleaningConfig:
    session_gap_minutes: int = 120
    turn_merge_minutes: int = 5
    media_collapse_seconds: int = 60
    context_turns: int = 12
    max_context_tokens: int = 1536
    min_other_turns_in_context: int = 1
    sticker_caption_as_target: bool = True
    exclude_media_dependent: bool = False
    exclude_target_privacy: Tuple[str, ...] = ("credential", "id_number", "bank_card", "phone", "email")
    flag_target_privacy: Tuple[str, ...] = ("address_detail", "income")
    redact_context_privacy: bool = True
    max_same_target_train: int = 20
    near_dup_threshold: float = 0.85
    near_dup_min_chars: int = 12
    split: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    split_buffer_hours: int = 24
    version: str = CLEANING_VERSION

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()}

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, d: dict) -> "CleaningConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__ and k != "version"}
        bad = [k for k in d if k not in cls.__dataclass_fields__]
        if bad:
            raise ValueError(f"unknown cleaning config keys: {bad}")
        for k in ("exclude_target_privacy", "flag_target_privacy", "split"):
            if k in known:
                known[k] = tuple(known[k])
        c = cls(**known)
        if abs(sum(c.split) - 1.0) > 1e-6 or min(c.split) <= 0:
            raise ValueError("split shares must be positive and sum to 1")
        if c.turn_merge_minutes < 0 or c.session_gap_minutes <= c.turn_merge_minutes:
            raise ValueError("session_gap_minutes must exceed turn_merge_minutes")
        return c


def privacy_hits(text: str) -> List[Tuple[str, str]]:
    """(category, substring) by the disclosure PATTERNS (an 18-digit ID is never also a bank card) plus credential shapes."""
    out = [("credential", m.group(0)) for m in CREDENTIAL.finditer(text or "")]
    ids = [m.span() for m in dict(PATTERNS)["id_number"].finditer(text or "")]
    for cat, pat in PATTERNS:
        for m in pat.finditer(text or ""):
            if cat == "bank_card" and (_ID_FULL.fullmatch(m.group(0)) or any(m.start() < e and s < m.end() for s, e in ids)):
                continue
            out.append((cat, m.group(0)))
    return out


def redact(text: str, cats: Sequence[str]) -> str:
    for cat, s in sorted(privacy_hits(text), key=lambda x: -len(x[1])):
        if cat in cats:
            text = text.replace(s, "（已隐藏）")
    return text


@dataclass
class Event:
    kind: str                      # text | sticker | media
    role: str                      # target | other
    bubbles: List[str]
    message_ids: List[str]
    time: int
    label: str = ""                # media label
    count: int = 1                 # collapsed identical media
    quote_of: Optional[str] = None  # platform id of the quoted message
    flags: List[str] = field(default_factory=list)

    def render(self) -> str:
        if self.kind == "media":
            return self.label + (f"×{self.count}" if self.count > 1 else "")
        return " / ".join(self.bubbles)


def events(messages: Sequence[Message], target_username: str, cfg: CleaningConfig) -> Tuple[List[Event], Dict[str, int]]:
    """Messages (sorted) -> events; ``role`` is ``target`` for the chosen username, ``other`` for the other person."""
    stats: Dict[str, int] = {}

    def bump(k, n=1):
        stats[k] = stats.get(k, 0) + n
    out: List[Event] = []
    for m in messages:
        if is_system(m):
            bump("dropped_system")
            continue
        role = "target" if m.sender == target_username else "other"
        content = m.content.strip()
        if m.type in TEXT_TYPES:
            if not content:
                bump("dropped_empty_text")
                continue
            if PLACEHOLDER.match(content):
                bump("placeholder_text_as_media")
                out.append(Event("media", role, [], [m.platform_id], m.create_time, label=content))
                continue
            bubbles = [ln.strip() for ln in content.splitlines() if ln.strip()]
            e = Event("text", role, bubbles, [m.platform_id], m.create_time, quote_of=m.reply_to if m.type == "引用消息" else None)
            out.append(e)
            bump("text_events")
            if m.type == "引用消息":
                bump("quote_events")
            continue
        if m.type == "动画表情" and m.emoji_caption:
            out.append(Event("sticker", role, [f"[表情：{m.emoji_caption.strip()}]"], [m.platform_id], m.create_time))
            bump("sticker_with_caption")
            continue
        label = MEDIA_LABEL.get(m.type, "[其他消息]")
        prev = out[-1] if out else None
        if prev and prev.kind == "media" and prev.role == role and prev.label == label and m.create_time - prev.time <= cfg.media_collapse_seconds:
            prev.count += 1
            prev.message_ids.append(m.platform_id)
            prev.time = m.create_time
            bump("media_collapsed")
            continue
        out.append(Event("media", role, [], [m.platform_id], m.create_time, label=label))
        bump("media_events")
    return out, stats
