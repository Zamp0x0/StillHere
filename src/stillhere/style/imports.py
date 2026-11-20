"""Authorised chat import into the existing ``CanonicalMessage`` contract, plus the two pipelines that read it.

Parsers (no network, no decryption, no account access; files the user exported themselves):

* ``stillhere_jsonl``: the native intermediate format, one JSON object per line (also the target for any other exporter);
* ``telegram_json``: Telegram Desktop "Export chat history" JSON (``messages`` with ``type``, ``from``, ``from_id``, ``text``,
  ``reply_to_message_id``, ``forwarded_from``);
* ``whatsapp_txt``: WhatsApp "Export chat" text (``dd/mm/yy, HH:MM - Name: text``; ``<Media omitted>``);
* ``wechat_jsonl``: an intermediate JSONL that third-party WeChat exporters can be mapped to (``talker``, ``sender``, ``is_self``,
  ``type`` 1 text / 3 image / 34 voice / 43 video / 47 sticker / 49 link or quote / 10000 system, ``content``, ``create_time``). Nothing
  here decrypts a WeChat database or asks for keys.

``ImportedMessage`` keeps what the canonical message has no field for (platform, type, recipients, reply, forward, bot, scope).

Two pipelines with different rights:

* **memory** may read the whole authorised conversation (``authorization.scope`` contains ``memory``);
* **style** reads only the target person's own typed text: never other speakers, forwards, quotes, bot output, system lines,
  media placeholders or voice transcripts (unless the scope allows them), and only when the scope contains ``style``.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from stillhere.contracts import CanonicalMessage

TEXT, MEDIA, SYSTEM, VOICE_TRANSCRIPT, LINK = "text", "media", "system", "voice_transcript", "link"


@dataclass(frozen=True)
class Authorization:
    """What the owner of the export allowed. ``style_speakers``: whose own words may form a style profile."""

    target_persona_id: str
    scope: tuple                              # subset of ("memory", "style")
    style_speakers: tuple                     # speaker ids
    consent_reference: str
    public: bool = False                      # an import is private unless it is synthetic test data
    allow_voice_transcripts_for_style: bool = False


@dataclass(frozen=True)
class ImportedMessage:
    platform: str
    source_file_id: str
    conversation_id: str
    message_id: str
    speaker_id: str
    speaker_name: str
    recipient_ids: tuple
    target_persona_id: str
    timestamp: datetime
    timezone: str
    message_type: str
    content: str
    reply_to: Optional[str] = None
    forwarded_from: Optional[str] = None
    is_bot: bool = False
    is_group: bool = False
    content_hash: str = ""

    def canonical(self, source_id: str) -> CanonicalMessage:
        loc = f"platform={self.platform};type={self.message_type}" + (f";reply_to={self.reply_to}" if self.reply_to else "") + (";forwarded=1" if self.forwarded_from else "")
        return CanonicalMessage(conversation_id=canonical_id(self.conversation_id), message_id=canonical_id(self.message_id), target_persona_id=self.target_persona_id,
                                speaker_id=canonical_id(self.speaker_id),
                                timestamp=self.timestamp, content=self.content, source_type=f"import_{self.platform}", source_id=source_id, locator=loc, recorded_at=self.timestamp)


_ID_BAD = re.compile(r"[^A-Za-z0-9._:-]")


def canonical_id(raw: str) -> str:
    """Platform ids (``123@chatroom``, ``unknown:阿明``) mapped onto the canonical id alphabet; a lossy mapping gets a hash suffix so two raw
    ids never collide."""
    s = str(raw)
    clean = _ID_BAD.sub("-", s).lstrip("._:-")[:100]
    if clean == s:
        return s
    return f"{clean or 'id'}.h{hashlib.sha256(s.encode('utf-8')).hexdigest()[:12]}"


def _hash(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def _mk(platform, file_id, conv, mid, speaker, name, recipients, target, ts, tz, kind, content, **kw) -> ImportedMessage:
    return ImportedMessage(platform, file_id, conv, f"{platform}.{file_id}.{mid}", speaker, name, tuple(recipients), target, ts, tz, kind, content,
                           content_hash=_hash(speaker, ts.isoformat(), content), **kw)


def parse_stillhere_jsonl(text: str, *, file_id: str, target: str) -> List[ImportedMessage]:
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ts = datetime.fromisoformat(r["timestamp"])
        out.append(_mk("stillhere", file_id, r["conversation_id"], r["message_id"], r["speaker_id"], r.get("speaker_name", r["speaker_id"]), r.get("recipient_ids", []), target, ts,
                       ts.utcoffset() and str(ts.tzinfo) or "UTC", r.get("message_type", TEXT), r["content"], reply_to=r.get("reply_to"), forwarded_from=r.get("forwarded_from"),
                       is_bot=bool(r.get("is_bot")), is_group=bool(r.get("is_group"))))
    return out


def _telegram_text(t) -> str:
    if isinstance(t, str):
        return t
    return "".join(x if isinstance(x, str) else x.get("text", "") for x in t)


def parse_telegram_json(text: str, *, file_id: str, target: str, tz: timezone) -> List[ImportedMessage]:
    data = json.loads(text)
    conv = f"telegram.{data.get('id', file_id)}"
    group = data.get("type") not in ("personal_chat", "bot_chat", None)
    speakers = sorted({m.get("from_id") for m in data["messages"] if m.get("from_id")})
    out = []
    for m in data["messages"]:
        ts = datetime.fromisoformat(m["date"]).replace(tzinfo=tz)
        if m.get("type") == "service":
            kind, content, speaker = SYSTEM, m.get("action", "service"), "system"
        else:
            content = _telegram_text(m.get("text", ""))
            kind = TEXT if content and not m.get("media_type") and not m.get("photo") else MEDIA
            speaker = m.get("from_id") or "unknown"
            content = content or f"<{m.get('media_type', 'media')}>"
        out.append(_mk("telegram", file_id, conv, m["id"], speaker, m.get("from") or speaker, [s for s in speakers if s != speaker], target, ts, str(tz), kind, content,
                       reply_to=str(m["reply_to_message_id"]) if m.get("reply_to_message_id") else None, forwarded_from=m.get("forwarded_from"),
                       is_bot=bool(m.get("via_bot")), is_group=group))
    return out


_WA = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{2,4}), (\d{1,2}):(\d{2}) - ([^:]+): (.*)$")


def parse_whatsapp_txt(text: str, *, file_id: str, target: str, names: Mapping[str, str], tz: timezone) -> List[ImportedMessage]:
    """``names``: display name -> speaker id, given by the importing user (the export has names only). Continuation lines join the message."""
    out, n = [], 0
    for line in text.splitlines():
        m = _WA.match(line)
        if not m:
            if out and line.strip():
                last = out[-1]
                out[-1] = ImportedMessage(**{**last.__dict__, "content": last.content + "\n" + line.strip(), "content_hash": _hash(last.speaker_id, last.timestamp.isoformat(), last.content + "\n" + line.strip())})
            continue
        d, mo, y, hh, mm, name, body = m.groups()
        year = int(y) + 2000 if len(y) == 2 else int(y)
        ts = datetime(year, int(mo), int(d), int(hh), int(mm), tzinfo=tz)
        speaker = names.get(name.strip(), f"unknown:{name.strip()}")
        kind = MEDIA if body.strip() in ("<Media omitted>", "<附件：图片>", "<媒体文件已省略>") else TEXT
        n += 1
        out.append(_mk("whatsapp", file_id, f"whatsapp.{file_id}", n, speaker, name.strip(), [v for v in names.values() if v != speaker], target, ts, str(tz), kind, body.strip()))
    return out


WECHAT_TYPES = {1: TEXT, 3: MEDIA, 34: VOICE_TRANSCRIPT, 43: MEDIA, 47: MEDIA, 49: LINK, 10000: SYSTEM, 10002: SYSTEM}


def parse_wechat_jsonl(text: str, *, file_id: str, target: str, self_id: str, tz: timezone) -> List[ImportedMessage]:
    out = []
    for i, line in enumerate(l for l in text.splitlines() if l.strip()):
        r = json.loads(line)
        kind = WECHAT_TYPES.get(int(r["type"]), MEDIA)
        speaker = self_id if r.get("is_self") else r.get("sender") or r["talker"]
        ts = datetime.fromtimestamp(int(r["create_time"]), tz)
        group = str(r["talker"]).endswith("@chatroom")
        quote = r.get("quote")
        out.append(_mk("wechat", file_id, f"wechat.{r['talker']}", r.get("msg_id", i), speaker, r.get("sender_name", speaker), [r["talker"]] if r.get("is_self") else [self_id], target, ts, str(tz), kind,
                       r.get("content", ""), reply_to=quote.get("msg_id") if isinstance(quote, dict) else None, forwarded_from=r.get("forwarded_from"), is_group=group))
    return out


def parse_csv(text: str, *, file_id: str, target: str, tz: timezone) -> List[ImportedMessage]:
    """Columns: conversation_id, message_id, speaker_id, timestamp (ISO), content[, message_type]."""
    out = []
    for r in csv.DictReader(io.StringIO(text)):
        ts = datetime.fromisoformat(r["timestamp"])
        ts = ts if ts.tzinfo else ts.replace(tzinfo=tz)
        out.append(_mk("csv", file_id, r["conversation_id"], r["message_id"], r["speaker_id"], r.get("speaker_name") or r["speaker_id"], [], target, ts, str(tz), r.get("message_type") or TEXT, r["content"]))
    return out


# ---------------------------------------------------------------- the two pipelines

class NotAuthorized(PermissionError):
    pass


def memory_messages(msgs: Sequence[ImportedMessage], auth: Authorization) -> List[ImportedMessage]:
    if "memory" not in auth.scope:
        raise NotAuthorized("this import is not authorised for memory")
    return [m for m in msgs if m.message_type != SYSTEM]


def style_samples(msgs: Sequence[ImportedMessage], auth: Authorization, speaker_id: str) -> List[ImportedMessage]:
    """Only the person's own typed words. Everything excluded is excluded for a reason a test checks."""
    if "style" not in auth.scope:
        raise NotAuthorized("this import is not authorised for style")
    if speaker_id not in auth.style_speakers:
        raise NotAuthorized(f"{speaker_id} has not authorised use of their words for style")
    ok_types = {TEXT} | ({VOICE_TRANSCRIPT} if auth.allow_voice_transcripts_for_style else set())
    return [m for m in msgs if m.speaker_id == speaker_id and m.message_type in ok_types and not m.forwarded_from and not m.is_bot and m.content.strip()
            and not m.content.startswith(("「", "“引用", "> "))]


def to_bubbles(msgs: Sequence[ImportedMessage], *, gap_seconds: int = 120) -> List[List[str]]:
    """Consecutive messages of one speaker within ``gap_seconds`` form one multi-bubble turn; a line break inside a message is a bubble too."""
    turns: List[List[str]] = []
    last: Optional[ImportedMessage] = None
    for m in sorted(msgs, key=lambda m: m.timestamp):
        parts = [p.strip() for p in m.content.split("\n") if p.strip()]
        if last and m.speaker_id == last.speaker_id and m.conversation_id == last.conversation_id and (m.timestamp - last.timestamp).total_seconds() <= gap_seconds:
            turns[-1].extend(parts)
        else:
            turns.append(parts)
        last = m
    return [t for t in turns if t]
