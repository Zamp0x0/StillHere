"""WeFlow JSON export reader (``weflow`` / ``session`` / ``messages`` / ``avatars``). One-to-one chats only.

The file is opened read-only and never changed. Rules:

* required message fields: ``localId createTime type localType content isSend senderUsername platformMessageId``; unknown fields are
  tolerated (counted, never fatal);
* ``createTime`` is a Unix time; WeFlow's ``formattedTime`` is local China time, so every time here is interpreted as +08:00;
* a repeated ``localId`` / ``platformMessageId`` whose record is identical (sender, time, type, content) is a duplicate import: dropped and
  counted; the same id with a DIFFERENT record is an error;
* messages are ordered by (createTime, localId); input that was out of order is reported, not rejected;
* a group chat is refused (``UNSUPPORTED_SESSION``): its members are not "the two people" of a private chat.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import PARSER_VERSION

TZ = timezone(timedelta(hours=8))
REQUIRED = ("localId", "createTime", "type", "localType", "content", "isSend", "senderUsername", "platformMessageId")
KNOWN = set(REQUIRED) | {"formattedTime", "senderDisplayName", "source", "senderAvatarKey", "emojiMd5", "emojiCdnUrl", "emojiCaption",
                         "replyToMessageId", "quotedContent", "quotedSender", "quotedType", "appMsgType", "appMsgKind", "appMsgDesc",
                         "appMsgAppName", "appMsgThumbUrl", "appMsgSourceName", "appMsgSourceUsername", "linkTitle", "linkUrl", "linkThumb",
                         "locationLat", "locationLng", "locationPoiname", "locationLabel"}
PRIVATE_SESSION_TYPES = ("私聊", "private", "friend")
MIN_TIME = 946684800                      # 2000-01-01


class WeFlowError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class Message:
    local_id: int
    platform_id: str
    create_time: int
    type: str
    local_type: int
    content: str
    is_send: int
    sender: str
    sender_display: str
    reply_to: Optional[str] = None
    quoted_type: Optional[str] = None
    emoji_caption: Optional[str] = None
    formatted_time: Optional[str] = None

    @property
    def time(self) -> datetime:
        return datetime.fromtimestamp(self.create_time, TZ)


@dataclass
class Export:
    path: str
    sha256: str
    size: int
    weflow_version: str
    generator: str
    session: Dict[str, object]
    avatars: Tuple[str, ...]
    messages: List[Message]
    report: Dict[str, object] = field(default_factory=dict)

    def by_platform_id(self) -> Dict[str, Message]:
        return {m.platform_id: m for m in self.messages}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def detect(obj) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("weflow"), dict) and isinstance(obj.get("session"), dict) and isinstance(obj.get("messages"), list)


def _record(m: dict) -> tuple:
    return (str(m.get("senderUsername")), m.get("createTime"), m.get("type"), m.get("content"))


def parse(obj: dict, *, path: str = "<memory>", sha256: str = "", size: int = 0) -> Export:
    if not detect(obj):
        raise WeFlowError("FORMAT", "not a WeFlow JSON export (needs 'weflow', 'session' and 'messages')")
    sess = obj["session"]
    stype = str(sess.get("type") or "")
    if stype not in PRIVATE_SESSION_TYPES:
        raise WeFlowError("UNSUPPORTED_SESSION", f"session type {stype!r} is not a one-to-one chat (group chats are not supported in P1)")
    if not sess.get("wxid"):
        raise WeFlowError("MISSING_FIELD", "session.wxid is missing")
    raw = obj["messages"]
    if not raw:
        raise WeFlowError("EMPTY", "the export has no messages")
    unknown = Counter()
    by_local: Dict[object, dict] = {}
    by_pid: Dict[str, dict] = {}
    kept: List[dict] = []
    dup_import = 0
    out_of_order = 0
    prev_t = None
    now = datetime.now(TZ).timestamp() + 86400
    for i, m in enumerate(raw):
        if not isinstance(m, dict):
            raise WeFlowError("FORMAT", f"messages[{i}] is not an object")
        missing = [k for k in REQUIRED if k not in m]
        if missing:
            raise WeFlowError("MISSING_FIELD", f"messages[{i}] lacks {missing}")
        t = m["createTime"]
        if not isinstance(t, (int, float)) or isinstance(t, bool) or not (MIN_TIME <= t <= now):
            raise WeFlowError("BAD_TIME", f"messages[{i}] has an invalid createTime")
        if m["isSend"] not in (0, 1):
            raise WeFlowError("BAD_FIELD", f"messages[{i}] isSend must be 0 or 1")
        for k in m:
            if k not in KNOWN:
                unknown[k] += 1
        if prev_t is not None and t < prev_t:
            out_of_order += 1
        prev_t = t
        lid, pid = m["localId"], str(m["platformMessageId"])
        seen = by_local.get(lid) or by_pid.get(pid)
        if seen is not None:
            if _record(seen) == _record(m):
                dup_import += 1
                continue
            raise WeFlowError("DUPLICATE_ID", f"messages[{i}] reuses an id of a different message")
        by_local[lid] = m
        by_pid[pid] = m
        kept.append(m)
    msgs = [Message(local_id=int(m["localId"]), platform_id=str(m["platformMessageId"]), create_time=int(m["createTime"]), type=str(m["type"]),
                    local_type=int(m["localType"]), content=str(m.get("content") or ""), is_send=int(m["isSend"]), sender=str(m["senderUsername"]),
                    sender_display=str(m.get("senderDisplayName") or ""), reply_to=str(m["replyToMessageId"]) if m.get("replyToMessageId") not in (None, "") else None,
                    quoted_type=str(m["quotedType"]) if m.get("quotedType") is not None else None, emoji_caption=m.get("emojiCaption") or None,
                    formatted_time=m.get("formattedTime")) for m in kept]
    msgs.sort(key=lambda x: (x.create_time, x.local_id))
    fmt_bad = sum(1 for x in msgs if x.formatted_time and x.time.strftime("%Y-%m-%d %H:%M:%S") != x.formatted_time)
    report = {"parser_version": PARSER_VERSION, "messages_in_file": len(raw), "messages": len(msgs), "duplicate_imports_dropped": dup_import,
              "out_of_order_in_file": out_of_order, "formattedTime_mismatch": fmt_bad, "unknown_fields": dict(unknown),
              "types": dict(Counter(x.type for x in msgs)), "first": msgs[0].time.isoformat(), "last": msgs[-1].time.isoformat(),
              "session_declared_count": sess.get("messageCount"), "timezone": "+08:00"}
    return Export(path=path, sha256=sha256, size=size, weflow_version=str(obj["weflow"].get("version", "")), generator=str(obj["weflow"].get("generator", "")),
                  session={k: sess.get(k) for k in ("wxid", "nickname", "remark", "displayName", "type")}, avatars=tuple(sorted((obj.get("avatars") or {}).keys())),
                  messages=msgs, report=report)


def load(path) -> Export:
    p = Path(path)
    if not p.is_file():
        raise WeFlowError("NOT_FOUND", f"input file not found: {p}")
    data = p.read_bytes()                     # read-only: the file is never opened for writing
    try:
        obj = json.loads(data.decode("utf-8-sig"))
    except UnicodeDecodeError as e:
        raise WeFlowError("ENCODING", f"not UTF-8: {e}") from None
    except json.JSONDecodeError as e:
        raise WeFlowError("FORMAT", f"invalid JSON: {e}") from None
    return parse(obj, path=str(p), sha256=hashlib.sha256(data).hexdigest(), size=len(data))
