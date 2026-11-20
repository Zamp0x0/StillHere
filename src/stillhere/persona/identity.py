"""Speaker identity resolution for a one-to-one WeFlow export. ``isSend`` is a direction relative to the exporting account, never an id.

Rules (each one tested):

1. System messages (``type`` 系统消息 or ``localType`` 10000 / 10002) belong to ``SYSTEM``, whatever ``senderUsername`` / ``isSend`` say
   (real exports attribute revoke notices to the exporting account with ``isSend = 0``);
2. the people are the distinct ``senderUsername`` of the non-system messages; exactly two are required;
3. the contact is ``session.wxid``; the exporting account ("self") is the other one;
4. ``isSend`` is a cross-check only: non-system ``isSend = 0`` must come from the contact and ``isSend = 1`` from self. A conflict is an
   ``IS_SEND_CONFLICT`` anomaly (blocking unless accepted);
5. display names are for display only (contact: session displayName -> remark -> nickname -> its senderDisplayName; self: its most common
   senderDisplayName). Several names for one username, or one name for both people, are reported (``MULTI_DISPLAY_NAME`` /
   ``SAME_DISPLAY_NAME``);
6. quotes are linked by ``replyToMessageId`` -> ``platformMessageId`` (``quotedSender`` is not used: it is not the display name).
"""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import IDENTITY_VERSION
from .weflow import Export, Message

SYSTEM = "SYSTEM"
SYSTEM_TYPES = ("系统消息",)
SYSTEM_LOCAL_TYPES = (10000, 10002)
BLOCKING = ("IS_SEND_CONFLICT", "SENDER_NOT_IN_AVATARS")


class IdentityError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def is_system(m: Message) -> bool:
    return m.type in SYSTEM_TYPES or m.local_type in SYSTEM_LOCAL_TYPES


def username_hash(username: str) -> str:
    return "u_" + hashlib.sha256(("stillhere.persona.speaker:" + username).encode("utf-8")).hexdigest()[:12]


@dataclass
class Speaker:
    username: str
    role: str                         # "contact" | "self"
    display_name: str
    display_source: str
    messages: int
    text_messages: int
    first: str
    last: str
    is_send: int

    @property
    def id(self) -> str:
        return username_hash(self.username)


@dataclass
class Identity:
    speakers: List[Speaker]
    system_messages: int
    system_by_attribution: Dict[str, int]
    anomalies: List[dict] = field(default_factory=list)
    quotes: Dict[str, int] = field(default_factory=dict)
    version: str = IDENTITY_VERSION

    def by_role(self, role: str) -> Speaker:
        return next(s for s in self.speakers if s.role == role)

    def find(self, key: str) -> Speaker:
        """``key``: 1 / 2 (list order), ``contact`` / ``self``, a speaker id (``u_...``) or a username."""
        k = str(key).strip()
        if k.isdigit() and 1 <= int(k) <= len(self.speakers):
            return self.speakers[int(k) - 1]
        for s in self.speakers:
            if k in (s.role, s.id, s.username):
                return s
        raise IdentityError("NO_SUCH_SPEAKER", f"no speaker matches {k!r}")

    @property
    def blocking(self) -> List[dict]:
        return [a for a in self.anomalies if a["code"] in BLOCKING]


def resolve(exp: Export) -> Identity:
    msgs = exp.messages
    people = [m for m in msgs if not is_system(m)]
    senders = sorted({m.sender for m in people})
    if len(senders) != 2:
        raise IdentityError("PARTICIPANTS", f"{len(senders)} non-system senders found; a one-to-one chat needs exactly 2")
    contact = str(exp.session["wxid"])
    if contact not in senders:
        raise IdentityError("CONTACT_NOT_FOUND", "session.wxid is not among the message senders")
    me = next(s for s in senders if s != contact)
    anomalies = []
    conflict = sum(1 for m in people if (m.is_send == 0) != (m.sender == contact))
    if conflict:
        anomalies.append({"code": "IS_SEND_CONFLICT", "count": conflict, "detail": "non-system messages whose isSend disagrees with the sender"})
    if exp.avatars and not set(senders) <= set(exp.avatars):
        anomalies.append({"code": "SENDER_NOT_IN_AVATARS", "count": len(set(senders) - set(exp.avatars)), "detail": "a sender has no avatar entry"})
    names = {s: Counter(m.sender_display for m in people if m.sender == s and m.sender_display) for s in senders}
    for s in senders:
        if len(names[s]) > 1:
            anomalies.append({"code": "MULTI_DISPLAY_NAME", "count": len(names[s]), "speaker": username_hash(s), "detail": "one username, several display names"})
    sess = exp.session

    def contact_name():
        for k in ("displayName", "remark", "nickname"):
            if sess.get(k) and len(str(sess[k]).strip()) > 0:
                return str(sess[k]), f"session.{k}"
        return (names[contact].most_common(1)[0][0], "senderDisplayName") if names[contact] else (username_hash(contact), "speaker id")

    def self_name():
        return (names[me].most_common(1)[0][0], "senderDisplayName") if names[me] else (username_hash(me), "speaker id")
    out = []
    for s, role, (dn, src) in ((contact, "contact", contact_name()), (me, "self", self_name())):
        mine = [m for m in people if m.sender == s]
        out.append(Speaker(username=s, role=role, display_name=dn, display_source=src, messages=len(mine),
                           text_messages=sum(1 for m in mine if m.type in ("文本消息", "引用消息")), first=mine[0].time.date().isoformat(),
                           last=mine[-1].time.date().isoformat(), is_send=0 if role == "contact" else 1))
    if out[0].display_name == out[1].display_name:
        anomalies.append({"code": "SAME_DISPLAY_NAME", "count": 2, "detail": "both people show the same display name; the role disambiguates"})
    system = [m for m in msgs if is_system(m)]
    attr = Counter(("contact" if m.sender == contact else "self" if m.sender == me else "other") + f"|isSend={m.is_send}" for m in system)
    idx = exp.by_platform_id()
    q = [m for m in msgs if m.reply_to]
    found = [m for m in q if m.reply_to in idx]
    quotes = {"quotes": len(q), "linked": len(found), "unlinked": len(q) - len(found),
              "target_earlier": sum(1 for m in found if idx[m.reply_to].create_time <= m.create_time),
              "quotes_other_speaker": sum(1 for m in found if idx[m.reply_to].sender != m.sender)}
    return Identity(speakers=out, system_messages=len(system), system_by_attribution=dict(attr), anomalies=anomalies, quotes=quotes)
