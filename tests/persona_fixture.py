"""Synthetic WeFlow exports for the persona pipeline tests. Invented people and text only (no real chat)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

TZ = timezone(timedelta(hours=8))
CONTACT, SELF = "wxid_synth_contact", "wxid_synth_self"
T0 = int(datetime(2025, 1, 1, 9, 0, tzinfo=TZ).timestamp())


class Chat:
    def __init__(self, contact_name="小合成", self_name="我自己", session_type="私聊"):
        self.msgs, self.lid = [], 0
        self.names = {CONTACT: contact_name, SELF: self_name}
        self.session_type = session_type

    def add(self, t, who, content="", type_="文本消息", local_type=1, is_send=None, **kw):
        self.lid += 1
        m = {"localId": self.lid, "createTime": int(t), "formattedTime": datetime.fromtimestamp(int(t), TZ).strftime("%Y-%m-%d %H:%M:%S"),
             "type": type_, "localType": local_type, "content": content, "isSend": (1 if who == SELF else 0) if is_send is None else is_send,
             "senderUsername": who, "senderDisplayName": kw.pop("display", self.names.get(who, who)), "source": "", "senderAvatarKey": who,
             "platformMessageId": kw.pop("pid", f"9{self.lid:08d}")}
        m.update(kw)
        self.msgs.append(m)
        return m

    def text(self, t, who, content, **kw):
        return self.add(t, who, content, **kw)

    def system(self, t, who, content="你撤回了一条消息", is_send=0):
        return self.add(t, who, content, type_="系统消息", local_type=10000, is_send=is_send)

    def media(self, t, who, type_="图片消息", content="[图片]", local_type=3, **kw):
        return self.add(t, who, content, type_=type_, local_type=local_type, **kw)

    def obj(self):
        return {"weflow": {"version": "1.0.3", "exportedAt": T0, "generator": "WeFlow"},
                "session": {"wxid": CONTACT, "nickname": "-", "remark": self.names[CONTACT], "displayName": self.names[CONTACT], "type": self.session_type,
                            "lastTimestamp": self.msgs[-1]["createTime"] if self.msgs else T0, "messageCount": len(self.msgs), "avatar": ""},
                "messages": self.msgs, "avatars": {CONTACT: "", SELF: ""}}

    def write(self, path: Path) -> Path:
        path.write_text(json.dumps(self.obj(), ensure_ascii=False), encoding="utf-8")
        return path


def standard(days: int = 80) -> Chat:
    """``days`` daily sessions. Each day: the self account opens, the contact answers (sometimes in two bubbles), a frequent short reply
    ("好"), a sticker with a caption, one image-only contact turn, one revoke notice attributed to SELF with isSend 0, and a quote."""
    c = Chat()
    for d in range(days):
        t = T0 + d * 86400
        c.text(t, SELF, f"第{d}天：今天过得怎么样？")
        c.text(t + 60, CONTACT, f"还行，第{d}天上班有点累")
        c.text(t + 90, CONTACT, f"晚上想吃火锅{d}")                        # merged into the same contact turn (30 s later)
        c.text(t + 600, SELF, f"那周末一起去吃吧{d}")
        c.text(t + 660, CONTACT, "好")                                     # frequent short reply, different ids every day
        c.text(t + 1200, SELF, f"顺便看个电影第{d}部？")
        c.add(t + 1260, CONTACT, f"[表情包：开心{d % 3}]", type_="动画表情", local_type=47, emojiCaption=f"开心{d % 3}")
        c.system(t + 1300, SELF)                                           # real exports: system line under SELF, isSend 0
        c.text(t + 1800, SELF, f"我发你链接{d}")
        c.media(t + 1860, CONTACT)                                         # contact turn with an image only: never a target
        q = c.text(t + 2400, SELF, f"你说的那家店叫什么{d}")
        c.add(t + 2460, CONTACT, f"就是老街那家{d}", type_="引用消息", local_type=244813135921, replyToMessageId=q["platformMessageId"],
              quotedContent=q["content"], quotedSender="不可靠的名字", quotedType="1")
    return c
