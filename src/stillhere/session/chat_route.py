"""Non-factual chat routing (``stillhere.session.chat_route/1.0``): a pure greeting / acknowledgement needs no memory answerability check.

Deterministic and deliberately narrow: the whole message, after removing punctuation, emoji, whitespace and trailing particles, must be ONE
whitelisted phrase (optionally repeated: "哈哈哈", "在吗在吗"). Length is never the criterion ("你手机号呢" is short and still a fact request).
Anything not on the list takes the normal memory path. A routed message still goes through every output guard (unsupported specifics,
must-not, leak, repetition, disclosure terms); only the "the memory holds no answer" gate does not apply, because nothing was asked.
"""
from __future__ import annotations

import re

ROUTE_VERSION = "stillhere.session.chat_route/1.0"
GREETINGS = ("你好", "您好", "嗨", "哈喽", "hi", "hello", "hey", "在吗", "在么", "在不在", "早", "早上好", "早安", "中午好", "下午好", "晚上好", "晚安",
             "嗯", "嗯嗯", "好", "好的", "好滴", "行", "ok", "okay", "收到", "谢谢", "多谢", "谢啦", "拜拜", "再见", "回头见", "哈", "嘿", "呵", "嘻",
             "哦", "噢", "哇", "吃了吗", "忙吗", "最近好吗", "辛苦了")
PARTICLES = "呀啊吖哈啦嘛呢哦噢喔~～"
_STRIP = re.compile(r"[\s\W_]+", re.UNICODE)
_EMOJI = re.compile(r"[\U0001F000-\U0001FAFF☀-➿️‍]")
_REPEATED_LAUGH = re.compile(r"^(哈|嘿|呵|嘻|hh|h)+$")


def normalise(text: str) -> str:
    t = _EMOJI.sub("", text or "").lower()
    t = _STRIP.sub("", t)
    return t


def is_small_talk(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    t = normalise(raw)
    if not t:
        return bool(_EMOJI.search(raw))                     # an emoji-only reply ("👍", "😂") is small talk
    if _REPEATED_LAUGH.match(t):
        return True
    core = t.rstrip(PARTICLES) or t
    for g in GREETINGS:
        if core == g or (len(core) % len(g) == 0 and core == g * (len(core) // len(g))):
            return True
    return False
