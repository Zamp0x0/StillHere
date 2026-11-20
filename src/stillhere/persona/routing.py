"""The V1 persona memory router (moved verbatim from ``persona.runtime`` so ``persona.memory_v2`` can extend it without an import cycle;
``persona.runtime`` re-exports every name, so callers are unchanged)."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

HISTORY_CUE = re.compile(r"以前|之前|那天|那次|上次|那时|记得|小时候|过去|曾经|去年|前年|上个月|上周|昨天|前天|\d{1,4}年|\d{1,2}月\d{1,2}[日号]|几号|什么时候|哪天|哪一年")
PERSONAL_CUE = re.compile(r"[你我](现在|以前|之前|最近|目前|当时|那时)?(在哪|在什么|的|家|住|工作|上班|上学|读书|喜欢|讨厌|养|妹|姐|哥|弟|爸|妈|爷|奶|老家|"
                          r"去过|做过|是不是|有没有)")
QUESTION_CUE = re.compile(r"[？?]|吗|呢$|什么|啥|怎么|哪|几|谁|多少|是不是|有没有")


def route_of(text: str, at: Optional[datetime] = None) -> str:
    """'date' for "what did we talk about on <day / month>", 'fact' for a history / personal-fact question, else 'chat'.
    Deterministic; small talk (``session.chat_route``) is never a memory query."""
    from stillhere.session.chat_route import is_small_talk
    from . import recall as R
    t = (text or "").strip()
    if not t or is_small_talk(t) or not QUESTION_CUE.search(t):
        return "chat"
    if R.is_date_question(t, at or datetime.now(R.TZ)):
        return "date"
    return "fact" if HISTORY_CUE.search(t) or PERSONAL_CUE.search(t) else "chat"
