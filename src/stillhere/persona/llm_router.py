"""Optional LLM router (off by default): the ``router`` provider turns one question into a STRUCTURED intent + time description.

It never answers, never sees the archive and never writes memory: the input is the question and the ask time, the output must match
``SCHEMA`` exactly. Anything else (no JSON, a wrong field, an unknown intent, an invalid date, confidence below ``min_confidence``, a
provider error) falls back to the rule router (``memory_v2.route_v2``, the default), and the decision records why.

    {"intent": "chat" | "fact" | "date", "time": {"kind": "none" | "absolute" | "relative" | "range", "start": "YYYY-MM-DD" | null,
     "end": "YYYY-MM-DD" | null}, "confidence": 0.0 - 1.0}

The rule router keeps resolving the actual date window (``recall.date_scope``); the LLM's time fields are recorded for diagnosis only,
so a model cannot widen what the date route reads.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Optional

INTENTS = ("chat", "fact", "date")
KINDS = ("none", "absolute", "relative", "range")
SYSTEM = ("你是一个对话路由器，只做分类，不回答问题。根据用户的一句话和提问时间，输出 JSON：intent 只能是 chat（普通闲聊、情绪、观点、常识）、"
          "fact（询问两人过去发生的事、说过的话、个人经历、偏好、计划）、date（询问某一天或某段时间聊了什么）；time.kind 为 none/absolute/relative/range，"
          "start/end 用 YYYY-MM-DD（按提问时间换算相对时间；没有则为 null）；confidence 是 0 到 1 的数字。只输出这一个 JSON 对象，不要任何解释。")
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def validate(obj) -> Optional[str]:
    """``None`` when ``obj`` matches the schema exactly, else the first problem."""
    if not isinstance(obj, dict) or set(obj) != {"intent", "time", "confidence"}:
        return "keys"
    if obj["intent"] not in INTENTS:
        return "intent"
    t = obj["time"]
    if not isinstance(t, dict) or set(t) != {"kind", "start", "end"} or t["kind"] not in KINDS:
        return "time"
    for k in ("start", "end"):
        v = t[k]
        if v is not None:
            if not isinstance(v, str) or not _DAY.match(v):
                return f"time.{k}"
            try:
                date.fromisoformat(v)
            except ValueError:
                return f"time.{k}"
    if t["kind"] == "none" and (t["start"] or t["end"]):
        return "time.none_with_dates"
    c = obj["confidence"]
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not 0 <= float(c) <= 1:
        return "confidence"
    return None


def route(text: str, at: Optional[datetime], gen, *, min_confidence: float = 0.6, version: str = "p1") -> dict:
    """``{"route", "source": "llm" | "rules", "why", "llm": {...}}``. ``gen``: a generator of the ``router`` role (or a test fake)."""
    from . import memory_v2 as V2
    rules = V2.route_v2(text, at, version)
    when = (at or datetime.now()).date().isoformat()
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps({"question": text, "ask_date": when}, ensure_ascii=False)}]
    try:
        r = gen(msgs, {"do_sample": False, "max_new_tokens": 160}, None)
        raw = r.get("text", "")
        m = re.search(r"\{.*\}", raw, re.S)
        obj = json.loads(m.group(0)) if m else None
    except Exception as e:  # noqa: BLE001 - provider failure -> rules
        return {"route": rules["route"], "source": "rules", "why": rules["why"], "llm": {"status": "ERROR", "error": type(e).__name__}}
    problem = validate(obj)
    if problem:
        return {"route": rules["route"], "source": "rules", "why": rules["why"], "llm": {"status": "INVALID", "problem": problem}}
    if float(obj["confidence"]) < min_confidence:
        return {"route": rules["route"], "source": "rules", "why": rules["why"], "llm": {"status": "LOW_CONFIDENCE", **obj}}
    return {"route": obj["intent"], "source": "llm", "why": [f"llm:{obj['intent']}@{obj['confidence']}"], "llm": {"status": "OK", **obj},
            "rules_route": rules["route"]}
