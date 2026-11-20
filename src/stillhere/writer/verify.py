"""Source verification: every candidate is checked against the raw messages it cites. Deterministic; no model.

Hard failures make a candidate UNSUPPORTED (never written): a span that is not verbatim in a cited message, a message outside
the group or the persona, a fact about the owner that the owner did not say, an entity or value not found in the cited text, an
inferred assertion, a "dislikes" with no negative wording. Doubt makes it AMBIGUOUS (never written): uncertain wording, a
"likes" whose source says the opposite, a negative statement that v1 does not store as a fact. A field without evidence (a date
the text does not name, a status the words do not carry, a daypart, a correction flag, an unknown reference) is removed and the
candidate becomes PARTIALLY_SUPPORTED: written without that field, never with a guessed one.
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence

from .contracts import AMBIGUOUS, PARTIALLY_SUPPORTED, SUPPORTED, UNSUPPORTED, Candidate, Verdict
from .dates import dates_in, dayparts_in, supported_date
from .extract import GroupMessage

NEGATIVE_LIKE = ("不喜欢", "不太喜欢", "不爱", "讨厌", "受不了", "不想再", "烦")
NEGATIVE_ANY = ("不", "没", "未", "别", "无", "非", "讨厌", "烦", "受不了")
UNCERTAIN = ("可能", "也许", "好像", "大概", "说不定", "不确定", "或许", "应该会")
CORRECTION_CUES = ("说错", "记错", "搞错", "弄错", "不对", "其实", "更正", "纠正", "应该是", "不是")
RETRACTION_CUES = ("不算", "收回", "并没有", "没有", "撤回", "删掉", "不对", "取消", "作废", "当我没说", "没去", "其实没", "没报上", "没参加", "没做")
DENIAL_CUES = ("不算", "其实没", "没报上", "并没有", "记错", "当我没说", "没去", "没参加", "撤回", "收回")
PREDICATE_CUES = {"lives_in": ("住", "家在", "家里", "搬", "定居", "老家"), "likes": ("喜欢", "爱", "最爱", "偏爱", "爱喝", "爱吃"),
                  "routine": ("每天", "通常", "习惯", "总是", "一般", "经常", "改成", "都会", "每周", "每晚", "每天早上")}
STATE_ASPECTS = ("睡眠", "心情", "情绪", "精力", "压力", "体重", "运动量")     # v1: everyday states only; nothing clinical, nothing diagnostic
STRONG_PLAN = ("打算", "计划", "准备", "想要", "将要", "明天", "后天", "下周", "下个月", "下星期")
STATUS_CUES = {
    "PLANNED": ("打算", "计划", "准备", "想", "要", "会", "将", "安排", "下周", "明天", "后天", "下个月", "周末", "改到", "改成", "推迟", "提前"),
    "SCHEDULED": ("订了", "订好", "约了", "约好", "预约", "报名", "买了票", "定了", "定好", "安排好", "挂号"),
    "IN_PROGRESS": ("正在", "在路上", "进行中", "还在"),
    "COMPLETED": ("已经", "刚", "完", "过了", "回来", "结束", "了"),
    "CANCELLED": ("取消", "不去了", "算了", "作废", "泡汤", "去不了", "不办了", "推掉", "不参加了", "不做了", "没去成", "退了", "退票", "没成"),
    "UNKNOWN": (),
}


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _bigrams(text: str) -> set:
    t = "".join(ch for ch in text if ch.isalnum())
    return {t[i:i + 2] for i in range(len(t) - 1)} or ({t} if t else set())


def grounded(value: str, text: str, need: float, short_share: float = 0.75) -> bool:
    """Every part of the value is in the text: verbatim, by character bigrams, or (short parts, ``读书`` in ``读了一会儿书``) by characters."""
    parts = [p for p in re.split(r"[\s:：/、，,]+", value) if p]
    tc = _compact(text)
    for p in parts:
        if p in tc:
            continue
        b = _bigrams(p)
        if b and len(b & _bigrams(text)) / len(b) >= need:
            continue
        chars = [ch for ch in p if ch.isalnum()]
        if len(chars) <= 4 and chars and sum(ch in tc for ch in chars) / len(chars) >= short_share:
            continue
        return False
    return bool(parts)


def verify(c: Candidate, group: Sequence[GroupMessage], persona_id: str, existing_ids: Sequence[str], strict_self: bool = False) -> Verdict:
    """``strict_self`` (two-person chats, e.g. WeFlow): a "self" fact needs a span in a message the owner wrote; an event merely
    ADDRESSED to the owner (the SMS rule below: every message of the other person is addressed to the owner there) is not enough.
    Default False: unchanged behaviour (LifeBench / SMS)."""
    by_id: Dict[str, GroupMessage] = {m.message_id: m for m in group}
    reasons: List[str] = []
    dropped: List[str] = []
    if c.persona_id != persona_id:
        return Verdict(c, UNSUPPORTED, ["persona"])
    if any(mid not in by_id for mid in c.source_message_ids):
        return Verdict(c, UNSUPPORTED, ["message outside the group"])
    cited = [by_id[mid] for mid in c.source_message_ids]
    located = []
    for span in c.source_spans:
        home = [m for m in cited if _compact(span) and _compact(span) in _compact(m.content)]
        if not home:
            return Verdict(c, UNSUPPORTED, [f"span not verbatim in a cited message: {span[:30]}"])
        located += home
    text = " ".join(m.content for m in cited)
    spans = " ".join(c.source_spans)
    if c.assertion_type == "inferred":
        return Verdict(c, UNSUPPORTED, ["inferred, not stated"])
    if c.subject == "self":
        addressed = (not strict_self) and c.predicate == "event" and any(m.recipient == "本人" for m in located)   # 今晚聚餐取消了, sent to the owner
        if not any(m.speaker_id == persona_id for m in located) and not addressed:
            return Verdict(c, UNSUPPORTED, ["a fact about the owner that the owner did not say"])
    else:
        name = c.subject.split(":", 1)[1].strip()
        if not name or not (name in text or any(name in m.sender or name in m.recipient for m in cited)):
            return Verdict(c, UNSUPPORTED, [f"entity {name!r} not in the cited messages"])
        if not any(m.speaker_id == persona_id or name in m.sender for m in located):
            return Verdict(c, UNSUPPORTED, ["said by neither the owner nor the person it is about"])
    obj_need = 0.3 if c.predicate in ("event", "state", "routine") else 0.5
    short = 0.5 if c.predicate == "event" else 0.75          # an activity noun from its verb: 跑步 from 跑了5公里
    if not grounded(c.object, text, obj_need, short):
        return Verdict(c, UNSUPPORTED, [f"value not found in the cited text: {c.object[:30]}"])
    if c.certainty == "uncertain" or any(w in spans for w in UNCERTAIN):
        return Verdict(c, AMBIGUOUS, ["uncertain wording"])
    if c.predicate == "likes" and (c.polarity == "negative" or any(w in spans for w in NEGATIVE_LIKE)):
        if any(w in spans for w in NEGATIVE_LIKE):
            c.predicate, c.polarity = "dislikes", "positive"         # 不太喜欢咖啡 is a dislike, said with a negation
            reasons.append("normalised likes+negation to dislikes")
        else:
            return Verdict(c, AMBIGUOUS, ["likes with negative polarity and no negative wording"])
    if c.predicate == "dislikes" and not any(w in spans for w in NEGATIVE_ANY):
        return Verdict(c, UNSUPPORTED, ["dislikes without negative wording"])
    cues = PREDICATE_CUES.get(c.predicate)
    if cues and not any(w in text for w in cues):
        return Verdict(c, UNSUPPORTED, [f"no wording for {c.predicate} in the cited text (inferred)"])
    if c.predicate == "state" and not any(c.object.replace("：", ":").split(":", 1)[0].startswith(a) for a in STATE_ASPECTS):
        return Verdict(c, UNSUPPORTED, ["state aspect outside the v1 list"])
    if c.predicate == "event" and c.retracts and c.event_status == "CANCELLED" and any(w in text for w in STATUS_CUES["CANCELLED"]) and not any(w in text for w in DENIAL_CUES):
        c.retracts = False                                   # 取消了 ends a plan; it does not deny that the plan existed
        reasons.append("cancellation is a state change, not a retraction")
    if c.polarity == "negative" and c.predicate not in ("dislikes",) and not c.retracts and not (c.predicate == "event" and c.event_status == "CANCELLED"):
        return Verdict(c, AMBIGUOUS, ["negative statement: v1 stores no negative facts"])
    if c.predicate == "event" and c.event_status:
        cues = STATUS_CUES[c.event_status]
        planned_words = any(w in text for w in STRONG_PLAN)       # not 会 / 要 / 想: 一会儿, 要不要
        if c.event_status == "COMPLETED":
            said = max(m.timestamp for m in cited)
            past_day = any(d <= said.astimezone(said.tzinfo).date() for d in dates_in(text, said)[0])
            ok = (any(w in text for w in cues) or past_day) and not (planned_words and not any(w in text for w in ("已经", "完", "回来", "结束", "过了")))
        else:
            ok = c.event_status == "UNKNOWN" or any(w in text for w in cues)   # the words of the cited messages must carry the status
            if not ok and c.event_status == "SCHEDULED" and any(w in text for w in STATUS_CUES["PLANNED"]):
                c.event_status, ok = "PLANNED", True        # no booking words, but a plan: keep the weaker claim the text supports
                reasons.append("SCHEDULED without booking words: kept as PLANNED")
        if not ok:
            dropped.append("event_status")            # 打算去了 / 要去了: a plan, the 了 does not make it done
            c.event_status = None
    for name in ("planned_time", "event_time", "valid_from", "valid_to"):
        value = getattr(c, name)
        if value and not any(supported_date(value, m.content + " " + spans, m.timestamp) for m in cited):
            dropped.append(name)
            setattr(c, name, None)
    if c.daypart and c.daypart not in dayparts_in(text):
        dropped.append("daypart")
        c.daypart = None
    if c.corrects and not any(w in spans for w in CORRECTION_CUES):
        dropped.append("corrects")
        c.corrects = False
    if c.retracts and not any(w in spans for w in RETRACTION_CUES):
        dropped.append("retracts")
        c.retracts = False
    if c.refers_to and c.refers_to not in existing_ids:
        dropped.append("refers_to")
        c.refers_to = None
    return Verdict(c, PARTIALLY_SUPPORTED if dropped else SUPPORTED, reasons, dropped)
