"""Batch 2.1 final confirmation world (``b21f-dataset-6``): three NEW invented personas, written and frozen (manifest hash) BEFORE any writer,
retrieval or reader run on it. Memory is written by the REAL extractor (DeepSeek, prompt v4) with the final writer (M2.2.1 + M2.2.2), so this
set confirms the writer rules (explicit corrections whose cue is outside the cited span; public-holiday dates), the holiday anchor in
retrieval, and the reader, end to end. Not derived from any earlier case (new names, places, facts and wording). It also holds controls:
a plan with no date (must not be read as a holiday plan) and a third party's holiday plan.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import List

from stillhere.contracts import CanonicalMessage, DataOrigin, SourceKind, SourceRef

from .writer_dataset import C, Checkpoint, Conv, Fact, ObsExpect  # noqa: F401  (same Gold format)

VERSION = "b21f-dataset-6"
TZ = timezone(timedelta(hours=8))
XA, XB, XC = "syn.xa", "syn.xb", "syn.xc"
NAMES = {XA: "郑岚", XB: "魏然", XC: "罗敏"}
BOT = {p: f"{p}.assistant" for p in NAMES}
FAN, TAO = f"{XB}.contact.fan", f"{XC}.contact.tao"
CONTACTS = {FAN: "阿凡", TAO: "小涛"}
SPEAKERS = {**{b: "AI 助手" for b in BOT.values()}, **CONTACTS}
CHAT, SMS = "b21f.src.chat", "b21f.src.sms"


def t(month: int, day: int, hour: int = 10, minute: int = 0) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=TZ)


def E(key, subject, obj, src, spans, alts, ekey, status, activity, planned=None, event_time=None, daypart=None, **kw):
    return Fact(key, subject, "event", obj, src, spans, alts, "EVENT", ekey, status, activity, planned, event_time, daypart, **kw)


def conversations() -> List[Conv]:
    return [
        # ---------------- 郑岚 (XA): 杭州，会计
        C("x01", XA, t(9, 2, 21), [("u", "我在一家物流公司做会计，住在杭州西湖区。"), ("a", "好的。")],
          [Fact("xa.job", "self", "job", "会计", (0,), ("在一家物流公司做会计",), ()),
           Fact("xa.work", "self", "works_at", "一家物流公司", (0,), ("在一家物流公司做会计",), ("物流公司",)),
           Fact("xa.home", "self", "lives_in", "杭州西湖区", (0,), ("住在杭州西湖区",), ("西湖区",))], tags=("multi_fact",)),
        C("x02", XA, t(9, 5, 21), [("u", "我刚才记错了，我住在拱墅区，不是西湖区。"), ("a", "已更正。")],
          [Fact("xa.home2", "self", "lives_in", "拱墅区", (0,), ("我住在拱墅区",), ("杭州拱墅区",), corrects=True, refers="xa.home", op="SUPERSEDE",
                relation="CORRECTION")], tags=("correction",)),
        C("x03", XA, t(9, 8, 20), [("u", "今年五一第二天我去了乌镇。"), ("a", "好玩吗？"), ("u", "人很多。")],
          [E("xa.wuzhen", "self", "去乌镇", (0,), ("今年五一第二天我去了乌镇",), ("乌镇",), "wuzhen", "COMPLETED", "旅行", None, "2026-05-02")], tags=("holiday_day",)),
        C("x04", XA, t(9, 15, 21), [("u", "国庆打算去厦门看海。"), ("a", "不错。")],
          [E("xa.xiamen", "self", "去厦门看海", (0,), ("国庆打算去厦门看海",), ("厦门",), "xiamen", "PLANNED", "旅行", "2026-10")], tags=("holiday_plan",)),
        C("x05", XA, t(9, 18, 21), [("u", "我不喝咖啡，喝了睡不着。"), ("a", "明白。")],
          [Fact("xa.coffee", "self", "dislikes", "咖啡", (0,), ("我不喝咖啡",), ())], tags=("negation",)),
        # ---------------- 魏然 (XB): 成都，大学生
        C("y01", XB, t(9, 3, 22), [("u", "我在四川大学读新闻，室友叫小邹。"), ("a", "好的。")],
          [Fact("xb.school", "self", "school", "四川大学", (0,), ("我在四川大学读新闻",), ("川大",)),
           Fact("xb.room", "self", "relationship", "室友 小邹", (0,), ("室友叫小邹",), ("小邹",))], tags=("multi_fact",)),
        C("y02", XB, t(9, 6, 22), [("u", "更正一下：我室友叫小周，不是小邹。"), ("a", "改好了。")],
          [Fact("xb.room2", "self", "relationship", "室友 小周", (0,), ("我室友叫小周",), ("小周",), corrects=True, refers="xb.room", op="SUPERSEDE",
                relation="CORRECTION")], tags=("correction",)),
        C("y03", XB, t(9, 10, 21), [("u", "我打算学吉他。"), ("a", "加油。")],
          [E("xb.guitar", "self", "学吉他", (0,), ("我打算学吉他",), ("吉他",), "guitar", "PLANNED", "学习")], tags=("undated_plan",)),
        C("y04", XB, t(9, 20, 21), [(FAN, "国庆最后一天我去重庆面试。"), ("u", "祝顺利。")],
          [E("xb.fan_interview", "person:阿凡", "去重庆面试", (0,), ("国庆最后一天我去重庆面试",), ("重庆面试", "面试"), "fan_interview", "PLANNED", "面试",
             "2026-10-07")], forbidden=(("event", "重庆:self"),), tags=("sender_recipient", "holiday_day"), kind="sms"),
        C("y05", XB, t(10, 12, 21), [("u", "元旦我要回绵阳老家。"), ("a", "好的。")],
          [E("xb.mianyang", "self", "回绵阳老家", (0,), ("元旦我要回绵阳老家",), ("绵阳",), "mianyang", "PLANNED", "回老家", "2027-01-01")], tags=("holiday_plan",)),
        # ---------------- 罗敏 (XC): 西安，退休教师
        C("z01", XC, t(9, 2, 9), [("u", "我退休前是中学物理老师，现在住在西安雁塔区。"), ("a", "您好。")],
          [Fact("xc.job", "self", "job", "中学物理老师", (0,), ("我退休前是中学物理老师",), ("物理老师",)),
           Fact("xc.home", "self", "lives_in", "西安雁塔区", (0,), ("现在住在西安雁塔区",), ("雁塔区",))], tags=("multi_fact",)),
        C("z02", XC, t(9, 4, 9), [("u", "我养了一只狗。"), ("a", "叫什么名字？"), ("u", "还没起。")],
          [Fact("xc.pet", "self", "pet", "狗", (0,), ("我养了一只狗",), ())], tags=("pet",)),
        C("z03", XC, t(9, 7, 9), [("u", "我之前说错了，我养的是一只猫，不是狗。"), ("a", "好的，已更正。")],
          [Fact("xc.pet2", "self", "pet", "猫", (0,), ("我养的是一只猫",), ("一只猫",), corrects=True, refers="xc.pet", op="SUPERSEDE", relation="CORRECTION")],
          tags=("correction",)),
        C("z04", XC, t(9, 9, 8), [("u", "我每天早上去公园打太极。"), ("a", "好习惯。")],
          [Fact("xc.taiji", "self", "routine", "每天早上去公园打太极", (0,), ("我每天早上去公园打太极",), ("打太极", "太极"))], tags=("routine",)),
        C("z05", XC, t(10, 9, 9), [("u", "国庆第三天我和老伴去了华山。"), ("a", "爬上去了吗？"), ("u", "坐缆车上去的。")],
          [E("xc.huashan", "self", "和老伴去华山", (0,), ("国庆第三天我和老伴去了华山",), ("华山",), "huashan", "COMPLETED", "旅行", None, "2026-10-03")],
          tags=("holiday_day",)),
        C("z06", XC, t(10, 10, 20), [(TAO, "妈，我下周六回家吃饭。"), ("u", "好，做你爱吃的。")],
          [E("xc.tao_visit", "person:小涛", "回家吃饭", (0,), ("我下周六回家吃饭",), ("回家",), "tao_visit", "PLANNED", "聚餐", "2026-10-17")],
          tags=("sender_recipient",), kind="sms"),
    ]


def checkpoints() -> List[Checkpoint]:
    return [Checkpoint("xa_home_gongshu_after_correction", XA, "xa.home", "lives_in", t(9, 10), t(9, 10), "拱墅"),
            Checkpoint("xc_pet_cat_after_correction", XC, "xc.pet", "pet", t(9, 10), t(9, 10), "猫")]


def observation_expectations() -> List[ObsExpect]:
    return []


def message_id(conv: Conv, i: int) -> str:
    return f"{conv.persona}.{conv.kind}.{conv.cid}.{i}"


def group_id(conv: Conv) -> str:
    if conv.kind == "chat":
        return f"{conv.persona}.{conv.kind}.{conv.cid}"
    contact = next(m.who for m in conv.messages if m.who not in ("u", "a"))
    return f"{conv.persona}.sms.{contact.rsplit('.', 1)[-1]}"


def raw_messages(conv: Conv) -> List[CanonicalMessage]:
    out = []
    for i, m in enumerate(conv.messages):
        speaker = conv.persona if m.who == "u" else BOT[conv.persona] if m.who == "a" else m.who
        loc = f"pos={i}" + ((";type=" + ("发送" if m.who == "u" else "接收")) if conv.kind == "sms" else "")
        out.append(CanonicalMessage(conversation_id=group_id(conv), message_id=message_id(conv, i), target_persona_id=conv.persona, speaker_id=speaker, timestamp=m.at,
                                    content=m.text, source_type="synthetic_chat_round" if conv.kind == "chat" else "synthetic_sms", source_id=CHAT if conv.kind == "chat" else SMS,
                                    locator=loc, recorded_at=m.recorded_at or m.at))
    return out


def sources() -> List[SourceRef]:
    return [SourceRef(source_id=CHAT, kind=SourceKind.CONVERSATION, origin=DataOrigin.SYNTHETIC, description="Batch 2.1 final confirmation dataset: chats."),
            SourceRef(source_id=SMS, kind=SourceKind.CONVERSATION, origin=DataOrigin.SYNTHETIC, description="Batch 2.1 final confirmation dataset: SMS.")]


def load(api) -> None:
    for s in sources():
        api.register_source(s)
    for conv in conversations():
        api.ingest_messages(raw_messages(conv))


def order() -> List[Conv]:
    return sorted(conversations(), key=lambda c: (max(m.recorded_at or m.at for m in c.messages), c.cid))


def manifest() -> dict:
    cs = conversations()
    body = {"version": VERSION, "conversations": [asdict(c) for c in cs], "checkpoints": [asdict(x) for x in checkpoints()]}
    digest = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return {"version": VERSION, "dataset_sha256": digest, "personas": NAMES, "conversations": len(cs), "messages": sum(len(c.messages) for c in cs),
            "gold_facts": sum(len(c.facts) for c in cs), "forbidden": sum(len(c.forbidden) for c in cs), "origin": "synthetic"}
