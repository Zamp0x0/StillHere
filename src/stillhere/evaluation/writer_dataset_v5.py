"""Batch 2.1 post-GPU retest world (``b21p-dataset-5``): three NEW invented personas, written and frozen (manifest hash) BEFORE the post-GPU
runtime fixes were run on it. Used ONLY for the targeted Colab retest's new confirmation set (memory written by the gold extractor, so the
retest isolates the RUNTIME: grounding, length, guard, fallback; writer quality is measured on its own sets).

Personas have different reply styles (see ``batch2_style_history``): 陆遥 (syn.ua) terse, 许曼 (syn.ub) long-winded, 程一 (syn.uc) writes in
several short bubbles. Not derived from any first-run failure case (new names, places, facts and wording).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import List

from stillhere.contracts import CanonicalMessage, DataOrigin, SourceKind, SourceRef

from .writer_dataset import C, Checkpoint, Conv, Fact, ObsExpect  # noqa: F401  (same Gold format)

VERSION = "b21p-dataset-5.1"   # 5.1: two lines reworded so the gold is writable (found by the gold-extractor check, before any runtime run)
TZ = timezone(timedelta(hours=8))
UA, UB, UC = "syn.ua", "syn.ub", "syn.uc"
NAMES = {UA: "陆遥", UB: "许曼", UC: "程一"}
BOT = {p: f"{p}.assistant" for p in NAMES}
HAO, JIE, MIAO = f"{UA}.contact.hao", f"{UB}.contact.jie", f"{UC}.contact.miao"
CONTACTS = {HAO: "浩子", JIE: "阿洁", MIAO: "苗苗"}
SPEAKERS = {**{b: "AI 助手" for b in BOT.values()}, **CONTACTS}
CHAT, SMS = "b21p.src.chat", "b21p.src.sms"


def t(month: int, day: int, hour: int = 10, minute: int = 0) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=TZ)


def E(key, subject, obj, src, spans, alts, ekey, status, activity, planned=None, event_time=None, daypart=None, **kw):
    return Fact(key, subject, "event", obj, src, spans, alts, "EVENT", ekey, status, activity, planned, event_time, daypart, **kw)


def conversations() -> List[Conv]:
    return [
        # ---------------- 陆遥 (UA): 外卖站长，长沙，话少
        C("u01", UA, t(9, 3, 21), [("u", "我住在长沙开福区，在一个外卖站当站长。"), ("a", "辛苦。")],
          [Fact("ua.job", "self", "job", "外卖站站长", (0,), ("在一个外卖站当站长",), ("站长",)),
           Fact("ua.home", "self", "lives_in", "长沙开福区", (0,), ("我住在长沙开福区",), ("开福区",))], tags=("multi_fact",)),
        C("u02", UA, t(9, 5, 22), [("u", "我不吃香菜。"), ("a", "记住了。")],
          [Fact("ua.cilantro", "self", "dislikes", "香菜", (0,), ("我不吃香菜",), ())], tags=("negation",)),
        C("u03", UA, t(9, 8, 20), [("u", "我报名了10月10日的骑行赛。"), ("a", "加油。")],
          [E("ua.ride", "self", "骑行赛", (0,), ("我报名了10月10日的骑行赛",), ("骑行",), "ride_race", "SCHEDULED", "骑行", "2026-10-10")], tags=("schedule",)),
        C("u04", UA, t(9, 20, 21), [("u", "骑行赛取消了，台风。"), ("a", "可惜。")],
          [E("ua.ride2", "self", "骑行赛", (0,), ("骑行赛取消了",), ("骑行",), "ride_race", "CANCELLED", "骑行", refers="ua.ride", op="SUPERSEDE",
             relation="TEMPORAL_CHANGE")], tags=("cancel",)),
        C("u05", UA, t(9, 22, 20), [(HAO, "哥，我下周二去武汉面试。"), ("u", "好好考。")],
          [E("ua.hao_interview", "person:浩子", "去武汉面试", (0,), ("我下周二去武汉面试",), ("武汉面试", "面试"), "hao_interview", "PLANNED", "面试", "2026-09-29")],
          forbidden=(("event", "武汉:self"),), tags=("sender_recipient",), kind="sms"),
        # ---------------- 许曼 (UB): 中学语文老师，苏州，话多
        C("v01", UB, t(9, 2, 20), [("u", "我住在苏州姑苏区，在一所中学当语文老师，已经教了八年了。"), ("a", "很资深了。")],
          [Fact("ub.job", "self", "job", "语文老师", (0,), ("在一所中学当语文老师",), ("语文",)),
           Fact("ub.home", "self", "lives_in", "苏州姑苏区", (0,), ("我住在苏州姑苏区",), ("姑苏区",))], tags=("multi_fact",)),
        C("v02", UB, t(9, 4, 21), [("u", "我女儿叫朵朵，今年上小学二年级。"), ("a", "可爱。")],
          [Fact("ub.rel", "self", "relationship", "女儿 朵朵", (0,), ("我女儿叫朵朵",), ("朵朵",)),
           Fact("ub.dd_school", "person:朵朵", "school", "小学二年级", (0,), ("今年上小学二年级",), ("小学",))], tags=("entity",)),
        C("v03", UB, t(9, 6, 21), [("u", "我之前说错了，我教了十年，不是八年。"), ("a", "好的。")], [], tags=("correction_of_detail",)),
        C("v04", UB, t(9, 9, 20), [("u", "我打算国庆带朵朵去杭州玩两天。"), ("a", "好主意。")],
          [E("ub.hz", "self", "带朵朵去杭州玩", (0,), ("我打算国庆带朵朵去杭州玩两天",), ("去杭州", "杭州"), "hz_trip", "PLANNED", "旅行")], tags=("plan",)),
        C("v05", UB, t(10, 8, 21), [("u", "国庆去杭州玩完了，朵朵特别喜欢西湖。"), ("a", "真好。")],
          [E("ub.hz2", "self", "带朵朵去杭州玩", (0,), ("国庆去杭州玩完了",), ("去杭州", "杭州"), "hz_trip", "COMPLETED", "旅行", refers="ub.hz", op="SUPERSEDE",
             relation="TEMPORAL_CHANGE")], tags=("plan_completed",)),
        C("v06", UB, t(10, 10, 20), [("u", "我们教研组可能下学期换组长，还没定。"), ("a", "等通知。")], [], tags=("uncertain",)),
        # ---------------- 程一 (UC): 程序员，深圳，分条发消息
        C("w01", UC, t(9, 2, 22), [("u", "我在深圳一家做支付的公司写后端。"), ("a", "好的。")],
          [Fact("uc.work", "self", "works_at", "一家做支付的公司", (0,), ("在深圳一家做支付的公司",), ("支付",)),
           Fact("uc.job", "self", "job", "后端", (0,), ("写后端",), ("后端开发",))], tags=("multi_fact",)),
        C("w02", UC, t(9, 4, 22), [("u", "我住南山区。"), ("a", "收到。")],
          [Fact("uc.home", "self", "lives_in", "南山区", (0,), ("我住南山区",), ())], tags=("simple",)),
        C("w03", UC, t(9, 10, 22), [("u", "我之前说错了，我住的是福田区，不是南山区。"), ("a", "已更正。")],
          [Fact("uc.home2", "self", "lives_in", "福田区", (0,), ("我住的是福田区",), (), corrects=True, refers="uc.home", op="SUPERSEDE", relation="CORRECTION")],
          tags=("correction",)),
        C("w04", UC, t(9, 12, 22), [("u", "我喜欢打篮球，周末常去。"), ("a", "好运动。")],
          [Fact("uc.ball", "self", "likes", "打篮球", (0,), ("我喜欢打篮球",), ("篮球",))], tags=("preference",)),
        C("w05", UC, t(9, 14, 21), [(MIAO, "哥，我十月要搬去上海。"), ("u", "到了说一声。")],
          [E("uc.miao_move", "person:苗苗", "搬去上海", (0,), ("我十月要搬去上海",), ("去上海",), "miao_move", "PLANNED", "搬家", "2026-10")],
          forbidden=(("lives_in", "上海"), ("event", "上海:self")), tags=("sender_recipient",), kind="sms"),
        C("w06", UC, t(9, 18, 22), [("u", "下周四约了牙医洗牙。"), ("a", "好。")],
          [E("uc.dentist", "self", "洗牙", (0,), ("下周四约了牙医洗牙",), ("牙医",), "dentist", "SCHEDULED", "看牙", "2026-09-24")], tags=("schedule",)),
    ]


def checkpoints() -> List[Checkpoint]:
    return [Checkpoint("uc_home_futian_after_correction", UC, "uc.home", "lives_in", t(9, 15), t(9, 15), "福田")]


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
    return [SourceRef(source_id=CHAT, kind=SourceKind.CONVERSATION, origin=DataOrigin.SYNTHETIC, description="Batch 2.1 post-GPU retest dataset: chats."),
            SourceRef(source_id=SMS, kind=SourceKind.CONVERSATION, origin=DataOrigin.SYNTHETIC, description="Batch 2.1 post-GPU retest dataset: SMS.")]


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
