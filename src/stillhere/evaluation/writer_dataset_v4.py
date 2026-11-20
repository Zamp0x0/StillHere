"""Batch 2.1 writer CONFIRMATION world (``b21-writer-dataset-4``): three NEW invented personas, written and frozen (manifest hash) BEFORE the
Batch 2.1 writer fix was written or run. Run once, after the fix configuration is locked; never used to tune.

Storage policy the gold follows (stated before the run; the same policy the Batch 2.1 writer implements):
* the persona's own stated facts, preferences, routines and events are stored; a relationship the persona states is stored;
* other people (contacts, stated relations): their job / workplace / home / school and their own events are stored under ``person:<name>``;
  their preferences, pets, habits and health are NOT stored (privacy minimisation: not the persona's memory);
* what someone else says ABOUT the persona ("我妈说我小时候不吃香菜", "阿雷说我肯定喜欢猫", "朋友都说我做的面好吃") is not the persona's own
  statement and is not stored;
* suggestions ("要不周末去…？"), hypotheticals ("如果…", "要是…就…"), wishes, advice ("医生建议…"), guesses ("我觉得他可能…") and uncertain
  plans ("可能…还没约") are not stored;
* one real occurrence = one event: reschedule / cancel / complete update it; causes and details are not separate events; two real occurrences
  on different days are two events.

High-risk cases per persona: third-party preference next to the persona's own, hearsay about the persona, suggestion, hypothetical, advice,
uncertain plan, plan -> reschedule -> completed, plan -> cancelled, two separate occurrences, preference change, duplicate mention,
contact's own plan in SMS (sender, not owner), negation, small talk.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import List

from stillhere.contracts import CanonicalMessage, DataOrigin, SourceKind, SourceRef

from .writer_dataset import C, Checkpoint, Conv, Fact, ObsExpect  # noqa: F401  (same Gold format)

VERSION = "b21-writer-dataset-4"
TZ = timezone(timedelta(hours=8))
TA, TB, TC = "syn.ta", "syn.tb", "syn.tc"
NAMES = {TA: "宋岚", TB: "韩松", TC: "白露"}
BOT = {p: f"{p}.assistant" for p in NAMES}
LEI, MOM, NING = f"{TA}.contact.lei", f"{TB}.contact.mom", f"{TC}.contact.ning"
CONTACTS = {LEI: "阿雷", MOM: "妈", NING: "宁宁"}
SPEAKERS = {**{b: "AI 助手" for b in BOT.values()}, **CONTACTS}
CHAT, SMS = "b21.src.chat", "b21.src.sms"


def t(month: int, day: int, hour: int = 10, minute: int = 0) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=TZ)


def E(key, subject, obj, src, spans, alts, ekey, status, activity, planned=None, event_time=None, daypart=None, **kw):
    return Fact(key, subject, "event", obj, src, spans, alts, "EVENT", ekey, status, activity, planned, event_time, daypart, **kw)


def conversations() -> List[Conv]:
    return [
        # ---------------- 宋岚 (TA): UI 设计师，成都
        C("a01", TA, t(9, 2, 20), [("u", "我在成都一家游戏公司做UI设计，住在高新区。"), ("a", "好的。")],
          [Fact("ta.work", "self", "works_at", "一家游戏公司", (0,), ("在成都一家游戏公司",), ("游戏公司",)),
           Fact("ta.job", "self", "job", "UI设计", (0,), ("做UI设计",), ("设计",)),
           Fact("ta.home", "self", "lives_in", "成都高新区", (0,), ("住在高新区",), ("高新区",))], tags=("multi_fact",)),
        C("a02", TA, t(9, 3, 21), [("u", "我朋友阿雷特别喜欢咖啡，我反而喜欢喝茶。"), ("a", "各有所好。")],
          [Fact("ta.rel_lei", "self", "relationship", "朋友 阿雷", (0,), ("我朋友阿雷",), ("阿雷",)),
           Fact("ta.tea", "self", "likes", "茶", (0,), ("我反而喜欢喝茶",), ("喝茶",))],
          forbidden=(("likes", "咖啡"),), tags=("third_party", "preference")),
        C("a03", TA, t(9, 4, 21), [("u", "阿雷说我肯定喜欢猫，其实我没养过宠物。"), ("a", "哈哈。")], [],
          forbidden=(("pet", "猫"), ("likes", "猫")), tags=("hearsay", "negation")),
        C("a04", TA, t(9, 5, 19), [("u", "要不周末去青城山？"), ("a", "可以考虑。"), ("u", "再看吧。")], [],
          forbidden=(("event", "青城山"),), tags=("suggestion",)),
        C("a05", TA, t(9, 8, 20), [("u", "我下周六要去重庆参加朋友婚礼。"), ("a", "记得带红包。")],
          [E("ta.wedding", "self", "去重庆参加婚礼", (0,), ("我下周六要去重庆参加朋友婚礼",), ("参加婚礼", "婚礼", "重庆"), "cq_wedding", "PLANNED", "婚礼", "2026-09-19")],
          tags=("plan",)),
        C("a06", TA, t(9, 12, 20), [("u", "重庆的婚礼改到9月20日了。"), ("a", "好。")],
          [E("ta.wedding2", "self", "去重庆参加婚礼", (0,), ("重庆的婚礼改到9月20日了",), ("参加婚礼", "婚礼", "重庆"), "cq_wedding", "PLANNED", "婚礼", "2026-09-20",
             refers="ta.wedding", op="SUPERSEDE", relation="TEMPORAL_CHANGE")], tags=("reschedule",)),
        C("a07", TA, t(9, 14, 22), [("u", "如果下个月有空，我想去西藏。"), ("a", "西藏很美。")], [], forbidden=(("event", "西藏"),), tags=("hypothetical",)),
        C("a08", TA, t(9, 16, 21), [("u", "我每周二和周四晚上去游泳。"), ("a", "坚持得不错。")],
          [Fact("ta.swim", "self", "routine", "每周二和周四晚上去游泳", (0,), ("我每周二和周四晚上去游泳",), ("游泳",))], tags=("routine",)),
        C("a09", TA, t(9, 18, 20), [(LEI, "我下个月要去上海出差一周。"), ("u", "好的，回来聚。")],
          [E("ta.lei_trip", "person:阿雷", "去上海出差", (0,), ("我下个月要去上海出差一周",), ("上海出差", "出差"), "lei_shanghai", "PLANNED", "出差", "2026-10")],
          forbidden=(("event", "上海:self"),), tags=("sender_recipient",), kind="sms"),
        C("a10", TA, t(9, 21, 21), [("u", "昨天在重庆参加完婚礼了，累但开心。"), ("a", "辛苦了。")],
          [E("ta.wedding3", "self", "去重庆参加婚礼", (0,), ("昨天在重庆参加完婚礼了",), ("参加婚礼", "婚礼", "重庆"), "cq_wedding", "COMPLETED", "婚礼", None, "2026-09-20",
             refers="ta.wedding", op="SUPERSEDE", relation="TEMPORAL_CHANGE")], tags=("plan_completed",)),
        C("a11", TA, t(9, 25, 21), [("u", "我不喝咖啡，喝了睡不着。"), ("a", "那就喝茶。")],
          [Fact("ta.coffee", "self", "dislikes", "咖啡", (0,), ("我不喝咖啡",), ("喝咖啡",))], tags=("negation",)),
        C("a12", TA, t(10, 8, 20), [("u", "人家都说那家火锅好吃，我还没去过。"), ("a", "可以去试试。")], [],
          forbidden=(("likes", "火锅"), ("event", "火锅")), tags=("hearsay",)),
        C("a13", TA, t(10, 12, 21), [("u", "国庆最后一天我去爬了峨眉山。"), ("a", "风景怎么样？"), ("u", "很美。")],
          [E("ta.emei", "self", "爬峨眉山", (0,), ("国庆最后一天我去爬了峨眉山",), ("峨眉山",), "emei", "COMPLETED", "爬山", None, "2026-10-07")], tags=("completed",)),
        C("a14", TA, t(10, 15, 22), [("u", "今天好热。"), ("a", "注意防暑。"), ("u", "嗯。")], [], tags=("noise",)),
        # ---------------- 韩松 (TB): 厨师，南京
        C("b01", TB, t(9, 2, 21), [("u", "我在南京一家酒店当厨师，住在鼓楼区。"), ("a", "了解。")],
          [Fact("tb.work", "self", "works_at", "一家酒店", (0,), ("在南京一家酒店",), ("酒店",)),
           Fact("tb.job", "self", "job", "厨师", (0,), ("当厨师",), ()),
           Fact("tb.home", "self", "lives_in", "南京鼓楼区", (0,), ("住在鼓楼区",), ("鼓楼区",))], tags=("multi_fact",)),
        C("b02", TB, t(9, 3, 21), [("u", "我妹妹韩梅在北京大学读研究生。"), ("a", "真厉害。")],
          [Fact("tb.rel_mei", "self", "relationship", "妹妹 韩梅", (0,), ("我妹妹韩梅",), ("韩梅",)),
           Fact("tb.mei_school", "person:韩梅", "school", "北京大学", (0,), ("在北京大学读研究生",), ("北大",))], tags=("entity",)),
        C("b03", TB, t(9, 5, 20), [("u", "我妹喜欢吃甜的，我不爱吃甜。"), ("a", "口味不一样。")],
          [Fact("tb.sweet", "self", "dislikes", "甜", (0,), ("我不爱吃甜",), ("吃甜", "甜食"))],
          forbidden=(("likes", "甜"),), tags=("third_party", "negation")),
        C("b04", TB, t(9, 7, 19), [(MOM, "儿子，我下周三去医院复查。"), ("u", "好的，路上注意。")],
          [E("tb.mom_check", "person:妈", "去医院复查", (0,), ("我下周三去医院复查",), ("复查",), "mom_checkup", "PLANNED", "复查", "2026-09-16")],
          forbidden=(("event", "复查:self"),), tags=("sender_recipient",), kind="sms"),
        C("b05", TB, t(9, 9, 21), [("u", "我妈之前说过我小时候不吃香菜。"), ("a", "现在呢？"), ("u", "早就无所谓了。")], [],
          forbidden=(("dislikes", "香菜"),), tags=("hearsay",)),
        C("b06", TB, t(9, 10, 20), [("u", "我报名了9月27日的厨艺比赛。"), ("a", "加油！")],
          [E("tb.contest", "self", "厨艺比赛", (0,), ("我报名了9月27日的厨艺比赛",), ("比赛",), "cook_contest", "SCHEDULED", "比赛", "2026-09-27")], tags=("schedule",)),
        C("b07", TB, t(9, 12, 20), [("u", "今天游了一次泳。"), ("a", "运动不错。")],
          [E("tb.swim1", "self", "游泳", (0,), ("今天游了一次泳",), ("游泳",), "swim_0912", "COMPLETED", "游泳", None, "2026-09-12")], tags=("occurrence",)),
        C("b08", TB, t(9, 19, 20), [("u", "今天又去游了一次泳。"), ("a", "坚持住。")],
          [E("tb.swim2", "self", "游泳", (0,), ("今天又去游了一次泳",), ("游泳",), "swim_0919", "COMPLETED", "游泳", None, "2026-09-19")], tags=("occurrence",)),
        C("b09", TB, t(9, 20, 21), [("u", "厨艺比赛取消了，主办方没拉到赞助。"), ("a", "可惜了。")],
          [E("tb.contest2", "self", "厨艺比赛", (0,), ("厨艺比赛取消了",), ("比赛",), "cook_contest", "CANCELLED", "比赛",
             refers="tb.contest", op="SUPERSEDE", relation="TEMPORAL_CHANGE")], tags=("cancel",)),
        C("b10", TB, t(9, 23, 21), [("u", "要是明天不下雨，我就去紫金山跑步。"), ("a", "看天气。")], [],
          forbidden=(("event", "紫金山"),), tags=("hypothetical",)),
        C("b11", TB, t(9, 26, 20), [("u", "医生建议我多睡觉少熬夜。"), ("a", "听医生的。")], [],
          forbidden=(("event", "熬夜"), ("routine", "睡")), tags=("advice",)),
        C("b12", TB, t(10, 2, 21), [("u", "我现在不喝酒了，戒了。"), ("a", "好事。")],
          [Fact("tb.alcohol", "self", "dislikes", "喝酒", (0,), ("我现在不喝酒了",), ("酒",))], tags=("preference_change",)),
        C("b13", TB, t(10, 6, 20), [("u", "推荐个下饭的菜谱吧。"), ("a", "鱼香肉丝。"), ("u", "会做。")], [], tags=("noise",)),
        # ---------------- 白露 (TC): 退休护士，厦门
        C("c01", TC, t(9, 2, 9), [("u", "我退休前是护士，现在住在厦门思明区。"), ("a", "好的。")],
          [Fact("tc.job", "self", "job", "退休的护士", (0,), ("我退休前是护士",), ("护士",)),
           Fact("tc.home", "self", "lives_in", "厦门思明区", (0,), ("现在住在厦门思明区",), ("思明区",))], tags=("multi_fact",)),
        C("c02", TC, t(9, 4, 9), [("u", "我每天傍晚去海边散步。"), ("a", "很惬意。")],
          [Fact("tc.walk", "self", "routine", "傍晚去海边散步", (0,), ("我每天傍晚去海边散步",), ("散步",))], tags=("routine",)),
        C("c03", TC, t(9, 6, 20), [("u", "我老伴喜欢下棋，我对下棋没兴趣。"), ("a", "各有爱好。")],
          [Fact("tc.chess", "self", "dislikes", "下棋", (0,), ("我对下棋没兴趣",), ())],
          forbidden=(("likes", "下棋"),), tags=("third_party", "negation")),
        C("c04", TC, t(9, 8, 19), [(NING, "姑姑，我10月份要参加教师资格考试。"), ("u", "加油！")],
          [E("tc.ning_exam", "person:宁宁", "教师资格考试", (0,), ("我10月份要参加教师资格考试",), ("资格考试", "考试"), "ning_exam", "PLANNED", "考试", "2026-10")],
          forbidden=(("event", "考试:self"),), tags=("sender_recipient",), kind="sms"),
        C("c05", TC, t(9, 10, 9), [("u", "我打算9月20日去鼓浪屿拍照。"), ("a", "带好相机。")],
          [E("tc.gulangyu", "self", "去鼓浪屿拍照", (0,), ("我打算9月20日去鼓浪屿拍照",), ("鼓浪屿",), "gulangyu", "PLANNED", "拍照", "2026-09-20")], tags=("plan",)),
        C("c06", TC, t(9, 15, 20), [("u", "可能下周去医院做个体检，还没约。"), ("a", "约好告诉我。")], [],
          forbidden=(("event", "体检"),), tags=("uncertain",)),
        C("c07", TC, t(9, 18, 20), [("u", "朋友都说我做的沙茶面好吃，我自己也挺喜欢做饭。"), ("a", "真棒。")],
          [Fact("tc.cook", "self", "likes", "做饭", (0,), ("我自己也挺喜欢做饭",), ())],
          forbidden=(("likes", "沙茶面"),), tags=("hearsay", "preference")),
        C("c08", TC, t(9, 21, 9), [("u", "鼓浪屿那天下大雨，没去成。"), ("a", "下次再去。")],
          [E("tc.gulangyu2", "self", "去鼓浪屿拍照", (0,), ("鼓浪屿那天下大雨，没去成",), ("鼓浪屿",), "gulangyu", "CANCELLED", "拍照",
             refers="tc.gulangyu", op="SUPERSEDE", relation="TEMPORAL_CHANGE")], tags=("cancel",)),
        C("c09", TC, t(9, 25, 9), [("u", "我养了一只鹦鹉，叫豆豆。"), ("a", "好可爱。")],
          [Fact("tc.pet", "self", "pet", "鹦鹉豆豆", (0,), ("我养了一只鹦鹉，叫豆豆",), ("鹦鹉", "豆豆"))], tags=("pet",)),
        C("c10", TC, t(10, 3, 9), [("u", "豆豆就是我那只鹦鹉，昨天又学会一句话。"), ("a", "真聪明。")],
          [Fact("tc.pet2", "self", "pet", "鹦鹉", (0,), ("豆豆就是我那只鹦鹉",), ("豆豆",), op="NOOP", relation="DUPLICATE")], tags=("duplicate",)),
        C("c11", TC, t(10, 13, 20), [(NING, "姑姑，我考完了，感觉还行。"), ("u", "辛苦了！")],
          [E("tc.ning_exam2", "person:宁宁", "教师资格考试", (0,), ("我考完了",), ("考试", "资格考试"), "ning_exam", "COMPLETED", "考试",
             refers="tc.ning_exam", op="SUPERSEDE", relation="TEMPORAL_CHANGE")], tags=("plan_completed",), kind="sms"),
        C("c12", TC, t(10, 14, 20), [("u", "我觉得宁宁可能喜欢画画。"), ("a", "可以问问她。")], [],
          forbidden=(("likes", "画画"),), tags=("guess",)),
        C("c13", TC, t(10, 16, 9), [("u", "今天的潮水好大。"), ("a", "注意安全。"), ("u", "嗯。")], [], tags=("noise",)),
    ]


def checkpoints() -> List[Checkpoint]:
    return [
        Checkpoint("ta_wedding_rescheduled_20", TA, "ta.wedding", "event.planned_time", t(9, 13), t(9, 13), "2026-09-20"),
        Checkpoint("ta_wedding_done", TA, "ta.wedding", "event.status", t(9, 22), t(9, 22), "COMPLETED"),
        Checkpoint("tb_contest_cancelled", TB, "tb.contest", "event.status", t(9, 21), t(9, 21), "CANCELLED"),
        Checkpoint("tc_gulangyu_cancelled", TC, "tc.gulangyu", "event.status", t(9, 22), t(9, 22), "CANCELLED"),
    ]


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
    return [SourceRef(source_id=CHAT, kind=SourceKind.CONVERSATION, origin=DataOrigin.SYNTHETIC, description="Batch 2.1 writer confirmation dataset: chats."),
            SourceRef(source_id=SMS, kind=SourceKind.CONVERSATION, origin=DataOrigin.SYNTHETIC, description="Batch 2.1 writer confirmation dataset: SMS.")]


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
            "gold_facts": sum(len(c.facts) for c in cs), "gold_facts_committing": sum(f.op != "NOOP" for c in cs for f in c.facts),
            "forbidden": sum(len(c.forbidden) for c in cs), "checkpoints": len(checkpoints()), "tags": sorted({tg for c in cs for tg in c.tags}), "origin": "synthetic",
            "storage_policy": "b21 (see module docstring)"}
