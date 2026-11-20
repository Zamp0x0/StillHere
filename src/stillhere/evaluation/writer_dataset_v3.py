"""Batch 2 independent public synthetic world (``b2-writer-dataset-3``): three NEW invented personas, written spec-first, before any Writer,
Retrieval or Reflection run on it, and frozen by its manifest hash. Nothing here was used to develop Batch 2 code (DEV = the M2 world and the
M2.1 v2 world, whose failures were inspected).

Same Gold format as ``writer_dataset`` / ``writer_dataset_v2`` (``writer_eval`` scores it with ``ds=writer_dataset_v3``). High-risk cases:
a friend's / colleague's facts said by the owner (must not become the owner's), plans that never happen, negation, contacts' own plans in
SMS (sender, not owner), a completed event that must update the planned one (same event), reschedule, cancellation, correction, retraction,
update of a residence, preference change, late-arriving evidence, a duplicate statement, uncertain statements and pure small talk.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import List

from stillhere.contracts import CanonicalMessage, DataOrigin, SourceKind, SourceRef

from .writer_dataset import C, Checkpoint, Conv, Fact, ObsExpect  # noqa: F401  (same Gold format)

TZ = timezone(timedelta(hours=8))
RA, RB, RC = "syn.ra", "syn.rb", "syn.rc"
NAMES = {RA: "方远", RB: "林雪", RC: "周默"}
BOT = {p: f"{p}.assistant" for p in NAMES}
KAI, MEI, BIN = f"{RA}.contact.kai", f"{RB}.contact.mei", f"{RC}.contact.bin"
CONTACTS = {KAI: "阿凯", MEI: "小梅", BIN: "周斌"}
SPEAKERS = {**{b: "AI 助手" for b in BOT.values()}, **CONTACTS}
CHAT, SMS = "b2.src.chat", "b2.src.sms"
VERSION = "b2-writer-dataset-3"


def t(month: int, day: int, hour: int = 10, minute: int = 0) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=TZ)


def E(key, subject, obj, src, spans, alts, ekey, status, activity, planned=None, event_time=None, daypart=None, **kw):
    return Fact(key, subject, "event", obj, src, spans, alts, "EVENT", ekey, status, activity, planned, event_time, daypart, **kw)


def conversations() -> List[Conv]:
    return [
        # ---------------- 方远 (RA): 软件测试工程师
        C("x01", RA, t(9, 2, 9), [("u", "我住在西安雁塔区，在一家汽车公司做软件测试。"), ("a", "好的。"), ("u", "平时我喜欢打羽毛球。")],
          [Fact("ra.home1", "self", "lives_in", "西安雁塔区", (0,), ("住在西安雁塔区",), ("雁塔区",)),
           Fact("ra.work1", "self", "works_at", "一家汽车公司", (0,), ("在一家汽车公司",), ("汽车公司",)),
           Fact("ra.job", "self", "job", "软件测试", (0,), ("做软件测试",), ("测试",)),
           Fact("ra.badminton", "self", "likes", "打羽毛球", (2,), ("平时我喜欢打羽毛球",), ("羽毛球",))], tags=("multi_fact",)),
        C("x02", RA, t(9, 3, 21), [("u", "我朋友阿凯特别喜欢喝咖啡，一天三杯。"), ("a", "挺能喝的。"), ("u", "我不喝咖啡，我喜欢喝茶。")],
          [Fact("ra.coffee", "self", "dislikes", "咖啡", (2,), ("我不喝咖啡",), ("喝咖啡",)),
           Fact("ra.tea", "self", "likes", "茶", (2,), ("我喜欢喝茶",), ("喝茶",))],
          forbidden=(("likes", "咖啡"),), tags=("third_party", "negation")),
        C("x03", RA, t(9, 5, 20), [("u", "我不吃鸡蛋，过敏。"), ("a", "那要注意。")],
          [Fact("ra.egg", "self", "dislikes", "鸡蛋", (0,), ("我不吃鸡蛋",), ("吃鸡蛋",))], forbidden=(("likes", "鸡蛋"),), tags=("negation",)),
        C("x04", RA, t(9, 7, 22), [("u", "我准备下周六去香港看演唱会。"), ("a", "听起来不错。")],
          [E("ra.hk", "self", "去香港看演唱会", (0,), ("我准备下周六去香港看演唱会",), ("看演唱会", "去香港"), "hongkong", "PLANNED", "看演唱会")],
          forbidden=(("event", "香港:COMPLETED"),), tags=("plan",)),
        C("x05", RA, t(9, 10, 19), [(KAI, "我下个月要搬去成都了，公司调岗。"), ("u", "那以后少了个球友。")],
          [E("ra.kai_move", "person:阿凯", "搬去成都", (0,), ("我下个月要搬去成都了",), ("去成都",), "kai_move", "PLANNED", "搬家", "2026-10")],
          forbidden=(("lives_in", "成都"), ("event", "成都:self")), tags=("sender_recipient",), kind="sms"),
        C("x06", RA, t(9, 11, 21), [("u", "香港那趟取消了，演唱会延期了。"), ("a", "可惜。")],
          [E("ra.hk_cancel", "self", "去香港看演唱会", (0,), ("香港那趟取消了",), ("看演唱会", "去香港"), "hongkong", "CANCELLED", "看演唱会",
             refers="ra.hk", op="SUPERSEDE", relation="TEMPORAL_CHANGE")], tags=("cancel",)),
        C("x07", RA, t(9, 14, 20), [("u", "我报名了10月18日的半程马拉松。"), ("a", "加油。")],
          [E("ra.run", "self", "半程马拉松", (0,), ("我报名了10月18日的半程马拉松",), ("马拉松", "半马"), "half_marathon", "SCHEDULED", "跑步", "2026-10-18")],
          tags=("schedule",)),
        C("x08", RA, t(9, 20, 9), [("u", "我之前说错了，我在一家新能源车企工作，不是传统汽车公司。"), ("a", "已更正。")],
          [Fact("ra.work2", "self", "works_at", "一家新能源车企", (0,), ("我之前说错了，我在一家新能源车企工作",), ("新能源车企",), corrects=True, refers="ra.work1",
                op="SUPERSEDE", relation="CORRECTION")], forbidden=(), tags=("correction",)),
        C("x09", RA, t(10, 8, 21), [("u", "这周搬家了，现在住在曲江新区。"), ("a", "新家怎么样？"), ("u", "离公司近。")],
          [Fact("ra.home2", "self", "lives_in", "曲江新区", (0,), ("现在住在曲江新区",), ("曲江",), refers="ra.home1", op="SUPERSEDE", relation="TEMPORAL_CHANGE")],
          tags=("knowledge_update",)),
        C("x10", RA, t(10, 18, 19), [("u", "今天跑完半程马拉松了，用时两小时出头。"), ("a", "厉害！")],
          [E("ra.run_done", "self", "半程马拉松", (0,), ("今天跑完半程马拉松了",), ("半马", "马拉松"), "half_marathon", "COMPLETED", "跑步", None, "2026-10-18",
             refers="ra.run", op="SUPERSEDE", relation="TEMPORAL_CHANGE")], tags=("plan_completed",)),
        C("x11", RA, t(10, 20, 8), [("u", "下个月可能去北京出差，还没定。"), ("a", "好。")], [], forbidden=(("event", "北京"),), tags=("uncertain",)),
        # ---------------- 林雪 (RB): 小学英语老师
        C("y01", RB, t(9, 1, 8), [("u", "我是小学英语老师，住在青岛市南区。"), ("a", "了解。"), ("u", "我养了一只叫年糕的柯基。")],
          [Fact("rb.job", "self", "job", "小学英语老师", (0,), ("我是小学英语老师",), ("英语老师", "老师")),
           Fact("rb.home", "self", "lives_in", "青岛市南区", (0,), ("住在青岛市南区",), ("市南区",)),
           Fact("rb.pet", "self", "pet", "柯基年糕", (2,), ("我养了一只叫年糕的柯基",), ("年糕", "柯基"))], tags=("multi_fact",)),
        C("y02", RB, t(9, 3, 20), [("u", "我同事小陈住在黄岛，每天坐轮渡上班。"), ("a", "挺远的。")], [],
          forbidden=(("lives_in", "黄岛"),), tags=("third_party",)),
        C("y03", RB, t(9, 4, 21), [("u", "我喜欢吃辣，越辣越好。"), ("a", "胃要紧。")],
          [Fact("rb.spicy", "self", "likes", "吃辣", (0,), ("我喜欢吃辣",), ("辣",))], tags=("preference",)),
        C("y04", RB, t(9, 6, 22), [("u", "这周批改作业到很晚，睡眠很差。"), ("a", "注意休息。")],
          [Fact("rb.sleep1", "self", "state", "睡眠:差", (0,), ("睡眠很差",), ("睡眠:很差",))], tags=("state",)),
        C("y05", RB, t(9, 8, 19), [("u", "我预约了9月20日去拔智齿。"), ("a", "祝顺利。")],
          [E("rb.tooth", "self", "拔智齿", (0,), ("我预约了9月20日去拔智齿",), ("智齿",), "tooth", "SCHEDULED", "看牙", "2026-09-20")], tags=("schedule",)),
        C("y06", RB, t(9, 12, 20), [(MEI, "姐，我国庆要去云南旅游，你帮我照看一下猫吧。"), ("u", "行，交给我。")],
          [E("rb.mei_trip", "person:小梅", "去云南旅游", (0,), ("我国庆要去云南旅游",), ("云南旅游",), "mei_yunnan", "PLANNED", "旅游")],
          forbidden=(("event", "云南:self"),), tags=("sender_recipient",), kind="sms"),
        C("y07", RB, t(9, 17, 20), [("u", "拔智齿改到9月24日了，医生临时有手术。"), ("a", "好。")],
          [E("rb.tooth2", "self", "拔智齿", (0,), ("拔智齿改到9月24日了",), ("智齿", "拔牙"), "tooth", "SCHEDULED", "看牙", "2026-09-24",
             refers="rb.tooth", op="SUPERSEDE", relation="TEMPORAL_CHANGE")], tags=("reschedule",)),
        C("y08", RB, t(9, 24, 18), [("u", "今天拔完智齿了，脸肿了。"), ("a", "冰敷一下。")],
          [E("rb.tooth3", "self", "拔智齿", (0,), ("今天拔完智齿了",), ("智齿", "拔牙"), "tooth", "COMPLETED", "看牙", None, "2026-09-24",
             refers="rb.tooth", op="SUPERSEDE", relation="TEMPORAL_CHANGE")], tags=("plan_completed",)),
        C("y09", RB, t(9, 29, 21), [("u", "国庆这几天睡眠好多了。"), ("a", "太好了。")],
          [Fact("rb.sleep2", "self", "state", "睡眠:好", (0,), ("睡眠好多了",), ("睡眠:好多了",), refers="rb.sleep1", op="SUPERSEDE", relation="TEMPORAL_CHANGE")],
          tags=("state",)),
        C("y10", RB, t(10, 6, 21), [("u", "最近胃不好，我现在不吃辣了。"), ("a", "养胃要紧。")],
          [Fact("rb.spicy2", "self", "dislikes", "吃辣", (0,), ("我现在不吃辣了",), ("辣",))], tags=("preference_change",)),
        C("y11", RB, t(10, 9, 20), [("u", "年糕的事我跟你说过吧，就是那只柯基。"), ("a", "说过。")],
          [Fact("rb.pet2", "self", "pet", "柯基", (0,), ("就是那只柯基",), ("年糕",), op="NOOP", relation="DUPLICATE")], tags=("duplicate",)),
        C("y12", RB, t(10, 12, 9), [("u", "学校可能让我带毕业班，还没正式通知。"), ("a", "等通知。")], [], tags=("uncertain",)),
        # ---------------- 周默 (RC): 退休会计，爱摄影
        C("z01", RC, t(9, 2, 8), [("u", "我退休前是会计，现在住在杭州拱墅区。"), ("a", "好的。"), ("u", "每天早上去运河边拍照。")],
          [Fact("rc.job", "self", "job", "退休的会计", (0,), ("我退休前是会计",), ("会计",)),
           Fact("rc.home", "self", "lives_in", "杭州拱墅区", (0,), ("住在杭州拱墅区",), ("拱墅区",)),
           Fact("rc.routine", "self", "routine", "早上去运河边拍照", (2,), ("每天早上去运河边拍照",), ("拍照", "运河"))], tags=("multi_fact", "routine")),
        C("z02", RC, t(9, 4, 19), [("u", "我儿子周斌在深圳做设计。"), ("a", "了解。")],
          [Fact("rc.son", "self", "relationship", "儿子 周斌", (0,), ("我儿子周斌",), ("周斌",)),
           Fact("rc.son_job", "person:周斌", "job", "设计", (0,), ("在深圳做设计",), ("做设计",))], tags=("entity",)),
        C("z03", RC, t(9, 6, 20), [("u", "我打算10月5日去黄山拍日出。"), ("a", "记得带厚衣服。")],
          [E("rc.huangshan", "self", "去黄山拍日出", (0,), ("我打算10月5日去黄山拍日出",), ("黄山", "拍日出"), "huangshan", "PLANNED", "摄影", "2026-10-05")],
          tags=("plan",)),
        C("z04", RC, t(9, 9, 9), [(BIN, "爸，我十一要加班，回不了杭州。"), ("u", "工作要紧。")],
          [E("rc.bin_work", "person:周斌", "加班", (0,), ("我十一要加班",), ("十一加班",), "bin_overtime", "PLANNED", "加班")],
          forbidden=(("event", "加班:self"),), tags=("sender_recipient",), kind="sms"),
        C("z05", RC, t(9, 12, 21), [("u", "老同事老王喜欢钓鱼，常叫我一起。"), ("a", "可以一起去。"), ("u", "我不喜欢钓鱼，坐不住。")],
          [Fact("rc.fishing", "self", "dislikes", "钓鱼", (2,), ("我不喜欢钓鱼",), ("钓鱼",))], forbidden=(("likes", "钓鱼"),), tags=("third_party", "negation")),
        C("z06", RC, t(9, 15, 20), [("u", "上个月体检血糖有点高，医生让我少吃甜的。"), ("a", "注意饮食。")], [], tags=("unsupported_aspect",)),
        C("z07", RC, t(9, 18, 20), [("u", "我之前说错了，我退休前是出纳，不是会计。"), ("a", "好的，已更正。")],
          [Fact("rc.job2", "self", "job", "退休的出纳", (0,), ("我之前说错了，我退休前是出纳",), ("出纳",), corrects=True, refers="rc.job", op="SUPERSEDE",
                relation="CORRECTION")], tags=("correction",)),
        C("z08", RC, t(10, 5, 19), [("u", "今天在黄山拍到了日出，太值了。"), ("a", "恭喜！")],
          [E("rc.huangshan_done", "self", "去黄山拍日出", (0,), ("今天在黄山拍到了日出",), ("黄山", "拍日出"), "huangshan", "COMPLETED", "摄影", None, "2026-10-05",
             refers="rc.huangshan", op="SUPERSEDE", relation="TEMPORAL_CHANGE")], tags=("plan_completed",)),
        C("z09", RC, t(10, 7, 21), [("u", "10月6日那天我在黄山下山时把相机摔了。", t(10, 6, 21), t(10, 7, 21)), ("a", "人没事吧？", t(10, 6, 21, 2), t(10, 7, 21))],
          [E("rc.camera", "self", "相机摔了", (0,), ("把相机摔了",), ("摔了相机",), "camera_drop", "COMPLETED", "意外", None, "2026-10-06")], tags=("late_arriving",)),
        C("z10", RC, t(10, 14, 9), [("u", "我报名了社区摄影班，每周三上课。"), ("a", "不错。")],
          [E("rc.class", "self", "社区摄影班", (0,), ("我报名了社区摄影班",), ("摄影班",), "photo_class", "SCHEDULED", "上课")], tags=("schedule",)),
        C("z11", RC, t(10, 16, 20), [("u", "摄影班那个其实没报上，名额满了。"), ("a", "下次吧。")],
          [E("rc.class_retract", "self", "社区摄影班", (0,), ("摄影班那个其实没报上",), ("摄影班",), "photo_class", None, "上课", retracts=True, refers="rc.class",
             op="RETRACT", relation="CORRECTION")], tags=("retraction",)),
        # ---------------- small talk / hypotheticals (no fact about the persona)
        C("w01", RA, t(9, 9, 22), [("u", "你觉得测试和开发哪个更累？"), ("a", "各有各的累。"), ("u", "也是。"), ("a", "嗯。")], [], tags=("noise",)),
        C("w02", RA, t(9, 26, 23), [("u", "要是中了大奖我就去环游世界。"), ("a", "好梦想。"), ("u", "做梦而已。"), ("a", "哈哈。")], [], tags=("noise",)),
        C("w03", RA, t(10, 12, 22), [("u", "推荐一部悬疑电影吧。"), ("a", "可以看看《消失的爱人》。"), ("u", "看过了。"), ("a", "那试试别的。")], [], tags=("noise",)),
        C("w04", RB, t(9, 10, 21), [("u", "英语单词怎么记得快？"), ("a", "间隔重复比较有效。"), ("u", "我回头试试。"), ("a", "好。")], [], tags=("noise",)),
        C("w05", RB, t(9, 22, 22), [("u", "隔壁班的王老师下学期要去支教。"), ("a", "很有意义。"), ("u", "是啊。"), ("a", "嗯。")], [],
          forbidden=(("event", "支教"),), tags=("noise", "third_party")),
        C("w06", RB, t(10, 3, 21), [("u", "给我讲个冷笑话。"), ("a", "为什么海是咸的？因为鱼在里面哭。"), ("u", "好冷。"), ("a", "哈哈。")], [], tags=("noise",)),
        C("w07", RC, t(9, 21, 9), [("u", "光圈和快门是什么关系？"), ("a", "一起决定曝光量。"), ("u", "明白了。"), ("a", "好。")], [], tags=("noise",)),
        C("w08", RC, t(10, 1, 20), [("u", "如果年轻十岁，我想去西藏拍星空。"), ("a", "现在也可以考虑。"), ("u", "随便说说。"), ("a", "好的。")], [],
          forbidden=(("event", "西藏"),), tags=("noise",)),
        C("w09", RC, t(10, 11, 9), [("u", "今天天气不错。"), ("a", "适合出门。"), ("u", "嗯。"), ("a", "好。")], [], tags=("noise",)),
    ]


def checkpoints() -> List[Checkpoint]:
    return [
        Checkpoint("ra_home_yanta_in_sept", RA, "ra.home1", "lives_in", t(9, 25), t(10, 20), "雁塔"),
        Checkpoint("ra_home_qujiang_after_move", RA, "ra.home1", "lives_in", t(10, 15), t(10, 20), "曲江"),
        Checkpoint("ra_hk_cancelled_on_sep12", RA, "ra.hk", "event.status", t(9, 12), t(9, 12), "CANCELLED"),
        Checkpoint("ra_run_scheduled_before_race", RA, "ra.run", "event.status", t(10, 1), t(10, 1), "SCHEDULED"),
        Checkpoint("rb_tooth_rescheduled_24", RB, "rb.tooth", "event.planned_time", t(9, 18), t(9, 18), "2026-09-24"),
        Checkpoint("rb_tooth_done_known_sep25", RB, "rb.tooth", "event.status", t(9, 25), t(9, 25), "COMPLETED"),
        Checkpoint("rc_huangshan_planned_on_oct1", RC, "rc.huangshan", "event.status", t(10, 1), t(10, 1), "PLANNED"),
        Checkpoint("rc_camera_not_known_on_oct6", RC, "rc.camera", "event.status", t(10, 6, 23), t(10, 6, 23), None),
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
    return [SourceRef(source_id=CHAT, kind=SourceKind.CONVERSATION, origin=DataOrigin.SYNTHETIC, description="Batch 2 independent synthetic dataset: chats."),
            SourceRef(source_id=SMS, kind=SourceKind.CONVERSATION, origin=DataOrigin.SYNTHETIC, description="Batch 2 independent synthetic dataset: SMS.")]


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
            "forbidden": sum(len(c.forbidden) for c in cs), "checkpoints": len(checkpoints()), "tags": sorted({tg for c in cs for tg in c.tags}), "origin": "synthetic"}
