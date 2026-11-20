"""Public synthetic Writer / Observation evaluation dataset (BUILD-M2). Three invented personas; nothing describes a real person.

The gold is a hand-written specification, independent of any Writer output: for every conversation, the atomic facts it states
(subject, predicate, value and accepted variants, event fields, the operation it should cause, the messages and verbatim spans
that support it), plus the facts it must NOT produce (``forbidden``), bitemporal checkpoints, and the observations that should
and should not form. ``reference_candidates`` turns the gold into candidates, so the deterministic part of the pipeline (verify,
resolve, commit, observations) is tested without a model; the real model is measured against the same gold.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from stillhere.contracts import CanonicalMessage, DataOrigin, SourceKind, SourceRef

TZ = timezone(timedelta(hours=8))
PA, PB, PC = "syn.pa", "syn.pb", "syn.pc"
NAMES = {PA: "林晓", PB: "陈默", PC: "周宁"}
BOT = {p: f"{p}.assistant" for p in NAMES}
CONTACTS = {f"{PA}.contact.lilei": "李雷", f"{PA}.contact.wang": "王经理", f"{PB}.contact.su": "苏晴"}
SPEAKERS = {**{b: "AI 助手" for b in BOT.values()}, **CONTACTS}
CHAT, SMS = "m2.src.chat", "m2.src.sms"
VERSION = "m2-writer-dataset-1"


def t(month: int, day: int, hour: int = 10, minute: int = 0) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=TZ)


@dataclass
class Msg:
    who: str                    # "u" owner, "a" assistant, or a contact speaker id
    text: str
    at: datetime
    recorded_at: Optional[datetime] = None


@dataclass
class Fact:
    key: str
    subject: str                # "self" | "person:<name>"
    predicate: str
    object: str
    src: Tuple[int, ...]        # message positions (0-based) in the conversation
    spans: Tuple[str, ...]
    alts: Tuple[str, ...] = ()  # accepted value variants
    type: str = "FACT"
    event_key: Optional[str] = None
    status: Optional[str] = None
    activity: Optional[str] = None
    planned: Optional[str] = None
    event_time: Optional[str] = None
    daypart: Optional[str] = None
    corrects: bool = False
    retracts: bool = False
    refers: Optional[str] = None            # gold key of the earlier fact or event this one updates
    op: str = "ADD"                         # expected logical operation
    relation: str = "INDEPENDENT_FACT"


@dataclass
class Conv:
    cid: str
    persona: str
    kind: str                   # "chat" | "sms"
    messages: List[Msg]
    facts: List[Fact] = field(default_factory=list)
    forbidden: Tuple[Tuple[str, str], ...] = ()     # (predicate, value fragment) that must not be committed
    tags: Tuple[str, ...] = ()


def C(cid, persona, start, lines, facts=(), forbidden=(), tags=(), kind="chat", step=2):
    msgs = []
    for i, line in enumerate(lines):
        who, text = line[0], line[1]
        at = line[2] if len(line) > 2 and line[2] else start + timedelta(minutes=step * i)
        rec = line[3] if len(line) > 3 else None
        msgs.append(Msg(who, text, at, rec))
    return Conv(cid, persona, kind, msgs, list(facts), tuple(forbidden), tuple(tags))


def conversations() -> List[Conv]:
    LI, WANG, SU = f"{PA}.contact.lilei", f"{PA}.contact.wang", f"{PB}.contact.su"
    cs = [
        # ---------------- persona A: 林晓
        C("a01", PA, t(10, 1, 9), [("u", "我现在住在杭州滨江区，在一家地图公司做后端开发。"), ("a", "好的，记住了。"), ("u", "我还养了一只叫团子的橘猫。")],
          [Fact("a.home1", "self", "lives_in", "杭州滨江区", (0,), ("住在杭州滨江区",), ("滨江区",)),
           Fact("a.work", "self", "works_at", "地图公司", (0,), ("在一家地图公司",), ("一家地图公司",)),
           Fact("a.job", "self", "job", "后端开发", (0,), ("做后端开发",), ("后端开发工程师",)),
           Fact("a.cat", "self", "pet", "橘猫团子", (2,), ("养了一只叫团子的橘猫",), ("团子", "橘猫", "叫团子的橘猫"))], tags=("multi_fact",)),
        C("a02", PA, t(10, 2, 21), [("u", "我不太喜欢咖啡，喝了晚上睡不着。"), ("a", "那可以试试茶。"), ("u", "我喜欢喝乌龙茶。")],
          [Fact("a.coffee", "self", "dislikes", "咖啡", (0,), ("我不太喜欢咖啡",)), Fact("a.tea", "self", "likes", "乌龙茶", (2,), ("我喜欢喝乌龙茶",), ("喝乌龙茶",))],
          forbidden=(("likes", "咖啡"),), tags=("negation",)),
        C("a03", PA, t(10, 3, 21, 30), [("u", "今晚读了一个小时《百年孤独》。"), ("a", "很不错的书。")],
          [Fact("a.read1", "self", "event", "读《百年孤独》", (0,), ("今晚读了一个小时《百年孤独》",), ("读书", "百年孤独"), "EVENT", "read_1003", "COMPLETED", "读书", None, "2026-10-03", "晚上")], tags=("habit",)),
        C("a04", PA, t(10, 5, 20), [("u", "我妹妹叫小雨，她在上海读大学。"), ("a", "她读什么专业？"), ("u", "学建筑的。")],
          [Fact("a.sister", "self", "relationship", "妹妹 小雨", (0,), ("我妹妹叫小雨",), ("妹妹小雨", "小雨")),
           Fact("a.sister_school", "person:小雨", "school", "上海的大学", (0,), ("她在上海读大学",), ("上海",))], tags=("entity",)),
        C("a05", PA, t(10, 6, 22), [("u", "今晚又读了半小时书，睡前读书很放松。"), ("a", "坚持得不错。")],
          [Fact("a.read2", "self", "event", "读书", (0,), ("今晚又读了半小时书",), ("读书半小时",), "EVENT", "read_1006", "COMPLETED", "读书", None, "2026-10-06", "晚上")], tags=("habit",)),
        C("a06", PA, t(10, 8, 12), [("u", "我报名了这周六的陶艺课。"), ("a", "听起来很有意思。")],
          [Fact("a.pottery", "self", "event", "陶艺课", (0,), ("我报名了这周六的陶艺课",), ("上陶艺课",), "EVENT", "pottery", "SCHEDULED", "陶艺", "2026-10-10")]),
        C("a07", PA, t(10, 9, 9), [("u", "陶艺课那个不算，我其实没报上名。"), ("a", "那下次再报。")],
          [Fact("a.pottery_retract", "self", "event", "陶艺课", (0,), ("陶艺课那个不算，我其实没报上名",), ("上陶艺课",), "EVENT", "pottery", None, "陶艺", retracts=True, refers="a.pottery", op="RETRACT", relation="CORRECTION")],
          tags=("retraction",)),
        C("a08", PA, t(10, 9, 21, 30), [("u", "今晚读书读到很晚。"), ("a", "读的什么？"), ("u", "还是《百年孤独》，晚上读书特别安静。"), ("u", "对，就是刚才说的今晚读书那会儿。")],
          [Fact("a.read3", "self", "event", "读《百年孤独》", (0, 2), ("今晚读书读到很晚",), ("读书", "百年孤独"), "EVENT", "read_1009", "COMPLETED", "读书", None, "2026-10-09", "晚上")],
          tags=("habit", "same_event_repeated")),
        C("a09", PA, t(10, 10, 20), [("u", "今天跑了5公里。"), ("a", "厉害！"), ("u", "跑完5公里真累。"), ("u", "我说的就是今天那5公里，只跑了这一次。")],
          [Fact("a.run", "self", "event", "跑5公里", (0,), ("今天跑了5公里",), ("跑步",), "EVENT", "run_1010", "COMPLETED", "跑步", None, "2026-10-10")], tags=("same_event_repeated",)),
        C("a10", PA, t(10, 11, 18), [(LI, "明天下午三点开会，材料我会准备好。"), ("u", "好的，我会准时到。")],
          [Fact("a.meeting", "self", "event", "开会", (0, 1), ("明天下午三点开会",), ("下午三点开会",), "EVENT", "meeting_1012", "PLANNED", "开会", "2026-10-12", None, "下午")],
          forbidden=(("job", "材料"),), tags=("sender_recipient",), kind="sms"),
        C("a11", PA, t(10, 11, 19), [(LI, "我下周一要去深圳出差。"), ("u", "一路顺风。")],
          [Fact("a.lilei_trip", "person:李雷", "event", "去深圳出差", (0,), ("我下周一要去深圳出差",), ("深圳出差",), "EVENT", "lilei_shenzhen", "PLANNED", "出差", "2026-10-12")],
          forbidden=(("event", "深圳出差:self"),), tags=("entity", "sender_recipient"), kind="sms"),
        C("a12", PA, t(10, 12, 21, 40), [("u", "今晚读了几页诗集。"), ("a", "哪位诗人的？"), ("u", "顾城的。")],
          [Fact("a.read4", "self", "event", "读诗集", (0,), ("今晚读了几页诗集",), ("读书", "读顾城的诗集"), "EVENT", "read_1012", "COMPLETED", "读书", None, "2026-10-12", "晚上")], tags=("habit",)),
        C("a13", PA, t(10, 14, 9), [("u", "我打算这周六去爬西山。"), ("a", "天气预报说周六晴。")],
          [Fact("a.hike_plan", "self", "event", "爬西山", (0,), ("我打算这周六去爬西山",), ("去爬西山",), "EVENT", "hike_xishan", "PLANNED", "爬山", "2026-10-17")], tags=("plan",)),
        C("a14", PA, t(10, 15, 21), [("u", "今晚读书一个半小时，终于把《百年孤独》读完了。"), ("a", "恭喜！")],
          [Fact("a.read5", "self", "event", "读完《百年孤独》", (0,), ("今晚读书一个半小时",), ("读书", "百年孤独"), "EVENT", "read_1015", "COMPLETED", "读书", None, "2026-10-15", "晚上")], tags=("habit",)),
        C("a15", PA, t(10, 17, 19), [("u", "今天去爬了西山，风景很好。"), ("a", "照片发我看看。")],
          [Fact("a.hike_done", "self", "event", "爬西山", (0,), ("今天去爬了西山",), ("去爬西山",), "EVENT", "hike_xishan", "COMPLETED", "爬山", None, "2026-10-17", refers="a.hike_plan", op="SUPERSEDE", relation="TEMPORAL_CHANGE")],
          tags=("plan_to_done",)),
        C("a16", PA, t(10, 18, 15), [("u", "我还是不喜欢咖啡的味道，太苦了。"), ("a", "那就多喝茶吧。")],
          [Fact("a.coffee2", "self", "dislikes", "咖啡", (0,), ("我还是不喜欢咖啡的味道",), ("咖啡的味道",), op="NOOP", relation="DUPLICATE")], tags=("preference_repeat",)),
        C("a17", PA, t(10, 20, 20), [("u", "我之前说错了，我妹妹小雨在南京读大学，不是上海。"), ("a", "好的，已更正。")],
          [Fact("a.sister_school2", "person:小雨", "school", "南京的大学", (0,), ("我妹妹小雨在南京读大学，不是上海",), ("南京",), corrects=True, refers="a.sister_school", op="SUPERSEDE", relation="CORRECTION"),
           Fact("a.sister2", "self", "relationship", "妹妹 小雨", (0,), ("我妹妹小雨",), ("妹妹小雨", "小雨"), op="NOOP", relation="DUPLICATE")], tags=("correction",)),
        C("a18", PA, t(10, 20, 23), [("u", "最近睡眠很差，经常半夜醒。"), ("a", "可以试试睡前不看手机。")],
          [Fact("a.sleep1", "self", "state", "睡眠:差", (0,), ("最近睡眠很差",), ("睡眠:很差",))], tags=("trend",)),
        C("a19", PA, t(10, 22, 12), [("u", "推荐一部电影吧。"), ("a", "你喜欢科幻片的话，可以看《星际穿越》。"), ("u", "好，我周末看看。")],
          [Fact("a.film", "self", "event", "看《星际穿越》", (1, 2), ("我周末看看",), ("看电影", "星际穿越"), "EVENT", "film_interstellar", "PLANNED", "看电影", "2026-10-24")],
          forbidden=(("likes", "科幻"),), tags=("speaker",)),
        C("a20", PA, t(10, 25, 10), [("u", "我上周搬家了，从滨江区搬到了西湖区。"), ("a", "新家住得习惯吗？")],
          [Fact("a.home2", "self", "lives_in", "西湖区", (0,), ("从滨江区搬到了西湖区",), ("杭州西湖区",), refers="a.home1", op="SUPERSEDE", relation="TEMPORAL_CHANGE")], tags=("temporal_change",)),
        C("a21", PA, t(10, 26, 9), [("u", "我可能下个月去北京出差，还不确定。"), ("a", "定了告诉我。")], [], forbidden=(("event", "北京"),), tags=("uncertainty",)),
        C("a22", PA, t(10, 27, 23), [("u", "这周睡眠一般，比上周好一点。"), ("a", "继续保持。")],
          [Fact("a.sleep2", "self", "state", "睡眠:一般", (0,), ("这周睡眠一般",), (), refers="a.sleep1", op="SUPERSEDE", relation="TEMPORAL_CHANGE")], tags=("trend",)),
        C("a23", PA, t(10, 28, 10), [("u", "我没有养狗，家里只有团子一只猫。"), ("a", "团子一定很受宠。")],
          [Fact("a.cat2", "self", "pet", "橘猫团子", (0,), ("家里只有团子一只猫",), ("团子", "猫", "团子一只猫"), op="NOOP", relation="DUPLICATE")], forbidden=(("pet", "狗"),), tags=("negation",)),
        C("a24", PA, t(11, 1, 8), [("u", "我现在已经改成早上读书了，晚上太累。"), ("a", "早上头脑更清醒。")],
          [Fact("a.routine", "self", "routine", "早上读书", (0,), ("我现在已经改成早上读书了",), ("早上读书了",))], tags=("habit_change",)),
        C("a25", PA, t(11, 2, 10), [("u", "我打算11月10日去日本旅行。"), ("a", "需要我提醒你办签证吗？"), ("u", "要的。")],
          [Fact("a.japan", "self", "event", "去日本旅行", (0,), ("我打算11月10日去日本旅行",), ("日本旅行",), "EVENT", "trip_japan", "PLANNED", "旅行", "2026-11-10")], tags=("plan",)),
        C("a26", PA, t(11, 2, 7, 30), [("u", "今天早上读了半小时书。"), ("a", "早起读书真好。")],
          [Fact("a.mread1", "self", "event", "读书", (0,), ("今天早上读了半小时书",), ("读书半小时",), "EVENT", "mread_1102", "COMPLETED", "读书", None, "2026-11-02", "早上")], tags=("habit",)),
        C("a27", PA, t(11, 3, 9), [("u", "11月5日晚上部门聚餐，我订好了位子。"), ("a", "好的。")],
          [Fact("a.dinner", "self", "event", "部门聚餐", (0,), ("11月5日晚上部门聚餐，我订好了位子",), ("聚餐",), "EVENT", "dept_dinner", "SCHEDULED", "聚餐", "2026-11-05", None, "晚上")], tags=("late_arriving",)),
        C("a28", PA, t(11, 3, 23), [("u", "这几天睡眠好多了，基本能睡整夜。"), ("a", "太好了。")],
          [Fact("a.sleep3", "self", "state", "睡眠:好", (0,), ("这几天睡眠好多了",), ("睡眠:好多了",), refers="a.sleep2", op="SUPERSEDE", relation="TEMPORAL_CHANGE")], tags=("trend",)),
        C("a29", PA, t(11, 4, 10), [("u", "日本旅行改到11月12日了。"), ("a", "已更新提醒。")],
          [Fact("a.japan2", "self", "event", "去日本旅行", (0,), ("日本旅行改到11月12日了",), ("日本旅行",), "EVENT", "trip_japan", "PLANNED", "旅行", "2026-11-12", refers="a.japan", op="SUPERSEDE", relation="TEMPORAL_CHANGE")],
          tags=("plan_change",)),
        C("a30", PA, t(11, 4, 7, 20), [("u", "今天早上又读了一会儿书。"), ("a", "坚持就是胜利。")],
          [Fact("a.mread2", "self", "event", "读书", (0,), ("今天早上又读了一会儿书",), (), "EVENT", "mread_1104", "COMPLETED", "读书", None, "2026-11-04", "早上")], tags=("habit",)),
        C("a31", PA, t(11, 5, 10), [("u", "日本的旅行取消了，签证没下来。"), ("a", "太可惜了。")],
          [Fact("a.japan3", "self", "event", "去日本旅行", (0,), ("日本的旅行取消了",), ("日本旅行",), "EVENT", "trip_japan", "CANCELLED", "旅行", refers="a.japan", op="SUPERSEDE", relation="TEMPORAL_CHANGE")],
          tags=("plan_cancel",)),
        C("a32", PA, t(11, 5, 18), [(WANG, "今晚的部门聚餐取消了，改天再约。", t(11, 5, 18), t(11, 8, 9))],
          [Fact("a.dinner2", "self", "event", "部门聚餐", (0,), ("今晚的部门聚餐取消了",), ("聚餐",), "EVENT", "dept_dinner", "CANCELLED", "聚餐", refers="a.dinner", op="SUPERSEDE", relation="TEMPORAL_CHANGE")],
          tags=("late_arriving", "sender_recipient"), kind="sms"),
        C("a33", PA, t(11, 6, 7, 40), [("u", "今天早上读书四十分钟，状态很好。"), ("a", "好习惯。")],
          [Fact("a.mread3", "self", "event", "读书", (0,), ("今天早上读书四十分钟",), ("读书四十分钟",), "EVENT", "mread_1106", "COMPLETED", "读书", None, "2026-11-06", "早上")], tags=("habit",)),
        C("a34", PA, t(11, 7, 20), [("u", "周末和妹妹小雨视频了，她在南京挺好的。"), ("a", "姐妹感情真好。")],
          [Fact("a.sister3", "self", "relationship", "妹妹 小雨", (0,), ("和妹妹小雨视频了",), ("妹妹小雨", "小雨"), op="NOOP", relation="DUPLICATE")], tags=("relationship",)),
        C("a35", PA, t(11, 7, 21), [("u", "今天天气不错。"), ("a", "是啊，适合散步。")], [], tags=("chit_chat",)),
        # ---------------- persona B: 陈默 (same city and similar wording as A, independent facts)
        C("b01", PB, t(10, 1, 9), [("u", "我住在杭州西湖区，在一所中学当语文老师。"), ("a", "老师辛苦了。")],
          [Fact("b.home", "self", "lives_in", "杭州西湖区", (0,), ("我住在杭州西湖区",), ("西湖区",)),
           Fact("b.job", "self", "job", "语文老师", (0,), ("当语文老师",), ("中学语文老师",)),
           Fact("b.work", "self", "works_at", "一所中学", (0,), ("在一所中学",), ("中学",))], tags=("persona_isolation",)),
        C("b02", PB, t(10, 2, 8), [("u", "我每天早上都要喝一杯咖啡，很喜欢咖啡。"), ("a", "手冲还是速溶？")],
          [Fact("b.coffee", "self", "likes", "咖啡", (0,), ("很喜欢咖啡",), ())], forbidden=(("dislikes", "咖啡"),), tags=("persona_isolation",)),
        C("b03", PB, t(10, 5, 19), [("u", "今天去游泳了，游了一千米。"), ("a", "好厉害。")],
          [Fact("b.swim1", "self", "event", "游泳一千米", (0,), ("今天去游泳了",), ("游泳",), "EVENT", "swim_1005", "COMPLETED", "游泳", None, "2026-10-05")], tags=("repeated",)),
        C("b04", PB, t(10, 12, 19), [("u", "今天又去游泳了。"), ("a", "坚持锻炼。")],
          [Fact("b.swim2", "self", "event", "游泳", (0,), ("今天又去游泳了",), (), "EVENT", "swim_1012", "COMPLETED", "游泳", None, "2026-10-12")], tags=("repeated",)),
        C("b05", PB, t(10, 19, 19), [("u", "今天游泳游了半小时。"), ("a", "很棒。")],
          [Fact("b.swim3", "self", "event", "游泳半小时", (0,), ("今天游泳游了半小时",), ("游泳",), "EVENT", "swim_1019", "COMPLETED", "游泳", None, "2026-10-19")], tags=("repeated",)),
        C("b06", PB, t(10, 21, 20), [("u", "我记错了，10月12日那次游泳其实没去，那天下雨。"), ("a", "好的，已经记下。")],
          [Fact("b.swim2_retract", "self", "event", "游泳", (0,), ("10月12日那次游泳其实没去",), (), "EVENT", "swim_1012", None, "游泳", None, "2026-10-12", retracts=True, refers="b.swim2", op="RETRACT", relation="CORRECTION")],
          tags=("dependency_invalidation",)),
        C("b07", PB, t(10, 9, 21), [("u", "今晚读了一会儿书。"), ("a", "读的什么？")],
          [Fact("b.read1", "self", "event", "读书", (0,), ("今晚读了一会儿书",), (), "EVENT", "bread_1009", "COMPLETED", "读书", None, "2026-10-09", "晚上")], tags=("no_overgeneralization",)),
        C("b08", PB, t(10, 16, 21), [("u", "今晚又读书了。"), ("a", "好习惯。")],
          [Fact("b.read2", "self", "event", "读书", (0,), ("今晚又读书了",), (), "EVENT", "bread_1016", "COMPLETED", "读书", None, "2026-10-16", "晚上")], tags=("no_overgeneralization",)),
        C("b09", PB, t(10, 20, 18), [(SU, "我下个月要结婚了，记得来喝喜酒！"), ("u", "恭喜恭喜，一定到。")],
          [Fact("b.su_wedding", "person:苏晴", "event", "结婚", (0,), ("我下个月要结婚了",), ("苏晴结婚",), "EVENT", "su_wedding", "PLANNED", "婚礼", "2026-11")],
          forbidden=(("event", "结婚:self"),), tags=("entity", "sender_recipient"), kind="sms"),
        C("b10", PB, t(10, 22, 9), [("u", "我昨天说的那个周末讲座，我报名了。"), ("a", "哪个讲座？"), ("u", "城市历史讲座，在图书馆。")],
          [Fact("b.lecture", "self", "event", "城市历史讲座", (0, 2), ("我报名了",), ("讲座", "图书馆的城市历史讲座"), "EVENT", "history_lecture", "SCHEDULED", "讲座")], tags=("partial_date",)),
        # ---------------- persona C: 周宁, one long round with several facts and a final state
        C("c01", PC, t(10, 10, 20), [("u", "我在一家银行做柜员。"), ("a", "工作忙吗？"), ("u", "挺忙的，不过我下个月要换工作了。"), ("a", "去哪里？"),
                                     ("u", "去一家出版社做编辑。"), ("a", "听起来不错。"), ("u", "对了，我养了两只仓鼠。"), ("u", "还有，下周六原本想去看话剧，后来改成了周日，最后还是没去成，票退了。")],
          [Fact("c.job1", "self", "job", "银行柜员", (0,), ("我在一家银行做柜员",), ("柜员",)),
           Fact("c.work1", "self", "works_at", "一家银行", (0,), ("在一家银行",), ("银行",)),
           Fact("c.newjob", "self", "event", "去出版社做编辑", (2, 4), ("我下个月要换工作了",), ("换工作", "去出版社做编辑"), "EVENT", "job_change", "PLANNED", "换工作", "2026-11"),
           Fact("c.hamster", "self", "pet", "两只仓鼠", (6,), ("我养了两只仓鼠",), ("仓鼠",)),
           Fact("c.play", "self", "event", "看话剧", (7,), ("下周六原本想去看话剧，后来改成了周日，最后还是没去成，票退了",), ("话剧",), "EVENT", "play", "CANCELLED", "看话剧")],
          forbidden=(("event", "话剧:COMPLETED"),), tags=("multi_fact", "plan_change_cancel", "long_round")),
        C("c02", PC, t(10, 12, 9), [("u", "我好像对花粉过敏，不确定。"), ("a", "可以去医院测一下。")], [], forbidden=(("state", "过敏"),), tags=("uncertainty",)),
    ]
    return cs


# ---------------------------------------------------------------- checkpoints and observations

@dataclass
class Checkpoint:
    name: str
    persona: str
    fact: str                   # gold key of the slot (its first fact)
    attribute: str
    as_of: datetime
    known_at: datetime
    expect: Optional[str]       # expected value fragment, None = nothing


def checkpoints() -> List[Checkpoint]:
    return [
        Checkpoint("japan_planned_on_nov3", PA, "a.japan", "event.status", t(11, 3), t(11, 3), "PLANNED"),
        Checkpoint("japan_cancelled_on_nov6", PA, "a.japan", "event.status", t(11, 6), t(11, 6), "CANCELLED"),
        Checkpoint("japan_date_nov12_on_nov4", PA, "a.japan", "event.planned_time", t(11, 4, 12), t(11, 4, 12), "2026-11-12"),
        Checkpoint("japan_date_nov10_on_nov3", PA, "a.japan", "event.planned_time", t(11, 3), t(11, 3), "2026-11-10"),
        Checkpoint("dinner_still_scheduled_known_nov6", PA, "a.dinner", "event.status", t(11, 6), t(11, 6), "SCHEDULED"),
        Checkpoint("dinner_cancelled_known_nov9", PA, "a.dinner", "event.status", t(11, 6), t(11, 9), "CANCELLED"),
        Checkpoint("home_binjiang_in_october", PA, "a.home1", "lives_in", t(10, 10), t(11, 20), "滨江"),
        Checkpoint("home_xihu_after_move", PA, "a.home1", "lives_in", t(10, 26), t(11, 20), "西湖"),
        Checkpoint("sister_school_shanghai_before_correction", PA, "a.sister_school", "school", t(10, 10), t(10, 10), "上海"),
        Checkpoint("sister_school_nanjing_after_correction", PA, "a.sister_school", "school", t(10, 10), t(10, 21), "南京"),
        Checkpoint("pottery_visible_before_retraction", PA, "a.pottery", "event.status", t(10, 8, 18), t(10, 8, 18), "SCHEDULED"),
        Checkpoint("pottery_gone_after_retraction", PA, "a.pottery", "event.status", t(10, 9, 12), t(10, 9, 12), None),
        Checkpoint("hike_planned_on_oct15", PA, "a.hike_plan", "event.status", t(10, 15), t(10, 15), "PLANNED"),
        Checkpoint("hike_done_on_oct18", PA, "a.hike_plan", "event.status", t(10, 18), t(10, 18), "COMPLETED"),
    ]


@dataclass
class ObsExpect:
    name: str
    persona: str
    rule: str                   # rule key prefix, e.g. "HABIT:读书:晚上"
    known_at: datetime
    expect: Optional[str]       # "active" | "invalidated" | "retracted" | None (must not exist as active)
    support: Optional[int] = None


def observation_expectations() -> List[ObsExpect]:
    return [
        ObsExpect("evening_reading_habit_formed", PA, "HABIT:读书:晚上", t(10, 10), "active", 3),
        ObsExpect("evening_reading_habit_grows", PA, "HABIT:读书:晚上", t(10, 16), "active", 5),
        ObsExpect("evening_reading_not_yet_on_oct7", PA, "HABIT:读书:晚上", t(10, 7), None),
        ObsExpect("evening_reading_invalidated_by_routine", PA, "HABIT:读书:晚上", t(11, 2), "invalidated"),
        ObsExpect("morning_reading_not_yet_on_nov5", PA, "HABIT:读书:早上", t(11, 5), None),
        ObsExpect("morning_reading_habit_formed", PA, "HABIT:读书:早上", t(11, 7), "active", 3),
        ObsExpect("coffee_dislike_pattern", PA, "PREFERENCE_PATTERN:dislikes:咖啡", t(10, 19), "active", 2),
        ObsExpect("coffee_dislike_not_from_one_statement", PA, "PREFERENCE_PATTERN:dislikes:咖啡", t(10, 10), None),
        ObsExpect("sister_stable_relationship", PA, "STABLE_RELATIONSHIP:妹妹 小雨", t(11, 8), "active", 3),
        ObsExpect("sleep_improving_trend", PA, "STATE_TREND:睡眠", t(11, 4), "active", 3),
        ObsExpect("run_once_is_not_a_habit", PA, "REPEATED_BEHAVIOR:跑步", t(11, 20), None),
        ObsExpect("swim_repeated_before_retraction", PB, "REPEATED_BEHAVIOR:游泳", t(10, 20), "active", 3),
        ObsExpect("swim_retracted_after_support_retracted", PB, "REPEATED_BEHAVIOR:游泳", t(10, 22), "retracted"),
        ObsExpect("two_readings_are_not_a_habit", PB, "HABIT:读书:晚上", t(11, 20), None),
        ObsExpect("no_observation_from_uncertainty", PC, "STATE_TREND:", t(11, 20), None),
    ]


# ---------------------------------------------------------------- building the raw data

def message_id(conv: Conv, i: int) -> str:
    return f"{conv.persona}.{conv.kind}.{conv.cid}.{i}"


def conversation_id(conv: Conv) -> str:
    return f"{conv.persona}.{conv.kind}.{conv.cid}"


def raw_messages(conv: Conv) -> List[CanonicalMessage]:
    out = []
    for i, m in enumerate(conv.messages):
        speaker = conv.persona if m.who == "u" else BOT[conv.persona] if m.who == "a" else m.who
        loc = f"pos={i}"
        if conv.kind == "sms":
            loc += ";type=" + ("发送" if m.who == "u" else "接收")
        out.append(CanonicalMessage(conversation_id=group_id(conv), message_id=message_id(conv, i),
                                    target_persona_id=conv.persona, speaker_id=speaker, timestamp=m.at, content=m.text,
                                    source_type="synthetic_chat_round" if conv.kind == "chat" else "synthetic_sms", source_id=CHAT if conv.kind == "chat" else SMS,
                                    locator=loc, recorded_at=m.recorded_at or m.at))
    return out


def group_id(conv: Conv) -> str:
    """A chat is its own round; SMS go to the thread with that contact (``<persona>.sms.<contact>``), as a phone keeps them."""
    if conv.kind == "chat":
        return conversation_id(conv)
    contact = next(m.who for m in conv.messages if m.who not in ("u", "a"))
    return f"{conv.persona}.sms.{contact.rsplit('.', 1)[-1]}"


def sources() -> List[SourceRef]:
    return [SourceRef(source_id=CHAT, kind=SourceKind.CONVERSATION, origin=DataOrigin.SYNTHETIC, description="M2 synthetic writer dataset: chats."),
            SourceRef(source_id=SMS, kind=SourceKind.CONVERSATION, origin=DataOrigin.SYNTHETIC, description="M2 synthetic writer dataset: SMS.")]


def load(api) -> None:
    for s in sources():
        api.register_source(s)
    for conv in conversations():
        api.ingest_messages(raw_messages(conv))


def order() -> List[Conv]:
    """Conversations in the order their content became known (latest recorded message), as a live system would see them."""
    return sorted(conversations(), key=lambda c: (max(m.recorded_at or m.at for m in c.messages), c.cid))


def manifest() -> dict:
    cs = conversations()
    body = {"version": VERSION, "conversations": [asdict(c) for c in cs], "checkpoints": [asdict(x) for x in checkpoints()], "observations": [asdict(x) for x in observation_expectations()]}
    digest = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return {"version": VERSION, "dataset_sha256": digest, "personas": NAMES, "conversations": len(cs), "messages": sum(len(c.messages) for c in cs),
            "gold_facts": sum(len(c.facts) for c in cs), "gold_facts_committing": sum(f.op != "NOOP" for c in cs for f in c.facts),
            "forbidden": sum(len(c.forbidden) for c in cs), "checkpoints": len(checkpoints()), "observation_expectations": len(observation_expectations()),
            "tags": sorted({tg for c in cs for tg in c.tags}), "origin": "synthetic"}
