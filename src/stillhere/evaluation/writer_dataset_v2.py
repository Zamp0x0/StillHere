"""BUILD-M2.1 / M3 independent public synthetic world (``m21-writer-dataset-2``): four NEW invented personas, written spec-first.

Independent of M2: different people, places, stories and wording; written before any Writer run on it, frozen by its manifest hash.
Same Gold format and interface as ``writer_dataset`` (so ``writer_eval`` scores it with ``ds=writer_dataset_v2``). Covers single and
multi-fact statements, contacts' own plans (speaker / recipient), negation, plan / reschedule / cancel / complete, correction, retraction,
late-arriving evidence (recorded after it happened), preference change, a joint event (one event with two people, not two events), a
repeated statement (duplicate), uncertain statements that must not become facts, and bitemporal checkpoints. Nothing describes a real person.
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
QA, QB, QC, QD = "syn.qa", "syn.qb", "syn.qc", "syn.qd"
NAMES = {QA: "顾川", QB: "沈曼", QC: "唐睿", QD: "许宁"}
BOT = {p: f"{p}.assistant" for p in NAMES}
ZHOU, JIE, LIU, QING = f"{QA}.contact.zhou", f"{QB}.contact.jie", f"{QC}.contact.liu", f"{QD}.contact.qing"
CONTACTS = {ZHOU: "老周", JIE: "阿杰", LIU: "小刘", QING: "许晴"}
SPEAKERS = {**{b: "AI 助手" for b in BOT.values()}, **CONTACTS}
CHAT, SMS = "m21.src.chat", "m21.src.sms"
VERSION = "m21-writer-dataset-2"


def t(month: int, day: int, hour: int = 10, minute: int = 0) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=TZ)


def E(key, subject, obj, src, spans, alts, ekey, status, activity, planned=None, event_time=None, daypart=None, **kw):
    return Fact(key, subject, "event", obj, src, spans, alts, "EVENT", ekey, status, activity, planned, event_time, daypart, **kw)


def conversations() -> List[Conv]:
    return [
        # ---------------- 顾川 (QA)
        C("q01", QA, t(9, 1, 9), [("u", "我住在成都武侯区，在一家游戏公司做前端工程师。"), ("a", "了解。"), ("u", "家里养了一只边牧，叫可乐。")],
          [Fact("qa.home1", "self", "lives_in", "成都武侯区", (0,), ("住在成都武侯区",), ("武侯区",)),
           Fact("qa.work1", "self", "works_at", "一家游戏公司", (0,), ("在一家游戏公司",), ("游戏公司",)),
           Fact("qa.job", "self", "job", "前端工程师", (0,), ("做前端工程师",), ("前端",)),
           Fact("qa.pet", "self", "pet", "边牧可乐", (2,), ("养了一只边牧，叫可乐",), ("可乐", "边牧"))], tags=("multi_fact",)),
        C("q02", QA, t(9, 2, 21), [("u", "我不吃香菜，闻到就难受。"), ("a", "点外卖记得备注。"), ("u", "我挺喜欢周末去爬山的。")],
          [Fact("qa.cilantro", "self", "dislikes", "香菜", (0,), ("我不吃香菜",)), Fact("qa.hiking", "self", "likes", "爬山", (2,), ("喜欢周末去爬山",), ("周末爬山",))],
          forbidden=(("likes", "香菜"),), tags=("negation",)),
        C("q03", QA, t(9, 4, 20), [("u", "我弟弟叫顾林，在重庆上高中。"), ("a", "高几了？"), ("u", "高二。")],
          [Fact("qa.brother", "self", "relationship", "弟弟 顾林", (0,), ("我弟弟叫顾林",), ("弟弟顾林", "顾林")),
           Fact("qa.brother_school", "person:顾林", "school", "重庆的高中", (0,), ("在重庆上高中",), ("重庆",))], tags=("entity",)),
        C("q04", QA, t(9, 6, 19), [("u", "今天去健身房练了一个小时。"), ("a", "坚持就是胜利。")],
          [E("qa.gym1", "self", "健身", (0,), ("今天去健身房练了一个小时",), ("去健身房",), "gym_0906", "COMPLETED", "健身", None, "2026-09-06")], tags=("habit",)),
        C("q05", QA, t(9, 8, 12), [("u", "我打算国庆10月1日去云南玩。"), ("a", "需要帮你看看机票吗？"), ("u", "先不用。")],
          [E("qa.yunnan_plan", "self", "去云南玩", (0,), ("我打算国庆10月1日去云南玩",), ("云南旅行", "去云南"), "yunnan", "PLANNED", "旅行", "2026-10-01")]),
        C("q06", QA, t(9, 9, 18), [(ZHOU, "我下周三要去西安出差，周会你帮我主持一下。"), ("u", "没问题。")],
          [E("qa.zhou_trip", "person:老周", "去西安出差", (0,), ("我下周三要去西安出差",), ("西安出差",), "zhou_xian", "PLANNED", "出差", "2026-09-16")],
          forbidden=(("event", "西安出差:self"),), tags=("sender_recipient",), kind="sms"),
        C("q07", QA, t(9, 10, 19), [("u", "今天又去健身房了，练腿。"), ("a", "明天腿会酸。")],
          [E("qa.gym2", "self", "健身", (0,), ("今天又去健身房了",), ("去健身房",), "gym_0910", "COMPLETED", "健身", None, "2026-09-10")], tags=("habit",)),
        C("q08", QA, t(9, 12, 10), [("u", "我报名了下个月的城市马拉松。"), ("a", "加油！")],
          [E("qa.marathon", "self", "城市马拉松", (0,), ("我报名了下个月的城市马拉松",), ("马拉松",), "marathon", "SCHEDULED", "跑马拉松", "2026-10")]),
        C("q09", QA, t(9, 13, 9), [("u", "马拉松那个其实没报成，名额满了。"), ("a", "下次早点。")],
          [E("qa.marathon_retract", "self", "城市马拉松", (0,), ("马拉松那个其实没报成",), ("马拉松",), "marathon", None, "跑马拉松", retracts=True, refers="qa.marathon", op="RETRACT", relation="CORRECTION")],
          tags=("retraction",)),
        C("q10", QA, t(9, 15, 20), [("u", "我之前说错了，我是在一家教育公司工作，不是游戏公司。"), ("a", "好的，已更正。")],
          [Fact("qa.work2", "self", "works_at", "一家教育公司", (0,), ("我是在一家教育公司工作，不是游戏公司",), ("教育公司",), corrects=True, refers="qa.work1", op="SUPERSEDE", relation="CORRECTION")],
          tags=("correction",)),
        C("q11", QA, t(9, 20, 10), [("u", "云南的行程改到10月3日出发了。"), ("a", "记下了。")],
          [E("qa.yunnan_resched", "self", "去云南玩", (0,), ("云南的行程改到10月3日出发了",), ("云南旅行", "去云南"), "yunnan", "PLANNED", "旅行", "2026-10-03", refers="qa.yunnan_plan", op="SUPERSEDE", relation="TEMPORAL_CHANGE")],
          tags=("reschedule",)),
        C("q12", QA, t(9, 22, 21), [("u", "我没有养猫，家里只有可乐一只狗。"), ("a", "可乐很幸福。")],
          [Fact("qa.pet2", "self", "pet", "可乐", (0,), ("家里只有可乐一只狗",), ("边牧可乐",), op="NOOP", relation="DUPLICATE")], forbidden=(("pet", "猫"),), tags=("negation",)),
        C("q13", QA, t(10, 5, 21), [("u", "云南玩回来了，大理特别美。"), ("a", "照片发我看看。")],
          [E("qa.yunnan_done", "self", "去云南玩", (0,), ("云南玩回来了",), ("云南旅行", "去云南"), "yunnan", "COMPLETED", "旅行", None, None, refers="qa.yunnan_plan", op="SUPERSEDE", relation="TEMPORAL_CHANGE")],
          tags=("plan_completed",)),
        C("q14", QA, t(10, 10, 11), [("u", "我搬家了，现在住在高新区。"), ("a", "离公司近吗？"), ("u", "近多了。")],
          [Fact("qa.home2", "self", "lives_in", "高新区", (0,), ("现在住在高新区",), ("成都高新区",), refers="qa.home1", op="SUPERSEDE", relation="TEMPORAL_CHANGE")], tags=("knowledge_update",)),
        C("q15", QA, t(10, 12, 9), [("u", "下个月可能要去上海培训，还没最后定。"), ("a", "定了告诉我。")], [], forbidden=(("event", "上海"),), tags=("uncertain",)),
        C("q16", QA, t(10, 14, 19), [("u", "今天健身练了胸和背。"), ("a", "练得很全面。")],
          [E("qa.gym3", "self", "健身", (0,), ("今天健身练了胸和背",), ("去健身房",), "gym_1014", "COMPLETED", "健身", None, "2026-10-14")], tags=("habit",)),
        # ---------------- 沈曼 (QB)
        C("r01", QB, t(9, 1, 8), [("u", "我在市第一医院当护士，住在南京鼓楼区。"), ("a", "护士很辛苦。")],
          [Fact("qb.job", "self", "job", "护士", (0,), ("当护士",)), Fact("qb.work", "self", "works_at", "市第一医院", (0,), ("在市第一医院",), ("第一医院",)),
           Fact("qb.home", "self", "lives_in", "南京鼓楼区", (0,), ("住在南京鼓楼区",), ("鼓楼区",))], tags=("multi_fact",)),
        C("r02", QB, t(9, 2, 20), [("u", "我老公叫阿杰，是一名消防员。"), ("a", "向他致敬。")],
          [Fact("qb.husband", "self", "relationship", "老公 阿杰", (0,), ("我老公叫阿杰",), ("老公阿杰", "阿杰", "丈夫 阿杰")),
           Fact("qb.husband_job", "person:阿杰", "job", "消防员", (0,), ("是一名消防员",))], tags=("entity",)),
        C("r03", QB, t(9, 3, 7), [("u", "我每天早上都要喝一杯拿铁。"), ("a", "提神。")],
          [Fact("qb.coffee1", "self", "routine", "每天早上喝一杯拿铁", (0,), ("每天早上都要喝一杯拿铁",), ("喝一杯拿铁", "拿铁"))], tags=("routine",)),
        C("r04", QB, t(9, 5, 22), [("u", "这周值了三个夜班，睡眠很差。"), ("a", "注意休息。")],
          [Fact("qb.sleep1", "self", "state", "睡眠:差", (0,), ("睡眠很差",), ("睡眠:很差",))], tags=("state",)),
        C("r05", QB, t(9, 7, 12), [("u", "我和阿杰打算这周六一起去看画展。"), ("a", "哪个画展？"), ("u", "省美术馆的莫奈展。")],
          [E("qb.exhibit_plan", "self", "看莫奈展", (0, 2), ("我和阿杰打算这周六一起去看画展",), ("看画展", "莫奈展"), "exhibit", "PLANNED", "看画展", "2026-09-12")],
          tags=("joint_event",)),
        C("r06", QB, t(9, 13, 21), [("u", "今天和阿杰看了莫奈展，人很多。"), ("a", "值得。")],
          [E("qb.exhibit_done", "self", "看莫奈展", (0,), ("今天和阿杰看了莫奈展",), ("看画展", "莫奈展"), "exhibit", "COMPLETED", "看画展", None, "2026-09-13", refers="qb.exhibit_plan", op="SUPERSEDE", relation="TEMPORAL_CHANGE")],
          tags=("joint_event", "plan_completed", "reschedule")),
        C("r07", QB, t(9, 16, 21), [("u", "今晚我们科室一起去吃了火锅。", t(9, 16, 21), t(9, 20, 9)), ("a", "热闹。", t(9, 16, 21, 2), t(9, 20, 9))],
          [E("qb.hotpot", "self", "科室聚餐吃火锅", (0,), ("今晚我们科室一起去吃了火锅",), ("吃火锅",), "hotpot", "COMPLETED", "聚餐", None, "2026-09-16")],
          tags=("late_arriving",)),
        C("r08", QB, t(9, 22, 19), [("u", "国庆打算回扬州老家住几天。"), ("a", "好好休息。")],
          [E("qb.home_trip", "self", "回扬州老家", (0,), ("国庆打算回扬州老家住几天",), ("回老家",), "yangzhou", "PLANNED", "回老家")]),
        C("r09", QB, t(9, 28, 20), [(JIE, "国庆我要值班，回不了扬州了。"), ("u", "那我也不回扬州老家了，计划取消，在家陪你。")],
          [E("qb.home_trip_cancel", "self", "回扬州老家", (1,), ("那我也不回扬州老家了，计划取消",), ("回老家",), "yangzhou", "CANCELLED", "回老家", refers="qb.home_trip", op="SUPERSEDE", relation="TEMPORAL_CHANGE")],
          tags=("cancel", "sender_recipient"), kind="sms"),
        C("r10", QB, t(10, 3, 8), [("u", "最近咖啡喝多了胃不舒服，我现在不喜欢喝咖啡了。"), ("a", "胃要紧。"), ("u", "我现在喜欢喝茶。")],
          [Fact("qb.coffee2", "self", "dislikes", "咖啡", (0,), ("我现在不喜欢喝咖啡了",), ("喝咖啡",)),
           Fact("qb.tea", "self", "likes", "茶", (2,), ("我现在喜欢喝茶",), ("喝茶",))], tags=("preference_change",)),
        C("r11", QB, t(10, 6, 21), [("u", "这周没上夜班，睡眠好多了。"), ("a", "太好了。")],
          [Fact("qb.sleep2", "self", "state", "睡眠:好", (0,), ("睡眠好多了",), ("睡眠:好多了",), refers="qb.sleep1", op="SUPERSEDE", relation="TEMPORAL_CHANGE")], tags=("state",)),
        C("r12", QB, t(10, 9, 20), [("u", "听说可能要调去急诊科，还没正式通知。"), ("a", "等通知吧。")], [], forbidden=(("works_at", "急诊"),), tags=("uncertain",)),
        # ---------------- 唐睿 (QC)
        C("s01", QC, t(9, 1, 15), [("u", "我在武汉大学读计算机，今年大三。"), ("a", "学业顺利。")],
          [Fact("qc.school", "self", "school", "武汉大学", (0,), ("我在武汉大学读计算机",), ("武大",))], tags=("entity",)),
        C("s02", QC, t(9, 3, 22), [(LIU, "我周末要回家，宿舍的钥匙放你那。"), ("u", "行。")],
          [E("qc.liu_home", "person:小刘", "回家", (0,), ("我周末要回家",), (), "liu_home", "PLANNED", "回家", "2026-09-05")],
          forbidden=(("event", "回家:self"),), tags=("sender_recipient",), kind="sms"),
        C("s03", QC, t(9, 5, 10), [("u", "我报名了10月20日的英语六级考试。"), ("a", "加油。")],
          [E("qc.cet6", "self", "英语六级考试", (0,), ("我报名了10月20日的英语六级考试",), ("六级考试", "六级"), "cet6", "SCHEDULED", "考试", "2026-10-20")]),
        C("s04", QC, t(9, 10, 18), [("u", "下周二下午有个实习面试，是一家做芯片的公司。"), ("a", "准备好简历。")],
          [E("qc.interview", "self", "实习面试", (0,), ("下周二下午有个实习面试",), ("面试",), "interview", "SCHEDULED", "面试", "2026-09-15", None, "下午")]),
        C("s05", QC, t(9, 14, 19), [("u", "面试改到周四了，对方临时有事。"), ("a", "正好多准备一天。")],
          [E("qc.interview_resched", "self", "实习面试", (0,), ("面试改到周四了",), ("面试",), "interview", "SCHEDULED", "面试", "2026-09-17", refers="qc.interview", op="SUPERSEDE", relation="TEMPORAL_CHANGE")],
          tags=("reschedule",)),
        C("s06", QC, t(9, 17, 21), [("u", "今天面试完了，感觉还行。"), ("a", "等好消息。")],
          [E("qc.interview_done", "self", "实习面试", (0,), ("今天面试完了",), ("面试",), "interview", "COMPLETED", "面试", None, "2026-09-17", refers="qc.interview", op="SUPERSEDE", relation="TEMPORAL_CHANGE")],
          tags=("plan_completed",)),
        C("s07", QC, t(9, 20, 12), [("u", "我不喝奶茶，太甜了。"), ("a", "健康。")],
          [Fact("qc.milktea", "self", "dislikes", "奶茶", (0,), ("我不喝奶茶",))], forbidden=(("likes", "奶茶"),), tags=("negation",)),
        C("s08", QC, t(9, 25, 21), [("u", "下学期可能去新加坡交换，还没确定。"), ("a", "好机会。")], [], forbidden=(("event", "交换"),), tags=("uncertain",)),
        C("s09", QC, t(10, 2, 16), [("u", "我养了一只仓鼠，叫豆豆。"), ("a", "可爱。")],
          [Fact("qc.pet", "self", "pet", "仓鼠豆豆", (0,), ("我养了一只仓鼠，叫豆豆",), ("豆豆", "仓鼠"))]),
        C("s10", QC, t(10, 8, 9), [("u", "我说错了，我读的是软件工程，不是计算机。"), ("a", "已更正。")],
          [Fact("qc.major", "self", "school", "软件工程", (0,), ("我说错了，我读的是软件工程",), ("软件工程专业",), corrects=True, refers="qc.school", op="SUPERSEDE", relation="CORRECTION")],
          tags=("correction",)),
        C("s11", QC, t(10, 20, 18), [("u", "今天考完六级了，听力有点难。"), ("a", "辛苦了。")],
          [E("qc.cet6_done", "self", "六级", (0,), ("今天考完六级了",), ("英语六级考试", "六级考试"), "cet6", "COMPLETED", "考试", None, "2026-10-20", refers="qc.cet6", op="SUPERSEDE", relation="TEMPORAL_CHANGE")],
          tags=("plan_completed",)),
        # ---------------- 许宁 (QD)
        C("t01", QD, t(9, 1, 8), [("u", "我退休前是中学数学老师，现在住在苏州。"), ("a", "苏州好地方。"), ("u", "每天早上在公园打太极拳。")],
          [Fact("qd.job", "self", "job", "退休的中学数学老师", (0,), ("我退休前是中学数学老师",), ("数学老师", "退休教师")),
           Fact("qd.home", "self", "lives_in", "苏州", (0,), ("现在住在苏州",)),
           Fact("qd.taichi", "self", "routine", "早上在公园打太极拳", (2,), ("每天早上在公园打太极拳",), ("打太极拳", "太极"))], tags=("multi_fact",)),
        C("t02", QD, t(9, 2, 10), [("u", "我养了一只鹦鹉，叫小绿，会说你好。"), ("a", "真聪明。")],
          [Fact("qd.pet", "self", "pet", "鹦鹉小绿", (0,), ("我养了一只鹦鹉，叫小绿",), ("小绿", "鹦鹉"))]),
        C("t03", QD, t(9, 4, 19), [("u", "女儿许晴在杭州工作，周末常回来。"), ("a", "真孝顺。")],
          [Fact("qd.daughter", "self", "relationship", "女儿 许晴", (0,), ("女儿许晴",), ("许晴",)),
           Fact("qd.daughter_work", "person:许晴", "works_at", "杭州", (0,), ("在杭州工作",))], tags=("entity",)),
        C("t04", QD, t(9, 8, 9), [("u", "医生说我血压有点偏高。"), ("a", "注意饮食。")],
          [], tags=("unsupported_aspect",)),
        C("t05", QD, t(9, 10, 20), [("u", "我约了9月25日去医院体检。"), ("a", "记得空腹。")],
          [E("qd.checkup", "self", "医院体检", (0,), ("我约了9月25日去医院体检",), ("体检",), "checkup", "SCHEDULED", "体检", "2026-09-25")]),
        C("t06", QD, t(9, 18, 21), [(QING, "爸，我十月中旬带你去北京玩几天。"), ("u", "好，我等你安排。")],
          [E("qd.beijing", "self", "去北京玩", (0, 1), ("我十月中旬带你去北京玩几天",), ("北京旅行", "去北京"), "beijing", "PLANNED", "旅行")],
          tags=("joint_event", "sender_recipient"), kind="sms"),
        C("t07", QD, t(9, 25, 15), [("u", "今天去医院体检了，抽了好几管血。"), ("a", "辛苦了。")],
          [E("qd.checkup_done", "self", "医院体检", (0,), ("今天去医院体检了",), ("体检",), "checkup", "COMPLETED", "体检", None, "2026-09-25", refers="qd.checkup", op="SUPERSEDE", relation="TEMPORAL_CHANGE")],
          tags=("plan_completed",)),
        C("t08", QD, t(9, 26, 9), [("u", "昨天体检的事我跟你说过了吧，抽了好几管血。"), ("a", "说过了。")],
          [E("qd.checkup_dup", "self", "医院体检", (0,), ("昨天体检的事",), ("体检",), "checkup", "COMPLETED", "体检", None, "2026-09-25", op="NOOP", relation="DUPLICATE")],
          tags=("duplicate",)),
        C("t09", QD, t(10, 8, 20), [(QING, "爸，北京那趟取消了，公司临时加班。"), ("u", "没事，工作要紧。")],
          [E("qd.beijing_cancel", "self", "去北京玩", (0,), ("北京那趟取消了",), ("北京旅行", "去北京"), "beijing", "CANCELLED", "旅行", refers="qd.beijing", op="SUPERSEDE", relation="TEMPORAL_CHANGE")],
          tags=("cancel",), kind="sms"),
        C("t10", QD, t(10, 15, 9), [("u", "这个月复查，血压已经正常了。"), ("a", "太好了。")],
          [], tags=("unsupported_aspect",)),
        C("t11", QD, t(10, 16, 10), [("u", "女儿说下个月可能换工作去上海，还说不准。"), ("a", "看她的选择。")], [], forbidden=(("works_at", "上海"),), tags=("uncertain",)),
        # ---------------- noise: no fact about the persona (questions, hypotheticals, other people's facts, small talk)
        C("n01", QA, t(9, 3, 21), [("u", "今天天气怎么样？"), ("a", "晴，二十六度。"), ("u", "那挺适合出门的。"), ("a", "是的。")], [], tags=("noise",)),
        C("n02", QA, t(9, 6, 22), [("u", "你觉得哪部科幻电影最好看？"), ("a", "很多人推荐《星际穿越》。"), ("u", "听说过，还没看。"), ("a", "有空可以看看。")], [], tags=("noise",)),
        C("n03", QA, t(9, 14, 23), [("u", "如果我中了彩票，就去冰岛看极光。"), ("a", "听起来很浪漫。"), ("u", "哈哈，只是想想。"), ("a", "梦想总是要有的。")], [], tags=("noise",)),
        C("n04", QA, t(9, 22, 12), [("u", "我同事小王住在上海浦东，每天通勤两小时。"), ("a", "真辛苦。"), ("u", "他说习惯了。"), ("a", "嗯。")], [], forbidden=(("lives_in", "上海浦东"),), tags=("noise",)),
        C("n05", QA, t(10, 2, 20), [("u", "帮我想个周报的标题吧。"), ("a", "《本周进展与下周计划》怎么样？"), ("u", "太普通了。"), ("a", "那就《冲刺第三周》。")], [], tags=("noise",)),
        C("n06", QA, t(10, 12, 22), [("u", "你会下围棋吗？"), ("a", "我懂规则。"), ("u", "改天教我。"), ("a", "好的。")], [], tags=("noise",)),
        C("n07", QB, t(9, 4, 13), [("u", "1加1为什么等于2？"), ("a", "这是自然数加法的定义。"), ("u", "好吧，太哲学了。"), ("a", "哈哈。")], [], tags=("noise",)),
        C("n08", QB, t(9, 9, 23), [("u", "科室新来的护士小李说她老家在哈尔滨。"), ("a", "东北人很热情。"), ("u", "是啊，她很开朗。"), ("a", "挺好的。")], [], forbidden=(("lives_in", "哈尔滨"),), tags=("noise",)),
        C("n09", QB, t(9, 17, 14), [("u", "要是能放一个月假就好了。"), ("a", "你想去哪里？"), ("u", "没想好，随便说说。"), ("a", "理解。")], [], tags=("noise",)),
        C("n10", QB, t(9, 26, 22), [("u", "给我讲个笑话。"), ("a", "程序员最讨厌的季节是什么？冬天，因为有很多bug……其实是没有bug。"), ("u", "不好笑。"), ("a", "我再努力。")], [], tags=("noise",)),
        C("n11", QB, t(10, 9, 21), [("u", "红烧肉怎么做比较好吃？"), ("a", "先焯水，再炒糖色，小火慢炖一小时。"), ("u", "听着不难。"), ("a", "关键是火候。")], [], tags=("noise",)),
        C("n12", QB, t(10, 14, 13), [("u", "你说人为什么会做梦？"), ("a", "有种说法是大脑在整理记忆。"), ("u", "有意思。"), ("a", "还有很多未解之谜。")], [], tags=("noise",)),
        C("n13", QC, t(9, 5, 23), [("u", "线性代数里特征值到底有什么用？"), ("a", "可以描述变换的伸缩方向。"), ("u", "还是有点抽象。"), ("a", "可以从二维的例子开始。")], [], tags=("noise",)),
        C("n14", QC, t(9, 12, 15), [("u", "室友说他下学期要去德国交换。"), ("a", "挺好的机会。"), ("u", "他德语已经学了两年。"), ("a", "很用心。")], [], forbidden=(("lives_in", "德国"),), tags=("noise",)),
        C("n15", QC, t(9, 21, 22), [("u", "推荐几首适合学习时听的歌。"), ("a", "可以试试轻音乐或者白噪音。"), ("u", "白噪音好像不错。"), ("a", "可以先试一周。")], [], tags=("noise",)),
        C("n16", QC, t(9, 30, 16), [("u", "假如我当初学了医，现在会怎么样？"), ("a", "也许在医院实习吧。"), ("u", "想想就累。"), ("a", "每条路都有辛苦。")], [], tags=("noise",)),
        C("n17", QC, t(10, 10, 23), [("u", "Python里列表和元组有什么区别？"), ("a", "列表可变，元组不可变。"), ("u", "懂了。"), ("a", "不客气。")], [], tags=("noise",)),
        C("n18", QC, t(10, 17, 14), [("u", "今天食堂人好多。"), ("a", "饭点总是这样。"), ("u", "下次错峰去。"), ("a", "好主意。")], [], tags=("noise",)),
        C("n19", QD, t(9, 2, 8), [("u", "今天新闻说要降温了。"), ("a", "记得添衣服。"), ("u", "知道了。"), ("a", "嗯。")], [], tags=("noise",)),
        C("n20", QD, t(9, 11, 9), [("u", "老邻居老李的孙子考上清华了。"), ("a", "真厉害。"), ("u", "老李高兴坏了。"), ("a", "可以理解。")], [], forbidden=(("school", "清华"),), tags=("noise",)),
        C("n21", QD, t(9, 19, 16), [("u", "象棋里马走日是什么意思？"), ("a", "马每步走一个日字形。"), ("u", "原来如此。"), ("a", "多下几盘就熟了。")], [], tags=("noise",)),
        C("n22", QD, t(9, 28, 10), [("u", "要是年轻二十岁，我也想去学跳舞。"), ("a", "现在学也不晚。"), ("u", "算了，随口一说。"), ("a", "好的。")], [], tags=("noise",)),
        C("n23", QD, t(10, 5, 17), [("u", "电视上那个养生节目说喝温水好。"), ("a", "适量饮水是好的。"), ("u", "我就随便看看。"), ("a", "嗯。")], [], tags=("noise",)),
        C("n24", QD, t(10, 13, 9), [("u", "帮我查一下重阳节是哪天。"), ("a", "今年的重阳节是10月25日左右，具体以日历为准。"), ("u", "好，到时候提醒我。"), ("a", "好的。")], [], tags=("noise",)),
    ]


def checkpoints() -> List[Checkpoint]:
    return [
        Checkpoint("qa_home_wuhou_in_sept", QA, "qa.home1", "lives_in", t(9, 20), t(10, 20), "武侯"),
        Checkpoint("qa_home_gaoxin_after_move", QA, "qa.home1", "lives_in", t(10, 15), t(10, 20), "高新"),
        Checkpoint("qa_work_game_before_correction", QA, "qa.work1", "works_at", t(9, 10), t(9, 10), "游戏"),
        Checkpoint("qa_work_edu_after_correction", QA, "qa.work1", "works_at", t(9, 10), t(9, 16), "教育"),
        Checkpoint("qb_trip_planned_known_sept25", QB, "qb.home_trip", "event.status", t(9, 25), t(9, 25), "PLANNED"),
        Checkpoint("qb_trip_cancelled_known_sept29", QB, "qb.home_trip", "event.status", t(9, 29), t(9, 29), "CANCELLED"),
        Checkpoint("qc_interview_on_17_after_reschedule", QC, "qc.interview", "event.planned_time", t(9, 15), t(9, 15), "2026-09-17"),
        Checkpoint("qd_beijing_cancelled_oct9", QD, "qd.beijing", "event.status", t(10, 9), t(10, 9), "CANCELLED"),
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
    return [SourceRef(source_id=CHAT, kind=SourceKind.CONVERSATION, origin=DataOrigin.SYNTHETIC, description="M2.1 independent synthetic dataset: chats."),
            SourceRef(source_id=SMS, kind=SourceKind.CONVERSATION, origin=DataOrigin.SYNTHETIC, description="M2.1 independent synthetic dataset: SMS.")]


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
