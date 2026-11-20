"""M3 DEV questions over the public synthetic M2 Writer world (``writer_dataset``), written from its hand-written gold and checkpoints
(never from Writer output). DEV only: the M2 data was used to build the Writer, so nothing here is an independent confirmation.

Each question: id, persona, ask time, text (first person, as the owner asks), kind, and the deterministic grading: ``all`` (every fragment
must appear), ``any`` (one of them must appear), ``none`` (none may appear), ``answerable`` (False: the right answer is an abstention).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

TZ = timezone(timedelta(hours=8))
PA, PB, PC = "syn.pa", "syn.pb", "syn.pc"


def t(month: int, day: int, hour: int = 20, minute: int = 0) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=TZ)


@dataclass(frozen=True)
class Q:
    qid: str
    persona: str
    ask: datetime
    text: str
    kind: str
    any: Tuple[str, ...] = ()
    all: Tuple[str, ...] = ()
    none: Tuple[str, ...] = ()
    answerable: bool = True


def dev_questions() -> List[Q]:
    return [
        Q("d01", PA, t(10, 2), "我住在哪个区？", "SIMPLE", any=("滨江",)),
        Q("d02", PA, t(10, 2), "我是做什么工作的？", "SIMPLE", any=("后端",)),
        Q("d03", PA, t(10, 2), "我养了什么宠物？", "SIMPLE", any=("团子", "橘猫"), none=("狗",)),
        Q("d04", PA, t(10, 3), "我喜欢喝什么茶？", "SIMPLE", any=("乌龙",)),
        Q("d05", PB, t(10, 3), "我是做什么工作的？", "SIMPLE", any=("语文老师", "老师")),
        Q("d06", PB, t(10, 3), "我喜欢喝咖啡吗？", "SIMPLE", any=("喜欢",), none=("不喜欢",)),
        Q("d07", PC, t(10, 11), "我现在在哪里工作？", "SIMPLE", any=("银行",)),
        Q("d08", PA, t(10, 6), "我妹妹叫什么名字？", "SIMPLE", any=("小雨",)),
        Q("d09", PA, t(10, 28), "我现在住在哪个区？", "KNOWLEDGE_UPDATE", any=("西湖",)),
        Q("d10", PA, t(11, 8), "10月10日的时候我住在哪个区？", "TEMPORAL", any=("滨江",)),
        Q("d11", PA, t(10, 12), "我妹妹在哪个城市读大学？", "TEMPORAL", any=("上海",)),
        Q("d12", PA, t(10, 25), "我妹妹在哪个城市读大学？", "CONFLICT", any=("南京",)),
        Q("d13", PA, t(11, 3), "我去日本旅行定在哪天？", "TEMPORAL", any=("11月10", "11-10", "10日", "10号")),
        Q("d14", PA, t(11, 4, 12), "我去日本旅行定在哪天？", "KNOWLEDGE_UPDATE", any=("11月12", "11-12", "12日", "12号")),
        Q("d15", PA, t(11, 6), "我的日本旅行现在是什么状态？", "KNOWLEDGE_UPDATE", any=("取消",)),
        Q("d16", PA, t(11, 4), "11月5日晚上我有什么安排？", "TEMPORAL", any=("聚餐",)),
        Q("d17", PA, t(11, 6), "11月5日的部门聚餐办成了吗？", "KNOWLEDGE_UPDATE", any=("取消", "没有办", "没办")),
        Q("d18", PA, t(10, 16), "我这周六有什么计划？", "TEMPORAL", any=("西山",)),
        Q("d19", PA, t(10, 18), "我去爬西山了吗？", "TEMPORAL", any=("爬了", "去了", "去过", "已经", "完成")),
        Q("d20", PA, t(10, 9, 12), "我报名陶艺课了吗？", "CONFLICT", any=("没", "未", "不算")),
        Q("d21", PA, t(10, 25), "我妹妹学校的记录更正过吗？现在是哪里？", "CONFLICT", any=("南京",)),
        Q("d22", PB, t(10, 25), "我10月12日去游泳了吗？", "CONFLICT", any=("没", "未")),
        Q("d23", PA, t(11, 4, 12), "日本旅行的日期改过吗？改成了哪天？", "CONFLICT", any=("12",)),
        Q("d24", PA, t(11, 2), "我现在习惯什么时候读书？", "KNOWLEDGE_UPDATE", any=("早上", "早晨")),
        Q("d25", PA, t(10, 2), "我住在哪里、做什么工作？", "MULTI_FACT", all=("滨江", "后端")),
        Q("d26", PB, t(10, 3), "我住在哪里，做什么工作？", "MULTI_FACT", all=("西湖", "老师")),
        Q("d27", PA, t(10, 16), "到现在为止我读完了哪本书？", "MULTI_FACT", any=("百年孤独",)),
        Q("d28", PB, t(10, 25), "10月份我去游泳了几次？", "MULTI_FACT", any=("2次", "两次", "2 次")),
        Q("d29", PA, t(11, 7), "11月以来我早上读书读了几次？", "MULTI_FACT", any=("3次", "三次", "3 次")),
        Q("d30", PC, t(10, 11), "我换工作的计划是什么？", "SIMPLE", any=("出版社", "编辑")),
        Q("d31", PA, t(10, 27, 23, 30), "最近我的睡眠怎么样？", "KNOWLEDGE_UPDATE", any=("一般", "好一点", "好一些")),
        Q("d32", PA, t(11, 4), "我最近的睡眠有改善吗？", "KNOWLEDGE_UPDATE", any=("好", "改善")),
        Q("d33", PA, t(10, 28, 12), "我养狗了吗？", "SIMPLE", any=("没", "没有")),
        Q("d34", PA, t(10, 27), "我下个月去北京出差是哪天？", "UNANSWERABLE", answerable=False),
        Q("d35", PA, t(10, 6), "我妹妹学什么专业？", "SIMPLE", any=("建筑",)),
        Q("d36", PB, t(10, 25), "我结婚的日期定在哪天？", "UNANSWERABLE", answerable=False),
        Q("d37", PC, t(10, 13), "我是不是确定对花粉过敏？", "UNANSWERABLE", answerable=False),
        Q("d38", PA, t(10, 5), "我的血型是什么？", "UNANSWERABLE", answerable=False),
        Q("d39", PB, t(10, 5), "我养了什么宠物？", "PERSONA_ISOLATION", answerable=False),
        Q("d40", PA, t(10, 30), "我已经看过《星际穿越》了吗？", "UNANSWERABLE", answerable=False),
        Q("d41", PC, t(10, 11), "我养了什么宠物？", "SIMPLE", any=("仓鼠",)),
        Q("d42", PC, t(10, 11), "我去看话剧了吗？", "KNOWLEDGE_UPDATE", any=("取消", "没")),
        Q("d43", PB, t(10, 25), "我报名了哪个讲座？", "SIMPLE", any=("城市历史",)),
        Q("d44", PA, t(10, 11, 20), "李雷什么时候去深圳出差？", "TEMPORAL", any=("下周一", "10月12", "12日", "12号", "周一")),
    ]


# ---------------------------------------------------------------------------------------------------------------------------------------
# M3 independent confirmation (NEW_INDEPENDENT_CONFIRMATION) over the M2.1 world ``writer_dataset_v2``: written from that dataset's
# hand-written gold, before any confirmation run, and frozen by hash (``confirm_manifest``). Reflection was repaired on DEV only; nothing
# below was looked at while building it. Personas: 顾川 / 沈曼 / 唐睿 / 许宁.
QA, QB, QC, QD = "syn.qa", "syn.qb", "syn.qc", "syn.qd"
D25 = ("9月25", "09-25", "25日", "25号")


def confirm_questions() -> List[Q]:
    return [
        # 顾川
        Q("c01", QA, t(9, 10), "我住在哪个区？", "SIMPLE", any=("武侯",)),
        Q("c02", QA, t(10, 15), "我现在住在哪里？", "KNOWLEDGE_UPDATE", any=("高新",)),
        Q("c03", QA, t(10, 15), "我搬家之前住在哪个区？", "TEMPORAL", any=("武侯",)),
        Q("c04", QA, t(9, 20), "我在哪家公司工作？", "CONFLICT", any=("教育",)),
        Q("c05", QA, t(9, 20), "我是做什么工作的？", "SIMPLE", any=("前端",)),
        Q("c06", QA, t(9, 25), "我养了什么宠物？", "SIMPLE", any=("可乐", "边牧")),
        Q("c07", QA, t(9, 25), "我不吃什么？", "SIMPLE", any=("香菜",)),
        Q("c08", QA, t(9, 25), "我周末喜欢做什么？", "SIMPLE", any=("爬山",)),
        Q("c09", QA, t(9, 25), "我弟弟叫什么？", "SIMPLE", any=("顾林",)),
        Q("c10", QA, t(9, 25), "我弟弟在哪里上学？", "SIMPLE", any=("重庆",)),
        Q("c11", QA, t(9, 25), "云南的行程定在哪天出发？", "KNOWLEDGE_UPDATE", any=("10月3", "10-03", "3日", "3号")),
        Q("c12", QA, t(9, 10), "我国庆打算去哪里玩？", "SIMPLE", any=("云南",)),
        Q("c13", QA, t(10, 8), "云南去成了吗？", "TEMPORAL", any=("回来", "去了", "完成", "已经", "玩过", "去成")),
        Q("c14", QA, t(9, 20), "我报名的城市马拉松是哪天？", "CONFLICT", any=("没报成", "没有报", "未报", "撤回", "没报上", "不成立", "取消", "没有报名")),
        Q("c15", QA, t(9, 20), "我9月6日做了什么？", "TEMPORAL", any=("健身",)),
        Q("c16", QA, t(10, 15), "我最近一次健身是哪天？", "TEMPORAL", any=("10月14", "10-14", "14日", "14号")),
        Q("c17", QA, t(10, 15), "我9月一共去了几次健身房？", "MULTI", any=("两次", "2次", "2 次", "二次")),
        Q("c18", QA, t(9, 12), "我要去哪里出差？", "SIMPLE", any=("西安",)),
        Q("c19", QA, t(10, 15), "我下个月去上海培训的事定了吗？", "ABSTAIN", answerable=False),
        Q("c20", QA, t(10, 15), "我妹妹叫什么？", "ABSTAIN", answerable=False),
        # 沈曼
        Q("c21", QB, t(9, 10), "我在哪家医院工作？", "SIMPLE", any=("第一医院",)),
        Q("c22", QB, t(9, 10), "我是做什么的？", "SIMPLE", any=("护士",)),
        Q("c23", QB, t(9, 10), "我住在哪里？", "SIMPLE", any=("鼓楼",)),
        Q("c24", QB, t(9, 10), "我老公叫什么？", "SIMPLE", any=("阿杰",)),
        Q("c25", QB, t(9, 10), "我老公是做什么工作的？", "SIMPLE", any=("消防",)),
        Q("c26", QB, t(9, 10), "我早上一般喝什么？", "SIMPLE", any=("拿铁",)),
        Q("c27", QB, t(10, 10), "我现在还喜欢喝咖啡吗？", "KNOWLEDGE_UPDATE", any=("不喜欢", "不喝", "茶")),
        Q("c28", QB, t(10, 10), "我现在喜欢喝什么？", "KNOWLEDGE_UPDATE", any=("茶",)),
        Q("c29", QB, t(9, 10), "我这周六有什么安排？", "TEMPORAL", any=("画展", "莫奈")),
        Q("c30", QB, t(9, 20), "莫奈展去看了吗？", "TEMPORAL", any=("看了", "去了", "已经", "完成", "看过")),
        Q("c31", QB, t(9, 20), "我是哪天看的莫奈展？", "TEMPORAL", any=("9月13", "09-13", "13日", "13号")),
        Q("c32", QB, t(9, 20), "9月16日晚上我做了什么？", "TEMPORAL", any=("火锅",)),
        Q("c33", QB, t(9, 25), "我国庆打算去哪里？", "SIMPLE", any=("扬州",)),
        Q("c34", QB, t(10, 2), "国庆回扬州的计划还在吗？", "KNOWLEDGE_UPDATE", any=("取消", "不回", "去不成", "回不了")),
        Q("c35", QB, t(9, 25), "回扬州老家的计划取消了吗？", "TEMPORAL", any=("计划", "打算", "没有取消", "未取消", "还没"), none=("已取消", "已经取消")),
        Q("c36", QB, t(9, 10), "我最近睡眠怎么样？", "TEMPORAL", any=("差",)),
        Q("c37", QB, t(10, 10), "我最近睡眠怎么样？", "KNOWLEDGE_UPDATE", any=("好多了", "好转", "变好", "睡眠好", "不错", "改善", "好了")),
        Q("c38", QB, t(10, 15), "我要调去急诊科了吗？", "ABSTAIN", answerable=False),
        Q("c39", QB, t(10, 10), "我老公叫什么名字，做什么工作？", "MULTI", all=("阿杰", "消防")),
        Q("c40", QB, t(10, 10), "我女儿叫什么？", "ABSTAIN", answerable=False),
        Q("c41", QB, t(10, 15), "我11月有什么安排？", "ABSTAIN", answerable=False),
        # 唐睿
        Q("c42", QC, t(9, 10), "我在哪所大学读书？", "SIMPLE", any=("武汉大学",)),
        Q("c43", QC, t(10, 15), "我读的是什么专业？", "CONFLICT", any=("软件工程",)),
        Q("c44", QC, t(9, 10), "英语六级考试是哪天？", "TEMPORAL", any=("10月20", "10-20", "20日", "20号")),
        Q("c45", QC, t(10, 25), "六级考完了吗？", "TEMPORAL", any=("考完", "已经", "完成", "考了", "考过")),
        Q("c46", QC, t(9, 12), "实习面试是哪天？", "TEMPORAL", any=("9月15", "09-15", "15日", "15号", "周二", "星期二")),
        Q("c47", QC, t(9, 16), "实习面试改到哪天了？", "KNOWLEDGE_UPDATE", any=("9月17", "09-17", "17日", "17号", "周四", "星期四")),
        Q("c48", QC, t(9, 20), "实习面试结束了吗？", "TEMPORAL", any=("面试完", "完成", "已经", "结束", "面完", "面过")),
        Q("c49", QC, t(9, 20), "我面试的是什么公司？", "SIMPLE", any=("芯片",)),
        Q("c50", QC, t(9, 25), "我不喝什么饮料？", "SIMPLE", any=("奶茶",)),
        Q("c51", QC, t(10, 10), "我养了什么宠物？", "SIMPLE", any=("仓鼠", "豆豆")),
        Q("c52", QC, t(9, 4), "我这周末有什么安排？", "TEMPORAL", any=("回家",)),
        Q("c53", QC, t(10, 15), "我下学期去新加坡交换定了吗？", "ABSTAIN", answerable=False),
        Q("c54", QC, t(10, 15), "我在哪家公司实习？", "ABSTAIN", answerable=False),
        Q("c55", QC, t(9, 20), "9月17日我做了什么？", "TEMPORAL", any=("面试",)),
        Q("c56", QC, t(10, 15), "我的仓鼠叫什么？", "SIMPLE", any=("豆豆",)),
        Q("c57", QC, t(10, 15), "我女朋友叫什么？", "ABSTAIN", answerable=False),
        # 许宁
        Q("c58", QD, t(9, 10), "我住在哪个城市？", "SIMPLE", any=("苏州",)),
        Q("c59", QD, t(9, 10), "我退休前是做什么的？", "SIMPLE", any=("老师",)),
        Q("c60", QD, t(9, 10), "我每天早上做什么？", "SIMPLE", any=("太极",)),
        Q("c61", QD, t(9, 10), "我养的鹦鹉叫什么？", "SIMPLE", any=("小绿",)),
        Q("c62", QD, t(9, 10), "我女儿在哪里工作？", "SIMPLE", any=("杭州",)),
        Q("c63", QD, t(9, 10), "我女儿叫什么？", "SIMPLE", any=("许晴",)),
        Q("c64", QD, t(9, 20), "我约了哪天体检？", "TEMPORAL", any=D25),
        Q("c65", QD, t(9, 30), "体检做了吗？", "TEMPORAL", any=("做了", "去了", "已经", "完成", "体检了", "做过")),
        Q("c66", QD, t(9, 30), "我哪天去的医院体检？", "TEMPORAL", any=D25),
        Q("c67", QD, t(9, 25), "女儿说十月要带我去哪里玩？", "SIMPLE", any=("北京",)),
        Q("c68", QD, t(10, 10), "去北京的计划还在吗？", "KNOWLEDGE_UPDATE", any=("取消", "去不成", "不去")),
        Q("c69", QD, t(10, 1), "去北京的计划取消了吗？", "TEMPORAL", any=("计划", "打算", "没有取消", "未取消", "还在", "还没"), none=("已取消", "已经取消")),
        Q("c70", QD, t(10, 20), "我女儿要去上海工作了吗？", "ABSTAIN", answerable=False),
        Q("c71", QD, t(10, 20), "我儿子叫什么？", "ABSTAIN", answerable=False),
        Q("c72", QD, t(10, 20), "我女儿叫什么，在哪个城市工作？", "MULTI", all=("许晴", "杭州")),
    ]


def confirm_manifest() -> dict:
    import hashlib
    import json
    from dataclasses import asdict
    qs = confirm_questions()
    body = json.dumps([asdict(q) for q in qs], ensure_ascii=False, sort_keys=True, default=str)
    return {"version": "m3-confirm-1", "questions": len(qs), "answerable": sum(q.answerable for q in qs), "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "by_kind": {k: sum(q.kind == k for q in qs) for k in sorted({q.kind for q in qs})}, "world": "writer_dataset_v2 (m21-writer-dataset-2)"}
