"""Batch 2.1 independent QA confirmation over the b21 writer world (``writer_dataset_v4``), written from its gold BEFORE any reflection run on
that world and frozen by hash. Covers the Batch 2.1 reflection fixes on unseen questions: relative weekdays / days (这周六, 下周六, 下周三,
今天, 上周六), dated plans after a reschedule, completion / cancellation, counting, attribution traps (a friend's coffee, a contact's trip) and
abstention traps for what the storage policy does not keep (a suggestion, hearsay, an uncertain plan, a guess).
"""
from __future__ import annotations

from typing import List

from .m3_questions import Q
from .writer_dataset_v4 import TA, TB, TC, t

D20 = ("9月20", "09-20", "20日", "20号")


def confirm3_questions() -> List[Q]:
    return [
        # 宋岚
        Q("r01", TA, t(9, 10, 20), "我这周六有什么安排？", "ABSTAIN", answerable=False),                  # 9-12: nothing (the wedding is 9-19)
        Q("r02", TA, t(9, 10, 20), "我下周六有什么安排？", "RELATIVE_WEEKDAY", any=("婚礼", "重庆")),
        Q("r03", TA, t(9, 13, 20), "重庆的婚礼是哪天？", "KNOWLEDGE_UPDATE", any=D20),
        Q("r04", TA, t(9, 22, 20), "婚礼参加了吗？", "TEMPORAL", any=("参加了", "参加完", "已经", "完成", "去了")),
        Q("r05", TA, t(9, 20, 20), "我平时喝咖啡吗？", "ABSTAIN", answerable=False),                     # only 阿雷's coffee is said before 9-25
        Q("r06", TA, t(9, 26, 20), "我喝咖啡吗？", "SIMPLE", any=("不喝", "不喜欢")),
        Q("r07", TA, t(10, 13, 20), "国庆最后一天我干什么了？", "TEMPORAL", any=("峨眉",)),
        Q("r08", TA, t(9, 20, 20), "阿雷什么时候去上海出差？", "ATTRIBUTION", any=("下个月", "10月", "十月")),
        Q("r09", TA, t(9, 20, 20), "我要去上海出差吗？", "ATTRIBUTION", any=("阿雷",)),
        Q("r10", TA, t(9, 15, 20), "我周末去青城山吗？", "ABSTAIN", answerable=False),
        # 韩松
        Q("r11", TB, t(9, 8, 20), "我妈下周三干什么？", "RELATIVE_WEEKDAY", any=("复查", "医院")),
        Q("r12", TB, t(9, 21, 20), "厨艺比赛还比吗？", "KNOWLEDGE_UPDATE", any=("取消", "不比", "没有")),
        Q("r13", TB, t(9, 25, 20), "我9月份游了几次泳？", "MULTI", any=("两次", "2次", "2 次")),
        Q("r14", TB, t(9, 12, 22), "我今天干什么了？", "RELATIVE_WEEKDAY", any=("游泳", "游了")),
        Q("r15", TB, t(9, 14, 20), "上周六我干什么了？", "RELATIVE_WEEKDAY", any=("游泳", "游了")),
        Q("r16", TB, t(10, 3, 20), "我现在喝酒吗？", "KNOWLEDGE_UPDATE", any=("不喝", "戒")),
        Q("r17", TB, t(9, 10, 20), "我小时候吃香菜吗？", "ABSTAIN", answerable=False),
        Q("r18", TB, t(9, 10, 20), "我妹妹在哪读书？", "SIMPLE", any=("北京大学", "北大")),
        # 白露
        Q("r19", TC, t(9, 10, 20), "我9月20日有什么安排？", "TEMPORAL", any=("鼓浪屿",)),
        Q("r20", TC, t(9, 22, 20), "鼓浪屿去成了吗？", "KNOWLEDGE_UPDATE", any=("没去成", "取消", "没有", "没去")),
        Q("r21", TC, t(10, 14, 20), "宁宁考试考完了吗？", "TEMPORAL", any=("考完", "完成", "已经")),
        Q("r22", TC, t(9, 20, 20), "我约了体检吗？", "ABSTAIN", answerable=False),
        Q("r23", TC, t(10, 5, 20), "我的鹦鹉叫什么？", "SIMPLE", any=("豆豆",)),
        Q("r24", TC, t(9, 10, 20), "我喜欢下棋吗？", "SIMPLE", any=("没兴趣", "不喜欢", "不")),
        Q("r25", TC, t(10, 15, 20), "宁宁喜欢画画吗？", "ABSTAIN", answerable=False),
    ]


def confirm3_manifest() -> dict:
    import hashlib
    import json
    from dataclasses import asdict
    qs = confirm3_questions()
    body = json.dumps([asdict(q) for q in qs], ensure_ascii=False, sort_keys=True, default=str)
    return {"version": "b21-confirm-3", "questions": len(qs), "answerable": sum(q.answerable for q in qs), "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "by_kind": {k: sum(q.kind == k for q in qs) for k in sorted({q.kind for q in qs})}, "world": "writer_dataset_v4 (b21-writer-dataset-4), written by M2.2.1"}
