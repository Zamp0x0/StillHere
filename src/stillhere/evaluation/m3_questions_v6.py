"""Batch 2.1 final confirmation questions (``b21f-confirm-6``) over the ``b21f-dataset-6`` world, written from its gold BEFORE any writer,
retrieval or reader run on that world and frozen by hash. Covers explicit corrections (residence, relationship, pet), public-holiday dates
(a day of 五一 / 国庆, a whole-holiday plan, 元旦), the undated-plan control (a plan with no time is not a 国庆 plan), a third party's holiday
plan, plus simple facts, a negation and complete / partial unknowns."""
from __future__ import annotations

from typing import List

from .m3_questions import Q
from .writer_dataset_v6 import XA, XB, XC, t


def confirm6_questions() -> List[Q]:
    return [
        # 郑岚
        Q("f01", XA, t(9, 10, 20), "我现在住哪个区？", "CORRECTION", any=("拱墅",), none=("西湖区",)),
        Q("f02", XA, t(9, 10, 20), "我做什么工作？", "SIMPLE", any=("会计",)),
        Q("f03", XA, t(9, 10, 20), "五一第二天我去哪了？", "HOLIDAY_DATE", any=("乌镇",)),
        Q("f04", XA, t(9, 20, 20), "我国庆打算去哪？", "HOLIDAY_PLAN", any=("厦门",)),
        Q("f05", XA, t(9, 20, 20), "我喝咖啡吗？", "NEGATION", any=("不喝", "不")),
        Q("f06", XA, t(9, 20, 20), "我在哪家物流公司？", "ABSTAIN_PARTIAL", answerable=False),       # knows "a logistics company", not its name
        Q("f07", XA, t(9, 20, 20), "五一第一天我去哪了？", "ABSTAIN_DAY", answerable=False),             # only the second day (乌镇) is recorded
        # 魏然
        Q("f08", XB, t(9, 10, 20), "我室友叫什么？", "CORRECTION", any=("小周",), none=("小邹",)),
        Q("f09", XB, t(9, 10, 20), "我在哪个学校？", "SIMPLE", any=("四川大学", "川大")),
        Q("f10", XB, t(9, 25, 20), "我国庆打算干什么？", "ABSTAIN_UNDATED", answerable=False),       # 学吉他 has no time: not a 国庆 plan
        Q("f11", XB, t(9, 25, 20), "国庆最后一天我去重庆面试吗？", "ATTRIBUTION", any=("阿凡",)),
        Q("f12", XB, t(10, 15, 20), "元旦我有什么安排？", "HOLIDAY_PLAN", any=("绵阳",)),
        Q("f13", XB, t(9, 25, 20), "我养宠物吗？", "ABSTAIN", answerable=False),
        Q("f14", XB, t(9, 25, 20), "阿凡哪天去重庆？", "HOLIDAY_DATE", any=("10月7", "10-07", "7日", "7号", "最后一天")),
        # 罗敏
        Q("f15", XC, t(9, 10, 20), "我养的是什么？", "CORRECTION", any=("猫",), none=("狗",)),
        Q("f16", XC, t(9, 10, 20), "我退休前做什么？", "SIMPLE", any=("物理",)),
        Q("f17", XC, t(10, 12, 20), "国庆第三天我去哪了？", "HOLIDAY_DATE", any=("华山",)),
        Q("f18", XC, t(10, 12, 20), "我早上一般干什么？", "SIMPLE", any=("太极",)),
        Q("f19", XC, t(10, 12, 20), "我老伴叫什么？", "ABSTAIN", answerable=False),
        Q("f20", XC, t(10, 12, 20), "我下周六回家吃饭吗？", "ATTRIBUTION", any=("小涛", "儿子")),
        Q("f21", XC, t(10, 12, 20), "华山是怎么上去的？", "SIMPLE_DETAIL", any=("缆车",)),
        Q("f22", XC, t(10, 12, 20), "我的猫叫什么名字？", "ABSTAIN_PARTIAL", answerable=False),
    ]


def confirm6_manifest() -> dict:
    import hashlib
    import json
    from dataclasses import asdict
    qs = confirm6_questions()
    body = json.dumps([asdict(q) for q in qs], ensure_ascii=False, sort_keys=True, default=str)
    return {"version": "b21f-confirm-6", "questions": len(qs), "answerable": sum(q.answerable for q in qs), "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "by_kind": {k: sum(q.kind == k for q in qs) for k in sorted({q.kind for q in qs})}, "world": "writer_dataset_v6 (b21f-dataset-6), real extractor + final writer"}
