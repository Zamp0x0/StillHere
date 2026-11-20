"""Batch 2.1 last-fix confirmation questions (``b21g-confirm-7``) over the ``b21g-dataset-7`` world, written from its gold BEFORE the
multi-valued correction rule and before any run, frozen by hash. Corrected multi-valued slots must give only the new value; additions must
keep both values; a change over time gives the current value; a single-valued correction still works; unknowns are declined."""
from __future__ import annotations

from typing import List

from .m3_questions import Q
from .writer_dataset_v7 import YA, YB, YC, t


def confirm7_questions() -> List[Q]:
    return [
        # 秦朗
        Q("g01", YA, t(9, 8, 20), "我的鹦鹉叫什么？", "CORRECTION_MULTI", any=("球球",), none=("豆豆",)),
        Q("g02", YA, t(9, 12, 20), "我养了哪些宠物？", "ADDITION_MULTI", all=("鹦鹉", "仓鼠")),
        Q("g03", YA, t(9, 12, 20), "我做什么工作？", "SIMPLE", any=("药剂师",)),
        Q("g04", YA, t(9, 12, 20), "我的仓鼠叫什么？", "ABSTAIN_PARTIAL", answerable=False),
        Q("g05", YA, t(9, 12, 20), "我在哪家医院？", "ABSTAIN_PARTIAL", answerable=False),
        # 白露
        Q("g06", YB, t(9, 8, 20), "我表妹叫什么？", "CORRECTION_MULTI", any=("思琪",), none=("思思",)),
        Q("g07", YB, t(9, 12, 20), "我同事都有谁？", "ADDITION_MULTI", all=("阿康", "阿伟")),
        Q("g08", YB, t(9, 15, 20), "我现在养什么宠物？", "CHANGE_OVER_TIME", any=("金毛",)),
        Q("g09", YB, t(9, 15, 20), "我在哪工作？", "SIMPLE", any=("广告",)),
        Q("g10", YB, t(9, 15, 20), "我表妹读几年级？", "ABSTAIN_PARTIAL", answerable=False),
        # 孟川
        Q("g11", YC, t(9, 12, 20), "我喜欢喝什么茶？", "CORRECTION_MULTI", any=("红茶",), none=("绿茶",)),
        Q("g12", YC, t(9, 12, 20), "我现在住哪个区？", "CORRECTION_SINGLE", any=("江夏",), none=("洪山区",)),
        Q("g13", YC, t(9, 12, 20), "我喜欢跑步吗？", "ADDITION_MULTI", any=("喜欢",), none=("不喜欢",)),
        Q("g14", YC, t(9, 12, 20), "我养宠物吗？", "ABSTAIN", answerable=False),
    ]


def confirm7_manifest() -> dict:
    import hashlib
    import json
    from dataclasses import asdict
    qs = confirm7_questions()
    body = json.dumps([asdict(q) for q in qs], ensure_ascii=False, sort_keys=True, default=str)
    return {"version": "b21g-confirm-7", "questions": len(qs), "answerable": sum(q.answerable for q in qs), "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "by_kind": {k: sum(q.kind == k for q in qs) for k in sorted({q.kind for q in qs})}, "world": "writer_dataset_v7 (b21g-dataset-7), real extractor + final writer"}
