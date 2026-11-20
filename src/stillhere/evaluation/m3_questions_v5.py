"""Batch 2.1 post-GPU retest: NEW confirmation questions (``b21p-confirm-5``) over the ``b21p-dataset-5`` world, written from its gold BEFORE
any runtime run on that world and frozen by hash. Each persona gets: known facts, a partial-unknown question (memory holds a related fact but
not the asked detail), a complete-unknown question, a third-party trap, a correction / negation / cancellation question."""
from __future__ import annotations

from typing import List

from .m3_questions import Q
from .writer_dataset_v5 import UA, UB, UC, t


def confirm5_questions() -> List[Q]:
    return [
        # 陆遥 (terse)
        Q("p01", UA, t(9, 10, 20), "我在哪儿工作？", "SIMPLE", any=("外卖站", "站长")),
        Q("p02", UA, t(9, 10, 20), "我吃香菜吗？", "NEGATION", any=("不吃", "不")),
        Q("p03", UA, t(9, 25, 20), "骑行赛还比吗？", "KNOWLEDGE_UPDATE", any=("取消", "不比", "台风")),
        Q("p04", UA, t(9, 25, 20), "我去武汉面试吗？", "ATTRIBUTION", any=("浩子",)),
        Q("p05", UA, t(9, 25, 20), "我的站里有多少个骑手？", "ABSTAIN_PARTIAL", answerable=False),      # knows the job, not the number
        Q("p06", UA, t(9, 25, 20), "我养猫吗？", "ABSTAIN", answerable=False),
        Q("p07", UA, t(9, 12, 20), "骑行赛是哪天？", "TEMPORAL", any=("10月10", "10-10", "10日", "10号")),
        # 许曼 (long-winded)
        Q("p08", UB, t(9, 10, 20), "我是教什么的？", "SIMPLE", any=("语文",)),
        Q("p09", UB, t(9, 10, 20), "朵朵上几年级？", "SIMPLE", any=("二年级", "小学")),
        Q("p10", UB, t(10, 9, 20), "杭州去了吗？", "KNOWLEDGE_UPDATE", any=("去了", "玩完", "已经", "完成")),
        Q("p11", UB, t(9, 10, 20), "我在哪所中学教书？", "ABSTAIN_PARTIAL", answerable=False),         # knows the city / subject, not the school's name
        Q("p12", UB, t(10, 12, 20), "我下学期当教研组长吗？", "ABSTAIN", answerable=False),            # uncertain, not stored
        Q("p13", UB, t(9, 10, 20), "朵朵在哪个小学？", "ABSTAIN_PARTIAL", answerable=False),           # knows the grade, not the school
        Q("p14", UB, t(9, 10, 20), "我住在哪？", "SIMPLE", any=("姑苏",)),
        # 程一 (multi-bubble)
        Q("p15", UC, t(9, 15, 20), "我现在住哪个区？", "CONFLICT", any=("福田",), none=("南山区",)),
        Q("p16", UC, t(9, 15, 20), "我是做什么的？", "SIMPLE", any=("后端",)),
        Q("p17", UC, t(9, 15, 20), "我在哪家公司？", "ABSTAIN_PARTIAL", answerable=False),             # knows "a payment company", not its name
        Q("p18", UC, t(9, 15, 20), "我喜欢什么运动？", "SIMPLE", any=("篮球",)),
        Q("p19", UC, t(9, 20, 20), "我要搬去上海吗？", "ATTRIBUTION", any=("苗苗",)),
        Q("p20", UC, t(9, 20, 20), "我约了什么时候洗牙？", "TEMPORAL", any=("24", "周四", "星期四")),
        Q("p21", UC, t(9, 20, 20), "我喜欢游泳吗？", "ABSTAIN", answerable=False),
        Q("p22", UC, t(9, 20, 20), "苗苗喜欢上海吗？", "ABSTAIN", answerable=False),
    ]


def confirm5_manifest() -> dict:
    import hashlib
    import json
    from dataclasses import asdict
    qs = confirm5_questions()
    body = json.dumps([asdict(q) for q in qs], ensure_ascii=False, sort_keys=True, default=str)
    return {"version": "b21p-confirm-5", "questions": len(qs), "answerable": sum(q.answerable for q in qs), "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "by_kind": {k: sum(q.kind == k for q in qs) for k in sorted({q.kind for q in qs})}, "world": "writer_dataset_v5 (b21p-dataset-5), gold-extractor memory"}
