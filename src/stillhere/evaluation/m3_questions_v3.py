"""Batch 2 independent QA confirmation over the v3 world (``writer_dataset_v3``), written from its hand-written gold BEFORE any run, frozen by
hash. Nothing here was used to develop Batch 2 retrieval / reflection (DEV = M3 dev questions on the M2 world; regression = M3 confirm-1 on the
v2 world). Includes synonym questions (no wording overlap with the stored fact), wrong-person traps (a contact's plan / a friend's
preference), late-arriving facts that must stay unknown before they were recorded, uncertain statements and never-mentioned facts.
"""
from __future__ import annotations

from typing import List

from .m3_questions import Q, t

RA, RB, RC = "syn.ra", "syn.rb", "syn.rc"
DATE18 = ("10月18", "10-18", "18日", "18号")


def confirm2_questions() -> List[Q]:
    return [
        # 方远
        Q("k01", RA, t(9, 10), "我住在哪个区？", "SIMPLE", any=("雁塔",)),
        Q("k02", RA, t(10, 15), "我现在住在哪里？", "KNOWLEDGE_UPDATE", any=("曲江",)),
        Q("k03", RA, t(10, 15), "搬家前我住在哪？", "TEMPORAL", any=("雁塔",)),
        Q("k04", RA, t(9, 25), "我在哪家公司工作？", "CONFLICT", any=("新能源",)),
        Q("k05", RA, t(9, 10), "我是做什么的？", "SYNONYM", any=("测试",)),
        Q("k06", RA, t(9, 10), "我平时喜欢什么运动？", "SYNONYM", any=("羽毛球",)),
        Q("k07", RA, t(9, 10), "我喜欢喝咖啡吗？", "ATTRIBUTION", any=("不喝", "不喜欢", "不爱")),
        Q("k08", RA, t(9, 10), "我平时喝什么饮品？", "SYNONYM", any=("茶",)),
        Q("k09", RA, t(9, 10), "有什么东西是我不吃的？", "SYNONYM", any=("鸡蛋",)),
        Q("k10", RA, t(9, 8), "我这周六有什么安排？", "TEMPORAL", any=("香港", "演唱会")),
        Q("k11", RA, t(9, 15), "香港去成了吗？", "KNOWLEDGE_UPDATE", any=("取消", "没去", "延期", "去不成")),
        Q("k12", RA, t(9, 20), "我报名的半马是哪天？", "TEMPORAL", any=DATE18),
        Q("k13", RA, t(10, 20), "半马跑完了吗？", "TEMPORAL", any=("跑完", "完成", "已经", "跑了")),
        Q("k14", RA, t(9, 12), "阿凯要搬去哪里？", "ATTRIBUTION", any=("成都",)),
        Q("k15", RA, t(9, 12), "我要搬去成都吗？", "ATTRIBUTION", any=("阿凯",)),
        Q("k16", RA, t(10, 25), "我下个月去北京出差吗？", "ABSTAIN", answerable=False),
        Q("k17", RA, t(9, 10), "我的工作单位和住址分别是什么？", "MULTI", all=("汽车", "雁塔")),
        # 林雪
        Q("k18", RB, t(9, 10), "我是做什么工作的？", "SIMPLE", any=("英语老师", "老师")),
        Q("k19", RB, t(9, 10), "我的狗叫什么？", "SYNONYM", any=("年糕",)),
        Q("k20", RB, t(9, 10), "我住在哪？", "SIMPLE", any=("市南",)),
        Q("k21", RB, t(9, 10), "我能吃辣吗？", "SIMPLE", any=("喜欢", "能", "可以"), none=("不喜欢", "不吃辣", "不能")),
        Q("k22", RB, t(10, 10), "我现在还吃辣吗？", "KNOWLEDGE_UPDATE", any=("不吃", "不再", "不能", "不了")),
        Q("k23", RB, t(9, 10), "我最近睡眠怎么样？", "TEMPORAL", any=("差",)),
        Q("k24", RB, t(10, 1), "我最近睡眠怎么样？", "KNOWLEDGE_UPDATE", any=("好多了", "好转", "变好", "不错", "改善", "好了")),
        Q("k25", RB, t(9, 12), "我约了哪天拔智齿？", "TEMPORAL", any=("9月20", "09-20", "20日", "20号")),
        Q("k26", RB, t(9, 20), "拔智齿改到哪天了？", "KNOWLEDGE_UPDATE", any=("9月24", "09-24", "24日", "24号")),
        Q("k27", RB, t(9, 28), "智齿拔了吗？", "TEMPORAL", any=("拔完", "拔了", "已经", "完成")),
        Q("k28", RB, t(9, 14), "国庆谁要去云南？", "ATTRIBUTION", any=("小梅",)),
        Q("k29", RB, t(9, 14), "我国庆去云南吗？", "ATTRIBUTION", any=("小梅",)),
        Q("k30", RB, t(10, 15), "学校让我带毕业班了吗？", "ABSTAIN", answerable=False),
        Q("k31", RB, t(9, 10), "我的职业和宠物分别是什么？", "MULTI", all=("老师", "年糕")),
        # 周默
        Q("k32", RC, t(9, 10), "我退休前做什么工作？", "SIMPLE", any=("会计",)),
        Q("k33", RC, t(9, 25), "我退休前做什么工作？", "CONFLICT", any=("出纳",)),
        Q("k34", RC, t(9, 10), "我住在哪？", "SIMPLE", any=("拱墅",)),
        Q("k35", RC, t(9, 10), "我每天早上干什么？", "SYNONYM", any=("拍照", "运河", "摄影")),
        Q("k36", RC, t(9, 10), "我儿子叫什么？", "SIMPLE", any=("周斌",)),
        Q("k37", RC, t(9, 10), "我儿子是做什么工作的？", "ATTRIBUTION", any=("设计",)),
        Q("k38", RC, t(9, 10), "我10月5日有什么安排？", "TEMPORAL", any=("黄山", "日出")),
        Q("k39", RC, t(10, 8), "黄山去了吗？", "TEMPORAL", any=("去了", "拍到", "已经", "完成", "日出")),
        Q("k40", RC, t(10, 8), "10月6日发生了什么？", "TEMPORAL", any=("相机", "摔")),
        Q("k41", RC, t(10, 6, 22), "我的相机怎么了？", "ABSTAIN", answerable=False),
        Q("k42", RC, t(9, 13), "我喜欢钓鱼吗？", "ATTRIBUTION", any=("不喜欢", "不爱", "坐不住")),
        Q("k43", RC, t(9, 12), "我儿子十一回杭州吗？", "ATTRIBUTION", any=("不回", "回不了", "加班", "不能")),
        Q("k44", RC, t(10, 20), "我报上社区摄影班了吗？", "CONFLICT", any=("没报上", "没有报上", "没报成", "名额满", "没有")),
        Q("k45", RC, t(9, 20), "我血糖怎么样？", "ABSTAIN", answerable=False),
        Q("k46", RC, t(10, 20), "我女儿叫什么？", "ABSTAIN", answerable=False),
        Q("k47", RC, t(10, 20), "我打算去西藏拍星空吗？", "ABSTAIN", answerable=False),
    ]


def confirm2_manifest() -> dict:
    import hashlib
    import json
    from dataclasses import asdict
    qs = confirm2_questions()
    body = json.dumps([asdict(q) for q in qs], ensure_ascii=False, sort_keys=True, default=str)
    return {"version": "b2-confirm-2", "questions": len(qs), "answerable": sum(q.answerable for q in qs), "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "by_kind": {k: sum(q.kind == k for q in qs) for k in sorted({q.kind for q in qs})}, "world": "writer_dataset_v3 (b2-writer-dataset-3)"}
