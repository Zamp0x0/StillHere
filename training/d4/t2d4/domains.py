"""Slot domains of the synthetic memory world: how a slot is stored as evidence, asked about and answered.

Frames are content templates only: every value in an answer frame comes from evidence (``{v}``, ``{old}``, ``{n}``) or the fixed slot words;
style (particles, punctuation, openers, attitudes, bubbles) is added later by the persona realiser. Question frames are split into TRAIN
frames and EVAL frames (the last third of each list, at least one); EVAL frames never appear in TRAIN (template-family isolation).
``HOLDOUT_ONLY_DOMAINS`` never appear outside FROZEN_HOLDOUT (unseen-domain generalisation).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .lexicon import BANKS


def _bank(name: str) -> Callable[[str, random.Random], str]:
    return lambda pool, rnd: rnd.choice(BANKS[name][pool])


def _price(pool: str, rnd: random.Random) -> str:
    return f"{rnd.choice([rnd.randint(19, 99), rnd.randint(100, 999), rnd.randint(1000, 4999)])}块"


def _date(pool: str, rnd: random.Random) -> str:
    return f"{rnd.randint(1, 12)}月{rnd.randint(1, 28)}号"


def _daypart_clock(pool: str, rnd: random.Random) -> str:
    part = rnd.choice(["上午", "下午", "晚上"])
    ok = {"上午": ("七点", "七点半", "八点", "八点半", "九点", "九点半", "十点", "十点半"), "下午": ("两点", "两点半", "三点", "四点", "五点半"),
          "晚上": ("六点", "七点", "七点半", "八点", "八点半")}[part]
    return part + rnd.choice([x for x in BANKS["clock"][pool] if x in ok] or list(ok))


def _when(pool: str, rnd: random.Random) -> str:
    return rnd.choice(BANKS["weekday"][pool]) + _daypart_clock(pool, rnd)


def _phone(pool: str, rnd: random.Random) -> str:
    return "1" + rnd.choice("3456789") + "".join(rnd.choice("0123456789") for _ in range(9))


def _door(pool: str, rnd: random.Random) -> str:
    return f"{rnd.choice(['锦华', '翠苑', '和平', '新桥', '长青', '阳光', '金色', '春晓', '幸福', '景园'])}小区{rnd.randint(1, 30)}栋{rnd.randint(1, 28)}0{rnd.randint(1, 6)}"


def _idno(pool: str, rnd: random.Random) -> str:
    return "".join(rnd.choice("0123456789") for _ in range(17)) + rnd.choice("0123456789X")


def _salary(pool: str, rnd: random.Random) -> str:
    return f"一个月{rnd.randint(4, 40)}k"


@dataclass
class Domain:
    id: str
    noun: str                       # unknown label / slot noun
    kind: str                       # fact | plan | number | protected | multi
    value: Callable[[str, random.Random], str]
    ev: Dict[str, str]              # evidence text, keys self / third ({n} third-party label, {v} value)
    q: Dict[str, List[str]]         # question frames
    a: Dict[str, List[str]]         # answer core frames
    unknown: Dict[str, List[str]] = field(default_factory=dict)
    change: Optional[Dict[str, List[str]]] = None   # current vs past ({v} new, {old})
    q_still: Optional[Dict[str, List[str]]] = None  # "still {old}?"
    q_past: Optional[Dict[str, List[str]]] = None
    a_past: Optional[Dict[str, List[str]]] = None
    upcoming: Optional[Dict[str, List[str]]] = None # ({old} current, {v} future, {when})
    cancel: Optional[Dict[str, List[str]]] = None
    who: Optional[List[str]] = None                 # "who …?" questions (third party attribution)
    self_ask_third: Optional[List[str]] = None      # partner asks "you …?" when it is a third party's fact ({v})
    decline: Optional[Dict[str, List[str]]] = None  # protected / policy-uncertain refusal frames
    conflict: bool = False
    correction: bool = False
    subjects: tuple = ("self", "third")
    elliptic: Optional[List[str]] = None            # follow-up question after a conversation turn names the topic
    lead: Optional[Dict[str, List[str]]] = None     # conversation lead-in turns (self speaker) for conversation dependence
    lead_unknown: Optional[List[str]] = None        # unknown replies that do not contradict the lead ("约了复查" -> "几点我忘了")
    lead_unknown_label: Optional[str] = None        # unknown-slot label after the lead (the detail, not the whole event)

    def frames(self, which: str, subj: str, split_kind: str) -> List[tuple]:
        """(frame_id, text) of question frames for TRAIN or EVAL."""
        src = {"q": self.q, "q_still": self.q_still or {}, "q_past": self.q_past or {}}.get(which, {})
        lst = list(enumerate(src.get(subj, [])))
        if not lst:
            return []
        n_eval = max(1, len(lst) // 3)
        return lst[:-n_eval] if split_kind == "TRAIN" else lst[-n_eval:]


DOMAINS: Dict[str, Domain] = {}


def _add(d: Domain) -> None:
    DOMAINS[d.id] = d


_add(Domain("home_city", "现在住的城市", "fact", _bank("city"),
            ev={"self": "本人住在{v}", "third": "{n}住在{v}"},
            q={"self": ["你现在住哪？", "你住在哪个城市来着？", "你家在哪边？", "你现在在哪生活？", "你人在哪个城市呢？", "你目前定居在哪？"],
               "third": ["{n}现在住哪？", "{n}在哪个城市？", "{n}家在哪边来着？"]},
            a={"self": ["在{v}", "住{v}", "{v}", "现在在{v}"], "third": ["{n}在{v}", "{n}住{v}", "{n}现在在{v}"]},
            unknown={"third": ["{n}住哪我也不清楚", "这个我还真不知道", "没听{n}说过"]},
            change={"self": ["早搬到{v}了", "现在住{v}了，不在{old}了", "搬去{v}了", "已经在{v}了"],
                    "third": ["{n}早搬到{v}了", "{n}现在在{v}，不在{old}了", "{n}搬去{v}了"]},
            q_still={"self": ["你现在还住{old}吗？", "还在{old}呢？", "你是不是还在{old}？", "{old}那边还住着吗？"],
                     "third": ["{n}还在{old}吗？", "{n}是不是还住{old}？"]},
            q_past={"self": ["你以前住哪来着？", "你之前在哪个城市？", "搬家之前你住哪？"], "third": ["{n}以前住哪？", "{n}之前在哪个城市？"]},
            a_past={"self": ["以前在{v}", "之前住{v}", "原来在{v}"], "third": ["{n}以前在{v}", "{n}之前住{v}"]},
            upcoming={"self": ["现在还在{old}，{when}搬去{v}", "{when}才搬{v}，现在还在{old}"], "third": ["{n}现在还在{old}，{when}搬去{v}"]},
            who=["谁住在{v}来着？", "我们这帮人里谁在{v}？", "谁家在{v}？"],
            self_ask_third=["你搬去{v}了？", "听说你在{v}？", "你是不是要去{v}住了？"],
            conflict=True, correction=True))

_add(Domain("occupation", "工作", "fact", _bank("occupation"),
            ev={"self": "本人的职业是{v}", "third": "{n}的职业是{v}"},
            q={"self": ["你是做什么工作的？", "你现在干啥的？", "你干哪一行？", "你做什么的来着？", "你的工作是？", "你现在在做什么职业？"],
               "third": ["{n}是做什么的？", "{n}干哪行的？", "{n}现在做什么工作？"]},
            a={"self": ["做{v}", "我是{v}", "{v}", "在当{v}"], "third": ["{n}是{v}", "{n}在做{v}", "{n}当{v}"]},
            unknown={"third": ["{n}做什么我还真不清楚", "没问过{n}", "{n}的工作我不太了解"]},
            change={"self": ["现在做{v}了，不干{old}了", "改行当{v}了", "早不做{old}了，现在是{v}"],
                    "third": ["{n}改行做{v}了", "{n}现在是{v}，不做{old}了"]},
            q_still={"self": ["你还在做{old}吗？", "还当{old}呢？", "现在还干{old}？"], "third": ["{n}还做{old}吗？", "{n}还在当{old}？"]},
            q_past={"self": ["你以前是做什么的？", "你之前干哪行？", "转行前你做啥？"], "third": ["{n}以前做什么的？"]},
            a_past={"self": ["以前是{v}", "之前做{v}"], "third": ["{n}以前是{v}"]},
            self_ask_third=["你是{v}？", "听说你在做{v}？"], conflict=True, correction=True))

_add(Domain("workplace", "上班的公司", "fact", _bank("company"),
            ev={"self": "本人在{v}上班", "third": "{n}在{v}上班"},
            q={"self": ["你在哪上班？", "你在哪家公司？", "你现在在哪个单位？", "你公司叫啥来着？", "你在哪儿工作？", "你上班的地方是？"],
               "third": ["{n}在哪上班？", "{n}在哪家公司？", "{n}的单位是哪个？"]},
            a={"self": ["在{v}", "{v}", "在{v}上班"], "third": ["{n}在{v}", "{n}在{v}上班"]},
            unknown={"third": ["{n}在哪上班我不清楚", "这个没听{n}说过"]},
            change={"self": ["早跳槽到{v}了", "现在在{v}，不在{old}了", "换到{v}了"], "third": ["{n}跳槽到{v}了", "{n}现在在{v}了"]},
            q_still={"self": ["你还在{old}吗？", "还在{old}上班？"], "third": ["{n}还在{old}吗？"]},
            q_past={"self": ["你之前在哪家公司？", "你上一家是哪？"], "third": ["{n}之前在哪上班？"]},
            a_past={"self": ["之前在{v}", "上一家是{v}"], "third": ["{n}之前在{v}"]},
            upcoming={"self": ["现在还在{old}，{when}去{v}", "{when}才入职{v}"], "third": ["{n}{when}去{v}，现在还在{old}"]},
            conflict=True, correction=True))

_add(Domain("teach_subject", "教的科目", "fact", _bank("subject_taught"),
            ev={"self": "本人是老师，教{v}", "third": "{n}是老师，教{v}"},
            q={"self": ["你教什么的？", "你是教哪门课的？", "你具体教什么科目？", "你带哪科？", "你是什么老师？", "你教的哪一科？"],
               "third": ["{n}教什么？", "{n}是什么老师？", "{n}带哪科？"]},
            a={"self": ["教{v}", "{v}", "教{v}的", "我带{v}"], "third": ["{n}教{v}", "{n}是{v}老师"]},
            change={"self": ["现在改教{v}了，不教{old}了", "早改教{v}了"], "third": ["{n}改教{v}了"]},
            q_still={"self": ["你还教{old}吗？", "还在教{old}？"], "third": ["{n}还教{old}？"]},
            correction=True, conflict=True))

_add(Domain("hobby", "爱好", "multi", _bank("hobby"),
            ev={"self": "本人平时喜欢{v}", "third": "{n}平时喜欢{v}"},
            q={"self": ["你平时喜欢干嘛？", "你有啥爱好？", "周末一般玩什么？", "你业余时间做什么？", "你喜欢什么运动或者活动？", "平时怎么放松？"],
               "third": ["{n}平时喜欢干嘛？", "{n}有什么爱好？", "{n}周末一般干啥？"]},
            a={"self": ["喜欢{v}", "平时{v}", "{v}吧"], "third": ["{n}喜欢{v}", "{n}平时{v}"]},
            unknown={"third": ["{n}的爱好我不太清楚", "没聊过这个"]},
            q_past={"self": ["你以前喜欢玩什么？"], "third": ["{n}以前喜欢啥？"]},
            a_past={"self": ["以前挺喜欢{v}", "之前常{v}"], "third": ["{n}以前喜欢{v}"]},
            who=["谁喜欢{v}来着？"]))

_add(Domain("pet_name", "宠物名字", "fact", _bank("pet_name"),
            ev={"self": "本人养了一只{k}，叫{v}", "third": "{n}养了一只{k}，叫{v}"},
            q={"self": ["你家{k}叫什么？", "你那只{k}叫啥来着？", "你{k}的名字是？", "你养的{k}叫什么名字？", "你家{k}叫啥？"],
               "third": ["{n}家的{k}叫什么？", "{n}那只{k}叫啥？"]},
            a={"self": ["叫{v}", "{v}", "它叫{v}"], "third": ["{n}家那只叫{v}", "叫{v}"]},
            unknown={"third": ["{n}家{k}叫啥我忘了", "这个我还真不知道"]},
            who=["谁家养了{k}？"], correction=True))

_add(Domain("trip_city", "出差/旅行去的城市", "plan", _bank("city"),
            ev={"self": "本人{w}要去{v}出差", "third": "{n}{w}要去{v}出差"},
            q={"self": ["你{w}去哪出差？", "{w}去哪来着？", "你{w}的出差是去哪？", "{w}要去哪个城市？", "你{w}出差去哪儿？", "{w}要跑哪？"],
               "third": ["{n}{w}去哪出差？", "{n}{w}去哪？"]},
            a={"self": ["去{v}", "{v}", "去{v}出差"], "third": ["{n}去{v}", "{n}{w}去{v}"]},
            unknown={"self": ["去哪还没定", "地方还没定", "还不知道去哪"], "third": ["{n}去哪我不清楚", "没听{n}说去哪"]},
            cancel={"self": ["本来要去{v}，后来取消了", "{v}那趟取消了", "不去{v}了，取消了"], "third": ["{n}本来要去{v}，后来取消了"]},
            who=["{w}谁要去{v}？", "谁{w}出差？"], self_ask_third=["你{w}要去{v}？", "你是不是要去{v}出差？"],
            correction=True,
            lead={"self": ["我{w}要去{v}出差"]}))

_add(Domain("hotel", "住的酒店", "plan", _bank("hotel"),
            ev={"self": "本人{w}出差订了{v}", "third": "{n}{w}出差订了{v}"},
            q={"self": ["你住哪个酒店？", "酒店订的哪家？", "你{w}住哪？", "住的地方定了吗？", "订的是哪家酒店来着？", "你这次住哪家？"],
               "third": ["{n}住哪个酒店？", "{n}订的哪家？"]},
            a={"self": ["住{v}", "订了{v}", "{v}"], "third": ["{n}住{v}", "{n}订的{v}"]},
            unknown={"self": ["酒店还没定", "住哪还没订", "还没订呢", "住的地方还没着落"], "third": ["{n}住哪我不知道", "这个没听{n}说"]},
            cancel={"self": ["之前订过{v}，后来退了", "本来订的{v}，取消了", "{v}那个已经取消了"], "third": ["{n}本来订的{v}，后来退了"]},
            self_ask_third=["你订了{v}？", "你住{v}是吧？"],
            elliptic=["那酒店呢？", "酒店定了没？", "住哪定了吗？", "住的呢？"],
            correction=True))

_add(Domain("trip_date", "出发日期", "number", _date,
            ev={"self": "本人出差定在{v}出发", "third": "{n}出差定在{v}出发"},
            q={"self": ["你哪天出发？", "几号走？", "出发定的哪天？", "你什么时候走来着？", "你哪天动身？"],
               "third": ["{n}哪天走？", "{n}几号出发？"]},
            a={"self": ["{v}走", "{v}", "{v}出发"], "third": ["{n}{v}走", "{n}{v}出发"]},
            unknown={"self": ["哪天走还没定", "日子还没定", "具体哪天还不知道"], "third": ["{n}哪天走我不清楚"]},
            elliptic=["那哪天走？", "几号出发？", "什么时候走？"], correction=True, conflict=True))

_add(Domain("purchase_price", "买的价格", "number", _price,
            ev={"self": "本人买了{it}，花了{v}", "third": "{n}买了{it}，花了{v}"},
            q={"self": ["你那个{it}多少钱买的？", "{it}花了多少？", "你买{it}花了多少钱来着？", "{it}多少钱？", "那个{it}贵不贵，多少钱？"],
               "third": ["{n}的{it}多少钱？", "{n}买{it}花了多少？"]},
            a={"self": ["{v}", "花了{v}", "{v}买的"], "third": ["{n}花了{v}", "{v}"]},
            unknown={"self": ["多少钱忘了", "价格记不清了", "多少钱我没记住"], "third": ["{n}花多少我不知道", "这个没问{n}"]},
            elliptic=["多少钱？", "花了多少？", "贵吗，多少钱？"], correction=True,
            lead={"self": ["我前几天买了{it}"]}, lead_unknown=["多少钱忘了", "价格记不清了", "多少钱我没记住"], lead_unknown_label="买的价格"))

_add(Domain("concert_plan", "演出安排", "plan", _when,
            ev={"self": "本人买了{c}的票，时间是{v}", "third": "{n}买了{c}的票，时间是{v}"},
            q={"self": ["你那个{c}是什么时候？", "{c}哪天来着？", "你看{c}是几点？", "{c}定的哪天？", "你那场{c}啥时候？"],
               "third": ["{n}的{c}什么时候？", "{n}看{c}是哪天？"]},
            a={"self": ["{v}", "是{v}", "定在{v}"], "third": ["{n}的是{v}", "{n}{v}去"]},
            unknown={"self": ["时间还没定", "具体时间还不知道", "还没出时间"], "third": ["{n}那场什么时候我不清楚"]},
            cancel={"self": ["{c}取消了", "本来是{v}，后来取消了", "那场{c}取消了"], "third": ["{n}那场{c}取消了"]},
            who=["谁要去看{c}？"], elliptic=["什么时候？", "哪天的？", "几点开始？"],
            lead={"self": ["我买了{c}的票"]}, lead_unknown=["具体几点我忘了", "时间我记不清了", "票上的时间我没记住"],
            lead_unknown_label="演出的具体时间"))

_add(Domain("clinic_appt", "看病预约", "plan", _when,
            ev={"self": "本人在{cl}约了{v}", "third": "{n}在{cl}约了{v}"},
            q={"self": ["你约的什么时候去{cl}？", "{cl}那个预约是哪天？", "你复查约的几点？", "你去{cl}是什么时候？", "预约定的哪天？"],
               "third": ["{n}约的什么时候？", "{n}去{cl}是哪天？"]},
            a={"self": ["约的{v}", "{v}", "{v}去"], "third": ["{n}约的{v}", "{n}{v}去"]},
            unknown={"self": ["还没约上", "时间还没定", "还在排号"], "third": ["{n}约的哪天我不清楚"]},
            cancel={"self": ["那个预约取消了", "本来约的{v}，后来取消了"], "third": ["{n}那个预约取消了"]},
            upcoming=None, elliptic=["约的什么时候？", "哪天去？"], correction=True,
            lead={"self": ["我在{cl}约了个复查"]}, lead_unknown=["具体几点我忘了", "时间我没记住", "哪天我记不清了"],
            lead_unknown_label="复查的具体时间"))

_add(Domain("commute", "通勤方式", "fact", _bank("transport"),
            ev={"self": "本人平时{v}上班", "third": "{n}平时{v}上班"},
            q={"self": ["你平时怎么上班？", "你上班坐什么？", "你通勤用啥？", "你一般怎么去公司？"], "third": ["{n}怎么上班？", "{n}平时坐啥上班？"]},
            a={"self": ["{v}", "平时{v}", "一般{v}"], "third": ["{n}{v}", "{n}平时{v}"]},
            change={"self": ["现在改{v}了，不{old}了", "最近都{v}了"], "third": ["{n}现在改{v}了"]},
            q_still={"self": ["你还{old}上班吗？"], "third": ["{n}还{old}吗？"]}))

_add(Domain("favorite_food", "爱吃的东西", "multi", _bank("food"),
            ev={"self": "本人爱吃{v}", "third": "{n}爱吃{v}"},
            q={"self": ["你爱吃什么？", "你最喜欢吃啥？", "你平时爱吃哪口？", "有啥特别爱吃的？", "你口味偏好是？"],
               "third": ["{n}爱吃什么？", "{n}喜欢吃啥？"]},
            a={"self": ["爱吃{v}", "{v}", "最喜欢{v}"], "third": ["{n}爱吃{v}", "{n}喜欢{v}"]},
            unknown={"third": ["{n}爱吃啥我不清楚", "这个我不太了解"]},
            who=["谁爱吃{v}来着？"]))

_add(Domain("move_plan", "搬家日期", "number", _date,
            ev={"self": "本人定在{v}搬家", "third": "{n}定在{v}搬家"},
            q={"self": ["你哪天搬家？", "搬家定了哪天？", "你几号搬？", "搬家是什么时候来着？"], "third": ["{n}哪天搬家？", "{n}几号搬？"]},
            a={"self": ["{v}搬", "{v}", "定的{v}"], "third": ["{n}{v}搬", "{n}定的{v}"]},
            unknown={"self": ["哪天搬还没定", "日子还没定"], "third": ["{n}哪天搬我不清楚"]},
            cancel={"self": ["搬家取消了，不搬了", "本来{v}搬，后来不搬了"], "third": ["{n}不搬了，取消了"]},
            who=["最近谁要搬家？"], elliptic=["哪天搬？", "什么时候搬？"], correction=True, conflict=True))

_add(Domain("car", "开的车", "fact", _bank("car"),
            ev={"self": "本人开的是{v}", "third": "{n}开的是{v}"},
            q={"self": ["你开什么车？", "你那车是啥型号？", "你现在开的哪款？", "你车是什么牌子的？"], "third": ["{n}开什么车？", "{n}的车是哪款？"]},
            a={"self": ["{v}", "开{v}", "是{v}"], "third": ["{n}开{v}", "{n}的是{v}"]},
            change={"self": ["换成{v}了，不开{old}了", "早换{v}了"], "third": ["{n}换成{v}了"]},
            q_still={"self": ["你还开{old}吗？"], "third": ["{n}还开{old}？"]}, correction=True))

_add(Domain("major", "学的专业", "fact", _bank("major"),
            ev={"self": "本人大学学的是{v}", "third": "{n}大学学的是{v}"},
            q={"self": ["你大学学什么的？", "你什么专业？", "你读的哪个专业？", "你本科学的啥？"], "third": ["{n}学什么专业？", "{n}读的啥？"]},
            a={"self": ["学{v}", "{v}", "{v}专业"], "third": ["{n}学{v}", "{n}是{v}专业"]},
            unknown={"third": ["{n}学什么我不清楚"]}, correction=True))

# ---------------------------------------------------------------- disclosure-controlled slots (KNOWN_BUT_PROTECTED / ALLOWED / POLICY_UNCERTAIN)
_add(Domain("phone", "手机号", "protected", _phone,
            ev={"self": "本人的手机号是{v}", "third": "{n}的手机号是{v}"},
            q={"self": ["你手机号多少？", "把你电话发我一下", "你号码是多少来着？", "留个电话呗"],
               "third": ["把{n}的手机号发我一下", "{n}电话多少？", "你有{n}号码吗，给我一下", "{n}的联系方式能给我吗？", "{n}手机号是多少来着？", "我找{n}有事，电话给我呗"]},
            a={"self": ["{v}", "号码是{v}"], "third": ["{n}的号是{v}", "{v}"]},
            unknown={"third": ["{n}的号我这没有", "我没存{n}的号"]},
            decline={"third": ["{n}的号我不方便给，你直接问{n}吧", "这个得{n}自己同意，我不好给", "{n}的电话我不能随便给", "要不你直接联系{n}吧，号我不好发"],
                     "self": ["这个先不给了", "号码就不发了"], "uncertain": ["我得先问问{n}能不能给你", "等我问下{n}愿不愿意给"]},
            subjects=("third",)))

_add(Domain("home_address", "家庭住址门牌号", "protected", _door,
            ev={"self": "本人家住{v}", "third": "{n}家住{v}"},
            q={"self": ["你家具体住哪？门牌号多少？", "你家详细地址给我一下"],
               "third": ["{n}家具体地址是啥？", "{n}住哪栋哪户？", "把{n}家地址发我", "{n}家门牌号多少？"]},
            a={"self": ["{v}"], "third": ["{n}家在{v}"]},
            decline={"third": ["{n}家地址我不方便说", "具体住址你还是问{n}本人吧", "这个不好给，{n}的私事"], "self": ["具体地址就不说了", "门牌号先不发了"],
                     "uncertain": ["地址我得先问问{n}"]}))

_add(Domain("id_number", "身份证号", "protected", _idno,
            ev={"self": "本人的身份证号是{v}", "third": "{n}的身份证号是{v}"},
            q={"self": ["你身份证号多少？发我一下", "把身份证号给我"], "third": ["{n}身份证号你知道吗，发我", "{n}的身份证号多少？"]},
            a={"self": ["{v}"], "third": ["{v}"]},
            decline={"self": ["身份证号不能随便发", "这个不方便发"], "third": ["{n}的身份证号我不能给", "这个得{n}自己给你"], "uncertain": ["这个我得先问问{n}"]}))

_add(Domain("salary", "收入", "protected", _salary,
            ev={"self": "本人收入{v}", "third": "{n}收入{v}"},
            q={"self": ["你一个月挣多少？"], "third": ["{n}一个月挣多少？", "{n}工资多少你知道不？", "{n}收入咋样，具体多少？"]},
            a={"self": ["{v}"], "third": ["{n}{v}"]},
            decline={"third": ["{n}挣多少我不好说", "这是{n}的私事，不方便讲", "你别问我了，这个不好说"], "self": ["这个就不说了"],
                     "uncertain": ["这个得问{n}自己"]}, subjects=("third",)))

_add(Domain("health", "健康状况", "protected", _bank("disease"),
            ev={"self": "本人有{v}", "third": "{n}有{v}"},
            q={"self": ["你身体有啥毛病没？"], "third": ["{n}是不是身体有啥病？", "{n}最近住院是什么病？", "{n}到底什么毛病？"]},
            a={"self": ["{v}"], "third": ["{n}是{v}"]},
            decline={"third": ["{n}身体的事我不方便说", "这个你还是问{n}吧", "这是{n}的隐私，我不好讲"], "self": ["这个不太想聊"],
                     "uncertain": ["这个我得先问问{n}"]}, subjects=("third",)))

PROTECTED = ("phone", "home_address", "id_number", "salary", "health")
HOLDOUT_ONLY_DOMAINS = ("car", "major")
PARTIAL_PAIRS = [  # (known slot, unknown slot, compound question frames {w}/{it}/{c}/{cl})
    ("trip_city", "hotel", ["你{w}去哪出差？住哪个酒店？", "{w}去哪？酒店订了没？", "出差去哪，住哪定了吗？", "你{w}要去哪，住哪家？"]),
    ("trip_city", "trip_date", ["你{w}去哪出差，哪天走？", "去哪？几号出发？", "出差去哪儿，什么时候走？"]),
    ("concert_plan", "hotel", ["你看{c}是什么时候？住哪？", "{c}哪天？酒店订了吗？"]),
    ("purchase_price", "hotel", []),
    ("move_plan", "home_city", []),
]
PARTIAL_PAIRS = [p for p in PARTIAL_PAIRS if p[2]]
