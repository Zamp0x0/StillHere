"""Public synthetic Context Regression Dataset (BUILD-M1) and its metrics. Two invented personas; nothing describes a real person.

``build_world`` writes one in-memory ``MemoryAPI``. ``cases`` are the regression cases: each is a ``ContextRequest`` plus what the
assembled context must and must not contain. ``evaluate`` runs the orchestrator on every case and computes the M1 metrics; it
also runs the frozen legacy renderer (``v3_bench.round_text``, three turns) on the same rounds, so the D6 defect is measured, not
assumed. No model and no network.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from stillhere.context import ContextOrchestrator, ContextRequest, DerivedItem, WorkingItem
from stillhere.context import contracts as cc
from stillhere.contracts import (CanonicalMessage, DataOrigin, EvidenceHit, HitKind, MemoryKind, MemoryRecord, Provenance, SourceKind,
                                 SourceRef)
from stillhere.memory import MemoryAPI

TZ = timezone(timedelta(hours=8))
PA, PB = "syn.pa", "syn.pb"
BOT_A, BOT_B = "syn.pa.assistant", "syn.pb.assistant"
LI = "syn.pa.contact.lilei"
SPEAKERS = {BOT_A: "AI 助手", BOT_B: "AI 助手", LI: "李雷（同事）"}
CHAT, SMS, CAL, DAILY, FACTS = "syn.src.chat", "syn.src.sms", "syn.src.calendar", "syn.src.daily_event", "syn.src.profile"
ROUND = "synthetic_chat_round"


def t(day: int, hour: int = 10, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2026, 11, day, hour, minute, second, tzinfo=TZ)


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


@dataclass
class World:
    api: MemoryAPI
    clock: Clock
    rounds: Dict[str, List[str]] = field(default_factory=dict)          # conversation id -> message ids in order
    owner: Dict[str, str] = field(default_factory=dict)                 # record id -> persona
    said: Dict[str, datetime] = field(default_factory=dict)
    recorded: Dict[str, datetime] = field(default_factory=dict)
    contents: Dict[str, str] = field(default_factory=dict)


def _round(w: World, persona: str, name: str, start: datetime, lines: Sequence[Tuple[str, str]], *, step_s: int = 60,
           times: Optional[Sequence[datetime]] = None, recorded_delay: Optional[timedelta] = None, locator_time: str = "given") -> str:
    conv = f"{persona}.agent_chat.{name}"
    bot = BOT_A if persona == PA else BOT_B
    msgs = []
    for i, (who, text) in enumerate(lines):
        when = times[i] if times else start + timedelta(seconds=step_s * i)
        mid = f"{conv}.{i}"
        rec = when + recorded_delay if recorded_delay else when
        msgs.append(CanonicalMessage(conversation_id=conv, message_id=mid, target_persona_id=persona, speaker_id=persona if who == "u" else bot, timestamp=when,
                                     content=text, source_type=ROUND, source_id=CHAT, locator=f"round={name};pos={i};time={locator_time}", recorded_at=rec))
        w.owner[mid], w.said[mid], w.recorded[mid], w.contents[mid] = persona, when, rec, text
    w.api.ingest_messages(msgs)
    w.rounds[conv] = [m.message_id for m in sorted(msgs, key=lambda m: (m.timestamp, int(m.message_id.rsplit(".", 1)[1])))]
    return conv


def _sms(w: World, mid: str, kind: Optional[str], when: datetime, text: str, speaker: str) -> str:
    loc = f"phone_id={mid};type={kind}" if kind else f"phone_id={mid}"
    m = CanonicalMessage(conversation_id=f"{PA}.sms.lilei", message_id=f"{PA}.sms.{mid}", target_persona_id=PA, speaker_id=speaker, timestamp=when, content=text,
                         source_type="synthetic_sms", source_id=SMS, locator=loc, recorded_at=when)
    w.api.ingest_messages([m])
    w.owner[m.message_id], w.said[m.message_id], w.recorded[m.message_id], w.contents[m.message_id] = PA, when, when, text
    return m.message_id


def _fact(w: World, mid: str, persona: str, statement: str, source: str, *, recorded: datetime, valid_from: Optional[datetime] = None,
          valid_to: Optional[datetime] = None, attribute: str = "note", value: str = "x", kind: MemoryKind = MemoryKind.EVENT) -> str:
    w.clock.now = recorded
    r = MemoryRecord(memory_id=mid, persona_id=persona, kind=kind, subject="persona", attribute=attribute, value=value, statement=statement,
                     provenance=(Provenance(source_id=source, locator=f"row={mid}"),), origin=DataOrigin.SYNTHETIC, recorded_at=recorded,
                     valid_from=valid_from, valid_to=valid_to)
    w.api.add_facts(persona, [r])
    w.owner[mid], w.recorded[mid], w.contents[mid] = persona, recorded, statement
    return mid


def build_world() -> World:
    clock = Clock(t(1, 0))
    api = MemoryAPI(":memory:", clock=clock)
    w = World(api, clock)
    for sid, kind in ((CHAT, SourceKind.CONVERSATION), (SMS, SourceKind.CONVERSATION), (CAL, SourceKind.DOCUMENT), (DAILY, SourceKind.DOCUMENT), (FACTS, SourceKind.DOCUMENT)):
        api.register_source(SourceRef(source_id=sid, kind=kind, origin=DataOrigin.SYNTHETIC, description=f"Synthetic M1 context regression source {sid}."))
    _round(w, PA, "r1", t(1, 9), [("u", "我想开始学吉他，先找个老师。")])
    _round(w, PA, "r3", t(2, 9), [("u", "周末去爬山吗？"), ("a", "好主意，天气预报说周六晴。"), ("u", "那就周六早上去西山。")])
    _round(w, PA, "r4", t(3, 9), [("u", "我报名了杭州半程马拉松。"), ("a", "加油！需要训练计划吗？"), ("u", "先每周跑三次吧。"),
                                  ("u", "哦对了，其实我报的是苏州半程马拉松，不是杭州。")])
    _round(w, PA, "r6_cancel", t(4, 9), [("u", "下周三我要去上海出差。"), ("a", "需要我帮你记一下行程吗？"), ("u", "订了周三早上八点的高铁。"),
                                         ("a", "好的，已经记下。"), ("u", "酒店订在静安区。"), ("u", "刚接到通知，上海出差取消了，高铁票也退了。")])
    _round(w, PA, "r6_done", t(6, 20), [("u", "季度报告还差两页没写。"), ("a", "今晚能写完吗？"), ("u", "尽量吧，数据表还没整理。"),
                                        ("a", "可以先整理数据表。"), ("u", "数据表整理好了。"), ("u", "报告终于写完，已经交给王经理了。")])
    _round(w, PA, "r10_move", t(7, 19), [("u", "我在考虑搬家。"), ("a", "想搬到哪里？"), ("u", "在看西湖区和滨江区。"), ("a", "两个区各有什么考虑？"),
                                         ("u", "西湖区离公司近，但是房租贵。"), ("a", "滨江区呢？"), ("u", "滨江区房租便宜一些，通勤四十分钟。"),
                                         ("a", "你更看重哪一点？"), ("u", "我再想想，周末去看房。"), ("u", "看完房了，最后决定搬到滨江区，下个月一号搬。")])
    _round(w, PA, "r5_modify", t(8, 21), [("u", "我女儿今年八岁了。"), ("a", "她在上几年级？"), ("u", "三年级。"), ("a", "这个年纪很可爱。"),
                                          ("u", "我刚才说错了，我女儿今年九岁，不是八岁。")])
    _round(w, PA, "r3_quote", t(9, 20), [("u", "我妈刚发微信说：“你爸周日来看你，记得去车站接。”"), ("a", "需要提醒你周日去车站吗？"), ("u", "要，周日下午两点提醒我。")])
    _round(w, PA, "ia", t(10, 10), [("u", "A轮第一句：我在准备面试。"), ("a", "A轮第二句：哪家公司？"), ("u", "A轮第三句：一家做地图的公司。")],
           times=[t(10, 10, 0), t(10, 10, 2), t(10, 10, 4)])
    _round(w, PA, "ib", t(10, 10), [("u", "B轮第一句：晚饭吃什么？"), ("a", "B轮第二句：要不要试试面馆？")], times=[t(10, 10, 1), t(10, 10, 3)])
    _round(w, PA, "order", t(10, 11), [("u", "这条存储在前，时间却在后。"), ("u", "这条存储在后，时间却在前。")], times=[t(10, 11, 5), t(10, 11, 0)])
    long = "这是一段很长的旅行讨论，" * 25
    lines = [("u", f"第{i + 1}条：{long}") for i in range(12)]
    lines[5] = ("u", "第6条：机票订在十一月二十八日上午，去成都。" + long)
    lines[11] = ("u", "第12条：成都之行取消了，机票全额退款。")
    _round(w, PA, "r12_long", t(10, 15), lines)
    _round(w, PA, "r_future", t(15, 9), [("u", "十五号的我去了动物园。"), ("a", "好玩吗？")])
    _round(w, PA, "r_late_known", t(5, 9), [("u", "今天公司年会，我拿了一等奖。")], recorded_delay=timedelta(days=1))
    _round(w, PA, "r_partial_future", t(10, 23, 50), [("u", "今晚十一点五十，我在写日记。"), ("u", "过了零点，我想起明天要早起。")], step_s=900)
    _round(w, PA, "r_dayonly", t(5, 23, 59), [("u", "那天我去了图书馆。")], locator_time="assumed_end_of_date")
    _round(w, PB, "r3", t(2, 9), [("u", "周末去爬山吗？"), ("a", "好主意。"), ("u", "那就周日去东山。")])
    _round(w, PB, "secret", t(3, 9), [("u", "我的体检报告有点问题，别告诉别人。")])
    _round(w, PA, "current", t(10, 20), [("u", "我下周要去北京见张伟。"), ("a", "好的，哪天去？"), ("u", "周三去。"), ("a", "记下了：周三去北京。"),
                                         ("u", "不对，我说错了，是周四去，不是周三。"), ("a", "好的，改成周四。"), ("u", "他住在朝阳区。"), ("a", "了解。")])
    _sms(w, "s1", "发送", t(4, 15), "明天下午三点开会，别忘了带材料。", PA)
    _sms(w, "s2", "接收", t(4, 15, 5), "好的，我会准时到，材料我也带上。", LI)
    _sms(w, "s3", None, t(4, 16), "会议改到四点了。", LI)
    _fact(w, "f_cal_hospital", PA, "日程：11月12日下午去医院复查。", CAL, recorded=t(5, 8), valid_from=t(12, 14), valid_to=t(12, 15))
    _fact(w, "f_gala", PA, "11月5日晚上参加了公司年会。", DAILY, recorded=t(5, 22), valid_from=t(5, 19), valid_to=t(5, 22))
    _fact(w, "f_dinner_4", PA, "11月4日晚上和朋友吃了火锅。", DAILY, recorded=t(4, 22), valid_from=t(4, 19), valid_to=t(4, 21))
    _fact(w, "f_film_6", PA, "11月6日下午看了电影。", DAILY, recorded=t(6, 18), valid_from=t(6, 15), valid_to=t(6, 17))
    _fact(w, "f_name", PA, "我的名字是林晓（虚构人物）。", FACTS, recorded=t(1, 8), valid_from=t(1, 0), attribute="name", value="林晓", kind=MemoryKind.FACT)
    _fact(w, "f_home_old", PA, "我住在西湖区。", FACTS, recorded=t(1, 9), valid_from=t(1, 0), attribute="home_district", value="西湖区", kind=MemoryKind.FACT)
    _fact(w, "f_job_wrong", PA, "我在银行工作。", FACTS, recorded=t(1, 9), valid_from=t(1, 0), attribute="job", value="银行", kind=MemoryKind.FACT)
    _fact(w, "f_pb_home", PB, "我住在东山脚下。", FACTS, recorded=t(1, 9), valid_from=t(1, 0), attribute="home_district", value="东山", kind=MemoryKind.FACT)
    _fact(w, "f_hobby_a", PA, "我喜欢游泳。", FACTS, recorded=t(1, 9), valid_from=t(1, 0), attribute="hobby", value="游泳", kind=MemoryKind.PREFERENCE)
    _fact(w, "f_hobby_b", PA, "我喜欢下围棋。", FACTS, recorded=t(2, 9), valid_from=t(2, 0), attribute="hobby", value="围棋", kind=MemoryKind.PREFERENCE)
    w.clock.now = t(8, 10)
    new = MemoryRecord(memory_id="f_home_new", persona_id=PA, kind=MemoryKind.FACT, subject="persona", attribute="home_district", value="滨江区",
                       statement="我住在滨江区。", provenance=(Provenance(source_id=FACTS, locator="row=f_home_new"),), origin=DataOrigin.SYNTHETIC,
                       recorded_at=t(8, 10), valid_from=t(8, 0), supersedes="f_home_old", correction_reason="moved")
    api.correct(PA, "f_home_old", new, "moved")
    w.owner["f_home_new"], w.recorded["f_home_new"], w.contents["f_home_new"] = PA, t(8, 10), new.statement
    w.clock.now = t(9, 10)
    api.retract(PA, "f_job_wrong", "never worked at a bank")
    w.clock.now = t(20, 0)
    return w


# ---------------------------------------------------------------- hits (the retriever's output, deterministic)

def message_hit(w: World, mid: str, score: float = 1.0) -> EvidenceHit:
    m = w.api.store.get_message(w.owner[mid], mid)
    return EvidenceHit(kind=HitKind.MESSAGE, persona_id=m.target_persona_id, text=m.content, source_id=m.source_id, score=score, retriever="synthetic_ranked",
                       message_id=mid, speaker_id=m.speaker_id, conversation_id=m.conversation_id, timestamp=m.timestamp, recorded_at=m.recorded_at, locator=m.locator)


def memory_hit(w: World, mid: str, score: float = 1.0) -> EvidenceHit:
    r = w.api.store.get(mid, persona_id=w.owner[mid])
    return EvidenceHit(kind=HitKind.MEMORY, persona_id=r.persona_id, text=r.statement, source_id=r.provenance[0].source_id, score=score, retriever="synthetic_ranked",
                       memory_id=mid, subject_id=r.subject, valid_from=r.valid_from, valid_to=r.valid_to, recorded_at=r.recorded_at, locator=r.provenance[0].locator)


def hits(w: World, *ids: str) -> Tuple[EvidenceHit, ...]:
    return tuple(message_hit(w, i, 1.0 - n / 100) if i in w.said else memory_hit(w, i, 1.0 - n / 100) for n, i in enumerate(ids))


def rerank(hs: Sequence[EvidenceHit], order: Sequence[int]) -> Tuple[EvidenceHit, ...]:
    """A deterministic stand-in for the Qwen reranker: a new order of the same hits."""
    return tuple(hs[i] for i in order)


# ---------------------------------------------------------------- cases

@dataclass
class Case:
    case_id: str
    category: str
    request: ContextRequest
    required: Tuple[str, ...] = ()                 # message or memory ids that must be in the context
    forbidden: Tuple[str, ...] = ()                # ids that must not be (other persona, later, retracted)
    forbidden_text: Tuple[str, ...] = ()
    full_rounds: Tuple[str, ...] = ()              # conversation ids that must be shown whole (no trimming needed)
    fits: bool = True                              # the required evidence fits the declared budget
    expect_status: str = cc.OK
    expect_warnings: Tuple[str, ...] = ()
    expect_boundary: Optional[Tuple[str, Optional[str]]] = None          # (status, operator)
    expect_relation: Dict[str, str] = field(default_factory=dict)        # record key -> SATISFIES / VIOLATES / UNKNOWN
    attribution: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # message id -> (sender, recipient)
    expect_order: Tuple[str, ...] = ()             # message ids that must appear in this order
    note: str = ""


def _req(question: str, as_of: datetime, hs=(), **kw) -> ContextRequest:
    return ContextRequest(persona_id=kw.pop("persona", PA), question=question, as_of=as_of, hits=tuple(hs), **kw)


def cases(w: World) -> List[Case]:
    R = w.rounds
    c: List[Case] = []
    a = t(11, 12)       # the default question time
    # --- D6: whole rounds of every length
    for name, size in (("r1", 1), ("r3", 3), ("r4", 4), ("r6_cancel", 6), ("r10_move", 10)):
        conv = f"{PA}.agent_chat.{name}"
        c.append(Case(f"d6_len_{size}", "d6_full_round", _req("我之前聊过什么？", a, hits(w, R[conv][0])), required=tuple(R[conv]), full_rounds=(conv,)))
    c.append(Case("d6_hit_last_msg", "d6_full_round", _req("马拉松是哪个城市的？", a, hits(w, R[f"{PA}.agent_chat.r4"][1])), required=(R[f"{PA}.agent_chat.r4"][3],),
                  full_rounds=(f"{PA}.agent_chat.r4",), note="the 4th message corrects the 1st"))
    c.append(Case("d6_modify_fact", "d6_correction", _req("我女儿几岁？", a, hits(w, R[f"{PA}.agent_chat.r5_modify"][0])), required=(R[f"{PA}.agent_chat.r5_modify"][4],),
                  full_rounds=(f"{PA}.agent_chat.r5_modify",)))
    c.append(Case("d6_cancel_plan", "d6_plan_vs_done", _req("我去上海出差了吗？", a, hits(w, R[f"{PA}.agent_chat.r6_cancel"][0])), required=(R[f"{PA}.agent_chat.r6_cancel"][5],),
                  full_rounds=(f"{PA}.agent_chat.r6_cancel",)))
    c.append(Case("d6_completed", "d6_plan_vs_done", _req("季度报告交了吗？", a, hits(w, R[f"{PA}.agent_chat.r6_done"][0])), required=(R[f"{PA}.agent_chat.r6_done"][5],),
                  full_rounds=(f"{PA}.agent_chat.r6_done",)))
    c.append(Case("d6_final_decision", "d6_correction", _req("我最后决定搬到哪个区？", a, hits(w, R[f"{PA}.agent_chat.r10_move"][2])), required=(R[f"{PA}.agent_chat.r10_move"][9],),
                  full_rounds=(f"{PA}.agent_chat.r10_move",)))
    c.append(Case("d6_two_hits_one_round", "duplicate", _req("上海出差的高铁和酒店？", a, hits(w, R[f"{PA}.agent_chat.r6_cancel"][2], R[f"{PA}.agent_chat.r6_cancel"][4])),
                  required=tuple(R[f"{PA}.agent_chat.r6_cancel"]), full_rounds=(f"{PA}.agent_chat.r6_cancel",), note="two hits on one round make one unit"))
    # --- speakers, senders, recipients
    q = R[f"{PA}.agent_chat.r3_quote"]
    c.append(Case("attr_quoted_third_party", "attribution", _req("谁周日来看我？", a, hits(w, q[0])), required=tuple(q), full_rounds=(f"{PA}.agent_chat.r3_quote",),
                  attribution={q[0]: ("本人", "AI 助手"), q[1]: ("AI 助手", "本人")}))
    c.append(Case("attr_sms_sent", "attribution", _req("我提醒李雷开会了吗？", a, hits(w, f"{PA}.sms.s1")), required=(f"{PA}.sms.s1",), attribution={f"{PA}.sms.s1": ("本人", "李雷（同事）")}))
    c.append(Case("attr_sms_received", "attribution", _req("李雷说他会来吗？", a, hits(w, f"{PA}.sms.s2")), required=(f"{PA}.sms.s2",), attribution={f"{PA}.sms.s2": ("李雷（同事）", "本人")}))
    c.append(Case("attr_sms_unknown_type", "attribution", _req("会议几点？", a, hits(w, f"{PA}.sms.s3")), required=(f"{PA}.sms.s3",), attribution={f"{PA}.sms.s3": ("李雷（同事）", "未知")},
                  note="no stored type: the recipient is unknown, not guessed"))
    ia, ib = R[f"{PA}.agent_chat.ia"], R[f"{PA}.agent_chat.ib"]
    c.append(Case("interleaved_rounds", "attribution", _req("面试和晚饭聊了什么？", a, hits(w, ia[0], ib[0])), required=tuple(ia + ib), full_rounds=(f"{PA}.agent_chat.ia", f"{PA}.agent_chat.ib"),
                  expect_order=tuple(ia), forbidden_text=(), note="two rounds whose messages interleave in time stay apart"))
    o = R[f"{PA}.agent_chat.order"]
    c.append(Case("order_anomaly", "integrity", _req("那两条说了什么？", a, hits(w, o[0])), required=tuple(o), full_rounds=(f"{PA}.agent_chat.order",), expect_warnings=(cc.ORDER_ANOMALY,),
                  expect_order=(f"{PA}.agent_chat.order.1", f"{PA}.agent_chat.order.0")))
    # --- persona isolation
    c.append(Case("persona_similar_message", "persona", _req("周末爬山去哪？", a, hits(w, f"{PB}.agent_chat.r3.0", R[f"{PA}.agent_chat.r3"][0])), required=tuple(R[f"{PA}.agent_chat.r3"]),
                  forbidden=tuple(R[f"{PB}.agent_chat.r3"]), forbidden_text=("东山",), full_rounds=(f"{PA}.agent_chat.r3",)))
    c.append(Case("persona_private_fact", "persona", _req("我住在哪？", a, hits(w, "f_pb_home", "f_home_new")), required=("f_home_new",), forbidden=("f_pb_home",), forbidden_text=("东山",)))
    c.append(Case("persona_private_round", "persona", _req("体检怎么样？", a, hits(w, f"{PB}.agent_chat.secret.0")), forbidden=(f"{PB}.agent_chat.secret.0",), forbidden_text=("体检",)))
    # --- time visibility
    c.append(Case("future_round_hidden", "temporal_visibility", _req("我去动物园了吗？", a, hits(w, f"{PA}.agent_chat.r_future.0")), forbidden=tuple(R[f"{PA}.agent_chat.r_future"]),
                  forbidden_text=("动物园",)))
    c.append(Case("known_at_hides_late_record", "known_at", _req("11月5日年会我拿奖了吗？", t(5, 18), hits(w, f"{PA}.agent_chat.r_late_known.0")),
                  forbidden=(f"{PA}.agent_chat.r_late_known.0",), forbidden_text=("一等奖",), note="said on the 5th, learned on the 6th: asked on the 5th it is not known"))
    c.append(Case("known_at_later_shows_record", "known_at", _req("11月5日年会我拿奖了吗？", t(5, 18), hits(w, f"{PA}.agent_chat.r_late_known.0"), known_at=t(7, 0)),
                  required=(f"{PA}.agent_chat.r_late_known.0",), note="an explicit later known_at may see it"))
    c.append(Case("known_at_nov4_event_nov5", "known_at", _req("11月4日时，系统知道11月5日的年会吗？", t(4, 23), hits(w, "f_gala")), forbidden=("f_gala",), forbidden_text=("年会",)))
    pf = R[f"{PA}.agent_chat.r_partial_future"]
    c.append(Case("round_part_after_as_of", "temporal_visibility", _req("那晚写日记了吗？", t(10, 23, 55), hits(w, pf[0])), required=(pf[0],), forbidden=(pf[1],),
                  forbidden_text=("零点",), expect_warnings=(cc.LATER_MESSAGES_HIDDEN,), note="the later message of the round is not yet said"))
    c.append(Case("retracted_not_current", "knowledge_update", _req("我在哪工作？", a, hits(w, "f_job_wrong")), forbidden=("f_job_wrong",), forbidden_text=("银行",)))
    c.append(Case("retracted_but_known_before", "knowledge_update", _req("我在哪工作？", t(8, 12), hits(w, "f_job_wrong")), required=("f_job_wrong",),
                  note="before the retraction it was the belief, so it is shown"))
    c.append(Case("superseded_old_value", "knowledge_update", _req("我住在哪个区？", a, hits(w, "f_home_old", "f_home_new")), required=("f_home_new",), forbidden=("f_home_old",),
                  forbidden_text=("西湖区。",)))
    c.append(Case("conflict_multi_value", "conflict", _req("我有什么爱好？", a, hits(w, "f_hobby_a", "f_hobby_b")), required=("f_hobby_a", "f_hobby_b")))
    c.append(Case("multi_fact_three_rounds", "multi_fact", _req("我最近有哪些计划？", a, hits(w, R[f"{PA}.agent_chat.r1"][0], R[f"{PA}.agent_chat.r3"][2], R[f"{PA}.agent_chat.r6_cancel"][0])),
                  required=R[f"{PA}.agent_chat.r1"] + R[f"{PA}.agent_chat.r3"] + R[f"{PA}.agent_chat.r6_cancel"], full_rounds=(f"{PA}.agent_chat.r1", f"{PA}.agent_chat.r3", f"{PA}.agent_chat.r6_cancel")))
    c.append(Case("reranked_order_kept", "reranker", _req("我最近有哪些计划？", a, rerank(hits(w, R[f"{PA}.agent_chat.r1"][0], R[f"{PA}.agent_chat.r3"][0]), (1, 0))),
                  required=R[f"{PA}.agent_chat.r1"] + R[f"{PA}.agent_chat.r3"], expect_order=(R[f"{PA}.agent_chat.r3"][0], R[f"{PA}.agent_chat.r1"][0])))
    c.append(Case("duplicate_memory_hit", "duplicate", _req("年会是哪天？", a, hits(w, "f_gala", "f_gala")), required=("f_gala",)))
    # --- temporal boundaries on evidence
    for cid, question, op, rel in (("tb_before_excludes_day", "11月5日之前我参加过年会吗？", "BEFORE", "VIOLATES"), ("tb_until_includes_day", "截至11月5日我参加过年会吗？", "ON_OR_BEFORE", "SATISFIES"),
                                   ("tb_on_day", "11月5日当天我做了什么？", "ON", "SATISFIES"), ("tb_after_excludes_day", "11月5日之后我参加过年会吗？", "AFTER", "VIOLATES"),
                                   ("tb_since_includes_day", "自11月5日起我参加过年会吗？", "ON_OR_AFTER", "SATISFIES"), ("tb_not_later_than", "不晚于11月5日的活动有哪些？", "ON_OR_BEFORE", "SATISFIES"),
                                   ("tb_not_earlier_than", "不早于11月5日的活动有哪些？", "ON_OR_AFTER", "SATISFIES"), ("tb_between", "11月4日到11月5日之间我做了什么？", "BETWEEN", "SATISFIES"),
                                   ("tb_before_inclusive_marker", "11月5日之前（含当天）我参加过年会吗？", "ON_OR_BEFORE", "SATISFIES")):
        c.append(Case(cid, "temporal_boundary", _req(question, a, hits(w, "f_gala")), required=("f_gala",), expect_boundary=("RESOLVED", op), expect_relation={"f_gala": rel}))
    c.append(Case("tb_before_keeps_earlier", "temporal_boundary", _req("11月5日之前我做了什么？", a, hits(w, "f_dinner_4", "f_gala", "f_film_6")), required=("f_dinner_4", "f_gala", "f_film_6"),
                  expect_boundary=("RESOLVED", "BEFORE"), expect_relation={"f_dinner_4": "SATISFIES", "f_gala": "VIOLATES", "f_film_6": "VIOLATES"},
                  note="the boundary labels, it does not delete: the reader still sees all three, each against the boundary"))
    c.append(Case("tb_ambiguous_until", "temporal_boundary", _req("到11月5日我参加过几次活动？", a, hits(w, "f_gala")), required=("f_gala",), expect_boundary=("AMBIGUOUS", None),
                  expect_relation={"f_gala": "UNKNOWN"}, expect_warnings=(cc.TEMPORAL_BOUNDARY_AMBIGUOUS,)))
    c.append(Case("tb_around", "temporal_boundary", _req("11月5日前后我在忙什么？", a, hits(w, "f_gala")), required=("f_gala",), expect_boundary=("AMBIGUOUS", None), expect_relation={"f_gala": "UNKNOWN"}))
    c.append(Case("tb_vague", "temporal_boundary", _req("前阵子我去哪了？", a, hits(w, "f_gala")), required=("f_gala",), expect_boundary=("UNKNOWN", None)))
    c.append(Case("tb_day_precision_unknown", "temporal_boundary", _req("11月5日晚上十点之前我去图书馆了吗？", a, hits(w, f"{PA}.agent_chat.r_dayonly.0")), required=(f"{PA}.agent_chat.r_dayonly.0",),
                  note="a record known only to the day is never compared to the minute"))
    c.append(Case("tb_planned_vs_occurred", "plan_vs_done", _req("11月12日之前我去医院了吗？", a, hits(w, "f_cal_hospital")), required=("f_cal_hospital",), expect_boundary=("RESOLVED", "BEFORE"),
                  expect_relation={"f_cal_hospital": "VIOLATES"}, note="a calendar entry is a plan: it is labelled 计划时间"))
    # --- recent conversation, pinned, working, derived
    cur = R[f"{PA}.agent_chat.current"]
    c.append(Case("recent_correction_kept", "recent", _req("他住在哪？我哪天去北京？", t(10, 21), conversation_id=f"{PA}.agent_chat.current"), required=(cur[4], cur[6]),
                  note="the latest correction and the referent are in the recent context"))
    c.append(Case("recent_tight_budget_keeps_correction", "recent", _req("我哪天去北京？他是谁？", t(10, 21), conversation_id=f"{PA}.agent_chat.current", input_token_budget=420),
                  required=(cur[4], cur[-1]), expect_warnings=(cc.RECENT_TRIMMED,), note="trimmed from the oldest end, the correction pulled back in"))
    c.append(Case("pinned_identity", "pinned", _req("我叫什么名字？", a, pinned_memory_ids=("f_name",)), required=("f_name",)))
    c.append(Case("pinned_retracted_rejected", "pinned", _req("我在哪工作？", a, pinned_memory_ids=("f_job_wrong",)), forbidden=("f_job_wrong",), forbidden_text=("银行",)))
    c.append(Case("working_not_persisted", "working", _req("帮我继续整理搬家清单", a, working=(WorkingItem("w1", "当前任务：整理搬家清单，已完成纸箱和搬家公司两项。"),)), required=()))
    c.append(Case("derived_disabled", "derived", _req("我最近在忙什么？", a, hits(w, "f_gala"), derived=(DerivedItem("obs1", "observation", "用户最近工作很忙。", ("f_gala",)),)),
                  required=("f_gala",), forbidden_text=("工作很忙",), expect_warnings=(cc.DERIVED_EXCLUDED,)))
    c.append(Case("derived_allowed_labelled", "derived", _req("我最近在忙什么？", a, hits(w, "f_gala"), derived=(DerivedItem("obs1", "observation", "用户最近工作很忙。", ("f_gala",)),), allow_derived=True),
                  required=("f_gala",)))
    # --- budget
    c.append(Case("overflow_long_round", "token_overflow", _req("我十一月底去成都吗？", a, hits(w, R[f"{PA}.agent_chat.r12_long"][5]), input_token_budget=1500),
                  required=(R[f"{PA}.agent_chat.r12_long"][5], R[f"{PA}.agent_chat.r12_long"][11]), fits=True, expect_warnings=(cc.CONTEXT_OVERFLOW,),
                  note="the round does not fit: the hit and the final cancellation are kept, every omitted message is marked"))
    c.append(Case("overflow_whole_round_fits_larger_budget", "token_overflow", _req("我十一月底去成都吗？", a, hits(w, R[f"{PA}.agent_chat.r12_long"][5]), input_token_budget=8000),
                  required=tuple(R[f"{PA}.agent_chat.r12_long"]), full_rounds=(f"{PA}.agent_chat.r12_long",)))
    c.append(Case("budget_fixed_parts_too_big", "token_overflow", _req("我" * 300 + "？", a, hits(w, "f_gala"), input_token_budget=200), fits=False, expect_status=cc.BUDGET_INSUFFICIENT,
                  expect_warnings=(cc.BUDGET_INSUFFICIENT,)))
    c.append(Case("budget_top_evidence_does_not_fit", "token_overflow", _req("马拉松是哪个城市的？", a, hits(w, R[f"{PA}.agent_chat.r4"][0]), input_token_budget=60, instruction_tokens=20),
                  fits=False, expect_status=cc.BUDGET_INSUFFICIENT, expect_warnings=(cc.TOP_EVIDENCE_EXCLUDED,)))
    c.append(Case("budget_skips_big_keeps_small", "token_overflow", _req("我最近的计划？", a, hits(w, R[f"{PA}.agent_chat.r12_long"][0], "f_gala"), input_token_budget=900),
                  required=("f_gala",), note="an oversized unit is shown in part or excluded, the next unit is still tried"))
    # --- abstention
    c.append(Case("abstain_no_hits", "abstention", _req("我养过猫吗？", a), required=(), note="no evidence: the reader sees that nothing was found"))
    c.append(Case("abstain_only_future_hits", "abstention", _req("我去动物园了吗？", t(12, 0), hits(w, f"{PA}.agent_chat.r_future.1")), forbidden=tuple(R[f"{PA}.agent_chat.r_future"]),
                  forbidden_text=("动物园",)))
    # --- retrieval integration (the real legacy FTS path of MemoryAPI)
    fts = tuple(w.api.search_messages(PA, "上海出差", as_of=a, k=5))
    c.append(Case("retrieval_fts_integration", "retrieval", _req("上海出差怎么样了？", a, fts), required=(R[f"{PA}.agent_chat.r6_cancel"][5],), full_rounds=(f"{PA}.agent_chat.r6_cancel",)))
    return c


# ---------------------------------------------------------------- legacy comparison

class _LegacyScope:
    def __init__(self, w: World, as_of: datetime, known_at: datetime) -> None:
        self.w, self.as_of, self.known_at = w, as_of, known_at

    def visible(self, persona_id: str, record_id: str, mask) -> bool:
        return record_id in self.w.said and self.w.owner[record_id] == persona_id and self.w.said[record_id] <= self.as_of and self.w.recorded[record_id] <= self.known_at


def legacy_round(w: World, persona: str, conv: str, as_of: datetime) -> Optional[str]:
    from stillhere.evaluation import v3_bench as vb
    return vb.round_text(w.api, _LegacyScope(w, as_of, as_of), None, persona, w.rounds[conv][0], SPEAKERS)


# ---------------------------------------------------------------- metrics

def check(w: World, case: Case, r) -> dict:
    req = case.request
    as_of, known = w.api.resolve_times(req.as_of, req.known_at)
    text = r.assembled_context
    included_ids = set(r.included_message_ids) | {m for p in r.provenance_map.values() for m in p.get("memory_ids", [])} | {p["key"] for p in r.provenance_map.values() if p["section"] == "pinned"}
    out = {"case_id": case.case_id, "category": case.category, "status": r.status, "warnings": r.warnings, "tokens": r.token_usage_estimate, "budget": req.input_token_budget,
           "token_mode": r.token_count_mode, "latency_ms": r.latency_ms, "failures": []}
    f = out["failures"]
    missing = [i for i in case.required if i not in included_ids]
    out["required_total"], out["required_kept"] = len(case.required), len(case.required) - len(missing)
    if case.fits and missing:
        f.append(f"required evidence missing: {missing}")
    persona_leaks = [i for i in included_ids if w.owner.get(i, req.persona_id) != req.persona_id] + [x for x in case.forbidden if x in included_ids and w.owner.get(x) != req.persona_id]
    persona_leaks += [s for s in case.forbidden_text if s in text and any(w.owner.get(i) != req.persona_id and s in w.contents.get(i, "") for i in case.forbidden)]
    temporal_leaks = [i for i in included_ids if i in w.said and (w.said[i] > as_of or w.recorded[i] > known)]
    temporal_leaks += [i for i in included_ids if i in w.recorded and i not in w.said and w.recorded[i] > known]
    forbidden_hit = [x for x in case.forbidden if x in included_ids] + [s for s in case.forbidden_text if s in text]
    out["persona_leakage"] = len(set(persona_leaks))
    out["temporal_leakage"] = len(set(temporal_leaks))
    if forbidden_hit:
        f.append(f"forbidden content present: {forbidden_hit}")
    # full rounds and silent truncation: every round shown is checked against the store
    silent = 0
    full_ok = []
    for cid, p in r.provenance_map.items():
        if p.get("kind") != "round":
            continue
        conv = w.api.store.get_message(req.persona_id, p["message_ids"][0]).conversation_id if p["message_ids"] else None
        if conv is None:
            continue
        visible = [m for m in w.rounds[conv] if w.said[m] <= as_of and w.recorded[m] <= known]
        shown = p["message_ids"]
        if len(shown) < len(visible):
            marked = p["key"] in r.omitted_message_ids and set(r.omitted_message_ids[p["key"]]) == set(visible) - set(shown) and "CONTEXT_OVERFLOW" in text
            if not marked:
                silent += 1
        if conv in case.full_rounds:
            full_ok.append(shown == visible)
    for conv in case.full_rounds:
        if not any(conv == (w.api.store.get_message(req.persona_id, p["message_ids"][0]).conversation_id if p.get("kind") == "round" and p["message_ids"] else None) for p in r.provenance_map.values()):
            full_ok.append(False)
    out["silent_truncation"] = silent
    out["full_rounds_expected"], out["full_rounds_ok"] = len(case.full_rounds), sum(full_ok)
    if case.full_rounds and sum(full_ok) < len(case.full_rounds):
        f.append("a round expected whole is not whole")
    out["token_budget_violation"] = int(r.token_usage_estimate > req.input_token_budget and r.status != cc.BUDGET_INSUFFICIENT) + int(r.status == cc.BUDGET_INSUFFICIENT and r.token_usage_estimate > req.input_token_budget and bool(r.provenance_map))
    if out["token_budget_violation"]:
        f.append("token budget exceeded")
    if r.status != case.expect_status:
        f.append(f"status {r.status} != {case.expect_status}")
    for wname in case.expect_warnings:
        if wname not in r.warnings:
            f.append(f"warning {wname} missing")
    if case.expect_boundary:
        st, op = case.expect_boundary
        if r.temporal_boundary["status"] != st or (op and r.temporal_boundary["operator"] != op):
            f.append(f"boundary {r.temporal_boundary['status']}/{r.temporal_boundary['operator']} != {st}/{op}")
    for key, rel in case.expect_relation.items():
        got = next((p.get("boundary_relation") for p in r.provenance_map.values() if p["key"] == key), None)
        if got != rel:
            f.append(f"relation of {key}: {got} != {rel}")
    attr_total = attr_ok = 0
    for mid, (snd, rcv) in case.attribution.items():
        attr_total += 1
        lines = text.splitlines()
        ok = False
        for i, ln in enumerate(lines):
            if f"{snd} → {rcv}：{w.contents[mid]}" in ln:                       # a line of a round or of the recent conversation
                ok = True
            elif ln == f"内容：{w.contents[mid]}" and i and f"发送方：{snd}" in lines[i - 1] and f"接收方：{rcv}" in lines[i - 1]:
                ok = True
        attr_ok += ok
        if not ok:
            f.append(f"attribution of {mid} wrong")
    out["attribution_total"], out["attribution_ok"] = attr_total, attr_ok
    if case.expect_order:
        pos = [text.find(w.contents[m]) for m in case.expect_order]
        if any(p < 0 for p in pos) or pos != sorted(pos):
            f.append("order wrong")
    occurrences = [m for p in r.provenance_map.values() for m in p.get("message_ids", [])]
    out["message_occurrences"], out["duplicate_occurrences"] = len(occurrences), len(occurrences) - len(set(occurrences))
    out["pass"] = not f and out["persona_leakage"] == 0 and out["temporal_leakage"] == 0 and silent == 0
    return out


def percentile(values: Sequence[float], q: float) -> float:
    v = sorted(values)
    return round(v[min(len(v) - 1, int(round(q * (len(v) - 1))))], 3) if v else 0.0


def evaluate(repeats: int = 5) -> dict:
    w = build_world()
    orch = ContextOrchestrator(w.api, speakers=SPEAKERS)
    rows, latencies = [], []
    for case in cases(w):
        r = orch.assemble(case.request)
        for _ in range(repeats - 1):
            started = time.perf_counter()
            orch.assemble(case.request)
            latencies.append((time.perf_counter() - started) * 1000)
        latencies.append(r.latency_ms)
        rows.append({**check(w, case, r), "excluded": r.excluded_evidence_ids, "selected": r.selected_evidence_ids, "boundary": r.temporal_boundary})
    applicable_full = [x for x in rows if x["full_rounds_expected"]]
    applicable_req = [x for x in rows if x["required_total"] and next(c for c in cases(w) if c.case_id == x["case_id"]).fits]
    attr = sum(x["attribution_total"] for x in rows)
    occ = sum(x["message_occurrences"] for x in rows)
    metrics = {
        "TOTAL_CASES": len(rows), "CASES_PASSED": sum(x["pass"] for x in rows),
        "FULL_TURN_PRESERVATION_RATE": round(sum(x["full_rounds_ok"] for x in applicable_full) / max(sum(x["full_rounds_expected"] for x in applicable_full), 1), 4),
        "REQUIRED_EVIDENCE_PRESERVATION_RATE": round(sum(x["required_kept"] for x in applicable_req) / max(sum(x["required_total"] for x in applicable_req), 1), 4),
        "SOURCE_ATTRIBUTION_ACCURACY": round(sum(x["attribution_ok"] for x in rows) / max(attr, 1), 4),
        "PERSONA_LEAKAGE_COUNT": sum(x["persona_leakage"] for x in rows), "TEMPORAL_LEAKAGE_COUNT": sum(x["temporal_leakage"] for x in rows),
        "SILENT_TRUNCATION_COUNT": sum(x["silent_truncation"] for x in rows), "TOKEN_BUDGET_VIOLATIONS": sum(x["token_budget_violation"] for x in rows),
        "CONTEXT_DUPLICATE_RATE": round(sum(x["duplicate_occurrences"] for x in rows) / max(occ, 1), 4),
        "CONTEXT_ASSEMBLY_LATENCY_P50": percentile(latencies, 0.5), "CONTEXT_ASSEMBLY_LATENCY_P95": percentile(latencies, 0.95),
        "TOKEN_COUNT_MODE": rows[0]["token_mode"] if rows else None,
    }
    # D6 before and after, on every round of the persona
    d6 = []
    for conv, ids in w.rounds.items():
        persona = w.owner[ids[0]]
        as_of = t(20, 0)
        visible = [m for m in ids if w.said[m] <= as_of]
        legacy = legacy_round(w, persona, conv, as_of) or ""
        legacy_kept = sum(1 for m in visible if w.contents[m] in legacy)
        r = orch.assemble(ContextRequest(persona_id=persona, question="回顾", as_of=as_of, hits=(message_hit(w, ids[0]),), input_token_budget=20000))
        new_kept = len(next((p["message_ids"] for p in r.provenance_map.values() if p.get("kind") == "round"), []))
        d6.append({"conversation_id": conv, "messages": len(visible), "legacy_shown": legacy_kept, "orchestrator_shown": new_kept, "legacy_silently_dropped": len(visible) - legacy_kept})
    d6_summary = {"rounds": len(d6), "rounds_over_3": sum(x["messages"] > 3 for x in d6), "legacy_rounds_truncated": sum(x["legacy_silently_dropped"] > 0 for x in d6),
                  "legacy_messages_dropped": sum(x["legacy_silently_dropped"] for x in d6), "orchestrator_rounds_truncated": sum(x["orchestrator_shown"] < x["messages"] for x in d6)}
    return {"metrics": metrics, "rows": rows, "d6": {"summary": d6_summary, "rounds": d6}}
