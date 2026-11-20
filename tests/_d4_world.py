"""Public synthetic memory for the Simple Context v1.1 D4 tests: one invented persona (林夏, ``syn.da``), every D4 evidence state built from
explicit Core records (no extractor, no model). Phone numbers / emails are invented test values in reserved-looking ranges."""
from datetime import datetime, timedelta, timezone

from stillhere.contracts import Cardinality, DataOrigin, MemoryKind, MemoryRecord, Provenance, SourceKind, SourceRef
from stillhere.memory.api import MemoryAPI

TZ = timezone(timedelta(hours=8))
P = "syn.da"
NOW = datetime(2026, 10, 10, 20, 0, tzinfo=TZ)
SRC = "d4ctx.synthetic"
FRIEND_PHONE = "13900000000"
OWN_PHONE = "13800000001"
EMAIL = "linxia@example.com"


def t(m, d, h=10):
    return datetime(2026, m, d, h, tzinfo=TZ)


def rec(mid, subject, attribute, value, statement, recorded, *, kind=MemoryKind.FACT, card=Cardinality.SINGLE, vf=None, vt=None, conf=1.0):
    return MemoryRecord(memory_id=mid, persona_id=P, kind=kind, subject=subject, attribute=attribute, value=value, statement=statement,
                        provenance=(Provenance(SRC, locator=mid),), origin=DataOrigin.SYNTHETIC, recorded_at=recorded, valid_from=vf, valid_to=vt,
                        confidence=conf, cardinality=card)


def event(eid, desc, status, recorded, *, actor="self", planned=None, done=None):
    out = [rec(f"{eid}.d", f"event:{eid}", "event.description", desc, f"事件：{desc}", recorded, kind=MemoryKind.EVENT),
           rec(f"{eid}.s", f"event:{eid}", "event.status", status, f"事件状态：{status}", recorded, kind=MemoryKind.EVENT),
           rec(f"{eid}.a", f"event:{eid}", "event.actor", actor, f"事件当事人：{actor}", recorded, kind=MemoryKind.EVENT)]
    if planned:
        out.append(rec(f"{eid}.p", f"event:{eid}", "event.planned_time", planned, f"事件计划时间：{planned}", recorded, kind=MemoryKind.EVENT))
    if done:
        out.append(rec(f"{eid}.t", f"event:{eid}", "event.time", done, f"事件时间：{done}", recorded, kind=MemoryKind.EVENT))
    return out


def build(clock_at=NOW):
    api = MemoryAPI(":memory:", clock=lambda: clock_at)
    api.register_source(SourceRef(source_id=SRC, kind=SourceKind.SCRIPT, origin=DataOrigin.SYNTHETIC, description="Simple Context v1.1 D4 test world"))
    recs = [
        rec("f.home.old", "persona", "lives_in", "武汉", "本人住在武汉", t(1, 5), vf=t(1, 1), vt=t(8, 31)),
        rec("f.home", "persona", "lives_in", "杭州西湖区", "本人住在杭州西湖区", t(9, 1), vf=t(9, 1)),
        rec("f.job.wrong", "persona", "job", "护士", "本人的职业是护士", t(9, 2)),
        rec("f.work.a", "persona", "works_at", "一家医院", "本人在一家医院工作", t(9, 3)),
        rec("f.work.b", "persona", "works_at", "一家诊所", "本人在一家诊所工作", t(9, 4)),
        rec("f.rel", "persona", "relationship", "室友 小邹", "本人的室友 小邹", t(9, 5), card=Cardinality.MULTI, kind=MemoryKind.RELATIONSHIP),
        rec("f.phone.friend", "person:小邹", "phone", FRIEND_PHONE, f"小邹的手机号是{FRIEND_PHONE}", t(9, 6)),
        rec("f.phone.own", "persona", "phone", OWN_PHONE, f"本人的手机号是{OWN_PHONE}", t(9, 6)),
        rec("f.email", "persona", "email", EMAIL, f"本人的邮箱是{EMAIL}", t(9, 6)),
        rec("f.cat", "persona", "likes", "爬山", "本人喜欢爬山", t(9, 7), card=Cardinality.MULTI, kind=MemoryKind.PREFERENCE),
    ]
    recs += event("e.hzhotel", "订杭州的酒店", "CANCELLED", t(9, 20))
    recs += event("e.cdtrip", "去成都出差", "PLANNED", t(10, 1), planned="2026-10-20")
    recs += event("e.linmove", "搬去成都", "PLANNED", t(10, 2), actor="person:小林", planned="2026-11")
    recs += event("e.emei", "去爬峨眉山", "COMPLETED", t(10, 8), done="2026-10-07")
    recs += event("e.hk", "去香港看演唱会", "PLANNED", t(10, 3))
    api.add_facts(P, recs)
    fixed = rec("f.job", "persona", "job", "药剂师", "本人的职业是药剂师", t(9, 9))
    api.correct(P, "f.job.wrong", fixed, "correction from syn.da.chat.x.0")          # the Writer's marker for an explicit correction
    return api
