"""Simple Context v1.1 — Core memory -> D4 Evidence View states (at query_time) and the disclosure policy / redaction (public synthetic)."""
from datetime import timedelta

import pytest

import _d4_world as W
from stillhere.runtime import pipeline as rp
from stillhere.session import d4_evidence as d4e
from stillhere.session import d4_renderer as dr
from stillhere.session import disclosure as ds

V2 = {"GROUNDING_V2_ENABLED": "1"}


def view(q, now=W.NOW, api=None, grant=True):
    api = api or W.build()
    pol = ds.PolicyResolver(api)
    if grant:
        pol.set(W.P, "persona", "email", "ALLOW", W.t(9, 1))
    m, _, _, g, _ = rp.ground(api, W.P, q, now, env=V2)
    return d4e.build(api, W.P, m, g, now, pol), api


def by_text(v, needle):
    return next(e for e in v.evidence if needle in e.evidence_text)


def test_current_and_historical_are_computed_at_query_time():
    v, _ = view("我现在住哪？")
    assert by_text(v, "杭州西湖区").status == "CURRENT" and by_text(v, "武汉").status == "HISTORICAL"
    early, _ = view("我现在住哪？", now=W.t(8, 1))                         # asked in August: Wuhan was still current
    assert by_text(early, "武汉").status == "CURRENT"


def test_corrected_keeps_the_wrong_value_only_as_replaces():
    v, _ = view("我是做什么的？")
    e = by_text(v, "药剂师")
    assert e.status == "CORRECTED" and e.replaces == "本人的职业是护士"
    assert not any(x.evidence_text == "本人的职业是护士" for x in v.evidence)     # the superseded record is never its own item


def test_conflict_group_is_complete():
    v, _ = view("我在哪工作？")
    conflicted = [e for e in v.evidence if e.status == "CONFLICTED"]
    assert len(conflicted) == 2 and len({e.conflict_group for e in conflicted}) == 1 and {e.value for e in conflicted} == {"一家医院", "一家诊所"}


def test_cancelled_upcoming_completed_and_third_party_events():
    v, _ = view("杭州的酒店还订着吗？")
    hz = by_text(v, "杭州的酒店")
    assert hz.status == "CANCELLED" and hz.note
    v2, _ = view("我什么时候去成都出差？")
    cd = by_text(v2, "去成都出差")
    assert cd.status == "UPCOMING" and cd.valid_from == "2026-10-20" and cd.subject == "本人"
    lin = by_text(v2, "搬去成都")
    assert lin.subject == "小林" and lin.subject_type == "friend" and lin.status == "UPCOMING"
    after, _ = view("我什么时候去成都出差？", now=W.NOW + timedelta(days=15))      # after 10-20 the plan is no longer "upcoming"
    assert by_text(after, "去成都出差").status == "CURRENT"
    v3, _ = view("我去爬峨眉山了吗？")
    assert by_text(v3, "峨眉山").status == "CURRENT" and by_text(v3, "峨眉山").valid_from == "2026-10-07"


def test_unknown_slots_come_from_the_grounding():
    api = W.build()
    pol = ds.PolicyResolver(api)
    m, _, _, g, _ = rp.ground(api, W.P, "我养猫吗？", W.NOW, env=V2, answerer=lambda q, t, ev: {"answer": "根据现有记忆，无法确定。", "abstain": True})
    v = d4e.build(api, W.P, m, g, W.NOW, pol)
    assert v.unknown and all(isinstance(u, dr.UnknownSlot) for u in v.unknown)


def test_runtime_ids_never_reach_items():
    v, _ = view("我是做什么的？")
    blob = repr([e.to_json() for e in v.evidence])
    assert "f.job" not in blob and "f.work" not in blob and v.id_map["F1"]
    assert all(e.fact_id.startswith("F") for e in v.evidence)


# ---------------------------------------------------------------- disclosure
def test_policy_categories_and_fail_closed_defaults():
    api = W.build()
    pol = ds.PolicyResolver(api)
    friend = pol.decide(W.P, "person:小邹", "phone", W.FRIEND_PHONE, f"小邹的手机号是{W.FRIEND_PHONE}", W.NOW)
    own = pol.decide(W.P, "persona", "phone", W.OWN_PHONE, f"本人的手机号是{W.OWN_PHONE}", W.NOW)
    idn = pol.decide(W.P, "persona", "", "110101199001010000", "本人的身份证号是110101199001010000", W.NOW)
    plain = pol.decide(W.P, "persona", "lives_in", "杭州西湖区", "本人住在杭州西湖区", W.NOW)
    assert (friend.policy, own.policy, idn.policy, plain.policy) == (ds.PROTECTED, ds.UNCERTAIN, ds.PROTECTED, ds.ALLOWED)
    assert W.FRIEND_PHONE in friend.guard_terms and not plain.guard_terms
    assert W.FRIEND_PHONE not in repr(friend.trace())                       # traces keep hashes only


def test_grant_deny_and_revocation_are_time_bound():
    api = W.build()
    pol = ds.PolicyResolver(api)
    pol.set(W.P, "persona", "phone", "ALLOW", W.t(9, 1), valid_to=W.t(10, 1))
    d = lambda at: pol.decide(W.P, "persona", "phone", W.OWN_PHONE, f"本人的手机号是{W.OWN_PHONE}", at).policy
    assert d(W.t(9, 15)) == ds.ALLOWED and d(W.NOW) == ds.UNCERTAIN          # the grant expired on 10-01
    pol.set(W.P, "persona", "phone", "ALLOW", W.t(10, 5))
    pol.set(W.P, "persona", "phone", "REVOKE", W.t(10, 8))
    assert d(W.t(10, 6)) == ds.ALLOWED and d(W.NOW) == ds.PROTECTED           # revoked consent wins


def test_redaction_leaves_no_raw_value_in_any_model_visible_field():
    v, _ = view("小邹手机号多少？")
    msgs = dr.render_messages(profile=None, examples=[], evidence=v.evidence, unknown=v.unknown, now="2026-10-10",
                              conversation=[{"speaker": "partner", "text": "小邹手机号多少？"}])
    joined = "\n".join(m["content"] for m in msgs)
    assert W.FRIEND_PHONE not in joined and W.OWN_PHONE not in joined and "不可透露" in joined and "未确认能否透露" in joined
    assert W.FRIEND_PHONE in v.guard_terms and W.OWN_PHONE in v.guard_terms
    for e in v.evidence:
        if e.disclosure_policy != ds.ALLOWED:
            assert e.value == "（不显示）" and not ds.raw_terms(e.evidence_text + (e.replaces or "") + (e.note or ""))


def test_allowed_contact_is_said_not_refused():
    v, _ = view("我邮箱是什么？")
    e = by_text(v, "邮箱")
    assert e.disclosure_policy == ds.ALLOWED and W.EMAIL in e.evidence_text
    no_grant, _ = view("我邮箱是什么？", grant=False)
    assert by_text(no_grant, "邮箱").disclosure_policy == ds.UNCERTAIN


def test_scrub_removes_guarded_values_from_nested_traces():
    tr = {"a": [f"x{W.FRIEND_PHONE}y", {"b": W.OWN_PHONE}], "c": 3}
    out = ds.scrub(tr, [W.FRIEND_PHONE, W.OWN_PHONE])
    assert W.FRIEND_PHONE not in repr(out) and W.OWN_PHONE not in repr(out) and out["c"] == 3


# ---------------------------------------------------------------- categories(): two false positives fixed
def test_an_18_digit_id_number_is_not_also_a_bank_card():
    idn = "110101199001010000"
    assert ds.categories(idn, f"本人的身份证号是{idn}") == ["id_number"]
    assert ds.categories("11010119900101000X", "本人的身份证号是11010119900101000X") == ["id_number"]
    assert ds.raw_terms(idn) == [idn]
    assert ds.categories("6222020200112233445", "一张卡") == ["bank_card"]          # a 19-digit card is still a bank card
    assert ds.categories("6222020200112233", "一张卡") == ["bank_card"]             # and a 16-digit one


def test_value_and_statement_are_matched_separately():
    assert ds.categories("3", "室友小王") == []                                      # was address_detail via the joined "3 室友小王"
    assert ds.categories("13900000000", "室友小邹的手机号") == ["phone"]             # was phone + address_detail
    assert "3 室" not in "".join(ds.raw_terms("3", "室友小王")) and ds.raw_terms("3", "室友小王") == []
    assert ds.categories("8号楼302室", "本人住在8号楼302室") == ["address_detail"]   # a real address inside one text is still found
    pol = ds.PolicyResolver(W.build())
    d = pol.decide(W.P, "person:小王", "relationship", "3", "室友小王", W.NOW)
    assert d.policy == ds.ALLOWED and not d.categories


def test_salary_card_is_not_income():
    assert ds.categories("", "本人用招商银行的工资卡") == []                                   # a card, no amount: not income
    assert ds.categories("6222020200112233445", "本人的工资卡") == ["bank_card"]               # its number: a bank card only
    assert ds.categories("", "本人的工资卡号是6222020200112233") == ["bank_card"]
    assert ds.categories("", "本人的工资是一万") == ["income"]                                 # a real salary statement stays income
    assert ds.categories("", "本人月薪8000") == ["income"]
    assert ds.categories("", "工资卡里的工资还没发") == ["income"]                             # the exclusion removes only the phrase "工资卡"
