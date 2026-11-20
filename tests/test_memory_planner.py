"""QueryPlanner and the query-plan contracts (Day 7, Query Understanding V2). Synthetic fixture persona only.

Development / regression tests: the sentences are written for the rules, they are not an independent holdout."""
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from conftest import at, make_memory
from memory_support import QIAOMO, QIAOMO_NOW, open_store, qiaomo_store, snapshot
from stillhere.contracts import (AtomicQuery, AtomicResult, BatchStatus, DialogueFocus, IntentStatus, MemoryQuery,
                                 Provenance, QueryDependency, QueryPlan, RetrievalBatch, RetrievalStatus, SpeechAct,
                                 SubjectStatus, Sufficiency, TimeScope, ValidationError)
from stillhere.contracts.planning import AnswerLevel, DependencyKind, DependencyOutcome, EvidenceState, FocusStatus
from stillhere.memory import QueryPlanner, SlotRetriever

ROOT = Path(__file__).resolve().parents[1]
NAMES = {QIAOMO: ("乔沫",)}
OK, NO_MATCH, AMBIGUOUS = RetrievalStatus.OK, RetrievalStatus.NO_MATCH, RetrievalStatus.AMBIGUOUS_SUBJECT


@pytest.fixture(scope="module")
def planner():
    return QueryPlanner(persona_names=NAMES)


def plan_of(planner, text, **kw):
    return planner.plan(MemoryQuery(persona_id=QIAOMO, text=text, **kw))


@pytest.fixture(params=[None, False], ids=["fts5", "scan"])
def store(request):
    s = qiaomo_store(fts=request.param)
    yield s
    s.close()


@pytest.fixture
def ask(store):
    r = SlotRetriever(store, persona_names=NAMES, clock=lambda: QIAOMO_NOW)
    return lambda text, **kw: r.retrieve(QIAOMO, text, **kw)


def ids(result):
    return [h.record.memory_id for h in result.hits]


def user_turn(text):
    return dict(context=(text,), context_roles=("user",))


def assert_no_cross_atom_records(result):
    """Every record of an atom belongs to that atom's subject, and every hit belongs to an atom."""
    batch = result.batch
    by_id = {h.record.memory_id: h.record for h in result.hits}
    for a in batch.atoms:
        for mid in a.memory_ids:
            assert by_id[mid].subject == a.subject, (a.query_id, mid)
    assert {m for a in batch.atoms for m in a.memory_ids} == set(by_id)


# ---------------------------------------------------------------- contracts

def test_plan_round_trips_and_its_id_is_stable(planner):
    plan = plan_of(planner, "你住哪？我住哪？")
    assert QueryPlan.from_json(plan.to_json()) == plan
    assert plan_of(planner, "你住哪？我住哪？").plan_id == plan.plan_id
    assert plan_of(planner, "你住哪？他住哪？").plan_id != plan.plan_id
    assert plan.planner == "local_rules_v1" and plan.version == "query_plan_v1"
    assert [a.query_id for a in plan.atoms] == ["q1", "q2"]


def test_plans_do_not_depend_on_hash_order():
    code = ("import sys; sys.path.insert(0, 'tests'); from memory_support import QIAOMO; "
            "from stillhere.contracts import MemoryQuery; from stillhere.memory import QueryPlanner; "
            "p = QueryPlanner(persona_names={QIAOMO: ('乔沫',)}); "
            "texts = ['你喜欢做什么？在哪上班？', '你住哪？我住哪？', '你和我谁生日更早', '你做什么工作？你爸爸呢？', "
            "'我差点忘了问，你喜欢什么', '你喜欢我，还是喜欢攀岩']; "
            "print([p.plan(MemoryQuery(persona_id=QIAOMO, text=t)).to_json() for t in texts] + "
            "[p.plan(MemoryQuery(persona_id=QIAOMO, text='那你呢', context=['我住在海港市'], "
            "context_roles=['user'])).to_json()])")
    outs = {subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True,
                           env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONIOENCODING": "utf-8"}).stdout for seed in ("0", "1", "12345")}
    assert len(outs) == 1


def atom(**kw):
    base = dict(query_id="q1", parts=(0,), span="你住哪", speaker="user", act=SpeechAct.QUESTION, subject="persona",
                subject_status=SubjectStatus.RESOLVED, subject_source="query", attributes=("home_city",),
                intent=IntentStatus.SLOT, intent_source="cue", time_scope=TimeScope.CURRENT,
                time_scope_source="default", retrievable=True)
    base.update(kw)
    return AtomicQuery(**base)


@pytest.mark.parametrize("kw", [
    dict(subject=None),                                                    # resolved without a subject
    dict(subject_status=SubjectStatus.AMBIGUOUS, ambiguity="ambiguous:pronoun", retrievable=False),
    dict(subject_source=None),
    dict(subject_source="guess"),
    dict(subject="Persona Mother"),                                        # not an id
    dict(attributes=()),                                                   # slot intent without attributes
    dict(intent=IntentStatus.KEYWORD),                                     # attributes without slot intent
    dict(attributes=("home_city", "home_city")),
    dict(speaker="assistant"),
    dict(retrievable=False),
    dict(subject=None, subject_source=None, subject_status=SubjectStatus.UNOWNED, retrievable=False),  # no reason
    dict(intent=IntentStatus.UNSUPPORTED, attributes=(), intent_source=None, retrievable=False),
    dict(period_unplaced=True),                                            # unplaced period cannot be retrievable
    dict(time_scope_source="model"),
])
def test_atomic_query_rejects_inconsistent_fields(kw):
    with pytest.raises(ValidationError):
        atom(**kw)


def test_atomic_query_accepts_closed_atoms_with_a_reason():
    a = atom(subject=None, subject_source=None, subject_status=SubjectStatus.UNOWNED, retrievable=False,
             ambiguity="unowned:妈妈")
    assert AtomicQuery.from_dict(a.to_dict()) == a
    u = atom(intent=IntentStatus.UNSUPPORTED, attributes=(), intent_source=None, retrievable=False,
             ambiguity="unsupported:social")
    assert not u.retrievable


def test_dependency_focus_plan_and_batch_validation(planner):
    with pytest.raises(ValidationError):
        QueryDependency(kind=DependencyKind.INDEPENDENT, query_ids=("q1", "q2"))
    with pytest.raises(ValidationError):
        QueryDependency(kind=DependencyKind.DEPENDENT_COMPARE, query_ids=("q1",))
    with pytest.raises(ValidationError):
        QueryDependency(kind=DependencyKind.UNKNOWN, query_ids=("q1", "q2"), supported=True, relation="earlier")
    with pytest.raises(ValidationError):
        DialogueFocus(status=FocusStatus.ACTIVE, reason="carried")                    # no attribute
    with pytest.raises(ValidationError):
        DialogueFocus(status=FocusStatus.INVALID, reason="quoted", attribute="home_city")
    with pytest.raises(ValidationError):
        DialogueFocus(status=FocusStatus.ACTIVE, reason="carried", attribute="home_city", turn_offset=1,
                      speaker="narrator")
    plan = plan_of(planner, "你和我谁生日更早")
    with pytest.raises(ValidationError):
        replace(plan, dependencies=(QueryDependency(kind=DependencyKind.DEPENDENT_COMPARE, query_ids=("q1", "q9")),))
    with pytest.raises(ValidationError):
        replace(plan, version="query_plan_v0")
    with pytest.raises(ValidationError):
        replace(plan, atoms=())
    with pytest.raises(ValidationError):
        replace(plan, confidence="medium")
    ok = AtomicResult(query_id="q1", status=OK, sufficiency=Sufficiency.CONFLICT, subject="persona",
                      memory_ids=("mem_birthday_nov",))
    missing = AtomicResult(query_id="q2", status=NO_MATCH, sufficiency=Sufficiency.NO_RECORD, subject="user",
                           missing="no_record")
    dep = plan.dependencies[0]
    outcome = DependencyOutcome(
        kind=dep.kind, query_ids=dep.query_ids, evidence=EvidenceState.MISSING, answer=AnswerLevel.SAFE_PARTIAL)
    RetrievalBatch(plan=plan, status=BatchStatus.PARTIAL_SUCCESS, atoms=(ok, missing), dependencies=(outcome,))
    with pytest.raises(ValidationError):                       # status must follow the answered count
        RetrievalBatch(plan=plan, status=BatchStatus.COMPLETE_SUCCESS, atoms=(ok, missing), dependencies=(outcome,))
    with pytest.raises(ValidationError):                       # one result per atom, in order
        RetrievalBatch(plan=plan, status=BatchStatus.PARTIAL_SUCCESS, atoms=(missing, ok), dependencies=(outcome,))
    with pytest.raises(ValidationError):                       # one outcome per dependency
        RetrievalBatch(plan=plan, status=BatchStatus.PARTIAL_SUCCESS, atoms=(ok, missing))
    with pytest.raises(ValidationError):
        replace(outcome, answer=AnswerLevel.COMPLETE)          # missing evidence never gives a complete answer
    with pytest.raises(ValidationError):
        AtomicResult(query_id="q1", status=OK, sufficiency=Sufficiency.SUFFICIENT)          # ok without records
    with pytest.raises(ValidationError):
        AtomicResult(query_id="q1", status=NO_MATCH, sufficiency=Sufficiency.NO_RECORD)     # missing without why
    with pytest.raises(ValidationError):
        AtomicResult(query_id="q1", status=RetrievalStatus.ERROR, sufficiency=Sufficiency.NO_RECORD, missing="x")


def test_an_empty_message_has_no_plan(planner, ask):
    with pytest.raises(ValueError):
        plan_of(planner, "   ")
    assert ask("").status is NO_MATCH


# ---------------------------------------------------------------- slot intents

@pytest.mark.parametrize("text, attributes", [
    ("你喜欢做什么？在哪上班？", ("hobby", "occupation")),
    ("我差点忘了问，你喜欢什么", ("hobby",)),
    ("你有什么爱好", ("hobby",)),
    ("你平时喜欢做什么", ("hobby",)),
    ("你喜欢哪些活动", ("hobby",)),
    ("你周末喜欢做些什么", ("hobby",)),                       # a habitual time word between owner and question
    ("你在哪里上班", ("occupation",)),
    ("你是做什么工作的", ("occupation",)),
])
def test_activity_and_work_questions_name_their_slots(planner, text, attributes):
    plan = plan_of(planner, text)
    asked = [a for a in plan.atoms if a.act is not SpeechAct.STATEMENT]
    assert tuple(x for a in asked for x in a.attributes) == attributes
    assert all(a.subject == "persona" and a.retrievable for a in asked)


def test_x39_x44_x50_shapes_are_retrieved(ask):
    r = ask("你喜欢做什么？在哪上班？")
    assert r.batch.status is BatchStatus.COMPLETE_SUCCESS
    assert ids(r) == ["mem_hobby_baking", "mem_hobby_climbing", "mem_job_cafe"]
    assert [a.memory_ids for a in r.batch.atoms] == [("mem_hobby_baking", "mem_hobby_climbing"), ("mem_job_cafe",)]
    r = ask("我差点忘了问，你喜欢什么")
    assert (r.subject, ids(r)) == ("persona", ["mem_hobby_baking", "mem_hobby_climbing"])
    r = ask("那你呢", **user_turn("我住在海港市"))
    assert (r.subject, ids(r)) == ("persona", ["mem_home_beichuan"])
    assert r.batch.plan.atoms[0].intent_source == "focus"


@pytest.mark.parametrize("text, intent, reason", [
    ("你喜欢我吗", IntentStatus.UNSUPPORTED, "unsupported:social"),
    ("你喜欢我，还是喜欢攀岩", IntentStatus.UNKNOWN, "unknown:mixed_social"),
    ("乔沫喜欢谁", IntentStatus.UNSUPPORTED, "unsupported:social"),
])
def test_social_likes_are_not_hobbies_and_take_no_nearest_slot(planner, ask, text, intent, reason):
    [a] = plan_of(planner, text).atoms
    assert (a.intent, a.ambiguity, a.attributes, a.retrievable) == (intent, reason, (), False)
    r = ask(text)
    assert (r.status, r.hits, r.slots) == (NO_MATCH, (), ())


def test_liking_an_object_is_a_keyword_lookup_not_a_hobby(planner, ask):
    [a] = plan_of(planner, "你喜欢吃香菜吗").atoms
    assert a.intent is IntentStatus.KEYWORD
    assert ids(ask("你喜欢吃香菜吗")) == ["mem_pref_coriander"]
    [a] = plan_of(planner, "你喜欢这首歌吗").atoms
    assert a.intent is IntentStatus.KEYWORD and "hobby" not in a.attributes
    assert ask("你喜欢这首歌吗").status is NO_MATCH


def test_a_user_statement_is_not_a_question_about_the_persona(planner, ask):
    [a] = plan_of(planner, "我喜欢攀岩").atoms
    assert a.act is SpeechAct.STATEMENT and a.subject == "user"
    r = ask("我喜欢攀岩")
    assert r.status is NO_MATCH and "mem_hobby_climbing" not in ids(r)


# ---------------------------------------------------------------- dialogue focus

@pytest.mark.parametrize("text, ctx, roles, subject, attribute, scope, expected", [
    ("那你呢", ("我住在海港市",), ("user",), "persona", "home_city", TimeScope.CURRENT, ["mem_home_beichuan"]),
    ("那你呢", ("我住在海港市", "真不错"), ("user", "persona"), "persona", "home_city", TimeScope.CURRENT,
     ["mem_home_beichuan"]),
    ("那你呢？", ("我现在住在海港市",), ("user",), "persona", "home_city", TimeScope.CURRENT, ["mem_home_beichuan"]),
    ("以前呢", ("你现在住哪",), ("user",), "persona", "home_city", TimeScope.HISTORICAL,
     ["mem_home_nanyu", "mem_home_beichuan"]),
    ("现在呢", ("你以前住哪", "我以前住南屿"), ("user", "persona"), "persona", "home_city", TimeScope.CURRENT,
     ["mem_home_beichuan"]),
    ("你也是吗", ("我在咖啡馆上班",), ("user",), "persona", "occupation", TimeScope.CURRENT, ["mem_job_cafe"]),
    ("你呢", ("我生日在十月",), ("user",), "persona", "birthday_month", TimeScope.CURRENT,
     ["mem_birthday_nov", "mem_birthday_oct"]),
    ("我呢", ("我住在北川",), ("persona",), "user", "home_city", TimeScope.CURRENT, []),        # role swap
])
def test_an_elliptical_question_carries_only_the_attribute(ask, text, ctx, roles, subject, attribute, scope,
                                                            expected):
    r = ask(text, context=ctx, context_roles=roles)
    [a] = r.batch.plan.atoms
    assert (a.subject, a.attributes, a.intent_source, a.time_scope) == (subject, (attribute,), "focus", scope)
    assert r.batch.plan.focus.status is FocusStatus.ACTIVE and r.batch.plan.focus.attribute == attribute
    assert ids(r) == expected
    assert all(h.record.subject == subject for h in r.hits)


@pytest.mark.parametrize("text, ctx, roles, reason", [
    ("那你呢", ("我住在海港市",), (), "no_roles"),
    ("那你呢", ("他说“我住在海港市”",), ("user",), "quoted"),
    ("那你呢", ("我住在海港市，在银行上班",), ("user",), "multiple"),
    ("那你呢", ("今天天气不错",), ("user",), "no_topic"),
    ("那你呢", ("我住在海港市", "你周末一般干嘛"), ("user", "persona"), "switched"),
    ("那你呢", ("你喜欢我吗",), ("persona",), "unclear"),
    ("那你呢", ("我住在海港市", "我住北川", "好的"), ("user", "persona", "user"), "no_topic"),   # too far back
])
def test_the_focus_is_dropped_when_the_topic_is_not_clear(planner, ask, text, ctx, roles, reason):
    focus = plan_of(planner, text, context=ctx, context_roles=roles).focus
    assert (focus.status, focus.reason) == (FocusStatus.INVALID, reason)
    r = ask(text, context=ctx, context_roles=roles)
    assert r.hits == () and r.slots == ()


@pytest.mark.parametrize("text, status, reason", [
    ("那她呢", AMBIGUOUS, "ambiguous:pronoun"),                # third person without a referent
    ("那你们呢", AMBIGUOUS, "ambiguous:ellipsis_owner"),       # plural
    ("那你妈妈呢", NO_MATCH, "unowned:妈妈"),                  # someone with no records
])
def test_a_carried_attribute_never_picks_an_owner(ask, text, status, reason):
    r = ask(text, **user_turn("我住在海港市"))
    [a] = r.batch.plan.atoms
    assert (r.status, r.hits, a.attributes, a.ambiguity) == (status, (), ("home_city",), reason)


def test_a_new_explicit_slot_and_an_explicit_subject_beat_the_focus(ask):
    r = ask("那你住哪", **user_turn("我在银行上班"))
    assert ids(r) == ["mem_home_beichuan"]
    r = ask("那你呢", subject="cat", **user_turn("我住在海港市"))
    assert (r.status, r.hits) == (AMBIGUOUS, ())
    assert r.batch.plan.atoms[0].ambiguity == "ambiguous:focus_subject_conflict"
    assert ids(ask("那你呢", subject="persona", **user_turn("我住在海港市"))) == ["mem_home_beichuan"]


def test_the_focus_is_never_written(store):
    before = snapshot(store)
    r = SlotRetriever(store, persona_names=NAMES, clock=lambda: QIAOMO_NOW)
    r.retrieve(QIAOMO, "那你呢", **user_turn("我住在海港市"))
    r.retrieve(QIAOMO, "那你呢")                                # no context: nothing is remembered
    assert snapshot(store) == before
    assert r.retrieve(QIAOMO, "那你呢").hits == ()


# ---------------------------------------------------------------- atoms, partial success, comparisons

@pytest.mark.parametrize("text, answered, closed, missing", [
    ("你住哪？我住哪？", ["mem_home_beichuan"], "user", "no_record"),
    ("你住哪？那我呢", ["mem_home_beichuan"], "user", "no_record"),
    ("你做什么工作？你爸爸呢？", ["mem_job_cafe"], None, "unowned:爸爸"),
    ("你和我分别住哪里", ["mem_home_beichuan"], "user", "no_record"),
    ("乔沫的猫叫什么，她住哪", ["mem_cat_name_v2"], None, "ambiguous:pronoun"),
])
def test_one_atom_answered_and_one_not_is_a_partial_success(ask, text, answered, closed, missing):
    r = ask(text)
    assert r.status is OK and r.batch.status is BatchStatus.PARTIAL_SUCCESS
    first, second = r.batch.atoms
    assert (first.status, list(first.memory_ids)) == (OK, answered)
    assert (second.status is not OK, second.subject, second.missing) == (True, closed, missing)
    assert ids(r) == answered and r.subject is None
    assert_no_cross_atom_records(r)


def test_a_comparison_reports_what_evidence_is_missing(ask, planner):
    plan = plan_of(planner, "你和我谁生日更早")
    [dep] = plan.dependencies
    assert (dep.kind, dep.query_ids, dep.relation, dep.supported) == (DependencyKind.DEPENDENT_COMPARE,
                                                                      ("q1", "q2"), "earlier", True)
    r = ask("你和我谁生日更早")
    [outcome] = r.batch.dependencies
    assert (outcome.evidence, outcome.answer) == (EvidenceState.MISSING, AnswerLevel.SAFE_PARTIAL)
    assert [a.sufficiency for a in r.batch.atoms] == [Sufficiency.CONFLICT, Sufficiency.NO_RECORD]
    assert_no_cross_atom_records(r)
    plan = plan_of(planner, "你和我谁住得远")
    assert plan.dependencies[0].supported is False
    r = ask("你和我谁住得远")
    assert r.batch.dependencies[0].evidence is EvidenceState.UNSUPPORTED_COMPARISON
    assert r.batch.dependencies[0].answer is AnswerLevel.SAFE_PARTIAL


@pytest.mark.parametrize("text, owners, relation", [
    ("你生日比我早还是比我晚", ["persona", "user"], "earlier"),
    ("我的生日比你晚吗", ["user", "persona"], "later"),
])
def test_a_comparison_with_bi_gets_one_atom_per_owner(ask, planner, text, owners, relation):
    """Phase 5B: "A 比 B" is a comparison as much as "A 和 B 谁": both owners' facts, and no conclusion while one
    side is missing or disputed."""
    plan = plan_of(planner, text)
    assert [a.subject for a in plan.atoms] == owners
    [dep] = plan.dependencies
    assert (dep.kind, dep.query_ids, dep.relation, dep.supported) == (DependencyKind.DEPENDENT_COMPARE,
                                                                      ("q1", "q2"), relation, True)
    r = ask(text)
    assert r.batch.dependencies[0].evidence is EvidenceState.MISSING
    assert_no_cross_atom_records(r)


@pytest.mark.parametrize("text", ["你比我大吗", "你家比我家远吗", "我觉得你比较忙"])
def test_bi_without_two_owners_and_a_recorded_attribute_is_not_a_comparison(planner, text):
    assert plan_of(planner, text).dependencies == ()


def user_birthday(source):
    return make_memory(memory_id="mem_user_birthday", persona_id=QIAOMO, subject="user", attribute="birthday_month",
                       value="十二月", statement="用户的生日在十二月。", cardinality="single",
                       provenance=(Provenance(source, "turn:1"),), recorded_at=at(200))


@pytest.mark.parametrize("fts", [None, False], ids=["fts5", "scan"])
def test_a_comparison_is_complete_only_with_one_value_on_each_side(fts):
    """The fixture user has no birthday, so the user's is added here. One value on each side: evidence available,
    a complete answer. The persona's two disagreeing months: conflict, only a safe partial answer."""
    clean = open_store(clock=lambda: QIAOMO_NOW, fts=fts)
    clean.add(make_memory(memory_id="mem_persona_birthday", persona_id=QIAOMO, subject="persona",
                          attribute="birthday_month", value="十月", statement="乔沫的生日在十月。",
                          cardinality="single", recorded_at=at(100)))
    clean.add(user_birthday("src_1"))
    fixture = qiaomo_store(fts=fts)
    fixture.add(user_birthday("src_character_sheet"))
    for store, evidence, answer, batch_status, sufficiency in (
            (clean, EvidenceState.AVAILABLE, AnswerLevel.COMPLETE, BatchStatus.COMPLETE_SUCCESS,
             [Sufficiency.SUFFICIENT, Sufficiency.SUFFICIENT]),
            (fixture, EvidenceState.CONFLICT, AnswerLevel.SAFE_PARTIAL, BatchStatus.COMPLETE_SUCCESS,
             [Sufficiency.CONFLICT, Sufficiency.SUFFICIENT])):
        r = SlotRetriever(store, persona_names=NAMES, clock=lambda: QIAOMO_NOW).retrieve(QIAOMO, "你和我谁生日更早")
        [outcome] = r.batch.dependencies
        assert (outcome.evidence, outcome.answer, outcome.relation) == (evidence, answer, "earlier")
        assert r.batch.status is batch_status and [a.sufficiency for a in r.batch.atoms] == sufficiency
        assert r.batch.atoms[1].memory_ids == ("mem_user_birthday",)
        assert_no_cross_atom_records(r)
        store.close()


def test_mixed_persona_and_relation_questions_answer_only_the_persona(ask):
    for text, persona_ids in (("你住哪？你妈妈住哪？", ["mem_home_beichuan"]),
                              ("你几月生日？你男朋友呢？", ["mem_birthday_nov", "mem_birthday_oct"]),
                              ("你做什么工作，你姐姐做什么工作", ["mem_job_cafe"])):
        r = ask(text)
        assert r.batch.status is BatchStatus.PARTIAL_SUCCESS, text
        assert ids(r) == persona_ids and r.batch.atoms[1].missing.startswith("unowned:"), text
        assert_no_cross_atom_records(r)


def test_atoms_with_the_same_owner_and_attribute_merge(planner):
    plan = plan_of(planner, "你住哪？以前呢？")
    [a] = plan.atoms
    assert (a.attributes, a.time_scope, a.parts) == (("home_city",), TimeScope.ALL, (0, 1))


def test_a_single_question_keeps_the_day6_result_shape(ask):
    r = ask("你现在住在哪个城市呀")
    assert r.batch is None and r.subject == "persona"
    assert "batch" not in r.to_dict()


def test_a_low_confidence_plan_says_so(planner):
    assert plan_of(planner, "你喜欢这首歌吗").confidence == "low"
    assert plan_of(planner, "你住哪？做什么工作？").confidence == "high"
