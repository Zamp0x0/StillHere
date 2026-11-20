"""BUILD-M1 Context Management V1: the regression dataset, D6, attribution, scope, time, boundaries, budget, compaction, contract,
reader compatibility, legacy compatibility, privacy and snapshot integrity. Synthetic data only; no model, no network."""
import dataclasses
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from stillhere.context import ContextOrchestrator, ContextRequest, ContextResult, DerivedItem, WorkingItem
from stillhere.context import attribution as at
from stillhere.context import boundary as bd
from stillhere.context import budget as bg
from stillhere.context import compaction as cp
from stillhere.context import contracts as cc
from stillhere.evaluation import answer_pilot as ap
from stillhere.evaluation import context_bridge as cb
from stillhere.evaluation import context_regression as cr
from stillhere.memory.sqlite_store import PersonaScopeError

ROOT = Path(__file__).resolve().parents[1]
TZ = timezone(timedelta(hours=8))


@pytest.fixture(scope="module")
def world():
    return cr.build_world()


@pytest.fixture(scope="module")
def orch(world):
    return ContextOrchestrator(world.api, speakers=cr.SPEAKERS)


def run(world, orch, case_id):
    case = next(c for c in cr.cases(world) if c.case_id == case_id)
    return case, orch.assemble(case.request)


# ---------------------------------------------------------------- the regression dataset: every case passes

def _ids():
    return [c.case_id for c in cr.cases(cr.build_world())]


@pytest.mark.parametrize("case_id", _ids())
def test_regression_case(world, orch, case_id):
    case, r = run(world, orch, case_id)
    out = cr.check(world, case, r)
    assert out["pass"], out["failures"]


def test_dataset_size_and_categories(world):
    cs = cr.cases(world)
    assert 40 <= len(cs) <= 70
    need = {"d6_full_round", "d6_correction", "d6_plan_vs_done", "attribution", "persona", "temporal_visibility", "known_at", "knowledge_update", "conflict", "multi_fact",
            "duplicate", "temporal_boundary", "token_overflow", "abstention", "retrieval", "reranker", "recent", "pinned", "working", "derived", "integrity"}
    assert need <= {c.category for c in cs}


def test_metrics_meet_every_gate():
    m = cr.evaluate(repeats=1)["metrics"]
    assert m["PERSONA_LEAKAGE_COUNT"] == 0 and m["TEMPORAL_LEAKAGE_COUNT"] == 0 and m["SILENT_TRUNCATION_COUNT"] == 0 and m["TOKEN_BUDGET_VIOLATIONS"] == 0
    assert m["REQUIRED_EVIDENCE_PRESERVATION_RATE"] == 1.0 and m["FULL_TURN_PRESERVATION_RATE"] == 1.0 and m["SOURCE_ATTRIBUTION_ACCURACY"] == 1.0
    assert m["TOKEN_COUNT_MODE"] == cc.TOKEN_ESTIMATED


# ---------------------------------------------------------------- D6

def test_d6_legacy_renderer_drops_messages_and_the_orchestrator_does_not():
    d6 = cr.evaluate(repeats=1)["d6"]["summary"]
    assert d6["rounds_over_3"] > 0 and d6["legacy_rounds_truncated"] == d6["rounds_over_3"] and d6["legacy_messages_dropped"] > 0
    assert d6["orchestrator_rounds_truncated"] == 0


@pytest.mark.parametrize("name,size", [("r1", 1), ("r3", 3), ("r4", 4), ("r6_cancel", 6), ("r10_move", 10), ("r12_long", 12)])
def test_every_message_of_a_round_is_rendered_in_order_with_its_ids(world, orch, name, size):
    conv = f"{cr.PA}.agent_chat.{name}"
    ids = world.rounds[conv]
    r = orch.assemble(ContextRequest(persona_id=cr.PA, question="回顾", as_of=cr.t(20), hits=cr.hits(world, ids[0]), input_token_budget=50000))
    p = r.provenance_map["E01"]
    assert len(ids) == size and p["message_ids"] == ids and p["complete"] and p["messages_total"] == size and p["source_ids"] == [cr.CHAT]
    positions = [r.assembled_context.find(world.contents[m]) for m in ids]
    assert all(x >= 0 for x in positions) and positions == sorted(positions)
    assert f"对话：完整，共 {size} 条" in r.assembled_context


def test_round_text_is_kept_as_the_frozen_three_turn_renderer(world):
    """The legacy renderer stays as it was, so every frozen result reproduces; the new path is the orchestrator."""
    text = cr.legacy_round(world, cr.PA, f"{cr.PA}.agent_chat.r6_cancel", cr.t(20))
    assert text.count(" / ") == 2 and "取消" not in text


def test_overflow_keeps_hits_final_state_and_marks_every_gap(world, orch):
    case, r = run(world, orch, "overflow_long_round")
    conv = f"{cr.PA}.agent_chat.r12_long"
    p = r.provenance_map["E01"]
    assert not p["complete"] and cc.CONTEXT_OVERFLOW in r.warnings
    assert world.rounds[conv][5] in p["message_ids"] and world.rounds[conv][11] in p["message_ids"]
    assert set(r.omitted_message_ids[p["key"]]) == set(world.rounds[conv]) - set(p["message_ids"])
    assert "部分显示" in r.assembled_context and "省略第" in r.assembled_context and r.token_usage_estimate <= case.request.input_token_budget


def test_an_oversized_round_does_not_starve_the_next_unit(world, orch):
    case, r = run(world, orch, "budget_skips_big_keeps_small")
    assert "f_gala" in r.selected_evidence_ids


def test_compaction_priority_and_two_passes(world):
    w = world
    conv = f"{cr.PA}.agent_chat.r12_long"
    from stillhere.context import units as un
    units, _, _ = un.expand(w.api, cr.PA, cr.hits(w, w.rounds[conv][5]), [w.rounds[conv][5]], cr.t(20), cr.t(20), cr.SPEAKERS)
    u = units[0]
    order = cp.priority(u)
    assert order[0].message_id == w.rounds[conv][5] and order[1].message_id == w.rounds[conv][11]
    first = cp.overflow_selection(u, lambda sub: True)
    assert {m.message_id for m in first} == {w.rounds[conv][5], w.rounds[conv][11]}           # essential only
    full = cp.overflow_selection(u, lambda sub: True, essential_only=False, start=first)
    assert len(full) == 12
    assert cp.overflow_selection(u, lambda sub: False) is None


# ---------------------------------------------------------------- attribution

def test_two_party_round_direction_and_unknown_otherwise():
    d = at.round_direction("p", ["p", "bot", "p"], "p", {"bot": "AI 助手"})
    assert (d.sender, d.recipient) == ("本人", "AI 助手")
    d3 = at.round_direction("p", ["p", "bot", "x"], "p", {"bot": "AI 助手"})
    assert d3.recipient == at.UNKNOWN
    alone = at.round_direction("p", ["p"], "p", {})
    assert (alone.sender, alone.recipient) == ("本人", at.UNKNOWN)


def test_sms_direction_never_guesses():
    assert at.sms_direction("phone_id=1;type=发送", "本人", "李雷") == at.Direction("本人", "李雷")
    assert at.sms_direction("phone_id=1;type=接收", "李雷", "李雷") == at.Direction("李雷", "本人")
    assert at.sms_direction("phone_id=1", "李雷", "李雷") is None
    assert at.sms_direction("phone_id=1;type=发送", "本人", None) is None


def test_one_sms_rule_shared_with_the_v34_renderer():
    from stillhere.evaluation import evidence_render as rv
    assert rv.sms_direction is at.sms_direction and rv.Direction is at.Direction


def test_quoted_person_is_not_the_sender(world, orch):
    _, r = run(world, orch, "attr_quoted_third_party")
    line = next(ln for ln in r.assembled_context.splitlines() if "我妈刚发微信" in ln)
    assert "本人 → AI 助手" in line and "我妈 →" not in line


# ---------------------------------------------------------------- scope and time through MemoryAPI

def test_conversation_read_is_persona_scoped(world):
    with pytest.raises(PersonaScopeError):
        world.api.conversation(cr.PA, f"{cr.PB}.agent_chat.secret", as_of=cr.t(20))
    msgs, hidden = world.api.conversation(cr.PB, f"{cr.PB}.agent_chat.secret", as_of=cr.t(20))
    assert len(msgs) == 1 and hidden == 0


def test_conversation_read_hides_later_messages_and_reports_how_many(world):
    conv = f"{cr.PA}.agent_chat.r_partial_future"
    msgs, hidden = world.api.conversation(cr.PA, conv, as_of=cr.t(10, 23, 55))
    assert [m.message_id for m in msgs] == world.rounds[conv][:1] and hidden == 1
    msgs, hidden = world.api.conversation(cr.PA, f"{cr.PA}.agent_chat.r_late_known", as_of=cr.t(5, 18))
    assert msgs == [] and hidden == 1                                    # said on the 5th, learned on the 6th


def test_hidden_later_messages_are_not_announced_to_the_reader(world, orch):
    _, r = run(world, orch, "round_part_after_as_of")
    assert cc.LATER_MESSAGES_HIDDEN in r.warnings and "零点" not in r.assembled_context and "之后还有" not in r.assembled_context


def test_assembly_never_writes_memory(world, orch):
    before = world.api.stats()["counts"]
    for case in cr.cases(world):
        orch.assemble(case.request)
    assert world.api.stats()["counts"] == before


# ---------------------------------------------------------------- temporal boundaries

ASKED = datetime(2026, 11, 20, 12, tzinfo=TZ)
NOV5, NOV6 = datetime(2026, 11, 5, tzinfo=TZ), datetime(2026, 11, 6, tzinfo=TZ)


@pytest.mark.parametrize("q,op,lower,upper", [
    ("11月5日之前", bd.BEFORE, None, NOV5), ("11月5日以前", bd.BEFORE, None, NOV5), ("早于11月5日", bd.BEFORE, None, NOV5), ("before 2026-11-05", bd.BEFORE, None, NOV5),
    ("截至11月5日", bd.ON_OR_BEFORE, None, NOV6), ("截止到11月5日", bd.ON_OR_BEFORE, None, NOV6), ("不晚于11月5日", bd.ON_OR_BEFORE, None, NOV6),
    ("到11月5日为止", bd.ON_OR_BEFORE, None, NOV6), ("on or before 2026-11-05", bd.ON_OR_BEFORE, None, NOV6), ("11月5日之前（含当天）", bd.ON_OR_BEFORE, None, NOV6),
    ("11月5日当天", bd.ON, NOV5, NOV6), ("11月5日", bd.ON, NOV5, NOV6), ("on 2026-11-05", bd.ON, NOV5, NOV6),
    ("11月5日之后", bd.AFTER, NOV6, None), ("晚于11月5日", bd.AFTER, NOV6, None), ("after 2026-11-05", bd.AFTER, NOV6, None),
    ("自11月5日起", bd.ON_OR_AFTER, NOV5, None), ("11月5日以来", bd.ON_OR_AFTER, NOV5, None), ("不早于11月5日", bd.ON_OR_AFTER, NOV5, None),
    ("since 2026-11-05", bd.ON_OR_AFTER, NOV5, None), ("11月5日之后（含当天）", bd.ON_OR_AFTER, NOV5, None),
    ("11月3日到11月5日之间", bd.BETWEEN, datetime(2026, 11, 3, tzinfo=TZ), NOV6), ("11月之前", bd.BEFORE, None, datetime(2026, 11, 1, tzinfo=TZ)),
    ("2026-11-05 之前", bd.BEFORE, None, NOV5), ("2026-11-05  以后", bd.AFTER, NOV6, None),
])
def test_boundary_operators(q, op, lower, upper):
    b = bd.parse_boundary(q, ASKED)
    assert (b.status, b.operator, b.lower, b.upper) == (bd.RESOLVED, op, lower, upper)


@pytest.mark.parametrize("q,status", [("11月5日前后", bd.AMBIGUOUS), ("到11月5日", bd.AMBIGUOUS), ("直到11月5日", bd.AMBIGUOUS), ("until 2026-11-05", bd.AMBIGUOUS),
                                      ("11月5日和11月7日", bd.AMBIGUOUS), ("前阵子", bd.UNKNOWN), ("2026年2月30日之前", bd.UNKNOWN), ("你好", bd.NONE)])
def test_boundaries_that_are_not_forced(q, status):
    assert bd.parse_boundary(q, ASKED).status == status


def test_relation_strict_before_on_and_until():
    event = datetime(2026, 11, 5, 19, tzinfo=TZ)
    assert bd.relation(bd.parse_boundary("11月5日之前", ASKED), event) == bd.VIOLATES
    assert bd.relation(bd.parse_boundary("截至11月5日", ASKED), event) == bd.SATISFIES
    assert bd.relation(bd.parse_boundary("11月5日当天", ASKED), event) == bd.SATISFIES
    assert bd.relation(bd.parse_boundary("11月4日之前", ASKED), event) == bd.VIOLATES
    assert bd.relation(bd.parse_boundary("11月6日之前", ASKED), event) == bd.SATISFIES
    assert bd.relation(bd.parse_boundary("到11月5日", ASKED), event) == bd.UNKNOWN


def test_relation_with_day_precision_is_unknown_inside_the_day():
    b = bd.parse_boundary("2026-11-05 之前", ASKED)
    assert bd.relation(b, *bd.day_of(datetime(2026, 11, 5, 23, 59, tzinfo=TZ))) == bd.VIOLATES
    tight = dataclasses.replace(b, upper=datetime(2026, 11, 5, 22, tzinfo=TZ))
    assert bd.relation(tight, *bd.day_of(datetime(2026, 11, 5, 23, 59, tzinfo=TZ))) == bd.UNKNOWN


def test_year_inferred_is_said():
    b = bd.parse_boundary("12月25日之前", ASKED)
    assert b.reason == "year inferred" and "年份按提问时间推断" in b.description


def test_planned_time_is_labelled_as_planned(world, orch):
    _, r = run(world, orch, "tb_planned_vs_occurred")
    assert "时间类型：计划时间" in r.assembled_context and r.provenance_map["E01"]["time_kind"] == "scheduled"
    _, r2 = run(world, orch, "tb_on_day")
    assert "时间类型：发生时间" in r2.assembled_context


def test_boundary_line_states_the_day_rule(world, orch):
    _, r = run(world, orch, "tb_before_excludes_day")
    assert "严格早于2026-11-05：不包括2026-11-05当天" in r.assembled_context


# ---------------------------------------------------------------- budget and counting

def test_token_counter_modes():
    est = bg.TokenCounter()
    assert est.mode == cc.TOKEN_ESTIMATED and est("一二三四五六七八九十" * 10) == 105
    exact = bg.TokenCounter(lambda s: len(s.split()), exact=True)
    assert exact.mode == cc.TOKEN_EXACT and exact("a b c") == 3
    with pytest.raises(ValueError):
        bg.TokenCounter(exact=True)


def test_budget_profiles():
    assert bg.profile_of("他住在哪？", has_boundary=False, has_conversation=True) == "conversational"
    assert bg.profile_of("11月5日之前？", has_boundary=True, has_conversation=False) == "temporal"
    assert bg.profile_of("我叫什么？", has_boundary=False, has_conversation=False) == "factual"
    b = bg.plan(1000, 400, 100, "temporal")
    assert b.available == 900 and b.cap("recent") == 90


def test_exact_counter_is_reported_and_respected(world):
    counter = bg.TokenCounter(lambda s: len(s) * 2, exact=True)
    o = ContextOrchestrator(world.api, speakers=cr.SPEAKERS, counter=counter)
    r = o.assemble(ContextRequest(persona_id=cr.PA, question="年会？", as_of=cr.t(11), hits=cr.hits(world, "f_gala", "f_dinner_4"), input_token_budget=200))
    assert r.token_count_mode == cc.TOKEN_EXACT and r.token_usage_estimate <= 200


def test_fixed_parts_are_never_dropped_for_evidence(world, orch):
    case, r = run(world, orch, "budget_fixed_parts_too_big")
    assert r.status == cc.BUDGET_INSUFFICIENT and not r.provenance_map


# ---------------------------------------------------------------- contract, sections, future slots

def test_context_result_contract(world, orch):
    _, r = run(world, orch, "multi_fact_three_rounds")
    for f in ("assembled_context", "selected_evidence_ids", "included_message_ids", "excluded_evidence_ids", "provenance_map", "token_usage_estimate", "token_budget",
              "truncation_policy", "warnings", "assembly_trace", "token_count_mode", "temporal_boundary"):
        assert hasattr(r, f)
    assert [s["step"] for s in r.assembly_trace][0] == "scope" and r.assembly_trace[-1]["step"] == "done"
    for cid, p in r.provenance_map.items():
        assert f"[{cid}]" in r.assembled_context and p["source_ids"]
    assert json.dumps(r.assembly_trace, ensure_ascii=False, default=str)
    assert not any("内容" in json.dumps(s, ensure_ascii=False) for s in r.assembly_trace)    # the trace holds ids and counts, not text


def test_request_fields_for_writer_observation_reflection():
    names = {f.name for f in dataclasses.fields(ContextRequest)}
    assert {"persona_id", "conversation_id", "question", "as_of", "known_at", "reader_model", "input_token_budget", "output_token_reserve", "working", "derived", "hits", "pinned_memory_ids"} <= names
    d = {f.name for f in dataclasses.fields(DerivedItem)}
    assert {"supporting_evidence_ids", "contradicting_evidence_ids", "valid_from", "valid_to", "confidence", "version"} <= d


def test_derived_is_labelled_and_below_raw(world, orch):
    _, r = run(world, orch, "derived_allowed_labelled")
    d = next(p for p in r.provenance_map.values() if p["section"] == "derived")
    assert d["authority"] == "derived_below_raw" and "【推导内容】（不是原始记录）" in r.assembled_context
    assert r.assembled_context.index("【推导内容】") < r.assembled_context.index("【记忆证据】")
    o = ContextOrchestrator(world.api, speakers=cr.SPEAKERS)
    lone = o.assemble(ContextRequest(persona_id=cr.PA, question="?", as_of=cr.t(11), allow_derived=True, derived=(DerivedItem("x", "reflection", "猜测", ("f_gala",)),)))
    assert lone.excluded_evidence_ids["x"] == "derived_without_shown_support"


def test_working_context_is_marked_session_only(world, orch):
    _, r = run(world, orch, "working_not_persisted")
    assert "仅本次会话，不是长期记忆" in r.assembled_context and r.provenance_map["W01"]["persisted"] is False


def test_evidence_alone_keeps_the_frozen_layout(world, orch):
    _, r = run(world, orch, "duplicate_memory_hit")
    assert r.assembled_context.startswith("[E01] | 来源：") and "【" not in r.assembled_context.split("\n", 1)[0]


# ---------------------------------------------------------------- reader and legacy compatibility

def test_flag_is_off_by_default():
    assert not cb.enabled({}) and cb.enabled({cb.FLAG: "1"}) and not cb.enabled({cb.FLAG: "0"})


def test_bridge_returns_an_answer_pilot_context(world):
    r = cb.orchestrated(world.api, persona_id=cr.PA, question="上海出差怎么样了？", ask_time="2026-11-11", when=cr.t(11), hits=cr.hits(world, world.rounds[f"{cr.PA}.agent_chat.r6_cancel"][0]),
                        keys=["k1"], speakers=cr.SPEAKERS)
    c = cb.as_reader_context(r)
    assert isinstance(c, ap.Context) and c.evidence_ids == ("k1",) and c.id_map["E01"] == "k1" and c.sha256 == ap.sha256_text(c.text)
    assert r.token_budget["input_total"] == cb.scaffolding_tokens(bg.TokenCounter(), "2026-11-11") + bg.TokenCounter()("上海出差怎么样了？") + ap.MAX_CONTEXT_TOKENS
    msgs = ap.build_messages("上海出差怎么样了？", "2026-11-11", c.text)
    assert msgs[0]["content"] == ap.SYSTEM_PROMPT and c.text in msgs[1]["content"]


def test_bridge_refuses_a_failed_integrity_check():
    bad = ContextResult("SCOPE_ERROR", "", [], [], {}, {}, 0, {}, cc.TOKEN_ESTIMATED, "", [], [])
    with pytest.raises(ValueError):
        cb.as_reader_context(bad)


def test_reader_scripts_keep_the_legacy_path_and_guard_frozen_outputs():
    v3 = (ROOT / "scripts" / "evaluate_v3.py").read_text(encoding="utf-8")
    assert "if not cb.enabled():\n            return ap.build_context(items)" in v3 and "context_management_m1" in v3
    v32 = (ROOT / "scripts" / "evaluate_v3_2.py").read_text(encoding="utf-8")
    assert "ap.build_context(items)" not in v32 and v32.count("b.context_for(") == 2


def test_frozen_reader_prompt_is_untouched():
    frozen = (ROOT / "src" / "stillhere" / "evaluation" / "answer_pilot.py").read_text(encoding="utf-8")
    assert "deepseek_c1_answerability" not in (ROOT / "src" / "stillhere" / "context" / "orchestrator.py").read_text(encoding="utf-8")
    assert ap.SYSTEM_PROMPT in frozen


def test_legacy_memory_api_search_still_works(world):
    hits = world.api.search(cr.PA, "滨江区", as_of=cr.t(11))
    assert hits and all(h.persona_id == cr.PA for h in hits)
    assert world.api.strategies() == ["legacy"]


# ---------------------------------------------------------------- privacy, secrets, snapshots

def test_privacy_guard_and_secret_scan_pass_on_the_new_files():
    files = [str(p.relative_to(ROOT)) for p in list((ROOT / "src" / "stillhere" / "context").glob("*.py")) + [ROOT / "src" / "stillhere" / "evaluation" / "context_regression.py",
             ROOT / "src" / "stillhere" / "evaluation" / "context_bridge.py", ROOT / "scripts" / "evaluate_context_m1.py"]]
    r = subprocess.run([sys.executable, "scripts/privacy_guard.py", *files], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_regression_world_is_synthetic_only(world):
    rows = world.api.store._conn.execute("SELECT origin FROM sources").fetchall()
    assert {r[0] for r in rows} == {"synthetic"}


def test_frozen_snapshots_did_not_move():
    def rev(name):
        r = subprocess.run(["git", "rev-parse", f"{name}^{{commit}}"], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    if rev("HEAD") is None:
        pytest.skip("not a git checkout")
    assert rev("memory-reader-deepseek-v3.3-audited-snapshot") == "82e5c241526c941c99ed8ea003b2a26f0b7afe52"
    assert rev("memory-retrieval-v3.2-research-snapshot") == "7bfaaf7d10924c80be96e44f1947cad907a2e00c"
    assert rev("memory-retrieval-v2-baseline") == "984bc260833dff9ec5978c1f51b342e3331184ed"
    assert rev("memory-retrieval-v1") == "61a26c008b35f162948f1388a03a7866382c4d32"
