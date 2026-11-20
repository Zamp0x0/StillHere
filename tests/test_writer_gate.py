"""Full Build Gate (P4-P5): severity rules, the FAST evidence as input, backup / restore and the gated build on the synthetic chat.
No real person, no model (the reference extractor is a test double)."""
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import persona_memory_fixture as MF  # noqa: E402
from stillhere.persona import memory_store as MS  # noqa: E402
from stillhere.persona import service as svc  # noqa: E402
from stillhere.persona import store as S  # noqa: E402
from stillhere.persona import writer_gate as G  # noqa: E402

ALL_OK = {k: True for k in G.SYNTHETIC_SEVERITY}


def test_clean_inputs_pass_and_missing_inputs_never_pass():
    d = G.decide(synthetic_checks=ALL_OK, sample={"sessions": 20, "candidates": 30, "verified": 12},
                 judge={"supported": 12, "unsupported": 0, "attribution_errors": 0, "temporal_errors": 0, "suspected_missing": 5})
    assert d.decision == "FULL_BUILD" and d.stats["judge_estimated_recall"] == pytest.approx(12 / 17, abs=1e-3)
    assert G.decide(sample={"verified": 5}).decision == "EVIDENCE_ONLY"                 # no synthetic gold
    assert G.decide(synthetic_checks=ALL_OK).decision == "EVIDENCE_ONLY"               # no real sample


def test_any_critical_finding_blocks_and_minor_findings_do_not():
    bad = dict(ALL_OK, wedding_not_self=False)
    d = G.decide(synthetic_checks=bad, sample={"verified": 10})
    assert d.decision == "EVIDENCE_ONLY" and any(f.severity == "CRITICAL" for f in d.findings)
    minor = dict(ALL_OK, wedding_event_third_party=False, coffee_negation=False)
    assert G.decide(synthetic_checks=minor, sample={"verified": 10}).decision == "FULL_BUILD"
    assert G.decide(synthetic_checks=ALL_OK, sample={"verified": 10}, judge={"unsupported": 1}).decision == "EVIDENCE_ONLY"
    assert G.decide(synthetic_checks=ALL_OK, sample={"verified": 10}, judge={"temporal_errors": 2}).decision == "EVIDENCE_ONLY"   # 20 % > 10 %
    assert G.decide(synthetic_checks=ALL_OK, sample={"verified": 30}, judge={"temporal_errors": 2}).decision == "FULL_BUILD"      # 6.7 %


def test_the_fast_eval_writer_evidence_gives_evidence_only():
    # FAST_EVAL fast-run-95acdf7e58: SYNTHETIC_GOLD_REPORT writer checks + WRITER_ERROR_BREAKDOWN (2 verified of 29 candidates)
    fast = dict(ALL_OK, wedding_event_third_party=False, wedding_not_self=False)
    d = G.decide(synthetic_checks=fast, sample={"sessions": 20, "candidates": 29, "verified": 2},
                 judge={"supported": 2, "unsupported": 0, "attribution_errors": 0, "temporal_errors": 0, "worthiness_errors": 1, "suspected_missing": 14})
    assert d.decision == "EVIDENCE_ONLY" and "wedding_not_self" in d.reasons[0] and d.stats["judge_estimated_recall"] == 0.125


def test_a_relatives_event_said_by_the_persona_is_never_committed_as_the_persona_own():
    """FAST synthetic gold ``wedding_not_self`` failed: the real model wrote "self / event / 结婚" for "我妹妹小雨下个月结婚". The strict
    speaker rule passes (the persona said it); the third-party-actor guard keeps it out of the reliable store."""
    from datetime import datetime, timedelta, timezone
    from stillhere.writer.contracts import AMBIGUOUS, Candidate
    from stillhere.writer.extract import GroupMessage
    from stillhere.writer.verify import verify
    T = datetime(2025, 1, 3, 9, tzinfo=timezone(timedelta(hours=8)))

    def run(text, **kw):
        base = dict(candidate_id="c.0", persona_id="p", type="EVENT", subject="self", predicate="event", object="结婚", polarity="positive",
                    assertion_type="stated", certainty="certain", event_key="e", event_status="PLANNED", activity="结婚", daypart=None,
                    planned_time=None, event_time=None, valid_from=None, valid_to=None, temporal_precision="unknown", corrects=False, retracts=False,
                    refers_to=None, source_message_ids=["w1"], source_spans=["下个月结婚"], extraction_version="t")
        base.update(kw)
        own = [GroupMessage("m1", "w1", "p", "本人", "对方", T, T, text, "s")]
        c = MS.fix_candidate(Candidate(**base), own, [])
        return MS.worthy(verify(c, own, "p", [], strict_self=True), own)
    bad = run("我妹妹小雨下个月结婚")
    assert not bad.committable and bad.support == AMBIGUOUS and "third-party actor" in bad.reasons[-1]
    assert MS.candidate_status(bad) == "AMBIGUOUS"
    assert run("我妹妹小雨下个月结婚", subject="person:小雨", source_spans=["小雨下个月结婚"]).committable          # the right subject still commits
    assert run("我和妹妹下个月去旅游", object="旅游", activity="旅游", source_spans=["我和妹妹下个月去旅游"]).committable   # the persona's own plan
    assert run("我下个月去看妹妹", object="看妹妹", activity="看望", source_spans=["我下个月去看妹妹"]).committable
    rel = run("我妹妹小雨下个月结婚", type="RELATIONSHIP", predicate="relationship", object="妹妹 小雨", event_key=None, event_status=None,
              activity=None, source_spans=["我妹妹小雨"])
    assert MS.third_party_actor(rel.candidate, []) is None                                              # the owner's relation is not affected


@pytest.fixture()
def prof(tmp_path):
    path = MF.chat().write(tmp_path / "c.json")
    root = tmp_path / "r"
    pid = svc.init_profile(path, speaker="contact", root=root)["profile_id"]
    MS.build_evidence(root, pid)
    return root, pid, MS.mdir(S.profile_dir(root, pid)) / "memory.sqlite"


def _count(db):
    c = sqlite3.connect(str(db))
    try:
        return c.execute("SELECT count(*) FROM memories").fetchone()[0]
    finally:
        c.close()


def test_gated_build_runs_only_on_full_build_and_restores_on_failure(prof, tmp_path):
    root, pid, db = prof
    no = G.decide(synthetic_checks=dict(ALL_OK, no_self_australia=False), sample={"verified": 3})
    r = G.gated_full_build(root, pid, no, complete=MF.reference_complete, reviewer="t", backup_dir=tmp_path / "bk0")
    assert r["mode"] == "EVIDENCE_ONLY" and not r["built"] and _count(db) == 0 and not (tmp_path / "bk0").exists()
    yes = G.decide(synthetic_checks=ALL_OK, sample={"verified": 3})
    ok = G.gated_full_build(root, pid, yes, complete=MF.reference_complete, reviewer="t", backup_dir=tmp_path / "bk1")
    assert ok["mode"] == "STRUCTURED_MEMORY" and _count(db) > 0 and (tmp_path / "bk1" / "BACKUP.json").is_file()
    before = _count(db)
    path2 = MF.chat(extra=[(9, 0, MF.F.CONTACT, "我搬到杭州了")]).write(tmp_path / "c2.json")
    pid2 = svc.init_profile(path2, speaker="contact", root=root)["profile_id"]
    MS.build_evidence(root, pid2)
    db2 = MS.mdir(S.profile_dir(root, pid2)) / "memory.sqlite"

    class Bad:
        def __call__(self, *a, **k):
            raise RuntimeError("model crashed")
    import stillhere.persona.memory_store as MSmod
    orig = MSmod.build_semantic

    def crash(*a, **k):
        raise RuntimeError("full build crashed")
    MSmod.build_semantic = crash
    try:
        fail = G.gated_full_build(root, pid2, yes, complete=Bad(), reviewer="t", backup_dir=tmp_path / "bk2")
    finally:
        MSmod.build_semantic = orig
    assert fail["mode"] == "EVIDENCE_ONLY" and fail["restored"]["sha256"] == fail["backup"]["sha256"] and _count(db2) == 0
    assert _count(db) == before
    with pytest.raises(FileExistsError):
        G.backup_store(db, tmp_path / "bk1")                             # never overwrites a backup
