"""P1 Reader (subject resolution / attribution gate, router past-speech cue) and P2 Writer (R3 relation names) on synthetic questions and the
synthetic memory chat. The old behaviour (reader_version "v1") is pinned so V4 can compare OLD vs NEW. No private data, no model."""
import json
import re
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stillhere.persona import memory_v2 as V2  # noqa: E402
from stillhere.persona import recall as R  # noqa: E402

AT = datetime(2026, 4, 5, tzinfo=timezone(timedelta(hours=8)))
P, C, J, T = R.PERSONA_SELF, R.COUNTERPART, R.JOINT, R.THIRD_PARTY

# (question, v1 kind, p1 kind): synthetic phrasings
CASES = [
    ("你去过哪儿？", P, P),                                      # the persona's own experience: unchanged, gate needs 本人 lines
    ("我以前在哪读书来着？", C, C),                              # the asker's own fact
    ("你还记得我叫你多睡会儿吗？", C, C),                          # 你记得 我…
    ("你妹妹的婚礼定在几月？", T, T),                            # third party of the persona
    ("我爸说我姐下个月搬家了吗？", T, T),                        # a relative reporting on another relative
    ("你说过你最喜欢吃什么？", P, P),                            # the persona's own words: unchanged
    ("我那次说就喜欢你哪种样子？", P, J),                        # the asker's own words (V3 known failure)
    ("我当时给你点外卖，我当时怕什么？", P, J),                    # the asker's own fear
    ("我说想跟你一起去旅行时，说我可以帮你做什么？", P, J),
    ("你给我做的那个手链后来怎么了？", P, J),                    # a shared thing named by a relative clause
    ("你上次跟我说的那场演出怎么样？", P, P),                    # '跟我说' (我 is an object): still the persona's words
    ("你问我周末去不去你家，我是怎么回答的？", P, J),
    ("我们上次去哪吃的饭？", J, J),
]


@pytest.mark.parametrize("q,v1,p1", CASES)
def test_subject_resolution_old_and_new(q, v1, p1):
    assert R.resolve_subject(q).kind == v1 == R.resolve_subject(q, "v1").kind
    assert R.resolve_subject(q, "p1").kind == p1


def test_p1_never_switches_to_the_other_speaker():
    # the fix only RELAXES an ambiguous mixed question to JOINT; it never makes a persona question require the counterpart's lines
    for q, v1, p1 in CASES:
        if v1 == P:
            assert p1 in (P, J)
        if v1 == C:
            assert p1 == C


def test_gate_behaviour_on_the_known_failure_shape():
    q = "我那次说就喜欢你哪种样子？"
    cited = [["对方"]]                                              # the answer cites the asker's own line
    assert not R.attribution_ok(R.resolve_subject(q, "v1"), cited)  # v1: a correct answer is rejected
    assert R.attribution_ok(R.resolve_subject(q, "p1"), cited)
    # safety kept: a persona-own question still rejects an answer that cites only the counterpart's line
    assert not R.attribution_ok(R.resolve_subject("你去过哪儿？", "p1"), [["对方"]])
    assert R.attribution_ok(R.resolve_subject("你去过哪儿？", "p1"), [["本人"]])


def test_unknown_reader_version_is_refused():
    with pytest.raises(ValueError):
        R.resolve_subject("你好吗", "p9")


def test_p1_speaker_note_names_the_speakers_relative():
    s = R.resolve_subject("2025年8月9日是谁在喂猫？", "p1")
    v1, p1 = R.perspective_notes(s), R.perspective_notes(s, "p1")
    assert v1[0] == R.SPEAKER_NOTE and p1[0].startswith(R.SPEAKER_NOTE) and "我弟" in p1[0] and len(p1) == len(v1)


ROUTE = [  # (text, v1 route, p1 route)
    ("你说过哪句话让我很难过？", "chat", "fact"),
    ("你问我周末去不去，我是怎么回答的？", "chat", "fact"),
    ("你跟我讲过的那家店在哪？", "chat", "fact"),
    ("你说过 Python 支持多继承吗？", "chat", "chat"),              # general knowledge stays chat
    ("当时那个公式是什么意思？", "chat", "chat"),
    ("在吗", "chat", "chat"),                                     # small talk
    ("那次真开心", "chat", "chat"),                               # not a question
    ("今天吃什么？", "chat", "chat"),                             # present-tense chat
]


@pytest.mark.parametrize("text,v1,p1", ROUTE)
def test_router_past_speech_cue(text, v1, p1):
    assert V2.route_of_v2(text, AT) == v1 == V2.route_of_v2(text, AT, "v1")
    assert V2.route_of_v2(text, AT, "p1") == p1


def test_runtime_rejects_an_unknown_reader_version(tmp_path):
    from stillhere.persona.runtime import PersonaRuntime
    with pytest.raises(ValueError):
        PersonaRuntime(tmp_path, "p-x", reader_version="p9")


# ---------------------------------------------------------------- P2 Writer: R3 relation names
def test_relation_names():
    from stillhere.writer import m22
    assert m22.relation_names("妹妹 小雨") >= {"小雨", "妹妹"}
    assert m22.relation_names("妹妹小雨") == {"妹妹小雨", "妹妹", "小雨"}
    assert m22.relation_names("同事") == {"同事"} and m22.relation_names("") == set()

    class M:
        def __init__(self, t):
            self.content = t
    assert m22.stated_in_message("小雨", [M("我妹妹小雨下个月结婚")]) and m22.stated_in_message("妹妹", [M("我妹妹下个月结婚")])
    assert not m22.stated_in_message("小雨", [M("小雨下个月结婚")]) and not m22.stated_in_message("小明", [M("我妹妹小雨下个月结婚")])


def _store(cands):
    import persona_memory_fixture as MF
    from stillhere.persona import memory_store as MS, service as svc, store as S
    W = "我妹妹小雨下个月结婚"
    tmp = Path(tempfile.mkdtemp())
    try:
        path = MF.chat().write(tmp / "chat.json")
        root = tmp / "profiles"
        pid = svc.init_profile(path, speaker="contact", root=root)["profile_id"]
        MS.build_evidence(root, pid)

        def complete(messages):
            user = messages[-1]["content"]
            if "只看下面标出的这条消息" in user:
                return {"text": json.dumps({"candidates": []})}
            out = []
            for line in user.splitlines():
                m = MF.LINE.match(line.strip())
                if m and W in line:
                    out += [dict(x, source_message_ids=[m.group(1)]) for x in cands]
            return {"text": json.dumps({"candidates": out}, ensure_ascii=False)}
        MS.build_semantic(root, pid, complete=complete, reviewer="test")
        con = sqlite3.connect(str(S.profile_dir(root, pid) / "memory" / "memory.sqlite"))
        rows = [tuple(x) for x in con.execute("SELECT subject, attribute, value FROM memories").fetchall()]
        con.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    actor = {x[0]: x[2] for x in rows if x[1] == "event.actor"}
    return [(actor.get(x[0]), x[2]) for x in rows if x[1] == "event.description" and "婚" in x[2]]


@pytest.mark.parametrize("rel,subject", [("妹妹小雨", "person:小雨"), ("妹妹 小雨", "person:小雨"), (None, "person:小雨"), (None, "person:妹妹")])
def test_a_correctly_attributed_third_party_wedding_is_stored(rel, subject):
    import persona_memory_fixture as MF
    c = MF.cand
    cands = ([c("self", "relationship", rel, "M", "我妹妹小雨")] if rel else []) + [
        c(subject, "event", "结婚", "M", "下个月结婚", type="EVENT", event_key="w", activity="结婚", event_status="PLANNED")]
    assert _store(cands) == [(subject, "结婚")]


@pytest.mark.parametrize("obj,span", [("结婚", "我妹妹小雨下个月结婚"), ("妹妹结婚", "下个月结婚"), ("小雨结婚", "小雨下个月结婚"), ("结婚", "结婚")])
def test_the_sisters_wedding_is_never_stored_as_the_persona_s_own(obj, span):
    import persona_memory_fixture as MF
    c = MF.cand
    cands = [c("self", "relationship", "妹妹 小雨", "M", "我妹妹小雨"),
             c("self", "event", obj, "M", span, type="EVENT", event_key="w", activity="结婚", event_status="PLANNED")]
    assert all(a != "self" for a, _ in _store(cands))


def test_a_stranger_is_still_outside_the_boundary():
    import persona_memory_fixture as MF
    c = MF.cand
    assert _store([c("person:小明", "event", "结婚", "M", "下个月结婚", type="EVENT", event_key="w", activity="结婚", event_status="PLANNED")]) == []


# ---------------------------------------------------------------- Colab reliability: the dense index works on local disk
def test_dense_index_lives_on_local_disk_and_resumes_from_the_drive_snapshot(tmp_path):
    from stillhere.persona import eval_v2 as EV2

    class Run:
        run_dir = tmp_path / "drive_run"
    work, persisted = EV2.dense_dirs(Run, tmp_path / "local")
    assert work == tmp_path / "local" and persisted == tmp_path / "drive_run" / "dense_index"
    assert EV2.dense_dirs(Run)[0] == persisted                       # the old layout without a local dir
    (persisted / "index").mkdir(parents=True)
    (persisted / "index" / "embedding_index_manifest.json").write_text("{}")
    (persisted / "query_cache_x.sqlite").write_text("stale")         # a writable cache is never copied back
    assert EV2._index_done(persisted) and not EV2._index_done(work)
