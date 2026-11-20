"""Final Colab L4 evaluation (CPU only, synthetic data): reader subject / date recall, third-party writer follow-up, chunking, QA builder
verification and freeze, DeepSeek judge (mock transport), notebook static validation, checkpoint / resume, data isolation, and the full
package -> Colab stages -> results ZIP -> local import flow with scripted model doubles (never presented as a real model)."""
import ast
import hashlib
import json
import re
import sqlite3
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in ("src", "training/persona", "training/d5", "training/d4"):
    sys.path.insert(0, str(ROOT / p))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import persona_fixture as F  # noqa: E402
import persona_memory_fixture as MF  # noqa: E402
from stillhere.persona import card as K  # noqa: E402
from stillhere.persona import colab_eval as CE  # noqa: E402
from stillhere.persona import evaluation as EV  # noqa: E402
from stillhere.persona import final_package as FP  # noqa: E402
from stillhere.persona import judge as J  # noqa: E402
from stillhere.persona import memory_store as MS  # noqa: E402
from stillhere.persona import qa_builder as QB  # noqa: E402
from stillhere.persona import recall as R  # noqa: E402
from stillhere.persona import results_import as RI  # noqa: E402
from stillhere.persona import service as svc  # noqa: E402
from stillhere.persona import store as S  # noqa: E402
from stillhere.persona.runtime import PersonaRuntime, route_of  # noqa: E402

TZ = timezone(timedelta(hours=8))
REPO_ONLY = pytest.mark.skipif(not (ROOT / ".git").exists(), reason="needs the git checkout (the package build reads the committed tree); skipped in Colab")


# ======================================================================= reader: subject resolution and routing
def test_subject_resolution_and_routes():
    assert R.resolve_subject("你以前在哪工作？").kind == R.PERSONA_SELF
    assert R.resolve_subject("我在哪读书来着？").kind == R.COUNTERPART
    assert R.resolve_subject("你还记得我以前在哪读书吗？").kind == R.COUNTERPART
    s = R.resolve_subject("你妹妹什么时候结婚？")
    assert s.kind == R.THIRD_PARTY and s.owner == R.PERSONA_SELF and s.name == "妹妹"
    assert R.resolve_subject("我们上次聊了什么？").kind == R.JOINT
    at = datetime(2025, 7, 10, 12, tzinfo=TZ)
    assert route_of("2025年6月1日我们聊了什么？", at) == "date" and route_of("我们6月1号聊了啥", at) == "date"
    assert route_of("2025年6月我们聊过什么", at) == "date"
    assert route_of("你以前在哪工作？", at) == "fact" and route_of("在吗", at) == "chat"
    sc = R.date_scope("2025年6月我们聊过什么", at)
    assert sc.granularity == "month" and sc.start == datetime(2025, 6, 1, tzinfo=TZ) and sc.end == datetime(2025, 7, 1, tzinfo=TZ)
    assert R.attribution_ok(R.Subject(R.COUNTERPART), [["对方"]]) and not R.attribution_ok(R.Subject(R.COUNTERPART), [["本人"]])
    assert R.attribution_ok(R.Subject(R.THIRD_PARTY), [["本人"]])


class Scripted:
    """Reader / persona double: answers the Fact Reader with fixed JSON (records the prompt), the persona pass with a fixed text."""
    label = "SCRIPTED_TEST_DOUBLE"

    def __init__(self, reader_json=None, styled="嗯", summary=None):
        self.reader_json, self.styled, self.summary, self.calls = reader_json, styled, summary, []

    def __call__(self, messages, cfg, adapter):
        self.calls.append(messages)
        sysm = messages[0]["content"]
        if "个人记忆助手" in sysm:
            return {"text": json.dumps(self.reader_json, ensure_ascii=False), "new_tokens": 10}
        if "聊天记录整理助手" in sysm:
            return {"text": json.dumps(self.summary(messages[1]["content"]), ensure_ascii=False), "new_tokens": 10}
        return {"text": self.styled, "new_tokens": 3}


@pytest.fixture()
def mem_rt(tmp_path):
    path = MF.chat().write(tmp_path / "c.json")
    root = tmp_path / "r"
    pid = svc.init_profile(path, speaker="contact", root=root)["profile_id"]
    MS.build_evidence(root, pid)
    MS.build_semantic(root, pid, complete=MF.reference_complete, reviewer="t")
    pdir = S.profile_dir(root, pid)
    card = {"schema": K.CARD_SCHEMA, "persona_id": pid, "display_name": "测试", "ai_identity": "你是 AI", "reply_length": {}, "style_traits": [], "examples": [],
            "evidence": {"split": "train", "dataset_version": "v", "train_sha256": "x", "profiler_version": "p", "example_ids": [], "method": "m"},
            "review_status": K.AUTO, "authored_prompt": "你是 AI 数字人。" + "说话很短。" * 60 + "没有依据时说不知道，不要编造。"}
    K.set_current(pdir, K._write_version(pdir, card, parent=None)["card_id"])
    return PersonaRuntime(root, pid, memory="production", generator=Scripted())


def test_counterpart_question_gets_the_perspective_note_and_a_counterpart_citation(mem_rt):
    at = datetime(2025, 2, 1, tzinfo=TZ)
    mem_rt.gen = Scripted({"answer": "对方在澳洲读书。", "answerable": True, "abstain": False, "evidence": ["E02"], "quotes": []}, "澳洲呀")
    pk = mem_rt.fact_pack("对方在哪读书来着？", at)
    e = next(i.eid for i in pk.items if "澳洲" in i.text)
    mem_rt.gen.reader_json["evidence"] = [e]
    r = mem_rt.reply([("other", "我在哪读书来着？")], now=at)
    reader_prompt = mem_rt.gen.calls[0][1]["content"]
    assert "对方在哪读书来着" in reader_prompt and R.SPEAKER_NOTE in reader_prompt and "对方”（聊天对象）自己的事" in reader_prompt
    assert r["route"] == "fact" and r["reader"]["answerable"] and r["reader"]["subject"]["kind"] == R.COUNTERPART and "澳洲" in r["text"]
    own = next(i.eid for i in pk.items if i.speakers == ["本人"])
    mem_rt.gen = Scripted({"answer": "对方在广州上班。", "answerable": True, "abstain": False, "evidence": [own], "quotes": []}, "广州")
    r = mem_rt.reply([("other", "我在哪读书来着？")], now=at)
    assert not r["reader"]["answerable"] and r["reader"]["fail"] == "SUBJECT_RESOLUTION_FAIL"          # attribution is not dropped for recall


def test_date_route_reads_the_whole_local_day_and_keeps_only_grounded_points(mem_rt):
    def summary(user):
        ids = re.findall(r"\[(D\d+)\]", user)
        return {"points": [{"text": "对方说自己在澳洲读书很忙", "who": "对方", "evidence": [ids[2]]},
                           {"text": "本人说在火星工作了30年", "who": "本人", "evidence": [ids[0]]},                    # not in the cited line: dropped
                           {"text": "没有编号的点", "who": "本人", "evidence": []}]}
    mem_rt.gen = Scripted(styled="那天你说在澳洲读书 挺忙的", summary=summary)
    r = mem_rt.reply([("other", "2025年1月1日我们聊了什么？")], now=datetime(2025, 2, 1, tzinfo=TZ))
    ds = r["reader"]["date_summary"]
    assert r["route"] == "date" and ds["status"] == "ANSWERED" and ds["messages"] == 4 and len(ds["points"]) == 1
    assert {d["reason"] for d in ds["dropped"]} == {"number not in the cited lines", "no valid evidence id"}
    assert r["reader"]["final_check"] == "ANSWER_KEPT" and "澳洲" in r["text"]
    r = mem_rt.reply([("other", "2025年1月20日我们聊了什么？")], now=datetime(2025, 2, 1, tzinfo=TZ))
    assert r["reader"]["date_summary"]["status"] == "NO_MESSAGES" and not r["reader"]["answerable"]


def test_window_messages_respect_the_local_timezone(tmp_path):
    db = tmp_path / "m.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE messages (message_id, conversation_id, target_persona_id, speaker_id, timestamp_utc, content)")
    rows = [("a", "c", "p", "p", "2025-05-31T15:59:00Z", "前一天"), ("b", "c", "p", "p.other", "2025-05-31T16:00:00Z", "当天零点"),
            ("c", "c", "p", "p", "2025-06-01T15:59:59Z", "当天最后"), ("d", "c", "p", "p", "2025-06-01T16:00:00Z", "第二天")]
    con.executemany("INSERT INTO messages VALUES (?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    sc = R.date_scope("2025年6月1日聊了什么", datetime(2025, 7, 1, tzinfo=TZ))
    got = R.window_messages(db, "p", sc)
    assert [m["message_id"] for m in got] == ["b", "c"] and got[0]["speaker"] == "对方" and got[1]["speaker"] == "本人"


def test_summary_is_segmented_and_failures_are_not_answers():
    sc = R.DateScope(datetime(2025, 6, 1, tzinfo=TZ), datetime(2025, 6, 2, tzinfo=TZ), "day", "x")
    msgs = [{"message_id": f"m{i}", "speaker": "本人", "time": "2025-06-01 10:00", "text": f"今天去公园散步了第{i}次", "conversation_id": "c"} for i in range(170)]
    calls = []
    s = R.summarize_window(sc, msgs, lambda m: calls.append(m) or {"text": "不是 JSON"})
    assert len(calls) == 3 and s.status == "SUMMARY_FAILED" and not s.answerable
    trivial = [{"message_id": "x", "speaker": "本人", "time": "t", "text": "[图片]", "conversation_id": "c"}, {"message_id": "y", "speaker": "对方", "time": "t", "text": "哈哈哈", "conversation_id": "c"}]
    assert R.summarize_window(sc, trivial, lambda m: 1 / 0).status == "NOTHING_SUBSTANTIVE"


# ======================================================================= writer: third-party follow-up, chunking, candidate log
def test_third_party_event_follow_up_is_merged_and_verified(tmp_path):
    seen = []

    def model(messages):
        user = messages[-1]["content"]
        if user.startswith("只看下面标出的这条消息"):
            seen.append(user)
            label = re.search(r"source_message_ids 只写 \[\"(m\d+)\"\]", user).group(1)
            return {"text": json.dumps({"candidates": [MF.cand("person:小雨", "event", "结婚", label, "小雨下个月结婚", type="EVENT", event_key="wedding_xy",
                                                                 activity="结婚", event_status="PLANNED", polarity="neutral")]}, ensure_ascii=False)}
        d = json.loads(MF.reference_complete(messages)["text"])
        d["candidates"] = [c for c in d["candidates"] if c["predicate"] != "event" or "结婚" not in c["object"]]     # the model's miss
        return {"text": json.dumps(d, ensure_ascii=False)}
    path = MF.chat().write(tmp_path / "c.json")
    root = tmp_path / "r"
    pid = svc.init_profile(path, speaker="contact", root=root)["profile_id"]
    MS.build_evidence(root, pid)
    MS.build_semantic(root, pid, complete=model, reviewer="t")
    con = sqlite3.connect(S.profile_dir(root, pid) / "memory" / "memory.sqlite")
    rows = con.execute("SELECT subject, attribute, value FROM memories").fetchall()
    actor = {s: v for s, a, v in rows if a == "event.actor"}
    assert len(seen) == 1                                                                   # only the wedding line triggered a follow-up
    assert any(a == "event.description" and "结婚" in v and "小雨" in actor.get(s, "") for s, a, v in rows)
    assert not any(a == "event.description" and "结婚" in v and actor.get(s) == "self" for s, a, v in rows)
    log = [json.loads(l) for l in (S.profile_dir(root, pid) / "memory" / "writer_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(r["followup_calls"] == 1 for r in log)
    statuses = {c["status"] for r in log for c in r["candidates"]}
    assert "AUTO_VERIFIED" in statuses and "REJECTED" in statuses                         # the planted wrong candidates stay rejected
    assert all(c["source_message_ids"] for r in log for c in r["candidates"])


def test_follow_up_is_bounded_and_repair_turns_pass_through():
    calls = []

    def model(messages):
        calls.append(messages)
        return {"text": json.dumps({"candidates": []})}
    lines = "\n".join(f"[m{i}] 2025-01-01 10:0{i} 本人 → 对方：我妹妹下个月结婚{i}" for i in range(6))
    MS.event_followup(model)([{"role": "system", "content": "s"}, {"role": "user", "content": lines}])
    assert len(calls) == 1 + MS.FOLLOWUP_MAX
    calls.clear()
    MS.event_followup(model)([{"role": "system", "content": "s"}, {"role": "user", "content": lines}, {"role": "assistant", "content": "x"}])
    assert len(calls) == 1


def test_long_sessions_are_chunked_with_overlap():
    assert MS.chunks(10) == [(0, 10)]
    c = MS.chunks(173)
    assert c[0] == (0, 60) and c[-1][1] == 173 and all(b - a <= MS.CHUNK_MESSAGES for a, b in c)
    assert all(c[i + 1][0] == c[i][1] - MS.CHUNK_OVERLAP for i in range(len(c) - 1))


# ======================================================================= QA builder: verification, freeze
def test_qa_verification_rules():
    lines = {"Q01": {"message_id": "a", "speaker": "本人", "text": "我之前在广州上班", "time": "2025-01-01 10:00", "date": "2025-01-01", "conversation_id": "c"},
             "Q02": {"message_id": "b", "speaker": "对方", "text": "我在澳洲读书", "time": "2025-01-02 10:00", "date": "2025-01-02", "conversation_id": "c"}}
    ok = {"question": "你以前在哪个城市上班？", "reference_answer": "广州", "reference_facts": ["本人之前在广州上班"], "evidence": ["Q01"]}
    assert QB.verify_row(ok, lines, "PERSONA_SELF_EXPERIENCE") is None
    assert QB.verify_row({**ok, "evidence": ["Q09"]}, lines, "PERSONA_SELF_EXPERIENCE") == "evidence id not in the window"
    assert QB.verify_row({**ok, "reference_facts": ["本人在广州上班了3年"]}, lines, "PERSONA_SELF_EXPERIENCE") == "number of a fact not in the cited lines"
    assert QB.verify_row({**ok, "reference_facts": ["本人喜欢吃火锅"]}, lines, "PERSONA_SELF_EXPERIENCE") == "fact wording not in the cited lines"
    assert QB.verify_row({**ok, "question": "你以前在广州上班吗？", "reference_answer": "广州上班"}, lines, "PERSONA_SELF_EXPERIENCE") == "answer leaked into the question"
    assert QB.verify_row({**ok, "evidence": ["Q02"], "reference_facts": ["在澳洲读书"]}, lines, "PERSONA_SELF_EXPERIENCE") == "persona fact not said by the persona"
    assert "does not fit" in QB.verify_row({**ok, "question": "我以前在哪个城市上班？"}, lines, "PERSONA_SELF_EXPERIENCE")
    assert QB.verify_row(ok, lines, "CROSS_MESSAGE") == "cross-message question cites one line"
    assert QB.verify_row(ok, lines, "TEMPORAL_CHANGE") == "temporal question cites one date only"


class ModelDouble:
    """One scripted stand-in for the shared local model in the dry run (writer / QA builder / summariser / reader / persona chat).
    Its outputs are derived from the lines it is shown; it is a test double, never reported as a model result."""
    label = "SCRIPTED_DRY_RUN"

    def __init__(self):
        self.prompts = []

    def is_oom(self, e):
        return False

    def after_error(self):
        pass

    def __call__(self, messages, cfg, adapter):
        self.prompts.append(json.dumps(messages, ensure_ascii=False))
        sysm, user = messages[0]["content"], messages[-1]["content"]
        if "评测题构建助手" in sysm:
            return {"text": json.dumps(self.qa(user), ensure_ascii=False), "new_tokens": 20}
        if "聊天记录整理助手" in sysm:
            ids = re.findall(r"\[(D\d+)\] \S+ \S+ (本人|对方)：(.+)", user)
            pts = [{"text": f"{who}说{t[:12]}", "who": who, "evidence": [i]} for i, who, t in ids[:3] if len(t) >= 2]
            return {"text": json.dumps({"points": pts}, ensure_ascii=False), "new_tokens": 20}
        if "个人记忆助手" in sysm:
            return {"text": json.dumps({"answer": "根据现有记录，无法确定。", "answerable": False, "abstain": True, "evidence": [], "quotes": []}, ensure_ascii=False)}
        if "candidates" in sysm:
            return {"text": json.dumps({"candidates": []}), "new_tokens": 5}
        return {"text": "不记得了" if "没有找到" in sysm else ("好" if (adapter and len(user) % 3) else "嗯嗯 好呀"), "new_tokens": 3, "hit_token_limit": False}

    @staticmethod
    def qa(user):
        cat = next((c for c, ask in QB.CATEGORY_ASK.items() if ask in user), None)
        lines = re.findall(r"\[(Q\d+)\] (\S+ \S+) (本人|对方)：(.+)", user)
        want = {"PERSONA_SELF_EXPERIENCE": "本人", "COUNTERPART_EXPERIENCE": "对方"}.get(cat)
        pick = [l for l in lines if (want is None or l[2] == want) and len(l[3]) >= 4]
        if not pick or cat == "THIRD_PARTY":
            return {"skip": True}
        ev = [pick[0]] if cat not in ("CROSS_MESSAGE", "TEMPORAL_CHANGE") else pick[:3]
        forms = ["你那次说的是什么？", "你当时提到过哪件事？", "你之前讲过什么安排？", "你还记得自己说过啥吗？", "你后来怎么说的？", "你那会儿聊起过什么？"]
        mine = ["我那次说的是什么来着？", "我当时提到过哪件事来着？", "我之前讲过什么安排来着？", "我后来怎么说的来着？"]
        pick_form = int(hashlib.sha256(ev[0][3].encode()).hexdigest()[:6], 16)
        q = mine[pick_form % len(mine)] if cat == "COUNTERPART_EXPERIENCE" else forms[pick_form % len(forms)]
        return {"question": q, "reference_answer": ev[0][3], "reference_facts": [e[3] for e in ev], "evidence": [e[0] for e in ev], "subject": "PERSONA_SELF",
                "difficulty": "easy"}


def mock_transport(fail_first=()):
    """DeepSeek stand-in: valid rubric JSON (varied per prompt) with usage; ``fail_first`` unit texts get a schema error once."""
    seen = set()

    def t(url, headers, body, timeout):
        req = json.loads(body)
        sysm, user = req["messages"][0]["content"], req["messages"][1]["content"]
        h = int(hashlib.sha256(user.encode()).hexdigest()[:4], 16)
        if "对话评审员" in sysm:
            d = {"style_fidelity": h % 6, "contextual_appropriateness": (h // 6) % 6, "naturalness": 4, "non_repetition": 5, "hallucination": False, "reason": "r"}
        elif "记忆问答评审员" in sysm:
            d = {"factual_correctness": h % 6, "evidence_grounding": 5, "temporal_attribution": 4, "appropriate_abstention": 3, "persona_style": 3,
                 "fabricated_private_fact": False, "reason": "r"}
        else:
            d = {"fact_correctness": 4, "attribution": 5, "temporal": 4, "coverage": 3, "memory_worthiness": 5, "supported": [], "unsupported": [],
                 "missing": [{"fact": "x", "messages": ["m1"]}], "attribution_errors": [], "temporal_errors": [], "worthiness_errors": [], "summary": "s"}
        key = user[:60]
        if any(f in user for f in fail_first) and key not in seen and len(req["messages"]) == 2:
            seen.add(key)
            d = {"style_fidelity": "high"}
        out = {"id": "x", "model": req["model"], "choices": [{"message": {"content": json.dumps(d, ensure_ascii=False)}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": len(user) // 2, "completion_tokens": 60, "prompt_cache_hit_tokens": 0}}
        return 200, json.dumps(out).encode()
    return t


ENV = {"DEEPSEEK_API_KEY": "sk-test-not-real"}


# ======================================================================= judge
def test_judge_normalises_blinds_repairs_and_never_scores_failures(tmp_path):
    d = {"style_fidelity": 5, "contextual_appropriateness": 4, "naturalness": 3, "non_repetition": 5, "hallucination": False}
    assert J.score("style", d)["score_100"] == round(0.35 * 100 + 0.35 * 80 + 0.2 * 60 + 0.1 * 100, 2)
    with pytest.raises(J.JudgeError):
        J.validate("style", json.dumps({**d, "naturalness": 7}))
    with pytest.raises(J.JudgeError):
        J.validate("memory", "no json here")
    with pytest.raises(J.JudgeError):
        J.make_client({}, tmp_path / "l.jsonl")
    u1 = J.unit_id("style", "salt", "s1", "B1")
    assert u1 != J.unit_id("style", "salt", "s1", "C1") and "B1" not in u1
    client, model = J.make_client(ENV, tmp_path / "ledger.jsonl", transport=mock_transport(fail_first=("REPAIR-ME",)))
    units = [{"unit_id": J.unit_id("style", "salt", str(i), "g"), "context": f"对方：问题{i}", "reference": "好", "response": "REPAIR-ME" if i == 0 else f"回复{i}",
              "style_profile": "p", "repetition": {"group_size": 3, "count": 1, "top": "x"}} for i in range(3)]
    res = J.run(client, model, "style", units, tmp_path / "s.jsonl", seed="s", concurrency=2, log=lambda s: None)
    rows = J.read_results(tmp_path / "s.jsonl")
    assert res["ok"] == 3 and rows[units[0]["unit_id"]]["repair_turns"] == 1
    assert all("B1" not in json.dumps(r) and "C1" not in json.dumps(r) for r in rows.values())
    n = len((tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines())
    J.run(client, model, "style", units, tmp_path / "s.jsonl", seed="s", log=lambda s: None)
    assert len((tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()) == n          # resume: nothing re-sent

    def broken(url, headers, body, timeout):
        return 400, b'{"error": {"code": "invalid_request"}}'
    client2, _ = J.make_client(ENV, tmp_path / "l2.jsonl", transport=broken)
    res = J.run(client2, model, "style", units[:1], tmp_path / "f.jsonl", seed="s", log=lambda s: None)
    row = J.read_results(tmp_path / "f.jsonl")[units[0]["unit_id"]]
    assert res["failed"] == 1 and row["status"] == "JUDGE_FAILED" and "score_100" not in row and "sk-test" not in json.dumps(row)
    st = J.describe([10, 20, 30, 100])
    assert st["n"] == 4 and st["median"] == 25 and sum(st["histogram"].values()) == 4
    pr = J.paired({"a": 10, "b": 20}, {"a": 20, "b": 30})
    assert pr["mean_diff"] == 10 and pr["wins"] == 2


def test_judge_calibration_runs_separately(tmp_path):
    client, model = J.make_client(ENV, tmp_path / "ledger.jsonl", transport=mock_transport())
    rep = J.calibrate(client, model, tmp_path / "cal.jsonl", log=lambda s: None)
    assert rep["total"] == len(J.CALIBRATION) and all(r["status"] == "OK" for r in rep["cases"]) and "NOT part" in rep["note"]


# ======================================================================= notebook (static)
@REPO_ONLY
def test_notebook_is_one_run_all_notebook_with_valid_cells():
    nb = FP.notebook()
    assert nb["nbformat"] == 4 and nb["metadata"]["colab"]["gpuType"] == "L4"
    code = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
    for c in code:
        ast.parse(c)
    text = "\n".join(code)
    order = [int(m) for m in re.findall(r"# STAGE (\d+)", text)]
    assert order[:7] == [0, 1, 2, 3, 4, 5, 6] and 20 in order
    for fn in re.findall(r"CE\.(stage_\w+)\(", text):
        assert hasattr(CE, fn), fn
    assert 'userdata.get("DEEPSEEK_API_KEY")' in text and not re.search(r"sk-[A-Za-z0-9]{8,}", text)
    assert "llamafactory" not in json.dumps(FP.PINS_EVAL) and "test.jsonl" not in text
    assert json.loads(FP.notebook_bytes()) == nb
    repo_nb = ROOT / "notebooks" / FP.NB_NAME
    if not (ROOT / "notebooks").is_dir():                  # the public tree does not ship the research evaluation notebooks
        return
    assert repo_nb.is_file() and json.loads(repo_nb.read_text(encoding="utf-8")) == nb, "notebooks/StillHere_Final_L4.ipynb is stale: regenerate it"


# ======================================================================= full flow: package -> Colab stages -> results -> import
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    import test_persona_p2p3 as T
    from stillhere.persona import adapters as AD
    from stillhere.persona import memory as M
    from stillhere.persona import package as P
    from stillhere.persona import prompt_writer as PW
    from personatrain import export as EX
    mp = pytest.MonkeyPatch()
    orig = M.build_api
    mp.setattr(M, "build_api", lambda *a, **k: orig(*a, **{**k, "synthetic": True}))
    tmp = tmp_path_factory.mktemp("final")
    chat = F.standard(days=80).write(tmp / "chat.json")
    root = tmp / "personas"
    pid = svc.init_profile(chat, speaker="contact", root=root)["profile_id"]
    prep = svc.prepare_all(root, pid)
    pkg = P.build(root, pid, target="cpu-only")
    w = {"chat": chat, "root": root, "pid": pid, "pdir": S.profile_dir(root, pid), "ddir": Path(prep["dataset_dir"]), "prep": prep, "pkg": pkg, "tmp": tmp}
    MS.build_evidence(root, pid)
    run, data = T.make_run(w, tmp)
    aid = AD.import_adapter(root, pid, EX.adapter(run, run / "train", "checkpoint-6", "run-final", data))["adapter_id"]
    old = K.current(w["pdir"])["card_id"]
    text = ("你是一个 AI 数字人，模仿一个说话很简短的人和朋友微信聊天。" * 3 + "先回应对方最新的一句话，短不等于敷衍。"
            "不知道的事就说不知道或不记得，不要编造任何经历、人物、地点、时间或数字。" * 2 + "一次最多发三条消息，不用句号。" * 4)
    new = PW.save_candidate(w["pdir"], text, author="test", activate=False)["card_id"]
    yield {**w, "aid": aid, "old": old, "new": new}
    mp.undo()


@REPO_ONLY
def test_final_package_has_no_test_split_and_a_verifiable_manifest(world, tmp_path):
    r = FP.build(world["root"], world["pid"], old_card=world["old"], new_card=world["new"], adapter_id=world["aid"], out_dir=tmp_path / "pkg")
    out = Path(r["package_dir"])
    assert {p.name for p in out.iterdir()} == {FP.ZIP_NAME, FP.NB_NAME, "PACKAGE_MANIFEST.json", "SHA256SUMS.txt", "README_COLAB_CN.md"}
    man = json.loads((out / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    zb = (out / FP.ZIP_NAME).read_bytes()
    assert hashlib.sha256(zb).hexdigest() == man["input_zip_sha256"]
    with zipfile.ZipFile(out / FP.ZIP_NAME) as z:
        names = z.namelist()
        inner = json.loads(z.read("INNER_MANIFEST.json"))
        assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in inner["files"].items())
    assert not any(n.rsplit("/", 1)[-1].startswith("test") and "/datasets/" in n for n in names)
    assert any(n.endswith("memory/memory.sqlite") for n in names) and any(n.endswith("adapter_model.safetensors") for n in names)
    assert any(n == "code/src/stillhere/persona/colab_eval.py" for n in names) and any(n == "code/tests/test_persona_colab_final.py" for n in names)
    assert not any(n.endswith(".env") or "DEEPSEEK_API_KEY=" in z_text for n in names for z_text in [""])
    modes = man["counts"]["modes"]
    assert man["counts"]["default_mode"] == "FAST" and modes["FAST"]["plan"]["total"] == 720 and modes["FAST"]["plan"]["generations"]["total"] == 700
    assert modes["FULL"]["plan"]["total"] == 1772 and modes["FULL"]["plan"]["generations"]["total"] == 1672 and modes["FULL"]["memory_build"] == "semantic_full"
    assert man["prompts"]["old"]["card_id"] == world["old"] and man["counts"]["memory_groups"] == ["M0", "M1", "M2"] and man["contains"]["test_split"] is False
    assert man["counts"]["archive_messages"] > 0 and man["counts"]["text_messages"] > 0


@REPO_ONLY
def test_full_flow_on_cpu_resume_export_and_import(world, tmp_path):
    pk = FP.build(world["root"], world["pid"], old_card=world["old"], new_card=world["new"], adapter_id=world["aid"], out_dir=tmp_path / "pkg")
    man = pk["manifest"]
    colab = tmp_path / "colab"
    with zipfile.ZipFile(pk["zip"]) as z:
        z.extractall(colab)
    gen = ModelDouble()
    small = {"PERSONA_SELF_EXPERIENCE": 2, "COUNTERPART_EXPERIENCE": 2, "THIRD_PARTY": 1, "DATE_SUMMARY": 2, "TEMPORAL_CHANGE": 1, "NEGATION_PLAN_CANCEL": 1,
             "CROSS_MESSAGE": 1, "UNANSWERABLE": 2}
    cfg = CE.defaults({"eval_mode": "FULL", "qa_targets": small, "writer_audit_n": 5, "judge_concurrency": 2, "judge_per_second": 0})
    run = CE.Run(work_root=colab / "data" / "personas", profile_id=world["pid"], run_dir=tmp_path / "drive" / "run-x", manifest=man, cfg=cfg, gen=gen, log=lambda s: None)
    assert CE.bind(run) == "NEW" and CE.bind(run) == "RESUME"
    bad = CE.Run(work_root=run.work_root, profile_id=run.profile_id, run_dir=run.run_dir, manifest={**man, "input_zip_sha256": "0" * 64}, cfg=cfg, gen=gen)
    with pytest.raises(EV.EvalError):
        CE.bind(bad)                                                                     # a mismatching resume is refused
    rep = CE.stage_07_memory(run)
    assert rep["input_messages"] > 0 and (run.run_dir / "checkpoints" / "memory" / "memory.sqlite").is_file()
    assert CE.stage_07_memory(run)["input_messages"] == rep["input_messages"]          # done before: no rework
    CE.stage_08_09_reports(run)
    qa = CE.stage_10_qa(run)
    rows, sha = QB.load_frozen(run.run_dir / "qa")
    assert qa["questions"] == len(rows) and all(r["verification_status"] == QB.LABEL for r in rows)
    qrep = json.loads((run.run_dir / "qa" / "QA_BUILD_REPORT.json").read_text(encoding="utf-8"))
    assert all(v["built"] + v["shortfall"] == v["target"] for v in qrep["categories"].values())        # a shortfall is reported, never padded
    assert {r["category"] for r in rows} >= {"DATE_SUMMARY", "UNANSWERABLE", "NEGATION_PLAN_CANCEL"} and qrep["total"] == len(rows)
    assert all(r["evidence_message_ids"] or r["answerability"] == "UNANSWERABLE" for r in rows)
    with pytest.raises(FileExistsError):
        QB.freeze(rows, {}, run.run_dir / "qa")                                         # frozen: never rebuilt
    CE.stage_11_writer_sample(run)
    CE.stage_12_style(run)
    assert all(EV.done_rows(run.run_dir / "style" / g / "answers.jsonl") for g in CE.STYLE_GROUPS)
    sel = CE.stage_14_style_judge(run, ENV, transport=mock_transport())["selection"]
    assert sel["selected"] in ("old", "new") and "pre-registered" in sel["rule"]
    before = len(gen.prompts)
    CE.stage_13_memory_qa(run)
    questions = {r["question"] for r in rows}
    for raw in gen.prompts[before:]:                       # only the normal chain reaches the model: the question, then its own retrieval
        msgs = json.loads(raw)
        if "个人记忆助手" in msgs[0]["content"] or "聊天记录整理助手" in msgs[0]["content"]:
            continue
        assert msgs[-1]["content"].startswith("对方：") and msgs[-1]["content"][3:] in questions
        assert not any(r["reference_answer"] in msgs[0]["content"] for r in rows if r["answerability"] == "UNANSWERABLE")
    sel_prompt = K.load(run.pdir, sel["selected_card_id"])["prompt"]
    m0 = [json.loads(l) for l in (run.run_dir / "memory_qa" / "M0" / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len([r for r in m0 if r["status"] == "OK"]) == len(rows) and all(r["route"] == "chat" and r["reader_answerable"] is None for r in m0)
    m0_calls = [json.loads(p) for p in gen.prompts[before:] if json.loads(p)[0]["content"] == sel_prompt]
    assert {c[-1]["content"][3:] for c in m0_calls} == questions          # M0 = the selected prompt + the question, nothing retrieved
    CE.stage_15_memory_judge(run, ENV, transport=mock_transport())
    CE.stage_16_writer_judge(run, ENV, transport=mock_transport())
    rep = CE.stage_17_18_analysis(run)
    assert set(rep["combined"]) == {"M0", "M1", "M2"} and rep["combined"]["M1"]["combined_score"] is not None
    assert "NOT human" in rep["writer"]["note"] and rep["reader"]["M1"]["outcomes"]
    comps = rep["ablation"]["comparisons"]
    assert set(comps) == {"A_memory_contribution_M1_vs_M0", "B_adapter_contribution_M2_vs_M1", "C_prompt_contribution_B2_vs_B1", "D1_adapter_style_C1_vs_B1",
                          "D2_adapter_style_C2_vs_B2"}
    assert all("ci95" in c["paired"] and "relative_improvement_pct" in c for c in comps.values())
    cases = json.loads((run.run_dir / "reports" / "CASE_STUDY_SELECTION.json").read_text(encoding="utf-8"))
    assert cases["n"] == min(len(rows), 20) and "pre-registered" in cases["rule"]
    demo = (run.run_dir / "reports" / "ACADEMIC_DEMO_CASES.md").read_text(encoding="utf-8")
    assert not any(r["question"] in demo for r in rows)                   # de-identified: no question / chat text
    z = CE.stage_19_export(run, {"gpu": "cpu-dry-run"})
    with zipfile.ZipFile(z) as zz:
        names = set(zz.namelist())
    for n in ("RUN_MANIFEST.json", "SHA256SUMS.txt", "FULL_MEMORY_BUILD_REPORT.json", "MEMORY_CANDIDATES.jsonl", "VERIFIED_MEMORIES.jsonl", "AMBIGUOUS_MEMORIES.jsonl",
              "REJECTED_MEMORIES.jsonl", "MEMORY_LINEAGE.jsonl", "TEMPORAL_REPORT.json", "EF1_REPORT.json", "REAL_HISTORY_QA_200.jsonl", "QA_BUILD_REPORT.json",
              "WRITER_AUDIT_100.jsonl", "WRITER_AUDIT_REPORT.json", "STYLE_DEV_GENERATIONS.jsonl", "STYLE_DEEPSEEK_SCORES.jsonl", "STYLE_V2_REPORT.json",
              "MEMORY_QA_GENERATIONS.jsonl", "MEMORY_DEEPSEEK_SCORES.jsonl", "READER_EVALUATION_REPORT.json", "COMBINED_EVALUATION_REPORT.json",
              "PROMPT_COMPARISON_REPORT.json", "RUNTIME_RECOMMENDATION.json", "JUDGE_USAGE_REPORT.json", "ERROR_QUEUE.json", "FINAL_SUMMARY_CN.md",
              "ENVIRONMENT_REPORT.json", "memory_snapshot.sqlite", "ABLATION_REPORT.json", "REAL_CASE_STUDIES_20_30.md", "ACADEMIC_DEMO_CASES.md",
              "STILLHERE_ABLATION_RESULTS.md"):
        assert n in names, n
    assert not any("test.jsonl" in n for n in names) and "sk-test" not in Path(z).read_bytes().decode("latin-1")
    cur_before = (world["pdir"] / "cards" / "CURRENT.json").read_bytes()
    res = RI.import_results(world["root"], world["pid"], z)
    assert res["status"] == "IMPORTED" and (world["pdir"] / "cards" / "CURRENT.json").read_bytes() == cur_before
    assert RI.import_results(world["root"], world["pid"], z)["status"] == "ALREADY_IMPORTED"
    lt = RI.latest(world["root"], world["pid"])
    assert lt["imported"] and lt["memory"]["human_reviewed"] == 0 and "not human review" in lt["label"]
    tampered = tmp_path / "t.zip"
    with zipfile.ZipFile(z) as src, zipfile.ZipFile(tampered, "w") as dst:
        for n in src.namelist():
            b = src.read(n)
            dst.writestr(n, b + b" " if n == "COMBINED_EVALUATION_REPORT.json" else b)
    with pytest.raises(RI.ImportError_) as e:
        RI.import_results(world["root"], world["pid"], tampered)
    assert "SHA256" in str(e.value)


def test_import_refuses_a_human_review_claim(tmp_path):
    db = tmp_path / "s.sqlite"
    con = sqlite3.connect(db)
    con.executescript("CREATE TABLE messages (message_id, target_persona_id, source_id, content_sha256);"
                      "CREATE TABLE memories (memory_id, persona_id, subject, attribute, value, status, valid_from_utc, valid_to_utc);"
                      "CREATE TABLE provenance (memory_id, source_id); CREATE TABLE memory_events (reviewer);"
                      "INSERT INTO messages VALUES ('m1','p','s1','h'); INSERT INTO memories VALUES ('x','p','persona','a','v','active',NULL,NULL);"
                      "INSERT INTO provenance VALUES ('x','s1'); INSERT INTO memory_events VALUES ('Alice (human review)');")
    con.commit()
    con.close()
    with pytest.raises(RI.ImportError_) as e:
        RI.check_snapshot(db, "p", db)
    assert "REVIEW_INTEGRITY" in str(e.value)


def test_checkpoint_restore_roundtrip(tmp_path):
    work = tmp_path / "w"
    pdir = work / "p-x"
    (pdir / "memory").mkdir(parents=True)
    con = sqlite3.connect(pdir / "memory" / "memory.sqlite")
    con.execute("CREATE TABLE t (x)")
    con.execute("INSERT INTO t VALUES (1)")
    con.commit()
    con.close()
    (pdir / "memory" / "writer_candidates.jsonl").write_text('{"a": 1}\n', encoding="utf-8")
    run = CE.Run(work_root=work, profile_id="p-x", run_dir=tmp_path / "d", manifest={}, log=lambda s: None)
    CE.checkpoint_memory(run)
    (pdir / "memory" / "memory.sqlite").unlink()
    (pdir / "memory" / "writer_candidates.jsonl").unlink()
    assert CE.restore_memory(run)
    con = sqlite3.connect(pdir / "memory" / "memory.sqlite")
    assert con.execute("SELECT x FROM t").fetchone()[0] == 1
    con.close()
    assert (pdir / "memory" / "writer_candidates.jsonl").read_text(encoding="utf-8") == '{"a": 1}\n'


# ======================================================================= FAST_EVAL
def test_mode_plans_and_defaults():
    fast, full = CE.defaults({}), CE.defaults({"eval_mode": "full"})
    assert fast["eval_mode"] == "FAST" and fast["memory_build"] == "evidence_only" and fast["style_n"] == 100 and fast["writer_audit_n"] == 20
    assert sum(fast["qa_targets"].values()) == 100 and fast["qa_targets"]["DATE_SUMMARY"] == 18 and fast["qa_targets"]["UNANSWERABLE"] == 10
    assert CE.plan_units(fast) == {"style": 400, "memory": 300, "writer": 20, "total": 720, "generations": {"style": 400, "memory": 300, "total": 700}}
    assert CE.plan_units(full)["total"] == 1772 and full["memory_build"] == "semantic_full" and full["qa_backfill"] is False
    with pytest.raises(ValueError):
        CE.defaults({"eval_mode": "TURBO"})
    assert CE.export_names("FAST")["REAL_HISTORY_QA_200.jsonl"] == "MEMORY_QA_100.jsonl" and CE.export_names("FULL") == {}


def test_qa_backfill_fills_a_shortfall_from_other_real_categories(tmp_path):
    path = MF.chat().write(tmp_path / "c.json")
    root = tmp_path / "r"
    pid = svc.init_profile(path, speaker="contact", root=root)["profile_id"]
    MS.build_evidence(root, pid)
    db = S.profile_dir(root, pid) / "memory" / "memory.sqlite"
    targets = {"PERSONA_SELF_EXPERIENCE": 1, "COUNTERPART_EXPERIENCE": 1, "THIRD_PARTY": 2, "DATE_SUMMARY": 0, "TEMPORAL_CHANGE": 0,
               "NEGATION_PLAN_CANCEL": 1, "CROSS_MESSAGE": 0, "UNANSWERABLE": 0}
    gen = ModelDouble()
    complete = lambda m: gen(m, {}, None)   # noqa: E731
    rows0, rep0 = QB.build(db, pid, complete, targets=targets, log=lambda s: None)
    rows1, rep1 = QB.build(db, pid, complete, targets=targets, log=lambda s: None, backfill=True)
    assert rep0["categories"]["THIRD_PARTY"]["shortfall"] == 2 and not rep0["backfill"]["added"]          # the double skips third-party windows
    assert sum(rep1["backfill"]["added"].values()) > 0 and len(rows1) > len(rows0)
    assert all(r["verification_status"] == QB.LABEL and r["evidence_message_ids"] for r in rows1)            # backfill rows are real, verified rows
    assert rep1["actual_distribution"] == dict(__import__("collections").Counter(r["category"] for r in rows1))


@REPO_ONLY
def test_fast_flow_evidence_memory_writer_audit_export_and_import(world, tmp_path):
    from stillhere.persona import dataset as D
    pk = FP.build(world["root"], world["pid"], old_card=world["old"], new_card=world["new"], adapter_id=world["aid"], out_dir=tmp_path / "pkg")
    man = pk["manifest"]
    colab = tmp_path / "colab"
    with zipfile.ZipFile(pk["zip"]) as z:
        z.extractall(colab)
    gen = ModelDouble()
    small = {"PERSONA_SELF_EXPERIENCE": 1, "COUNTERPART_EXPERIENCE": 1, "THIRD_PARTY": 1, "DATE_SUMMARY": 2, "TEMPORAL_CHANGE": 1, "NEGATION_PLAN_CANCEL": 1,
             "CROSS_MESSAGE": 1, "UNANSWERABLE": 2}
    cfg = CE.defaults({"style_n": 6, "qa_targets": small, "writer_audit_n": 4, "judge_concurrency": 2, "judge_per_second": 0})
    assert cfg["eval_mode"] == "FAST"
    run = CE.Run(work_root=colab / "data" / "personas", profile_id=world["pid"], run_dir=tmp_path / "drive" / "fast-run-x", manifest=man, cfg=cfg, gen=gen, log=lambda s: None)
    assert CE.bind(run) == "NEW"
    full_cfg = CE.defaults({"eval_mode": "FULL", "qa_targets": small, "writer_audit_n": 4})
    with pytest.raises(EV.EvalError):
        CE.bind(CE.Run(work_root=run.work_root, profile_id=run.profile_id, run_dir=run.run_dir, manifest=man, cfg=full_cfg))   # FULL may not reuse a FAST run
    reads = []
    real = D.load_split
    import unittest.mock as um
    with um.patch.object(D, "load_split", lambda d, sp, pu: reads.append(sp) or real(d, sp, pu)):
        with CE.timer(run, "STAGE 07 memory"):
            mm = CE.stage_07_memory(run)
        assert mm["memory_mode"] == "EVIDENCE_MEMORY" and "NOT_BUILT" in mm["structured_semantic_memory"] and mm["evidence_archive"]["messages"] > 0
        con = sqlite3.connect(run.pdir / "memory" / "memory.sqlite")
        assert con.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0                      # M1 / M2 read the evidence archive only
        con.close()
        wlog = [json.loads(l) for l in (CE.writer_pdir(run) / "memory" / "writer_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
        sample = json.loads((run.run_dir / "writer_audit" / "WRITER_AUDIT_SAMPLE.json").read_text(encoding="utf-8"))
        assert {r["conversation_id"] for r in wlog} <= {x["conversation_id"] for x in sample["sessions"]} and sample["n"] == 4   # the real Writer ran on the sample only
        assert CE.stage_07_memory(run)["memory_mode"] == "EVIDENCE_MEMORY"                         # done before
        CE.stage_08_09_reports(run)
        CE.stage_10_qa(run)
        CE.stage_11_writer_sample(run)
        CE.stage_12_style(run)
        ids = json.loads((run.run_dir / "reports" / "STYLE_SAMPLE_MANIFEST.json").read_text(encoding="utf-8"))["sample_ids"]
        assert len(ids) == 6 and all(len(EV.done_rows(run.run_dir / "style" / g / "answers.jsonl")) == 6 for g in CE.STYLE_GROUPS)
        assert CE.style_sample(run, EV.eval_items(next((run.pdir / "datasets").iterdir()), "dev")) == ids          # frozen, deterministic
        CE.stage_14_style_judge(run, ENV, transport=mock_transport())
        CE.stage_13_memory_qa(run)
        CE.stage_15_memory_judge(run, ENV, transport=mock_transport())
        CE.stage_16_writer_judge(run, ENV, transport=mock_transport())
        rep = CE.stage_17_18_analysis(run)
        z = CE.stage_19_export(run, {"gpu": "cpu-dry-run"})
    assert "test" not in reads                                                                      # the sealed test is never read
    rows, _ = QB.load_frozen(run.run_dir / "qa")
    m1 = [json.loads(l) for l in (run.run_dir / "memory_qa" / "M1" / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(r["route"] in ("fact", "date") and r["retrieved"] > 0 for r in m1)                   # memory ON retrieves from the full archive
    assert "memory_switch_note" in rep["ablation"] and rep["ablation"]["memory_mode"] == "EVIDENCE_MEMORY"
    assert z.name == "StillHere_FAST_L4_Results.zip"
    with zipfile.ZipFile(z) as zz:
        names = set(zz.namelist())
        mode = json.loads(zz.read("EVAL_MODE.json"))
        usage = json.loads(zz.read("JUDGE_USAGE_REPORT.json"))
    for n in ("RUN_MANIFEST.json", "EVAL_MODE.json", "ENVIRONMENT_REPORT.json", "FAST_EVAL_SUMMARY_CN.md", "STYLE_FAST_GENERATIONS.jsonl", "STYLE_FAST_DEEPSEEK_SCORES.jsonl",
              "STYLE_V2_REPORT.json", "MEMORY_QA_100.jsonl", "MEMORY_FAST_GENERATIONS.jsonl", "MEMORY_FAST_DEEPSEEK_SCORES.jsonl", "MEMORY_ABLATION_REPORT.json",
              "WRITER_AUDIT_20.jsonl", "WRITER_FAST_REPORT.json", "WRITER_ERROR_BREAKDOWN.json", "WRITER_FAST_REPORT.md", "FAST_ACADEMIC_RESULTS.md",
              "FAST_ABLATION_REPORT.md", "FAST_CASE_STUDIES.md", "FAST_INTERVIEW_SUMMARY_EN_CN.md", "JUDGE_USAGE_REPORT.json", "ERROR_QUEUE.json", "SHA256SUMS.txt",
              "STAGE_TIMINGS.json", "MEMORY_MODE.json", "STYLE_SAMPLE_MANIFEST.json"):
        assert n in names, n
    assert "memory_snapshot.sqlite" not in names and mode["memory_mode"] == "EVIDENCE_MEMORY" and "Semantic Memory Full Build Not Executed" in mode["labels"]
    assert usage["target_total"] == 4 * 6 + 3 * sum(small.values()) + 4 and usage["ok_total"] == usage["planned_total"]
    local_db = world["pdir"] / "memory" / "memory.sqlite"
    before = hashlib.sha256(local_db.read_bytes()).hexdigest()
    res = RI.import_results(world["root"], world["pid"], z)
    assert res["status"] == "IMPORTED" and res["eval_mode"] == "FAST" and res["memories"] is None and "NOT_APPLICABLE" in res["memory_action"]
    assert hashlib.sha256(local_db.read_bytes()).hexdigest() == before                              # FAST never touches the memory store
    lt = RI.latest(world["root"], world["pid"])
    assert lt["eval_mode"] == "FAST" and "evidence-only" in lt["label"] and lt["ablation"]
    assert RI.import_results(world["root"], world["pid"], z)["status"] == "ALREADY_IMPORTED"


def test_engineering_pytest_subprocess_sees_the_notebook_import_path(tmp_path, monkeypatch):
    t = tmp_path / "test_probe.py"
    t.write_text("import stillhere.persona.colab_eval\n\ndef test_ok():\n    assert True\n", encoding="utf-8")
    monkeypatch.delenv("PYTHONPATH", raising=False)              # as in Colab: the code is only on this process's sys.path
    rc, tail = CE.run_pytest([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(t)])
    assert rc == 0, tail
    rc, tail = CE.run_pytest([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path / "missing_test.py")])
    assert rc != 0 and tail.strip()                               # a failure always carries its message
