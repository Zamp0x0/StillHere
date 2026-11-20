"""Final quality round: strict two-person attribution, cancellation-by-id repair, memory-worthiness, honest review records, Style Evaluator V2
(person-adaptive), Fact Reader routing in the persona runtime, prompt generator providers + validator, local model configuration.
Synthetic data only; no model, no network."""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import persona_fixture as F  # noqa: E402
import persona_memory_fixture as MF  # noqa: E402
from stillhere.persona import card as K  # noqa: E402
from stillhere.persona import config as CF  # noqa: E402
from stillhere.persona import evaluation as EV  # noqa: E402
from stillhere.persona import memory_store as MS  # noqa: E402
from stillhere.persona import prompt_writer as PW  # noqa: E402
from stillhere.persona import runtime as RT  # noqa: E402
from stillhere.persona import service as svc  # noqa: E402
from stillhere.persona import store as S  # noqa: E402
from stillhere.persona import style_v2 as SV  # noqa: E402
from stillhere.writer.contracts import UNSUPPORTED, Candidate, Verdict  # noqa: E402
from stillhere.writer.extract import GroupMessage  # noqa: E402
from stillhere.writer.verify import verify  # noqa: E402

TZ = timezone(timedelta(hours=8))
T = datetime(2025, 1, 1, 9, tzinfo=TZ)


def cand(**kw):
    base = dict(candidate_id="c.0", persona_id="p", type="EVENT", subject="self", predicate="event", object="上班", polarity="positive", assertion_type="stated",
                certainty="certain", event_key="work", event_status="IN_PROGRESS", activity="上班", daypart=None, planned_time=None, event_time=None, valid_from=None,
                valid_to=None, temporal_precision="unknown", corrects=False, retracts=False, refers_to=None, source_message_ids=["w1"], source_spans=["我在上班"],
                extraction_version="t")
    base.update(kw)
    return Candidate(**base)


OTHER_SAYS = [GroupMessage("m1", "w1", "p.other", "对方", "本人", T, T, "我在上班呢", "s")]


# ---------------------------------------------------------------- Writer attribution / repair / worthiness / review
def test_strict_two_person_attribution_rejects_the_other_persons_event_default_unchanged():
    assert verify(cand(), OTHER_SAYS, "p", []).committable                     # the SMS rule (addressed to the owner) - default behaviour unchanged
    v = verify(cand(), OTHER_SAYS, "p", [], strict_self=True)
    assert v.support == UNSUPPORTED and "owner did not say" in v.reasons[0]
    own = [GroupMessage("m1", "w1", "p", "本人", "对方", T, T, "我在上班呢", "s")]
    assert verify(cand(), own, "p", [], strict_self=True).committable


def test_cancellation_by_event_id_is_repaired_and_momentary_states_are_not_memories():
    existing = [("x1", "event:abc123", "event.description", "看牙医")]
    c = cand(object="abc123", event_status="CANCELLED", refers_to=None)
    fixed = MS.fix_candidate(c, [], existing)
    assert fixed.object == "看牙医" and fixed.refers_to == "event:abc123"
    momentary = Verdict(cand(object="去啦", event_status=None, source_spans=["去啦"]), "SUPPORTED")
    assert MS.worthy(momentary, []).support == UNSUPPORTED
    planned = Verdict(cand(object="看牙", event_status="PLANNED", source_spans=["看牙"]), "SUPPORTED")
    assert MS.worthy(planned, []).committable                                   # short wording with a change is still a memory


def test_writer_format_repair_is_bounded_and_inverted_kin_relation_is_the_owners():
    raw = {"candidates": [{"predicate": "event", "event_status": "CANCELLED", "polarity": "neutral", "certainty": "maybe"},
                          {"predicate": "state", "event_status": None, "polarity": "neutral"}]}
    out = json.loads(MS.normalize_writer_json(json.dumps(raw, ensure_ascii=False)))["candidates"]
    assert out[0]["polarity"] == "positive" and out[0]["certainty"] == "uncertain"
    assert out[1]["polarity"] == "neutral"                                      # not a status-bearing event: left for the schema check to reject
    assert MS.normalize_writer_json("not json") == "not json"
    call = MS.repaired(lambda m: {"text": json.dumps(raw), "model": "x"})
    assert json.loads(call([])["text"])["candidates"][0]["polarity"] == "positive"
    inv = cand(type="RELATIONSHIP", subject="person:小雨", predicate="relationship", object="妹妹", event_key=None, event_status=None, activity=None)
    fixed = MS.fix_candidate(inv, [], [])
    assert fixed.subject == "self" and fixed.object == "妹妹 小雨"
    other = cand(type="RELATIONSHIP", subject="person:小雨", predicate="relationship", object="很熟", event_key=None, event_status=None, activity=None)
    assert MS.fix_candidate(other, [], []) is other                             # not a kin word: unchanged


def test_review_record_is_auto_verified_not_human_reviewed(tmp_path, monkeypatch):
    path = MF.chat().write(tmp_path / "c.json")
    root = tmp_path / "r"
    pid = svc.init_profile(path, speaker="contact", root=root)["profile_id"]
    MS.build_evidence(root, pid)
    seen = {}
    from stillhere.writer import pipeline as WP
    orig = WP.Writer.__init__

    def spy(self, *a, **k):
        seen.update(k)
        orig(self, *a, **k)
    monkeypatch.setattr(WP.Writer, "__init__", spy)
    r = MS.build_semantic(root, pid, complete=MF.reference_complete, reviewer="Owner")
    assert seen["reviewer"] == "auto_verified;accepted_by:Owner" and seen["strict_self_attribution"] is True and seen["prompt_version"] == "writer-extract-v5"
    assert r["memories_active"] > 0
    with pytest.raises(MS.MemoryError_) as e:
        MS.build_semantic(root, pid, complete=MF.reference_complete)
    assert "accept-auto-verified-by" in str(e.value)


# ---------------------------------------------------------------- Style Evaluator V2
class Ref:
    def __init__(self, targets):
        self.train_targets = targets


def _run(tmp, name, texts, refs, queries):
    run = tmp / name
    (run / "C").mkdir(parents=True)
    items = []
    with open(run / "C" / "answers.jsonl", "w", encoding="utf-8") as f:
        for i, (t, r, q) in enumerate(zip(texts, refs, queries)):
            f.write(json.dumps({"sample_id": f"s{i}", "group": "C", "status": "OK", "text": t}, ensure_ascii=False) + "\n")
            items.append({"sample_id": f"s{i}", "reference": r, "query": q, "user": "对方：" + q})
    return run, items


SHORT_TRAIN = ["好", "行", "嗯嗯", "哈哈哈", "没吃呢", "好的呀", "在呢", "去啦", "到家了", "累死", "明天吧", "想你"] * 20 + [f"刚到{i}号楼" for i in range(30)]


def test_v2_accepts_authentic_short_replies_and_flags_real_collapse(tmp_path):
    refs = ["好", "没吃呢", "在呢", "刚到3号楼", "明天吧", "嗯嗯", "去啦", "累死", "好的呀", "到家了"] * 3
    qs = ["吃了吗", "你在哪", "睡了吗", "到哪了？", "什么时候", "好不好", "走吗", "累不累", "行吗", "回家没"] * 3
    varied, items = _run(tmp_path, "varied", ["好", "还没", "在呢", "快到了", "明天", "嗯嗯", "走", "累", "行呀", "到了"] * 3, refs, qs)
    r = SV.evaluate(varied, items, ["C"], Ref(SHORT_TRAIN))
    assert r["groups"]["C"]["checks"]["length_in_person_range"] and r["groups"]["C"]["checks"]["no_conditional_collapse"]     # 1-5 chars are fine
    col, items2 = _run(tmp_path, "collapse", ["好"] * 30, refs, qs)
    r2 = SV.evaluate(col, items2, ["C"], Ref(SHORT_TRAIN))
    assert not r2["groups"]["C"]["checks"]["no_conditional_collapse"] and r2["groups"]["C"]["excess_repetition"]
    long_train = [f"我今天下午去了一趟图书馆借了几本书{i}" for i in range(60)]
    lr, items3 = _run(tmp_path, "long", ["好"] * 10 + ["嗯"] * 10 + ["行"] * 10, [x for x in long_train[:30]], qs)
    r3 = SV.evaluate(lr, items3, ["C"], Ref(long_train))
    assert not r3["groups"]["C"]["checks"]["length_in_person_range"]                                                       # adaptive, not "short is good"


# ---------------------------------------------------------------- Fact Reader routing
class Scripted(EV.FakeGenerator):
    """Answers the Fact Reader with a fixed JSON and the persona style pass with a fixed text (records what it was asked)."""

    def __init__(self, reader_json, styled):
        self.reader_json, self.styled, self.calls = reader_json, styled, []

    def __call__(self, messages, cfg, adapter):
        self.calls.append(messages)
        if "个人记忆助手" in messages[0]["content"]:
            return {"text": json.dumps(self.reader_json, ensure_ascii=False), "new_tokens": 10}
        return {"text": self.styled, "new_tokens": 3}


@pytest.fixture()
def rt(tmp_path):
    path = MF.chat().write(tmp_path / "c.json")
    root = tmp_path / "r"
    pid = svc.init_profile(path, speaker="contact", root=root)["profile_id"]
    svc.prepare  # dataset not needed for production memory
    MS.build_evidence(root, pid)
    MS.build_semantic(root, pid, complete=MF.reference_complete, reviewer="t")
    from stillhere.persona import dataset as D  # noqa: F401
    pdir = S.profile_dir(root, pid)
    card = {"schema": K.CARD_SCHEMA, "persona_id": pid, "display_name": "测试", "ai_identity": "你是 AI", "reply_length": {}, "style_traits": [], "examples": [],
            "evidence": {"split": "train", "dataset_version": "v", "train_sha256": "x", "profiler_version": "p", "example_ids": [], "method": "m"},
            "review_status": K.AUTO, "authored_prompt": "你是 AI 数字人。" + "说话很短。" * 60 + "没有依据时说不知道，不要编造。"}
    res = K._write_version(pdir, card, parent=None)
    K.set_current(pdir, res["card_id"])
    return RT.PersonaRuntime(root, pid, backend="fake", memory="production")


def test_history_question_goes_through_the_fact_reader_and_keeps_the_answer(rt):
    rt.gen = Scripted({"answer": "本人以前在广州工作，现在在深圳。", "answerable": True, "abstain": False, "evidence": ["E02", "E05"], "quotes": []}, "广州啊 现在深圳")
    r = rt.reply([("other", "你以前在哪个城市工作？")], now=datetime(2025, 2, 1, tzinfo=TZ))
    assert r["route"] == "fact" and r["reader"]["answerable"] and r["reader"]["final_check"] == "ANSWER_KEPT" and "广州" in r["text"]
    reader_call = rt.gen.calls[0]
    assert "个人记忆助手" in reader_call[0]["content"] and "本人以前在哪个城市工作" in reader_call[1]["content"]     # Fact Reader B, owner frame
    assert "【记忆证据】" in reader_call[1]["content"] and "要表达的内容" in rt.gen.calls[1][0]["content"]
    rt.gen = Scripted({"answer": "本人以前在广州工作。", "answerable": True, "abstain": False, "evidence": ["E02"], "quotes": []}, "北京")
    r = rt.reply([("other", "你以前在哪个城市工作？")], now=datetime(2025, 2, 1, tzinfo=TZ))
    assert r["reader"]["final_check"] == "FALLBACK_TO_READER_ANSWER" and "广州" in r["text"] and "北京" not in r["text"]      # a changed fact is not kept
    rt.gen = Scripted({"answer": "本人以前在广州工作。", "answerable": True, "abstain": False, "evidence": ["E01"], "quotes": []}, "广州")
    r = rt.reply([("other", "你以前在哪个城市工作？")], now=datetime(2025, 2, 1, tzinfo=TZ))
    assert not r["reader"]["answerable"] and r["reader"]["fail"] == "SUBJECT_RESOLUTION_FAIL"     # E01 is the asker's question, not the persona's words


def test_unknown_question_never_fabricates_and_small_talk_skips_the_reader(rt):
    rt.gen = Scripted({"answer": "根据现有记录，无法确定。", "answerable": False, "abstain": True, "evidence": [], "quotes": []}, "养过一只猫叫咪咪")
    r = rt.reply([("other", "你小时候养过什么宠物吗？")])
    assert r["route"] == "fact" and r["reader"]["abstain"] and "咪咪" not in r["text"] and r["reader"]["final_check"] == "FALLBACK_MINIMAL_UNCERTAIN"
    rt.gen = Scripted({"answer": "根据现有记录，无法确定。", "answerable": False, "abstain": True, "evidence": [], "quotes": []}, "不记得了呀")
    assert rt.reply([("other", "你小时候养过什么宠物吗？")])["text"] == "不记得了呀"                                        # natural, in style
    rt.gen = Scripted({}, "在呢")
    r = rt.reply([("other", "在吗")])
    assert r["route"] == "chat" and r["reader"] is None and len(rt.gen.calls) == 1


# ---------------------------------------------------------------- prompt generator + configuration
def test_prompt_validator_and_providers(tmp_path, monkeypatch):
    path = F.standard(days=80).write(tmp_path / "c.json")
    root = tmp_path / "r"
    pid = svc.init_profile(path, speaker="contact", root=root)["profile_id"]
    svc.prepare_all(root, pid)
    pdir = S.profile_dir(root, pid)
    cur = K.current(pdir)
    assert "biography" in " ".join(PW.validate_prompt("你是 AI。" + "不要编造。" * 40 + "她今年28岁，住在杭州。", cur["card"]))
    assert PW.validate_prompt("说话很短" * 80, cur["card"])                                                             # no AI disclosure / rule
    good = "你是一个 AI 数字人。" + "回复要短，先回应对方最新一句。" * 15 + "没有依据时说不知道，不要编造经历。"
    res = PW.save_candidate(pdir, good, author="test")
    assert not res["activated"] and K.current(pdir)["card_id"] == cur["card_id"] and K.load(pdir, res["card_id"])["prompt"].startswith("你是一个 AI")
    sent = {}

    class Resp:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": self.body}}]}).encode()

    def opener(req, timeout=0):
        sent["payload"] = json.loads(req.data)
        return Resp(good + "（API）")
    cfg = CF.Config(prompt_provider="openai_compatible", prompt_api_base_url="http://example.invalid/v1", prompt_api_model="m")
    r = PW.generate(root, pid, cfg=cfg, opener=opener)
    user = json.loads(sent["payload"]["messages"][1]["content"])
    assert r["provider"] == "openai_compatible" and "examples" not in user and not r["activated"]                      # statistics only by default
    r2 = PW.generate(root, pid, cfg=cfg, opener=lambda req, timeout=0: (_ for _ in ()).throw(OSError("down")))
    assert r2["provider"] == "deterministic" and "fallback" in r2
    assert PW.generate(root, pid, cfg=CF.Config())["provider"] == "deterministic"


def test_configuration_precedence_secrets_and_no_gpu_message(tmp_path, monkeypatch):
    dot = tmp_path / ".env"
    dot.write_text("STILLHERE_BASE_MODEL_DIR=/from/dotenv\nSTILLHERE_DEVICE=cuda:1\nSTILLHERE_PROMPT_API_KEY=sk-secret\n", encoding="utf-8")
    c = CF.load(dotenv=dot, env={"STILLHERE_DEVICE": "cuda:2"})
    assert c.base_model_dir == "/from/dotenv" and c.device == "cuda:2" and c.device_index == 2 and c.public()["prompt_api_key"] == "***set***"
    pdir = tmp_path / "p"
    (pdir / "runtime").mkdir(parents=True)
    (pdir / "runtime" / "settings.json").write_text(json.dumps({"base_model_dir": "/from/profile", "prompt_api_key": "ignored"}), encoding="utf-8")
    c2 = CF.load(dotenv=dot, env={}, profile_dir=pdir, overrides={"backend": "hf-lowvram"})
    assert c2.base_model_dir == "/from/profile" and c2.backend == "hf-lowvram" and c2.prompt_api_key == "sk-secret" and c2.writer_dir() == "/from/profile"
    with pytest.raises(ValueError):
        CF.load(dotenv=dot, env={"STILLHERE_LOCAL_BACKEND": "cloud"})
    from stillhere.persona import doctor as DR
    monkeypatch.setattr(DR, "gpu_check", lambda nf4=False: {"cuda_available": False})
    with pytest.raises(RuntimeError) as e:
        CF.resolve_backend(CF.Config())
    assert "NO_GPU" in str(e.value)
    monkeypatch.setattr(DR, "gpu_check", lambda nf4=False: {"cuda_available": True, "vram_total_gib": 8.0})
    assert CF.resolve_backend(CF.Config()) == "hf-lowvram"
