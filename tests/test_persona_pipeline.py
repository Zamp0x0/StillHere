"""BUILD P1 persona pipeline: WeFlow parsing, speaker identity, cleaning, dialogue reconstruction, splits, leakage, local-only privacy,
idempotency and the CLI (simplified: no consent form for local processing). Synthetic exports only (tests/persona_fixture.py); no real chat is ever read here."""
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import persona_fixture as F  # noqa: E402
from stillhere.persona import cleaning as C  # noqa: E402
from stillhere.persona import dataset as D  # noqa: E402
from stillhere.persona import dialogue as G  # noqa: E402
from stillhere.persona import identity as I  # noqa: E402
from stillhere.persona import service as svc  # noqa: E402
from stillhere.persona import store as S  # noqa: E402
from stillhere.persona import weflow as W  # noqa: E402
from stillhere.persona import cli as PC  # noqa: E402

@pytest.fixture()
def chat_file(tmp_path):
    return F.standard().write(tmp_path / "chat.json")


@pytest.fixture()
def root(tmp_path):
    return tmp_path / "personas"


def ready_profile(path, root, **kw):
    return svc.init_profile(path, speaker="contact", root=root, **kw)


def rows(ddir, split):
    return [json.loads(l) for l in (Path(ddir) / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def metas(ddir, split):
    return [json.loads(l) for l in (Path(ddir) / f"{split}_meta.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------- parsing
def test_parse_standard_and_report(chat_file):
    exp = W.load(chat_file)
    assert exp.report["messages"] == len(exp.messages) and exp.session["type"] == "私聊" and exp.report["timezone"] == "+08:00"
    assert exp.report["formattedTime_mismatch"] == 0 and exp.sha256 == hashlib.sha256(chat_file.read_bytes()).hexdigest()


def test_group_chat_missing_fields_and_bad_time_are_refused(tmp_path):
    with pytest.raises(W.WeFlowError) as e:
        W.parse(F.Chat(session_type="群聊").obj() | {"messages": F.standard(1).msgs})
    assert e.value.code == "UNSUPPORTED_SESSION"
    o = F.standard(1).obj()
    del o["messages"][0]["senderUsername"]
    with pytest.raises(W.WeFlowError) as e:
        W.parse(o)
    assert e.value.code == "MISSING_FIELD"
    o = F.standard(1).obj()
    o["messages"][0]["createTime"] = 5
    with pytest.raises(W.WeFlowError) as e:
        W.parse(o)
    assert e.value.code == "BAD_TIME"
    with pytest.raises(W.WeFlowError) as e:
        W.load(tmp_path / "missing.json")
    assert e.value.code == "NOT_FOUND"


def test_duplicate_ids_identical_dropped_conflicting_refused_unknown_fields_tolerated():
    o = F.standard(2).obj()
    o["messages"].append(dict(o["messages"][3]))                     # a re-imported identical record
    o["messages"][0]["newExporterField"] = 1
    exp = W.parse(o)
    assert exp.report["duplicate_imports_dropped"] == 1 and exp.report["unknown_fields"] == {"newExporterField": 1}
    bad = dict(o["messages"][3])
    bad["content"] = "不同的内容"
    o["messages"].append(bad)
    with pytest.raises(W.WeFlowError) as e:
        W.parse(o)
    assert e.value.code == "DUPLICATE_ID"


def test_out_of_order_sorted_and_same_second_messages_kept():
    c = F.Chat()
    c.text(F.T0 + 100, F.SELF, "后发的")
    c.text(F.T0, F.CONTACT, "先发的")
    c.text(F.T0, F.CONTACT, "同一秒另一条")
    exp = W.parse(c.obj())
    assert exp.report["out_of_order_in_file"] == 1 and [m.create_time for m in exp.messages] == sorted(m.create_time for m in exp.messages)
    assert len(exp.messages) == 3


# ---------------------------------------------------------------- identity
def test_identity_uses_username_not_isSend_and_system_is_separate(chat_file):
    ident = I.resolve(W.load(chat_file))
    c, s = ident.by_role("contact"), ident.by_role("self")
    assert c.username == F.CONTACT and s.username == F.SELF and c.is_send == 0 and s.is_send == 1
    assert ident.system_messages == 80 and ident.system_by_attribution == {"self|isSend=0": 80}       # system under SELF with isSend 0
    assert not ident.anomalies and c.display_name == "小合成" and c.display_source == "session.displayName"
    assert ident.quotes["linked"] == ident.quotes["quotes"] == 80


def test_identity_anomalies_same_name_multi_name_and_isSend_conflict():
    c = F.Chat(contact_name="同名", self_name="同名")
    c.text(F.T0, F.SELF, "嗨")
    c.text(F.T0 + 5, F.CONTACT, "嗨嗨")
    c.text(F.T0 + 9, F.CONTACT, "换了个名字", display="另一个名字")
    ident = I.resolve(W.parse(c.obj()))
    codes = {a["code"] for a in ident.anomalies}
    assert {"SAME_DISPLAY_NAME", "MULTI_DISPLAY_NAME"} <= codes and not ident.blocking
    c.text(F.T0 + 20, F.CONTACT, "方向不对", is_send=1)
    assert [a["code"] for a in I.resolve(W.parse(c.obj())).blocking] == ["IS_SEND_CONFLICT"]


def test_three_people_refused_and_speaker_lookup():
    c = F.standard(1)
    c.text(F.T0 + 5000, "wxid_third", "我是第三个人", is_send=0)
    with pytest.raises(I.IdentityError) as e:
        I.resolve(W.parse(c.obj()))
    assert e.value.code == "PARTICIPANTS"
    ident = I.resolve(W.parse(F.standard(1).obj()))
    assert ident.find("1").role == "contact" and ident.find("self").username == F.SELF and ident.find(ident.speakers[0].id).role == "contact"
    with pytest.raises(I.IdentityError):
        ident.find("9")


def test_init_blocks_isSend_conflict_unless_accepted(tmp_path, root):
    c = F.standard(4)
    c.text(F.T0 + 99999, F.CONTACT, "方向不对", is_send=1)
    p = c.write(tmp_path / "c.json")
    with pytest.raises(S.StoreError) as e:
        svc.init_profile(p, speaker="contact", root=root)
    assert e.value.code == "IDENTITY_ANOMALY"
    assert svc.init_profile(p, speaker="contact", root=root, accept_anomalies=True)["profile_id"]


# ---------------------------------------------------------------- cleaning and dialogue
def _build(chat, cfg=None, target=F.CONTACT):
    cfg = cfg or C.CleaningConfig()
    exp = W.parse(chat.obj())
    evs, cst = C.events(exp.messages, target, cfg)
    samples, sst, shape = G.build(evs, cfg, count=len, quoted_text=svc._quoted(exp))
    return exp, evs, samples, cst, sst, shape


def test_media_system_sticker_quote_rules():
    exp, evs, samples, cst, sst, shape = _build(F.standard(3))
    targets = [b for s in samples for b in s.target_bubbles]
    assert "[图片]" not in targets and not any("撤回" in b for b in targets)
    assert any(b.startswith("[表情：开心") for b in targets)                    # caption sticker may be a target bubble
    assert sst["excluded_target_media_only"] == 3                              # the image-only contact turn every day
    assert cst["dropped_system"] == 3 and not any("撤回" in l for s in samples for l in s.context_lines)
    q = [s for s in samples if "target_quotes" in s.flags]
    assert len(q) == 3 and all(s.target_bubbles == [f"就是老街那家{d}"] for d, s in enumerate(q))       # only the new words are target
    assert all("（对方引用了" not in s.target_text and any("引用了" in l for l in [s.context_lines[-1]] + s.target_bubbles) is False for s in q)
    assert not any("不可靠的名字" in l for s in samples for l in s.context_lines)          # quotedSender is never used


def test_quote_is_rendered_in_context_when_the_quoting_turn_is_context():
    c = F.Chat()
    q = c.text(F.T0, F.CONTACT, "原话在这里")
    c.add(F.T0 + 60, F.SELF, "回复原话", type_="引用消息", local_type=1, replyToMessageId=q["platformMessageId"], quotedContent="原话在这里")
    c.text(F.T0 + 120, F.CONTACT, "嗯嗯")
    _, _, samples, *_ = _build(c)
    assert samples[-1].context_lines[-1] == "对方：（对方引用了：原话在这里）回复原话"
    assert samples[-1].target_bubbles == ["嗯嗯"]


def test_sticker_without_caption_and_placeholder_text_are_never_targets():
    c = F.Chat()
    c.text(F.T0, F.SELF, "在吗")
    c.add(F.T0 + 10, F.CONTACT, "[表情包]", type_="动画表情", local_type=47)
    c.text(F.T0 + 20, F.CONTACT, "[图片]")                                    # typed text that is only a placeholder
    _, evs, samples, cst, sst, _ = _build(c)
    assert samples == [] and cst["placeholder_text_as_media"] == 1 and sst["excluded_target_media_only"] == 1


def test_short_replies_kept_and_capped_in_train_only(chat_file, root):
    res = svc.prepare(root, ready_profile(chat_file, root)["profile_id"])
    d = res["dataset_dir"]
    hao = {k: sum(1 for r in rows(d, k) if r["conversations"][1]["value"] == "好") for k in D.SPLITS}
    assert 0 < hao["train"] <= C.CleaningConfig().max_same_target_train and hao["dev"] > 0 and hao["test"] > 0


def test_session_boundary_and_turn_merge():
    c = F.Chat()
    c.text(F.T0, F.SELF, "第一段")
    c.text(F.T0 + 60, F.CONTACT, "一")
    c.text(F.T0 + 60 + 299, F.CONTACT, "二")                                 # 4m59s: same turn
    c.text(F.T0 + 60 + 299 + 301, F.CONTACT, "三")                           # 5m01s: new turn of the same person (no other turn before it in-window)
    c.text(F.T0 + 4 * 3600, F.CONTACT, "隔了很久")                            # > 2 h: new session, starts with the target -> no context
    _, _, samples, _, sst, shape = _build(c)
    assert shape["sessions"] == 2 and samples[0].target_bubbles == ["一", "二"] and len(samples) == 2
    assert samples[1].target_bubbles == ["三"] and samples[1].context_lines == ["对方：第一段", "我：一 / 二"]   # the target person is "我"
    assert sst["excluded_no_context"] == 1


def test_no_target_session_and_context_never_crosses_sessions():
    c = F.Chat()
    c.text(F.T0, F.SELF, "只有我在说")
    c.text(F.T0 + 10, F.SELF, "还是我")
    c.text(F.T0 + 5 * 3600, F.SELF, "新的一段")
    c.text(F.T0 + 5 * 3600 + 30, F.CONTACT, "回复")
    _, _, samples, *_ = _build(c)
    assert len(samples) == 1 and samples[0].context_lines == ["对方：新的一段"]


def test_privacy_target_excluded_context_redacted_ordinary_life_kept():
    c = F.Chat()
    c.text(F.T0, F.SELF, "我的号码是13812345678")
    c.text(F.T0 + 30, F.CONTACT, "我昨天去了杭州西湖，5月3号回来")             # ordinary first-person experience, date, place: KEPT
    c.text(F.T0 + 600, F.SELF, "那你电话多少")
    c.text(F.T0 + 630, F.CONTACT, "13987654321")                              # a phone number as the target: dropped
    _, _, samples, _, sst, _ = _build(c)
    assert [s.target_text for s in samples] == ["我昨天去了杭州西湖，5月3号回来"]
    assert samples[0].context_lines == ["对方：我的号码是（已隐藏）"] and sst["excluded_target_privacy"] == 1


def test_context_limits_and_target_never_truncated():
    c = F.Chat()
    for k in range(20):
        c.text(F.T0 + k * 20, F.SELF if k % 2 == 0 else F.CONTACT, f"第{k}句" + "很长" * 30)
    cfg = C.CleaningConfig(context_turns=4, max_context_tokens=150)
    _, _, samples, *_ = _build(c, cfg)
    s = samples[-1]
    assert len(s.context_lines) <= 4 and s.context_tokens <= 150 and s.target_text.endswith("很长" * 30)


# ---------------------------------------------------------------- dataset, splits, leakage, sealing
def test_dataset_splits_isolated_chronological_and_only_target_supervised(chat_file, root):
    res = svc.prepare(root, ready_profile(chat_file, root)["profile_id"])
    d = Path(res["dataset_dir"])
    assert all(res["checks"].values()), res["checks"]
    exp = W.load(chat_file)
    sender = {m.platform_id: m.sender for m in exp.messages}
    by_time = {m.platform_id: m.create_time for m in exp.messages}
    seen_sess, seen_ids = {}, {}
    for k in D.SPLITS:
        for r, m in zip(rows(d, k), metas(d, k)):
            conv = r["conversations"]
            assert [x["from"] for x in conv] == ["human", "gpt"] and r["system"] == D.SYSTEM_PROMPT       # one supervised assistant turn
            assert all(sender[i] == F.CONTACT for i in m["target_message_ids"])                          # target = the chosen person only
            assert all(by_time[i] <= by_time[m["target_message_ids"][0]] for i in m["context_message_ids"])
            assert not set(m["target_message_ids"]) & set(m["context_message_ids"])
            for b in conv[1]["value"].split("\n"):
                assert b not in conv[0]["value"].split("\n")
            assert seen_sess.setdefault(m["session_id"], k) == k
            for i in m["source_message_ids"]:
                assert seen_ids.setdefault(i, k) == k
    t = [max(by_time[i] for m in metas(d, k) for i in m["target_message_ids"]) for k in D.SPLITS]
    f = [min(by_time[i] for m in metas(d, k) for i in m["target_message_ids"]) for k in D.SPLITS]
    assert t[0] < f[1] and t[1] < f[2]
    assert any(m["context_message_ids"] and any(sender[i] == F.SELF for i in m["context_message_ids"]) for m in metas(d, "train"))  # other person = context


def test_leakage_filter_removes_train_side_only():
    def mk(i, t, ctx, tgt):
        return G.Sample(sample_id=f"s{i}", session_id=f"x{i}", session_start=t, time=t, context_lines=[ctx], target_bubbles=[tgt],
                        context_message_ids=[f"c{i}"], target_message_ids=[f"t{i}"])
    cfg = C.CleaningConfig()
    held = [mk(9, 9, "对方：明天一起去公园散步怎么样呀", "好啊我们早上八点在门口见吧然后一起去吃早饭再散步")]
    train = [mk(1, 1, "对方：明天一起去公园散步怎么样呀", "好啊我们早上八点在门口见吧然后一起去吃早饭再散步"),
             mk(2, 2, "对方：明天一起去公园散步怎么样呀！", "好啊我们早上八点在门口见吧然后一起去吃早饭再散步哦"),    # one added particle: a near duplicate
             mk(3, 3, "对方：吃了吗", "吃了")]
    kept, info = D.leakage_filter(train, held, cfg)
    assert [s.sample_id for s in kept] == ["s3"] and info == {"exact_cross_split_removed_from_train": 1, "near_dup_cross_split_removed_from_train": 1}


def test_test_split_sealed_and_load_split_guard(chat_file, root):
    d = Path(svc.prepare(root, ready_profile(chat_file, root)["profile_id"])["dataset_dir"])
    assert not os.access(d / "test.jsonl", os.W_OK) and not os.access(d / "test_meta.jsonl", os.W_OK)
    man = json.loads((d / "SPLIT_MANIFEST.json").read_text(encoding="utf-8"))
    assert man["sealed"] and man["test_sha256"] == hashlib.sha256((d / "test.jsonl").read_bytes()).hexdigest()
    for split, purpose in (("test", "training"), ("test", "persona_prompt"), ("dev", "persona_prompt")):
        with pytest.raises(D.DatasetError):
            D.load_split(d, split, purpose)
    assert D.load_split(d, "train", "persona_prompt") and D.load_split(d, "test", "evaluation")


def test_too_small_dataset_is_an_error():
    s = [G.Sample(f"a{i}", "only", 1, i, ["对方：x"], ["y"], ["c"], [f"t{i}"]) for i in range(5)]
    with pytest.raises(D.DatasetError):
        D.assemble(s, C.CleaningConfig(), count=len, suffix_tokens=2, msg_time={"c": 0}, persona_id="p", speaker_hash="u", version_id="v")


# ---------------------------------------------------------------- local-only privacy, idempotency, storage
def test_local_prepare_needs_no_consent_form_and_stays_local_only(chat_file, root):
    pid = svc.init_profile(chat_file, speaker="contact", root=root)["profile_id"]
    res = svc.prepare(root, pid)                                     # choosing the person + prepare is enough for LOCAL processing
    assert res["dataset_version"] and all(res["gates"].values())
    pdir = S.profile_dir(root, pid)
    prof = S.read_json(pdir / "profile.json")
    assert svc.privacy_of(prof) == {"local_only": True, "cloud_upload": False, "external_api": False, "publish": False}
    man = S.read_json(Path(res["dataset_dir"]) / "DATASET_MANIFEST.json")
    assert man["privacy_policy"]["local_only"] and not man["privacy_policy"]["cloud_upload"] and not (pdir / "consent").exists()
    assert not hasattr(S, "Consent") and not hasattr(svc, "record_consent") and not hasattr(svc, "AuthorizationBlocked")


def test_profile_from_before_the_simplification_is_compatible(chat_file, root):
    """A P1 profile (an ``authorization`` block, no ``privacy`` block, maybe an old consent folder) prepares unchanged and is not rewritten."""
    pid = svc.init_profile(chat_file, speaker="contact", root=root)["profile_id"]
    pdir = S.profile_dir(root, pid)
    prof = S.read_json(pdir / "profile.json")
    prof.pop("privacy")
    S.write_json(pdir / "profile.json", prof)
    (pdir / "consent").mkdir()
    (pdir / "consent" / "1.json").write_text("{}", encoding="utf-8")
    before = (pdir / "profile.json").read_bytes()
    res = svc.prepare(root, pid)
    assert res["dataset_version"] and (pdir / "profile.json").read_bytes() == before and (pdir / "consent" / "1.json").is_file()
    assert svc.privacy_of(prof)["local_only"]


def test_idempotent_init_and_prepare_and_new_version_on_config_change(chat_file, root, tmp_path):
    a = ready_profile(chat_file, root)
    b = ready_profile(chat_file, root)
    assert a["profile_id"] == b["profile_id"] and b["reused"]
    r1 = svc.prepare(root, a["profile_id"])
    r2 = svc.prepare(root, a["profile_id"])
    assert r1["dataset_version"] == r2["dataset_version"] and r2["reused"]
    other = tmp_path / "other_root"
    r3 = svc.prepare(other, ready_profile(chat_file, other)["profile_id"])
    m1 = json.loads((Path(r1["dataset_dir"]) / "DATASET_MANIFEST.json").read_text(encoding="utf-8"))["files"]
    m3 = json.loads((Path(r3["dataset_dir"]) / "DATASET_MANIFEST.json").read_text(encoding="utf-8"))["files"]
    assert {k: v for k, v in m1.items() if k.endswith(".jsonl")} == {k: v for k, v in m3.items() if k.endswith(".jsonl")}   # deterministic
    cfgp = tmp_path / "cfg.json"
    cfgp.write_text(json.dumps({"turn_merge_minutes": 10}), encoding="utf-8")
    r4 = svc.prepare(root, a["profile_id"], config_path=cfgp)
    assert r4["dataset_version"] != r1["dataset_version"] and Path(r1["dataset_dir"]).is_dir()             # old version kept
    lineage = (S.profile_dir(root, a["profile_id"]) / "LINEAGE.jsonl").read_text(encoding="utf-8")
    assert lineage.count('"kind": "dataset"') == 2


def test_profile_conflict_and_path_safety(chat_file, root):
    pid = svc.init_profile(chat_file, speaker="contact", root=root, profile_id="mine")["profile_id"]
    with pytest.raises(S.StoreError) as e:
        svc.init_profile(chat_file, speaker="self", root=root, profile_id=pid)
    assert e.value.code == "PROFILE_EXISTS"
    for bad in ("../escape", "A", "x/y", "..", "有中文"):
        with pytest.raises(S.StoreError):
            S.profile_dir(root, bad)


def test_source_file_stays_read_only_and_unchanged(chat_file, root):
    os.chmod(chat_file, stat.S_IREAD)
    before = hashlib.sha256(chat_file.read_bytes()).hexdigest()
    svc.prepare(root, ready_profile(chat_file, root)["profile_id"])
    assert not os.access(chat_file, os.W_OK) and hashlib.sha256(chat_file.read_bytes()).hexdigest() == before
    os.chmod(chat_file, stat.S_IREAD | stat.S_IWRITE)


def test_source_change_is_detected(chat_file, root):
    pid = ready_profile(chat_file, root)["profile_id"]
    chat_file.write_text(chat_file.read_text(encoding="utf-8").replace("还行", "还可以"), encoding="utf-8")
    with pytest.raises(S.StoreError) as e:
        svc.prepare(root, pid)
    assert e.value.code == "SOURCE_CHANGED"


def test_root_inside_tracked_git_tree_is_refused(tmp_path, chat_file):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    with pytest.raises(S.StoreError) as e:
        svc.init_profile(chat_file, speaker="contact", root=repo / "personas")
    assert e.value.code == "ROOT_NOT_PRIVATE"
    (repo / ".gitignore").write_text("personas/\n", encoding="utf-8")
    assert svc.init_profile(chat_file, speaker="contact", root=repo / "personas")["profile_id"]


def test_reports_contain_counts_not_chat_text(chat_file, root):
    d = Path(svc.prepare(root, ready_profile(chat_file, root)["profile_id"])["dataset_dir"])
    rep = (d / "QUALITY_REPORT.json").read_text(encoding="utf-8")
    for m in json.loads(chat_file.read_text(encoding="utf-8"))["messages"]:
        if len(m["content"]) >= 4:
            assert m["content"] not in rep
    assert "小合成" not in rep and F.CONTACT not in rep


# ---------------------------------------------------------------- CLI
def cli(*args, stdin=None):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, "-m", "stillhere.cli", "persona", *args], capture_output=True, text=True, encoding="utf-8",
                          env=env, input=stdin, timeout=300)


def test_cli_end_to_end_non_interactive(chat_file, root):
    r = cli("init", "--input", str(chat_file), "--speaker", "contact", "--root", str(root), "--non-interactive")
    assert r.returncode == 0 and "next: stillhere persona prepare" in r.stdout and "小合成" not in r.stdout   # names are not printed without a terminal
    pid = next(l.split(": ")[1].split(" ")[0] for l in r.stdout.splitlines() if l.startswith("profile: "))
    assert cli("consent", "--profile", pid, "--root", str(root)).returncode == 2           # the consent command is gone
    r = cli("prepare", "--profile", pid, "--root", str(root), "--json")
    out = json.loads(r.stdout)
    assert r.returncode == 0 and out["gates_pass"] and set(out["samples"]) == set(D.SPLITS)
    for m in json.loads(chat_file.read_text(encoding="utf-8"))["messages"]:
        if len(m["content"]) >= 4:
            assert m["content"] not in r.stdout


def test_cli_errors_and_help(tmp_path, chat_file, root):
    assert cli("init", "--input", str(tmp_path / "nope.json"), "--speaker", "1", "--root", str(root)).returncode == 2
    r = cli("init", "--input", str(chat_file), "--root", str(root), "--non-interactive")
    assert r.returncode == 2 and "--speaker is required" in r.stderr
    assert cli("init", "--input", str(chat_file), "--root", str(root), "--speaker", "9", "--non-interactive").returncode == 4
    assert cli("prepare", "--profile", "unknown-profile", "--root", str(root)).returncode == 5
    assert cli("prepare", "--profile", "../x", "--root", str(root)).returncode == 5
    assert cli("init", "--help").returncode == 0 and cli("bogus").returncode == 2
    r = cli("inspect", "--input", str(chat_file), "--json")
    assert r.returncode == 0 and json.loads(r.stdout)["identity"]["speakers"][0]["role"] == "contact"


def test_cli_interactive_selection_does_not_wait_without_a_terminal(chat_file, root):
    r = cli("init", "--input", str(chat_file), "--root", str(root), stdin="1\n")    # stdin is a pipe, not a TTY: never prompts
    assert r.returncode == 2


def test_cli_interactive_flow_with_simulated_terminal(chat_file, root, monkeypatch, capsys):
    answers = iter(["7", "1"])                                 # an invalid choice, then person 1; no consent questions any more
    monkeypatch.setattr(PC.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    from stillhere import cli as main_cli
    assert main_cli.main(["persona", "init", "--input", str(chat_file), "--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "小合成" in out and "invalid choice" in out and "next: stillhere persona prepare" in out   # names only on the selection screen


# ---------------------------------------------------------------- existing helpers now covered
def test_style_imports_style_samples_and_to_bubbles():
    from datetime import datetime, timedelta
    from stillhere.style import imports as SI
    tz = SI.timezone(timedelta(hours=8))
    t = datetime(2025, 1, 1, 9, tzinfo=tz)
    mk = lambda i, who, s, kind=SI.TEXT, dt=0, fwd=None: SI.ImportedMessage("x", "f", "c", f"m{i}", who, who, (), "p", t + timedelta(seconds=dt), "+08", kind, s,
                                                                           forwarded_from=fwd)
    msgs = [mk(1, "a", "你好"), mk(2, "a", "第二条\n第二行", dt=60), mk(3, "b", "别人的话", dt=70), mk(4, "a", "[图片]", SI.MEDIA, dt=80),
            mk(5, "a", "转发的", dt=90, fwd="z"), mk(6, "a", "很久以后", dt=1000)]
    auth = SI.Authorization("p", ("style",), ("a",), "ref")
    assert [m.message_id for m in SI.style_samples(msgs, auth, "a")] == ["m1", "m2", "m6"]
    with pytest.raises(SI.NotAuthorized):
        SI.style_samples(msgs, auth, "b")
    assert SI.to_bubbles([msgs[0], msgs[1], msgs[5]]) == [["你好", "第二条", "第二行"], ["很久以后"]]
    assert SI.to_bubbles([msgs[0], msgs[1]], gap_seconds=30) == [["你好"], ["第二条", "第二行"]]


def test_target_quote_of_an_old_message_is_shown_in_context_not_target():
    c = F.Chat()
    old = c.text(F.T0, F.SELF, "很早以前说的那句话")
    for k in range(14):                                       # push the quoted message out of the 12-turn window
        c.text(F.T0 + 10 + k * 10, F.CONTACT if k % 2 else F.SELF, f"闲聊{k}")
    c.text(F.T0 + 600, F.SELF, "还记得吗"); c.add(F.T0 + 660, F.CONTACT, "对，就是那句", type_="引用消息", local_type=1, replyToMessageId=old["platformMessageId"], quotedContent="x")
    _, _, samples, *_ = _build(c)
    s = samples[-1]
    assert s.target_bubbles == ["对，就是那句"] and s.context_lines[-1] == "（我接下来引用了：很早以前说的那句话）"
    assert "target_quote_shown_in_context" in s.flags and old["platformMessageId"] not in s.target_message_ids
