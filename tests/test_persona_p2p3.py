"""BUILD P2 + P3 persona pipeline: persona card / prompt (train only), training vs inference prompt, label audit, Colab package and notebook,
GPU doctor, trainer gates / resume / checkpoints, adapter export / import, A-E evaluation, checkpoint selection, blind review, paired
statistics, test sealing, persona runtime, CLI. Synthetic exports only (tests/persona_fixture.py); FakeGenerator results are DRY_RUN_ONLY."""
import ast
import hashlib
import json
import os
import re
import stat
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in ("src", "training/persona", "training/d5", "training/d4"):
    sys.path.insert(0, str(ROOT / p))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import persona_fixture as F  # noqa: E402
from stillhere.persona import adapters as AD  # noqa: E402
from stillhere.persona import card as K  # noqa: E402
from stillhere.persona import dataset as D  # noqa: E402
from stillhere.persona import doctor as DR  # noqa: E402
from stillhere.persona import evaluation as EV  # noqa: E402
from stillhere.persona import memory as M  # noqa: E402
from stillhere.persona import notebook as NB  # noqa: E402
from stillhere.persona import package as P  # noqa: E402
from stillhere.persona import service as svc  # noqa: E402
from stillhere.persona import store as S  # noqa: E402
from stillhere.persona.runtime import PersonaRuntime  # noqa: E402

LF_PY = Path("<data-dir>/envs/cpcd_t1/Scripts/python.exe")
QWEN = Path(os.environ.get("STILLHERE_QWEN3_DIR", "<models-dir>/Qwen3-8B"))


def _patch_memory_synthetic(monkeypatch):
    orig = M.build_api
    monkeypatch.setattr(M, "build_api", lambda *a, **k: orig(*a, **{**k, "synthetic": True}))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """One synthetic profile, prepared (dataset + auto card) and packaged for cpu-only; shared by the read-only tests."""
    mp = pytest.MonkeyPatch()
    _patch_memory_synthetic(mp)
    tmp = tmp_path_factory.mktemp("p2p3")
    chat = F.standard(days=80).write(tmp / "chat.json")
    root = tmp / "personas"
    pid = svc.init_profile(chat, speaker="contact", root=root)["profile_id"]
    prep = svc.prepare_all(root, pid)
    pkg = P.build(root, pid, target="cpu-only")
    yield {"chat": chat, "root": root, "pid": pid, "pdir": S.profile_dir(root, pid), "ddir": Path(prep["dataset_dir"]), "prep": prep, "pkg": pkg, "tmp": tmp}
    mp.undo()


def unzip(p):
    with zipfile.ZipFile(p) as z:
        return {n: z.read(n) for n in z.namelist()}


# ======================================================================= P2: persona card / prompt
def test_card_reads_train_only_and_records_it(world, monkeypatch):
    calls = []
    real_split, real_meta = D.load_split, D.load_meta
    monkeypatch.setattr(D, "load_split", lambda d, s, p: calls.append(("split", s, p)) or real_split(d, s, p))
    monkeypatch.setattr(D, "load_meta", lambda d, s, p: calls.append(("meta", s, p)) or real_meta(d, s, p))
    prof = S.read_json(world["pdir"] / "profile.json")
    card = K.generate(world["ddir"], persona_id=world["pid"], display_name="小合成", speaker_hash=prof["target"]["speaker_id"])
    assert calls and all(s == "train" and p == "persona_prompt" for _, s, p in calls)
    train_ids = {r["id"] for r in real_split(world["ddir"], "train", "persona_prompt")}
    assert card["evidence"]["split"] == "train" and card["evidence"]["dev_test_read"] is False
    assert set(card["evidence"]["example_ids"]) <= train_ids and card["examples"]
    with pytest.raises(D.DatasetError):
        D.load_split(world["ddir"], "dev", "persona_prompt")
    with pytest.raises(D.DatasetError):
        D.load_meta(world["ddir"], "test", "persona_prompt")


def test_other_persons_habits_are_not_written_into_the_target_card(tmp_path):
    c = F.Chat()
    t = F.T0
    things = ["电影", "火锅", "跑步", "开会", "下雨", "快递", "猫咪", "作业", "地铁", "咖啡"]
    for day in range(40):
        base = t + day * 86400
        x, y = things[day % 10], things[(day * 3 + 1) % 10]
        c.text(base, F.SELF, f"哈哈哈哈今天{x}好呀～{day}")
        c.text(base + 30, F.SELF, f"你那边{y}怎么样呀")
        c.text(base + 400, F.CONTACT, f"得嘞 {y}第{day}次")
        c.text(base + 900, F.SELF, f"哈哈哈那就好呀 {x}")
        c.text(base + 1300, F.CONTACT, f"得嘞 {x}明天说{day % 7}")
    path = c.write(tmp_path / "c.json")
    root = tmp_path / "r"
    pid = svc.init_profile(path, speaker="contact", root=root)["profile_id"]
    res = svc.prepare(root, pid)
    card = K.generate(res["dataset_dir"], persona_id=pid, display_name="小合成", speaker_hash="u_x")
    expr = [x["text"] for x in card["common_expressions"]]
    assert "得嘞" in expr and not any(x in ("哈哈", "好呀", "呀") for x in expr)
    assert card["emoji"]["laugh_share"] == 0 and card["punctuation"]["tilde_share"] == 0
    laugh = [t for t in card["style_traits"] if t["trait"] == "laugh"]
    assert not laugh or laugh[0]["target_rate"] == 0
    prompt = K.render_prompt(card)
    assert "哈哈" not in prompt.split("【示例对话】")[0] and "得嘞" in prompt


def test_card_and_prompt_are_deterministic_and_reused(world):
    prof = S.read_json(world["pdir"] / "profile.json")
    a = K.generate(world["ddir"], persona_id=world["pid"], display_name=prof["target"]["display_name"], speaker_hash=prof["target"]["speaker_id"])
    b = K.generate(world["ddir"], persona_id=world["pid"], display_name=prof["target"]["display_name"], speaker_hash=prof["target"]["speaker_id"])
    assert K.canonical(a) == K.canonical(b) and K.render_prompt(a) == K.render_prompt(b)
    r1 = K.build_auto(world["pdir"], world["ddir"], prof)
    r2 = K.build_auto(world["pdir"], world["ddir"], prof)
    assert r1["card_id"] == r2["card_id"] and r2["reused"] and r1["prompt_sha256"] == hashlib.sha256(K.render_prompt(a).encode()).hexdigest()
    cur = K.current(world["pdir"])
    assert cur["manifest"]["review_status"] in (K.AUTO, K.REVIEWED) and cur["manifest"]["training_split_sha256"] == a["evidence"]["train_sha256"]


def test_prompt_never_invents_facts_and_user_facts_only_after_review(tmp_path):
    chat = F.standard(days=60).write(tmp_path / "chat.json")
    root = tmp_path / "r"
    pid = svc.init_profile(chat, speaker="contact", root=root)["profile_id"]
    svc.prepare_all(root, pid)
    pdir = S.profile_dir(root, pid)
    cur = K.current(pdir)
    head = cur["prompt"].split("【示例对话】")[0]
    for w in ("岁", "职业", "工作于", "住在", "老家", "男朋友", "女朋友", "结婚", "爱好是"):
        assert w not in head
    assert "\n【用户确认的人物设定】\n" not in cur["prompt"] and "不要编造" in cur["prompt"] and "AI" in cur["prompt"]   # no such section until reviewed
    edited = dict(cur["card"], user_confirmed_facts=["喜欢别人叫她小合"], addressing_confirmed=["你"])
    new = K.review(pdir, edited=edited)
    assert new["review_status"] == K.REVIEWED and new["parent_card"] == cur["card_id"] and new["card_id"] != cur["card_id"]
    assert (pdir / "cards" / cur["card_id"] / "persona_prompt.txt").is_file()                       # the old version is kept
    now = K.current(pdir)
    assert now["card_id"] == new["card_id"] and "【用户确认的人物设定】\n- 喜欢别人叫她小合" in now["prompt"]
    with pytest.raises(K.CardError) as e:
        K.review(pdir, edited=dict(now["card"], reply_length={"chars_p50": 99}))
    assert e.value.code == "FIELD_NOT_EDITABLE"
    ap = K.review(pdir, approve=True)
    assert ap["reused"] and ap["card_id"] == new["card_id"]


def test_training_prompt_and_inference_prompt_are_separate_and_layouts_compatible(world):
    from d5pilot import common as DK
    from stillhere.session.tokens import PROMPT_FMT
    rows = D.load_split(world["ddir"], "train", "persona_prompt")
    assert {r["system"] for r in rows} == {D.SYSTEM_PROMPT}
    cur = K.current(world["pdir"])
    assert cur["prompt"] != D.SYSTEM_PROMPT and D.SYSTEM_PROMPT not in cur["prompt"]
    items = EV.read_eval_set(world["pdir"] / "evals" / "sets" / f"{world['ddir'].name}_{cur['card_id']}_dev.jsonl")
    dev = {r["id"]: r for r in D.load_split(world["ddir"], "dev", "evaluation")}
    for it in items:
        m = it["messages"]
        assert m["B"] == m["C"] and m["D"] == m["E"] and m["B"][0]["content"] == cur["prompt"]
        assert m["A"][0]["content"] == EV.NEUTRAL_PROMPT and m["D"][0]["content"].startswith(cur["prompt"].rstrip("\n")) and "【记忆证据】" in m["D"][0]["content"]
        assert all(m[g][1]["content"] == dev[it["sample_id"]]["conversations"][0]["value"] for g in m)     # the training user turn, byte for byte
    s, u = cur["prompt"], items[0]["user"]
    assert DK.PRE.format(s=s, u=u) == PROMPT_FMT.format(system=s, user=u)                                # one layout: training == inference


@pytest.mark.skipif(not (LF_PY.is_file() and (QWEN / "tokenizer.json").is_file()), reason="needs the LLaMA-Factory env and the pinned Qwen3 tokenizer")
def test_real_llamafactory_label_audit_all_rows_synthetic(world, tmp_path):
    data = tmp_path / "data"
    with zipfile.ZipFile(Path(world["pkg"]["package_dir"]) / "persona_data.zip") as z:
        z.extractall(data)
    res = P.run_label_audit(data, tmp_path / "audit", str(QWEN), python=str(LF_PY))
    assert res["LLAMAFACTORY_LABELS"] == "PASS", res
    for split in ("train", "dev"):
        s = res["splits"][split]
        assert s["status"] == "PASS" and s["failures_total"] == 0 and s["negative_control_caught"] == s["rows"] > 0
        assert s["inference_template_equals_training_layout"] == s["rows"] == s["persona_prompt_template_equals_training_layout"]
        assert s["max_input_ids"] <= 2048 and s["passed"]["no_truncation"] == s["rows"] and s["passed"]["prompt_fully_masked"] == s["rows"]
    assert res["splits"]["train"]["rows_with_earlier_target_replies_in_prompt"] > 0       # earlier own replies exist and are masked


# ======================================================================= P2: package, notebook, doctor
def test_package_has_no_test_no_raw_export_and_is_deterministic(world):
    pkg = Path(world["pkg"]["package_dir"])
    data = unzip(pkg / "persona_data.zip")
    names = set(data)
    assert not any(n.split("/")[-1].startswith("test") for n in names) and "train/train.jsonl" in names and "eval/dev_eval.jsonl" in names
    raw = world["chat"].read_bytes()
    assert all(b != raw for b in data.values()) and hashlib.sha256((world["ddir"] / "test.jsonl").read_bytes()).hexdigest() not in \
        {hashlib.sha256(b).hexdigest() for b in data.values()}
    dm = json.loads(data["DATA_MANIFEST.json"])
    assert dm["contains_test"] is False and dm["contains_raw_export"] is False and all(hashlib.sha256(data[n]).hexdigest() == h for n, h in dm["files"].items())
    test_ids = {r["id"] for r in D.load_split(world["ddir"], "test", "evaluation")}
    assert not any(i in data["train/train.jsonl"].decode() + data["eval/dev_eval.jsonl"].decode() for i in test_ids)
    code = unzip(pkg / "persona_code.zip")
    assert "training/persona/personatrain/train_persona.py" in code and "src/stillhere/persona/card.py" in code
    assert not any(x in n for n in code for x in ("/private/", ".jsonl", ".env", "__pycache__", ".safetensors"))
    sums = dict((l.split()[1], l.split()[0]) for l in (pkg / "SHA256SUMS.txt").read_text().splitlines())
    assert all(hashlib.sha256((pkg / n).read_bytes()).hexdigest() == h for n, h in sums.items())
    again = P.build(world["root"], world["pid"], target="cpu-only")
    assert again["package_id"] == world["pkg"]["package_id"] and again["reused"]


def test_cloud_package_needs_explicit_confirmation(world):
    with pytest.raises(P.PackageError) as e:
        P.build(world["root"], world["pid"], target="colab-l4")
    assert e.value.code == "CLOUD_NOT_CONFIRMED" and "PRIVATE CHAT" in str(e.value)
    res = P.build(world["root"], world["pid"], target="colab-l4", confirm_cloud=True)
    rm = S.read_json(Path(res["package_dir"]) / "RELEASE_MANIFEST.json")
    assert rm["cloud_transfer_confirmed_by_operator"] is True and rm["contains"] == {"test": False, "raw_export": False, "other_conversations": False}
    lineage = (world["pdir"] / "LINEAGE.jsonl").read_text(encoding="utf-8")
    assert "cloud_transfer_confirmed_by_operator" in lineage and res["notebook"].endswith("StillHere_Persona_L4.ipynb")


def test_training_plan_and_resolved_config():
    p = P.plan(2101, {"gradient_accumulation_steps": 8, "per_device_train_batch_size": 1, "num_train_epochs": 1.0})
    assert p["steps"] == 263 and p["save_steps"] == 33 and p["candidate_steps"] == [132, 263] and p["checkpoints_expected"][-1] == 263
    for rows in (5, 57, 200, 333, 1999, 4001):
        q = P.plan(rows, {"gradient_accumulation_steps": 8, "per_device_train_batch_size": 1, "num_train_epochs": 1.0})
        assert all(c <= q["steps"] and c in q["checkpoints_expected"] for c in q["candidate_steps"]) and q["save_steps"] >= 1
    import yaml
    b, pl = P.resolved_config(2101)
    c = yaml.safe_load(b)
    for k, v in dict(model_revision=DR.BASE_REVISION, quantization_bit=4, quantization_type="nf4", double_quantization=True, lora_rank=16, lora_alpha=32,
                     lora_dropout=0.05, learning_rate=5e-5, num_train_epochs=1.0, cutoff_len=2048, per_device_train_batch_size=1, gradient_accumulation_steps=8,
                     gradient_checkpointing=True, flash_attn="sdpa", train_on_prompt=False, template="qwen3", enable_thinking=False, bf16=True).items():
        assert c[k] == v, k
    assert c["persona_expected_steps"] == 263 and c["persona_candidate_steps"] == [132, 263] and c["save_steps"] == 33


def test_notebook_is_valid_staged_and_generic():
    nb = NB.build("pk-test", "run-1", "colab-l4")
    json.loads(NB.dumps(nb))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for src in code:
        ast.parse(src)
    titles = [re.match(r"# STAGE (\d+) · ([^（(\n]+)", s) for s in code]
    nums = [int(m.group(1)) for m in titles if m]
    assert nums == list(range(15))
    want = ["Environment Check", "Dataset Import", "SHA256 Verification", "Install Compatible Dependencies", "Qwen3-8B Base Verification", "GPU Preflight",
            "Loss Mask Verification", "Short GPU Smoke", "QLoRA Training", "Checkpoint Save / Resume", "Dev Evaluation", "Adapter Export", "Release Manifest", "Return Artifacts"]
    joined = "\n".join(code)
    assert all(w in joined for w in want)
    assert 'CONFIRM_PRIVATE_DATA_ON_CLOUD = False' in joined and 'PACKAGE_ID = "pk-test"' in joined and "assert CONFIRM_PRIVATE_DATA_ON_CLOUD" in joined
    assert not re.search(r"hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{16,}|F:\\\\|private/|@gmail", joined)
    assert "--lf-result" in joined and "--stop-at-step 1" in joined and "persona_smoke" in joined and "test.jsonl" not in joined
    if NB.REPO_NOTEBOOK.is_file():
        assert NB.REPO_NOTEBOOK.read_bytes() == NB.dumps(NB.build())                       # the repository copy is the generator output


def _gpu(name, vram, bf16=True, nf4=True):
    return {"torch": "2.8", "cuda_available": True, "gpu": name, "vram_total_gib": vram, "bf16": bf16, "nf4_roundtrip_ok": nf4}


def test_doctor_targets(monkeypatch):
    monkeypatch.setattr(DR, "_ver", lambda m: DR.EXPECT.get(m, "1.0"))
    monkeypatch.setattr(DR, "gpu_check", lambda nf4=True: _gpu("NVIDIA L4", 22.0))
    r = DR.doctor("colab-l4")
    assert r["status"] == "PASS" and r["capabilities"]["qwen3_8b_qlora_training"]
    assert DR.doctor("colab-a100")["checks"]["gpu_model_ok"] is False
    assert DR.doctor("colab-a100", allow_other_gpu=True)["checks"]["gpu_model_ok"] is True
    monkeypatch.setattr(DR, "gpu_check", lambda nf4=True: _gpu("NVIDIA GeForce RTX 4060 Laptop GPU", 8.0))
    r = DR.doctor("local-cuda")
    assert r["status"] == "BLOCKED" and not r["capabilities"]["qwen3_8b_qlora_training"] and r["capabilities"]["qwen3_8b_nf4_inference"]
    assert any("QLoRA training refused" in b for b in r["blocked"])
    monkeypatch.setattr(DR, "gpu_check", lambda nf4=True: _gpu("NVIDIA L4", 22.0, nf4=False))
    assert "nf4" in DR.doctor("colab-l4")["blocked"]
    monkeypatch.setattr(DR, "gpu_check", lambda nf4=True: {"torch": None, "cuda_available": False})
    assert DR.doctor("cpu-only")["status"] == "PASS" and DR.doctor("colab-l4")["status"] == "BLOCKED"


# ======================================================================= P3: trainer gates, resume, checkpoints
def _pkg_data(world, tmp):
    data = tmp / "data"
    with zipfile.ZipFile(Path(world["pkg"]["package_dir"]) / "persona_data.zip") as z:
        z.extractall(data)
    return data


def _trainer(*args):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(str(ROOT / x) for x in ("training/persona", "training/d5", "training/d4", "src")), "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, "-m", "personatrain.train_persona", *args], capture_output=True, text=True, encoding="utf-8", env=env, timeout=300)


def test_trainer_gates_fingerprint_resume_and_checkpoints(world, tmp_path):
    from personatrain import train_persona as TP
    data = _pkg_data(world, tmp_path)
    cfg = data / "config" / "training_config.yaml"
    out = tmp_path / "run" / "train"
    r = _trainer("--data", str(data), "--config", str(cfg), "--output-dir", str(out), "--dry-run")
    assert r.returncode == 3 and "llamafactory_labels_pass" in r.stdout                       # no label audit -> no formal training
    tsha = hashlib.sha256((data / "train" / "train.jsonl").read_bytes()).hexdigest()
    lf = tmp_path / "LF.json"
    lf.write_text(json.dumps({"LLAMAFACTORY_LABELS": "PASS", "splits": {"train": {"file_sha256": "0" * 64}}}), encoding="utf-8")
    r = _trainer("--data", str(data), "--config", str(cfg), "--output-dir", str(out), "--dry-run", "--lf-result", str(lf))
    assert r.returncode == 3 and "labels_checked_this_train_file" in r.stdout                 # an audit of ANOTHER train file does not count
    lf.write_text(json.dumps({"LLAMAFACTORY_LABELS": "PASS", "splits": {"train": {"file_sha256": tsha}}}), encoding="utf-8")
    r = _trainer("--data", str(data), "--config", str(cfg), "--output-dir", str(out), "--dry-run", "--lf-result", str(lf))
    plan = json.loads(r.stdout)
    assert r.returncode == 0 and plan["gates"]["status"] == "PASS" and plan["newest_complete_checkpoint"] is None
    assert "persona_expected_steps" not in plan["resolved_config_keys"] and "dataset_dir" in plan["resolved_config_keys"]
    out.mkdir(parents=True, exist_ok=True)                                                     # checkpoints: newest COMPLETE one is chosen
    for n, files in (("checkpoint-33", TP_FILES), ("checkpoint-66", [f for f in TP_FILES if f != "optimizer.pt"])):
        for f in files:
            (out / n).mkdir(exist_ok=True)
            (out / n / f).write_bytes(b"x")
    r = _trainer("--data", str(data), "--config", str(cfg), "--output-dir", str(out), "--dry-run", "--lf-result", str(lf))
    plan = json.loads(r.stdout)
    assert plan["newest_complete_checkpoint"] == "checkpoint-33" and plan["skipped_incomplete_checkpoints"][0]["checkpoint"] == "checkpoint-66"
    (out / "run_fingerprint.json").write_text(json.dumps({"sha256": "different", "fields": {"learning_rate": 1.0}}), encoding="utf-8")
    r = _trainer("--data", str(data), "--config", str(cfg), "--output-dir", str(out), "--dry-run", "--lf-result", str(lf))
    assert r.returncode == 1 and "RESUME REFUSED" in r.stdout                                  # another configuration never resumes this run
    c = TP.resolve({"persona_expected_steps": 1, "dataset": "persona_train", "eval_dataset": "persona_dev"}, data, out, max_steps=2, save_steps=1,
                   dataset="persona_smoke")
    assert c["dataset"] == "persona_smoke" and "eval_dataset" not in c and c["max_steps"] == 2 and "persona_expected_steps" not in c
    (data / "DATA_MANIFEST.json").write_text((data / "DATA_MANIFEST.json").read_text().replace('"contains_test": false', '"contains_test": true'))
    r = _trainer("--data", str(data), "--config", str(cfg), "--output-dir", str(tmp_path / "x"), "--dry-run", "--lf-result", str(lf))
    assert r.returncode == 1


TP_FILES = ["adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth"]


# ======================================================================= P3: adapter export / import
def fake_safetensors(names=("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight")):
    head = {n: {"dtype": "F32", "shape": [2, 2], "data_offsets": [i * 16, i * 16 + 16]} for i, n in enumerate(names)}
    h = json.dumps(head).encode()
    return struct.pack("<Q", len(h)) + h + b"\0" * 16 * len(names)


def make_run(world, tmp, *, selected="ckpt-6", tensors=None):
    data = _pkg_data(world, tmp)
    run = tmp / "run"
    ck = run / "train" / "checkpoint-6"
    ck.mkdir(parents=True)
    (ck / "adapter_model.safetensors").write_bytes(fake_safetensors(tensors) if tensors else fake_safetensors())
    (ck / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
                                                        "target_modules": ["q_proj"], "base_model_name_or_path": "Qwen/Qwen3-8B"}), encoding="utf-8")
    (run / "train" / "run_fingerprint.json").write_text(json.dumps({"sha256": "f" * 64, "fields": {"data_manifest_sha256": "d" * 64}}), encoding="utf-8")
    (run / "eval" / "dev").mkdir(parents=True)
    (run / "eval" / "dev" / "CHECKPOINT_SELECTION.json").write_text(json.dumps({"status": "SELECTED", "selected": selected}), encoding="utf-8")
    return run, data


def test_adapter_export_import_and_verification(world, tmp_path):
    from personatrain import export as EX
    run, data = make_run(world, tmp_path)
    z = EX.adapter(run, run / "train", "checkpoint-6", "run-1", data)
    names = set(unzip(z))
    assert names == AD.MEMBERS and not any("model-0000" in n for n in names)                    # no base weights
    man = json.loads(unzip(z)["persona_adapter/persona_manifest.json"])
    assert man["name"] == "PERSONA_ADAPTER_CANDIDATE" and man["base"]["revision"] == DR.BASE_REVISION and man["profile_id"] == world["pid"]
    with pytest.raises(SystemExit):
        EX.adapter(run, run / "train", "checkpoint-6", "run-1", data)                            # never overwritten
    res = AD.import_adapter(world["root"], world["pid"], z)
    assert not res["reused"] and res["tensors"] == 2 and (Path(res["adapter_dir"]) / "adapter" / "adapter_model.safetensors").is_file()
    assert AD.import_adapter(world["root"], world["pid"], z)["reused"]
    assert AD.resolve(world["pdir"], res["adapter_id"])["manifest"]["checkpoint"] == "checkpoint-6"

    def tampered(fn, name):
        src = unzip(z)
        fn(src)
        p = tmp_path / name
        with zipfile.ZipFile(p, "w") as w:
            for n, b in src.items():
                w.writestr(n, b)
        return p
    cases = {"SHA256_MISMATCH": lambda s: s.__setitem__("persona_adapter/persona_prompt.txt", b"changed"),
             "ZIP_LAYOUT": lambda s: s.__setitem__("persona_adapter/extra.bin", b"x"),
             "ZIP_UNSAFE": lambda s: s.__setitem__("../evil.txt", b"x")}
    for code, fn in cases.items():
        with pytest.raises(AD.AdapterError) as e:
            AD.verify(tampered(fn, code + ".zip"), world["pdir"])
        assert e.value.code == code
    other = S.profile_dir(world["root"], "p-otherperson")
    other.mkdir(parents=True, exist_ok=True)
    S.write_json(other / "profile.json", {"profile_id": "p-otherperson"})
    with pytest.raises(AD.AdapterError) as e:
        AD.verify(z, other)
    assert e.value.code == "PERSONA_MISMATCH"
    with pytest.raises(AD.AdapterError):
        AD.verify(tmp_path / "missing.zip", world["pdir"])


def test_adapter_with_non_lora_weights_or_unselected_checkpoint_is_refused(world, tmp_path):
    from personatrain import export as EX
    run, data = make_run(world, tmp_path, selected="ckpt-99", tensors=("model.embed_tokens.weight",))
    with pytest.raises(SystemExit):
        EX.adapter(run, run / "train", "checkpoint-6", "run-2", data)                            # not the pre-registered selection
    z = EX.adapter(run, run / "train", "checkpoint-6", "run-2", data, unqualified=True)
    assert "UNQUALIFIED" in z.name
    with pytest.raises(AD.AdapterError) as e:
        AD.verify(z, world["pdir"])
    assert e.value.code == "ADAPTER_WEIGHTS"


# ======================================================================= P3: evaluation
@pytest.fixture()
def items(world):
    cur = K.current(world["pdir"])
    return EV.read_eval_set(world["pdir"] / "evals" / "sets" / f"{world['ddir'].name}_{cur['card_id']}_dev.jsonl")


def test_memory_is_train_period_only_and_identical_for_d_and_e(world, items):
    prof = S.read_json(world["pdir"] / "profile.json")
    cut = M.cutoff_from_train(world["ddir"])
    api, meta = M.build_api(prof["source"]["path"], persona_id=world["pid"], target_username=prof["target"]["username"], cutoff=cut, synthetic=True)
    rows = api.store._conn.execute("SELECT MAX(timestamp) FROM messages").fetchone() if hasattr(api.store, "_conn") else None
    api.close()
    assert meta["messages"] > 0 and meta["cutoff"] == cut.isoformat()
    dev_first = min(it["time"] for it in items)
    assert cut.isoformat() < dev_first
    for it in items:
        assert it["memory"] and it["messages"]["D"] == it["messages"]["E"] and it["memory"] in it["messages"]["D"][0]["content"]
        for line in it["memory"].splitlines():
            m = re.match(r"\[\w+\] (\d{4}-\d{2}-\d{2})", line)
            if m:
                assert m.group(1) <= cut.date().isoformat()                                   # no dev / test period message in memory


def test_fake_generation_resume_scoring_and_dry_run_label(world, items, tmp_path):
    gen = EV.FakeGenerator(["好", "好的呀", "明天一起吃饭吧"])
    fp = {"x": 1}
    s1 = EV.generate(gen, items, ["A", "B", "C"], tmp_path / "r", fp, adapters={"C": "C"}, limit=3, log=lambda s: None)
    assert s1["A"]["ok"] == 3
    s2 = EV.generate(gen, items, ["A", "B", "C"], tmp_path / "r", fp, adapters={"C": "C"}, log=lambda s: None)
    lines = (tmp_path / "r" / "A" / "answers.jsonl").read_text(encoding="utf-8").splitlines()
    assert s2["A"]["ok"] == len(items) and len(lines) == len(items)                           # resumed: the first 3 were not generated again
    with pytest.raises(EV.EvalError) as e:
        EV.generate(gen, items, ["A"], tmp_path / "r", {"x": 2}, adapters={}, log=lambda s: None)
    assert e.value.code == "FINGERPRINT_MISMATCH"
    sc = EV.score_run(tmp_path / "r", items, ["A", "B", "C"], EV.Reference(world["ddir"], K.current(world["pdir"])["card"]))
    assert sc["status"] == EV.DRY and sc["verdicts"]["B_vs_C"]["verdict"] == EV.DRY and set(sc["paired"]) == {"A_vs_B", "B_vs_C"}
    assert sc["groups"]["A"]["n"] == len(items) and 0 <= sc["groups"]["C"]["top1_reply_share"] <= 1


def test_score_row_detects_leaks_copies_collapse_and_unsupported_claims(world):
    ref = EV.Reference(world["ddir"], K.current(world["pdir"])["card"])
    long = next(t for t in ref.long_targets if len(t) >= 10)
    it = {"user": "对方：今天怎么样", "reference": "还行吧", "query": "今天怎么样", "memory": "【记忆证据】\n[e1] 2025-01-02｜对方：我们约了3号见面"}
    assert EV.score_row("我的电话是13812345678", it, ref)["privacy_leak"]
    assert EV.score_row(long, it, ref)["train_copy"]
    assert EV.score_row("嗯", it, ref)["generic"] and not EV.score_row("还行吧，你呢", it, ref)["generic"]
    r = EV.score_row("我下周五要去5号楼开会", it, ref)
    assert r["unsupported_specific"] >= 1
    assert EV.score_row("那就3号见", it, ref)["memory_specific_used"] == 1
    assert EV.score_row("<think>x</think>好", it, ref)["special_token_leak"] and EV.score_row("我：好", it, ref)["prefix_leak"]


def _answers(run, group, texts):
    (run / group).mkdir(parents=True, exist_ok=True)
    with open(run / group / "answers.jsonl", "w", encoding="utf-8") as f:
        for sid, t in texts.items():
            f.write(json.dumps({"sample_id": sid, "group": group, "status": "OK", "text": t, "generator": "QWEN3_8B_NF4_D5_REAL_GPU"}, ensure_ascii=False) + "\n")


def test_checkpoint_selection_rule_and_paired_stats(world, items, tmp_path):
    ref = EV.Reference(world["ddir"], K.current(world["pdir"])["card"])
    run = tmp_path / "sel"
    refs = {it["sample_id"]: it["reference"] for it in items}
    varied = {s: f"收到第{i}个" for i, s in enumerate(sorted(refs))}                   # short, varied, never a train sentence
    _answers(run, "B", {s: "这个问题可以先看看再说，希望对你有帮助" for s in refs})
    _answers(run, "ckpt-3", varied)
    _answers(run, "ckpt-6", {s: "嗯" for s in refs})                                    # collapsed to one reply
    sel = EV.select_checkpoint(run, items, ["ckpt-3", "ckpt-6"], ref)
    assert sel["status"] == "SELECTED" and sel["selected"] == "ckpt-3" and "collapse" in sel["candidates"]["ckpt-6"]["reasons"], sel
    _answers(run, "ckpt-6", varied)                                                     # identical quality -> tie -> the later checkpoint
    assert EV.select_checkpoint(run, items, ["ckpt-3", "ckpt-6"], ref)["selected"] == "ckpt-6"
    _answers(run, "C", {s: r for s, r in refs.items()})
    sc = EV.score_run(run, items, ["B", "C"], ref)
    p = sc["paired"]["B_vs_C"]
    assert sc["status"] == "REAL_MODEL" and p["n"] == len(items) and p["style_margin"]["mean_diff"] > 0 and p["style_margin"]["win"] > p["style_margin"]["loss"]
    assert sc["verdicts"]["B_vs_C"]["verdict"] in ("AUTOMATIC_METRICS_ONLY_PASS", "NOT_BETTER")       # never BETTER without a blind review


def test_blind_review_is_randomised_hidden_and_scored(world, items, tmp_path):
    run = tmp_path / "b"
    _answers(run, "B", {it["sample_id"]: "左边的回复" for it in items})
    _answers(run, "C", {it["sample_id"]: "右边的回复" for it in items})
    res = EV.make_blind(run, items, ["B", "C"], n=20)
    csv_text = Path(res["csv"]).read_text(encoding="utf-8-sig")
    key = S.read_json(Path(res["key"]))["key"]
    assert res["rows"] == min(20, len(items)) and "B" not in csv_text.split("\n")[0] and {v["left"] for v in key.values()} == {"B", "C"}
    import csv
    import io
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    for row in rows:                                                                     # the rater always prefers C's reply
        row["choice(L/R/=)"] = "L" if key[row["id"]]["left"] == "C" else "R"
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    Path(res["csv"]).write_text(buf.getvalue(), encoding="utf-8-sig")
    sc = EV.score_blind(Path(res["csv"]), Path(res["key"]))
    assert sc["C_wins"] == res["rows"] and sc["B_wins"] == 0 and sc["win_rate"] == 1.0 and sc["sign_p"] < 0.05


def test_test_split_is_sealed_locked_once_and_logged(world, tmp_path):
    with pytest.raises(EV.EvalError) as e:
        svc.evaluate(world["root"], world["pid"], split="test", backend="fake")
    assert e.value.code == "TEST_SEALED"
    root = tmp_path / "evals"
    root.mkdir()
    l1 = EV.lock_for_test(root, dataset_version="v", prompt_sha256="p", adapter_sha256="a", checkpoint="c", groups=["B", "C"], backend="hf")
    assert EV.lock_for_test(root, dataset_version="v", prompt_sha256="p", adapter_sha256="a", checkpoint="c", groups=["B", "C"], backend="hf")["sha256"] == l1["sha256"]
    with pytest.raises(EV.EvalError) as e:
        EV.lock_for_test(root, dataset_version="v", prompt_sha256="p", adapter_sha256="OTHER", checkpoint="c", groups=["B", "C"], backend="hf")
    assert e.value.code == "TEST_ALREADY_LOCKED"
    assert EV.log_test_open(root, l1["sha256"], "evaluation") == 1 and EV.log_test_open(root, l1["sha256"], "evaluation") == 2
    t = world["ddir"] / "test.jsonl"
    assert not os.access(t, os.W_OK)
    import shutil
    copy = tmp_path / "ds"
    shutil.copytree(world["ddir"], copy)
    os.chmod(copy / "test.jsonl", stat.S_IWRITE | stat.S_IREAD)
    (copy / "test.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(D.DatasetError):
        EV.eval_items(copy, "test")


def test_persona_runtime_switches_adapter_and_memory_without_reloading(world, tmp_path, monkeypatch):
    _patch_memory_synthetic(monkeypatch)
    from personatrain import export as EX
    run, data = make_run(world, tmp_path)
    aid = AD.import_adapter(world["root"], world["pid"], EX.adapter(run, run / "train", "checkpoint-6", "run-rt", data))["adapter_id"]
    rt = PersonaRuntime(world["root"], world["pid"], backend="fake", adapters=[aid], memory=True)
    hist = [("other", "今天吃什么"), ("me", "还没想好"), ("other", "那一起去吃面吧")]
    m = rt.messages(hist)
    assert m[0]["content"].startswith(rt.prompt.rstrip("\n")) and "【记忆证据】" in m[0]["content"] and m[1]["content"].endswith("对方：那一起去吃面吧")
    assert rt.reply(hist)["mode"] == "D"
    rt.use_adapter(aid)
    gen = rt.gen
    assert rt.reply(hist)["mode"] == "E" and rt.gen is gen
    rt.use_adapter(None)
    assert rt.reply(hist)["mode"] == "D"
    with pytest.raises(AD.AdapterError):
        rt.use_adapter("a-not-loaded")
    plain = PersonaRuntime(world["root"], world["pid"], backend="fake")
    assert plain.messages(hist)[0]["content"] == rt.prompt and plain.reply(hist)["mode"] == "B"


def test_gpu_runner_cli_with_fake_backend_on_the_package(world, tmp_path):
    data = _pkg_data(world, tmp_path)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(str(ROOT / x) for x in ("training/persona", "training/d5", "training/d4", "src")), "PYTHONIOENCODING": "utf-8"}
    es = data / "eval" / "dev_eval.jsonl"
    run = tmp_path / "eval" / "dev"
    base = [sys.executable, "-m", "personatrain.evaluate"]
    r = subprocess.run(base + ["generate", "--eval-set", str(es), "--groups", "B,ckpt-3,ckpt-6", "--candidates", "ckpt-3,ckpt-6", "--run-dir", str(run),
                               "--backend", "fake"], capture_output=True, text=True, encoding="utf-8", env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-800:]
    r = subprocess.run(base + ["select", "--eval-set", str(es), "--run-dir", str(run), "--train-dir", str(data / "train"), "--card",
                               str(data / "persona" / "persona_card.yaml"), "--candidates", "ckpt-3,ckpt-6"], capture_output=True, text=True, encoding="utf-8", env=env, timeout=300)
    sel = S.read_json(run / "CHECKPOINT_SELECTION.json")
    assert r.returncode == 0 and sel["dry_run"] and sel["status"] in (EV.DRY, "NO_QUALIFIED_CHECKPOINT")


# ======================================================================= CLI
def cli(*args, stdin=None):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, "-m", "stillhere.cli", "persona", *args], capture_output=True, text=True, encoding="utf-8", env=env, input=stdin, timeout=600)


def test_cli_pipeline_review_doctor_evaluate_chat(tmp_path):
    chat = F.standard(days=80).write(tmp_path / "chat.json")
    root = tmp_path / "personas"
    r = cli("pipeline", "--input", str(chat), "--target", "cpu-only", "--speaker", "contact", "--root", str(root), "--non-interactive", "--json")
    assert r.returncode == 0, r.stderr[-1500:]
    out = json.loads(r.stdout[r.stdout.index("{"):])
    pid = out["profile_id"]
    st = out["stages"]
    assert st["05"]["status"] == "PASS" and st["06"]["status"] == "PASS" and st["12"]["status"] == "PASS" and Path(st["12"]["notebook"]).is_file()
    assert K.current(S.profile_dir(root, pid))["manifest"]["review_status"] == K.AUTO
    r2 = cli("pipeline", "--input", str(chat), "--target", "cpu-only", "--speaker", "contact", "--root", str(root), "--non-interactive", "--json")
    assert json.loads(r2.stdout[r2.stdout.index("{"):])["stages"]["12"]["package_id"] == st["12"]["package_id"]          # every stage reused
    r3 = cli("pipeline", "--input", str(chat), "--target", "colab-l4", "--speaker", "contact", "--root", str(root), "--non-interactive", "--json")
    assert json.loads(r3.stdout[r3.stdout.index("{"):])["stages"]["12"]["status"] == "BLOCKED"            # no cloud package without confirmation
    assert cli("package", "--profile", pid, "--root", str(root), "--target", "colab-l4", "--non-interactive").returncode == 3
    assert cli("pipeline", "--input", str(chat), "--root", str(root), "--non-interactive").returncode == 2    # no --speaker without a terminal
    r = cli("review", "--profile", pid, "--root", str(root), "--json")
    rv = json.loads(r.stdout[: r.stdout.rindex("}") + 1])
    assert r.returncode == 0 and rv["review_status"] == K.AUTO
    assert json.loads(cli("review", "--profile", pid, "--root", str(root), "--approve", "--json").stdout)["review_status"] == K.REVIEWED
    r = cli("doctor", "--target", "cpu-only", "--json")
    assert r.returncode == 0 and json.loads(r.stdout)["status"] == "PASS"
    r = cli("evaluate", "--profile", pid, "--root", str(root), "--groups", "A,B,C", "--backend", "fake", "--limit", "3", "--json")
    assert r.returncode == 0 and json.loads(r.stdout)["status"] == EV.DRY
    assert cli("evaluate", "--profile", pid, "--root", str(root), "--split", "test", "--backend", "fake").returncode == 3
    r = cli("chat", "--profile", pid, "--root", str(root), "--backend", "fake", "--message", "在吗")
    assert r.returncode == 0 and "DRY_RUN_ONLY" in r.stdout
    assert cli("import-adapter", "--profile", pid, "--root", str(root), "--file", str(tmp_path / "nope.zip")).returncode == 6
    assert cli("train", "--profile", pid, "--root", str(root), "--package", st["12"]["package_id"], "--device", "cpu").returncode == 3
    assert cli("consent", "--profile", pid).returncode == 2
    texts = [m["content"] for m in json.loads(chat.read_text(encoding="utf-8"))["messages"] if len(m["content"]) >= 6]
    for x in (r.stdout, rv, out):
        blob = json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x
        assert not any(t in blob for t in texts if "在吗" not in t)                                   # no chat text on stdout (except chat replies)
