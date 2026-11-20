"""The documented user workflows on the public SYNTHETIC demo export (Alice / Bob), CPU only, through the real CLI:
quickstart (--json is pure JSON), status (ACTIVE_MEMORY_PIPELINE), memory-query, a labelled fake chat, the NO_GPU message, the
'another export of the same chat' note + incremental update, a second persona, download-model, and the active-pipeline labels."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stillhere import cli  # noqa: E402
from stillhere.persona import config as CF  # noqa: E402
from stillhere.persona import model_download as MD  # noqa: E402
from stillhere.persona import product_config as PC  # noqa: E402

DEMO = ROOT / "examples" / "synthetic" / "weflow_demo"
EXPORT, UPDATE = DEMO / "alice_bob_weflow.json", DEMO / "alice_bob_weflow_update.json"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("STILLHERE_EMBEDDING_MODEL_DIR", "")
    mp.setenv("HF_HOME", str(tmp_path_factory.mktemp("hf")))        # no embedding model in this cache: the product runs V2L + P2
    root = tmp_path_factory.mktemp("profiles")
    yield root
    mp.undo()


def run(capsys, *argv):
    rc = cli.main(list(argv))
    out = capsys.readouterr()
    return rc, out.out, out.err


def test_demo_export_is_deterministic_and_parses(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_demo", DEMO / "make_demo_export.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert json.dumps(m.build(100), ensure_ascii=False, indent=1) + "\n" == EXPORT.read_text(encoding="utf-8")
    assert json.dumps(m.build(110), ensure_ascii=False, indent=1) + "\n" == UPDATE.read_text(encoding="utf-8")
    from stillhere.persona import weflow as W
    exp = W.load(EXPORT)
    assert exp.report["messages"] > 900 and {x["senderDisplayName"] for x in json.loads(EXPORT.read_text(encoding="utf-8"))["messages"]} == {"Alice", "Bob"}


def test_quickstart_status_query_and_fake_chat(world, capsys):
    rc, out, err = run(capsys, "persona", "quickstart", "--input", str(EXPORT), "--speaker", "contact", "--root", str(world), "--non-interactive", "--json")
    res = json.loads(out)                                                       # stdout is pure JSON with --json
    assert rc == 0 and res["overall"] == "PARTIAL_READY" and res["stages"]["05"] == "PASS" and "Detected speakers" in err
    pid = res["profile_id"]
    rc, out, _ = run(capsys, "persona", "status", "--root", str(world), "--json")
    assert [p["profile_id"] for p in json.loads(out)["profiles"]] == [pid]
    rc, out, _ = run(capsys, "persona", "status", "--root", str(world), "--profile", pid, "--json")
    st = json.loads(out)
    assert st["memory_pipeline"]["ACTIVE_MEMORY_PIPELINE_expected"] == "V2L_P2_FALLBACK" and st["memory"]["archive_messages"] > 900
    assert st["adapter"]["enabled"] is False and st["card"]["card_id"].startswith("c-")
    rc, out, _ = run(capsys, "persona", "memory-query", "--root", str(world), "--profile", pid, "--q", "西湖改到几号了？")
    assert rc == 0 and "西湖改到25号" in out
    rc, out, _ = run(capsys, "persona", "chat", "--root", str(world), "--profile", pid, "--backend", "fake", "--message", "你领养的猫叫什么？")
    assert rc == 0 and "ACTIVE_MEMORY_PIPELINE: V2L_P2_FALLBACK" in out and "SYNTHETIC" in out
    rc, out, _ = run(capsys, "persona", "quickstart", "--input", str(EXPORT), "--speaker", "contact", "--root", str(world), "--non-interactive")
    assert rc == 0 and "not PASS" in out and "persona status --profile" in out        # a re-run is idempotent and says what is pending


def test_no_gpu_gives_options_instead_of_a_traceback(world, capsys, monkeypatch):
    pid = json.loads(run(capsys, "persona", "status", "--root", str(world), "--json")[1])["profiles"][0]["profile_id"]

    def no_gpu(cfg):
        raise RuntimeError("NO_GPU: no CUDA GPU was found.")
    monkeypatch.setattr(CF, "resolve_backend", no_gpu)
    rc, out, err = run(capsys, "persona", "chat", "--root", str(world), "--profile", pid, "--message", "你好")
    assert rc == 3 and "--answer-provider deepseek_api" in err and "--backend fake" in err


def test_newer_export_is_an_update_not_a_new_persona(world, capsys):
    pid = json.loads(run(capsys, "persona", "status", "--root", str(world), "--json")[1])["profiles"][0]["profile_id"]
    rc, out, _ = run(capsys, "persona", "memory-build", "--root", str(world), "--profile", pid, "--source", str(UPDATE), "--json")
    ev = json.loads(out)["evidence"]
    assert rc == 0 and ev["added"] > 0
    rc, out, _ = run(capsys, "persona", "memory-build", "--root", str(world), "--profile", pid, "--source", str(UPDATE), "--json")
    assert json.loads(out)["evidence"]["added"] == 0                              # repeat import: nothing duplicated
    rc, out, err = run(capsys, "persona", "quickstart", "--input", str(UPDATE), "--speaker", "contact", "--root", str(world), "--non-interactive", "--json")
    assert rc == 0 and f"memory-build --profile {pid} --source" in err             # the user is told how to UPDATE instead


def test_second_persona_has_its_own_memory(world, capsys):
    rc, out, _ = run(capsys, "persona", "quickstart", "--input", str(EXPORT), "--speaker", "self", "--root", str(world), "--non-interactive", "--json")
    bob = json.loads(out)["profile_id"]
    profiles = json.loads(run(capsys, "persona", "status", "--root", str(world), "--json")[1])["profiles"]
    assert bob in {p["profile_id"] for p in profiles} and len({p["profile_id"] for p in profiles}) == len(profiles)
    st = json.loads(run(capsys, "persona", "status", "--root", str(world), "--profile", bob, "--json")[1])
    assert st["memory"]["db"].startswith(str(world / bob))


def test_active_pipeline_labels_and_download_model(tmp_path, capsys):
    assert PC.active_pipeline(PC.PRODUCT_MEMORY_PIPELINE, True) == "V2D_P2" and PC.active_pipeline(PC.PRODUCT_MEMORY_PIPELINE, False) == "V2L_P2_FALLBACK"
    assert PC.active_pipeline(PC.PRODUCT_V2D_MEMORY_PIPELINE, False) == "V2L_FALLBACK" and PC.active_pipeline(None, False) == "V1"
    seen = {}

    def fake_snapshot(**kw):
        seen.update(kw)
        d = tmp_path / "m"
        d.mkdir(exist_ok=True)
        (d / "config.json").write_text("{}", encoding="utf-8")
        return str(d)
    r = MD.download("embedding", snapshot=fake_snapshot)
    assert seen == {"repo_id": "Qwen/Qwen3-Embedding-0.6B", "revision": PC.EMBEDDING["revision"]} and r["env"] == "STILLHERE_EMBEDDING_MODEL_DIR"
    with pytest.raises(MD.DownloadError):
        MD.download("other")


def test_training_package_code_list_works_without_git(tmp_path):
    from stillhere.persona import package as P
    (tmp_path / "src" / "stillhere").mkdir(parents=True)
    (tmp_path / "src" / "stillhere" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "src" / "stillhere" / "__pycache__").mkdir()
    (tmp_path / "src" / "stillhere" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\0")
    (tmp_path / "training" / "persona").mkdir(parents=True)
    (tmp_path / "training" / "persona" / "t.py").write_text("y = 2\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    assert P.code_files(tmp_path) == ["pyproject.toml", "src/stillhere/a.py", "training/persona/t.py"]     # a ZIP download, no .git


def test_package_build_without_pyyaml_is_a_clear_error(monkeypatch, tmp_path):
    import builtins
    from stillhere.persona import package as P
    real = builtins.__import__

    def no_yaml(name, *a, **k):
        if name == "yaml":
            raise ImportError("no yaml")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_yaml)
    with pytest.raises(P.PackageError, match="NEEDS_PYYAML"):
        P.build(tmp_path, "p-0000000000", target="cpu-only")
