"""The frozen final configuration (persona.final_config) and its runtime wiring: runtime_config.json with the ``memory_pipeline`` section
builds the V2 lexical pipeline with the P1 reader; a config without it keeps the historical V1 / reader v1 path. Synthetic chats only."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import persona_fixture as F  # noqa: E402
from stillhere.persona import final_config as FC  # noqa: E402
from stillhere.persona import memory as M  # noqa: E402
from stillhere.persona import pipeline as PL  # noqa: E402
from stillhere.persona import store as S  # noqa: E402
from stillhere.persona.runtime import PersonaRuntime  # noqa: E402


def test_final_config_is_the_evidence_backed_combination():
    c = FC.FINAL_CONFIG
    assert c["adapter"]["enabled_by_default"] is False and c["memory_pipeline"]["pipeline"] == "v2" and c["memory_pipeline"]["dense"] is False
    assert c["memory_pipeline"]["reader_version"] == "p1" and c["memory_pipeline"]["writer_mode"] == "EVIDENCE_ONLY"
    assert c["memory_pipeline"]["strip_qform"] is False                    # BEST_STABLE_V2 was not adopted
    assert len(FC.fingerprint()) == 64 and FC.fingerprint() == FC.fingerprint(json.loads(json.dumps(c)))


def _profile(tmp_path, monkeypatch):
    orig = M.build_api
    monkeypatch.setattr(M, "build_api", lambda *a, **k: orig(*a, **{**k, "synthetic": True}))
    chat = F.standard(days=80).write(tmp_path / "c.json")
    root = tmp_path / "r"
    pid = PL.run(chat, speaker="contact", root=root, target="cpu-only")["profile_id"]
    return root, pid, S.profile_dir(root, pid) / "runtime" / "runtime_config.json"


def test_runtime_config_without_the_section_keeps_the_historical_path(tmp_path, monkeypatch):
    root, pid, path = _profile(tmp_path, monkeypatch)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    assert cfg["memory_pipeline"]["version"].startswith("stillhere.persona.product/")   # a NEW profile is finalized with the product default
    cfg.pop("memory_pipeline")                                                          # an older config without the section
    path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    rt = PersonaRuntime.from_config(root, pid, backend="fake")
    assert rt.memory_pipeline == "v1" and rt.reader_version == "v1" and rt.retrieval == "V1"


def test_the_final_section_builds_v2_lexical_with_the_p1_reader(tmp_path, monkeypatch):
    root, pid, path = _profile(tmp_path, monkeypatch)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    new = FC.apply(cfg)
    assert {k: v for k, v in new.items() if k != "memory_pipeline"} == {k: v for k, v in cfg.items() if k != "memory_pipeline"}   # nothing else changes
    path.write_text(json.dumps(new, ensure_ascii=False), encoding="utf-8")
    rt = PersonaRuntime.from_config(root, pid, backend="fake")
    assert rt.memory_pipeline == "v2" and rt.reader_version == "p1" and rt.v2cfg.use_dense is False and rt.v2cfg.neighbors == (2, 2)
    assert rt.active is None                                                          # the adapter stays off
    new["memory_pipeline"]["dense"] = True                                           # no dense index here: V2L fallback, said in the notes
    path.write_text(json.dumps(new, ensure_ascii=False), encoding="utf-8")
    rt = PersonaRuntime.from_config(root, pid, backend="fake")
    assert rt.dense is None and rt.v2cfg.use_dense is False and rt.retrieval == "V2L"
    assert any("DENSE_UNAVAILABLE" in n and "V2L" in n for n in rt.memory_notes)
    new["memory_pipeline"]["dense_fallback"] = "none"                                # a pipeline that forbids the fallback refuses
    path.write_text(json.dumps(new, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        PersonaRuntime.from_config(root, pid, backend="fake")


def test_a_re_finalize_keeps_the_final_memory_pipeline(tmp_path, monkeypatch):
    from stillhere.persona import quality as Q
    root, pid, path = _profile(tmp_path, monkeypatch)
    path.write_text(json.dumps(FC.apply(json.loads(path.read_text(encoding="utf-8"))), ensure_ascii=False), encoding="utf-8")
    Q.select_runtime(root, pid)                                                       # what `persona finalize` runs
    cfg = json.loads(path.read_text(encoding="utf-8"))
    assert cfg["memory_pipeline"]["reader_version"] == "p1" and cfg["memory_pipeline"]["pipeline"] == "v2"
