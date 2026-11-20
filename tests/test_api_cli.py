"""Public API + CLI smoke (CPU, synthetic, FakeGenerator: no model is loaded)."""
import json

import pytest

from stillhere import api, cli


@pytest.mark.parametrize("context", ["FULL_CONTEXT", "D4_COMPATIBLE"])
def test_demo_runtime_answers_from_memory_in_both_context_configs(context):
    rt = api.demo_runtime(context, loop_guard_enabled=True)
    s = rt.create_session(api.DEMO_PERSONA)
    r = rt.reply_session(api.DEMO_PERSONA, s, "我下周去哪出差？", "r1", generate=api.FakeGenerator(), backend="fake", env=api.GUARDS_ON)
    assert r.status == "GENERATED" and "深圳" in r.text
    assert r.trace["context_config"]["render_mode"] == ("d4" if context == "D4_COMPATIBLE" else "prompt2")


def test_unknown_context_is_refused():
    with pytest.raises(ValueError):
        api.demo_runtime("BIG")


def test_cli_demo_json(capsys):
    assert cli.main(["demo", "--json", "--context", "D4_COMPATIBLE"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["context"] == "D4_COMPATIBLE" and len(out["turns"]) == 4
    assert out["turns"][-1]["loop_guard"] == "TRIMMED" and out["turns"][-1]["reply"].count("杭州西湖区") == 1


def test_cli_loop_guard_off_keeps_the_raw_text(capsys):
    assert cli.main(["demo", "--json", "--loop-guard", "off"]) == 0
    last = json.loads(capsys.readouterr().out)["turns"][-1]
    assert last["loop_guard"] == "DISABLED"


def test_cli_version_and_loop_guard(capsys):
    assert cli.main(["version"]) == 0 and capsys.readouterr().out.strip() == __import__("stillhere").__version__
    assert cli.main(["loop-guard", "先试试，先试试，先试试，先试试，"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "TRIMMED"
