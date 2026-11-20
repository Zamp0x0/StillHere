"""r2 training preparation: the r2 config template, the every-save candidate grid, prompt-consistent rows and the r2 selection rule.
CPU only; uses the synthetic persona world of test_persona_p2p3 (no private data)."""
from __future__ import annotations

import json

import pytest
import yaml

from stillhere.persona import evaluation as EV
from stillhere.persona import package as P

from test_persona_p2p3 import _answers, items, world  # noqa: F401  (fixtures)
from stillhere.persona import card as K


def test_r1_plan_is_unchanged_and_r2_plan_has_six_candidates():
    t = {"gradient_accumulation_steps": 8, "per_device_train_batch_size": 1, "num_train_epochs": 1.0}
    assert P.plan(2101, t)["candidate_steps"] == [132, 263]                     # r1 behaviour is the default
    p = P.plan(2101, {**t, "persona_candidate_policy": "every_save"})
    assert p["steps"] == 263 and p["save_steps"] == 44 and p["candidate_steps"] == [44, 88, 132, 176, 220, 263]
    assert p["checkpoints_expected"] == p["candidate_steps"]
    for rows in (5, 57, 200, 4001):
        q = P.plan(rows, {**t, "persona_candidate_policy": "every_save"})
        assert q["candidate_steps"][-1] == q["steps"] and all(1 <= c <= q["steps"] for c in q["candidate_steps"])


def test_r2_template_changes_only_the_registered_variables():
    r1 = yaml.safe_load(P.CONFIG_TEMPLATE.read_text(encoding="utf-8"))
    r2 = yaml.safe_load(P.CONFIG_TEMPLATE_R2.read_text(encoding="utf-8"))
    changed = {k for k in set(r1) | set(r2) if r1.get(k) != r2.get(k)}
    assert changed == {"persona_candidate_policy", "persona_selection_rule", "persona_train_system", "persona_run_label"}, changed
    assert r2["persona_train_system"] == "persona_prompt" and r2["persona_selection_rule"] == "r2"
    b, pl = P.resolved_config(2101, P.CONFIG_TEMPLATE_R2)
    c = yaml.safe_load(b)
    assert c["save_steps"] == 44 and c["persona_candidate_steps"] == [44, 88, 132, 176, 220, 263] and c["per_device_train_batch_size"] == 1


def test_r2_persona_keys_never_reach_llamafactory():
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location("tp", pathlib.Path(P.ROOT) / "training/persona/personatrain/train_persona.py")
    tp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tp)
    r2 = yaml.safe_load(P.CONFIG_TEMPLATE_R2.read_text(encoding="utf-8"))
    assert {k for k in r2 if k.startswith("persona_")} <= set(tp.PERSONA_KEYS)


def test_with_system_replaces_only_the_system_message():
    rows = [{"id": "a", "system": "old", "conversations": [{"from": "human", "value": "对方：在吗"}, {"from": "gpt", "value": "在"}]},
            {"id": "b", "system": "old", "conversations": [{"from": "human", "value": "对方：吃了吗"}, {"from": "gpt", "value": "没呢"}]}]
    b = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    out = [json.loads(l) for l in P.with_system(b, "PERSONA").decode("utf-8").splitlines()]
    assert [r["id"] for r in out] == ["a", "b"] and all(r["system"] == "PERSONA" for r in out)
    assert [r["conversations"] for r in out] == [r["conversations"] for r in rows]


def test_r2_rule_rejects_partial_collapse_that_r1_accepts(world, items, tmp_path):  # noqa: F811
    ref = EV.Reference(world["ddir"], K.current(world["pdir"])["card"])
    run = tmp_path / "sel"
    refs = {it["sample_id"]: it["reference"] for it in items}
    sids = sorted(refs)
    varied = {s: f"收到第{i}个" for i, s in enumerate(sids)}
    partial = dict(varied)
    gen = ["好", "嗯", "哦", "行", "可以", "收到", "嗯嗯", "好的"]                     # half the answers -> generic replies, spread so top-1 stays low
    for i, s in enumerate(sids[: len(sids) // 2]):
        partial[s] = gen[i % len(gen)]
    _answers(run, "B", {s: "这个问题可以先看看再说，希望对你有帮助" for s in refs})
    _answers(run, "ckpt-3", varied)
    _answers(run, "ckpt-6", partial)
    r1 = EV.select_checkpoint(run, items, ["ckpt-3", "ckpt-6"], ref, rule="r1")
    r2 = EV.select_checkpoint(run, items, ["ckpt-3", "ckpt-6"], ref, rule="r2")
    assert r1["rule_id"] == "r1" and r2["rule_id"] == "r2"
    assert r2["selected"] == "ckpt-3" and not r2["candidates"]["ckpt-6"]["qualified"], (len(sids), r2["reference_top1_share"], r2["reference_distinct_ratio"], r2["candidates"])
    assert any(x.endswith("_r2") for x in r2["candidates"]["ckpt-6"]["reasons"]), r2["candidates"]["ckpt-6"]
    assert set(r1["candidates"]["ckpt-6"]["reasons"]) == {x for x in r2["candidates"]["ckpt-6"]["reasons"] if not x.endswith("_r2")}
    with pytest.raises(EV.EvalError):
        EV.select_checkpoint(run, items, ["ckpt-3"], ref, rule="r9")
