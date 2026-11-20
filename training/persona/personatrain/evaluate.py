"""Persona evaluation runner on a GPU (Colab L4 / A100, or a local NVIDIA GPU with >= 7 GiB for NF4 inference). The metrics, groups,
selection rule and statistics live in ``stillhere.persona.evaluation`` (CPU, tested); this module only attaches the real generator.

    python -m personatrain.evaluate generate --eval-set E --base-dir B [--adapter NAME=DIR ...] --groups B,C --run-dir R [--alias ckpt-132=C ...]
    python -m personatrain.evaluate select   --eval-set E --run-dir R --train-dir D/train --card D/persona/persona_card.yaml --candidates ckpt-132,ckpt-263
    python -m personatrain.evaluate score    --eval-set E --run-dir R --train-dir D/train --card ... --groups A,B,C,D,E
    (--backend fake: the CPU dry run; every output is labelled DRY_RUN_ONLY)

Generator: ``d5pilot.evaluate.Gen`` (the GPU-verified D5 path: ONE NF4 base, adapters attached by name, base groups with every adapter
disabled, same render / seed / stop tokens). Group -> adapter: A, B, D -> none; C, E -> the adapter named ``C`` / ``E`` (or ``--adapter``);
an ``--alias ckpt-N=C`` group uses the messages of C with the adapter named ckpt-N (checkpoint selection).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stillhere.persona import card as K
from stillhere.persona import evaluation as EV


def _items(eval_set: Path, aliases: dict):
    items = EV.read_eval_set(eval_set)
    for it in items:
        for name, src in aliases.items():
            it["messages"][name] = it["messages"][src]
            it["prompt_hash"][name] = it["prompt_hash"][src]
    return items


def _kv(xs):
    out = {}
    for x in xs or []:
        k, v = x.split("=", 1)
        out[k] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "select", "score"])
    ap.add_argument("--eval-set", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--base-dir")
    ap.add_argument("--adapter", action="append")
    ap.add_argument("--alias", action="append")
    ap.add_argument("--groups", default="B,C")
    ap.add_argument("--candidates")
    ap.add_argument("--train-dir")
    ap.add_argument("--card")
    ap.add_argument("--backend", choices=["hf", "fake"], default="hf")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--fingerprint-json")
    ap.add_argument("--rule", choices=sorted(EV.SELECTION_RULES), default="r1", help="checkpoint selection rule (select only)")
    a = ap.parse_args(argv)
    aliases = _kv(a.alias)
    if a.candidates:
        aliases.update({c: "C" for c in a.candidates.split(",") if c not in aliases})
    items = _items(Path(a.eval_set), aliases)
    run = Path(a.run_dir)
    groups = [g for g in a.groups.split(",") if g]
    if a.cmd == "generate":
        adapters = _kv(a.adapter)
        if a.backend == "fake":
            gen = EV.FakeGenerator()
        else:
            from d5pilot.evaluate import Gen
            gen = Gen(Path(a.base_dir), {n: Path(d) for n, d in adapters.items()})
        with_adapter = [g for g in groups if g in aliases or EV.GROUPS.get(g, {}).get("adapter")]
        missing = [g for g in with_adapter if a.backend == "hf" and g not in adapters]
        if missing:
            raise SystemExit(f"groups {missing} need --adapter <group>=<adapter dir>")
        amap = {g: (g if g in with_adapter else None) for g in groups}
        head = json.loads(Path(a.eval_set).with_suffix(".meta.json").read_text(encoding="utf-8"))
        fp = {"eval_set_sha256": head["file_sha256"], "persona_prompt_sha256": head["persona_prompt_sha256"], "decode": EV.DECODE, "generator": gen.label,
              "adapters": {n: (gen.info[n]["sha256"] if a.backend == "hf" else n) for n in adapters}, "eval_version": EV.EVAL_VERSION}
        if a.fingerprint_json:
            fp["extra"] = json.loads(Path(a.fingerprint_json).read_text(encoding="utf-8"))
        s = EV.generate(gen, items, groups, run, fp, adapters=amap, limit=a.limit)
        print(json.dumps(s, ensure_ascii=False))
        return 0 if all(v["ok"] == v["of"] for v in s.values()) or a.limit else 1
    card = K.from_yaml(Path(a.card).read_text(encoding="utf-8")) if a.card else None
    ref = EV.Reference(Path(a.train_dir), card)
    if a.cmd == "select":
        res = EV.select_checkpoint(run, items, a.candidates.split(","), ref, rule=a.rule)
    else:
        res = EV.score_run(run, items, groups, ref)
    print(json.dumps({k: v for k, v in res.items() if k not in ("success_rule", "rule")}, ensure_ascii=False, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
