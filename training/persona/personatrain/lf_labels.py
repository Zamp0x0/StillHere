"""Full loss-mask audit with the REAL LLaMA-Factory 0.9.5 preprocessing (tokenizer only, no weights; CPU or Colab).

python -m personatrain.lf_labels --data <package data dir> --model-dir <Qwen3-8B snapshot> --work <scratch dir> --out <dir> [--cutoff 2048]
        [--splits train,dev] [--persona-prompt <persona_prompt.txt>]

Every row of every split is checked (``d5pilot.lf_labels.check``, unchanged):
* LLaMA-Factory input ids == the static layout ``<|im_start|>system … user … assistant <think></think>`` + target + ``<|im_end|>\\n``;
* the whole prompt is masked (system tokens, the other person's lines, and the target person's EARLIER replies, which sit inside the
  human turn), the supervised span is exactly the current reply + EOS, nothing is truncated at ``cutoff``;
* a negative control (``train_on_prompt=true``) must be caught on every row.
Extra (persona): the HF chat template of the Qwen3 tokenizer, rendered for inference (``add_generation_prompt``, ``enable_thinking=False``),
must equal the training prefix for the training system prompt AND for the persona inference prompt (same role / turn / think layout;
only the system text differs). Output: ids and counts only, no text.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import io
import json
import sys
import time
from pathlib import Path

from d5pilot import common as K
from d5pilot.lf_labels import SETTINGS, check

CUTOFF = 2048


def _rows(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cutoff", type=int, default=CUTOFF)
    ap.add_argument("--splits", default="train,dev")
    ap.add_argument("--persona-prompt")
    ap.add_argument("--negative-control", choices=["on", "off"], default="on")
    a = ap.parse_args(argv)
    t0 = time.time()
    d, work, out = Path(a.data), Path(a.work), Path(a.out)
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((d / "train" / "dataset_info.json").read_text(encoding="utf-8"))
    import llamafactory
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer

    def run(ds: Path, over, tag):
        args = dict(SETTINGS, **over, cutoff_len=a.cutoff, model_name_or_path=a.model_dir, dataset_dir=str(ds), dataset="persona_check",
                    output_dir=str(work / f"o_{tag}"))
        with contextlib.redirect_stdout(io.StringIO()):
            m, da, tr, ft, _ = get_train_args(args)
            tm = load_tokenizer(m)
            tpl = get_template_and_fix_tokenizer(tm["tokenizer"], da)
            data = get_dataset(tpl, m, da, tr, stage="sft", **tm)["train_dataset"]
        return tm["tokenizer"], data

    persona_prompt = Path(a.persona_prompt).read_text(encoding="utf-8") if a.persona_prompt else None
    res = {"check": "persona LLaMA-Factory labels (all rows)", "framework_version": llamafactory.__version__, "cutoff_len": a.cutoff, "splits": {}}
    ok_all = True
    for split in [s.strip() for s in a.splits.split(",") if s.strip()]:
        key = {"train": "persona_train", "dev": "persona_dev", "smoke": "persona_smoke"}[split]
        f = d / "train" / info[key]["file_name"]
        rows = _rows(f)
        ds = work / f"lf_{split}"
        ds.mkdir(exist_ok=True)
        (ds / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        (ds / "dataset_info.json").write_text(json.dumps({"persona_check": {**info[key], "file_name": "rows.jsonl"}}), encoding="utf-8")
        tok, data = run(ds, {}, f"{split}_main")
        per, fails, st = collections.Counter(), [], []
        tpl_train = tpl_persona = 0
        hist = 0
        for r, ex in zip(rows, data):
            c, s = check(tok, ex, r, a.cutoff)
            st.append(s)
            per.update(k for k, v in c.items() if v)
            if not all(c.values()):
                fails.append({"id": r["id"], "failed": [k for k, v in c.items() if not v]})
            user = r["conversations"][0]["value"]
            hist += any(ln.startswith("我：") for ln in user.split("\n"))
            msgs = [{"role": "system", "content": r["system"]}, {"role": "user", "content": user}]
            tpl_train += tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False) == K.PRE.format(s=r["system"], u=user)
            if persona_prompt is not None:
                pm = [{"role": "system", "content": persona_prompt}, {"role": "user", "content": user}]
                tpl_persona += tok.apply_chat_template(pm, tokenize=False, add_generation_prompt=True, enable_thinking=False) == K.PRE.format(s=persona_prompt, u=user)
        caught = None
        if a.negative_control == "on":
            _, neg = run(ds, {"train_on_prompt": True, "mask_history": False}, f"{split}_neg")
            caught = sum(1 for r, ex in zip(rows, neg) if not check(tok, ex, r, a.cutoff)[0]["prompt_fully_masked"])
        ok = (len(data) == len(rows) and not fails and tpl_train == len(rows) and (persona_prompt is None or tpl_persona == len(rows))
              and (caught is None or caught == len(rows)))
        ok_all &= ok
        res["splits"][split] = {"rows": len(rows), "rows_returned": len(data), "passed": dict(per), "failures": fails[:20], "failures_total": len(fails),
                                "rows_with_earlier_target_replies_in_prompt": hist, "inference_template_equals_training_layout": tpl_train,
                                "persona_prompt_template_equals_training_layout": tpl_persona if persona_prompt is not None else None,
                                "negative_control_caught": caught, "max_input_ids": max((x["n_ids"] for x in st), default=None),
                                "max_supervised": max((x["n_sup"] for x in st), default=None), "supervised_tokens": sum(x["n_sup"] for x in st),
                                "file_sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "status": "PASS" if ok else "FAIL"}
    res.update(persona_prompt_sha256=hashlib.sha256(persona_prompt.encode("utf-8")).hexdigest() if persona_prompt else None,
               thinking="enable_thinking=false: empty <think></think> block in the prompt (masked), same as inference",
               eos="<|im_end|> (151645) + \\n (198) supervised as the last two tokens", LLAMAFACTORY_LABELS="PASS" if ok_all else "FAIL",
               seconds=round(time.time() - t0, 1))
    (out / "LF_LABELS_RESULT.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "splits"} | {"splits": {k: {x: v[x] for x in ("rows", "status", "failures_total", "negative_control_caught", "max_input_ids")}
                                                                                     for k, v in res["splits"].items()}}, ensure_ascii=False, indent=1))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
