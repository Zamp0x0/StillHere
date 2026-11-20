"""Loss-mask check with the REAL LLaMA-Factory preprocessing (Colab cell 08; tokenizer only, no weights).

python -m d5pilot.lf_labels --data <pilot data dir> --model-dir <Qwen3-8B snapshot> --work <empty dir> --out <run dir> [--cutoff 8192]

Rows: the 16 longest pilot rows + up to 8 per source (seeded) + every abstention kind. For each row LLaMA-Factory's labels must equal
the static layout: prompt (system, user, empty think block) fully masked; supervised span == tokens(target + "<|im_end|>\\n");
no truncation; a negative control (train_on_prompt=true) must be caught. Output carries ids and counts only.
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

from . import common as K

SETTINGS = dict(stage="sft", do_train=True, finetuning_type="lora", template="qwen3", enable_thinking=False, train_on_prompt=False, mask_history=True,
                packing=False, overwrite_cache=True, preprocessing_num_workers=None, per_device_train_batch_size=1, report_to="none", use_cpu=True)


def choose(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["source"]].append(r)
    out = sorted(rows, key=lambda r: -len(r["conversations"][0]["value"]))[:16]
    for s, rr in sorted(by.items()):
        out += sorted(rr, key=lambda r: K.frac("lf:" + r["id"]))[:8]
    out += [r for r in rows if '"abstain": true' in r["conversations"][1]["value"]][:6]
    seen, res = set(), []
    for r in out:
        if r["id"] not in seen:
            seen.add(r["id"])
            res.append(r)
    return res


def check(tok, ex, row, cutoff):
    ids, labels = list(ex["input_ids"]), list(ex["labels"])
    pre = K.PRE.format(s=row["system"], u=row["conversations"][0]["value"])
    target = row["conversations"][1]["value"]
    exp_p = tok.encode(pre, add_special_tokens=False)
    exp_s = tok.encode(target + K.SUP_SUFFIX, add_special_tokens=False)
    sup = [i for i, l in enumerate(labels) if l != -100]
    n = len(exp_p)
    return {"ids_equal_static_layout": ids == exp_p + exp_s, "prompt_fully_masked": all(l == -100 for l in labels[:n]),
            "supervised_span_is_target": sup == list(range(n, n + len(exp_s))), "labels_equal_ids_on_target": all(labels[i] == ids[i] for i in sup),
            "eos_last": ids[-2:] == [151645, 198] and labels[-2:] == [151645, 198], "no_truncation": len(ids) == len(exp_p) + len(exp_s) <= cutoff,
            "nonzero_supervision": len(sup) > 0}, {"n_ids": len(ids), "n_sup": len(sup)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cutoff", type=int, default=K.CUTOFF)
    ap.add_argument("--train-file", default="pilot_train.jsonl")
    ap.add_argument("--all-rows", action="store_true", help="check every row instead of the representative sample")
    a = ap.parse_args(argv)
    t0 = time.time()
    d, work, out = Path(a.data), Path(a.work), Path(a.out)
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (d / "train" / a.train_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    chosen = rows if a.all_rows else choose(rows)
    ds = work / "lf_check"
    ds.mkdir(exist_ok=True)
    with open(ds / "rows.jsonl", "w", encoding="utf-8") as f:
        for r in chosen:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    info = json.loads((d / "train" / "dataset_info.json").read_text(encoding="utf-8"))["d5_pilot_train"]
    (ds / "dataset_info.json").write_text(json.dumps({"d5_check": {**info, "file_name": "rows.jsonl"}}), encoding="utf-8")
    import llamafactory
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer

    def run(over, tag):
        args = dict(SETTINGS, **over, cutoff_len=a.cutoff, model_name_or_path=a.model_dir, dataset_dir=str(ds), dataset="d5_check",
                    output_dir=str(work / f"o_{tag}"))
        with contextlib.redirect_stdout(io.StringIO()):
            m, da, tr, ft, _ = get_train_args(args)
            tm = load_tokenizer(m)
            tpl = get_template_and_fix_tokenizer(tm["tokenizer"], da)
            data = get_dataset(tpl, m, da, tr, stage="sft", **tm)["train_dataset"]
        return tm["tokenizer"], data
    tok, data = run({}, "main")
    per, fails, st = collections.Counter(), [], []
    for r, ex in zip(chosen, data):
        res, s = check(tok, ex, r, a.cutoff)
        st.append(s)
        per.update(k for k, v in res.items() if v)
        if not all(res.values()):
            fails.append({"id": r["id"], "failed": [k for k, v in res.items() if not v]})
    _, neg = run({"train_on_prompt": True, "mask_history": False}, "neg")      # LLaMA-Factory rejects train_on_prompt with mask_history
    caught = sum(1 for r, ex in zip(chosen, neg) if not check(tok, ex, r, a.cutoff)[0]["prompt_fully_masked"])
    ok = len(data) == len(chosen) and not fails and caught == len(chosen)
    res = {"check": "D5 LLaMA-Factory labels", "scope": "all rows" if a.all_rows else "representative sample", "framework_version": llamafactory.__version__, "cutoff_len": a.cutoff, "rows": len(chosen),
           "rows_returned": len(data), "passed": dict(per), "failures": fails, "negative_control_caught": caught,
           "max_input_ids": max(x["n_ids"] for x in st) if st else None, "train_file_sha256": hashlib.sha256((d / "train" / a.train_file).read_bytes()).hexdigest(),
           "LLAMAFACTORY_LABELS": "PASS" if ok else "FAIL", "seconds": round(time.time() - t0, 1)}
    (out / "LF_LABELS_RESULT.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
