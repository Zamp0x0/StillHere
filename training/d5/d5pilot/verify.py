"""Independent re-check of the WRITTEN pilot / full files (CPU, real Qwen3 tokenizer). Writes <full>/PILOT_VERIFY.json (aggregates only).

python -m d5pilot.verify --full <full_v1> --pilot <pilot_v1> --tokenizer <Qwen3-8B dir> --t2d4 <training/d4>
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

from . import common as K


def read(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def check_rows(rows, tok, M, cutoff):
    im_end = tok.token_to_id("<|im_end|>")
    c = collections.Counter()
    sup_total = inp_total = 0
    for r in rows:
        system, user, target = r["system"], r["conversations"][0]["value"], r["conversations"][1]["value"]
        c["rows"] += 1
        c["system_is_fact_v11"] += K.sha256_bytes(system.encode()) == K.fact_system_sha()
        pre = K.PRE.format(s=system, u=user)
        p = tok.encode(pre, add_special_tokens=False).ids
        s = tok.encode(target + K.SUP_SUFFIX, add_special_tokens=False).ids
        j = tok.encode(pre + target + K.SUP_SUFFIX, add_special_tokens=False).ids
        c["mask_boundary_ok"] += j == p + s
        c["eos_ok"] += s[-2] == im_end
        c["fits_cutoff"] += len(p) + len(s) <= cutoff
        sup_total += len(s)
        inp_total += len(p)
        try:
            t = json.loads(target)
            c["target_json_valid"] += list(t) == ["answer", "answerable", "abstain", "evidence", "quotes"]
        except Exception:
            continue
        eids = set(re.findall(r"^\[(E\d{2,3})\]", user, re.M))
        c["evidence_ids_exist"] += all(e in eids for e in t["evidence"])
        c["quotes_verbatim"] += all(q in user for q in t["quotes"])
        if t["abstain"]:
            c["abstain"] += 1
            c["abstain_contract"] += t["answer"].startswith(K.ABSTAIN) and t["answer"] in K.ABSTAIN_VARIANTS and not t["evidence"]
        else:
            c["answerable_with_evidence"] += bool(t["evidence"])
        c["no_target_in_prompt"] += target not in user and '"answerable"' not in user
        c["repetitive_target"] += K.repetitive(t["answer"], K.lang_of(r.get("source", "")), M)
        c["persona_filler_in_zh_answer"] += bool(re.search(r"哈哈|hhh|orz|整挺好", t["answer"]))
    return dict(c), sup_total, inp_total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    for k in ("full", "pilot", "tokenizer", "t2d4"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, a.t2d4)
    from t2d4.colab import metrics as M
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(a.tokenizer) / "tokenizer.json"))
    full, pilot = Path(a.full), Path(a.pilot)
    res = {}
    for name, p, cut in (("pilot_train", pilot / "train" / "pilot_train.jsonl", 8192), ("pilot_dev", pilot / "train" / "pilot_dev.jsonl", 8192),
                         ("pilot_train_le4096", pilot / "train" / "pilot_train_le4096.jsonl", 4096), ("full_train", full / "train.jsonl", 8192),
                         ("full_dev", full / "dev.jsonl", 8192), ("full_internal_holdout", full / "internal_holdout.jsonl", 8192)):
        rows = read(p)
        c, sup, inp = check_rows(rows, tok, M, cut)
        res[name] = {**c, "supervised_tokens": sup, "input_tokens": inp}
    ids = {k: {r["id"] for r in read(p)} for k, p in (("train", full / "train.jsonl"), ("dev", full / "dev.jsonl"), ("holdout", full / "internal_holdout.jsonl"),
                                                      ("pilot_train", pilot / "train" / "pilot_train.jsonl"), ("pilot_dev", pilot / "train" / "pilot_dev.jsonl"))}
    meta = {k: {m["id"]: m for m in read(full / f"{k}_meta.jsonl")} for k in ("train", "dev", "internal_holdout")}
    groups = {k: {m["group"] for m in v.values()} for k, v in meta.items()}
    res["isolation"] = {"pilot_train_subset_of_full_train": ids["pilot_train"] <= ids["train"], "pilot_dev_subset_of_full_dev": ids["pilot_dev"] <= ids["dev"],
                        "pilot_contains_holdout_ids": len((ids["pilot_train"] | ids["pilot_dev"]) & ids["holdout"]),
                        "id_overlap_train_dev": len(ids["train"] & ids["dev"]), "id_overlap_train_holdout": len(ids["train"] & ids["holdout"]),
                        "group_overlap": {"train&dev": len(groups["train"] & groups["dev"]), "train&holdout": len(groups["train"] & groups["internal_holdout"]),
                                          "dev&holdout": len(groups["dev"] & groups["internal_holdout"])},
                        "pairs_split_consistent": all(meta["train"].get(m["pair_id"], {"split": m["split"]})["split"] == m["split"]
                                                      for m in meta["train"].values() if m.get("pair_id") and m["pair_id"] != m["id"])}
    ok = all(v.get("rows") == v.get("mask_boundary_ok") == v.get("eos_ok") == v.get("fits_cutoff") == v.get("target_json_valid") == v.get("evidence_ids_exist")
             == v.get("quotes_verbatim") == v.get("no_target_in_prompt") == v.get("system_is_fact_v11") and v.get("abstain") == v.get("abstain_contract")
             and v.get("repetitive_target", 0) == 0 for k, v in res.items() if k != "isolation")
    iso = res["isolation"]
    ok &= iso["pilot_train_subset_of_full_train"] and iso["pilot_dev_subset_of_full_dev"] and iso["pilot_contains_holdout_ids"] == 0 and \
        not any(iso["group_overlap"].values()) and iso["id_overlap_train_dev"] == 0 and iso["id_overlap_train_holdout"] == 0
    res["VERIFY"] = "PASS" if ok else "FAIL"
    (full / "PILOT_VERIFY.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
