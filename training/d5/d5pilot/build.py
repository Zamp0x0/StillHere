"""Build the D5 full candidate pool (Windows-local) and the D5 pilot subset (the only part that goes to Colab).  CPU only.

python -m d5pilot.build --raw <data-dir>/research/d5_dataset_audit/raw --tokenizer <models-dir>/Qwen3-8B \
    --d4 <data-dir>/research/t2d4_dataset/v1.1 --d4-labels <data-dir>/research/d5_dataset_audit/d4_audit/d4_reuse_labels.jsonl \
    --t2d4 <repo>/training/d4 --lifebench <dev LifeBench json (conversation block only is read)> \
    --full <data-dir>/research/d5_dataset/full_v1 --pilot <data-dir>/research/d5_dataset/pilot_v1
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import common as K
from . import sources as S

QUOTA = {  # full candidate pool, per split (targets, not obligations)
    "perltqa": {"train": {"events": 650, "dialogues": 550, "social_relationship": 320, "profile": 180},
                "dev": {"events": 60, "dialogues": 50, "social_relationship": 25, "profile": 15},
                "holdout": {"events": 60, "dialogues": 50, "social_relationship": 25, "profile": 15}},
    "locomo": {"train": {"temporal": 150, "multi_hop": 110, "adversarial_unanswerable": 100, "single_hop": 90},
               "dev": {"temporal": 12, "multi_hop": 10, "adversarial_unanswerable": 10, "single_hop": 8},
               "holdout": {"temporal": 20, "multi_hop": 15, "adversarial_unanswerable": 15, "single_hop": 10}},
    "hotpotqa": {"train": 300, "dev": 30, "holdout": 30},
    "qasper": {"train": 150, "dev": 25, "holdout": 25},
    "d4": {"train": 300, "dev": 30, "holdout": 30},
}
PILOT = {"perltqa": 650, "d4": 180, "locomo": 170, "hotpotqa": 120, "qasper": 80}
PILOT_UA = {"locomo": 65, "d4": 40, "qasper": 25, "perltqa": 75}          # target abstentions inside each source quota
PILOT_DEV = {"perltqa": 50, "locomo": 25, "d4": 15, "hotpotqa": 15, "qasper": 15}
SELECT_SUBSET = 48
UA_CANDIDATES = 100
BUCKET_W = {"perltqa": [("short", .12), ("medium", .63), ("long", .25)], "locomo": [("medium", .6), ("long", .4)]}
SYS_ROOM = 640        # system prompt + layout + question line, subtracted from the target input length


def pick(rows, n, key, strat=None):
    rows = sorted(rows, key=lambda c: K.frac(key + c["id"]))
    if strat is None:
        return rows[:n]
    out = []
    for k, q in strat.items():
        out += [c for c in rows if c["qtype"] == k][:q]
    return out


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else None


def dist(xs):
    return {"n": len(xs), "sum": sum(xs), "p10": q(xs, .1), "p50": q(xs, .5), "p90": q(xs, .9), "p95": q(xs, .95), "max": max(xs) if xs else None,
            "mean": round(statistics.mean(xs), 1) if xs else None}


# ------------------------------------------------------------------ assemble candidates per source
def build_candidates(a, tok):
    raw = Path(a.raw)
    cands, stats, ua_log = [], {}, []

    # PerLTQA
    P = S.perltqa(raw, tok)
    stats["perltqa"] = {"parse": P["stats"], "split_persons": {k: len(v) for k, v in P["split_persons"].items()}}
    sup_drop = collections.Counter()
    by_split = collections.defaultdict(list)
    for c in P["rows"]:
        s = S.perltqa_support(c)
        if not s["supported"]:
            sup_drop[c["qtype"]] += 1
            continue
        c["_support"] = s
        by_split[c["split"]].append(c)
    stats["perltqa"]["drop_unsupported_by_gold_memory"] = dict(sup_drop)
    for sp in ("train", "dev", "holdout"):
        chosen = pick(by_split[sp], 0, "perltqa:" + sp, QUOTA["perltqa"][sp])
        for c in chosen:
            b = S.bucket_of(c["id"], BUCKET_W["perltqa"])
            for bb in [b] + [x for x in ("medium", "long") if x != b and K.BUCKETS[x][1] > K.BUCKETS[b][1]]:
                units = S.choose_units(c["_units"], c["gold_uids"], S.target_input(c["id"], bb) - SYS_ROOM, c["id"])
                if units:
                    break
            if not units:
                stats["perltqa"].setdefault("drop_gold_too_long", 0)
                stats["perltqa"]["drop_gold_too_long"] += 1
                continue
            f = S.perltqa_finish(c, units, 0)
            cands.append({**{k: v for k, v in c.items() if not k.startswith("_")}, **f, "answerable": True, "ua_kind": None,
                          "quotes": c["_support"]["quotes"], "support": {k: c["_support"][k] for k in ("anchor_hits", "answer_bigram_coverage")},
                          "_units_ref": c})
    # G2: evidence-removal unanswerables (train: UA_CANDIDATES; dev: 30 for the dev set) - verified deterministically
    for sp, n in (("train", UA_CANDIDATES), ("dev", 30)):
        pool = [c for c in cands if c["source"] == "perltqa" and c["split"] == sp]
        strat = {"events": int(n * .4), "dialogues": int(n * .3), "social_relationship": int(n * .15), "profile": n - int(n * .4) - int(n * .3) - int(n * .15)}
        for c in pick(pool, 0, "perltqa-ua:" + sp, strat):
            ref = c["_units_ref"]
            by = {u["uid"]: u for u in ref["_units"]}
            excl = set(ref["gold_uids"]) | set().union(*[by[g]["links"] for g in ref["gold_uids"] if g in by])
            b = S.bucket_of("ua" + c["id"], BUCKET_W["perltqa"])
            units = S.choose_units(ref["_units"], ref["gold_uids"], S.target_input("ua" + c["id"], b) - SYS_ROOM, "ua" + c["id"], exclude=excl)
            v = S.verify_unanswerable(ref, units or [])
            ua_log.append({"split": sp, "source_id": c["id"], "qtype": c["qtype"], "removed_units": len(excl), **v})
            if v["status"] == "VERIFIED" and units:
                f = S.perltqa_finish(ref, units, 0, unanswerable=True)
                cid = c["id"] + ":ua"
                cands.append({**{k: v2 for k, v2 in c.items() if not k.startswith("_") and k not in ("pack", "gold_eids", "quotes")}, **f, "id": cid,
                              "answerable": False, "ua_kind": "PERLTQA_EVIDENCE_REMOVED_VERIFIED", "pair_id": c["id"], "answer": None, "quotes": [],
                              "verification": v})
                c["pair_id"] = c["id"]

    # LoCoMo
    L = S.locomo(raw, tok)
    stats["locomo"] = {"parse": L["stats"], "split_conversations": L["split_conversations"]}
    drop = collections.Counter()
    for sp in ("train", "dev", "holdout"):
        rows = [c for c in L["rows"] if c["split"] == sp]
        # single-hop / multi-hop answers must be supported by the gold turns (token overlap); temporal needs a dated gold session
        keep = []
        for c in rows:
            ses = {se["sid"]: se for se in c["_sessions"]}
            gtext = " ".join(t["text"] for g in c["gold_sessions"] for t in ses[g]["turns"] if t["dia"] in c["evidence_dia"])
            if c["qtype"] in ("single_hop", "multi_hop"):
                toks = [t for t in K.norm(c["answer"]).split() if t] or [c["answer"]]
                import re as _re
                words = _re.findall(r"[a-z0-9]+", str(c["answer"]).lower())
                hit = sum(1 for w in words if w in gtext.lower()) / max(1, len(words))
                if hit < 0.5:
                    drop["answer_not_supported_by_gold_turns"] += 1
                    continue
            if c["qtype"] == "temporal" and not all(ses[g]["time"] for g in c["gold_sessions"]):
                drop["temporal_without_session_date"] += 1
                continue
            keep.append(c)
        for c in pick(keep, 0, "locomo:" + sp, QUOTA["locomo"][sp]):
            b = S.bucket_of(c["id"], BUCKET_W["locomo"])
            res = None
            for bb in [b, "long"]:
                res = S.locomo_pack(c, S.target_input(c["id"], bb) - SYS_ROOM)
                if res:
                    break
            if not res:
                res = S.locomo_pack(c, K.CUTOFF - SYS_ROOM - 300)
            if not res:
                drop["gold_sessions_exceed_cutoff"] += 1
                continue
            cands.append({**{k: v for k, v in c.items() if not k.startswith("_")}, **res, "ua_kind": None if c["answerable"] else "LOCOMO_ADVERSARIAL"})
    stats["locomo"]["drops"] = dict(drop)

    # HotpotQA
    H = S.hotpot(S.hotpot_rows(raw, Path(a.full) / "_cache" / "hotpot_extra_rowgroups.jsonl"))
    stats["hotpotqa"] = {"parse": H["stats"]}
    for sp in ("train", "dev", "holdout"):
        rows = [c for c in H["rows"] if c["split"] == sp]
        yn = [c for c in rows if c["qtype"].endswith("yes_no")]
        other = [c for c in rows if not c["qtype"].endswith("yes_no")]
        n = QUOTA["hotpotqa"][sp]
        chosen = pick(yn, max(1, n // 20), "hotpot-yn:" + sp) + pick(other, n - max(1, n // 20), "hotpot:" + sp)
        for c in chosen:
            cands.append({**c, "ua_kind": None})

    # QASPER
    Q = S.qasper(raw, tok)
    stats["qasper"] = {"parse": Q["stats"]}
    qd = collections.Counter()
    for sp in ("train", "dev", "holdout"):
        rows = [c for c in Q["rows"] if c["split"] == sp]
        n = QUOTA["qasper"][sp]
        ua = pick([c for c in rows if not c["answerable"]], int(n * 0.27), "qasper-ua:" + sp)
        yn = pick([c for c in rows if c["qtype"] == "yes_no"], int(n * 0.08), "qasper-yn:" + sp)
        rest = pick([c for c in rows if c["answerable"] and c["qtype"] != "yes_no"], n - len(ua) - len(yn), "qasper:" + sp)
        for c in ua + yn + rest:
            res = S.qasper_pack(c, K.CUTOFF - SYS_ROOM - 320)
            if not res:
                qd["gold_paragraphs_exceed_cutoff"] += 1
                continue
            cands.append({**{k: v for k, v in c.items() if not k.startswith("_")}, **res, "ua_kind": None if c["answerable"] else "QASPER_UNANSWERABLE"})
    stats["qasper"]["drops"] = dict(qd)

    # D4 synthetic
    D = S.d4(Path(a.d4), Path(a.d4_labels))
    stats["d4"] = {"parse": D["stats"], "personas": D["personas"]}
    for sp in ("train", "dev", "holdout"):
        rows = [c for c in D["rows"] if c["split"] == sp]
        n = QUOTA["d4"][sp]
        ua = pick([c for c in rows if c["qtype"] == "FULL_UNKNOWN"], int(n * 0.2), "d4-ua:" + sp)
        cats = [x for x in S.D4_CATS if x != "FULL_UNKNOWN"]
        per = (n - len(ua)) // len(cats)
        rest = []
        for cat in cats:
            rest += pick([c for c in rows if c["qtype"] == cat], per, f"d4:{sp}:{cat}")
        for c in ua + rest:
            cands.append({**c, "ua_kind": None if c["answerable"] else "D4_FULL_UNKNOWN"})
    return cands, stats, ua_log


# ------------------------------------------------------------------ render, tokenizer audit, validation
def render_and_audit(cands, tok, M):
    from tokenizers import Tokenizer  # noqa: F401  (tok wraps it)
    im_end = tok.t.token_to_id("<|im_end|>")
    rows, rejects = [], collections.Counter()
    for c in cands:
        pack = c["pack"]
        msgs = K.messages(pack, c["header"])
        user = msgs[1]["content"]
        eids = [i.eid for i in pack.items]
        target = K.target_json(c["id"], c["answer"], c["gold_eids"], c.get("quotes") or [], c["answerable"])
        t = json.loads(target)
        problems = []
        if c["answerable"] and not c["gold_eids"]:
            problems.append("answerable_without_gold_evidence")
        if any(e not in eids or f"[{e}]" not in user for e in t["evidence"]):
            problems.append("evidence_id_not_in_prompt")
        if any(qq not in user for qq in t["quotes"]):
            problems.append("quote_not_verbatim_in_prompt")
        if target in user or '"answer"' in user:
            problems.append("gold_in_prompt")
        if K.repetitive(t["answer"], K.lang_of(c["source"]), M):
            problems.append("repetitive_target")
        pre = K.PRE.format(s=msgs[0]["content"], u=user)
        p_ids = tok.t.encode(pre, add_special_tokens=False).ids
        s_ids = tok.t.encode(target + K.SUP_SUFFIX, add_special_tokens=False).ids
        j_ids = tok.t.encode(pre + target + K.SUP_SUFFIX, add_special_tokens=False).ids
        mask_ok = j_ids[len(p_ids):] == s_ids and j_ids[:len(p_ids)] == p_ids
        eos_ok = len(s_ids) >= 2 and s_ids[-2] == im_end
        full = len(p_ids) + len(s_ids)
        if not mask_ok:
            problems.append("mask_boundary")
        if not eos_ok:
            problems.append("eos")
        if full > K.CUTOFF:
            problems.append("over_cutoff")
        if len(s_ids) - 2 > 600:
            problems.append("target_too_long")
        if problems:
            for p in problems:
                rejects[f"{c['source']}:{p}"] += 1
            continue
        bucket = "short" if len(p_ids) < 1500 else "medium" if len(p_ids) < 4000 else "long"
        rows.append({"c": c, "msgs": msgs, "target": target, "input_tokens": len(p_ids), "supervised_tokens": len(s_ids), "full_tokens": full,
                     "target_answer_tokens": len(tok.t.encode(t["answer"], add_special_tokens=False).ids), "bucket": bucket,
                     "fits_4096": full <= K.CUTOFF_FALLBACK})
    return rows, dict(rejects)


def minhash_sig(text, perms=32):
    t = K.norm(text)
    g = {t[i:i + 3] for i in range(max(1, len(t) - 2))}
    hs = [int(hashlib.blake2b(x.encode(), digest_size=8).hexdigest(), 16) for x in g]
    rnd = random.Random(11)
    salts = [rnd.getrandbits(32) for _ in range(perms)]
    return tuple(min((x ^ s) for x in hs) for s in salts), g


def quality_gates(rows):
    """Exact / normalised duplicates, duplicate questions, answer concentration, near-duplicate questions across splits."""
    out, log = [], collections.Counter()
    seen_row, seen_q = set(), {}
    ans_count = collections.Counter()
    for r in sorted(rows, key=lambda r: (r["c"]["split"] != "train", K.frac("qg:" + r["c"]["id"]))):
        c = r["c"]
        key = K.h(r["msgs"][1]["content"] + r["target"], 24)
        if key in seen_row:
            log["exact_duplicate_row"] += 1
            continue
        qk = (c["source"], K.norm(c["question"]), c["answerable"])
        if qk in seen_q and c.get("pair_id") is None:
            log["duplicate_question_same_answerability"] += 1
            continue
        a = json.loads(r["target"])["answer"]
        if c["answerable"]:
            ak = (c["source"], K.norm(a))
            if ans_count[ak] >= 3 and len(K.norm(a)) > 0:
                log["answer_concentration_cap3"] += 1
                continue
            ans_count[ak] += 1
        seen_row.add(key)
        seen_q[qk] = c["split"]
        out.append(r)
    # near-duplicate questions across splits (train vs dev/holdout) within a source: drop the TRAIN copy
    sig = {}
    bands = collections.defaultdict(list)
    for i, r in enumerate(out):
        s, g = minhash_sig(r["c"]["question"])
        sig[i] = g
        for b in range(8):
            bands[(r["c"]["source"], b, s[b * 4:(b + 1) * 4])].append(i)
    drop = set()
    for ids in bands.values():
        if len(ids) < 2:
            continue
        for x in ids:
            for y in ids:
                if x < y and out[x]["c"]["split"] != out[y]["c"]["split"]:
                    j = len(sig[x] & sig[y]) / max(1, len(sig[x] | sig[y]))
                    if j >= 0.8:
                        tr = x if out[x]["c"]["split"] == "train" else y if out[y]["c"]["split"] == "train" else None
                        if tr is not None:
                            drop.add(tr)
    log["near_duplicate_question_across_splits_train_dropped"] = len(drop)
    return [r for i, r in enumerate(out) if i not in drop], dict(log)


def lifebench_overlap(rows, lb_path):
    """40-char normalised substrings shared with the LifeBench conversation corpus (the qa block is never read)."""
    lb = json.load(open(lb_path, encoding="utf-8"))
    S40 = set()
    for s in lb:
        for k, v in (s.get("conversation") or {}).items():
            if isinstance(v, list):
                for d in v:
                    t = K.norm(d.get("text", ""))
                    for i in range(0, max(0, len(t) - 39), 5):
                        S40.add(t[i:i + 40])
    hits = []
    for r in rows:
        t = K.norm(r["msgs"][1]["content"] + r["target"])
        if any(t[i:i + 40] in S40 for i in range(0, max(0, len(t) - 39))):
            hits.append(r["c"]["id"])
    return set(hits), len(S40)


# ------------------------------------------------------------------ pilot selection
def select_pilot(rows):
    tr = [r for r in rows if r["c"]["split"] == "train"]
    chosen = []
    for src, n in PILOT.items():
        pool = [r for r in tr if r["c"]["source"] == src]
        ua = sorted([r for r in pool if not r["c"]["answerable"]], key=lambda r: K.frac("pilot-ua:" + r["c"]["id"]))[:PILOT_UA.get(src, 0)]
        # PerLTQA unanswerables come with their answerable sibling (pairs)
        pair_ids = {r["c"]["pair_id"] for r in ua if r["c"].get("pair_id")}
        sib = [r for r in pool if r["c"]["answerable"] and r["c"]["id"] in pair_ids]
        rest = sorted([r for r in pool if r["c"]["answerable"] and r["c"]["id"] not in pair_ids], key=lambda r: K.frac("pilot:" + r["c"]["id"]))
        # spread over question types
        by = collections.defaultdict(list)
        for r in rest:
            by[r["c"]["qtype"]].append(r)
        need = n - len(ua) - len(sib)
        picked = []
        while len(picked) < need and any(by.values()):
            for k in sorted(by):
                if by[k] and len(picked) < need:
                    picked.append(by[k].pop(0))
        chosen += ua + sib + picked
    # yes / no cap 5 % of the pilot
    yn = [r for r in chosen if K.norm(json.loads(r["target"])["answer"]) in ("yes", "no")]
    cap = int(0.05 * len(chosen))
    drop = {id(r) for r in yn[cap:]}
    chosen = [r for r in chosen if id(r) not in drop]
    dev = [r for r in rows if r["c"]["split"] == "dev"]
    dsel = []
    for src, n in PILOT_DEV.items():
        pool = sorted([r for r in dev if r["c"]["source"] == src], key=lambda r: K.frac("pdev:" + r["c"]["id"]))
        ua = [r for r in pool if not r["c"]["answerable"]][: max(2, n // 5)]
        an = [r for r in pool if r["c"]["answerable"] and r not in ua][: n - len(ua)]
        dsel += ua + an
    sel = sorted(dsel, key=lambda r: K.frac("select:" + r["c"]["id"]))
    subset = []
    per_src = collections.Counter()
    for r in sel:
        if len(subset) >= SELECT_SUBSET:
            break
        if per_src[r["c"]["source"]] < SELECT_SUBSET * PILOT_DEV[r["c"]["source"]] // sum(PILOT_DEV.values()) + 1:
            subset.append(r["c"]["id"])
            per_src[r["c"]["source"]] += 1
    return chosen, dsel, subset, len(yn) - min(len(yn), cap)


# ------------------------------------------------------------------ outputs
def summarize(rows, name):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["c"]["source"]].append(r)
    tot_sup = sum(r["supervised_tokens"] for r in rows)
    res = {"rows": len(rows), "unanswerable": sum(not r["c"]["answerable"] for r in rows),
           "unanswerable_rate": round(sum(not r["c"]["answerable"] for r in rows) / max(1, len(rows)), 4),
           "input_tokens": dist([r["input_tokens"] for r in rows]), "supervised_tokens": dist([r["supervised_tokens"] for r in rows]),
           "full_tokens": dist([r["full_tokens"] for r in rows]),
           "buckets": dict(collections.Counter(r["bucket"] for r in rows)), "rows_over_4096": sum(not r["fits_4096"] for r in rows),
           "language": dict(collections.Counter("zh" if r["c"]["source"] in ("perltqa", "d4") else "en" for r in rows)),
           "abstain_variants": dict(collections.Counter(json.loads(r["target"])["answer"] for r in rows if not r["c"]["answerable"])),
           "yes_no": sum(1 for r in rows if K.norm(json.loads(r["target"])["answer"]) in ("yes", "no")),
           "answer_le3_tokens": sum(1 for r in rows if r["c"]["answerable"] and r["target_answer_tokens"] <= 3),
           "by_source": {}}
    for s, rr in sorted(by.items()):
        sup = sum(r["supervised_tokens"] for r in rr)
        res["by_source"][s] = {"rows": len(rr), "unanswerable": sum(not r["c"]["answerable"] for r in rr), "qtypes": dict(collections.Counter(r["c"]["qtype"] for r in rr)),
                               "input_tokens_p50": q([r["input_tokens"] for r in rr], .5), "input_tokens_p95": q([r["input_tokens"] for r in rr], .95),
                               "supervised_tokens": sup, "supervised_token_share": round(sup / max(1, tot_sup), 4),
                               "input_token_share": round(sum(r["input_tokens"] for r in rr) / max(1, sum(x["input_tokens"] for x in rows)), 4),
                               "buckets": dict(collections.Counter(r["bucket"] for r in rr)), "groups": len({r["c"]["group"] for r in rr})}
    return res


def meta_row(r):
    c = r["c"]
    return {"id": c["id"], "source": c["source"], "split": c["split"], "group": c["group"], "qtype": c["qtype"], "answerable": c["answerable"],
            "ua_kind": c.get("ua_kind"), "pair_id": c.get("pair_id"), "header": c["header"], "gold_eids": c["gold_eids"], "answer": c["answer"],
            "input_tokens": r["input_tokens"], "supervised_tokens": r["supervised_tokens"], "bucket": r["bucket"], "provenance": c["provenance"],
            "support": c.get("support"), "verification": c.get("verification")}


def write_jsonl(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def lf_row(r):
    return K.sharegpt(r["c"]["id"], r["msgs"], r["target"], source=r["c"]["source"])


DATASET_INFO = {name: {"file_name": f, "formatting": "sharegpt", "columns": {"messages": "conversations", "system": "system"},
                       "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"}}
                for name, f in (("d5_pilot_train", "pilot_train.jsonl"), ("d5_pilot_dev", "pilot_dev.jsonl"),
                                ("d5_pilot_train_c4096", "pilot_train_le4096.jsonl"), ("d5_pilot_dev_c4096", "pilot_dev_le4096.jsonl"),
                                ("d5_smoke", "pilot_smoke.jsonl"), ("d5_smoke_c4096", "pilot_smoke_le4096.jsonl"))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    for k in ("raw", "tokenizer", "d4", "d4-labels", "t2d4", "lifebench", "full", "pilot"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, a.t2d4)
    from t2d4.colab import metrics as M           # the D4 repetition detector
    tok = S.Tok(Path(a.tokenizer))
    full, pilot = Path(a.full), Path(a.pilot)
    cands, stats, ua_log = build_candidates(a, tok)
    rows, rejects = render_and_audit(cands, tok, M)
    rows, qlog = quality_gates(rows)
    hits, nlb = lifebench_overlap(rows, a.lifebench)
    rows = [r for r in rows if r["c"]["id"] not in hits]
    # unanswerables whose answerable sibling did not survive keep their verification but lose the pair link
    alive = {r["c"]["id"] for r in rows}
    for r in rows:
        if r["c"].get("pair_id") and r["c"]["pair_id"] not in alive:
            r["c"]["pair_id"] = None
    splits = {s: [r for r in rows if r["c"]["split"] == s] for s in ("train", "dev", "holdout")}
    # split integrity
    groups = {s: {r["c"]["group"] for r in v} for s, v in splits.items()}
    leak = {f"{x}&{y}": len(groups[x] & groups[y]) for x, y in (("train", "dev"), ("train", "holdout"), ("dev", "holdout"))}
    hot_titles = {s: {t for r in v if r["c"]["source"] == "hotpotqa" for t in r["c"].get("titles", [])} for s, v in splits.items()}
    pilot_rows, pdev, subset, yn_dropped = select_pilot(rows)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    from . import D5_VERSION
    # ---- full pool
    for s, name in (("train", "train"), ("dev", "dev"), ("holdout", "internal_holdout")):
        write_jsonl(full / f"{name}.jsonl", [lf_row(r) for r in splits[s]])
        write_jsonl(full / f"{name}_meta.jsonl", [meta_row(r) for r in splits[s]])
    ua_counts = collections.Counter((x["split"], x["status"]) for x in ua_log)
    ua_train = [x for x in ua_log if x["split"] == "train"]
    ua_summary = {"CANDIDATES": len(ua_train), "VERIFIED": sum(x["status"] == "VERIFIED" for x in ua_train),
                  "REJECTED": sum(x["status"] == "REJECTED" for x in ua_train), "AMBIGUOUS": sum(x["status"] == "AMBIGUOUS" for x in ua_train),
                  "ACCEPTANCE_RATE": round(sum(x["status"] == "VERIFIED" for x in ua_train) / max(1, len(ua_train)), 3),
                  "dev": {k[1]: v for k, v in ua_counts.items() if k[0] == "dev"},
                  "by_qtype": {t: dict(collections.Counter(x["status"] for x in ua_train if x["qtype"] == t)) for t in sorted({x["qtype"] for x in ua_train})},
                  "rejection_reasons": dict(collections.Counter(r.split(" (")[0].split(":")[0] for x in ua_train for r in x["reasons"] if x["status"] != "VERIFIED"))}
    (full / "UNANSWERABLE_VALIDATION.json").write_text(json.dumps({"summary": ua_summary, "rows": ua_log}, ensure_ascii=False, indent=1), encoding="utf-8")
    fman = {"schema": "stillhere.d5.full_dataset_manifest/1.0", "version": D5_VERSION + "/full_v1", "created_utc": now, "seed": K.SEED,
            "usage_scope": "PRIVATE_NONCOMMERCIAL_RESEARCH", "fact_system_sha256": K.fact_system_sha(), "reader_version": "stillhere.reader.fact/1.1",
            "evidence_headers": {"memory": K.HEADER_MEMORY, "document": K.HEADER_DOC}, "cutoff": K.CUTOFF, "abstain_variants": list(K.ABSTAIN_VARIANTS),
            "splits": {s: summarize(v, s) for s, v in splits.items()}, "source_stats": stats, "render_rejects": rejects, "quality_gates": qlog,
            "lifebench_overlap": {"rows_dropped": len(hits), "corpus_40char_shingles": nlb}, "group_overlap_between_splits": leak,
            "hotpot_title_overlap": {"train&dev": len(hot_titles["train"] & hot_titles["dev"]), "train&holdout": len(hot_titles["train"] & hot_titles["holdout"])},
            "unanswerable_validation": ua_summary,
            "files": {}}
    for f in ("train.jsonl", "dev.jsonl", "internal_holdout.jsonl", "train_meta.jsonl", "dev_meta.jsonl", "internal_holdout_meta.jsonl", "UNANSWERABLE_VALIDATION.json"):
        fman["files"][f] = {"sha256": K.sha256_file(full / f), "bytes": (full / f).stat().st_size}
    (full / "D5_FULL_DATASET_MANIFEST.json").write_text(json.dumps(fman, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    # ---- pilot
    pilot.mkdir(parents=True, exist_ok=True)
    tr_rows = [lf_row(r) for r in pilot_rows]
    dv_rows = [lf_row(r) for r in pdev]
    write_jsonl(pilot / "train" / "pilot_train.jsonl", tr_rows)
    write_jsonl(pilot / "train" / "pilot_dev.jsonl", dv_rows)
    write_jsonl(pilot / "train" / "pilot_train_le4096.jsonl", [lf_row(r) for r in pilot_rows if r["fits_4096"]])
    write_jsonl(pilot / "train" / "pilot_dev_le4096.jsonl", [lf_row(r) for r in pdev if r["fits_4096"]])
    longest = sorted(pilot_rows, key=lambda r: -r["full_tokens"])[:16]
    write_jsonl(pilot / "train" / "pilot_smoke.jsonl", [lf_row(r) for r in longest])
    longest4 = sorted([r for r in pilot_rows if r["fits_4096"]], key=lambda r: -r["full_tokens"])[:16]
    write_jsonl(pilot / "train" / "pilot_smoke_le4096.jsonl", [lf_row(r) for r in longest4])
    (pilot / "train" / "dataset_info.json").write_text(json.dumps(DATASET_INFO, ensure_ascii=False, indent=1), encoding="utf-8")
    # evaluation inputs (no answers) and the scorer's gold (dev split only)
    write_jsonl(pilot / "eval" / "pilot_dev_eval.jsonl", [{"question_id": r["c"]["id"], "source": r["c"]["source"], "qtype": r["c"]["qtype"],
                                                          "header": r["c"]["header"], "pack": r["c"]["pack"].to_json(), "messages": r["msgs"],
                                                          "input_tokens": r["input_tokens"], "prompt_hash": K.h(json.dumps(r["msgs"], ensure_ascii=False), 32)}
                                                         for r in pdev])
    write_jsonl(pilot / "eval" / "pilot_dev_gold.jsonl", [{"question_id": r["c"]["id"], "source": r["c"]["source"], "qtype": r["c"]["qtype"],
                                                          "answerable": r["c"]["answerable"], "answer": r["c"]["answer"], "gold_eids": r["c"]["gold_eids"],
                                                          "language": "zh" if r["c"]["source"] in ("perltqa", "d4") else "en"} for r in pdev])
    (pilot / "eval" / "selection_subset.json").write_text(json.dumps({"rule": "pre-registered: fixed 48-question subset of pilot_dev for checkpoint selection",
                                                                      "question_ids": subset}, ensure_ascii=False, indent=1), encoding="utf-8")
    write_jsonl(pilot / "pilot_train_meta.jsonl", [meta_row(r) for r in pilot_rows])
    rows_n = len(tr_rows)
    pman = {"schema": "stillhere.d5.pilot_dataset_manifest/1.0", "version": D5_VERSION, "created_utc": now, "seed": K.SEED,
            "usage_scope": "PRIVATE_NONCOMMERCIAL_RESEARCH", "status": "D5_PILOT_CANDIDATE data", "fact_system_sha256": K.fact_system_sha(),
            "reader_version": "stillhere.reader.fact/1.1", "evidence_headers": {"memory": K.HEADER_MEMORY, "document": K.HEADER_DOC},
            "cutoff": K.CUTOFF, "train": summarize(pilot_rows, "pilot_train"), "dev": summarize(pdev, "pilot_dev"),
            "train_rows_le4096": sum(r["fits_4096"] for r in pilot_rows), "dev_rows_le4096": sum(r["fits_4096"] for r in pdev),
            "yes_no_dropped_by_cap": yn_dropped, "selection_subset": len(subset),
            "expected_steps_bs1_ga8": -(-rows_n // 8), "full_pool_manifest_sha256": K.sha256_file(full / "D5_FULL_DATASET_MANIFEST.json"),
            "gates": {"loss_mask": "PASS (every row: joint == separate tokenisation; prompt fully masked by layout)", "eos": "PASS",
                      "truncation": f"0 rows over {K.CUTOFF} (rows over the cutoff were dropped, never cut)",
                      "gold_in_prompt": "0", "evidence_ids_exist": "PASS", "quotes_verbatim": "PASS", "repetitive_targets": "0 (D4 detector)"}}
    (pilot / "D5_PILOT_DATASET_MANIFEST.json").write_text(json.dumps(pman, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    files = {}
    for p in sorted(pilot.rglob("*")):
        if p.is_file() and p.name not in ("DATA_MANIFEST.json", "pilot_train_meta.jsonl"):
            files[p.relative_to(pilot).as_posix()] = K.sha256_file(p)
    (pilot / "DATA_MANIFEST.json").write_text(json.dumps({"version": D5_VERSION, "data_origin": "d5_pilot", "contains_internal_holdout": False,
                                                          "contains_full_pool": False, "files": files}, indent=1), encoding="utf-8")
    print(json.dumps({"full": {s: fman["splits"][s]["rows"] for s in fman["splits"]}, "pilot_train": rows_n, "pilot_dev": len(dv_rows),
                      "ua": ua_summary, "rejects": rejects, "quality": qlog, "lifebench_hits": len(hits), "leak": leak,
                      "pilot_ua_rate": pman["train"]["unanswerable_rate"], "pilot_buckets": pman["train"]["buckets"],
                      "pilot_input": pman["train"]["input_tokens"], "pilot_sup": pman["train"]["supervised_tokens"]["sum"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
