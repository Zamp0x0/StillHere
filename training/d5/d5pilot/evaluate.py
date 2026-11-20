"""D5 evaluation: Base (B) vs D5 (E) under identical conditions; checkpoint selection; local scoring; paired comparison.

    python -m d5pilot.evaluate generate --base-dir B --adapter E=<adapter dir> --set d5dev|select|p3dev200 --groups B,E --run-dir R \
        [--pilot-data P] [--p3-data <extracted ef1_phase3_data>]
    python -m d5pilot.evaluate score    --run-dir R --set d5dev --pilot-data P --groups B,E
    python -m d5pilot.evaluate select   --run-dir R --pilot-data P --candidates ckpt-75,ckpt-125,ckpt-150
    python -m d5pilot.evaluate compare  --run-dir R --p3-data D --data <LifeBench json> --groups B,E      (after the optional judge)

Fairness: ONE NF4 base; B = every adapter disabled (PEFT ``disable_adapter``), E / ckpt-N = that adapter enabled. Same frozen messages,
same reader 1.1 FACT generation config, same per-prompt seed (identical to the Phase 3 B code path), same guard (annotate). Only the
adapter differs. Phase 3 dev200 prompts are the FROZEN Phase 3 prompts (no retrieval here, no gold read while generating).
Every finished question is appended (fsync) to ``<run-dir>/eval/<set>/<group>/answers.jsonl``; a resumed run skips OK rows with the same
prompt hash; a different fingerprint is refused.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from . import common as K

_ABSTAIN_RE = re.compile(r"无法确定|不知道|没有记录|记不清|不清楚|没有相关|无从得知|没有提到|没印象")
CJK = re.compile(r"[一-鿿]")
SELECTION_RULE = ("pre-registered: among the candidate checkpoints, drop any with JSON-valid rate < 0.95 or repetition rate > Base + 0.01 on the "
                  "48-question selection subset; pick the highest local correct rate; ties -> the later checkpoint; if none qualifies -> "
                  "NO_QUALIFIED_CHECKPOINT (report, do not train more). Train / eval loss never selects.")


def P3():
    """Phase 3 helpers (evaluation/lifebench/phase3) - imported from the code package."""
    from phase3 import analyze as A
    from phase3 import common as C
    from phase3 import runner as R
    return A, C, R


# ------------------------------------------------------------------ generator (same semantics as stillhere_lifebench.backends.D4Generator)
class Gen:
    label = "QWEN3_8B_NF4_D5_REAL_GPU"

    def __init__(self, base_dir: Path, adapters: Mapping[str, Path]):
        from t2d4.colab.backend_hf import HFBackend
        self.b = HFBackend(Path(base_dir))
        self.info = {name: self.b.add_adapter(name, Path(d)) for name, d in adapters.items()}
        self.torch = self.b.torch

    def count(self, messages, thinking=False) -> int:
        return self.b.prompt_length(messages)

    def __call__(self, messages, cfg, adapter: Optional[str]) -> dict:
        from t2d4.colab.backend_hf import STOP_IDS, batch_seed
        b, torch = self.b, self.torch
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        enc = b.tok([b.render(messages)], return_tensors="pt", padding=True, add_special_tokens=False).to(0)
        mx = int(cfg["max_new_tokens"])
        sample = bool(cfg.get("do_sample", True))
        kw = dict(max_new_tokens=mx, eos_token_id=list(STOP_IDS), pad_token_id=b.tok.pad_token_id, do_sample=sample,
                  repetition_penalty=float(cfg.get("repetition_penalty", 1.0)))
        if sample:
            kw.update(temperature=float(cfg["temperature"]), top_p=float(cfg["top_p"]), top_k=int(cfg["top_k"]), min_p=float(cfg.get("min_p", 0.0)))
        seed_key = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        torch.manual_seed(batch_seed(int(cfg.get("seed", 20261112)), seed_key))
        with torch.inference_mode():
            if adapter is None:
                if self.info:
                    with b.model.disable_adapter():
                        out = b.model.generate(**enc, **kw)
                else:
                    out = b.model.generate(**enc, **kw)
            else:
                b.model.set_adapter(adapter)
                out = b.model.generate(**enc, **kw)
        n = enc["input_ids"].shape[1]
        new = [int(t) for t in out[0][n:].tolist()]
        stop = next((i for i, t in enumerate(new) if t in STOP_IDS), None)
        body = new[:stop] if stop is not None else new
        dt = time.perf_counter() - t0
        return {"text": b.tok.decode(body, skip_special_tokens=False).strip(), "hit_token_limit": stop is None and len(body) >= mx,
                "new_tokens": len(body), "prompt_tokens": int(n), "latency_s": round(dt, 3),
                "tokens_per_s": round(len(body) / dt, 2) if dt > 0 else None, "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)}

    def is_oom(self, e) -> bool:
        return isinstance(e, self.torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()

    def after_error(self):
        import gc
        gc.collect()
        self.torch.cuda.empty_cache()


# ------------------------------------------------------------------ evaluation sets (no gold)
def load_set(name: str, pilot: Optional[Path], p3: Optional[Path]) -> Dict[str, dict]:
    from stillhere.context.fact_pack import FactPack
    rows = {}
    if name in ("d5dev", "select"):
        keep = None
        if name == "select":
            keep = set(json.loads((pilot / "eval" / "selection_subset.json").read_text(encoding="utf-8"))["question_ids"])
        for l in (pilot / "eval" / "pilot_dev_eval.jsonl").read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                if keep is None or r["question_id"] in keep:
                    rows[r["question_id"]] = {"messages": r["messages"], "pack": FactPack.from_json(r["pack"]), "prompt_hash": r["prompt_hash"],
                                              "source": r["source"], "qtype": r["qtype"]}
    elif name == "p3dev200":
        A, C, R = P3()
        frozen = C.load_frozen(p3 / "ef1_frozen.jsonl")
        samples = json.loads((p3 / "samples.json").read_text(encoding="utf-8"))
        for q in samples["dev200"]["question_ids"]:
            fr = frozen[q]
            rows[q] = {"messages": fr["prompts"]["fact"], "pack": FactPack.from_json(fr["pack"]), "prompt_hash": fr["hashes"]["fact_prompt"],
                       "source": "lifebench_p3_dev200", "qtype": None}
    else:
        raise KeyError(name)
    return rows


def gen_config(pack) -> dict:
    from stillhere.reader import fact as rf
    return dict(rf.generation_for("FACT", pack))


def run_one(gen, q: str, item: dict, group: str) -> dict:
    from stillhere.reader import fact as rf
    A, C, R = P3()
    pack = item["pack"]
    cfg = gen_config(pack)
    adapter = None if group == "B" else group
    n_in = gen.count(item["messages"])
    budget = C.allowed_new_tokens(n_in, int(cfg["max_new_tokens"]))
    row = {"question_id": q, "group": group, "adapter": adapter, "prompt_hash": item["prompt_hash"], "config_hash": C.sha256_json(cfg),
           "generation": cfg, "input_tokens": n_in, "budget": budget, "source": item["source"], "qtype": item["qtype"]}
    if budget["context_limit"]:
        return {**row, "status": "CONTEXT_LIMIT", "final_answer": ""}
    out = gen(item["messages"], {**cfg, "max_new_tokens": budget["max_new_tokens"]}, adapter)
    row.update(status="OK", **{k: out[k] for k in ("new_tokens", "hit_token_limit", "latency_s", "tokens_per_s", "peak_vram_gib")})
    row["raw"] = out["text"]
    reading = rf.parse(row["raw"])
    fin = rf.finalize(reading, pack, guard_action="annotate")
    row["parsed"] = {"ok": reading.ok, "answer": reading.answer, "abstain": reading.abstain, "answerable": reading.answerable, "evidence": reading.evidence,
                     "quotes": reading.quotes, "error": reading.error, "answer_complete": bool(reading.ok or "salvaged (complete)" in (reading.error or ""))}
    row["repeated_line_ratio"] = R.repeated_line_ratio(row["raw"])
    row["audit"] = {k: fin[k] for k in ("action", "support", "citations")}
    row["final_answer"] = fin["answer"]
    return row


def generate(gen, items: Dict[str, dict], groups: List[str], out_dir: Path, fingerprint: dict, limit=None, log=print) -> dict:
    A, C, R = P3()
    mode = C.bind_fingerprint(out_dir, fingerprint)
    log(f"{mode} {out_dir}")
    summary = {}
    for g in groups:
        path = out_dir / g / "answers.jsonl"
        done = C.done_rows(path)
        todo = [q for q in items if (q,) not in done or done[(q,)].get("prompt_hash") != items[q]["prompt_hash"]]
        if limit:
            todo = todo[:limit]
        log(f"group {g}: {len(items) - len(todo)} done, {len(todo)} to run")
        for i, q in enumerate(todo):
            t0 = time.time()
            try:
                row = run_one(gen, q, items[q], g)
            except Exception as e:  # noqa: BLE001 - recorded, retried on the next run
                oom = gen.is_oom(e)
                gen.after_error()
                row = {"question_id": q, "group": g, "status": "OOM" if oom else "ERROR", "error": f"{type(e).__name__}: {str(e)[:300]}",
                       "trace": None if oom else traceback.format_exc()[-1200:], "prompt_hash": items[q]["prompt_hash"]}
            row.update(run_fingerprint=C.sha256_json(fingerprint), generator=gen.label, wall_s=round(time.time() - t0, 3),
                       finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
            C.append_jsonl(path, row)
            log(f"[{g}] {i + 1}/{len(todo)} {q[:40]} {row.get('status')} in={row.get('input_tokens')} out={row.get('new_tokens')} "
                f"{row.get('latency_s')}s peak={row.get('peak_vram_gib')}GiB")
        done = C.done_rows(path)
        summary[g] = {"ok": sum(1 for q in items if (q,) in done and done[(q,)].get("status") == "OK"), "of": len(items)}
    C.write_json_atomic(out_dir / "GENERATION_STATUS.json", {"summary": summary, "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return summary


# ------------------------------------------------------------------ local scoring (d5 dev; gold lives only in pilot_dev_gold.jsonl)
def _toks(s: str, lang: str):
    s = K.norm(s) if lang == "zh" else str(s or "").lower()
    return list(s) if lang == "zh" else re.findall(r"[a-z0-9]+", s)


def f1(pred: str, gold: str, lang: str) -> float:
    p, g = _toks(pred, lang), _toks(gold, lang)
    if not p or not g:
        return 0.0
    common = sum((collections.Counter(p) & collections.Counter(g)).values())
    if common == 0:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def abstained(row) -> bool:
    return bool((row.get("parsed") or {}).get("abstain")) or bool(_ABSTAIN_RE.search(row.get("final_answer") or "")) and len(row.get("final_answer") or "") < 60


def score_row(row, gold, M=None) -> dict:
    lang = gold["language"]
    ans = row.get("final_answer") or ""
    ok = row.get("status") == "OK"
    ab = abstained(row) if ok else False
    if gold["answerable"]:
        s = f1(ans, gold["answer"], lang)
        contained = K.norm(gold["answer"]) and K.norm(gold["answer"]) in K.norm(ans)
        correct = ok and not ab and (s >= 0.5 or bool(contained))
    else:
        s, correct = None, ok and ab
    cited = set((row.get("parsed") or {}).get("evidence") or [])
    g = set(gold["gold_eids"])
    cit = (row.get("audit") or {}).get("citations") or {}
    return {"question_id": row["question_id"], "status": row.get("status"), "answerable": gold["answerable"], "source": gold["source"], "qtype": gold["qtype"],
            "correct": bool(correct), "f1": None if s is None else round(s, 3), "abstained": ab,
            "hallucinated_unanswerable": (not gold["answerable"]) and ok and not ab, "wrong_abstention": gold["answerable"] and ab,
            "evidence_precision": (len(cited & g) / len(cited)) if (cited and g) else None, "evidence_recall": (len(cited & g) / len(g)) if g else None,
            "cites_unknown_id": bool(cit.get("unknown_ids")), "quotes_not_in_evidence": bool(cit.get("quotes_not_in_evidence")),
            "json_valid": bool((row.get("parsed") or {}).get("answer_complete")), "repetitive": K.repetitive(row.get("raw") or "", lang, M),
            "repeated_line_ratio": row.get("repeated_line_ratio"), "hit_token_limit": bool(row.get("hit_token_limit")),
            "chinese_output": (bool(CJK.search(ans)) if lang == "zh" else None), "new_tokens": row.get("new_tokens"),
            "latency_s": row.get("latency_s"), "peak_vram_gib": row.get("peak_vram_gib")}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def aggregate(scored: List[dict]) -> dict:
    an = [s for s in scored if s["answerable"]]
    ua = [s for s in scored if not s["answerable"]]
    long_ = [s for s in scored if (s.get("new_tokens") or 0) > 128]
    lat = sorted(s["latency_s"] for s in scored if s.get("latency_s") is not None)
    return {"n": len(scored), "ok": sum(s["status"] == "OK" for s in scored), "accuracy": mean([s["correct"] for s in scored]),
            "answerable_accuracy": mean([s["correct"] for s in an]), "unanswerable_accuracy": mean([s["correct"] for s in ua]),
            "by_source": {k: mean([s["correct"] for s in scored if s["source"] == k]) for k in sorted({s["source"] for s in scored})},
            "by_qtype": {k: {"n": sum(1 for s in scored if s["qtype"] == k), "acc": mean([s["correct"] for s in scored if s["qtype"] == k])}
                         for k in sorted({str(s["qtype"]) for s in scored})},
            "f1_answerable": mean([s["f1"] for s in an]), "hallucination_unanswerable": mean([s["hallucinated_unanswerable"] for s in ua]),
            "wrong_abstention_answerable": mean([s["wrong_abstention"] for s in an]), "abstention_rate": mean([s["abstained"] for s in scored]),
            "evidence_precision": mean([s["evidence_precision"] for s in an]), "evidence_recall": mean([s["evidence_recall"] for s in an]),
            "cites_unknown_id": sum(s["cites_unknown_id"] for s in scored), "quotes_not_in_evidence": sum(s["quotes_not_in_evidence"] for s in scored),
            "json_valid": mean([s["json_valid"] for s in scored]), "repetition_rate": mean([s["repetitive"] for s in scored]),
            "repetition_rate_long_answers": mean([s["repetitive"] for s in long_]), "long_answers": len(long_),
            "hit_token_limit_rate": mean([s["hit_token_limit"] for s in scored]), "chinese_output_rate_zh": mean([s["chinese_output"] for s in scored]),
            "latency_p50": lat[len(lat) // 2] if lat else None, "latency_p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else None,
            "peak_vram_gib_max": max((s["peak_vram_gib"] or 0 for s in scored), default=None)}


def latest_ok(path: Path) -> Dict[str, dict]:
    A, C, R = P3()
    rows = {}
    for r in C.read_jsonl(path):
        if rows.get(r["question_id"], {}).get("status") != "OK":
            rows[r["question_id"]] = r
    return rows


def score(run_dir: Path, set_name: str, pilot: Path, groups: List[str], M=None) -> dict:
    A, C, R = P3()
    gold = {json.loads(l)["question_id"]: json.loads(l) for l in (pilot / "eval" / "pilot_dev_gold.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    items = load_set(set_name, pilot, None)
    res, correct = {"set": set_name, "label": "EXPLORATORY local scorer (token F1 >= 0.5 or containment; abstention for unanswerable); not the LifeBench judge",
                    "groups": {}}, {}
    for g in groups:
        rows = latest_ok(run_dir / "eval" / set_name / g / "answers.jsonl")
        sc = [score_row(rows[q], gold[q], M) for q in items if q in rows]
        res["groups"][g] = aggregate(sc)
        res["groups"][g]["missing"] = len([q for q in items if q not in rows])
        correct[g] = {s["question_id"]: s["correct"] for s in sc}
    if "B" in correct:
        res["paired_vs_B"] = {g: A.paired(correct["B"], correct[g], list(items)) for g in groups if g != "B"}
    C.write_json_atomic(run_dir / "eval" / set_name / "LOCAL_SCORE.json", res)
    return res


def select(run_dir: Path, pilot: Path, candidates: List[str], M=None) -> dict:
    res = score(run_dir, "select", pilot, ["B"] + candidates, M)
    base_rep = res["groups"]["B"]["repetition_rate"] or 0.0
    table, ok = [], []
    for c in candidates:
        g = res["groups"][c]
        q = g["json_valid"] is not None and g["json_valid"] >= 0.95 and (g["repetition_rate"] or 0) <= base_rep + 0.01 and g["missing"] == 0
        table.append({"candidate": c, "accuracy": g["accuracy"], "json_valid": g["json_valid"], "repetition_rate": g["repetition_rate"], "qualified": q})
        if q:
            ok.append((g["accuracy"] or 0, int(c.split("-")[1]), c))
    chosen = max(ok)[2] if ok else None
    out = {"rule": SELECTION_RULE, "base": {"accuracy": res["groups"]["B"]["accuracy"], "repetition_rate": base_rep}, "candidates": table,
           "selected": chosen, "status": "SELECTED" if chosen else "NO_QUALIFIED_CHECKPOINT"}
    A, C, R = P3()
    C.write_json_atomic(run_dir / "eval" / "CHECKPOINT_SELECTION.json", out)
    return out


def compare_p3(run_dir: Path, p3: Path, data_path: Path, groups: List[str], M=None) -> dict:
    """LifeBench dev200 (EXPLORATORY): judged accuracy via the Phase 3 analysis (groups B, E) + local diagnostics + paired B vs E."""
    A, C, R = P3()
    d = run_dir / "eval" / "p3dev200"
    frozen = C.load_frozen(p3 / "ef1_frozen.jsonl")
    samples = json.loads((p3 / "samples.json").read_text(encoding="utf-8"))
    data = json.load(open(data_path, encoding="utf-8"))
    baseline = {json.loads(l)["question_id"]: json.loads(l) for l in (p3 / "v3_baseline.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    res = A.analyze(d, samples, "dev200", frozen, data, baseline, groups)
    corr = {g: {q: bool(v["is_correct"]) for q, v in A.load_verdicts(d, g, "dev200").items()} for g in groups}
    qids = list(samples["dev200"]["question_ids"])
    if corr.get("B") and corr.get("E"):
        res["paired"]["B_vs_E"] = A.paired(corr["B"], corr["E"], qids)
    diag = {}
    for g in groups:
        rows = latest_ok(d / g / "answers.jsonl")
        ok = [r for r in rows.values() if r.get("status") == "OK"]
        diag[g] = {"n_ok": len(ok), "json_valid": mean([bool((r.get("parsed") or {}).get("answer_complete")) for r in ok]),
                   "repetition_rate": mean([K.repetitive(r.get("raw") or "", "zh", M) for r in ok]),
                   "repetition_rate_long": mean([K.repetitive(r.get("raw") or "", "zh", M) for r in ok if (r.get("new_tokens") or 0) > 128]),
                   "hit_token_limit_rate": mean([bool(r.get("hit_token_limit")) for r in ok]),
                   "chinese_output_rate": mean([bool(CJK.search(r.get("final_answer") or "")) for r in ok]),
                   "unsupported_numbers_or_quotes": sum(1 for r in ok if not ((r.get("audit") or {}).get("support") or {}).get("supported", True)),
                   "cites_unknown_eid": sum(1 for r in ok if ((r.get("audit") or {}).get("citations") or {}).get("unknown_ids"))}
    res["diagnostics"] = diag
    res["judged"] = {g: len(corr.get(g) or {}) for g in groups}
    C.write_json_atomic(d / "D5_COMPARISON.json", res)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "score", "select", "compare"])
    ap.add_argument("--base-dir")
    ap.add_argument("--adapter", action="append", default=[], help="NAME=DIR (repeatable)")
    ap.add_argument("--set", default="d5dev")
    ap.add_argument("--groups", default="B,E")
    ap.add_argument("--candidates", default="")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--pilot-data")
    ap.add_argument("--p3-data")
    ap.add_argument("--data")
    ap.add_argument("--fingerprint-json", help="extra identity fields (release / config / adapter sha) merged into the run fingerprint")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    pilot = Path(a.pilot_data) if a.pilot_data else None
    p3 = Path(a.p3_data) if a.p3_data else None
    try:
        from t2d4.colab import metrics as M
    except Exception:  # noqa: BLE001
        M = None
    groups = [g.strip() for g in a.groups.split(",") if g.strip()]
    if a.cmd == "generate":
        adapters = dict(x.split("=", 1) for x in a.adapter)
        missing = [g for g in groups if g != "B" and g not in adapters]
        if missing:
            print(f"REFUSED: no adapter given for groups {missing}")
            return 1
        items = load_set(a.set, pilot, p3)
        gen = Gen(Path(a.base_dir), {k: Path(v) for k, v in adapters.items()})
        from stillhere.reader import fact as rf
        fp = {"d5": "d5-pilot-v1", "set": a.set, "generator": gen.label, "reader_version": rf.READER_VERSION, "fact_generation": rf.FACT_GENERATION,
              "fact_generation_long": rf.FACT_GENERATION_LONG, "fact_system_sha256": K.fact_system_sha(),
              "adapters": {k: v["sha256"] for k, v in gen.info.items()}, "items_sha256": K.h(json.dumps(sorted((q, i["prompt_hash"]) for q, i in items.items())), 32)}
        if a.fingerprint_json:
            fp.update(json.loads(Path(a.fingerprint_json).read_text(encoding="utf-8")))
        s = generate(gen, items, groups, run / "eval" / a.set, fp, a.limit)
        print(json.dumps(s, ensure_ascii=False, indent=1))
        return 0 if all(v["ok"] == v["of"] for v in s.values()) else 2
    if a.cmd == "score":
        print(json.dumps(score(run, a.set, pilot, groups, M), ensure_ascii=False, indent=1))
        return 0
    if a.cmd == "select":
        cands = [c.strip() for c in a.candidates.split(",") if c.strip()]
        print(json.dumps(select(run, pilot, cands, M), ensure_ascii=False, indent=1))
        return 0
    print(json.dumps(compare_p3(run, p3, Path(a.data), groups, M), ensure_ascii=False, indent=1, default=str)[:20000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
