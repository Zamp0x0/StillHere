"""Persona evaluation: A-E ablation, checkpoint selection on DEV, blind review, paired statistics, a sealed TEST.

Groups (one NF4 base; the adapter is switched on / off by name, as in ``d5pilot.evaluate.Gen``):

    A  Base + minimal neutral prompt                     (no persona prompt, no adapter, no memory)
    B  Base + Persona Inference Prompt                   (byte-identical prompt for B, C, D, E; sha256 pinned)
    C  Base + the same prompt + Persona Adapter          -> B vs C: what fine-tuning adds
    D  Base + the same prompt + SQLite memory evidence   (EF1 over the train period)
    E  Base + the same prompt + adapter + the SAME memory evidence  -> D vs E: does the adapter still help with memory

Everything else is identical for every group: the conversation context (the training ``human`` turn, byte for byte), the decode
configuration ``DECODE`` (seed per prompt), the token budget, no output rewriting (the guard only annotates). The reference reply is a
reference for metrics, never a target the model must reproduce.

DEV selects checkpoints and debugs; TEST is read only through ``dataset.load_split(..., "evaluation")`` (seal verified), only after
``lock_for_test`` froze adapter / prompt / decode / groups, and every test opening is logged; a second test run with another
configuration is refused (no re-trying until the numbers look better).

A run without a real model (``FakeGenerator``) is labelled ``DRY_RUN_ONLY`` everywhere; it tests the infrastructure, never the persona.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import random
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from stillhere.style.safety import SPECIFIC, repetition
from stillhere.style_eval import metrics as SM

from . import cleaning as C
from . import dataset as D
from . import store as S

EVAL_VERSION = "stillhere.persona.eval/1.0"
NEUTRAL_PROMPT = "请根据下面的聊天记录，以“我”的身份回复对方的最新消息。只输出回复本身。"
NEUTRAL_PROMPT_VERSION = "stillhere.persona.neutral_prompt/1.0"
GROUPS = {"A": {"prompt": "neutral", "adapter": False, "memory": False}, "B": {"prompt": "persona", "adapter": False, "memory": False},
          "C": {"prompt": "persona", "adapter": True, "memory": False}, "D": {"prompt": "persona", "adapter": False, "memory": True},
          "E": {"prompt": "persona", "adapter": True, "memory": True}}
PAIRS = (("A", "B"), ("B", "C"), ("D", "E"))
DECODE = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0, "max_new_tokens": 96,
          "seed": 20261116, "enable_thinking": False}
GENERIC = {"嗯", "嗯嗯", "好", "好的", "好滴", "哦", "哦哦", "噢", "行", "ok", "okk", "可以", "收到", "嗯呢", "是", "对"}
DRY = "DRY_RUN_ONLY"
SELECTION_RULE = ("pre-registered (DEV, group C messages): drop a candidate whose repetition rate > B + 0.01, train-copy rate > 0.02, median "
                  "length ratio outside [0.5, 2.0], or top-1 identical-reply share > max(0.30, reference top-1 + 0.10); among the rest pick the "
                  "highest mean style margin; a difference < 0.005 is a tie -> the later checkpoint; none left -> NO_QUALIFIED_CHECKPOINT. "
                  "Training / validation loss is recorded, never used to select.")
SELECTION_RULE_R2 = (SELECTION_RULE + " r2 (registered 2026-11-18, before any r2 training, from the r1 / FAST collapse evidence) additionally "
                     "drops a candidate whose top-1 identical-reply share > max(2 x reference top-1, reference top-1 + 0.03), whose generic-reply "
                     "rate > reference generic-reply rate + 0.05, or whose distinct-reply ratio < reference distinct-reply ratio - 0.15 "
                     "(reference = the real DEV replies of the same items). Applied to r1's ckpt-132 it would have rejected it "
                     "(top-1 0.093 vs 0.045, generic 0.134 vs 0.056, distinct 0.51).")
SELECTION_RULES = {"r1": SELECTION_RULE, "r2": SELECTION_RULE_R2}
SUCCESS_RULE = ("pre-registered (C vs B, and E vs D): style margin paired mean diff > 0 with a 95% CI above 0; repetition <= baseline + 0.01; "
                "train-copy rate <= 0.02; median length ratio in [0.7, 1.4]; privacy leaks 0; and, for the verdict BETTER, a blind review win "
                "rate > 50% with sign-test p < 0.05. Without a blind review the verdict is at most AUTOMATIC_METRICS_ONLY.")


class EvalError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha_obj(o) -> str:
    return sha(json.dumps(o, ensure_ascii=False, sort_keys=True))


# ------------------------------------------------------------------ evaluation sets
def _last_other(human: str) -> str:
    lines = [ln[3:] for ln in human.split("\n") if ln.startswith("对方：")]
    return lines[-1] if lines else ""


def eval_items(dataset_dir, split: str) -> List[dict]:
    """DEV or TEST rows as evaluation items (purpose ``evaluation``: the test seal is verified on every read)."""
    if split not in ("dev", "test"):
        raise EvalError("BAD_SPLIT", "evaluation reads dev or test")
    rows = D.load_split(dataset_dir, split, "evaluation")
    metas = {m["sample_id"]: m for m in D.load_meta(dataset_dir, split, "evaluation")}
    return [{"sample_id": r["id"], "user": r["conversations"][0]["value"], "reference": r["conversations"][1]["value"], "query": _last_other(r["conversations"][0]["value"]),
             "time": metas[r["id"]]["time"], "flags": metas[r["id"]]["flags"]} for r in rows]


def messages_for(item: Mapping, group: str, persona_prompt: str) -> List[dict]:
    g = GROUPS[group]
    system = NEUTRAL_PROMPT if g["prompt"] == "neutral" else persona_prompt
    if g["memory"]:
        mem = item.get("memory")
        if mem is None:
            raise EvalError("NO_MEMORY", f"group {group} needs the memory evidence of {item['sample_id']}")
        system = system.rstrip("\n") + "\n\n" + mem
    return [{"role": "system", "content": system}, {"role": "user", "content": item["user"]}]


def write_eval_set(path: Path, items: Sequence[Mapping], *, persona_prompt: str, groups: Sequence[str], meta: Mapping) -> dict:
    """One JSONL line per item with the exact messages of every group (and their hashes); B/C and D/E identity is asserted here."""
    rows = []
    for it in items:
        msgs = {g: messages_for(it, g, persona_prompt) for g in groups}
        if "B" in msgs and "C" in msgs:
            assert msgs["B"] == msgs["C"]
        if "D" in msgs and "E" in msgs:
            assert msgs["D"] == msgs["E"]
        rows.append({"sample_id": it["sample_id"], "messages": msgs, "prompt_hash": {g: sha_obj(m) for g, m in msgs.items()}, "user": it["user"],
                     "reference": it["reference"], "query": it["query"], "time": it["time"], "memory": it.get("memory"), "flags": it.get("flags", [])})
    head = {**meta, "items": len(rows), "groups": list(groups), "persona_prompt_sha256": sha(persona_prompt), "neutral_prompt_sha256": sha(NEUTRAL_PROMPT),
            "decode": DECODE, "eval_version": EVAL_VERSION}
    S.atomic_write(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8"))
    S.write_json(path.with_suffix(".meta.json"), {**head, "file_sha256": D.sha256_bytes(path.read_bytes())})
    return head


def read_eval_set(path: Path) -> List[dict]:
    head = S.read_json(Path(path).with_suffix(".meta.json"))
    if D.sha256_bytes(Path(path).read_bytes()) != head["file_sha256"]:
        raise EvalError("EVAL_SET_CHANGED", f"{path} does not match its meta")
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:                                     # the user turn is identical in every group
        r.setdefault("user", next(iter(r["messages"].values()))[1]["content"])
    return rows


# ------------------------------------------------------------------ generators
class FakeGenerator:
    """CPU stand-in with the generator interface of ``d5pilot.evaluate.Gen``. Deterministic; NOT a model. Every row it produces is
    labelled FAKE_DRY_RUN and every report built from it DRY_RUN_ONLY. Base groups answer a generic line, adapter groups a train reply
    picked by hash (so copy detection and collapse metrics are exercised)."""
    label = "FAKE_DRY_RUN"

    def __init__(self, pool: Sequence[str] = ("好", "我也不知道诶", "哈哈哈哈", "明天再说吧")):
        self.pool = list(pool) or ["好"]

    def __call__(self, messages, cfg, adapter: Optional[str]) -> dict:
        h = int(sha(json.dumps(messages, ensure_ascii=False, sort_keys=True) + str(adapter))[:8], 16)
        text = self.pool[h % len(self.pool)] if adapter else ("嗯" if h % 3 == 0 else "好的，我知道了。")
        return {"text": text, "hit_token_limit": False, "new_tokens": max(1, len(text)), "prompt_tokens": None, "latency_s": 0.0, "peak_vram_gib": 0.0}

    def is_oom(self, e) -> bool:
        return False

    def after_error(self):
        pass


def _append(path: Path, row: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def done_rows(path: Path) -> Dict[str, dict]:
    out = {}
    if path.is_file():
        for l in path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("status") == "OK":
                    out[r["sample_id"]] = r
    return out


def bind_fingerprint(out_dir: Path, fp: Mapping) -> str:
    p = out_dir / "RUN_FINGERPRINT.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    if p.is_file():
        old = S.read_json(p)
        if old["sha256"] != sha_obj(fp):
            diff = sorted(k for k in set(old["fields"]) | set(fp) if old["fields"].get(k) != fp.get(k))
            raise EvalError("FINGERPRINT_MISMATCH", f"{out_dir} was started with another configuration ({diff}); use a new run directory")
        return "RESUME"
    S.write_json(p, {"fields": dict(fp), "sha256": sha_obj(fp)})
    return "NEW"


def generate(gen, items: Sequence[Mapping], groups: Sequence[str], out_dir: Path, fingerprint: Mapping, *, adapters: Mapping[str, Optional[str]],
             limit: Optional[int] = None, log: Callable[[str], None] = print, decodes: Optional[Mapping[str, Mapping]] = None) -> dict:
    """``adapters``: group -> adapter name (None for the base groups). Resumable: OK rows with the same prompt hash are skipped.
    ``decodes``: optional group -> generation config (exploratory decoding groups); every other group uses ``DECODE``."""
    decodes = decodes or {}
    mode = bind_fingerprint(out_dir, fingerprint)
    log(f"{mode} {out_dir}")
    summary = {}
    for g in groups:
        path = out_dir / g / "answers.jsonl"
        done = done_rows(path)
        todo = [it for it in items if it["sample_id"] not in done or done[it["sample_id"]].get("prompt_hash") != it["prompt_hash"][g]]
        if limit:
            todo = todo[:limit]
        log(f"group {g}: {len(items) - len(todo)} done, {len(todo)} to run")
        for i, it in enumerate(todo):
            t0 = time.time()
            try:
                r = gen(it["messages"][g], decodes.get(g, DECODE), adapters.get(g))
                row = {"sample_id": it["sample_id"], "group": g, "status": "OK", **r}
            except Exception as e:  # noqa: BLE001 - recorded, retried on the next run
                oom = gen.is_oom(e)
                gen.after_error()
                row = {"sample_id": it["sample_id"], "group": g, "status": "OOM" if oom else "ERROR", "error": f"{type(e).__name__}: {str(e)[:300]}"}
            row.update(prompt_hash=it["prompt_hash"][g], generator=gen.label, run_fingerprint=sha_obj(fingerprint), wall_s=round(time.time() - t0, 3))
            _append(path, row)
            if (i + 1) % 25 == 0 or i + 1 == len(todo):
                log(f"[{g}] {i + 1}/{len(todo)}")
        d = done_rows(path)
        summary[g] = {"ok": sum(1 for it in items if it["sample_id"] in d), "of": len(items)}
    S.write_json(out_dir / "GENERATION_STATUS.json", {"summary": summary, "generator": gen.label, "at": S.now_iso()})
    return summary


# ------------------------------------------------------------------ metrics
def _vec(text: str) -> Dict[str, float]:
    return SM.raw_vector(text, "char")


def _centroid(texts: Iterable[str]) -> Dict[str, float]:
    c: Counter = Counter()
    n = 0
    for t in texts:
        c.update(_vec(t))
        n += 1
    return {k: v / max(1, n) for k, v in c.items()}


def _cos(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na, nb = math.sqrt(sum(v * v for v in a.values())), math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


class Reference:
    """Train-only references for the metrics: the target's train replies (style centroid, memorisation set) and the other person's
    train-context lines (the contrast centroid: imitating the other person lowers the style margin)."""

    def __init__(self, dataset_dir, card: Optional[Mapping] = None):
        rows = D.load_split(dataset_dir, "train", "persona_prompt")
        self.train_targets = [r["conversations"][1]["value"].replace("\n", "") for r in rows]
        others = sorted({ln[3:] for r in rows for ln in r["conversations"][0]["value"].split("\n") if ln.startswith("对方：")})
        self.target_c, self.other_c = _centroid(self.train_targets), _centroid(others)
        self.long_targets = sorted({t for t in self.train_targets if len(t) >= 8})
        self.norm_targets = {D.norm(t) for t in self.train_targets if len(D.norm(t)) >= 6}
        self.markers = [x["text"] for x in (card or {}).get("common_expressions", [])][:12]

    def style_margin(self, text: str) -> float:
        v = _vec(text.replace("\n", ""))
        return round(_cos(v, self.target_c) - _cos(v, self.other_c), 4)


def score_row(text: str, item: Mapping, ref: Reference, *, hit_token_limit: bool = False) -> dict:
    flat = text.replace("\n", "")
    refflat = item["reference"].replace("\n", "")
    evidence = item["user"] + "\n" + (item.get("memory") or "")
    specific = [m.group(0) for m in SPECIFIC.finditer(flat)]
    unsupported = [s for s in specific if s not in evidence]
    in_memory = [s for s in specific if item.get("memory") and s in item["memory"] and s not in item["user"]]
    q = set(SM._grams(D.norm(item.get("query") or ""), 2)) if item.get("query") else set()
    g = set(SM._grams(D.norm(flat), 2))
    return {"chars": len(flat), "bubbles": len([x for x in text.split("\n") if x.strip()]), "ref_chars": len(refflat),
            "length_log_ratio": round(math.log((len(flat) + 1) / (len(refflat) + 1)), 4), "generic": flat.strip().lower() in GENERIC,
            "ref_generic": refflat.strip().lower() in GENERIC, "repetition": repetition(text, hit_token_limit=hit_token_limit).degenerate,
            "train_copy": SM.copies(flat, ref.long_targets) if len(flat) >= 8 else False, "train_exact": D.norm(flat) in ref.norm_targets,
            "style_margin": ref.style_margin(text), "marker": any(m in flat for m in ref.markers),
            "unsupported_specific": len(unsupported), "memory_specific_used": len(in_memory),
            "privacy_leak": bool([c for c, _ in C.privacy_hits(text) if c in C.CleaningConfig().exclude_target_privacy]),
            "special_token_leak": "<think>" in text or "<|" in text, "prefix_leak": flat.startswith(("我：", "我:")),
            "context_overlap": round(len(q & g) / len(q | g), 4) if q and g else 0.0, "empty": not flat.strip(), "hit_token_limit": hit_token_limit}


def _rate(xs) -> float:
    xs = list(xs)
    return round(sum(1 for x in xs if x) / len(xs), 4) if xs else 0.0


def _mean(xs) -> Optional[float]:
    xs = list(xs)
    return round(statistics.fmean(xs), 4) if xs else None


def aggregate(scored: Sequence[Mapping], texts: Sequence[str]) -> dict:
    if not scored:
        return {"n": 0}
    top = Counter(t.strip() for t in texts).most_common(1)[0]
    ratios = sorted(math.exp(s["length_log_ratio"]) for s in scored)
    return {"n": len(scored), "style_margin_mean": _mean(s["style_margin"] for s in scored), "marker_rate": _rate(s["marker"] for s in scored),
            "chars_median": statistics.median(s["chars"] for s in scored), "ref_chars_median": statistics.median(s["ref_chars"] for s in scored),
            "length_ratio_median": round(statistics.median(ratios), 3), "abs_length_log_ratio_mean": _mean(abs(s["length_log_ratio"]) for s in scored),
            "bubbles_mean": _mean(s["bubbles"] for s in scored), "generic_reply_rate": _rate(s["generic"] for s in scored),
            "ref_generic_reply_rate": _rate(s["ref_generic"] for s in scored), "top1_reply_share": round(top[1] / len(scored), 4),
            "distinct_reply_ratio": round(len(set(t.strip() for t in texts)) / len(texts), 4), "repetition_rate": _rate(s["repetition"] for s in scored),
            "train_copy_rate": _rate(s["train_copy"] for s in scored), "train_exact_rate": _rate(s["train_exact"] for s in scored),
            "unsupported_specific_rate": _rate(s["unsupported_specific"] > 0 for s in scored), "memory_specific_used_rate": _rate(s["memory_specific_used"] > 0 for s in scored),
            "privacy_leaks": sum(1 for s in scored if s["privacy_leak"]), "special_token_leak_rate": _rate(s["special_token_leak"] for s in scored),
            "prefix_leak_rate": _rate(s["prefix_leak"] for s in scored), "context_overlap_mean": _mean(s["context_overlap"] for s in scored),
            "empty_rate": _rate(s["empty"] for s in scored), "hit_token_limit_rate": _rate(s["hit_token_limit"] for s in scored)}


def ref_top1(items: Sequence[Mapping]) -> float:
    c = Counter(it["reference"].replace("\n", "").strip() for it in items)
    return round(c.most_common(1)[0][1] / len(items), 4) if items else 0.0


def score_run(out_dir: Path, items: Sequence[Mapping], groups: Sequence[str], ref: Reference) -> dict:
    per, agg, gens = {}, {}, set()
    by_id = {it["sample_id"]: it for it in items}
    for g in groups:
        rows = done_rows(out_dir / g / "answers.jsonl")
        sc, texts = {}, []
        for sid, r in sorted(rows.items()):
            if sid in by_id:
                sc[sid] = score_row(r["text"], by_id[sid], ref, hit_token_limit=bool(r.get("hit_token_limit")))
                texts.append(r["text"])
                gens.add(r.get("generator"))
        per[g] = sc
        agg[g] = aggregate(list(sc.values()), texts)
    paired = {}
    for a, b in PAIRS:
        if a in per and b in per and per[a] and per[b]:
            keys = sorted(set(per[a]) & set(per[b]))
            sm = SM.paired({k: per[a][k]["style_margin"] for k in keys}, {k: per[b][k]["style_margin"] for k in keys}, seed="persona")
            ln = SM.paired({k: -abs(per[a][k]["length_log_ratio"]) for k in keys}, {k: -abs(per[b][k]["length_log_ratio"]) for k in keys}, seed="persona")
            bins = {}
            for m in ("repetition", "train_copy", "generic", "privacy_leak", "unsupported_specific"):
                worse = sum(1 for k in keys if bool(per[b][k][m]) and not bool(per[a][k][m]))
                better = sum(1 for k in keys if bool(per[a][k][m]) and not bool(per[b][k][m]))
                bins[m] = {"b_worse": worse, "b_better": better, "mcnemar_sign_p": SM.sign_p(better, worse)}
            paired[f"{a}_vs_{b}"] = {"n": len(keys), "style_margin": sm, "length_closeness": ln, "binary": bins}
    dry = gens <= {FakeGenerator.label} or not gens
    out = {"schema": "stillhere.persona.eval_score/1.0", "status": DRY if dry else "REAL_MODEL", "generators": sorted(x for x in gens if x),
           "groups": agg, "paired": paired, "reference_top1_share": ref_top1(items), "success_rule": SUCCESS_RULE}
    out["verdicts"] = {p: verdict(out, *p.split("_vs_")) for p in paired if p != "A_vs_B"}
    S.write_json(out_dir / "SCORE.json", out)
    return out


def verdict(score: Mapping, base: str, cand: str, blind: Optional[Mapping] = None) -> dict:
    if score["status"] == DRY:
        return {"verdict": DRY, "note": "no real model: infrastructure test only"}
    a, b = score["groups"][base], score["groups"][cand]
    p = score["paired"][f"{base}_vs_{cand}"]
    ci = (p["style_margin"] or {}).get("ci")
    checks = {"style_margin_ci_above_0": bool(ci and ci[0] > 0), "repetition_ok": b["repetition_rate"] <= a["repetition_rate"] + 0.01,
              "copy_ok": b["train_copy_rate"] <= 0.02, "length_ok": 0.7 <= b["length_ratio_median"] <= 1.4, "privacy_ok": b["privacy_leaks"] == 0}
    if blind:
        checks["blind_win"] = blind.get("win_rate", 0) > 0.5 and (blind.get("sign_p") or 1) < 0.05
    ok = all(checks.values())
    v = ("BETTER" if ok else "NOT_BETTER") if blind else ("AUTOMATIC_METRICS_ONLY_PASS" if ok else "NOT_BETTER")
    return {"verdict": v, "checks": checks}


def ref_distinct(items: Sequence[Mapping]) -> float:
    refs = [it["reference"].strip() for it in items]
    return round(len(set(refs)) / len(refs), 4) if refs else 0.0


def select_checkpoint(out_dir: Path, items: Sequence[Mapping], candidates: Sequence[str], ref: Reference, rule: str = "r1") -> dict:
    """``candidates``: group-like names of checkpoint answer directories (ckpt-N, messages of C), compared with B on DEV.
    ``rule``: ``r1`` (the rule r1 was selected with, unchanged) or ``r2`` (r1 + the collapse guards of ``SELECTION_RULE_R2``)."""
    if rule not in SELECTION_RULES:
        raise EvalError("BAD_RULE", f"selection rule must be one of {sorted(SELECTION_RULES)}")
    sc = score_run(out_dir, items, ["B", *candidates], ref)
    base, rt1 = sc["groups"]["B"], sc["reference_top1_share"]
    rdist = ref_distinct(items)
    table = {}
    for c in candidates:
        g = sc["groups"][c]
        reasons = []
        if g["n"] == 0:
            reasons.append("no answers")
        else:
            if g["repetition_rate"] > base["repetition_rate"] + 0.01:
                reasons.append("repetition")
            if g["train_copy_rate"] > 0.02:
                reasons.append("train copy")
            if not 0.5 <= g["length_ratio_median"] <= 2.0:
                reasons.append("length")
            if g["top1_reply_share"] > max(0.30, rt1 + 0.10):
                reasons.append("collapse")
            if rule == "r2":
                if g["top1_reply_share"] > max(2 * rt1, rt1 + 0.03):
                    reasons.append("collapse_top1_r2")
                if (g.get("generic_reply_rate") or 0.0) > (g.get("ref_generic_reply_rate") or 0.0) + 0.05:
                    reasons.append("collapse_generic_r2")
                if g["distinct_reply_ratio"] < rdist - 0.15:
                    reasons.append("collapse_distinct_r2")
        table[c] = {"qualified": not reasons, "reasons": reasons, "style_margin_mean": g.get("style_margin_mean"),
                    **({k: g.get(k) for k in ("top1_reply_share", "generic_reply_rate", "distinct_reply_ratio")} if g["n"] else {})}
    q = [c for c in candidates if table[c]["qualified"]]
    best = None
    for c in sorted(q, key=lambda c: int(c.split("-")[1])):            # ascending steps: a later checkpoint wins a tie
        if best is None or table[c]["style_margin_mean"] > table[best]["style_margin_mean"] - 0.005:
            best = c
    res = {"rule": SELECTION_RULES[rule], "rule_id": rule, "reference_distinct_ratio": rdist, "reference_top1_share": rt1, "status": (DRY if sc["status"] == DRY else "SELECTED") if best else "NO_QUALIFIED_CHECKPOINT", "selected": best,
           "candidates": table, "base": {k: base.get(k) for k in ("style_margin_mean", "repetition_rate", "train_copy_rate", "length_ratio_median")},
           "dry_run": sc["status"] == DRY}
    S.write_json(out_dir / "CHECKPOINT_SELECTION.json", res)
    return res


# ------------------------------------------------------------------ blind review
def make_blind(out_dir: Path, items: Sequence[Mapping], pair: Sequence[str], *, n: int = 60, seed: str = "persona-blind", context_lines: int = 6,
               dest: Optional[Path] = None) -> dict:
    """A CSV for a human rater (left / right shuffled per row, groups hidden) and a separate key file. Both are PRIVATE (real chat)."""
    a, b = pair
    ra, rb = done_rows(out_dir / a / "answers.jsonl"), done_rows(out_dir / b / "answers.jsonl")
    ids = sorted(set(ra) & set(rb) & {it["sample_id"] for it in items})
    rng = random.Random(f"{seed}:{a}:{b}")
    rng.shuffle(ids)
    ids = ids[:n]
    by = {it["sample_id"]: it for it in items}
    key, rows = {}, []
    for i, sid in enumerate(ids, 1):
        left_is_a = rng.random() < 0.5
        rid = f"r{i:03d}"
        key[rid] = {"sample_id": sid, "left": a if left_is_a else b, "right": b if left_is_a else a}
        ctx = "\n".join(by[sid]["user"].split("\n")[-context_lines:])
        rows.append({"id": rid, "context": ctx, "reply_left": (ra if left_is_a else rb)[sid]["text"], "reply_right": (rb if left_is_a else ra)[sid]["text"],
                     "choice(L/R/=)": "", "note": ""})
    bdir = Path(dest) if dest else out_dir / "blind" / f"{a}_vs_{b}"
    if (bdir / "BLIND_KEY.json").is_file():
        raise EvalError("BLIND_EXISTS", f"{bdir} already holds a blind review (its key is never overwritten)")
    bdir.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]) if rows else ["id"])
    w.writeheader()
    w.writerows(rows)
    S.atomic_write(bdir / "BLIND_REVIEW.csv", buf.getvalue().encode("utf-8-sig"))
    S.write_json(bdir / "BLIND_KEY.json", {"pair": [a, b], "seed": seed, "key": key, "note": "do not open before rating"})
    return {"csv": str(bdir / "BLIND_REVIEW.csv"), "key": str(bdir / "BLIND_KEY.json"), "rows": len(rows)}


def score_blind(csv_path: Path, key_path: Path) -> dict:
    key = S.read_json(key_path)
    a, b = key["pair"]
    wa = wb = tie = blank = 0
    with open(csv_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            ch = (r.get("choice(L/R/=)") or "").strip().upper()
            k = key["key"].get(r["id"])
            if not k or ch not in ("L", "R", "="):
                blank += 1
                continue
            if ch == "=":
                tie += 1
            elif (ch == "L") == (k["left"] == b):
                wb += 1
            else:
                wa += 1
    decided = wa + wb
    return {"pair": [a, b], f"{b}_wins": wb, f"{a}_wins": wa, "ties": tie, "unrated": blank, "win_rate": round(wb / decided, 4) if decided else None,
            "sign_p": SM.sign_p(wb, wa)}


# ------------------------------------------------------------------ test seal
def lock_for_test(eval_root: Path, *, dataset_version: str, prompt_sha256: str, adapter_sha256: Optional[str], checkpoint: Optional[str],
                  groups: Sequence[str], backend: str) -> dict:
    lock = {"dataset_version": dataset_version, "persona_prompt_sha256": prompt_sha256, "neutral_prompt_sha256": sha(NEUTRAL_PROMPT),
            "adapter_sha256": adapter_sha256, "checkpoint": checkpoint, "groups": list(groups), "decode": DECODE, "backend": backend, "eval_version": EVAL_VERSION}
    p = eval_root / "TEST_LOCK.json"
    if p.is_file():
        old = S.read_json(p)
        if old["sha256"] != sha_obj(lock):
            raise EvalError("TEST_ALREADY_LOCKED", "the test set was locked for another configuration; test is evaluated once (see TEST_LOCK.json)")
        return old
    S.write_json(p, {"lock": lock, "sha256": sha_obj(lock), "locked_at": S.now_iso()})
    return S.read_json(p)


def log_test_open(eval_root: Path, lock_sha: str, purpose: str) -> int:
    p = eval_root / "TEST_OPENINGS.jsonl"
    _append(p, {"at": S.now_iso(), "lock_sha256": lock_sha, "purpose": purpose})
    return sum(1 for l in p.read_text(encoding="utf-8").splitlines() if l.strip())
