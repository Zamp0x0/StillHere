"""STYLE_EVALUATOR_V2: person-adaptive judgement of generated replies (EXPLORATORY; V1 results and verdicts are never rewritten).

V1 judged length by a fixed ratio band (median model / reference length in [0.7, 1.4]) and generic replies by a fixed list. For a person
who mostly answers in 1-5 characters that punishes authentic style. V2 compares with the PERSON's own distributions:

* length: the share of replies inside the person's TRAIN length range [p5, p95] (chars), against the same share of the DEV references;
* authentic short replies: replies the person wrote >= 3 times in TRAIN ("好", "嗯嗯", ...) are allowed at the person's own rate;
* collapse is CONDITIONAL repetition: one reply used across different questions far beyond (a) the person's train rate of that reply and
  (b) the references' concentration; low reply diversity relative to the references; an authentic phrase repeated is not collapse by itself;
* insufficient answer: on DEV items whose reference carries content (not an authentic short phrase, >= 4 chars) after a question, a
  generic / authentic-short reply is counted as a possibly missing answer (reported, not a verdict on its own).

Nothing here is per-person code: every threshold is relative to the person's train / dev data.
"""
from __future__ import annotations

import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from . import dataset as D
from . import evaluation as EV

VERSION = "stillhere.persona.style_evaluator/2.0"
QUESTION = re.compile(r"[？?]|吗|呢$|什么|怎么|哪|几|谁|多少|是不是|有没有")


def _flat(t: str) -> str:
    return (t or "").replace("\n", "").strip()


def person_profile(ref: EV.Reference) -> dict:
    lens = sorted(len(t) for t in ref.train_targets)
    q = lambda p: lens[min(len(lens) - 1, int(p * (len(lens) - 1)))]
    cnt = Counter(D.norm(t) for t in ref.train_targets if t)
    n = len(ref.train_targets)
    authentic = {t: c / n for t, c in cnt.items() if c >= 3 and 0 < len(t) <= 6}
    return {"train_n": n, "len_p5": q(.05), "len_p25": q(.25), "len_p50": q(.5), "len_p95": q(.95), "authentic_short": authentic,
            "authentic_short_rate": round(sum(authentic.values()), 4)}


def _shape(texts: Sequence[str], prof: Mapping) -> dict:
    flat = [_flat(t) for t in texts]
    lo, hi = prof["len_p5"], prof["len_p95"]
    norm = [D.norm(t) for t in flat]
    c = Counter(norm)
    top, topn = c.most_common(1)[0] if c else ("", 0)
    return {"n": len(flat), "in_person_range": round(sum(lo <= len(t) <= hi for t in flat) / max(1, len(flat)), 4),
            "authentic_short_rate": round(sum(x in prof["authentic_short"] for x in norm) / max(1, len(flat)), 4),
            "distinct_ratio": round(len(c) / max(1, len(flat)), 4), "top_reply_share": round(topn / max(1, len(flat)), 4), "top_reply_norm": top,
            "chars_median": statistics.median(len(t) for t in flat) if flat else None}


def evaluate(run: Path, items: Sequence[Mapping], groups: Sequence[str], ref: EV.Reference, *, score: Optional[Mapping] = None) -> dict:
    prof = person_profile(ref)
    by = {it["sample_id"]: it for it in items}
    refs = _shape([it["reference"] for it in items], prof)
    out = {"version": VERSION, "label": "EXPLORATORY (new metric; V1 results and verdicts unchanged)", "person": {k: v for k, v in prof.items() if k != "authentic_short"},
           "authentic_short_phrases": len(prof["authentic_short"]), "reference": refs, "groups": {}}
    for g in groups:
        rows = EV.done_rows(run / g / "answers.jsonl")
        texts = [rows[s]["text"] for s in sorted(rows) if s in by]
        sh = _shape(texts, prof)
        norm = {s: D.norm(_flat(rows[s]["text"])) for s in rows if s in by}
        # conditional repetition: a reply over different questions vs the person's own train rate of that reply
        rep = Counter(norm.values())
        excess = []
        for t, k in rep.items():
            if k >= 3:
                share = k / max(1, len(norm))
                base = prof["authentic_short"].get(t, 0.0)
                if share > max(3 * base, refs["top_reply_share"] + 0.10, 0.15):
                    excess.append({"reply_len": len(t), "share": round(share, 4), "train_rate": round(base, 4), "authentic": t in prof["authentic_short"]})
        content_q = [s for s, it in by.items() if s in norm and QUESTION.search(it.get("query") or "") and len(_flat(it["reference"])) >= 4
                     and D.norm(_flat(it["reference"])) not in prof["authentic_short"]]
        insufficient = [s for s in content_q if norm[s] in prof["authentic_short"] or _flat(rows[s]["text"]).lower() in EV.GENERIC or len(norm[s]) <= 1]
        checks = {"length_in_person_range": sh["in_person_range"] >= refs["in_person_range"] - 0.10,
                  "no_conditional_collapse": not excess and sh["distinct_ratio"] >= refs["distinct_ratio"] - 0.25,
                  "authentic_short_rate_plausible": sh["authentic_short_rate"] <= max(2 * refs["authentic_short_rate"], prof["authentic_short_rate"] * 2) + 0.05}
        out["groups"][g] = {**sh, "excess_repetition": excess, "content_questions": len(content_q),
                            "possibly_insufficient_answers": round(len(insufficient) / max(1, len(content_q)), 4), "checks": checks}
    if score:
        out["verdicts_v2"] = {}
        for pair in ("B_vs_C", "D_vs_E"):
            a, b = pair.split("_vs_")
            if pair not in (score.get("paired") or {}) or b not in out["groups"]:
                continue
            ci = (score["paired"][pair]["style_margin"] or {}).get("ci")
            gb, ga = score["groups"][b], score["groups"][a]
            ch = {"style_margin_ci_above_0": bool(ci and ci[0] > 0), "repetition_ok": gb["repetition_rate"] <= ga["repetition_rate"] + 0.01,
                  "copy_ok": gb["train_copy_rate"] <= 0.02, "privacy_ok": gb["privacy_leaks"] == 0, **out["groups"][b]["checks"]}
            out["verdicts_v2"][pair] = {"verdict_v2": "PASS_AUTOMATIC_V2" if all(ch.values()) else "NOT_PASS_V2", "checks": ch,
                                        "note": "exploratory re-scoring of existing DEV generations; the original V1 verdict stands"}
    return out
