"""D5 candidate builders, one per source (pure Python, deterministic, no LLM, no API).

Every builder returns candidate dicts:
  id, source, split, group (persona / conversation / paper / question / persona_id), qtype, answerable, ua_kind,
  pack (FactPack), header, gold_eids, answer (gold, verbatim), quotes, provenance, bucket, pair_id
Gold evidence LOCATIONS may steer which units enter the pack (training-set construction); the answer text never does,
and nothing gold is written into the prompt.
"""
from __future__ import annotations

import collections
import json
import random
import re
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from . import common as K

# ------------------------------------------------------------------ shared
_PACK_SYSTEM_TOKENS = 600     # FACT_SYSTEM (~550) + chat layout; the exact count is measured later by the tokenizer audit


def split_of(key: str, train=0.8, dev=0.1) -> str:
    f = K.frac("split:" + key)
    return "train" if f < train else "dev" if f < train + dev else "holdout"


def bucket_of(key: str, weights) -> str:
    f = K.frac("bucket:" + key)
    acc = 0.0
    for b, w in weights:
        acc += w
        if f < acc:
            return b
    return weights[-1][0]


def target_input(key: str, bucket: str) -> int:
    lo, hi = K.BUCKETS[bucket]
    return int(lo + (hi - lo) * K.frac("len:" + key))


class Tok:
    def __init__(self, tokenizer_dir: Path):
        from tokenizers import Tokenizer
        self.t = Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer.json"))

    def n(self, s: str) -> int:
        return len(self.t.encode(s, add_special_tokens=False).ids)


def mk_item(n_total: int, idx: int, text: str, *, time: Optional[str], label: str, unit: str):
    from stillhere.context.fact_pack import PackItem
    eid = f"E{idx:02d}" if n_total < 100 else f"E{idx:03d}"
    return PackItem(eid=eid, unit_key=unit, rank=idx, score=0.0, source_type="d5_public", source_label=label, source_id=unit, event_time=time,
                    recorded_at=None, speakers=[], text=text, message_ids=[], channels={})


def mk_pack(items, question: str, ask_time: str = "未知"):
    from stillhere.context.fact_pack import FactPack
    return FactPack(schema="d5.pilot.v1", persona_id="d5", question=question, body=question, ask_time=ask_time, k=len(items), items=items,
                    notes=[], subqueries=[], shape=[])


def rendered(text: str, time: Optional[str], label: str, wide: bool = False) -> str:
    """The text ``fact_pack.render_item`` produces for one item (E000 / E00 placeholder id)."""
    head = f"[{'E000' if wide else 'E00'}] {(time or '时间未知')[:16]}｜{label}"
    return "\n".join([head] + ["    " + ln for ln in text.splitlines() if ln.strip()])


def sentences(text: str) -> List[str]:
    """Clauses ending at 。！？；/ newline (the trailing ；/。 is kept so every clause stays a verbatim substring of the record)."""
    return [s.strip() for s in re.split(r"(?<=[。！？!?；\n])", text) if s.strip()]


def bigrams(s: str) -> set:
    t = K.norm(s)
    return {t[i:i + 2] for i in range(len(t) - 1)}


# ------------------------------------------------------------------ PerLTQA (zh)
PROFILE_CN = {"Protagonist": "姓名", "Gender": "性别", "Nickname": "昵称", "Title": "头衔", "Age": "年龄", "Occupation": "职业", "Nationality": "国籍",
              "Physical Characteristics": "外貌特征", "Hobbies": "爱好", "Achievements": "成就", "Ethnic Background": "民族背景",
              "Education Background": "教育背景", "Employer": "雇主", "Awards and Role Models": "奖项与榜样"}
PERLT_TYPES = ("profile", "social_relationship", "events", "dialogues")


def _cn_date(s):
    m = re.match(r"(\d{4})年(\d{1,2})月(\d{1,2})日", s or "")
    return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}" if m else None


def perltqa_units(m: dict, tok: Tok) -> List[dict]:
    """One unit per profile / relationship / event / dialogue; time, characters, theme and links come from the data."""
    names = {sid: s["Supporting Characters"] for sid, s in m["social_relationship"].items()}
    prof = "人物档案：" + "；".join(f"{PROFILE_CN.get(k, k)}：{v}" for k, v in m["profile"].items() if str(v).strip())
    units = [{"uid": "PROFILE", "kind": "profile", "time": None, "text": prof, "chars": set(), "theme": None, "links": set()}]
    for sid, s in m["social_relationship"].items():
        units.append({"uid": sid, "kind": "social", "time": None, "text": f"{s['Supporting Characters']}（{s['Relationship']}）：{s['Description']}",
                      "chars": {s["Supporting Characters"]}, "theme": None, "links": set()})
    for eid, e in m["events"].items():
        units.append({"uid": eid, "kind": "event", "time": _cn_date(e.get("Creation Time")), "text": f"事件《{e.get('summary', '')}》：{e['content']}",
                      "chars": {names.get(str(c), str(c)) for c in (e.get("Characters") or []) if isinstance(c, (str, int))},
                      "theme": e.get("Theme") if isinstance(e.get("Theme"), str) else None, "links": set()})
    by = {u["uid"]: u for u in units}
    for did, dg in m["dialogues"].items():
        stamps = sorted(dg["contents"])
        lines = []
        for ts in stamps:
            lines.append(f"[{ts}]")
            stack = list(dg["contents"][ts]) if isinstance(dg["contents"][ts], list) else [dg["contents"][ts]]
            while stack:                       # a few dialogues nest lists of lines
                x = stack.pop(0)
                if isinstance(x, list):
                    stack = list(x) + stack
                elif str(x).strip():
                    lines.append(str(x))
        evs = dg.get("events")
        evs = [str(e) for e in (evs if isinstance(evs, list) else [evs]) if isinstance(e, (str, int)) and str(e) in by]
        u = {"uid": did, "kind": "dialogue", "time": stamps[0].replace(" ", "T") if stamps else None, "text": "对话：\n" + "\n".join(lines),
             "chars": set().union(*[by[e]["chars"] for e in evs]) if evs else set(), "theme": by[evs[0]]["theme"] if evs else None,
             "links": set(evs)}
        units.append(u)
        for e in evs:
            by[e]["links"].add(did)
    for u in units:
        u["tokens"] = tok.n(rendered(u["text"], u["time"], "个人记忆")) + 1      # exactly as render_item lays it out (+ newline)
    return units


def _days(a, b):
    try:
        return abs((datetime.fromisoformat(a[:10]) - datetime.fromisoformat(b[:10])).days)
    except Exception:
        return 10 ** 6


def choose_units(units: List[dict], gold: List[str], budget: int, key: str, exclude: set = frozenset()) -> Optional[List[dict]]:
    """Gold units + deterministic hard negatives until the token budget: linked records, records about the same people, the same theme,
    records within 60 days, then the rest in a seeded order. Uses gold LOCATIONS only (never the answer)."""
    by = {u["uid"]: u for u in units}
    req = [by[g] for g in gold if g in by and g not in exclude]
    used = sum(u["tokens"] for u in req)
    if used > budget:
        return None
    gset = {g for g in gold if g in by}
    chars = set().union(*[by[g]["chars"] for g in gset]) if gset else set()
    themes = {by[g]["theme"] for g in gset if by[g]["theme"]}
    times = [by[g]["time"] for g in gset if by[g]["time"]]
    links = set().union(*[by[g]["links"] for g in gset]) if gset else set()

    def prio(u):
        p = 4
        if u["uid"] in links:
            p = 0
        elif u["chars"] & chars:
            p = 1
        elif u["theme"] and u["theme"] in themes:
            p = 2
        elif u["time"] and times and min(_days(u["time"], t) for t in times) <= 60:
            p = 3
        return (p, K.frac(key + ":" + u["uid"]))
    rest = sorted((u for u in units if u["uid"] not in gset and u["uid"] not in exclude), key=prio)
    chosen = list(req)
    for u in rest:
        if used + u["tokens"] > budget:
            continue
        chosen.append(u)
        used += u["tokens"]
    kind_order = {"profile": 0, "social": 1, "event": 2, "dialogue": 2}
    chosen.sort(key=lambda u: (u["time"] is not None, kind_order[u["kind"]], u["time"] or "", u["uid"]))
    return chosen


def perltqa(raw: Path, tok: Tok) -> Dict[str, list]:
    P = json.load(open(raw / "perltqa/Dataset/zh/perltqa.json", encoding="utf-8"))
    PM = {m["profile"]["Protagonist"]: m for m in json.load(open(raw / "perltqa/Dataset/zh/perltmem.json", encoding="utf-8"))}
    persons = sorted({n for x in P for n in x})
    order = sorted(persons, key=lambda n: K.frac("perltqa-person:" + n))
    split = {n: ("train" if i < 24 else "dev" if i < 28 else "holdout") for i, n in enumerate(order)}
    out, stats = [], collections.Counter()
    for x in P:
        for name, d in x.items():
            units = perltqa_units(PM[name], tok)
            uids = {u["uid"] for u in units}
            for typ in PERLT_TYPES:
                v = d.get(typ) or []
                recs = v if isinstance(v, list) else [r for rs in v.values() for r in rs]
                for r in recs:
                    ref = r["Reference Memory"]
                    gold = re.findall(r"[\d_#]+", ref) if ref.startswith("[") else ["PROFILE"]
                    stats["qa"] += 1
                    if not gold or any(g not in uids for g in gold):
                        stats["drop_unresolved_ref"] += 1
                        continue
                    rid = "perltqa:" + K.h(f"{name}|{typ}|{ref}|{r['Question']}|{r['Answer']}")
                    anchors = [k for a in r.get("Memory Anchors", []) for k in a if len(K.norm(k)) >= 2]
                    out.append({"id": rid, "source": "perltqa", "split": split[name], "group": "perltqa:" + name, "qtype": typ, "question": r["Question"],
                                "answer": r["Answer"], "gold_uids": gold, "anchors": anchors, "_units": units, "person": name,
                                "provenance": {"dataset": "PerLTQA zh", "person": name, "reference_memory": ref}})
    return {"rows": out, "stats": dict(stats), "split_persons": {s: sorted(n for n in persons if split[n] == s) for s in ("train", "dev", "holdout")}}


def perltqa_support(c: dict) -> dict:
    by = {u["uid"]: u for u in c["_units"]}
    gtext = "\n".join(by[g]["text"] for g in c["gold_uids"])
    ng = K.norm(gtext)
    anchor_hit = [a for a in c["anchors"] if K.norm(a) in ng]
    ab = bigrams(c["answer"]) - bigrams(c["question"])
    cov = len([b for b in ab if b in ng]) / len(ab) if ab else 0.0
    quotes = []
    for s in sentences(gtext):
        if any(K.norm(a) in K.norm(s) for a in anchor_hit) and not s.startswith("["):
            quotes.append(s)
    return {"anchor_hits": len(anchor_hit), "answer_bigram_coverage": round(cov, 3), "supported": bool(anchor_hit) or cov >= 0.5,
            "quotes": quotes[:2], "answer_specific_bigrams": len(ab)}


def perltqa_finish(c: dict, units: List[dict], budget: int, unanswerable: bool = False) -> dict:
    """Materialise the FactPack (memory header, items in time order) for a chosen unit list."""
    items = [mk_item(len(units), i + 1, u["text"], time=u["time"], label="个人记忆", unit=u["uid"]) for i, u in enumerate(units)]
    gold = [] if unanswerable else [it.eid for it, u in zip(items, units) if u["uid"] in c["gold_uids"]]
    return {"pack": mk_pack(items, c["question"]), "header": K.HEADER_MEMORY, "gold_eids": gold}


def verify_unanswerable(c: dict, units: List[dict]) -> dict:
    """Deterministic check that the answer is NOT recoverable from the remaining (visible) records after removing the reference memory
    and its linked records. VERIFIED only when every test passes; doubt -> AMBIGUOUS (never used)."""
    vis = "\n".join(u["text"] for u in units)
    nv = K.norm(vis)
    reasons = []
    anchors_left = [a for a in c["anchors"] if K.norm(a) in nv]
    if anchors_left:
        reasons.append(f"anchor still visible ({len(anchors_left)})")
    ab = bigrams(c["answer"]) - bigrams(c["question"])
    if not ab:
        return {"status": "AMBIGUOUS", "reasons": ["answer adds nothing beyond the question text"], "max_unit_coverage": None}
    per_unit = max((len([b for b in ab if b in K.norm(u["text"])]) / len(ab) for u in units), default=0.0)
    union = len([b for b in ab if b in nv]) / len(ab)
    if K.norm(c["answer"]) and K.norm(c["answer"]) in nv:
        reasons.append("answer string visible")
    if per_unit >= 0.5:
        reasons.append(f"one record covers {per_unit:.2f} of the answer-specific bigrams")
    if reasons:
        status = "REJECTED"
    elif per_unit >= 0.25 or union >= 0.5:
        status, reasons = "AMBIGUOUS", [f"partial coverage: unit {per_unit:.2f}, union {union:.2f}"]
    else:
        status = "VERIFIED"
    return {"status": status, "reasons": reasons, "max_unit_coverage": round(per_unit, 3), "union_coverage": round(union, 3)}


# ------------------------------------------------------------------ LoCoMo (en)
LOCOMO_CATS = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial_unanswerable"}


def _locomo_time(s):
    try:
        return datetime.strptime(s.strip(), "%I:%M %p on %d %B, %Y").isoformat(timespec="minutes")
    except Exception:
        return None


def locomo(raw: Path, tok: Tok) -> Dict[str, list]:
    L = json.load(open(raw / "locomo/data/locomo10.json", encoding="utf-8"))
    ids = sorted(s["sample_id"] for s in L)
    order = sorted(ids, key=lambda i: K.frac("locomo-conv:" + i))
    split = {c: ("train" if i < 7 else "dev" if i < 8 else "holdout") for i, c in enumerate(order)}
    out, stats = [], collections.Counter()
    for s in L:
        conv = s["conversation"]
        sess = sorted([k for k in conv if re.fullmatch(r"session_\d+", k)], key=lambda k: int(k.split("_")[1]))
        sessions = []
        for k in sess:
            t = _locomo_time(conv.get(k + "_date_time", ""))
            turns = []
            for d in conv[k]:
                txt = f"{d['speaker']}：{d['text']}" + (f"（分享图片：{d['blip_caption']}）" if d.get("blip_caption") else "")
                turns.append({"dia": d["dia_id"], "text": txt, "raw": d["text"], "tokens": tok.n(rendered(txt, t, "对话", wide=True)) + 1})
            sessions.append({"sid": k, "idx": int(k.split("_")[1]), "time": t, "turns": turns, "tokens": sum(x["tokens"] for x in turns)})
        dia_sess = {t["dia"]: se["sid"] for se in sessions for t in se["turns"]}
        for qi, qa in enumerate(s["qa"]):
            cat = qa["category"]
            stats[f"cat_{LOCOMO_CATS[cat]}"] += 1
            if cat == 3:
                stats["drop_open_domain"] += 1
                continue
            ev = [e for x in qa.get("evidence", []) for e in re.split(r"[;,\s]+", str(x)) if e]
            if not ev or any(e not in dia_sess for e in ev):
                stats["drop_evidence_unresolved"] += 1
                continue
            answerable = cat != 5
            ans = qa.get("answer") if answerable else None
            if answerable and (ans is None or str(ans).strip() == ""):
                stats["drop_no_answer"] += 1
                continue
            rid = f"locomo:{s['sample_id']}#{qi}"
            out.append({"id": rid, "source": "locomo", "split": split[s["sample_id"]], "group": "locomo:" + s["sample_id"], "qtype": LOCOMO_CATS[cat],
                        "question": qa["question"], "answer": None if ans is None else str(ans), "answerable": answerable, "evidence_dia": ev,
                        "gold_sessions": sorted({dia_sess[e] for e in ev}), "_sessions": sessions,
                        "provenance": {"dataset": "LoCoMo", "sample_id": s["sample_id"], "qa_index": qi, "category": cat}})
    return {"rows": out, "stats": dict(stats), "split_conversations": {k: sorted(c for c in ids if split[c] == k) for k in ("train", "dev", "holdout")}}


def locomo_pack(c: dict, budget: int) -> Optional[dict]:
    S = {se["sid"]: se for se in c["_sessions"]}
    gold = [S[g] for g in c["gold_sessions"]]
    used = sum(x["tokens"] for x in gold)
    if used > budget:
        return None
    gidx = [g["idx"] for g in gold]
    rest = sorted((se for se in c["_sessions"] if se["sid"] not in c["gold_sessions"]),
                  key=lambda se: (min(abs(se["idx"] - i) for i in gidx), K.frac(c["id"] + se["sid"])))
    chosen = list(gold)
    for se in rest:
        if used + se["tokens"] > budget:
            continue
        chosen.append(se)
        used += se["tokens"]
    chosen.sort(key=lambda se: se["idx"])
    turns = [(se, t) for se in chosen for t in se["turns"]]
    items = [mk_item(len(turns), i + 1, t["text"], time=se["time"], label="对话", unit=t["dia"]) for i, (se, t) in enumerate(turns)]
    gold_eids = [it.eid for it, (se, t) in zip(items, turns) if t["dia"] in c["evidence_dia"]]
    quotes = [t["raw"] for se, t in turns if t["dia"] in c["evidence_dia"]]
    return {"pack": mk_pack(items, c["question"]), "header": K.HEADER_MEMORY, "gold_eids": gold_eids if c["answerable"] else [],
            "quotes": quotes, "support_text": "\n".join(t["text"] for se, t in turns if t["dia"] in c["evidence_dia"])}


# ------------------------------------------------------------------ HotpotQA (en, official train split, distractor setting)
def hotpot(rows: List[dict]) -> Dict[str, list]:
    out, stats, seen = [], collections.Counter(), set()
    for r in rows:
        stats["rows"] += 1
        if r["id"] in seen:
            stats["drop_duplicate_id"] += 1
            continue
        seen.add(r["id"])
        titles, sents = r["context"]["title"], r["context"]["sentences"]
        sf = list(zip(r["supporting_facts"]["title"], r["supporting_facts"]["sent_id"]))
        gt = sorted({t for t, _ in sf})
        if any(t not in titles for t in gt):
            stats["drop_gold_paragraph_missing"] += 1
            continue
        quotes = []
        for t, k in sf:
            ss = sents[titles.index(t)]
            if k < len(ss):
                quotes.append(ss[k].strip())
            else:
                quotes = None
                break
        if quotes is None:
            stats["drop_bad_sentence_id"] += 1
            continue
        items = [mk_item(len(titles), i + 1, f"{t}：{''.join(s).strip()}", time=None, label="维基百科段落", unit=t) for i, (t, s) in enumerate(zip(titles, sents))]
        gold = [it.eid for it in items if it.unit_key in gt]
        ans = r["answer"]
        yn = K.norm(ans) in ("yes", "no")
        if not yn and K.norm(ans) not in K.norm(" ".join(quotes)) and K.norm(ans) not in K.norm(" ".join(it.text for it in items if it.eid in gold)):
            stats["drop_answer_not_in_gold"] += 1
            continue
        out.append({"id": "hotpotqa:" + r["id"], "source": "hotpotqa", "split": split_of("hotpot:" + r["id"]), "group": "hotpotqa:" + r["id"],
                    "qtype": f"{r['type']}/{r['level']}" + ("/yes_no" if yn else ""), "question": r["question"], "answer": ans, "answerable": True,
                    "pack": mk_pack(items, r["question"]), "header": K.HEADER_DOC, "gold_eids": gold, "quotes": quotes, "titles": titles,
                    "provenance": {"dataset": "HotpotQA distractor (train split)", "id": r["id"]}})
    return {"rows": out, "stats": dict(stats)}


# ------------------------------------------------------------------ QASPER (en, train split, grouped by paper)
def qasper(raw: Path, tok: Tok) -> Dict[str, list]:
    import pyarrow.parquet as pq
    papers = pq.read_table(raw / "qasper/train-0000.parquet").to_pylist()
    out, stats = [], collections.Counter()
    for p in papers:
        paras = [(sec, x.strip()) for sec, ps in zip(p["full_text"]["section_name"], p["full_text"]["paragraphs"]) for x in ps if x and x.strip()]
        ptoks = [tok.n(f"〔{sec}〕{x}") + 10 for sec, x in paras]
        sp = split_of("qasper-paper:" + p["id"])
        for k, q in enumerate(p["qas"]["question"]):
            stats["questions"] += 1
            anns = p["qas"]["answers"][k]["answer"]
            if len({bool(a["unanswerable"]) for a in anns}) > 1:
                stats["drop_annotators_disagree_answerability"] += 1
                continue
            a = anns[0]
            unans = bool(a["unanswerable"])
            evs = [e.strip() for e in a["evidence"] if e and e.strip()]
            if any(e.startswith("FLOAT SELECTED") for e in evs):
                stats["drop_table_or_figure_evidence"] += 1
                continue
            gold_idx = sorted({i for i, (sec, x) in enumerate(paras) for e in evs if e == x or (len(e) > 40 and e in x)})
            if not unans:
                if not evs or len(gold_idx) < len(set(evs)):
                    stats["drop_evidence_unmapped"] += 1
                    continue
                if a["yes_no"] is not None:
                    ans, qt = ("Yes" if a["yes_no"] else "No"), "yes_no"
                elif a["extractive_spans"]:
                    ans, qt = "; ".join(s.strip() for s in a["extractive_spans"]), "extractive"
                elif (a["free_form_answer"] or "").strip():
                    ans, qt = a["free_form_answer"].strip(), "abstractive"
                else:
                    stats["drop_empty_answer"] += 1
                    continue
            else:
                ans, qt = None, "unanswerable"
            out.append({"id": "qasper:" + p["qas"]["question_id"][k], "source": "qasper", "split": sp, "group": "qasper:" + p["id"], "qtype": qt,
                        "question": q, "answer": ans, "answerable": not unans, "_paras": paras, "_ptoks": ptoks, "gold_idx": gold_idx,
                        "quotes": [x for x in (a.get("highlighted_evidence") or []) if x and not x.startswith("FLOAT")],
                        "provenance": {"dataset": "QASPER (train split)", "paper_id": p["id"], "question_id": p["qas"]["question_id"][k]}})
    return {"rows": out, "stats": dict(stats)}


def qasper_pack(c: dict, budget: int) -> Optional[dict]:
    """Whole paper in original order; if too long, drop the non-gold paragraphs farthest from any gold paragraph (never a gold one)."""
    n = len(c["_paras"])
    keep = set(range(n))
    total = sum(c["_ptoks"])
    anchor = c["gold_idx"] or [0]
    far = sorted((i for i in range(n) if i not in c["gold_idx"]), key=lambda i: -min(abs(i - g) for g in anchor))
    for i in far:
        if total <= budget:
            break
        keep.discard(i)
        total -= c["_ptoks"][i]
    if total > budget:
        return None
    idx = sorted(keep)
    items = [mk_item(len(idx), j + 1, f"〔{c['_paras'][i][0]}〕{c['_paras'][i][1]}", time=None, label="论文段落", unit=f"p{i}") for j, i in enumerate(idx)]
    gold = [it.eid for it, i in zip(items, idx) if i in c["gold_idx"]] if c["answerable"] else []
    return {"pack": mk_pack(items, c["question"]), "header": K.HEADER_DOC, "gold_eids": gold, "trimmed_paragraphs": n - len(idx),
            "support_text": "\n".join(c["_paras"][i][1] for i in c["gold_idx"])}


# ------------------------------------------------------------------ D4 synthetic (zh, REUSE_CANDIDATE only)
D4_CATS = ("CONFLICT", "CORRECTION", "CANCELLATION", "TEMPORAL_CHANGE", "PARTIAL_UNKNOWN", "KNOWN_FACT", "FULL_UNKNOWN")
STATUS_CN = {"CURRENT": "当前有效", "PAST": "过去", "CORRECTED": "已更正", "CANCELLED": "已取消", "PLANNED": "计划中"}
STYLE_TOKENS = re.compile(r"哈哈|hhh|orz|emm|害|整挺好|吧！|哦！|嘿|嘻|～|~|\(.*[◡ω].*\)|呗|啦")


def d4(d4dir: Path, labels: Path) -> Dict[str, list]:
    lab = {}
    for l in open(labels, encoding="utf-8"):
        o = json.loads(l)
        lab[o["episode_id"]] = o["label"]
    out, stats = [], collections.Counter()
    personas = set()
    rows = [json.loads(l) for l in open(Path(d4dir) / "train_meta.jsonl", encoding="utf-8")]
    for r in rows:
        if lab.get(r["episode_id"]) != "REUSE_CANDIDATE" or r["category"] not in D4_CATS:
            continue
        stats["reuse_in_scope"] += 1
        personas.add(r["persona_id"])
        plan = "；".join(r.get("content_plan") or []).strip()
        unknown = r["category"] == "FULL_UNKNOWN"
        if not unknown and (not plan or STYLE_TOKENS.search(plan)):
            stats["drop_plan_empty_or_styled"] += 1
            continue
        ev = sorted(r["evidence"], key=lambda e: (e.get("observed_at") or "", e["fact_id"]))
        if not ev and not unknown:
            stats["drop_no_evidence"] += 1
            continue
        items = [mk_item(len(ev), i + 1, f"关于{e['subject']}：{e['evidence_text']}（状态：{STATUS_CN.get(e['status'], e['status'])}）",
                         time=e.get("observed_at"), label="记忆", unit=e["fact_id"]) for i, e in enumerate(ev)]
        req = r["answer_state"].get("required_values") or []
        gold = [it.eid for it, e in zip(items, ev) if any(K.norm(v) and K.norm(v) in K.norm(e["evidence_text"]) for v in req)]
        if not gold:
            gold = [it.eid for it, e in zip(items, ev) if K.norm(e.get("value")) and K.norm(e.get("value")) in K.norm(plan)]
        if not unknown and not gold:
            stats["drop_gold_unmapped"] += 1
            continue
        out.append({"id": "d4:" + r["episode_id"], "source": "d4", "group": "d4:" + r["persona_id"], "persona": r["persona_id"], "qtype": r["category"],
                    "question": r["conversation"][-1]["text"], "answer": None if unknown else plan, "answerable": not unknown,
                    "pack": mk_pack(items, r["conversation"][-1]["text"], ask_time=r["now"]), "header": K.HEADER_MEMORY,
                    "gold_eids": [] if unknown else gold, "quotes": [],
                    "provenance": {"dataset": "t2d4-data-v1.1 TRAIN (synthetic, REUSE_CANDIDATE)", "episode_id": r["episode_id"], "category": r["category"]}})
    order = sorted(personas, key=lambda p: K.frac("d4-persona:" + p))
    n = len(order)
    sp = {p: ("train" if i < int(n * 0.8) else "dev" if i < int(n * 0.9) else "holdout") for i, p in enumerate(order)}
    for c in out:
        c["split"] = sp[c["persona"]]
    return {"rows": out, "stats": dict(stats), "personas": {k: sum(1 for p in order if sp[p] == k) for k in ("train", "dev", "holdout")}}


def hotpot_rows(raw: Path, cache: Path) -> List[dict]:
    """Phase 0 sample (60 rows) + one more seeded row group per official TRAIN parquet file (range-read once, cached)."""
    rows = [json.loads(l)["row"] for l in open(raw / "hotpotqa/train_sample.jsonl", encoding="utf-8")]
    if not cache.exists():
        import io
        import pyarrow.parquet as pq
        import urllib.request
        base = "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/1908d6afbbead072334abe2965f91bd2709910ab/"

        class RF(io.RawIOBase):
            def __init__(s, u):
                s.u, s.pos = u, 0
                s.size = int(urllib.request.urlopen(urllib.request.Request(u, method="HEAD", headers={"User-Agent": "dataset-build/1.0"}), timeout=60).headers["Content-Length"])

            def seekable(s):
                return True

            def readable(s):
                return True

            def tell(s):
                return s.pos

            def seek(s, o, w=0):
                s.pos = {0: o, 1: s.pos + o, 2: s.size + o}[w]
                return s.pos

            def read(s, n=-1):
                n = s.size - s.pos if n is None or n < 0 else n
                if not n:
                    return b""
                b = urllib.request.urlopen(urllib.request.Request(s.u, headers={"User-Agent": "dataset-build/1.0", "Range": f"bytes={s.pos}-{s.pos + n - 1}"}), timeout=180).read()
                s.pos += len(b)
                return b

            def readinto(s, b):
                d = s.read(len(b))
                b[:len(d)] = d
                return len(d)
        extra, picks = [], []
        rnd = random.Random(K.SEED + ":hotpot-rowgroups")
        for f in ("distractor/train-00000-of-00002.parquet", "distractor/train-00001-of-00002.parquet"):
            pf = pq.ParquetFile(RF(base + f))
            g = rnd.randrange(pf.metadata.num_row_groups)
            picks.append([f, g])
            extra += pf.read_row_group(g).to_pylist()
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"row_groups": picks, "revision": "1908d6afbbead072334abe2965f91bd2709910ab"}) + "\n")
            for r in extra:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    lines = open(cache, encoding="utf-8").read().splitlines()
    return rows + [json.loads(l) for l in lines[1:]]
