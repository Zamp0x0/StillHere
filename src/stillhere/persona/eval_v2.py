"""Second-round Colab evaluation (needs the r2 adapter): r1 vs r2 style, base-capability retention, Memory V1 vs V2 (router, evidence
windows, local dense / hybrid retrieval), the Writer Full Build Gate, and the DeepSeek judge on the SAME rubric as FAST_EVAL.

Frozen inputs are reused, never rebuilt: the FAST real-history QA (100 questions, sha ``01af8c54…``), the FAST 100-item DEV style sample,
the FAST 20 Writer audit sessions, and the Memory V2 DEV / HOLDOUT split frozen on 2026-11-18 (``scripts/freeze_baselines.py``).
Every group writes ``answers.jsonl`` in the FAST layout, so ``colab_eval`` judge units / taxonomy / metrics are reused unchanged.

Decision rules (registered 2026-11-18, before r2 exists; ``R2_RULE`` / ``MEMORY_V2_RULE`` / ``writer_gate``) are applied by ``analysis``;
the result is a recommendation, nothing local changes automatically.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from . import colab_eval as CE
from . import evaluation as EV
from . import judge as J
from . import qa_builder as QB
from . import store as S

SCHEMA = "stillhere.persona.eval_v2/1.0"
T10 = dict(EV.DECODE, temperature=1.0, top_p=0.95, top_k=0)              # exploratory decoding (r1 analysis: sampling sharpens the mode)
STYLE_SPECS: Dict[str, dict] = {
    "A": {"prompt": "neutral", "adapter": None},                            # Base
    "B1": {"prompt": "skill", "adapter": None},                             # Base + Persona Skill (the runtime persona prompt)
    "R1T": {"prompt": "sft", "adapter": "r1"},                              # r1 with the prompt it was trained on
    "R1S": {"prompt": "skill", "adapter": "r1"},                            # r1 + Skill (= FAST C1)
    "R2S": {"prompt": "skill", "adapter": "r2"},                            # r2 + Skill (r2 was trained with the Skill prompt)
    "R1S_T10": {"prompt": "skill", "adapter": "r1", "decode": "t10"},
    "R2S_T10": {"prompt": "skill", "adapter": "r2", "decode": "t10"},
}
GENERAL_SPECS: Dict[str, dict] = {"GA": {"prompt": "neutral", "adapter": None}, "GB1": {"prompt": "skill", "adapter": None},
                                  "GR1": {"prompt": "skill", "adapter": "r1"}, "GR2": {"prompt": "skill", "adapter": "r2"}}
MEMORY_SPECS: Dict[str, dict] = {
    "M0": {"memory": None, "pipeline": "v1", "adapter": None},
    "M1": {"memory": "production", "pipeline": "v1", "adapter": None},                       # the FAST M1 path (+ the header-scope fix)
    "V2L": {"memory": "production", "pipeline": "v2", "dense": False, "adapter": None},     # router + windows + date prefer, lexical
    "V2D": {"memory": "production", "pipeline": "v2", "dense": True, "adapter": None},      # + local dense (hybrid RRF)
    "V2D_R2": {"memory": "production", "pipeline": "v2", "dense": True, "adapter": "r2"},
    # budget-matched control (added 2026-11-18 after the DEV fairness audit, before any answer was generated): about the same number of
    # shown messages as V2D (40 anchors, no windows). DEV recall: windows ~ more anchors, so only answer quality can show whether
    # coherent windows help the reader. Reported as V2D-V2D_K40; NOT part of MEMORY_V2_RULE.
    "V2D_K40": {"memory": "production", "pipeline": "v2", "dense": True, "adapter": None, "k": 40, "neighbors": (0, 0)},
}
STRUCTURED_GROUP = "V2D_S"                                                  # only when the Writer gate allowed a full build
UNSEEN_THRESHOLD = 0.2
UNSEEN_RULE = ("pre-registered: a DEV item is UNSEEN when the char-3gram Jaccard similarity of its latest partner line with EVERY train "
               f"context's latest line is < {UNSEEN_THRESHOLD}; otherwise SEEN")
R2_RULE = ("pre-registered 2026-11-18: r2 is BETTER than r1 iff (1) the paired style score R2S-R1S has a 95% CI above 0, (2) Style V2 "
           "no_conditional_collapse passes for R2S, (3) general-knowledge accuracy GR2 >= GR1 - 0.05, and (4) memory V2D_R2-V2D has a 95% CI "
           "lower bound > -2 points with a fabricated-private-fact rate <= V2D + 0.02; r2 is RECOMMENDED for the runtime only if additionally "
           "R2S-B1 has a CI above 0 and (2) holds; otherwise the adapter stays off")
MEMORY_V2_RULE = ("pre-registered 2026-11-18, decided on the HOLDOUT questions only: adopt V2 as the default memory pipeline iff the chosen "
                  "V2 group minus M1 has a paired memory-score CI lower bound > -2 points (non-inferior), a mean > 0, a grounded-answer accuracy "
                  "not lower and a fabricated-private-fact rate not higher by more than 0.02; the chosen V2 group is V2D iff V2D-V2L has a CI "
                  "lower bound > 0 on HOLDOUT, else the simpler V2L")
GENERAL_FILE = "examples/synthetic/general_knowledge_zh_v1.jsonl"


def defaults(cfg: Mapping) -> dict:
    base = {"eval_mode": "V2", "style_groups": list(STYLE_SPECS), "memory_groups": list(MEMORY_SPECS), "general_groups": list(GENERAL_SPECS),
            "style_n": 100, "seed": 20261117, "writer_audit_n": 20, "memory_build": "evidence_only", "allow_full_build": False,
            "judge_concurrency": 4, "judge_per_second": 4.0, "accept_auto_verified_by": "owner (second Colab run)", "writer_max_new_tokens": 1536,
            "dense_device": "cuda", "dense_batch": 64, "case_min": 15, "case_max": 20}
    return {**base, **dict(cfg)}


# ---------------------------------------------------------------- frozen inputs
def _put(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"FROZEN_INPUT_CHANGED: {path} exists with different content")
    else:
        path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def seed_frozen(run: CE.Run, *, fast_zip: Optional[Path] = None, fast_run_dir: Optional[Path] = None, split: Optional[Mapping] = None,
                expect_qa_sha256: Optional[str] = None) -> dict:
    """Copy the frozen FAST inputs into this run (never overwritten with different bytes). ``split``: {"dev": [...], "holdout": [...]}."""
    R = Path(run.run_dir)
    if fast_zip:
        with zipfile.ZipFile(fast_zip) as z:
            name = lambda suffix: next(n for n in z.namelist() if n.endswith(suffix))          # noqa: E731
            qa = z.read(name("MEMORY_QA_100.jsonl"))
            qb = z.read(name("QA_BUILD_REPORT.json"))
            ss = z.read(name("STYLE_SAMPLE_MANIFEST.json"))
            wa = [json.loads(l) for l in z.read(name("WRITER_AUDIT_20.jsonl")).decode("utf-8").splitlines() if l.strip()]
        sample = {"seed": run.cfg["seed"], "n": len(wa), "strata": dict(Counter(w["stratum"] for w in wa)),
                  "sessions": [{"conversation_id": w["conversation_id"], "stratum": w["stratum"]} for w in wa], "source": "FAST WRITER_AUDIT_20"}
        ws = json.dumps(sample, ensure_ascii=False, indent=1).encode("utf-8")
    elif fast_run_dir:
        F = Path(fast_run_dir)
        qa, qb = (F / "qa" / "REAL_HISTORY_QA_200.jsonl").read_bytes(), (F / "qa" / "QA_BUILD_REPORT.json").read_bytes()
        ss, ws = (F / "reports" / "STYLE_SAMPLE_MANIFEST.json").read_bytes(), (F / "writer_audit" / "WRITER_AUDIT_SAMPLE.json").read_bytes()
    else:
        raise ValueError("seed_frozen needs fast_zip or fast_run_dir")
    qsha = _put(R / "qa" / "REAL_HISTORY_QA_200.jsonl", qa)
    if expect_qa_sha256 and qsha != expect_qa_sha256:
        raise RuntimeError("QA_SET_CHANGED: not the frozen FAST question set")
    _put(R / "qa" / "QA_FROZEN.sha256", f"{qsha}  REAL_HISTORY_QA_200.jsonl\n".encode())
    _put(R / "qa" / "QA_BUILD_REPORT.json", qb)
    _put(R / "reports" / "STYLE_SAMPLE_MANIFEST.json", ss)
    _put(R / "writer_audit" / "WRITER_AUDIT_SAMPLE.json", ws)
    ids = [json.loads(l)["question_id"] for l in qa.decode("utf-8").splitlines() if l.strip()]
    sp = {"dev": sorted(split["dev"]), "holdout": sorted(split["holdout"])} if split else {"dev": sorted(ids), "holdout": []}
    _put(R / "reports" / "MEMORY_V2_SPLIT.json", json.dumps(sp, indent=1).encode("utf-8"))
    run.status("V2 STAGE 06 frozen inputs", "PASS", qa_sha256=qsha, qa=len(ids), dev=len(sp["dev"]), holdout=len(sp["holdout"]))
    return {"qa_sha256": qsha, "questions": len(ids), "split": {k: len(v) for k, v in sp.items()}}


# ---------------------------------------------------------------- prompts
def _skill_card(run: CE.Run):
    from . import card as K
    return K.load(run.pdir, run.manifest["prompts"]["old"]["card_id"])


def prompt_text(run: CE.Run, kind: str) -> str:
    from . import dataset as D
    return {"neutral": EV.NEUTRAL_PROMPT, "sft": D.SYSTEM_PROMPT, "skill": _skill_card(run)["prompt"]}[kind]


def _adapter_id(run: CE.Run, key: Optional[str]) -> Optional[str]:
    return None if key is None else run.manifest["adapters"][key]["adapter_id"]


# ---------------------------------------------------------------- style (+ unseen strata)
def style_items_v2(run: CE.Run) -> List[dict]:
    items = CE.dev_items(run)
    for it in items:
        it["messages"] = {g: [{"role": "system", "content": prompt_text(run, s["prompt"])}, {"role": "user", "content": it["user"]}]
                          for g, s in STYLE_SPECS.items() if g in run.cfg["style_groups"]}
        it["prompt_hash"] = {g: EV.sha_obj({"m": m, "decode": STYLE_SPECS[g].get("decode")}) for g, m in it["messages"].items()}
    return items


def stage_style(run: CE.Run) -> dict:
    items = style_items_v2(run)
    groups = run.cfg["style_groups"]
    fp = {"base": run.manifest["base"], "adapters": {k: v["sha256"] for k, v in run.manifest["adapters"].items()}, "skill_card": run.manifest["prompts"]["old"],
          "decode": EV.DECODE, "t10": T10, "dataset_version": run.manifest["dataset_version"], "items": len(items), "groups": groups}
    summary = EV.generate(run.gen, items, groups, Path(run.run_dir) / "style", fp, adapters={g: _adapter_id(run, STYLE_SPECS[g]["adapter"]) for g in groups},
                          decodes={g: T10 for g in groups if STYLE_SPECS[g].get("decode") == "t10"}, log=run.log)
    ok = all(v["ok"] == v["of"] for v in summary.values())
    run.status("V2 STAGE 12 style generation", "PASS" if ok else "INCOMPLETE", **{g: f"{v['ok']}/{v['of']}" for g, v in summary.items()})
    return summary


def _grams(s: str) -> set:
    from . import dataset as D
    t = D.norm(s)
    return {t[i:i + 3] for i in range(max(1, len(t) - 2))}


def unseen_strata(run: CE.Run) -> dict:
    from . import dataset as D
    from . import package as P
    ddir = P.latest_dataset(run.pdir)
    last = lambda ctx: re.sub(r"^(对方|我)：", "", (ctx.strip().splitlines() or [""])[-1])          # noqa: E731
    tg = [_grams(last(r["conversations"][0]["value"])) for r in D.load_split(ddir, "train", "training")]
    out = {}
    for it in CE.dev_items(run):
        g = _grams(re.sub(r"^(对方|我)：", "", it.get("query") or ""))
        best = max((len(g & h) / max(1, len(g | h)) for h in tg), default=0.0)
        out[it["sample_id"]] = "UNSEEN" if best < UNSEEN_THRESHOLD else "SEEN"
    res = {"rule": UNSEEN_RULE, "counts": dict(Counter(out.values())), "items": out}
    CE._write(run.p("reports", "STYLE_UNSEEN_STRATA.json"), res)
    return res


# ---------------------------------------------------------------- base capability retention (public questions, keyword scoring)
def general_items(run: CE.Run) -> List[dict]:
    path = Path(run.manifest.get("code_dir", "code")) / GENERAL_FILE
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    items = []
    for r in rows:
        it = {"sample_id": r["id"], "keywords": r["keywords"], "question": r["question"]}
        it["messages"] = {g: [{"role": "system", "content": prompt_text(run, s["prompt"])}, {"role": "user", "content": f"对方：{r['question']}"}]
                          for g, s in GENERAL_SPECS.items() if g in run.cfg["general_groups"]}
        it["prompt_hash"] = {g: EV.sha_obj(m) for g, m in it["messages"].items()}
        items.append(it)
    return items


def keyword_ok(text: str, keywords) -> bool:
    t = re.sub(r"\s+", "", (text or "")).lower()
    return any(k.lower() in t for k in keywords)


def stage_general(run: CE.Run) -> dict:
    items = general_items(run)
    groups = run.cfg["general_groups"]
    fp = {"file": GENERAL_FILE, "n": len(items), "groups": groups, "decode": EV.DECODE}
    EV.generate(run.gen, items, groups, Path(run.run_dir) / "general", fp, adapters={g: _adapter_id(run, GENERAL_SPECS[g]["adapter"]) for g in groups}, log=run.log)
    acc = {}
    per = {}
    for g in groups:
        rows = EV.done_rows(Path(run.run_dir) / "general" / g / "answers.jsonl")
        per[g] = {it["sample_id"]: 1.0 if keyword_ok(rows.get(it["sample_id"], {}).get("text"), it["keywords"]) else 0.0 for it in items}
        acc[g] = round(sum(per[g].values()) / len(items), 4)
    paired = {f"{b}-{a}": J.paired(per[a], per[b]) for a, b in (("GB1", "GR1"), ("GB1", "GR2"), ("GR1", "GR2"), ("GA", "GB1")) if a in per and b in per}
    res = {"accuracy": acc, "paired": paired, "n": len(items), "scoring": "keyword match (public facts), no judge"}
    CE._write(run.p("reports", "GENERAL_CAPABILITY_REPORT.json"), res)
    run.status("V2 STAGE 12b general capability", "PASS", **acc)
    return res


# ---------------------------------------------------------------- dense index (rebuilt on the Colab GPU from the packaged archive)
def dense_dirs(run: CE.Run, local_dir: Optional[Path] = None):
    """(working dir, persisted dir). With ``local_dir`` (the Colab /content disk) the index and its WRITABLE SQLite query cache live on
    local disk and ``<run>/dense_index`` on Drive only keeps a finished snapshot (V3: 28 / 35 K40 answers failed with 'disk I/O error' on a
    Drive-hosted SQLite cache). Without it, both are ``<run>/dense_index`` (the old layout)."""
    persisted = Path(run.run_dir) / "dense_index"
    return (Path(local_dir) if local_dir else persisted), persisted


def _index_done(d: Path) -> bool:
    return (d / "index" / "embedding_index_manifest.json").is_file()


def stage_dense(run: CE.Run, model_dir: Path, local_dir: Optional[Path] = None) -> dict:
    from stillhere.memory.api import MemoryAPI
    from stillhere.retrieval import local_embedding as LE
    from stillhere.retrieval.ef1 import local_dense as LD
    from . import memory_store as MS
    out, persisted = dense_dirs(run, local_dir)
    if local_dir and not _index_done(out) and _index_done(persisted):         # resume: the Drive snapshot back to local disk
        shutil.rmtree(out, ignore_errors=True)
        shutil.copytree(persisted, out, ignore=shutil.ignore_patterns("query_cache_*", "*-wal", "*-shm"))
    if _index_done(out):
        ch, why = LD.open_local(out, model=LE.CACHE_MODEL_NAME, dimension=LE.EXPECTED_DIMENSION)
        if ch is not None:
            run.status("V2 STAGE 07b dense index", "DONE_BEFORE", working_dir=str(out))
            return {"status": "DONE_BEFORE"}
    api = MemoryAPI(str(run.pdir / "memory" / "memory.sqlite"), public_only=False, persona_names={run.profile_id: ["TA"]})
    try:
        emb = LE.get_embedder(model_dir, device=run.cfg["dense_device"])
        res = LD.build_local(api, run.profile_id, out, emb, names={run.profile_id: "本人", MS.other_id(run.profile_id): "对方"},
                             model=LE.CACHE_MODEL_NAME, dimension=LE.EXPECTED_DIMENSION, batch_size=int(run.cfg["dense_batch"]))
    finally:
        api.close()
    res.update(model_id=LE.MODEL_ID, model_revision=LE.MODEL_REVISION, device=run.cfg["dense_device"], embedder=emb.stats.to_dict(), working_dir=str(out))
    if local_dir and res.get("status") == "COMPLETE":                          # persist a finished snapshot (atomic rename on Drive)
        tmp = persisted.with_name(persisted.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(out, tmp, ignore=shutil.ignore_patterns("query_cache_*", "*-wal", "*-shm"))
        shutil.rmtree(persisted, ignore_errors=True)
        tmp.rename(persisted)
    CE._write(run.p("reports", "DENSE_INDEX_REPORT.json"), res)
    run.status("V2 STAGE 07b dense index", "PASS" if res["status"] == "COMPLETE" else "INCOMPLETE", documents=res["documents"])
    return res


def dense_channel(run: CE.Run, model_dir: Path, local_dir: Optional[Path] = None):
    from stillhere.retrieval import local_embedding as LE
    from stillhere.retrieval.ef1 import local_dense as LD
    enc = LE.LocalQueryEncoder(LE.get_embedder(model_dir, device=run.cfg["dense_device"]))
    ch, why = LD.open_local(dense_dirs(run, local_dir)[0], model=LE.CACHE_MODEL_NAME, dimension=LE.EXPECTED_DIMENSION, query_encoder=enc)
    return ch, why


# ---------------------------------------------------------------- memory groups
def stage_memory(run: CE.Run, *, dense=None, groups: Optional[List[str]] = None, specs: Optional[Mapping[str, dict]] = None) -> dict:
    """``specs``: group definitions (default: this round's ``MEMORY_SPECS`` + the structured group; ``eval_v3`` passes its own)."""
    from . import memory_v2 as V2
    from .runtime import PersonaRuntime
    rows, qa_sha = QB.load_frozen(Path(run.run_dir) / "qa")
    groups = groups or run.cfg["memory_groups"]
    ask_time = datetime.fromisoformat(json.loads((Path(run.run_dir) / "qa" / "QA_BUILD_REPORT.json").read_text(encoding="utf-8"))["ask_time"])
    card_id = run.manifest["prompts"]["old"]["card_id"]
    adapters = [v["adapter_id"] for v in (run.manifest.get("adapters") or {}).values()]
    specs = dict(specs) if specs is not None else dict(MEMORY_SPECS, **{STRUCTURED_GROUP: {"memory": "production", "pipeline": "v2", "dense": True,
                                                                                           "adapter": None}})
    out = {}
    for g in groups:
        sp = specs[g]
        if sp.get("dense") and dense is None:
            run.status(f"V2 STAGE 13 memory {g}", "SKIP", reason="no dense channel")
            continue
        rt = PersonaRuntime(run.work_root, run.profile_id, adapters=adapters, memory=sp["memory"], generator=run.gen, card_id=card_id,
                            memory_pipeline=sp["pipeline"], memory_v2=V2.V2Config(use_dense=bool(sp.get("dense")), k=int(sp.get("k", 8)),
                                                                                   neighbors=tuple(sp.get("neighbors", (2, 2))),
                                                                                   strip_qform=bool(sp.get("strip_qform", False))),
                            dense=dense if sp.get("dense") else None, reader_version=sp.get("reader_version", "v1"))
        rt.use_adapter(_adapter_id(run, sp["adapter"]))
        path = run.p("memory_qa", g, "answers.jsonl")
        done = {r["question_id"] for r in CE._jsonl(path) if r.get("status") == "OK"}
        for q in [q for q in rows if q["question_id"] not in done]:
            t0 = time.time()
            try:
                r = rt.reply([("other", q["question"])], now=ask_time)
                rd = r.get("reader") or {}
                retrieved = set(rd.get("retrieved_message_ids") or [])
                gold = set(q.get("evidence_message_ids") or [])
                row = {"question_id": q["question_id"], "group": g, "status": "OK", "text": r["text"], "raw_text": r.get("raw_text"), "route": r["route"],
                       "final_check": rd.get("final_check"), "reader_answerable": rd.get("answerable"), "reader_answer": rd.get("answer"),
                       "reader_fail": rd.get("fail"), "rejected_answer": rd.get("rejected_answer"), "subject": rd.get("subject"), "retrieved": len(retrieved),
                       "gold_retrieved": (len(gold & retrieved) / len(gold)) if gold else None, "date_summary": rd.get("date_summary"),
                       "adapter": r.get("adapter"), "generator": r.get("generator"), "pipeline": sp["pipeline"], "v2_meta": getattr(rt, "last_v2_meta", None),
                       "reader_version": rt.reader_version,
                       "tokens": {"reader_prompt": rd.get("reader_prompt_tokens"), "reader_new": rd.get("reader_tokens"),
                                  "reply_prompt": r.get("prompt_tokens"), "reply_new": r.get("new_tokens"),
                                  "reader_latency_s": rd.get("reader_latency_s"), "reply_latency_s": r.get("latency_s")}}
            except Exception as e:  # noqa: BLE001 - recorded; retried on the next run
                row = {"question_id": q["question_id"], "group": g, "status": "ERROR", "error": f"{type(e).__name__}: {str(e)[:300]}"}
            row["wall_s"] = round(time.time() - t0, 2)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        out[g] = sum(1 for r in CE._jsonl(path) if r.get("status") == "OK")
    run.status("V2 STAGE 13 memory generation", "PASS" if all(v >= len(rows) for v in out.values()) else "INCOMPLETE",
               **{g: f"{v}/{len(rows)}" for g, v in out.items()}, qa_sha256=qa_sha)
    return out


# ---------------------------------------------------------------- writer gate (+ conditional full build)
THIRD_PERSON_LINES = ((9, 0, "contact", "我爸说我姐下个月搬家"), (9, 60, "self", "这么突然"), (9, 120, "contact", "我妹妹的婚礼定在五月"))


def third_person_probe(run: CE.Run, tmp: Path) -> dict:
    """The real Writer on the public synthetic chat + third-person lines said BY the persona about relatives ("我爸说我姐下个月搬家",
    "我妹妹的婚礼定在五月"). PASS iff no memory of the persona itself (subject persona, or an event whose actor is self) mentions 搬家 / 婚.
    Synthetic and public: no private text."""
    import sqlite3 as _sq
    import persona_memory_fixture as MF
    from . import memory_store as MS
    from . import service as svc
    from .models import local_complete
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    who = {"contact": MF.F.CONTACT, "self": MF.F.SELF}
    path = MF.chat(extra=[(d, s, who[w], t) for d, s, w, t in THIRD_PERSON_LINES]).write(tmp / "third_person_chat.json")
    root = tmp / "profiles"
    pid = svc.init_profile(path, speaker="contact", root=root)["profile_id"]
    MS.build_evidence(root, pid)
    MS.build_semantic(root, pid, complete=local_complete(run.gen, int(run.cfg["writer_max_new_tokens"])), reviewer="third-person-probe")
    con = _sq.connect(str(S.profile_dir(root, pid) / "memory" / "memory.sqlite"))
    rows = [tuple(x) for x in con.execute("SELECT subject, attribute, value FROM memories").fetchall()]
    con.close()
    actor = {x[0]: x[2] for x in rows if x[1] == "event.actor"}
    bad = [x for x in rows if any(w in str(x[2]) for w in ("搬家", "婚")) and (x[0] == "persona" or actor.get(x[0]) == "self")]
    res = {"third_person_not_self": not bad, "violations": len(bad), "memories": len(rows), "lines": [t for *_, t in THIRD_PERSON_LINES]}
    CE._write(run.p("reports", "WRITER_THIRD_PERSON_PROBE.json"), res)
    return res


def stage_writer_gate(run: CE.Run, *, synthetic_dir: Path, env: Mapping[str, str], transport=None) -> dict:
    from . import writer_gate as G
    from .models import local_complete
    syn = CE.synthetic_gold(run, synthetic_dir)
    syn["writer_checks"]["third_person_not_self"] = third_person_probe(run, Path(synthetic_dir).with_name(Path(synthetic_dir).name + "_third_person"))["third_person_not_self"]
    CE._write(run.p("reports", "SYNTHETIC_GOLD_REPORT.json"), syn)
    CE.stage_07_evidence(run)                                              # the real Writer on the frozen 20 audit sessions (isolated copy)
    CE.stage_16_writer_judge(run, env, transport=transport)
    wri = CE._scores(run, "writer", "writer_unit_key.json")
    rows = [r for g in wri.values() for r in g.values()]
    judge = {k: sum(len(r["judgement"].get(src) or []) for r in rows) for k, src in (("supported", "supported"), ("unsupported", "unsupported"),
                                                                                       ("attribution_errors", "attribution_errors"), ("temporal_errors", "temporal_errors"),
                                                                                       ("worthiness_errors", "worthiness_errors"), ("suspected_missing", "missing"))}
    cand = CE._jsonl(CE.writer_pdir(run) / "memory" / "writer_candidates.jsonl")
    allc = [c for r in cand for c in r.get("candidates", [])]
    sample = {"sessions": len({r["conversation_id"] for r in cand}), "candidates": len(allc), "verified": sum(1 for c in allc if c.get("status") == "AUTO_VERIFIED")}
    d = G.decide(synthetic_checks=syn["writer_checks"], sample=sample, judge=judge)
    res = {"decision": d.to_dict(), "full_build": "NOT_RUN"}
    if d.decision == "FULL_BUILD" and run.cfg.get("allow_full_build"):
        r = G.gated_full_build(run.work_root, run.profile_id, d, complete=local_complete(run.gen, int(run.cfg["writer_max_new_tokens"])),
                               reviewer=run.cfg["accept_auto_verified_by"], backup_dir=Path(run.run_dir) / "memory_backup_before_full_build", log=run.log)
        res["full_build"] = {k: v for k, v in r.items() if k != "result"}
        CE.checkpoint_memory(run)
    elif d.decision == "FULL_BUILD":
        res["full_build"] = "ALLOWED_BY_GATE_BUT_DISABLED (cfg allow_full_build=False)"
    CE._write(run.p("reports", "WRITER_GATE_REPORT.json"), res)
    run.status("V2 STAGE 07 writer gate", d.decision, reasons=d.reasons[:3], full_build=str(res["full_build"])[:80])
    return res


# ---------------------------------------------------------------- judge (same rubric as FAST)
def stage_judge(run: CE.Run, env: Mapping[str, str], transport=None) -> dict:
    client, model = CE.judge_client(run, env, transport)
    cal = J.calibrate(client, model, run.p("judge", "CALIBRATION_SCORES.jsonl"), log=run.log)
    CE._write(run.p("reports", "JUDGE_CALIBRATION_REPORT.json"), cal)
    out = {"calibration": f"{cal['passed']}/{cal['total']}"}
    for kind, units, keyname in (("style", CE.style_units(run), "style_unit_key.json"), ("memory", CE.memory_units(run), "memory_unit_key.json")):
        key_field = "sample_id" if kind == "style" else "question_id"
        CE._write(run.p("judge", keyname), {u["unit_id"]: {key_field: u[key_field], "group": u["_group"]} for u in units})
        res = J.run(client, model, kind, [{k: v for k, v in u.items() if not k.startswith("_")} for u in units],
                    run.p("judge", f"{kind.upper()}_DEEPSEEK_SCORES.jsonl"), seed=run.manifest["run_salt"], concurrency=int(run.cfg["judge_concurrency"]),
                    per_second=float(run.cfg["judge_per_second"]), log=run.log)
        out[kind] = {"ok": res["ok"], "failed": res["failed"]}
    run.status("V2 STAGE 14-15 judge", "PASS" if all(v["failed"] == 0 for k, v in out.items() if k != "calibration") else "PARTIAL", **out)
    return out


# ---------------------------------------------------------------- analysis + decision rules
def _sub(d: Mapping[str, float], keep) -> Dict[str, float]:
    return {k: v for k, v in d.items() if k in keep}


def analysis(run: CE.Run) -> dict:
    from . import package as P
    from . import style_v2 as SV
    sty = CE._scores(run, "style", "style_unit_key.json")
    mem = CE._scores(run, "memory", "memory_unit_key.json")
    rows, _ = QB.load_frozen(Path(run.run_dir) / "qa")
    by_q = {q["question_id"]: q for q in rows}
    split = json.loads(run.p("reports", "MEMORY_V2_SPLIT.json").read_text(encoding="utf-8"))
    sv = lambda g: {k: r["score_100"] for k, r in sty.get(g, {}).items()}          # noqa: E731
    mv = lambda g: {k: r["score_100"] for k, r in mem.get(g, {}).items()}          # noqa: E731
    style = {g: J.describe(list(sv(g).values())) for g in sty}
    pairs_s = [("B1", "A"), ("R1S", "B1"), ("R2S", "B1"), ("R2S", "R1S"), ("R1S", "R1T"), ("R1S_T10", "R1S"), ("R2S_T10", "R2S")]
    style_paired = {f"{b}-{a}": J.paired(sv(a), sv(b)) for b, a in pairs_s if a in sty and b in sty}
    items = CE.dev_items(run)
    ref = EV.Reference(P.latest_dataset(run.pdir), _skill_card(run)["card"])
    v2 = SV.evaluate(Path(run.run_dir) / "style", items, [g for g in run.cfg["style_groups"] if g in sty], ref)
    strata = unseen_strata(run)
    by_stratum = {g: {s: J.describe([v for k, v in sv(g).items() if strata["items"].get(k) == s]).get("mean") for s in ("SEEN", "UNSEEN")} for g in sty}
    general = json.loads(run.p("reports", "GENERAL_CAPABILITY_REPORT.json").read_text(encoding="utf-8")) if run.p("reports", "GENERAL_CAPABILITY_REPORT.json").is_file() else {}
    memory = {}
    for part, keep in (("ALL", set(by_q)), ("DEV", set(split["dev"])), ("HOLDOUT", set(split["holdout"]))):
        if not keep:
            continue
        memory[part] = {"groups": {g: {**J.describe(list(_sub(mv(g), keep).values())),
                                       **CE.memory_metrics(run, g, {q: r for q, r in mem[g].items() if q in keep}, by_q)} for g in mem},
                        "paired": {f"{b}-{a}": J.paired(_sub(mv(a), keep), _sub(mv(b), keep))
                                   for b, a in (("M1", "M0"), ("V2L", "M1"), ("V2D", "M1"), ("V2D", "V2L"), ("V2D_R2", "V2D"), (STRUCTURED_GROUP, "V2D"),
                                                ("V2D", "V2D_K40"))
                                   if a in mem and b in mem}}
    tax = CE.reader_taxonomy(run)
    cost = memory_cost(run, mem)
    failures = failure_classes(run, mem, by_q)
    decisions = {"r2": r2_decision(style_paired, v2, general, memory.get("ALL", {})), "memory_v2": memory_v2_decision(memory.get("HOLDOUT", {}))}
    rep = {"schema": SCHEMA, "style": style, "style_paired": style_paired, "style_v2": v2, "style_by_stratum": by_stratum,
           "unseen_counts": strata["counts"], "general": general, "memory": memory, "reader_taxonomy": tax, "memory_cost": cost,
           "failure_classes": failures, "decisions": decisions,
           "rules": {"r2": R2_RULE, "memory_v2": MEMORY_V2_RULE, "unseen": UNSEEN_RULE}}
    CE._write(run.p("reports", "ANALYSIS_V2.json"), rep)
    run.status("V2 STAGE 17 analysis", "PASS", r2=decisions["r2"]["verdict"], memory_v2=decisions["memory_v2"]["verdict"])
    return rep


def memory_cost(run: CE.Run, mem: Mapping) -> dict:
    """Per memory group: mean input / output tokens (reader + reply), wall time p50 / p95, and memory score per 1,000 input tokens.
    Equal message counts are not equal token counts: this is the budget actually spent."""
    import statistics as st
    out = {}
    for g in mem:
        rows = [r for r in CE._jsonl(Path(run.run_dir) / "memory_qa" / g / "answers.jsonl") if r.get("status") == "OK"]
        tin = [((r.get("tokens") or {}).get("reader_prompt") or 0) + ((r.get("tokens") or {}).get("reply_prompt") or 0) for r in rows]
        tout = [((r.get("tokens") or {}).get("reader_new") or 0) + ((r.get("tokens") or {}).get("reply_new") or 0) for r in rows]
        wall = sorted(r.get("wall_s") or 0 for r in rows)
        score = st.fmean(r["score_100"] for r in mem[g].values()) if mem[g] else None
        out[g] = {"n": len(rows), "input_tokens_mean": round(st.fmean(tin), 1) if tin else None, "output_tokens_mean": round(st.fmean(tout), 1) if tout else None,
                  "wall_s_p50": wall[len(wall) // 2] if wall else None, "wall_s_p95": wall[min(len(wall) - 1, int(.95 * len(wall)))] if wall else None,
                  "score_per_1k_input_tokens": round(1000 * score / st.fmean(tin), 2) if tin and st.fmean(tin) and score is not None else None}
    return out


FAILURE_CLASSES = {"A": "routing", "B": "retrieval", "C": "evidence selection (gold partly shown, answer wrong)", "D": "reader reasoning",
                   "E": "abstention", "F": "attribution / temporal", "G": "judge / evaluation item (yes / no reference, judge leniency)",
                   "N": "no memory (the M0 baseline has no router, retrieval or reader: its failures are not retrieval failures)"}


def failure_classes(run: CE.Run, mem: Mapping, by_q: Mapping) -> dict:
    """Every judged memory answer with a low score (< 60) gets ONE class, first match in A..G order; answers >= 60 are OK. G is a flag
    for items the FAST audit showed to be unreliable (reference answer only 记得 / 是 / 对 / 有), not a judge error proven per item."""
    from collections import Counter as _C
    out = {}
    for g in mem:
        ans = {r["question_id"]: r for r in CE._jsonl(Path(run.run_dir) / "memory_qa" / g / "answers.jsonl") if r.get("status") == "OK"}
        c = _C()
        for qid, sc in mem[g].items():
            q, a, j = by_q[qid], ans.get(qid, {}), sc["judgement"]
            if sc["score_100"] >= 60:
                c["OK"] += 1
                continue
            yes_no = str(q.get("reference_answer") or "").strip() in ("记得", "是", "对", "有", "是的")
            gr = a.get("gold_retrieved")
            abstained = a.get("final_check") in ("UNCERTAINTY_KEPT", "FALLBACK_MINIMAL_UNCERTAIN") or a.get("reader_fail") == "READER_ABSTAINED"
            if MEMORY_SPECS.get(g, {}).get("memory", "production") is None:
                k = "N"
            elif q["answerability"] != "ANSWERABLE":
                k = "E"
            elif a.get("route") == "chat":
                k = "A"
            elif a.get("reader_fail") == "SUBJECT_RESOLUTION_FAIL":
                k = "F"
            elif gr == 0:
                k = "B"
            elif abstained or a.get("reader_fail") == "GROUNDEDNESS_FAIL":
                # an abstention: the judge scores "不记得" low on EVERY dimension, so it is checked before attribution / temporal
                k = "G" if yes_no else "E"
            elif gr is not None and gr < 1 and j.get("factual_correctness", 0) <= 2:
                k = "C"
            elif j.get("temporal_attribution", 5) <= 1:
                k = "F"
            else:
                k = "G" if yes_no else "D"
            c[k] += 1
        out[g] = dict(c)
    return {"classes": FAILURE_CLASSES, "rule": ("score < 60 -> N (no memory) | E unanswerable | A chat route | F attribution gate | B no gold "
                                                 "shown | E/G abstained | C partial evidence | F temporal / person | D/G answered wrong; abstentions are "
                                                 "classified before the judge's attribution dimension (revised 2026-11-18 after the first V2 run)"), "by_group": out}


def _ci_above(p: Mapping, x: float = 0.0) -> bool:
    return bool(p and p.get("n") and p["ci95"][0] > x)


def r2_decision(style_paired: Mapping, v2: Mapping, general: Mapping, mem_all: Mapping) -> dict:
    g = (v2.get("groups") or {})
    no_collapse = bool(((g.get("R2S") or {}).get("checks") or {}).get("no_conditional_collapse"))
    acc = (general or {}).get("accuracy") or {}
    mp = (mem_all.get("paired") or {}).get("V2D_R2-V2D") or {}
    mg = mem_all.get("groups") or {}
    fab = lambda k: (mg.get(k) or {}).get("unsupported_claim_rate")                 # noqa: E731
    checks = {"style_ci_above_0": _ci_above(style_paired.get("R2S-R1S")), "no_conditional_collapse": no_collapse,
              "general_not_worse": (acc.get("GR2") is not None and acc.get("GR1") is not None and acc["GR2"] >= acc["GR1"] - 0.05),
              "memory_non_inferior": bool(mp.get("n")) and mp["ci95"][0] > -2,
              "fabrication_ok": fab("V2D_R2") is not None and fab("V2D") is not None and fab("V2D_R2") <= fab("V2D") + 0.02}
    better = all(checks.values())
    recommend = better and _ci_above(style_paired.get("R2S-B1"))
    return {"verdict": "R2_BETTER" if better else "R2_NOT_BETTER", "recommend_enable": recommend, "checks": checks}


def memory_v2_decision(hold: Mapping) -> dict:
    if not hold:
        return {"verdict": "NOT_EVALUATED", "reason": "no HOLDOUT questions"}
    p, gs = hold.get("paired") or {}, hold.get("groups") or {}
    chosen = "V2D" if _ci_above(p.get("V2D-V2L")) else "V2L"
    d = p.get(f"{chosen}-M1") or {}
    gm = lambda g, k: (gs.get(g) or {}).get(k)                                        # noqa: E731
    checks = {"non_inferior": bool(d.get("n")) and d["ci95"][0] > -2, "mean_above_0": bool(d.get("n")) and d["mean_diff"] > 0,
              "grounded_not_lower": (gm(chosen, "grounded_answer_accuracy") or 0) >= (gm("M1", "grounded_answer_accuracy") or 0),
              "fabrication_ok": (gm(chosen, "unsupported_claim_rate") or 0) <= (gm("M1", "unsupported_claim_rate") or 0) + 0.02}
    return {"verdict": "ADOPT_V2" if all(checks.values()) else "KEEP_V1", "chosen_v2_group": chosen, "checks": checks, "paired": d}


# ---------------------------------------------------------------- export
EXPORT = ("STAGE_STATUS.json", "STAGE_TIMINGS.json", "reports/ANALYSIS_V2.json", "reports/GENERAL_CAPABILITY_REPORT.json", "reports/WRITER_GATE_REPORT.json",
          "reports/WRITER_THIRD_PERSON_PROBE.json", "reports/MEMORY_MODE.json",
          "reports/SYNTHETIC_GOLD_REPORT.json", "reports/DENSE_INDEX_REPORT.json", "reports/STYLE_UNSEEN_STRATA.json", "reports/MEMORY_V2_SPLIT.json",
          "reports/JUDGE_CALIBRATION_REPORT.json", "reports/STYLE_SAMPLE_MANIFEST.json", "judge/STYLE_DEEPSEEK_SCORES.jsonl", "judge/MEMORY_DEEPSEEK_SCORES.jsonl",
          "judge/WRITER_DEEPSEEK_SCORES.jsonl", "judge/style_unit_key.json", "judge/memory_unit_key.json", "judge/writer_unit_key.json", "qa/QA_FROZEN.sha256")


def stage_export(run: CE.Run, env_report: Mapping) -> Path:
    R = Path(run.run_dir)
    out = R / "export" / "StillHere_V2_Results.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for n in EXPORT:
        if (R / n).is_file():
            entries.append((n, (R / n).read_bytes()))
    for sub in ("style", "general", "memory_qa"):
        for f in sorted((R / sub).rglob("answers.jsonl")) if (R / sub).is_dir() else []:
            entries.append((f.relative_to(R).as_posix(), f.read_bytes()))
    usage = J.make_ledger_summary(run.p("judge", "ledger.jsonl")) if run.p("judge", "ledger.jsonl").is_file() else {}
    man = {"schema": SCHEMA, "run_dir": str(R), "manifest": {k: run.manifest.get(k) for k in ("package_id", "input_zip_sha256", "git_commit", "base", "adapters", "prompts")},
           "cfg": run.cfg, "environment": dict(env_report), "judge_usage": usage, "contains_test": False, "human_review": "NONE",
           "created_at": S.now_iso()}
    entries.append(("RUN_MANIFEST_V2.json", json.dumps(man, ensure_ascii=False, indent=1, default=str).encode("utf-8")))
    sums = "".join(f"{hashlib.sha256(b).hexdigest()}  {n}\n" for n, b in entries)
    from . import package as P
    out.write_bytes(P.det_zip(entries + [("SHA256SUMS.txt", sums.encode("utf-8"))]))
    run.status("V2 STAGE 19 export", "PASS", zip=str(out), files=len(entries) + 1)
    return out
