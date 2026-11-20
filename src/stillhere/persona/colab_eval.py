"""The final Colab L4 evaluation run: full memory build, real-history QA, Style Dev (B1/B2/C1/C2), Memory ablation (M0/M1/M2), DeepSeek judging,
combined analysis, recommendation and the results ZIP. One base model is loaded once and shared by the Writer, the QA builder, the
Fact Reader, the date summariser and persona chat (adapter on / off by name). No training, no gradients, no weight updates.

Every stage is a function of ``Run`` and is resumable: outputs and checkpoints live in the run directory on Google Drive; a stage
whose result file exists is not redone; generation / judging skip finished units; the memory store is checkpointed with SQLite's
backup API every ``CHECKPOINT_EVERY`` sessions and restored on a new session. A run directory is bound to a fingerprint (input ZIP,
code, base revision, adapter, prompts, configuration): a mismatching resume is refused.

Groups (only the prompt version, the adapter switch or the memory switch differs inside each comparison):
    B1 Base + old prompt   B2 Base + new prompt   C1 Base + old prompt + r1   C2 Base + new prompt + r1        (268 DEV each, no memory)
    M0 Base + selected prompt, NO memory   M1 Base + selected prompt + memory   M2 Base + selected prompt + r1 + memory   (200 QA each)
Style DEV groups use no memory at all, so no DEV target reply can reach the model through memory (time-cutoff by construction).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional

from . import evaluation as EV
from . import judge as J
from . import qa_builder as QB
from . import store as S

RUN_SCHEMA = "stillhere.persona.final_colab_run/1.0"
STYLE_GROUPS = ("B1", "B2", "C1", "C2")
MEMORY_GROUPS = ("M0", "M1", "M2")
CHECKPOINT_EVERY = 20
STYLE_WEIGHT, MEMORY_WEIGHT = 0.40, 0.60
PROMPT_RULE = ("pre-registered: choose the NEW prompt iff the pooled paired DeepSeek Style Score difference (B2-B1 and C2-C1, per DEV item) "
               "has mean > 0 and a 95% bootstrap CI above 0, AND STYLE_V2 'no_conditional_collapse' does not newly fail for B2 or C2; else keep OLD")
ADAPTER_RULE = ("pre-registered: recommend enabling the r1 adapter iff Combined(M2) > Combined(M1), the paired Memory Score difference M2-M1 has a 95% CI "
                "lower bound > -2, the fabricated-private-fact rate of M2 is <= M1 + 0.02, the paired Style Score difference C_sel-B_sel has a 95% CI above 0, "
                "and STYLE_V2 'no_conditional_collapse' passes for C_sel; otherwise keep it disabled (r1 NOT_BETTER stays as recorded)")


@dataclass
class Run:
    work_root: Path                     # local copy of the profile root (fast disk)
    profile_id: str
    run_dir: Path                       # Drive: outputs + checkpoints (survives a disconnect)
    manifest: dict                      # PACKAGE_MANIFEST.json
    cfg: dict = field(default_factory=dict)
    gen: object = None
    log: Callable[[str], None] = print

    @property
    def pdir(self) -> Path:
        return S.profile_dir(self.work_root, self.profile_id)

    def p(self, *parts) -> Path:
        q = Path(self.run_dir).joinpath(*parts)
        q.parent.mkdir(parents=True, exist_ok=True)
        return q

    def status(self, name: str, status: str, **info) -> dict:
        path = self.p("STAGE_STATUS.json")
        st = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        st[name] = {"status": status, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **info}
        path.write_text(json.dumps(st, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")
        self.log(f"===== {name}: {status} ===== {json.dumps(info, ensure_ascii=False, default=str)[:700]}")
        return st[name]


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _jsonl(path: Path) -> List[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()] if Path(path).is_file() else []


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


FULL_DEV_ITEMS = 268
MODES = {
    # the complete design: full structured semantic memory build + 268 DEV + 200 QA + 100 writer audits (1,772 judge units)
    "FULL": {"style_n": None, "qa_targets": dict(QB.TARGETS), "writer_audit_n": 100, "memory_build": "semantic_full", "qa_backfill": False,
             "case_min": 20, "case_max": 30, "results_zip": "StillHere_Final_L4_Results.zip", "memory_mode": "STRUCTURED_SEMANTIC_MEMORY"},
    # the fast formal evaluation: EVIDENCE memory (the full archive + EF1 + Fact Reader, no structured semantic build), 100 DEV, 100 QA,
    # 20 writer sessions run through the real Writer in an isolated copy (720 judge units)
    "FAST": {"style_n": 100, "qa_targets": dict(QB.FAST_TARGETS), "writer_audit_n": 20, "memory_build": "evidence_only", "qa_backfill": True,
             "case_min": 15, "case_max": 20, "results_zip": "StillHere_FAST_L4_Results.zip", "memory_mode": "EVIDENCE_MEMORY"},
}
DEFAULT_MODE = "FAST"


def defaults(cfg: Mapping) -> dict:
    mode = str(cfg.get("eval_mode") or DEFAULT_MODE).upper()
    if mode not in MODES:
        raise ValueError(f"EVAL_MODE must be one of {sorted(MODES)}, not {mode!r}")
    return {"eval_mode": mode, "style_groups": list(STYLE_GROUPS), **MODES[mode], "judge_concurrency": 4, "judge_per_second": 4.0, "seed": 20261117,
            "accept_auto_verified_by": "owner (final Colab run)", "writer_max_new_tokens": 1536, **{k: v for k, v in dict(cfg).items() if k != "eval_mode"}}


def plan_units(cfg: Mapping) -> dict:
    """The formal DeepSeek units of a configuration (calibration, QA building and retries are counted separately)."""
    style = len(cfg["style_groups"]) * (cfg.get("style_n") or FULL_DEV_ITEMS)
    memory = len(MEMORY_GROUPS) * sum(cfg["qa_targets"].values())
    writer = int(cfg["writer_audit_n"])
    return {"style": style, "memory": memory, "writer": writer, "total": style + memory + writer,
            "generations": {"style": style, "memory": memory, "total": style + memory}}


class timer:
    """Wall-clock per stage (``STAGE_TIMINGS.json`` on Drive; printed): progress and timing without promising a duration."""

    def __init__(self, run: "Run", name: str):
        self.run, self.name = run, name

    def __enter__(self):
        self.t0 = time.time()
        self.run.log(f">>> {self.name} started")
        return self

    def __exit__(self, et, ev, tb):
        path = self.run.p("STAGE_TIMINGS.json")
        t = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        dt = round(time.time() - self.t0, 1)
        t.setdefault(self.name, []).append({"seconds": dt, "ok": et is None, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        path.write_text(json.dumps(t, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
        self.run.log(f"<<< {self.name} {'done' if et is None else 'FAILED'} in {dt / 60:.1f} min")
        return False


# ---------------------------------------------------------------- fingerprint
def bind(run: Run) -> str:
    """Bind the run directory to its inputs; refuse a mismatching resume (use a new RUN_NAME for a new configuration)."""
    m = run.manifest
    fp = {"input_zip_sha256": m.get("input_zip_sha256"), "code_tree_sha256": m.get("code_tree_sha256"), "base": m.get("base"), "adapter": m.get("adapter"),
          "prompts": m.get("prompts"), "dataset_version": m.get("dataset_version"), "profile_id": run.profile_id,
          "cfg": {k: run.cfg.get(k) for k in ("eval_mode", "qa_targets", "writer_audit_n", "seed", "style_groups", "style_n", "memory_build", "qa_backfill")},
          "judge_rubric_sha256": J.rubric_sha()}
    return EV.bind_fingerprint(Path(run.run_dir), fp)


# ---------------------------------------------------------------- memory store checkpoints (SQLite backup API)
MEM_FILES = ("writer_cache.jsonl", "writer_candidates.jsonl", "WRITER_FAILED_QUEUE.json", "WRITER_PROGRESS.json")


def checkpoint_memory(run: Run, pdir: Optional[Path] = None, name: str = "memory") -> None:
    pdir = pdir or run.pdir
    src_db = pdir / "memory" / "memory.sqlite"
    dst = run.p("checkpoints", name, "memory.sqlite")
    tmp = dst.with_suffix(".sqlite.tmp")
    a, b = sqlite3.connect(str(src_db)), sqlite3.connect(str(tmp))
    try:
        a.backup(b)
    finally:
        b.close()
        a.close()
    os.replace(tmp, dst)
    for n in MEM_FILES:
        f = pdir / "memory" / n
        if f.is_file():
            shutil.copyfile(f, dst.parent / (n + ".tmp"))
            os.replace(dst.parent / (n + ".tmp"), dst.parent / n)


def restore_memory(run: Run, pdir: Optional[Path] = None, name: str = "memory") -> bool:
    pdir = pdir or run.pdir
    ck = Path(run.run_dir) / "checkpoints" / name / "memory.sqlite"
    if not ck.is_file():
        return False
    (pdir / "memory").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ck, pdir / "memory" / "memory.sqlite")
    for n in MEM_FILES:
        if (ck.parent / n).is_file():
            shutil.copyfile(ck.parent / n, pdir / "memory" / n)
    return True


# ---------------------------------------------------------------- stage 06: engineering + synthetic gold (real model)
def stage_06_checks(run: Run, *, synthetic_dir: Path, pytest_cmd: Optional[List[str]] = None) -> dict:
    """Engineering checks block the run (they mean the code / data are broken); the synthetic gold result is recorded, not a gate."""
    out = {}
    if pytest_cmd:
        rc, tail = run_pytest(pytest_cmd)
        out["pytest_rc"] = rc
        out["pytest_tail"] = tail
        if rc != 0:
            run.status("STAGE 06 checks", "FAIL", **out)
            raise RuntimeError("engineering checks failed: see STAGE_STATUS.json (pytest)")
    out["synthetic_gold"] = synthetic_gold(run, Path(synthetic_dir))
    _write(run.p("reports", "SYNTHETIC_GOLD_REPORT.json"), out["synthetic_gold"])
    run.status("STAGE 06 checks", "PASS", pytest_rc=out.get("pytest_rc"), synthetic_gold=out["synthetic_gold"]["summary"])
    return out


def run_pytest(cmd: List[str]) -> tuple:
    """Run the engineering tests in a subprocess that sees the SAME import path as this process (the notebook puts the shipped code on
    ``sys.path``, which a child process does not inherit). Returns (exit code, the tail of stdout + stderr)."""
    import subprocess
    import sys
    paths = [p for p in sys.path if p and Path(p).is_dir()]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(dict.fromkeys(paths + [x for x in os.environ.get("PYTHONPATH", "").split(os.pathsep) if x])),
           "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return r.returncode, ((r.stdout or "") + ("\n[stderr]\n" + r.stderr if r.stderr else ""))[-3000:]


def synthetic_gold(run: Run, tmp: Path) -> dict:
    """The public synthetic chat through the real Writer + Reader + date route with the loaded model (same assertions as the local GPU
    smoke, plus the third-party event and the date summary). Failure taxonomy per question."""
    import sys
    from datetime import timedelta
    sys.path.insert(0, str(Path(run.manifest.get("tests_dir", "tests"))))
    import persona_memory_fixture as MF
    from . import card as K
    from . import memory_store as MS
    from . import service as svc
    from .models import local_complete
    from .runtime import UNCERTAIN_CUE, PersonaRuntime
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    path = MF.chat().write(tmp / "synthetic_memory_chat.json")
    root = tmp / "profiles"
    pid = svc.init_profile(path, speaker="contact", root=root)["profile_id"]
    MS.build_evidence(root, pid)
    MS.build_semantic(root, pid, complete=local_complete(run.gen, 1536), reviewer="synthetic-gold")
    pdir = S.profile_dir(root, pid)
    con = sqlite3.connect(str(pdir / "memory" / "memory.sqlite"))
    rows = [tuple(x) for x in con.execute("SELECT subject, attribute, value, status, valid_to FROM memories").fetchall()]
    con.close()
    actor = {x[0]: x[2] for x in rows if x[1] == "event.actor"}
    selfrows = [x for x in rows if x[0] == "persona"]
    writer = {"no_self_australia": not any("澳洲" in str(x) for x in selfrows),
              "shenzhen_current": any(x[1] == "works_at" and "深圳" in x[2] and x[3] == "active" and x[4] is None for x in selfrows),
              "guangzhou_closed": any(x[1] == "works_at" and "广州" in x[2] and (x[3] != "active" or x[4] is not None) for x in selfrows),
              "dentist_cancelled": any(x[1] == "event.status" and x[2] == "CANCELLED" and x[3] == "active" for x in rows),
              # the sister may be named by her name or by the kin word (the follow-up prompt's own example is person:妹妹); P2 2026-11-19
              "wedding_event_third_party": any(x[1] == "event.description" and "婚" in x[2] and str(actor.get(x[0])) in ("person:小雨", "person:妹妹")
                                               for x in rows),
              "wedding_not_self": not any(x[1] == "event.description" and "婚" in x[2] and actor.get(x[0]) == "self" for x in rows),
              "coffee_negation": any(x[1] == "dislikes" and "咖啡" in x[2] for x in selfrows)}
    card = {"schema": K.CARD_SCHEMA, "persona_id": pid, "display_name": "小合成", "ai_identity": "你是 AI", "reply_length": {}, "style_traits": [], "examples": [],
            "evidence": {"split": "train", "dataset_version": "v", "train_sha256": "x", "profiler_version": "p", "example_ids": [], "method": "synthetic gold"},
            "review_status": K.AUTO, "authored_prompt": ("你是一个 AI 数字人，模仿一个说话很简短的人和对方微信聊天。回复通常只有几个字，先回应对方最新一句。"
                                                       "被问到过去的事时只说【要表达的内容】里的事实；没有依据时自然地说不记得，不要编造任何经历。") * 2}
    K.set_current(pdir, K._write_version(pdir, card, parent=None)["card_id"])
    rt = PersonaRuntime(root, pid, memory="production", generator=run.gen)
    at = datetime(2025, 2, 1, tzinfo=QB.TZ)
    qs = [("past_job", "你以前在哪个城市工作？", lambda r: "广州" in r["text"]),
          ("current_job", "你现在在哪工作？", lambda r: "深圳" in r["text"] and "广州" not in r["text"]),
          ("counterpart_experience", "我在哪读书来着？", lambda r: "澳洲" in r["text"]),
          ("third_party", "你妹妹什么时候结婚？", lambda r: "下个月" in r["text"] or "结婚" in r["text"]),
          ("date_summary", "2025年1月1日我们聊了什么？", lambda r: r["route"] == "date" and bool((r["reader"] or {}).get("answerable"))),
          ("unknown", "你小时候养过什么宠物吗？", lambda r: bool(UNCERTAIN_CUE.search(r["text"])) or (r["reader"] or {}).get("final_check") == "FALLBACK_MINIMAL_UNCERTAIN"),
          ("small_talk", "在吗", lambda r: r["route"] == "chat")]
    reader = {}
    for name, q, ok in qs:
        r = rt.reply([("other", q)], now=at)
        rd = r["reader"] or {}
        reader[name] = {"pass": bool(ok(r)), "route": r["route"], "final_check": rd.get("final_check"), "fail": rd.get("fail"), "reply": r["text"],
                        "reader_answer": rd.get("answer")}
    summary = {"writer_pass": sum(writer.values()), "writer_total": len(writer), "reader_pass": sum(v["pass"] for v in reader.values()), "reader_total": len(reader)}
    return {"writer_checks": writer, "reader_checks": reader, "summary": summary, "note": "public synthetic chat; recorded, not a gate",
            "wedding_diagnostics": wedding_diagnostics(pdir, rows, actor)}


def wedding_diagnostics(pdir: Path, rows, actor) -> dict:
    """Synthetic chat only: every stored row and every Writer candidate (with its verdict) about the wedding line, so a failed
    ``wedding_not_self`` / ``wedding_event_third_party`` on the real model can be traced (the V2 run could not: the store was not exported)."""
    stored = [{"subject": x[0], "attribute": x[1], "value": x[2], "actor": actor.get(x[0])} for x in rows if "婚" in str(x[2]) or "小雨" in str(x[2])]
    cands = []
    log = Path(pdir) / "memory" / "writer_candidates.jsonl"
    for line in (log.read_text(encoding="utf-8").splitlines() if log.is_file() else []):
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        for c in rec.get("candidates") or []:
            if any("婚" in str(c.get(k)) or "小雨" in str(c.get(k)) for k in ("object", "activity", "subject", "source_spans")):
                cands.append({k: c.get(k) for k in ("subject", "predicate", "object", "activity", "event_status", "source_spans", "support", "reasons", "status")})
    return {"stored": stored, "candidates": cands}


# ---------------------------------------------------------------- stage 07-09: full memory build
def stage_07_memory(run: Run) -> dict:
    if run.cfg.get("memory_build") == "evidence_only":
        return stage_07_evidence(run)
    return stage_07_semantic(run)


def _archive_counts(db: Path, pid: str) -> dict:
    con = sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True)
    try:
        q = lambda sql, *a: con.execute(sql, a).fetchone()[0]   # noqa: E731
        return {"messages": q("SELECT COUNT(*) FROM messages WHERE target_persona_id = ?", pid),
                "text_messages": q("SELECT COUNT(*) FROM messages WHERE target_persona_id = ? AND locator NOT LIKE '%kind=media%'", pid),   # = memory_store.status
                "sessions": q("SELECT COUNT(DISTINCT conversation_id) FROM messages WHERE target_persona_id = ?", pid),
                "messages_fts_rows": q("SELECT COUNT(*) FROM messages_fts"), "semantic_memories": q("SELECT COUNT(*) FROM memories"),
                "first": q("SELECT MIN(timestamp_utc) FROM messages WHERE target_persona_id = ?", pid),
                "last": q("SELECT MAX(timestamp_utc) FROM messages WHERE target_persona_id = ?", pid)}
    finally:
        con.close()


def writer_pdir(run: Run) -> Path:
    """Where the audited Writer ran: the production store in FULL; an isolated copy of the evidence archive in FAST (the FAST writer
    output is an evaluation candidate set only and never reaches the store M1 / M2 read)."""
    if run.cfg.get("memory_build") != "evidence_only":
        return run.pdir
    w = Path(run.work_root).parent / "writer_audit_root" / run.profile_id
    if not (w / "memory" / "memory.sqlite").is_file():
        (w / "memory").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run.pdir / "profile.json", w / "profile.json")
        if not restore_memory(run, w, "writer_audit"):
            shutil.copyfile(run.pdir / "memory" / "memory.sqlite", w / "memory" / "memory.sqlite")
    return w


def stage_07_evidence(run: Run) -> dict:
    """FAST: no structured semantic build. M1 / M2 read the COMPLETE conversation evidence archive (every message, its source, speaker
    and date) through EF1 + the Fact Reader; the real Writer runs only on the 20 sampled audit sessions, in an isolated copy."""
    from . import memory_store as MS
    from .models import local_complete
    done = run.p("reports", "MEMORY_MODE.json")
    if done.is_file():
        run.status("STAGE 07 memory (FAST)", "DONE_BEFORE")
        writer_pdir(run)
        return json.loads(done.read_text(encoding="utf-8"))
    counts = _archive_counts(run.pdir / "memory" / "memory.sqlite", run.profile_id)
    if counts["messages"] == 0 or counts["messages_fts_rows"] == 0:
        raise RuntimeError("EVIDENCE_ARCHIVE_MISSING: the package has no searchable evidence archive; FAST memory would degrade to no memory")
    sample = stage_11_writer_sample(run)
    w = writer_pdir(run)
    complete = local_complete(run.gen, int(run.cfg["writer_max_new_tokens"]))
    only = {x["conversation_id"] for x in sample["sessions"]}
    fq = w / "memory" / "WRITER_FAILED_QUEUE.json"
    t0 = time.time()
    for _ in range(MS.MAX_GROUP_ATTEMPTS):
        res = MS.build_semantic(w.parent, run.profile_id, complete=complete, reviewer=run.cfg["accept_auto_verified_by"], log=run.log, only=only,
                                on_group=lambda n: checkpoint_memory(run, w, "writer_audit") if n % 5 == 0 else None)
        checkpoint_memory(run, w, "writer_audit")
        pending = S.read_json(fq) if fq.is_file() else {}
        if not any(v.get("attempts", 0) < MS.MAX_GROUP_ATTEMPTS for v in pending.values()):
            break
    after = _archive_counts(run.pdir / "memory" / "memory.sqlite", run.profile_id)
    if after["semantic_memories"] != counts["semantic_memories"]:
        raise RuntimeError("EVIDENCE_STORE_CHANGED: the audit Writer must not write into the store M1 / M2 read")
    rep_ = {"memory_mode": "EVIDENCE_MEMORY", "structured_semantic_memory": "NOT_BUILT (FAST mode)", "evidence_archive": counts,
            "retrieval": "EF1 over every archived message (lexical + time window; no dense index) -> fact pack -> Fact Reader B",
            "writer_audit": {"sessions": len(only), "run": res.get("run"), "isolated_store": "writer_audit_root (copy of the archive; never imported)",
                             "seconds": round(time.time() - t0, 1)},
            "note": "FAST evaluates evidence memory; it does NOT establish a production structured semantic memory"}
    _write(done, rep_)
    run.status("STAGE 07 memory (FAST)", "PASS", messages=counts["messages"], text=counts["text_messages"], writer_sessions=len(only))
    return rep_


def stage_07_semantic(run: Run) -> dict:
    from . import memory_store as MS
    from .models import local_complete
    done = Path(run.run_dir) / "reports" / "FULL_MEMORY_BUILD_REPORT.json"
    if done.is_file():
        run.status("STAGE 07 memory", "DONE_BEFORE")
        restore_memory(run)
        return json.loads(done.read_text(encoding="utf-8"))
    restored = restore_memory(run)
    run.log(f"memory: {'resumed from the Drive checkpoint' if restored else 'new build'}")

    def on_group(n):
        if n % CHECKPOINT_EVERY == 0:
            checkpoint_memory(run)
    t0 = time.time()
    complete = local_complete(run.gen, int(run.cfg["writer_max_new_tokens"]))
    fq = run.pdir / "memory" / "WRITER_FAILED_QUEUE.json"
    for _ in range(MS.MAX_GROUP_ATTEMPTS):             # a failed session is retried inside the stage, at most MAX_GROUP_ATTEMPTS times in total
        res = MS.build_semantic(run.work_root, run.profile_id, complete=complete, reviewer=run.cfg["accept_auto_verified_by"], log=run.log, on_group=on_group)
        checkpoint_memory(run)
        pending = S.read_json(fq) if fq.is_file() else {}
        if not any(v.get("attempts", 0) < MS.MAX_GROUP_ATTEMPTS for v in pending.values()):
            break
    con = sqlite3.connect(str(run.pdir / "memory" / "memory.sqlite"))
    try:
        n_msgs = con.execute("SELECT COUNT(*) FROM messages WHERE target_persona_id = ?", (run.profile_id,)).fetchone()[0]
        n_sess = con.execute("SELECT COUNT(DISTINCT conversation_id) FROM messages WHERE target_persona_id = ?", (run.profile_id,)).fetchone()[0]
        n_batches = con.execute("SELECT COUNT(*) FROM writer_batches").fetchone()[0]
    finally:
        con.close()
    failed = S.read_json(run.pdir / "memory" / "WRITER_FAILED_QUEUE.json") if (run.pdir / "memory" / "WRITER_FAILED_QUEUE.json").is_file() else {}
    rep = {"input_messages": n_msgs, "sessions": n_sess, "written_batches": n_batches, "this_run": res["run"], "failed_queue": failed,
           "memories_active": res.get("memories_active"), "status": res.get("memory_status"), "seconds_this_session": round(time.time() - t0, 1),
           "review_record": "auto_verified;accepted_by:<name> (NOT human-reviewed)"}
    rep["given_up_sessions"] = sorted(k for k, v in failed.items() if v.get("attempts", 0) >= MS.MAX_GROUP_ATTEMPTS)
    _write(done, rep)
    run.status("STAGE 07 memory", "PASS", messages=n_msgs, sessions=n_sess, active=rep["memories_active"], gave_up=len(failed))
    return rep


def memory_snapshot(run: Run) -> dict:
    """A deterministic fingerprint of the memory content (D and E must read the same snapshot)."""
    con = sqlite3.connect(str(run.pdir / "memory" / "memory.sqlite"))
    try:
        rows = con.execute("SELECT memory_id, subject, attribute, value, status, valid_from_utc, valid_to_utc FROM memories ORDER BY memory_id").fetchall()
    except sqlite3.OperationalError:
        rows = con.execute("SELECT memory_id, subject, attribute, value, status FROM memories ORDER BY memory_id").fetchall()
    finally:
        con.close()
    h = hashlib.sha256(json.dumps(rows, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()
    con = sqlite3.connect(str(run.pdir / "memory" / "memory.sqlite"))
    try:
        arch = con.execute("SELECT message_id, content_sha256 FROM messages WHERE target_persona_id = ? ORDER BY message_id", (run.profile_id,)).fetchall()
    finally:
        con.close()
    a = hashlib.sha256(json.dumps(arch).encode("utf-8")).hexdigest()
    return {"memories": len(rows), "archive_messages": len(arch), "content_sha256": hashlib.sha256((h + a).encode()).hexdigest(), "memories_sha256": h,
            "archive_sha256": a, "memory_mode": run.cfg.get("memory_mode")}


def stage_08_09_reports(run: Run) -> dict:
    """TEMPORAL_REPORT (closed / superseded / conflicts) and EF1_REPORT (index sizes, latency) over the built store."""
    con = sqlite3.connect(str(run.pdir / "memory" / "memory.sqlite"))
    try:
        q = lambda sql: con.execute(sql).fetchone()[0]   # noqa: E731
        cols = {r[1] for r in con.execute("PRAGMA table_info(memories)")}
        temporal = {"memories_total": q("SELECT COUNT(*) FROM memories"), "active": q("SELECT COUNT(*) FROM memories WHERE status = 'active'"),
                    "closed_intervals": q("SELECT COUNT(*) FROM memories WHERE valid_to_utc IS NOT NULL") if "valid_to_utc" in cols else None,
                    "conflicts": q("SELECT COUNT(*) FROM memory_conflicts"), "memory_events": q("SELECT COUNT(*) FROM memory_events"),
                    "by_attribute": dict(con.execute("SELECT attribute, COUNT(*) FROM memories GROUP BY attribute ORDER BY 2 DESC LIMIT 40").fetchall())}
        ef1 = {"messages_indexed": q("SELECT COUNT(*) FROM messages_fts"), "memories_indexed": q("SELECT COUNT(*) FROM memory_fts"), "dense": False}
    finally:
        con.close()
    from . import memory_store as MS
    t0 = time.time()
    probe = MS.query(run.work_root, run.profile_id, "最近聊了什么", k=8)
    ef1["probe_latency_ms"] = round((time.time() - t0) * 1000, 1)
    ef1["probe_items"] = len(probe.get("items", []))
    snap = memory_snapshot(run)
    _write(run.p("reports", "TEMPORAL_REPORT.json"), temporal)
    _write(run.p("reports", "EF1_REPORT.json"), ef1)
    _write(run.p("reports", "MEMORY_SNAPSHOT.json"), snap)
    run.status("STAGE 08-09 memory reports", "PASS", **snap)
    return {"temporal": temporal, "ef1": ef1, "snapshot": snap}


# ---------------------------------------------------------------- stage 10-11: QA + writer audit sample
def stage_10_qa(run: Run) -> dict:
    from .models import local_complete
    qdir = Path(run.run_dir) / "qa"
    if (qdir / "REAL_HISTORY_QA_200.jsonl").is_file():
        rows, sha = QB.load_frozen(qdir)
        run.status("STAGE 10 QA", "FROZEN_BEFORE", questions=len(rows), sha=sha[:16])
        return {"questions": len(rows), "sha256": sha}
    rows, rep = QB.build(run.pdir / "memory" / "memory.sqlite", run.profile_id, local_complete(run.gen, 768), seed=int(run.cfg["seed"]),
                         targets=run.cfg["qa_targets"], log=run.log, progress=run.p("qa", "qa_build_progress.jsonl"), backfill=bool(run.cfg.get("qa_backfill")))
    rep = QB.freeze(rows, rep, qdir)
    run.status("STAGE 10 QA", "FROZEN", questions=len(rows), sha=rep["qa_sha256"][:16],
               shortfall={k: v["shortfall"] for k, v in rep["categories"].items() if v["shortfall"]})
    return {"questions": len(rows), "sha256": rep["qa_sha256"]}


def stage_11_writer_sample(run: Run) -> dict:
    path = run.p("writer_audit", "WRITER_AUDIT_SAMPLE.json")
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    if run.cfg.get("memory_build") == "evidence_only":
        written = sorted({m["conversation_id"] for m in QB.load_messages(run.pdir / "memory" / "memory.sqlite", run.profile_id)})
        per = 2
    else:
        written = sorted({r["conversation_id"] for r in _jsonl(run.pdir / "memory" / "writer_candidates.jsonl")})
        per = 4
    sample = QB.audit_sample(run.pdir / "memory" / "memory.sqlite", run.profile_id, written, n=int(run.cfg["writer_audit_n"]), seed=int(run.cfg["seed"]),
                             min_per_stratum=per)
    out = {"seed": run.cfg["seed"], "n": len(sample), "strata": dict(Counter(s["stratum"] for s in sample)), "sessions": sample}
    _write(path, out)
    run.status("STAGE 11 writer sample", "PASS", n=len(sample), strata=out["strata"])
    return out


# ---------------------------------------------------------------- stage 12: style DEV generation (B1 / B2 / C1 / C2)
def _cards(run: Run):
    from . import card as K
    return K.load(run.pdir, run.manifest["prompts"]["old"]["card_id"]), K.load(run.pdir, run.manifest["prompts"]["new"]["card_id"])


def style_sample(run: Run, items: List[dict]) -> List[str]:
    """Deterministic DEV sample (FAST): strata = reply length (<=2, <=5, <=12, >12 chars) x time tertile; largest-remainder allocation;
    inside a stratum, scenarios take turns, each ordered by sha256(seed:sample_id). Chosen without looking at any generation or score;
    frozen in STYLE_SAMPLE_MANIFEST.json (a later call must return the same ids)."""
    from . import card as K
    path = run.p("reports", "STYLE_SAMPLE_MANIFEST.json")
    n = int(run.cfg["style_n"])
    if path.is_file():
        m = json.loads(path.read_text(encoding="utf-8"))
        if m["n"] != n or m["seed"] != run.cfg["seed"]:
            raise RuntimeError("STYLE_SAMPLE_CHANGED: this run directory froze another DEV sample")
        return m["sample_ids"]
    times = sorted(it["time"] for it in items)
    cut = [times[len(times) // 3], times[2 * len(times) // 3]]

    def stratum(it):
        L = len((it["reference"] or "").replace("\n", ""))
        lb = "len<=2" if L <= 2 else "len<=5" if L <= 5 else "len<=12" if L <= 12 else "len>12"
        tb = "t1" if it["time"] < cut[0] else "t2" if it["time"] < cut[1] else "t3"
        return f"{lb}|{tb}"
    key = lambda it: hashlib.sha256(f"{run.cfg['seed']}:{it['sample_id']}".encode()).hexdigest()   # noqa: E731
    groups: Dict[str, List[dict]] = defaultdict(list)
    for it in items:
        groups[stratum(it)].append(it)
    quota = {k: n * len(v) / len(items) for k, v in groups.items()}
    alloc = {k: int(q) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: (-(quota[k] - alloc[k]), k))[: n - sum(alloc.values())]:
        alloc[k] += 1
    chosen = []
    for k in sorted(groups):
        by_sc: Dict[str, List[dict]] = defaultdict(list)
        for it in sorted(groups[k], key=key):
            by_sc[K.scenario_of(it.get("query") or "")].append(it)
        order = []
        while any(by_sc.values()):
            for sc in sorted(by_sc):
                if by_sc[sc]:
                    order.append(by_sc[sc].pop(0))
        chosen += [it["sample_id"] for it in order[:alloc[k]]]
    _write(path, {"n": len(chosen), "seed": run.cfg["seed"], "of": len(items), "sample_ids": sorted(chosen), "strata_alloc": alloc,
                  "rule": "reply-length x time-tertile strata, largest remainder, scenario round-robin, sha256(seed:sample_id) order; no scores used"})
    return sorted(chosen)


def dev_items(run: Run) -> List[dict]:
    """The DEV items of this mode (all 268 in FULL; the frozen 100-item sample in FAST). The sealed test is never read."""
    from . import package as P
    items = EV.eval_items(P.latest_dataset(run.pdir), "dev")
    if run.cfg.get("style_n"):
        keep = set(style_sample(run, items))
        items = [it for it in items if it["sample_id"] in keep]
    return items


def style_items(run: Run) -> List[dict]:
    old, new = _cards(run)
    items = dev_items(run)
    prompt = {"B1": old["prompt"], "C1": old["prompt"], "B2": new["prompt"], "C2": new["prompt"]}
    for it in items:
        it["messages"] = {g: [{"role": "system", "content": prompt[g]}, {"role": "user", "content": it["user"]}] for g in run.cfg["style_groups"]}
        it["prompt_hash"] = {g: EV.sha_obj(m) for g, m in it["messages"].items()}
        assert it["messages"].get("B1") == it["messages"].get("C1") and it["messages"].get("B2") == it["messages"].get("C2")   # only the adapter differs
    return items


def stage_12_style(run: Run) -> dict:
    items = style_items(run)
    aid = run.manifest["adapter"]["adapter_id"]
    fp = {"base": run.manifest["base"], "adapter_sha256": run.manifest["adapter"]["sha256"], "prompts": run.manifest["prompts"], "decode": EV.DECODE,
          "dataset_version": run.manifest["dataset_version"], "items": len(items)}
    summary = EV.generate(run.gen, items, run.cfg["style_groups"], Path(run.run_dir) / "style", fp,
                          adapters={"B1": None, "B2": None, "C1": aid, "C2": aid}, log=run.log)
    ok = all(v["ok"] == v["of"] for v in summary.values())
    run.status("STAGE 12 style generation", "PASS" if ok else "INCOMPLETE", **{g: f"{v['ok']}/{v['of']}" for g, v in summary.items()})
    return summary


# ---------------------------------------------------------------- stage 14: style judge (+ prompt selection)
def style_profile_text(card: Mapping) -> str:
    rl = card.get("reply_length") or {}
    ex = "、".join(x["text"] for x in (card.get("common_expressions") or [])[:8])
    sr = "、".join(f"{x['text']}({x['share']:.0%})" for x in (card.get("frequent_short_replies") or [])[:6])
    return (f"回复长度（字）：p25={rl.get('chars_p25')}，中位数={rl.get('chars_p50')}，p90={rl.get('chars_p90')}；常用表达：{ex or '无'}；"
            f"高频短回复：{sr or '无'}。这些统计只来自训练集。")


def style_units(run: Run) -> List[dict]:
    from . import card as K
    from . import package as P
    old, _ = _cards(run)
    prof = style_profile_text(old["card"])           # the person's train-only statistics (identical for every group: blind)
    items = {it["sample_id"]: it for it in dev_items(run)}
    units = []
    for g in run.cfg["style_groups"]:
        rows = EV.done_rows(Path(run.run_dir) / "style" / g / "answers.jsonl")
        texts = {s: (r["text"] or "").strip() for s, r in rows.items()}
        cnt = Counter(texts.values())
        top = "、".join(f"{t[:12]}({c})" for t, c in cnt.most_common(5))
        for sid, t in sorted(texts.items()):
            if sid not in items:
                continue
            it = items[sid]
            units.append({"unit_id": J.unit_id("style", run.manifest["run_salt"], sid, g), "sample_id": sid, "_group": g, "context": it["user"],
                          "reference": it["reference"], "response": t, "style_profile": prof,
                          "repetition": {"group_size": len(texts), "count": cnt[t], "top": top}})
    return units


def judge_client(run: Run, env: Mapping[str, str], transport=None):
    return J.make_client(env, run.p("judge", "ledger.jsonl"), transport=transport)


def stage_14_style_judge(run: Run, env: Mapping[str, str], transport=None) -> dict:
    client, model = judge_client(run, env, transport)
    cal_path = run.p("judge", "CALIBRATION_SCORES.jsonl")
    cal = J.calibrate(client, model, cal_path, log=run.log)
    _write(run.p("reports", "JUDGE_CALIBRATION_REPORT.json"), cal)
    units = style_units(run)
    _write(run.p("judge", "style_unit_key.json"), {u["unit_id"]: {"sample_id": u["sample_id"], "group": u["_group"]} for u in units})  # key, not sent
    res = J.run(client, model, "style", [{k: v for k, v in u.items() if not k.startswith("_")} for u in units], run.p("judge", "STYLE_DEEPSEEK_SCORES.jsonl"),
                seed=run.manifest["run_salt"], concurrency=int(run.cfg["judge_concurrency"]), per_second=float(run.cfg["judge_per_second"]), log=run.log)
    sel = select_prompt(run)
    run.status("STAGE 14 style judge", "PASS" if res["failed"] == 0 else "PARTIAL", ok=res["ok"], failed=res["failed"], calibration=f"{cal['passed']}/{cal['total']}",
               selected_prompt=sel["selected"])
    return {"judge": res, "selection": sel}


def _scores(run: Run, kind: str, key_name: str) -> Dict[str, Dict[str, dict]]:
    """group -> item id -> OK judge row (unblinded through the local key file)."""
    key = json.loads(run.p("judge", key_name).read_text(encoding="utf-8"))
    res = J.read_results(run.p("judge", f"{kind.upper()}_DEEPSEEK_SCORES.jsonl"))
    out: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for uid, meta in key.items():
        r = res.get(uid)
        if r and r.get("status") == "OK":
            out[meta["group"]][meta.get("sample_id") or meta.get("question_id")] = r
    return out


def style_v2_report(run: Run) -> dict:
    from . import card as K
    from . import package as P
    from . import style_v2 as SV
    old, _ = _cards(run)
    ddir = P.latest_dataset(run.pdir)
    items = dev_items(run)
    ref = EV.Reference(ddir, old["card"])
    v2 = SV.evaluate(Path(run.run_dir) / "style", items, run.cfg["style_groups"], ref)
    v1 = EV.score_run(Path(run.run_dir) / "style", items, run.cfg["style_groups"], ref)
    return {"style_v2": v2, "v1_signals": {g: {k: v1["groups"][g].get(k) for k in ("repetition_rate", "train_copy_rate", "privacy_leaks", "unsupported_specific_rate",
                                                                                 "generic_rate", "median_chars", "style_margin")} for g in v1["groups"]}}


def select_prompt(run: Run) -> dict:
    sc = _scores(run, "style", "style_unit_key.json")
    v = lambda g: {k: r["score_100"] for k, r in sc.get(g, {}).items()}   # noqa: E731
    diffs = {}
    for a, b in (("B1", "B2"), ("C1", "C2")):
        for k in set(v(a)) & set(v(b)):
            diffs[f"{a}:{k}"] = (v(a)[k], v(b)[k])
    pooled = J.paired({k: x[0] for k, x in diffs.items()}, {k: x[1] for k, x in diffs.items()})
    v2 = style_v2_report(run)["style_v2"]["groups"]
    newly_collapsed = [b for a, b in (("B1", "B2"), ("C1", "C2"))
                       if v2.get(a, {}).get("checks", {}).get("no_conditional_collapse") and not v2.get(b, {}).get("checks", {}).get("no_conditional_collapse")]
    new_ok = pooled.get("n", 0) > 0 and pooled["mean_diff"] > 0 and pooled["ci95"][0] > 0 and not newly_collapsed
    out = {"rule": PROMPT_RULE, "pooled_paired_new_minus_old": pooled, "newly_collapsed_groups": newly_collapsed,
           "selected": "new" if new_ok else "old", "selected_card_id": run.manifest["prompts"]["new" if new_ok else "old"]["card_id"],
           "per_pair": {f"{b}-{a}": J.paired(v(a), v(b)) for a, b in (("B1", "B2"), ("C1", "C2"))},
           "note": "recommendation only: the local current prompt is not changed by the Colab run"}
    _write(run.p("reports", "PROMPT_COMPARISON_REPORT.json"), out)
    return out


# ---------------------------------------------------------------- stage 13: memory QA generation (M0 / M1 / M2)
def stage_13_memory_qa(run: Run) -> dict:
    from .runtime import PersonaRuntime
    rows, qa_sha = QB.load_frozen(Path(run.run_dir) / "qa")
    sel = json.loads(run.p("reports", "PROMPT_COMPARISON_REPORT.json").read_text(encoding="utf-8"))
    card_id = sel["selected_card_id"]
    snap = json.loads(run.p("reports", "MEMORY_SNAPSHOT.json").read_text(encoding="utf-8"))
    if memory_snapshot(run)["content_sha256"] != snap["content_sha256"]:
        raise RuntimeError("MEMORY_SNAPSHOT_CHANGED: M1 and M2 must read the same memory snapshot")
    ask_time = datetime.fromisoformat(json.loads((Path(run.run_dir) / "qa" / "QA_BUILD_REPORT.json").read_text(encoding="utf-8"))["ask_time"])
    aid = run.manifest["adapter"]["adapter_id"]
    fp = {"qa_sha256": qa_sha, "card_id": card_id, "memory_snapshot": snap["content_sha256"], "adapter_sha256": run.manifest["adapter"]["sha256"],
          "decode": EV.DECODE, "ask_time": ask_time.isoformat(), "groups": list(MEMORY_GROUPS)}
    EV.bind_fingerprint(Path(run.run_dir) / "memory_qa", fp)
    # one loaded model; M0 differs from M1 only in the memory switch (no evidence, no reader, no retrieval: plain persona chat with the
    # same prompt, question and decode), M2 from M1 only in the adapter switch
    rts = {"M0": PersonaRuntime(run.work_root, run.profile_id, adapters=[aid], memory=None, generator=run.gen, card_id=card_id),
           "M1": PersonaRuntime(run.work_root, run.profile_id, adapters=[aid], memory="production", generator=run.gen, card_id=card_id)}
    rts["M2"] = rts["M1"]
    out = {}
    for g in MEMORY_GROUPS:
        path = run.p("memory_qa", g, "answers.jsonl")
        done = {r["question_id"] for r in _jsonl(path) if r.get("status") == "OK"}
        rt = rts[g]
        rt.use_adapter(aid if g == "M2" else None)
        todo = [q for q in rows if q["question_id"] not in done]
        run.log(f"memory QA {g}: {len(done)} done, {len(todo)} to run")
        for i, q in enumerate(todo):
            t0 = time.time()
            try:
                r = rt.reply([("other", q["question"])], now=ask_time)
                rd = r.get("reader") or {}
                retrieved = set(rd.get("retrieved_message_ids") or [])
                gold = set(q.get("evidence_message_ids") or [])
                row = {"question_id": q["question_id"], "group": g, "status": "OK", "text": r["text"], "raw_text": r.get("raw_text"), "route": r["route"],
                       "final_check": rd.get("final_check"), "reader_answerable": rd.get("answerable"), "reader_answer": rd.get("answer"), "reader_fail": rd.get("fail"),
                       "subject": rd.get("subject"), "retrieved": len(retrieved), "gold_retrieved": (len(gold & retrieved) / len(gold)) if gold else None,
                       "date_summary": rd.get("date_summary"), "adapter": r.get("adapter"), "generator": r.get("generator"), "loop_guard": r.get("loop_guard")}
            except Exception as e:  # noqa: BLE001 - recorded; retried on the next run
                row = {"question_id": q["question_id"], "group": g, "status": "ERROR", "error": f"{type(e).__name__}: {str(e)[:300]}"}
            row["wall_s"] = round(time.time() - t0, 2)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            if (i + 1) % 20 == 0:
                run.log(f"[{g}] {i + 1}/{len(todo)}")
        out[g] = sum(1 for r in _jsonl(path) if r.get("status") == "OK")
    ok = all(v >= len(rows) for v in out.values())
    run.status("STAGE 13 memory QA generation", "PASS" if ok else "INCOMPLETE", **{g: f"{v}/{len(rows)}" for g, v in out.items()}, card_id=card_id)
    return out


def _qa_answers(run: Run, g: str) -> Dict[str, dict]:
    out = {}
    for r in _jsonl(Path(run.run_dir) / "memory_qa" / g / "answers.jsonl"):
        if r.get("status") == "OK":
            out[r["question_id"]] = r
    return out


def _messages_by_id(run: Run) -> Dict[str, dict]:
    return {m["message_id"]: m for m in QB.load_messages(run.pdir / "memory" / "memory.sqlite", run.profile_id)}


def memory_groups(run: Run) -> List[str]:
    """The memory groups of this run (FAST / FULL: M0 / M1 / M2; the second-round evaluation sets ``cfg["memory_groups"]``)."""
    return list(run.cfg.get("memory_groups") or MEMORY_GROUPS)


def memory_units(run: Run) -> List[dict]:
    rows, _ = QB.load_frozen(Path(run.run_dir) / "qa")
    msgs = _messages_by_id(run)
    units = []
    for g in memory_groups(run):
        ans = _qa_answers(run, g)
        for q in rows:
            if q["question_id"] not in ans:
                continue
            ids = q.get("day_message_ids") or q.get("evidence_message_ids") or []
            ev = [{"id": f"M{i + 1:03d}", "time": msgs[m]["time"], "speaker": msgs[m]["speaker"], "text": msgs[m]["text"]} for i, m in enumerate(ids[:150]) if m in msgs]
            units.append({"unit_id": J.unit_id("memory", run.manifest["run_salt"], q["question_id"], g), "question_id": q["question_id"], "_group": g,
                          "question": q["question"], "category": q["category"], "subject": q["subject"], "time_scope": q.get("time_scope"),
                          "answerability": q["answerability"], "unanswerable_basis": q.get("unanswerable_basis"), "reference_answer": q["reference_answer"],
                          "reference_facts": q.get("reference_facts"), "evidence": ev, "response": ans[q["question_id"]]["text"]})
    return units


def stage_15_memory_judge(run: Run, env: Mapping[str, str], transport=None) -> dict:
    client, model = judge_client(run, env, transport)
    units = memory_units(run)
    _write(run.p("judge", "memory_unit_key.json"), {u["unit_id"]: {"question_id": u["question_id"], "group": u["_group"]} for u in units})
    res = J.run(client, model, "memory", [{k: v for k, v in u.items() if not k.startswith("_")} for u in units], run.p("judge", "MEMORY_DEEPSEEK_SCORES.jsonl"),
                seed=run.manifest["run_salt"], concurrency=int(run.cfg["judge_concurrency"]), per_second=float(run.cfg["judge_per_second"]), log=run.log)
    run.status("STAGE 15 memory judge", "PASS" if res["failed"] == 0 else "PARTIAL", ok=res["ok"], failed=res["failed"])
    return res


# ---------------------------------------------------------------- stage 16: writer audit judge
def writer_units(run: Run) -> List[dict]:
    sample = json.loads(run.p("writer_audit", "WRITER_AUDIT_SAMPLE.json").read_text(encoding="utf-8"))["sessions"]
    msgs = QB.load_messages(run.pdir / "memory" / "memory.sqlite", run.profile_id)
    by_conv: Dict[str, List[dict]] = defaultdict(list)
    for m in msgs:
        by_conv[m["conversation_id"]].append(m)
    logs: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for r in _jsonl(writer_pdir(run) / "memory" / "writer_candidates.jsonl"):
        logs[r["conversation_id"]][r["group_key"]] = r                  # latest line per chunk
    units = []
    for s in sample:
        cid = s["conversation_id"]
        ms = by_conv.get(cid, [])
        label = {m["message_id"]: f"m{i + 1}" for i, m in enumerate(ms)}
        cands = []
        for key in sorted(logs.get(cid, {})):
            for c in logs[cid][key]["candidates"]:
                cands.append({"status": c["status"], "subject": c["subject"], "predicate": c["predicate"], "object": c["object"], "event_status": c.get("event_status"),
                              "time": c.get("event_time") or c.get("planned_time") or c.get("valid_from"),
                              "labels": [label.get(x, x) for x in c.get("source_message_ids") or []],
                              "reasons": "; ".join(c.get("reasons") or []) if c["status"] != "AUTO_VERIFIED" else ""})
        units.append({"unit_id": J.unit_id("writer", run.manifest["run_salt"], cid), "conversation_id": cid, "stratum": s["stratum"],
                      "messages": [{"label": label[m["message_id"]], "time": m["time"], "speaker": m["speaker"], "text": m["text"]} for m in ms], "candidates": cands})
    return units


def stage_16_writer_judge(run: Run, env: Mapping[str, str], transport=None) -> dict:
    client, model = judge_client(run, env, transport)
    units = writer_units(run)
    with open(run.p("writer_audit", "WRITER_AUDIT_UNITS.jsonl"), "w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")
    _write(run.p("judge", "writer_unit_key.json"), {u["unit_id"]: {"conversation_id": u["conversation_id"], "group": "writer", "sample_id": u["conversation_id"]} for u in units})
    res = J.run(client, model, "writer", [{k: v for k, v in u.items() if k in ("unit_id", "messages", "candidates")} for u in units],
                run.p("judge", "WRITER_DEEPSEEK_SCORES.jsonl"), seed=run.manifest["run_salt"], concurrency=int(run.cfg["judge_concurrency"]),
                per_second=float(run.cfg["judge_per_second"]), log=run.log)
    run.status("STAGE 16 writer judge", "PASS" if res["failed"] == 0 else "PARTIAL", ok=res["ok"], failed=res["failed"])
    return res


# ---------------------------------------------------------------- stage 17-18: analysis + recommendation
READER_FAILS = ("RETRIEVAL_FAIL", "SUBJECT_RESOLUTION_FAIL", "READER_FAIL", "GROUNDEDNESS_FAIL", "ANSWER_PRESERVATION_FAIL")


def reader_taxonomy(run: Run) -> dict:
    rows, _ = QB.load_frozen(Path(run.run_dir) / "qa")
    by_q = {q["question_id"]: q for q in rows}
    out = {}
    for g in memory_groups(run):
        c = Counter()
        per_cat: Dict[str, Counter] = defaultdict(Counter)
        for qid, a in _qa_answers(run, g).items():
            q = by_q[qid]
            if q["answerability"] != "ANSWERABLE":
                kind = "UNANSWERABLE_ABSTAINED" if not a.get("reader_answerable") else "UNANSWERABLE_ANSWERED"
            elif a.get("reader_fail") == "SUBJECT_RESOLUTION_FAIL":
                kind = "SUBJECT_RESOLUTION_FAIL"
            elif a.get("reader_fail") == "GROUNDEDNESS_FAIL":
                kind = "GROUNDEDNESS_FAIL"
            elif a.get("route") == "chat":
                kind = "ROUTED_TO_CHAT"
            elif a.get("gold_retrieved") == 0:
                kind = "RETRIEVAL_FAIL"
            elif not a.get("reader_answerable"):
                kind = "READER_FAIL"
            elif a.get("final_check") == "FALLBACK_TO_READER_ANSWER":
                kind = "ANSWER_PRESERVATION_FAIL"
            else:
                kind = "READER_ANSWERED"
            c[kind] += 1
            per_cat[q["category"]][kind] += 1
        gr = [a.get("gold_retrieved") for a in _qa_answers(run, g).values() if a.get("gold_retrieved") is not None]
        out[g] = {"outcomes": dict(c), "by_category": {k: dict(v) for k, v in per_cat.items()},
                  "gold_evidence_recall_mean": round(sum(gr) / len(gr), 4) if gr else None,
                  "routes": dict(Counter(a.get("route") for a in _qa_answers(run, g).values()))}
    return out


def stage_17_18_analysis(run: Run) -> dict:
    rows, qa_sha = QB.load_frozen(Path(run.run_dir) / "qa")
    by_q = {q["question_id"]: q for q in rows}
    sty = _scores(run, "style", "style_unit_key.json")
    mem = _scores(run, "memory", "memory_unit_key.json")
    wri = J.read_results(run.p("judge", "WRITER_DEEPSEEK_SCORES.jsonl"))
    sel = json.loads(run.p("reports", "PROMPT_COMPARISON_REPORT.json").read_text(encoding="utf-8"))
    v2 = style_v2_report(run)
    _write(run.p("reports", "STYLE_V2_REPORT.json"), v2)
    style = {g: {"deepseek_style_score": J.describe([r["score_100"] for r in sty.get(g, {}).values()]),
                 "dimensions_mean_100": {d: round(sum(r["dimensions_100"][d] for r in sty[g].values()) / len(sty[g]), 2) for d in J.STYLE_WEIGHTS} if sty.get(g) else {},
                 "hallucination_rate": round(sum(bool(r["judgement"].get("hallucination")) for r in sty[g].values()) / len(sty[g]), 4) if sty.get(g) else None,
                 "style_v2_checks": v2["style_v2"]["groups"].get(g, {}).get("checks"), "v1_signals": v2["v1_signals"].get(g)} for g in run.cfg["style_groups"]}
    memory = {}
    for g in MEMORY_GROUPS:
        rs = mem.get(g, {})
        memory[g] = {"deepseek_memory_score": J.describe([r["score_100"] for r in rs.values()]),
                     "dimensions_mean_100": {d: round(sum(r["dimensions_100"][d] for r in rs.values()) / len(rs), 2) for d in J.MEMORY_WEIGHTS} if rs else {},
                     "persona_style_mean_0_5": round(sum(r["judgement"]["persona_style"] for r in rs.values()) / len(rs), 2) if rs else None,
                     "fabricated_private_fact_rate": round(sum(bool(r["judgement"].get("fabricated_private_fact")) for r in rs.values()) / len(rs), 4) if rs else None,
                     "by_category": {c: J.describe([r["score_100"] for q, r in rs.items() if by_q[q]["category"] == c]) for c in QB.TARGETS}}
    sel_b, sel_c = ("B2", "C2") if sel["selected"] == "new" else ("B1", "C1")
    combined = {}
    for g, sg in (("M0", sel_b), ("M1", sel_b), ("M2", sel_c)):
        s_mean, m_mean = style[sg]["deepseek_style_score"].get("mean"), memory[g]["deepseek_memory_score"].get("mean")
        combined[g] = {"style_component_group": sg, "style_score": s_mean, "memory_score": m_mean,
                       "combined_score": round(STYLE_WEIGHT * s_mean + MEMORY_WEIGHT * m_mean, 2) if s_mean is not None and m_mean is not None else None}
    paired_mem = J.paired({k: r["score_100"] for k, r in mem.get("M1", {}).items()}, {k: r["score_100"] for k, r in mem.get("M2", {}).items()})
    paired_sty = J.paired({k: r["score_100"] for k, r in sty.get(sel_b, {}).items()}, {k: r["score_100"] for k, r in sty.get(sel_c, {}).items()})
    fab = {g: memory[g]["fabricated_private_fact_rate"] for g in MEMORY_GROUPS}
    c_ok = (style[sel_c]["style_v2_checks"] or {}).get("no_conditional_collapse")
    enable = (combined["M2"]["combined_score"] is not None and combined["M1"]["combined_score"] is not None and combined["M2"]["combined_score"] > combined["M1"]["combined_score"]
              and paired_mem.get("n", 0) > 0 and paired_mem["ci95"][0] > -2 and fab["M2"] is not None and fab["M1"] is not None and fab["M2"] <= fab["M1"] + 0.02
              and paired_sty.get("n", 0) > 0 and paired_sty["ci95"][0] > 0 and bool(c_ok))
    writer_rows = [r for r in wri.values() if r.get("status") == "OK"]
    sup = sum(len(r["judgement"].get("supported") or []) for r in writer_rows)
    uns = sum(len(r["judgement"].get("unsupported") or []) for r in writer_rows)
    mis = sum(len(r["judgement"].get("missing") or []) for r in writer_rows)
    writer = {"deepseek_writer_score": J.describe([r["score_100"] for r in writer_rows]),
              "JUDGE_ESTIMATED_PRECISION": round(sup / (sup + uns), 4) if sup + uns else None, "JUDGE_ESTIMATED_RECALL": round(sup / (sup + mis), 4) if sup + mis else None,
              "supported": sup, "unsupported": uns, "suspected_missing": mis,
              "attribution_errors": sum(len(r["judgement"].get("attribution_errors") or []) for r in writer_rows),
              "temporal_errors": sum(len(r["judgement"].get("temporal_errors") or []) for r in writer_rows),
              "worthiness_errors": sum(len(r["judgement"].get("worthiness_errors") or []) for r in writer_rows),
              "note": "DeepSeek-assisted audit estimates, NOT human-annotated precision / recall"}
    reader = reader_taxonomy(run)
    rec = {"prompt": {"selected": sel["selected"], "card_id": sel["selected_card_id"], "rule": PROMPT_RULE},
           "adapter": {"recommend_enable": bool(enable), "adapter_id": run.manifest["adapter"]["adapter_id"], "rule": ADAPTER_RULE,
                       "evidence": {"combined": combined, "paired_memory_M2_minus_M1": paired_mem, "paired_style_C_minus_B": paired_sty,
                                    "fabricated_rate": fab, "C_sel_no_conditional_collapse": c_ok}},
           "memory": {"enable": True, "view": "production_full_history"},
           "note": "recommendation only; importing the results changes nothing until the owner applies it locally (persona runtime / review)"}
    abl = ablation(run, sty, mem, by_q, sel_b, sel_c, style)
    abl["memory_switch_note"] = ("M0 -> M1 switches the whole memory pathway: retrieval over the evidence archive, the Fact Reader route for history / fact / "
                                 "date questions and the answer-preservation render. M1 vs M0 therefore measures that pathway, not retrieval alone; the "
                                 "route distribution per group is reported in reader.<group>.routes.")
    abl["memory_mode"] = run.cfg.get("memory_mode")
    report = {"schema": RUN_SCHEMA, "qa_sha256": qa_sha, "style": style, "memory": memory, "combined": combined, "paired": {"memory_M2_minus_M1": paired_mem,
              "style_selected_C_minus_B": paired_sty}, "ablation": abl, "writer": writer, "reader": reader, "weights": {"style": STYLE_WEIGHT, "memory": MEMORY_WEIGHT},
              "combined_note": "StillHere product metric (0.40 Style + 0.60 Memory); NOT a LifeBench score and NOT the only acceptance criterion: read it "
                               "with the separate scores, the fabrication rate and the attribution errors"}
    _write(run.p("reports", "COMBINED_EVALUATION_REPORT.json"), report)
    _write(run.p("reports", "READER_EVALUATION_REPORT.json"), {"taxonomy": reader, "memory": memory})
    _write(run.p("reports", "WRITER_AUDIT_REPORT.json"), writer)
    _write(run.p("reports", "RUNTIME_RECOMMENDATION.json"), rec)
    _write(run.p("reports", "ABLATION_REPORT.json"), abl)
    wb = writer_breakdown(run, wri)
    _write(run.p("reports", "WRITER_ERROR_BREAKDOWN.json"), wb)
    run.p("reports", "WRITER_REPORT.md").write_text(writer_md(run, writer, wb), encoding="utf-8", newline="\n")
    run.p("reports", "ACADEMIC_RESULTS.md").write_text(academic_md(run, report), encoding="utf-8", newline="\n")
    run.p("reports", "INTERVIEW_SUMMARY_EN_CN.md").write_text(interview_md(run, report), encoding="utf-8", newline="\n")
    cases = case_studies(run, mem, by_q)
    (run.p("reports", "REAL_CASE_STUDIES_20_30.md")).write_text(cases["private_md"], encoding="utf-8", newline="\n")
    (run.p("reports", "ACADEMIC_DEMO_CASES.md")).write_text(cases["public_md"], encoding="utf-8", newline="\n")
    _write(run.p("reports", "CASE_STUDY_SELECTION.json"), cases["selection"])
    (run.p("reports", "STILLHERE_ABLATION_RESULTS.md")).write_text(ablation_md(report), encoding="utf-8", newline="\n")
    run.status("STAGE 17-18 analysis", "PASS", combined={g: v["combined_score"] for g, v in combined.items()}, prompt=sel["selected"], adapter=enable)
    return report


# ---------------------------------------------------------------- stage 19-20: export
def memory_exports(run: Run, out: Path) -> dict:
    wp = writer_pdir(run)
    rows = _jsonl(wp / "memory" / "writer_candidates.jsonl")
    latest = {}
    for r in rows:
        latest[r["group_key"]] = r
    counts = Counter()
    files = {n: open(out / n, "w", encoding="utf-8") for n in ("MEMORY_CANDIDATES.jsonl", "AMBIGUOUS_MEMORIES.jsonl", "REJECTED_MEMORIES.jsonl")}
    try:
        for key in sorted(latest):
            r = latest[key]
            for c in r["candidates"]:
                line = json.dumps({"persona_id": run.profile_id, "conversation_id": r["conversation_id"], "group_key": key, **c}, ensure_ascii=False, default=str) + "\n"
                files["MEMORY_CANDIDATES.jsonl"].write(line)
                counts[c["status"]] += 1
                if c["status"] == "AMBIGUOUS":
                    files["AMBIGUOUS_MEMORIES.jsonl"].write(line)
                elif c["status"] == "REJECTED":
                    files["REJECTED_MEMORIES.jsonl"].write(line)
    finally:
        for f in files.values():
            f.close()
    con = sqlite3.connect(str(wp / "memory" / "memory.sqlite"))
    con.row_factory = sqlite3.Row
    try:
        with open(out / "VERIFIED_MEMORIES.jsonl", "w", encoding="utf-8") as f:
            for m in con.execute("SELECT * FROM memories WHERE persona_id = ? ORDER BY memory_id", (run.profile_id,)):
                d = dict(m)
                f.write(json.dumps({**d, "verification_status": "AUTO_VERIFIED", "human_reviewed": False}, ensure_ascii=False, default=str) + "\n")
        with open(out / "MEMORY_LINEAGE.jsonl", "w", encoding="utf-8") as f:
            for m in con.execute("SELECT * FROM provenance ORDER BY rowid"):
                f.write(json.dumps({"type": "provenance", **dict(m)}, ensure_ascii=False, default=str) + "\n")
            for m in con.execute("SELECT * FROM memory_events ORDER BY rowid"):
                f.write(json.dumps({"type": "event", **dict(m)}, ensure_ascii=False, default=str) + "\n")
    finally:
        con.close()
    return dict(counts)


def stage_19_export(run: Run, env_report: Mapping) -> Path:
    out = Path(run.run_dir) / "export" / "files"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    counts = memory_exports(run, out)
    fast = run.cfg.get("memory_build") == "evidence_only"
    copy = {"reports/MEMORY_MODE.json" if fast else "reports/FULL_MEMORY_BUILD_REPORT.json": "MEMORY_MODE.json" if fast else "FULL_MEMORY_BUILD_REPORT.json",
            "reports/TEMPORAL_REPORT.json": "TEMPORAL_REPORT.json", "reports/STYLE_SAMPLE_MANIFEST.json": "STYLE_SAMPLE_MANIFEST.json",
            "reports/WRITER_ERROR_BREAKDOWN.json": "WRITER_ERROR_BREAKDOWN.json", "reports/WRITER_REPORT.md": "WRITER_REPORT.md",
            "reports/ACADEMIC_RESULTS.md": "ACADEMIC_RESULTS.md", "reports/INTERVIEW_SUMMARY_EN_CN.md": "INTERVIEW_SUMMARY_EN_CN.md",
            "STAGE_TIMINGS.json": "STAGE_TIMINGS.json",
            "reports/EF1_REPORT.json": "EF1_REPORT.json", "qa/REAL_HISTORY_QA_200.jsonl": "REAL_HISTORY_QA_200.jsonl", "qa/QA_BUILD_REPORT.json": "QA_BUILD_REPORT.json",
            "writer_audit/WRITER_AUDIT_UNITS.jsonl": "WRITER_AUDIT_100.jsonl", "reports/WRITER_AUDIT_REPORT.json": "WRITER_AUDIT_REPORT.json",
            "judge/STYLE_DEEPSEEK_SCORES.jsonl": "STYLE_DEEPSEEK_SCORES.jsonl", "reports/STYLE_V2_REPORT.json": "STYLE_V2_REPORT.json",
            "judge/MEMORY_DEEPSEEK_SCORES.jsonl": "MEMORY_DEEPSEEK_SCORES.jsonl", "judge/WRITER_DEEPSEEK_SCORES.jsonl": "WRITER_DEEPSEEK_SCORES.jsonl",
            "reports/READER_EVALUATION_REPORT.json": "READER_EVALUATION_REPORT.json", "reports/COMBINED_EVALUATION_REPORT.json": "COMBINED_EVALUATION_REPORT.json",
            "reports/PROMPT_COMPARISON_REPORT.json": "PROMPT_COMPARISON_REPORT.json", "reports/RUNTIME_RECOMMENDATION.json": "RUNTIME_RECOMMENDATION.json",
            "reports/JUDGE_CALIBRATION_REPORT.json": "JUDGE_CALIBRATION_REPORT.json", "reports/SYNTHETIC_GOLD_REPORT.json": "SYNTHETIC_GOLD_REPORT.json",
            "reports/MEMORY_SNAPSHOT.json": "MEMORY_SNAPSHOT.json", "judge/style_unit_key.json": "judge_keys/style_unit_key.json",
            "reports/ABLATION_REPORT.json": "ABLATION_REPORT.json", "reports/REAL_CASE_STUDIES_20_30.md": "REAL_CASE_STUDIES_20_30.md",
            "reports/ACADEMIC_DEMO_CASES.md": "ACADEMIC_DEMO_CASES.md", "reports/CASE_STUDY_SELECTION.json": "CASE_STUDY_SELECTION.json",
            "reports/STILLHERE_ABLATION_RESULTS.md": "STILLHERE_ABLATION_RESULTS.md",
            "judge/memory_unit_key.json": "judge_keys/memory_unit_key.json", "judge/writer_unit_key.json": "judge_keys/writer_unit_key.json",
            "STAGE_STATUS.json": "STAGE_STATUS.json", "RUN_FINGERPRINT.json": "RUN_FINGERPRINT.json"}
    missing = []
    optional = {"STYLE_SAMPLE_MANIFEST.json"} if not fast else set()
    for src, dst in copy.items():
        p = Path(run.run_dir) / src
        if p.is_file():
            (out / dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, out / dst)
        elif dst not in optional:
            missing.append(dst)
    for kind, groups in (("STYLE_DEV_GENERATIONS.jsonl", ("style", run.cfg["style_groups"])), ("MEMORY_QA_GENERATIONS.jsonl", ("memory_qa", MEMORY_GROUPS))):
        with open(out / kind, "w", encoding="utf-8") as f:
            for g in groups[1]:
                for r in _jsonl(Path(run.run_dir) / groups[0] / g / "answers.jsonl"):
                    f.write(json.dumps({**r, "group": g}, ensure_ascii=False, default=str) + "\n")
    if not fast:                                     # FAST built no semantic memory: nothing to import, no snapshot
        shutil.copyfile(run.pdir / "memory" / "memory.sqlite", out / "memory_snapshot.sqlite")
    ledger = J.make_ledger_summary(run.p("judge", "ledger.jsonl"))
    n_qa = len(QB.load_frozen(Path(run.run_dir) / "qa")[0]) if (Path(run.run_dir) / "qa" / "REAL_HISTORY_QA_200.jsonl").is_file() else 0
    planned = {"style": len(style_units(run)) if (Path(run.run_dir) / "style").is_dir() else len(run.cfg["style_groups"]) * 268,
               "memory": len(MEMORY_GROUPS) * n_qa, "writer": int(run.cfg["writer_audit_n"])}
    target = {k: v for k, v in plan_units(run.cfg).items() if k in ("style", "memory", "writer")}
    ok = {k: sum(1 for r in J.read_results(run.p("judge", f"{k.upper()}_DEEPSEEK_SCORES.jsonl")).values() if r.get("status") == "OK") for k in planned}
    usage = {"target_units": target, "target_total": sum(target.values()), "planned_units": planned, "planned_total": sum(planned.values()), "ok_units": ok,
             "ok_total": sum(ok.values()), "planned_note": "planned = units that exist (frozen QA count x 3 groups, generated DEV rows x 4 groups, sampled sessions); "
                                                           "a QA category shortfall is reported in QA_BUILD_REPORT, never padded",
             "failed_units": {k: planned[k] - ok[k] for k in planned}, "api": ledger, "judge_model": J.judge_model(os.environ),
             "rubric_version": J.RUBRIC_VERSION, "prompt_version": J.PROMPT_VERSION, "rubric_sha256": J.rubric_sha(),
             "note": "planned units are the target; actual HTTP requests include repair turns, retries and calibration (see api.requests)"}
    _write(out / "JUDGE_USAGE_REPORT.json", usage)
    errors = {"stage_status": json.loads((Path(run.run_dir) / "STAGE_STATUS.json").read_text(encoding="utf-8")) if (Path(run.run_dir) / "STAGE_STATUS.json").is_file() else {},
              "judge_failed": {k: J.run_failures(run.p("judge", f"{k.upper()}_DEEPSEEK_SCORES.jsonl")) for k in planned},
              "writer_failed_queue": S.read_json(writer_pdir(run) / "memory" / "WRITER_FAILED_QUEUE.json") if (writer_pdir(run) / "memory" / "WRITER_FAILED_QUEUE.json").is_file() else {},
              "missing_files": missing}
    _write(out / "ERROR_QUEUE.json", errors)
    _write(out / "ENVIRONMENT_REPORT.json", dict(env_report))
    man = {"schema": RUN_SCHEMA, "profile_id": run.profile_id, "run_dir": str(run.run_dir), "package": {k: run.manifest.get(k) for k in (
        "package_id", "input_zip_sha256", "code_tree_sha256", "git_commit", "dataset_version", "base", "adapter", "prompts")},
           "judge": {"model": J.judge_model(os.environ), "rubric_version": J.RUBRIC_VERSION, "rubric_sha256": J.rubric_sha()},
           "qa_sha256": json.loads((Path(run.run_dir) / "qa" / "QA_BUILD_REPORT.json").read_text(encoding="utf-8"))["qa_sha256"] if (Path(run.run_dir) / "qa" / "QA_BUILD_REPORT.json").is_file() else None,
           "memory_snapshot": json.loads(run.p("reports", "MEMORY_SNAPSHOT.json").read_text(encoding="utf-8")) if run.p("reports", "MEMORY_SNAPSHOT.json").is_file() else None,
           "candidate_counts": counts, "judge_ok_units": usage["ok_total"], "judge_planned_units": usage["planned_total"], "judge_target_units": usage["target_total"],
           "memory_groups": list(MEMORY_GROUPS), "style_groups": list(run.cfg["style_groups"]), "missing_files": missing,
           "eval_mode": run.cfg.get("eval_mode", "FULL"), "memory_mode": run.cfg.get("memory_mode"),
           "labels": (["Evidence-only Memory Evaluation", "Semantic Memory Full Build Not Executed", "Experimental Evaluation Results"] if fast
                      else ["Full Structured Semantic Memory Build", "Experimental Evaluation Results"]),
           "sizes": {"style_items": len(dev_items(run)) if (Path(run.run_dir) / "style").is_dir() else None, "qa": n_qa, "writer_sessions": int(run.cfg["writer_audit_n"])},
           "contains_test": False, "human_review": "NONE (machine judge + deterministic verification only)", "created_at": datetime.now(timezone.utc).isoformat()}
    _write(out / "RUN_MANIFEST.json", man)
    _write(out / "EVAL_MODE.json", {"eval_mode": man["eval_mode"], "memory_mode": man["memory_mode"], "labels": man["labels"], "sizes": man["sizes"],
                                    "judge_target_units": usage["target_total"], "generations_target": plan_units(run.cfg)["generations"]})
    (out / "FINAL_SUMMARY_CN.md").write_text(summary_cn(run, man, usage), encoding="utf-8", newline="\n")
    for src, dst in export_names(man["eval_mode"]).items():      # FAST names (the content is the same schema)
        if (out / src).is_file() and src != dst:
            os.replace(out / src, out / dst)
    names = sorted(str(p.relative_to(out)).replace("\\", "/") for p in out.rglob("*") if p.is_file())
    (out / "SHA256SUMS.txt").write_text("".join(f"{_sha_file(out / n)}  {n}\n" for n in names), encoding="utf-8", newline="\n")
    dest = Path(run.run_dir) / "export" / run.cfg.get("results_zip", "StillHere_Final_L4_Results.zip")
    tmp = dest.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names + ["SHA256SUMS.txt"]:
            z.write(out / n, n)
    os.replace(tmp, dest)
    run.status("STAGE 19 export", "PASS", zip=str(dest), size_mb=round(dest.stat().st_size / 2**20, 1), missing=missing)
    return dest


FAST_NAMES = {"REAL_HISTORY_QA_200.jsonl": "MEMORY_QA_100.jsonl", "STYLE_DEV_GENERATIONS.jsonl": "STYLE_FAST_GENERATIONS.jsonl",
              "STYLE_DEEPSEEK_SCORES.jsonl": "STYLE_FAST_DEEPSEEK_SCORES.jsonl", "MEMORY_QA_GENERATIONS.jsonl": "MEMORY_FAST_GENERATIONS.jsonl",
              "MEMORY_DEEPSEEK_SCORES.jsonl": "MEMORY_FAST_DEEPSEEK_SCORES.jsonl", "ABLATION_REPORT.json": "MEMORY_ABLATION_REPORT.json",
              "WRITER_AUDIT_100.jsonl": "WRITER_AUDIT_20.jsonl", "WRITER_AUDIT_REPORT.json": "WRITER_FAST_REPORT.json", "WRITER_REPORT.md": "WRITER_FAST_REPORT.md",
              "STILLHERE_ABLATION_RESULTS.md": "FAST_ABLATION_REPORT.md", "REAL_CASE_STUDIES_20_30.md": "FAST_CASE_STUDIES.md",
              "ACADEMIC_RESULTS.md": "FAST_ACADEMIC_RESULTS.md", "INTERVIEW_SUMMARY_EN_CN.md": "FAST_INTERVIEW_SUMMARY_EN_CN.md",
              "FINAL_SUMMARY_CN.md": "FAST_EVAL_SUMMARY_CN.md", "MEMORY_CANDIDATES.jsonl": "WRITER_AUDIT_CANDIDATES.jsonl",
              "VERIFIED_MEMORIES.jsonl": "WRITER_AUDIT_VERIFIED.jsonl", "AMBIGUOUS_MEMORIES.jsonl": "WRITER_AUDIT_AMBIGUOUS.jsonl",
              "REJECTED_MEMORIES.jsonl": "WRITER_AUDIT_REJECTED.jsonl", "MEMORY_LINEAGE.jsonl": "WRITER_AUDIT_LINEAGE.jsonl"}


def export_names(mode: str) -> Dict[str, str]:
    """canonical export name -> the name in the results ZIP of this mode (identity in FULL)."""
    return dict(FAST_NAMES) if str(mode).upper() == "FAST" else {}


def summary_cn(run: Run, man: Mapping, usage: Mapping) -> str:
    def rd(name):
        p = run.p("reports", name)
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    comb, rec, mb = rd("COMBINED_EVALUATION_REPORT.json"), rd("RUNTIME_RECOMMENDATION.json"), rd("FULL_MEMORY_BUILD_REPORT.json")
    fast = man.get("eval_mode") == "FAST"
    mm = rd("MEMORY_MODE.json").get("evidence_archive", {}) if fast else {}
    mem_line = (f"- 记忆模式：**EVIDENCE_MEMORY**（完整证据档案 {mm.get('messages')} 条消息、{mm.get('text_messages')} 条文字可检索；"
                f"**没有建立结构化语义记忆**）；Writer 只在 {run.cfg['writer_audit_n']} 个审计会话上真实运行，结果不导入生产记忆") if fast else \
        f"- 全量记忆：输入 {mb.get('input_messages')} 条消息，{mb.get('sessions')} 个会话；有效记忆 {mb.get('memories_active')} 条（自动验证，**不是人工审核**）"
    lines = [f"# StillHere {'FAST_EVAL' if fast else '最终'} Colab L4 评测摘要", "", f"- 模式：{man.get('eval_mode')}；规模：{json.dumps(man.get('sizes'), ensure_ascii=False)}",
             f"- Profile：{run.profile_id}", f"- 评测题集 sha256：{(man.get('qa_sha256') or '')[:16]}",
             f"- Judge：{man['judge']['model']}（rubric {J.RUBRIC_VERSION}）；成功评分 {usage['ok_total']}/{usage['planned_total']}（目标 {usage['target_total']}）",
             mem_line, ""]
    if comb:
        lines += ["## Style（DeepSeek 0–100）", "", "| 组 | 均值 | 中位数 | 标准差 | n |", "|---|---:|---:|---:|---:|"]
        for g, v in comb["style"].items():
            d = v["deepseek_style_score"]
            lines.append(f"| {g} | {d.get('mean')} | {d.get('median')} | {d.get('sd')} | {d.get('n')} |")
        lines += ["", "## Memory（DeepSeek 0–100）与综合分", "", "| 组 | Memory 均值 | Style 分（对应组） | 综合分 0.4S+0.6M | 编造率 |", "|---|---:|---:|---:|---:|"]
        for g in MEMORY_GROUPS:
            c, m = comb["combined"][g], comb["memory"][g]
            lines.append(f"| {g} | {c['memory_score']} | {c['style_score']}（{c['style_component_group']}） | {c['combined_score']} | {m['fabricated_private_fact_rate']} |")
        w = comb["writer"]
        lines += ["", f"## Writer 审计（{w['deepseek_writer_score'].get('n')} 个会话）", "",
                  f"- 平均分 {w['deepseek_writer_score'].get('mean')}；估计精确率 {w['JUDGE_ESTIMATED_PRECISION']}，估计召回率 {w['JUDGE_ESTIMATED_RECALL']}（DeepSeek 辅助估计，不是人工标注）"]
    if rec:
        lines += ["", "## 推荐（只是推荐，导入后不会自动生效）", "", f"- Prompt：{rec['prompt']['selected']}（{rec['prompt']['card_id']}）",
                  f"- r1 Adapter：{'建议开启' if rec['adapter']['recommend_enable'] else '保持关闭'}"]
    lines += ["", "综合分是 StillHere 自定义的产品指标，不是 LifeBench 分数；请同时看各项单独的分数。"]
    return "\n".join(lines) + "\n"



# ---------------------------------------------------------------- ablation (addendum): M1 vs M0, M2 vs M1, B2 vs B1, C1 vs B1, C2 vs B2
CASE_BUCKETS = ("memory_improved", "memory_no_help", "memory_introduced_error", "correct_abstention", "wrong_abstention", "third_party", "date_summary",
                "past_present", "adapter_style_better", "adapter_regression")
CASE_RULE = ("pre-registered: buckets in this order, each question in at most one bucket, at most 3 per bucket, ordered by sha256(run_salt + question_id); "
             "memory_improved M1-M0>=30; memory_no_help |M1-M0|<10 and M1<60; memory_introduced_error M1-M0<=-10 or M1 fabricates while M0 does not; "
             "correct_abstention UNANSWERABLE and M1 abstention>=4 (0-5); wrong_abstention ANSWERABLE and the M1 reader abstained; third_party / date_summary / "
             "past_present by category; adapter_style_better M2-M1 persona_style>=2 with M2 factual>=M1; adapter_regression M2-M1<=-20 or the M2 reply fell "
             "back to the reader answer; filled up to case_min (FULL 20 / FAST 15) from the remaining questions in the same order when buckets are small; "
             "capped at case_max (FULL 30 / FAST 20)")


def _cmp(a: Mapping[str, float], b: Mapping[str, float]) -> dict:
    da, db = J.describe(list(a.values())), J.describe(list(b.values()))
    pr = J.paired(a, b)
    rel = round(100.0 * (db["mean"] - da["mean"]) / da["mean"], 2) if da.get("n") and db.get("n") and da["mean"] else None
    return {"baseline": {k: da.get(k) for k in ("n", "mean", "median", "sd")}, "treatment": {k: db.get(k) for k in ("n", "mean", "median", "sd")},
            "paired": pr, "absolute_diff_points": pr.get("mean_diff"), "relative_improvement_pct": rel,
            "note": "absolute_diff_points = paired mean difference in score points (0-100 scale); relative_improvement_pct = (treatment - baseline) / baseline"}


def _abstains(text: str) -> bool:
    from .runtime import UNCERTAIN_CUE
    return bool(UNCERTAIN_CUE.search(text or ""))


def _rate(xs, f):
    return round(sum(1 for x in xs if f(x)) / len(xs), 4) if xs else None


def memory_metrics(run: Run, g: str, rs: Mapping[str, dict], by_q: Mapping[str, dict]) -> dict:
    ans = _qa_answers(run, g)
    A = [q for q in rs if by_q[q]["answerability"] == "ANSWERABLE"]
    U = [q for q in rs if by_q[q]["answerability"] != "ANSWERABLE"]

    def j(q, k):
        return rs[q]["judgement"].get(k)
    errs = Counter()
    for q in rs:
        if j(q, "factual_correctness") <= 2 and by_q[q]["answerability"] == "ANSWERABLE":
            errs["factual_error"] += 1
        if j(q, "temporal_attribution") <= 2:
            errs["temporal_or_attribution_error"] += 1
        if j(q, "evidence_grounding") <= 2:
            errs["ungrounded"] += 1
    by_cat = {c: [q for q in rs if by_q[q]["category"] == c] for c in QB.TARGETS}
    return {"n": len(rs), "grounded_answer_accuracy": _rate(A, lambda q: j(q, "factual_correctness") >= 4 and j(q, "evidence_grounding") >= 4),
            "unsupported_claim_rate": _rate(list(rs), lambda q: bool(j(q, "fabricated_private_fact"))),
            "abstention_rate_answerable": _rate(A, lambda q: _abstains(ans.get(q, {}).get("text"))),
            "abstention_rate_unanswerable": _rate(U, lambda q: _abstains(ans.get(q, {}).get("text"))),
            "judge_error_breakdown": dict(errs),
            "by_category": {c: J.describe([rs[q]["score_100"] for q in qq]).get("mean") for c, qq in by_cat.items()},
            "temporal_attribution_by_category_0_5": {c: (round(sum(j(q, "temporal_attribution") for q in qq) / len(qq), 2) if qq else None) for c, qq in by_cat.items()}}


def ablation(run: Run, sty, mem, by_q, sel_b: str, sel_c: str, style: Mapping) -> dict:
    def sv(g):
        return {k: r["score_100"] for k, r in sty.get(g, {}).items()}

    def mv(g):
        return {k: r["score_100"] for k, r in mem.get(g, {}).items()}

    def sd(g, d):
        return {k: r["dimensions_100"][d] for k, r in sty.get(g, {}).items()}

    def fallback(a):
        return round(sum(1 for r in a.values() if r.get("final_check") == "FALLBACK_TO_READER_ANSWER") / len(a), 4) if a else None
    mm = {g: memory_metrics(run, g, mem.get(g, {}), by_q) for g in MEMORY_GROUPS}
    m1, m2 = _qa_answers(run, "M1"), _qa_answers(run, "M2")
    broke = [q for q in m1 if q in m2 and m1[q].get("reader_answerable") and q in mem.get("M1", {}) and q in mem.get("M2", {})
             and mem["M2"][q]["judgement"]["factual_correctness"] <= mem["M1"][q]["judgement"]["factual_correctness"] - 2]
    v2g = style_v2_report(run)["style_v2"]["groups"]
    groups = run.cfg["style_groups"]
    comps = {
        "A_memory_contribution_M1_vs_M0": {"question": "same model and prompt: does memory improve historical QA?", **_cmp(mv("M0"), mv("M1"))},
        "B_adapter_contribution_M2_vs_M1": {"question": "same memory and prompt: does r1 improve the persona's expression without losing facts?",
                                            **_cmp(mv("M1"), mv("M2")),
                                            "persona_style_0_5": {g: round(sum(r["judgement"]["persona_style"] for r in mem.get(g, {}).values()) / max(1, len(mem.get(g, {}))), 2)
                                                                  for g in ("M1", "M2")}},
        "C_prompt_contribution_B2_vs_B1": {"question": "without the adapter: does the new prompt improve persona style?", **_cmp(sv("B1"), sv("B2"))},
        "D1_adapter_style_C1_vs_B1": {"question": "old prompt: does QLoRA improve style (or cause short-reply repetition)?", **_cmp(sv("B1"), sv("C1"))},
        "D2_adapter_style_C2_vs_B2": {"question": "new prompt: does QLoRA improve style (or cause short-reply repetition)?", **_cmp(sv("B2"), sv("C2"))},
    }
    diag = {
        "memory_reduces_wrong_abstention": {g: mm[g]["abstention_rate_answerable"] for g in ("M0", "M1")},
        "memory_reduces_fabrication": {g: mm[g]["unsupported_claim_rate"] for g in ("M0", "M1")},
        "memory_past_vs_present_TEMPORAL_CHANGE_score": {g: mm[g]["by_category"].get("TEMPORAL_CHANGE") for g in MEMORY_GROUPS},
        "memory_attribution_0_5": {g: {c: mm[g]["temporal_attribution_by_category_0_5"].get(c) for c in ("PERSONA_SELF_EXPERIENCE", "COUNTERPART_EXPERIENCE", "THIRD_PARTY")}
                                   for g in MEMORY_GROUPS},
        "adapter_breaks_reader_answer": {"fallback_to_reader_answer_rate": {"M1": fallback(m1), "M2": fallback(m2)},
                                         "reader_answerable_but_M2_factual_down_by_2plus": len(broke)},
        "adapter_short_reply_style": {"style_fidelity_points": {"C1_vs_B1": J.paired(sd("B1", "style_fidelity"), sd("C1", "style_fidelity")),
                                                                "C2_vs_B2": J.paired(sd("B2", "style_fidelity"), sd("C2", "style_fidelity"))},
                                      "authentic_short_rate": {g: v2g.get(g, {}).get("authentic_short_rate") for g in groups}},
        "adapter_unconditional_repetition": {"non_repetition_points": {"C1_vs_B1": J.paired(sd("B1", "non_repetition"), sd("C1", "non_repetition")),
                                                                       "C2_vs_B2": J.paired(sd("B2", "non_repetition"), sd("C2", "non_repetition"))},
                                             "distinct_ratio": {g: v2g.get(g, {}).get("distinct_ratio") for g in groups},
                                             "excess_repetition_patterns": {g: len(v2g.get(g, {}).get("excess_repetition") or []) for g in groups},
                                             "style_v2_checks": {g: (style.get(g) or {}).get("style_v2_checks") for g in groups}},
    }
    return {"comparisons": comps, "memory_metrics": mm, "diagnostics": diag, "selected_style_groups": {"B_sel": sel_b, "C_sel": sel_c},
            "reading_rule": "prefer paired differences with their 95% CI; a CI that includes 0 is NOT an established improvement"}


def case_studies(run: Run, mem, by_q) -> dict:
    ans = {g: _qa_answers(run, g) for g in MEMORY_GROUPS}

    def sc(g, q):
        return (mem.get(g, {}).get(q) or {}).get("score_100")

    def jd(g, q, k):
        return ((mem.get(g, {}).get(q) or {}).get("judgement") or {}).get(k)
    qs = [q for q in by_q if all(sc(g, q) is not None for g in MEMORY_GROUPS)]
    qs.sort(key=lambda q: hashlib.sha256((run.manifest.get("run_salt", "") + q).encode()).hexdigest())
    tests = {
        "memory_improved": lambda q: sc("M1", q) - sc("M0", q) >= 30,
        "memory_no_help": lambda q: abs(sc("M1", q) - sc("M0", q)) < 10 and sc("M1", q) < 60,
        "memory_introduced_error": lambda q: sc("M1", q) - sc("M0", q) <= -10 or bool(jd("M1", q, "fabricated_private_fact") and not jd("M0", q, "fabricated_private_fact")),
        "correct_abstention": lambda q: by_q[q]["answerability"] != "ANSWERABLE" and (jd("M1", q, "appropriate_abstention") or 0) >= 4,
        "wrong_abstention": lambda q: by_q[q]["answerability"] == "ANSWERABLE" and ans["M1"].get(q, {}).get("reader_answerable") is False,
        "third_party": lambda q: by_q[q]["category"] == "THIRD_PARTY",
        "date_summary": lambda q: by_q[q]["category"] == "DATE_SUMMARY",
        "past_present": lambda q: by_q[q]["category"] == "TEMPORAL_CHANGE",
        "adapter_style_better": lambda q: (jd("M2", q, "persona_style") or 0) - (jd("M1", q, "persona_style") or 0) >= 2
        and (jd("M2", q, "factual_correctness") or 0) >= (jd("M1", q, "factual_correctness") or 0),
        "adapter_regression": lambda q: sc("M2", q) - sc("M1", q) <= -20 or ans["M2"].get(q, {}).get("final_check") == "FALLBACK_TO_READER_ANSWER",
    }
    chosen: List[tuple] = []
    used = set()
    for b in CASE_BUCKETS:
        for q in [q for q in qs if q not in used and tests[b](q)][:3]:
            chosen.append((b, q))
            used.add(q)
    for q in qs:
        if len(chosen) >= int(run.cfg.get("case_min", 20)):
            break
        if q not in used:
            chosen.append(("fill", q))
            used.add(q)
    chosen = chosen[:int(run.cfg.get("case_max", 30))]

    def outcome(a):
        return a.get("reader_fail") or ("answered" if a.get("reader_answerable") else ("abstained" if a.get("route") in ("fact", "date") else "no reader"))

    def analysis(q):
        d01, d12 = sc("M1", q) - sc("M0", q), sc("M2", q) - sc("M1", q)
        a1 = ans["M1"].get(q, {})
        gr = a1.get("gold_retrieved")
        return (f"Memory changed the score by {d01:+.1f} points (M1 vs M0); the adapter by {d12:+.1f} (M2 vs M1). M1 route {a1.get('route')}, "
                f"reader {outcome(a1)}{', gold evidence retrieved ' + format(gr, '.0%') if gr is not None else ''}.")
    priv = ["# Real case studies (PRIVATE: real chat text; never commit or publish)", "", f"Selection rule: {CASE_RULE}", ""]
    pub = ["# StillHere academic demo cases (de-identified)", "",
           "Derived from the real-history QA run. No question text, chat text, names or answers are shown: only the category, the subject kind, "
           "the pre-registered case bucket, the judge scores and the system's route / outcome. Cases are chosen by a fixed rule, not by result.", "",
           "| # | Bucket | Category | Subject | M0 | M1 | M2 | M1 route / outcome | Analysis |", "|---:|---|---|---|---:|---:|---:|---|---|"]
    for i, (b, q) in enumerate(chosen, 1):
        r = by_q[q]
        priv += [f"## {i}. [{b}] {r['category']} · {q}", "", f"- Question: {r['question']}",
                 f"- Reference facts: {'；'.join(r.get('reference_facts') or []) or r['reference_answer']}",
                 f"- Evidence message ids: {', '.join(r.get('evidence_message_ids') or []) or '(none: ' + str(r.get('unanswerable_basis')) + ')'}"]
        for g in MEMORY_GROUPS:
            j = mem[g][q].get("judgement") or {}
            priv.append(f"- {g}: {ans[g].get(q, {}).get('text')!r} — score {sc(g, q)} (factual {j.get('factual_correctness')}, grounding {j.get('evidence_grounding')}, "
                        f"time/attribution {j.get('temporal_attribution')}, abstention {j.get('appropriate_abstention')}, style {j.get('persona_style')}, "
                        f"fabricated {j.get('fabricated_private_fact')})")
        priv += [f"- Error category: {b}; M1 reader outcome: {outcome(ans['M1'].get(q, {}))}", f"- Short analysis: {analysis(q)}", ""]
        a1 = ans["M1"].get(q, {})
        pub.append(f"| {i} | {b} | {r['category']} | {r['subject']} | {sc('M0', q)} | {sc('M1', q)} | {sc('M2', q)} | {a1.get('route')} / {outcome(a1)} | {analysis(q)} |")
    sel = {"rule": CASE_RULE, "n": len(chosen), "by_bucket": dict(Counter(b for b, _ in chosen)), "cases": [{"bucket": b, "question_id": q} for b, q in chosen]}
    return {"private_md": "\n".join(priv) + "\n", "public_md": "\n".join(pub) + "\n", "selection": sel}


def ablation_md(report: Mapping) -> str:
    a = report["ablation"]
    L = ["# StillHere ablation results (filled from this run)", "",
         "Absolute differences are in score points on the 0-100 judge scale; relative improvement is (treatment - baseline) / baseline. "
         "A 95% CI that includes 0 is not an established difference.", "",
         "| Comparison | Baseline mean | Treatment mean | Paired diff (points) | 95% CI | Relative | W / T / L | n |", "|---|---:|---:|---:|---|---:|---|---:|"]
    for k, c in a["comparisons"].items():
        p = c["paired"]
        L.append(f"| {k} | {c['baseline'].get('mean')} | {c['treatment'].get('mean')} | {p.get('mean_diff')} | {p.get('ci95')} | "
                 f"{c['relative_improvement_pct']}% | {p.get('wins')} / {p.get('ties')} / {p.get('losses')} | {p.get('n')} |")
    L += ["", "## Memory metrics", "", "| Group | Grounded answer accuracy | Unsupported claim rate | Abstention (answerable) | Abstention (unanswerable) | Error breakdown |",
          "|---|---:|---:|---:|---:|---|"]
    for g, m in a["memory_metrics"].items():
        L.append(f"| {g} | {m['grounded_answer_accuracy']} | {m['unsupported_claim_rate']} | {m['abstention_rate_answerable']} | {m['abstention_rate_unanswerable']} | "
                 f"{json.dumps(m['judge_error_breakdown'], ensure_ascii=False)} |")
    L += ["", "## Diagnostics", "", "```json", json.dumps(a["diagnostics"], ensure_ascii=False, indent=1, default=str), "```", "",
          "Reader outcome taxonomy (M1 / M2): READER_EVALUATION_REPORT.json. The combined score (0.4 Style + 0.6 Memory) is a product metric, "
          "not an acceptance criterion on its own."]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- writer breakdown + generated academic / interview reports (from data only)
def writer_breakdown(run: Run, wri: Mapping[str, dict]) -> dict:
    logs = _jsonl(writer_pdir(run) / "memory" / "writer_candidates.jsonl")
    latest = {r["group_key"]: r for r in logs}
    cands = [c for r in latest.values() for c in r["candidates"]]
    rows = [r for r in wri.values() if r.get("status") == "OK"]
    n = lambda k: sum(len(r["judgement"].get(k) or []) for r in rows)   # noqa: E731
    return {"sessions_written": len({r["conversation_id"] for r in latest.values()}), "schema_valid_rate": round(sum(1 for r in latest.values() if r.get("schema_ok")) / len(latest), 4) if latest else None,
            "followup_calls": sum(int(r.get("followup_calls") or 0) for r in latest.values()),
            "candidates": dict(Counter(c["status"] for c in cands)), "third_party_events_auto_verified": sum(1 for c in cands if c["status"] == "AUTO_VERIFIED"
                                                                                                         and c["predicate"] == "event" and str(c["subject"]).startswith("person:")),
            "reject_reasons": dict(Counter(r.split(":")[0][:60] for c in cands if c["status"] != "AUTO_VERIFIED" for r in (c.get("reasons") or ["(none)"])).most_common(12)),
            "judge": {"supported": n("supported"), "unsupported": n("unsupported"), "suspected_missing": n("missing"), "attribution_errors": n("attribution_errors"),
                      "temporal_errors": n("temporal_errors"), "worthiness_errors": n("worthiness_errors"), "units_ok": len(rows), "units_planned": int(run.cfg["writer_audit_n"])},
            "note": "deterministic counts from the writer log + DeepSeek-assisted audit estimates; not human annotation"}


def _fmt(c: Mapping) -> str:
    p = c.get("paired") or {}
    if not p.get("n"):
        return "no paired data"
    ci = p["ci95"]
    verdict = "an established improvement" if ci[0] > 0 else "an established decline" if ci[1] < 0 else "NOT established (the CI includes 0)"
    return (f"{c['baseline']['mean']} -> {c['treatment']['mean']} (paired diff {p['mean_diff']:+} points, 95% CI [{ci[0]}, {ci[1]}], "
            f"relative {c['relative_improvement_pct']}%, W/T/L {p['wins']}/{p['ties']}/{p['losses']}, n={p['n']}): {verdict}")


def writer_md(run: Run, writer: Mapping, wb: Mapping) -> str:
    d = writer["deepseek_writer_score"]
    return "\n".join([f"# Writer audit ({run.cfg.get('eval_mode')}, {wb['sessions_written']} real sessions)", "",
                      f"- Writer quality (DeepSeek 0-100): mean {d.get('mean')}, median {d.get('median')}, n {d.get('n')}",
                      f"- Judge-estimated precision {writer['JUDGE_ESTIMATED_PRECISION']}, recall {writer['JUDGE_ESTIMATED_RECALL']} (estimates, not human annotation)",
                      f"- Schema-valid sessions: {wb['schema_valid_rate']}; follow-up calls: {wb['followup_calls']}",
                      f"- Candidates by status: {json.dumps(wb['candidates'], ensure_ascii=False)}; third-party events auto-verified: {wb['third_party_events_auto_verified']}",
                      f"- Judge error counts: {json.dumps(wb['judge'], ensure_ascii=False)}", f"- Main reject reasons: {json.dumps(wb['reject_reasons'], ensure_ascii=False)}",
                      "", "FAST writer output is an evaluation candidate set only; it is not imported into the production memory."]) + "\n"


def academic_md(run: Run, report: Mapping) -> str:
    a, fast = report["ablation"], run.cfg.get("eval_mode") == "FAST"
    c = a["comparisons"]
    mm = a["memory_metrics"]
    w = report["writer"]
    L = [f"# StillHere {'FAST_EVAL ' if fast else ''}academic results (generated from this run's data)", "",
         f"Mode {run.cfg.get('eval_mode')}: {report.get('sizes_note', '')}memory = {run.cfg.get('memory_mode')}"
         + (" (the complete evidence archive through EF1 + Fact Reader; NO structured semantic memory was built)" if fast else ""), "",
         "## RQ1 Does memory retrieval improve factual recall? (M1 vs M0)", "", _fmt(c["A_memory_contribution_M1_vs_M0"]),
         f"Grounded-answer accuracy {mm['M0']['grounded_answer_accuracy']} -> {mm['M1']['grounded_answer_accuracy']}; unsupported-claim rate "
         f"{mm['M0']['unsupported_claim_rate']} -> {mm['M1']['unsupported_claim_rate']}; abstention on answerable {mm['M0']['abstention_rate_answerable']} -> "
         f"{mm['M1']['abstention_rate_answerable']}. {a['memory_switch_note']}", "",
         "## RQ2 Does the revised persona prompt improve style fidelity? (B2 vs B1)", "", _fmt(c["C_prompt_contribution_B2_vs_B1"]), "",
         "## RQ3 Does the r1 QLoRA adapter improve persona style? (C1 vs B1, C2 vs B2)", "", "C1 vs B1: " + _fmt(c["D1_adapter_style_C1_vs_B1"]),
         "", "C2 vs B2: " + _fmt(c["D2_adapter_style_C2_vs_B2"]), f"Repetition diagnostics: {json.dumps(a['diagnostics']['adapter_unconditional_repetition']['distinct_ratio'], ensure_ascii=False)}", "",
         "## RQ4 Does the adapter preserve factual accuracy with memory? (M2 vs M1)", "", _fmt(c["B_adapter_contribution_M2_vs_M1"]),
         f"Reader-answer fallback rate {json.dumps(a['diagnostics']['adapter_breaks_reader_answer'], ensure_ascii=False)}", "",
         "## RQ5 Can the Writer extract grounded memories from real conversations?", "",
         f"Writer score mean {w['deepseek_writer_score'].get('mean')} (n {w['deepseek_writer_score'].get('n')}); judge-estimated precision {w['JUDGE_ESTIMATED_PRECISION']}, "
         f"recall {w['JUDGE_ESTIMATED_RECALL']}; attribution errors {w['attribution_errors']}, temporal errors {w['temporal_errors']} (machine estimates, not human annotation).", "",
         "## Standard benchmark (kept separate)", "", "LifeBench: on the same 200 development questions StillHere reader B 62.0% vs official Mem0 64.5% "
         "(-2.5 points, 95% CI [-10.0, +5.5]); NOT_DIRECTLY_COMPARABLE (different answer model and judge runs). The real-history QA above is a custom "
         "evaluation and is never compared with LifeBench numbers.", "",
         "Reading rule: a 95% CI that includes 0 is not an established difference. The combined score is a product metric only."]
    return "\n".join(L) + "\n"


def interview_md(run: Run, report: Mapping) -> str:
    c = report["ablation"]["comparisons"]
    fast = run.cfg.get("eval_mode") == "FAST"
    def short(k):
        p = c[k].get("paired") or {}
        return f"{p.get('mean_diff'):+} points (95% CI {p.get('ci95')})" if p.get("n") else "n/a"
    scope = "a fast formal run (100 DEV turns, 100 real-history questions, 20 writer sessions; evidence memory, no structured semantic build)" if fast \
        else "the full run (268 DEV turns, 200 questions, 100 writer sessions, full semantic memory)"
    return "\n".join([f"# Interview summary (generated from {scope})", "",
                      f"**EN.** With the same model and prompt, memory changed historical-QA scores by {short('A_memory_contribution_M1_vs_M0')}; the new prompt changed "
                      f"style by {short('C_prompt_contribution_B2_vs_B1')}; the adapter changed style by {short('D1_adapter_style_C1_vs_B1')} (old prompt) and "
                      f"{short('D2_adapter_style_C2_vs_B2')} (new prompt), and memory scores by {short('B_adapter_contribution_M2_vs_M1')}. A CI that includes 0 is "
                      "reported as not established. LifeBench vs Mem0 is not directly comparable.", "",
                      f"**中文。** 同一模型和 Prompt 下，记忆使历史问答分数变化 {short('A_memory_contribution_M1_vs_M0')}；新 Prompt 使风格分变化 "
                      f"{short('C_prompt_contribution_B2_vs_B1')}；Adapter 使风格分变化 {short('D1_adapter_style_C1_vs_B1')}（旧 Prompt）、"
                      f"{short('D2_adapter_style_C2_vs_B2')}（新 Prompt），使记忆分变化 {short('B_adapter_contribution_M2_vs_M1')}。置信区间包含 0 的差异记为“未确立”。"
                      "LifeBench 与 Mem0 不能直接比较。"]) + "\n"
