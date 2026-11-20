"""P2 evaluation stages (the Colab notebook calls them in order; resumable; PASS / INCOMPLETE / FAIL / SKIP).

Two experiments, both through the same day-by-day rule (a date's events are added, then that date's questions are searched):
    NEWP   the fresh person: OLD_V2D and NEW_P2_V2D share ONE StillHere store and ONE dense index; contexts are frozen per group
    KNOWN  于晓薇 pilot 100 QA: NEW_P2_V2D only; compared with the STORED pilot M2_STILLHERE_V2D records
Durable records live in ``run_dir`` (Drive); stores in ``scratch`` (/content) with SQLite-backup snapshots in ``run_dir/snapshots``.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional

from lbpilot import answer as A
from lbpilot import context as CX
from lbpilot import judge as J
from lbpilot import timeline as TL
from lbpilot.records import Store, write_json_atomic
from lbpilot.stillhere_memory import ask_at

from . import answer_p2 as A2
from . import protocol as P
from . import stillhere_p2 as SP

PASS, INCOMPLETE, FAIL, SKIP = "PASS", "INCOMPLETE", "FAIL", "SKIP"
EXPERIMENTS = {"NEWP": ("OLD_V2D", "NEW_P2_V2D"), "KNOWN": ("NEW_P2_V2D",)}
PRICE = {"input_miss": 0.30, "input_hit": 0.006, "output": 1.20}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def usd(r) -> float:
    inp, hit, out = r.get("input_tokens") or 0, r.get("cache_hit_tokens") or 0, r.get("output_tokens") or 0
    return (inp - hit) / 1e6 * PRICE["input_miss"] + hit / 1e6 * PRICE["input_hit"] + out / 1e6 * PRICE["output"]


class BudgetExceeded(RuntimeError):
    pass


class Run:
    def __init__(self, run_dir: Path, scratch: Path, *, log: Callable = print):
        self.run_dir, self.scratch, self.log = Path(run_dir), Path(scratch), log
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "STATE.json"
        self.state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.is_file() else \
            {"protocol_sha256": P.P2_PROTOCOL_SHA256, "created_at": now(), "stages": {}, "added_through": {}, "snapshots": {}}
        if self.state.get("protocol_sha256") != P.P2_PROTOCOL_SHA256:
            raise RuntimeError("PROTOCOL_CHANGED: this run directory belongs to another protocol; use a new run directory")
        self.contexts = Store(self.run_dir / "contexts.jsonl", ("experiment", "group", "question_id"))
        self.answers = Store(self.run_dir / "answers.jsonl", ("experiment", "group", "question_id"))
        self.judgments = Store(self.run_dir / "judge.jsonl", ("experiment", "group", "question_id", "repeat"))
        self.inputs: Dict[str, dict] = {}

    def save(self):
        write_json_atomic(self.state_path, self.state)

    def stage(self, name, status, **info):
        self.state["stages"][name] = {"status": status, "at": now(), **info}
        self.save()
        self.log(f"[{status}] {name} {json.dumps(info, ensure_ascii=False)[:300]}")
        return self.state["stages"][name]

    # ------------------------------------------------------------------ cost guard
    def spent(self) -> float:
        rows = [r for r in self.answers.rows() if r.get("status") == "OK"] + [r for r in self.judgments.rows() if r.get("status") == "OK"]
        return round(sum(usd(r) for r in rows), 4)

    def guard(self):
        s = self.spent()
        if s > P.COST_BUDGET_USD:
            raise BudgetExceeded(f"spent USD {s} > budget {P.COST_BUDGET_USD} (peak-price estimate): stopped; raise COST_BUDGET_USD in the protocol only deliberately")

    # ------------------------------------------------------------------ inputs
    def set_inputs(self, experiment: str, person: Mapping, qs: List[dict], slug: str):
        last = max(q["ask_time"][:10] for q in qs)
        events = TL.build(person, until=last)
        conv = person["conversation"]
        self.inputs[experiment] = {"person": person, "qs": qs, "events": events, "last": last, "slug": slug, "name": conv["speaker_a"], "assistant": conv["speaker_b"]}
        return self.inputs[experiment]

    # ------------------------------------------------------------------ snapshots (one live dir per experiment)
    def live(self, experiment: str) -> Path:
        return self.scratch / f"live_{experiment}"

    def snapshot(self, experiment: str, date: str):
        sd = self.run_dir / "snapshots"
        sd.mkdir(parents=True, exist_ok=True)
        stage = self.scratch / f"_snap_{experiment}"
        shutil.rmtree(stage, ignore_errors=True)
        for src in self.live(experiment).rglob("*"):
            if src.is_dir() or src.name.endswith(("-wal", "-shm", "-journal", ".lock")):
                continue
            dst = stage / src.relative_to(self.live(experiment))
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.suffix in (".sqlite", ".db"):
                s, d = sqlite3.connect(str(src)), sqlite3.connect(str(dst))
                with d:
                    s.backup(d)
                s.close()
                d.close()
            else:
                shutil.copy2(src, dst)
        final, tmp = sd / f"{experiment}_{date}.tar", sd / f"{experiment}_{date}.tar.partial"
        with tarfile.open(tmp, "w") as t:
            t.add(stage, arcname=f"live_{experiment}")
        os.replace(tmp, final)
        shutil.rmtree(stage, ignore_errors=True)
        snaps = [s for s in self.state["snapshots"].get(experiment, []) if s["date"] != date] + [{"date": date, "file": final.name}]
        snaps = sorted(snaps, key=lambda s: s["date"])
        for old in snaps[:-2]:
            (sd / old["file"]).unlink(missing_ok=True)
        self.state["snapshots"][experiment] = snaps[-2:]
        self.save()

    def restore_if_needed(self, experiment: str) -> Optional[str]:
        live = self.live(experiment)
        marker = live / "LIVE_DATE"
        live_at = marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
        want = self.state["added_through"].get(experiment)
        if live_at == want and not (live / "DAY_IN_PROGRESS").is_file():
            return live_at
        shutil.rmtree(live, ignore_errors=True)
        snaps = self.state["snapshots"].get(experiment) or []
        if not snaps:
            self.state["added_through"][experiment] = None
            self.save()
            return None
        s = snaps[-1]
        with tarfile.open(self.run_dir / "snapshots" / s["file"]) as t:
            t.extractall(self.scratch, filter="data")
        (live / "LIVE_DATE").write_text(s["date"], encoding="utf-8")
        self.state["added_through"][experiment] = s["date"]
        self.save()
        return s["date"]

    # ------------------------------------------------------------------ chronological ADD + SEARCH (StillHere only)
    def chronological(self, experiment: str, *, stillhere, dense, count, snapshot_every: int = 30) -> dict:
        inp = self.inputs[experiment]
        days = TL.by_date(inp["events"])
        qd: Dict[str, List[dict]] = {}
        for q in inp["qs"]:
            qd.setdefault(q["ask_time"][:10], []).append(q)
        start = self.state["added_through"].get(experiment)
        since = 0
        for d in sorted(set(days) | set(qd)):
            if d > inp["last"] or (start and d <= start):
                continue
            live = self.live(experiment)
            live.mkdir(parents=True, exist_ok=True)
            (live / "DAY_IN_PROGRESS").write_text(d, encoding="utf-8")
            stillhere.add_day(days.get(d, []))
            for q in qd.get(d, []):
                self.search_question(experiment, q, stillhere=stillhere, dense=dense, count=count)
            (live / "LIVE_DATE").write_text(d, encoding="utf-8")
            self.state["added_through"][experiment] = d
            self.save()
            (live / "DAY_IN_PROGRESS").unlink(missing_ok=True)
            since += 1
            if qd.get(d) or since >= snapshot_every:
                self.snapshot(experiment, d)
                since = 0
        need = [(experiment, g, q["question_id"]) for g in EXPERIMENTS[experiment] for q in inp["qs"]]
        done = self.contexts.done()
        miss = [k for k in need if k not in done]
        return self.stage(f"chronological_{experiment}", PASS if not miss else INCOMPLETE, contexts=len(need) - len(miss), missing=len(miss),
                          added_through=self.state["added_through"].get(experiment))

    def search_question(self, experiment, q, *, stillhere, dense, count):
        done = self.contexts.done()
        query = TL.search_query(q["question"])
        at = ask_at(q["ask_time"])
        for g in EXPERIMENTS[experiment]:
            key = (experiment, g, q["question_id"])
            if key in done:
                continue
            row = {"experiment": experiment, "group": g, "question_id": q["question_id"], "ask_time": q["ask_time"], "query": query}
            err = None
            for attempt in range(1, 4):
                try:
                    if g == "OLD_V2D":
                        res = stillhere.search(query, at, dense=dense)
                        ctx = CX.render(res["entries"], budget=int(P.EVIDENCE["token_budget"]), count=count)
                    else:
                        res = SP.search(stillhere, query, at, dense=dense)
                        ctx = SP.render(res, count)
                    row.update(status="OK", attempts=attempt, **ctx, retrieval_ms=res["meta"].get("retrieval_ms"), meta=res["meta"],
                               context_sha256=hashlib.sha256(ctx["text"].encode("utf-8")).hexdigest(), future_messages_in_store=stillhere.future_messages(at))
                    err = None
                    break
                except Exception as e:  # noqa: BLE001 - recorded; a search cannot be redone later (the store moves on)
                    err = f"{type(e).__name__}: {str(e)[:300]}"
                    time.sleep(min(5, attempt))
            if err:
                row.update(status="ERROR", error=err)
            self.contexts.append(row)

    # ------------------------------------------------------------------ answers / judge
    def run_answers(self, chat: Callable, *, concurrency: int = 4) -> dict:
        ctx = self.contexts.done()
        items = []
        for exp, inp in self.inputs.items():
            for q in inp["qs"]:
                for g in EXPERIMENTS[exp]:
                    c = ctx.get((exp, g, q["question_id"]))
                    if c:
                        items.append((exp, g, q, c, inp["name"]))

        def one(it):
            exp, g, q, c, name = it
            msgs = (A2.messages if g == "NEW_P2_V2D" else A.messages)(name, q["question"], q["ask_time"], c["text"])
            base = {"experiment": exp, "group": g, "question_id": q["question_id"], "context_sha256": c["context_sha256"],
                    "prompt_version": A2.PROMPT_VERSION if g == "NEW_P2_V2D" else A.PROMPT_VERSION,
                    "prompt_sha256": hashlib.sha256(msgs[0]["content"].encode("utf-8")).hexdigest()}
            try:
                r = chat(msgs)
                return {**base, "status": "OK" if (r.text or "").strip() else "EMPTY", "answer": (r.text or "").strip(), "abstained": A.is_abstention(r.text), **r.usage()}
            except Exception as e:  # noqa: BLE001
                return {**base, "status": "ERROR", "error": f"{type(e).__name__}: {str(e)[:200]}"}
        for rnd in range(2):
            done = self.answers.done()
            todo = [it for it in items if (it[0], it[1], it[2]["question_id"]) not in done]
            if not todo:
                break
            self.guard()
            self.log(f"answer round {rnd + 1}: {len(todo)} requests")
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                for f in as_completed([ex.submit(one, it) for it in todo]):
                    self.answers.append(f.result())
        per = {f"{e}:{g}": sum(1 for k in self.answers.done() if k[0] == e and k[1] == g) for e in self.inputs for g in EXPERIMENTS[e]}
        return self.stage("answers", PASS if all(v == 100 for v in per.values()) else INCOMPLETE, per_group=per, spent_usd=self.spent())

    def run_judge(self, official: J.Official, chat: Callable, *, concurrency: int = 6) -> dict:
        if official.prompt_sha256 != J.OFFICIAL_PROMPT_SHA256:
            self.stage("judge", FAIL, reason="official judge prompt changed")
            raise RuntimeError("JUDGE_PROMPT_MISMATCH")
        qa = {(e, q["question_id"]): q for e, inp in self.inputs.items() for q in inp["qs"]}
        jobs = [(k, a, r) for k, a in self.answers.done().items() for r in range(3)]

        def one(job):
            (e, g, qid), a, rep = job
            base = {"experiment": e, "group": g, "question_id": qid, "repeat": rep, "judge_prompt_sha256": official.prompt_sha256}
            last = None
            for attempt in range(1, 4):
                try:
                    r = chat(official.messages(qa[(e, qid)], a["answer"]))
                except Exception as ex:  # noqa: BLE001
                    last = {**base, "status": "ERROR", "error": f"{type(ex).__name__}: {str(ex)[:200]}"}
                    continue
                v = official.parse(r.text)
                if v is not None:
                    return {**base, "status": "OK", "parse_attempts": attempt, "raw": r.text, **v, **r.usage()}
                last = {**base, "status": "PARSE_FAIL", "raw": r.text, **r.usage()}
            return last
        for rnd in range(2):
            done = self.judgments.done()
            todo = [j for j in jobs if (j[0][0], j[0][1], j[0][2], j[2]) not in done]
            if not todo:
                break
            self.guard()
            self.log(f"judge round {rnd + 1}: {len(todo)} requests")
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                for f in as_completed([ex.submit(one, j) for j in todo]):
                    self.judgments.append(f.result())
        per = {f"{e}:{g}": sum(1 for k in self.judgments.done() if k[0] == e and k[1] == g) for e in self.inputs for g in EXPERIMENTS[e]}
        return self.stage("judge", PASS if all(v == 300 for v in per.values()) else INCOMPLETE, per_group=per, spent_usd=self.spent())
