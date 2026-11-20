"""The pilot's stages (the Colab notebook calls them in order; each is resumable and records PASS / INCOMPLETE / FAIL / SKIP).

Durable state lives in ``run_dir`` (Google Drive): ``STATE.json``, the result stores (``contexts.jsonl``, ``answers.jsonl``,
``judge.jsonl``), logs, snapshots. Active databases live in ``scratch`` (the Colab local disk): the StillHere run store, the dense index,
Mem0's Qdrant collection + history db. Snapshots are tar files of ``scratch/live`` written to ``run_dir/snapshots`` (sqlite files copied
with the sqlite backup API, the tar written to ``.partial`` and renamed), so a lost runtime restarts from the newest snapshot.

Time rule (official date order): for every date up to the last ask date, ADD that date's events to BOTH StillHere and Mem0, then search
the questions asked on that date in every group and FREEZE the rendered contexts. Answers and judgments are computed later from the
frozen contexts only. A finished search / answer / judge record is never requested again.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional

from . import answer as A
from . import context as CX
from . import judge as J
from . import metrics as MT
from . import protocol as P
from . import select as SEL
from . import timeline as TL
from .records import Store, write_json_atomic

PASS, INCOMPLETE, FAIL, SKIP = "PASS", "INCOMPLETE", "FAIL", "SKIP"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Run:
    def __init__(self, run_dir: Path, scratch: Path, *, log: Callable = print):
        self.run_dir, self.scratch, self.log = Path(run_dir), Path(scratch), log
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "STATE.json"
        self.state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.is_file() else \
            {"protocol_sha256": P.PROTOCOL_SHA256, "created_at": now(), "stages": {}, "added_through": None, "snapshots": []}
        if self.state.get("protocol_sha256") != P.PROTOCOL_SHA256:
            raise RuntimeError("PROTOCOL_CHANGED: this run directory belongs to another protocol; use a new run directory")
        self.contexts = Store(self.run_dir / "contexts.jsonl", ("group", "question_id"))
        self.answers = Store(self.run_dir / "answers.jsonl", ("group", "question_id"))
        self.judgments = Store(self.run_dir / "judge.jsonl", ("group", "question_id", "repeat"))

    # ------------------------------------------------------------------ state
    def save(self) -> None:
        write_json_atomic(self.state_path, self.state)

    def stage(self, name: str, status: str, **info) -> dict:
        self.state["stages"][name] = {"status": status, "at": now(), **info}
        self.save()
        self.log(f"[{status}] {name} {json.dumps(info, ensure_ascii=False)[:300]}")
        return self.state["stages"][name]

    # ------------------------------------------------------------------ inputs
    def load_inputs(self, dataset_path: Path, frozen: Mapping, expected_timeline_sha: Optional[str], *, verify_dataset: bool = True) -> dict:
        data = SEL.load_dataset(dataset_path, verify=verify_dataset)
        again = SEL.sample_questions(data[frozen["person"]["position"]])
        if again["question_ids"] != frozen["question_ids"] or again["qa_sha256"] != frozen["qa_sha256"]:
            self.stage("06_frozen_100qa", FAIL, reason="the sample recomputed from the dataset differs from the frozen manifest")
            raise RuntimeError("FROZEN_SAMPLE_MISMATCH")
        person, qs = SEL.selected(data, frozen)
        last = max(q["ask_time"][:10] for q in qs)
        events = TL.build(person, until=last)
        tsha = TL.sha(events)
        if expected_timeline_sha and tsha != expected_timeline_sha:
            self.stage("07_timeline", FAIL, sha256=tsha, expected=expected_timeline_sha)
            raise RuntimeError("TIMELINE_SHA_MISMATCH")
        self.stage("06_frozen_100qa", PASS, person=frozen["person"]["sample_id"], questions=len(qs), qa_sha256=frozen["qa_sha256"])
        self.stage("07_timeline", PASS, sha256=tsha, **{k: v for k, v in TL.stats(events).items() if k != "speakers"})
        conv = person["conversation"]
        self.inputs = {"person": person, "qs": qs, "events": events, "last": last, "name": conv["speaker_a"], "assistant": conv["speaker_b"],
                       "slug": frozen["person"]["slug"]}
        return self.inputs

    # ------------------------------------------------------------------ snapshots
    def live(self) -> Path:
        return self.scratch / "live"

    def snapshot(self, date: str) -> dict:
        snap_dir = self.run_dir / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        stage_dir = self.scratch / "_snap"
        shutil.rmtree(stage_dir, ignore_errors=True)
        for src in self.live().rglob("*"):
            if src.is_dir():
                continue
            rel = src.relative_to(self.live())
            dst = stage_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.suffix in (".sqlite", ".db") or src.name == "storage.sqlite":
                s, d = sqlite3.connect(str(src)), sqlite3.connect(str(dst))
                with d:
                    s.backup(d)
                s.close()
                d.close()
            elif src.name.endswith(("-wal", "-shm", "-journal", ".lock")):
                continue                                  # WAL / journal are folded in by the backup API; a lock file is not data
            else:
                shutil.copy2(src, dst)
        (stage_dir / "SNAPSHOT_DATE").write_text(date, encoding="utf-8")
        final = snap_dir / f"live_{date}.tar"
        tmp = snap_dir / f"live_{date}.tar.partial"
        with tarfile.open(tmp, "w") as t:
            t.add(stage_dir, arcname="live")
        os.replace(tmp, final)
        shutil.rmtree(stage_dir, ignore_errors=True)
        info = {"date": date, "file": final.name, "bytes": final.stat().st_size, "at": now()}
        self.state["snapshots"] = [s for s in self.state["snapshots"] if s["date"] != date] + [info]
        for old in sorted(self.state["snapshots"], key=lambda s: s["date"])[:-3]:              # keep the newest three
            (snap_dir / old["file"]).unlink(missing_ok=True)
        self.state["snapshots"] = sorted(self.state["snapshots"], key=lambda s: s["date"])[-3:]
        self.save()
        return info

    def restore_if_needed(self) -> Optional[str]:
        """The live databases must be at ``added_through``. Returns the date the live state is at (None: empty)."""
        marker = self.live() / "LIVE_DATE"
        live_at = marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
        want = self.state.get("added_through")
        dirty = (self.live() / "DAY_IN_PROGRESS").is_file()          # stopped in the middle of a date's ADD: never continue on top of it
        if live_at == want and not dirty:
            return live_at
        snaps = sorted(self.state.get("snapshots") or [], key=lambda s: s["date"])
        shutil.rmtree(self.live(), ignore_errors=True)
        if not snaps:
            self.state["added_through"] = None
            self.save()
            return None
        s = snaps[-1]
        with tarfile.open(self.run_dir / "snapshots" / s["file"]) as t:
            t.extractall(self.scratch, filter="data")
        (self.live() / "LIVE_DATE").write_text(s["date"], encoding="utf-8")
        self.state["added_through"] = s["date"]
        self.save()
        self.log(f"restored live state from snapshot {s['file']}")
        return s["date"]

    # ------------------------------------------------------------------ chronological ADD + SEARCH
    def chronological(self, *, stillhere, mem0, dense, count: Callable[[str], int], snapshot_every: int = 20) -> dict:
        """ADD date by date into both systems; freeze the four contexts of every question on its ask date."""
        inp = self.inputs
        days = TL.by_date(inp["events"])
        q_by_date: Dict[str, List[dict]] = {}
        for q in inp["qs"]:
            q_by_date.setdefault(q["ask_time"][:10], []).append(q)
        dates = sorted(set(days) | set(q_by_date))
        dates = [d for d in dates if d <= inp["last"]]
        start = self.state.get("added_through")
        timing = self.state.setdefault("timing", {"stillhere_add_s": 0.0, "mem0_add_s": 0.0, "search_s": 0.0})
        since = 0
        from .stillhere_memory import ask_at
        for d in dates:
            if start and d <= start:
                continue
            self.live().mkdir(parents=True, exist_ok=True)
            (self.live() / "DAY_IN_PROGRESS").write_text(d, encoding="utf-8")
            t0 = time.perf_counter()
            stillhere.add_day(days.get(d, []))
            t1 = time.perf_counter()
            r = mem0.add_day(days.get(d, []))
            t2 = time.perf_counter()
            timing["stillhere_add_s"] += t1 - t0
            timing["mem0_add_s"] += t2 - t1
            self.state.setdefault("mem0_add", {"batches": 0, "failed": 0, "memories_returned": 0})
            self.state["mem0_add"]["batches"] += r.get("batches", 0)
            self.state["mem0_add"]["failed"] += r.get("failed", 0)
            self.state["mem0_add"]["memories_returned"] += r.get("memories", 0)
            for q in q_by_date.get(d, []):
                self.search_question(q, stillhere=stillhere, mem0=mem0, dense=dense, count=count, at=ask_at(q["ask_time"]))
            timing["search_s"] += time.perf_counter() - t2
            (self.live() / "LIVE_DATE").write_text(d, encoding="utf-8")
            self.state["added_through"] = d
            self.save()
            (self.live() / "DAY_IN_PROGRESS").unlink(missing_ok=True)
            since += 1
            if q_by_date.get(d) or since >= snapshot_every:
                self.snapshot(d)
                since = 0
        done = self.contexts.done()
        need = [(g, q["question_id"]) for g in P.GROUP_ORDER for q in inp["qs"]]
        missing = [k for k in need if k not in done]
        st = PASS if not missing and not self.state["mem0_add"]["failed"] else INCOMPLETE
        return self.stage("11_16_chronological_add_search", st, added_through=self.state["added_through"], contexts=len(need) - len(missing),
                          missing=len(missing), mem0_failed_batches=self.state["mem0_add"]["failed"], timing={k: round(v, 1) for k, v in timing.items()})

    def search_question(self, q: Mapping, *, stillhere, mem0, dense, count, at) -> None:
        done = self.contexts.done()
        query = TL.search_query(q["question"])
        for g in P.GROUP_ORDER:
            if (g, q["question_id"]) in done:
                continue
            row = {"group": g, "question_id": q["question_id"], "ask_time": q["ask_time"], "query": query, "added_through": q["ask_time"][:10]}
            err = None
            for attempt in range(1, 4):
                try:
                    if g == "M0_NO_MEMORY":
                        res = {"entries": [], "meta": {"retrieval_ms": 0.0}}
                    elif g == "M1_STILLHERE_V2L":
                        res = stillhere.search(query, at, dense=None)
                    elif g == "M2_STILLHERE_V2D":
                        res = stillhere.search(query, at, dense=dense)
                    else:
                        res = mem0.search(query)
                        late = [e.date for e in res["entries"] if e.date and e.date > q["ask_time"][:10]]
                        if late:
                            raise AssertionError(f"mem0 returned memories dated after the ask date: {late[:3]}")
                    ctx = CX.render(res["entries"], budget=int(P.EVIDENCE["token_budget"]), count=count)
                    row.update(status="OK", attempts=attempt, **ctx, retrieval_ms=res["meta"].get("retrieval_ms"), meta=res["meta"],
                               context_sha256=hashlib.sha256(ctx["text"].encode("utf-8")).hexdigest(),
                               future_messages_in_store=stillhere.future_messages(at) if g in ("M1_STILLHERE_V2L", "M2_STILLHERE_V2D") else None)
                    err = None
                    break
                except Exception as e:  # noqa: BLE001 - recorded; a search cannot be redone later (the store moves on)
                    err = f"{type(e).__name__}: {str(e)[:300]}"
                    time.sleep(min(10, 2 ** attempt) if attempt < 3 else 0)
            if err:
                row.update(status="ERROR", error=err)
            self.contexts.append(row)

    # ------------------------------------------------------------------ answers / judge
    def answer_items(self) -> List[dict]:
        ctx = self.contexts.done()
        out = []
        for q in self.inputs["qs"]:
            for g in P.GROUP_ORDER:
                c = ctx.get((g, q["question_id"]))
                if c is None:
                    continue
                out.append({"group": g, "question_id": q["question_id"], "question": q["question"], "ask_time": q["ask_time"], "context": c["text"],
                            "context_sha256": c["context_sha256"]})
        return out

    def run_answers(self, chat: Callable) -> dict:
        r = A.run(self.answer_items(), chat=chat, store=self.answers, name=self.inputs["name"], log=self.log)
        need = 100 * len(P.GROUP_ORDER)
        per = {g: sum(1 for (gg, _q) in self.answers.done() if gg == g) for g in P.GROUP_ORDER}
        return self.stage("13_16_answers", PASS if all(v == P.COMPLETENESS["answers_per_group"] for v in per.values()) else INCOMPLETE,
                          per_group=per, requested=r["requested"], expected=need)

    def calibrate(self, official: J.Official, chat: Callable) -> dict:
        res = []
        for c in J.calibration_items():
            r = chat(official.messages(c["qa"], c["answer"]))
            v = official.parse(r.text)
            res.append({"id": c["id"], "expect": c["expect"], "got": None if v is None else v["is_correct"], "ok": v is not None and v["is_correct"] == c["expect"],
                        "raw": r.text, **r.usage()})
        write_json_atomic(self.run_dir / "JUDGE_CALIBRATION.json", res)
        ok = sum(x["ok"] for x in res)
        return self.stage("17_judge_calibration", PASS if ok == len(res) else INCOMPLETE, passed=ok, of=len(res),
                          note="recorded, not a gate on the formal judging (a failed anchor is reported as a limitation)")

    def judge_jobs(self) -> List[dict]:
        ans = self.answers.done()
        qa = {q["question_id"]: q for q in self.inputs["qs"]}
        return [{"group": g, "question_id": qid, "repeat": r, "qa": qa[qid], "answer": a["answer"]}
                for (g, qid), a in sorted(ans.items()) for r in range(int(P.JUDGE["repeats"]))]

    def run_judge(self, official: J.Official, chat: Callable) -> dict:
        if official.prompt_sha256 != J.OFFICIAL_PROMPT_SHA256:
            self.stage("18_judge", FAIL, reason="the official judge prompt differs from the pinned one", got=official.prompt_sha256)
            raise RuntimeError("JUDGE_PROMPT_MISMATCH")
        J.run(self.judge_jobs(), official=official, chat=chat, store=self.judgments, log=self.log)
        per = {g: sum(1 for (gg, _q, _r) in self.judgments.done() if gg == g) for g in P.GROUP_ORDER}
        return self.stage("18_judge", PASS if all(v == P.COMPLETENESS["judge_records_per_group"] for v in per.values()) else INCOMPLETE,
                          per_group=per, prompt_sha256=official.prompt_sha256)

    # ------------------------------------------------------------------ metrics
    def compute_metrics(self, official: J.Official, extra: Optional[Mapping] = None) -> dict:
        out = compute(self.inputs["qs"], self.inputs["events"], self.contexts.done(), self.answers.done(), self.judgments.done(), official)
        out["run"] = {"stages": {k: v["status"] for k, v in self.state["stages"].items()}, "mem0_add": self.state.get("mem0_add"),
                      "timing_s": self.state.get("timing"), **(extra or {})}
        write_json_atomic(self.run_dir / "METRICS.json", out)
        return self.stage("19_metrics", PASS if out["formal"] else INCOMPLETE, formal=out["formal"],
                          micro={g: m["micro_accuracy"] for g, m in out["groups"].items()})


def compute(qs, events, contexts: Mapping, answers: Mapping, judgments: Mapping, official) -> dict:
    """Pure: every number in METRICS.json from the stored records (``scripts`` can rerun this on a downloaded Results ZIP)."""
    gold_texts: Dict[str, List[dict]] = {}
    for e in events:
        gold_texts.setdefault(e["gold_id"], []).append(e)
    rows: Dict[str, List[dict]] = {}
    complete: Dict[str, dict] = {}
    for g in P.GROUP_ORDER:
        rs = []
        for q in qs:
            recs = [judgments[(g, q["question_id"], r)] for r in range(int(P.JUDGE["repeats"])) if (g, q["question_id"], r) in judgments]
            verdict = J.aggregate(recs, q, official) if len(recs) == int(P.JUDGE["repeats"]) else None
            ask = q["ask_time"][:10]
            gt = {k: [e["content"] for e in v if e["date"] <= ask] for k, v in gold_texts.items() if k in (q.get("evidence") or [])}
            rs.append(MT.per_question(q, answers.get((g, q["question_id"])), verdict, contexts.get((g, q["question_id"])), gt, g))
        rows[g] = rs
        complete[g] = {"contexts": sum(1 for q in qs if (g, q["question_id"]) in contexts), "answers": sum(1 for q in qs if (g, q["question_id"]) in answers),
                       "judge_records": sum(1 for q in qs for r in range(int(P.JUDGE["repeats"])) if (g, q["question_id"], r) in judgments)}
        complete[g]["complete"] = (complete[g]["contexts"] == len(qs) and complete[g]["answers"] == P.COMPLETENESS["answers_per_group"]
                                   and complete[g]["judge_records"] == P.COMPLETENESS["judge_records_per_group"])
    groups = {g: {**MT.group_metrics(rows[g]), "completeness": complete[g]} for g in P.GROUP_ORDER}
    pairs = {}
    for a, b in P.PAIRS:
        p = MT.paired(rows[a], rows[b])
        p["decidable"] = complete[a]["complete"] and complete[b]["complete"]
        pairs[f"{a}__vs__{b}"] = p
    return {"schema": "stillhere.lifebench_pilot.metrics/1.0", "protocol_sha256": P.PROTOCOL_SHA256, "scope": P.SCOPE,
            "formal": all(c["complete"] for c in complete.values()), "groups": groups, "pairs": pairs, "primary_pair": "__vs__".join(P.PRIMARY_PAIR),
            "per_question": rows}
