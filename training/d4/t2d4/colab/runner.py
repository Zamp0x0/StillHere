"""Resumable generation engine (T2-D3 Mini runner, A100-proven; D4 changes: a batch never mixes models OR sampling seeds, and each row records
its seed and whether it stopped at max_new_tokens). Backend pluggable: ``backend_hf.HFBackend`` on Colab, ``MockBackend`` in CPU tests.

Guarantees:
* resume: before running, the local and the persistent (Drive) outputs are merged (valid JSON, known job ids, first wins); done jobs are skipped;
* idempotent: a job id is written at most once; a finished stage with a valid STAGE_COMPLETE marker is verified, never regenerated;
* memory: batches are packed by a TOKEN budget (prompt + max_new_tokens per row) and a size cap, both identical for every group; on CUDA OOM
  (``torch.cuda.OutOfMemoryError`` OR any RuntimeError whose message says "out of memory") the cache is emptied, the budget is halved for the
  rest of the stage and the batch is retried; a single job that still fails at batch size 1 after ``max_retries`` is recorded as failed
  (CUDA_OOM) and the stage ends FAILED_OOM -- never silently "completed";
* persistence: rows are appended to the local file as soon as a batch ends; the persistent copy is replaced atomically every ``sync_every``
  rows or ``sync_seconds`` seconds and at the end;
* progress: an atomically replaced JSON (status, phase, group, persona, completed / total, %, elapsed, throughput, ETA, VRAM, peak VRAM,
  failed jobs, last saved, heartbeat) after every batch and from a heartbeat thread every 20 s.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from . import io

STATUSES = ("STARTING", "LOADING", "RUNNING", "SCORING", "SYNCING", "COMPLETED", "FAILED_OOM", "FAILED_OTHER", "INTERRUPTED")
EXIT = {"COMPLETED": 0, "FAILED_OTHER": 1, "CONFIG_MISMATCH": 2, "LOCKED": 3, "FAILED_OOM": 4, "MODEL_LOAD_FAILURE": 5, "INTERRUPTED": 130}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_oom(exc: BaseException) -> bool:
    name = type(exc).__name__
    return name == "OutOfMemoryError" or (isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower())


@dataclass
class Settings:
    max_batch_size: int = 16
    max_batch_tokens: int = 24000
    max_new_tokens: int = 128
    max_prompt_tokens: int = 1536
    max_retries: int = 4
    sync_every: int = 64
    sync_seconds: float = 120.0


@dataclass
class Progress:
    path: Path
    stage: str
    gpu: str = "unknown"
    state: Dict = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, **kw):
        with self.lock:
            self.state.update(kw)
            self.state["heartbeat_at"] = now()
            self.state["stage"] = self.stage
            self.state["gpu"] = self.gpu
            io.atomic_write_json(self.path, self.state)


class Heartbeat(threading.Thread):
    def __init__(self, prog: Progress, vram: Callable[[], dict], every: float = 20.0):
        super().__init__(daemon=True)
        self.prog, self.vram, self.every, self.stop = prog, vram, every, threading.Event()

    def run(self):
        while not self.stop.wait(self.every):
            try:
                self.prog.update(**self.vram())
            except Exception:  # noqa: BLE001 - the heartbeat never kills the run
                pass


def pack_batches(jobs: Sequence[dict], lengths: Dict[str, int], s: Settings, budget: int) -> List[List[dict]]:
    """Consecutive jobs of ONE model (base / lora) packed so that n * (longest prompt + max_new_tokens) <= budget and n <= max_batch_size."""
    out, cur, longest = [], [], 0
    for j in jobs:
        L = lengths[j["job_id"]]
        new_longest = max(longest, L)
        if cur and (cur[0]["model"] != j["model"] or cur[0].get("seed") != j.get("seed") or (len(cur) + 1) * (new_longest + s.max_new_tokens) > budget or len(cur) >= s.max_batch_size):
            out.append(cur)
            cur, new_longest = [], L
        cur.append(j)
        longest = new_longest
    if cur:
        out.append(cur)
    return out


def run_stage(stage: str, jobs: Sequence[dict], backend, local_out: Path, persistent_out: Path, progress_path: Path, s: Settings,
              done_marker: Optional[Path] = None, gpu: str = "unknown", log: Callable[[str], None] = print) -> str:
    """Generate every job of ``jobs`` not yet done. Returns the final status (one of STATUSES)."""
    prog = Progress(progress_path, stage, gpu)
    ids = [j["job_id"] for j in jobs]
    if len(set(ids)) != len(ids):
        prog.update(status="FAILED_OTHER", error="duplicate job ids in the jobs file")
        return "FAILED_OTHER"
    prog.update(status="STARTING", phase="merge", total=len(ids), started_at=now())
    merged = io.merge_outputs(local_out, persistent_out, ids)
    done = {r["job_id"] for r in io.read_jsonl(local_out)[0]}
    todo = [j for j in jobs if j["job_id"] not in done]
    log(f"[{stage}] {len(done)}/{len(ids)} already done ({merged}); {len(todo)} to generate")
    if not todo:
        prog.update(status="COMPLETED", completed=len(done), progress_pct=100.0, remaining=0, failed_jobs=0, finished_at=now())
        _mark_done(done_marker, stage, local_out, ids)
        return "COMPLETED"
    prog.update(status="LOADING", phase="tokenize", completed=len(done))
    lengths = {j["job_id"]: backend.prompt_length(j["messages"]) for j in todo}
    over = [k for k, v in lengths.items() if v > s.max_prompt_tokens]
    if over:
        prog.update(status="FAILED_OTHER", error=f"CONFIG_MISMATCH: {len(over)} prompts over {s.max_prompt_tokens} tokens", failed_job_id=over[0])
        return "FAILED_OTHER"
    todo = sorted(todo, key=lambda j: (j["model"], j.get("seed") or 0, lengths[j["job_id"]], j["job_id"]))
    hb = Heartbeat(prog, backend.vram)
    hb.start()
    budget, failed, made = s.max_batch_tokens, [], 0
    t0 = last_sync = time.time()
    since_sync = 0
    status = "RUNNING"
    try:
        queue = pack_batches(todo, lengths, s, budget)
        while queue:
            batch = queue.pop(0)
            tries = 0
            while True:
                try:
                    t1 = time.time()
                    outs = backend.generate(batch, s.max_new_tokens)
                    break
                except Exception as exc:  # noqa: BLE001
                    if not is_oom(exc):
                        raise
                    backend.empty_cache()
                    tries += 1
                    log(f"[{stage}] CUDA OOM on a batch of {len(batch)} (try {tries}); token budget {budget} -> {budget // 2}")
                    if len(batch) == 1:
                        if tries > s.max_retries:
                            failed.append({"job_id": batch[0]["job_id"], "error": "CUDA_OOM", "at": now()})
                            outs = None
                            break
                        continue
                    budget = budget // 2
                    # repack everything still to do with the smaller budget (same rule for every group), then retry the new first batch
                    flat = batch + [j for b in queue for j in b]
                    queue = pack_batches(flat, lengths, s, budget)
                    batch = queue.pop(0)
            if outs is not None:
                stamp = now()
                with Path(local_out).open("a", encoding="utf-8") as f:
                    for j, o in zip(batch, outs):
                        f.write(json.dumps({"job_id": j["job_id"], "group": j["group"], "set": j["set"], "episode_id": j["episode_id"], "persona_id": j["persona_id"],
                                            "model": j["model"], "prompt_sha256": j["prompt_sha256"], "seed": j.get("seed"),
                                            "output": o["text"], "new_tokens": o["new_tokens"], "hit_max": o.get("hit_max"), "raw_special": o.get("raw_special"), "prompt_tokens": lengths[j["job_id"]], "batch_size": len(batch),
                                            "batch_seconds": round(time.time() - t1, 2), "generated_at": stamp}, ensure_ascii=False) + "\n")
                    f.flush()
                made += len(batch)
                since_sync += len(batch)
            el = time.time() - t0
            rate = made / el if el else 0.0
            completed = len(done) + made
            remaining = len(ids) - completed - len(failed)
            if since_sync >= s.sync_every or time.time() - last_sync >= s.sync_seconds:
                prog.update(status="SYNCING")
                io.merge_outputs(local_out, persistent_out, ids)
                last_sync, since_sync = time.time(), 0
                prog.update(last_saved=now())
            prog.update(status="RUNNING", phase="generate", current_group=batch[0]["group"], current_persona=batch[0]["persona_id"], completed=completed,
                        progress_pct=round(100.0 * completed / len(ids), 2), remaining=remaining, elapsed_s=round(el), throughput_per_min=round(rate * 60, 2),
                        eta_s=round(remaining / rate) if rate else None, failed_jobs=len(failed), failed=failed[-20:], token_budget=budget,
                        last_completed_job=batch[-1]["job_id"] if outs is not None else None, **backend.vram())
        status = "COMPLETED" if not failed else "FAILED_OOM"
    except KeyboardInterrupt:
        status = "INTERRUPTED"
    except Exception as exc:  # noqa: BLE001
        status = "FAILED_OOM" if is_oom(exc) else "FAILED_OTHER"
        prog.update(error=f"{type(exc).__name__}: {str(exc)[:500]}")
    finally:
        hb.stop.set()
        prog.update(status="SYNCING")
        final = io.merge_outputs(local_out, persistent_out, ids)
        prog.update(status=status, finished_at=now(), rows=final["rows"], failed_jobs=len(failed), failed=failed, last_saved=now(),
                    completed=final["rows"], progress_pct=round(100.0 * final["rows"] / len(ids), 2), remaining=len(ids) - final["rows"])
    if status == "COMPLETED":
        _mark_done(done_marker, stage, local_out, ids)
    return status


def _mark_done(marker: Optional[Path], stage: str, out: Path, ids: Sequence[str]) -> None:
    if not marker:
        return
    rows = io.read_jsonl(out)[0]
    got = {r["job_id"] for r in rows}
    if got != set(ids):
        raise RuntimeError(f"{stage}: refusing to mark complete, {len(set(ids) - got)} jobs missing")
    io.atomic_write_json(marker, {"stage": stage, "jobs": len(ids), "rows": len(rows), "rows_sha256": io.sha256_rows(rows), "completed_at": now()})


def verify_done(marker: Path, out: Path, ids: Sequence[str]) -> dict:
    """A rerun of a finished stage checks the marker against the rows instead of generating again."""
    m = json.loads(Path(marker).read_text(encoding="utf-8"))
    rows = io.read_jsonl(out)[0]
    ok = {r["job_id"] for r in rows} == set(ids) and io.sha256_rows(rows) == m["rows_sha256"]
    return {"valid": ok, **m}
