"""T2-D4 live training progress (ported from the T2-D3 Mini module, which ran on A100; only titles changed): structured events, a pure progress / ETA model, a fixed Colab panel, live process streaming, and the
reconnect monitor. Standard library only (IPython is used when present).

Events (one JSON line each, written by the training callbacks in t2d4/colab/train_d4.py): ``phase`` (STARTING, LOADING, READY, TRAINING, VALIDATING,
STYLE_PROBE, SAVING, FINAL_VALIDATION, FINISHED, EARLY_STOPPED, FAILED), ``train_begin`` (segment start: GPU, max_steps, epochs, eval / save intervals, resumed
step), ``step`` (one optimizer step: global_step, epoch, step_seconds measured between on_step_begin and on_step_end, i.e. without
validation or save, non-padding tokens), ``log`` (loss, learning_rate, grad_norm), ``eval`` (one validation set loss), ``selection`` (mean
validation loss, probe metrics, best tracks, early-stopping counters), ``save`` (checkpoint, seconds), ``train_end``, ``error``.

Events go to a local file every time (fast, /content) and are mirrored to Google Drive at most every ``drive_every`` seconds and at every
validation, save and the end, so a new runtime can rebuild the progress of earlier segments. The panel shows only what events say: it never
advances a step on its own, and when no event has arrived for a while it says so (with the last event time) instead of guessing.

ETA: rolling mean of the last ``window`` valid optimizer steps (the first ``skip_first`` steps of each segment are excluded as warm-up);
CALIBRATING until ``min_steps`` valid steps exist; recalibration when the GPU changes (new segment) or when the last 10 steps differ from
the window by more than ``shift``. Training ETA = remaining steps x step time. Overall ETA adds the remaining validation points and saves
at their MEASURED mean duration (shown as "overhead not measured yet" until one of each has happened). Both are predictions, not promises,
and assume no early stop.
"""
from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

PHASES = ("STARTING", "LOADING", "READY", "TRAINING", "VALIDATING", "STYLE_PROBE", "SAVING", "FINAL_VALIDATION", "FINISHED", "EARLY_STOPPED", "FAILED")
WINDOW, SKIP_FIRST, MIN_STEPS, SHIFT, STALE_SECONDS = 30, 2, 10, 0.35, 180


def gpu_class(name: str | None, vram_gb: float | None) -> str:
    n = (name or "").upper()
    if not name or n == "CPU":
        return "CPU"
    if "L4" in n:
        return "L4_24GB"
    if "A100" in n:
        return "A100_80GB" if (vram_gb or 0) > 60 else "A100_40GB"
    return f"OTHER ({name}, {vram_gb} GB)"


# ---------------------------------------------------------------- writing (training side)

class EventWriter:
    def __init__(self, local_dir: Path, drive_dir: Path | None = None, drive_every: float = 60.0):
        self.local = Path(local_dir)
        self.local.mkdir(parents=True, exist_ok=True)
        self.drive = Path(drive_dir) if drive_dir else None
        self.drive_every, self._last_drive = drive_every, 0.0
        self.path = self.local / "events.jsonl"
        self._lock = threading.Lock()
        if self.drive and not self.path.exists() and (self.drive / "events.jsonl").exists():
            shutil.copy2(self.drive / "events.jsonl", self.path)  # new runtime: continue the persisted history

    def emit(self, event: str, **fields):
        row = {"t": time.time(), "event": event, **fields}
        try:
            with self._lock, self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
                f.flush()
            milestone = event in ("selection", "save", "train_end", "error") or (event == "phase" and fields.get("phase") in ("TRAINING", "FINISHED", "EARLY_STOPPED", "FAILED"))
            if milestone or time.time() - self._last_drive > self.drive_every:
                self.persist()
        except OSError:
            pass  # progress must never break training
        return row

    def persist(self):
        if not self.drive:
            return
        try:
            self.drive.mkdir(parents=True, exist_ok=True)
            tmp = self.drive / "events.jsonl.partial"
            shutil.copy2(self.path, tmp)
            os.replace(tmp, self.drive / "events.jsonl")
            state = ProgressModel.from_file(self.path).state(time.time())
            (self.drive / "progress_state.json").write_text(json.dumps(state, indent=1, default=str) + "\n", encoding="utf-8")
            self._last_drive = time.time()
        except OSError:
            pass


# ---------------------------------------------------------------- model (pure)

class ProgressModel:
    def __init__(self):
        self.events = []
        self.phase, self.phase_t = "STARTING", None
        self.gpu = {}
        self.max_steps = self.epochs = self.eval_steps = self.save_steps = None
        self.step = 0
        self.epoch = 0.0
        self.train_loss = self.lr = None
        self.val = {}
        self.val_mean = None
        self.segments = []          # [{"t0", "gpu", "begin_step", "durations": [(step, sec, tokens)]}]
        self.window = deque(maxlen=WINDOW)
        self.recalibrations = []
        self.eval_seconds, self.save_seconds, self.probe_seconds = [], [], []
        self.latest_ckpt = None
        self.best_loss = self.best_style = None
        self.early = {}
        self.stopped = None
        self.final = None
        self.error = None
        self.last_t = None
        self._eval_t0 = self._save_t0 = None
        self.train_seconds = 0.0
        self.tokens_seen = None

    @classmethod
    def from_events(cls, rows):
        m = cls()
        for r in rows:
            m.ingest(r)
        return m

    @classmethod
    def from_file(cls, path: Path):
        rows = []
        if Path(path).exists():
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue  # a half-written last line
        return cls.from_events(rows)

    def ingest(self, r: dict):
        self.events.append(r)
        t, ev = r.get("t"), r.get("event")
        self.last_t = t
        if ev == "phase":
            self.phase, self.phase_t = r["phase"], t
            if r["phase"] == "VALIDATING" and self._eval_t0 is None:
                self._eval_t0 = t
            if r["phase"] == "SAVING":
                self._save_t0 = t
        elif ev == "train_begin":
            self.gpu = r.get("gpu_info") or {}
            for k in ("max_steps", "epochs", "eval_steps", "save_steps"):
                if r.get(k) is not None:
                    setattr(self, k, r[k])
            prev = self.segments[-1]["gpu"] if self.segments else None
            self.segments.append({"t0": t, "gpu": self.gpu.get("gpu"), "begin_step": r.get("global_step", 0), "durations": []})
            self.window.clear()  # new segment (resume, maybe another GPU): recalibrate
            if prev is not None:
                self.recalibrations.append({"at_step": r.get("global_step"), "why": "resume on " + str(self.gpu.get("gpu")) + (" (GPU changed)" if prev != self.gpu.get("gpu") else "")})
            self.step = r.get("global_step", self.step)
            self.phase, self.phase_t = "TRAINING", t
        elif ev == "step":
            self.step, self.epoch = r["global_step"], r.get("epoch", self.epoch)
            seg = self.segments[-1] if self.segments else None
            if seg is not None:
                seg["durations"].append((r["global_step"], r["step_seconds"], r.get("tokens")))
                if len(seg["durations"]) > SKIP_FIRST:
                    self.window.append((r["step_seconds"], r.get("tokens")))
                    self._maybe_recalibrate(r["global_step"])
            self.train_seconds += r["step_seconds"]
            if r.get("tokens_seen") is not None:
                self.tokens_seen = r["tokens_seen"]
            if r.get("gpu_mem"):
                self.gpu.update(r["gpu_mem"])
            if self.phase not in ("VALIDATING", "STYLE_PROBE", "SAVING"):
                self.phase = "TRAINING"
        elif ev == "log":
            if r.get("loss") is not None:
                self.train_loss = r["loss"]
            if r.get("learning_rate") is not None:
                self.lr = r["learning_rate"]
            if r.get("gpu_mem"):
                self.gpu.update(r["gpu_mem"])
        elif ev == "eval":
            if self.phase != "FINAL_VALIDATION":  # the post-training record of the best model is not the latest validation point
                self.val[r["name"]] = r["loss"]
        elif ev == "selection":
            self.val_mean = r.get("mean_val_loss")
            self.best_loss, self.best_style = r.get("best_loss"), r.get("best_style")
            self.early = r.get("early_stopping") or {}
            if r.get("probe_seconds") is not None:
                self.probe_seconds.append(r["probe_seconds"])
            if self._eval_t0 is not None:
                self.eval_seconds.append(t - self._eval_t0)
                self._eval_t0 = None
            if r.get("early_stop"):
                self.stopped = {"step": r["global_step"], "reason": r["early_stop"]}
        elif ev == "save":
            self.latest_ckpt = r.get("checkpoint")
            if self._save_t0 is not None:
                self.save_seconds.append(t - self._save_t0)
                self._save_t0 = None
            self.phase = "TRAINING"
        elif ev == "train_end":
            self.final = r.get("final_selection")
            self.stopped = r.get("stopped") or self.stopped
            self.phase = "EARLY_STOPPED" if self.stopped else "FINISHED"
        elif ev == "error":
            self.error = {"type": r.get("type"), "message": r.get("message")}
            self.phase = "FAILED"

    def _maybe_recalibrate(self, step):
        if len(self.window) >= 20:
            all_mean = statistics.fmean(s for s, _ in self.window)
            recent = statistics.fmean(s for s, _ in list(self.window)[-10:])
            if all_mean and abs(recent - all_mean) / all_mean > SHIFT:
                keep = list(self.window)[-10:]
                self.window.clear()
                self.window.extend(keep)
                self.recalibrations.append({"at_step": step, "why": f"throughput shift ({all_mean:.2f} -> {recent:.2f} s/step)"})

    # ---- derived values
    def sec_per_step(self):
        return statistics.fmean(s for s, _ in self.window) if len(self.window) >= MIN_STEPS else None

    def tokens_per_second(self):
        w = [(s, k) for s, k in self.window if k]
        return sum(k for _, k in w) / sum(s for s, _ in w) if len(w) >= MIN_STEPS and sum(s for s, _ in w) > 0 else None

    def remaining_points(self, every):
        if not every or not self.max_steps:
            return 0
        return self.max_steps // every - self.step // every

    def state(self, now: float) -> dict:
        sps = self.sec_per_step()
        rem = max((self.max_steps or 0) - self.step, 0)
        train_eta = rem * sps if sps else None
        ev_mean = statistics.fmean(self.eval_seconds) if self.eval_seconds else None
        sv_mean = statistics.fmean(self.save_seconds) if self.save_seconds else None
        n_eval, n_save = self.remaining_points(self.eval_steps), self.remaining_points(self.save_steps)
        overall = None
        if train_eta is not None and (n_eval == 0 or ev_mean is not None) and (n_save == 0 or sv_mean is not None):
            overall = train_eta + n_eval * (ev_mean or 0) + n_save * (sv_mean or 0)
        seg = self.segments[-1] if self.segments else None
        done = self.phase in ("FINISHED", "EARLY_STOPPED", "FAILED")
        stale = (not done) and self.last_t is not None and now - self.last_t > STALE_SECONDS
        nxt = lambda every: (every - self.step % every) if every else None
        return {"phase": self.phase, "gpu": self.gpu.get("gpu"), "gpu_class": gpu_class(self.gpu.get("gpu"), self.gpu.get("vram_gb")), "vram_total_gb": self.gpu.get("vram_gb"),
                "vram_allocated_gb": self.gpu.get("gpu_mem_allocated_gb"), "vram_reserved_gb": self.gpu.get("gpu_mem_reserved_gb"), "vram_peak_gb": self.gpu.get("gpu_mem_peak_gb"),
                "step": self.step, "max_steps": self.max_steps, "percent": round(100 * self.step / self.max_steps, 1) if self.max_steps else None,
                "epoch": self.epoch, "epochs": self.epochs, "train_loss": self.train_loss, "val_mean_loss": self.val_mean, "val_losses": dict(self.val), "learning_rate": self.lr,
                "sec_per_step": sps, "calibrating": sps is None, "valid_steps_in_window": len(self.window), "tokens_per_second": self.tokens_per_second(),
                "training_seconds": self.train_seconds, "segment_wall_seconds": (now - seg["t0"]) if seg else None, "segments": len(self.segments),
                "train_eta_seconds": train_eta, "overall_eta_seconds": overall, "eval_overhead_seconds": ev_mean, "save_overhead_seconds": sv_mean,
                "probe_seconds": statistics.fmean(self.probe_seconds) if self.probe_seconds else None, "remaining_validations": n_eval, "remaining_saves": n_save,
                "finish_at_utc": (datetime.fromtimestamp(now, timezone.utc) + timedelta(seconds=overall)).isoformat(timespec="minutes") if overall is not None else None,
                "next_validation_in": nxt(self.eval_steps), "next_save_in": nxt(self.save_steps), "latest_checkpoint": self.latest_ckpt,
                "best_loss": self.best_loss, "best_style": self.best_style, "early_stopping": self.early, "stopped": self.stopped, "final_selection": self.final, "error": self.error,
                "recalibrations": self.recalibrations[-3:], "last_event_utc": datetime.fromtimestamp(self.last_t, timezone.utc).isoformat(timespec="seconds") if self.last_t else None,
                "seconds_since_last_event": round(now - self.last_t, 1) if self.last_t else None, "stale": stale}


# ---------------------------------------------------------------- rendering

STATUS_LABEL = {"STARTING": "LOADING (starting)", "LOADING": "LOADING (tokenizer, data, model)", "READY": "LOADING (done)", "TRAINING": "TRAINING",
                "VALIDATING": "EVALUATING (validation losses)", "STYLE_PROBE": "EVALUATING (style / fact probe)", "SAVING": "SAVING (checkpoint)",
                "FINAL_VALIDATION": "EVALUATING (final record of the best model)", "FINISHED": "COMPLETED", "EARLY_STOPPED": "COMPLETED (early stopped)", "FAILED": "FAILED"}


def status_label(phase: str) -> str:
    return STATUS_LABEL.get(phase, phase)


def _hms(s):
    if s is None:
        return "-"
    s = int(s)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _f(x, fmt="{:.4f}"):
    return "-" if x is None else fmt.format(x)


def render_text(st: dict, tz=None) -> str:
    bar_n = int((st["percent"] or 0) / 5)
    finish = "-"
    if st["finish_at_utc"]:
        dt = datetime.fromisoformat(st["finish_at_utc"])
        finish = (dt.astimezone(tz).strftime("%Y-%m-%d %H:%M %Z") if tz else dt.strftime("%Y-%m-%d %H:%M UTC")) + " (estimate)"
    eta_train = "CALIBRATING (%d/%d valid steps)" % (st["valid_steps_in_window"], MIN_STEPS) if st["calibrating"] else _hms(st["train_eta_seconds"]) + " (estimate)"
    if st["overall_eta_seconds"] is not None:
        eta_all = _hms(st["overall_eta_seconds"]) + " (estimate, incl. validation + saves)"
    elif st["calibrating"]:
        eta_all = "CALIBRATING"
    else:
        eta_all = "overhead not measured yet (needs one validation and one save)"
    es = st["early_stopping"] or {}
    es_txt = "-" if not es else (f"warm-up until step {es.get('warmup_steps')}" if st["step"] < (es.get("warmup_steps") or 0)
                                  else f"{es.get('no_improve', 0)}/{es.get('patience')} evaluations without improvement")
    if st["stopped"]:
        es_txt = f"STOPPED at step {st['stopped']['step']}: {st['stopped']['reason']}"
    lines = [
        "StillHere T2-D4 - Qwen3-8B QLoRA (Memory-Grounded Persona Adapter)",
        "",
        f"GPU: {st['gpu'] or '-'} [{st['gpu_class']}]",
        f"Status: {status_label(st['phase'])}" + (f"   (no new event for {_hms(st['seconds_since_last_event'])}, last {st['last_event_utc']})" if st["stale"] else ""),
        "",
        f"Progress: {st['step']} / {st['max_steps'] or '-'} steps ({_f(st['percent'], '{:.1f}')}%)  [{'#' * bar_n}{'.' * (20 - bar_n)}]",
        f"Epoch: {_f(st['epoch'], '{:.2f}')} / {_f(st['epochs'], '{:.2f}')}",
        "",
        f"Train Loss: {_f(st['train_loss'])}    Eval Loss (mean): {_f(st['val_mean_loss'])}    Learning Rate: {_f(st['learning_rate'], '{:.2e}')}",
        "Eval by set: " + (", ".join(f"{k.replace('eval_style_val_', '').replace('_loss', '')} {v:.4f}" for k, v in st["val_losses"].items()) or "-"),
        "",
        f"Average: {_f(st['sec_per_step'], '{:.2f}')} sec/optimizer step (last {st['valid_steps_in_window']} valid steps)    Tokens/s: {_f(st['tokens_per_second'], '{:.0f}')}",
        f"Training time: {_hms(st['training_seconds'])}    This segment (wall): {_hms(st['segment_wall_seconds'])}    Segments: {st['segments']}",
        f"Training ETA: {eta_train}",
        f"Overall ETA: {eta_all}",
        f"Estimated Finish: {finish}",
        f"Validation + probe: {_hms(st['eval_overhead_seconds'])} each, {st['remaining_validations']} left; save: {_hms(st['save_overhead_seconds'])} each, {st['remaining_saves']} left",
        "",
        f"VRAM: allocated {_f(st['vram_allocated_gb'], '{:.1f}')} / reserved {_f(st['vram_reserved_gb'], '{:.1f}')} / peak {_f(st['vram_peak_gb'], '{:.1f}')} of {_f(st['vram_total_gb'], '{:.0f}')} GB",
        f"Latest Checkpoint: {st['latest_checkpoint'] or '-'}    next validation in {st['next_validation_in'] or '-'} steps, next save in {st['next_save_in'] or '-'} steps",
        f"Best Val-Loss Checkpoint: {('checkpoint-%s (%.4f)' % (st['best_loss']['step'], st['best_loss']['mean_loss'])) if st['best_loss'] else '-'}",
        "Best Style Checkpoint: chosen after training on DEV_PROBE generations (stages 12-13), never by loss alone",
        f"Early Stopping: {es_txt}",
    ]
    if st["recalibrations"]:
        lines.append("ETA recalibrated: " + "; ".join(f"step {r['at_step']}: {r['why']}" for r in st["recalibrations"]))
    if st["final_selection"]:
        f = st["final_selection"]
        lines.append(f"SELECTED: checkpoint-{f['step']} ({f['track']}): {f['reason']}")
    if st["error"]:
        lines.append(f"ERROR: {st['error']['type']}: {st['error']['message']}")
    return "\n".join(lines)


def render_html(st: dict, log_lines=(), tz=None, exit_code=None) -> str:
    import html
    head = render_text(st, tz)
    if exit_code is not None:
        head += f"\n\nPROCESS EXITED with code {exit_code}" + ("" if exit_code == 0 else "  <-- see the log below")
    log = "\n".join(log_lines)
    return (f"<pre style='font-size:12px;line-height:1.35;margin:0'>{html.escape(head)}</pre>"
            f"<details {'open' if exit_code not in (None, 0) else ''}><summary>log (last {len(log_lines)} lines)</summary>"
            f"<pre style='max-height:260px;overflow:auto;font-size:11px;background:#f6f6f6'>{html.escape(log)}</pre></details>")


# ---------------------------------------------------------------- live run (notebook side)

def live_run(cmd, events_path: Path, log_path: Path, *, refresh: float = 10.0, handle=None, tz=None, max_log_lines: int = 40, pid_path: Path | None = None,
             clock=time.time, sleep=time.sleep) -> int:
    """Start ``cmd`` unbuffered, stream its output into a bounded log and a log file, and refresh ONE display (``handle.update``) every
    ``refresh`` seconds from the real events. Returns the exit code. A user interrupt stops the child (resume later from the latest checkpoint)."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail = deque(maxlen=max_log_lines)
    proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1, env=env)
    if pid_path:
        Path(pid_path).write_text(str(proc.pid), encoding="utf-8")

    def pump():
        with log_path.open("a", encoding="utf-8") as lf:
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
                tail.append(line.rstrip("\n")[:300])
    th = threading.Thread(target=pump, daemon=True)
    th.start()
    if handle is None:
        try:
            from IPython.display import HTML, display
            handle = display(HTML("<pre>starting...</pre>"), display_id=True)
            wrap = HTML
        except ImportError:
            handle, wrap = None, None
    else:
        wrap = lambda x: x

    def show(code=None):
        st = ProgressModel.from_file(events_path).state(clock())
        if handle is not None:
            handle.update(wrap(render_html(st, list(tail), tz, code)))
        return st
    try:
        while proc.poll() is None:
            show()
            sleep(refresh)
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait(timeout=60)
        tail.append("INTERRUPTED BY USER: training process stopped; re-run the cell to resume from the latest complete checkpoint")
    th.join(timeout=10)
    code = proc.wait()
    show(code)
    return code


def monitor(events_path: Path, pid_path: Path | None = None, tz=None) -> str:
    """Reconnect view: the persisted state as last written, and whether the training process is actually alive. It never implies that
    training resumed: only a running process (and new events) means training is going on."""
    st = ProgressModel.from_file(events_path).state(time.time())
    alive = None
    if pid_path and Path(pid_path).exists():
        pid = Path(pid_path).read_text().strip()
        alive = Path(f"/proc/{pid}").exists() if Path("/proc").exists() else None
    head = {True: "TRAINING PROCESS: RUNNING", False: "TRAINING PROCESS: NOT RUNNING (this runtime) - the state below is the last persisted one",
            None: "TRAINING PROCESS: unknown (no pid in this runtime) - the state below is the last persisted one"}[alive]
    return head + "\n" + render_text(st, tz)
