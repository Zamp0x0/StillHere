"""P4-P5 Full Build Gate: small real-model Writer test -> evidence validation -> quality decision -> full structured build OR evidence-only.

The Writer's output is a CANDIDATE, not a fact (deterministic verification + strict attribution already run inside ``memory_store``).
Whether the structured layer may be built over the WHOLE private archive is decided here from typed errors with a severity, not from
one composite score:

    CRITICAL  a fact of someone else stored as the persona's (attribution), a committed fact the evidence does not support, a negation
              stored as its opposite                                  -> EVIDENCE_ONLY (any one)
    MAJOR     a wrong temporal state (old value still current, cancelled event still planned)  -> EVIDENCE_ONLY if > max_major_rate of
              the committed facts, else a warning
    MINOR     a missed fact (low recall), a not-memory-worthy write  -> reported; never blocks (the evidence archive still answers)
    INFO      no verified fact at all in the sample                  -> EVIDENCE_ONLY (nothing to gain from a 13-hour build)

Inputs, each optional: the synthetic-gold Writer checks (``colab_eval.synthetic_gold``), the small real sample's candidate statistics
(``memory_store`` writer run + ``writer_candidates.jsonl``) and a judge audit (``WRITER_DEEPSEEK_SCORES`` aggregates). The decision
records which inputs were present; a missing input is never counted as a pass.

``gated_full_build`` backs the production store up (SQLite backup API, sha256 verified) before a full build and restores it if the
build raises; the evidence archive is never modified by the Writer.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional

GATE_VERSION = "stillhere.persona.writer_gate/1.0"
SYNTHETIC_SEVERITY = {
    "no_self_australia": ("CRITICAL", "attribution: the other person's study abroad stored as the persona's"),
    "wedding_not_self": ("CRITICAL", "attribution: the sister's wedding stored as the persona's own event"),
    "shenzhen_current": ("MAJOR", "temporal: the current workplace is not the active value"),
    "guangzhou_closed": ("MAJOR", "temporal: the former workplace is still active"),
    "dentist_cancelled": ("MAJOR", "temporal: a cancelled plan is not recorded as cancelled"),
    "wedding_event_third_party": ("MINOR", "recall: the third-party event was not extracted"),
    "coffee_negation": ("MINOR", "recall: the negated preference was not extracted"),
    "third_person_not_self": ("CRITICAL", "attribution: a relative's event reported by the persona stored as the persona's own (probe)"),
}


@dataclass
class Finding:
    check: str
    severity: str
    detail: str
    source: str


@dataclass
class GateDecision:
    decision: str                                   # FULL_BUILD | EVIDENCE_ONLY
    reasons: List[str]
    findings: List[Finding] = field(default_factory=list)
    inputs: Dict[str, bool] = field(default_factory=dict)
    stats: Dict[str, object] = field(default_factory=dict)
    version: str = GATE_VERSION

    def to_dict(self) -> dict:
        d = asdict(self)
        d["findings"] = [asdict(f) for f in self.findings]
        return d


def decide(*, synthetic_checks: Optional[Mapping[str, bool]] = None, sample: Optional[Mapping] = None, judge: Optional[Mapping] = None,
           max_major_rate: float = 0.10, require_synthetic: bool = True) -> GateDecision:
    """``sample``: {"sessions", "candidates", "verified"} of the small real-model run. ``judge``: {"supported", "unsupported",
    "attribution_errors", "temporal_errors", "worthiness_errors", "suspected_missing"} (counts)."""
    findings: List[Finding] = []
    for k, ok in (synthetic_checks or {}).items():
        if not ok:
            sev, why = SYNTHETIC_SEVERITY.get(k, ("MAJOR", f"synthetic check {k} failed"))
            findings.append(Finding(k, sev, why, "synthetic_gold"))
    committed = int((sample or {}).get("verified") or 0)
    if judge:
        for key, sev, why in (("attribution_errors", "CRITICAL", "judge: committed fact attributed to the wrong person"),
                              ("unsupported", "CRITICAL", "judge: committed fact not supported by the cited messages"),
                              ("temporal_errors", "MAJOR", "judge: wrong temporal state"),
                              ("worthiness_errors", "MINOR", "judge: not worth remembering"),
                              ("suspected_missing", "MINOR", "judge: suspected missed facts")):
            n = int(judge.get(key) or 0)
            if n:
                findings.append(Finding(key, sev, f"{why} (n={n})", "judge"))
    reasons: List[str] = []
    crit = [f for f in findings if f.severity == "CRITICAL"]
    major = [f for f in findings if f.severity == "MAJOR"]
    if crit:
        reasons.append(f"{len(crit)} critical finding(s): " + "; ".join(f.check for f in crit))
    major_n = int((judge or {}).get("temporal_errors") or 0) + sum(1 for f in major if f.source == "synthetic_gold")
    if committed and major_n / max(committed, 1) > max_major_rate:
        reasons.append(f"major temporal findings {major_n} / {committed} committed facts > {max_major_rate:.0%}")
    elif not committed and major:
        reasons.append(f"{len(major)} major synthetic finding(s) and no committed real fact to weigh them against")
    if sample is not None and committed == 0:
        reasons.append("no verified fact in the real-model sample: a full build would add nothing")
    inputs = {"synthetic_gold": synthetic_checks is not None, "real_sample": sample is not None, "judge_audit": judge is not None}
    if require_synthetic and synthetic_checks is None:
        reasons.append("synthetic gold checks missing: the gate does not pass without them")
    if sample is None:
        reasons.append("no real-model sample: the gate does not pass without one")
    stats = {"committed": committed, "candidates": (sample or {}).get("candidates"), "sessions": (sample or {}).get("sessions")}
    if judge and (judge.get("supported") is not None):
        sup, miss = int(judge.get("supported") or 0), int(judge.get("suspected_missing") or 0)
        stats["judge_estimated_recall"] = round(sup / (sup + miss), 3) if sup + miss else None
    return GateDecision("EVIDENCE_ONLY" if reasons else "FULL_BUILD", reasons or ["no blocking finding"], findings, inputs, stats)


# ---------------------------------------------------------------- backup / restore of the production store
def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def backup_store(db: Path, dest_dir: Path) -> dict:
    """A consistent single-file copy (SQLite online backup, includes WAL content) + its sha256. Never overwrites an existing backup."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / "memory.sqlite"
    if out.exists():
        raise FileExistsError(f"{out} exists: backups are never overwritten")
    src = sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True)
    dst = sqlite3.connect(str(out))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    rec = {"file": str(out), "sha256": _sha(out), "source": str(db)}
    (dest_dir / "BACKUP.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def restore_store(backup_dir: Path, db: Path) -> dict:
    rec = json.loads((backup_dir / "BACKUP.json").read_text(encoding="utf-8"))
    src = backup_dir / "memory.sqlite"
    if _sha(src) != rec["sha256"]:
        raise ValueError("the backup changed since it was written: not restored")
    for suffix in ("-wal", "-shm"):
        Path(str(db) + suffix).unlink(missing_ok=True)
    shutil.copyfile(src, db)
    return {"restored": str(db), "sha256": _sha(db)}


def gated_full_build(root, profile_id: str, decision: GateDecision, *, complete: Callable, reviewer: str, backup_dir: Path,
                     log=lambda s: None) -> dict:
    """Run the full structured build only for ``FULL_BUILD``; otherwise report EVIDENCE_ONLY (the archive keeps answering). The store is
    backed up first and restored if the build raises; a build that finishes is kept (it is resumable and idempotent)."""
    from . import memory_store as MS
    from . import store as S
    root = Path(root) if root else S.default_root()
    pdir = S.profile_dir(root, profile_id)
    db = MS.mdir(pdir) / "memory.sqlite"
    if decision.decision != "FULL_BUILD":
        return {"mode": "EVIDENCE_ONLY", "gate": decision.to_dict(), "built": False}
    bk = backup_store(db, backup_dir)
    try:
        res = MS.build_semantic(root, profile_id, complete=complete, reviewer=reviewer, log=log)
    except Exception as e:  # noqa: BLE001 - restore, then report
        restored = restore_store(backup_dir, db)
        return {"mode": "EVIDENCE_ONLY", "gate": decision.to_dict(), "built": False, "error": f"{type(e).__name__}: {str(e)[:300]}",
                "backup": bk, "restored": restored}
    return {"mode": "STRUCTURED_MEMORY", "gate": decision.to_dict(), "built": True, "backup": bk, "result": res}
