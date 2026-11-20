"""Split guard for the frozen FAST memory QA (``FREEZE_MANIFEST.json`` -> ``fast_qa_split``).

The HOLDOUT half was used ONCE, for the pre-registered ``MEMORY_V2_RULE`` decision (2026-11-18). From then on every retrieval / reader
tuning reads DEV only. ``select`` is the one place a script turns a split name into questions:

- ``dev``: always allowed;
- ``holdout`` / ``all``: refused unless the caller passes ``allow_holdout`` = a written reason (recorded in the output by the caller);
  tuning scripts never pass it;
- the manifest itself is checked: DEV and HOLDOUT disjoint, and every frozen question in exactly one of them.

It never reads question text, gold or answers: only ids.
"""
from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Sequence


class SplitGuardError(RuntimeError):
    pass


def check_manifest(split: Mapping, question_ids: Iterable[str]) -> None:
    dev, hold = set(split.get("dev") or ()), set(split.get("holdout") or ())
    if not dev or not hold:
        raise SplitGuardError("SPLIT_GUARD: the manifest has no DEV or no HOLDOUT list")
    if dev & hold:
        raise SplitGuardError(f"SPLIT_GUARD: {len(dev & hold)} question ids are in both DEV and HOLDOUT")
    ids = set(question_ids)
    if ids - dev - hold:
        raise SplitGuardError(f"SPLIT_GUARD: {len(ids - dev - hold)} frozen questions are in neither split")
    if (dev | hold) - ids:
        raise SplitGuardError(f"SPLIT_GUARD: {len((dev | hold) - ids)} split ids are not in the frozen QA (wrong QA file?)")


def select(qa: Sequence[Mapping], split: Mapping, name: str, *, allow_holdout: Optional[str] = None) -> List[Mapping]:
    """The frozen QA rows of split ``name`` (``dev`` | ``holdout`` | ``all``), after the manifest checks."""
    check_manifest(split, (q["question_id"] for q in qa))
    if name == "dev":
        keep = set(split["dev"])
    elif name in ("holdout", "all"):
        if not (allow_holdout or "").strip():
            raise SplitGuardError("REFUSED: the HOLDOUT was used once for the registered decision (2026-11-18); tuning may only read DEV. "
                                  "Pass a written reason (allow_holdout / --allow-holdout-reuse) only for a pre-registered re-run.")
        keep = set(split["holdout"]) if name == "holdout" else set(split["dev"]) | set(split["holdout"])
    else:
        raise SplitGuardError(f"unknown split {name!r}")
    return [q for q in qa if q["question_id"] in keep]
