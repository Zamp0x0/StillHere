"""NEW group: the same StillHere store / dense index as OLD, the P2 pack (``stillhere.persona.memory_p2``), the temporal view and the
coverage note, rendered in the SAME evidence frame and token budget as OLD (the note is counted inside the budget)."""
from __future__ import annotations

import time
from typing import Callable

from lbpilot import context as CX
from lbpilot import protocol as LP
from lbpilot.stillhere_memory import TZ, FutureEvidence

from stillhere.persona import memory_p2 as MP

from . import protocol as P


def p2config() -> MP.P2Config:
    c = P.P2["config"]
    return MP.P2Config(**{k: (tuple(v) if k == "neighbors" else v) for k, v in c.items()})          # every registered field, nothing defaulted


def search(sh, query: str, at, *, dense=None) -> dict:
    """``sh``: an lbpilot ``StillHereMemory`` (the same store OLD uses). Returns entries (with the P2 temporal note) + coverage + meta."""
    from stillhere.persona import memory_v2 as V2
    from stillhere.persona import product_config as PC
    mp = PC.PRODUCT_MEMORY_PIPELINE
    cfg = V2.V2Config(use_dense=dense is not None, k=int(mp["k"]), neighbors=tuple(mp["neighbors"]), date_mode=mp["date_mode"], strip_qform=bool(mp["strip_qform"]))
    t0 = time.perf_counter()
    pk, meta = MP.pack_p2(sh.api, sh.pid, query, at, cfg, p2config(), dense=dense, names=sh.names)
    ms = round((time.perf_counter() - t0) * 1000, 1)
    limit = at.astimezone(TZ).isoformat()[:19]
    late = [i.eid for i in pk.items if (i.event_time or "")[:19] > limit]
    if late:
        raise FutureEvidence(f"items after the ask time: {late}")
    entries = []
    for i in pk.items:
        day = (i.event_time or "")[:10] or None
        ids = [i.unit_key] if i.unit_key else []
        note = MP.event_time_note(i.text, day)
        text = f"（{note}）{i.text}" if note else i.text
        entries.append(CX.Entry(rank=int(i.rank), date=day, source=i.source_label, text=text, ids=ids + [m for m in (i.message_ids or []) if m not in ids],
                                score=float(i.score) if i.score is not None else None))
    p2 = meta.get("p2") or {}
    return {"entries": entries, "coverage": p2.get("coverage") or [],
            "meta": {"retrieval_ms": ms, "items": len(pk.items), "dense": bool(meta.get("dense")), "p2_kinds": p2.get("kinds"), "p2_requirements": p2.get("requirements"),
                     "p2_trace": p2.get("trace"), "second_pass": p2.get("second_pass", False)}}


def render(res: dict, count: Callable[[str], int]) -> dict:
    """Coverage note + entries within ONE budget (the note's tokens are subtracted first)."""
    note = MP.coverage_note(res["coverage"])
    budget = int(LP.EVIDENCE["token_budget"])
    nt = count(note + "\n") if note else 0
    ctx = CX.render(res["entries"], budget=budget - nt, count=count)
    if note:
        ctx["text"] = note + "\n" + ctx["text"]
        ctx["tokens"] = count(ctx["text"])
    ctx["coverage"] = res["coverage"]
    return ctx
