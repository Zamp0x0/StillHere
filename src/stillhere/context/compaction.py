"""Compaction: lightweight, traceable and lossless about what it leaves out. No model, no summary, nothing written to memory.

* ``dedupe_units``: a unit whose every message is already in the recent conversation is not injected twice.
* ``overflow_selection``: when a whole round does not fit, choose which of its messages to show. The order of priority keeps
  what can change the answer: the messages the retriever hit, the last message (the final state of the round), messages that
  change a state (取消, 改成, 已经完成, 其实不是 ...), the neighbours of the hits, the first message, then the rest nearest a hit
  first. Messages are added in that order while the rendered unit fits. The caller lists every omitted message id, and the
  rendered round marks each omitted stretch. A round whose hit messages alone do not fit is not shown in part at all.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Set

from .units import Unit, UnitMessage

STATE_CUES = ("取消", "不去", "不用了", "改成", "改到", "改为", "换成", "推迟", "延期", "提前", "已经", "完成", "搞定", "做完", "办好", "其实",
              "不是", "没有", "算了", "决定", "最终", "最后", "确定", "不再", "放弃", "结束", "改天", "作废")


def has_state_change(text: str) -> bool:
    return any(c in text for c in STATE_CUES)


def dedupe_units(units: Sequence[Unit], already: Set[str]):
    kept, dropped = [], {}
    for u in units:
        ids = set(u.message_ids)
        if ids and ids <= already:
            dropped[u.key] = "duplicate_of_recent"
        else:
            kept.append(u)
    return kept, dropped


def priority(u: Unit) -> List[UnitMessage]:
    msgs = u.messages
    if not msgs:
        return []
    hits = [m for m in msgs if m.message_id in set(u.hit_message_ids)]
    hit_pos = [m.position for m in hits] or [msgs[0].position]
    order: List[UnitMessage] = []

    def add(m: UnitMessage) -> None:
        if m not in order:
            order.append(m)

    for m in hits:
        add(m)
    add(msgs[-1])
    for m in reversed(msgs):
        if has_state_change(m.content):
            add(m)
    for m in msgs:
        if any(abs(m.position - p) == 1 for p in hit_pos):
            add(m)
    add(msgs[0])
    for m in sorted(msgs, key=lambda m: (min(abs(m.position - p) for p in hit_pos), m.position)):
        add(m)
    return order


def essential(u: Unit) -> List[UnitMessage]:
    """What an overflowing round must keep if it is shown at all: its hits, its final message and every state change."""
    hits = set(u.hit_message_ids)
    return [m for m in u.messages if m.message_id in hits or m is u.messages[-1] or has_state_change(m.content)]


def overflow_selection(u: Unit, fits: Callable[[Sequence[UnitMessage]], bool], *, essential_only: bool = True,
                       start: Optional[Sequence[UnitMessage]] = None) -> Optional[List[UnitMessage]]:
    """The messages to show, in round order, or None if even the hit messages do not fit.

    With ``essential_only`` (the first pass) only hits, the final message and state changes are taken, so one oversized round cannot
    use up the space later units need. The caller then extends each partial round with the rest, in ``priority`` order, from what
    is left (``essential_only=False`` with ``start``)."""
    order = priority(u)
    if start is None:
        must = [m for m in order if m.message_id in set(u.hit_message_ids)] or order[:1]
        chosen = sorted(must, key=lambda m: m.position)
        if not fits(chosen):
            return None
    else:
        chosen = sorted(start, key=lambda m: m.position)
    allowed = set(m.message_id for m in essential(u)) if essential_only else None
    for m in order:
        if m in chosen or (allowed is not None and m.message_id not in allowed):
            continue
        trial = sorted(chosen + [m], key=lambda x: x.position)
        if fits(trial):
            chosen = trial
    return chosen
