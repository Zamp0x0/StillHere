"""Evidence rendering, version 2 (BUILD-V3.4): every SMS names who sent it and who received it.

Version 1 (``answer_pilot.render_item``) shows the speaker of a message. For a message the user sent, the person it went to appears only inside the text, so a
reader can take the person named in the text for its sender. Version 2 adds both ends of every SMS from the stored message metadata. Everything else is
version 1 exactly, and version 1 stays the frozen renderer of every earlier result.
"""
from __future__ import annotations

from dataclasses import replace  # noqa: F401
from typing import List, Mapping, Optional, Sequence, Set, Tuple

from . import answer_pilot as ap

VERSION = "v3-round-2: v3-round-1 plus sender and recipient of every SMS"
SELF = "本人"


# One implementation of the SMS direction rule: it lives in the context layer, which the orchestrator also uses.
from stillhere.context.attribution import Direction, sms_direction  # noqa: E402,F401


def render_item(evidence_id: str, e: ap.Evidence, direction: Optional[Direction] = None) -> str:
    if direction is None:
        return ap.render_item(evidence_id, e)
    head = [f"[{evidence_id}]", f"来源：{e.source_type}"]
    if e.event_time:
        head.append(f"时间：{e.event_time[:16].replace('T', ' ')}")
    head.append(f"发送方：{direction.sender}")
    head.append(f"接收方：{direction.recipient}")
    if e.locator:
        head.append(f"定位：{e.locator}")
    return " | ".join(head) + f"\n内容：{e.text}"


def build_context(items: Sequence[ap.Evidence], directions: Mapping[str, Direction], *, max_tokens: int = ap.MAX_CONTEXT_TOKENS, top_k: int = ap.TOP_K) -> ap.Context:
    """``answer_pilot.build_context`` with the version 2 header. Same order, same ceiling, the same rule of leaving out whole records."""
    seen: Set[str] = set()
    unique = [e for e in items if not (e.key in seen or seen.add(e.key))][:top_k]
    parts: List[str] = []
    kept: List[str] = []
    used = 0
    dropped: List[str] = []
    for e in unique:
        block = render_item(f"E{len(parts) + 1:02d}", e, directions.get(e.key))
        if dropped or used + ap.estimate_tokens(block) > max_tokens:
            dropped.append(e.key)
            continue
        parts.append(block)
        kept.append(e.key)
        used += ap.estimate_tokens(block)
    text = "\n\n".join(parts) if parts else "（没有找到相关记忆）"
    return ap.Context(text, tuple(kept), {f"E{i:02d}": k for i, k in enumerate(kept, 1)}, tuple(dropped), used, bool(dropped), ap.sha256_text(text))
