"""Rendering for the reader. Plain text, deterministic, one block per unit; the header layout follows ``answer_pilot.render_item``.

A round lists every message on its own line, numbered in order, with its time, sender and recipient (unknown stays 未知). A
round shown in part says so in its header and has one line per omitted stretch, so the reader can never take a part for the
whole. That a round has messages after ``as_of`` / ``known_at`` is not rendered: it would leak that later messages exist.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Sequence

from stillhere.memory.temporal import CALENDAR_TZ

from .units import TIME_KIND_LABELS, Unit, UnitMessage

SOURCE_LABELS = {"lifebench_agent_chat": "AI 对话", "synthetic_chat_round": "AI 对话", "lifebench_sms": "短信", "synthetic_sms": "短信"}
NO_EVIDENCE = "（没有找到相关记忆）"
SECTION = {"pinned": "【身份信息】（已确认的记录）", "working": "【当前任务】（仅本次会话，不是长期记忆）", "recent": "【近期对话】（当前会话，按时间顺序）",
           "boundary": "【时间边界】", "derived": "【推导内容】（不是原始记录）", "evidence": "【记忆证据】"}


def when(t: Optional[datetime], day_only: bool = False) -> str:
    if t is None:
        return "未知"
    local = t.astimezone(CALENDAR_TZ)
    return local.strftime("%Y-%m-%d") + "（只知道日期）" if day_only else local.strftime("%Y-%m-%d %H:%M")


def message_line(m: UnitMessage) -> str:
    return f"  {m.position}. {m.timestamp.astimezone(CALENDAR_TZ).strftime('%H:%M')} {m.sender} → {m.recipient}：{m.content}"


def omitted_line(first: int, last: int) -> str:
    n = last - first + 1
    return f"  …（省略第 {first} 条，因上下文预算）" if n == 1 else f"  …（省略第 {first}–{last} 条，共 {n} 条，因上下文预算）"


def render_round(cid: str, u: Unit, shown: Optional[Sequence[UnitMessage]] = None) -> str:
    shown = list(u.messages if shown is None else shown)
    total = len(u.messages)
    complete = len(shown) == total
    state = f"完整，共 {total} 条" if complete else f"部分显示 {len(shown)}/{total} 条（CONTEXT_OVERFLOW：因上下文预算，省略的消息已标出）"
    head = [f"[{cid}]", f"来源：{SOURCE_LABELS.get(u.source_type, u.source_type)}", f"时间：{when(u.event_time, u.day_precision)}", f"对话：{state}"]
    if u.order_anomaly:
        head.append("注意：存储顺序与时间顺序不一致，按时间排列")
    lines = [" | ".join(head)]
    keep = {m.position for m in shown}
    gap: List[int] = []
    for m in u.messages:
        if m.position in keep:
            if gap:
                lines.append(omitted_line(gap[0], gap[-1]))
                gap = []
            lines.append(message_line(m))
        else:
            gap.append(m.position)
    if gap:
        lines.append(omitted_line(gap[0], gap[-1]))
    return "\n".join(lines)


def render_unit(cid: str, u: Unit, shown: Optional[Sequence[UnitMessage]] = None) -> str:
    if u.kind == "round":
        return render_round(cid, u, shown)
    head = [f"[{cid}]", f"来源：{SOURCE_LABELS.get(u.source_type, u.source_type)}", f"时间：{when(u.event_time, u.day_precision)}"]
    if u.kind == "message":
        m = u.messages[0]
        head += [f"发送方：{m.sender}", f"接收方：{m.recipient}"]
        body = m.content
    else:
        head.append(f"时间类型：{TIME_KIND_LABELS.get(u.time_kind, u.time_kind)}")
        body = u.statement or ""
    if u.locator:
        head.append(f"定位：{u.locator}")
    return " | ".join(head) + f"\n内容：{body}"


def recent_line(cid: str, m: UnitMessage) -> str:
    return f"[{cid}] {m.timestamp.astimezone(CALENDAR_TZ).strftime('%Y-%m-%d %H:%M')} {m.sender} → {m.recipient}：{m.content}"


def join_sections(sections: Sequence[tuple]) -> str:
    """``(name, [blocks])`` in order. With evidence alone and nothing else, no headers: the frozen layout of ``answer_pilot``."""
    present = [(n, b) for n, b in sections if b]
    if not present:
        return NO_EVIDENCE
    if len(present) == 1 and present[0][0] == "evidence":
        return "\n\n".join(present[0][1])
    parts = []
    for name, blocks in present:
        parts.append(SECTION[name] + "\n" + ("\n\n" if name == "evidence" else "\n").join(blocks))
    if not any(n == "evidence" for n, _ in present):
        parts.append(SECTION["evidence"] + "\n" + NO_EVIDENCE)
    return "\n\n".join(parts)
