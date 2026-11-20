"""Memory V2 (P1-P3): router coverage, date-aware first-hop retrieval, local dense / hybrid fusion and evidence windows.

Every part is a switch over the unchanged V1 path (``runtime.route_of`` + ``memory_store.config_for`` + EF1 + ``fact_pack.build``), so
V1 stays the default and both can be compared on the same frozen questions (``persona.retrieval_bench``).

P1 router (``route_of_v2``): V1 sends a question to memory only with a history / personal cue. The FAST audit found 32 of 90 answerable
    questions routed to chat; their cues were the recall particle 来着 / 还记得, recent-past deixis (刚才 / 昨晚), plans (打算 / 准备),
    "why did you" (为什么 + 你 / 我) and preferences with a degree word (你最喜欢) ("with whom" was tried and dropped). These are added as
    patterns of the QUESTION FORM, never of any question's content; small talk is still never a memory query.
P2 date-aware first hop (``explicit_windows`` / ``date_mode``): a question that names an absolute date / month (digits or a Chinese month)
    gets the EF1 time-window channel boosted (V1 already does this for ANY resolved window) and, in ``prefer`` mode, candidates inside the
    window (± ``slack_days``) are moved to the front WITHOUT dropping the others (a hard filter can only lose recall when the date is
    approximate). Deictic times (今天 / 刚才) are never treated as explicit: in a digital-human chat they refer to the conversation moment
    of the asker, not to an archive date.
P3 evidence (``expand_pack``): one EF1 unit of a WeFlow archive is one message (median 4 characters), so each pack item gets up to
    ``before`` / ``after`` neighbouring messages of the same conversation (visible at the ask time, never duplicated across items).
    Dense (``dense``): an optional ``ef1.channels.DenseChannel`` (``retrieval.ef1.local_dense``) fused by EF1's weighted RRF.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence

RECALL_CUE = re.compile(r"来着|还记得|记不记得|记得吗|忘了吗")
RECENT_PAST_CUE = re.compile(r"刚才|刚刚|昨晚|昨天晚上|前几天|那会儿|当初|后来|一开始")
PLAN_CUE = re.compile(r"打算|准备|计划|想去|要去|决定")
WHY_SELF_CUE = re.compile(r"[你我](?:们)?[^，。？?]{0,8}为什么|为什么[^，。？?]{0,6}[你我]")
PREFERENCE_CUE = re.compile(r"[你我](?:最|比较|特别|很|最近)?(?:喜欢|讨厌|爱吃|爱喝|爱看|想要|希望)")
# "with whom" (跟谁 / 是谁) was dropped on 2026-11-18 after the DEV router audit: it was never the only cue of a DEV memory question, but
# it sent 4 present-tense chat lines of the persona DEV split to memory ("在跟谁打电话"); "who" in a recall question is caught by 来着 etc.
SOFT_CUES = (("recall", RECALL_CUE), ("recent_past", RECENT_PAST_CUE), ("plan", PLAN_CUE), ("why_self", WHY_SELF_CUE),
             ("preference", PREFERENCE_CUE))
# P1 (2026-11-19, reader_version "p1"): past speech between the two people ("你说过 / 我问你…的 / 那次 / 我是怎么回答的"). V3 known
# failures: answerable questions with these forms went to chat. Needs a personal pronoun AND must not ask about general knowledge
# ("你说过 Python 支持多继承吗" stays chat): a Latin-letter / digit technical term or a knowledge verb blocks it.
PAST_SPEECH_CUE = re.compile(r"(?:说过|讲过|提过|问过|聊过|答应过|告诉过)|(?:那次|上次|当时|那天)|怎么(?:回答|回|解释|说)的")
PERSONAL = re.compile(r"[你我]|咱")
KNOWLEDGE_BLOCK = re.compile(r"[A-Za-z]{2,}|支持|原理|定义|是什么意思|怎么用|用法|公式|翻译|英文|单词|代码|函数|多少钱一")
EXPLICIT_DATE = re.compile(r"\d|[一二三四五六七八九十]{1,3}月|[一二三四五六七八九十]{1,3}[号日]")


def route_v2(text: str, at: Optional[datetime] = None, version: str = "v1") -> Dict[str, object]:
    """``{"route": chat | date | fact, "why": [...]}``. V1 decisions are kept; V2 only turns some V1 'chat' into 'fact'. ``version`` "p1"
    adds the past-speech cue (default "v1": unchanged)."""
    from . import routing as RT
    from stillhere.session.chat_route import is_small_talk
    t = (text or "").strip()
    v1 = RT.route_of(t, at)
    if v1 != "chat":
        return {"route": v1, "why": ["v1"]}
    if not t or is_small_talk(t) or not RT.QUESTION_CUE.search(t):
        return {"route": "chat", "why": ["no question" if t else "empty"]}
    why = [name for name, rx in SOFT_CUES if rx.search(t)]
    if version == "p1" and not why and PAST_SPEECH_CUE.search(t) and PERSONAL.search(t) and not KNOWLEDGE_BLOCK.search(t):
        why = ["past_speech"]
    return {"route": "fact", "why": why} if why else {"route": "chat", "why": ["no memory cue"]}


def route_of_v2(text: str, at: Optional[datetime] = None, version: str = "v1") -> str:
    return str(route_v2(text, at, version)["route"])


def explicit_windows(question: str, at: datetime):
    """EF1 windows that come from an explicit date / month in the question (not from 今天 / 刚才 / 上周)."""
    from stillhere.retrieval.ef1.query import prepare
    from stillhere.retrieval.ef1.temporal import resolve
    ti = resolve(prepare(question, at).body, at)
    explicit = [m for m in ti.mentions if EXPLICIT_DATE.search(m.text or "")]
    if not explicit:
        return []
    spans = [(m.start, m.end) for m in explicit]
    return [w for w in ti.windows if any(w.start < e and s < w.end for s, e in spans)] or [w for w in ti.windows if w.role == "mention"]


@dataclass
class V2Config:
    router: str = "v2"                  # "v1" | "v2"
    k: int = 8                          # pack units (V1: 8)
    date_mode: str = "prefer"           # "off" | "boost" (V1 behaviour) | "prefer" | "filter"
    slack_days: int = 1
    neighbors: tuple = (2, 2)           # (before, after) messages added to each item; (0, 0) = V1 items
    use_dense: bool = True
    dense_weight: float = 1.0
    strip_qform: bool = False           # P0 (2026-11-18): lexical channels without the question form (EF1Config.strip_question_form)
    notes: List[str] = field(default_factory=list)


# P0 retrieval round (2026-11-18, DEV only): the two FUTURE candidates to be confirmed on the fresh human-reviewed question set. Neither
# is a default: ``V2Config()`` and the registered eval groups are unchanged, and the production choice stays V2L (MEMORY_V2_RULE).
#   BEST_STABLE_V2          the adopted V2L + question-form removal in the lexical channels: DEV Reader-input recall +0.069
#                           [0.015, 0.138] over V2L (5 wins / 41 ties / 0 losses), MRR not worse, +20 input tokens
#   BEST_EVIDENCE_FOCUSED   40 hybrid anchors, no windows (= eval group V2D_K40): DEV recall = V2D (+0.015, n.s.); better answers on the
#                           HOLDOUT were post-hoc and exploratory. The question-form switch is NOT added (it lowered hybrid MRR on DEV)
FUTURE_CANDIDATES: Dict[str, dict] = {
    "BEST_STABLE_V2": {"use_dense": False, "strip_qform": True},
    "BEST_EVIDENCE_FOCUSED": {"use_dense": True, "k": 40, "neighbors": (0, 0)},
}


def ef1_config(question: str, at: datetime, cfg: V2Config, dense_available: bool):
    from stillhere.retrieval.ef1.pipeline import DEFAULT_WEIGHTS, EF1Config
    from . import memory_store as MS
    w = dict(DEFAULT_WEIGHTS)
    if cfg.date_mode != "off" and explicit_windows(question, at):
        w["time_window"] = MS.DATE_WEIGHT
    elif cfg.date_mode == "boost":                                  # V1: any resolved window boosts the channel
        from stillhere.retrieval.ef1.query import prepare
        from stillhere.retrieval.ef1.temporal import resolve
        if resolve(prepare(question, at).body, at).windows:
            w["time_window"] = MS.DATE_WEIGHT
    use_dense = bool(cfg.use_dense and dense_available)
    if use_dense:
        w["dense"] = cfg.dense_weight
        w.setdefault("sub_dense", cfg.dense_weight)
    return EF1Config(top_k=cfg.k, adaptive_k=False, use_dense=use_dense, weights=w, strip_question_form=cfg.strip_qform)


def _unit_day(api, persona_id: str, key: str):
    row = api.store._conn.execute("SELECT timestamp FROM messages WHERE target_persona_id = ? AND (locator = ? OR message_id = ?) LIMIT 1",
                                  (persona_id, key, key)).fetchone()
    return datetime.fromisoformat(row[0]).date() if row else None


def apply_date_mode(api, cs, question: str, at: datetime, cfg: V2Config):
    """``prefer``: stable re-order, in-window candidates first; ``filter``: keep only in-window ones (when any exist). Returns notes."""
    if cfg.date_mode not in ("prefer", "filter"):
        return []
    wins = explicit_windows(question, at)
    if not wins:
        return []
    slack = timedelta(days=cfg.slack_days)
    pool = cs.candidates[: max(cs.k * 25, 200)]
    inside = []
    for c in pool:
        d = None if c.key.startswith("mem:") else _unit_day(api, cs.persona_id, c.key)
        inside.append(d is not None and any(w.start - slack <= d < w.end + slack for w in wins))
    if not any(inside):
        return ["date_mode: no candidate inside the explicit window; order unchanged"]
    first = [c for c, ok in zip(pool, inside) if ok]
    rest = [c for c, ok in zip(pool, inside) if not ok]
    cs.candidates[: len(pool)] = first + ([] if cfg.date_mode == "filter" else rest)
    if cfg.date_mode == "filter":
        del cs.candidates[len(first):]
    return [f"date_mode {cfg.date_mode}: {len(first)} in-window candidates first"]


def _window_rows(conn, persona_id: str, mid_first: str, mid_last: str, before: int, after: int, at_utc: str):
    row = conn.execute("SELECT conversation_id, timestamp_utc FROM messages WHERE message_id = ?", (mid_first,)).fetchone()
    last = conn.execute("SELECT timestamp_utc FROM messages WHERE message_id = ?", (mid_last,)).fetchone()
    if not row:
        return None, [], []
    conv, t_first = row
    t_last = last[0] if last else t_first
    cols = "message_id, speaker_id, content, timestamp, timestamp_utc, source_type, source_id, recorded_at, locator, conversation_id"
    prev = conn.execute(f"SELECT {cols} FROM messages WHERE conversation_id = ? AND target_persona_id = ? AND timestamp_utc < ? "
                        "ORDER BY timestamp_utc DESC, message_id DESC LIMIT ?", (conv, persona_id, t_first, max(before, 0))).fetchall()
    nxt = conn.execute(f"SELECT {cols} FROM messages WHERE conversation_id = ? AND target_persona_id = ? AND timestamp_utc > ? "
                       "ORDER BY timestamp_utc, message_id LIMIT ?", (conv, persona_id, t_last, max(after, 0))).fetchall()
    keep = lambda r: r[4][:19] <= at_utc                                          # noqa: E731 - never a message after the ask time
    return conv, [r for r in reversed(prev) if keep(r)], [r for r in nxt if keep(r)]


def expand_pack(api, pack, persona_id: str, at: datetime, before: int, after: int, names: Dict[str, str]) -> int:
    """Add up to ``before`` / ``after`` neighbouring messages of the same conversation around every message item, each as its OWN
    context item (``kind="context"``, ``channels={"context_of": anchor rank}``, the anchor's rank): every line keeps its own speaker,
    message id, source id and time, so the attribution gate and citations stay exact (a merged item would carry only the anchor's
    speaker). Visible at ``at`` only; no message twice. Items are re-sorted by time and eids renumbered, as ``fact_pack.build`` does.
    Returns the number of added messages."""
    if before <= 0 and after <= 0:
        return 0
    from stillhere.context.fact_pack import PackItem
    from stillhere.memory.labels import SOURCE_LABELS, speaker_label
    conn = api.store._conn
    at_utc = at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    seen = {m for it in pack.items for m in (it.message_ids or [])}
    added: list = []
    for it in list(pack.items):
        if not it.message_ids or it.kind != "record":
            continue
        _, pre, post = _window_rows(conn, persona_id, it.message_ids[0], it.message_ids[-1], before, after, at_utc)
        for r in pre + post:
            if r[0] in seen:
                continue
            seen.add(r[0])
            label = speaker_label(persona_id, r[1], names)
            added.append(PackItem("", r[8] or r[0], it.rank, 0.0, r[5], SOURCE_LABELS.get(r[5], r[5]), r[6], r[3], r[7], [label],
                                  f"{label}：{r[2]}", [r[0]], {"context_of": it.rank}, [], list(it.in_windows), kind="context"))
    if not added:
        return 0
    pack.items.extend(added)
    pack.items.sort(key=lambda i: ((i.event_time or "9999"), i.rank))
    for n, i in enumerate(pack.items, 1):
        i.eid = f"E{n:02d}" if len(pack.items) < 100 else f"E{n:03d}"
    return len(added)


def pack_v2(api, persona_id: str, question: str, at: datetime, cfg: V2Config, *, dense=None, names: Optional[Dict[str, str]] = None):
    """EF1 (+ optional dense) -> date mode -> fact_pack -> evidence windows. ``pack.notes`` records what V2 did."""
    from stillhere.context import fact_pack as FP
    from stillhere.retrieval.ef1.pipeline import retrieve
    from . import memory_store as MS
    names = names or {persona_id: "本人", MS.other_id(persona_id): "对方"}
    ecfg = ef1_config(question, at, cfg, dense is not None)
    cs = retrieve(api, persona_id, question, at, ecfg, dense=dense if ecfg.use_dense else None)
    notes = apply_date_mode(api, cs, question, at, cfg)
    pk = FP.build(api, cs, names=names)
    added = expand_pack(api, pk, persona_id, at, cfg.neighbors[0], cfg.neighbors[1], names)
    pk.notes.extend(n for n in notes if n)
    return pk, {"ef1_weights": dict(ecfg.weights), "dense": bool(ecfg.use_dense), "date_notes": notes, "neighbors_added": added,
                "timings_ms": dict(cs.timings_ms)}


def retrieved_ids(pack) -> List[str]:
    return [m for it in pack.items for m in (it.message_ids or [])]


def ranked_unit_messages(pack) -> List[List[str]]:
    """Message ids per ANCHOR in rank order: an anchor item and its context items (same rank) form one group, so a k-anchor pack
    scores as k groups whatever the window size."""
    groups: Dict[int, List[str]] = {}
    for it in sorted(pack.items, key=lambda i: (i.rank, (i.event_time or ""))):
        groups.setdefault(it.rank, []).extend(it.message_ids or [])
    return [groups[r] for r in sorted(groups)]


def summary_flags(questions: Sequence[str], at: datetime) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for q in questions:
        r = route_v2(q, at)
        for w in r["why"]:
            out[w] = out.get(w, 0) + 1
    return out
