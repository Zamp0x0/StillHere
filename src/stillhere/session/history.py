"""L2: on-demand search of the EARLIER messages of the same session (never another session, never another persona).

Search engine: the store's existing message search (FTS5 over Python-made character bigrams / unigrams when the build has FTS5; a pure
Python bigram overlap otherwise), restricted to this ``conversation_id``. Plain trigram FTS is NOT used for Chinese: it cannot match a
two-character word ("深圳"). Long messages are also searched chunk by chunk. Results: at most ``history_search_top_k`` excerpts with
message id, role, time, relevance, source and a GENERATED label for assistant text; the neighbouring message of the same turn is attached.
NOT_FOUND / AMBIGUOUS are results, not errors: a missing excerpt is never invented.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from stillhere.memory.search import index_terms, overlap_score, query_units

from .config import DEFAULT, ContextConfig
from .store import SessionStore


def _turns(messages):
    from .window import turns_of
    return turns_of(messages)

# a match must share at least one CONTENT bigram with the question: bigrams made of function characters ("什么", "之前", "说的") match almost
# any message and would turn "I can't find it" into a false FOUND
FUNCTION_CHARS = set("的了吗呢吧啊呀哦嘛是我你他她它们这那么什之前说过记得还想问来着个一有没在和就都也要会能吗么怎样哪谁啥")


def content_bigrams(text: str) -> set:
    t = "".join(ch for ch in text if "一" <= ch <= "鿿" or ch.isalnum())
    return {t[i:i + 2] for i in range(len(t) - 1) if not (t[i] in FUNCTION_CHARS or t[i + 1] in FUNCTION_CHARS)}


HISTORY_CUES = ("之前", "以前说", "上次", "刚才", "刚刚", "前面", "早些时候", "一开始", "还记得", "你说过", "我说过", "你说的", "我说的", "第一个", "第二个",
                "第三个", "那个方案", "说过")


@dataclass
class Excerpt:
    message_id: str
    seq: int
    role: str
    timestamp: str
    relevance: float
    source: str
    text: str
    generated: bool
    chunk_index: Optional[int] = None
    neighbour_ids: List[str] = field(default_factory=list)


@dataclass
class HistoryResult:
    status: str                   # NOT_NEEDED, FOUND, NOT_FOUND, AMBIGUOUS
    excerpts: List[Excerpt]
    engine: str                   # fts5_bigram | python_bigram | none
    reason: str = ""


def needed(question: str, window_ids: Sequence[str], window_text: str, older_exists: bool) -> Optional[str]:
    if any(c in question for c in HISTORY_CUES):
        return "explicit reference to earlier conversation"
    if not older_exists:
        return None
    units = query_units(question)
    if units and overlap_score(units, index_terms(window_text)) <= 0:
        return "question shares no words with the recent window and earlier messages exist"
    return None


def search(store: SessionStore, persona_id: str, session_id: str, question: str, query: str, exclude_ids: Sequence[str],
           window_text: str, older_exists: bool, cfg: ContextConfig = DEFAULT, force_python: bool = False, at=None) -> HistoryResult:
    """``at``: the asking time (live messages are bitemporal rows: said and recorded at their send time; the store's own clock may be
    earlier, so the search must be anchored on the asking time, never the store default)."""
    why = needed(question, exclude_ids, window_text, older_exists)
    if why is None:
        return HistoryResult("NOT_NEEDED", [], "none")
    msgs = {m.message_id: m for m in store.messages(persona_id, session_id)}
    excl = set(exclude_ids)
    units = query_units(query)
    if not units:
        return HistoryResult("NOT_FOUND", [], "none", why)
    st = store.api.store
    engine = "fts5_bigram" if (st.message_fts_enabled and not force_python) else "python_bigram"
    scored = {}
    if engine == "fts5_bigram":
        for m, score in st.search_messages(persona_id, query, as_of=at, known_at=at, limit=200, conversation_id=session_id):
            if m.message_id in msgs and m.message_id not in excl:
                scored[m.message_id] = (float(score), None)
    else:
        for mid, m in msgs.items():
            if mid not in excl:
                s = overlap_score(units, index_terms(m.content))
                if s > 0:
                    scored[mid] = (s, None)
    for mid, m in msgs.items():                               # long messages: the best chunk, so the excerpt is the relevant part
        if m.chunked and mid not in excl:
            best = max(((overlap_score(units, index_terms((c.heading or "") + " " + c.text)), c) for c in store.chunks(mid)), key=lambda x: x[0],
                       default=(0, None))
            if best[0] > 0:
                scored[mid] = (max(best[0], scored.get(mid, (0, None))[0]), best[1])
    need = content_bigrams(query)
    def content_hits(mid):
        ch = scored[mid][1]
        return len(need & content_bigrams(ch.text if ch is not None else msgs[mid].content))
    scored = {mid: v for mid, v in scored.items() if content_hits(mid) > 0}
    if not scored:
        return HistoryResult("NOT_FOUND", [], engine, why)
    # best first: content words shared, then the engine score, then what the USER said before generated text, then the newer message
    ranked = sorted(scored.items(), key=lambda kv: (-content_hits(kv[0]), msgs[kv[0]].role != "user", -kv[1][0], -msgs[kv[0]].seq))
    turn = {}
    for i, t in enumerate(_turns(sorted(msgs.values(), key=lambda m: m.seq))):
        for m in t:
            turn[m.message_id] = i
    status = "FOUND"
    if (len(ranked) > 1 and content_hits(ranked[0][0]) == content_hits(ranked[1][0]) and turn[ranked[0][0]] != turn[ranked[1][0]]
            and any(c in question for c in ("那个", "那件", "这个"))):
        status = "AMBIGUOUS"                                   # two different turns fit a demonstrative equally well: do not pick one
    out: List[Excerpt] = []
    by_seq = {m.seq: m for m in msgs.values()}
    for mid, (score, ch) in ranked[:cfg.history_search_top_k]:
        m = msgs[mid]
        text = (f"〔长消息第{ch.index + 1}段〕{ch.text.strip()}" if ch is not None else m.content)
        nb = by_seq.get(m.seq + 1 if m.role == "user" else m.seq - 1)
        neigh = [nb.message_id] if nb is not None and nb.message_id not in excl and nb.message_id not in scored else []
        out.append(Excerpt(mid, m.seq, m.role, m.created_at, round(score, 4), m.source, text, m.generated, ch.index if ch else None, neigh))
    return HistoryResult(status, out, engine, why)
