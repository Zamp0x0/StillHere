"""L1: the token-aware recent window of one session.

* Turns: a user message and the assistant replies to it; the current (unanswered) user message is NOT part of the window (it is the
  question, kept apart and never cut).
* At most ``recent_turn_limit`` whole turns, within ``recent_context_token_budget``; a turn is never split.
* Over budget: the oldest turn that shares no words with the question goes first, then the oldest turn; the newest turn stays.
* A long message (stored chunked) is shown by excerpt: the chunks the question refers to and their neighbours, under an explicit marker
  that says how many parts exist and that the rest is stored and not shown. Status flags tell the caller when the reference is ambiguous
  or nothing in the long message relates to the question.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

from stillhere.memory.search import index_terms, overlap_score, query_units

import stillhere.session.chunking as chunking
from .config import DEFAULT, ContextConfig
from .store import LiveMessage, SessionStore


@dataclass
class WindowMessage:
    message_id: str
    seq: int
    role: str
    text: str                         # what the prompt shows (the whole message, or the marked excerpt of a long one)
    generated: bool
    excerpt_of: int = 0               # total chunks when ``text`` is an excerpt
    shown_chunks: List[int] = field(default_factory=list)


@dataclass
class Window:
    turns: List[List[WindowMessage]]
    tokens: int
    token_mode: str
    dropped_turns: List[List[str]]    # message ids of turns left out (oldest / unrelated first)
    status: List[str]                 # LONG_MESSAGE_EXCERPT, AMBIGUOUS_CONTEXT, CONTEXT_INSUFFICIENT, RECENT_TRIMMED
    older_exists: bool                # messages earlier than the window exist in the session

    @property
    def messages(self) -> List[WindowMessage]:
        return [m for t in self.turns for m in t]

    @property
    def message_ids(self) -> List[str]:
        return [m.message_id for m in self.messages]


def turns_of(messages: Sequence[LiveMessage]) -> List[List[LiveMessage]]:
    out: List[List[LiveMessage]] = []
    for m in messages:
        if m.role == "user" or not out:
            out.append([m])
        else:
            out[-1].append(m)
    return out


def excerpt(store: SessionStore, m: LiveMessage, question: str, cfg: ContextConfig):
    chunks = store.chunks(m.message_id)
    picked, status = chunking.select(chunks, question, cfg.long_message_excerpt_chunks)
    if not picked:                                    # nothing located: the opening part only, clearly marked
        picked = [0]
    shown = "\n".join(chunks[i].text.rstrip("\n") for i in picked)
    parts = "、".join(str(i + 1) for i in picked)
    text = f"〔长消息，共{len(chunks)}段；这里只显示第{parts}段，其余已保存但未显示〕\n{shown}"
    return text, picked, len(chunks), status


def build(store: SessionStore, persona_id: str, session_id: str, question: str, current_message_id: str,
          count: Callable[[str], int], cfg: ContextConfig = DEFAULT, token_mode: str = "ESTIMATED") -> Window:
    msgs = [m for m in store.messages(persona_id, session_id) if m.message_id != current_message_id]
    all_turns = turns_of(msgs)
    cand = all_turns[-cfg.recent_turn_limit:] if cfg.recent_turn_limit > 0 else []
    status: List[str] = []
    rendered: List[List[WindowMessage]] = []
    for t in cand:
        row = []
        for m in t:
            if m.chunked:
                text, picked, total, st = excerpt(store, m, question, cfg)
                status += ["LONG_MESSAGE_EXCERPT"] + ([st] if st in ("AMBIGUOUS_CONTEXT", "CONTEXT_INSUFFICIENT") else [])
                row.append(WindowMessage(m.message_id, m.seq, m.role, text, m.generated, total, picked))
            else:
                row.append(WindowMessage(m.message_id, m.seq, m.role, m.content, m.generated))
        rendered.append(row)
    units = query_units(question)
    rel = [overlap_score(units, index_terms(*[x.text for x in t])) for t in rendered]
    cost = [sum(count(x.text) for x in t) for t in rendered]
    keep = list(range(len(rendered)))
    dropped: List[List[str]] = []
    while keep and sum(cost[i] for i in keep) > cfg.recent_context_token_budget and len(keep) > 1:
        unrelated = [i for i in keep[:-1] if rel[i] <= 0]
        drop = unrelated[0] if unrelated else keep[0]          # oldest unrelated turn first, then the oldest turn; never the newest
        keep.remove(drop)
        dropped.append([x.message_id for x in rendered[drop]])
    if dropped:
        status.append("RECENT_TRIMMED")
    turns = [rendered[i] for i in keep]
    return Window(turns, sum(cost[i] for i in keep), token_mode, dropped, sorted(set(status), key=status.index),
                  older_exists=len(all_turns) > len(cand) or bool(dropped))
