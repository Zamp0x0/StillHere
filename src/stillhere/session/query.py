"""Context-aware retrieval query (a retrieval aid, never a fact).

A follow-up ("酒店定了没？", "他什么时候搬？", "我不是说已经取消了吗？") names neither the trip, the person nor the cancelled thing, so memory
retrieval on the bare question misses them. The builder borrows the words of the last ``query_context_user_turns`` USER messages of the
recent window (assistant messages are never used: generated text must not steer what memory is asked) and returns:

* ``retrieval_query``: those words + the question, for B2 / M3.1 and grounding relevance;
* ``reader_question``: the question with the earlier user words marked as context ("（前文：…）…"), for the frozen reader (its prompt and
  rules are unchanged; only the question text it is given carries the context);
* the original question, unchanged, and the ids of the messages borrowed from (trace).

The memory reader stays the only judge of facts, time, status, conflicts and answerability.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Sequence

from stillhere.context.budget import BACK_REFERENCES

from .config import DEFAULT, ContextConfig

FOLLOWUP_CUES = tuple(BACK_REFERENCES) + ("呢", "那家", "那次", "那件", "这次", "还", "也", "不是说", "不是说了", "取消了吗", "怎么样了", "定了没", "定了吗")
NAMED = re.compile(r"(小|老|阿)[一-鿿]|[一-鿿]{2,}(市|省|区|县|公司|大学|医院|学校)")


@dataclass
class CompletedQuery:
    original_question: str
    retrieval_query: str
    reader_question: str
    completed: bool
    borrowed_message_ids: List[str] = field(default_factory=list)
    reason: str = ""


def is_followup(question: str, cfg: ContextConfig = DEFAULT) -> bool:
    q = question.strip()
    if any(c in q for c in FOLLOWUP_CUES):
        return True
    core = re.sub(r"[？?！!。，,\s]", "", q)
    return len(core) <= cfg.followup_max_chars and not NAMED.search(core)


def complete(question: str, window_messages: Sequence, cfg: ContextConfig = DEFAULT) -> CompletedQuery:
    """``window_messages``: the recent window, oldest first (objects with ``role``, ``text``, ``message_id``)."""
    users = [m for m in window_messages if m.role == "user"][-cfg.query_context_user_turns:]
    if not users or not is_followup(question, cfg):
        return CompletedQuery(question, question, question, False, [], "no follow-up cue or no earlier user message")
    ctx = " ".join(re.sub(r"\s+", " ", m.text.splitlines()[-1] if m.text.startswith("〔长消息") else m.text).strip() for m in users)
    return CompletedQuery(question, f"{ctx} {question}", f"（前文：{ctx}）{question}", True, [m.message_id for m in users], "follow-up")
