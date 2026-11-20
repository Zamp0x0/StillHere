"""Token counting and the context budget.

**Counting.** The DeepSeek reader's tokenizer is not available locally, so the default count is ESTIMATED: one token per
character plus a 5% margin, rounded up. One per character is already an upper bound for Chinese (the frozen
``answer_pilot.estimate_tokens`` rule); the margin covers punctuation and mixed text. A caller with the reader's real tokenizer
injects it and the mode becomes EXACT. The mode is always reported; an estimate is never called exact.

**Budget.** ``input_token_budget`` covers everything sent. The caller's fixed scaffolding (``instruction_tokens``) and the
question are taken first and can never be dropped; if they alone do not fit, the result is BUDGET_INSUFFICIENT. What is left
is shared by the sections. Each section has a cap that depends on the question's profile; space a section does not use goes
to the long-term evidence:

==============  ======  ======  =======  ===============================================
profile         pinned  recent  working  when
==============  ======  ======  =======  ===============================================
factual         10%     15%     10%      default
temporal        10%     10%     10%      the question has a dated boundary: timelines need the room
conversational  10%     45%     10%      a current conversation and a reference back (刚才, 他, 那个, ...)
==============  ======  ======  =======  ===============================================

The latest whole turn of the recent conversation is guaranteed (it is never cut for evidence) as long as the total fits.
``output_token_reserve`` is reported with the budget; it is part of the model window, not of the input.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Optional

from .contracts import TOKEN_ESTIMATED, TOKEN_EXACT

ESTIMATE_MARGIN = 0.05
CAPS = {"factual": {"pinned": 0.10, "recent": 0.15, "working": 0.10},
        "temporal": {"pinned": 0.10, "recent": 0.10, "working": 0.10},
        "conversational": {"pinned": 0.10, "recent": 0.45, "working": 0.10}}
BACK_REFERENCES = ("刚才", "刚刚", "上面", "前面说", "你说的", "我说的", "那个", "这个", "那件事", "这件事", "他", "她", "它", "那边", "那里")


class TokenCounter:
    def __init__(self, count: Optional[Callable[[str], int]] = None, *, exact: bool = False) -> None:
        if exact and count is None:
            raise ValueError("an exact count needs the reader's tokenizer")
        self._count = count
        self.mode = TOKEN_EXACT if exact else TOKEN_ESTIMATED

    def __call__(self, text: str) -> int:
        if self.mode == TOKEN_EXACT:
            return int(self._count(text))
        base = self._count(text) if self._count else len(text)
        return int(math.ceil(base * (1 + ESTIMATE_MARGIN)))


@dataclass
class Budget:
    input_total: int
    output_reserve: int
    fixed: int                    # instructions + question
    available: int                # what the sections share
    profile: str
    caps: dict

    def cap(self, section: str) -> int:
        return int(self.available * self.caps.get(section, 0.0))

    def to_dict(self) -> dict:
        return asdict(self)


def profile_of(question: str, *, has_boundary: bool, has_conversation: bool) -> str:
    if has_conversation and any(w in question for w in BACK_REFERENCES):
        return "conversational"
    return "temporal" if has_boundary else "factual"


def plan(input_total: int, output_reserve: int, fixed: int, profile: str) -> Budget:
    return Budget(input_total, output_reserve, fixed, max(input_total - fixed, 0), profile, dict(CAPS[profile]))
