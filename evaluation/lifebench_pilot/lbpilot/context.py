"""One evidence format and one token budget for every memory system.

Every system returns ranked ``Entry`` objects (rank 1 = best, its own text, a date, a source label, the ids it came from). ``render``
keeps the best-ranked entries that fit ``budget`` tokens (dropping from the bottom of the RANKING) and prints the kept ones oldest
first. Mem0's memory text and StillHere's raw records keep their own form: only the frame and the budget are shared. The rendered
context, its token count, the kept / dropped counts and every source id are recorded for the audit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional, Sequence

EMPTY = "（没有检索到相关记忆）"
HEADER = "【记忆记录】（按时间先后排列；每条前面是记录日期和来源）"


@dataclass
class Entry:
    rank: int
    date: Optional[str]                 # YYYY-MM-DD or None
    source: str                         # e.g. 短信 / 日历 / mem0
    text: str
    ids: List[str] = field(default_factory=list)       # provenance ids (StillHere: dia_ids; Mem0: the memory id and its batch dia_ids)
    score: Optional[float] = None

    def line(self, n: int) -> str:
        body = "\n    ".join(x.strip() for x in self.text.splitlines() if x.strip())
        return f"[{n}] {self.date or '日期未知'}｜{self.source}｜{body}"


def estimate_tokens(text: str) -> int:
    """CPU fallback: about one token per CJK character, one per four other characters (an upper-side estimate)."""
    cjk = sum(1 for c in text if "㐀" <= c <= "鿿")
    return cjk + (len(text) - cjk + 3) // 4


def render(entries: Sequence[Entry], *, budget: int, count: Callable[[str], int] = estimate_tokens) -> dict:
    ranked = sorted(entries, key=lambda e: e.rank)
    kept: List[Entry] = []

    def text_of(es):
        es = sorted(es, key=lambda e: ((e.date or "9999"), e.rank))
        return (HEADER + "\n" + "\n".join(e.line(i + 1) for i, e in enumerate(es))) if es else EMPTY
    for e in ranked:
        trial = kept + [e]
        if count(text_of(trial)) > budget:
            break                                       # a ranking prefix: nothing ranked lower than a dropped entry is kept
        kept = trial
    text = text_of(kept)
    return {"text": text, "tokens": count(text), "entries_in": len(ranked), "entries_kept": len(kept), "entries_dropped": len(ranked) - len(kept),
            "kept": [asdict(e) for e in sorted(kept, key=lambda e: e.rank)], "dropped_ranks": [e.rank for e in ranked if e not in kept]}
