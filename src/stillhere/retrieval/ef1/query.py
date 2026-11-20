"""Query preprocessing: separate WHEN a question is asked from WHAT time it asks about.

Benchmark questions (LifeBench 2.0) carry the ask date as a prefix, ``（提问时间：2025-12-15）``. That date is the moment of asking, never the
date of the asked event: it must not reach a date parser that selects evidence (Phase 1: the prefix decided the date window of 542/566
answerable questions). ``prepare`` returns the body without the prefix; the prefix date is only checked against the ask time the caller
passes (a mismatch is a warning, the caller's ask time wins). A question without a prefix is returned unchanged.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional

# full-width or half-width brackets, 提问时间 / 提问日期 / 当前时间 / ask time, ISO or 年月日 date, optional time of day
_PREFIX = re.compile(
    r"^\s*[（(]\s*(?:提问时间|提问日期|当前时间|ask(?:ed)?[ _]?(?:time|date)|question[ _]date)\s*[：:]\s*"
    r"(?P<y>\d{4})\s*(?:[-/.]|年)\s*(?P<m>\d{1,2})\s*(?:[-/.]|月)\s*(?P<d>\d{1,2})\s*日?"
    r"(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?\s*[）)]\s*", re.IGNORECASE)


@dataclass(frozen=True)
class PreparedQuery:
    original: str
    body: str                       # the question without the ask-time prefix (event dates in the body are kept verbatim)
    prefix_date: Optional[date]     # the date in the prefix, if any
    ask_time: Optional[datetime]    # the caller's ask time (authoritative)
    warnings: List[str] = field(default_factory=list)

    @property
    def ask_date(self) -> Optional[date]:
        return self.ask_time.date() if self.ask_time is not None else self.prefix_date


def normalize(text: str) -> str:
    """NFKC: full-width digits, letters and brackets -> half-width (used for matching only; returned text is never normalised)."""
    return unicodedata.normalize("NFKC", text or "")


def strip_ask_prefix(text: str):
    """(body, prefix_date or None). Only a LEADING ask-time prefix is removed; nothing inside the question is touched."""
    s = text or ""
    m = _PREFIX.match(normalize(s))
    if not m:
        return s.strip(), None
    try:
        d = date(int(m.group("y")), int(m.group("m")), int(m.group("d")))
    except ValueError:
        return s.strip(), None
    # cut the same number of characters from the ORIGINAL: NFKC maps each prefix character (brackets, digits, CJK, colon, space) to
    # exactly one character, so the offsets agree; the body keeps its own punctuation
    if len(normalize(s[:m.end()])) != m.end():
        return s.strip(), None
    return s[m.end():].strip(), d


# The FORM of a question (interrogatives, the recall frame 还记得 / 来着, sentence-final particles), never its content. In the lexical
# channels these characters form bigrams (什么 / 么时 / 时候 / 了吗 ...) that match thousands of chat lines and push the content words
# down; P0 DEV audit 2026-11-18: gold in the msg_bm25 top 8 for 17 -> 20 of 37 fact questions when they are removed. Used only when
# ``EF1Config.strip_question_form`` is on, and only for the lexical channels (dense keeps the full question).
QUESTION_FORM = re.compile(r"还记得|记不记得|记得吗|来着|什么时候|什么|怎么样|怎么|为什么|是不是|有没有|多少|哪里|哪儿|哪个|吗|呢|吧|呀|啊")


def lexical_body(body: str) -> str:
    """The body with its question form replaced by spaces (so no bigram spans a removed word); the body itself if nothing is left."""
    s = QUESTION_FORM.sub(" ", body or "")
    return s if re.search(r"[^\W_]", s) else (body or "")


def prepare(text: str, ask_time: Optional[datetime] = None) -> PreparedQuery:
    body, d = strip_ask_prefix(text)
    warnings: List[str] = []
    if d is not None and ask_time is not None and ask_time.date() != d:
        warnings.append(f"ASK_PREFIX_MISMATCH prefix={d.isoformat()} ask_time={ask_time.date().isoformat()} (ask_time used)")
    if d is None and ask_time is None:
        warnings.append("NO_ASK_TIME")
    return PreparedQuery(original=text or "", body=body, prefix_date=d, ask_time=ask_time, warnings=warnings)
