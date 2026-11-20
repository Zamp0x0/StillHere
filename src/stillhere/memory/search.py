"""Keyword segmentation for memory search: CJK character bigrams plus whole non-CJK words.

This is lexical matching, not semantic search: a memory is found only when it shares characters or words
with the query. There are no embeddings and no model calls.

Why bigrams instead of the FTS5 ``trigram`` tokenizer: trigram needs SQLite 3.34+ and cannot match a query
shorter than three characters, while most useful Chinese keywords ("年糕", "北川", "香菜") are two characters.
Chinese has no spaces, so ``unicode61`` alone would index a whole sentence run as one token. The store
therefore segments text itself and hands FTS5 space-separated terms:

* index side: every CJK run becomes its single characters (``unigrams`` column) and its overlapping
  character pairs (``bigrams`` column); other words (Latin letters, digits) go whole into ``bigrams``.
* query side: CJK runs become bigrams (a one-character run stays a unigram), plus single characters that are
  not common function characters; other words stay whole. Units are OR-ed, so partial overlap still
  recalls a record, and ``bm25`` weights a bigram hit above a single-character hit.

Text is NFKC-normalised and case-folded on both sides, so full-width letters and digits match half-width.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Tuple

# Han ideographs (with extension A and compatibility forms), kana and Hangul: scripts written without spaces
# between words, segmented into characters and character pairs.
_CJK = "぀-ヿ㐀-䶿一-鿿가-힯豈-﫿"
_TOKEN = re.compile(rf"[{_CJK}]+|[^\W_{_CJK}]+")
_CJK_RUN = re.compile(rf"[{_CJK}]+")

# Single characters too common to be useful as a query unit on their own (they still count inside bigrams).
QUERY_STOP_CHARS = frozenset("的了是在和与及或吗呢吧啊呀么嘛哦呃这那哪谁什怎几个有没不也就都很还又把被给我你他她它们")

BIGRAM_WEIGHT = 1.0
UNIGRAM_WEIGHT = 0.25


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _runs(text: str) -> Iterable[Tuple[bool, str]]:
    for m in _TOKEN.finditer(normalize(text)):
        tok = m.group(0)
        yield bool(_CJK_RUN.fullmatch(tok)), tok


def _bigrams(run: str) -> List[str]:
    return [run[i:i + 2] for i in range(len(run) - 1)]


def _dedupe(items: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


@dataclass(frozen=True)
class IndexTerms:
    bigrams: Tuple[str, ...]   # CJK character pairs and whole non-CJK words
    unigrams: Tuple[str, ...]  # single CJK characters

    @property
    def bigram_text(self) -> str:
        return " ".join(self.bigrams)

    @property
    def unigram_text(self) -> str:
        return " ".join(self.unigrams)


def index_terms(*texts: str) -> IndexTerms:
    bigrams: List[str] = []
    unigrams: List[str] = []
    for text in texts:
        for is_cjk, tok in _runs(text):
            if is_cjk:
                unigrams.extend(tok)
                bigrams.extend(_bigrams(tok))
            else:
                bigrams.append(tok)
    return IndexTerms(tuple(bigrams), tuple(unigrams))


@dataclass(frozen=True)
class QueryUnits:
    bigrams: Tuple[str, ...]
    unigrams: Tuple[str, ...]

    def __bool__(self) -> bool:
        return bool(self.bigrams or self.unigrams)


def query_units(query: str) -> QueryUnits:
    bigrams: List[str] = []
    unigrams: List[str] = []
    for is_cjk, tok in _runs(query):
        if not is_cjk:
            bigrams.append(tok)
        elif len(tok) == 1:
            unigrams.append(tok)
        else:
            bigrams.extend(_bigrams(tok))
            unigrams.extend(c for c in tok if c not in QUERY_STOP_CHARS)
    return QueryUnits(tuple(_dedupe(bigrams)), tuple(_dedupe(unigrams)))


def _quote(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def match_expression(units: QueryUnits) -> str:
    """An FTS5 MATCH expression: any query unit in its column. Terms are quoted, so no query syntax leaks."""
    parts = []
    if units.bigrams:
        parts.append("bigrams : (" + " OR ".join(_quote(t) for t in units.bigrams) + ")")
    if units.unigrams:
        parts.append("unigrams : (" + " OR ".join(_quote(t) for t in units.unigrams) + ")")
    return " OR ".join(parts)


def overlap_score(units: QueryUnits, terms: IndexTerms) -> float:
    """Fallback score when FTS5 is unavailable: weighted count of distinct query units present (no IDF)."""
    bi, uni = set(terms.bigrams), set(terms.unigrams)
    return (BIGRAM_WEIGHT * sum(t in bi for t in units.bigrams)
            + UNIGRAM_WEIGHT * sum(t in uni for t in units.unigrams))
