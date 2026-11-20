"""Deterministic chunking of a long message (headings, then paragraphs, then sentences), with the original offsets kept, and the
selection of the chunks a question refers to ("第三部分", "三、…", a heading word, or shared words). Nothing is ever dropped silently: the
caller learns which chunks are shown and how many exist."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from stillhere.memory.search import index_terms, overlap_score, query_units

CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
HEADING = re.compile(r"^\s*(#{1,6}\s*\S.*|第[一二三四五六七八九十\d]+(?:部分|章|节|条|点|段)\S*.*|[一二三四五六七八九十]+、\S.*|\d+[.、]\s*\S.*|【[^】]{1,20}】.*)$")
SECTION_REF = re.compile(r"第([一二三四五六七八九十\d]+)(?:部分|章|节|条|点|段)")
SENT_END = re.compile(r"(?<=[。！？!?；;\n])")


@dataclass(frozen=True)
class Chunk:
    index: int
    offset: int
    heading: Optional[str]
    text: str


def _num(s: str) -> Optional[int]:
    if s.isdigit():
        return int(s)
    if s == "十":
        return 10
    if s.startswith("十"):
        return 10 + CN_NUM.get(s[1:], 0)
    if s.endswith("十"):
        return CN_NUM.get(s[0], 0) * 10
    if len(s) == 2 and s[1] in CN_NUM and s[0] in CN_NUM:
        return CN_NUM[s[0]] * 10 + CN_NUM[s[1]]
    return CN_NUM.get(s)


def section_number(heading: Optional[str]) -> Optional[int]:
    if not heading:
        return None
    h = heading.strip().lstrip("#").strip()
    m = re.match(r"第([一二三四五六七八九十\d]+)", h) or re.match(r"([一二三四五六七八九十]+)、", h) or re.match(r"(\d+)[.、]", h)
    return _num(m.group(1)) if m else None


def split(text: str, count: Callable[[str], int], max_tokens: int) -> List[Chunk]:
    """Sections at headings; a section above ``max_tokens`` is cut at paragraph, then sentence boundaries. Offsets index ``text``."""
    lines = text.splitlines(keepends=True)
    sections: List[Tuple[int, Optional[str], str]] = []
    pos, start, head, buf = 0, 0, None, []
    for ln in lines:
        if HEADING.match(ln.rstrip("\n")) and buf:
            sections.append((start, head, "".join(buf)))
            start, buf = pos, []
        if HEADING.match(ln.rstrip("\n")):
            head = ln.strip()
        buf.append(ln)
        pos += len(ln)
    if buf:
        sections.append((start, head, "".join(buf)))
    out: List[Chunk] = []
    for off, h, body in sections:
        if count(body) <= max_tokens:
            out.append(Chunk(len(out), off, h, body))
            continue
        piece_off, piece = off, ""
        for sent in [s for s in SENT_END.split(body) if s]:
            if piece and count(piece + sent) > max_tokens:
                out.append(Chunk(len(out), piece_off, h, piece))
                piece_off, piece = piece_off + len(piece), ""
            piece += sent
        if piece:
            out.append(Chunk(len(out), piece_off, h, piece))
    return out


def select(chunks: Sequence[Chunk], question: str, max_chunks: int) -> Tuple[List[int], str]:
    """The chunk indices to show for ``question`` (matched ones and their neighbours, in order) and a status:
    SECTION (an explicit section reference matched), MATCH (shared words), AMBIGUOUS_CONTEXT (a section reference that matches no heading),
    CONTEXT_INSUFFICIENT (nothing in the message relates to the question)."""
    ref = SECTION_REF.search(question)
    if ref:
        n = _num(ref.group(1))
        hit = [c.index for c in chunks if section_number(c.heading) == n]
        if not hit:
            return [], "AMBIGUOUS_CONTEXT"
        first = hit[0]
        same = [i for i in hit][:max_chunks]                    # a long section spans several chunks: its own first
        extra = [i for i in (first - 1, same[-1] + 1) if 0 <= i < len(chunks) and i not in same]
        return sorted(same + extra)[:max_chunks + 1], "SECTION"
    units = query_units(question)
    scored = sorted(((overlap_score(units, index_terms((c.heading or "") + " " + c.text)), c.index) for c in chunks), key=lambda x: (-x[0], x[1]))
    if not scored or scored[0][0] <= 0:
        return [], "CONTEXT_INSUFFICIENT"
    best = scored[0][1]
    picked = [i for i in (best, best - 1, best + 1) if 0 <= i < len(chunks)]
    return sorted(set(picked))[:max_chunks], "MATCH"
