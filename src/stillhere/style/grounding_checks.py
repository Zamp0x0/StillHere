"""Shared grounding checks for T2-D2 data QC, package audits and the scorer: one definition of "a fact was kept" and "a fact was invented".

* ``key_term_present``: a key term is kept when it appears verbatim (whitespace ignored), or, for a term with numbers, when every number of
  the term appears in the reply as a whole number (``5000元`` is kept by ``5000块`` or ``5000``; ``50`` is NOT kept by ``500``). A text term
  (a name, a place) must appear verbatim.
* ``unsupported_specifics``: digit runs and Latin names in a reply that the facts and the conversation do not contain.
* ``UNCERTAIN`` / ``CORRECTION``: the markers used for unknown / conflicting facts and for mentioning an old (corrected) value.
"""
from __future__ import annotations

import re
from typing import List

UNCERTAIN = re.compile(r"不知道|不清楚|不确定|记不清|记不得|不记得|没记|没印象|说不准|说不好|不太清楚|没查|不晓得|唔清楚|唔知|唔记得|记岔|拿不准|不好说|没准|不一定|待确认|"
                       r"确认一下|再核实|再问问|问问|没底|两种说法|对不上")
CORRECTION = re.compile(r"记错|搞错|弄错|记岔|改|更正|纠正|之前以为|一开始|其实|不是|原来")
LATIN = re.compile(r"[A-Za-z][A-Za-z0-9+#.\-]{1,}")
LATIN_OK = {"ok", "app", "wifi", "hh", "hhh", "emm", "emmm", "lol", "get"}
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def norm(t: str) -> str:
    return re.sub(r"\s+", "", t or "")


def key_term_present(term: str, reply: str) -> bool:
    k, r = norm(term), norm(reply)
    if not k:
        return True
    if k in r:
        return True
    nums = NUMBER.findall(k)
    return bool(nums) and all(re.search(rf"(?<![\d.]){re.escape(n)}(?![\d.])", r) for n in nums)


def unsupported_specifics(reply: str, *sources: str) -> List[str]:
    hay = norm("\n".join(sources)).lower()
    flat = norm(reply)
    out = [m.group(0) for m in NUMBER.finditer(flat) if m.group(0) not in hay]
    out += [m.group(0) for m in LATIN.finditer(flat) if m.group(0).strip(".-").lower() not in LATIN_OK and m.group(0).strip(".-").lower() not in hay]
    return out
