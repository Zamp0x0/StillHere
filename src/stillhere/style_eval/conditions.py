"""Prompt conditions for the Style Evaluation v2 groups. Every condition renders through the T2-D2 builder
(``stillhere.style.prompt_builder.build_style_conditioned_messages``, schema ``stillhere.style.prompt/2.0``); only the inputs differ.

Fixed before any DEV generation:

* ``same_style``     the episode's own profile + its 5 support examples: byte-identical to the T2-D2 training / test rendering.   (groups A, B)
* ``length_ctrl``    ``same_style`` + ONE extra generation rule asking for the length of the shown examples. The number comes from the
                     shown examples only (median char length, rounded up to a multiple of 5, at least 10), never from the target.  (group C)
* ``no_examples``    the profile, no examples.                                                                                     (group D)
* ``swap``           the donor author's profile + examples (another author of the same split, no shared topic).                    (group E)

Candidate conditions (Phase 7, DEV only, new experiment versions; they change the prompt, so they are NOT v2-compatible by construction and
carry their own version tag):

* ``ex_rep``         examples re-selected from the same pre-target pool: the k = 5 replies closest to the author's own pool centroid in the
                     char view (most typical of the author), still cross-topic and mutually dissimilar.   version ``sev2-cand-ex-rep-1``
* ``prof_lex``       ``same_style`` + one profile line listing up to 6 words / punctuation habits that are frequent in this author's pool and
                     rare in the split's pool overall (log-odds), each with its count of pool replies.         version ``sev2-cand-prof-lex-1``
"""
from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from typing import List, Mapping, Optional, Sequence

from stillhere.style import prompt_builder as pb
from stillhere.style.features import char_grams, jaccard

CONDITIONS = ("same_style", "length_ctrl", "no_examples", "swap", "ex_rep", "prof_lex")
CANDIDATE_VERSION = {"ex_rep": "sev2-cand-ex-rep-1", "prof_lex": "sev2-cand-prof-lex-1"}
LENGTH_RULE = "回复长度：和上面的示例差不多长，大约{n}个字以内，像平时随手回帖那样说，不要写成长段落。"
TOKEN = re.compile(r"[A-Za-z]{2,}|[一-鿿]{2}|[^\w\s]")


def _flat(bubbles: Sequence[str]) -> str:
    return bubbles[0] if len(bubbles) == 1 else " / ".join(bubbles)


def length_target(examples: Sequence[str]) -> int:
    m = statistics.median(len(re.sub(r"\s|/", "", e)) for e in examples) if examples else 20
    return max(10, int(math.ceil(m / 5.0) * 5))


def _char_vec(t: str) -> Counter:
    t = re.sub(r"\s", "", t)
    return Counter(t[i:i + n] for n in (1, 2) for i in range(len(t) - n + 1))


def _cos(a: Counter, b: Counter) -> float:
    num = sum(v * b.get(k, 0) for k, v in a.items())
    na, nb = math.sqrt(sum(v * v for v in a.values())), math.sqrt(sum(v * v for v in b.values()))
    return num / (na * nb) if na and nb else 0.0


def representative_examples(pool: Sequence[Mapping], target: str, k: int = 5) -> List[str]:
    """The k pool replies most typical of the author (char 1-2 gram cosine to the pool centroid), one per topic, mutually dissimilar,
    never a near duplicate of the target (the same exclusions as ``select_examples``)."""
    texts = [_flat(m["bubbles"]) for m in pool]
    cent = Counter()
    for t in texts:
        cent.update(_char_vec(t))
    tg = char_grams(target)
    ranked = sorted(range(len(pool)), key=lambda i: (-_cos(_char_vec(texts[i]), cent), pool[i]["message_id"]))
    out, topics = [], set()
    for i in ranked:
        t = texts[i]
        if pool[i]["item_id"] in topics or jaccard(char_grams(t), tg) >= 0.5 or any(jaccard(char_grams(t), char_grams(o)) >= 0.5 for o in out):
            continue
        out.append(t)
        topics.add(pool[i]["item_id"])
        if len(out) == k:
            break
    return out


def lexical_markers(pool: Sequence[Mapping], background: Counter, bg_docs: int, k: int = 6) -> List[str]:
    """Words / punctuation habits frequent in this author's pool and rare in the split's pool overall: smoothed log-odds on document
    frequency, at least 3 of the author's replies. Returns ``token（N 条）`` strings (counts are real counts, nothing inferred)."""
    docs = [set(TOKEN.findall(_flat(m["bubbles"]))) for m in pool]
    df = Counter(t for d in docs for t in d)
    n = len(docs)
    scored = []
    for t, c in df.items():
        if c < 3:
            continue
        p_a = (c + 0.5) / (n + 1)
        p_b = (background.get(t, 0) + 0.5) / (bg_docs + 1)
        scored.append((math.log(p_a / (1 - p_a)) - math.log(p_b / (1 - p_b)), t, c))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [f"“{t}”（{c} 条）" for s, t, c in scored[:k] if s > 1.0]


def background_counts(episodes: Sequence[Mapping]) -> tuple:
    seen, bg = set(), Counter()
    for e in episodes:
        for m in e["pool"]:
            if m["message_id"] in seen:
                continue
            seen.add(m["message_id"])
            bg.update(set(TOKEN.findall(_flat(m["bubbles"]))))
    return bg, len(seen)


def messages(ep: Mapping, condition: str, *, donor: Optional[Mapping] = None, conversation=None, grounded=None, background=None) -> List[dict]:
    conv = conversation if conversation is not None else ep["conversation"]
    g = grounded if grounded is not None else ep["grounded"]
    prof, ex = ep["style_profile"], ep["style_examples"]
    if condition == "same_style":
        return pb.build_style_conditioned_messages(conv, grounded_content=g, style_profile=prof, style_examples=ex)
    if condition == "length_ctrl":
        return pb.build_style_conditioned_messages(conv, grounded_content=g, style_profile=prof, style_examples=ex, constraints=[LENGTH_RULE.format(n=length_target(ex))])
    if condition == "no_examples":
        return pb.build_style_conditioned_messages(conv, grounded_content=g, style_profile=prof, style_examples=None)
    if condition == "swap":
        return pb.build_style_conditioned_messages(conv, grounded_content=g, style_profile=donor["style_profile"], style_examples=donor["style_examples"])
    if condition == "ex_rep":
        return pb.build_style_conditioned_messages(conv, grounded_content=g, style_profile=prof, style_examples=representative_examples(ep["pool"], ep["reference"] or ""))
    if condition == "prof_lex":
        bg, nd = background
        marks = lexical_markers(ep["pool"], bg, nd)
        prof2 = dict(prof)
        if marks:
            prof2["traits"] = list(prof["traits"]) + [{"trait": "lexical", "description": "此人比别人更常用：" + "、".join(marks), "type": "deterministic_statistic",
                                                      "evidence_ids": [], "evidence_count": len(marks), "of_messages": len(ep["pool"]), "stability": "stable"}]
        return pb.build_style_conditioned_messages(conv, grounded_content=g, style_profile=prof2, style_examples=ex)
    raise ValueError(condition)
