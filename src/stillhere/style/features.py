"""Deterministic, content-agnostic style features of chat messages (Chinese first, code-mixed and Cantonese markers included).

A message is a list of bubbles. Every feature here is about HOW something is said: length and splitting, punctuation, symbols,
particles, register markers, discourse devices. Content words are deliberately not features, so a person who often talks about
games is not "styled" by the word 游戏. ``content_profile`` gives the opposite view (content character bigrams with every style
marker removed) and exists only to measure topic confounding.

Nothing here is a calibrated probability: values are rates per message, per bubble or per 100 characters.
"""
from __future__ import annotations

import math
import re
import statistics
from typing import Dict, List, Mapping, Sequence

EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿⬀-⯿]")
KAOMOJI = re.compile(r"[\(（][^()（）一-鿿]{1,12}[\)）]|[ﾉ٩و๑ᗜωﾟ∀≧≦´`•̀́ᴗ･ˊˋ]")
LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z']+")
LAUGH = re.compile(r"哈{2,}|呵{2,}|嘿{2,}|hhh+|233+")

MARKERS: Dict[str, Sequence[str]] = {
    "particle_ya": ("呀",), "particle_ne": ("呢",), "particle_ba": ("吧",), "particle_la": ("啦",), "particle_ma": ("嘛",), "particle_o": ("哦", "喔"),
    "particle_a": ("啊",), "particle_en": ("嗯",), "particle_bei": ("呗",),
    "formal": ("您", "请您", "请查收", "麻烦您", "特此", "诚邀", "尚未", "已于", "敬请", "查收", "需要说明的是"),
    "hedge": ("可能", "也许", "好像", "或许", "大概", "不太确定", "说不好", "要不", "应该", "不知道"),
    "enumeration": ("第一", "第二", "首先", "其次", "一是", "二是", "三件事", "两件事", "1.", "2.", "分三步"),
    "rhetorical": ("难道", "不会吧", "你确定", "还用问", "就这", "惊不惊喜", "不是，", "是不是"),
    "care": ("记得", "别忘", "注意", "身体", "别着急", "好好休息", "别淋着"),
    "slang": ("笑死", "绝了", "离谱", "yyds", "躺平", "人麻了", "会谢", "冲冲冲", "原地去世", "上头", "破防", "绝绝子"),
    "cantonese": ("咗", "嘅", "唔", "喺", "嚟", "啲", "咁", "冇", "攞", "嗰", "仲", "嘢", "睇", "琴日", "听日"),
    "northern": ("咋整", "咋了", "咋办", "咋样", "咋回事", "咋说", "整挺好", "贼", "嘎嘎", "俩", "瞅", "老大了", "啥玩意", "得劲"),
    "metaphor": ("就像", "像是", "像一", "像个", "像极了", "仿佛", "如同", "似的"),
    "parenthesis": ("（", "("),
    "joke": ("开玩笑", "说人话", "哈哈", "梗"),
    "second_formal": ("您",),
}
FEATURE_NAMES: List[str] = []


def _per100(count: int, chars: int) -> float:
    return 100.0 * count / chars if chars else 0.0


def message_features(bubbles: Sequence[str]) -> Dict[str, float]:
    text = "".join(bubbles)
    n = max(len(text), 1)
    f: Dict[str, float] = {
        "bubbles": float(len(bubbles)),
        "chars_per_bubble": statistics.fmean(len(b) for b in bubbles) if bubbles else 0.0,
        "log_chars": math.log1p(len(text)),
        "comma": _per100(text.count("，") + text.count(","), n),
        "period": _per100(text.count("。"), n),
        "exclaim": _per100(text.count("！") + text.count("!"), n),
        "question": _per100(text.count("？") + text.count("?"), n),
        "ellipsis": _per100(text.count("…") + text.count("..."), n),
        "tilde": _per100(text.count("～") + text.count("~"), n),
        "space_as_pause": _per100(sum(1 for b in bubbles for _ in re.finditer(r"[一-鿿] [一-鿿]", b)), n),
        "emoji": _per100(len(EMOJI.findall(text)), n),
        "kaomoji": _per100(len(KAOMOJI.findall(text)), n),
        "latin_words": _per100(len(LATIN_WORD.findall(text)), n),
        "digits": _per100(sum(ch.isdigit() for ch in text), n),
        "laugh": _per100(len(LAUGH.findall(text)), n),
        "no_end_punct": sum(1 for b in bubbles if b and b[-1] not in "。！？…～!?~）)」") / max(len(bubbles), 1),
        "double_exclaim": _per100(len(re.findall(r"[！!]{2,}", text)), n),
    }
    for name, words in MARKERS.items():
        f[name] = _per100(sum(text.count(w) for w in words), n)
    return f


FEATURE_NAMES.extend(message_features(["占位"]).keys())


def aggregate(messages: Sequence[Sequence[str]]) -> Dict[str, float]:
    rows = [message_features(m) for m in messages]
    return {k: statistics.fmean(r[k] for r in rows) for k in FEATURE_NAMES} if rows else {k: 0.0 for k in FEATURE_NAMES}


STYLE_STRIP = re.compile(r"[\s\W\d_A-Za-z]+")
STOP_CHARS = set("的了是我你他她它们在有就也都还很和吗吧呢呀啦嘛哦啊嗯呗哈这那个一不没要会能去来说看想好得着过把被让给")


def content_profile(bubbles: Sequence[str]) -> Dict[str, int]:
    """Content character bigrams with punctuation, latin, digits, particles and every style marker removed: a topic view, for confounding checks only."""
    text = "".join(bubbles)
    for words in MARKERS.values():
        for w in words:
            text = text.replace(w, " ")
    text = STYLE_STRIP.sub(" ", text)
    out: Dict[str, int] = {}
    for chunk in text.split():
        chunk = "".join(ch for ch in chunk if ch not in STOP_CHARS)
        for i in range(len(chunk) - 1):
            out[chunk[i:i + 2]] = out.get(chunk[i:i + 2], 0) + 1
    return out


def char_grams(text: str, n: int = 3) -> set:
    t = re.sub(r"\s+", "", text)
    return {t[i:i + n] for i in range(max(len(t) - n + 1, 1))} if t else set()


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def standardise(vectors: Mapping[str, Dict[str, float]], basis: Sequence[Dict[str, float]]) -> Dict[str, List[float]]:
    """z-scores with means and spreads taken from ``basis`` only (the training personas), never from the people being tested."""
    mu = {k: statistics.fmean(v[k] for v in basis) for k in FEATURE_NAMES}
    sd = {k: (statistics.pstdev([v[k] for v in basis]) or 1.0) for k in FEATURE_NAMES}
    return {key: [(v[k] - mu[k]) / sd[k] for k in FEATURE_NAMES] for key, v in vectors.items()}


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0
