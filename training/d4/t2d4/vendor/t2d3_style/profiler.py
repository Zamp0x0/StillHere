"""Style Profiler V1: a person's own authorised messages in, a structured, evidence-backed style profile out. Deterministic.

The profile is built from the messages it is given and nothing else: the episode builder passes the history that precedes the
target, never the target, and the profile records exactly which message ids it read (``source_ids``), so its provenance can be
checked by recomputing it.

Each trait has a type:

* ``deterministic_statistic``: a measured rate (length, splitting, punctuation, symbols, digits);
* ``rule_induced``: a lexicon family seen in at least two messages (particles, hedges, enumeration, register markers, dialect
  markers, code-mixing, laughter);
* ``unconfirmed_hypothesis``: seen once only; kept in the profile for transparency, never rendered into a prompt.

``evidence_count`` is how many messages show the trait. It is not a probability and is never presented as one.

A trait seen in messages of two or more different scenario types is ``stable``; one seen in a single type is
``context_dependent`` (it may belong to the situation, not to the person). Content words are never traits.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from typing import Dict, List, Mapping, Optional, Sequence

from .features import aggregate, char_grams, jaccard, message_features

PROFILER_VERSION = "style-profiler-v1"

# (trait key, Chinese description, feature, threshold per message, kind)
PER_MESSAGE_RULES = [
    ("exclaim", "常用感叹号", "exclaim", 2.0, "deterministic_statistic"),
    ("double_exclaim", "常用连续的感叹号", "double_exclaim", 0.5, "deterministic_statistic"),
    ("question", "常用问句或反问", "question", 2.0, "deterministic_statistic"),
    ("ellipsis", "常用省略号，语气留白", "ellipsis", 1.5, "deterministic_statistic"),
    ("tilde", "常用波浪号", "tilde", 1.0, "deterministic_statistic"),
    ("emoji", "常用表情符号", "emoji", 0.5, "deterministic_statistic"),
    ("kaomoji", "常用颜文字", "kaomoji", 1.0, "deterministic_statistic"),
    ("latin", "中文里夹英文词", "latin_words", 1.0, "deterministic_statistic"),
    ("digits", "喜欢写具体的数字和时间", "digits", 4.0, "deterministic_statistic"),
    ("laugh", "常用拉长的笑声（哈哈哈）", "laugh", 0.5, "rule_induced"),
    ("space_pause", "用空格代替逗号断句", "space_as_pause", 1.0, "deterministic_statistic"),
    ("particles_soft", "常用语气词“呀、呢”", None, None, "rule_induced"),
    ("formal", "用词正式，礼貌用语多", "formal", 1.0, "rule_induced"),
    ("hedge", "说话委婉，常用“可能、要不、好像”", "hedge", 2.0, "rule_induced"),
    ("enumeration", "条理分明，常分点说明", "enumeration", 1.0, "rule_induced"),
    ("rhetorical", "爱用反问和揶揄", "rhetorical", 1.0, "rule_induced"),
    ("care", "常叮嘱和关心对方", "care", 1.0, "rule_induced"),
    ("slang", "常用网络流行语", "slang", 1.0, "rule_induced"),
    ("cantonese", "用粤语口语", "cantonese", 3.0, "rule_induced"),
    ("northern", "用北方口语", "northern", 2.0, "rule_induced"),
    ("parenthesis", "常用括号补充说明", "parenthesis", 1.0, "rule_induced"),
    ("metaphor", "常用比喻，带画面感", "metaphor", 1.0, "rule_induced"),
    ("joke", "爱开玩笑、玩梗", "joke", 1.0, "rule_induced"),
]


def _has(rule, f: Mapping[str, float]) -> bool:
    key, _, feature, threshold, _ = rule
    if key == "particles_soft":
        return f["particle_ya"] + f["particle_ne"] >= 2.0
    return f[feature] >= threshold


def profile(history: Sequence[Mapping], *, scenario_of: Optional[Mapping[str, str]] = None) -> dict:
    """``history``: [{"message_id", "bubbles", "scenario_type"}], the person's own messages strictly before the target."""
    if not history:
        return {"profiler_version": PROFILER_VERSION, "source_ids": [], "message_count": 0, "traits": [], "stats": {}, "notes": ["no history: no profile"]}
    feats = {m["message_id"]: message_features(m["bubbles"]) for m in history}
    stype = {m["message_id"]: m.get("scenario_type", "unknown") for m in history}
    agg = aggregate([m["bubbles"] for m in history])
    traits = []
    for rule in PER_MESSAGE_RULES:
        ids = sorted(mid for mid, f in feats.items() if _has(rule, f))
        if not ids:
            continue
        kinds = {stype[i] for i in ids}
        kind = "unconfirmed_hypothesis" if len(ids) < 2 else rule[4]
        traits.append({"trait": rule[0], "description": rule[1], "type": kind, "evidence_ids": ids, "evidence_count": len(ids), "of_messages": len(history),
                       "stability": "stable" if len(kinds) >= 2 else "context_dependent", "scenario_types": sorted(kinds)})
    bubbles = [len(m["bubbles"]) for m in history]
    per_bubble = [len(b) for m in history for b in m["bubbles"]]
    mean_b, mean_c = statistics.fmean(bubbles), statistics.fmean(per_bubble)
    shape = []
    if mean_b >= 2.2:
        shape.append(("multi_bubble", f"常把话拆成几条消息发（平均 {mean_b:.1f} 条）"))
    elif mean_b <= 1.3:
        shape.append(("single_bubble", f"通常一次只发一条消息（平均 {mean_b:.1f} 条）"))
    if mean_c <= 9:
        shape.append(("short", f"每条消息很短（平均 {mean_c:.0f} 字）"))
    elif mean_c >= 24:
        shape.append(("long", f"每条消息较长、句子完整（平均 {mean_c:.0f} 字）"))
    end_open = statistics.fmean(f["no_end_punct"] for f in feats.values())
    if end_open >= 0.8:
        shape.append(("no_end_punct", "句末通常不加标点"))
    elif agg["period"] >= 2.0 and end_open <= 0.3:
        shape.append(("full_punct", "标点规范，句末常用句号"))
    for key, desc in shape:
        traits.append({"trait": key, "description": desc, "type": "deterministic_statistic", "evidence_ids": sorted(feats), "evidence_count": len(feats), "of_messages": len(history),
                       "stability": "stable" if len(set(stype.values())) >= 2 else "context_dependent", "scenario_types": sorted(set(stype.values()))})
    stats = {"messages": len(history), "bubbles_per_message": {"mean": round(mean_b, 2), "min": min(bubbles), "max": max(bubbles)},
             "chars_per_bubble": {"mean": round(mean_c, 1), "sd": round(statistics.pstdev(per_bubble), 1) if len(per_bubble) > 1 else 0.0},
             "features": {k: round(v, 3) for k, v in agg.items()}}
    notes = []
    if len(history) < 5:
        notes.append(f"only {len(history)} messages: traits are weakly evidenced")
    if len(set(stype.values())) < 2:
        notes.append("all history from one scenario type: stability cannot be judged")
    out = {"profiler_version": PROFILER_VERSION, "source_ids": sorted(m["message_id"] for m in history), "message_count": len(history), "traits": traits, "stats": stats, "notes": notes}
    out["profile_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in out.items()}, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return out


def render(p: Mapping) -> str:
    """The profile as prompt text: confirmed traits only, stable ones first; context-dependent ones say so."""
    lines = []
    for t in sorted(p.get("traits", []), key=lambda t: (t["stability"] != "stable", t["trait"])):
        if t["type"] == "unconfirmed_hypothesis":
            continue
        lines.append(f"- {t['description']}" + ("（只在部分场合出现）" if t["stability"] == "context_dependent" else ""))
    return "\n".join(lines) or "- （样本不足，无法总结稳定的说话方式）"


def select_examples(pool: Sequence[Mapping], k: int, *, seed: str, exclude_text: Sequence[str] = ()) -> List[Mapping]:
    """Up to ``k`` representative messages: spread over scenario types, mutually dissimilar, never close to ``exclude_text``
    (the target). Deterministic: ties break on sha256(seed:message_id)."""
    order = sorted(pool, key=lambda m: hashlib.sha256(f"{seed}:{m['message_id']}".encode()).hexdigest())
    banned = [char_grams("".join(t)) for t in exclude_text]
    chosen: List[Mapping] = []
    used_types = set()
    for pass_ in (0, 1):
        for m in order:
            if len(chosen) >= k or m in chosen:
                continue
            g = char_grams("".join(m["bubbles"]))
            if any(jaccard(g, b) >= 0.5 for b in banned):
                continue
            if any(jaccard(g, char_grams("".join(c["bubbles"]))) >= 0.5 for c in chosen):
                continue
            if pass_ == 0 and m.get("scenario_type") in used_types:
                continue
            chosen.append(m)
            used_types.add(m.get("scenario_type"))
    return chosen
