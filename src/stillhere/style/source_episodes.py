"""Style Episodes from existing multi-party dialogue corpora (T2-D0.1), built on the T2-D0 contract (``episodes.StyleEpisode``).

Input: utterance records ``{source_id, dialogue_id, turn, speaker_id, text}`` from a corpus whose speaker ids are stable across
dialogues. Nothing is generated: every target, context line and support example is an original utterance.

Isolation rules (all checked by ``audit``):

* splits are by **speaker**, over every speaker in the corpus; a **dialogue** belongs to the most restricted split of its
  participants (test > validation > train), and an episode uses only dialogues of its own split, so no test speaker's words ever
  appear in a train episode's context or support (a train speaker may appear in a test dialogue's context: that text never trains);
* the target is one utterance; the context is the preceding turns of the same dialogue, oldest dropped first to stay within a
  character budget (so no rendered episode reaches the training cutoff); support utterances are at most ``max_support_chars``;
* support examples and the style profile come from the same speaker's utterances in **other** dialogues of the same split, never
  the target's dialogue, never a near-duplicate of the target (char 3-gram Jaccard ≥ 0.5) or of each other;
* grounded content has an explicit mode:
  - ``CONTEXT_ONLY``: no content condition; the reply must follow from the conversation (realistic for chat; no memory facts);
  - ``ORACLE_CONTENT_CONDITION``: the content words of the target itself (tokens outside the corpus-frequent keep set, in order).
    It is derived from the target and exists only for offline style-vs-content control experiments. It is never evidence for
    Memory-to-Style generation, and is reported separately.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Sequence

from . import episodes as ep
from .features import char_grams, jaccard
from .learnability import keep_set, tokens
from .profiler import profile, render, select_examples

CONTEXT_ONLY = "CONTEXT_ONLY"
ORACLE = "ORACLE_CONTENT_CONDITION"
MODES = (CONTEXT_ONLY, ORACLE)
RANK = {"test": 2, "validation": 1, "train": 0}


def split_speakers(speakers: Sequence[str], *, seed: str, val_share: float, test_share: float) -> Dict[str, str]:
    order = sorted(speakers, key=lambda s: hashlib.sha256(f"{seed}:speaker:{s}".encode()).hexdigest())
    n_test, n_val = round(len(order) * test_share), round(len(order) * val_share)
    return {s: "test" if i < n_test else "validation" if i < n_test + n_val else "train" for i, s in enumerate(order)}


def dialogue_splits(utts: Sequence[Mapping], speaker_split: Mapping[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for u in utts:
        s = speaker_split[u["speaker_id"]]
        if RANK[s] >= RANK.get(out.get(u["dialogue_id"], "train"), 0):
            out[u["dialogue_id"]] = s
    return out


def oracle_content(text: str, keep: set) -> List[str]:
    seen, out = set(), []
    for t in tokens(text):
        if t not in keep and (t.isalpha() or len(t) == 1) and t not in seen and not t.isdigit():
            seen.add(t)
            out.append(t)
    return out


def build(utts: Sequence[Mapping], *, dataset: str, revision: str, license_status: str, speaker_split: Mapping[str, str], persons: Sequence[str],
          per_person: int, seed: str, k: int = 5, history: int = 30, context_turns: int = 6, mode: str = CONTEXT_ONLY, language: str,
          min_chars: int = 4, keep: Optional[set] = None, max_support_chars: int = 300, max_context_chars: int = 1200,
          max_targets_per_dialogue: int = 1, min_turn_gap: int = 8) -> tuple:
    """``max_targets_per_dialogue`` > 1 allows several targets of one speaker in one dialogue, each at least ``min_turn_gap`` turns
    from the others, so their contexts never overlap (``context_turns`` < ``min_turn_gap``)."""
    assert max_targets_per_dialogue == 1 or min_turn_gap > context_turns
    """Returns (episodes as dicts, rejected [{reason, ...}])."""
    assert mode in MODES
    dsplit = dialogue_splits(utts, speaker_split)
    by_dlg = defaultdict(list)
    for u in utts:
        by_dlg[u["dialogue_id"]].append(u)
    for v in by_dlg.values():
        v.sort(key=lambda u: u["turn"])
    own = defaultdict(list)  # speaker -> utterances in dialogues of the speaker's own split
    for u in utts:
        if dsplit[u["dialogue_id"]] == speaker_split[u["speaker_id"]]:
            own[u["speaker_id"]].append(u)
    if keep is None:
        keep = keep_set([u["text"] for u in utts if speaker_split[u["speaker_id"]] == "train"], 300)
    out, rejected = [], []
    for p in persons:
        split = speaker_split[p]
        pool_all = own[p]
        cands = sorted((u for u in pool_all if len(u["text"]) >= min_chars and u["turn"] >= 2 and u.get("eligible", True)),
                       key=lambda u: hashlib.sha256(f"{seed}:{u['source_id']}".encode()).hexdigest())
        used_dialogues: Dict[str, List[int]] = defaultdict(list)
        made = 0
        for t in cands:
            if made >= per_person:
                break
            taken = used_dialogues[t["dialogue_id"]]
            if len(taken) >= max_targets_per_dialogue or any(abs(t["turn"] - x) < min_turn_gap for x in taken):
                continue  # at most max_targets_per_dialogue per dialogue, far enough apart that no two episodes share context
            tg = char_grams(t["text"])
            others = [u for u in pool_all if u["dialogue_id"] != t["dialogue_id"] and len(u["text"]) <= max_support_chars and jaccard(char_grams(u["text"]), tg) < ep.NEAR_DUP]
            others = sorted(others, key=lambda u: hashlib.sha256(f"{seed}:h:{u['source_id']}".encode()).hexdigest())
            hist = others[:history]
            pool = [{"message_id": u["source_id"], "persona_id": p, "item_id": u["dialogue_id"], "scenario_type": f"dlg:{u['dialogue_id']}", "bubbles": [u["text"]]} for u in hist]
            support = select_examples(pool, k, seed=f"{seed}:{t['source_id']}", exclude_text=[[t["text"]]])
            if len(support) < k:
                rejected.append({"reason": "fewer than k support utterances outside the target dialogue", "persona_id": p, "target": t["source_id"]})
                continue
            prof = profile([{"message_id": m["message_id"], "bubbles": m["bubbles"], "scenario_type": m["scenario_type"]} for m in pool])
            ctx = [u for u in by_dlg[t["dialogue_id"]] if u["turn"] < t["turn"]][-context_turns:]
            while len(ctx) > 1 and sum(len(u["text"]) for u in ctx) > max_context_chars:  # token budget: drop the oldest turns first
                ctx = ctx[1:]
            if sum(len(u["text"]) for u in ctx) > max_context_chars:
                rejected.append({"reason": "preceding turn alone exceeds the context budget", "persona_id": p, "target": t["source_id"]})
                continue
            if not ctx:
                rejected.append({"reason": "no preceding context", "persona_id": p, "target": t["source_id"]})
                continue
            facts = oracle_content(t["text"], keep) if mode == ORACLE else []
            if mode == ORACLE and not facts:
                rejected.append({"reason": "oracle content empty (target has no content words)", "persona_id": p, "target": t["source_id"]})
                continue
            e = ep.StyleEpisode(
                episode_id=f"{dataset}.{t['source_id']}.{mode}", persona_id=p, conversation_id=f"{dataset}.{t['dialogue_id']}",
                source_ids=[t["source_id"]] + [s["message_id"] for s in support], style_profile=prof,
                support_examples=[{k2: s[k2] for k2 in ("message_id", "persona_id", "item_id", "scenario_type", "bubbles")} for s in support],
                current_context=[{"speaker": "self" if u["speaker_id"] == p else "other", "speaker_id": u["speaker_id"], "source_id": u["source_id"], "text": u["text"]} for u in ctx],
                grounded_content={"kind": mode, "mode": mode, "item_id": t["dialogue_id"], "facts": facts},
                target_response={"message_id": t["source_id"], "item_id": t["dialogue_id"], "bubbles": [b for b in t["text"].split("\n") if b.strip()] or [t["text"]]},
                language=language, scenario_type="chat", split=split,
                provenance={"origin": "existing_corpus", "source": dataset, "revision": revision, "license": license_status, "generator": "none (original utterances)"},
                quality_flags=["oracle_content_target_derived"] if mode == ORACLE else [])
            d = e.to_dict()
            d.update({"source_dataset": dataset, "source_revision": revision, "license_status": license_status, "support_source_ids": [s["message_id"] for s in support]})
            out.append(d)
            used_dialogues[t["dialogue_id"]].append(t["turn"])
            made += 1
    return out, rejected


def validate(d: Mapping) -> List[str]:
    probs = ep.validate(d)
    if d["grounded_content"].get("mode") == CONTEXT_ONLY:
        probs = [x for x in probs if x != "no grounded content"]
    if d["grounded_content"].get("mode") not in MODES:
        probs.append("grounded content mode")
    return probs


def audit(eps: Sequence[Mapping], speaker_split: Mapping[str, str]) -> dict:
    """T2-D0 leakage audit plus the multi-party rules: every context speaker is in the episode's split; no dialogue in two splits."""
    objs = [ep.StyleEpisode(**{k: v for k, v in d.items() if k in ep.REQUIRED}) for d in eps]
    base = ep.leakage(objs)
    base["counts"]["episode_invalid"] = sum(1 for d in eps if validate(d))
    extra = {"context_speaker_in_more_restricted_split": 0, "support_from_target_dialogue": 0, "oracle_content_unflagged": 0}
    for d in eps:
        # a context line may come from a less restricted split (a train speaker talking to a test speaker inside a test dialogue),
        # never from a more restricted one: no validation or test speaker's words in a train episode, no test words in validation
        if any(RANK[speaker_split[c["speaker_id"]]] > RANK[d["split"]] for c in d["current_context"]):
            extra["context_speaker_in_more_restricted_split"] += 1
        if any(s["item_id"] == d["target_response"]["item_id"] for s in d["support_examples"]):
            extra["support_from_target_dialogue"] += 1
        if d["grounded_content"]["mode"] == ORACLE and "oracle_content_target_derived" not in d["quality_flags"]:
            extra["oracle_content_unflagged"] += 1
    base["counts"].update(extra)
    base["pass"] = not any(base["counts"].values())
    return base


def _content_text(d: Mapping) -> str:
    if d["grounded_content"]["mode"] == CONTEXT_ONLY:
        return "- （本轮没有给定事实：根据对话自然回复，不要编造具体事实）"
    return "- 需要用到的内容词（取自原回复，仅用于离线对照实验）：" + "、".join(d["grounded_content"]["facts"])


def render_system(d: Mapping, *, style: Optional[Mapping] = None, no_style: bool = False) -> str:
    """The condition text. ``style``: another person's ``{style_profile, support_examples}`` (Style Swap); ``no_style``: content only."""
    if no_style:
        return ep.NO_STYLE_SYSTEM.format(content=_content_text(d))
    s = style or d
    examples = "\n".join(f"例{i}：" + " / ".join(x["bubbles"]) for i, x in enumerate(s["support_examples"], 1))
    return ep.SYSTEM_TEMPLATE.format(profile=render(s["style_profile"]), examples=examples, content=_content_text(d))


def render_context(d: Mapping, *, self_label: str = "我", other_label: str = "对方") -> str:
    return "\n".join(f"{self_label if c['speaker'] == 'self' else other_label}：{c['text']}" for c in d["current_context"])


def to_sharegpt(d: Mapping) -> dict:
    """One system turn (instruction + profile + support + content), one user turn (the dialogue so far), one supervised reply."""
    return {"id": d["episode_id"], "persona_id": d["persona_id"], "split": d["split"], "system": render_system(d),
            "conversations": [{"from": "human", "value": render_context(d)}, {"from": "gpt", "value": "\n".join(d["target_response"]["bubbles"])}]}


def swap_style(d: Mapping, donor: str, utts: Sequence[Mapping], speaker_split: Mapping[str, str], *, seed: str, k: int = 5, history: int = 30,
               max_support_chars: int = 300, dsplit: Optional[Mapping[str, str]] = None, dialogues_of: Optional[Mapping[str, set]] = None) -> Optional[dict]:
    """The donor's style condition for episode ``d``: profile and support from the donor's own utterances in dialogues of the donor's split,
    excluding the target dialogue and every dialogue the episode's own speaker took part in (so the swapped condition never carries a reply
    to, or from, the person being evaluated)."""
    dsplit = dsplit or dialogue_splits(utts, speaker_split)
    if dialogues_of is None:
        dialogues_of = defaultdict(set)
        for u in utts:
            dialogues_of[u["speaker_id"]].add(u["dialogue_id"])
    mine = dialogues_of[d["persona_id"]]
    pool_u = [u for u in utts if u["speaker_id"] == donor and dsplit[u["dialogue_id"]] == speaker_split[donor] and u["dialogue_id"] not in mine
              and u["dialogue_id"] != d["target_response"]["item_id"] and len(u["text"]) <= max_support_chars]
    pool_u = sorted(pool_u, key=lambda u: hashlib.sha256(f"{seed}:h:{u['source_id']}".encode()).hexdigest())[:history]
    pool = [{"message_id": u["source_id"], "persona_id": donor, "item_id": u["dialogue_id"], "scenario_type": f"dlg:{u['dialogue_id']}", "bubbles": [u["text"]]} for u in pool_u]
    support = select_examples(pool, k, seed=f"{seed}:{d['episode_id']}:{donor}", exclude_text=[d["target_response"]["bubbles"]])
    if len(support) < k:
        return None
    prof = profile([{"message_id": m["message_id"], "bubbles": m["bubbles"], "scenario_type": m["scenario_type"]} for m in pool])
    return {"persona_id": donor, "style_profile": prof, "support_examples": support}
