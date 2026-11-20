"""Style Episodes: the one contract shared by training and inference, its builder, the person-level split, the leakage audit and
the Qwen3 rendering.

    P(target_response | style_profile, support_examples, current_context, grounded_content)

* ``support_examples`` and ``style_profile`` come from the same person's history only, never from the target, its own item
  (scenario or content id), or anything nearly equal to it. The profile records the ids it read (provenance).
* ``grounded_content`` is what must be said (facts); the target is how this person says it. A target must keep every fact.
* Splits are by person: train, validation and sealed test personas are disjoint. The profiler needs no corpus statistics (per-message
  thresholds); the pilot quality checks z-score features on train personas only.

Rendering (``to_sharegpt``) puts everything but the target in the system and user turns, so with LLaMA-Factory's ``qwen3``
template, ``enable_thinking: false`` and ``mask_history: true`` only the target reply is supervised (audited with the real
tokenizer by training/cpcd_colab/scripts/audit_loss_mask.py with training/universal_style_t2/configs/qwen3_8b_style_poc_DRAFT.yaml).
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .features import char_grams, jaccard
from .profiler import profile, render, select_examples

SCHEMA = "stillhere.style.episode/1"
SPLITS = ("train", "validation", "test")
NEAR_DUP = 0.5
SYSTEM_TEMPLATE = ("你要用一位特定说话者的口吻回复对方。只学习下面描述和示例里的说话方式，不要照抄示例的内容；"
                   "只表达“要表达的内容”里的事实，不要增加、删改或编造。\n"
                   "【说话方式】\n{profile}\n"
                   "【此人过去的消息】（只示范说话方式，不是本次的内容）\n{examples}\n"
                   "【要表达的内容】（必须全部保留）\n{content}\n"
                   "回复可以分成几条消息，每条一行。")
NO_STYLE_SYSTEM = ("请回复对方。只表达“要表达的内容”里的事实，不要增加、删改或编造。\n"
                   "【要表达的内容】（必须全部保留）\n{content}\n回复可以分成几条消息，每条一行。")


@dataclass
class StyleEpisode:
    episode_id: str
    persona_id: str
    conversation_id: str
    source_ids: List[str]
    style_profile: dict
    support_examples: List[dict]
    current_context: List[dict]
    grounded_content: dict
    target_response: dict
    language: str
    scenario_type: str
    split: str
    provenance: dict
    quality_flags: List[str] = field(default_factory=list)
    schema: str = SCHEMA

    def to_dict(self) -> dict:
        return asdict(self)


REQUIRED = ("episode_id", "persona_id", "conversation_id", "source_ids", "style_profile", "support_examples", "current_context", "grounded_content",
            "target_response", "language", "scenario_type", "split", "provenance", "quality_flags", "schema")


def validate(ep: Mapping) -> List[str]:
    """Schema and isolation rules for one episode. An empty list means valid."""
    p = [f"missing {k}" for k in REQUIRED if k not in ep]
    if p:
        return p
    if ep["schema"] != SCHEMA:
        p.append("schema version")
    if ep["split"] not in SPLITS:
        p.append("split")
    t = ep["target_response"]
    if not t.get("bubbles") or not all(isinstance(b, str) and b.strip() and "\n" not in b for b in t["bubbles"]):
        p.append("target bubbles must be non-empty single lines")
    support_ids = [s["message_id"] for s in ep["support_examples"]]
    if t.get("message_id") in support_ids:
        p.append("target is in its own support")
    if t.get("message_id") in ep["style_profile"].get("source_ids", []):
        p.append("profile read the target")
    if any(s.get("persona_id") != ep["persona_id"] for s in ep["support_examples"]):
        p.append("support from another speaker")
    if any(s.get("item_id") == t.get("item_id") for s in ep["support_examples"]):
        p.append("support shares the target's item")
    tg = char_grams("".join(t["bubbles"]))
    if any(jaccard(tg, char_grams("".join(s["bubbles"]))) >= NEAR_DUP for s in ep["support_examples"]):
        p.append("support nearly duplicates the target")
    if not ep["grounded_content"].get("facts"):
        p.append("no grounded content")
    if not ep["provenance"].get("origin") or not ep["provenance"].get("license"):
        p.append("provenance")
    return p


def fact_kept(fact, text: str) -> bool:
    """A fact is a list of accepted surface forms (``[["下午三点", "15:00"]]``) or one plain string for natural episodes."""
    forms = fact if isinstance(fact, list) else [fact]
    return any(f in text for f in forms)


# ---------------------------------------------------------------- split

def split_personas(persona_ids: Sequence[str], *, seed: str, n_val: int, n_test: int) -> Dict[str, str]:
    order = sorted(persona_ids, key=lambda p: hashlib.sha256(f"{seed}:persona:{p}".encode()).hexdigest())
    return {p: "test" if i < n_test else "validation" if i < n_test + n_val else "train" for i, p in enumerate(order)}


# ---------------------------------------------------------------- building

def build(messages: Sequence[Mapping], contexts: Mapping[str, str], contents: Mapping[str, Mapping], split: Mapping[str, str], *, seed: str, k: int = 5,
          provenance: Mapping) -> List[StyleEpisode]:
    """``messages``: [{message_id, persona_id, item_id, scenario_type, kind ("natural"|"controlled"), bubbles, facts?, language}]."""
    by_person: Dict[str, List[Mapping]] = defaultdict(list)
    for m in messages:
        by_person[m["persona_id"]].append(m)
    out = []
    for m in messages:
        pool = [x for x in by_person[m["persona_id"]] if x["message_id"] != m["message_id"] and x["item_id"] != m["item_id"]
                and jaccard(char_grams("".join(x["bubbles"])), char_grams("".join(m["bubbles"]))) < NEAR_DUP]
        history = [{"message_id": x["message_id"], "bubbles": x["bubbles"], "scenario_type": x["scenario_type"]} for x in pool]
        prof = profile(history)
        support = select_examples(pool, k, seed=f"{seed}:{m['message_id']}", exclude_text=[m["bubbles"]])
        facts = m["facts"] if m["kind"] == "natural" else contents[m["item_id"]]["facts"]
        context = contexts[m["item_id"]] if m["kind"] == "natural" else contents[m["item_id"]]["context"]
        flags = []
        text = "".join(m["bubbles"])
        if m["kind"] == "controlled" and not all(fact_kept(f, text) for f in facts):
            flags.append("target_misses_a_fact")
        out.append(StyleEpisode(
            episode_id=f"ep.{m['message_id']}", persona_id=m["persona_id"], conversation_id=f"conv.{m['persona_id']}.{m['item_id']}",
            source_ids=[m["message_id"]] + [s["message_id"] for s in support], style_profile=prof,
            support_examples=[{"message_id": s["message_id"], "persona_id": s["persona_id"], "item_id": s["item_id"], "scenario_type": s["scenario_type"], "bubbles": list(s["bubbles"])} for s in support],
            current_context=[{"speaker": "other", "text": context}], grounded_content={"kind": m["kind"], "item_id": m["item_id"], "facts": facts},
            target_response={"message_id": m["message_id"], "item_id": m["item_id"], "bubbles": list(m["bubbles"])}, language=m.get("language", "zh"),
            scenario_type=m["scenario_type"], split=split[m["persona_id"]], provenance=dict(provenance), quality_flags=flags))
    return out


# ---------------------------------------------------------------- leakage

def leakage(episodes: Sequence[StyleEpisode]) -> dict:
    """Every way a test example could be seen, or a target could be given away. All counts must be zero (``template`` is a warning)."""
    found = defaultdict(list)
    persona_splits = defaultdict(set)
    conv_splits = defaultdict(set)
    for e in episodes:
        persona_splits[e.persona_id].add(e.split)
        conv_splits[e.conversation_id].add(e.split)
        problems = validate(e.to_dict())
        if problems:
            found["episode_invalid"].append([e.episode_id, problems])
        sup = [s["message_id"] for s in e.support_examples]
        if e.target_response["message_id"] in sup:
            found["target_in_support"].append(e.episode_id)
        if e.target_response["message_id"] in e.style_profile.get("source_ids", []):
            found["target_in_profile"].append(e.episode_id)
        target_text = "".join(e.target_response["bubbles"])
        if target_text in json.dumps(e.style_profile, ensure_ascii=False) or any(target_text == "".join(s["bubbles"]) for s in e.support_examples):
            found["target_text_in_condition"].append(e.episode_id)
        tg = char_grams(target_text)
        if any(jaccard(tg, char_grams("".join(s["bubbles"]))) >= NEAR_DUP for s in e.support_examples):
            found["near_duplicate_support"].append(e.episode_id)
        if any(s["persona_id"] != e.persona_id for s in e.support_examples):
            found["other_speaker_in_support"].append(e.episode_id)
        if any(s["item_id"] == e.target_response["item_id"] for s in e.support_examples):
            found["fact_leak_same_item_in_support"].append(e.episode_id)
    found["persona_in_two_splits"] = sorted(p for p, s in persona_splits.items() if len(s) > 1)
    found["conversation_in_two_splits"] = sorted(c for c, s in conv_splits.items() if len(s) > 1)
    # near-duplicate targets across different personas on the same item (template contamination) and across splits
    by_item = defaultdict(list)
    for e in episodes:
        by_item[e.target_response["item_id"]].append(e)
    template, cross = [], []
    for item, eps in by_item.items():
        for i in range(len(eps)):
            for j in range(i + 1, len(eps)):
                s = jaccard(char_grams("".join(eps[i].target_response["bubbles"])), char_grams("".join(eps[j].target_response["bubbles"])))
                if s >= 0.6:
                    template.append([eps[i].episode_id, eps[j].episode_id, round(s, 3)])
                    if eps[i].split != eps[j].split:
                        cross.append([eps[i].episode_id, eps[j].episode_id, round(s, 3)])
    found["template_near_duplicates_warning"] = template
    found["near_duplicate_across_splits"] = cross
    hard = ("episode_invalid", "target_in_support", "target_in_profile", "target_text_in_condition", "near_duplicate_support", "other_speaker_in_support",
            "fact_leak_same_item_in_support", "persona_in_two_splits", "conversation_in_two_splits", "near_duplicate_across_splits")
    counts = {k: len(found.get(k, [])) for k in hard}
    return {"counts": counts, "pass": not any(counts.values()), "template_warnings": len(template), "detail": {k: v for k, v in found.items() if v}}


# ---------------------------------------------------------------- rendering

def system_text(e: StyleEpisode, *, style: Optional[StyleEpisode] = None, no_style: bool = False) -> str:
    """The condition. ``style`` swaps in another person's profile and examples (Style Swap); ``no_style`` drops both."""
    facts = e.grounded_content["facts"]
    content = "\n".join("- " + (f[0] if isinstance(f, list) else f) for f in facts)
    if no_style:
        return NO_STYLE_SYSTEM.format(content=content)
    s = style or e
    examples = "\n".join(f"例{i}：" + " / ".join(x["bubbles"]) for i, x in enumerate(s.support_examples, 1))
    return SYSTEM_TEMPLATE.format(profile=render(s.style_profile), examples=examples, content=content)


def to_sharegpt(e: StyleEpisode) -> dict:
    return {"id": e.episode_id, "persona_id": e.persona_id, "split": e.split, "system": system_text(e),
            "conversations": [{"from": "human", "value": c["text"]} for c in e.current_context] + [{"from": "gpt", "value": "\n".join(e.target_response["bubbles"])}]}


def eval_fixtures(episodes: Sequence[StyleEpisode], *, seed: str) -> List[dict]:
    """Style Swap and No-Style variants of every sealed-test episode: same content and question, another person's (or no) style."""
    test = [e for e in episodes if e.split == "test"]
    persons = sorted({e.persona_id for e in test})
    by_person_item = {(e.persona_id, e.target_response["item_id"]): e for e in test}
    out = []
    for e in test:
        others = [p for p in persons if p != e.persona_id]
        other = sorted(others, key=lambda p: hashlib.sha256(f"{seed}:{e.episode_id}:{p}".encode()).hexdigest())[0]
        item = e.target_response["item_id"]
        # the donor's condition must not show the other person's reply to this same item (that would hand over the content)
        donor = next((x for x in test if x.persona_id == other and x.target_response["item_id"] != item and all(s["item_id"] != item for s in x.support_examples)), None)
        base = {"episode_id": e.episode_id, "content_from": e.persona_id, "question": e.current_context[0]["text"], "facts": e.grounded_content["facts"]}
        out.append({**base, "condition": "same_style", "style_from": e.persona_id, "system": system_text(e)})
        if donor is not None:
            out.append({**base, "condition": "style_swap", "style_from": other, "style_donor_episode": donor.episode_id, "system": system_text(e, style=donor),
                        "swap_reference": by_person_item.get((other, e.target_response["item_id"])) and "\n".join(by_person_item[(other, e.target_response["item_id"])].target_response["bubbles"])})
        out.append({**base, "condition": "no_style", "style_from": None, "system": system_text(e, no_style=True)})
    return out
