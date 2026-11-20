"""UniversalStylePromptContract v2: the ONE way StillHere builds a style-conditioned prompt, for training, validation, evaluation and runtime.

It does not re-implement the prompt: the messages come from ``stillhere.style.prompt_builder.build_style_conditioned_messages`` (schema
``stillhere.style.prompt/2.0``), the function every T2-D2 training, validation and test row was rendered with, and the style examples are
chosen by ``stillhere.style.profiler.select_examples`` with the training's rules (k = 5, deterministic order, spread over topics,
mutually dissimilar, never close to the reply being produced). The contract adds what a runtime needs around that core:

* persona isolation: every example and the profile's evidence must belong to the target persona; anything else is refused;
* answer-leak guard: an example equal or near-equal to the reference reply (when one is known, e.g. in evaluation) is refused;
* duplicate / near-duplicate example removal (char 3-gram Jaccard >= 0.5, as the training selection);
* token budget (default 1536 prompt tokens, the T2-D2 cutoff / generation budget), measured with the caller's tokenizer when given;
* explicit ablation semantics: no examples or no profile is allowed and marked ``ablation``;
* model metadata: base model, revision, chat template hash, enable_thinking False (the Qwen3 template is applied by the tokenizer at
  inference exactly as in training: ``apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)``);
* a stable prompt hash (sha256 of the canonical JSON of the messages) and the selected example ids for traces.

Memory decides WHAT to say (``grounded_content``: facts with roles / unknowns, or no-memory); style decides HOW (profile + examples).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Mapping, Optional, Sequence

from stillhere.style import prompt_builder as pb
from stillhere.style.features import char_grams, jaccard
from stillhere.style.profiler import profile as build_profile
from stillhere.style.profiler import select_examples

CONTRACT_VERSION = "stillhere.style.contract/2.0"
PROMPT_SCHEMA_VERSION = pb.PROMPT_SCHEMA_VERSION            # stillhere.style.prompt/2.0
TRAINING_K_EXAMPLES = 5
TRAINING_PROFILE_HISTORY = 30
DEFAULT_PROMPT_TOKEN_BUDGET = 1536
NEAR_DUP = 0.5
MODEL = {"base_model": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218", "chat_template": "qwen3 official (tokenizer.apply_chat_template)",
         "chat_template_sha256": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8", "enable_thinking": False, "add_generation_prompt": True}


class ContractError(ValueError):
    """The request violates the style contract (persona isolation, answer leak, schema)."""


@dataclass(frozen=True)
class StyleMessage:
    """One earlier message of the persona: a candidate style example / profile evidence."""
    message_id: str
    persona_id: str
    text: str
    topic: str = ""            # conversation or topic id; examples are spread over topics
    source: str = ""


@dataclass
class ContractPayload:
    contract_version: str
    prompt_schema_version: str
    persona_id: str
    messages: List[dict]
    prompt_hash: str
    selected_example_ids: List[str]
    profile_source_ids: List[str]
    ablation: List[str]
    prompt_tokens: Optional[int]
    model: dict = field(default_factory=lambda: dict(MODEL))

    def trace(self) -> dict:
        """What a run trace records: ids and hashes, never the texts."""
        d = asdict(self)
        d.pop("messages")
        return d


def prompt_hash(messages: Sequence[Mapping]) -> str:
    return hashlib.sha256(json.dumps(list(messages), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _pool(history: Sequence[StyleMessage]):
    return [{"message_id": m.message_id, "persona_id": m.persona_id, "item_id": m.topic or m.message_id, "scenario_type": m.topic or m.message_id,
             "bubbles": [x for x in m.text.split("\n") if x.strip()] or [m.text]} for m in history]


def build(persona_id: str, conversation: Sequence[Mapping], *, grounded_content: Optional[Mapping] = None, history: Sequence[StyleMessage] = (),
          use_profile: bool = True, use_examples: bool = True, k_examples: int = TRAINING_K_EXAMPLES, reference_reply: Optional[str] = None,
          seed: str = "stillhere-runtime", constraints: Sequence[str] = (), relation: Optional[str] = None,
          count_tokens: Optional[Callable[[Sequence[Mapping]], int]] = None, token_budget: int = DEFAULT_PROMPT_TOKEN_BUDGET) -> ContractPayload:
    """Build the style-conditioned chat for one reply. ``history``: the persona's earlier messages (profile evidence and example pool),
    most recent last; only the last ``TRAINING_PROFILE_HISTORY`` are used, as in training."""
    foreign = [m.message_id for m in history if m.persona_id != persona_id]
    if foreign:
        raise ContractError(f"persona isolation: {len(foreign)} history message(s) belong to another persona")
    hist = list(history)[-TRAINING_PROFILE_HISTORY:]
    if reference_reply:
        ref = char_grams(reference_reply.replace("\n", ""))
        hist = [m for m in hist if m.text.replace("\n", "") != reference_reply.replace("\n", "") and jaccard(char_grams(m.text.replace("\n", "")), ref) < NEAR_DUP]
    pool = _pool(hist)
    ablation = []
    prof = None
    if use_profile and pool:
        prof = build_profile([{"message_id": p["message_id"], "bubbles": p["bubbles"], "scenario_type": p["scenario_type"]} for p in pool])
    else:
        ablation.append("no_profile")
    examples, selected = None, []
    if use_examples and pool:
        chosen = select_examples(pool, k_examples, seed=seed, exclude_text=[[reference_reply]] if reference_reply else ())
        selected = [c["message_id"] for c in chosen]
        examples = [" / ".join(c["bubbles"]) for c in chosen]
    else:
        ablation.append("no_examples")
    if use_examples and pool and len(selected) < k_examples:
        ablation.append(f"fewer_examples:{len(selected)}")
    msgs = pb.build_style_conditioned_messages(conversation, grounded_content=grounded_content, style_profile=prof, style_examples=examples,
                                               constraints=constraints, relation=relation)
    tokens = count_tokens(msgs) if count_tokens else None
    if tokens is not None and tokens > token_budget:
        raise ContractError(f"prompt is {tokens} tokens, over the {token_budget}-token budget")
    return ContractPayload(CONTRACT_VERSION, PROMPT_SCHEMA_VERSION, persona_id, msgs, prompt_hash(msgs), selected,
                           list((prof or {}).get("source_ids", [])), ablation, tokens)


def from_training_episode(episode: Mapping) -> List[dict]:
    """Re-render a T2-D2 real training / validation / test episode from its stored fields with the shared builder (the reproduction check:
    the result must equal the episode's stored ``messages``)."""
    conv = [{"speaker": "other", "text": c["text"].replace("\n", " ")} for c in episode["current_context"]]
    examples = [" / ".join(x["bubbles"]) for x in episode["support_examples"]]
    return pb.build_style_conditioned_messages(conv, grounded_content=episode["grounded"], style_profile=episode["style_profile"], style_examples=examples)
