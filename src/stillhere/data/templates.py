"""Qwen3 chat-template rendering shared by training and inference.

One renderer turns ``ChatMessage`` turns into the exact text the official Qwen3-8B chat template
(``tokenizer.apply_chat_template``) produces in non-thinking mode, so a training sample and an inference
prompt built from the same history are identical up to the supervised target.

Rules (each one is checked by tests against the official template and tokenizer):

- Multi-bubble serialisation: the bubbles of one turn are joined with ``"\\n"`` into one message. A turn is
  never split into several messages, so several bubbles never become several assistant turns. A bubble
  must not contain a line break, start or end with whitespace, or contain a special-token or think/tool
  marker; such input is rejected instead of being silently rewritten.
- Roles alternate strictly, starting with ``user``. System messages are not accepted as input: the only
  system message a template may emit is its own situation line (see ``SituationMode``).
- Inference prompt: history ending with a user turn, then ``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n``
  (what ``add_generation_prompt=True, enable_thinking=False`` emits).
- Training example: the same prompt text followed by the target ``content<|im_end|>\\n``. Only the target
  is supervised; earlier assistant turns are context and carry no think block, exactly as the official
  template renders them.
- Thinking mode is not supported.
- Persona-card templates (``persona_card=True``, Phase 5B ``qwen3_situation_nothink`` 0.2.0): the system message
  is the situation line followed by a persona card, the memory block and answer rules, composed by
  ``inference.prompting`` and passed through the same official template. This module renders only their situation
  line; they have no training format yet, so ``render_training_example`` refuses them.
- Stage J (``qwen3_situation_nothink`` 0.2.1): the same layout with ``persona_card_v2`` (a card without time-bound
  clauses, and the AI disclosure rule without an example sentence) and the ``compact_v1`` memory layout (dates
  instead of instants, shared instructions once in the header, evidence of every answered query placed first in the
  memory budget). Both are recorded in the definition, so the hash says which prompt a run used; 0.1.0 and 0.2.0
  keep their definitions and hashes.

Standard library only. Token-level checks take any tokenizer object with the Hugging Face interface; this
module never imports ``transformers``.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from stillhere.contracts import ChatMessage, PromptTemplateRef, Role
from stillhere.contracts._base import canonical_json, require_id

QWEN3_BASE_MODEL_ID = "Qwen/Qwen3-8B"
QWEN3_BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
# sha256 of ``tokenizer.chat_template`` (UTF-8) in the revision above.
QWEN3_CHAT_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
ENDOFTEXT = "<|endoftext|>"
EMPTY_THINK = "<think>\n\n</think>\n\n"
BUBBLE_SEPARATOR = "\n"

# Token ids in the Qwen3 tokenizer (checked by the tokenizer tests).
IM_START_ID = 151644
IM_END_ID = 151645            # eos for chat generation
ENDOFTEXT_ID = 151643         # pad, and the second eos in generation_config.json
THINK_ID = 151667
END_THINK_ID = 151668
EOS_TOKEN_IDS = (IM_END_ID, ENDOFTEXT_ID)

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|<>\s]{1,64}\|>")
_RESERVED_TAGS = ("<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>")


PERSONA_CARD_FORMAT = "persona_card_v1"
PERSONA_CARD_FORMAT_V2 = "persona_card_v2"     # Stage J: no time-bound clauses, no disclosure example sentence
CARD_FORMATS = (PERSONA_CARD_FORMAT, PERSONA_CARD_FORMAT_V2)
MEMORY_LAYOUT_COMPACT = "compact_v1"           # Stage J: dates, shared header, evidence-first budget
MEMORY_LAYOUTS = ("standard", MEMORY_LAYOUT_COMPACT)


class TemplateError(ValueError):
    """Raised when turns cannot be rendered under a template's rules."""


class SituationMode(str, Enum):
    NONE = "none"      # no conditioning: the official template exactly (P1 baseline format)
    SYSTEM = "system"  # one system message holding the situation line, in training and inference alike


@dataclass(frozen=True)
class TemplateSpec:
    """Everything that decides how turns become text. Training and inference must use the same spec."""

    name: str
    version: str
    situation_mode: SituationMode
    situation_format: str = "situation: {situation}"
    persona_card: bool = False     # Phase 5B: the system message also holds a persona card and answer rules
    card_format: str = PERSONA_CARD_FORMAT     # Stage J: PERSONA_CARD_FORMAT_V2
    memory_layout: str = "standard"            # Stage J: MEMORY_LAYOUT_COMPACT

    def __post_init__(self) -> None:
        require_id("TemplateSpec.name", self.name)
        if "{situation}" not in self.situation_format or "\n" in self.situation_format:
            raise TemplateError("situation_format must contain {situation} and no line break")
        if self.persona_card and self.situation_mode is not SituationMode.SYSTEM:
            raise TemplateError("a persona card goes in the situation system message")
        if self.card_format not in CARD_FORMATS or self.memory_layout not in MEMORY_LAYOUTS:
            raise TemplateError(f"card_format must be one of {CARD_FORMATS}, memory_layout one of {MEMORY_LAYOUTS}")
        if not self.persona_card and (self.card_format != PERSONA_CARD_FORMAT or self.memory_layout != "standard"):
            raise TemplateError("only a persona-card template takes another card format or memory layout")

    def definition(self) -> Dict[str, Any]:
        d = {
            "base_chat_template_sha256": QWEN3_CHAT_TEMPLATE_SHA256,
            "base_model": QWEN3_BASE_MODEL_ID,
            "base_revision": QWEN3_BASE_REVISION,
            "bubble_separator": BUBBLE_SEPARATOR,
            "enable_thinking": False,
            "name": self.name,
            "situation_format": self.situation_format if self.situation_mode is SituationMode.SYSTEM else None,
            "situation_mode": self.situation_mode.value,
            "supervised_target": "final assistant turn: content + <|im_end|>\\n",
            "version": self.version,
        }
        if self.persona_card:      # absent otherwise, so earlier templates keep their hash
            d["persona_card"] = self.card_format
            d["system_layout"] = PERSONA_SYSTEM_LAYOUT
            d["supervised_target"] = None
        if self.memory_layout != "standard":    # absent for the standard layout, so 0.2.0 keeps its hash
            d["memory_layout"] = self.memory_layout
        return d

    @property
    def compact_memory(self) -> bool:
        return self.memory_layout == MEMORY_LAYOUT_COMPACT

    @property
    def sha256(self) -> str:
        """The official template's hash when rendering is that template unchanged, else the definition's hash."""
        if self.situation_mode is SituationMode.NONE:
            return QWEN3_CHAT_TEMPLATE_SHA256
        return hashlib.sha256(canonical_json(self.definition()).encode("utf-8")).hexdigest()

    def ref(self) -> PromptTemplateRef:
        return PromptTemplateRef(name=self.name, version=self.version, sha256=self.sha256)

    def situation_line(self, situation: str) -> str:
        return self.situation_format.format(situation=situation)


# The P1 baseline format: no system message, no situation, official template only.
QWEN3_NATIVE_NOTHINK = TemplateSpec(name="qwen3_native_nothink", version="hf-b968826d",
                                    situation_mode=SituationMode.NONE)
# The public situation-conditioned format for synthetic training data. Never used for the P1 private adapter.
QWEN3_SITUATION_NOTHINK = TemplateSpec(name="qwen3_situation_nothink", version="0.1.0",
                                       situation_mode=SituationMode.SYSTEM)
# Phase 5B: the same situation line first (the prefix the pilot adapter was trained on), then who is speaking, the
# memory block and the answer rules. Inference only (see the module docstring).
PERSONA_SYSTEM_LAYOUT = "situation line; persona card; memory block (if any); answer rules; answer check (retry only)"
QWEN3_SITUATION_NOTHINK_V2 = TemplateSpec(name="qwen3_situation_nothink", version="0.2.0",
                                          situation_mode=SituationMode.SYSTEM, persona_card=True)
# Stage J: the 0.2.0 layout with the time-free persona card and the compact memory block (inference only).
QWEN3_SITUATION_NOTHINK_V2_1 = TemplateSpec(name="qwen3_situation_nothink", version="0.2.1",
                                            situation_mode=SituationMode.SYSTEM, persona_card=True,
                                            card_format=PERSONA_CARD_FORMAT_V2, memory_layout=MEMORY_LAYOUT_COMPACT)


def check_bubble(bubble: str, where: str = "bubble") -> None:
    if not isinstance(bubble, str) or not bubble.strip():
        raise TemplateError(f"{where} must be a non-empty string")
    if "\n" in bubble or "\r" in bubble:
        raise TemplateError(f"{where} contains a line break, which is the bubble separator")
    if bubble != bubble.strip():
        raise TemplateError(f"{where} starts or ends with whitespace")
    if _SPECIAL_TOKEN_RE.search(bubble) or any(tag in bubble for tag in _RESERVED_TAGS):
        raise TemplateError(f"{where} contains a special-token or reserved marker")


def serialize_bubbles(bubbles: Sequence[str]) -> str:
    """Join one turn's bubbles into message content. Rejects bubbles that would make the split ambiguous."""
    if isinstance(bubbles, str) or not bubbles:
        raise TemplateError("a turn needs a non-empty sequence of bubbles")
    for i, b in enumerate(bubbles):
        check_bubble(b, f"bubble[{i}]")
    return BUBBLE_SEPARATOR.join(bubbles)


def parse_bubbles(content: str) -> Tuple[str, ...]:
    """Split generated or serialised content back into bubbles: split on newlines, strip, drop empties."""
    return tuple(part.strip() for part in content.split(BUBBLE_SEPARATOR) if part.strip())


def check_turns(messages: Sequence[ChatMessage], *, last_role: Optional[Role] = None) -> None:
    """Strict alternation starting with ``user``; no system turns; optionally fixes the last role."""
    if not messages:
        raise TemplateError("at least one turn is required")
    for i, m in enumerate(messages):
        if not isinstance(m, ChatMessage):
            raise TemplateError(f"turn {i} is not a ChatMessage")
        if m.role is Role.SYSTEM:
            raise TemplateError("system turns are not accepted; the template owns the only system message")
        expected = Role.USER if i % 2 == 0 else Role.ASSISTANT
        if m.role is not expected:
            raise TemplateError(f"turn {i} should be {expected.value}, got {m.role.value} (roles must alternate)")
        serialize_bubbles(m.bubbles)
    if last_role is not None and messages[-1].role is not last_role:
        raise TemplateError(f"the last turn must be {last_role.value}")


def _system_content(spec: TemplateSpec, situation: Optional[str]) -> Optional[str]:
    if spec.situation_mode is SituationMode.NONE:
        if situation is not None:
            raise TemplateError(f"template {spec.name} does not condition on situation; got {situation!r}")
        return None
    if situation is None:
        raise TemplateError(f"template {spec.name} requires a situation label")
    require_id("situation", situation)
    return spec.situation_line(situation)


def to_hf_messages(messages: Sequence[ChatMessage], spec: TemplateSpec, situation: Optional[str] = None,
                   target: Optional[ChatMessage] = None) -> List[Dict[str, str]]:
    """The ``[{"role", "content"}]`` list to give ``apply_chat_template`` for the same rendering."""
    turns = list(messages) + ([target] if target is not None else [])
    check_turns(turns, last_role=Role.ASSISTANT if target is not None else Role.USER)
    out: List[Dict[str, str]] = []
    system = _system_content(spec, situation)
    if system is not None:
        out.append({"role": "system", "content": system})
    out.extend({"role": m.role.value, "content": serialize_bubbles(m.bubbles)} for m in turns)
    return out


def _render_hf(hf_messages: Sequence[Dict[str, str]], add_generation_prompt: bool) -> str:
    """The subset of the official Qwen3 template these messages can reach (no tools, no reasoning content)."""
    parts: List[str] = []
    last_user = max(i for i, m in enumerate(hf_messages) if m["role"] == "user")
    for i, m in enumerate(hf_messages):
        role, content = m["role"], m["content"]
        if role in ("system", "user"):
            parts.append(f"{IM_START}{role}\n{content}{IM_END}\n")
        elif i > last_user and i == len(hf_messages) - 1:
            parts.append(f"{IM_START}assistant\n{EMPTY_THINK}{content}{IM_END}\n")
        else:
            parts.append(f"{IM_START}assistant\n{content}{IM_END}\n")
    if add_generation_prompt:
        parts.append(f"{IM_START}assistant\n{EMPTY_THINK}")
    return "".join(parts)


@dataclass(frozen=True)
class RenderedExample:
    """A training example: ``prompt_text`` is context, ``target_text`` is the only supervised text."""

    prompt_text: str
    target_text: str

    @property
    def full_text(self) -> str:
        return self.prompt_text + self.target_text

    @property
    def target_start(self) -> int:
        """Character offset in ``full_text`` where supervision starts."""
        return len(self.prompt_text)


def render_prompt(messages: Sequence[ChatMessage], spec: TemplateSpec, situation: Optional[str] = None) -> str:
    """Inference prompt for a history that ends with the user turn to answer."""
    return _render_hf(to_hf_messages(messages, spec, situation), add_generation_prompt=True)


def render_training_example(messages: Sequence[ChatMessage], target: ChatMessage, spec: TemplateSpec,
                            situation: Optional[str] = None) -> RenderedExample:
    """Prompt and supervised target for ``target`` answering ``messages``."""
    if spec.persona_card:
        raise TemplateError(f"template {spec.name} {spec.version} is composed at inference and has no training format")
    full = _render_hf(to_hf_messages(messages, spec, situation, target), add_generation_prompt=False)
    prompt = render_prompt(messages, spec, situation)
    if not full.startswith(prompt):  # guards the renderer itself; cannot happen for valid input
        raise TemplateError("training text does not start with the inference prompt")
    return RenderedExample(prompt_text=prompt, target_text=full[len(prompt):])


def check_prefix(prompt_text: str, full_text: str) -> List[str]:
    """String-level check that a training text starts with the inference prompt. Returns problems found."""
    if full_text.startswith(prompt_text):
        return []
    n = next((i for i, (a, b) in enumerate(zip(prompt_text, full_text)) if a != b), min(len(prompt_text), len(full_text)))
    return [f"training text diverges from the inference prompt at character {n}: "
            f"prompt has {prompt_text[n:n + 24]!r}, training has {full_text[n:n + 24]!r}"]


# ---- token-level verification against a real tokenizer -------------------------------------------------

def _encode(tokenizer: Any, text: str) -> Tuple[int, ...]:
    return tuple(tokenizer(text, add_special_tokens=False)["input_ids"])


def _official_ids(tokenizer: Any, hf_messages: List[Dict[str, str]], add_generation_prompt: bool) -> Tuple[int, ...]:
    out = tokenizer.apply_chat_template(hf_messages, tokenize=True, add_generation_prompt=add_generation_prompt,
                                        enable_thinking=False)
    ids = out["input_ids"] if not isinstance(out, (list, tuple)) else out
    return tuple(int(i) for i in ids)


def compare_token_prefix(tokenizer: Any, prompt_text: str, full_text: str) -> List[str]:
    """Token-level check: the training text's ids start with the prompt's ids, and encoding prompt and
    target separately (as LLaMA-Factory does) gives the same ids as encoding the whole text."""
    issues = check_prefix(prompt_text, full_text)
    p, f = _encode(tokenizer, prompt_text), _encode(tokenizer, full_text)
    if f[:len(p)] != p:
        issues.append("token ids of the training text do not start with the token ids of the inference prompt")
    if full_text.startswith(prompt_text) and p + _encode(tokenizer, full_text[len(prompt_text):]) != f:
        issues.append("encoding prompt and target separately differs from encoding the whole text")
    return issues


@dataclass(frozen=True)
class TokenConsistencyReport:
    prompt_ids: Tuple[int, ...]
    target_ids: Tuple[int, ...]
    issues: Tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.issues


def verify_with_tokenizer(tokenizer: Any, messages: Sequence[ChatMessage], target: ChatMessage,
                          spec: TemplateSpec, situation: Optional[str] = None) -> TokenConsistencyReport:
    """Compare this renderer with the official ``apply_chat_template`` at string and token-id level."""
    issues: List[str] = []
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or hashlib.sha256(template.encode("utf-8")).hexdigest() != QWEN3_CHAT_TEMPLATE_SHA256:
        issues.append("tokenizer chat_template is not the pinned Qwen3 template")
    example = render_training_example(messages, target, spec, situation)
    hf_prompt = to_hf_messages(messages, spec, situation)
    hf_full = to_hf_messages(messages, spec, situation, target)
    official_prompt = tokenizer.apply_chat_template(hf_prompt, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False)
    official_full = tokenizer.apply_chat_template(hf_full, tokenize=False, add_generation_prompt=False,
                                                  enable_thinking=False)
    if official_prompt != example.prompt_text:
        issues.append("inference prompt text differs from apply_chat_template")
    if official_full != example.full_text:
        issues.append("training text differs from apply_chat_template")
    prompt_ids = _encode(tokenizer, example.prompt_text)
    full_ids = _encode(tokenizer, example.full_text)
    if _official_ids(tokenizer, hf_prompt, True) != prompt_ids:
        issues.append("inference prompt token ids differ from apply_chat_template(tokenize=True)")
    if _official_ids(tokenizer, hf_full, False) != full_ids:
        issues.append("training token ids differ from apply_chat_template(tokenize=True)")
    issues += compare_token_prefix(tokenizer, example.prompt_text, example.full_text)
    target_ids = full_ids[len(prompt_ids):]
    if target_ids[-2:] != (IM_END_ID, _encode(tokenizer, "\n")[0]):
        issues.append("supervised target does not end with <|im_end|> and a newline")
    return TokenConsistencyReport(prompt_ids=prompt_ids, target_ids=target_ids, issues=tuple(issues))
