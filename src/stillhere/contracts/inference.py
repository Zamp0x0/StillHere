"""InferenceRequest / InferenceResponse: everything a backend needs to produce one multi-bubble reply.

The request is self-contained: persona reference, chat history, retrieved memories, model, adapter,
template and generation settings. A backend implements ``stillhere.contracts.protocols.InferenceBackend``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import ClassVar, Optional, Tuple

from ._base import (SCHEMA_VERSION, Contract, ValidationError, require_id, require_semver, require_text,
                    require_unique)
from .common import AdapterRef, ChatMessage, ModelRef, PromptTemplateRef, Role
from .grounding import AnswerNote
from .memory import MemoryStatus, RetrievedMemory
from .persona import PersonaCard
from .retrieval import MemoryContextView

MAX_NOTES = 12


@dataclass(frozen=True)
class GenerationConfig(Contract):
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    seed: Optional[int] = None
    max_bubbles: int = 3
    enable_thinking: bool = False  # Qwen3 thinking mode; off for persona chat

    def validate(self) -> None:
        if self.max_new_tokens < 1:
            raise ValidationError("max_new_tokens must be >= 1")
        if self.temperature < 0:
            raise ValidationError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValidationError("top_p must be within (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValidationError("repetition_penalty must be > 0")
        if self.seed is not None and self.seed < 0:
            raise ValidationError("seed must be >= 0")
        if self.max_bubbles < 1:
            raise ValidationError("max_bubbles must be >= 1")


@dataclass(frozen=True)
class InferenceRequest(Contract):
    request_id: str
    persona_id: str
    persona_version: str
    messages: Tuple[ChatMessage, ...]   # history, ending with the user turn to answer
    base_model: ModelRef
    prompt_template: PromptTemplateRef
    adapter: Optional[AdapterRef] = None  # None = base model only
    situation: Optional[str] = None       # situation label used for conditioning
    memory_context: Tuple[RetrievedMemory, ...] = ()
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    created_at: Optional[datetime] = None
    schema_version: str = SCHEMA_VERSION
    # How memory_context is to be read (memory_block_v1). None: the plain memory_block_v0 list, or no memory.
    memory_view: Optional[MemoryContextView] = None
    # Phase 5B: who is speaking, for a persona-card template (qwen3_situation_nothink 0.2.0), and the answer
    # notes of a regenerated reply (closed codes, see contracts.grounding). Both absent from earlier requests.
    persona_card: Optional[PersonaCard] = None
    notes: Tuple[AnswerNote, ...] = ()
    _omit_when_empty: ClassVar[Tuple[str, ...]] = ("persona_card", "notes")

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValidationError(f"unsupported schema_version {self.schema_version!r}")
        require_id("request_id", self.request_id)
        require_id("persona_id", self.persona_id)
        require_semver("persona_version", self.persona_version)
        if self.situation is not None:
            require_id("situation", self.situation)
        if not self.messages:
            raise ValidationError("messages must not be empty")
        if self.messages[-1].role is not Role.USER:
            raise ValidationError("the last message must be a user turn")
        if any(m.role is Role.SYSTEM for m in self.messages[1:]):
            raise ValidationError("a system message may only appear first")
        for rm in self.memory_context:
            if rm.record.persona_id != self.persona_id:
                raise ValidationError(f"memory {rm.record.memory_id} belongs to another persona")
        require_unique("memory_context ids", self.memory_context_ids)
        if self.memory_view is not None:
            if self.memory_context_ids != self.memory_view.context_ids:
                raise ValidationError("memory_context must hold exactly the view's current, past and unverified "
                                      "facts, in view order")
            for s in self.memory_view.slots:
                for f in s.facts:
                    if f.record is not None and f.record.persona_id != self.persona_id:
                        raise ValidationError(f"memory {f.memory_id} belongs to another persona")
            for rm in self.memory_context:
                if rm.record.status is not MemoryStatus.ACTIVE:
                    raise ValidationError(f"memory {rm.record.memory_id} is not active and cannot be shown as a fact")
        if self.persona_card is not None and (self.persona_card.persona_id, self.persona_card.persona_version) != \
                (self.persona_id, self.persona_version):
            raise ValidationError("persona_card belongs to another persona or persona version")
        if self.persona_card is not None and self.persona_card.situation != self.situation:
            raise ValidationError("persona_card was read for another situation")
        if len(self.notes) > MAX_NOTES:
            raise ValidationError(f"at most {MAX_NOTES} answer notes")
        require_unique("notes", [n.to_json(indent=None) for n in self.notes])
        if self.adapter is not None:
            if self.adapter.base_model.model_id != self.base_model.model_id:
                raise ValidationError("adapter was trained on a different base model")
            if self.adapter.prompt_template.name != self.prompt_template.name:
                raise ValidationError("adapter was trained with a different prompt template")

    @property
    def memory_context_ids(self) -> Tuple[str, ...]:
        return tuple(rm.record.memory_id for rm in self.memory_context)

    @property
    def history(self) -> Tuple[ChatMessage, ...]:
        return self.messages[:-1]

    @property
    def latest_user_message(self) -> ChatMessage:
        return self.messages[-1]


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    ERROR = "error"


@dataclass(frozen=True)
class TokenUsage(Contract):
    prompt_tokens: int
    completion_tokens: int

    def validate(self) -> None:
        if self.prompt_tokens < 0 or self.completion_tokens < 0:
            raise ValidationError("token counts must be >= 0")


@dataclass(frozen=True)
class InferenceResponse(Contract):
    request_id: str
    bubbles: Tuple[str, ...]
    finish_reason: FinishReason = FinishReason.STOP
    used_memory_ids: Tuple[str, ...] = ()  # memories the reply relied on (provenance for the answer)
    usage: Optional[TokenUsage] = None
    latency_ms: Optional[float] = None
    raw_text: Optional[str] = None         # model output before bubble splitting, for debugging
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValidationError(f"unsupported schema_version {self.schema_version!r}")
        require_id("request_id", self.request_id)
        if self.finish_reason is FinishReason.ERROR:
            require_text("error", self.error or "")
        else:
            if self.error is not None:
                raise ValidationError("error is only allowed when finish_reason is 'error'")
            if not self.bubbles:
                raise ValidationError("a successful response needs at least one bubble")
        for i, b in enumerate(self.bubbles):
            require_text(f"bubbles[{i}]", b)
        require_unique("used_memory_ids", self.used_memory_ids)
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValidationError("latency_ms must be >= 0")

    def check_against(self, request: InferenceRequest) -> None:
        """Cross-check a response with the request that produced it."""
        if self.request_id != request.request_id:
            raise ValidationError("response.request_id does not match the request")
        if len(self.bubbles) > request.generation.max_bubbles:
            raise ValidationError(f"{len(self.bubbles)} bubbles exceed max_bubbles={request.generation.max_bubbles}")
        unknown = set(self.used_memory_ids) - set(request.memory_context_ids)
        if unknown:
            raise ValidationError(f"response cites memories not in the request context: {', '.join(sorted(unknown))}")

    def as_message(self, created_at: Optional[datetime] = None) -> ChatMessage:
        """The reply as an assistant turn, ready to append to the next request's history."""
        return ChatMessage(role=Role.ASSISTANT, bubbles=self.bubbles, created_at=created_at)
