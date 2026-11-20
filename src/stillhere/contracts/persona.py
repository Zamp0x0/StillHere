"""PersonaProfile: who the persona is, how it behaves in each situation, where that came from, and which version."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

from ._base import (SCHEMA_VERSION, Contract, ValidationError, require_id, require_semver, require_text,
                    require_unique)
from .common import DataOrigin, SourceRef


@dataclass(frozen=True)
class PersonaTrait(Contract):
    """A stable characteristic, with the sources that support it."""

    name: str
    description: str
    evidence: Tuple[str, ...] = ()  # source_ids

    def validate(self) -> None:
        require_text("PersonaTrait.name", self.name)
        require_text("PersonaTrait.description", self.description)


@dataclass(frozen=True)
class SituationBehavior(Contract):
    """How the persona typically responds in a labelled situation.

    The same situation labels condition fine-tuning samples and inference requests.
    """

    situation: str                         # label, e.g. "comfort", "small_talk", "disagreement"
    behavior: str                          # what the persona tends to do
    example_bubbles: Tuple[str, ...] = ()  # an illustrative multi-bubble reply
    evidence: Tuple[str, ...] = ()

    def validate(self) -> None:
        require_id("SituationBehavior.situation", self.situation)
        require_text("SituationBehavior.behavior", self.behavior)
        for i, b in enumerate(self.example_bubbles):
            require_text(f"SituationBehavior.example_bubbles[{i}]", b)


@dataclass(frozen=True)
class StyleProfile(Contract):
    languages: Tuple[str, ...]
    tone: str
    max_bubbles_per_reply: int = 3
    typical_phrases: Tuple[str, ...] = ()
    avoid: Tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.languages:
            raise ValidationError("StyleProfile.languages must not be empty")
        require_text("StyleProfile.tone", self.tone)
        if self.max_bubbles_per_reply < 1:
            raise ValidationError("StyleProfile.max_bubbles_per_reply must be >= 1")


@dataclass(frozen=True)
class PersonaProfile(Contract):
    persona_id: str
    version: str                 # MAJOR.MINOR.PATCH of this profile
    display_name: str
    origin: DataOrigin
    summary: str
    style: StyleProfile
    traits: Tuple[PersonaTrait, ...] = ()
    behaviors: Tuple[SituationBehavior, ...] = ()
    sources: Tuple[SourceRef, ...] = ()
    created_at: Optional[datetime] = None
    parent_version: Optional[str] = None
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValidationError(f"unsupported schema_version {self.schema_version!r}")
        require_id("persona_id", self.persona_id)
        require_semver("version", self.version)
        require_text("display_name", self.display_name)
        require_text("summary", self.summary)
        if self.parent_version is not None:
            require_semver("parent_version", self.parent_version)
            if self.parent_version == self.version:
                raise ValidationError("parent_version must differ from version")
        require_unique("sources.source_id", [s.source_id for s in self.sources])
        require_unique("traits.name", [t.name for t in self.traits])
        require_unique("behaviors.situation", [b.situation for b in self.behaviors])
        known = {s.source_id for s in self.sources}
        for owner, ids in [(f"trait {t.name!r}", t.evidence) for t in self.traits] + \
                          [(f"behavior {b.situation!r}", b.evidence) for b in self.behaviors]:
            missing = [i for i in ids if i not in known]
            if missing:
                raise ValidationError(f"{owner} cites unknown source(s): {', '.join(missing)}")
        for b in self.behaviors:
            if len(b.example_bubbles) > self.style.max_bubbles_per_reply:
                raise ValidationError(f"behavior {b.situation!r} example exceeds max_bubbles_per_reply")
        if self.origin is DataOrigin.SYNTHETIC:
            real = [s.source_id for s in self.sources if s.origin is not DataOrigin.SYNTHETIC]
            if real:
                raise ValidationError(f"synthetic persona cites non-synthetic source(s): {', '.join(real)}")

    @property
    def situations(self) -> Tuple[str, ...]:
        return tuple(b.situation for b in self.behaviors)

    def behavior_for(self, situation: str) -> Optional[SituationBehavior]:
        return next((b for b in self.behaviors if b.situation == situation), None)

    def source(self, source_id: str) -> Optional[SourceRef]:
        return next((s for s in self.sources if s.source_id == source_id), None)


# Fields of a profile that a persona card renders as one prompt line each; a line break would start a new line.
def _require_line(name: str, value: str) -> None:
    require_text(name, value)
    if "\n" in value or "\r" in value:
        raise ValidationError(f"{name} must be a single line")


@dataclass(frozen=True)
class PersonaCard(Contract):
    """Phase 5B: who is speaking, as a persona-card prompt (``qwen3_situation_nothink`` 0.2.0) renders it.

    Read from a ``PersonaProfile`` by ``from_profile``, so there is one persona configuration: nothing here is
    written by hand. ``name`` is what the persona calls herself (by default the first word of ``display_name``,
    the same rule the retriever uses for the persona's names). ``background`` is the profile's summary; it is
    background only, and dated facts still come from the memory block. ``behavior`` and ``example_bubbles`` are
    the profile's entry for the request's situation. ``ai_disclosure`` is what the persona says when the user
    sincerely asks whether they are talking to an AI.
    """

    persona_id: str
    persona_version: str
    name: str
    background: str
    traits: Tuple[str, ...] = ()
    tone: str = ""
    typical_phrases: Tuple[str, ...] = ()
    avoid: Tuple[str, ...] = ()
    situation: Optional[str] = None
    behavior: Optional[str] = None
    example_bubbles: Tuple[str, ...] = ()
    ai_disclosure: str = ""

    def validate(self) -> None:
        require_id("persona_id", self.persona_id)
        require_semver("persona_version", self.persona_version)
        _require_line("name", self.name)
        if any(c.isspace() for c in self.name):
            raise ValidationError("name must be one word")
        _require_line("background", self.background)
        _require_line("ai_disclosure", self.ai_disclosure)
        for label, items in (("traits", self.traits), ("typical_phrases", self.typical_phrases),
                             ("avoid", self.avoid), ("example_bubbles", self.example_bubbles)):
            for i, item in enumerate(items):
                _require_line(f"{label}[{i}]", item)
        if self.tone:
            _require_line("tone", self.tone)
        if self.behavior is not None:
            _require_line("behavior", self.behavior)
            if self.situation is None:
                raise ValidationError("a behavior belongs to a situation")
        if self.situation is not None:
            require_id("situation", self.situation)

    @classmethod
    def from_profile(cls, profile: PersonaProfile, situation: Optional[str] = None, *,
                     name: Optional[str] = None, ai_disclosure: Optional[str] = None) -> "PersonaCard":
        name = name or profile.display_name.split()[0]
        behavior = profile.behavior_for(situation) if situation is not None else None
        return cls(persona_id=profile.persona_id, persona_version=profile.version, name=name,
                   background=profile.summary, traits=tuple(t.description for t in profile.traits),
                   tone=profile.style.tone, typical_phrases=profile.style.typical_phrases, avoid=profile.style.avoid,
                   situation=situation, behavior=behavior.behavior if behavior is not None else None,
                   example_bubbles=behavior.example_bubbles if behavior is not None else (),
                   ai_disclosure=ai_disclosure or f"我是 AI，在按{name}这个虚构角色和你聊天，不是真人。")
