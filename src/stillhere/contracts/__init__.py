"""Shared data contracts and service interfaces for every StillHere module.

This package is the only module every other module may import. It depends on the standard library only.
Status: v0.1 contracts, implemented and tested. ``retrieval`` (Day 5) adds the temporal retrieval result
and the memory context view a request carries; ``planning`` (Day 7) the query plan of a message, its atomic
queries and their outcomes; ``grounding`` (Phase 5B) the evidence state of each atom and the output guard's
audit record; ``evaluation`` (Day 6) adds the rubric, protocol, score, failure
attribution and human review records (import them from ``stillhere.contracts.evaluation``);
the services behind the protocols live in their own modules (see ``protocols``).
"""
from ._base import SCHEMA_VERSION, Contract, ValidationError, canonical_json
from .common import (AdapterRef, ChatMessage, DataOrigin, ModelRef, PromptTemplateRef, Provenance,
                     PublicBoundaryError, Role, SourceKind, SourceRef, assert_public_safe, find_non_synthetic)
from .grounding import (AnswerNote, AtomEvidence, AtomGrounding, DependencyGrounding, FallbackAudit, GroundingIssue,
                        GuardAttempt, GuardOutcome, GuardReport, TurnGrounding, TurnKind)
from .experiment import (ArtifactRef, DatasetRef, DatasetSplit, EvaluationConfig, ExperimentKind,
                         ExperimentManifest, ExperimentStatus, LossMasking, MetricResult, TrainingConfig)
from .inference import FinishReason, GenerationConfig, InferenceRequest, InferenceResponse, TokenUsage
from .memory import Cardinality, MemoryKind, MemoryRecord, MemoryStatus, RetrievedMemory
from .messages import (MESSAGE_LOCATOR_PREFIX, CanonicalMessage, Citation, EvidenceBundle, EvidenceHit, EvidenceItem, EvidencePack,
                       HitKind, IngestResult, SnapshotInfo, message_locator)
from .persona import PersonaCard, PersonaProfile, PersonaTrait, SituationBehavior, StyleProfile
from .protocols import InferenceBackend, MemoryRetriever, MemoryStore, PersonaRepository
from .planning import (AtomicQuery, AtomicResult, BatchStatus, DialogueFocus, IntentStatus, QueryDependency,
                       QueryPart, QueryPlan, RetrievalBatch, SpeechAct, SubjectStatus, Sufficiency, TimeExpression,
                       TimeKind)
from .retrieval import (ContextDependency, ContextFact, ContextQuery, ContextSlot, FactRole, MemoryContextView,
                        MemoryHit, MemoryQuery, RetrievalResult, RetrievalStatus, SlotResolution, SlotStatus,
                        TemporalRole, TimeScope)

__all__ = [
    "SCHEMA_VERSION", "Contract", "ValidationError", "canonical_json",
    "AdapterRef", "ChatMessage", "DataOrigin", "ModelRef", "PromptTemplateRef", "Provenance",
    "PublicBoundaryError", "Role", "SourceKind", "SourceRef", "assert_public_safe", "find_non_synthetic",
    "ArtifactRef", "DatasetRef", "DatasetSplit", "EvaluationConfig", "ExperimentKind", "ExperimentManifest",
    "ExperimentStatus", "LossMasking", "MetricResult", "TrainingConfig",
    "FinishReason", "GenerationConfig", "InferenceRequest", "InferenceResponse", "TokenUsage",
    "Cardinality", "MemoryKind", "MemoryRecord", "MemoryStatus", "RetrievedMemory",
    "MESSAGE_LOCATOR_PREFIX", "CanonicalMessage", "Citation", "EvidenceBundle", "EvidenceHit", "EvidenceItem", "EvidencePack", "HitKind",
    "IngestResult", "SnapshotInfo", "message_locator",
    "PersonaCard", "PersonaProfile", "PersonaTrait", "SituationBehavior", "StyleProfile",
    "AnswerNote", "AtomEvidence", "AtomGrounding", "DependencyGrounding", "FallbackAudit", "GroundingIssue",
    "GuardAttempt",
    "GuardOutcome", "GuardReport", "TurnGrounding", "TurnKind",
    "InferenceBackend", "MemoryRetriever", "MemoryStore", "PersonaRepository",
    "MemoryHit", "MemoryQuery", "RetrievalResult", "RetrievalStatus", "SlotResolution", "SlotStatus",
    "TemporalRole", "TimeScope", "ContextFact", "ContextSlot", "FactRole", "MemoryContextView",
    "ContextQuery", "ContextDependency", "AtomicQuery", "AtomicResult", "BatchStatus", "DialogueFocus",
    "IntentStatus", "QueryDependency", "QueryPart", "QueryPlan", "RetrievalBatch", "SpeechAct", "SubjectStatus",
    "Sufficiency", "TimeExpression", "TimeKind",
]
