"""Evaluation scoring contracts (Day 6): rubric, protocol, scores, objective checks, failures, human reviews.

What a scored evaluation of one generation looks like on disk, kept apart from the generation record itself
(``evaluation.recorder.BaselineRecord``), which a score only points to (``EvidenceRef``).

Rules the contracts enforce:

* A dimension score is a number 0-4 only when its status is ``scored``; every other status (not applicable,
  not evaluated, generation failed, truncated, invalid output, review pending) carries no number, so a missing
  score can never be read as a zero.
* Dimension scores are judgements (``human``, or a future ``llm_judge`` the protocol must enable). Deterministic
  code results are ``ObjectiveCheckResult`` rows, never dimension scores.
* A dimension score is the minimum of its criterion scores (``AGGREGATION``): a reply that breaks persona
  identity cannot reach a high persona score through good style.
* A failure is ``confirmed`` only with record, trace or review evidence; a retrieval failure needs the trace, and
  no reviewer can confirm a retrieval or context failure from the reply. Otherwise the attribution is
  ``suspected`` or ``insufficient_evidence``.
* A human review names an anonymous reviewer and a blind response id, never the case's arm.

Standard library only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from ._base import Contract, ValidationError, require_id, require_semver, require_sha256, require_text, require_unique
from .common import ModelRef

SCORE_MIN, SCORE_MAX = 0, 4
AGGREGATION = "min_of_criteria"
_REVIEWER_RE = re.compile(r"^rev_[a-z0-9]{2,16}$")
_BLIND_RE = re.compile(r"^blind_[0-9a-f]{12}$")


class EvalDimension(str, Enum):
    PERSONA_FIDELITY = "persona_fidelity"
    SITUATION_AWARENESS = "situation_awareness"
    MEMORY_FAITHFULNESS = "memory_faithfulness"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    UNSUPPORTED_FACT = "unsupported_fact"
    MULTITURN_CONTINUITY = "multiturn_continuity"


class ScoreStatus(str, Enum):
    SCORED = "scored"
    NOT_APPLICABLE = "not_applicable"      # the dimension does not apply to this case; never in a denominator
    NOT_EVALUATED = "not_evaluated"        # applies, but nobody scored it (e.g. the arm was not run)
    GENERATION_FAILED = "generation_failed"
    TRUNCATED = "truncated"                # cut at max_new_tokens; not scored in protocol 1.x
    INVALID_OUTPUT = "invalid_output"      # violates the request (too many bubbles)
    REVIEW_PENDING = "review_pending"      # waiting for a human review


# Applicable but without a score: counted as missing, never as zero.
MISSING_STATUSES = frozenset(s for s in ScoreStatus if s not in (ScoreStatus.SCORED, ScoreStatus.NOT_APPLICABLE))


class ScoringMethod(str, Enum):
    OBJECTIVE = "objective"
    HUMAN = "human"
    LLM_JUDGE = "llm_judge"


class CheckOutcome(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"
    NOT_EVALUATED = "not_evaluated"        # applies, but the evidence it needs is not there (e.g. no trace)
    ERROR = "error"                        # the scorer could not run on this input


class GenerationStatus(str, Enum):
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    INVALID_OUTPUT = "invalid_output"
    GENERATION_FAILED = "generation_failed"
    REJECTED = "rejected"                  # refused before generation (input over budget)
    NOT_RUN = "not_run"


class FailureCategory(str, Enum):
    RETRIEVAL = "retrieval"
    TEMPORAL = "temporal"
    CONTEXT = "context"
    GENERATION = "generation"
    EXECUTION = "execution"


class FailureTag(str, Enum):
    NO_RELEVANT_MEMORY = "NO_RELEVANT_MEMORY"            # a state, not a failure (see NORMAL_STATES)
    WRONG_SLOT = "WRONG_SLOT"
    WRONG_SUBJECT = "WRONG_SUBJECT"
    MISSED_RELEVANT_MEMORY = "MISSED_RELEVANT_MEMORY"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
    STALE_FACT_USED = "STALE_FACT_USED"
    CORRECTED_FACT_TREATED_AS_HISTORY = "CORRECTED_FACT_TREATED_AS_HISTORY"
    CONFLICT_RESOLVED_WITHOUT_EVIDENCE = "CONFLICT_RESOLVED_WITHOUT_EVIDENCE"
    WRONG_TIME_SCOPE = "WRONG_TIME_SCOPE"
    RELEVANT_MEMORY_NOT_INJECTED = "RELEVANT_MEMORY_NOT_INJECTED"
    MEMORY_CONTEXT_TRUNCATED = "MEMORY_CONTEXT_TRUNCATED"
    CONFLICT_GROUP_SPLIT = "CONFLICT_GROUP_SPLIT"
    UNSAFE_MEMORY_INJECTION = "UNSAFE_MEMORY_INJECTION"
    CONTEXT_OVER_BUDGET = "CONTEXT_OVER_BUDGET"
    UNSUPPORTED_PERSONAL_CLAIM = "UNSUPPORTED_PERSONAL_CLAIM"
    CONTRADICTED_MEMORY = "CONTRADICTED_MEMORY"
    PERSONA_DRIFT = "PERSONA_DRIFT"
    INAPPROPRIATE_SITUATION_RESPONSE = "INAPPROPRIATE_SITUATION_RESPONSE"
    MULTITURN_INCONSISTENCY = "MULTITURN_INCONSISTENCY"
    GENERATION_FAILED = "GENERATION_FAILED"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    TRUNCATED = "TRUNCATED"
    SCORER_ERROR = "SCORER_ERROR"
    REVIEW_PENDING = "REVIEW_PENDING"
    UNKNOWN_FAILURE = "UNKNOWN_FAILURE"                  # something went wrong; the evidence does not say what


_C = FailureCategory
_T = FailureTag
FAILURE_CATEGORY: Dict[FailureTag, FailureCategory] = {
    **{t: _C.RETRIEVAL for t in (_T.NO_RELEVANT_MEMORY, _T.WRONG_SLOT, _T.WRONG_SUBJECT, _T.MISSED_RELEVANT_MEMORY,
                                 _T.RETRIEVAL_ERROR)},
    **{t: _C.TEMPORAL for t in (_T.STALE_FACT_USED, _T.CORRECTED_FACT_TREATED_AS_HISTORY,
                                _T.CONFLICT_RESOLVED_WITHOUT_EVIDENCE, _T.WRONG_TIME_SCOPE)},
    **{t: _C.CONTEXT for t in (_T.RELEVANT_MEMORY_NOT_INJECTED, _T.MEMORY_CONTEXT_TRUNCATED, _T.CONFLICT_GROUP_SPLIT,
                               _T.UNSAFE_MEMORY_INJECTION, _T.CONTEXT_OVER_BUDGET)},
    **{t: _C.GENERATION for t in (_T.UNSUPPORTED_PERSONAL_CLAIM, _T.CONTRADICTED_MEMORY, _T.PERSONA_DRIFT,
                                  _T.INAPPROPRIATE_SITUATION_RESPONSE, _T.MULTITURN_INCONSISTENCY)},
    **{t: _C.EXECUTION for t in (_T.GENERATION_FAILED, _T.INVALID_OUTPUT, _T.TRUNCATED, _T.SCORER_ERROR,
                                 _T.REVIEW_PENDING, _T.UNKNOWN_FAILURE)},
}
NORMAL_STATES = frozenset({FailureTag.NO_RELEVANT_MEMORY})
# What happened before generation. A reviewer reading the reply can never confirm these; retrieval failures need the
# turn's trace, context failures the trace or the stored request (which shows what was injected).
SYSTEM_SIDE = frozenset(t for t, c in FAILURE_CATEGORY.items() if c in (_C.RETRIEVAL, _C.CONTEXT))
TRACE_REQUIRED = frozenset(t for t, c in FAILURE_CATEGORY.items() if c is _C.RETRIEVAL)


class AttributionLevel(str, Enum):
    CONFIRMED = "confirmed"                        # record or trace evidence shows it
    SUSPECTED = "suspected"                        # an output rule or a reviewer saw it; the cause is not proven
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"


class EvidenceKind(str, Enum):
    RECORD = "record"      # a generation record (run id + sample id)
    TRACE = "trace"        # a TurnTrace
    CASE = "case"          # the evaluation case definition
    CHECK = "check"        # an objective check result
    REVIEW = "review"      # a human review record


def require_score(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not SCORE_MIN <= value <= SCORE_MAX:
        raise ValidationError(f"{name} must be an integer {SCORE_MIN}-{SCORE_MAX}, got {value!r}")


def require_reviewer(name: str, value: str) -> None:
    if not isinstance(value, str) or not _REVIEWER_RE.match(value):
        raise ValidationError(f"{name} must be an anonymous reviewer id like 'rev_a1', got {value!r}")


@dataclass(frozen=True)
class EvidenceRef(Contract):
    kind: EvidenceKind
    ref: str
    sha256: Optional[str] = None

    def validate(self) -> None:
        require_text("ref", self.ref)
        if self.sha256 is not None:
            require_sha256("sha256", self.sha256)


# ---- rubric and protocol ----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RubricAnchor(Contract):
    score: int
    description: str

    def validate(self) -> None:
        require_score("score", self.score)
        require_text("description", self.description)


@dataclass(frozen=True)
class RubricCriterion(Contract):
    criterion_id: str
    name: str
    anchors: Tuple[RubricAnchor, ...]

    def validate(self) -> None:
        require_id("criterion_id", self.criterion_id)
        require_text("name", self.name)
        if [a.score for a in self.anchors] != list(range(SCORE_MIN, SCORE_MAX + 1)):
            raise ValidationError(f"{self.criterion_id}: anchors must cover scores {SCORE_MIN}-{SCORE_MAX} in order")


@dataclass(frozen=True)
class RubricDimension(Contract):
    dimension: EvalDimension
    definition: str
    measures: str
    evidence_required: Tuple[str, ...]
    applicable_when: str
    not_applicable_when: str
    method: ScoringMethod
    criteria: Tuple[RubricCriterion, ...]
    failure_conditions: Tuple[str, ...]
    failure_tags: Tuple[FailureTag, ...]
    positive_example: str
    negative_example: str
    limitations: Tuple[str, ...]

    def validate(self) -> None:
        for name in ("definition", "measures", "applicable_when", "not_applicable_when", "positive_example",
                     "negative_example"):
            require_text(name, getattr(self, name))
        if self.method is not ScoringMethod.HUMAN:
            raise ValidationError(f"{self.dimension.value}: dimensions are human-scored in this schema")
        for name in ("evidence_required", "criteria", "failure_conditions", "limitations"):
            if not getattr(self, name):
                raise ValidationError(f"{self.dimension.value}: {name} must not be empty")
        require_unique(f"{self.dimension.value} criteria", [c.criterion_id for c in self.criteria])

    @property
    def criterion_ids(self) -> Tuple[str, ...]:
        return tuple(c.criterion_id for c in self.criteria)


@dataclass(frozen=True)
class Rubric(Contract):
    rubric_id: str
    version: str
    scale: Dict[str, Any]
    aggregation: str
    statuses: Dict[str, Any]           # ScoreStatus value -> when to use it
    dimensions: Tuple[RubricDimension, ...]

    def validate(self) -> None:
        require_id("rubric_id", self.rubric_id)
        require_semver("version", self.version)
        if dict(self.scale) != {"min": SCORE_MIN, "max": SCORE_MAX}:
            raise ValidationError(f"scale must be {{'min': {SCORE_MIN}, 'max': {SCORE_MAX}}}")
        if self.aggregation != AGGREGATION:
            raise ValidationError(f"aggregation must be {AGGREGATION!r}")
        if set(self.statuses) != {s.value for s in ScoreStatus}:
            raise ValidationError("statuses must define every ScoreStatus")
        if sorted(d.dimension.value for d in self.dimensions) != sorted(d.value for d in EvalDimension):
            raise ValidationError("the rubric must define each of the six dimensions exactly once")

    def dimension(self, dimension: EvalDimension) -> RubricDimension:
        return next(d for d in self.dimensions if d.dimension is dimension)


@dataclass(frozen=True)
class EvaluationArm(Contract):
    arm_id: str
    label: str
    base_model: ModelRef
    memory_enabled: bool
    adapter_id: Optional[str] = None
    adapter_sha256: Optional[str] = None

    def validate(self) -> None:
        require_id("arm_id", self.arm_id)
        require_text("label", self.label)
        if self.adapter_sha256 is not None:
            if self.adapter_id is None:
                raise ValidationError("adapter_sha256 needs adapter_id")
            require_sha256("adapter_sha256", self.adapter_sha256)


@dataclass(frozen=True)
class EvaluationProtocol(Contract):
    protocol_id: str
    version: str
    rubric_id: str
    rubric_version: str
    arms: Tuple[EvaluationArm, ...]
    comparisons: Tuple[Tuple[str, ...], ...]     # (treatment arm, reference arm, what the difference isolates)
    objective_checks: Tuple[str, ...]
    controls: Dict[str, Any]                       # everything held fixed across arms
    dataset_roles: Dict[str, Any]                  # development / validation / holdout and what each may be used for
    statistics: Dict[str, Any]
    llm_judge: str
    known_limitations: Tuple[str, ...]

    def validate(self) -> None:
        require_id("protocol_id", self.protocol_id)
        require_semver("version", self.version)
        require_id("rubric_id", self.rubric_id)
        require_semver("rubric_version", self.rubric_version)
        if not self.arms:
            raise ValidationError("a protocol needs at least one arm")
        ids = [a.arm_id for a in self.arms]
        require_unique("arm ids", ids)
        for c in self.comparisons:
            if len(c) != 3 or c[0] not in ids or c[1] not in ids or c[0] == c[1]:
                raise ValidationError(f"comparison {c!r} must be (arm, other arm, description)")
        require_unique("objective_checks", self.objective_checks)
        for check in self.objective_checks:
            require_id("objective check", check)
        if self.llm_judge != "disabled":
            raise ValidationError("llm_judge must be 'disabled' until a judge model, rubric, cost and data permission "
                                  "are chosen in a new protocol version")

    def arm(self, arm_id: str) -> EvaluationArm:
        for a in self.arms:
            if a.arm_id == arm_id:
                return a
        raise ValidationError(f"unknown arm {arm_id!r}")


# ---- results ----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ObjectiveCheckResult(Contract):
    check_id: str
    outcome: CheckOutcome
    detail: str = ""
    heuristic: bool = False            # a text rule (keyword match): a hint for reviewers, not a fact verdict
    evidence: Tuple[EvidenceRef, ...] = ()

    def validate(self) -> None:
        require_id("check_id", self.check_id)
        if self.outcome is CheckOutcome.ERROR and not self.detail:
            raise ValidationError(f"{self.check_id}: an error outcome needs a detail")


@dataclass(frozen=True)
class DimensionScore(Contract):
    dimension: EvalDimension
    status: ScoreStatus
    rubric_version: str
    method: Optional[ScoringMethod] = None
    score: Optional[int] = None
    criterion_scores: Dict[str, Any] = field(default_factory=dict)
    reviewer_id: Optional[str] = None
    rationale: str = ""

    def validate(self) -> None:
        require_semver("rubric_version", self.rubric_version)
        if self.reviewer_id is not None:
            require_reviewer("reviewer_id", self.reviewer_id)
        if self.status is not ScoreStatus.SCORED:
            if self.score is not None or self.criterion_scores:
                raise ValidationError(f"{self.dimension.value}: only a scored dimension carries numbers "
                                      f"(status {self.status.value})")
            return
        if self.method not in (ScoringMethod.HUMAN, ScoringMethod.LLM_JUDGE):
            raise ValidationError(f"{self.dimension.value}: a dimension score is a judgement (human or llm_judge); "
                                  "deterministic results are ObjectiveCheckResult rows")
        if self.method is ScoringMethod.HUMAN and self.reviewer_id is None:
            raise ValidationError(f"{self.dimension.value}: a human score needs reviewer_id")
        if not self.criterion_scores:
            raise ValidationError(f"{self.dimension.value}: a scored dimension needs its criterion scores")
        for name, value in self.criterion_scores.items():
            require_id("criterion", name)
            require_score(f"{self.dimension.value}.{name}", value)
        require_score(f"{self.dimension.value}.score", self.score)
        if self.score != min(self.criterion_scores.values()):
            raise ValidationError(f"{self.dimension.value}: score must be the minimum of its criterion scores "
                                  f"({AGGREGATION})")


@dataclass(frozen=True)
class FailureAttribution(Contract):
    tag: FailureTag
    level: AttributionLevel
    source: ScoringMethod
    evidence: Tuple[EvidenceRef, ...] = ()
    note: str = ""

    @property
    def category(self) -> FailureCategory:
        return FAILURE_CATEGORY[self.tag]

    @property
    def is_failure(self) -> bool:
        return self.tag not in NORMAL_STATES

    def validate(self) -> None:
        if self.level is AttributionLevel.CONFIRMED:
            kinds = {e.kind for e in self.evidence}
            if self.tag in TRACE_REQUIRED and EvidenceKind.TRACE not in kinds:
                raise ValidationError(f"{self.tag.value}: confirming a {self.category.value} failure needs trace evidence")
            if not kinds & {EvidenceKind.TRACE, EvidenceKind.RECORD, EvidenceKind.REVIEW}:
                raise ValidationError(f"{self.tag.value}: a confirmed attribution needs record, trace or review evidence")
            if self.source is ScoringMethod.HUMAN and self.tag in SYSTEM_SIDE:
                raise ValidationError(f"{self.tag.value}: a reviewer reading the reply cannot confirm a "
                                      f"{self.category.value} failure")
        if self.level is AttributionLevel.INSUFFICIENT_EVIDENCE and self.tag is not FailureTag.UNKNOWN_FAILURE \
                and not self.note:
            raise ValidationError(f"{self.tag.value}: say what evidence is missing (note)")


@dataclass(frozen=True)
class HumanReviewRecord(Contract):
    """One reviewer's scores for one blind response. It names neither the case's arm nor its model."""

    review_id: str
    blind_id: str
    reviewer_id: str
    protocol_version: str
    rubric_version: str
    status: ReviewStatus
    scores: Tuple[DimensionScore, ...] = ()
    failures: Tuple[FailureAttribution, ...] = ()
    submitted_at: Optional[datetime] = None
    notes: str = ""

    def validate(self) -> None:
        require_id("review_id", self.review_id)
        if not _BLIND_RE.match(self.blind_id):
            raise ValidationError(f"blind_id must look like 'blind_<12 hex>', got {self.blind_id!r}")
        require_reviewer("reviewer_id", self.reviewer_id)
        require_semver("protocol_version", self.protocol_version)
        require_semver("rubric_version", self.rubric_version)
        require_unique("reviewed dimensions", [s.dimension for s in self.scores])
        if self.status is ReviewStatus.PENDING:
            if self.scores or self.failures or self.submitted_at is not None:
                raise ValidationError("a pending review carries no scores, failures or submission time")
            return
        if self.submitted_at is None or not self.scores:
            raise ValidationError("a submitted review needs submitted_at and at least one dimension")
        for s in self.scores:
            if s.status not in (ScoreStatus.SCORED, ScoreStatus.NOT_APPLICABLE):
                raise ValidationError(f"{s.dimension.value}: a reviewer scores a dimension or marks it not applicable")
            if s.status is ScoreStatus.SCORED and (s.method is not ScoringMethod.HUMAN or s.reviewer_id != self.reviewer_id):
                raise ValidationError(f"{s.dimension.value}: a review holds this reviewer's human scores only")
            if s.rubric_version != self.rubric_version:
                raise ValidationError(f"{s.dimension.value}: rubric version differs from the review's")
        if any(f.source is not ScoringMethod.HUMAN for f in self.failures):
            raise ValidationError("a review records human failure tags only")


@dataclass(frozen=True)
class CaseEvaluationResult(Contract):
    """Every dimension of one case under one arm: scores or the reason there is none, checks and failures."""

    protocol_version: str
    rubric_version: str
    case_id: str
    dataset_id: str
    dataset_version: str
    arm_id: str
    generation_status: GenerationStatus
    dimension_scores: Tuple[DimensionScore, ...]
    objective_checks: Tuple[ObjectiveCheckResult, ...] = ()
    failures: Tuple[FailureAttribution, ...] = ()
    record: Optional[EvidenceRef] = None
    blind_id: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    _NO_OUTPUT = {GenerationStatus.GENERATION_FAILED: ScoreStatus.GENERATION_FAILED,
                  GenerationStatus.REJECTED: ScoreStatus.GENERATION_FAILED,
                  GenerationStatus.NOT_RUN: ScoreStatus.NOT_EVALUATED,
                  GenerationStatus.TRUNCATED: ScoreStatus.TRUNCATED,
                  GenerationStatus.INVALID_OUTPUT: ScoreStatus.INVALID_OUTPUT}

    def validate(self) -> None:
        require_semver("protocol_version", self.protocol_version)
        require_semver("rubric_version", self.rubric_version)
        for name in ("case_id", "dataset_id", "arm_id"):
            require_id(name, getattr(self, name))
        require_semver("dataset_version", self.dataset_version)
        if sorted(s.dimension.value for s in self.dimension_scores) != sorted(d.value for d in EvalDimension):
            raise ValidationError(f"{self.case_id}/{self.arm_id}: every dimension appears exactly once "
                                  "(not applicable ones included)")
        require_unique("objective check ids", [c.check_id for c in self.objective_checks])
        if self.blind_id is not None and not _BLIND_RE.match(self.blind_id):
            raise ValidationError(f"blind_id must look like 'blind_<12 hex>', got {self.blind_id!r}")
        if (self.record is None) != (self.generation_status is GenerationStatus.NOT_RUN):
            raise ValidationError("a result points to its generation record exactly when the arm ran")
        if self.record is not None and self.record.kind is not EvidenceKind.RECORD:
            raise ValidationError("record must be an EvidenceRef of kind 'record'")
        forced = self._NO_OUTPUT.get(self.generation_status)
        for s in self.dimension_scores:
            if s.rubric_version != self.rubric_version:
                raise ValidationError(f"{s.dimension.value}: rubric version differs from the result's")
            if s.status is ScoreStatus.NOT_APPLICABLE:
                continue
            if forced is not None and s.status is not forced:
                raise ValidationError(f"{self.case_id}/{self.arm_id}: generation {self.generation_status.value} means "
                                      f"applicable dimensions are {forced.value}, {s.dimension.value} is {s.status.value}")
            if forced is None and s.status in self._NO_OUTPUT.values() and s.status is not ScoreStatus.NOT_EVALUATED:
                raise ValidationError(f"{s.dimension.value}: status {s.status.value} contradicts a completed generation")

    def score(self, dimension: EvalDimension) -> DimensionScore:
        return next(s for s in self.dimension_scores if s.dimension is dimension)
