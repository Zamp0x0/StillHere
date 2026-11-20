"""ExperimentManifest: the reproducibility record of one training or evaluation run.

It pins the base model, dataset hashes, adapter, prompt template, seed and configuration. Results may only
be attached once the run is ``completed``, so a planned manifest can never carry numbers.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from ._base import (SCHEMA_VERSION, Contract, ValidationError, canonical_json, require_id, require_sha256,
                    require_text, require_unique)
from .common import AdapterRef, DataOrigin, ModelRef, PromptTemplateRef


class ExperimentKind(str, Enum):
    TRAINING = "training"
    EVALUATION = "evaluation"


class ExperimentStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    EVAL = "eval"


class LossMasking(str, Enum):
    ASSISTANT_ONLY = "assistant_only"  # loss on assistant tokens only
    FULL = "full"


@dataclass(frozen=True)
class DatasetRef(Contract):
    dataset_id: str
    version: str
    split: DatasetSplit
    sha256: str           # hash of the exact file used
    num_records: int
    origin: DataOrigin
    locator: Optional[str] = None

    def validate(self) -> None:
        require_id("dataset_id", self.dataset_id)
        require_text("version", self.version)
        require_sha256("sha256", self.sha256)
        if self.num_records < 0:
            raise ValidationError("num_records must be >= 0")


@dataclass(frozen=True)
class TrainingConfig(Contract):
    learning_rate: float
    num_epochs: float
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    per_device_batch_size: int
    gradient_accumulation_steps: int
    max_seq_length: int
    framework: str = "llama-factory"
    method: str = "qlora"
    quantization_bits: int = 4
    loss_masking: LossMasking = LossMasking.ASSISTANT_ONLY
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        require_text("framework", self.framework)
        require_text("method", self.method)
        if self.quantization_bits not in (4, 8, 16):
            raise ValidationError("quantization_bits must be 4, 8 or 16")
        if self.learning_rate <= 0 or self.num_epochs <= 0:
            raise ValidationError("learning_rate and num_epochs must be > 0")
        if not 0 <= self.lora_dropout < 1:
            raise ValidationError("lora_dropout must be within [0, 1)")
        for name in ("lora_rank", "lora_alpha", "per_device_batch_size", "gradient_accumulation_steps",
                     "max_seq_length"):
            if getattr(self, name) < 1:
                raise ValidationError(f"{name} must be >= 1")


@dataclass(frozen=True)
class EvaluationConfig(Contract):
    suite: str
    systems: Tuple[str, ...]   # e.g. ("base", "finetuned", "finetuned_memory")
    metrics: Tuple[str, ...]   # e.g. ("semantic_appropriateness", "persona_fidelity", "factual_grounding")
    judge: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        require_id("suite", self.suite)
        if not self.systems or not self.metrics:
            raise ValidationError("systems and metrics must not be empty")
        for s in self.systems + self.metrics:
            require_id("systems/metrics entry", s)
        require_unique("systems", self.systems)
        require_unique("metrics", self.metrics)


@dataclass(frozen=True)
class MetricResult(Contract):
    metric: str
    system: str
    value: float
    n: Optional[int] = None
    note: Optional[str] = None

    def validate(self) -> None:
        require_id("metric", self.metric)
        require_id("system", self.system)
        if self.n is not None and self.n < 0:
            raise ValidationError("n must be >= 0")


@dataclass(frozen=True)
class ArtifactRef(Contract):
    name: str
    sha256: Optional[str] = None
    locator: Optional[str] = None

    def validate(self) -> None:
        require_text("name", self.name)
        if self.sha256 is not None:
            require_sha256("sha256", self.sha256)


# Fields that are bookkeeping, outcomes or machine description rather than declared inputs. Excluded from
# fingerprint(). ``environment`` (hardware, driver and library versions) is recorded but not fingerprinted, so
# the same configuration keeps one identity on different machines.
_NON_FINGERPRINT_FIELDS = ("experiment_id", "title", "status", "created_at", "results", "artifacts", "notes",
                           "environment")


def _fingerprint_view(value: Any) -> Any:
    """JSON view of a contract without its ``locator`` fields (machine-specific paths).

    Only contract fields named ``locator`` are dropped; keys inside free-form dicts are kept as they are.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _fingerprint_view(getattr(value, f.name)) for f in fields(value) if f.name != "locator"}
    if isinstance(value, (list, tuple)):
        return [_fingerprint_view(v) for v in value]
    return value


@dataclass(frozen=True)
class ExperimentManifest(Contract):
    experiment_id: str
    kind: ExperimentKind
    title: str
    base_model: ModelRef
    prompt_template: PromptTemplateRef
    seed: int
    datasets: Tuple[DatasetRef, ...]
    created_at: datetime
    status: ExperimentStatus = ExperimentStatus.PLANNED
    adapter: Optional[AdapterRef] = None    # training: the adapter produced; evaluation: the adapter evaluated
    training: Optional[TrainingConfig] = None
    evaluation: Optional[EvaluationConfig] = None
    code_revision: Optional[str] = None     # git commit of this repository
    environment: Dict[str, Any] = field(default_factory=dict)  # library versions, hardware
    results: Tuple[MetricResult, ...] = ()
    artifacts: Tuple[ArtifactRef, ...] = ()
    notes: str = ""
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValidationError(f"unsupported schema_version {self.schema_version!r}")
        require_id("experiment_id", self.experiment_id)
        require_text("title", self.title)
        if self.seed < 0:
            raise ValidationError("seed must be >= 0")
        if not self.datasets:
            raise ValidationError("at least one dataset is required")
        require_unique("datasets (id, split)", [(d.dataset_id, d.split) for d in self.datasets])
        if self.kind is ExperimentKind.TRAINING:
            if self.training is None:
                raise ValidationError("a training experiment needs a training config")
            if not any(d.split is DatasetSplit.TRAIN for d in self.datasets):
                raise ValidationError("a training experiment needs a train split")
        if self.kind is ExperimentKind.EVALUATION and self.evaluation is None:
            raise ValidationError("an evaluation experiment needs an evaluation config")
        if self.adapter is not None:
            if self.adapter.base_model.model_id != self.base_model.model_id:
                raise ValidationError("adapter base model does not match base_model")
            if self.adapter.prompt_template.name != self.prompt_template.name:
                raise ValidationError("adapter prompt template does not match prompt_template")
        if self.results and self.status is not ExperimentStatus.COMPLETED:
            raise ValidationError("results may only be recorded on a completed experiment")
        if self.results and self.evaluation is not None:
            for r in self.results:
                if r.system not in self.evaluation.systems or r.metric not in self.evaluation.metrics:
                    raise ValidationError(f"result {r.metric}/{r.system} is not declared in the evaluation config")

    def fingerprint(self) -> str:
        """sha256 identifying the declared inputs of this run.

        Covered: kind, base model (id, revision, dtype), prompt template, seed, datasets (id, version, split,
        sha256, size, origin; order-independent), training or evaluation config, code revision, schema version,
        and for evaluation runs the adapter under test. For training runs the adapter is the run's output and
        is left out, so filling in its hash after training does not change the identity.

        A shared fingerprint means the same declared configuration. It does not promise identical results:
        hardware, drivers, library versions and non-deterministic GPU kernels can still change the outcome.
        """
        data = {f.name: _fingerprint_view(getattr(self, f.name)) for f in fields(self)
                if f.name not in _NON_FINGERPRINT_FIELDS}
        data["datasets"] = sorted(data["datasets"], key=lambda d: (d["dataset_id"], d["split"]))
        if self.kind is ExperimentKind.TRAINING:
            data.pop("adapter")
        return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()
