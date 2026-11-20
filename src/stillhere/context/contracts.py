"""The Context Contract: what a caller asks for (``ContextRequest``) and what it gets (``ContextResult``).

Every future producer of context (Memory Writer, Observation, Reflection, a CPCD style adapter's caller) uses the same two
types. What exists now and what is only a slot for later:

* implemented: pinned records (by memory id, through ``MemoryAPI``), the recent conversation (by conversation id), long-term
  evidence (ranked ``EvidenceHit`` from any existing retriever or reranker), working context (session-only, never stored);
* slot only: ``DerivedItem`` (an observation or a reflection result). V1 validates it and keeps it out of the context unless
  ``allow_derived`` is set, and even then it is labelled as derived and never outranks the raw evidence it cites.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Mapping, Optional, Tuple

from stillhere.contracts import EvidenceHit

# result status
OK = "OK"
BUDGET_INSUFFICIENT = "BUDGET_INSUFFICIENT"      # the fixed parts (instructions, question) do not fit, or no evidence could be shown at all
SCOPE_ERROR = "SCOPE_ERROR"

# warnings (stable strings: tests and reports match on them)
CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"            # a unit was too large and is shown in part, with every omitted message listed
EVIDENCE_OVER_BUDGET = "EVIDENCE_OVER_BUDGET"    # a unit could not be shown at all and was left out
TOP_EVIDENCE_EXCLUDED = "TOP_EVIDENCE_EXCLUDED"  # the best ranked unit is not in the context
RECENT_TRIMMED = "RECENT_TRIMMED"                # earlier whole turns of the recent conversation left out
TEMPORAL_BOUNDARY_AMBIGUOUS = "TEMPORAL_BOUNDARY_AMBIGUOUS"
ORDER_ANOMALY = "ORDER_ANOMALY"                  # stored order and time order of a round disagree
LATER_MESSAGES_HIDDEN = "LATER_MESSAGES_HIDDEN"  # a round has messages after as_of / known_at; they are not shown
PROVENANCE_MISSING = "PROVENANCE_MISSING"
DERIVED_EXCLUDED = "DERIVED_EXCLUDED"

TOKEN_EXACT = "EXACT"
TOKEN_ESTIMATED = "ESTIMATED"


@dataclass(frozen=True)
class WorkingItem:
    """Session state for the current task. Shown as working context, never persisted as a memory."""

    key: str
    text: str
    source_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class DerivedItem:
    """A future observation or reflection result. Supporting evidence must be raw records the context also shows."""

    key: str
    kind: str                                   # "observation" | "reflection"
    text: str
    supporting_evidence_ids: Tuple[str, ...]
    contradicting_evidence_ids: Tuple[str, ...] = ()
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    confidence: Optional[float] = None
    version: str = "0"


@dataclass(frozen=True)
class ContextRequest:
    persona_id: str
    question: str
    as_of: Optional[datetime] = None
    known_at: Optional[datetime] = None
    conversation_id: Optional[str] = None       # the current conversation, for the recent context
    reader_model: str = "deepseek-flash"
    input_token_budget: int = 6000              # everything sent: instructions, question, context
    output_token_reserve: int = 400             # kept for the reader's answer; part of the model window, not of the input
    instruction_tokens: int = 0                 # the reader's fixed system and user scaffolding, counted by the caller
    hits: Tuple[EvidenceHit, ...] = ()          # ranked long-term evidence from the existing retriever / reranker
    hit_keys: Tuple[str, ...] = ()              # the caller's key for each hit (e.g. the V3 record key); defaults to the record id
    pinned_memory_ids: Tuple[str, ...] = ()
    working: Tuple[WorkingItem, ...] = ()
    derived: Tuple[DerivedItem, ...] = ()
    allow_derived: bool = False
    top_k: int = 10
    recent_max_turns: Optional[int] = None      # None: as many whole turns as the budget allows


@dataclass
class ContextResult:
    status: str
    assembled_context: str
    selected_evidence_ids: List[str]            # record keys of the evidence shown, in order
    included_message_ids: List[str]             # every message whose text is in the context
    excluded_evidence_ids: Dict[str, str]       # key -> reason
    provenance_map: Dict[str, dict]             # context id (E01, R01, P01, W01) -> record key, source ids, message ids, completeness
    token_usage_estimate: int
    token_budget: dict
    token_count_mode: str
    truncation_policy: str
    warnings: List[str]
    assembly_trace: List[dict]
    temporal_boundary: Optional[dict] = None
    omitted_message_ids: Dict[str, List[str]] = field(default_factory=dict)   # unit key -> messages left out by an overflow
    latency_ms: float = 0.0

    @property
    def id_map(self) -> Dict[str, str]:
        return {cid: p["key"] for cid, p in self.provenance_map.items()}

    @property
    def truncated(self) -> bool:
        return bool(self.omitted_message_ids) or any(r.startswith("budget") for r in self.excluded_evidence_ids.values())
