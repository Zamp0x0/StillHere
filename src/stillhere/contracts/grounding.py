"""Phase 5B grounding: what the evidence says about each asked atom, what the reply did with it, and what the
guard did about it.

* ``AtomEvidence``: the state of one atom's evidence. Each missing-evidence state is its own value, because each
  calls for a different answer: no record of a user's birthday is not "the user has no birthday", no record of a
  boyfriend's job is not "there is no boyfriend", no job record for a date is not unemployment.
* ``TurnGrounding``: the turn's kind (a fact question, the persona's identity, whether the persona is an AI, or
  ordinary conversation) and one ``AtomGrounding`` per atom of the query plan.
* ``GroundingIssue``: one finding of the output validator, tied to an atom when it is about one.
* ``AnswerNote``: one constraint a regenerated reply is given. Notes come from a closed list of codes with atom
  ids, subjects and attribute names as parameters; the rejected reply's text never becomes a note.
* ``GuardReport``: the audit record of one turn: every candidate reply with its raw text and issues, the notes
  of the retry, the outcome and the reply the user sees.
* ``FallbackAudit`` (Stage J): every stage of a fallback: the repair and its re-check, the evidence-only template
  and its re-check, which one was sent and why, and whether the sent reply is safe (no fact-bearing issue) and
  complete (no coverage issue). Safety is recorded, never inferred from a passing task score.
* Stage K: ``GuardOutcome.SHORT_CIRCUIT`` is a turn the model was never asked about, because no atom's evidence can
  answer it and one of them is a time gap, a budget-blocked or an unplaceable time: the reply is the evidence-only
  template, ``generation_skipped`` is set with the reason, and there is no model attempt to score.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import ClassVar, Optional, Tuple

from ._base import Contract, ValidationError, require_id, require_text, require_unique


class AtomEvidence(str, Enum):
    HAS_EVIDENCE = "has_evidence"                # records (or the persona card, for the persona's name) answer it
    NO_MATCH = "no_match"                        # nothing is recorded: unknown, never "does not exist"
    TIME_GAP = "time_gap"                        # records exist for other times only (a date, a life period)
    CONFLICT = "conflict"                        # records disagree; no value is chosen
    CORRECTED = "corrected"                      # answered, and the slot holds a mis-record that is not a past value
    AMBIGUOUS_SUBJECT = "ambiguous_subject"      # whose fact it is cannot be told
    UNRESOLVED_TIME = "unresolved_time"          # the time asked cannot be placed on the calendar
    DEPENDENCY_WITHHELD = "dependency_withheld"  # a comparison whose inputs are missing or disputed: no conclusion
    BUDGET_BLOCKED = "budget_blocked"            # records answer it, but the memory budget left them all out of the
                                                 # prompt: the reply must not state them; it says it cannot check


ANSWERED = (AtomEvidence.HAS_EVIDENCE, AtomEvidence.CORRECTED)


class TurnKind(str, Enum):
    FACT = "fact"                      # at least one atom asks for a recorded fact
    IDENTITY = "identity"              # who the persona is (name); answered from the persona card
    AI_NATURE = "ai_nature"            # whether the user is talking to an AI or a real person
    CONVERSATIONAL = "conversational"  # chit-chat, feelings, news, social questions: nothing to look up


ISSUE_CODES = (
    # identity (every turn)
    "NAME_MISMATCH",          # the persona gives herself another name (我叫小满)
    "ENTITY_AS_SELF",         # the persona takes a recorded name of someone or something else (我叫年糕)
    "AI_VOICE",               # assistant self-description when the user did not ask about it
    "DECEPTIVE_DENIAL",       # denies being an AI when sincerely asked
    # coverage
    "MISSING_VALUE",          # an answered atom's value is not in the reply
    "MISSING_UNKNOWN",        # an unanswered atom is neither said to be unknown nor asked back
    "ECHO",                   # a bubble repeats the question instead of answering it
    # grounding
    "UNSUPPORTED_FACT",       # a specific claim for an atom without evidence
    "UNSUPPORTED_ENTITY",     # a title, place or name that no evidence, card or message contains
    "SUBJECT_TRANSFER",       # another subject's recorded value given for this atom's subject
    "NEGATIVE_EXISTENCE",     # "does not exist / never had / unemployed" without evidence
    "BODY_CLAIM",             # a body reaction or health detail no evidence contains
    "CONTRADICTS_EVIDENCE",   # the opposite stance of a recorded preference
    "WRONG_TIME_VALUE",       # a value recorded for another time given as the answer for the time asked
    "CORRECTED_VALUE",        # a corrected mis-record given as a fact (now or in the past)
    "CONFLICT_COLLAPSE",      # disputed values given as settled
    "COMPARISON_CONCLUSION",  # a comparison decided while its inputs are missing or disputed
    # Stage J (grounding_validator_v2)
    "VALUE_NOT_VISIBLE",      # a value stated for a budget-blocked atom: the prompt did not show its records
    "WRONG_OWNER",            # a relative turned into someone else's (the user's "your sister" answered as theirs)
    "CONTRADICTS_RECORD",     # a claim the persona's records contradict (owning the business she manages)
    "TEMPORAL_MISMATCH",      # a recorded change placed in a year, season or month its record does not allow
    "INVENTED_SHARED_HISTORY",  # things the user and the persona did, or the user's habits, that nothing records
    "UNSUPPORTED_SELF_EVENT",   # a new purchase or event of the persona's that nothing records
    "VALUE_SUBSTITUTED",      # a recorded value left out and a look-alike given in its place (爬山 for 攀岩)
    # Stage K (grounding_validator_v3)
    "UNKNOWN_BOUNDARY",       # a claim, hedged or not, for an atom its evidence cannot answer (a time gap)
    "INFERRED_EVENT",         # an event (a move, a new job) put at a time no record dates: validity is no event
)

NOTE_CODES = ("cover_all", "answer_value", "answer_unknown", "keep_conflict", "withhold_comparison", "time_only",
              "time_gap", "no_negative", "no_new_entities", "no_body_claims", "keep_stance", "own_name",
              "no_assistant_voice", "no_denial", "no_corrected", "no_echo",
              # Stage J
              "keep_owner", "keep_record", "keep_dates", "no_new_events",
              # Stage K
              "no_event_dates")

_NOTE_SUBJECT = re.compile(r"^(?:[a-z][a-z0-9_]{0,31}|[\u4e00-\u9fff]{1,4})$")


@dataclass(frozen=True)
class AtomGrounding(Contract):
    query_id: str
    evidence: AtomEvidence
    fact_seeking: bool
    subject: Optional[str] = None              # "persona", "user", "cat", ...; None when unowned or ambiguous
    owner: Optional[str] = None                # the kinship word of an unowned atom ("妈妈")
    attributes: Tuple[str, ...] = ()
    time_scope: Optional[str] = None
    span: str = ""                             # the user's words the atom was read from
    memory_ids: Tuple[str, ...] = ()
    values: Tuple[str, ...] = ()               # values that answer the atom (for its time)
    context_values: Tuple[str, ...] = ()       # evidence shown alongside, true at another time (contrast)
    other_values: Tuple[str, ...] = ()         # the slot's values for other times that the evidence does not show
    corrected_values: Tuple[str, ...] = ()     # mis-records: never a fact, now or in the past
    foreign_values: Tuple[str, ...] = ()       # the same attribute's values for other subjects
    statements: Tuple[str, ...] = ()           # the evidence statements (the fallback's only facts)
    multi: bool = False                        # several values coexist and each is part of the answer
    source: str = "memory"                     # "memory" or "persona_card"
    detail: Optional[str] = None               # "time_gap", "life_period", "unowned:妈妈", ...
    hidden_values: Tuple[str, ...] = ()        # BUDGET_BLOCKED: the recorded values the prompt did not show
    time_start: Optional[datetime] = None      # Stage K: the asked time, when the atom names one
    time_end: Optional[datetime] = None
    _omit_when_empty: ClassVar[Tuple[str, ...]] = ("hidden_values", "time_start", "time_end")

    def validate(self) -> None:
        require_id("query_id", self.query_id)
        if self.hidden_values and self.evidence is not AtomEvidence.BUDGET_BLOCKED:
            raise ValidationError(f"{self.query_id}: only a budget-blocked atom has hidden values")
        if self.source not in ("memory", "persona_card"):
            raise ValidationError(f"source must be memory or persona_card, got {self.source!r}")
        if (self.evidence in ANSWERED or self.evidence is AtomEvidence.CONFLICT) != bool(self.values):
            raise ValidationError(f"{self.query_id}: values are listed exactly when the atom is answered or disputed")


@dataclass(frozen=True)
class DependencyGrounding(Contract):
    query_ids: Tuple[str, ...]
    evidence: AtomEvidence                     # HAS_EVIDENCE or DEPENDENCY_WITHHELD
    relation: Optional[str] = None

    def validate(self) -> None:
        if len(self.query_ids) < 2:
            raise ValidationError("a dependency links at least two atoms")
        if self.evidence not in (AtomEvidence.HAS_EVIDENCE, AtomEvidence.DEPENDENCY_WITHHELD):
            raise ValidationError("a dependency has evidence or is withheld")


@dataclass(frozen=True)
class TurnGrounding(Contract):
    kind: TurnKind
    atoms: Tuple[AtomGrounding, ...] = ()
    dependencies: Tuple[DependencyGrounding, ...] = ()
    plan_id: Optional[str] = None
    as_of: Optional[datetime] = None           # Stage J: the retrieval's as_of, for relative years (去年, 前年)
    _omit_when_empty: ClassVar[Tuple[str, ...]] = ("as_of",)

    def validate(self) -> None:
        require_unique("atom query ids", [a.query_id for a in self.atoms])
        ids = {a.query_id for a in self.atoms}
        for d in self.dependencies:
            if not set(d.query_ids) <= ids:
                raise ValidationError("a dependency names an atom the turn does not have")

    @property
    def fact_atoms(self) -> Tuple[AtomGrounding, ...]:
        return tuple(a for a in self.atoms if a.fact_seeking)


@dataclass(frozen=True)
class GroundingIssue(Contract):
    code: str
    query_id: Optional[str] = None
    span: str = ""             # the clause of the reply the issue was found in; "" when something is missing
    detail: str = ""

    def validate(self) -> None:
        if self.code not in ISSUE_CODES:
            raise ValidationError(f"unknown issue code {self.code!r}")


@dataclass(frozen=True)
class AnswerNote(Contract):
    code: str
    query_id: Optional[str] = None
    subject: Optional[str] = None
    attributes: Tuple[str, ...] = ()

    def validate(self) -> None:
        if self.code not in NOTE_CODES:
            raise ValidationError(f"unknown note code {self.code!r}")
        if self.query_id is not None:
            require_id("query_id", self.query_id)
        if self.subject is not None and not _NOTE_SUBJECT.match(self.subject):
            raise ValidationError(f"note subject {self.subject!r} is not a subject id or a kinship word")
        for a in self.attributes:
            require_id("attribute", a)


@dataclass(frozen=True)
class GuardAttempt(Contract):
    attempt: int                               # 1: the first reply; 2: the one regeneration
    bubbles: Tuple[str, ...]
    finish_reason: str
    issues: Tuple[GroundingIssue, ...] = ()
    notes: Tuple[AnswerNote, ...] = ()         # what the attempt's request carried
    raw_text: Optional[str] = None
    request_id: str = ""

    def validate(self) -> None:
        if self.attempt not in (1, 2):
            raise ValidationError("a turn has at most one regeneration")

    @property
    def passed(self) -> bool:
        return not self.issues


class GuardOutcome(str, Enum):
    PASSED = "passed"              # the first reply passed
    RETRY_PASSED = "retry_passed"  # the first reply failed, the regeneration passed
    FALLBACK = "fallback"          # no passing reply: the deterministic answer was sent
    SKIPPED = "skipped"            # nothing to check (the backend failed or retrieval errored)
    SHORT_CIRCUIT = "short_circuit"  # Stage K: no evidence can answer the turn; the model was not called


FALLBACK_MODES = ("repair", "template", "persona_example", "minimal")   # minimal: Stage J, no facts at all
# Structural defects of a deterministic reply (Stage J), checked besides the validator: a bubble that trails off
# with a bare topic (电影嘛…), a clause that starts with a connective whose clause was removed, a 她/他 whose
# person the reply never names.
FALLBACK_DEFECTS = ("FRAGMENT", "DANGLING_CONNECTIVE", "DANGLING_PRONOUN")
NATURALNESS_WARNINGS = ("RECORD_READOUT", "ECHO_QUESTION", "FRAGMENT", "REPETITION", "DANGLING_PRONOUN")


@dataclass(frozen=True)
class FallbackAudit(Contract):
    repair_bubbles: Tuple[str, ...] = ()
    repair_issues: Tuple[GroundingIssue, ...] = ()
    repair_defects: Tuple[str, ...] = ()
    repair_safe: bool = False                  # the repair passed the full check with no defect
    template_bubbles: Tuple[str, ...] = ()     # empty when the repair was sent
    template_issues: Tuple[GroundingIssue, ...] = ()
    template_defects: Tuple[str, ...] = ()
    template_safe: Optional[bool] = None       # None: the template was not needed
    sent: str = "repair"                       # the fallback mode that was sent
    reason: str = ""                           # why the repair (and the template) were not sent
    fallback_safe: bool = False                # the sent reply has no fact-bearing issue and no defect
    fallback_complete: bool = False            # the sent reply has no coverage issue

    def validate(self) -> None:
        if self.sent not in FALLBACK_MODES:
            raise ValidationError(f"sent must be one of {FALLBACK_MODES}")
        for d in self.repair_defects + self.template_defects:
            if d not in FALLBACK_DEFECTS:
                raise ValidationError(f"unknown fallback defect {d!r}")
        if self.sent == "repair" and not self.repair_safe:
            raise ValidationError("a repair is sent only when it passed its re-check")
        if self.sent != "repair" and self.template_safe is None:
            raise ValidationError("a repair that was not sent leaves a checked template")


@dataclass(frozen=True)
class GuardReport(Contract):
    validator: str
    grounding: TurnGrounding
    attempts: Tuple[GuardAttempt, ...]
    outcome: GuardOutcome
    final_bubbles: Tuple[str, ...]
    fallback_mode: Optional[str] = None
    triggers: Tuple[str, ...] = ()             # issue codes that led to the regeneration or the fallback
    note: str = ""
    fallback: Optional[FallbackAudit] = None   # Stage J: every stage of the fallback
    naturalness_warnings: Tuple[str, ...] = ()  # Stage J: of the reply the user sees (recorded, not gating)
    generation_skipped: bool = False           # Stage K: the model was not called (SHORT_CIRCUIT)
    skip_reason: str = ""
    locked_atoms: Tuple[str, ...] = ()         # Stage K: atoms answered from the evidence only (no regeneration)
    _omit_when_empty: ClassVar[Tuple[str, ...]] = ("note", "fallback", "naturalness_warnings", "generation_skipped",
                                                   "skip_reason", "locked_atoms")

    def validate(self) -> None:
        require_text("validator", self.validator)
        if len(self.attempts) > 2:
            raise ValidationError("a turn has at most one regeneration")
        if (self.outcome is GuardOutcome.FALLBACK) != (self.fallback_mode is not None):
            raise ValidationError("fallback_mode is set exactly when the outcome is a fallback")
        if self.fallback_mode is not None and self.fallback_mode not in FALLBACK_MODES:
            raise ValidationError(f"fallback_mode must be one of {FALLBACK_MODES}")
        if self.outcome is GuardOutcome.PASSED and (len(self.attempts) != 1 or not self.attempts[0].passed):
            raise ValidationError("passed means one attempt without issues")
        if self.outcome is GuardOutcome.RETRY_PASSED and (len(self.attempts) != 2 or not self.attempts[1].passed):
            raise ValidationError("retry_passed means a second attempt without issues")
        if self.outcome is not GuardOutcome.SKIPPED and not self.final_bubbles:
            raise ValidationError("a checked turn ends with a reply")
        if (self.outcome is GuardOutcome.SHORT_CIRCUIT) != self.generation_skipped:
            raise ValidationError("generation_skipped is set exactly when the turn was short-circuited")
        if self.outcome is GuardOutcome.SHORT_CIRCUIT:
            if self.attempts:
                raise ValidationError("a short-circuited turn has no model attempt")
            if self.fallback is None or self.fallback.sent not in ("template", "minimal"):
                raise ValidationError("a short-circuited turn sends the checked template or the minimal answer")
            if not self.skip_reason:
                raise ValidationError("a short-circuited turn says why")
        elif self.fallback is not None and (self.outcome is not GuardOutcome.FALLBACK
                                            or self.fallback.sent != self.fallback_mode):
            raise ValidationError("a fallback audit belongs to a fallback and names the mode that was sent")
        for w in self.naturalness_warnings:
            if w not in NATURALNESS_WARNINGS:
                raise ValidationError(f"unknown naturalness warning {w!r}")

    @property
    def intervened(self) -> bool:
        return self.outcome in (GuardOutcome.RETRY_PASSED, GuardOutcome.FALLBACK, GuardOutcome.SHORT_CIRCUIT)
