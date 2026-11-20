"""Query understanding: what one user message asks, split into atomic queries, and what each one found.

A message can ask several things at once ("你住哪？我住哪？", "你做什么工作？你爸爸呢？"). Reading it as one query
forces one subject on all of it: either a fact is given to the wrong owner, or the whole message is refused. A
``QueryPlan`` keeps the parts apart:

* ``QueryPart``: a span of the normalized message (a bubble or a clause) and its speech act. Remarks
  ("我最近也在找工作") are parts too, so a plan shows what was read and what was left out.
* ``AtomicQuery``: one asked fact: the parts it comes from, whose fact it is (``subject`` and how sure that is),
  which memory attributes it asks for (or why none), the time it asks about, and whether it may return records.
  A subject is never guessed: an unidentified reference is ``ambiguous``, someone with no subject in memory
  (``你妈妈``) is ``unowned``, and neither may return records.
* ``QueryDependency``: atoms that must be answered together (``你和我谁生日更早`` compares two atoms). Atoms
  without one are independent.
* ``DialogueFocus``: the attribute an elliptical part ("那你呢", "以前呢") takes over from the last turns. Only the
  attribute is carried, never the subject; it lives in the plan of one message and is never stored.

``RetrievalBatch`` is the outcome: one ``AtomicResult`` per atom (status, records, whether they suffice for what
was asked and, if not, why), the outcome of each dependency, and the batch status (all, some or none of the atoms
answered: ``partial_success`` when "你住哪？我住哪？" finds the persona's city and no record of the user's).

``TimeScope`` and ``RetrievalStatus`` live here because atoms use them; ``stillhere.contracts.retrieval``
re-exports both. Standard library only; the planner is ``stillhere.memory.planner``.

Day 7 Phase 5B (Temporal Query Engine): an atom whose words name a calendar time (``去年``, ``2026年3月10号``,
``今年三月初``) carries a ``TimeExpression``: the half-open interval ``[start, end)`` those words mean in the
calendar's timezone, as of the query's ``as_of``, with its kind, granularity and how it was resolved. Its scope is
``at``. Words that name a time no calendar places (``前阵子``, ``2026年2月30号``) give scope ``unresolved`` and an
atom that returns nothing; nothing falls back to the current value. Atoms without such words have no
``time`` field and plan exactly as before (``current``, ``historical``, ``all``, a life period).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import ClassVar, Optional, Tuple

from ._base import Contract, ValidationError, require_id, require_text, require_unique

PLAN_VERSION = "query_plan_v1"
ROLES = ("user", "persona")
SUBJECT_SOURCES = ("explicit", "query", "context", "slot")


class TimeScope(str, Enum):
    CURRENT = "current"        # what holds at as_of
    HISTORICAL = "historical"  # what held before as_of (the current value is returned alongside)
    ALL = "all"                # both
    AT = "at"                  # what held during the atom's TimeExpression interval (Phase 5B)
    UNRESOLVED = "unresolved"  # the words name a time no calendar places: nothing is returned (Phase 5B)


class TimeKind(str, Enum):
    """What a time expression names. Only the interval kinds carry ``start``/``end``."""

    CURRENT = "current"                      # 现在, 目前
    HISTORICAL = "historical"                # 以前, 之前, 过去 (open: past entries and the current one)
    ALL = "all"                              # 一直 (the whole stretch)
    ABSOLUTE_POINT = "absolute_point"        # one calendar day: 2026年3月10号, 3月10号 (year inferred)
    ABSOLUTE_INTERVAL = "absolute_interval"  # a year, a month or part of one: 2025年, 2026年3月, 3月初
    RELATIVE_INTERVAL = "relative_interval"  # counted from as_of: 去年, 前年, 今年三月初, 上个月, 上周
    LIFE_PERIOD = "life_period"              # 小时候: no record places it, so no interval
    UNRESOLVED_TIME = "unresolved_time"      # a time that cannot be placed: 前阵子, 2026年2月30号, a future date


INTERVAL_KINDS = (TimeKind.ABSOLUTE_POINT, TimeKind.ABSOLUTE_INTERVAL, TimeKind.RELATIVE_INTERVAL)
TIME_GRANULARITIES = ("day", "week", "month", "part_of_month", "year", "part_of_year", "open", "life_period",
                      "unknown")
TIME_RESOLUTIONS = ("resolved", "inferred_year", "unresolved", "invalid", "future")


class RetrievalStatus(str, Enum):
    OK = "ok"                                  # at least one hit
    NO_MATCH = "no_match"                      # nothing in memory answers the query
    AMBIGUOUS_SUBJECT = "ambiguous_subject"    # the subject could not be identified; nothing returned
    INVALID_REQUEST = "invalid_request"        # the query was malformed (bad time, persona id, limit)
    ERROR = "error"                            # the store failed; nothing returned


class SpeechAct(str, Enum):
    QUESTION = "question"      # ends with "?" or has a question word
    REQUEST = "request"        # a request word and a fact it asks for ("我想知道你住哪")
    STATEMENT = "statement"    # asks nothing ("我今天搬家了"); read only when no part of the message asks


class SubjectStatus(str, Enum):
    RESOLVED = "resolved"      # the owner is known (explicit, from the query or from the context)
    OPEN = "open"              # no owner named; an attribute whose records all share one subject supplies it
    AMBIGUOUS = "ambiguous"    # plural, quoted or unidentified reference: never guessed
    UNOWNED = "unowned"        # someone with no subject in memory (你妈妈): nothing is theirs to return


class IntentStatus(str, Enum):
    SLOT = "slot"              # one or more memory attributes
    KEYWORD = "keyword"        # no attribute; shared words decide (the store's keyword search)
    UNSUPPORTED = "unsupported"  # recognised, but not something memory records (你喜欢我吗)
    UNKNOWN = "unknown"        # mixed or undecidable (你喜欢我还是喜欢攀岩; an ellipsis with two candidate topics)


class DependencyKind(str, Enum):
    INDEPENDENT = "independent"              # the default: no QueryDependency names the atom
    DEPENDENT_COMPARE = "dependent_compare"  # the answer compares the atoms' values (你和我谁生日更早)
    UNKNOWN = "unknown"                      # the atoms depend on each other in a way that is not understood


class FocusStatus(str, Enum):
    ACTIVE = "active"          # one attribute from the last turns may be carried
    INVALID = "invalid"        # nothing may be carried (see ``reason``)


class Sufficiency(str, Enum):
    """Whether an atom's records answer what it asked, for the time it asked about."""

    SUFFICIENT = "sufficient"          # records answer it
    CONFLICT = "conflict"              # records exist and disagree: no value is chosen
    PARTIAL = "partial"                # some of the atom's attributes are answered, others not
    TIME_UNKNOWN = "time_unknown"      # records exist, none for the time asked (a life period, no earlier value)
    NO_RECORD = "no_record"            # a known owner, nothing recorded
    NOT_ANSWERABLE = "not_answerable"  # unowned, ambiguous, unsupported or unknown: nothing was looked up


class BatchStatus(str, Enum):
    COMPLETE_SUCCESS = "complete_success"  # every atom returned records
    PARTIAL_SUCCESS = "partial_success"    # some atoms returned records, others none
    NO_SUCCESS = "no_success"              # no atom returned records


class EvidenceState(str, Enum):
    AVAILABLE = "available"                    # every compared atom has records
    MISSING = "missing"                        # an atom has none
    CONFLICT = "conflict"                      # an atom's records disagree
    UNSUPPORTED_COMPARISON = "unsupported_comparison"  # the relation cannot be decided from these attributes


class AnswerLevel(str, Enum):
    COMPLETE = "complete"          # the dependent question can be answered in full
    SAFE_PARTIAL = "safe_partial"  # only what is known may be said; the comparison itself is not answered
    NONE = "none"                  # nothing known


def _require_role(name: str, value: str) -> None:
    if value not in ROLES:
        raise ValidationError(f"{name} must be one of {ROLES}, got {value!r}")


def _require_source(name: str, value: Optional[str]) -> None:
    if value is not None and value not in SUBJECT_SOURCES:
        raise ValidationError(f"{name} must be one of {SUBJECT_SOURCES}, got {value!r}")


@dataclass(frozen=True)
class TimeExpression(Contract):
    """The time one atom asks about. Intervals are half-open, ``[start, end)``, in the calendar's timezone (a
    calendar day is local midnight to local midnight, never UTC midnights); an interval that runs past ``as_of``
    ends at ``as_of``. ``source_text`` is the user's wording, kept for audit: it never enters a prompt."""

    kind: TimeKind
    source_text: str
    as_of: Optional[datetime]              # the instant relative words were counted from (None: not given)
    timezone: str                          # the calendar's UTC offset, "+08:00"
    granularity: str                       # day, week, month, part_of_month, year, part_of_year, open, ...
    resolution_status: str                 # resolved, inferred_year (a date without its year), unresolved,
                                           # invalid (2026年2月30号), future (starts after as_of)
    start: Optional[datetime] = None       # inclusive; interval kinds only
    end: Optional[datetime] = None         # exclusive; interval kinds only

    def validate(self) -> None:
        require_text("source_text", self.source_text)
        require_text("timezone", self.timezone)
        if self.granularity not in TIME_GRANULARITIES:
            raise ValidationError(f"granularity must be one of {TIME_GRANULARITIES}, got {self.granularity!r}")
        if self.resolution_status not in TIME_RESOLUTIONS:
            raise ValidationError(f"resolution_status must be one of {TIME_RESOLUTIONS}, got "
                                  f"{self.resolution_status!r}")
        interval = self.kind in INTERVAL_KINDS
        if interval != (self.start is not None and self.end is not None):
            raise ValidationError(f"{self.kind.value}: start and end are set exactly for a calendar interval")
        if interval:
            if self.resolution_status not in ("resolved", "inferred_year"):
                raise ValidationError("a calendar interval is resolved (or has an inferred year)")
            for name in ("start", "end"):
                value = getattr(self, name)
                if value.tzinfo is None or value.utcoffset() is None:
                    raise ValidationError(f"{name} must be timezone-aware")
            if not self.start < self.end:
                raise ValidationError(f"start {self.start.isoformat()} must be before end {self.end.isoformat()}")
        elif self.start is not None or self.end is not None:
            raise ValidationError(f"{self.kind.value} has no calendar interval")
        if (self.kind is TimeKind.UNRESOLVED_TIME) != (self.resolution_status in ("unresolved", "invalid", "future")):
            raise ValidationError("an unresolved time, and only that, has status unresolved, invalid or future")

    @property
    def placed(self) -> bool:
        """True for a calendar interval: records can be looked up for it."""
        return self.kind in INTERVAL_KINDS


@dataclass(frozen=True)
class QueryPart(Contract):
    """A span of the normalized message: a bubble or a clause between punctuation marks."""

    index: int
    text: str
    start: int                 # offsets into QueryPlan.text
    end: int
    act: SpeechAct

    def validate(self) -> None:
        require_text("text", self.text)
        if self.index < 0 or not 0 <= self.start < self.end:
            raise ValidationError(f"part {self.index}: bad offsets {self.start}..{self.end}")


@dataclass(frozen=True)
class AtomicQuery(Contract):
    query_id: str                          # "q1", "q2", ... in message order
    parts: Tuple[int, ...]                 # QueryPart indices the atom was read from
    span: str                              # the text of those parts
    speaker: str                           # who said them: always "user" for the message being answered
    act: SpeechAct
    subject: Optional[str]                 # set exactly when subject_status is resolved
    subject_status: SubjectStatus
    subject_source: Optional[str]          # "explicit", "query" or "context" for a resolved subject
    attributes: Tuple[str, ...]            # set exactly when intent is slot
    intent: IntentStatus
    intent_source: Optional[str]           # "cue", "phrase", "focus", "previous" (an earlier atom), "parser"
    time_scope: TimeScope
    time_scope_source: str                 # "explicit", "query" or "default"
    period_unplaced: bool = False          # a life period (小时候) no record places in time, whatever the scope
    retrievable: bool = False              # may return records
    matched: Tuple[str, ...] = ()          # the cue words or phrase that decided the intent
    reason: str = ""                       # the rules that decided owner and intent, for audit
    ambiguity: Optional[str] = None        # why the atom cannot return records ("unowned:妈妈", "plural", ...)
    time: Optional[TimeExpression] = None  # Phase 5B: the calendar time asked, for scope "at" or "unresolved"
    _omit_when_empty: ClassVar[Tuple[str, ...]] = ("time",)   # absent from atoms without one (earlier plans)

    def validate(self) -> None:
        require_id("query_id", self.query_id)
        require_text("span", self.span)
        _require_role("speaker", self.speaker)
        if not self.parts or any(i < 0 for i in self.parts):
            raise ValidationError(f"{self.query_id}: parts must list part indices")
        resolved = self.subject_status is SubjectStatus.RESOLVED
        if resolved != (self.subject is not None):
            raise ValidationError(f"{self.query_id}: subject is set exactly when it is resolved")
        if self.subject is not None:
            require_id("subject", self.subject)
        if resolved != (self.subject_source is not None):
            raise ValidationError(f"{self.query_id}: a resolved subject names its source, others none")
        _require_source("subject_source", self.subject_source)
        if (self.intent is IntentStatus.SLOT) != bool(self.attributes):
            raise ValidationError(f"{self.query_id}: attributes are listed exactly for a slot intent")
        for a in self.attributes:
            require_id("attribute", a)
        require_unique("attributes", self.attributes)
        if self.time_scope_source not in ("explicit", "query", "default"):
            raise ValidationError(f"time_scope_source must be explicit, query or default, got {self.time_scope_source!r}")
        if (self.time_scope is TimeScope.AT) != (self.time is not None and self.time.placed):
            raise ValidationError(f"{self.query_id}: scope 'at' goes with a calendar interval in time")
        if (self.time_scope is TimeScope.UNRESOLVED) != (self.time is not None and not self.time.placed):
            raise ValidationError(f"{self.query_id}: scope 'unresolved' goes with a time no calendar places")
        can = (self.subject_status in (SubjectStatus.RESOLVED, SubjectStatus.OPEN)
               and self.intent in (IntentStatus.SLOT, IntentStatus.KEYWORD) and not self.period_unplaced
               and self.time_scope is not TimeScope.UNRESOLVED)
        if self.retrievable != can:
            raise ValidationError(f"{self.query_id}: retrievable must be {can} for this subject, intent and time")
        needs_reason = self.subject_status in (SubjectStatus.AMBIGUOUS, SubjectStatus.UNOWNED) or self.intent in (
            IntentStatus.UNSUPPORTED, IntentStatus.UNKNOWN)
        if needs_reason and not self.ambiguity:
            raise ValidationError(f"{self.query_id}: an atom that cannot be answered says why (ambiguity)")


@dataclass(frozen=True)
class QueryDependency(Contract):
    kind: DependencyKind
    query_ids: Tuple[str, ...]
    relation: Optional[str] = None         # DEPENDENT_COMPARE: "earlier" or "later"
    supported: bool = False                # the relation can be decided for these attributes
    reason: str = ""

    def validate(self) -> None:
        if self.kind is DependencyKind.INDEPENDENT:
            raise ValidationError("independent atoms need no dependency record")
        if len(self.query_ids) < 2:
            raise ValidationError("a dependency links at least two atoms")
        require_unique("dependency query_ids", self.query_ids)
        if self.supported and (self.kind is not DependencyKind.DEPENDENT_COMPARE or self.relation is None):
            raise ValidationError("only a comparison with a relation can be supported")


@dataclass(frozen=True)
class DialogueFocus(Contract):
    """The attribute the last turns were about, derived from the context of one message. Never stored."""

    status: FocusStatus
    reason: str                            # "carried", "no_roles", "multiple", "quoted", "switched", "no_topic"
    attribute: Optional[str] = None        # ACTIVE only
    noun_subject: Optional[str] = None     # the thing that attribute belonged to ("猫" -> "cat"), if any
    turn_offset: int = 0                   # 1 = the newest context turn
    speaker: Optional[str] = None          # who said that turn
    act: Optional[SpeechAct] = None
    owner: Optional[str] = None            # whose fact that turn was, read with its speaker ("persona", "user")

    def validate(self) -> None:
        require_text("reason", self.reason)
        active = self.status is FocusStatus.ACTIVE
        if active != (self.attribute is not None):
            raise ValidationError("an active focus names exactly one attribute; an invalid one none")
        if active and (self.turn_offset < 1 or self.speaker is None):
            raise ValidationError("an active focus names its turn and speaker")
        if self.speaker is not None:
            _require_role("speaker", self.speaker)
        if self.owner is not None:
            _require_role("owner", self.owner)


@dataclass(frozen=True)
class QueryPlan(Contract):
    plan_id: str                           # a hash of the input and the atoms: same input, same id
    persona_id: str
    text: str                              # the normalized message
    planner: str                           # "local_rules_v1", or with "+parser:<model>" when a candidate was used
    parts: Tuple[QueryPart, ...]
    atoms: Tuple[AtomicQuery, ...]
    dependencies: Tuple[QueryDependency, ...] = ()
    focus: Optional[DialogueFocus] = None
    confidence: str = "high"               # "low": an asked part has no attribute or an unknown intent
    notes: Tuple[str, ...] = ()            # e.g. "parser:timeout" when the optional parser fell back
    version: str = PLAN_VERSION

    def validate(self) -> None:
        if self.version != PLAN_VERSION:
            raise ValidationError(f"unsupported query plan version {self.version!r}")
        require_id("plan_id", self.plan_id)
        require_id("persona_id", self.persona_id)
        require_text("planner", self.planner)
        if not self.atoms:
            raise ValidationError("a plan has at least one atom")
        if [p.index for p in self.parts] != list(range(len(self.parts))):
            raise ValidationError("parts are numbered 0..n-1 in order")
        ids = [a.query_id for a in self.atoms]
        require_unique("query ids", ids)
        if any(i >= len(self.parts) for a in self.atoms for i in a.parts):
            raise ValidationError("an atom names a part the plan does not have")
        for d in self.dependencies:
            if not set(d.query_ids) <= set(ids):
                raise ValidationError(f"dependency names unknown atoms: {list(d.query_ids)}")
        if self.confidence not in ("high", "low"):
            raise ValidationError(f"confidence must be 'high' or 'low', got {self.confidence!r}")

    def atom(self, query_id: str) -> AtomicQuery:
        for a in self.atoms:
            if a.query_id == query_id:
                return a
        raise KeyError(query_id)

    @property
    def multi_intent(self) -> bool:
        return len(self.atoms) > 1 or bool(self.dependencies)


# ---------------------------------------------------------------- outcome of a plan

@dataclass(frozen=True)
class AtomicResult(Contract):
    query_id: str
    status: RetrievalStatus                # ok, no_match or ambiguous_subject
    sufficiency: Sufficiency
    subject: Optional[str] = None          # the subject the records belong to (after an open subject is decided)
    subject_source: Optional[str] = None
    slot_keys: Tuple[str, ...] = ()        # "subject/attribute" of each slot resolved for the atom
    memory_ids: Tuple[str, ...] = ()       # the atom's records in the result's hits, in hit order
    missing: Optional[str] = None          # why records are missing ("no_record", "unowned:妈妈", "life_period")
    candidates: Tuple[str, ...] = ()       # ambiguous: subjects in memory that could fit

    def validate(self) -> None:
        require_id("query_id", self.query_id)
        if self.status not in (RetrievalStatus.OK, RetrievalStatus.NO_MATCH, RetrievalStatus.AMBIGUOUS_SUBJECT):
            raise ValidationError(f"an atom's status is ok, no_match or ambiguous_subject, got {self.status.value}")
        if (self.status is RetrievalStatus.OK) != bool(self.memory_ids):
            raise ValidationError(f"{self.query_id}: status 'ok' and records go together")
        require_unique("atom memory ids", self.memory_ids)
        _require_source("subject_source", self.subject_source)
        answered = self.sufficiency in (Sufficiency.SUFFICIENT, Sufficiency.CONFLICT, Sufficiency.PARTIAL)
        if answered != bool(self.memory_ids):
            raise ValidationError(f"{self.query_id}: sufficiency {self.sufficiency.value} does not fit "
                                  f"{len(self.memory_ids)} record(s)")
        if self.sufficiency is not Sufficiency.SUFFICIENT and self.sufficiency is not Sufficiency.CONFLICT \
                and not self.missing:
            raise ValidationError(f"{self.query_id}: missing evidence must say why")


@dataclass(frozen=True)
class DependencyOutcome(Contract):
    kind: DependencyKind
    query_ids: Tuple[str, ...]
    evidence: EvidenceState
    answer: AnswerLevel
    relation: Optional[str] = None

    def validate(self) -> None:
        if self.kind is DependencyKind.INDEPENDENT or len(self.query_ids) < 2:
            raise ValidationError("a dependency outcome links at least two dependent atoms")
        if self.answer is AnswerLevel.COMPLETE and self.evidence is not EvidenceState.AVAILABLE:
            raise ValidationError("only available evidence gives a complete answer")


@dataclass(frozen=True)
class RetrievalBatch(Contract):
    """What each atom of a plan found. Records of one atom are never another atom's evidence."""

    plan: QueryPlan
    status: BatchStatus
    atoms: Tuple[AtomicResult, ...]
    dependencies: Tuple[DependencyOutcome, ...] = ()
    _omit_when_empty: ClassVar[Tuple[str, ...]] = ("dependencies",)

    def validate(self) -> None:
        if [a.query_id for a in self.atoms] != [a.query_id for a in self.plan.atoms]:
            raise ValidationError("a batch has one result per plan atom, in plan order")
        answered = sum(1 for a in self.atoms if a.status is RetrievalStatus.OK)
        expected = (BatchStatus.NO_SUCCESS if not answered else
                    BatchStatus.COMPLETE_SUCCESS if answered == len(self.atoms) else BatchStatus.PARTIAL_SUCCESS)
        if self.status is not expected:
            raise ValidationError(f"batch status must be {expected.value} for {answered}/{len(self.atoms)} answered")
        if [d.query_ids for d in self.dependencies] != [d.query_ids for d in self.plan.dependencies]:
            raise ValidationError("a batch has one outcome per plan dependency")

    @property
    def per_query_view(self) -> bool:
        """True when one list of slots would blur the queries: a comparison, queries about different owners, or a
        query without (complete) records next to others. Queries that all have records for the same owner
        ("你住哪？做什么工作？") read the same as one slot list."""
        if self.dependencies:
            return True
        if len(self.atoms) < 2:
            return False
        if len({a.subject for a in self.atoms}) > 1:
            return True
        return any(a.sufficiency not in (Sufficiency.SUFFICIENT, Sufficiency.CONFLICT) for a in self.atoms)
