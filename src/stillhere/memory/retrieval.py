"""SlotRetriever: read-only memory retrieval with temporal resolution (implements ``contracts.MemoryRetriever``).

One message goes through these steps:

1. **Request.** ``MemoryQuery`` validates the persona id, limit, time and speakers. An unparseable or
   timezone-less ``as_of`` gives ``invalid_request``; it is never guessed. ``as_of=None`` uses the clock.
2. **Plan** (``stillhere.memory.planner.QueryPlanner``, Day 7 Query Understanding V2). The message becomes a
   ``QueryPlan`` of atomic queries: which parts ask something, whose fact each one asks for, which attributes,
   which time, and the attribute an elliptical part ("那你呢") carries from the last turns. Unidentified owners are
   ``ambiguous``, someone with no subject in memory (``你妈妈``) is ``unowned``, an intent memory does not record
   (``你喜欢我吗``) is ``unsupported``; none of them returns records.
3. **Atoms, one at a time.** Each retrievable atom is looked up on its own, so one atom's owner, attribute or time
   never reaches another's records:
   * **Slots.** For each attribute: the atom's subject, else (subject open) the only subject whose active records
     have that attribute (``subject_source="slot"``); records under several subjects make the atom ambiguous.
   * **Keywords** (an atom with no attribute): the store's CJK bigram segmentation, after removing the persona's
     name, pronouns, time words and ``喜欢`` (they appear in every statement or none). A record needs a shared
     character pair or word; single characters count only when the query has no pair at all. A selected record
     brings its active conflict partners along.
   * **Temporal resolution** per slot at ``as_of``: ``current`` (active, valid at as_of), ``past`` (active,
     ended at or before as_of, half-open intervals), ``corrected`` (superseded by a correction: a former belief,
     never a former fact). Disagreeing current values of a single slot give ``conflict`` with no value. Recording
     time never picks a winner; ordering uses valid time and ids only. A life period no record places
     (``你小时候住哪``) makes every slot of the atom ``unknown`` with no records.
   * **Sufficiency**: whether the atom's records answer it (``sufficient``), disagree (``conflict``), answer only
     some of its attributes (``partial``), miss the time asked (``time_unknown``), or are absent (``no_record``,
     ``not_answerable`` with the reason).
4. **Aggregate.** Hits in atom order, deterministic within an atom, cut at ``limit`` by whole groups (a conflict
   is never split; each atom keeps its first group). The result status is ``ok`` when any atom has records,
   else ``ambiguous_subject`` when an atom's owner is unclear, else ``no_match``. A message that asks several
   things (or carries an attribute from the context) also gets a ``RetrievalBatch``: every atom's records,
   sufficiency and missing evidence, and each comparison's outcome. "你住哪？我住哪？" is a ``partial_success``:
   the persona's city for one atom, nothing for the other, never one value for both. A message that asks one
   thing keeps the Day 6 result shape.

**A calendar time** (Phase 5B, scope ``at``: ``去年住哪``, ``2026年3月10号在哪上班``): the slot is resolved
against the atom's ``[start, end)`` interval instead of ``as_of``. The store's ``facts_during`` (an SQL interval
overlap on the stored UTC validity) returns the active records valid at some instant of it; records without a
validity interval count for every time. One value over the whole interval is ``at_time``; a value that changed
during it is ``segmented`` (each entry keeps its valid time, none is picked); records covering only part of it
are ``partial_time``; none is a **time gap** (``unknown``, missing ``time_gap``): no neighbouring value is
returned, and the prompt says that no record is not "nothing then" (no job record is not unemployment).
Conflicts stay conflicts and corrected values stay corrections at any time. A time no calendar places (scope
``unresolved``) looks nothing up (missing ``time_unresolved``). Relative words are counted from ``as_of``: the
planner is given the instant the lookup uses.

**Knowledge time.** ``known_at`` is what the system knew. Every lookup of one call reads the belief at that instant:
a record learned later is absent, and one corrected or retracted later reads as it stood then, so ``corrected``
lists only corrections already made. Left out, it defaults to ``as_of`` (``resolve_times``), so a look at the past
cannot see later knowledge; "where did she live then, as far as we know now" passes a newer ``known_at``.

Not semantic search, no model calls, no writes: the retriever keeps only the store's ``facts_at``,
``facts_during`` and ``history`` read methods (an optional semantic parser is the planner's, off by default). A
store without ``facts_during`` is filtered in Python with the same half-open rule. Store failures give ``error``.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from stillhere.contracts import (AtomicQuery, AtomicResult, BatchStatus, Cardinality, IntentStatus, MemoryHit,
                                 MemoryQuery, MemoryRecord, MemoryStatus, QueryPlan, RetrievalBatch,
                                 RetrievalResult, RetrievalStatus, SlotResolution, SlotStatus, SubjectStatus,
                                 Sufficiency, TemporalRole, TimeExpression, TimeScope, ValidationError)
from stillhere.contracts.retrieval import EVIDENCE_REASONS
from stillhere.contracts.planning import AnswerLevel, DependencyKind, DependencyOutcome, EvidenceState

from .planner import (DEFAULT_SLOT_CUES, DEFAULT_SUBJECT_NOUNS, PERSON_PRONOUNS, PERSONA, THING_PRONOUN, TIME_WORDS,
                      USER, QueryPlanner, SlotCue, _pronoun)
from .search import BIGRAM_WEIGHT, QUERY_STOP_CHARS, UNIGRAM_WEIGHT, index_terms, normalize, query_units
from .sqlite_store import MemoryStoreError, resolve_times

__all__ = ["SlotRetriever", "SlotCue", "DEFAULT_SLOT_CUES", "DEFAULT_SUBJECT_NOUNS"]

_EPOCH = datetime(1, 1, 1, tzinfo=timezone.utc)
# Words removed from keyword lookups: a predicate shared by preference statements, it names no object.
KEYWORD_STOP_WORDS = ("不喜欢", "喜欢")
_SOURCE_ORDER = ("explicit", "query", "context", "slot")


def _valid_key(r: MemoryRecord):
    return (r.valid_from is not None, r.valid_from or _EPOCH, r.memory_id)


@dataclass
class _AtomRun:
    atom: AtomicQuery
    status: RetrievalStatus
    subject: Optional[str] = None
    subject_source: Optional[str] = None
    slots: List[SlotResolution] = field(default_factory=list)
    groups: List[List[MemoryHit]] = field(default_factory=list)
    candidates: Tuple[str, ...] = ()
    missing: Optional[str] = None


class SlotRetriever:
    """Structured and keyword retrieval with temporal resolution over a ``SQLiteMemoryStore`` (or any object
    with its ``facts_at`` and ``history`` methods).

    ``persona_names`` maps a persona id to the names it is called by (e.g. ``{"synthetic_qiaomo": ("乔沫",)}``);
    they identify the persona as the subject and are ignored as keywords. ``clock`` supplies the default
    ``as_of``. ``planner`` replaces the default local ``QueryPlanner`` built from the same vocabulary (for example
    one with a semantic parser).
    """

    def __init__(self, store, *, persona_names: Optional[Mapping[str, Sequence[str]]] = None,
                 slot_cues: Sequence[SlotCue] = DEFAULT_SLOT_CUES,
                 subject_nouns: Mapping[str, str] = DEFAULT_SUBJECT_NOUNS,
                 clock: Optional[Callable[[], datetime]] = None, planner: Optional[QueryPlanner] = None) -> None:
        self._facts_at = store.facts_at      # read access only
        self._facts_during = getattr(store, "facts_during", None)
        self._history = store.history
        self._names = {k: tuple(normalize(n) for n in v) for k, v in (persona_names or {}).items()}
        self._nouns = sorted(((normalize(k), v) for k, v in subject_nouns.items()), key=lambda kv: -len(kv[0]))
        self._planner = planner or QueryPlanner(persona_names=persona_names, slot_cues=slot_cues,
                                                subject_nouns=subject_nouns)
        self._thing_subjects = self._planner.thing_subjects
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    # ------------------------------------------------------------------ public

    def retrieve(self, persona_id: str, text: str, *, context: Sequence[str] = (), context_roles: Sequence[str] = (),
                 as_of: Union[datetime, str, None] = None, subject: Optional[str] = None,
                 time_scope: Union[TimeScope, str, None] = None, limit: int = 5,
                 known_at: Union[datetime, str, None] = None) -> RetrievalResult:
        try:
            query = MemoryQuery(persona_id=persona_id, text=text, context=context, as_of=as_of, subject=subject,
                                time_scope=time_scope, limit=limit, context_roles=context_roles, known_at=known_at)
        except (ValidationError, TypeError) as exc:
            return RetrievalResult(status=RetrievalStatus.INVALID_REQUEST, errors=(str(exc),))
        when, known = resolve_times(query.as_of, query.known_at, self._clock)
        try:
            plan = self._planner.plan(replace(query, as_of=when))   # relative times count from the lookup's instant
        except ValueError:                                # no words at all
            scope = query.time_scope or TimeScope.CURRENT
            return RetrievalResult(status=RetrievalStatus.NO_MATCH, query=query, as_of=when, time_scope=scope,
                                   time_scope_source="explicit" if query.time_scope else "default")
        scope, scope_source = self._summary_scope(plan)
        base = dict(query=query, as_of=when, time_scope=scope, time_scope_source=scope_source,
                    known_at=query.known_at)
        try:
            return self._execute(replace(query, known_at=known), plan, when, base)
        except (sqlite3.Error, MemoryStoreError) as exc:
            return RetrievalResult(status=RetrievalStatus.ERROR, errors=(f"{type(exc).__name__}: {exc}",), **base)

    def plan(self, query: MemoryQuery) -> QueryPlan:
        """The plan ``retrieve`` would follow for ``query`` (no store access)."""
        return self._planner.plan(query if query.as_of is not None else replace(query, as_of=self._clock()))

    # ------------------------------------------------------------------ atoms

    def _execute(self, query: MemoryQuery, plan: QueryPlan, when: datetime, base: dict) -> RetrievalResult:
        active = [r for r in self._facts_at(query.persona_id, known_at=query.known_at)
                  if r.status is MemoryStatus.ACTIVE]
        runs = [self._run_atom(query, atom, when, active) for atom in plan.atoms]
        hits, truncated = self._hits(runs, query.limit)
        hit_ids = [h.record.memory_id for h in hits]
        slots: List[SlotResolution] = []
        for run in runs:
            for s in run.slots:
                if all((s.subject, s.attribute) != (x.subject, x.attribute) for x in slots):
                    slots.append(s)
        results = [self._atom_result(run, hit_ids) for run in runs]
        if hits:
            status = RetrievalStatus.OK
        elif any(r.status is RetrievalStatus.AMBIGUOUS_SUBJECT for r in runs):
            status = RetrievalStatus.AMBIGUOUS_SUBJECT
        else:
            status = RetrievalStatus.NO_MATCH
        candidates: Tuple[str, ...] = ()
        if status is RetrievalStatus.AMBIGUOUS_SUBJECT:
            candidates = tuple(sorted({c for r in runs for c in r.candidates}))
        if len(runs) == 1 and status is not RetrievalStatus.OK and runs[0].status is not RetrievalStatus.NO_MATCH:
            slots = []                                    # an unidentified owner shows no slot (Day 6 shape)
        subject, subject_source = self._summary_subject(runs)
        batch = None
        if plan.multi_intent or any(a.intent_source == "focus" for a in plan.atoms):
            answered = sum(1 for r in results if r.status is RetrievalStatus.OK)
            batch_status = (BatchStatus.NO_SUCCESS if not answered else BatchStatus.COMPLETE_SUCCESS
                            if answered == len(results) else BatchStatus.PARTIAL_SUCCESS)
            batch = RetrievalBatch(plan=plan, status=batch_status, atoms=tuple(results),
                                   dependencies=tuple(self._dependency(d, results) for d in plan.dependencies))
        times = {a.time for a in plan.atoms}
        time = times.pop() if len(times) == 1 else None
        evidence = None
        if len(runs) == 1 and not hits and runs[0].missing in EVIDENCE_REASONS:
            evidence = runs[0].missing
        return RetrievalResult(status=status, subject=subject, subject_source=subject_source,
                               subject_candidates=candidates, slots=tuple(slots), hits=tuple(hits),
                               truncated=truncated, batch=batch, time=time, evidence=evidence, **base)

    def _run_atom(self, query: MemoryQuery, atom: AtomicQuery, when: datetime, active: List[MemoryRecord]) -> _AtomRun:
        run = _AtomRun(atom, RetrievalStatus.NO_MATCH, atom.subject, atom.subject_source)
        if atom.subject_status is SubjectStatus.AMBIGUOUS:
            run.status, run.missing = RetrievalStatus.AMBIGUOUS_SUBJECT, atom.ambiguity
            run.candidates = self._candidates(atom, active)
            return run
        if atom.subject_status is SubjectStatus.UNOWNED or atom.intent not in (IntentStatus.SLOT,
                                                                                 IntentStatus.KEYWORD):
            run.missing = atom.ambiguity or "not_answerable"
            return run
        if atom.time_scope is TimeScope.UNRESOLVED:       # 前阵子, 2026年2月30号: no time to look up, no fallback
            run.missing = "time_unresolved"
            return run
        subject = atom.subject
        source = atom.subject_source
        if atom.intent is IntentStatus.SLOT:
            subjects = []
            for attr in atom.attributes:
                subj, src = subject, source
                if subj is None:
                    owners = sorted({r.subject for r in active if r.attribute == attr})
                    if len(owners) > 1:
                        run.status, run.missing = RetrievalStatus.AMBIGUOUS_SUBJECT, "ambiguous:several_owners"
                        run.candidates, run.slots, run.groups = tuple(owners), [], []
                        return run
                    if not owners:
                        continue
                    subj, src = owners[0], "slot"
                subjects.append((subj, src))
                slot = self._resolve(query, subj, attr, active, when, atom.time_scope, atom.time)
                run.slots.append(slot)
                matched = atom.matched + self._subject_words(subj, query)
                for role, records in self._slot_groups(slot, atom.time_scope):
                    if records:
                        run.groups.append([MemoryHit(record=r, role=role, via="slot", matched=matched, score=1.0)
                                           for r in records])
            distinct = {s for s, _ in subjects}
            run.subject, run.subject_source = subjects[0] if len(distinct) == 1 else (subject, source)
        else:
            run.groups, keys = self._by_keyword(query, atom, when, active, subject)
            run.slots = [self._resolve(query, s, a, active, when, atom.time_scope, atom.time) for s, a in keys]
            run.subject, run.subject_source = subject, source
        if atom.period_unplaced:                          # 你小时候住哪: no valid time falls in that period
            run.slots = [replace(s, status=SlotStatus.UNKNOWN, value=None, conflict_ids=()) for s in run.slots]
            run.groups, run.missing = [], "life_period"
        if run.groups:
            run.status = RetrievalStatus.OK
        elif run.missing is None and atom.time_scope is TimeScope.AT:
            run.missing = "time_gap" if any(s.current or s.past for s in run.slots) else "no_record"   # other times
        elif run.missing is None:
            run.missing = "time_unknown" if any(s.current or s.past for s in run.slots) else "no_record"
        return run

    def _candidates(self, atom: AtomicQuery, active: List[MemoryRecord]) -> Tuple[str, ...]:
        attrs = set(atom.attributes)
        subjects = {r.subject for r in active if not attrs or r.attribute in attrs}
        span = atom.span
        if _pronoun(span, PERSON_PRONOUNS) and not _pronoun(span, THING_PRONOUN):
            subjects -= self._thing_subjects            # 她/他 is not the cat
        elif _pronoun(span, THING_PRONOUN) and not _pronoun(span, PERSON_PRONOUNS):
            subjects -= {PERSONA, USER}                 # 它 is not a person
        return tuple(sorted(subjects))

    @staticmethod
    def _hits(runs: List[_AtomRun], limit: int) -> Tuple[List[MemoryHit], bool]:
        """Hits in atom order. Within an atom, groups are added while they fit the limit; the atom's first group is
        kept whole even above it, so every answered atom keeps its evidence. A conflict group is never split."""
        hits: List[MemoryHit] = []
        seen = set()
        truncated = False
        for run in runs:
            for i, group in enumerate(run.groups):
                group = [h for h in group if h.record.memory_id not in seen]
                if i == 0 or len(hits) + len(group) <= limit:
                    hits.extend(group)
                    seen.update(h.record.memory_id for h in group)
                else:
                    truncated = True
                    break
        return hits, truncated

    @staticmethod
    def _atom_result(run: _AtomRun, hit_ids: List[str]) -> AtomicResult:
        own = {h.record.memory_id for g in run.groups for h in g}
        ids = tuple(i for i in hit_ids if i in own)
        status = run.status
        keys = tuple(f"{s.subject}/{s.attribute}" for s in run.slots)
        missing = run.missing
        if ids:
            shown = {(h.record.subject, h.record.attribute) for g in run.groups for h in g if h.record.memory_id in ids}
            unanswered = [f"{s.subject}/{s.attribute}" for s in run.slots if (s.subject, s.attribute) not in shown]
            if any(s.status is SlotStatus.CONFLICT for s in run.slots if (s.subject, s.attribute) in shown):
                sufficiency = Sufficiency.CONFLICT
            else:
                sufficiency = Sufficiency.SUFFICIENT
            if unanswered:
                sufficiency, missing = Sufficiency.PARTIAL, "no_record:" + ",".join(unanswered)
            elif any(s.status is SlotStatus.PARTIAL_TIME for s in run.slots):
                sufficiency, missing = Sufficiency.PARTIAL, "time_partial"   # part of the asked time has no record
            else:
                missing = None
        elif missing in ("time_unknown", "life_period", "time_gap", "time_unresolved"):
            sufficiency = Sufficiency.TIME_UNKNOWN
        elif missing == "no_record":
            sufficiency = Sufficiency.NO_RECORD
        else:
            sufficiency = Sufficiency.NOT_ANSWERABLE
        return AtomicResult(query_id=run.atom.query_id, status=status, sufficiency=sufficiency, subject=run.subject,
                            subject_source=run.subject_source, slot_keys=keys, memory_ids=ids, missing=missing,
                            candidates=run.candidates)

    @staticmethod
    def _dependency(dep, results: List[AtomicResult]) -> DependencyOutcome:
        atoms = [r for r in results if r.query_id in dep.query_ids]
        have = [r.status is RetrievalStatus.OK for r in atoms]
        if not dep.supported:
            evidence = EvidenceState.UNSUPPORTED_COMPARISON
        elif not all(have):
            evidence = EvidenceState.MISSING
        elif any(r.sufficiency is Sufficiency.CONFLICT for r in atoms):
            evidence = EvidenceState.CONFLICT
        else:
            evidence = EvidenceState.AVAILABLE
        if evidence is EvidenceState.AVAILABLE and dep.kind is DependencyKind.DEPENDENT_COMPARE:
            answer = AnswerLevel.COMPLETE
        else:
            answer = AnswerLevel.SAFE_PARTIAL if any(have) else AnswerLevel.NONE
        return DependencyOutcome(kind=dep.kind, query_ids=dep.query_ids, evidence=evidence, answer=answer,
                                 relation=dep.relation)

    # ------------------------------------------------------------------ summaries

    @staticmethod
    def _summary_scope(plan: QueryPlan) -> Tuple[TimeScope, str]:
        scopes = {a.time_scope for a in plan.atoms}
        sources = [a.time_scope_source for a in plan.atoms]
        if len(plan.atoms) == 1 or len(scopes) == 1:
            first = plan.atoms[0]
            source = next((s for s in ("explicit", "query") if s in sources), first.time_scope_source)
            return first.time_scope, source
        return TimeScope.ALL, "explicit" if "explicit" in sources else "query"

    @staticmethod
    def _summary_subject(runs: List[_AtomRun]) -> Tuple[Optional[str], Optional[str]]:
        """One subject when every atom has the same one; its source is the strongest any atom has."""
        subjects = {r.subject for r in runs}
        if len(subjects) != 1 or None in subjects:
            return None, None
        sources = {r.subject_source for r in runs}
        return runs[0].subject, next((s for s in _SOURCE_ORDER if s in sources), runs[0].subject_source)

    def _subject_words(self, subj: str, query: MemoryQuery) -> Tuple[str, ...]:
        q = normalize(query.text)
        if subj == PERSONA:
            return tuple(n for n in self._names.get(query.persona_id, ()) if n in q)
        return tuple(w for w, s in self._nouns if s == subj and w in q)[:1]

    @staticmethod
    def _slot_groups(slot: SlotResolution, scope: TimeScope):
        if scope is TimeScope.AT:
            return [(TemporalRole.AT_TIME, slot.at_time)]
        if scope is TimeScope.CURRENT:
            return [(TemporalRole.CURRENT, slot.current)]
        return [(TemporalRole.PAST, slot.past), (TemporalRole.CURRENT, slot.current)]

    def _by_keyword(self, query: MemoryQuery, atom: AtomicQuery, when: datetime, active: List[MemoryRecord],
                    subject: Optional[str]) -> Tuple[List[List[MemoryHit]], List[Tuple[str, str]]]:
        cleaned = atom.span
        for word in self._names.get(query.persona_id, ()) + TIME_WORDS + KEYWORD_STOP_WORDS:
            if word:
                cleaned = cleaned.replace(normalize(word), " ")
        for c in "我你她他它们":
            cleaned = cleaned.replace(c, " ")
        units = query_units(cleaned)
        bigrams = [b for b in units.bigrams if not all(ch in QUERY_STOP_CHARS for ch in b)]
        unigrams = list(units.unigrams)
        pool = [r for r in active if subject is None or r.subject == subject]
        scored = []
        during = None
        if atom.time_scope is TimeScope.AT:
            during = {r.memory_id for r in self._during(query, atom.time, None, None, active)}
        for r in pool:
            role = self._role(r, when) if during is None else (TemporalRole.AT_TIME if r.memory_id in during else None)
            if role is None or (role is TemporalRole.PAST and atom.time_scope is TimeScope.CURRENT):
                continue
            terms = index_terms(r.statement, r.value, r.subject, r.attribute)
            bi = [b for b in bigrams if b in set(terms.bigrams)]
            uni = [u for u in unigrams if u in set(terms.unigrams)]
            if bi or (not bigrams and uni):
                score = BIGRAM_WEIGHT * len(bi) + UNIGRAM_WEIGHT * len(uni)
                scored.append((-score, r.subject, r.attribute, _valid_key(r), r, role, tuple(bi + uni), score))
        scored.sort(key=lambda x: x[:4])
        by_id = {r.memory_id: r for r in active}
        groups: List[List[MemoryHit]] = []
        seen = set()
        slot_keys: List[Tuple[str, str]] = []
        for *_, r, role, matched, score in scored:
            if r.memory_id in seen:
                continue
            group = [MemoryHit(record=r, role=role, via="keyword", matched=matched, score=score)]
            seen.add(r.memory_id)
            if role in (TemporalRole.CURRENT, TemporalRole.AT_TIME):
                for other_id in sorted(r.conflicts_with):
                    other = by_id.get(other_id)
                    valid = other is not None and (other.is_valid_at(when) if during is None else other_id in during)
                    if valid and other_id not in seen:
                        group.append(MemoryHit(record=other, role=role, via="keyword", matched=(), score=0.0))
                        seen.add(other_id)
            groups.append(group)
            for h in group:
                key = (h.record.subject, h.record.attribute)
                if key not in slot_keys:
                    slot_keys.append(key)
        return groups, slot_keys

    # ------------------------------------------------------------------ temporal resolution

    @staticmethod
    def _role(r: MemoryRecord, when: datetime) -> Optional[TemporalRole]:
        if r.is_valid_at(when):
            return TemporalRole.CURRENT
        if r.valid_to is not None and r.valid_to <= when:
            return TemporalRole.PAST
        return None                                   # starts after as_of

    def _during(self, query: MemoryQuery, time: TimeExpression, subject: Optional[str], attribute: Optional[str],
                active: List[MemoryRecord]) -> List[MemoryRecord]:
        """Active records valid at some instant of ``[time.start, time.end)``: the store's SQL interval lookup, or
        the same half-open rule in Python for a store without one."""
        if self._facts_during is not None:
            return [r for r in self._facts_during(query.persona_id, start=time.start, end=time.end, subject=subject,
                                                  attribute=attribute, known_at=query.known_at)
                    if r.status is MemoryStatus.ACTIVE]
        return [r for r in active if (subject is None or r.subject == subject)
                and (attribute is None or r.attribute == attribute)
                and (r.valid_from is None or r.valid_from < time.end)
                and (r.valid_to is None or time.start < r.valid_to)]

    @staticmethod
    def _covers(records: List[MemoryRecord], time: TimeExpression) -> bool:
        """True when the records' validity together leaves no instant of the asked interval uncovered."""
        cursor = time.start
        for r in sorted(records, key=_valid_key):
            start = r.valid_from if r.valid_from is not None and r.valid_from > time.start else time.start
            if start > cursor:
                return False
            if r.valid_to is None:
                return True
            cursor = max(cursor, r.valid_to)
            if cursor >= time.end:
                return True
        return cursor >= time.end

    def _at_time(self, query: MemoryQuery, subject: str, attribute: str, active: List[MemoryRecord],
                 time: TimeExpression, cardinality: Cardinality) -> Tuple[SlotStatus, Tuple[MemoryRecord, ...],
                                                                         Tuple[str, ...], Optional[str]]:
        """(status, records valid during the asked time, conflict ids, value) for scope "at"."""
        during = sorted(self._during(query, time, subject, attribute, active), key=_valid_key)
        conflict_ids = sorted({x.memory_id for x in during for y in during
                               if x.memory_id != y.memory_id
                               and (y.memory_id in x.conflicts_with or x.memory_id in y.conflicts_with
                                    or x.contradicts(y))})
        if not during:
            return SlotStatus.UNKNOWN, (), (), None
        if conflict_ids:
            return SlotStatus.CONFLICT, tuple(during), tuple(conflict_ids), None
        if not self._covers(during, time):
            return SlotStatus.PARTIAL_TIME, tuple(during), (), None
        values = {r.value for r in during}
        if len(values) == 1 and (cardinality is Cardinality.SINGLE or len(during) == 1):
            return SlotStatus.AT_TIME, tuple(during), (), during[0].value
        if all(self._covers([r], time) for r in during):
            return SlotStatus.MULTI_VALUE, tuple(during), (), None   # several values true all along (hobbies)
        return SlotStatus.SEGMENTED, tuple(during), (), None

    def _resolve(self, query: MemoryQuery, subject: str, attribute: str, active: List[MemoryRecord], when: datetime,
                 scope: TimeScope, time: Optional[TimeExpression] = None) -> SlotResolution:
        records = [r for r in active if r.subject == subject and r.attribute == attribute]
        current = sorted((r for r in records if self._role(r, when) is TemporalRole.CURRENT), key=_valid_key)
        past = sorted((r for r in records if self._role(r, when) is TemporalRole.PAST), key=_valid_key)
        corrected = [r for r in self._history(query.persona_id, subject, attribute, known_at=query.known_at)
                     if r.status is MemoryStatus.SUPERSEDED]
        kinds = {r.cardinality for r in records}
        cardinality = kinds.pop() if len(kinds) == 1 else Cardinality.UNKNOWN
        conflict_ids = sorted({x.memory_id for x in current for y in current
                               if x.memory_id != y.memory_id
                               and (y.memory_id in x.conflicts_with or x.memory_id in y.conflicts_with
                                    or x.contradicts(y))})
        value = None
        if scope is TimeScope.AT and time is not None:
            status, at_time, at_conflicts, value = self._at_time(query, subject, attribute, active, time, cardinality)
            return SlotResolution(subject=subject, attribute=attribute, cardinality=cardinality, status=status,
                                  current=tuple(current), past=tuple(past), corrected=tuple(corrected),
                                  conflict_ids=at_conflicts, value=value, at_time=at_time)
        if conflict_ids:
            status = SlotStatus.CONFLICT
        elif scope is TimeScope.HISTORICAL:
            status = SlotStatus.HISTORICAL if past else SlotStatus.UNKNOWN
        elif not current:
            status = SlotStatus.UNKNOWN
        elif len({r.value for r in current}) == 1 and (cardinality is Cardinality.SINGLE or len(current) == 1):
            status, value = SlotStatus.RESOLVED, current[0].value
        else:
            status = SlotStatus.MULTI_VALUE
        return SlotResolution(subject=subject, attribute=attribute, cardinality=cardinality, status=status,
                              current=tuple(current), past=tuple(past), corrected=tuple(corrected),
                              conflict_ids=tuple(conflict_ids), value=value)
