"""Context builder: a retrieval result and a chat history in, the memory context, its view and the turns to send
out, all within a deterministic token budget.

Steps, in order:

1. **Lookup status.** No result means memory is off: nothing is added and the prompt is the plain renderer
   output. ``error`` and ``invalid_request`` fail closed: nothing is added (the error text never reaches the
   prompt) and the turn is tagged ``MEMORY_RETRIEVAL_ERROR``. ``no_match`` and ``ambiguous_subject`` give a view
   with a status line and no records.
2. **Slot groups.** Hits are grouped by slot in retrieval order. Each group is indivisible: a fact is never cut,
   every side of a conflict is kept together (missing sides are taken from the slot resolution), and a corrected
   record only appears next to the slot that corrected it, labelled as a mis-record. Corrected records are shown
   only when the user asks about corrections (``CORRECTION_CUES``), and never enter ``memory_context``.
3. **Safety.** Every line of a group must pass the renderer's checks (no line break, special token or reserved
   tag). A group with an unsafe statement is dropped whole and tagged ``MEMORY_FACT_REJECTED``.
4. **Budget** (only with a ``ContextBudget``). ``max_prompt_tokens = context_window - max_new_tokens -
   reserved_tokens``. The smallest legal prompt (system line and the latest user turn, no memory) must fit or
   ``ContextOverBudget`` is raised. Slot groups are added in order while the memory block stays within
   ``max_memory_tokens`` and the prompt within ``max_prompt_tokens``; the first group that does not fit and every
   later one are left out and counted in the block (``MEMORY_CONTEXT_OVER_BUDGET``). Then the earliest whole turns
   are dropped two at a time (so the history still starts with a user turn) until the prompt and the history fit.
   Every count is taken on the final rendered text with the injected counter (the real tokenizer for a real
   model); the counts are reproducible for the same input.

Only the injected ``count_tokens`` is used for counting; nothing is truncated inside a turn, a fact or a special
token.

**Several asked queries** (Day 7 V2: the result carries a ``RetrievalBatch`` whose queries one slot list would
blur: different owners, a query without records next to others, a comparison; ``RetrievalBatch.per_query_view``).
The view is ``memory_block_v2``: besides the slot groups it lists every asked query (``ContextQuery``)
with its owner, attributes, time scope, sufficiency and why records are missing (no record, someone with no
records, unclear owner, unsupported intent, unknown time, partial, conflict), and each comparison's evidence. A
shown slot must belong to an asked query, so no fact can stand in for another query's answer. The query lines are
always rendered (they are the negative evidence: "no record of the user's city"); the budget only removes whole
slot groups, and a query whose groups were left out says so instead of looking unanswered. A partial batch adds
the normal state ``MEMORY_PARTIAL_ANSWER``.

**Evidence first** (Stage J, templates with the ``compact_v1`` memory layout): the groups are tried in priority
order instead of retrieval order: the first group of every query that has records (in query order), then the
rest; a group that does not fit is skipped and the next one is still tried, and the chosen groups are rendered in
retrieval order. A query whose groups were all left out is blocked (``MEMORY_BUDGET_BLOCKED``): its line says the
entries could not be shown (not "no record"), and ``BuiltContext.allocation`` records every group's token cost,
whether it was shown and the blocked queries, so the allocation can be audited. Other layouts keep the step 4 order
exactly.

**A calendar time** (Phase 5B, scope ``at``): the facts are the records valid during the asked interval, marked
``at_time``; the view carries the interval (``time``, per query in v2) and, for a one-query message without
records, why (``evidence``: a time gap, a time no calendar places, a life period). The model sees what held at
the time asked and what is missing, and never has to work dates out of the current facts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from stillhere.contracts import (AnswerNote, BatchStatus, ChatMessage, ContextDependency, ContextFact, ContextQuery,
                                 ContextSlot, FactRole, MemoryContextView, PersonaCard, RetrievalBatch, RetrievalResult,
                                 RetrievalStatus, RetrievedMemory, SlotResolution, SlotStatus, Sufficiency, TemporalRole)
from stillhere.contracts.retrieval import MEMORY_VIEW_V2
from stillhere.data.templates import SituationMode, TemplateSpec

from .prompting import InputOverBudget, PromptError, RenderedPrompt, render_messages, slot_lines, system_text

# Words that ask about the correction history ("did you get it wrong", "was it corrected").
CORRECTION_CUES = ("记错", "更正", "纠正", "改正", "写错", "弄错", "搞错", "记混", "错记")

# Memory states that are normal outcomes, not failures.
MEMORY_NO_MATCH = "MEMORY_NO_MATCH"
MEMORY_AMBIGUOUS_SUBJECT = "MEMORY_AMBIGUOUS_SUBJECT"
MEMORY_CONFLICT = "MEMORY_CONFLICT"
MEMORY_INJECTION_SKIPPED = "MEMORY_INJECTION_SKIPPED"   # memory was on but nothing could be injected
MEMORY_PARTIAL_ANSWER = "MEMORY_PARTIAL_ANSWER"         # some asked queries have records, others none
# Failures.
MEMORY_RETRIEVAL_ERROR = "MEMORY_RETRIEVAL_ERROR"
MEMORY_CONTEXT_OVER_BUDGET = "MEMORY_CONTEXT_OVER_BUDGET"
MEMORY_FACT_REJECTED = "MEMORY_FACT_REJECTED"
MEMORY_BUDGET_BLOCKED = "MEMORY_BUDGET_BLOCKED"         # a query with records has none shown (budget)
MEMORY_STATUS_TAGS = (MEMORY_NO_MATCH, MEMORY_AMBIGUOUS_SUBJECT, MEMORY_CONFLICT, MEMORY_INJECTION_SKIPPED,
                      MEMORY_PARTIAL_ANSWER)
MEMORY_FAILURE_TAGS = (MEMORY_RETRIEVAL_ERROR, MEMORY_CONTEXT_OVER_BUDGET, MEMORY_FACT_REJECTED, MEMORY_BUDGET_BLOCKED)
EVIDENCE_FIRST = "evidence_first"


class ContextOverBudget(InputOverBudget):
    """The smallest legal prompt (system line plus the latest user turn, no memory) exceeds the budget."""


@dataclass(frozen=True)
class ContextBudget:
    count_tokens: Callable[[str], int]
    context_window: int
    max_new_tokens: int
    max_memory_tokens: int
    max_history_tokens: Optional[int] = None   # None: the history may use whatever the prompt budget leaves
    reserved_tokens: int = 0                    # tokens a backend adds outside the rendered text (Qwen3: none)

    def __post_init__(self) -> None:
        if min(self.context_window, self.max_new_tokens) < 1 or self.max_memory_tokens < 0 or self.reserved_tokens < 0:
            raise ValueError("context_window and max_new_tokens must be >= 1; memory and reserved tokens >= 0")
        if self.max_history_tokens is not None and self.max_history_tokens < 0:
            raise ValueError("max_history_tokens must be >= 0")
        if self.max_prompt_tokens < 1:
            raise ValueError(f"no room for a prompt: context_window {self.context_window} - max_new_tokens "
                             f"{self.max_new_tokens} - reserved {self.reserved_tokens} < 1")

    @property
    def max_prompt_tokens(self) -> int:
        return self.context_window - self.max_new_tokens - self.reserved_tokens

    def describe(self) -> Dict[str, Optional[int]]:
        return {"context_window": self.context_window, "max_new_tokens": self.max_new_tokens,
                "reserved_tokens": self.reserved_tokens, "max_prompt_tokens": self.max_prompt_tokens,
                "max_memory_tokens": self.max_memory_tokens, "max_history_tokens": self.max_history_tokens}


@dataclass(frozen=True)
class TokenCounts:
    """Token counts of the final prompt and its parts, from the budget's counter."""

    prompt: int        # the whole rendered prompt
    base: int          # system line + latest user turn + generation prompt, no memory, no history
    memory: int        # what the memory block adds
    history: int       # what the earlier turns add
    user: int          # the latest user turn, with its role markers
    system: int        # the system message without memory (persona situation line)


@dataclass(frozen=True)
class BuiltContext:
    messages: Tuple[ChatMessage, ...]               # the turns to send, ending with the latest user turn
    memory_context: Tuple[RetrievedMemory, ...]
    memory_view: Optional[MemoryContextView]
    prompt: RenderedPrompt                          # the final rendered text (what the backend will render)
    memory_tags: Tuple[str, ...] = ()               # normal memory states
    failure_tags: Tuple[str, ...] = ()
    rejected_ids: Tuple[str, ...] = ()              # records of groups dropped as unsafe
    omitted_ids: Tuple[str, ...] = ()               # records of groups left out by the memory budget
    tokens: Optional[TokenCounts] = None            # None without a budget
    budget: Dict[str, Optional[int]] = field(default_factory=dict)
    allocation: Optional[Dict[str, Any]] = None     # evidence-first layouts with a budget: the audit of step 4
    blocked_ids: Tuple[str, ...] = ()               # records of queries the budget blocked (MEMORY_BUDGET_BLOCKED)

    @property
    def dropped_turns(self) -> int:
        return self.prompt.dropped_turns


def asks_about_corrections(text: str) -> bool:
    return any(cue in text for cue in CORRECTION_CUES)


@dataclass(frozen=True)
class _Group:
    slot: ContextSlot
    memories: Tuple[RetrievedMemory, ...]           # the group's non-corrected records, in fact order

    @property
    def ids(self) -> Tuple[str, ...]:
        return tuple(f.memory_id for f in self.slot.facts)


def _fallback_status(hits) -> SlotStatus:
    current = [h for h in hits if h.role is TemporalRole.CURRENT]
    if not current:
        return SlotStatus.HISTORICAL
    return SlotStatus.RESOLVED if len(current) == 1 else SlotStatus.MULTI_VALUE


def slot_groups(result: RetrievalResult, *, show_corrections: bool) -> List[_Group]:
    """The result's hits as indivisible slot groups, in retrieval order."""
    resolutions: Dict[Tuple[str, str], SlotResolution] = {(s.subject, s.attribute): s for s in result.slots}
    order: List[Tuple[str, str]] = []
    by_slot: Dict[Tuple[str, str], list] = {}
    for h in result.hits:
        key = (h.record.subject, h.record.attribute)
        if key not in by_slot:
            order.append(key)
            by_slot[key] = []
        by_slot[key].append(h)
    groups: List[_Group] = []
    for key in order:
        hits = by_slot[key]
        resolution = resolutions.get(key)
        status = resolution.status if resolution is not None else _fallback_status(hits)
        conflict = set(resolution.conflict_ids) if resolution is not None else set()
        entries: List[Tuple[RetrievedMemory, FactRole]] = []
        seen = set()
        for h in hits:
            role = FactRole.PAST if h.role is TemporalRole.PAST else (
                FactRole.UNVERIFIED if h.record.memory_id in conflict else
                FactRole.AT_TIME if h.role is TemporalRole.AT_TIME else FactRole.CURRENT)
            entries.append((RetrievedMemory(record=h.record, score=h.score, retriever=result.retriever), role))
            seen.add(h.record.memory_id)
        if status is SlotStatus.CONFLICT and resolution is not None:
            for r in resolution.current + resolution.at_time:   # never split a conflict: bring every side
                if r.memory_id in conflict and r.memory_id not in seen:
                    entries.append((RetrievedMemory(record=r, score=0.0, retriever=result.retriever),
                                    FactRole.UNVERIFIED))
                    seen.add(r.memory_id)
        # past first, then what holds now; a stable order that never depends on recording time
        entries.sort(key=lambda e: 0 if e[1] is FactRole.PAST else 1)
        facts = [ContextFact(memory_id=m.record.memory_id, role=role) for m, role in entries]
        if show_corrections and resolution is not None:
            facts += [ContextFact(memory_id=r.memory_id, role=FactRole.CORRECTED, corrected_by=r.superseded_by,
                                  record=r) for r in resolution.corrected
                      if r.superseded_by is not None and r.memory_id not in seen]
        slot = ContextSlot(subject=key[0], attribute=key[1], status=status, facts=tuple(facts))
        groups.append(_Group(slot, tuple(m for m, _ in entries)))
    return groups


def _key(group: _Group) -> str:
    return f"{group.slot.subject}/{group.slot.attribute}"


def _queries(result: RetrievalResult, batch: RetrievalBatch, chosen: Sequence[_Group]) -> Tuple[ContextQuery, ...]:
    """One line per asked query: the shown slots that answer it, or why there are none. A query whose records were
    left out (budget or an unsafe statement) says so instead of looking unanswered."""
    shown = {_key(g) for g in chosen}
    found = {f"{h.record.subject}/{h.record.attribute}" for h in result.hits}
    out = []
    for atom, planned in zip(batch.atoms, batch.plan.atoms):
        answered = bool(atom.memory_ids)
        slots = tuple(k for k in atom.slot_keys if k in shown) if answered else ()
        omitted = answered and any(k in found and k not in shown for k in atom.slot_keys)
        out.append(ContextQuery(query_id=atom.query_id, subject=atom.subject, attributes=planned.attributes,
                                time_scope=planned.time_scope, sufficiency=atom.sufficiency, reason=atom.missing,
                                slots=slots, omitted=omitted, time=planned.time))
    return tuple(out)


def _view(result: RetrievalResult, groups: Sequence[_Group], omitted: int) -> MemoryContextView:
    batch = result.batch if result.batch is not None and result.batch.per_query_view else None
    if batch is None:
        return MemoryContextView(status=result.status, as_of=result.as_of, time_scope=result.time_scope,
                                 slots=tuple(g.slot for g in groups), omitted_slots=omitted, time=result.time,
                                 evidence=result.evidence)
    deps = tuple(ContextDependency(kind=d.kind, query_ids=d.query_ids, evidence=d.evidence, answer=d.answer,
                                   relation=d.relation) for d in batch.dependencies)
    return MemoryContextView(status=result.status, as_of=result.as_of, time_scope=result.time_scope,
                             slots=tuple(g.slot for g in groups), omitted_slots=omitted, version=MEMORY_VIEW_V2,
                             queries=_queries(result, batch, groups), dependencies=deps)


def _answered_keys(result: RetrievalResult) -> List[Tuple[str, Tuple[str, ...]]]:
    """(query id, slot keys) of every query that has records, in query order (one entry for a one-slot-list view)."""
    batch = result.batch if result.batch is not None and result.batch.per_query_view else None
    if batch is None:
        keys = tuple(dict.fromkeys(f"{h.record.subject}/{h.record.attribute}" for h in result.hits))
        return [("q1", keys)] if keys else []
    return [(a.query_id, tuple(a.slot_keys)) for a in batch.atoms if a.memory_ids]


def _evidence_first(groups: Sequence[_Group], result: RetrievalResult,
                    memory_tokens: Callable[[List[_Group]], Optional[int]]) -> Tuple[List[_Group], Dict[str, Any]]:
    """Choose groups in priority order (the first group of each query with records, then the rest), skipping a
    group that does not fit; ``memory_tokens`` gives a trial's memory tokens, or None when it does not fit."""
    order = {id(g): i for i, g in enumerate(groups)}
    answered = _answered_keys(result)
    first = []
    for _, keys in answered:
        g = next((g for g in groups if _key(g) in keys), None)
        if g is not None and g not in first:
            first.append(g)
    priority = first + [g for g in groups if g not in first]
    chosen: List[_Group] = []
    used = memory_tokens([]) or 0
    rows = {}
    for g in priority:
        trial = sorted(chosen + [g], key=lambda x: order[id(x)])
        n = memory_tokens(trial)
        rows[id(g)] = {"slot": _key(g), "ids": list(g.ids), "priority": priority.index(g) + 1,
                       "tokens": None if n is None else n - used, "shown": n is not None}
        if n is not None:
            chosen, used = trial, n
    shown, present = {_key(g) for g in chosen}, {_key(g) for g in groups}
    blocked = [q for q, keys in answered if any(k in present for k in keys) and not any(k in shown for k in keys)]
    return chosen, {"policy": EVIDENCE_FIRST, "groups": [rows[id(g)] for g in groups], "memory_tokens": used,
                    "blocked_queries": blocked}


def _memories(groups: Sequence[_Group]) -> Tuple[RetrievedMemory, ...]:
    return tuple(m for g in groups for m in g.memories)


def _safe(group: _Group) -> bool:
    try:
        slot_lines(group.slot, {m.record.memory_id: m.record for m in group.memories})
    except PromptError:
        return False
    return True


def build_context(messages: Sequence[ChatMessage], spec: TemplateSpec, situation: Optional[str],
                  result: Optional[RetrievalResult], *, budget: Optional[ContextBudget] = None,
                  show_corrections: Optional[bool] = None, card: Optional[PersonaCard] = None,
                  notes: Sequence[AnswerNote] = ()) -> BuiltContext:
    """Build what one request sends. ``result`` None means memory is off. ``show_corrections`` None: decided from
    the latest user turn (``CORRECTION_CUES``). ``card`` and ``notes`` go to a persona-card template's system
    message (Phase 5B); they count as system tokens, not memory."""
    messages = tuple(messages)
    if not messages:
        raise ValueError("messages must end with the user turn to answer")
    if result is not None and spec.situation_mode is SituationMode.NONE:
        raise PromptError(f"template {spec.name} is the fixed P1 baseline format: it takes no memory context")
    memory_tags: List[str] = []
    failure_tags: List[str] = []
    groups: List[_Group] = []
    rejected: Tuple[str, ...] = ()
    with_view = False
    if result is None:
        pass
    elif result.status in (RetrievalStatus.ERROR, RetrievalStatus.INVALID_REQUEST):
        failure_tags.append(MEMORY_RETRIEVAL_ERROR)
        memory_tags.append(MEMORY_INJECTION_SKIPPED)
    else:
        with_view = True
        if result.status is RetrievalStatus.NO_MATCH:
            memory_tags.append(MEMORY_NO_MATCH)
        elif result.status is RetrievalStatus.AMBIGUOUS_SUBJECT:
            memory_tags.append(MEMORY_AMBIGUOUS_SUBJECT)
        else:
            if show_corrections is None:
                show_corrections = asks_about_corrections("".join(messages[-1].bubbles))
            candidates = slot_groups(result, show_corrections=show_corrections)
            groups = [g for g in candidates if _safe(g)]
            rejected = tuple(i for g in candidates if g not in groups for i in g.ids)
            if rejected:
                failure_tags.append(MEMORY_FACT_REJECTED)
            if not groups:
                with_view = False
                memory_tags.append(MEMORY_INJECTION_SKIPPED)

    def render(turns, chosen, omitted, view_on=True) -> RenderedPrompt:
        if not (with_view and view_on):
            return render_messages(turns, spec, situation, (), card=card, notes=notes)
        return render_messages(turns, spec, situation, _memories(chosen), _view(result, chosen, omitted), card=card,
                               notes=notes)

    last = messages[-1:]
    omitted_ids: Tuple[str, ...] = ()
    blocked_ids: Tuple[str, ...] = ()
    allocation: Optional[Dict[str, Any]] = None
    tokens = None
    if budget is None:
        chosen, omitted = groups, 0
        prompt = render(messages, chosen, omitted)
    else:
        count = budget.count_tokens
        base = count(render(last, [], 0, view_on=False).text)
        if base > budget.max_prompt_tokens:
            raise ContextOverBudget(f"the latest user turn alone needs {base} prompt tokens; the budget allows "
                                    f"{budget.max_prompt_tokens}")
        chosen, omitted = [], len(groups)
        if with_view and spec.compact_memory:
            def memory_tokens(trial: List[_Group]) -> Optional[int]:
                n = count(render(last, trial, len(groups) - len(trial)).text)
                fits = n - base <= budget.max_memory_tokens and n <= budget.max_prompt_tokens
                return n - base if fits else None

            chosen, allocation = _evidence_first(groups, result, memory_tokens)
            omitted = len(groups) - len(chosen)
        elif with_view:
            for i, g in enumerate(groups):
                trial = groups[:i + 1]
                n = count(render(last, trial, len(groups) - len(trial)).text)
                if n - base > budget.max_memory_tokens or n > budget.max_prompt_tokens:
                    break
                chosen, omitted = trial, len(groups) - len(trial)
        if with_view:
            n = count(render(last, chosen, omitted).text)
            if n - base > budget.max_memory_tokens or n > budget.max_prompt_tokens:
                with_view, chosen = False, []       # not even the status lines fit
                memory_tags.append(MEMORY_INJECTION_SKIPPED)
            if omitted or not with_view:
                failure_tags.append(MEMORY_CONTEXT_OVER_BUDGET)
            omitted_ids = tuple(i for g in groups if g not in chosen for i in g.ids)
            if allocation is not None and allocation["blocked_queries"]:
                failure_tags.append(MEMORY_BUDGET_BLOCKED)
                blocked_keys = {k for q, keys in _answered_keys(result) if q in allocation["blocked_queries"]
                                for k in keys}
                blocked_ids = tuple(i for g in groups if _key(g) in blocked_keys and g not in chosen for i in g.ids)
        minimal = count(render(last, chosen, omitted).text)
        turns, dropped = messages, 0
        while True:
            prompt = render(turns, chosen, omitted)
            n = count(prompt.text)
            history = n - minimal
            if n <= budget.max_prompt_tokens and (budget.max_history_tokens is None
                                                  or history <= budget.max_history_tokens):
                break
            keep = turns[2:] if len(turns) > 2 else last    # whole turns only; the latest user turn alone fits
            turns, dropped = keep, dropped + len(turns) - len(keep)
        prompt = RenderedPrompt(prompt.text, prompt.format, spec, prompt.memory_ids, dropped)
        messages = turns
        system = _system_tokens(count, spec, situation, card, notes)
        tokens = TokenCounts(prompt=n, base=base, memory=minimal - base, history=history, user=base - system,
                             system=system)
    if with_view and any(g.slot.status is SlotStatus.CONFLICT for g in chosen):
        memory_tags.append(MEMORY_CONFLICT)
    if with_view and result.batch is not None and result.batch.status is BatchStatus.PARTIAL_SUCCESS:
        memory_tags.append(MEMORY_PARTIAL_ANSWER)
    memory_context = _memories(chosen) if with_view else ()
    view = _view(result, chosen, omitted) if with_view else None
    return BuiltContext(messages=messages, memory_context=memory_context, memory_view=view, prompt=prompt,
                        memory_tags=tuple(dict.fromkeys(memory_tags)), failure_tags=tuple(dict.fromkeys(failure_tags)),
                        rejected_ids=rejected, omitted_ids=omitted_ids, tokens=tokens,
                        budget=budget.describe() if budget is not None else {}, allocation=allocation,
                        blocked_ids=blocked_ids)


def _system_tokens(count: Callable[[str], int], spec: TemplateSpec, situation: Optional[str],
                   card: Optional[PersonaCard] = None, notes: Sequence[AnswerNote] = ()) -> int:
    if spec.situation_mode is SituationMode.NONE:
        return 0
    return count(system_text(spec, situation, card=card, notes=notes))
