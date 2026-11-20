"""The experimental reader path through Context Orchestrator V1, behind ``CONTEXT_ORCHESTRATOR_V1_ENABLED``.

Off (the default): nothing changes; ``legacy_context`` is ``answer_pilot.build_context`` on the frozen v3-round-1 items, byte for
byte. On: the same ranked candidates (V3.2 pool, Qwen rerank) go to the orchestrator as ``EvidenceHit``; full rounds replace the
three-turn ``round_text``, and the result comes back as an ``answer_pilot.Context`` so the reader prompt
(``deepseek_c1_answerability``, unchanged), the cache keys and the result rows keep their shapes.

The evidence room is the frozen one: the input budget is the reader's own scaffolding plus the question plus
``answer_pilot.MAX_CONTEXT_TOKENS`` (estimated the same conservative way), so the orchestrator never gets more room than the legacy
context had.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Mapping, Optional, Sequence

from stillhere.context import ContextOrchestrator, ContextRequest, ContextResult
from stillhere.context.budget import TokenCounter

from . import answer_pilot as ap

FLAG = "CONTEXT_ORCHESTRATOR_V1_ENABLED"
RENDERER = "context-v1: Context Orchestrator V1, full rounds with sender and recipient, temporal boundary line"


def enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    return (env if env is not None else os.environ).get(FLAG, "").strip().lower() in ("1", "true", "yes", "on")


def scaffolding_tokens(counter: TokenCounter, ask_time: str) -> int:
    """The reader's system prompt and the user message around the context and the question, counted like everything else."""
    shell = ap.build_messages("", ask_time, "")
    return counter(shell[0]["content"]) + counter(shell[1]["content"])


def orchestrated(api, *, persona_id: str, question: str, ask_time: str, when: datetime, hits: Sequence, keys: Sequence[str],
                 speakers: Mapping[str, str], counter: Optional[TokenCounter] = None) -> ContextResult:
    counter = counter or TokenCounter()
    fixed = scaffolding_tokens(counter, ask_time)
    req = ContextRequest(persona_id=persona_id, question=question, as_of=when, known_at=when, hits=tuple(hits), hit_keys=tuple(keys), top_k=ap.TOP_K,
                         instruction_tokens=fixed, input_token_budget=fixed + counter(question) + ap.MAX_CONTEXT_TOKENS, output_token_reserve=ap.MAX_NEW_TOKENS)
    return ContextOrchestrator(api, speakers=speakers, counter=counter).assemble(req)


def as_reader_context(r: ContextResult) -> ap.Context:
    """The orchestrator's result in the shape every reader script stores. Only evidence ids are numbered E01..; other sections keep their own ids."""
    if r.status == "SCOPE_ERROR":
        raise ValueError("the orchestrator refused a context that failed its integrity check; it is never sent to a reader")
    evidence = tuple(r.selected_evidence_ids)
    dropped = tuple(k for k, why in r.excluded_evidence_ids.items() if why.startswith("budget"))
    text = r.assembled_context
    return ap.Context(text, evidence, r.id_map, dropped, r.token_usage_estimate, r.truncated, ap.sha256_text(text))
