"""StillHere Simple Context v1: the ONE place its parameters live (every module reads them from here)."""
from __future__ import annotations

from dataclasses import dataclass, field

CONTEXT_VERSION = "stillhere.session.context/1.0"


@dataclass(frozen=True)
class ContextConfig:
    recent_turn_limit: int = 8                      # whole turns (a user message and the replies to it); not a quota to fill
    recent_context_token_budget: int = 2048         # the recent window only (L1), not the model input
    history_search_top_k: int = 2                   # L2: earlier excerpts of the same session, on demand
    style_examples_k: int = 5
    total_input_token_budget: int = 6144            # everything sent to the model (system + user), measured on the rendered messages
    default_max_new_tokens: int = 160               # output reserve when the length plan gives none
    long_message_tokens: int = 600                  # a message above this is chunked (stored whole, shown by excerpt)
    chunk_tokens: int = 300
    long_message_excerpt_chunks: int = 3            # the matched chunk(s) and their neighbours shown in the window
    followup_max_chars: int = 12                    # a short question with no named subject is treated as elliptical
    query_context_user_turns: int = 2               # the earlier USER messages a follow-up query may borrow words from
    session_history_summarization: bool = False     # DISABLED (no summary model)
    vector_search: bool = False                     # DISABLED (no embeddings / vector store)
    additional_llm_calls: bool = False              # DISABLED
    estimate_margin: float = 0.05                   # ESTIMATED token count = chars x (1 + margin); never reported as exact
    flags: dict = field(default_factory=dict)


DEFAULT = ContextConfig()

# For a D4 adapter backend (T2-D4.1 interface handoff, read-only: docs/D4_1_CONTEXT_THREAD_INTERFACE_HANDOFF.md in the T2-D4 workspace):
# D4 was trained on 1-4 turns of the other side and inputs <= 1,169 tokens with cutoff_len 1536, and its 【对话】 block has no earlier-session
# excerpts. Keep the runtime inside that distribution when the backend is the D4 adapter.
D4_COMPATIBLE = ContextConfig(recent_turn_limit=4, total_input_token_budget=1536, history_search_top_k=0)
