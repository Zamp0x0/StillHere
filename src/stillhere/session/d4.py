"""Runtime -> D4 adapter input record: ``stillhere.response.grounded_persona/1.0`` — **PROPOSED**, not frozen.

The T2-D4 thread has since published its D4.1 interface (T2-D4 workspace, ``docs/D4_1_CONTEXT_THREAD_INTERFACE_HANDOFF.md``; renderer
``t2d4/render.py`` ``render_messages``, same schema name). That renderer is the authority for the D4 prompt; this record is the runtime side
of the mapping and is still PROPOSED until both threads align the gaps in ``PENDING_ALIGNMENT``. The record splits:

* MODEL_VISIBLE fields: what the renderer turns into the prompt (the same for Base and Adapter, for training and for inference);
* RUNTIME fields: ids, hashes and decisions for traces; never rendered into the prompt.

``record_from_reply`` builds the record from a ``SessionReply`` (runtime side). For a D4 backend the prompt must come from the D4.1 renderer
(``render_messages``) fed with these fields after the mapping in ``PENDING_ALIGNMENT``; for Base / the Batch 1 LoRA the runtime keeps the frozen
``stillhere.style.prompt/2.0`` builder.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

SCHEMA_VERSION = "stillhere.response.grounded_persona/1.0"
STATUS = "PROPOSED"
COMPATIBILITY_STATUS = ("PARTIAL: current message / recent turns / profile / k=5 examples map one to one to D4.1 inputs; memory evidence and "
                        "disclosure need the mapping below; the D4 prompt must be rendered by t2d4.render.render_messages, not by prompt/2.0")
MODEL_VISIBLE = ("current_user_message", "recent_conversation", "retrieved_session_context", "memory_evidence", "style_profile", "style_examples",
                 "temporal_context", "disclosure_constraints")
RUNTIME_ONLY = ("schema_version", "status", "persona_id", "session_id", "generation_config", "trace_metadata")
D4_REQUIRED_FIELDS = ("current_user_message", "recent_conversation", "memory_evidence", "style_profile", "style_examples")
RUNTIME_PROVIDED_FIELDS = MODEL_VISIBLE + ("generation_config", "trace_metadata")
PENDING_ALIGNMENT = ("memory_evidence -> D4.1 evidence entries: map grounding roles current/past/corrected/unverified + unknown to the D4 status view "
                     "CURRENT/HISTORICAL/CORRECTED/CONFLICTED/UNKNOWN; CANCELLED and UPCOMING need event status / valid_from at `now`",
                     "disclosure_policy (ALLOWED / PROTECTED / POLICY_UNCERTAIN): not in Core; the runtime must attach it before a D4 backend",
                     "recent turns: D4 trained on 1-4 turns and <= 1536 tokens -> use session.config.D4_COMPATIBLE for a D4 backend",
                     "retrieved_session_context: D4's 【对话】 block has no earlier-session excerpts -> L2 off for a D4 backend until trained",
"temporal_context rendering (today: dates live inside memory statements; no separate line in prompt/2.0)",
                     "disclosure_constraints content (today: must-not from corrected values + constraints lines)",
                     "retrieved_session_context marker wording (today: 〔较早的对话，第N条〕 prefix inside the conversation)",
                     "recent_conversation maximum (runtime: 8 turns / 2048 tokens) must equal the training maximum")


def record_from_reply(persona_id: str, session_id: str, reply, *, style_profile: Optional[Mapping] = None,
                      style_examples: Sequence[str] = (), temporal_context: Optional[Mapping] = None,
                      disclosure_constraints: Sequence[str] = ()) -> Dict:
    conv = [{"role": "assistant" if c["speaker"] == "self" else "user", "text": c["text"], "generated": bool(c.get("generated")),
             "layer": c.get("layer")} for c in _conversation_of(reply)]
    current = conv[-1]["text"] if conv else reply.user_message.content
    return {
        "schema_version": SCHEMA_VERSION, "status": STATUS, "persona_id": persona_id, "session_id": session_id,
        "current_user_message": current,
        "recent_conversation": [c for c in conv[:-1] if c["layer"] == "L1"],
        "retrieved_session_context": [c for c in conv[:-1] if c["layer"] == "L2"],
        "memory_evidence": {k: reply.trace["grounded"].get(k) for k in ("facts", "unknown", "answer_status")},
        "style_profile": style_profile, "style_examples": list(style_examples),
        "temporal_context": temporal_context or {}, "disclosure_constraints": list(disclosure_constraints),
        "generation_config": {"output_reserve": reply.trace.get("output_reserve")},
        "trace_metadata": {"context_version": reply.trace.get("context_version"), "prompt_hash": reply.trace.get("contract", {}).get("prompt_hash"),
                           "token_mode": reply.trace.get("token_mode")},
    }


def _conversation_of(reply) -> List[dict]:
    return getattr(reply, "conversation", None) or reply.trace.get("conversation", [])


def model_visible(record: Mapping) -> Dict:
    return {k: record[k] for k in MODEL_VISIBLE}
