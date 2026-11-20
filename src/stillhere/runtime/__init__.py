"""The Batch 2 runtime chain: memory (B2 retrieval + M3.1 reflection) -> grounded content -> style prompt contract -> generator -> safety
guards -> reply. A thin orchestrator over the existing Core modules; it adds no storage, no model and no second memory."""
from .pipeline import RUNTIME_PIPELINE_VERSION, ReplyTrace, grounded_from_evidence, reply

__all__ = ["RUNTIME_PIPELINE_VERSION", "ReplyTrace", "grounded_from_evidence", "reply"]
