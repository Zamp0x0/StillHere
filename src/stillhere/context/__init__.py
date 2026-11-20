"""Context Management V1: the deterministic assembly of what a reader sees (``ContextOrchestrator``), its contract
(``ContextRequest`` / ``ContextResult``), full chat rounds (D6), token budgets, temporal boundaries and the checks that keep
other personas, later knowledge and retracted records out. Reads through ``MemoryAPI`` only; writes nothing."""
from .contracts import ContextRequest, ContextResult, DerivedItem, WorkingItem
from .orchestrator import VERSION, ContextOrchestrator

__all__ = ["ContextOrchestrator", "ContextRequest", "ContextResult", "DerivedItem", "WorkingItem", "VERSION"]
