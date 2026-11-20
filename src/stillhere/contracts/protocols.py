"""Service interfaces between modules. Implementations live in their owning module; callers depend only on these.

Implementations: ``memory.SQLiteMemoryStore`` (MemoryStore), ``memory.SlotRetriever`` (MemoryRetriever),
``inference`` backends (InferenceBackend). A persona file repository is not implemented yet.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Protocol, Sequence, Union, runtime_checkable

from .inference import InferenceRequest, InferenceResponse
from .memory import MemoryRecord, RetrievedMemory
from .persona import PersonaProfile
from .retrieval import RetrievalResult, TimeScope


@runtime_checkable
class PersonaRepository(Protocol):
    def get(self, persona_id: str, version: Optional[str] = None) -> PersonaProfile:
        """Return the requested version, or the latest when ``version`` is None. Raise KeyError if absent."""
        ...


@runtime_checkable
class MemoryStore(Protocol):
    """Long-term memory. Every write is transactional: it fully applies or leaves the store unchanged."""

    def add(self, record: MemoryRecord) -> MemoryRecord:
        """Insert a new record. Returns it with ``conflicts_with`` filled for any contradicting active record."""
        ...

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        ...

    def correct(self, memory_id: str, replacement: MemoryRecord, reason: str) -> MemoryRecord:
        """Atomically mark ``memory_id`` superseded and insert ``replacement`` with ``supersedes=memory_id``."""
        ...

    def retract(self, memory_id: str, reason: str) -> MemoryRecord:
        """Mark a record retracted without replacing it."""
        ...

    def history(self, persona_id: str, subject: str, attribute: str) -> List[MemoryRecord]:
        """All records for one slot, including superseded ones, ordered by ``recorded_at``."""
        ...

    def search(self, persona_id: str, query: str, *, as_of: Optional[datetime] = None,
               known_at: Optional[datetime] = None, limit: int = 5) -> List[RetrievedMemory]:
        """Records relevant to ``query``, valid at ``as_of`` and believed at ``known_at``, best first. ``known_at``
        defaults to ``as_of``; with neither, both are now."""
        ...


@runtime_checkable
class MemoryRetriever(Protocol):
    """Read-only retrieval with temporal resolution (see ``contracts.retrieval``). Never raises for a bad
    query or a failing store: the result's status says ``invalid_request`` or ``error``."""

    def retrieve(self, persona_id: str, text: str, *, context: Sequence[str] = (), context_roles: Sequence[str] = (),
                 as_of: Union[datetime, str, None] = None, subject: Optional[str] = None,
                 time_scope: Union[TimeScope, str, None] = None, limit: int = 5,
                 known_at: Union[datetime, str, None] = None) -> RetrievalResult:
        ...


@runtime_checkable
class InferenceBackend(Protocol):
    def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Produce one reply. Must return a response that passes ``response.check_against(request)``."""
        ...
