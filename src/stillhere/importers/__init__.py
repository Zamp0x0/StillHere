"""Importers turn datasets into contracts and write them through ``memory.MemoryAPI``.

May import: stillhere.contracts, stillhere.memory.

- ``base``: stable ids, time normalisation, ``ImportReport``.
- ``generic_chat``: ``GenericConversationImporter``, raw chat history from standardised JSON or JSONL.
- ``lifebench``: LifeBench 2.0 Chinese data: metadata scan, frozen selection, ``LifeBenchImporter``.
"""
from .base import ImportReport, stable_id
from .generic_chat import GenericConversationImporter
from .lifebench import LifeBenchImporter, ensure_selection, parse_persona

__all__ = ["GenericConversationImporter", "ImportReport", "LifeBenchImporter", "ensure_selection", "parse_persona",
           "stable_id"]
