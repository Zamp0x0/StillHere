"""Evidence identity, shared by the pipeline and the relation index."""
from __future__ import annotations


def evidence_key(persona_id: str, stored_id: str) -> str:
    """One key per evidence item. All messages of one LifeBench chat share a key; other ids stand for themselves."""
    parts = stored_id.split(".")
    if len(parts) >= 4 and parts[0] == "lb":
        return f"{persona_id}:{parts[2]}:{parts[3]}"
    return f"{persona_id}:record:{stored_id}"


def record_type_of(stored_id: str) -> str:
    parts = stored_id.split(".")
    return parts[2] if len(parts) >= 4 and parts[0] == "lb" else "record"
