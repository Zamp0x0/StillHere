"""Inference (BUILD-1 subset): only the memory context builder and its prompt renderer.

BUILD-1 migrates ``context`` and ``prompting`` unchanged from the legacy project because the migrated
memory tests (``tests/test_memory_temporal.py``) check how temporal retrieval results are rendered into the
memory block. They depend only on ``stillhere.contracts`` and ``stillhere.data.templates``.

Not here yet: the chat session, grounding guard, backends (Hugging Face / fake), output parsing and traces.
They are migrated in a later BUILD together with their tests.

May import: stillhere.contracts, stillhere.data.
"""
