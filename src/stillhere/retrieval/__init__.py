"""Hybrid retrieval over the memory store. May import: stillhere.contracts, stillhere.memory.

- ``alibaba_client``: DashScope embedding and rerank calls, the cost ledger and the budget. Standard library only.
- ``dense``: document text, scope masks, the embedding cache and the exact NumPy index (needs the ``hybrid`` extra).
- ``hybrid``: ``RetrievalConfig`` and ``HybridRetriever``, one pipeline for every profile.
- ``profiles``: the configurations chosen on development questions.

SQLite stays the authority: everything here is derived and can be rebuilt.
"""
