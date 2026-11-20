"""Model providers for every StillHere role, each configured on its own (no single global ``model``).

Roles (``roles.ROLES``): ``answer`` (persona chat + Fact Reader + date summaries), ``card_assist`` (LLM-assisted persona card draft),
``memory_extraction`` (the optional Memory Writer), ``router`` (the optional LLM router), ``judge`` (evaluation only). Training base
model and embedding model are not chat providers and keep their own pinned configuration (``persona.final_config`` /
``retrieval.local_embedding``).

Provider kinds (``llm.PROVIDERS``):
    deepseek_api               DeepSeek's official OpenAI-compatible API (thinking off, the key from the environment only)
    openai_compatible_custom   any OpenAI-compatible endpoint the user names (base URL + model + key variable)
    huggingface_local          a local Hugging Face causal LM (``persona.models``: Qwen3-8B by default, or a user model directory)
    fake                       a labelled CPU stand-in for tests (never a silent fallback)

Standard library only. Keys are read from the environment / ``.env`` at call time and never written to a log, a manifest or a result.
"""
