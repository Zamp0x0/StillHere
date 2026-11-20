"""LifeBench 2.0 one-persona 100-QA pilot: StillHere V2L / V2D vs official Mem0 OSS vs no memory, one Answer model, judge x 3.

Modules: ``protocol`` (frozen constants), ``select`` (person + 100 QA), ``timeline`` (canonical events), ``context`` (one evidence format and
token budget for every group), ``stillhere_memory`` / ``mem0_memory`` (the memory systems), ``llm_proxy`` (Mem0's LLM traffic, counted),
``answer`` / ``judge`` (DeepSeek calls, resumable), ``metrics`` (micro / macro / paired bootstrap), ``runner`` (the Colab stages), ``export``.
Status labels: PASS / INCOMPLETE / FAIL / SKIP. Nothing here reads the persona project's private data.
"""
