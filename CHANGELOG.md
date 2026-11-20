# Changelog

All notable changes are listed here. The project follows [Semantic Versioning](https://semver.org/).

## [0.1.0] - unreleased (public release candidate)

First public release of StillHere.

### Added

- **Persona pipeline:** WeFlow JSON import, speaker identity checks, cleaning, versioned persona cards and prompts
  (automatic, manual edit, API-assisted draft), activation and rollback.
- **Evidence memory:** SQLite archive with FTS5 / BM25, incremental and de-duplicated re-import, optional dense index
  (Qwen3-Embedding-0.6B), hybrid fusion.
- **Memory P2:** time-aware evidence requirements and one bounded second retrieval pass, with no LLM call. The product default is
  `V2D + P2`, falling back to `V2L + P2` when the dense index is unavailable.
- **Fact Reader p1:** attribution and grounding gates, answer preservation, writer in EVIDENCE_ONLY mode.
- **Model providers:** local Qwen3-8B (CUDA, NF4), the DeepSeek API, OpenAI-compatible endpoints, with explicit opt-in for
  external APIs.
- **Optional QLoRA training:** Colab training packages with resumable notebooks, local CUDA training (>= 20 GiB VRAM), adapter
  import with base checks.
- **CLI:**
  - `stillhere persona quickstart / status / chat / memory-build / memory-query / card / review / provider / download-model / package / train / import-adapter / doctor / studio`;
  - `ACTIVE_MEMORY_PIPELINE` reporting.
- **Synthetic Alice / Bob WeFlow demo** with a later export for incremental import.
- **Evaluation code:** the LifeBench 2.0 pilot (vs. Mem0 OSS) and the P2 memory evaluation, with aggregate results in
  `docs/benchmark.md`.
