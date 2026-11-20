# Supported workflows

What each status means:

| Status | Meaning |
|---|---|
| ✅ **Verified** | run end to end by the project: in CI / CPU tests, on a real API with synthetic data, or on a real GPU |
| 🟡 **Implemented, not hardware-verified** | code and CPU dry-run tests exist, but the project has not run it on that hardware |
| 🛠️ **Manual** | StillHere prepares everything; you run a step yourself |
| ❌ **Not supported** | |

| # | Workflow | Status | How | Notes |
|---|---|---|---|---|
| 1 | Try the synthetic demo, no GPU / no key | ✅ Verified (CPU, CI) | [quickstart B](quickstart.md#b-build-alices-persona-cpu-only) | `--backend fake` is a labelled dry run, not a model |
| 2 | No GPU, chat through the DeepSeek API | ✅ Verified (real API, synthetic data) | [quickstart C](quickstart.md#c-chat-through-the-deepseek-api-no-gpu) | sends the persona prompt, evidence and message to DeepSeek, only with `--allow-external-api` |
| 3 | Another OpenAI-compatible API | 🟡 CPU tests only | [providers.md](providers.md) | `--answer-provider openai_compatible_custom --answer-base-url ...` |
| 4 | Import a WeFlow JSON export | ✅ Verified (CPU) | `persona inspect`, `persona quickstart` | WeFlow JSON only |
| 5 | Update with a newer export of the same chat | ✅ Verified (CPU) | `persona memory-build --source` | repeat imports add nothing |
| 6 | Create a second persona | ✅ Verified (CPU) | `persona quickstart --speaker self`, `persona status` | separate card, memory and config |
| 7 | Write or edit the persona description | ✅ Verified (CPU) | `persona review --edit / --from-file`, `persona finalize` | versions, rollback with `card activate` |
| 8 | API-assisted persona card draft | 🟡 implemented; the provider path is API-tested, `card assist` itself was not run in this release check | `persona card assist --allow-external-api` | style statistics only unless `--send-examples` |
| 9 | Dense retrieval on / off | ✅ Verified (CPU, real Qwen3-Embedding-0.6B) | `persona download-model embedding`, `memory-build --dense` | `ACTIVE_MEMORY_PIPELINE` shows `V2D_P2` or `V2L_P2_FALLBACK` |
| 10 | No model downloaded at all | ✅ Verified (CPU) | lexical memory + API, or `--backend fake` | dense needs the 1.2 GB embedding model |
| 11 | Local GPU chat with Qwen3-8B | ✅ Verified on Colab GPUs (L4 / A100) during the project's evaluations; 🟡 not verified on consumer local GPUs | [quickstart H](quickstart.md#h-chat-with-a-local-qwen3-8b-cuda-gpu) | `--backend auto` picks `hf-lowvram` below 20 GB VRAM |
| 12 | Already have a local Qwen3-8B | ✅ (as 11) | `--base-dir <dir>` or `STILLHERE_BASE_MODEL_DIR` | the chat template is checked; other models are not supported |
| 13 | Train an adapter on Google Colab | ✅ Verified (real L4 / A100 runs during the project) | [colab.md](colab.md) | resumable; you upload the package and download the adapter |
| 14 | Train on a local GPU | 🟡 Implemented, not hardware-verified | [training.md](training.md) | refused below 20 GiB VRAM |
| 15 | Train on your own remote GPU | 🛠️ Manual | `persona package --target remote-gpu` | package + instructions; no remote scheduler |
| 16 | Import an existing adapter | ✅ Verified | `persona import-adapter`, `persona finalize --adapter on` | base must match; not on API models |
| 17 | Local web UI | 🟡 Implemented (CPU tests); not part of this release's visual check | `persona studio` (127.0.0.1) | minimal tool |
| 18 | Other chat export formats (Telegram, WhatsApp, …) | ❌ Not supported | | roadmap |
| 19 | Arbitrary Hugging Face local models | ❌ Not supported | | Qwen3-8B / compatible template only |
| 20 | Automatic remote / multi-GPU training orchestration | ❌ Not supported | | |

Error recovery for API keys, paths, VRAM and missing indexes: [troubleshooting.md](troubleshooting.md).
