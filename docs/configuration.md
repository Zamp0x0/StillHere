# Configuration

## Where settings come from

| Layer | Example | Scope |
|---|---|---|
| defaults | built in | everything |
| `.env` | `.env` in the current directory (copy `.env.example`) | your machine |
| environment variables | `export STILLHERE_ANSWER_PROVIDER=deepseek_api` | your shell |
| profile settings | `<root>/<profile-id>/runtime/settings.json` (`providers` block) | one profile |
| runtime config | `<root>/<profile-id>/runtime/runtime_config.json` (written by `quickstart` / `finalize`) | one profile: card, adapter, memory pipeline |
| command-line flags | `--answer-provider`, `--backend`, `--root`, ... | one command |

Later layers win. `stillhere persona config [--profile <id>]` and `stillhere persona provider show` print the effective values with
keys masked.

## Important variables

| Variable | Meaning |
|---|---|
| `STILLHERE_PERSONA_ROOT` | the profile root (default `~/.stillhere/personas`; must be outside git or git-ignored) |
| `DEEPSEEK_API_KEY` | DeepSeek key (environment / `.env` only) |
| `STILLHERE_ALLOW_EXTERNAL_API` | `1` = allow API providers to receive private content (same as `--allow-external-api`) |
| `STILLHERE_ANSWER_PROVIDER`, `STILLHERE_ANSWER_MODEL`, `STILLHERE_ANSWER_BASE_URL` | the chat model (see [providers.md](providers.md)) |
| `STILLHERE_BASE_MODEL_DIR` | local Qwen3-8B directory |
| `STILLHERE_LOCAL_BACKEND` | `auto` (default) / `hf` / `hf-lowvram` / `fake` |
| `STILLHERE_DEVICE` | `cuda:0` (default), `cuda:1`, ... |
| `STILLHERE_EMBEDDING_MODEL_DIR` | local Qwen3-Embedding-0.6B directory (otherwise the Hugging Face cache is searched) |
| `HF_HOME` | Hugging Face cache location |
| `STILLHERE_ROUTER_MODE` | `rules` (default) / `llm` |

The full list, all empty, is in [`.env.example`](../.env.example).

## The memory pipeline section

`runtime_config.json` → `memory_pipeline`:

```json
{"pipeline": "v2", "dense": true, "dense_fallback": "v2l", "k": 8, "neighbors": [2, 2], "date_mode": "prefer",
 "reader_version": "p1", "writer_mode": "EVIDENCE_ONLY", "p2": true, "version": "stillhere.persona.product/2026-11-19-v2d-p2"}
```

Choose it with `stillhere persona finalize --profile <id> --memory-pipeline product|product-v2d|research-final|v1`. A re-finalize
without the flag keeps the profile's current choice.

## Safe defaults

- adapter **off**;
- writer **EVIDENCE_ONLY**;
- **no external API** without opt-in;
- Studio listens on **127.0.0.1** only;
- cloud training packages need `--confirm-cloud`, plus `CONFIRM_PRIVATE_DATA_ON_CLOUD = True` in the notebook.
