# Troubleshooting

Start with `stillhere persona status --profile <id>`. It shows the card, the memory, the dense index, the expected
`ACTIVE_MEMORY_PIPELINE`, the adapter and every setup stage that did not pass, with the reason.

## Installation

| Problem | Fix |
|---|---|
| `stillhere: command not found` | activate the virtual environment (`source .venv/bin/activate`, or `.venv\Scripts\Activate.ps1` on Windows), or use `python -m stillhere.cli` |
| PowerShell refuses to run `Activate.ps1` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use `cmd` with `.venv\Scripts\activate.bat` |
| Python version error | Python 3.10+ is required (`python --version`) |
| `ImportError` / `ModuleNotFoundError` | reinstall from the repository root: `pip install -e .` (add the extra you need: `.[embedding]`, `.[gpu]`, `.[download]`) |

## Import

| Problem | Fix |
|---|---|
| `NOT_FOUND: input file not found` | check the path (quote paths with spaces) |
| JSON / format error | the file must be a WeFlow JSON export; `stillhere persona inspect --input <file>` shows what was parsed |
| `IDENTITY_ANOMALY` | the export has speaker attributions the parser cannot trust; read the anomalies and rerun with `--accept-anomalies` only if they are harmless |
| wrong person selected | quickstart again with `--speaker 1/2/contact/self`; a different speaker makes a different profile |
| no profile id | `stillhere persona status` lists every profile |
| `PROFILE_EXISTS` | the profile id is bound to another file or person; it is never overwritten: use another `--profile` |
| duplicate import | importing the same file again adds nothing (`added: 0`); a newer export: `memory-build --profile <id> --source <file>` |
| a second profile appeared for the same person | you ran quickstart on a newer export; keep the first and update it with `memory-build --source` (you can delete the extra profile's folder) |
| stage 05 `FAILED` (`train_samples_ge_200`) | not enough replies to train an adapter; chat is not affected |

## Chat and providers

| Problem | Fix |
|---|---|
| `NO_GPU` | no CUDA GPU: chat through the API (`--answer-provider deepseek_api --allow-external-api`) or dry-run with `--backend fake` |
| `EXTERNAL_API_NOT_ALLOWED` | add `--allow-external-api` (or `STILLHERE_ALLOW_EXTERNAL_API=1`) if you accept sending the data |
| `NO_API_KEY` | set `DEEPSEEK_API_KEY` (or the variable your provider uses) in the shell or `.env` |
| `HTTP 401` | wrong or revoked key |
| `HTTP 429` / timeouts | rate limit or no credit; retries are bounded, so wait and try again |
| `CONFIG_STALE` | the active card changed: `stillhere persona finalize --profile <id>` |
| `ADAPTER_NEEDS_LOCAL_BASE` | adapters only run on a local base model, not on an API model |
| adapter refused at import | it was trained on another base or is not a LoRA package; rebuild it from this profile's package |
| CUDA out of memory | close other GPU programs; `--backend hf-lowvram`; or use the API |
| model / tokenizer / chat template not found | `stillhere persona download-model base`, then `--base-dir <printed-path>`; other models are not supported |

## Memory

| Problem | Fix |
|---|---|
| `note: DENSE_UNAVAILABLE ... -> V2L` | no embedding model or no index: `pip install -e ".[embedding,download]"`, `persona download-model embedding`, `persona memory-build --profile <id> --dense` |
| `dense_status: STALE` | new messages were imported after the index was built: run `memory-build --dense` again |
| answers miss something you know is in the chat | `persona memory-query --profile <id> --q "<question>"` shows what retrieval finds; try wording closer to the chat |
| `database is locked` (SQLite) | another StillHere process (chat, Studio, a build) is using the profile; close it and retry |
| `ROOT_NOT_PRIVATE` | the profile root is inside a git repository: use the default root or a git-ignored folder |

## Colab

| Problem | Fix |
|---|---|
| disconnected | reconnect and *Run all* with the same `RUN_ID`; finished stages are reused |
| sha256 / revision mismatch | re-upload the four files of one package; do not mix packages or edit them |
| `CONFIRM_PRIVATE_DATA_ON_CLOUD` | set it to `True` in STAGE 00 only if you accept the upload |

## Paths

| Platform | Notes |
|---|---|
| Windows | quote paths with spaces; Chinese characters in paths work (UTF-8). If output shows garbled Chinese, use Windows Terminal or set `PYTHONIOENCODING=utf-8` |
| macOS / Linux | the default root is `~/.stillhere/personas`; `export STILLHERE_PERSONA_ROOT=...` to move it |
