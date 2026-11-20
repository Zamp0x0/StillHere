# Contributing to StillHere

Thank you for helping. This page covers the development setup, the tests and the rules that keep the project safe.

## Ground rules

- **Never commit real chat data, keys, adapters or databases.** Use the synthetic examples in `examples/synthetic/`, or invent new
  data. `scripts/privacy_guard.py` blocks the common cases, and CI runs it on every pull request.
- **Keep behaviour honest.** A feature that only works on some hardware or with a remote API must say so in its docs and in
  `docs/SUPPORTED_WORKFLOWS.md`. Do not add benchmark numbers without the protocol and the raw records behind them.
- **The core stays light.** The only required dependency is PyYAML; heavy libraries (torch, transformers, huggingface_hub) are
  optional extras imported lazily.

## Development setup

```bash
git clone https://github.com/Zamp0x0/StillHere.git StillHere && cd StillHere
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python scripts/privacy_guard.py --all
```

Optional: install the pre-commit hook so the privacy guard runs before each commit.

```bash
git config core.hooksPath .githooks
```

## Running the tests

```bash
python -m pytest -q                  # CPU only; no GPU, no network, no API key
```

- Tests use synthetic data and fake model backends.
- Tests that need an optional external checkout (for example the official LifeBench evaluator) are skipped when it is absent.

## Project structure

| Path | What it contains |
|---|---|
| `src/stillhere/persona/` | the user workflow: WeFlow import, profiles, persona cards, memory, runtime, CLI, Studio |
| `src/stillhere/retrieval/`, `src/stillhere/memory/`, `src/stillhere/context/` | retrieval (EF1, dense), the SQLite memory store, fact packs |
| `src/stillhere/providers/` | model providers (DeepSeek, OpenAI-compatible, local) |
| `training/` | QLoRA training code shipped inside training packages |
| `evaluation/` | LifeBench pilot and P2 evaluation code |
| `examples/synthetic/` | synthetic data for demos and tests |
| `tests/` | the CPU test suite |

## Pull requests

1. Open an issue first for larger changes, so the design can be discussed.
2. Keep pull requests focused, add or update tests, and update the docs for any user-visible change.
3. Run `python -m pytest -q` and `python scripts/privacy_guard.py --all` before you push.
4. Describe what you tested, and on what hardware if it matters.

## Good first issues

- Add another chat export format behind the same profile interface as the WeFlow parser.
- Improve English-language persona statistics (today's style statistics are tuned for Chinese chats).
- Expand `docs/troubleshooting.md` with errors you hit and how you fixed them.
