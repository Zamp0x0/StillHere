# Privacy

StillHere is built for **private** conversations. The defaults keep everything on your machine.

## What stays local

| Data | Where |
|---|---|
| your chat export | wherever you keep it; StillHere reads it and records its sha256 |
| profiles, persona cards, datasets, memory databases, dense indexes, adapters, logs | `<root>/<profile-id>/`, default root `~/.stillhere/personas` |
| API keys | environment variables or `.env` only; never written to configs, logs or results |

The profile root must be outside any git work tree, or ignored by git (`ROOT_NOT_PRIVATE` otherwise). This repository's
`.gitignore` ignores `profiles/`, `.env`, databases and weights. `scripts/privacy_guard.py` (also a pre-commit hook:
`git config core.hooksPath .githooks`) refuses to commit them.

## What can leave your machine, and only when you opt in

| You choose | What is sent | Guard |
|---|---|---|
| an API answer provider (DeepSeek, an OpenAI-compatible endpoint) | the persona prompt, the retrieved evidence records, your message | `--allow-external-api` / `STILLHERE_ALLOW_EXTERNAL_API=1`; otherwise refused with a description of what would be sent |
| `memory-build --writer deepseek` | whole chat sessions | the same opt-in |
| `card assist` | style statistics; example replies only with `--send-examples` | the same opt-in |
| a Colab / remote training package | the training data (`persona_data.zip`) that you upload yourself | `--confirm-cloud`, and `CONFIRM_PRIVATE_DATA_ON_CLOUD = True` in the notebook |
| `download-model` | nothing personal (downloads from Hugging Face) | — |

## Good practice

- Only use chats you have the right to use, and respect the other person in the conversation.
- Do not paste real chats, persona cards or keys into issues, pull requests or screenshots. Reproduce problems with
  `examples/synthetic/`.
- `persona memory-query` and `review --show-prompt` print private text to your terminal; avoid them on shared screens.
- To delete a persona, delete its folder `<root>/<profile-id>/`. Imported exports are not modified.
