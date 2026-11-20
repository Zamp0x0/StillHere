# Security and privacy policy

StillHere processes private conversations, so privacy problems count as security problems.

## Reporting a vulnerability

Please **do not open a public issue** for:

- a security vulnerability;
- a way to leak private data, such as chats, persona cards, memories, keys or adapters;
- a way to send data to an external service without the user's explicit opt-in.

Use GitHub's **private vulnerability reporting** ("Security" tab → "Report a vulnerability") on this repository. Include the
affected version or commit, steps to reproduce with **synthetic** data, and the impact. We aim to acknowledge reports within 7 days.

## What StillHere does by design

- Profiles, chats, memories and adapters are stored only in your local profile root (default `~/.stillhere/personas`). The root
  must be outside any git work tree unless git ignores it.
- Nothing is sent to an external API unless you opt in with `--allow-external-api` or `STILLHERE_ALLOW_EXTERNAL_API=1`. The
  refusal message states exactly what would be sent and to whom.
- API keys are read only from environment variables or `.env`. They are never written to configs, logs or results, and keys that
  appear in errors are masked.
- `scripts/privacy_guard.py` blocks weights, databases, `.env` files and secret-like content from being committed.

## Supported versions

Only the latest release receives security fixes during the 0.x series.
