"""Credential-shaped strings (API tokens, private-key blocks). One pattern shared by the dataset builder (a real chat episode that carries one
is never used for training) and the release / secret scans. A token prefix must start a token, so task-... or my_hf_... inside a longer
identifier is not a credential."""
from __future__ import annotations

import re

CREDENTIAL = re.compile(r"(?<![A-Za-z0-9_-])(hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16})"
                        r"|-----BEGIN [A-Z ]*PRIVATE KEY-----")


def has_credential(*texts: str) -> bool:
    return any(t and CREDENTIAL.search(t) for t in texts)
