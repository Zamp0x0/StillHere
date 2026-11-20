"""Token counting: EXACT with the real Qwen3-8B tokenizer (revision b968826d, sha-pinned ``tokenizer.json``) or ESTIMATED as a conservative
fallback. The mode is always reported; an estimate is never labelled exact.

The chat layout is the one D4 trains on (LLaMA-Factory ``qwen3``, ``enable_thinking`` false), so a prompt's count here equals the training
count of the same rendered messages:

    <|im_start|>system\\n{system}<|im_end|>\\n<|im_start|>user\\n{user}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n

A D4-compatible request must fit ``prompt_tokens + max_new_tokens + len(<|im_end|>\\n) <= cutoff_len`` (1536).
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Mapping, Optional, Sequence

PROMPT_FMT = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
SUP_SUFFIX = "<|im_end|>\n"
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"          # = T2-D4 t2d4/tokens.py
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
TOKENIZER_ENV = "STILLHERE_QWEN3_TOKENIZER"          # path to the pinned Qwen3-8B tokenizer.json; no machine-specific default


def default_tokenizer() -> str:
    return os.environ.get(TOKENIZER_ENV, "")
D4_CUTOFF_LEN = 1536


class TokenizerUnavailable(RuntimeError):
    pass


class ExactTokenCounter:
    mode = "EXACT"

    def __init__(self, tokenizer_json: Optional[str] = None) -> None:
        tokenizer_json = tokenizer_json or default_tokenizer()
        if not tokenizer_json:
            raise TokenizerUnavailable(f"no tokenizer configured (set {TOKENIZER_ENV})")
        p = Path(tokenizer_json)
        if not p.exists():
            raise TokenizerUnavailable(f"no tokenizer file at {tokenizer_json}")
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != TOKENIZER_SHA256:
            raise TokenizerUnavailable(f"TOKENIZER_MISMATCH {got[:12]}")
        try:
            from tokenizers import Tokenizer
        except ImportError as e:                                 # the CPU environment has no tokenizers library
            raise TokenizerUnavailable("the 'tokenizers' library is not installed") from e
        self.sha256, self.tok = got, Tokenizer.from_file(str(p))

    def __call__(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False).ids)

    def chat(self, messages: Sequence[Mapping]) -> int:
        sysm = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        return self(PROMPT_FMT.format(system=sysm, user=user))

    def suffix(self) -> int:
        return self(SUP_SUFFIX)


class EstimatedTokenCounter:
    """chars x (1 + margin), rounded up: an upper bound for Chinese text; English and digits are over-counted (safe side)."""
    mode = "ESTIMATED"

    def __init__(self, margin: float = 0.05) -> None:
        self.margin = margin

    def __call__(self, text: str) -> int:
        return int(math.ceil(len(text) * (1 + self.margin)))

    def chat(self, messages: Sequence[Mapping]) -> int:
        sysm = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        return self(PROMPT_FMT.format(system=sysm, user=user))

    def suffix(self) -> int:
        return self(SUP_SUFFIX)


def best_counter(tokenizer_json: Optional[str] = None):
    """EXACT when the pinned tokenizer and the library are available, else ESTIMATED (the caller sees ``mode``)."""
    try:
        return ExactTokenCounter(tokenizer_json)
    except TokenizerUnavailable:
        return EstimatedTokenCounter()
