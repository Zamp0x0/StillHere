"""Token counting with the real Qwen3-8B tokenizer (revision b968826d) and the qwen3 chat layout with thinking disabled.

Rendered training text (LLaMA-Factory ``template: qwen3``, ``enable_thinking: false``, one system + one user + one assistant turn):

    <|im_start|>system\\n{system}<|im_end|>\\n<|im_start|>user\\n{user}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n{target}<|im_end|>\\n

Supervised tokens (train_on_prompt false, mask_history true) = tokens of ``{target}<|im_end|>\\n`` (the qwen3 assistant slot). The empty think
block belongs to the masked prompt. The conversation history is inside the single user turn, so there are no earlier assistant turns to mask.
"""
from __future__ import annotations

import hashlib
from typing import List

PROMPT_FMT = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
SUP_SUFFIX = "<|im_end|>\n"
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
TOKENIZER_CONFIG_SHA256 = "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
IM_END = "<|im_end|>"


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Counter:
    """Real tokenizer only: refuses a tokenizer.json whose sha256 differs (no mock counts)."""

    def __init__(self, tokenizer_json: str):
        from tokenizers import Tokenizer
        got = sha256_file(tokenizer_json)
        if got != TOKENIZER_SHA256:
            raise RuntimeError(f"TOKENIZER_MISMATCH: {got}")
        self.sha256 = got
        self.tok = Tokenizer.from_file(tokenizer_json)
        self.im_end_id = self.tok.token_to_id(IM_END)

    def ids(self, s: str) -> List[int]:
        return self.tok.encode(s, add_special_tokens=False).ids

    def n(self, s: str) -> int:
        return len(self.ids(s))

    def row(self, system: str, user: str, target: str) -> dict:
        p = self.n(PROMPT_FMT.format(system=system, user=user))
        sup = self.n(target + SUP_SUFFIX)
        return {"prompt_tokens": p, "supervised_tokens": sup, "full_tokens": p + sup, "target_tokens": self.n(target)}
