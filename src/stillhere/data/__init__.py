"""Datasets and prompt templates: synthetic dataset building, versioning, hashing, split checks and
LLaMA-Factory export. Owns the chat template so training and inference render prompts identically.

May import: stillhere.contracts.

- ``templates``: Qwen3 non-thinking renderer (multi-bubble serialisation, situation conditioning, training
  and inference text, token-level checks against a real tokenizer).
- ``sft``: synthetic SFT dataset builder, conversation-level split, manifest with sha256 of written bytes.
- ``llamafactory``: ShareGPT export and ``dataset_info.json`` for LLaMA-Factory.
- ``build``: ``python -m stillhere.data.build [--check]`` rebuilds ``examples/synthetic/qiaomo_sft_v0_1``.

Not here: tokenisation for training, token-level loss-mask audit, training runs (``stillhere.training``).
"""
from .templates import (QWEN3_NATIVE_NOTHINK, QWEN3_SITUATION_NOTHINK, RenderedExample, SituationMode, TemplateError,
                        TemplateSpec, parse_bubbles, render_prompt, render_training_example, serialize_bubbles,
                        verify_with_tokenizer)

__all__ = [
    "QWEN3_NATIVE_NOTHINK", "QWEN3_SITUATION_NOTHINK", "RenderedExample", "SituationMode", "TemplateError",
    "TemplateSpec", "parse_bubbles", "render_prompt", "render_training_example", "serialize_bubbles",
    "verify_with_tokenizer",
]
