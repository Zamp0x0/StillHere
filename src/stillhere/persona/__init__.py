"""StillHere Persona pipeline (BUILD P1): authorised WeFlow chat export -> speaker identity -> deterministic cleaning -> dialogue
reconstruction -> persona SFT dataset with time / session isolated train, dev and (sealed) test splits, quality audit and manifests.

Standard library only (the exact Qwen3 token count is used when ``tokenizers`` and the pinned tokenizer are available). No model, no
network, no external API. ``service`` is the one entry point for the CLI and any later UI.
"""
PARSER_VERSION = "stillhere.persona.weflow/1.0"
IDENTITY_VERSION = "stillhere.persona.identity/1.0"
CLEANING_VERSION = "stillhere.persona.cleaning/1.0"
DIALOGUE_VERSION = "stillhere.persona.dialogue/1.0"
DATASET_VERSION = "stillhere.persona.dataset/1.0"
