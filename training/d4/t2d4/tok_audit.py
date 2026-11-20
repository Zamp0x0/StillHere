"""Tokenizer, chat template, loss mask (static), EOS, padding and cutoff audit with the REAL Qwen3-8B tokenizer files (revision b968826d).

No model weights, no GPU, no LLaMA-Factory: the chat template is rendered with jinja2 exactly as Hugging Face does (sandboxed environment,
trim_blocks, lstrip_blocks) and compared with the training layout; the loss mask is reproduced statically the way LLaMA-Factory's
supervised preprocessing builds it (prompt and response encoded separately, prompt labels -100, response labels = response ids).
The LLaMA-Factory token-by-token comparison was done in T2-D3 for 0.9.5 and is repeated in D4.2's G0 on the training machine.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .tokens import IM_END, MODEL_REVISION, PROMPT_FMT, SUP_SUFFIX, TOKENIZER_CONFIG_SHA256, Counter, sha256_file

TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"  # T2-D3 recorded sha of tokenizer.chat_template
CUTOFF_CANDIDATES = (1024, 1536, 2048, 3072, 4096)


def _env():
    import jinja2
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    return env


def chat_template_check(model_dir: str, rows: Sequence[Mapping], n: int = 200) -> dict:
    cfg = json.loads(Path(model_dir, "tokenizer_config.json").read_text(encoding="utf-8"))
    tpl = cfg.get("chat_template") or ""
    sha = hashlib.sha256(tpl.encode("utf-8")).hexdigest()
    t = _env().from_string(tpl)
    bad_gen, bad_full = 0, 0
    for r in rows[:n]:
        msgs = [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}]
        gen = t.render(messages=msgs, add_generation_prompt=True, enable_thinking=False, bos_token="", eos_token=cfg.get("eos_token"))
        bad_gen += gen != PROMPT_FMT.format(system=r["system"], user=r["user"])
        full = t.render(messages=msgs + [{"role": "assistant", "content": r["target"]}], add_generation_prompt=False, enable_thinking=False,
                        bos_token="", eos_token=cfg.get("eos_token"))
        bad_full += full != PROMPT_FMT.format(system=r["system"], user=r["user"]) + r["target"] + SUP_SUFFIX
    return {"tokenizer_config_sha256": sha256_file(str(Path(model_dir, "tokenizer_config.json"))),
            "tokenizer_config_sha_ok": sha256_file(str(Path(model_dir, "tokenizer_config.json"))) == TOKENIZER_CONFIG_SHA256,
            "chat_template_sha256": sha, "chat_template_sha_ok": sha == TEMPLATE_SHA256, "rows_checked": min(n, len(rows)),
            "generation_prompt_mismatch": bad_gen, "full_render_mismatch": bad_full, "enable_thinking": False,
            "pass": sha == TEMPLATE_SHA256 and bad_gen == 0 and bad_full == 0,
            "layout": "prompt = " + PROMPT_FMT.replace("\n", "\\n") + " ; supervised = {target}" + SUP_SUFFIX.replace("\n", "\\n")}


def loss_mask_check(counter: Counter, rows: Sequence[Mapping]) -> dict:
    nl = counter.ids("\n")
    im_end = counter.im_end_id
    zero, eos_bad, special_in_target, boundary_diff, ledger_bad = 0, 0, 0, 0, 0
    specials = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>")
    for r in rows:
        p = counter.ids(PROMPT_FMT.format(system=r["system"], user=r["user"]))
        s = counter.ids(r["target"] + SUP_SUFFIX)
        labels = [-100] * len(p) + s
        sup = sum(1 for x in labels if x != -100)
        zero += sup <= len(nl) + 1  # nothing but <|im_end|>\n is supervised
        eos_bad += s[-1 - len(nl):] != [im_end] + nl
        special_in_target += any(x in r["target"] for x in specials)
        boundary_diff += counter.ids(PROMPT_FMT.format(system=r["system"], user=r["user"]) + r["target"] + SUP_SUFFIX) != p + s
        if r.get("supervised_tokens") is not None and r["supervised_tokens"] != len(s):
            ledger_bad += 1
    return {"rows": len(rows), "zero_loss_targets": zero, "eos_not_last_supervised": eos_bad, "special_token_in_target": special_in_target,
            "joint_vs_separate_tokenisation_differs": boundary_diff, "ledger_supervised_mismatch": ledger_bad,
            "im_end_id": im_end, "newline_ids": nl,
            "rule": "labels = -100 for every prompt token (system, user, empty think block); labels = ids(target + '<|im_end|>\\n'); the last "
                    "supervised tokens are <|im_end|> + newline; mask_history is moot (history is inside the single user turn)",
            "pass": zero == 0 and eos_bad == 0 and special_in_target == 0 and ledger_bad == 0}


def eos_padding_check(model_dir: str, counter: Counter) -> dict:
    cfg = json.loads(Path(model_dir, "tokenizer_config.json").read_text(encoding="utf-8"))
    gen = json.loads(Path(model_dir, "generation_config.json").read_text(encoding="utf-8")) if Path(model_dir, "generation_config.json").exists() else {}
    eos_ids = gen.get("eos_token_id")
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    pad = cfg.get("pad_token")
    pad_id = counter.tok.token_to_id(pad) if pad else None
    ok = counter.im_end_id in eos_ids and pad_id is not None and pad_id != counter.im_end_id
    return {"tokenizer_eos_token": cfg.get("eos_token"), "generation_eos_token_ids": eos_ids, "im_end_id": counter.im_end_id,
            "pad_token": pad, "pad_token_id": pad_id, "pad_differs_from_im_end": pad_id != counter.im_end_id,
            "padding_side_training": "right (LLaMA-Factory default for training)", "padding_side_generation": "left",
            "generation_config_sha256": sha256_file(str(Path(model_dir, "generation_config.json"))) if gen else None,
            "model_revision": MODEL_REVISION, "pass": ok}


def _pct(xs: List[int], q: float) -> int:
    if not xs:
        return 0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def cutoff_report(full_lens: Sequence[int], target_lens: Sequence[int], eval_prompt_lens: Sequence[int], max_new_tokens: int) -> dict:
    mx = max(full_lens) if full_lens else 0
    rec = next((c for c in CUTOFF_CANDIDATES if c >= mx), None)
    ev_need = max(eval_prompt_lens) + max_new_tokens if eval_prompt_lens else 0
    return {"train_full_tokens": {"p50": _pct(list(full_lens), .5), "p90": _pct(list(full_lens), .9), "p95": _pct(list(full_lens), .95),
                                  "p99": _pct(list(full_lens), .99), "max": mx},
            "target_tokens": {"p10": _pct(list(target_lens), .1), "p50": _pct(list(target_lens), .5), "p90": _pct(list(target_lens), .9),
                              "max": max(target_lens) if target_lens else 0},
            "rows_over_1024": sum(1 for x in full_lens if x > 1024), "recommended_cutoff_len": rec,
            "truncated_targets_at_recommended": 0 if rec else None,
            "eval_prompt_tokens": {"p50": _pct(list(eval_prompt_lens), .5), "p95": _pct(list(eval_prompt_lens), .95),
                                   "p99": _pct(list(eval_prompt_lens), .99), "max": max(eval_prompt_lens) if eval_prompt_lens else 0},
            "eval_context_needed_with_max_new_tokens": ev_need,
            "rule": "cutoff_len must be >= the longest training row, so no evidence or target is ever truncated; rows are never cut silently"}
