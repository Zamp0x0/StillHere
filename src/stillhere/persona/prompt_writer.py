"""Persona Prompt Generator: Train-only Persona Card -> Prompt Writer -> Prompt Validator -> a NEW card version (never overwrites).

Providers (none needs Claude Code, an IDE or an account):
* ``deterministic`` (default): the card's own renderer (``card.render_prompt``); no network, no key.
* ``openai_compatible`` (optional): any OpenAI-compatible chat endpoint (``STILLHERE_PROMPT_API_*``). It receives the card's STRUCTURED
  style statistics only; train example dialogues are added only with ``prompt_send_examples=true``. Used for prompts and nothing else.
  Any failure falls back to the deterministic prompt and says so.
* ``authored``: a prompt text written by a person or an assistant (e.g. a Claude Code session) from the same card, submitted as a file.

Every non-deterministic prompt goes through ``validate_prompt``: AI disclosure, no fabrication rule, no special tokens, no biography the
card does not hold (age, job, home, relationship ... unless the user confirmed it), no number that is not in the card, a sane length.
The result is a new card version (``authored_prompt`` + ``prompt_author``) that is NOT made current unless the caller asks.
"""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Mapping, Optional

from . import card as K
from . import store as S

BIOGRAPHY = re.compile(r"\d+岁|岁了|职业是|是一名|工作于|在\S{1,8}(公司|医院|学校|银行)工作|住在|老家|毕业于|大学生|研究生|结婚了|已婚|离婚|男朋友|女朋友|"
                       r"老公|老婆|星座|血型|生日是")
NO_FABRICATION = re.compile(r"不要编造|不编造|不能编造|别编造|不要瞎编|不确定|不知道")
WRITER_INSTRUCTIONS = (
    "你要为一个“数字人”写系统提示词（system prompt）。下面是从这个人真实聊天的训练集中用程序统计出来的说话风格（JSON）。"
    "只根据这些统计写，不要推测或编造这个人的年龄、职业、家庭、住址、经历和性格。要求：1）开头说明这是模仿此人说话方式的 AI，不是本人；"
    "2）把长度、分条、标点、表情、常用表达等写成清楚可执行的说话规则，尊重此人真实的短句习惯，但要根据上下文调整，优先回应对方最新一句；"
    "3）不要机械重复口头禅；4）事实只能来自提供的记忆，没有依据时自然地说不知道或不确定，不要编造经历；5）只输出提示词正文。")


class PromptError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def style_payload(card: Mapping, send_examples: bool) -> dict:
    keep = ("language", "reply_length", "style_traits", "punctuation", "emoji", "common_expressions", "frequent_short_replies", "response_patterns",
            "emotional_response", "conversation_behavior", "addressing_confirmed", "user_confirmed_facts", "extra_style_notes", "avoid", "display_name")
    out = {k: card.get(k) for k in keep}
    if send_examples:
        out["examples"] = card.get("examples")
    return out


def validate_prompt(text: str, card: Mapping) -> list:
    errs = []
    t = (text or "").strip()
    if not (200 <= len(t) <= 6000):
        errs.append("length must be 200-6000 characters")
    if "AI" not in t and "人工智能" not in t:
        errs.append("must state that this is an AI (not the real person)")
    if not NO_FABRICATION.search(t):
        errs.append("must tell the model not to fabricate / to say it does not know without evidence")
    from stillhere.style.prompt_builder import _RESERVED_TAGS, _SPECIAL_TOKEN_RE
    if _SPECIAL_TOKEN_RE.search(t) or any(x in t for x in _RESERVED_TAGS):
        errs.append("special tokens / reserved tags")
    confirmed = " ".join((card.get("user_confirmed_facts") or []) + [card.get("display_name") or ""])   # the persona's own display name is not biography
    for m in BIOGRAPHY.finditer(t):
        if m.group(0) not in confirmed:
            errs.append(f"biography not in the card: {m.group(0)}")
    blob = json.dumps(card, ensure_ascii=False)
    for n in set(re.findall(r"\d{2,}", t)):
        if n not in blob:
            errs.append(f"number {n} is not in the card statistics")
    return errs


def save_candidate(pdir: Path, text: str, *, author: str, activate: bool = False) -> dict:
    cur = K.current(pdir)
    if cur is None:
        raise PromptError("NO_CARD", "no persona card yet: run persona prepare")
    base = dict(cur["card"])
    base.pop("authored_prompt", None)
    errs = validate_prompt(text, base)
    if errs:
        raise PromptError("PROMPT_INVALID", "; ".join(errs))
    card = {**base, "authored_prompt": text.strip(), "prompt_author": author}
    res = K._write_version(pdir, card, parent=cur["card_id"])
    if activate:
        K.set_current(pdir, res["card_id"])
    S.append_lineage(pdir, {"artefact": f"cards/{res['card_id']}", "kind": "prompt_candidate", "author": author, "activated": activate})
    return {"card_id": res["card_id"], "prompt_sha256": res["prompt_sha256"], "author": author, "activated": activate, "card_dir": res["card_dir"]}


def generate(root, profile_id: str, *, cfg=None, activate: bool = False, opener=None) -> dict:
    """Build a prompt with the configured provider. ``opener`` (tests) replaces ``urllib.request.urlopen``."""
    from . import config as CF
    pdir = S.profile_dir(Path(root) if root else S.default_root(), profile_id)
    cfg = cfg or CF.load(profile_dir=pdir)
    cur = K.current(pdir)
    if cur is None:
        raise PromptError("NO_CARD", "no persona card yet: run persona prepare")
    if cfg.prompt_provider == "deterministic":
        return {"provider": "deterministic", "card_id": cur["card_id"], "prompt_sha256": cur["manifest"]["prompt_sha256"], "note": "the card's deterministic prompt"}
    if not (cfg.prompt_api_base_url and cfg.prompt_api_model):
        return {"provider": "deterministic", "fallback": "STILLHERE_PROMPT_API_BASE_URL / _MODEL not set", "card_id": cur["card_id"]}
    body = {"model": cfg.prompt_api_model, "temperature": 0.3,
            "messages": [{"role": "system", "content": WRITER_INSTRUCTIONS},
                         {"role": "user", "content": json.dumps(style_payload(cur["card"], cfg.prompt_send_examples), ensure_ascii=False)}]}
    req = urllib.request.Request(cfg.prompt_api_base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {cfg.prompt_api_key}"} if cfg.prompt_api_key else {})})
    try:
        with (opener or urllib.request.urlopen)(req, timeout=120) as r:
            text = json.loads(r.read())["choices"][0]["message"]["content"]
        res = save_candidate(pdir, text, author=f"openai_compatible:{cfg.prompt_api_model}", activate=activate)
        return {"provider": "openai_compatible", "examples_sent": cfg.prompt_send_examples, **res}
    except Exception as e:  # noqa: BLE001 - the deterministic prompt stays in use
        return {"provider": "deterministic", "fallback": f"prompt API failed or the prompt was invalid: {type(e).__name__}: {str(e)[:200]}", "card_id": cur["card_id"]}
