"""Persona Card management on top of the versioned card store (``persona.card``): list, preview, activate, roll back, LLM-assisted draft.

Three sources, one schema (every version is a ``card.py`` version: validated, content-addressed, never overwritten):

    AUTO          ``card.build_auto``: Python statistics of the TRAIN split (length, bubbles, punctuation, emoji, log-odds expressions, ...)
    MANUAL        ``persona review --edit / --from-file`` (only the EDITABLE fields) or ``persona prompt --from-file`` (an authored prompt)
    LLM_ASSISTED  ``assist``: the ``card_assist`` provider (``stillhere.providers.roles``) drafts a prompt from the card statistics; only
                  with an API provider the user configured and the external-API opt-in; examples are sent only if the user allows them

The runtime always uses exactly ONE card: ``cards/CURRENT.json``. Activating a version never deletes another. Whatever the user writes,
the fact-safety rules are not in the card: the Fact Reader, its gates and the "no evidence -> say you do not know" rule are applied by
the runtime around any prompt (``persona.runtime``), and ``prompt_writer.validate_prompt`` rejects invented biography / numbers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from . import card as K
from . import store as S

SOURCES = ("AUTO", "MANUAL", "LLM_ASSISTED")


def source_of(man: dict, card: dict) -> str:
    author = str(card.get("prompt_author") or "")
    if author.startswith(("llm_assisted:", "openai_compatible:")):
        return "LLM_ASSISTED"
    if card.get("authored_prompt") or man.get("review_status") == K.REVIEWED:
        return "MANUAL"
    return "AUTO"


def changed_fields(pdir: Path, cid: str) -> List[str]:
    """User-provided fields that differ from the parent version (the rest are data-derived and recomputed from train)."""
    v = K.load(pdir, cid)
    parent = v["manifest"].get("parent_card")
    if not parent:
        return []
    p = K.load(pdir, parent)["card"]
    return sorted(k for k in set(K.EDITABLE) | {"authored_prompt", "prompt_author"} if v["card"].get(k) != p.get(k))


def list_versions(pdir: Path) -> List[dict]:
    pdir = Path(pdir)
    cur = (S.read_json(pdir / "cards" / "CURRENT.json") if (pdir / "cards" / "CURRENT.json").is_file() else {}).get("card_id")
    out = []
    for d in sorted((pdir / "cards").glob("c-*")):
        if not (d / "persona_card_manifest.json").is_file():
            continue
        try:
            v = K.load(pdir, d.name)
            ok = True
        except K.CardError:
            v, ok = {"manifest": S.read_json(d / "persona_card_manifest.json"), "card": {}}, False
        m = v["manifest"]
        out.append({"card_id": d.name, "current": d.name == cur, "source": source_of(m, v["card"]) if ok else "UNREADABLE", "valid": ok,
                    "review_status": m.get("review_status"), "parent_card": m.get("parent_card"), "dataset_version": m.get("dataset_version"),
                    "prompt_sha256": m.get("prompt_sha256"), "prompt_tokens": m.get("prompt_tokens"), "created_at": m.get("created_at"),
                    "author": v["card"].get("prompt_author") if ok else None, "user_fields_changed": changed_fields(pdir, d.name) if ok else None})
    return sorted(out, key=lambda r: r["created_at"] or "")


def preview(pdir: Path, cid: Optional[str] = None) -> dict:
    v = K.load(pdir, cid) if cid else K.current(pdir)
    if v is None:
        raise K.CardError("NO_CARD", "no persona card yet: run persona prepare")
    return {"card_id": v["card_id"], "source": source_of(v["manifest"], v["card"]), "prompt": v["prompt"], "prompt_sha256": v["manifest"]["prompt_sha256"]}


def activate(pdir: Path, cid: str) -> dict:
    """Make ``cid`` the one active card (a rollback is the same call with an older id). The card must load and verify."""
    pdir = Path(pdir)
    prev = K.current(pdir)
    v = K.load(pdir, cid)
    K.set_current(pdir, cid)
    S.append_lineage(pdir, {"artefact": f"cards/{cid}", "kind": "card_activated", "previous": prev["card_id"] if prev else None})
    stale = (pdir / "runtime" / "runtime_config.json").is_file() and \
        S.read_json(pdir / "runtime" / "runtime_config.json")["persona_prompt"]["sha256"] != v["manifest"]["prompt_sha256"]
    return {"card_id": cid, "previous": prev["card_id"] if prev else None, "source": source_of(v["manifest"], v["card"]),
            "next": "stillhere persona finalize --profile <id>  (runtime_config.json must select the new card)" if stale else None}


def assist(root, profile_id: str, *, allow_external: Optional[bool] = None, send_examples: bool = False, activate_new: bool = False,
           generator=None) -> dict:
    """LLM_ASSISTED draft: the ``card_assist`` provider writes a prompt from the card statistics; it is validated like a manual prompt
    (no invented biography / numbers) and saved as a NEW version (activated only on request). ``generator``: tests inject a fake."""
    from stillhere.providers import roles as PR
    from . import prompt_writer as PW
    root = Path(root) if root else S.default_root()
    pdir = S.profile_dir(root, profile_id)
    cur = K.current(pdir)
    if cur is None:
        raise K.CardError("NO_CARD", "no persona card yet: run persona prepare")
    cfg = PR.resolve("card_assist", profile_dir=pdir)
    gen = generator or PR.api_generator("card_assist", cfg, allow_external=allow_external, usage_log=pdir / "runtime" / "api_usage.jsonl")
    base = dict(cur["card"])
    base.pop("authored_prompt", None)
    base.pop("prompt_author", None)
    msgs = [{"role": "system", "content": PW.WRITER_INSTRUCTIONS},
            {"role": "user", "content": json.dumps(PW.style_payload(base, send_examples), ensure_ascii=False)}]
    r = gen(msgs, {"do_sample": cfg.temperature > 0, "temperature": cfg.temperature, "max_new_tokens": cfg.max_output_tokens}, None)
    res = PW.save_candidate(pdir, r["text"], author=f"llm_assisted:{cfg.provider}:{r.get('returned_model') or cfg.model}", activate=activate_new)
    return {**res, "source": "LLM_ASSISTED", "examples_sent": send_examples, "provider": cfg.provider, "model": cfg.model}
