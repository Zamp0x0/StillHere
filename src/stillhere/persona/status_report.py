"""``stillhere persona status``: what exists for each local profile, read-only (no model is loaded, nothing is written).

One profile: the active persona card, the imported messages, the memory archive, the dense index, the configured and the EXPECTED active
memory pipeline (``ACTIVE_MEMORY_PIPELINE``; ``persona chat`` prints the one that actually runs), installed adapters, the runtime config
and every setup stage that did not pass, with its reason. Without a profile id: one line per profile under the root.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import store as S

# setup stages that only matter for OPTIONAL adapter training / evaluation (chat works without them)
TRAINING_ONLY = {"05": "dataset gates are for adapter training only; chat, the persona card and memory do not need them",
                 "12": "training package (optional)", "13": "adapter (optional)", "14": "GPU evaluation (optional)", "15": "prompt refinement (optional)",
                 "09": "semantic memory needs a writer model (optional); the evidence archive is enough for chat"}


def _root(root) -> Path:
    return Path(root) if root else S.default_root()


def profiles(root=None) -> list:
    r = _root(root)
    out = []
    for p in sorted(r.glob("*/profile.json")) if r.is_dir() else []:
        d = S.read_json(p)
        out.append({"profile_id": d["profile_id"], "display_name": (d.get("target") or {}).get("display_name"), "role": (d.get("target") or {}).get("role"),
                    "created_at": d.get("created_at"), "dir": str(p.parent)})
    return out


def profile(root, profile_id: str) -> dict:
    from . import adapters as AD
    from . import card as K
    from . import card_admin as CA
    from . import dense_memory as DM
    from . import memory_store as MS
    from . import product_config as PC
    r = _root(root)
    pdir = S.profile_dir(r, profile_id)
    if not (pdir / "profile.json").is_file():
        raise S.StoreError("NO_PROFILE", f"no profile {profile_id} under {r} (list them with: stillhere persona status)")
    out = {"profile_id": profile_id, "profile_dir": str(pdir)}
    try:
        cur = K.current(pdir)
        out["card"] = {"card_id": cur["card_id"], "review_status": cur["manifest"].get("review_status"), "versions": len(CA.list_versions(pdir))}
    except Exception as e:  # noqa: BLE001 - a status view never fails on one missing part
        out["card"] = {"card_id": None, "error": f"{type(e).__name__}: {str(e)[:120]}"}
    ms = MS.status(r, profile_id)
    out["memory"] = {k: ms.get(k) for k in ("memory_status", "archive_messages", "searchable_text_messages", "sessions", "first", "last", "semantic_extraction", "db")}
    ds = DM.status(r, profile_id)
    model = DM.resolve_model_dir()
    out["dense_index"] = {**{k: ds.get(k) for k in ("dense_status", "chunks", "dir")}, "embedding_model": str(model) if model else None}
    cfgp = pdir / "runtime" / "runtime_config.json"
    cfg = S.read_json(cfgp) if cfgp.is_file() else None
    mp = (cfg or {}).get("memory_pipeline")
    enabled = bool(cfg and cfg["memory"]["enabled"])
    dense_expected = bool(mp and mp.get("dense") and ds.get("dense_status") == "READY" and model is not None)
    out["memory_pipeline"] = {"configured": (mp or {}).get("version", "V1 (no section)") if cfg else None, "describe": PC.describe(mp) if cfg else None,
                              "ACTIVE_MEMORY_PIPELINE_expected": (PC.active_pipeline(mp, dense_expected) if enabled else "OFF") if cfg else "NOT_FINALIZED",
                              "note": "expected from the files on disk; persona chat prints the pipeline that actually runs"}
    out["adapter"] = {"enabled": bool(cfg and cfg["adapter"]["enabled"]), "active": (cfg or {}).get("adapter", {}).get("adapter_id") if cfg and cfg["adapter"]["enabled"] else None,
                      "installed": [a["adapter_id"] for a in AD.installed(pdir)]}
    out["runtime_config"] = str(cfgp) if cfgp.is_file() else None
    st = S.read_json(pdir / "PIPELINE_STATE.json") if (pdir / "PIPELINE_STATE.json").is_file() else {"stages": {}}
    notes = []
    for code, v in sorted(st.get("stages", {}).items()):
        if v.get("status") in ("PASS", None) or code == "17":
            continue
        why = v.get("reason") or v.get("error") or (f"failed gates: {', '.join(v['failed_gates'])}" if v.get("failed_gates") else "")
        notes.append({"stage": f"{code} {v.get('stage')}", "status": v["status"], "why": why, "meaning": TRAINING_ONLY.get(code, "")})
    out["setup_overall"] = (st.get("stages", {}).get("17") or {}).get("status")
    out["setup_notes"] = notes
    out["next"] = ([f"stillhere persona chat --profile {profile_id} --answer-provider deepseek_api --answer-model deepseek-flash --allow-external-api",
                    f"stillhere persona chat --profile {profile_id} --base-dir <Qwen3-8B dir>   (local CUDA GPU)",
                    f"stillhere persona chat --profile {profile_id} --backend fake --message \"...\"   (labelled dry run, no model)"]
                   if cfg else [f"stillhere persona finalize --profile {profile_id}"])
    return out
