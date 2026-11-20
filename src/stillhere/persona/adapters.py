"""``persona import-adapter``: verify a persona package ZIP from Colab (or a local run) and install it under the private profile.

Checked before anything is installed: ZIP integrity and layout (no absolute / ``..`` paths, exactly the expected members), every member
against SHA256SUMS and the manifest, the persona id, the dataset version and card (both must exist locally, prompt sha256 identical),
the pinned base revision / tokenizer / chat template, the adapter config (LoRA, causal LM, Qwen3-8B), the safetensors header (LoRA A/B
tensors, no other weights), and - when PEFT is installed - that PEFT itself can read the adapter config. Installed to
``<profile>/adapters/<adapter_id>/``; an adapter is never overwritten (the same file again is reused).
"""
from __future__ import annotations

import hashlib
import json
import struct
import zipfile
from pathlib import Path, PurePosixPath
from typing import Dict, Optional

from . import card as K
from . import doctor as DR
from . import store as S

ROOT = "persona_adapter"
MEMBERS = {f"{ROOT}/adapter/adapter_model.safetensors", f"{ROOT}/adapter/adapter_config.json", f"{ROOT}/persona_prompt.txt", f"{ROOT}/persona_card.yaml",
           f"{ROOT}/persona_manifest.json", f"{ROOT}/evaluation_summary.json", f"{ROOT}/SHA256SUMS.txt"}
SCHEMA = "stillhere.persona.adapter_manifest/1.0"
MAX_ADAPTER_BYTES = 2 * 2**30


class AdapterError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def safetensors_header(b: bytes) -> dict:
    if len(b) < 8:
        raise AdapterError("ADAPTER_INVALID", "safetensors file too short")
    n = struct.unpack("<Q", b[:8])[0]
    if n <= 0 or n > min(len(b) - 8, 100 * 2**20):
        raise AdapterError("ADAPTER_INVALID", "bad safetensors header length")
    try:
        return json.loads(b[8:8 + n].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise AdapterError("ADAPTER_INVALID", "unreadable safetensors header") from None


def verify(zip_path, pdir: Path) -> dict:
    zp = Path(zip_path)
    if not zp.is_file():
        raise AdapterError("NOT_FOUND", f"{zp} not found")
    try:
        z = zipfile.ZipFile(zp)
    except zipfile.BadZipFile:
        raise AdapterError("ZIP_INVALID", "not a ZIP file") from None
    with z:
        names = {n for n in z.namelist() if not n.endswith("/")}
        for n in names:
            p = PurePosixPath(n)
            if p.is_absolute() or ".." in p.parts or "\\" in n:
                raise AdapterError("ZIP_UNSAFE", f"unsafe member path {n!r}")
        if names != MEMBERS:
            raise AdapterError("ZIP_LAYOUT", f"unexpected members: missing {sorted(MEMBERS - names)}, extra {sorted(names - MEMBERS)}")
        if z.testzip() is not None:
            raise AdapterError("ZIP_CRC", "a member is corrupt")
        if z.getinfo(f"{ROOT}/adapter/adapter_model.safetensors").file_size > MAX_ADAPTER_BYTES:
            raise AdapterError("ADAPTER_TOO_LARGE", "the adapter is larger than any LoRA of this pipeline (base weights are never shipped)")
        data: Dict[str, bytes] = {n[len(ROOT) + 1:]: z.read(n) for n in names}
    sums = {l.split(None, 1)[1].strip(): l.split(None, 1)[0] for l in data["SHA256SUMS.txt"].decode("utf-8").splitlines() if l.strip()}
    for n, b in data.items():
        if n != "SHA256SUMS.txt" and sums.get(n) != hashlib.sha256(b).hexdigest():
            raise AdapterError("SHA256_MISMATCH", f"{n} does not match SHA256SUMS.txt")
    man = json.loads(data["persona_manifest.json"])
    if man.get("schema") != SCHEMA:
        raise AdapterError("MANIFEST_INVALID", "not a persona adapter manifest")
    for n, h in man["files"].items():
        if hashlib.sha256(data[n]).hexdigest() != h:
            raise AdapterError("SHA256_MISMATCH", f"{n} does not match the manifest")
    prof = S.read_json(pdir / "profile.json")
    if man["profile_id"] != prof["profile_id"]:
        raise AdapterError("PERSONA_MISMATCH", f"the adapter belongs to persona {man['profile_id']}, not {prof['profile_id']}")
    if not (pdir / "datasets" / man["dataset_version"] / "DATASET_MANIFEST.json").is_file():
        raise AdapterError("DATASET_UNKNOWN", f"dataset version {man['dataset_version']} does not exist in this profile")
    try:
        local = K.load(pdir, man["card_id"])
    except (FileNotFoundError, K.CardError):
        raise AdapterError("CARD_UNKNOWN", f"card {man['card_id']} does not exist (or is corrupt) in this profile") from None
    if hashlib.sha256(data["persona_prompt.txt"]).hexdigest() != man["persona_prompt_sha256"] or local["manifest"]["prompt_sha256"] != man["persona_prompt_sha256"]:
        raise AdapterError("PROMPT_MISMATCH", "the adapter's persona prompt is not the local card's prompt")
    b = man["base"]
    if (b.get("revision"), b.get("tokenizer_sha256"), b.get("chat_template_sha256")) != (DR.BASE_REVISION, DR.TOKENIZER_SHA256, DR.CHAT_TEMPLATE_SHA256):
        raise AdapterError("BASE_MISMATCH", "the adapter was trained on another base revision / tokenizer / chat template")
    acfg = json.loads(data["adapter/adapter_config.json"])
    if acfg.get("peft_type") != "LORA" or acfg.get("task_type") not in ("CAUSAL_LM", None) or "Qwen3-8B" not in str(acfg.get("base_model_name_or_path", "Qwen3-8B")):
        raise AdapterError("ADAPTER_CONFIG", "adapter_config.json is not a LoRA for Qwen3-8B causal LM")
    head = safetensors_header(data["adapter/adapter_model.safetensors"])
    tensors = [k for k in head if k != "__metadata__"]
    if not tensors or not all(("lora_A" in k or "lora_B" in k) for k in tensors):
        raise AdapterError("ADAPTER_WEIGHTS", "the safetensors file holds tensors that are not LoRA A / B weights")
    peft_ok = None
    try:
        import tempfile
        from peft import PeftConfig
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "adapter_config.json").write_bytes(data["adapter/adapter_config.json"])
            pc = PeftConfig.from_pretrained(td)
            peft_ok = pc.peft_type.value == "LORA" if hasattr(pc.peft_type, "value") else str(pc.peft_type) == "LORA"
    except ImportError:
        peft_ok = None
    except Exception as e:  # noqa: BLE001
        raise AdapterError("PEFT_INCOMPATIBLE", f"PEFT cannot read adapter_config.json: {type(e).__name__}: {str(e)[:200]}") from None
    return {"manifest": man, "data": data, "tensors": len(tensors), "r": acfg.get("r"), "alpha": acfg.get("lora_alpha"), "peft_readable": peft_ok,
            "adapter_sha256": man["files"]["adapter/adapter_model.safetensors"]}


def adapter_id(man: dict) -> str:
    return f"a-{man['run_id']}-{man['checkpoint']}".lower().replace("_", "-")[:64]


def import_adapter(root, profile_id: str, zip_path) -> dict:
    root = Path(root) if root else S.default_root()
    S.check_private_root(root)
    pdir = S.profile_dir(root, profile_id)
    v = verify(zip_path, pdir)
    man = v["manifest"]
    aid = adapter_id(man)
    adir = pdir / "adapters" / aid
    if adir.exists():
        old = S.read_json(adir / "persona_manifest.json")
        if old["files"] != man["files"]:
            raise AdapterError("ADAPTER_EXISTS", f"{adir} holds another adapter; it is never overwritten")
        return {"adapter_id": aid, "adapter_dir": str(adir), "reused": True, "name": man["name"], "checkpoint": man["checkpoint"], "tensors": v["tensors"]}
    import os
    import shutil
    tmp = pdir / "adapters" / f".tmp-{aid}"
    shutil.rmtree(tmp, ignore_errors=True)
    for n, b in v["data"].items():
        S.atomic_write(tmp / n, b)
    S.write_json(tmp / "IMPORT_RECORD.json", {"imported_at": S.now_iso(), "zip_sha256": hashlib.sha256(Path(zip_path).read_bytes()).hexdigest(),
                                              "checks": {"zip": "PASS", "sha256": "PASS", "persona": "PASS", "dataset": "PASS", "card_prompt": "PASS",
                                                         "base": "PASS", "adapter_config": "PASS", "safetensors_lora_only": "PASS",
                                                         "peft_readable": {True: "PASS", None: "NOT_CHECKED (peft not installed)"}.get(v["peft_readable"], "FAIL")}})
    os.replace(tmp, adir)
    rdir = pdir / "runs" / man["run_id"]
    if not (rdir / "RUN_IMPORT.json").exists():
        S.write_json(rdir / "RUN_IMPORT.json", {"run_id": man["run_id"], "adapters": [aid], "training": man.get("training"), "run_fingerprint": man.get("run_fingerprint"),
                                                "imported_at": S.now_iso()})
    S.append_lineage(pdir, {"artefact": f"adapters/{aid}", "kind": "adapter", "run_id": man["run_id"], "checkpoint": man["checkpoint"],
                            "adapter_sha256": v["adapter_sha256"], "name": man["name"]})
    return {"adapter_id": aid, "adapter_dir": str(adir), "reused": False, "name": man["name"], "checkpoint": man["checkpoint"], "tensors": v["tensors"],
            "peft_readable": v["peft_readable"]}


def installed(pdir: Path) -> list:
    out = []
    for d in sorted((pdir / "adapters").glob("a-*")) if (pdir / "adapters").is_dir() else []:
        m = S.read_json(d / "persona_manifest.json")
        out.append({"adapter_id": d.name, "name": m["name"], "run_id": m["run_id"], "checkpoint": m["checkpoint"], "card_id": m["card_id"],
                    "adapter_sha256": m["files"]["adapter/adapter_model.safetensors"]})
    return out


def resolve(pdir: Path, aid: Optional[str]) -> Optional[dict]:
    if not aid or aid == "none":
        return None
    d = pdir / "adapters" / aid
    if not (d / "persona_manifest.json").is_file():
        raise AdapterError("ADAPTER_UNKNOWN", f"adapter {aid} is not installed (stillhere persona import-adapter)")
    m = S.read_json(d / "persona_manifest.json")
    if hashlib.sha256((d / "adapter" / "adapter_model.safetensors").read_bytes()).hexdigest() != m["files"]["adapter/adapter_model.safetensors"]:
        raise AdapterError("ADAPTER_CORRUPT", f"{d} does not match its manifest")
    return {"adapter_id": aid, "dir": str(d / "adapter"), "manifest": m}
