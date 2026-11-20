"""Persona pipeline service: the one API the CLI (and a later UI) calls. No printing, no prompting, no private text in return values
except the display names a caller explicitly asks for (``speakers``).

    inspect(path)                                   read-only aggregate audit (writes nothing)
    speakers(path)                                  the two people, for a selection screen (display names included: local use only)
    init_profile(path, speaker=..., ...)            bind a profile to one username (local, private)
    prepare(root, profile_id)                       clean -> dialogues -> splits -> dataset version (idempotent; local processing only)
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from stillhere.session.tokens import best_counter

from . import CLEANING_VERSION, DATASET_VERSION, DIALOGUE_VERSION, IDENTITY_VERSION, PARSER_VERSION
from . import cleaning as C
from . import dataset as D
from . import dialogue as G
from . import identity as I
from . import store as S
from . import weflow as W

BUILDER_SHA = hashlib.sha256("|".join((PARSER_VERSION, IDENTITY_VERSION, CLEANING_VERSION, DIALOGUE_VERSION, DATASET_VERSION,
                                       D.SYSTEM_PROMPT_VERSION)).encode()).hexdigest()[:16]
QUOTE_CHARS = 40


def _quoted(exp: W.Export):
    idx = exp.by_platform_id()

    def f(pid: str) -> Optional[str]:
        m = idx.get(pid)
        if m is None or I.is_system(m):
            return None
        if m.type in C.TEXT_TYPES:
            t = " ".join(ln.strip() for ln in m.content.splitlines() if ln.strip())
            return t if len(t) <= QUOTE_CHARS else t[:QUOTE_CHARS] + "…"
        if m.type == "动画表情" and m.emoji_caption:
            return f"[表情：{m.emoji_caption}]"
        return C.MEDIA_LABEL.get(m.type, "[其他消息]")
    return f


def _samples(exp, ident, username, cfg, count):
    evs, cstats = C.events(exp.messages, username, cfg)
    samples, sstats, shape = G.build(evs, cfg, count=count, quoted_text=_quoted(exp))
    return evs, samples, cstats, sstats, shape


def speakers(path) -> List[dict]:
    exp = W.load(path)
    ident = I.resolve(exp)
    return [{"index": i + 1, "id": s.id, "role": s.role, "display_name": s.display_name, "display_source": s.display_source,
             "messages": s.messages, "text_messages": s.text_messages, "first": s.first, "last": s.last} for i, s in enumerate(ident.speakers)]


def _identity_summary(ident: I.Identity) -> dict:
    return {"version": ident.version, "speakers": [{"id": s.id, "role": s.role, "isSend": s.is_send, "messages": s.messages, "text_messages": s.text_messages,
                                                    "first": s.first, "last": s.last, "display_source": s.display_source} for s in ident.speakers],
            "system_messages": ident.system_messages, "system_by_attribution": ident.system_by_attribution, "anomalies": ident.anomalies,
            "quotes": ident.quotes}


def inspect(path, *, cfg: Optional[C.CleaningConfig] = None, tokenizer: Optional[str] = None, merge_minutes=(2, 5, 10)) -> dict:
    """Read-only aggregate audit of an export for both people and several turn-merge windows. Writes nothing; returns counts only."""
    cfg = cfg or C.CleaningConfig()
    count = best_counter(tokenizer)
    exp = W.load(path)
    ident = I.resolve(exp)
    out = {"source_sha256": exp.sha256, "weflow_version": exp.weflow_version, "parse": exp.report, "identity": _identity_summary(ident),
           "token_mode": count.mode, "merge_comparison": {}, "dataset_preview": {}}
    for sp in ident.speakers:
        for mm in merge_minutes:
            c2 = dataclasses.replace(cfg, turn_merge_minutes=mm)
            _, samples, cst, sst, shape = _samples(exp, ident, sp.username, c2, count)
            tt = [count(s.target_text) for s in samples]
            out["merge_comparison"][f"{sp.role}|{mm}min"] = {
                "samples": len(samples), "sessions": shape["sessions"], "target_turns": shape["target_turns"], "merged_turns": shape["merged_turns"],
                "merged_with_gap_over_2min": shape["merged_long_gap"], "bubbles_per_target_mean": round(sum(len(s.target_bubbles) for s in samples) / max(1, len(samples)), 2),
                "target_chars_p50": D.q([len(s.target_text) for s in samples], .5), "target_tokens_p50": D.q(tt, .5), "target_tokens_p90": D.q(tt, .9),
                "supervised_tokens": sum(t + count.suffix() for t in tt), "excluded": sst}
        _, samples, cst, sst, shape = _samples(exp, ident, sp.username, cfg, count)
        try:
            _, rep = D.assemble(samples, cfg, count=count, suffix_tokens=count.suffix(), msg_time={m.platform_id: m.create_time for m in exp.messages},
                                persona_id="inspect", speaker_hash=sp.id, version_id="inspect")
            out["dataset_preview"][sp.role] = {"splits": {k: {x: v[x] for x in ("samples", "sessions", "supervised_tokens")} for k, v in rep["splits"].items()},
                                               "checks": rep["checks"], "leakage_control": rep["leakage_control"], "train_cap_removed": rep["train_cap_removed"],
                                               "buffer_removed_samples": rep["buffer_removed_samples"]}
        except D.DatasetError as e:
            out["dataset_preview"][sp.role] = {"error": str(e)}
    return out


def _profile_id(source_sha: str, username: str) -> str:
    return "p-" + hashlib.sha256(f"{source_sha}|{username}".encode()).hexdigest()[:10]


def init_profile(path, *, speaker: str, root=None, profile_id: Optional[str] = None, display_name: Optional[str] = None,
                 accept_anomalies: bool = False) -> dict:
    root = Path(root) if root else S.default_root()
    S.check_private_root(root)
    exp = W.load(path)
    ident = I.resolve(exp)
    if ident.blocking and not accept_anomalies:
        raise S.StoreError("IDENTITY_ANOMALY", "; ".join(f"{a['code']} x{a['count']}" for a in ident.blocking) + " (review, then re-run with --accept-anomalies)")
    sp = ident.find(speaker)
    pid = profile_id or _profile_id(exp.sha256, sp.username)
    pdir = S.profile_dir(root, pid)
    pfile = pdir / "profile.json"
    reused = False
    if pfile.exists():
        old = S.read_json(pfile)
        if old["source"]["sha256"] != exp.sha256 or old["target"]["username"] != sp.username:
            raise S.StoreError("PROFILE_EXISTS", f"profile {pid} is bound to another file or person; it is never overwritten: choose a new --profile")
        reused = True
    else:
        session_fp = hashlib.sha256(json.dumps({"wxid": exp.session["wxid"], "type": exp.session["type"], "avatars": exp.avatars}, sort_keys=True).encode()).hexdigest()
        profile = {"schema": "stillhere.persona.profile/1.0", "profile_id": pid, "created_at": S.now_iso(),
                   "source": {"path": str(Path(path).resolve()), "sha256": exp.sha256, "size": exp.size, "format": "weflow", "weflow_version": exp.weflow_version},
                   "session_fingerprint": session_fp, "parser_version": PARSER_VERSION, "identity_version": IDENTITY_VERSION,
                   "target": {"username": sp.username, "speaker_id": sp.id, "role": sp.role, "is_send": sp.is_send,
                              "display_name": display_name or sp.display_name, "display_name_source": "user" if display_name else sp.display_source},
                   "identity_anomalies_accepted": [a["code"] for a in ident.blocking] if accept_anomalies else [],
                   "authorization": {"target_persona_id": pid, "scope": ["style"], "style_speakers": [sp.id], "public": False},
                   "privacy": dict(S.PRIVACY_DEFAULTS)}
        pdir.mkdir(parents=True, exist_ok=True)
        S.write_json(pfile, profile)
        S.append_lineage(pdir, {"artefact": "profile.json", "kind": "profile", "source_sha256": exp.sha256})
    return {"profile_id": pid, "profile_dir": str(pdir), "reused": reused, "speaker": {"id": sp.id, "role": sp.role, "is_send": sp.is_send},
            "source_sha256": exp.sha256}


def privacy_of(prof: dict) -> dict:
    """Profiles from before the simplification (no ``privacy`` block) get the local-only defaults; nothing is rewritten."""
    return {**S.PRIVACY_DEFAULTS, **(prof.get("privacy") or {})}


def _rmtree(p: Path) -> None:
    """Remove a half-built temporary directory, including sealed (read-only) files."""
    if not p.exists():
        return

    def onerror(func, path, _exc):
        os.chmod(path, 0o600)
        func(path)
    shutil.rmtree(p, onerror=onerror)


def load_config(path=None) -> C.CleaningConfig:
    if not path:
        return C.CleaningConfig()
    return C.CleaningConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _verify_dataset(ddir: Path) -> bool:
    mpath = ddir / "DATASET_MANIFEST.json"
    if not mpath.is_file():
        return False
    man = S.read_json(mpath)
    return all((ddir / n).is_file() and D.sha256_bytes((ddir / n).read_bytes()) == h for n, h in man["files"].items())


def prepare(root, profile_id: str, *, config_path=None, tokenizer: Optional[str] = None) -> dict:
    root = Path(root) if root else S.default_root()
    S.check_private_root(root)
    pdir = S.profile_dir(root, profile_id)
    if not (pdir / "profile.json").is_file():
        raise S.StoreError("NO_PROFILE", f"profile {profile_id} not found under {root}")
    prof = S.read_json(pdir / "profile.json")
    src = Path(prof["source"]["path"])
    if not src.is_file():
        raise S.StoreError("SOURCE_MISSING", f"source file not found: {src}")
    if W.sha256_file(src) != prof["source"]["sha256"]:
        raise S.StoreError("SOURCE_CHANGED", "the source file changed since the profile was created (sha256 differs)")
    cfg = load_config(config_path)
    count = best_counter(tokenizer)
    vid = hashlib.sha256("|".join((prof["source"]["sha256"], prof["target"]["username"], cfg.sha256(), BUILDER_SHA, count.mode,
                                   getattr(count, "sha256", "estimated"))).encode()).hexdigest()[:12]
    ddir = pdir / "datasets" / vid
    if ddir.exists():
        if _verify_dataset(ddir):
            rep = S.read_json(ddir / "QUALITY_REPORT.json")
            return {"profile_id": profile_id, "dataset_version": vid, "dataset_dir": str(ddir), "reused": True, "splits": rep["dataset"]["splits"],
                    "checks": rep["dataset"]["checks"], "gates": rep["gates"]}
        raise S.StoreError("DATASET_CORRUPT", f"{ddir} exists but does not match its manifest; it is not overwritten (inspect or remove it yourself)")
    exp = W.load(src)
    ident = I.resolve(exp)
    sp = next((s for s in ident.speakers if s.username == prof["target"]["username"]), None)
    if sp is None or sp.role != prof["target"]["role"]:
        raise S.StoreError("IDENTITY_CHANGED", "the bound speaker is no longer resolved with the same role")
    evs, samples, cstats, sstats, shape = _samples(exp, ident, sp.username, cfg, count)
    files, rep = D.assemble(samples, cfg, count=count, suffix_tokens=count.suffix(), msg_time={m.platform_id: m.create_time for m in exp.messages},
                            persona_id=profile_id, speaker_hash=sp.id, version_id=vid)
    target_turns_text = shape["target_turns"]
    funnel = {"raw_messages": exp.report["messages_in_file"], "duplicate_imports_dropped": exp.report["duplicate_imports_dropped"],
              "messages": exp.report["messages"], "system_messages_dropped": cstats.get("dropped_system", 0),
              "text_events": cstats.get("text_events", 0), "quote_events": cstats.get("quote_events", 0),
              "sticker_with_caption": cstats.get("sticker_with_caption", 0), "media_events": cstats.get("media_events", 0),
              "media_collapsed_repeats": cstats.get("media_collapsed", 0), "placeholder_text_as_media": cstats.get("placeholder_text_as_media", 0),
              "sessions": shape["sessions"], "turns": shape["turns"], "target_turns": target_turns_text, "other_turns": shape["other_turns"],
              "excluded": sstats, "samples_before_split": len(samples), "buffer_removed": rep["buffer_removed_samples"],
              "leakage_removed_from_train": rep["leakage_control"], "train_cap_removed": rep["train_cap_removed"],
              "final": {k: v["samples"] for k, v in rep["splits"].items()}}
    gates = {**rep["checks"], "two_speakers_resolved": len(ident.speakers) == 2, "no_blocking_identity_anomaly": not ident.blocking or
             bool(prof.get("identity_anomalies_accepted")), "train_samples_ge_200": rep["splits"]["train"]["samples"] >= 200}
    quality = {"schema": "stillhere.persona.quality_report/1.0", "label": "FACT (computed by the pipeline on this source); counts only, no text",
               "profile_id": profile_id, "dataset_version": vid, "source_sha256": prof["source"]["sha256"], "target": {"speaker_id": sp.id, "role": sp.role},
               "parse": exp.report, "identity": _identity_summary(ident), "cleaning": cstats, "dialogue": shape, "funnel": funnel, "dataset": rep,
               "token_mode": count.mode, "gates": gates, "gates_pass": all(gates.values())}
    tmp = pdir / "datasets" / f".tmp-{vid}-{os.getpid()}"
    _rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        for n, b in files.items():
            S.atomic_write(tmp / n, b)
        S.write_json(tmp / "cleaning_config.json", cfg.to_dict())
        S.write_json(tmp / "QUALITY_REPORT.json", quality)
        S.write_json(tmp / "dataset_info.json", {"persona_train": {"file_name": "train.jsonl", "formatting": "sharegpt", "columns": {"messages": "conversations", "system": "system"},
                                                                    "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"}},
                                                  "persona_dev": {"file_name": "dev.jsonl", "formatting": "sharegpt", "columns": {"messages": "conversations", "system": "system"},
                                                                  "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"}}})
        names = sorted(p.name for p in tmp.iterdir() if p.is_file())
        manifest = {"schema": "stillhere.persona.dataset_manifest/1.0", "profile_id": profile_id, "dataset_version": vid, "created_at": S.now_iso(),
                    "lineage": {"source_sha256": prof["source"]["sha256"], "parser_version": PARSER_VERSION, "identity_version": IDENTITY_VERSION,
                                "cleaning_config_sha256": cfg.sha256(), "dialogue_version": DIALOGUE_VERSION, "dataset_version": DATASET_VERSION,
                                "system_prompt_sha256": rep["system_prompt_sha256"], "builder_sha": BUILDER_SHA, "token_mode": count.mode},
                    "privacy_policy": privacy_of(prof),
                    "files": {n: D.sha256_bytes((tmp / n).read_bytes()) for n in names},
                    "sealed": ["test.jsonl", "test_meta.jsonl"], "privacy": "PRIVATE: real chat text; never commit; local only unless the operator confirms a cloud package"}
        S.write_json(tmp / "DATASET_MANIFEST.json", manifest)
        for n in ("test.jsonl", "test_meta.jsonl"):
            D.seal(tmp / n)
        os.replace(tmp, ddir)
    except BaseException:
        _rmtree(tmp)
        raise
    S.append_lineage(pdir, {"artefact": f"datasets/{vid}", "kind": "dataset", "manifest_sha256": D.sha256_bytes((ddir / "DATASET_MANIFEST.json").read_bytes())})
    return {"profile_id": profile_id, "dataset_version": vid, "dataset_dir": str(ddir), "reused": False, "splits": rep["splits"], "checks": rep["checks"],
            "gates": gates, "funnel": funnel, "token_mode": count.mode}


def prepare_all(root, profile_id: str, *, config_path=None, tokenizer: Optional[str] = None) -> dict:
    """``persona prepare``: the dataset (above) and then the AUTO_GENERATED persona card + prompt for it (train split only)."""
    from . import card as K
    res = prepare(root, profile_id, config_path=config_path, tokenizer=tokenizer)
    pdir = S.profile_dir(Path(root) if root else S.default_root(), profile_id)
    prof = S.read_json(pdir / "profile.json")
    c = K.build_auto(pdir, Path(res["dataset_dir"]), prof, counter=best_counter(tokenizer))
    cur = K.current(pdir)
    res["card"] = {"card_id": c["card_id"], "reused": c["reused"], "prompt_sha256": c["prompt_sha256"], "review_status": c["review_status"],
                   "current_card": cur["card_id"], "current_review_status": cur["manifest"]["review_status"], "card_dir": c["card_dir"],
                   "prompt_tokens": c.get("prompt_tokens")}
    return res


def evaluate(root, profile_id: str, *, split: str = "dev", groups=("A", "B", "C", "D", "E"), adapter_id: Optional[str] = None, backend: str = "fake",
             base_dir: Optional[str] = None, limit: Optional[int] = None, unseal_test: bool = False, blind=None, log=lambda s: None) -> dict:
    """Local A-E evaluation (``--backend hf`` on a CUDA GPU, or ``fake``: the labelled CPU dry run). TEST only with ``unseal_test`` after a lock."""
    from . import adapters as AD
    from . import card as K
    from . import evaluation as EV
    from . import memory as M
    from . import package as P
    from .models import make_generator
    root = Path(root) if root else S.default_root()
    pdir = S.profile_dir(root, profile_id)
    prof = S.read_json(pdir / "profile.json")
    ddir = P.latest_dataset(pdir)
    cur = K.current(pdir)
    if cur is None or cur["manifest"]["dataset_version"] != ddir.name:
        raise S.StoreError("NO_CARD", "no persona card for the latest dataset: run persona prepare")
    groups = [g for g in groups if g]
    bad = [g for g in groups if g not in EV.GROUPS]
    if bad:
        raise EV.EvalError("BAD_GROUP", f"unknown groups {bad}")
    need_adapter = [g for g in groups if EV.GROUPS[g]["adapter"]]
    ad = AD.resolve(pdir, adapter_id) if adapter_id else None
    if need_adapter and backend.startswith("hf") and ad is None:
        raise EV.EvalError("NO_ADAPTER", f"groups {need_adapter} need --adapter <installed adapter id>")
    if ad and ad["manifest"]["persona_prompt_sha256"] != cur["manifest"]["prompt_sha256"]:
        raise EV.EvalError("PROMPT_MISMATCH", "the adapter was selected with another persona prompt than the current card (B / C must use the same prompt)")
    if split == "dev":
        es = P.dev_eval_set(pdir, ddir, prof, cur, with_memory=True)
    elif split == "test":
        if not backend.startswith("hf"):
            raise EV.EvalError("TEST_SEALED", "the sealed test set is opened only for the real model (--backend hf), never for a dry run")
        if not unseal_test:
            raise EV.EvalError("TEST_SEALED", "test is sealed: freeze the configuration on dev first, then pass --unseal-test (evaluated once, logged)")
        eroot = pdir / "evals"
        lock = EV.lock_for_test(eroot, dataset_version=ddir.name, prompt_sha256=cur["manifest"]["prompt_sha256"],
                                adapter_sha256=ad["manifest"]["files"]["adapter/adapter_model.safetensors"] if ad else None,
                                checkpoint=ad["manifest"]["checkpoint"] if ad else None, groups=groups, backend=backend)
        n = EV.log_test_open(eroot, lock["sha256"], "evaluation")
        es = pdir / "evals" / "sets" / f"{ddir.name}_{cur['card_id']}_test.jsonl"
        if not es.is_file():
            items = EV.eval_items(ddir, "test")
            api, mm = M.build_api(prof["source"]["path"], persona_id=profile_id, target_username=prof["target"]["username"], cutoff=M.cutoff_from_train(ddir))
            blocks = M.blocks_for(items, api, profile_id)
            api.close()
            for it in items:
                it["memory"] = blocks[it["sample_id"]]
            EV.write_eval_set(es, items, persona_prompt=cur["prompt"], groups=list(EV.GROUPS),
                              meta={"split": "test", "dataset_version": ddir.name, "card_id": cur["card_id"], "memory": mm})
        log(f"TEST opening #{n} (lock {lock['sha256'][:12]})")
    else:
        raise EV.EvalError("BAD_SPLIT", "split must be dev or test")
    items = EV.read_eval_set(es)
    head = S.read_json(es.with_suffix(".meta.json"))
    adapters = {g: g for g in need_adapter}
    pool = [r["conversations"][1]["value"] for r in D.load_split(ddir, "train", "persona_prompt")][:200]
    gen = make_generator(backend, base_dir, {g: ad["dir"] for g in need_adapter} if (ad and backend.startswith("hf")) else {}, pool=pool)
    fp = {"eval_set_sha256": head["file_sha256"], "persona_prompt_sha256": head["persona_prompt_sha256"], "decode": EV.DECODE, "generator": gen.label,
          "adapter_sha256": ad["manifest"]["files"]["adapter/adapter_model.safetensors"] if ad else None, "eval_version": EV.EVAL_VERSION}
    run = pdir / "evals" / "runs" / f"{split}_{EV.sha_obj(fp)[:12]}"
    summ = EV.generate(gen, items, groups, run, fp, adapters=adapters, limit=limit, log=log)
    score = EV.score_run(run, items, groups, EV.Reference(ddir, cur["card"]))
    keys = ("n", "style_margin_mean", "length_ratio_median", "generic_reply_rate", "top1_reply_share", "repetition_rate", "train_copy_rate",
            "unsupported_specific_rate", "privacy_leaks")
    out = {"run_dir": str(run), "split": split, "status": score["status"], "generation": summ,
           "groups": {g: {k: score["groups"][g].get(k) for k in keys} for g in groups}, "verdicts": score["verdicts"]}
    if blind:
        out["blind"] = EV.make_blind(run, items, blind)
    return out


def pipeline(input_path, *, speaker: str, root=None, target: str = "colab-l4", confirm_cloud: bool = False, base_dir: Optional[str] = None,
             tokenizer: Optional[str] = None, profile_id: Optional[str] = None, display_name: Optional[str] = None, accept_anomalies: bool = False) -> dict:
    """init -> prepare (dataset + auto card) -> package (label audit on CPU when possible). Every stage reuses an identical earlier result."""
    from . import package as P
    if target in P.CLOUD and not confirm_cloud:
        raise P.PackageError("CLOUD_NOT_CONFIRMED", S.CLOUD_NOTICE + " Re-run with --confirm-cloud (or answer yes), or use --target cpu-only.")
    ini = init_profile(input_path, speaker=speaker, root=root, profile_id=profile_id, display_name=display_name, accept_anomalies=accept_anomalies)
    prep = prepare_all(root, ini["profile_id"], tokenizer=tokenizer)
    pkg = P.build(root, ini["profile_id"], target=target, confirm_cloud=confirm_cloud, base_dir=base_dir)
    summary = {k: prep[k] for k in ("dataset_version", "dataset_dir", "reused", "card")}
    summary.update(samples={k: v["samples"] for k, v in prep["splits"].items()}, gates_pass=all(prep["gates"].values()))
    return {"profile": ini, "prepare": summary, "package": pkg}


def same_chat_profiles(path, *, speaker: str, root=None) -> List[dict]:
    """Profiles under ``root`` built from ANOTHER export of the same chat for the same person (same session fingerprint and username,
    different file). A newer export is normally an UPDATE (``persona memory-build --profile <id> --source <file>``), not a new persona."""
    root = Path(root) if root else S.default_root()
    exp = W.load(path)
    sp = I.resolve(exp).find(speaker)
    fp = hashlib.sha256(json.dumps({"wxid": exp.session["wxid"], "type": exp.session["type"], "avatars": exp.avatars}, sort_keys=True).encode()).hexdigest()
    out = []
    for p in sorted(root.glob("*/profile.json")) if root.is_dir() else []:
        d = S.read_json(p)
        if d.get("session_fingerprint") == fp and d["target"]["username"] == sp.username and d["source"]["sha256"] != exp.sha256:
            out.append({"profile_id": d["profile_id"], "created_at": d.get("created_at"), "display_name": d["target"].get("display_name")})
    return out
