"""Persona SFT dataset: samples -> chronological, session-grouped train / dev / test -> leakage control -> LLaMA-Factory sharegpt rows,
provenance, quality report and manifests.

Rendering (one training row = one target reply):
    system    = the fixed P1 training instruction (``SYSTEM_PROMPT``; no persona card yet: P2 builds it from TRAIN only)
    human     = the conversation so far, one line per turn (``对方：…`` / ``我：…``), both people
    gpt       = the target person's real reply (bubbles joined by a line break)
With LLaMA-Factory 0.9.5 (template ``qwen3``, ``train_on_prompt: false``) only the gpt turn + ``<|im_end|>\\n`` is supervised; earlier
replies of either person sit inside the human turn and are never supervised (the D4 / D5 layout, checked token by token in D5).

Splits: sessions ordered by start time; train takes whole sessions until it holds ``split[0]`` of the samples, then dev, then test.
Sessions that END within ``split_buffer_hours`` of the next split's first session are left out (a topic buffer; only the earlier split
loses rows). Leakage control removes TRAIN rows only: an exact (last context line, target) pair that also occurs in dev / test, and
near-duplicates (char 3-gram Jaccard >= threshold, both texts >= ``near_dup_min_chars``). The train-only cap keeps at most
``max_same_target_train`` rows per normalised target, evenly spread over time (frequent short replies are kept, not deleted).
The test file is sealed: written once, read-only, its sha256 in the split manifest; ``load_split`` refuses it for training or prompt use.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

from . import DATASET_VERSION
from .cleaning import PLACEHOLDER, CleaningConfig
from .dialogue import Sample
from .weflow import TZ

SYSTEM_PROMPT = ("下面是你和对方此前的微信聊天记录，“我”是你自己，“对方”是和你聊天的人。请用你平时的说话方式回复对方。"
                 "只根据聊天里已有的信息回复，不要编造聊天里没有的事实；看不到的图片、语音、视频等内容不要猜测。回复可以分成几条消息，每条一行。")
SYSTEM_PROMPT_VERSION = "stillhere.persona.sft_system/1.0"
SPLITS = ("train", "dev", "test")
_NORM = re.compile(r"[\s　，。！？、；：,.!?;:~～…“”\"'‘’()（）【】\[\]<>《》·\-—_/]+")


class DatasetError(ValueError):
    pass


def norm(s: str) -> str:
    return _NORM.sub("", s or "").lower()


def grams(s: str) -> set:
    t = norm(s)
    return {t[i:i + 3] for i in range(max(1, len(t) - 2))}


def q(xs: Sequence[int], p: float):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def split_sessions(samples: Sequence[Sample], cfg: CleaningConfig) -> Tuple[Dict[str, str], Dict[str, int]]:
    by: Dict[str, List[Sample]] = defaultdict(list)
    for s in samples:
        by[s.session_id].append(s)
    order = sorted(by, key=lambda k: (by[k][0].session_start, k))
    if len(order) < 3:
        raise DatasetError(f"DATASET_TOO_SMALL: {len(order)} session(s) with samples; at least 3 are needed for train / dev / test")
    total = len(samples)
    assign, cum = {}, 0
    for k in order:
        frac = cum / total
        assign[k] = "train" if frac < cfg.split[0] else "dev" if frac < cfg.split[0] + cfg.split[1] else "test"
        cum += len(by[k])
    for need, prev in (("dev", "train"), ("test", "dev")):          # every split non-empty
        if need not in assign.values():
            last = [k for k in order if assign[k] == prev]
            if len(last) < 2:
                raise DatasetError("DATASET_TOO_SMALL: cannot form non-empty train / dev / test from whole sessions")
            assign[last[-1]] = need
    buffered = {"train": 0, "dev": 0}
    end = {k: max(s.time for s in by[k]) for k in order}
    for earlier, later in (("train", "dev"), ("dev", "test")):
        start_later = min(by[k][0].session_start for k in order if assign[k] == later)
        cand = [k for k in order if assign[k] == earlier and end[k] > start_later - cfg.split_buffer_hours * 3600]
        keep_one = [k for k in order if assign[k] == earlier and k not in cand]
        if not keep_one:                                             # never empty a split for the buffer
            cand = cand[1:]
        for k in cand:
            assign[k] = "buffer"
            buffered[earlier] += len(by[k])
    return assign, buffered


def train_cap(rows: List[Sample], cap: int) -> Tuple[List[Sample], int]:
    if cap <= 0:
        return rows, 0
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(rows):
        groups[norm(s.target_text)].append(i)
    drop = set()
    for idx in groups.values():
        if len(idx) > cap:
            keep = {idx[round(j * (len(idx) - 1) / (cap - 1))] for j in range(cap)} if cap > 1 else {idx[0]}
            drop |= set(idx) - keep
    return [s for i, s in enumerate(rows) if i not in drop], len(drop)


def leakage_filter(train: List[Sample], held: Sequence[Sample], cfg: CleaningConfig) -> Tuple[List[Sample], Dict[str, int]]:
    key = lambda s: norm(s.context_lines[-1] if s.context_lines else "") + "␞" + norm(s.target_text)
    held_keys = {key(s) for s in held}
    held_g = [(grams(s.target_text + s.context_lines[-1]), len(norm(s.target_text + s.context_lines[-1]))) for s in held if s.context_lines]
    out, exact, near = [], 0, 0
    for s in train:
        if key(s) in held_keys:
            exact += 1
            continue
        text = s.target_text + (s.context_lines[-1] if s.context_lines else "")
        n = len(norm(text))
        if n >= cfg.near_dup_min_chars:
            g = grams(text)
            if any(m >= cfg.near_dup_min_chars and min(n, m) / max(n, m) >= cfg.near_dup_threshold and
                   len(g & h) / max(1, len(g | h)) >= cfg.near_dup_threshold for h, m in held_g):
                near += 1
                continue
        out.append(s)
    return out, {"exact_cross_split_removed_from_train": exact, "near_dup_cross_split_removed_from_train": near}


def row(s: Sample) -> dict:
    return {"id": s.sample_id, "system": SYSTEM_PROMPT,
            "conversations": [{"from": "human", "value": "\n".join(s.context_lines)}, {"from": "gpt", "value": s.target_text}]}


def meta(s: Sample, split: str, persona_id: str, speaker_hash: str, cfg_sha: str, version_id: str, tgt_tokens: int, sup: int) -> dict:
    return {"sample_id": s.sample_id, "persona_id": persona_id, "split": split, "session_id": s.session_id, "speaker_hash": speaker_hash,
            "time": datetime.fromtimestamp(s.time, TZ).isoformat(), "context_message_ids": s.context_message_ids,
            "target_message_ids": s.target_message_ids, "source_message_ids": s.context_message_ids + s.target_message_ids,
            "context_tokens": s.context_tokens, "target_tokens": tgt_tokens, "supervised_tokens": sup, "flags": s.flags,
            "cleaning_config_sha256": cfg_sha, "dataset_version": version_id}


def _jsonl(rows: Iterable[dict]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def assemble(samples: List[Sample], cfg: CleaningConfig, *, count: Callable[[str], int], suffix_tokens: int, msg_time: Mapping[str, int],
             persona_id: str, speaker_hash: str, version_id: str) -> Tuple[Dict[str, bytes], dict]:
    """Everything that goes to disk (file name -> bytes) plus the aggregate report. Pure: no I/O."""
    assign, buffered = split_sessions(samples, cfg)
    parts: Dict[str, List[Sample]] = {k: [s for s in samples if assign[s.session_id] == k] for k in SPLITS}
    held = parts["dev"] + parts["test"]
    parts["train"], leak = leakage_filter(parts["train"], held, cfg)
    parts["train"], capped = train_cap(parts["train"], cfg.max_same_target_train)
    cfg_sha = cfg.sha256()
    files, tok = {}, {}
    for k in SPLITS:
        rows_, metas = [], []
        tt, ct, sup = [], [], []
        for s in parts[k]:
            t = count(s.target_text)
            tt.append(t)
            ct.append(s.context_tokens)
            sup.append(t + suffix_tokens)
            rows_.append(row(s))
            metas.append(meta(s, k, persona_id, speaker_hash, cfg_sha, version_id, t, t + suffix_tokens))
        files[f"{k}.jsonl"] = _jsonl(rows_)
        files[f"{k}_meta.jsonl"] = _jsonl(metas)
        tok[k] = {"samples": len(rows_), "target_tokens_p50": q(tt, .5), "target_tokens_p90": q(tt, .9), "target_tokens_p99": q(tt, .99),
                  "context_tokens_p50": q(ct, .5), "context_tokens_p95": q(ct, .95), "supervised_tokens": sum(sup),
                  "short_reply_le2_chars": sum(1 for s in parts[k] if len(norm(s.target_text)) <= 2),
                  "multi_bubble": sum(1 for s in parts[k] if len(s.target_bubbles) > 1),
                  "first": datetime.fromtimestamp(min(s.time for s in parts[k]), TZ).date().isoformat() if parts[k] else None,
                  "last": datetime.fromtimestamp(max(s.time for s in parts[k]), TZ).date().isoformat() if parts[k] else None,
                  "sessions": len({s.session_id for s in parts[k]}), "flags": dict(Counter(f for s in parts[k] for f in s.flags))}
    # property checks (computed, not assumed)
    sess = {k: {s.session_id for s in parts[k]} for k in SPLITS}
    mids = {k: {i for s in parts[k] for i in s.context_message_ids + s.target_message_ids} for k in SPLITS}
    all_s = [s for k in SPLITS for s in parts[k]]
    checks = {
        "sessions_disjoint": all(not (sess[a] & sess[b]) for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))),
        "message_ids_disjoint": all(not (mids[a] & mids[b]) for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))),
        "chronological": all(max((s.time for s in parts[a]), default=0) < min((s.time for s in parts[b]), default=1 << 62)
                             for a, b in (("train", "dev"), ("dev", "test"))),
        "context_before_target": all(msg_time[c] <= s.time for s in all_s for c in s.context_message_ids),
        "target_not_in_context": all(not (set(s.target_message_ids) & set(s.context_message_ids)) for s in all_s),
        "no_placeholder_target": all(not PLACEHOLDER.match(b) for s in all_s for b in s.target_bubbles),
        "every_split_non_empty": all(parts[k] for k in SPLITS),
    }
    report = {"dataset_version": DATASET_VERSION, "system_prompt_version": SYSTEM_PROMPT_VERSION,
              "system_prompt_sha256": sha256_bytes(SYSTEM_PROMPT.encode("utf-8")),
              "splits": tok, "session_assignment": dict(Counter(assign.values())), "buffer_removed_samples": buffered,
              "leakage_control": leak, "train_cap_removed": capped, "checks": checks,
              "supervised_tokens_total": sum(tok[k]["supervised_tokens"] for k in SPLITS)}
    split_manifest = {"assignment": {k: sorted(sess[k]) for k in SPLITS}, "buffer_sessions": sorted(s for s, v in assign.items() if v == "buffer"),
                      "test_sha256": sha256_bytes(files["test.jsonl"]), "test_meta_sha256": sha256_bytes(files["test_meta.jsonl"]), "sealed": True,
                      "rule": "chronological whole-session split; P2 prompt / example selection may read TRAIN only; test is evaluation-only"}
    files["SPLIT_MANIFEST.json"] = json.dumps(split_manifest, ensure_ascii=False, indent=1).encode("utf-8")
    return files, report


PURPOSES = {"training": ("train", "dev"), "persona_prompt": ("train",), "evaluation": ("dev", "test")}


def load_split(dataset_dir, split: str, purpose: str) -> List[dict]:
    """The only reader later stages should use: TRAIN for the persona prompt; TRAIN / DEV for training; TEST for evaluation only, after
    its sha256 is checked against the sealed split manifest."""
    if split not in PURPOSES.get(purpose, ()):
        raise DatasetError(f"SPLIT_FORBIDDEN: {split!r} may not be read for {purpose!r}")
    d = Path(dataset_dir)
    data = (d / f"{split}.jsonl").read_bytes()
    if split == "test":
        man = json.loads((d / "SPLIT_MANIFEST.json").read_text(encoding="utf-8"))
        if sha256_bytes(data) != man["test_sha256"]:
            raise DatasetError("TEST_SEAL_BROKEN: test.jsonl does not match the sealed manifest")
    return [json.loads(l) for l in data.decode("utf-8").splitlines() if l.strip()]


def load_meta(dataset_dir, split: str, purpose: str) -> List[dict]:
    """``<split>_meta.jsonl`` under the same purpose rule as ``load_split`` (the test meta is sealed with the test rows)."""
    if split not in PURPOSES.get(purpose, ()):
        raise DatasetError(f"SPLIT_FORBIDDEN: {split!r} may not be read for {purpose!r}")
    if split == "test":
        load_split(dataset_dir, split, purpose)            # verifies the seal first
    data = (Path(dataset_dir) / f"{split}_meta.jsonl").read_text(encoding="utf-8")
    return [json.loads(l) for l in data.splitlines() if l.strip()]


def seal(path: Path) -> None:
    os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
