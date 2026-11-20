"""Real chat supervision (IRC + V2EX) from the validated T2-D3 Mini Build-2 episodes, re-rendered for D4. RUNS ON THE OWNER'S DEVICE ONLY.

Reuse, not rebuild: the episodes, author splits, K=5 example selection and the style-profiler-v1 profiles all come unchanged from
``t2d3_mini_build2/build`` (validated builder, B3 package-3). D4 only
  1. filters targets that state specifics the conversation does not support (digits, dates / times / ages / money, names of known speakers or
     @mentions, place names, first-person biographical claims) and episodes whose target, conversation or style examples carry a
     credential-shaped string (API token, private-key block; t2d4/credentials.py), recording a reason per dropped episode; targets are
     never rewritten or cut;
  2. projects the stored profiler output onto Style Profile Schema v1 (profile_v1.build, persona_kind REAL_AUTHOR, persona id = hash);
  3. renders with ``stillhere.response.grounded_persona/1.0`` and NO memory evidence (no-evidence block, or the empty-retrieval block with the
     conversation date), because real chat has no memory store: these rows teach HOW to say things, never facts.
The B3 HOLDOUT is not read here; it is referenced by file hash only.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .credentials import has_credential
from .lexicon import BANKS
from .profile_v1 import build as build_profile
from .renderer import RenderError, render_messages
from . import renderer as RD
from .validate import hygiene
from .vendor.t2d3mini_privacy import _generic_tokens

DIGIT = re.compile(r"\d+")
CN_QTY = re.compile(r"[零一二三四五六七八九十两百千万\d]+\s*(?:月份?|号|日|岁|周岁|块钱?|元|万|千块|年级|点钟|点半|点\d+|个月)")
WEEKDAY = re.compile(r"(?:周|星期|礼拜)[一二三四五六日天]")
MENTION = re.compile(r"@[\w\-\.]+")
CLAIM = re.compile(r"我(?:家(?:在|里)|老婆|老公|媳妇|女朋友|男朋友|对象|妈|爸|儿子|女儿|们?公司|老板|们?学校|们?单位|住在|在[一-鿿]{1,4}(?:上班|工作|读书|上学)"
                   r"|是[一-鿿]{0,6}(?:工程师|程序员|学生|老师|医生|设计师|码农|运维)|今年|毕业于|工资|月薪|年薪|结婚|离婚)")
PLACES = sorted({c for pool in BANKS["city"].values() for c in pool} | {"北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "武汉", "西安", "天津",
                                                                           "重庆", "苏州", "长沙", "郑州", "青岛", "厦门", "大连", "沈阳", "哈尔滨", "香港",
                                                                           "台湾", "美国", "日本", "国外", "老家"}, key=len, reverse=True)


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def known_speakers(rows: Iterable[dict]) -> Set[str]:
    out = set()
    for r in rows:
        out.update(x for x in r.get("context_speakers", []) if x)
        for k in ("partner", "author"):
            if isinstance(r.get(k), str) and r["source"] == "irc":
                out.add(r[k])
    # ordinary words used as nicks ("root", "pointer", ...) are not names: same rule as the T2-D3 privacy guard (>= 4 chars, not a token
    # of the Python standard library sources). No nick is ever written into this code.
    generic = _generic_tokens()
    return {x.lower() for x in out if len(x) >= 4 and x.lower() not in generic}


def ungrounded(target: str, conv_text: str, speakers: Set[str]) -> List[str]:
    """Reasons a real target states specifics its conversation does not support (empty list = keep)."""
    t, c = target, conv_text
    cl = c.lower()
    reasons = []
    if any(d not in DIGIT.findall(c) for d in DIGIT.findall(t)):
        reasons.append("UNGROUNDED_NUMBER")
    if any(_norm(m.group(0)) not in _norm(c) for m in CN_QTY.finditer(t)) or any(m.group(0) not in c for m in WEEKDAY.finditer(t)):
        reasons.append("UNGROUNDED_DATE_TIME_AMOUNT")
    if any(m.group(0).lower() not in cl for m in MENTION.finditer(t)):
        reasons.append("UNGROUNDED_NAME")
    else:
        tl = t.lower()
        for s in speakers:
            if s in tl and s not in cl and re.search(r"(?<![a-z0-9_])" + re.escape(s) + r"(?![a-z0-9_])", tl):
                reasons.append("UNGROUNDED_NAME")
                break
    if any(p in t and p not in c for p in PLACES):
        reasons.append("UNGROUNDED_PLACE")
    if CLAIM.search(t):
        reasons.append("PERSONAL_CLAIM")
    return reasons


def persona_key(author: str, source: str) -> str:
    return "REAL-" + hashlib.sha256(f"t2d4-real:{source}:{author}".encode()).hexdigest()[:12]


def _conversation(r: dict) -> List[dict]:
    return [{"speaker": "partner" if t["speaker"] == "other" else "self", "text": t["text"]} for t in r["conversation"]]


def render_real(r: dict, seed: str, split: str, data_version: str) -> Tuple[Optional[dict], List[str]]:
    target = "\n".join(b for b in r["target_bubbles"] if b.strip())
    conv = _conversation(r)
    conv_text = "".join(t["text"] for t in conv)
    reasons = list(ungrounded(target, conv_text, r["_speakers"]))  # TRAIN: drop; evaluation splits: keep the case, record the flags
    reasons += ["HYGIENE:" + x for x in hygiene(target, max_chars=400) if x != "target_too_long"]
    if has_credential(target, conv_text, *r.get("style_examples", [])):
        reasons.append("CREDENTIAL")
    if reasons and split == "TRAIN":
        return None, reasons
    pid = persona_key(r["author"], r["source"])
    prof = build_profile(r["style_profile"], persona_id=pid, persona_kind="REAL_AUTHOR", history_before=r.get("history_max_key"))
    rnd = random.Random(hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest())
    empty_retrieval = rnd.random() < 0.5
    try:
        msgs = render_messages(profile=prof, examples=r["style_examples"], evidence=[] if empty_retrieval else None, unknown=(),
                               now=(r.get("target_key") if empty_retrieval and re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(r.get("target_key"))) else None),
                               conversation=conv)
    except RenderError as e:
        return None, [f"RENDER:{e}"]
    base_length = None
    if split != "TRAIN" and RD.base_length_enabled():
        bl = RD.base_length_messages(profile=prof, examples=r["style_examples"], evidence=[] if empty_retrieval else None, unknown=(),
                                     now=(r.get("target_key") if empty_retrieval and re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(r.get("target_key"))) else None),
                                     conversation=conv)
        base_length = {"system": bl[0]["content"], "user": bl[1]["content"]}
    eid = "D4R-" + hashlib.sha256(f"{data_version}:{r['id']}".encode()).hexdigest()[:16]
    return {
        "episode_id": eid, "data_version": data_version, "split": split, "source": r["source"].upper(), "persona_id": pid,
        "persona_kind": "REAL_AUTHOR", "group_id": "R-" + eid[4:], "group_type": "REAL", "category": "REAL_STYLE", "family": "S_STYLE",
        "domain": "real_chat", "subject_type": "self", "now": None, "evidence": [] if empty_retrieval else None, "unknown": [],
        "conversation": conv, "partner_turns": sum(1 for t in conv if t["speaker"] == "partner"), "examples": r["style_examples"],
        "profile_v1": prof, "profile_v1_sha256": prof["profile_v1_sha256"], "answer_state": None, "template_family": f"REAL|{r['source']}",
        "semantic_key": f"REAL|{r['source']}|{pid}", "source_episode_id": r["id"], "target_key": r.get("target_key"),
        "history_max_key": r.get("history_max_key"), "example_ids": r.get("example_ids"), "profile_source_ids": r.get("profile_source_ids"),
        "system": msgs[0]["content"], "user": msgs[1]["content"], "target": target, "target_bubbles": [b for b in target.split("\n") if b.strip()],
        "grounding_flags": reasons, "validators": "PASS" if not reasons else "FLAGGED_EVAL_ONLY",
        **({"base_length": base_length} if base_length else {}),
    }, reasons


def load_jsonl(p: Path) -> List[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_real(build_dir: Path, seed: str, data_version: str) -> Dict[str, object]:
    """TRAIN (filtered), DEV_PROBE and DEV_EVAL (B3 cases, re-rendered, never filtered, flags recorded). HOLDOUT: hash only."""
    train = load_jsonl(build_dir / "train" / "train_episodes_meta.jsonl")
    probe = load_jsonl(build_dir / "cases" / "dev_probe.jsonl")
    dev = load_jsonl(build_dir / "cases" / "dev_eval.jsonl")
    speakers = known_speakers(train + probe + dev)
    out: Dict[str, object] = {"TRAIN": [], "DEV_PROBE": [], "DEV_EVAL": [], "dropped": {}, "drop_examples_by_reason": {}}
    dropped: Dict[str, int] = {}
    for split, rows in (("TRAIN", train), ("DEV_PROBE", probe), ("DEV_EVAL", dev)):
        for r in rows:
            r["_speakers"] = speakers
            rec, reasons = render_real(r, seed, split, data_version)
            if rec is None:
                for x in reasons:
                    k = x.split(":")[0]
                    dropped[f"{split}:{k}"] = dropped.get(f"{split}:{k}", 0) + 1
                dropped[f"{split}:EPISODES_DROPPED"] = dropped.get(f"{split}:EPISODES_DROPPED", 0) + 1
                continue
            if split != "TRAIN":
                rec["reference_target"] = rec.pop("target")
                rec["episode_id"] = rec["episode_id"].replace("D4R-", "D4RE-")
                for x in reasons:
                    dropped[f"{split}:FLAG:{x}"] = dropped.get(f"{split}:FLAG:{x}", 0) + 1
            out[split].append(rec)
    out["dropped"] = dropped
    out["source_counts"] = {"TRAIN_IN": len(train), "DEV_PROBE_IN": len(probe), "DEV_EVAL_IN": len(dev)}
    return out
