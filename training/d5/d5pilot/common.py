"""Shared D5 contracts: the reader v1.1 prompt (unchanged), the evidence headers, the target JSON, the token layout.

* System prompt: ``stillhere.reader.fact.FACT_SYSTEM`` (reader 1.1, Phase 3 B), byte-identical; its sha256 is pinned in every manifest.
* Memory sources (PerLTQA, LoCoMo, D4) are rendered with ``fact_pack.render_evidence`` / ``fact_user`` exactly as Phase 3 (items in time order).
* Document sources (HotpotQA, QASPER) use header ``doc/1.0``: identical wording except two metadata phrases that are false for documents
  ("按时间先后排列" -> "按原文顺序排列"; the "括号里是系统据此整理的记忆" clause is dropped because there are no derived memories).
  No answering rule is changed; Base and D5 are always evaluated on the same rendered prompt.
* Target: reader 1.1 answer-first JSON. Abstention keeps the contract sentence "根据现有记录，无法确定。" as its first sentence; two
  fixed, meaning-identical follow-up sentences are allowed (chosen by a hash of the row id) so that not every abstention is one string.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Iterable, List, Sequence

SEED = "stillhere-d5-pilot-v1-20261116"
CUTOFF = 8192
CUTOFF_FALLBACK = 4096
TARGET_MARGIN = 8
ABSTAIN = "根据现有记录，无法确定。"
ABSTAIN_VARIANTS = (ABSTAIN, ABSTAIN + "现有证据中没有所问的具体信息。", ABSTAIN + "相关记录里没有提到这一点。")
PRE = "<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
SUP_SUFFIX = "<|im_end|>\n"
HEADER_MEMORY = "reader-1.1/memory (fact_pack.render_evidence, unchanged)"
HEADER_DOC = "doc/1.0"
BUCKETS = {"short": (512, 1500), "medium": (1500, 4000), "long": (4000, 7600)}   # input tokens; 8192 cutoff minus room for the target
PUNCT = re.compile(r"[\s　，。！？、；：,.!?;:~～…“”\"'‘’()（）【】\[\]<>《》·\-—_/|]+")


def norm(s) -> str:
    return PUNCT.sub("", str(s or "")).lower()


def h(s: str, n: int = 16) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p) -> str:
    x = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            x.update(c)
    return x.hexdigest()


def frac(key: str) -> float:
    """Deterministic uniform [0, 1) from a key."""
    return int(hashlib.sha256((SEED + ":" + key).encode()).hexdigest()[:12], 16) / 16 ** 12


def abstain_text(row_id: str) -> str:
    f = frac("abstain:" + row_id)
    return ABSTAIN_VARIANTS[0] if f < 0.5 else ABSTAIN_VARIANTS[1] if f < 0.75 else ABSTAIN_VARIANTS[2]


def is_abstain(answer: str) -> bool:
    return (answer or "").startswith(ABSTAIN)


def target_json(row_id: str, answer, eids: Sequence[str], quotes: Iterable[str], answerable: bool) -> str:
    if not answerable:
        return json.dumps({"answer": abstain_text(row_id), "answerable": False, "abstain": True, "evidence": [], "quotes": []}, ensure_ascii=False)
    qs = []
    for q in quotes:
        q = (q or "").strip()
        if q and len(q) <= 160 and q not in qs:
            qs.append(q)
        if len(qs) == 2:
            break
    return json.dumps({"answer": str(answer).strip(), "answerable": True, "abstain": False, "evidence": list(eids), "quotes": qs},
                      ensure_ascii=False)


def doc_evidence(pack) -> str:
    from stillhere.context.fact_pack import render_item
    head = f"【记忆证据】（提问时间：{pack.ask_time[:10]}；按原文顺序排列，共 {len(pack.items)} 条；每条是一段原始资料）"
    body = "\n".join(render_item(i) for i in pack.items) if pack.items else "（没有检索到相关记录）"
    return head + "\n" + body


def user_text(pack, header: str) -> str:
    from stillhere.reader import fact as rf
    if header == HEADER_MEMORY:
        return rf.fact_user(pack)
    parts = [doc_evidence(pack), f"【问题】（提问时间：{pack.ask_time[:10]}）{pack.body}"]
    return "\n\n".join(parts)


def messages(pack, header: str) -> List[dict]:
    from stillhere.reader import fact as rf
    return [{"role": "system", "content": rf.FACT_SYSTEM}, {"role": "user", "content": user_text(pack, header)}]


def fact_system_sha() -> str:
    from stillhere.reader import fact as rf
    return sha256_bytes(rf.FACT_SYSTEM.encode("utf-8"))


_LOOP = re.compile(r"(.{4,}?)\1{2,}", re.S)          # the D4 validator's loop patterns (t2d4/validate.py LOOP / SHORT_LOOP)
_SHORT_LOOP = re.compile(r"(.{1,3}?)\1{5,}", re.S)


def lang_of(source: str) -> str:
    return "zh" if source in ("perltqa", "d4") else "en"


def repetitive(text: str, lang: str, M=None) -> bool:
    """Chinese: the D4 detector itself (``t2d4.colab.metrics.repetitive``: loops, or any character 4-gram x3).
    English: the character 4-gram rule fires on ordinary prose ("tion" x3), so English uses the same loop patterns plus a WORD 4-gram
    repeated 3 times."""
    if lang == "zh" and M is not None:
        return bool(M.repetitive(text))
    t = text or ""
    if _LOOP.search(t) or _SHORT_LOOP.search(t):
        return True
    w = re.findall(r"[a-z0-9']+", t.lower())
    grams = {}
    for i in range(len(w) - 3):
        g = tuple(w[i:i + 4])
        grams[g] = grams.get(g, 0) + 1
    return bool(grams) and max(grams.values()) >= 3


def sharegpt(row_id: str, msgs: List[dict], target: str, **extra) -> dict:
    return {"id": row_id, "system": msgs[0]["content"], "conversations": [{"from": "human", "value": msgs[1]["content"]},
                                                                           {"from": "gpt", "value": target}], **extra}
