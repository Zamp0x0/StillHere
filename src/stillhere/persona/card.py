"""Persona Card + Persona Inference Prompt, generated deterministically from the TRAIN split only.

Inputs (``dataset.load_split`` / ``load_meta`` with purpose ``persona_prompt``: dev and test can not be read here):

* the target person's real replies (the gpt turns of train);
* the other person's lines inside the train contexts, used ONLY as a background corpus: an expression is "characteristic" when the
  target uses it clearly more often than the other person (smoothed log-odds on document frequency, the formula of
  ``style_eval.conditions.lexical_markers``), so the other person's habits are never written into the target's card;
* ``style.profiler.profile`` for the evidence-backed style traits (run on both people; each target trait carries the other person's
  rate for contrast) and ``style.profiler.select_examples`` for few-shot examples (train only; train rows that were near duplicates
  of dev / test were already removed by the dataset builder).

What the card never contains: age, job, family, hobbies, address, relationship status or events. Language statistics say HOW the
person talks; facts come from SQLite memory (+ EF1) at inference time, or from ``user_confirmed_facts`` that the user types in during
``persona review``. Every card version is a new directory; nothing is overwritten.

    <profile>/cards/<card_id>/persona_card.yaml        the editable card (YAML when PyYAML is installed, else JSON, which is YAML)
    <profile>/cards/<card_id>/persona_prompt.txt       the rendered inference prompt (sha256 in the manifest)
    <profile>/cards/<card_id>/persona_card_manifest.json
    <profile>/cards/CURRENT.json                       pointer to the card in use (metadata only)
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from stillhere.style import profiler as PR
from stillhere.style_eval.conditions import TOKEN

from . import cleaning as C
from . import dataset as D
from . import store as S

CARD_SCHEMA = "stillhere.persona.card/1.0"
CARD_GENERATOR_VERSION = "stillhere.persona.card_generator/1.0"
PROMPT_VERSION = "stillhere.persona.inference_prompt/1.0"
MANIFEST_SCHEMA = "stillhere.persona.card_manifest/1.0"
AUTO, REVIEWED = "AUTO_GENERATED", "USER_REVIEWED"
SEED = "stillhere-persona-card-v1"
K_EXAMPLES = 6
MIN_DOCS = 3                    # an expression must occur in at least this many target replies
LOG_ODDS_MIN = 1.0
SHORT_CHARS = 4
QUOTE_RE = re.compile(r"（(?:我|对方)(?:接下来)?引用了：[^）]*）")
PUNCT_ONLY = re.compile(r"^[^\w一-鿿]+$")
EMOJI_CODE = re.compile(r"\[[一-鿿A-Za-z]{1,6}\]")
STICKER = re.compile(r"\[表情[：:]")
SECOND_PERSON = ("你", "您", "你们")
EMOTION_NEG = re.compile(r"难过|伤心|累|烦|哭|不开心|郁闷|生气|压力|焦虑|崩溃|难受|委屈|失眠|害怕|担心|痛")
EMOTION_POS = re.compile(r"开心|高兴|太好了|耶|棒|好耶|激动|幸福|哈哈")
QUESTION = re.compile(r"[？?]|吗|呢$|什么|怎么|为什么|为啥|几点|哪|多少|谁|是不是|要不要|能不能|可不可以")
GREETING = re.compile(r"^(早|早安|晚安|在吗|在不|hi|hello|你好|嗨|哈喽)", re.I)
REQUEST = re.compile(r"帮我|麻烦|能不能|可以吗|要不要|一起|陪我|给我")
EDITABLE = ("display_name", "ai_identity", "user_confirmed_facts", "addressing_confirmed", "extra_style_notes", "avoid")


class CardError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


# ------------------------------------------------------------------ helpers
def _q(xs: Sequence[float], p: float):
    return D.q(list(xs), p)


def _other_lines(human: str) -> List[str]:
    return [ln[3:] for ln in human.split("\n") if ln.startswith("对方：")]


def _clean_line(s: str) -> str:
    return QUOTE_RE.sub("", s).strip()


def scenario_of(other_line: str) -> str:
    t = _clean_line(other_line)
    if EMOTION_NEG.search(t):
        return "emotion_negative"
    if GREETING.search(t):
        return "greeting"
    if QUESTION.search(t):
        return "question"
    if REQUEST.search(t):
        return "request"
    if EMOTION_POS.search(t):
        return "emotion_positive"
    return "statement"


def _share(xs: Sequence[bool]) -> float:
    return round(sum(1 for x in xs if x) / len(xs), 4) if xs else 0.0


def _log_odds(target_docs: Sequence[set], bg_docs: Sequence[set], k: int) -> List[dict]:
    """``style_eval.conditions.lexical_markers``' smoothed log-odds on document frequency, structured (token, counts, score)."""
    df = Counter(t for d in target_docs for t in d)
    bg = Counter(t for d in bg_docs for t in d)
    n, m = len(target_docs), len(bg_docs)
    out = []
    for t, c in df.items():
        if c < MIN_DOCS or PUNCT_ONLY.match(t):
            continue
        p_a = (c + 0.5) / (n + 1)
        p_b = (bg.get(t, 0) + 0.5) / (m + 1)
        s = math.log(p_a / (1 - p_a)) - math.log(p_b / (1 - p_b))
        if s > LOG_ODDS_MIN:
            out.append({"text": t, "target_replies": c, "other_lines": bg.get(t, 0), "log_odds": round(s, 3)})
    out.sort(key=lambda x: (-x["log_odds"], -x["target_replies"], x["text"]))
    return out[:k]


def _docs(texts: Sequence[str]) -> List[set]:
    return [set(TOKEN.findall(t)) for t in texts]


def _punct(texts: Sequence[str]) -> dict:
    ends = [t.strip()[-1:] if t.strip() else "" for t in texts]
    return {"no_end_punct_share": _share([not re.match(r"[，。！？!?,.~～…]", e) for e in ends]),
            "question_mark_share": _share(["？" in t or "?" in t for t in texts]), "exclaim_share": _share(["！" in t or "!" in t for t in texts]),
            "ellipsis_share": _share(["…" in t or "..." in t for t in texts]), "tilde_share": _share(["~" in t or "～" in t for t in texts]),
            "comma_share": _share(["，" in t or "," in t for t in texts]), "period_share": _share(["。" in t for t in texts]),
            "space_as_pause_share": _share([bool(re.search(r"[一-鿿] [一-鿿]", t)) for t in texts])}


def _emoji(texts: Sequence[str]) -> dict:
    from stillhere.style.features import EMOJI, KAOMOJI, LAUGH
    return {"unicode_emoji_share": _share([bool(EMOJI.search(t)) for t in texts]), "wechat_emoji_code_share": _share([bool(EMOJI_CODE.search(t)) and not STICKER.search(t) for t in texts]),
            "sticker_share": _share([bool(STICKER.search(t)) for t in texts]), "kaomoji_share": _share([bool(KAOMOJI.search(t)) for t in texts]),
            "laugh_share": _share([bool(LAUGH.search(t)) for t in texts])}


def _language(texts: Sequence[str]) -> dict:
    joined = "".join(texts)
    cjk = len(re.findall(r"[一-鿿]", joined))
    latin = len(re.findall(r"[A-Za-z]", joined))
    total = max(1, cjk + latin)
    primary = "zh" if cjk / total >= 0.7 else "en" if latin / total >= 0.7 else "mixed"
    return {"primary": primary, "cjk_char_share": round(cjk / total, 4), "replies_with_latin_word": _share([bool(re.search(r"[A-Za-z]{2,}", t)) for t in texts])}


def _display(name: Optional[str]) -> Tuple[str, bool]:
    n = (name or "").strip()
    ok = bool(re.search(r"[0-9A-Za-z一-鿿]", n)) and "\n" not in n and len(n) <= 32
    return (n, False) if ok else ("TA", True)


def canonical(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_obj(obj) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


# ------------------------------------------------------------------ generation
def generate(dataset_dir, *, persona_id: str, display_name: Optional[str], speaker_hash: str) -> dict:
    """The AUTO_GENERATED card for one dataset version. Pure and deterministic (same train split -> byte-identical card)."""
    ddir = Path(dataset_dir)
    rows = D.load_split(ddir, "train", "persona_prompt")
    metas = {m["sample_id"]: m for m in D.load_meta(ddir, "train", "persona_prompt")}
    man = S.read_json(ddir / "DATASET_MANIFEST.json")
    if not rows:
        raise CardError("EMPTY_TRAIN", "the train split is empty")
    replies = [r["conversations"][1]["value"] for r in rows]
    bubbles = [x.split("\n") for x in replies]
    flat = ["".join(b) for b in bubbles]
    others_by_row = [_other_lines(r["conversations"][0]["value"]) for r in rows]
    last_other = [o[-1] if o else "" for o in others_by_row]
    other_unique = sorted({_clean_line(x) for o in others_by_row for x in o if _clean_line(x) and "（已隐藏）" not in x})
    scen = [scenario_of(x) for x in last_other]

    history = [{"message_id": r["id"], "bubbles": b, "scenario_type": s} for r, b, s in zip(rows, bubbles, scen)]
    prof = PR.profile(history)
    other_prof = PR.profile([{"message_id": f"o{i}", "bubbles": [t], "scenario_type": scenario_of(t)} for i, t in enumerate(other_unique)]) if other_unique else {"traits": []}
    other_rate = {t["trait"]: t["evidence_count"] / max(1, t["of_messages"]) for t in other_prof["traits"]}
    traits = []
    for t in prof["traits"]:
        rate = t["evidence_count"] / max(1, t["of_messages"])
        o = other_rate.get(t["trait"], 0.0)
        traits.append({"trait": t["trait"], "description": t["description"], "type": t["type"], "stability": t["stability"],
                       "target_rate": round(rate, 4), "other_rate": round(o, 4), "distinctive_vs_other": rate >= 1.5 * o and rate > 0,
                       "evidence_count": t["evidence_count"], "of_replies": t["of_messages"]})

    chars = [len(t) for t in flat]
    nb = [len(b) for b in bubbles]
    short = Counter(t for t in flat if 0 < len(t) <= SHORT_CHARS)
    expressions = _log_odds(_docs(flat), _docs(other_unique), 12)
    starts = Counter()
    for b in bubbles:
        m = re.match(r"^([一-鿿A-Za-z]{1,3})[，,!！~～ ]", b[0]) if b and b[0] else None
        if m:
            starts[m.group(1)] += 1
    other_starts = Counter()
    for t in other_unique:
        m = re.match(r"^([一-鿿A-Za-z]{1,3})[，,!！~～ ]", t)
        if m:
            other_starts[m.group(1)] += 1
    address_candidates = [{"text": w, "target_replies": c, "other_lines": other_starts.get(w, 0)} for w, c in starts.most_common(30)
                          if c >= MIN_DOCS and c / len(rows) >= 2 * other_starts.get(w, 0) / max(1, len(other_unique)) and not QUESTION.search(w)][:6]

    by_scen = {}
    for s in sorted(set(scen)):
        idx = [i for i, x in enumerate(scen) if x == s]
        by_scen[s] = {"replies": len(idx), "chars_p50": _q([chars[i] for i in idx], .5), "multi_bubble_share": _share([nb[i] > 1 for i in idx]),
                      "asks_back_share": _share([bool(QUESTION.search(flat[i])) for i in idx])}
    emo = [i for i, x in enumerate(last_other) if EMOTION_NEG.search(_clean_line(x))]

    flags = {sid: set(metas.get(sid, {}).get("flags", [])) for sid in metas}
    lo, hi = _q(chars, .25) or 1, _q(chars, .9) or 40
    pool = []
    for r, b, s, o in zip(rows, bubbles, scen, last_other):
        f = flags.get(r["id"], set())
        text = "".join(b)
        oc = _clean_line(o)
        if not oc or any(x.startswith("privacy:") for x in f) or f & {"media_dependent", "target_turn_has_media", "target_quotes", "target_quote_shown_in_context"}:
            continue
        if "（已隐藏）" in oc or "[" in oc or C.privacy_hits(text) or C.privacy_hits(oc) or not (lo <= len(text) <= hi) or len(oc) > 60:
            continue
        pool.append({"message_id": r["id"], "bubbles": b, "scenario_type": s, "other": oc})
    chosen = PR.select_examples(pool, K_EXAMPLES, seed=SEED)
    examples = [{"sample_id": m["message_id"], "scenario": m["scenario_type"], "other": m["other"], "reply": list(m["bubbles"])} for m in chosen]

    name, needs_review = _display(display_name)
    card = {
        "schema": CARD_SCHEMA, "persona_id": persona_id, "display_name": name, "display_name_needs_review": needs_review,
        "source_speaker_hash": speaker_hash, "review_status": AUTO,
        "ai_identity": f"你是一个 AI 数字人，按照{name}在微信聊天里的说话方式和对方聊天；你不是{name}本人。对方真诚地问你是不是 AI 时，如实说明。",
        "language": _language(flat),
        "reply_length": {"replies": len(rows), "chars_p25": _q(chars, .25), "chars_p50": _q(chars, .5), "chars_p90": _q(chars, .9),
                         "bubbles_mean": round(sum(nb) / len(nb), 2), "bubbles_share": {"1": _share([n == 1 for n in nb]), "2": _share([n == 2 for n in nb]), "3+": _share([n >= 3 for n in nb])},
                         "short_reply_share": _share([0 < c <= 2 for c in chars])},
        "style_traits": traits,
        "punctuation": _punct(flat), "emoji": _emoji(flat),
        "common_expressions": expressions,
        "frequent_short_replies": [{"text": t, "count": c, "share": round(c / len(rows), 4)} for t, c in short.most_common(8)],
        "addressing": {"second_person_share": {p: _share([p in t for t in flat]) for p in SECOND_PERSON}, "candidates": address_candidates,
                       "note": "candidates are vocative-looking openings, more frequent for the target than for the other person; NOT rendered until confirmed in persona review"},
        "addressing_confirmed": [],
        "response_patterns": by_scen,
        "emotional_response": {"contexts": len(emo), "chars_p50": _q([chars[i] for i in emo], .5) if emo else None,
                               "asks_back_share": _share([bool(QUESTION.search(flat[i])) for i in emo]), "multi_bubble_share": _share([nb[i] > 1 for i in emo])},
        "conversation_behavior": {"asks_question_share": _share([bool(QUESTION.search(t)) for t in flat]), "multi_bubble_share": _share([n > 1 for n in nb]),
                                  "quotes_share": _share(["target_quotes" in flags.get(r["id"], set()) for r in rows]),
                                  "turn_with_media_share": _share(["target_turn_has_media" in flags.get(r["id"], set()) for r in rows])},
        "examples": examples,
        "user_confirmed_facts": [], "extra_style_notes": [], "avoid": [],
        "privacy_rules": ["不说出电话、证件号、银行卡、邮箱、住址、密码等隐私信息", "不透露聊天中第三方的隐私"],
        "memory_rules": ["只有【记忆证据】或【用户确认的人物设定】里写明的内容才是事实", "没有证据时不要假装记得具体的过去事件，自然地说不清楚或反问"],
        "evidence": {"split": "train", "dataset_version": man["dataset_version"], "train_sha256": man["files"]["train.jsonl"],
                     "train_meta_sha256": man["files"]["train_meta.jsonl"], "target_replies_used": len(rows), "other_lines_used_as_background": len(other_unique),
                     "profiler_version": prof.get("profiler_version"), "profile_sha256": prof.get("profile_sha256"), "generator_version": CARD_GENERATOR_VERSION,
                     "example_ids": [e["sample_id"] for e in examples], "dev_test_read": False,
                     "method": "deterministic rules: style.profiler (both people) + log-odds vs the other person + style.profiler.select_examples (train only)"},
    }
    return card


# ------------------------------------------------------------------ rendering
def _pct(x) -> str:
    return f"{round(100 * (x or 0))}%"


def render_prompt(card: Mapping) -> str:
    """The Persona Inference Prompt: used verbatim (byte-identical, sha256-pinned) by B / C / D / E and by persona chat. A card version
    with an ``authored_prompt`` (validated by ``prompt_writer.validate_prompt`` when it was saved) renders that text."""
    if card.get("authored_prompt"):
        return card["authored_prompt"].strip() + "\n"
    n = card["display_name"]
    rl = card["reply_length"]
    opt = card.get("prompt_options") or {}            # Q1 refinement knobs; absent = the original rendering, byte for byte
    if opt.get("length") == "soft":
        length_line = (f"- 回复长度跟着对话走：一半的回复在 {rl['chars_p50']} 个字以内，九成在 {rl['chars_p90']} 个字以内；"
                       "该多说一句时就多说一句，先把对方的话接住。")
    elif opt.get("length") == "cap":
        length_line = f"- 回复一般不超过 {rl['chars_p90']} 个字（九成回复的长度），一半的回复在 {rl['chars_p50']} 个字以内。"
    else:
        length_line = f"- 回复通常很短：一半的回复不超过 {rl['chars_p50']} 个字，九成不超过 {rl['chars_p90']} 个字。"
    lines = ["【你是谁】", card["ai_identity"], "", f"【说话方式】（根据{n}在训练集里的 {rl['replies']} 条真实回复统计）", length_line,
             f"- 平均每次发 {rl['bubbles_mean']} 条消息；{_pct(1 - rl['bubbles_share']['1'])} 的时候会分成几条发，每条一行。"]
    if opt.get("bubble_cap"):
        lines.append(f"- 一次最多发 {int(opt['bubble_cap'])} 条。")
    for t in sorted(card["style_traits"], key=lambda t: (t["stability"] != "stable", t["trait"])):
        if t["type"] == "unconfirmed_hypothesis" or t["target_rate"] < 0.05:
            continue
        lines.append(f"- {t['description']}" + ("（只在部分场合出现）" if t["stability"] == "context_dependent" else ""))
    p = card["punctuation"]
    if p["no_end_punct_share"] >= 0.6:
        lines.append(f"- 句末大多不加标点（{_pct(p['no_end_punct_share'])} 的回复）。")
    e = card["emoji"]
    for key, label in (("sticker_share", "表情包"), ("wechat_emoji_code_share", "微信表情（如 [捂脸]）"), ("unicode_emoji_share", "emoji"), ("laugh_share", "笑声（哈哈）")):
        if e[key] >= 0.05:
            lines.append(f"- 常用{label}（{_pct(e[key])} 的回复）。")
    lines += [f"- {x}" for x in card.get("extra_style_notes") or []]
    ex = card["common_expressions"][:8]
    sr = card["frequent_short_replies"][:5]
    if ex or sr:
        lines += ["", "【常用表达】"]
        if ex:
            lines.append("- 比对方明显更常用的说法：" + "、".join(f"“{x['text']}”" for x in ex) + "（自然地用，不要每句都用）")
        if sr:
            lines.append("- 常见的简短回复：" + "、".join(f"“{x['text']}”" for x in sr))
    if card.get("addressing_confirmed"):
        lines += ["", "【称呼】", "- 称呼对方：" + "、".join(card["addressing_confirmed"])]
    else:
        lines += ["", "【称呼】", "- 不确定怎么称呼对方时，不要自己编昵称。"]
    if card.get("user_confirmed_facts"):
        lines += ["", "【用户确认的人物设定】"] + [f"- {x}" for x in card["user_confirmed_facts"]]
    if card["examples"]:
        lines += ["", "【示例对话】（过去聊天里的真实片段，只示范说话方式；里面的内容不是现在的事实）"]
        for x in card["examples"]:
            lines.append(f"对方：{x['other']}")
            lines.append("我：" + " / ".join(x["reply"]))
    lines += ["", "【事实与隐私】", "- 只根据当前聊天和提供的记忆回复；年龄、职业、家人、住址、经历等个人信息，没有依据就不要编造，可以自然地说不清楚或反问。"]
    lines += [f"- {x}" for x in card["privacy_rules"]] + [f"- 避免：{x}" for x in card.get("avoid") or []]
    lines += ["", "【记忆使用】"] + [f"- {x}" for x in card["memory_rules"]]
    lines += ["", "【回复要求】", f"- 像微信聊天一样，用{n}平时的长度和语气回复对方的最新消息；可以分成几条，每条一行。", "- 只输出回复本身，不要加“我：”前缀，不要解释。"]
    if opt.get("respond_first"):
        lines.append("- 先回应对方最新一句话的内容，回答要和上下文接得上。")
    if opt.get("anti_generic"):
        lines.append("- 上面列出的简短回复只在真的合适时用，不要每次都用同一句。")
    return "\n".join(lines) + "\n"


def to_yaml(card: Mapping) -> str:
    try:
        import yaml
        return yaml.safe_dump(dict(card), allow_unicode=True, sort_keys=False, width=1000)
    except ImportError:                                   # JSON is valid YAML
        return json.dumps(card, ensure_ascii=False, indent=1) + "\n"


def from_yaml(text: str) -> dict:
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        return json.loads(text)


def validate(card: Mapping) -> List[str]:
    errs = []
    for k in ("schema", "persona_id", "display_name", "ai_identity", "reply_length", "style_traits", "examples", "evidence", "review_status"):
        if k not in card:
            errs.append(f"missing {k}")
    if card.get("schema") != CARD_SCHEMA:
        errs.append("wrong schema")
    if (card.get("evidence") or {}).get("split") != "train":
        errs.append("evidence.split must be train")
    if card.get("review_status") not in (AUTO, REVIEWED):
        errs.append("review_status must be AUTO_GENERATED or USER_REVIEWED")
    for k in ("user_confirmed_facts", "addressing_confirmed", "extra_style_notes", "avoid"):
        if not isinstance(card.get(k, []), list) or not all(isinstance(x, str) and x.strip() and "\n" not in x for x in card.get(k, [])):
            errs.append(f"{k} must be a list of single-line strings")
    from stillhere.style.prompt_builder import _RESERVED_TAGS, _SPECIAL_TOKEN_RE
    blob = json.dumps(card, ensure_ascii=False)
    if _SPECIAL_TOKEN_RE.search(blob) or any(t in blob for t in _RESERVED_TAGS):
        errs.append("special tokens / reserved tags are not allowed in a card")
    return errs


# ------------------------------------------------------------------ storage (versioned; never overwritten)
def card_id(card: Mapping) -> str:
    return "c-" + sha256_obj(card)[:12]


def _write_version(pdir: Path, card: dict, *, parent: Optional[str], counter=None) -> dict:
    errs = validate(card)
    if errs:
        raise CardError("CARD_INVALID", "; ".join(errs))
    cid = card_id(card)
    cdir = pdir / "cards" / cid
    prompt = render_prompt(card)
    man = {"schema": MANIFEST_SCHEMA, "card_id": cid, "persona_id": card["persona_id"], "dataset_version": card["evidence"]["dataset_version"],
           "training_split_sha256": card["evidence"]["train_sha256"], "profiler_version": card["evidence"]["profiler_version"],
           "generator_version": CARD_GENERATOR_VERSION, "prompt_version": PROMPT_VERSION, "example_ids": card["evidence"]["example_ids"],
           "card_sha256": sha256_obj(card), "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
           "prompt_tokens": counter(prompt) if counter else None, "token_mode": getattr(counter, "mode", None),
           "generation_method": card["evidence"]["method"], "review_status": card["review_status"], "parent_card": parent,
           "privacy": "PRIVATE: derived from a real chat; never commit, never publish"}
    if cdir.exists():
        old = S.read_json(cdir / "persona_card_manifest.json")
        if old["card_sha256"] != man["card_sha256"] or old["prompt_sha256"] != man["prompt_sha256"]:
            raise CardError("CARD_CORRUPT", f"{cdir} exists with other content; it is not overwritten")
        return {**old, "reused": True, "card_dir": str(cdir)}
    tmp = pdir / "cards" / f".tmp-{cid}"
    tmp.mkdir(parents=True, exist_ok=True)
    S.atomic_write(tmp / "persona_card.yaml", to_yaml(card).encode("utf-8"))
    S.atomic_write(tmp / "persona_prompt.txt", prompt.encode("utf-8"))
    man["created_at"] = S.now_iso()
    man["files"] = {n: D.sha256_bytes((tmp / n).read_bytes()) for n in ("persona_card.yaml", "persona_prompt.txt")}
    S.write_json(tmp / "persona_card_manifest.json", man)
    import os
    os.replace(tmp, cdir)
    S.append_lineage(pdir, {"artefact": f"cards/{cid}", "kind": "persona_card", "review_status": card["review_status"], "prompt_sha256": man["prompt_sha256"]})
    return {**man, "reused": False, "card_dir": str(cdir)}


def set_current(pdir: Path, cid: str) -> None:
    S.write_json(pdir / "cards" / "CURRENT.json", {"card_id": cid, "at": S.now_iso()})


def current(pdir: Path) -> Optional[dict]:
    p = pdir / "cards" / "CURRENT.json"
    if not p.is_file():
        return None
    cid = S.read_json(p)["card_id"]
    return load(pdir, cid)


def load(pdir: Path, cid: str) -> dict:
    cdir = pdir / "cards" / cid
    man = S.read_json(cdir / "persona_card_manifest.json")
    for n, h in man["files"].items():
        if D.sha256_bytes((cdir / n).read_bytes()) != h:
            raise CardError("CARD_CORRUPT", f"{cdir / n} does not match its manifest")
    card = from_yaml((cdir / "persona_card.yaml").read_text(encoding="utf-8"))
    prompt = (cdir / "persona_prompt.txt").read_text(encoding="utf-8")
    if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != man["prompt_sha256"] or render_prompt(card) != prompt:
        raise CardError("CARD_CORRUPT", "the prompt is not the rendering of the card")
    return {"card_id": cid, "card": card, "prompt": prompt, "manifest": man, "dir": str(cdir)}


def build_auto(pdir: Path, dataset_dir, prof: Mapping, counter=None) -> dict:
    card = generate(dataset_dir, persona_id=prof["profile_id"], display_name=prof["target"].get("display_name"),
                    speaker_hash=prof["target"]["speaker_id"])
    res = _write_version(pdir, card, parent=None, counter=counter)
    cur = current(pdir) if (pdir / "cards" / "CURRENT.json").is_file() else None
    # a reviewed card of the same dataset stays current; a new dataset version makes its own auto card current
    if cur is None or cur["manifest"]["dataset_version"] != res["dataset_version"]:
        set_current(pdir, res["card_id"])
    return res


def review(pdir: Path, *, edited: Optional[Mapping] = None, approve: bool = False, counter=None) -> dict:
    """A new USER_REVIEWED version from the current card: ``edited`` (the user's edited card; only the EDITABLE fields may change) or
    ``approve`` (same content, status USER_REVIEWED). The previous version stays on disk."""
    cur = current(pdir)
    if cur is None:
        raise CardError("NO_CARD", "no card yet: run persona prepare / pipeline first")
    base = dict(cur["card"])
    new = dict(base)
    if edited is not None:
        changed = sorted(k for k in set(edited) | set(base) if edited.get(k) != base.get(k) and k != "review_status")
        locked = [k for k in changed if k not in EDITABLE]
        if locked:
            raise CardError("FIELD_NOT_EDITABLE", f"only {', '.join(EDITABLE)} may be edited (data-derived fields are recomputed from train): {locked}")
        for k in EDITABLE:
            if k in edited:
                new[k] = edited[k]
        if "display_name" in changed:
            new["display_name_needs_review"] = False
            new["ai_identity"] = edited.get("ai_identity", base["ai_identity"]).replace(base["display_name"], new["display_name"]) \
                if "ai_identity" not in changed else edited["ai_identity"]
    elif not approve:
        raise CardError("NOTHING_TO_DO", "pass an edited card or approve")
    new["review_status"] = REVIEWED
    res = _write_version(pdir, new, parent=cur["card_id"], counter=counter)
    set_current(pdir, res["card_id"])
    return res
