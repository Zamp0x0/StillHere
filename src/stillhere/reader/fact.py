"""Fact reader interface for EF1 (``stillhere.reader.fact/1.0``): prompts, reader configurations, generation configs, output checks.

Reader configurations (Phase 3 compares them on the SAME fact pack; only the listed field differs between B and C):

  A_d4_persona   D4 adapter + the pinned D4 grounded-persona renderer, fed with the EF1 items (no 5-item cap)  - the original prompt family
  B_base_fact    Base model (adapter disabled) + FACT prompt + FACT generation
  C_d4_fact      D4 adapter enabled       + FACT prompt + FACT generation               (B vs C: the adapter's effect on fact answering)
  D_base_d4_style  B, then the D4 adapter restates B's answer in the persona's style (style stage; facts must be preserved)

The legacy v3 chain (D4 renderer, 5 evidence items, 1536 tokens, temperature 0.7 / 128 tokens) stays available in the LifeBench adapter
as ``pipeline = "d4_v3"``.

FACT prompt semantics: the asker is the OWNER of the memory ("我" = the user whose history this is). This applies to the memory-assistant
calling mode only; the persona chat runtime keeps its own speaker roles.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

from stillhere.context.fact_pack import FactPack, output_tokens_for, render_evidence, render_notes

READER_VERSION = "stillhere.reader.fact/1.1"   # 1.1: answer-first JSON, cited ids only, Qwen non-thinking sampling (Phase 3 smoke)

FACT_SYSTEM = """你是用户本人的个人记忆助手。用户正在查询自己过去的记录，问题中的“我”指用户本人。
你只能依据下面【记忆证据】中的记录回答；【检索说明】里的日期计算和子问题清单由系统给出，可以直接使用，但它们不是答案。

回答规则：
1. 先找出问题真正要的东西（人名、数字、金额、日期、原话、顺序、原因、状态或结论），再到证据中逐条核对。
2. 证据里有答案时直接回答，不要因为证据措辞与问题不同就拒答；只知道一部分时，回答已知部分并说明缺什么。
3. 证据中没有所问的具体信息时，回答“根据现有记录，无法确定”，不要用相近的事实、常识或猜测代替。
4. 时间：只使用提问时间之前的记录；“34天后”“后的第N天”等日期以【检索说明】的计算为准；问某一天的事，只用那一天的记录。
5. 排序题：先写出每件事在证据中的日期，再按日期先后排列；不要照抄题目中的顺序。
6. 计数和汇总题：逐条列出证据中符合条件的独立事件（不同日期的同类事件分别计数），再给出结论。
7. 知识更新与冲突：以提问时间当时有效的记录为准；标为“已更正”“过去”“已撤回”的不能当作现在的事实；同一事项有矛盾时说明并以最新的实际发生记录为准。
8. 询问某个计划或安排是否合适时，检查同一时间是否已有其他日程、计划或周期性安排，有冲突就指出冲突的记录。
9. 因果题要依据证据中写明的经过，不能只因两件事先后发生就认定因果。

只输出一个 JSON 对象，不要输出其他内容。字段按下面的顺序写，答案写在最前面：
- answer：给用户的完整回答。列举、排序、汇总题在这里逐条写，每条一行，同一件事只写一次。
- answerable：证据能否回答问题，true 或 false。
- abstain：是否无法回答，true 或 false。
- evidence：直接支持答案的证据编号（如 E03），只列真正用到的，不要把全部编号都列出来。
- quotes：最关键的一两句证据原文，可以为空数组，同一句不要重复。
不能回答时：answer 写“根据现有记录，无法确定。”，answerable 为 false，abstain 为 true，evidence 与 quotes 为空数组。"""

# Qwen3 non-thinking sampling (the model card advises against greedy decoding: on 16K-31K prompts greedy looped in the Phase 3 smoke),
# a mild repetition penalty, a fixed seed (the backend seeds per prompt, so B and C - identical prompts - draw the same seed)
FACT_GENERATION = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.05,
                   "max_new_tokens": 512, "enable_thinking": False, "seed": 20261112}
FACT_GENERATION_LONG = {**FACT_GENERATION, "max_new_tokens": 1024}
STYLE_GENERATION = {"do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0,
                    "max_new_tokens": 512, "enable_thinking": False, "seed": 20261112}
# the D4 training-time sampling (legacy chain); kept only for configuration A
D4_PERSONA_GENERATION = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0,
                         "max_new_tokens": 512, "enable_thinking": False, "seed": 20261112}

READER_CONFIGS: Dict[str, dict] = {
    "A_d4_persona": {"prompt": "d4_persona", "adapter_enabled": True, "generation": "D4_PERSONA"},
    "B_base_fact": {"prompt": "fact", "adapter_enabled": False, "generation": "FACT"},
    "C_d4_fact": {"prompt": "fact", "adapter_enabled": True, "generation": "FACT"},
    "D_base_d4_style": {"prompt": "fact", "adapter_enabled": False, "generation": "FACT",
                        "style": {"prompt": "d4_style", "adapter_enabled": True, "generation": "STYLE"}},
}


def generation_for(name: str, pack: FactPack) -> dict:
    long_ = pack.output_tokens > 512
    if name == "FACT":
        return dict(FACT_GENERATION_LONG if long_ else FACT_GENERATION)
    if name == "STYLE":
        return dict(STYLE_GENERATION)
    if name == "D4_PERSONA":
        return {**D4_PERSONA_GENERATION, "max_new_tokens": 1024 if long_ else 512}
    raise KeyError(name)


def fact_user(pack: FactPack) -> str:
    parts = [render_evidence(pack)]
    notes = render_notes(pack)
    if notes:
        parts.append(notes)
    parts.append(f"【问题】（提问时间：{pack.ask_time[:10]}）{pack.body}")
    return "\n\n".join(parts)


def fact_messages(pack: FactPack) -> List[dict]:
    return [{"role": "system", "content": FACT_SYSTEM}, {"role": "user", "content": fact_user(pack)}]


# ---------------------------------------------------------------- configuration A: the D4 renderer fed with EF1 items
def d4_persona_messages(pack: FactPack, *, profile=None, examples: Sequence[str] = (), owner_frame: bool = True) -> List[dict]:
    """The pinned D4 grounded-persona renderer with ALL pack items as evidence (status CURRENT, source = the record, date = event time).
    ``owner_frame`` adds one rule that the asker is the persona themself (the LifeBench setting); the renderer text is otherwise unchanged."""
    import stillhere.session.d4_renderer as dr
    ev = []
    for n, i in enumerate(pack.items, 1):
        text = i.text.replace("\n", " ／ ")
        if i.memories:
            text += "（" + "；".join(f"{m.statement}：{m.status}" for m in i.memories) + "）"
        e = dr.Evidence(fact_id=f"F{n}", subject="本人", subject_type="self", predicate="record", value=text, evidence_text=text, status="CURRENT",
                        observed_at=(i.event_time or "")[:10] or None, source_ref=i.unit_key, confidence="high", disclosure_policy="ALLOWED")
        e.validate()
        ev.append(e)
    rules = ("对方就是记忆的主人本人，正在问自己过去的事；“我”指对方本人，直接用证据回答，不要反问。",) if owner_frame else ()
    return dr.render_messages(profile=profile, examples=list(examples), evidence=ev, unknown=[], now=pack.ask_time[:10],
                              conversation=[{"speaker": "partner", "text": f"（提问时间：{pack.ask_time[:10]}）{pack.body}"}], extra_rules=rules)


# ---------------------------------------------------------------- configuration D: style stage
def d4_style_messages(fact_answer: str, pack: FactPack, cited: Sequence[str], *, profile=None, examples: Sequence[str] = ()) -> List[dict]:
    """The D4 renderer restates a checked fact answer: the answer is the only evidence item (plus the cited records as context)."""
    import stillhere.session.d4_renderer as dr
    ev = [dr.Evidence(fact_id="F1", subject="本人", subject_type="self", predicate="answer", value=fact_answer, evidence_text=fact_answer,
                      status="CURRENT", source_ref="fact_reader", confidence="high", disclosure_policy="ALLOWED")]
    by = {i.eid: i for i in pack.items}
    for n, eid in enumerate([e for e in cited if e in by][:8], 2):
        t = by[eid].text.replace("\n", " ／ ")
        ev.append(dr.Evidence(fact_id=f"F{n}", subject="本人", subject_type="self", predicate="record", value=t, evidence_text=t, status="CURRENT",
                              observed_at=(by[eid].event_time or "")[:10] or None, source_ref=by[eid].unit_key, confidence="high", disclosure_policy="ALLOWED"))
    for e in ev:
        e.validate()
    rules = ("F1 是已经核对过的答案：用这个人的说话方式把它说出来，不得增加、删除或改动其中的数字、日期、人名、地点和结论。",)
    return dr.render_messages(profile=profile, examples=list(examples), evidence=ev, unknown=[], now=pack.ask_time[:10],
                              conversation=[{"speaker": "partner", "text": pack.body}], extra_rules=rules)


def third_party_redactor(resolver, persona_id: str):
    """A ``fact_pack.build(redactor=...)`` for any audience other than the owner: the disclosure policy decides per item (fail closed)."""
    from stillhere.session import disclosure as ds

    def redact(text: str, at):
        dec = resolver.decide(persona_id, "persona", "", text, text, at)
        return None if dec.policy == ds.ALLOWED else ds.redacted_text("本人", dec)
    return redact


# ---------------------------------------------------------------- output parsing and checks
@dataclass
class FactReading:
    ok: bool                     # parsed as the JSON object
    answer: str
    abstain: bool
    answerable: Optional[bool]
    evidence: List[str]
    quotes: List[str]
    needed: str
    raw: str
    error: Optional[str] = None


_JSON = re.compile(r"\{.*\}", re.S)


_ANSWER_FIELD = re.compile(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)("?)', re.S)


def _salvage(text: str) -> Optional[tuple]:
    """The ``answer`` string of a JSON object cut off later (e.g. at the token limit): (answer, complete?) or None."""
    m = _ANSWER_FIELD.search(text)
    if not m:
        return None
    body = m.group(1)
    if body.endswith("\\"):                       # cut inside an escape sequence
        body = body[:-1]
    try:
        ans = json.loads('"' + body + '"')
    except json.JSONDecodeError:
        ans = body.replace("\\n", "\n").replace('\\"', '"')
    return ans.strip(), bool(m.group(2))


def parse(raw: str) -> FactReading:
    text = (raw or "").strip()
    text = re.sub(r"^<think>.*?</think>\s*", "", text, flags=re.S)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    m = _JSON.search(text)
    if m:
        try:
            d = json.loads(m.group(0))
            ev = [str(x) for x in (d.get("evidence") or d.get("citations") or []) if x is not None]
            quotes = [str(x) for x in (d.get("quotes") or []) if x is not None]
            return FactReading(True, str(d.get("answer") or "").strip(), bool(d.get("abstain")), d.get("answerable"), ev, quotes,
                               str(d.get("needed") or ""), raw)
        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            err = f"{type(e).__name__}"
        else:
            err = None
    else:
        err = "no JSON object"
    got = _salvage(text)                                  # truncated JSON: keep the answer if it was written
    if got and got[0]:
        ans, complete = got
        ab = bool(re.search(r'"abstain"\s*:\s*true', text)) or bool(re.search(r"无法确定|不知道|没有记录|记不清", ans))
        return FactReading(False, ans, ab, None, re.findall(r'"(E\d{2,3})"', text), [], "", raw,
                           f"{err}; answer salvaged ({'complete' if complete else 'cut off'})")
    return FactReading(False, text, bool(re.search(r"无法确定|不知道|没有记录|记不清", text)), None, [], [], "", raw, err)


_NUM = re.compile(r"\d+(?:\.\d+)?")
_QUOTE = re.compile(r"[“\"『「](.{2,30}?)[”\"』」]")


def citation_check(reading: FactReading, pack: FactPack) -> dict:
    eids = {i.eid for i in pack.items}
    unknown = [e for e in reading.evidence if e not in eids]
    text = pack.evidence_text()
    quotes_missing = [q for q in reading.quotes if q and q.strip() and q.strip() not in text]
    return {"cited": len(reading.evidence), "unknown_ids": unknown, "quotes": len(reading.quotes), "quotes_not_in_evidence": len(quotes_missing)}


def support_check(answer: str, pack: FactPack) -> dict:
    """Numbers and quoted phrases in the answer that appear NOWHERE in the evidence scope (all pack items, their memories, the notes,
    the question and the ask date). The whole pack is the scope - not the first 5 items (the v3 guard rejected correct numbers that way).
    Returns the unsupported terms; the caller decides what to do (``guard_action``)."""
    # the item headers the reader is SHOWN ("[E01] 2026-03-18T02:29｜…", render_item) belong to the scope too: without them a correct
    # date / clock time read from a header was rejected as unsupported (FAST M1 GROUNDEDNESS_FAIL cases with the gold evidence retrieved)
    headers = "\n".join((i.event_time or "")[:16] for i in pack.items)
    scope = pack.evidence_text() + "\n" + headers + "\n" + pack.body + "\n" + pack.ask_time
    digits = set(_NUM.findall(scope))
    unsupported = [n for n in _NUM.findall(answer or "") if n not in digits and len(n) >= 2]
    quoted = [q for q in _QUOTE.findall(answer or "") if q not in scope]
    return {"unsupported_numbers": sorted(set(unsupported)), "unsupported_quotes": quoted,
            "supported": not unsupported and not quoted}


def fact_preserved(fact: str, styled: str) -> dict:
    """Every number of the fact answer must survive the style stage (dates and amounts are digits); returns the missing ones."""
    missing = [n for n in dict.fromkeys(_NUM.findall(fact or "")) if n not in (styled or "")]
    return {"preserved": not missing, "missing_numbers": missing}


def finalize(reading: FactReading, pack: FactPack, *, guard_action: str = "annotate") -> dict:
    """The answer and its audit. ``guard_action``: "annotate" (keep the answer, record the problems - the Phase 3 default, so the guard
    cannot raise accuracy by refusing) or "abstain_unsupported" (replace an answer with unsupported specifics and no citation by an
    abstention)."""
    sup = support_check(reading.answer, pack)
    cit = citation_check(reading, pack)
    answer = reading.answer or ("根据现有记录，无法确定。" if reading.abstain else "")
    action = "kept"
    if guard_action == "abstain_unsupported" and not reading.abstain and not sup["supported"] and not reading.evidence:
        answer, action = "根据现有记录，无法确定。", "abstained_unsupported"
    return {"answer": answer, "action": action, "parsed": reading.ok, "abstain": reading.abstain, "support": sup, "citations": cit,
            "reader_version": READER_VERSION}
