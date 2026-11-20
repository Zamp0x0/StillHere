"""StillHere Style Profile Schema v1 (``stillhere.style.profile/1.0``).

A deterministic projection of the T2-D3 Style Profiler v1 output (``style-profiler-v1``: per-message feature rates + evidence-backed traits).
Real authors keep the profile the validated T2-D3 builder computed from their own earlier messages; synthetic personas are run through the
same vendored profiler on their synthetic history, so both kinds of persona reach the prompt through one code path.

Only HOW a person writes is represented. Facts (place, job, family, events) are never profile fields: they belong to Memory Evidence.
Traits the profiler marks ``unconfirmed_hypothesis`` (seen once) are kept for transparency and never rendered.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Mapping, Optional

SCHEMA_VERSION = "stillhere.style.profile/1.0"
EXTRACTION_VERSION = "t2d4-profile-v1 over style-profiler-v1"

PARTICLES = {"particle_ya": "呀", "particle_ne": "呢", "particle_ba": "吧", "particle_la": "啦", "particle_ma": "嘛", "particle_o": "哦",
             "particle_a": "啊", "particle_en": "嗯", "particle_bei": "呗"}
PARTICLE_MIN_PER100 = 1.5
REGISTER_TRAITS = {"slang": "常用网络流行语", "northern": "带北方口语", "cantonese": "用粤语口语", "formal": "用词正式、礼貌用语多",
                   "laugh": "常用拉长的笑声（哈哈哈）", "emoji": "常用表情符号", "kaomoji": "常用颜文字"}
INTERACTION_TRAITS = {"hedge": "说话委婉，常用“可能、要不、好像”", "rhetorical": "爱用反问和揶揄", "care": "常叮嘱和关心对方",
                      "joke": "爱开玩笑、玩梗", "enumeration": "条理分明，常分点说明", "parenthesis": "常用括号补充说明", "metaphor": "常用比喻"}
PUNCT_TRAITS = {"exclaim": "常用感叹号", "double_exclaim": "常用连续的感叹号", "ellipsis": "常用省略号", "tilde": "常用波浪号",
                "space_pause": "用空格代替逗号断句"}

FIELDS = ("schema_version", "persona_id", "persona_kind", "evidence_window", "sample_count", "observation_span", "response_length_distribution",
          "sentence_structure", "punctuation_habits", "colloquialism", "common_particles", "question_frequency", "multi_bubble_behavior",
          "language_mixing", "tone_by_context", "interaction_style", "confidence", "source_hash", "extraction_version")


def _band(x: float, cuts, names) -> str:
    for c, n in zip(cuts, names):
        if x <= c:
            return n
    return names[-1]


def _traits(p: Mapping) -> Dict[str, dict]:
    return {t["trait"]: t for t in p.get("traits", [])}


def _confirmed(t: Optional[dict]) -> bool:
    return bool(t) and t["type"] != "unconfirmed_hypothesis"


def build(profiler_out: Mapping, *, persona_id: str, persona_kind: str, history_before: Optional[str] = None) -> dict:
    """``profiler_out``: the ``style-profiler-v1`` dict. ``history_before``: the target's time key (profile evidence strictly precedes it)."""
    if persona_kind not in ("REAL_AUTHOR", "SYNTHETIC_PERSONA"):
        raise ValueError(persona_kind)
    st = profiler_out.get("stats") or {}
    f = st.get("features") or {}
    tr = _traits(profiler_out)
    n = int(profiler_out.get("message_count") or 0)
    stypes = sorted({s for t in profiler_out.get("traits", []) for s in t.get("scenario_types", [])})
    cpb = (st.get("chars_per_bubble") or {}).get("mean", 0.0)
    bpm = (st.get("bubbles_per_message") or {}).get("mean", 1.0)
    end = "no_end_punct" if _confirmed(tr.get("no_end_punct")) else "full_punct" if _confirmed(tr.get("full_punct")) else "mixed"
    parts = sorted(((k, round(f.get(k, 0.0), 3)) for k in PARTICLES if f.get(k, 0.0) >= PARTICLE_MIN_PER100), key=lambda kv: -kv[1])
    q = f.get("question", 0.0)
    lat = f.get("latin_words", 0.0)
    ctx_dep = sorted(t["trait"] for t in profiler_out.get("traits", []) if t["type"] != "unconfirmed_hypothesis" and t["stability"] == "context_dependent")
    conf = "high" if n >= 20 and len(stypes) >= 2 else "medium" if n >= 8 else "low"
    out = {
        "schema_version": SCHEMA_VERSION,
        "persona_id": persona_id,
        "persona_kind": persona_kind,
        "evidence_window": {"source_message_count": n, "source_ids_sha256": hashlib.sha256(json.dumps(sorted(profiler_out.get("source_ids", []))).encode()).hexdigest(),
                            "history_before": history_before},
        "sample_count": n,
        "observation_span": {"scenario_types": len(stypes)},
        "response_length_distribution": {"chars_per_bubble_mean": cpb, "chars_per_bubble_sd": (st.get("chars_per_bubble") or {}).get("sd", 0.0),
                                         "band": _band(cpb, (9, 16, 28), ("very_short", "short", "medium", "long"))},
        "sentence_structure": {"comma_per100": round(f.get("comma", 0.0), 3), "period_per100": round(f.get("period", 0.0), 3),
                               "enumeration": _confirmed(tr.get("enumeration")), "parenthesis": _confirmed(tr.get("parenthesis"))},
        "punctuation_habits": {"sentence_end": end, **{k: _confirmed(tr.get(k)) for k in PUNCT_TRAITS}},
        "colloquialism": {k: _confirmed(tr.get(k)) for k in REGISTER_TRAITS},
        "common_particles": [{"particle": PARTICLES[k], "per100_chars": v} for k, v in parts[:3]],
        "question_frequency": {"per100_chars": round(q, 3), "band": "frequent" if _confirmed(tr.get("question")) else "rare" if q < 0.5 else "occasional"},
        "multi_bubble_behavior": {"bubbles_per_message_mean": bpm, "max": (st.get("bubbles_per_message") or {}).get("max", 1),
                                  "band": "multi" if bpm >= 2.2 else "single" if bpm <= 1.3 else "sometimes"},
        "language_mixing": {"latin_words_per100": round(lat, 3), "band": "frequent" if _confirmed(tr.get("latin")) else "none" if lat < 0.3 else "occasional"},
        "tone_by_context": {"context_dependent_traits": ctx_dep},
        "interaction_style": {k: _confirmed(tr.get(k)) for k in INTERACTION_TRAITS},
        "confidence": conf,
        "source_hash": profiler_out.get("profile_sha256", ""),
        "extraction_version": EXTRACTION_VERSION,
    }
    out["profile_v1_sha256"] = hashlib.sha256(json.dumps(out, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return out


def render(p: Mapping) -> str:
    """Prompt text of a Schema v1 profile. Deterministic; only confirmed traits; never facts."""
    if not p or not p.get("sample_count"):
        return "- （没有足够的历史消息总结说话方式：用自然、简洁的口吻）"
    L: List[str] = []
    ld = p["response_length_distribution"]
    mb = p["multi_bubble_behavior"]
    size = {"very_short": "很短", "short": "偏短", "medium": "中等", "long": "较长、句子完整"}[ld["band"]]
    bub = {"single": "通常一次只发一条消息", "sometimes": "有时会拆成两三条发", "multi": f"常把话拆成几条消息发（平均 {mb['bubbles_per_message_mean']:.1f} 条）"}[mb["band"]]
    L.append(f"- 长度：每条消息{size}（平均约 {ld['chars_per_bubble_mean']:.0f} 字），{bub}")
    ph = p["punctuation_habits"]
    end = {"no_end_punct": "句末通常不加标点", "full_punct": "标点规范，句末常用句号", "mixed": "句末标点不固定"}[ph["sentence_end"]]
    extra = [PUNCT_TRAITS[k] for k in PUNCT_TRAITS if ph.get(k)]
    L.append("- 标点：" + "；".join([end] + extra))
    if p["common_particles"]:
        L.append("- 语气词：常用" + "、".join(f"“{x['particle']}”" for x in p["common_particles"]))
    else:
        L.append("- 语气词：用得不多")
    qf = p["question_frequency"]["band"]
    L.append("- 问句：" + {"frequent": "常用问句或反问", "occasional": "偶尔用问句", "rare": "很少用问句"}[qf])
    lm = p["language_mixing"]["band"]
    L.append("- 中英混用：" + {"frequent": "常在中文里夹英文词", "occasional": "偶尔夹英文词", "none": "基本不夹英文"}[lm])
    reg = [REGISTER_TRAITS[k] for k, v in p["colloquialism"].items() if v]
    if reg:
        L.append("- 口语：" + "；".join(reg))
    inter = [INTERACTION_TRAITS[k] for k, v in p["interaction_style"].items() if v]
    if p["sentence_structure"]["enumeration"] and INTERACTION_TRAITS["enumeration"] not in inter:
        inter.append(INTERACTION_TRAITS["enumeration"])
    if inter:
        L.append("- 表达习惯：" + "；".join(inter))
    cd = p["tone_by_context"]["context_dependent_traits"]
    if cd:
        L.append("- 以上部分特点只在某些场合出现，不要每句都用")
    L.append(f"- （依据此人此前 {p['sample_count']} 条消息，置信度：{ {'high': '高', 'medium': '中', 'low': '低'}[p['confidence']] }）")
    return "\n".join(L)


def check_no_facts(p: Mapping) -> List[str]:
    """Schema guard: the profile must not carry factual slots."""
    banned = ("city", "home", "job", "occupation", "family", "spouse", "address", "phone", "school", "company", "age", "birthday")
    bad = [k for k in p if any(b in k.split("_") for b in banned)]  # whole words: "language_mixing" is not "age"
    missing = [k for k in FIELDS if k not in p]
    return [f"factual_field:{k}" for k in bad] + [f"missing_field:{k}" for k in missing]


def json_schema() -> dict:
    """Machine-readable field list (profile_schema_v1.json)."""
    return {"schema_version": SCHEMA_VERSION, "extraction_version": EXTRACTION_VERSION, "fields": {
        "schema_version": "constant stillhere.style.profile/1.0",
        "persona_id": "persona key (real author key or SYN-… id)",
        "persona_kind": "REAL_AUTHOR | SYNTHETIC_PERSONA",
        "evidence_window": "source_message_count, sha256 of the sorted source message ids, history_before (target time key; all sources precede it)",
        "sample_count": "messages the profiler read",
        "observation_span": "number of distinct scenario types (days / topics) the sources cover",
        "response_length_distribution": "chars_per_bubble mean / sd and band very_short<=9<short<=16<medium<=28<long",
        "sentence_structure": "comma and period per 100 chars; enumeration / parenthesis traits",
        "punctuation_habits": "sentence_end no_end_punct|full_punct|mixed + exclaim, double_exclaim, ellipsis, tilde, space_pause traits",
        "colloquialism": "slang, northern, cantonese, formal, laugh, emoji, kaomoji traits",
        "common_particles": f"up to 3 particles with rate >= {PARTICLE_MIN_PER100} per 100 chars",
        "question_frequency": "question marks per 100 chars; band frequent (profiler trait) | occasional | rare (<0.5)",
        "multi_bubble_behavior": "bubbles per message mean / max; band single<=1.3<sometimes<2.2<=multi",
        "language_mixing": "latin words per 100 chars; band frequent (profiler trait) | occasional | none (<0.3)",
        "tone_by_context": "traits the profiler saw in one scenario type only (context_dependent)",
        "interaction_style": "hedge, rhetorical, care, joke, enumeration, parenthesis, metaphor traits",
        "confidence": "high (>=20 msgs and >=2 scenario types) | medium (>=8) | low",
        "source_hash": "profile_sha256 of the style-profiler-v1 output it was projected from",
        "extraction_version": EXTRACTION_VERSION,
        "profile_v1_sha256": "sha256 of this object without the field itself"},
        "rules": ["deterministic: same profiler output -> same profile", "traits of type unconfirmed_hypothesis are never rendered",
                  "no factual fields (place, job, family, events, contact) - those are Memory Evidence"]}
