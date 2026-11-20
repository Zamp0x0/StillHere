"""One reply, end to end, on top of an existing ``MemoryAPI`` (the memory the Writer filled):

    question (last message of the other side) + ask time
      -> M3.1 reflection over the B2 first hop (read-only, known_at = ask time)
      -> [optional reader: ``answerer(question, ask_time, evidence_text) -> {"answer", "abstain"}``, checked by the M3.1 claim check]
      -> grounded content (``stillhere.style`` grounded schema: current / past / corrected facts with the old value, unknown items)
      -> style prompt contract (``stillhere.style.prompt/2.0``, unchanged) with the persona's own history
      -> ``generate(messages, config)``: base model or adapter, local or remote: one interface
      -> runtime safety guards (repetition / must-not / claims / length policy, bounded retry, fallback)

Every stage leaves ids and decisions in the trace, never another persona's text. With every flag off the chain is the plain one (neutral
answer of the grounded content, no generation) so it can run on CPU without a model.

Runtime fields added here are versioned as ``RUNTIME_PIPELINE_VERSION``; the prompt schema is not changed. A future adapter (e.g. a later
training round) plugs in through ``generate`` and ``backend`` only.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from stillhere.memory import reflection as r2
from stillhere.memory import reflection_m31 as m31
from stillhere.style import contract as sc
from stillhere.style import realization as rz
from stillhere.style import safety as safety
import stillhere.runtime.loop_guard as lg

RUNTIME_PIPELINE_VERSION = "stillhere.runtime.pipeline/1.0"
MAX_FACTS = 6


@dataclass
class ReplyTrace:
    version: str
    persona_id: str
    ask_time: str
    question: str
    route: str
    evidence_units: List[str]
    evidence_ids: List[str]
    reflection_notes: List[str]
    reader: Optional[dict]
    grounded: dict
    contract: Optional[dict]
    safety: Optional[dict]
    backend: str
    text: str
    status: str
    latency_ms: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def grounded_from_evidence(m: "m31.M31", ask_time: datetime, answer: Optional[str] = None, max_facts: int = MAX_FACTS) -> dict:
    """M3.1 evidence units -> the style layer's grounded content. Nothing is invented: every statement is a stored statement (or the
    checked reader answer, marked as such).

    * a fact slot's current record -> ``current``; if an earlier record of the same slot was corrected, the current one is ``corrected`` with
      the old value in ``replaces`` (the safety guard forbids that old value as a current claim);
    * a record no longer valid at the asking time -> ``past``; retracted records are dropped;
    * an event -> one ``current`` statement: description, status and date (another person's event says whose it is);
    * no evidence -> one unknown item.
    """
    by_id = {r.memory_id: r for r in m.evidence.records}
    facts: List[dict] = []
    if answer:
        facts.append({"statement": answer, "role": "current", "source": "reader(M3.1-checked)"})
    slots: Dict[tuple, list] = {}
    for key, ids in m.evidence.units:
        rs = [by_id[i] for i in ids if i in by_id]
        if not rs:
            continue
        if key.startswith("event:"):
            line = m31._event_line(key, rs, ask_time)
            facts.append({"statement": line, "role": "current", "source": key})
            continue
        r = rs[0]
        slots.setdefault((r.subject, r.attribute), []).append(r)
    for (_subject, _attr), rs in slots.items():
        # an old value counts as CORRECTED (known to be wrong) only if it is not also a valid past value of the slot: a move closes the old
        # address (a past fact, still true for its time) and must not be turned into a forbidden value
        past_vals = {r.value for r in rs if r2._status(r, ask_time).startswith("过去")}
        old = [r for r in rs if getattr(r.status, "value", r.status) == "superseded" and r2._status(r, ask_time).startswith("已更正") and r.value not in past_vals]
        for r in rs:
            st = r2._status(r, ask_time)
            if st == "当前":
                f = {"statement": r.statement, "role": "current", "source": r.memory_id}
                if old:
                    f.update(role="corrected", replaces=old[-1].statement)
                facts.append(f)
            elif st.startswith("过去"):
                facts.append({"statement": r.statement, "role": "past", "source": r.memory_id, "time": r2._fmt(r.valid_from) + "~" + r2._fmt(r.valid_to)})
    facts = facts[:max_facts]
    unknown = ["对方问的这件事"] if "no_evidence" in m.notes and not answer else []
    clean = [{k: v for k, v in f.items() if v is not None} for f in facts]
    return {"status": "facts" if clean or unknown else "no_memory", "facts": clean, "unknown": unknown}


# ---------------------------------------------------------------- grounding v2 (Batch 2.1 post-GPU fix)
GROUNDING_V2_FLAG = "GROUNDING_V2_ENABLED"
GROUNDING_V2_VERSION = "stillhere.runtime.grounding/2"
_STATUS_PHRASE = {"PLANNED": "计划中", "SCHEDULED": "已经约好", "IN_PROGRESS": "正在进行", "COMPLETED": "已经完成", "CANCELLED": "已经取消"}
_CHANGE = re.compile(r"改到|改成|推迟|提前|取消|延期")
_VERB_START = re.compile(r"^(去|来|回|参加|搬|拔|看|跑|报名|带|买|做|考|学|爬|游|拍|洗|吃|见|开|请|约|陪|帮|办|练|跳|骑|飞|坐|住|写|读|搞|装|修|换)")
# runtime question -> attribute aliases (post-retest; the B2 retrieval aliases are unchanged): question forms the first retest missed
RUNTIME_ALIASES = ((r"教什么|教哪|什么老师|当什么|干哪行|哪一行|是做什么|做什么的|什么职业|什么工作", ("job", "works_at")),
                   (r"在哪(家|个)?(公司|单位|上班|工作|实习)|什么公司|哪家公司|哪个单位", ("works_at", "job")),
                   (r"住(在)?哪|哪个区|家在哪|住什么地方", ("lives_in",)),
                   (r"(读|学)(的)?什么|什么专业|哪个学校|哪所(学校|大学|中学|小学)|几年级|在哪(上学|读书)", ("school",)),
                   (r"养(了)?(什么|猫|狗|宠物)|宠物", ("pet",)),
                   (r"(爸|妈|哥|姐|弟|妹|儿子|女儿|老公|老婆|丈夫|妻子|朋友|同事)\S{0,3}(叫什么|是谁)", ("relationship",)))
GENERIC_BIGRAMS = {"喜欢", "知道", "觉得", "安排", "时候", "平时", "现在", "一下", "有没", "最近", "以前", "之前", "今天", "明天", "这周", "下周", "上周"}
_STOP = re.compile(r"[我你他她的了吗呢吧啊是在有什么哪些哪个哪里么怎么样没过还要会想几？?，,。！!\s]")


def grounding_v2_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    import os
    return (env if env is not None else os.environ).get(GROUNDING_V2_FLAG, "").strip().lower() in ("1", "true", "yes", "on")


def _md(v: Optional[str]) -> str:
    m_ = re.match(r"(\d{4})-(\d{2})(?:-(\d{2}))?", v or "")
    if not m_:
        return ""
    return f"{int(m_.group(2))}月{int(m_.group(3))}日" if m_.group(3) else f"{int(m_.group(2))}月"


def natural_event_statement(key: str, rs, ask_time: datetime) -> str:
    """One plain sentence for an event unit (no internal labels, ids or record times): who, what, current status, the relevant date."""
    rs = sorted(rs, key=lambda r: (r.recorded_at, r.memory_id))
    live = [r for r in rs if r2._status(r, ask_time) == "当前"] or rs
    descs = [r.value for r in rs if r.attribute == "event.description"] or ["一件事"]
    plain = [d for d in descs if not _CHANGE.search(d)] or descs
    desc = max(plain, key=len)                       # the most informative description that is not a change note ("实习面试", not "面试")
    actor = next((r.value for r in reversed(rs) if r.attribute == "event.actor"), "self")
    who = "本人" if actor in ("self", "persona", "") else actor.split(":", 1)[-1]
    status = next((r.value for r in reversed(live) if r.attribute == "event.status"), None)
    planned = next((r.value for r in reversed(live) if r.attribute == "event.planned_time"), None)
    done = next((r.value for r in reversed(live) if r.attribute == "event.time"), None)
    # plain Chinese (post-retest): a description that starts with a verb reads as an action ("去重庆参加朋友婚礼"), otherwise as a thing
    # ("实习面试"); no quotation marks, no "的…计划中" constructions
    act = bool(_VERB_START.match(desc))
    when_p = f"（{_md(planned)}）" if _md(planned) else ""
    when_d = f"（{_md(done)}）" if _md(done) else ""
    if status == "COMPLETED":
        return (f"{who}已经{desc}了" if act else f"{who}的{desc}已经结束了") + when_d
    if status == "CANCELLED":
        return (f"{who}原本打算{desc}，后来取消了" if act else f"{who}的{desc}取消了") + when_p
    if status == "SCHEDULED":
        return (f"{who}已经约好{desc}" if act else f"{who}已经约好了{desc}") + when_p
    if status == "IN_PROGRESS":
        return f"{who}正在{desc}" if act else f"{who}的{desc}正在进行"
    if status == "PLANNED":
        return (f"{who}打算{desc}" if act else f"{who}有{desc}的安排") + when_p
    return (f"{who}提过{desc}" if act else f"{who}提过{desc}") + "，没说后来怎么样"


def question_core(question: str) -> str:
    """The asked thing, for an unknown item: "我在哪家公司实习？" -> "在哪家公司实习"."""
    q = re.sub(r"[？?。！!\s]+$", "", question or "").strip()
    q = re.sub(r"^(我|请问|那)", "", q)
    q = re.sub(r"(吗|呢|吧|了吗|没有)$", "", q)
    return q or "对方问的这件事"


def relevant(statement: str, question: str) -> bool:
    """Lexical relevance: the statement (with every stored description of an event) shares a non-generic 2-character piece with the
    question's content words."""
    core = _STOP.sub("", question or "")
    st = statement or ""
    return any(core[i:i + 2] in st and core[i:i + 2] not in GENERIC_BIGRAMS for i in range(max(0, len(core) - 1)))


def grounded_from_evidence_v2(m: "m31.M31", ask_time: datetime, question: str, answer: Optional[str] = None, reader_abstained: Optional[bool] = None,
                              max_facts: int = 3) -> dict:
    """Grounding v2 (fix of the first L4 run): the style model is never handed unrelated memory as if it answered the question.

    * natural statements only (no internal labels, record ids or record times; provenance stays in ``provenance``, not in the prompt);
    * the reader answered (and passed the M3.1 claim check): the answer first + up to 2 RELATED units -> ``answer_status`` = answered;
    * the reader found no answer: only units related to the question (partial knowledge, e.g. "参加过实习面试") + an explicit unknown item for
      the asked thing -> ``partial`` (some related facts) or ``unknown`` (none);
    * no reader: related units (or all when none is related), unknown only when there is no evidence.
    Corrected slots keep the old value in ``replaces`` (the must-not guard forbids it as a current claim)."""
    v1 = grounded_from_evidence(m, ask_time, None, max_facts=50)
    by_id = {r.memory_id: r for r in m.evidence.records}
    natural, match = {}, {}
    for key, ids in m.evidence.units:
        if key.startswith("event:"):
            rs = [by_id[i] for i in ids if i in by_id]
            if rs:
                natural[key] = natural_event_statement(key, rs, ask_time)
                match[key] = natural[key] + " " + " ".join(r.value or "" for r in rs if r.attribute in ("event.description", "event.activity"))
    # a single-valued slot with two CURRENT values is an unresolved conflict: both become "unverified" (the prompt then says 待确认 and the
    # fallback says there are two versions), never two current facts
    single = ("lives_in", "works_at", "job", "school", "age", "birthday")
    cur = {}
    for r in m.evidence.records:
        if r.attribute in single and not r.subject.startswith("event:") and r2._status(r, ask_time) == "当前":
            cur.setdefault((r.subject, r.attribute), set()).add(r.memory_id)
    conflicted = {mid for ids in cur.values() if len(ids) > 1 for mid in ids}
    facts, prov = [], []
    for f in v1["facts"]:
        src = f.get("source", "")
        g = {k: v for k, v in f.items() if k != "source"}
        if src in conflicted and g.get("role") == "current":
            g["role"] = "unverified"
        if src in natural:
            g["statement"] = natural[src]
        prov.append(src)
        facts.append((g, match.get(src, g["statement"])))
    # relevance: lexical, or the fact's stored attribute is one B2's question aliases point to ("住哪个区" -> lives_in)
    from stillhere.memory import retrieval_b2 as b2
    attrs = set(b2.attributes_for(question or ""))
    for pat, names in RUNTIME_ALIASES:
        if re.search(pat, question or ""):
            attrs |= set(names)
    if b2.people_in(question or "", m.evidence.records):
        attrs = set()                                                                       # not for a question about someone else
    src_attr = {src: by_id[src].attribute for src in prov if src in by_id}
    rel = [g for (g, mt), src in zip(facts, prov) if relevant(mt, question) or (src_attr.get(src) in attrs and src_attr.get(src) != "event")]
    facts = [g for g, _mt in facts]
    # a corrected slot always travels (at most one): its old value in ``replaces`` is what keeps the must-not guard on it
    corr = [g for g in facts if g.get("role") == "corrected" and g not in rel][:1]
    rel = corr + rel
    unknown = []
    if answer:
        chosen = [{"statement": answer, "role": "current"}] + rel[:2]
        status = "answered"
    elif reader_abstained:
        chosen = rel[:2]
        unknown = [question_core(question)]
        status = "partial" if chosen else "unknown"
    else:
        chosen = (rel or facts)[:max_facts]
        if "no_evidence" in m.notes or not chosen:
            unknown = [question_core(question)]
        status = "unknown" if not chosen else ("partial" if unknown else "facts")
    return {"status": "facts" if chosen or unknown else "no_memory", "facts": chosen, "unknown": unknown, "answer_status": status,
            "grounding_version": GROUNDING_V2_VERSION, "provenance": [p for p in prov if p]}


def ground(api, persona_id: str, question: str, ask_time: datetime, *, answerer: Optional[Callable[[str, datetime, str], Mapping]] = None,
           env: Optional[Mapping[str, str]] = None, retrieval_query: Optional[str] = None, reader_question: Optional[str] = None):
    """Memory -> (optional) reader -> grounded content, for one question. ``retrieval_query`` / ``reader_question`` (Simple Context v1):
    a follow-up's question completed with earlier user words, used for retrieval / grounding relevance and given to the reader; both
    default to the question itself, which is exactly the single-turn behaviour. Returns (m31 result, reader record, answer, grounded, latency)."""
    rq = retrieval_query or question
    lat: Dict[str, float] = {}
    t0 = time.perf_counter()
    m = m31.reflect(api, persona_id, rq, ask_time)
    lat["memory_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    reader = None
    answer = None
    if answerer is not None:
        t1 = time.perf_counter()
        q_reader = reader_question or question
        out = dict(answerer(q_reader, ask_time, m.context + ("\n\n" + m.brief if m.brief else "")))
        chk = m31.verify(out.get("answer", ""), bool(out.get("abstain")), m, q_reader)
        # "text": the reader's own words (also when it abstains) for runtime compatibility rules; a caller must not log it unfiltered
        reader = {"abstain": bool(out.get("abstain")), "claim_check": chk, "text": out.get("answer")}
        if chk["ok"] and not out.get("abstain"):
            answer = out.get("answer")
        lat["reader_ms"] = round((time.perf_counter() - t1) * 1000, 1)
    if grounding_v2_enabled(env):
        abstained = None if answerer is None else (answer is None)
        grounded = grounded_from_evidence_v2(m, ask_time, rq, answer, abstained)
    else:
        grounded = grounded_from_evidence(m, ask_time, answer)
    return m, reader, answer, grounded, lat


def reply(api, persona_id: str, conversation: Sequence[Mapping], ask_time: datetime, *, history: Sequence[sc.StyleMessage] = (),
          generate: Optional[Callable[[List[dict], dict], dict]] = None, backend: str = "neutral",
          answerer: Optional[Callable[[str, datetime, str], Mapping]] = None, env: Optional[Mapping[str, str]] = None,
          generation: Optional[Mapping] = None, max_retries: int = 1, retrieval_query: Optional[str] = None,
          reader_question: Optional[str] = None, loop_guard: Optional[bool] = None) -> ReplyTrace:
    t0 = time.perf_counter()
    question = next((c.get("text", "") for c in reversed(list(conversation)) if c.get("speaker") != "self"), "")
    m, reader, answer, grounded, lat = ground(api, persona_id, question, ask_time, answerer=answerer, env=env, retrieval_query=retrieval_query,
                                             reader_question=reader_question)
    trace = ReplyTrace(RUNTIME_PIPELINE_VERSION, persona_id, ask_time.isoformat(), question, m.route.kind, [k for k, _ in m.evidence.units], m.evidence_ids,
                       list(m.notes), reader, grounded, None, None, backend, "", "")
    if generate is None or backend == "neutral":
        trace.text, trace.status = rz.neutral(grounded), "NEUTRAL"
        lat["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        trace.latency_ms = lat
        return trace
    t2 = time.perf_counter()
    cfg = {**rz.GENERATION, **(generation or {})}
    plan = None
    if safety.flags(env)["LENGTH_POLICY_ENABLED"]:
        # pre-generation persona length condition (Batch 2.1): decided from the persona's own history BEFORE generation; its rule goes into
        # the contract's existing ``constraints`` input (stillhere.style.prompt/2.0 unchanged), its cap into the generation config
        plan = safety.pre_generation_plan([h.text for h in history], grounded=grounded, default_max_new_tokens=int(cfg["max_new_tokens"]))
    payload = sc.build(persona_id, conversation, grounded_content=grounded, history=history, constraints=[plan.constraint] if plan and plan.constraint else ())
    trace.contract = payload.trace()
    lg_on = lg.enabled(env) if loop_guard is None else loop_guard
    lg_records: List[dict] = []
    res = safety.safe_generate(lg.wrap_generate(generate, enabled_=lg_on, records=lg_records), payload.messages, cfg, backend=backend,
                               grounded=grounded, conversation=conversation,
                               history_texts=[h.text for h in history], fallback_text=rz.neutral(grounded), max_retries=max_retries, env=env,
                               length_plan=plan)
    trace.safety = {**res.audit(), "loop_guard": {"enabled": lg_on, "version": lg.VERSION, "calls": lg_records}}
    trace.text, trace.status = res.text, res.status
    lat["style_ms"] = round((time.perf_counter() - t2) * 1000, 1)
    lat["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    trace.latency_ms = lat
    return trace
