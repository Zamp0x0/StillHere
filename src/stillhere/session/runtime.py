"""StillHere Simple Context v1: the session-aware reply (Python runtime API; a web layer can wrap it one to one).

    create_session / send_message / get_session_messages / reply_session

    user message -> session check -> persist (idempotent by request id)
      -> L1 recent window (whole turns, token budget, long messages by marked excerpt)
      -> follow-up query completion (earlier USER words only; a retrieval aid, never a fact)
      -> L2 session history search (same session, same persona; on demand; NOT_FOUND / AMBIGUOUS honest)
      -> L3 existing memory: M3.1 reflection + optional frozen reader + claim check -> grounding v2   (runtime.pipeline.ground)
      -> existing style contract (stillhere.style.prompt/2.0, unchanged) with profile + k examples
      -> total input budget: L2 excerpts, then style examples, then the oldest recent turns give way; system rules, memory facts and the
         current message are never cut (if they alone do not fit: CONTEXT_INSUFFICIENT and the safe fallback, no generation)
      -> existing generate(messages, config) + guard + safe fallback (runtime.safety)
      -> assistant message persisted as GENERATED / FALLBACK / NEUTRAL; the user message's status records the outcome

Three layers stay apart: L1/L2 chat text is never promoted to memory (no Writer call here), assistant text is never fact evidence, style
examples are never facts. Base and adapter receive the same messages (only ``generate`` / ``backend`` differ).
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from stillhere.context.budget import TokenCounter
from stillhere.runtime import pipeline as rp
from stillhere.style import contract as sc
from stillhere.style import realization as rz
from stillhere.style import safety

import stillhere.runtime.loop_guard as lg
import stillhere.session.chat_route as cr
import stillhere.session.d4_evidence as d4e
import stillhere.session.d4_renderer as dr
import stillhere.session.disclosure as ds
import stillhere.session.grounding_compat as gcompat
import stillhere.session.history as hs
import stillhere.session.query as qb
import stillhere.session.tokens as tk
import stillhere.session.window as wn
from .config import CONTEXT_VERSION, D4_COMPATIBLE, DEFAULT, ContextConfig
from .store import LiveMessage, SessionStore

EARLIER_MARK = "〔较早的对话，第{seq}条{gen}〕"
HISTORY_UNKNOWN = "对方提到的较早对话内容（本次会话记录里没有找到）"
LONG_UNKNOWN = "对方问到的长消息里具体是哪一部分（无法确定）"
_SPECIAL = re.compile(r"<\|[^|<>\s]{1,64}\|>")
_RESERVED = ("<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>")


def data_text(text: str) -> str:
    """User / history text enters the prompt as DATA: chat-template control tokens and reserved tags are defused (not removed silently: the
    visible characters change from ``<|x|>`` to ``‹|x|›``), so pasted text can never open a system turn."""
    out = _SPECIAL.sub(lambda m: "‹" + m.group(0)[1:-1] + "›", text)
    for t in _RESERVED:
        out = out.replace(t, t.replace("<", "‹").replace(">", "›"))
    return out


@dataclass
class SessionReply:
    status: str                         # GENERATED | FALLBACK | NEUTRAL | CONTEXT_INSUFFICIENT | FAILED | REPLAYED
    text: Optional[str]
    user_message: LiveMessage
    assistant_message: Optional[LiveMessage]
    messages: List[dict] = field(default_factory=list)      # what the model received (empty when it was not called)
    trace: dict = field(default_factory=dict)


@dataclass
class PreparedReply:
    """A reply prepared up to the model call (``SessionRuntime._prepare``). JSON-serialisable except ``length_plan``."""
    messages: List[dict]
    conversation: List[dict]
    guard_grounded: dict
    guard_terms: List[str]
    cfg_gen: dict
    length_plan: object
    small: bool
    ok: bool
    pre_leak: bool
    history_texts: List[str]
    trace: dict

    def to_json(self) -> dict:
        if self.length_plan is not None:
            raise ValueError("a prepared reply with a length plan is not serialisable (LENGTH_POLICY_ENABLED must be off)")
        return {k: v for k, v in asdict(self).items()}

    @classmethod
    def from_json(cls, d: Mapping) -> "PreparedReply":
        return cls(**{**d, "length_plan": None})


class SessionRuntime:
    def __init__(self, api, *, config: Optional[ContextConfig] = None, count: Optional[Callable[[str], int]] = None, exact_tokens: bool = False,
                 clock: Optional[Callable[[], datetime]] = None, render_mode: str = "prompt2", requester: str = "live:user",
                 persona_kind: str = "SYNTHETIC_PERSONA", tokenizer_json: Optional[str] = None, loop_guard: Optional[bool] = None) -> None:
        if render_mode not in ("prompt2", "d4"):
            raise ValueError(f"render_mode {render_mode!r}")
        self.render_mode = render_mode
        self.cfg = config or (D4_COMPATIBLE if render_mode == "d4" else DEFAULT)
        if count is not None or exact_tokens:
            self.count = TokenCounter(count, exact=exact_tokens)
        elif render_mode == "d4":
            self.count = tk.best_counter(tokenizer_json)          # EXACT with the pinned Qwen3 tokenizer when available, else ESTIMATED
        else:
            self.count = TokenCounter()
        self.token_mode = self.count.mode
        self.store = SessionStore(api, self.cfg, self.count)
        self.api = api
        self.policy = ds.PolicyResolver(api, requester)
        self.persona_kind = persona_kind
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.loop_guard = loop_guard                           # None: the LOOP_GUARD_ENABLED flag decides (default off)

    # ------------------------------------------------------------------ thin session API
    def create_session(self, persona_id: str, session_id: Optional[str] = None) -> str:
        return self.store.create_session(persona_id, self.clock(), session_id)

    def send_message(self, persona_id: str, session_id: str, text: str, request_id: str) -> LiveMessage:
        return self.store.append(persona_id, session_id, "user", text, self.clock(), request_id, status="RECEIVED")

    def get_session_messages(self, persona_id: str, session_id: str) -> List[LiveMessage]:
        return self.store.messages(persona_id, session_id)

    # ------------------------------------------------------------------ prompt assembly (prompt/2.0)
    def _conversation(self, earlier: Sequence[hs.Excerpt], window: wn.Window, current: str) -> List[dict]:
        conv = []
        for e in sorted(earlier, key=lambda x: x.seq):
            mark = EARLIER_MARK.format(seq=e.seq, gen="，AI生成" if e.generated else "")
            conv.append({"speaker": "self" if e.role == "assistant" else "other", "text": mark + data_text(e.text), "generated": e.generated,
                         "message_id": e.message_id, "layer": "L2"})
        for m in window.messages:
            conv.append({"speaker": "self" if m.role == "assistant" else "other", "text": data_text(m.text), "generated": m.generated,
                         "message_id": m.message_id, "layer": "L1"})
        conv.append({"speaker": "other", "text": data_text(current), "generated": False, "layer": "current"})
        return conv

    def _tokens(self, messages: Sequence[Mapping]) -> int:
        return sum(self.count(m["content"]) for m in messages)

    def assemble(self, persona_id: str, grounded: Mapping, history: Sequence[sc.StyleMessage], earlier: List[hs.Excerpt], window: wn.Window,
                 current: str, constraints: Sequence[str], max_new_tokens: int):
        """Fit everything into ``total_input_token_budget - max_new_tokens``. Gives way in this order: L2 excerpts, style examples, the oldest
        recent turns. Never cut: system rules, memory facts, the current message. Returns (payload, conversation, decisions, ok)."""
        limit = self.cfg.total_input_token_budget - max_new_tokens
        earlier, turns, k = list(earlier), list(window.turns), self.cfg.style_examples_k
        decisions: List[str] = []
        while True:
            w = wn.Window(turns, window.tokens, window.token_mode, window.dropped_turns, window.status, window.older_exists)
            conv = self._conversation(earlier, w, current)
            payload = sc.build(persona_id, [{"speaker": c["speaker"], "text": c["text"]} for c in conv], grounded_content=grounded, history=history,
                               k_examples=max(k, 1), use_examples=k > 0, constraints=constraints)
            used = self._tokens(payload.messages)
            if used <= limit:
                return payload, conv, decisions, True, used
            if earlier:
                decisions.append(f"drop L2 excerpt {earlier[-1].message_id}")
                earlier.pop()
            elif k > 0:
                decisions.append(f"style examples {k} -> {k - 1}")
                k -= 1
            elif len(turns) > 1:
                decisions.append(f"drop oldest recent turn {[m.message_id for m in turns[0]]}")
                turns.pop(0)
            elif turns:
                decisions.append(f"drop last recent turn {[m.message_id for m in turns[0]]}")
                turns.pop(0)
            else:
                return payload, conv, decisions, False, used     # system rules + memory facts + current message alone exceed the budget

    # ------------------------------------------------------------------ prompt assembly (D4 renderer)
    def _style_inputs(self, persona_id: str, history: Sequence[sc.StyleMessage]):
        pool = sc._pool(list(history)[-sc.TRAINING_PROFILE_HISTORY:])
        if not pool:
            return None, []
        prof = sc.build_profile([{"message_id": p["message_id"], "bubbles": p["bubbles"], "scenario_type": p["scenario_type"]} for p in pool])
        profile = dr.build_profile(prof, persona_id=persona_id, persona_kind=self.persona_kind)
        chosen = sc.select_examples(pool, self.cfg.style_examples_k, seed="stillhere-runtime")
        return profile, [" / ".join(c["bubbles"]) for c in chosen]

    def assemble_d4(self, persona_id: str, profile, examples, view, window: wn.Window, current: str, now: datetime, max_new_tokens: int):
        """D4 renderer (``stillhere.response.grounded_persona/1.0``, pinned). Must fit ``prompt + max_new_tokens + <|im_end|>\\n <= cutoff``.
        Gives way: the oldest recent turns, then the lowest-ranked evidence items (a conflict group leaves whole); never the current message,
        never inside a text. Returns (messages, conversation, decisions, ok, prompt_tokens, evidence_shown)."""
        cutoff = self.cfg.total_input_token_budget
        turns = list(window.turns)
        evidence = None if view is None else list(view.evidence)
        unknown = [] if view is None else list(view.unknown)
        decisions: List[str] = []
        while True:
            conv = [{"speaker": "self" if m.role == "assistant" else "partner", "text": data_text(m.text), "generated": m.generated,
                     "message_id": m.message_id, "layer": "L1"} for t in turns for m in t]
            conv.append({"speaker": "partner", "text": data_text(current), "generated": False, "layer": "current"})
            msgs = dr.render_messages(profile=profile, examples=examples, evidence=evidence, unknown=unknown, now=now.date().isoformat(),
                                      conversation=[{"speaker": c["speaker"], "text": c["text"]} for c in conv])
            used = self.count.chat(msgs) if hasattr(self.count, "chat") else self._tokens(msgs)
            reserve = max_new_tokens + (self.count.suffix() if hasattr(self.count, "suffix") else 4)
            if used + reserve <= cutoff:
                return msgs, conv, decisions, True, used, evidence
            if turns:
                decisions.append(f"drop oldest recent turn {[m.message_id for m in turns[0]]}")
                turns.pop(0)
            elif evidence and len(evidence) > 1:
                last = evidence[-1]
                group = [e for e in evidence if last.conflict_group and e.conflict_group == last.conflict_group]
                drop = group or [last]
                if len(drop) >= len(evidence):
                    return msgs, conv, decisions, False, used, evidence
                decisions.append(f"drop evidence {[e.fact_id for e in drop]}")
                evidence = [e for e in evidence if e not in drop]
            else:
                return msgs, conv, decisions, False, used, evidence

    # ------------------------------------------------------------------ disclosure on the grounding (both modes)
    def _redact_grounded(self, persona_id: str, grounded: dict, at: datetime):
        out, terms, trace = dict(grounded), [], {}
        facts = []
        for i, f in enumerate(grounded.get("facts", [])):
            st = f.get("statement", "")
            subject = "persona" if st.startswith(("本人", "你", "我")) else "person:?"
            texts = [st, f.get("replaces") or ""]
            dec = self.policy.decide(persona_id, subject, "", st, " ".join(texts), at)
            if dec.policy != ds.ALLOWED:
                head = st.split("的", 1)[0] if "的" in st[:8] else ""
                label = "本人" if subject == "persona" else (head if 0 < len(head) <= 6 and not ds.raw_terms(head) else "这个人")
                f = {**f, "statement": ds.redacted_text(label, dec), "replaces": None, "disclosure_policy": dec.policy}
                terms += dec.guard_terms
            facts.append(f)
            trace[f"fact{i + 1}"] = dec.trace()
        out["facts"] = facts
        return out, terms, trace

    # ------------------------------------------------------------------ the reply
    def reply_session(self, persona_id: str, session_id: str, text: str, request_id: str, *, history: Sequence[sc.StyleMessage] = (),
                      generate: Optional[Callable[[List[dict], dict], dict]] = None, backend: str = "neutral",
                      answerer: Optional[Callable[[str, datetime, str], Mapping]] = None, env: Optional[Mapping[str, str]] = None,
                      generation: Optional[Mapping] = None, max_retries: int = 1, force_python_search: bool = False) -> SessionReply:
        t0 = time.perf_counter()
        user = self.send_message(persona_id, session_id, text, request_id)
        done = self.store.reply_for(persona_id, session_id, user.message_id)
        if done is not None:                                   # a retried request: the stored reply, never a second generation
            return SessionReply("REPLAYED", done.content, user, done, [], {"context_version": CONTEXT_VERSION, "replayed": True})
        ask_time = self.clock()
        window = wn.build(self.store, persona_id, session_id, text, user.message_id, self.count, self.cfg, self.token_mode)
        win_text = "\n".join(m.text for m in window.messages)
        small = cr.is_small_talk(text)
        hist = hs.HistoryResult("NOT_NEEDED", [], "none", "small talk") if small else hs.search(
            self.store, persona_id, session_id, text, text, window.message_ids + [user.message_id], win_text, window.older_exists, self.cfg,
            force_python=force_python_search, at=ask_time)
        prep = self._prepare(persona_id, text, ask_time, window, hist, small, history=history, answerer=answerer, env=env, generation=generation)

        def finish(status: str, out_text: Optional[str], assistant_status: Optional[str], user_status: str, model_messages: List[dict]):
            a = None
            if out_text is not None and assistant_status:
                a = self.store.append(persona_id, session_id, "assistant", out_text, self.clock(), request_id, status=assistant_status,
                                      reply_to=user.message_id)
            self.store.set_status(user.message_id, user_status)
            prep.trace.update(status=status, total_ms=round((time.perf_counter() - t0) * 1000, 1))
            return SessionReply(status, out_text, self.store.get(user.message_id), a, model_messages, ds.scrub(prep.trace, prep.guard_terms))

        return finish(*self._generate_and_guard(prep, generate, backend, env, max_retries))

    # ------------------------------------------------------------------ the two halves of a reply (also used without a session store)
    def _prepare(self, persona_id: str, text: str, ask_time: datetime, window: wn.Window, hist: hs.HistoryResult, small: bool, *,
                 history: Sequence[sc.StyleMessage] = (), answerer=None, env: Optional[Mapping[str, str]] = None,
                 generation: Optional[Mapping] = None) -> "PreparedReply":
        """Everything up to the model call: memory (L3), grounding, disclosure, the rendered messages and the guard inputs."""
        cq = qb.complete(text, window.messages, self.cfg)
        m, reader, lat, compat = None, None, {}, False
        if small:                                              # nothing is asked: no memory answerability gate (every output guard still runs)
            grounded = {"status": "no_memory", "facts": [], "unknown": [], "route": "SMALL_TALK"}
        else:
            m, reader, answer, grounded, lat = rp.ground(self.api, persona_id, text, ask_time, answerer=answerer, env=env,
                                                         retrieval_query=cq.retrieval_query, reader_question=cq.reader_question)
            grounded = dict(grounded)
            reader_text = reader.pop("text", None) if reader else None
            kept = gcompat.keep_reader_named_units(m, grounded, reader, reader_text, ask_time)
            compat, grounded = kept is not grounded, kept
            extra_unknown = []
            if hist.status in ("NOT_FOUND", "AMBIGUOUS") and any(c in text for c in hs.HISTORY_CUES):
                extra_unknown.append(HISTORY_UNKNOWN)           # the reply must admit it, not quote a history that was not found
            if any(s in window.status for s in ("AMBIGUOUS_CONTEXT", "CONTEXT_INSUFFICIENT")):
                extra_unknown.append(LONG_UNKNOWN)
            if extra_unknown:
                grounded["unknown"] = list(grounded.get("unknown", [])) + extra_unknown
                grounded["answer_status"] = "partial" if grounded.get("facts") else "unknown"
                grounded["status"] = "facts"
        grounded, guard_terms, policy_trace = self._redact_grounded(persona_id, grounded, ask_time) if not small else (grounded, [], {})
        cfg_gen = {**rz.GENERATION, **(generation or {})}
        plan = None
        if safety.flags(env)["LENGTH_POLICY_ENABLED"] and not small:
            plan = safety.pre_generation_plan([h.text for h in history], grounded=grounded, default_max_new_tokens=int(cfg_gen["max_new_tokens"]))
        max_new = int(plan.max_new_tokens if plan else cfg_gen.get("max_new_tokens", self.cfg.default_max_new_tokens))
        view = None
        guard_grounded = grounded
        if self.render_mode == "d4":
            view = None if (small or m is None) else d4e.build(self.api, persona_id, m, grounded, ask_time, self.policy)
            guard_terms = sorted(set(guard_terms) | set(view.guard_terms if view else []))
            profile, examples = self._style_inputs(persona_id, history)
            messages, conv, decisions, ok, used, shown = self.assemble_d4(persona_id, profile, examples, view, window, text, ask_time, max_new)
            extra = [{"statement": e.evidence_text, "role": "current"} for e in (shown or []) if e.disclosure_policy == ds.ALLOWED]
            known = {f.get("statement") for f in grounded.get("facts", [])}
            guard_grounded = {**grounded, "facts": list(grounded.get("facts", [])) + [x for x in extra if x["statement"] not in known]}
            contract_trace = {"renderer_version": dr.RENDERER_VERSION, "renderer_sha256": dr.renderer_sha256(), "profile_schema": "stillhere.style.profile/1.0",
                              "evidence_schema_version": d4e.EVIDENCE_SCHEMA_VERSION, "examples": len(examples), "evidence_items": len(shown or [])}
        else:
            payload, conv, decisions, ok, used = self.assemble(persona_id, grounded, history, hist.excerpts if hist.status in ("FOUND", "AMBIGUOUS") else [],
                                                               window, text, [plan.constraint] if plan and plan.constraint else (), max_new)
            messages, contract_trace = payload.messages, payload.trace()
        visible = [messages[0]["content"]] + [c["text"] for c in conv if c["speaker"] == "self"]
        pre_leak = ds.leaked(visible, guard_terms)
        trace = {"context_version": CONTEXT_VERSION, "context_config": {"render_mode": self.render_mode, "recent_turn_limit": self.cfg.recent_turn_limit,
                                                                         "input_budget": self.cfg.total_input_token_budget},
                 "ask_time": ask_time.isoformat(), "token_mode": self.token_mode, "input_tokens": used, "input_budget": self.cfg.total_input_token_budget,
                 "output_reserve": max_new, "route": "SMALL_TALK" if small else "MEMORY", "policy_version": ds.POLICY_VERSION, "policy": policy_trace,
                 "evidence_view": view.trace() if view else None, "reader_named_units_kept": compat,
                 "window": {"message_ids": window.message_ids, "tokens": window.tokens, "dropped_turns": window.dropped_turns, "status": window.status},
                 "query": asdict(cq), "history": {"status": hist.status, "engine": hist.engine, "reason": hist.reason,
                                                  "excerpts": [asdict(e) | {"text": None} for e in hist.excerpts]},
                 "memory": None if m is None else {"route": m.route.kind, "evidence_units": [k for k, _ in m.evidence.units], "notes": list(m.notes),
                                                   "reader": reader, "evidence_ids": list(m.evidence_ids)},
                 "grounded": grounded, "budget_decisions": decisions, "contract": contract_trace, "latency_ms": lat, "conversation": conv}
        return PreparedReply(messages=messages, conversation=conv, guard_grounded=guard_grounded, guard_terms=list(guard_terms), cfg_gen=cfg_gen,
                             length_plan=plan, small=small, ok=ok, pre_leak=pre_leak, history_texts=[h.text for h in history], trace=trace)

    def _generate_and_guard(self, prep: "PreparedReply", generate, backend: str, env: Optional[Mapping[str, str]], max_retries: int):
        """The model call and every output guard. Returns (status, text, assistant_status, user_status, model_messages); ``prep.trace`` is
        updated in place. Writes nothing."""
        trace, messages, guard_grounded, guard_terms = prep.trace, prep.messages, prep.guard_grounded, prep.guard_terms
        if prep.pre_leak:                                      # fail closed: a protected value reached a model-visible field
            fb, kind = safety.safe_fallback(guard_grounded, must_not_terms=guard_terms)
            trace.update(policy_blocked="PROTECTED_VALUE_IN_PROMPT", fallback_kind=kind)
            return "POLICY_BLOCKED", fb, "FALLBACK", "CONTEXT_INSUFFICIENT", []
        if not prep.ok:
            fb, kind = safety.safe_fallback(guard_grounded, must_not_terms=guard_terms)
            trace.update(fallback_kind=kind)
            return "CONTEXT_INSUFFICIENT", fb, "FALLBACK", "CONTEXT_INSUFFICIENT", []
        if generate is None or backend == "neutral":
            out_text, status = (rz.neutral(guard_grounded) if not prep.small else "你好"), "NEUTRAL"
        else:
            try:
                lg_on = lg.enabled(env) if self.loop_guard is None else self.loop_guard
                lg_records: List[dict] = []
                gen = lg.wrap_generate(generate, enabled_=lg_on, records=lg_records)    # LLM text -> Loop Guard -> every output guard below
                res = safety.safe_generate(gen, messages, prep.cfg_gen, backend=backend, grounded=guard_grounded,
                                           conversation=[{"speaker": "self" if c["speaker"] == "self" else "other", "text": c["text"],
                                                          "generated": c["generated"]} for c in prep.conversation],
                                           must_not_terms=guard_terms, history_texts=list(prep.history_texts),
                                           fallback_text=rz.neutral(guard_grounded) if not prep.small else None, max_retries=max_retries, env=env,
                                           length_plan=prep.length_plan)
            except Exception as e:                              # the model call failed: keep the user message, no invented reply
                trace.update(error=f"{type(e).__name__}")
                return "FAILED", None, None, "FAILED", messages
            trace["safety"] = res.audit()
            trace["loop_guard"] = {"enabled": lg_on, "version": lg.VERSION, "calls": lg_records}
            if res.status == "FAILED_NO_FALLBACK" or not res.text:
                return "FAILED", None, None, "FAILED", messages
            out_text, status = res.text, ("FALLBACK" if res.status == "FALLBACK" else "GENERATED")
        if ds.leaked([out_text], guard_terms):                 # always on, whatever the guard flags: a protected value never leaves
            out_text, kind = safety.safe_fallback(guard_grounded, must_not_terms=guard_terms)
            trace.update(policy_output_blocked=True, fallback_kind=kind)
            status = "FALLBACK"
        return status, out_text, status, "ANSWERED", messages

    # ------------------------------------------------------------------ one question, no session store (benchmarks)
    def prepare_single_turn(self, persona_id: str, text: str, ask_time: datetime, *, history: Sequence[sc.StyleMessage] = (), answerer=None,
                            env: Optional[Mapping[str, str]] = None, generation: Optional[Mapping] = None) -> "PreparedReply":
        """Exactly what ``reply_session`` prepares for the FIRST message of a new session, at ``ask_time``, without storing anything: an empty
        recent window, and an empty session history (searched only when the question asks for earlier conversation, which then is NOT_FOUND)."""
        window = wn.Window([], 0, self.token_mode, [], [], False)
        small = cr.is_small_talk(text)
        if small:
            hist = hs.HistoryResult("NOT_NEEDED", [], "none", "small talk")
        else:
            why = hs.needed(text, [], "", False)
            hist = hs.HistoryResult("NOT_NEEDED", [], "none") if why is None else hs.HistoryResult("NOT_FOUND", [], "none", why)
        return self._prepare(persona_id, text, ask_time, window, hist, small, history=history, answerer=answerer, env=env, generation=generation)

    def finish_single_turn(self, prep: "PreparedReply", generate, *, backend: str, env: Optional[Mapping[str, str]] = None,
                           max_retries: int = 1) -> SessionReply:
        """The model call and the output guards for a prepared turn; nothing is stored (``user_message`` / ``assistant_message`` are None)."""
        t0 = time.perf_counter()
        status, out_text, _a, _u, model_messages = self._generate_and_guard(prep, generate, backend, env, max_retries)
        prep.trace.update(status=status, total_ms=round((time.perf_counter() - t0) * 1000, 1))
        return SessionReply(status, out_text, None, None, model_messages, ds.scrub(prep.trace, prep.guard_terms))
