"""Persona Runtime: Qwen3-8B base + Persona Inference Prompt + optional persona adapter + optional SQLite memory (EF1).

    rt = PersonaRuntime.from_config(root, profile_id, backend="hf", base_dir=...)     # runtime/runtime_config.json (Q1 selection)
    rt = PersonaRuntime(root, profile_id, backend="hf", base_dir=..., adapters=["a-..."], memory="production")
    rt.reply(history)       # history: [("other", text) | ("me", text), ...]  ->  {"text", "mode", "evidence", "loop_guard", ...}

The four modes are exactly the B / C / D / E rendering of ``stillhere.persona.evaluation`` (same system prompt, same user-turn
transcript as training, same decode). One base stays loaded; adapters are attached by name and switched with ``use_adapter``.
Memory views never mix: ``production`` = the full-history store ``<profile>/memory/memory.sqlite`` (chat); ``eval`` = the in-memory
train-cutoff store of the A-E ablation. Every reply goes through the existing Loop Guard (``runtime.loop_guard.guard``) and has
contact-detail / credential values redacted (``cleaning.privacy_hits``); the trace records both. The existing D4 renderer, fact reader
and SessionRuntime are not changed.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from stillhere.runtime.loop_guard import guard

from . import adapters as AD
from . import card as K
from . import cleaning as C
from . import evaluation as EV
from . import memory as M
from . import store as S
from .models import LowVramGen, local_complete, make_generator  # noqa: F401 - re-exported (public API)

REPO = Path(__file__).resolve().parents[3]
MAX_TURNS = 12
REDACT = ("credential", "id_number", "bank_card", "phone", "email")


from .routing import HISTORY_CUE, PERSONAL_CUE, QUESTION_CUE, route_of  # noqa: F401,E402 - re-exported (moved to routing.py)
UNCERTAIN_CUE = re.compile(r"不记得|记不清|记不太清|不知道|忘了|不清楚|没印象|不确定|想不起|不太记得|没聊|没有聊|没说过|好像没")
GREEDY = {"do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0, "max_new_tokens": 512,
          "enable_thinking": False, "seed": 20261112}


def to_owner_frame(question: str) -> str:
    """The Fact Reader's frame is the owner's own question (its '我' = the owner = the persona, '本人'). In persona chat the asker is the
    other person: '你' means the persona and '我' the asker. Pronouns are mapped on the QUESTION only (the reader prompt is unchanged)."""
    return question.replace("你们", "本人和对方").replace("我们", "本人和对方").replace("你", "本人").replace("我", "对方")


def check_fact_reply(text: str, reader: dict):
    """The styled reply must carry the checked answer (its numbers, and at least one of its 2-char content units) or, without evidence,
    express uncertainty; otherwise the deterministic fallback is used (the answer itself / the existing MINIMAL_REPLY)."""
    from stillhere.reader.fact import fact_preserved
    from stillhere.style.safety import MINIMAL_REPLY
    flat = (text or "").replace("\n", "")
    if reader["answerable"]:
        ans = re.sub(r"[，。！？、,.!?\s]|本人|对方|根据现有记录", "", reader["answer"] or "")
        units = {ans[i:i + 2] for i in range(len(ans) - 1)}
        missing = fact_preserved(reader["answer"], flat)["missing_numbers"]
        if not missing and (not units or any(u in flat for u in units)):
            return text, "ANSWER_KEPT"
        return reader["answer"].replace("本人", "我").replace("对方", "你"), "FALLBACK_TO_READER_ANSWER"
    if UNCERTAIN_CUE.search(flat) and not re.search(r"\d", flat):
        return text, "UNCERTAINTY_KEPT"
    return MINIMAL_REPLY, "FALLBACK_MINIMAL_UNCERTAIN"


def transcript(history: Sequence[Tuple[str, str]]) -> str:
    lab = {"other": "对方", "me": "我"}
    return "\n".join(f"{lab[r]}：{' / '.join(x.strip() for x in t.splitlines() if x.strip())}" for r, t in history[-MAX_TURNS:])


class PersonaRuntime:
    def __init__(self, root, profile_id: str, *, backend: Optional[str] = None, base_dir: Optional[str] = None, adapters: Sequence[str] = (), memory=None,
                 generator=None, card_id: Optional[str] = None, memory_pipeline: str = "v1", memory_v2=None, dense=None, reader_version: str = "v1",
                 answer: Optional[dict] = None, allow_external_api: Optional[bool] = None, router: str = "rules", router_generator=None):
        """``generator``: an already-loaded generator (the Colab evaluation shares ONE base across Writer / Reader / chat); ``card_id``:
        a specific card version (prompt comparison) instead of ``cards/CURRENT.json``. Both default to the normal runtime.
        ``memory_pipeline``: ``v1`` (default, the FAST-evaluated path) or ``v2`` (``persona.memory_v2``: wider router, date-aware first
        hop, evidence windows, optional local dense channel ``dense``; production memory only). ``memory_v2``: a ``memory_v2.V2Config``.
        ``answer``: overrides of the ``answer`` provider role (``stillhere.providers.roles``). With an API provider (DeepSeek or a custom
        OpenAI-compatible endpoint) the persona and the Fact Reader run on that model through the SAME reader chain (grounding and
        attribution gates, answer preservation); it needs ``allow_external_api`` (or STILLHERE_ALLOW_EXTERNAL_API=1) and no adapter.
        An explicit ``backend`` (hf / hf-lowvram / fake) always means the local path."""
        if memory_pipeline not in ("v1", "v2"):
            raise ValueError("memory_pipeline must be 'v1' or 'v2'")
        self.memory_pipeline, self.dense = memory_pipeline, dense
        self.memory_notes: List[str] = []
        if router not in ("rules", "llm"):
            raise ValueError("router must be 'rules' or 'llm'")
        self.router, self.router_gen, self.last_route = router, router_generator, None
        self.retrieval = ("V2D" if dense is not None else "V2L") if memory_pipeline == "v2" else "V1"
        from . import recall as _R
        if reader_version not in _R.READER_VERSIONS:
            raise ValueError(f"reader_version must be one of {_R.READER_VERSIONS}")
        self.reader_version = reader_version          # "v1" (FAST / V2 / V3 reader, default) | "p1" (2026-11-19 fixes; V4 decides)
        if memory_pipeline == "v2":
            from . import memory_v2 as V2
            self.v2cfg = memory_v2 or V2.V2Config()
        self.root = Path(root) if root else S.default_root()
        self.profile_id = profile_id
        self.pdir = S.profile_dir(self.root, profile_id)
        self.prof = S.read_json(self.pdir / "profile.json")
        cur = K.load(self.pdir, card_id) if card_id else K.current(self.pdir)
        if cur is None:
            raise K.CardError("NO_CARD", "no persona card yet: run stillhere persona prepare / pipeline")
        self.card, self.prompt = cur, cur["prompt"]
        self.adapters = {a: AD.resolve(self.pdir, a) for a in adapters}
        lineage = {cur["card_id"], (cur["card"].get("refinement") or {}).get("parent_card"), cur["manifest"].get("parent_card")} - {None}
        for a, info in self.adapters.items():     # a prompt-only child card (refinement / prompt candidate) keeps its parent's adapter
            if info["manifest"]["card_id"] not in lineage:
                raise AD.AdapterError("CARD_MISMATCH", f"adapter {a} belongs to card {info['manifest']['card_id']}, not to the current card lineage")
        from . import config as CF
        from stillhere.providers import llm as PL
        from stillhere.providers import roles as PR
        self.answer_cfg = PR.resolve("answer", profile_dir=self.pdir, overrides=answer or {})
        self.cfg = CF.load(profile_dir=self.pdir, overrides={"backend": backend, "base_model_dir": base_dir or self.answer_cfg.model_dir})
        if generator is not None:
            self.backend, self.gen = "injected", generator
        elif backend in (None, "auto") and self.answer_cfg.provider in PL.API_PROVIDERS:
            if self.adapters:
                raise AD.AdapterError("ADAPTER_NEEDS_LOCAL_BASE", "a persona adapter runs only on the local Qwen3-8B base: turn the adapter off "
                                      "(persona finalize --adapter off) to chat with an API model")
            self.backend = "api"
            self.gen = PR.api_generator("answer", self.answer_cfg, allow_external=allow_external_api, usage_log=self.pdir / "runtime" / "api_usage.jsonl")
        else:
            self.backend = CF.resolve_backend(self.cfg)
            self.gen = make_generator(self.backend, self.cfg.base_model_dir, {a: i["dir"] for a, i in self.adapters.items()}, device_index=self.cfg.device_index)
        self.active: Optional[str] = None
        self.memory_view = {True: "eval", False: None}.get(memory, memory) if isinstance(memory, bool) else memory
        self.api = None
        if self.memory_view == "eval":
            from . import package as P
            ddir = P.latest_dataset(self.pdir)
            self.api, self.memory_meta = M.build_api(self.prof["source"]["path"], persona_id=profile_id, target_username=self.prof["target"]["username"],
                                                     cutoff=M.cutoff_from_train(ddir))
        elif self.memory_view == "production":
            from . import memory_store as MS
            if MS.status(self.root, profile_id)["memory_status"] == "NOT_BUILT":
                raise S.StoreError("MEMORY_NOT_BUILT", "run stillhere persona memory-build first (or chat without memory)")
        elif self.memory_view is not None:
            raise ValueError("memory must be None, 'production' or 'eval'")

    @classmethod
    def from_config(cls, root, profile_id: str, *, backend: Optional[str] = None, base_dir: Optional[str] = None, answer: Optional[dict] = None,
                    allow_external_api: Optional[bool] = None, generator=None, dense_encoder=None, router: Optional[str] = None, router_generator=None):
        """``runtime_config.json`` -> runtime. A ``memory_pipeline`` with ``dense`` (the product V2D, ``persona.product_config``) opens the
        profile's local dense index; when it is unavailable (no index, no embedding model, no torch) the SAME pipeline runs lexical-only
        (V2L) and ``rt.memory_notes`` says why. Without the section: the historical V1 path."""
        pdir = S.profile_dir(Path(root) if root else S.default_root(), profile_id)
        cfg = S.read_json(pdir / "runtime" / "runtime_config.json")
        ad = [cfg["adapter"]["adapter_id"]] if cfg["adapter"]["enabled"] and cfg["adapter"]["adapter_id"] else []
        mp = cfg.get("memory_pipeline") or {}                 # absent: the historical V1 pipeline / reader v1 (``persona.final_config``)
        kw, notes, dense = {}, [], None
        if mp:
            from . import memory_v2 as V2
            if mp.get("dense") and cfg["memory"]["enabled"]:
                from . import dense_memory as DM
                dense, why = DM.open_channel(root, profile_id, encoder=dense_encoder)
                if dense is None:
                    if mp.get("dense_fallback", "v2l") != "v2l":
                        raise ValueError(f"dense pipeline without a fallback and {why}")
                    notes.append(f"{why} -> V2L (lexical) fallback")
                elif why != "OK":
                    notes.append(why)
            kw = {"memory_pipeline": mp.get("pipeline", "v1"), "reader_version": mp.get("reader_version", "v1"), "dense": dense,
                  "memory_v2": V2.V2Config(use_dense=dense is not None, k=int(mp.get("k", 8)), neighbors=tuple(mp.get("neighbors", (2, 2))),
                                           date_mode=mp.get("date_mode", "prefer"), strip_qform=bool(mp.get("strip_qform", False)))}
        import os as _os
        rt = cls(root, profile_id, backend=backend, base_dir=base_dir, adapters=ad, memory="production" if cfg["memory"]["enabled"] else None,
                 answer=answer, allow_external_api=allow_external_api, generator=generator,
                 router=router or _os.environ.get("STILLHERE_ROUTER_MODE", "rules"), router_generator=router_generator, **kw)
        rt.memory_notes = notes
        rt.p2 = bool(mp.get("p2", False))           # Memory P2 retrieval: on in the product section (2026-11-19-v2d-p2), off in product-v2d / research-final
        rt.retrieval = ("V2D" if dense is not None else "V2L") if mp else "V1"      # + ``rt.p2`` (Memory P2 on top of either)
        from . import product_config as PC
        rt.active_memory_pipeline = PC.active_pipeline(mp, dense is not None) if cfg["memory"]["enabled"] else "OFF"
        if rt.card["manifest"]["prompt_sha256"] != cfg["persona_prompt"]["sha256"]:
            raise K.CardError("CONFIG_STALE", "the current card is not the one runtime_config.json selected: run persona finalize again")
        rt.use_adapter(ad[0] if ad else None)
        rt.config = cfg
        return rt

    def use_adapter(self, name: Optional[str]) -> None:
        if name is not None and name not in self.adapters:
            raise AD.AdapterError("ADAPTER_UNKNOWN", f"{name} is not loaded")
        self.active = name

    def evidence(self, history, now: Optional[datetime]) -> Optional[str]:
        last = next((t for r, t in reversed(history) if r == "other"), "") or "最近"
        at = now or datetime.now(M.TZ)
        if self.memory_view == "eval":
            return M.evidence(self.api, self.profile_id, last, at)
        if self.memory_view == "production":
            from . import memory_store as MS
            return MS.query(self.root, self.profile_id, last, ask_time=at)["evidence"]
        return None

    def messages(self, history: Sequence[Tuple[str, str]], now: Optional[datetime] = None) -> List[dict]:
        system = self.prompt
        ev = self.evidence(history, now)
        if ev:
            system = system.rstrip("\n") + "\n\n" + ev
        return [{"role": "system", "content": system}, {"role": "user", "content": transcript(history)}]

    def fact_pack(self, question: str, now: Optional[datetime]):
        at = now or datetime.now(M.TZ)
        if self.memory_view == "eval":
            return M.pack(self.api, self.profile_id, question, at)
        if self.memory_view == "production":
            from . import memory_store as MS
            if self.memory_pipeline == "v2":
                from . import memory_v2 as V2
                api = MS.open_api(self.pdir, self.prof)
                try:
                    if getattr(self, "p2", False):     # Memory P2 (opt-in, ``memory_pipeline.p2``): requirements + one bounded second pass
                        from . import memory_p2 as MP2
                        pk, meta = MP2.pack_p2(api, self.profile_id, question, at, self.v2cfg, dense=self.dense)
                    else:
                        pk, meta = V2.pack_v2(api, self.profile_id, question, at, self.v2cfg, dense=self.dense)
                finally:
                    api.close()
                self.last_v2_meta = meta
                return pk
            return MS.pack(self.root, self.profile_id, question, ask_time=at)
        return None

    def route(self, text: str, now: Optional[datetime] = None) -> str:
        if self.memory_view is None:
            return "chat"
        if self.memory_pipeline == "v2" and self.memory_view == "production":
            from . import memory_v2 as V2
            if self.router == "llm":
                if self.router_gen is None:
                    from stillhere.providers import roles as PR
                    try:
                        self.router_gen = PR.api_generator("router", PR.resolve("router", profile_dir=self.pdir), usage_log=self.pdir / "runtime" / "api_usage.jsonl")
                    except Exception as e:  # noqa: BLE001 - the rule router is always there
                        self.router, self.last_route = "rules", {"source": "rules", "llm": {"status": "UNAVAILABLE", "error": str(e)[:200]}}
                        self.memory_notes.append("LLM router unavailable -> rule router")
                        return V2.route_of_v2(text, now, version=self.reader_version)
                from . import llm_router as LR
                self.last_route = LR.route(text, now, self.router_gen, version=self.reader_version)
                return str(self.last_route["route"])
            return V2.route_of_v2(text, now, version=self.reader_version)
        return route_of(text, now)

    def reply(self, history: Sequence[Tuple[str, str]], now: Optional[datetime] = None) -> dict:
        last = next((t for r, t in reversed(history) if r == "other"), "")
        route = self.route(last, now)
        if route == "date" and self.memory_view != "production":
            route = "fact"                                             # date recall reads the full archive: production view only
        reader = None
        if route == "date":
            msgs, reader = self.date_route(history, last, now)
        elif route == "fact":
            msgs, reader = self.fact_route(history, last, now)
        else:
            msgs = self.messages(history, now)
        out = self.gen(msgs, EV.DECODE, self.active)
        text = out.get("text", "")
        if reader is not None:
            text, reader["final_check"] = check_fact_reply(text, reader)
        lg = guard(text, hit_max=out.get("hit_token_limit"))
        final = C.redact(lg.final_response, REDACT)
        mem = self.memory_view is not None
        return {**out, "text": final, "raw_text": out.get("text"), "loop_guard": lg.trace(), "redacted": final != lg.final_response, "route": route, "route_detail": self.last_route,
                "reader": reader, "generator": self.gen.label, "synthetic": self.gen.label == EV.FakeGenerator.label, "adapter": self.active,
                "memory": self.memory_view, "retrieval": self.retrieval if self.memory_view else None,
                "memory_p2": bool(getattr(self, "p2", False)) if self.memory_view else None,
                "active_memory_pipeline": getattr(self, "active_memory_pipeline", None) if self.memory_view else "OFF", "memory_notes": list(self.memory_notes),
                "evidence_chars": len(msgs[0]["content"]) - len(self.prompt), "prompt_sha256": self.card["manifest"]["prompt_sha256"],
                "mode": {(False, False): "B", (True, False): "C", (False, True): "D", (True, True): "E"}[(self.active is not None, mem)]}

    def fact_route(self, history, question: str, now: Optional[datetime]):
        """History / personal-fact questions: EF1 -> fact_pack -> Fact Reader B (unchanged FACT_SYSTEM, base weights, the reader's own
        generation config) -> the persona states the checked answer in its own words (or, without evidence, says it does not know)."""
        from stillhere.reader import fact as RF
        from . import recall as R
        subject = R.resolve_subject(question, self.reader_version)
        q = to_owner_frame(question)
        pk = self.fact_pack(q, now)
        pk.notes.extend(R.perspective_notes(subject, self.reader_version))   # the pack's system-provided notes; FACT_SYSTEM is unchanged
        rmsgs = RF.fact_messages(pk)
        cfg = RF.generation_for("FACT", pk)
        r = self.gen(rmsgs, cfg, None)                                 # the reader never runs through the persona adapter
        rd = RF.parse(r.get("text", ""))
        answerable = bool(rd.answer) and not rd.abstain and rd.answerable is not False
        sup = RF.support_check(rd.answer, pk) if answerable else {}
        fail = None
        if answerable and (sup.get("unsupported_numbers") or sup.get("unsupported_quotes")):
            answerable, fail = False, "GROUNDEDNESS_FAIL"
        speakers = {i.eid: list(i.speakers or []) for i in pk.items}
        if answerable and not R.attribution_ok(subject, [speakers.get(e, []) for e in (rd.evidence or [])]):
            answerable, fail = False, "SUBJECT_RESOLUTION_FAIL"          # cited lines are not the asked person's own: no recall by dropping attribution
        content = (f"【要表达的内容】（已经核对过的记忆答案，必须如实说出来，可以很短，不能改动其中的地点、人名、数字和时间）：{rd.answer}"
                   if answerable else "【要表达的内容】记忆里没有找到这件事的记录：自然地说不记得或不确定，不要编造任何过去的经历、人物、地点或细节。")
        system = self.prompt.rstrip("\n") + "\n\n" + content
        reader = {"answerable": answerable, "answer": rd.answer if answerable else None, "abstain": not answerable, "evidence": rd.evidence,
                  "parsed": rd.ok, "items": len(pk.items), "reader_tokens": r.get("new_tokens"), "reader_prompt_tokens": r.get("prompt_tokens"), "reader_latency_s": r.get("latency_s"), "support": sup,
                  "question_owner_frame": q, "subject": subject.to_dict(), "fail": fail if fail else (None if answerable else "READER_ABSTAINED"),
                  "retrieved_message_ids": sorted({m for i in pk.items for m in (i.message_ids or [])}), "route": "fact",
                  "rejected_answer": rd.answer if (fail and rd.answer) else None,         # kept for error analysis only, never shown
                  "reader_version": self.reader_version}
        return [{"role": "system", "content": system}, {"role": "user", "content": transcript(history)}], reader

    def date_route(self, history, question: str, now: Optional[datetime]):
        """'What did we talk about on <date>': every archived message of that local-time window, oldest first -> segmented grounded
        summary (cited ids; unsupported points dropped) -> the persona says it. No messages -> it says there was no chat that day."""
        from . import memory_store as MS
        from . import recall as R
        at = now or datetime.now(M.TZ)
        scope = R.date_scope(question, at)
        msgs = R.window_messages(MS.mdir(self.pdir) / "memory.sqlite", self.profile_id, scope, before=at)
        acc = {"prompt_tokens": 0, "new_tokens": 0, "latency_s": 0.0, "calls": 0}

        def summarise_call(m):                                         # base weights, greedy; token use is summed for the cost report
            r = self.gen(m, GREEDY, None)
            acc["calls"] += 1
            for k in ("prompt_tokens", "new_tokens", "latency_s"):
                acc[k] += r.get(k) or 0
            return r
        ds = R.summarize_window(scope, msgs, summarise_call)
        ans = ds.answer()
        if ds.answerable:
            content = f"【要表达的内容】（已经核对过的聊天记录，必须如实说出来，可以简短，但不能改动其中的人、事、地点、数字和时间）：{ans}"
        elif ds.status == "NO_MESSAGES":
            content = f"【要表达的内容】记录里{scope.label()}没有聊天：自然地说那天好像没聊天或不记得了，不要编造任何内容。"
        else:
            content = "【要表达的内容】那段时间的记录整理不出可靠的内容：自然地说记不太清了，不要编造任何内容。"
        system = self.prompt.rstrip("\n") + "\n\n" + content
        reader = {"answerable": ds.answerable, "answer": ds.answer(label=False) if ds.answerable else None, "abstain": not ds.answerable, "evidence": [],
                  "items": len(msgs), "date_summary": ds.to_dict(), "route": "date", "subject": R.resolve_subject(question, self.reader_version).to_dict(),
                  "fail": None if ds.answerable or ds.status == "NO_MESSAGES" else ds.status,
                  "retrieved_message_ids": [m["message_id"] for m in msgs], "reader_tokens": acc["new_tokens"],
                  "reader_prompt_tokens": acc["prompt_tokens"], "reader_latency_s": round(acc["latency_s"], 3), "reader_calls": acc["calls"]}
        return [{"role": "system", "content": system}, {"role": "user", "content": transcript(history)}], reader
