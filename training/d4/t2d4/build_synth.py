"""Synthetic memory-grounded episodes (SYNTHETIC_PERSONA x synthetic memory world).

Every episode = one scenario (evidence, unknown slots, conversation, answer key, content plan) + one synthetic persona (profile v1, K=5 examples)
+ a target realised in that persona's way of speaking, rendered by ``stillhere.response.grounded_persona/1.0`` and accepted only when every
validator passes. A rejected surface is re-realised (new surface seed) up to ``MAX_SURFACE_TRIES`` times, then the episode is dropped and counted.

Group types (TRAIN):
    SINGLE             one episode
    COUNTERFACTUAL     same persona, same question, different memory states -> different targets
    PERSONA_VARIATION  same scenario and content plan, different personas -> fact-equivalent targets in different styles
    PARAPHRASE         same persona and facts, different question frames
Evaluation splits carry a reference target plus ablation inputs (NO_MEMORY, NO_PROFILE, NO_EXAMPLES, SWAPPED_PERSONA, WRONG_MEMORY_STATE).

Isolation: each split draws personas and entity values from its own pool (lexicon.pool_of), eval splits use EVAL question frames only, and
HOLDOUT_ONLY_DOMAINS appear only in FROZEN_HOLDOUT.
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
from typing import Dict, List, Optional, Sequence, Tuple

from . import validate as V
from .domains import DOMAINS, HOLDOUT_ONLY_DOMAINS
from .evidence import Evidence
from .lexicon import BANKS
from .persona_synth import Persona, make_persona, realise
from .profile_v1 import check_no_facts
from .renderer import render_messages
from . import renderer as RD
from .scenarios import CATEGORIES, FAMILY, Ctx, NotApplicable, build as build_scenario, semantic_key, stable_seed, template_family

PERSONA_COUNTS = {"TRAIN": 150, "DEV": 40, "SAFETY": 40, "HOLDOUT": 40}
POOL_TAG = {"TRAIN": "T", "DEV": "D", "SAFETY": "S", "HOLDOUT": "H"}
MAX_SURFACE_TRIES = 6
TRAIN_TARGET_CAP = 3          # the same exact target text at most this often in TRAIN
EVAL_TARGET_CAP = 2

TRAIN_WEIGHTS = {"KNOWN_FACT": 13, "FULL_UNKNOWN": 9, "PARTIAL_UNKNOWN": 7, "TEMPORAL_CHANGE": 6, "HISTORICAL_ASK": 4, "UPCOMING": 4,
                 "CORRECTION": 6, "CANCELLATION": 5, "CONFLICT": 5, "THIRD_PARTY": 9, "MULTI_SUBJECT": 4, "MUST_NOT": 7,
                 "DISCLOSURE_ALLOWED": 3, "POLICY_UNCERTAIN": 3, "OUTDATED_CONTACT": 2, "CONVERSATION_DEPENDENCE": 6, "EXAMPLE_FACT_TRAP": 3}
EVAL_WEIGHTS = {k: 1 for k in CATEGORIES}  # balanced
SAFETY_WEIGHTS = {"MUST_NOT": 10, "POLICY_UNCERTAIN": 5, "DISCLOSURE_ALLOWED": 4, "OUTDATED_CONTACT": 3, "CORRECTION": 4, "CANCELLATION": 3,
                  "CONFLICT": 3, "TEMPORAL_CHANGE": 2, "HISTORICAL_ASK": 2, "UPCOMING": 2, "THIRD_PARTY": 4, "MULTI_SUBJECT": 2,
                  "EXAMPLE_FACT_TRAP": 2, "FULL_UNKNOWN": 2, "PARTIAL_UNKNOWN": 2}
FACT_STATES = ("KNOWN_FACT", "FULL_UNKNOWN", "CORRECTION", "TEMPORAL_CHANGE", "UPCOMING", "CANCELLATION", "CONFLICT")
DISCLOSURE_STATES = ("MUST_NOT", "DISCLOSURE_ALLOWED", "POLICY_UNCERTAIN", "OUTDATED_CONTACT")
GROUP_WEIGHTS = {"SINGLE": 0.45, "COUNTERFACTUAL": 0.25, "PERSONA_VARIATION": 0.15, "PARAPHRASE": 0.15}
CHITCHAT = [("在吗", "在"), ("忙不忙", "还行"), ("最近咋样", "老样子"), ("诶", "嗯？"), ("问你个事", "你说"), ("有空不", "咋了"),
            ("吃了没", "吃了"), ("睡了没", "还没")]


# ------------------------------------------------------------------------------------------------------------------------------ personas
def make_personas(seed: str) -> Dict[str, List[Persona]]:
    out: Dict[str, List[Persona]] = {}
    for pool, n in PERSONA_COUNTS.items():
        lst = []
        for i in range(n):
            pid = f"SYN-{POOL_TAG[pool]}-{i:03d}"
            r = random.Random(stable_seed(seed, "trap", pid))
            trap = {"city": r.choice(BANKS["city"][pool]), "item": r.choice(BANKS["item"][pool]), "pet": r.choice(BANKS["pet_kind"][pool])}
            p = make_persona(pid, pool, seed, trap_values=trap)
            bad = check_no_facts(p.profile)
            if bad:
                raise RuntimeError(f"profile schema: {pid}: {bad}")
            lst.append(p)
        out[pool] = lst
    return out


def persona_record(p: Persona) -> dict:
    return {"persona_id": p.style.persona_id, "persona_kind": "SYNTHETIC_PERSONA", "pool": p.style.pool, "style_params": p.style.to_json(),
            "profile_v1": p.profile, "history": p.history, "trap_values": p.trap_values}


def _trap_line(p: Persona, city: str) -> Optional[str]:
    for m in p.history:
        t = "\n".join(m["bubbles"])
        if m.get("scenario_type") == "trap" and city in t:
            return t
    return None


def examples_for(p: Persona, seed: str, trap_city: Optional[str] = None) -> List[str]:
    if trap_city:
        line = _trap_line(p, trap_city)
        if not line:
            raise NotApplicable("persona has no trap line")
        ex = [line] + [x for x in p.pick_examples(seed, exclude=[line], k=5) if x != line][:4]
        random.Random(seed).shuffle(ex)
        return ex
    # a trap line may also appear in an ordinary episode: the evidence decides what is true, the examples never do
    return p.pick_examples(seed, k=5)


# ------------------------------------------------------------------------------------------------------------------------------ validation
def validate_episode(target: str, scn: dict, examples: Sequence[str], persona: Optional[Persona]) -> List[str]:
    ev, key, conv = scn["evidence"], scn["answer_state"], scn["conversation"]
    errs = V.grounding(target, ev, key, conv) + V.temporal(ev, scn["now"]) + V.disclosure(target, ev, conv, examples) \
        + V.conversation_consistency(conv, ev) + V.hygiene(target)
    errs += V.persona(target, examples, persona.style.to_json() if persona else None)
    return errs


class Caps:
    def __init__(self, cap: int):
        self.cap, self.c = cap, collections.Counter()

    def ok(self, t: str) -> bool:
        return self.c[V._norm(t)] < self.cap

    def add(self, t: str) -> None:
        self.c[V._norm(t)] += 1


def surface(scn: dict, persona: Persona, examples: Sequence[str], seed: str, caps: Caps, rejects: collections.Counter) -> Optional[str]:
    for k in range(MAX_SURFACE_TRIES):
        t = realise(scn["clauses"], persona.style, random.Random(stable_seed(seed, "surface", k)))
        errs = validate_episode(t, scn, examples, persona)
        if not caps.ok(t):
            errs.append("target_cap")
        if not errs:
            caps.add(t)
            return t
        for e in errs:
            rejects[e.split(":")[0]] += 1
    rejects["episode_dropped"] += 1
    return None


# ------------------------------------------------------------------------------------------------------------------------------ scenarios
def _domains_for(split: str) -> List[str]:
    return [d for d in DOMAINS if d not in HOLDOUT_ONLY_DOMAINS or split == "FROZEN_HOLDOUT"]


def _add_chitchat(scn: dict, rnd: random.Random) -> None:
    """Fact-free exchanges before the topic (conversation depth 1-4 partner turns; history stays separate from memory)."""
    k = 2 if rnd.random() < 0.06 else 1 if rnd.random() < 0.15 else 0
    for a, b in rnd.sample(CHITCHAT, k):
        scn["conversation"] = [{"speaker": "partner", "text": a}, {"speaker": "self", "text": b}] + scn["conversation"]


def trap_persona(cat: str, personas: List[Persona], rnd: random.Random) -> Tuple[Optional[Persona], Optional[dict]]:
    """EXAMPLE_FACT_TRAP: a persona whose examples will contain an earlier, unrelated city remark; the evidence holds another city."""
    if cat != "EXAMPLE_FACT_TRAP":
        return None, None
    cands = [p for p in personas if p.trap_values and _trap_line(p, p.trap_values["city"])]
    if not cands:
        return None, None
    p = rnd.choice(cands)
    return p, {"trap": p.trap_values["city"]}


def try_scenario(category: str, split: str, pool: str, seed: str, *, dom: Optional[str] = None, subj_kind: Optional[str] = None,
                 qframe: Optional[int] = None, sv: Optional[dict] = None, nd: Optional[int] = None, tries: int = 30) -> Optional[dict]:
    split_kind = "TRAIN" if split == "TRAIN" else "EVAL"
    doms = [dom] if dom else _domains_for(split)
    r = random.Random(stable_seed(seed, "dom"))
    for t in range(tries if not dom else 1):
        d = dom or r.choice(doms)
        c = Ctx(pool, random.Random(stable_seed(seed, "ctx", t)))
        try:
            s = build_scenario(category, d, c, split_kind, subj_kind=subj_kind, qframe=qframe, seed_values=sv, n_distract=nd)
        except NotApplicable:
            continue
        if split != "FROZEN_HOLDOUT" and any(e.predicate in HOLDOUT_ONLY_DOMAINS for e in s["evidence"]):
            continue
        s["ctx_seed"] = stable_seed(seed, "ctx", t)
        return s
    return None


def rebuild(anchor: dict, category: str, split: str, pool: str, *, qframe: Optional[int] = None, keep_sv=("subj", "v", "old", "third")) -> Optional[dict]:
    """Same context seed, subject, values and distractor count as ``anchor``, another memory state (or another question frame)."""
    split_kind = "TRAIN" if split == "TRAIN" else "EVAL"
    sv = {k: v for k, v in anchor["seed_values"].items() if k in keep_sv}
    c = Ctx(pool, random.Random(anchor["ctx_seed"]))
    sk = anchor["seed_values"]["subj"][0]
    sk = "self" if sk == "self" else "third"
    try:
        s = build_scenario(category, anchor["domain"], c, split_kind, subj_kind=sk, qframe=qframe, seed_values=sv, n_distract=anchor["n_distract"])
    except NotApplicable:
        return None
    s["ctx_seed"] = anchor["ctx_seed"]
    return s


def _weighted(rnd: random.Random, w: Dict[str, float]) -> str:
    ks = list(w)
    return rnd.choices(ks, weights=[w[k] for k in ks])[0]


# ------------------------------------------------------------------------------------------------------------------------------ episodes
def _ev_json(ev: Sequence[Evidence]) -> List[dict]:
    return [e.to_json() for e in ev]


def episode_record(scn: dict, persona: Persona, examples: List[str], target: str, *, split: str, group_id: str, group_type: str,
                   data_version: str) -> dict:
    msgs = render_messages(profile=persona.profile, examples=examples, evidence=scn["evidence"], unknown=scn["unknown"], now=scn["now"],
                           conversation=scn["conversation"])
    key = dict(scn["answer_state"])
    eid = "D4S-" + hashlib.sha256(json.dumps([split, group_id, persona.style.persona_id, msgs, target], ensure_ascii=False).encode()).hexdigest()[:16]
    return {
        "episode_id": eid, "data_version": data_version, "split": split, "source": "SYNTH", "persona_id": persona.style.persona_id,
        "persona_kind": "SYNTHETIC_PERSONA", "persona_pool": persona.style.pool, "group_id": group_id, "group_type": group_type,
        "category": scn["category"], "family": FAMILY[scn["category"]], "domain": scn["domain"], "subject_type": scn["subject_type"],
        "now": scn["now"], "evidence": _ev_json(scn["evidence"]), "unknown": [u.__dict__ for u in scn["unknown"]],
        "conversation": scn["conversation"], "partner_turns": sum(1 for t in scn["conversation"] if t["speaker"] == "partner"),
        "examples": examples, "profile_v1_sha256": persona.profile["profile_v1_sha256"], "answer_state": key,
        "template_family": template_family(scn), "question_template": scn["question_template"], "semantic_key": semantic_key(scn), "content_plan": scn["clauses"],
        "system": msgs[0]["content"], "user": msgs[1]["content"], "target": target, "target_bubbles": [b for b in target.split("\n") if b.strip()],
        "validators": "PASS",
    }


def _scn_add_meta(s: dict, nd: int) -> dict:
    s["n_distract"] = nd
    return s


def gen_group(gtype: str, split: str, pool: str, personas: List[Persona], weights: Dict[str, float], seed: str, caps: Caps,
              rejects: collections.Counter, data_version: str) -> List[dict]:
    rnd = random.Random(stable_seed(seed, "group"))
    nd = rnd.choice([0, 1, 1, 2, 3])
    gid = "G-" + stable_seed(seed)[:12]
    out: List[dict] = []

    def emit(scn: dict, persona: Persona, gt: str) -> bool:
        trap = scn["answer_state"].get("trap_value") if scn["category"] == "EXAMPLE_FACT_TRAP" else None
        try:
            ex = examples_for(persona, stable_seed(seed, "ex", persona.style.persona_id, len(out)), trap_city=trap)
        except NotApplicable:
            rejects["no_trap_line"] += 1
            return False
        t = surface(scn, persona, ex, stable_seed(seed, persona.style.persona_id, len(out)), caps, rejects)
        if t is None:
            return False
        out.append(episode_record(scn, persona, ex, t, split=split, group_id=gid, group_type=gt, data_version=data_version))
        return True

    if gtype == "COUNTERFACTUAL":
        disclosure = rnd.random() < 0.25
        states = list(DISCLOSURE_STATES if disclosure else FACT_STATES)
        anchor_cat = rnd.choice(states)
        anchor = try_scenario(anchor_cat, split, pool, seed, nd=nd)
        if not anchor:
            rejects["no_scenario"] += 1
            return out
        _scn_add_meta(anchor, nd)
        persona = rnd.choice(personas)
        members = [anchor]
        q0 = anchor["conversation"][-1]["text"]
        same_answer = {"KNOWN_FACT": "CORRECTION", "CORRECTION": "KNOWN_FACT"}  # same correct reply: not a contrast
        for cat in rnd.sample([s for s in states if s != anchor_cat], k=len(states) - 1):
            if len(members) >= rnd.choice([2, 3, 3]):
                break
            if any(same_answer.get(cat) == m["category"] for m in members):
                continue
            s = rebuild(anchor, cat, split, pool, qframe=anchor["question_frame"] % 1000)
            if s and s["conversation"][-1]["text"] == q0:
                members.append(_scn_add_meta(s, nd))
        if len(members) < 2:
            rejects["counterfactual_incomplete"] += 1
            return out
        conv0 = anchor["conversation"]
        for s in members:
            s["conversation"] = conv0  # identical conversation (pre-turns included)
            emit(s, persona, "COUNTERFACTUAL")
        seen, kept = set(), []
        for e in out:  # a contrast group needs distinct targets
            if V._norm(e["target"]) not in seen:
                seen.add(V._norm(e["target"]))
                kept.append(e)
        if len(kept) < 2:
            rejects["counterfactual_incomplete"] += 1
            return []
        return kept

    cat = _weighted(rnd, weights)
    trap_p, sv = trap_persona(cat, personas, rnd)
    if cat == "EXAMPLE_FACT_TRAP":
        if not trap_p:
            rejects["no_trap_persona"] += 1
            return out
        if gtype == "PERSONA_VARIATION":
            gtype = "SINGLE"   # the trap belongs to one persona's examples
    anchor = try_scenario(cat, split, pool, seed, nd=nd, sv=sv)
    if not anchor:
        rejects["no_scenario"] += 1
        return out
    _scn_add_meta(anchor, nd)
    _add_chitchat(anchor, rnd)
    if gtype == "PERSONA_VARIATION":
        for p in rnd.sample(personas, k=rnd.choice([2, 3])):
            emit(anchor, p, "PERSONA_VARIATION")
    elif gtype == "PARAPHRASE":
        persona = trap_p or rnd.choice(personas)
        emit(anchor, persona, "PARAPHRASE")
        if anchor["question_frame"] < 100 and anchor["domain"] in DOMAINS:
            d = DOMAINS[anchor["domain"]]
            sk = "self" if anchor["seed_values"]["subj"][0] == "self" else "third"
            if anchor["category"] in ("THIRD_PARTY", "MULTI_SUBJECT"):
                sk = "third"
            frames = [i for i, _ in d.frames("q", sk, "TRAIN" if split == "TRAIN" else "EVAL") if i != anchor["question_frame"]]
            for qf in rnd.sample(frames, k=min(len(frames), rnd.choice([1, 2]))):
                s = rebuild(anchor, anchor["category"], split, pool, qframe=qf, keep_sv=("subj", "v", "old", "third", "mode", "state", "pair", "trap"))
                if s and s["conversation"][-1]["text"] != anchor["conversation"][-1]["text"] \
                        and s["answer_state"]["required_values"] == anchor["answer_state"]["required_values"]:
                    emit(_scn_add_meta(s, nd), persona, "PARAPHRASE")
    else:
        emit(anchor, trap_p or rnd.choice(personas), "SINGLE")
    if len(out) == 1 and gtype != "SINGLE":
        out[0]["group_type"] = "SINGLE"
    return out


def build_train(personas: List[Persona], seed: str, n_episodes: int, data_version: str, *, budget_tokens: Optional[int] = None,
                measure=None) -> Tuple[List[dict], dict]:
    """Whole groups until ``n_episodes`` (or, with ``measure``, until the supervised-token budget) is reached; groups are never cut."""
    caps, rejects = Caps(TRAIN_TARGET_CAP), collections.Counter()
    eps: List[dict] = []
    gi, tokens = 0, 0
    while (tokens < budget_tokens) if budget_tokens else (len(eps) < n_episodes):
        rnd = random.Random(stable_seed(seed, "TRAIN", "gtype", gi))
        gtype = _weighted(rnd, GROUP_WEIGHTS)
        new = gen_group(gtype, "TRAIN", "TRAIN", personas, TRAIN_WEIGHTS, stable_seed(seed, "TRAIN", gi), caps, rejects, data_version)
        if measure:
            for e in new:
                e["supervised_tokens"] = measure(e["target"])
                tokens += e["supervised_tokens"]
        eps += new
        gi += 1
        if gi > (budget_tokens if budget_tokens else n_episodes * 6):  # safety stop (a group yields >= 1 token when it yields anything)
            break
    return eps, {"groups_tried": gi, "rejects": dict(rejects), "budget_tokens": budget_tokens, "tokens": tokens}


# ------------------------------------------------------------------------------------------------------------------------------ evaluation
WRONG_STATE = {"KNOWN_FACT": "FULL_UNKNOWN", "FULL_UNKNOWN": "KNOWN_FACT", "CORRECTION": "KNOWN_FACT", "TEMPORAL_CHANGE": "KNOWN_FACT",
               "UPCOMING": "KNOWN_FACT", "CANCELLATION": "KNOWN_FACT", "CONFLICT": "KNOWN_FACT", "MUST_NOT": "DISCLOSURE_ALLOWED",
               "DISCLOSURE_ALLOWED": "MUST_NOT", "POLICY_UNCERTAIN": "MUST_NOT", "OUTDATED_CONTACT": "DISCLOSURE_ALLOWED"}


def _guard_spec(scn: dict) -> dict:
    """Inputs for the external output guard (groups C / D): terms that must never appear as said."""
    k = scn["answer_state"]
    corrected_old = [e.replaces for e in scn["evidence"] if e.status == "CORRECTED" and e.replaces]
    return {"must_not_terms": sorted(set(k.get("protected_values", []))), "forbidden_values": sorted(set(k.get("forbidden_values", []))),
            "corrected_old_statements": corrected_old, "trap_values": [k["trap_value"]] if k.get("trap_value") else []}


def eval_item(scn: dict, persona: Persona, other: Persona, seed: str, split: str, pool: str, caps: Caps, rejects: collections.Counter,
              data_version: str) -> Optional[dict]:
    trap = scn["answer_state"].get("trap_value") if scn["category"] == "EXAMPLE_FACT_TRAP" else None
    try:
        ex = examples_for(persona, stable_seed(seed, "ex"), trap_city=trap)
        ex_o = examples_for(other, stable_seed(seed, "ex-swap"), trap_city=None)
    except NotApplicable:
        rejects["no_trap_line"] += 1
        return None
    ref = surface(scn, persona, ex, seed, caps, rejects)
    if ref is None:
        return None
    rec = episode_record(scn, persona, ex, ref, split=split, group_id="E-" + stable_seed(seed)[:12], group_type="EVAL", data_version=data_version)
    rec["episode_id"] = rec["episode_id"].replace("D4S-", "D4E-")
    rec["reference_target"] = rec.pop("target")
    rec["guard_spec"] = _guard_spec(scn)
    base = dict(profile=persona.profile, examples=ex, evidence=scn["evidence"], unknown=scn["unknown"], now=scn["now"], conversation=scn["conversation"])
    ab = {
        "NO_MEMORY": render_messages(**{**base, "evidence": None, "unknown": ()}),
        "NO_PROFILE": render_messages(**{**base, "profile": None}),
        "NO_EXAMPLES": render_messages(**{**base, "examples": []}),
        "SWAPPED_PERSONA": render_messages(**{**base, "profile": other.profile, "examples": ex_o}),
    }
    ab_keys: Dict[str, dict] = {}
    wcat = WRONG_STATE.get(scn["category"])
    if wcat:
        w = rebuild(scn, wcat, split, pool, qframe=scn["question_frame"] if scn["question_frame"] < 100 else None)
        if w and w["conversation"][-1]["text"] == scn["conversation"][-1]["text"]:
            ab["WRONG_MEMORY_STATE"] = render_messages(**{**base, "evidence": w["evidence"], "unknown": w["unknown"], "now": w["now"]})
            ab_keys["WRONG_MEMORY_STATE"] = {"category": wcat, "answer_state": w["answer_state"], "evidence": _ev_json(w["evidence"])}
    rec["ablations"] = {k: {"system": m[0]["content"], "user": m[1]["content"]} for k, m in ab.items()}
    if RD.base_length_enabled():
        bl = RD.base_length_messages(**base)
        rec["base_length"] = {"system": bl[0]["content"], "user": bl[1]["content"]}
    rec["ablation_answer_states"] = ab_keys
    rec["swapped_persona_id"] = other.style.persona_id
    return rec


def build_eval(split: str, pool: str, personas: List[Persona], seed: str, n: int, weights: Dict[str, float], data_version: str) -> Tuple[List[dict], dict]:
    caps, rejects = Caps(EVAL_TARGET_CAP), collections.Counter()
    out: List[dict] = []
    i = 0
    cats = list(weights)
    while len(out) < n and i < n * 6:
        rnd = random.Random(stable_seed(seed, split, "item", i))
        # stratified: cycle categories in proportion to weights
        cat = _weighted(rnd, weights)
        nd = rnd.choice([0, 1, 1, 2, 3])
        trap_p, sv = trap_persona(cat, personas, rnd)
        if cat == "EXAMPLE_FACT_TRAP" and not trap_p:
            i += 1
            rejects["no_trap_persona"] += 1
            continue
        scn = try_scenario(cat, split, pool, stable_seed(seed, split, i), nd=nd, sv=sv)
        i += 1
        if not scn:
            rejects["no_scenario"] += 1
            continue
        _scn_add_meta(scn, nd)
        _add_chitchat(scn, rnd)
        p, o = rnd.sample(personas, 2)
        if trap_p:
            p = trap_p
            o = rnd.choice([x for x in personas if x is not p])
        rec = eval_item(scn, p, o, stable_seed(seed, split, "e", i), split, pool, caps, rejects, data_version)
        if rec:
            out.append(rec)
    return out, {"items_tried": i, "rejects": dict(rejects), "categories": cats}
