"""Scenario generator: memory states x slot domains x subjects -> evidence, question, conversation, answer key and content plan.

The answer key (``answer_state``) is derived from the evidence and the question's semantics, never from the realised text, and the grounding
validator checks every target against it. The content plan (``clauses``) is fact-equivalent across personas; the persona realiser only adds
fact-free surface material, so PERSONA_VARIATION siblings share one plan and differ in style only.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import random
from typing import Dict, List, Optional, Tuple

from .domains import DOMAINS, PARTIAL_PAIRS, PROTECTED, Domain
from .evidence import Evidence, UnknownSlot, check_conflict_groups
from .lexicon import BANKS, RELATIONS

CATEGORIES = ("KNOWN_FACT", "FULL_UNKNOWN", "PARTIAL_UNKNOWN", "TEMPORAL_CHANGE", "HISTORICAL_ASK", "UPCOMING", "CORRECTION", "CANCELLATION",
              "CONFLICT", "THIRD_PARTY", "MULTI_SUBJECT", "MUST_NOT", "DISCLOSURE_ALLOWED", "POLICY_UNCERTAIN", "OUTDATED_CONTACT",
              "CONVERSATION_DEPENDENCE", "EXAMPLE_FACT_TRAP")
# which B3 failure / user scenario family each category covers
FAMILY = {"KNOWN_FACT": "A_KNOWN", "FULL_UNKNOWN": "B_FULL_UNKNOWN", "PARTIAL_UNKNOWN": "C_PARTIAL_UNKNOWN", "TEMPORAL_CHANGE": "D_TEMPORAL",
          "HISTORICAL_ASK": "D_TEMPORAL", "UPCOMING": "D_TEMPORAL", "CORRECTION": "D_CORRECTION", "CANCELLATION": "D_CANCELLATION",
          "CONFLICT": "D_CONFLICT", "THIRD_PARTY": "E_THIRD_PARTY", "MULTI_SUBJECT": "E_THIRD_PARTY", "MUST_NOT": "F_MUST_NOT",
          "DISCLOSURE_ALLOWED": "F_DISCLOSURE", "POLICY_UNCERTAIN": "F_DISCLOSURE", "OUTDATED_CONTACT": "F_DISCLOSURE",
          "CONVERSATION_DEPENDENCE": "G_CONVERSATION", "EXAMPLE_FACT_TRAP": "H_EXAMPLE_TRAP"}
WHEN_TRIP = ["下周", "下个月", "月底", "这周末", "过两天"]
UPCOMING_WHEN = ["下个月", "月底", "明年初", "过完年"]


class NotApplicable(Exception):
    pass


def _iso(d: dt.date) -> str:
    return d.isoformat()


class Ctx:
    """One scenario instance: pool, rng, conversation date, slot extras (item, pet kind, concert, clinic, trip time word)."""

    def __init__(self, pool: str, rnd: random.Random):
        self.pool, self.rnd = pool, rnd
        self.now = dt.date(2026, rnd.randint(1, 12), rnd.randint(1, 28))
        self.extra = {"w": rnd.choice(WHEN_TRIP), "it": rnd.choice(BANKS["item"][pool]), "k": rnd.choice(BANKS["pet_kind"][pool]),
                      "c": rnd.choice(BANKS["concert"][pool]), "cl": rnd.choice(BANKS["clinic"][pool]), "when": rnd.choice(UPCOMING_WHEN)}
        self._fid = 0
        self.used_values = set()
        self.q_list = "-"

    def fid(self) -> str:
        self._fid += 1
        return f"F{self._fid}"

    def value(self, d: Domain, avoid=()) -> str:
        for _ in range(50):
            v = d.value(self.pool, self.rnd)
            if v not in avoid and v not in self.used_values:
                self.used_values.add(v)
                return v
        raise NotApplicable(f"value bank exhausted for {d.id}")

    def person(self, avoid=()) -> Tuple[str, str, str]:
        """(subject_type, label shown in evidence, name used in speech)"""
        st = self.rnd.choice(["friend", "friend", "colleague", "family", "spouse", "neighbor"])
        rel = self.rnd.choice(RELATIONS[st])
        for _ in range(50):
            name = self.rnd.choice(BANKS["person"][self.pool])
            if name not in avoid and name not in self.used_values:
                self.used_values.add(name)
                break
        if rel in ("老婆", "老公", "对象", "我妈", "我爸"):
            return st, rel, rel.replace("我", "")  # spoken as 老婆 / 妈
        return st, f"{rel}{name}", name

    def fmt(self, frame: str, **kw) -> str:
        return frame.format(**{**self.extra, **kw})

    def past_window(self) -> Tuple[str, str]:
        a = self.now - dt.timedelta(days=self.rnd.randint(400, 1500))
        b = self.now - dt.timedelta(days=self.rnd.randint(20, 380))
        return _iso(a), _iso(b)

    def since(self) -> str:
        return _iso(self.now - dt.timedelta(days=self.rnd.randint(10, 900)))

    def future(self) -> str:
        return _iso(self.now + dt.timedelta(days=self.rnd.randint(15, 120)))


def _ev(c: Ctx, d: Domain, subj: Tuple[str, str, str], v: str, status: str = "CURRENT", **kw) -> Evidence:
    st, label, _ = subj
    text = c.fmt(d.ev["self" if st == "self" else "third"], n=label, v=v)
    if status == "CURRENT" and "valid_from" not in kw and d.kind in ("fact", "multi", "protected"):
        kw["valid_from"] = c.since()
    e = Evidence(fact_id=c.fid(), subject="本人" if st == "self" else label, subject_type=st, predicate=d.id, value=v, evidence_text=text,
                 status=status, observed_at=_iso(c.now - dt.timedelta(days=c.rnd.randint(1, 60))), **kw)
    e.validate()
    return e


SELF = ("self", "本人", "我")


def _distractors(c: Ctx, exclude: set, n: int) -> List[Evidence]:
    """Unrelated, disclosable facts (self or third party) the reply must not need: evidence selection practice."""
    pool = [k for k, d in DOMAINS.items() if k not in exclude and d.kind != "protected" and k not in ("car", "major")]
    out = []
    for k in c.rnd.sample(pool, k=min(n, len(pool))):
        d = DOMAINS[k]
        subj = SELF if (c.rnd.random() < 0.5 and "self" in d.subjects) else c.person()
        try:
            v = c.value(d)
        except NotApplicable:
            continue
        status = "HISTORICAL" if (d.kind == "fact" and c.rnd.random() < 0.25) else "CURRENT"
        kw = {}
        if status == "HISTORICAL":
            kw["valid_from"], kw["valid_to"] = c.past_window()
        out.append(_ev(c, d, subj, v, status, **kw))
    return out


def _key(category: str, **kw) -> dict:
    base = {"category": category, "family": FAMILY[category], "required_values": [], "allowed_values": [], "forbidden_values": [],
            "marker_values": {}, "must_express": [], "third_party_names": [], "self_claim_forbidden_values": []}
    base.update(kw)
    return base


def _pick(c: Ctx, lst: List[str]) -> Tuple[int, str]:
    i = c.rnd.randrange(len(lst))
    return i, lst[i]


def _question(d: Domain, which: str, subj_key: str, split_kind: str, c: Ctx, forced: Optional[int] = None) -> Tuple[int, str]:
    c.q_list = f"{which}:{subj_key}"  # the frame list the question came from (template identity = domain + list + index)
    fr = d.frames(which, subj_key, split_kind)
    if not fr:
        raise NotApplicable(f"{d.id}:{which}:{subj_key}")
    if forced is not None:
        fr = [x for x in fr if x[0] == forced] or fr
    return c.rnd.choice(fr)


def _split_frames(lst: Optional[List[str]], split_kind: str) -> List[Tuple[int, str]]:
    """Same TRAIN / EVAL rule as Domain.frames for the other question lists (who / self-ask / elliptic); a single frame is TRAIN only."""
    lst = list(enumerate(lst or []))
    if len(lst) < 2:
        return lst if split_kind == "TRAIN" else []
    n_eval = max(1, len(lst) // 3)
    return lst[:-n_eval] if split_kind == "TRAIN" else lst[-n_eval:]


def _q(c: "Ctx", qi_q: Tuple[int, str], **kw) -> Tuple[int, str]:
    return qi_q[0], c.fmt(qi_q[1], **kw)


def build(category: str, dom_id: str, c: Ctx, split_kind: str, *, subj_kind: Optional[str] = None, qframe: Optional[int] = None,
          seed_values: Optional[dict] = None, n_distract: Optional[int] = None) -> dict:
    """One scenario (without persona surface). ``split_kind`` TRAIN|EVAL selects question frames. Raises NotApplicable for impossible combos."""
    d = DOMAINS[dom_id]
    sv = dict(seed_values or {})
    c.used_values.update(x for x in sv.values() if isinstance(x, str))  # seeded values are taken: no other slot may reuse them
    nd = c.rnd.choice([0, 1, 1, 2, 3]) if n_distract is None else n_distract
    conv_pre: List[dict] = []
    unknown: List[UnknownSlot] = []
    sk = subj_kind or ("self" if "self" in d.subjects and c.rnd.random() < 0.6 else "third")
    if sk not in d.subjects and category not in ("THIRD_PARTY", "MULTI_SUBJECT"):
        raise NotApplicable("subject")
    subj = SELF if sk == "self" else (sv.get("subj") or c.person())
    sv["subj"] = subj
    n = subj[2]
    ev: List[Evidence] = []
    ak = None
    aframes: List[int] = []

    def answer(frames: List[str], need_name: bool = False, **kw) -> str:
        if need_name:
            frames = [f for f in frames if "{n}" in f] or frames
        i, f = _pick(c, frames)
        aframes.append(i)
        return c.fmt(f, n=n, **kw)

    if category == "KNOWN_FACT":
        if d.kind == "protected":
            raise NotApplicable("protected")
        v = sv["v"] = sv.get("v") or c.value(d)
        ev.append(_ev(c, d, subj, v))
        qi, q = _q(c, _question(d, "q", sk, split_kind, c, qframe), n=n)
        clauses = [answer(d.a[sk], v=v)]
        ak = _key(category, required_values=[v], third_party_names=[] if sk == "self" else [n])
    elif category == "FULL_UNKNOWN":
        frames = d.unknown.get(sk)
        if not frames:
            raise NotApplicable("no unknown frames")
        qi, q = _q(c, _question(d, "q", sk, split_kind, c, qframe), n=n)
        if c.rnd.random() < 0.6:
            unknown.append(UnknownSlot(slot_id="U1", subject="本人" if sk == "self" else subj[1], predicate=d.id,
                                       label=(c.fmt(d.noun) if sk == "self" else f"{subj[1]}的{d.noun}")))
        clauses = [answer(frames, noun=d.noun)]
        ak = _key(category, must_express=["UNKNOWN"])
    elif category == "PARTIAL_UNKNOWN":
        pair = sv.get("pair") or c.rnd.choice([p for p in PARTIAL_PAIRS if p[0] == dom_id] or PARTIAL_PAIRS)
        d1, d2 = DOMAINS[pair[0]], DOMAINS[pair[1]]
        if "self" not in d1.subjects:
            raise NotApplicable("partial self")
        subj, sk, n = SELF, "self", "我"
        v1 = sv["v"] = sv.get("v") or c.value(d1)
        ev.append(_ev(c, d1, subj, v1))
        unknown.append(UnknownSlot(slot_id="U1", subject="本人", predicate=d2.id, label=c.fmt(d2.noun)))
        frames = pair[2]
        n_eval = max(1, len(frames) // 3)
        lst = list(enumerate(frames))
        lst = lst[:-n_eval] if split_kind == "TRAIN" else lst[-n_eval:]
        qi, qf = c.rnd.choice(lst)
        q = c.fmt(qf)
        clauses = [answer(d1.a["self"], v=v1), answer(d2.unknown["self"], noun=d2.noun)]
        ak = _key(category, required_values=[v1], must_express=["UNKNOWN"])
        dom_id = f"{d1.id}+{d2.id}"
    elif category in ("TEMPORAL_CHANGE", "HISTORICAL_ASK"):
        if not d.change:
            raise NotApplicable("no change")
        old = sv["old"] = sv.get("old") or c.value(d)
        v = sv["v"] = sv.get("v") or c.value(d, avoid={old})
        a, b = c.past_window()
        ev.append(_ev(c, d, subj, old, "HISTORICAL", valid_from=a, valid_to=b))
        ev.append(_ev(c, d, subj, v, "CURRENT", valid_from=b))
        if category == "TEMPORAL_CHANGE":
            if d.q_still and c.rnd.random() < 0.6:
                qi, qf = _question(d, "q_still", sk, split_kind, c, qframe)
                q = c.fmt(qf, n=n, old=old)
                qi += 100
            else:
                qi, q = _q(c, _question(d, "q", sk, split_kind, c, qframe), n=n)
            clauses = [answer(d.change[sk], v=v, old=old)]
            ak = _key(category, required_values=[v], marker_values={old: "PAST"}, third_party_names=[] if sk == "self" else [n])
        else:
            if not (d.q_past and d.a_past and d.q_past.get(sk)):
                raise NotApplicable("no past q")
            qi, q = _q(c, _question(d, "q_past", sk, split_kind, c, qframe), n=n)
            qi += 200
            clauses = [answer(d.a_past[sk], v=old)]
            ak = _key(category, required_values=[old], must_express=["PAST"], allowed_values=[v], third_party_names=[] if sk == "self" else [n])
    elif category == "UPCOMING":
        if not d.upcoming:
            raise NotApplicable("no upcoming")
        old = sv["old"] = sv.get("old") or c.value(d)
        v = sv["v"] = sv.get("v") or c.value(d, avoid={old})
        when = c.extra["when"]
        ev.append(_ev(c, d, subj, old, "CURRENT", valid_from=c.since()))
        e2 = _ev(c, d, subj, v, "UPCOMING", valid_from=c.future())
        ev.append(Evidence(**{**e2.to_json(), "evidence_text": e2.evidence_text + f"（定在{when}）"}))
        if d.q_still and c.rnd.random() < 0.5:
            qi, qf = _question(d, "q_still", sk, split_kind, c, qframe)
            q, qi = c.fmt(qf, n=n, old=old), qi + 100
        else:
            qi, q = _q(c, _question(d, "q", sk, split_kind, c, qframe), n=n)
        clauses = [answer(d.upcoming[sk], v=v, old=old, when=when)]
        req = [old, v] if old in clauses[0] else [v]
        ak = _key(category, required_values=req, allowed_values=[old], marker_values={v: "FUTURE"}, must_express=["FUTURE"],
                  third_party_names=[] if sk == "self" else [n])
    elif category == "CORRECTION":
        if not d.correction or d.kind == "protected":
            raise NotApplicable("no correction")
        old = sv["old"] = sv.get("old") or c.value(d)
        v = sv["v"] = sv.get("v") or c.value(d, avoid={old})
        ev.append(_ev(c, d, subj, v, "CORRECTED", replaces=c.fmt(d.ev["self" if sk == "self" else "third"], n=subj[1], v=old)))
        qi, q = _q(c, _question(d, "q", sk, split_kind, c, qframe), n=n)
        clauses = [answer(d.a[sk], v=v)]
        ak = _key(category, required_values=[v], forbidden_values=[old], third_party_names=[] if sk == "self" else [n])
    elif category == "CANCELLATION":
        if not d.cancel or not d.cancel.get(sk):
            raise NotApplicable("no cancel")
        v = sv["v"] = sv.get("v") or c.value(d)
        cd = c.now - dt.timedelta(days=c.rnd.randint(1, 20))
        ev.append(_ev(c, d, subj, v, "CANCELLED", note=f"{cd.month}月{cd.day}号已经取消", valid_to=_iso(cd)))
        qi, q = _q(c, _question(d, "q", sk, split_kind, c, qframe), n=n)
        clauses = [answer(d.cancel[sk], v=v)]
        ak = _key(category, marker_values={v: "CANCELLED"}, must_express=["CANCELLED"], third_party_names=[] if sk == "self" else [n])
    elif category == "CONFLICT":
        if not d.conflict:
            raise NotApplicable("no conflict")
        v1 = sv["v"] = sv.get("v") or c.value(d)
        v2 = c.value(d, avoid={v1})
        for v in (v1, v2):
            ev.append(_ev(c, d, subj, v, "CONFLICTED", conflict_group="G1", confidence="medium"))
        qi, q = _q(c, _question(d, "q", sk, split_kind, c, qframe), n=n)
        if sk == "self":
            frames = ["好像是{a}还是{b}来着，记不太清了", "{a}或者{b}吧，我也说不准", "这个有两个说法，{a}和{b}，还没确认", "记得不太清，可能是{a}，也可能是{b}"]
        else:  # the uncertainty is the speaker's, never the third party's ("老许记得不太清" would mean 老许 forgot)
            frames = ["{n}的话，好像是{a}还是{b}来着，我记不太清了", "{n}是{a}还是{b}，我也说不准", "{n}这个有两个说法，{a}和{b}，还没确认",
                      "{n}可能是{a}，也可能是{b}，我记不清了"]
            frames = [x.replace("{n}", n) for x in frames]
        i, f = _pick(c, frames)
        aframes.append(i)
        a_, b_ = (v1, v2) if c.rnd.random() < 0.5 else (v2, v1)
        clauses = [c.fmt(f, a=a_, b=b_)]
        ak = _key(category, marker_values={v1: "UNCERTAIN", v2: "UNCERTAIN"}, must_express=["UNCERTAIN"], third_party_names=[] if sk == "self" else [n])
    elif category == "THIRD_PARTY":
        if "third" not in d.a or d.kind == "protected":
            raise NotApplicable("no third")
        subj = sv.get("third") or c.person()
        sv["third"] = subj
        n = subj[2]
        v = sv["v"] = sv.get("v") or c.value(d)
        ev.append(_ev(c, d, subj, v))
        who_f, self_f = _split_frames(d.who, split_kind), _split_frames(d.self_ask_third, split_kind)
        mode = sv.get("mode") or c.rnd.choice([m for m, ok in (("who", bool(who_f)), ("self_ask", bool(self_f)),
                                                                 ("same_q", d.kind in ("plan", "number") and bool(d.unknown.get("self")))) if ok] or ["direct"])
        if mode == "who":
            if not who_f:
                raise NotApplicable("who frames")
            i, f = c.rnd.choice(who_f)
            qi, q = 300 + i, c.fmt(f, v=v)
            clauses = [answer(d.a["third"], need_name=True, v=v)]
        elif mode == "self_ask":
            if not self_f:
                raise NotApplicable("self-ask frames")
            i, f = c.rnd.choice(self_f)
            qi, q = 400 + i, c.fmt(f, v=v)
            lead = c.rnd.choice(["是{n}", "那是{n}", "说的是{n}吧"])
            clauses = [c.fmt(lead, n=n), answer(d.a["third"], need_name=True, v=v)]
        elif mode == "same_q":
            qi, q = _q(c, _question(d, "q", "self", split_kind, c, qframe), n=n)
            clauses = [answer(d.unknown["self"], noun=d.noun), c.fmt(c.rnd.choice(["{n}那边是{v}", "倒是{n}定了{v}", "{v}是{n}的"]), n=n, v=v)]
            ak_extra = {"must_express": ["UNKNOWN"]}
        else:
            qi, q = _q(c, _question(d, "q", "third", split_kind, c, qframe), n=n)
            clauses = [answer(d.a["third"], need_name=True, v=v)]
        ak = _key(category, required_values=[v, n], third_party_names=[n], self_claim_forbidden_values=[v], subtype=mode)
        if mode == "same_q":
            ak["must_express"] = ["UNKNOWN"]
        sk = "third"
    elif category == "MULTI_SUBJECT":
        if "third" not in d.a or d.kind == "protected":
            raise NotApplicable("no multi")
        st1, l1, name = c.person()
        rel2 = c.rnd.choice([r for rs in RELATIONS.values() for r in rs if r not in ("老婆", "老公", "对象", "我妈", "我爸") and not l1.startswith(r)])
        subj1 = (st1, l1, l1 if l1 != name else name)
        subj2 = ("friend", f"{rel2}{name}", f"{rel2}{name}")
        if l1 in ("老婆", "老公", "对象", "我妈", "我爸"):
            raise NotApplicable("relation label")
        v1, v2 = c.value(d), None
        v2 = c.value(d, avoid={v1})
        ev.append(_ev(c, d, subj1, v1))
        ev.append(_ev(c, d, subj2, v2))
        qi, qf = _question(d, "q", "third", split_kind, c, qframe)
        q = c.fmt(qf, n=l1)
        named = [f for f in d.a["third"] if "{n}" in f] or d.a["third"]
        i, f = _pick(c, named)
        aframes.append(i)
        clauses = [c.fmt(f, n=l1, v=v1)]
        ak = _key(category, required_values=[v1], forbidden_values=[v2], third_party_names=[l1])
        sk, n = "third", l1
    elif category in ("MUST_NOT", "DISCLOSURE_ALLOWED", "POLICY_UNCERTAIN", "OUTDATED_CONTACT"):
        if d.kind != "protected":
            raise NotApplicable("not protected")
        if sk not in d.subjects:
            raise NotApplicable("subject")
        v = sv["v"] = sv.get("v") or c.value(d)
        qi, q = _q(c, _question(d, "q", sk, split_kind, c, qframe), n=n)
        if category == "MUST_NOT":
            ev.append(_ev(c, d, subj, v, disclosure_policy="PROTECTED"))
            clauses = [answer(d.decline[sk])]
            ak = _key(category, forbidden_values=[v], must_express=["DECLINE"])
        elif category == "POLICY_UNCERTAIN":
            if sk == "self":
                raise NotApplicable("self uncertain")
            ev.append(_ev(c, d, subj, v, disclosure_policy="POLICY_UNCERTAIN"))
            clauses = [answer(d.decline["uncertain"])]
            ak = _key(category, forbidden_values=[v], must_express=["PERMISSION"])
        elif category == "DISCLOSURE_ALLOWED":
            if d.id not in ("phone", "home_address"):
                raise NotApplicable("allowed only for contact")
            e = _ev(c, d, subj, v, disclosure_policy="ALLOWED")
            note = "（本人同意告诉对方）" if sk == "self" else f"（{subj[1]}同意把这个告诉对方）"
            ev.append(Evidence(**{**e.to_json(), "evidence_text": e.evidence_text + note}))
            clauses = [answer(d.a[sk], v=v)]
            ak = _key(category, required_values=[v])
        else:  # OUTDATED_CONTACT: only an old, no longer used number is recorded
            if d.id != "phone" or sk == "self":
                raise NotApplicable("outdated phone only")
            a, b = c.past_window()
            e = _ev(c, d, subj, v, "HISTORICAL", valid_from=a, valid_to=b)
            ev.append(Evidence(**{**e.to_json(), "evidence_text": e.evidence_text + "（这个号已经停用）"}))
            clauses = [answer(["{n}换号了，新号我这没有", "那个号{n}早不用了，新号我不知道", "{n}的旧号停了，新的我没有"])]
            ak = _key(category, forbidden_values=[v], must_express=["UNKNOWN"])
        if c.rnd.random() < 0.5 and category != "DISCLOSURE_ALLOWED":
            # an answerable side question next to the protected one: the reply must answer it (no blanket refusal)
            d2 = DOMAINS["home_city"]
            v2 = c.value(d2)
            ev.append(_ev(c, d2, subj, v2))
            side = d2.frames("q", sk, split_kind)
            if not side:
                raise NotApplicable("side question")
            si, sq = c.rnd.choice(side)
            q = q.rstrip("？?") + "，" + c.fmt(sq, n=n)
            qi = qi + 1000 * (si + 1)
            clauses = [c.fmt(d2.a[sk][0], n=n, v=v2)] + clauses
            ak["required_values"] = ak["required_values"] + [v2]
            ak["subtype"] = "with_answerable_part"
        ak["third_party_names"] = [] if sk == "self" else [n]
    elif category == "CONVERSATION_DEPENDENCE":
        if not d.elliptic or "self" not in d.subjects:
            raise NotApplicable("no elliptic")
        subj, sk, n = SELF, "self", "我"
        # the earlier turn names the topic; the follow-up question only makes sense with it
        if d.id in ("hotel", "trip_date"):
            dt_ = DOMAINS["trip_city"]
            v0 = c.value(dt_)
            ev.append(_ev(c, dt_, SELF, v0))
            lead = c.fmt(dt_.lead["self"][0], v=v0)
        elif d.lead:
            lead = c.fmt(d.lead["self"][0], v="")
            ev.append(Evidence(fact_id=c.fid(), subject="本人", subject_type="self", predicate=f"{d.id}_context", value=lead.replace("我", "", 1),
                               evidence_text="本人" + lead.replace("我", "", 1), status="CURRENT", valid_from=c.since(),
                               observed_at=_iso(c.now - dt.timedelta(days=c.rnd.randint(1, 30)))))
        else:
            raise NotApplicable("no lead")
        state = sv.get("state") or c.rnd.choice(["known", "unknown"])
        conv_pre = [{"speaker": "partner", "text": c.rnd.choice(["最近忙啥呢？", "在干嘛呢", "这几天咋样？", "好久没聊了"])},
                    {"speaker": "self", "text": lead}]
        ell = _split_frames(d.elliptic, split_kind)
        if not ell:
            raise NotApplicable("elliptic frames")
        i, q = c.rnd.choice(ell)
        qi = 500 + i
        if state == "known":
            v = sv["v"] = sv.get("v") or c.value(d)
            ev.append(_ev(c, d, SELF, v))
            clauses = [answer(d.a["self"], v=v)]
            ak = _key(category, required_values=[v], subtype="known")
        else:
            has_lead_fact = d.id not in ("hotel", "trip_date")  # the lead turn itself states an event (appointment, ticket, purchase)
            label = d.lead_unknown_label if (has_lead_fact and d.lead_unknown_label) else c.fmt(d.noun)
            unknown.append(UnknownSlot(slot_id="U1", subject="本人", predicate=d.id, label=label))
            clauses = [answer(d.lead_unknown if (has_lead_fact and d.lead_unknown) else d.unknown["self"], noun=d.noun)]
            ak = _key(category, must_express=["UNKNOWN"], subtype="unknown")
    elif category == "EXAMPLE_FACT_TRAP":
        if d.id not in ("home_city", "trip_city"):
            raise NotApplicable("trap domains")
        subj, sk, n = SELF, "self", "我"
        trap = sv.get("trap")
        v = sv["v"] = sv.get("v") or c.value(d, avoid={trap} if trap else ())
        if not trap:
            trap = c.value(d, avoid={v})
        sv["trap"] = trap
        ev.append(_ev(c, d, SELF, v))
        qi, q = _q(c, _question(d, "q", "self", split_kind, c, qframe), n=n)
        clauses = [answer(d.a["self"], v=v)]
        ak = _key(category, required_values=[v], forbidden_values=[trap], trap_value=trap)
    else:
        raise ValueError(category)

    if category not in ("CONVERSATION_DEPENDENCE",):
        if c.rnd.random() < 0.2:
            conv_pre = [{"speaker": "partner", "text": c.rnd.choice(["在吗", "忙不忙", "问你个事", "诶"])},
                        {"speaker": "self", "text": c.rnd.choice(["在", "咋了", "你说", "嗯"])}]
    exclude = {d.id} | {e.predicate for e in ev}
    ev += _distractors(c, exclude, nd)
    c.rnd.shuffle(ev)
    # renumber fact ids in display order so position does not leak the answer
    ren = {e.fact_id: f"F{i + 1}" for i, e in enumerate(ev)}
    ev = [Evidence(**{**e.to_json(), "fact_id": ren[e.fact_id]}) for e in ev]
    check_conflict_groups(ev)
    allowed = sorted({e.value for e in ev if e.disclosure_policy == "ALLOWED" and e.status in ("CURRENT", "UPCOMING", "CORRECTED")})
    ak["allowed_values"] = sorted(set(ak["allowed_values"]) | set(allowed))
    ak["protected_values"] = sorted({e.value for e in ev if e.disclosure_policy != "ALLOWED"})
    ak["forbidden_values"] = sorted(set(ak["forbidden_values"]) | set(ak["protected_values"]))
    ak["historical_values"] = sorted({e.value for e in ev if e.status in ("HISTORICAL", "CANCELLED")} - set(ak["required_values"]))
    conversation = conv_pre + [{"speaker": "partner", "text": q}]
    return {"category": category, "domain": dom_id, "subject_type": subj[0] if category not in ("THIRD_PARTY", "MULTI_SUBJECT") else "third",
            "now": _iso(c.now), "evidence": ev, "unknown": unknown, "conversation": conversation, "question_frame": qi, "answer_frames": aframes,
            "clauses": clauses, "answer_state": ak, "seed_values": sv,
            "question_template": f"{dom_id}|{c.q_list if qi < 300 or qi >= 1000 else 'special'}|{qi}"}


def template_family(s: dict) -> str:
    return f"{s['domain']}|{s['category']}|q{s['question_frame']}|a{'-'.join(map(str, s['answer_frames']))}"


def semantic_key(s: dict) -> str:
    """The semantic scenario: what is being tested, independent of names, values and surface."""
    ev = s["evidence"]
    sig = sorted(f"{e.subject_type}:{e.predicate}:{e.status}:{e.disclosure_policy}" for e in ev)
    return "|".join([s["category"], s["domain"], s["subject_type"], str(len(s["conversation"])), f"unk{len(s['unknown'])}",
                     s["answer_state"].get("subtype", ""), ";".join(sig)])


def stable_seed(*parts) -> str:
    return hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()
