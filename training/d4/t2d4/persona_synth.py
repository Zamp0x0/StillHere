"""Synthetic personas (SYNTHETIC_PERSONA): a style parameter set, a fact-free synthetic message history in that style, the T2-D3
Profiler v1 run on that history (vendored, unchanged), K=5 examples chosen by the T2-D3 example selector, and the D4 surface realiser that
turns a fact-equivalent content plan into this persona's way of saying it.

Nothing here is a real person's text. Histories are built from a fact-free chit-chat bank (attitudes, reactions, generic remarks); they
never carry the memory facts of an episode, so examples cannot leak answers. A small, labelled share of personas also has "example trap"
lines (an earlier, unrelated personal remark) used by EXAMPLE_FACT_TRAP scenarios to teach that examples are not facts.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

from .profile_v1 import build as build_profile_v1
from .vendor.t2d3_style.profiler import profile as profiler_v1, select_examples

PARTICLE_SETS = [[], ["呀", "呢"], ["啊", "吧"], ["嘛", "啦"], ["哦"], ["呗", "啊"], ["嗯"], ["呢"], ["吧"], ["啦", "呀"]]
ENDS = ["none", "period", "tilde", "exclaim", "mixed"]
REGISTERS = ["plain", "plain", "northern", "slang", "formal", "warm"]
LENGTHS = ["very_short", "short", "short", "medium", "long"]

# fact-free chit-chat bases, grouped by scenario type (the profiler's stability unit)
CHAT_BANK: Dict[str, List[str]] = {
    "weather": ["今天好冷", "这天说变就变", "外面下雨了", "太阳好大", "风好大", "终于不下雨了", "这雾霾也太重了", "热得不想动"],
    "food": ["中午吃什么好", "这家味道一般", "饿死了", "刚吃完饭", "好想吃辣的", "这个有点咸", "外卖还没到", "晚上随便吃点"],
    "work": ["今天事情好多", "又要开会", "终于下班了", "这周好忙", "明天再说吧", "先忙完这个", "改了一下午", "等会儿再看"],
    "tech": ["这个软件又卡了", "重启试试", "网好慢", "更新完就好了", "这个设置藏得好深", "换个浏览器试试", "电脑风扇好吵", "手机又没电了"],
    "reaction": ["真的假的", "可以可以", "笑死", "这也太离谱了", "还行吧", "确实", "有道理", "不至于吧", "好家伙", "没毛病", "也是", "那挺好"],
    "ask": ["你吃了没", "在干嘛", "怎么了", "啥时候的事", "你觉得呢", "然后呢", "那你怎么办", "要不要一起"],
    "plan": ["改天一起吃饭", "周末再约", "到时候再说", "我想想", "回头聊", "有空再看", "先这样", "晚点再弄"],
    "mood": ["有点困", "今天心情不错", "好累", "不想动", "还挺开心的", "有点烦", "算了不想了", "慢慢来吧"],
    "media": ["这剧有点拖", "这歌还挺好听", "那个视频你看了没", "最后那段好感人", "看到一半睡着了", "这游戏太难了"],
}
TRAP_BANK = ["我前阵子去了一趟{city}", "我那时候在{city}上班", "之前买过一个{item}", "我以前养过一只{pet}"]
LATIN = ["ok", "emm", "anyway", "fine", "nice", "orz", "lol"]
REGISTER_OPENERS = {"plain": ["嗯", "哦", "噢"], "northern": ["嗐", "害", "得嘞"], "slang": ["好家伙", "诶"], "formal": ["是的", "好的"],
                    "warm": ["嗯嗯", "好呀"]}
ADDONS = {  # attitudes / discourse with no facts (allowed by the no-new-specifics rule)
    "plain": ["还行吧", "就这样", "挺好的", "也就那样", "说来话长"],
    "northern": ["咋突然问这个", "还凑合", "那可不", "整挺好"],
    "slang": ["咋说呢", "害", "反正就这样", "就那样吧"],
    "formal": ["如有需要再联系", "感谢关心", "情况就是这样"],
    "warm": ["谢谢关心", "你也注意休息", "别担心"],
}
QUESTION_BACK = {"plain": ["你呢", "怎么了", "问这个干嘛"], "northern": ["咋了", "你问这干啥"], "slang": ["你呢", "怎么突然问"],
                 "formal": ["请问有什么事吗", "您有什么需要吗"], "warm": ["你最近还好吗", "怎么啦"]}


@dataclass
class Style:
    persona_id: str
    pool: str
    length: str
    particles: List[str]
    particle_rate: float
    end: str
    pause: str           # "comma" | "space"
    multi_bubble: float  # probability to split clauses into lines
    latin: float
    laugh: float
    question_back: float
    register: str
    kaomoji: float
    trap: bool = False

    def to_json(self) -> dict:
        return asdict(self)


def sample_style(persona_id: str, pool: str, rnd: random.Random) -> Style:
    return Style(persona_id=persona_id, pool=pool, length=rnd.choice(LENGTHS), particles=rnd.choice(PARTICLE_SETS),
                 particle_rate=rnd.choice([0.3, 0.5, 0.8]), end=rnd.choice(ENDS), pause=rnd.choice(["comma", "comma", "comma", "space"]),
                 multi_bubble=rnd.choice([0.0, 0.0, 0.15, 0.6]), latin=rnd.choice([0.0, 0.0, 0.1, 0.35]), laugh=rnd.choice([0.0, 0.0, 0.15, 0.4]),
                 question_back=rnd.choice([0.0, 0.1, 0.25]), register=rnd.choice(REGISTERS), kaomoji=rnd.choice([0.0, 0.0, 0.0, 0.08]),
                 trap=rnd.random() < 0.12)


def _end(style: Style, rnd: random.Random, question: bool = False) -> str:
    if question:
        return "？" if style.end != "none" or rnd.random() < 0.5 else ""
    e = style.end if style.end != "mixed" else rnd.choice(["none", "period", "exclaim"])
    return {"none": "", "period": "。", "tilde": "～", "exclaim": "！"}[e]


def _particle(style: Style, rnd: random.Random, clause: str) -> str:
    if style.particles and rnd.random() < style.particle_rate and clause and clause[-1] not in "呀呢啊吧嘛啦哦呗嗯了":
        return clause + rnd.choice(style.particles)
    return clause


def realise(clauses: Sequence[str], style: Style, rnd: random.Random, *, allow_addons: bool = True, allow_question_back: bool = True) -> str:
    """Surface form of a content plan in this persona's way of speaking. ``clauses`` carry ALL the content (fact-equivalent across personas);
    everything added here is fact-free (openers, particles, punctuation, attitudes, question-backs, laughter, code-mixed fillers)."""
    cl = [c for c in clauses if c]
    if not cl:
        raise ValueError("empty plan")
    cl = [_particle(style, rnd, c) for c in cl]
    pre = []
    if style.register in REGISTER_OPENERS and rnd.random() < (0.25 if style.length != "very_short" else 0.1):
        pre.append(rnd.choice(REGISTER_OPENERS[style.register]))
    if style.latin and rnd.random() < style.latin:
        pre.append(rnd.choice(LATIN))
    post = []
    add_p = {"very_short": 0.0, "short": 0.1, "medium": 0.35, "long": 0.65}[style.length]
    if allow_addons and rnd.random() < add_p:
        post.append(rnd.choice(ADDONS[style.register]))
    if style.laugh and rnd.random() < style.laugh:
        post.append(rnd.choice(["哈哈", "哈哈哈", "hhh"]))
    q = None
    if allow_question_back and style.question_back and rnd.random() < style.question_back:
        q = rnd.choice(QUESTION_BACK[style.register])
    body = cl + post
    sep = "\n" if (len(body) > 1 and rnd.random() < style.multi_bubble) else (" " if style.pause == "space" else "，")
    if sep == "\n":
        text = "\n".join(b + _end(style, rnd) for b in body)
    else:
        text = sep.join(body) + _end(style, rnd)
    if pre:
        opener = "，".join(pre) if style.pause == "comma" else " ".join(pre)
        text = opener + ("\n" if sep == "\n" else ("，" if style.pause == "comma" else " ")) + text
    if q:
        text = text + ("\n" if sep == "\n" else ("" if text[-1:] in "。！～？" else "，")) + q + _end(style, rnd, question=True)
    if style.kaomoji and rnd.random() < style.kaomoji:
        text += rnd.choice(["(｀・ω・´)", "(●'◡'●)", "(╯▽╰)"])
    text = re.sub(r"[，,]{2,}", "，", text).strip()
    return text


def history(style: Style, rnd: random.Random, n: int = 30, trap_values: Optional[Dict[str, str]] = None) -> List[dict]:
    """``n`` synthetic earlier messages of this persona, spread over scenario types (fact-free; trap lines only for trap personas)."""
    msgs = []
    types = list(CHAT_BANK)
    for i in range(n):
        st = types[i % len(types)] if i < len(types) else rnd.choice(types)
        k = 1 if style.length in ("very_short", "short") else rnd.choice([1, 2])
        base = rnd.sample(CHAT_BANK[st], k=min(k, len(CHAT_BANK[st])))
        if style.trap and trap_values and i in (3, 11):
            bank = [x for x in TRAP_BANK if "{city}" in x] if i == 3 else TRAP_BANK  # the first trap line always names a city
            base = [rnd.choice(bank).format(**trap_values)]
            st = "trap"
        text = realise(base, style, rnd, allow_addons=style.length in ("medium", "long"), allow_question_back=st != "ask")
        msgs.append({"message_id": f"{style.persona_id}:h{i:02d}", "bubbles": [b for b in text.split("\n") if b.strip()], "scenario_type": st})
    return msgs


@dataclass
class Persona:
    style: Style
    history: List[dict]
    profiler_out: dict
    profile: dict
    examples: List[str] = field(default_factory=list)
    trap_values: Optional[Dict[str, str]] = None

    def pick_examples(self, seed: str, exclude: Sequence[str] = (), k: int = 5) -> List[str]:
        chosen = select_examples(self.history, k, seed=seed, exclude_text=[[x] for x in exclude])
        return ["\n".join(m["bubbles"]) for m in chosen]


def make_persona(persona_id: str, pool: str, seed: str, trap_values: Optional[Dict[str, str]] = None) -> Persona:
    rnd = random.Random(hashlib.sha256(f"{seed}:{persona_id}".encode()).hexdigest())
    st = sample_style(persona_id, pool, rnd)
    h = history(st, rnd, 30, trap_values if st.trap else None)
    po = profiler_v1(h)
    pr = build_profile_v1(po, persona_id=persona_id, persona_kind="SYNTHETIC_PERSONA", history_before="synthetic")
    return Persona(style=st, history=h, profiler_out=po, profile=pr, trap_values=trap_values if st.trap else None)
