"""Batch 2.1 last-fix confirmation world (``b21g-dataset-7``): three NEW invented personas, written and frozen (manifest hash) BEFORE the
multi-valued correction rule was written and before any run on it. Memory is written by the REAL extractor (DeepSeek, prompt v4) with the
final writer. Focus: an explicit correction of a MULTI-valued slot (pet, relationship, likes) against its controls: an ADDITION ("又养了",
"还有个同事"), a CHANGE OVER TIME ("以前…现在…"), a single-valued correction (regression), and plain facts / unknowns.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import List

from stillhere.contracts import CanonicalMessage, DataOrigin, SourceKind, SourceRef

from .writer_dataset import C, Checkpoint, Conv, Fact, ObsExpect  # noqa: F401  (same Gold format)

VERSION = "b21g-dataset-7"
TZ = timezone(timedelta(hours=8))
YA, YB, YC = "syn.ya", "syn.yb", "syn.yc"
NAMES = {YA: "秦朗", YB: "白露", YC: "孟川"}
BOT = {p: f"{p}.assistant" for p in NAMES}
CONTACTS: dict = {}
SPEAKERS = {**{b: "AI 助手" for b in BOT.values()}, **CONTACTS}
CHAT, SMS = "b21g.src.chat", "b21g.src.sms"


def t(month: int, day: int, hour: int = 10, minute: int = 0) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=TZ)


def conversations() -> List[Conv]:
    return [
        # ---------------- 秦朗 (YA): 南京，药剂师; pet correction + pet addition
        C("g01", YA, t(9, 2, 21), [("u", "我在南京一家医院当药剂师。"), ("a", "好的。")],
          [Fact("ya.job", "self", "job", "药剂师", (0,), ("在南京一家医院当药剂师",), ())], tags=("simple",)),
        C("g02", YA, t(9, 4, 21), [("u", "我养了一只鹦鹉，叫豆豆。"), ("a", "可爱。")],
          [Fact("ya.pet", "self", "pet", "鹦鹉豆豆", (0,), ("我养了一只鹦鹉，叫豆豆",), ("豆豆", "鹦鹉"))], tags=("pet",)),
        C("g03", YA, t(9, 6, 21), [("u", "我说错了，鹦鹉叫球球，不是豆豆。"), ("a", "已更正。")],
          [Fact("ya.pet2", "self", "pet", "鹦鹉球球", (0,), ("鹦鹉叫球球",), ("球球",), corrects=True, refers="ya.pet", op="SUPERSEDE", relation="CORRECTION")],
          tags=("correction_multi",)),
        C("g04", YA, t(9, 9, 21), [("u", "我又养了一只仓鼠。"), ("a", "热闹了。")],
          [Fact("ya.pet3", "self", "pet", "仓鼠", (0,), ("我又养了一只仓鼠",), ())], tags=("addition_multi",)),
        # ---------------- 白露 (YB): 厦门，设计师; relationship correction + relationship addition + change over time
        C("h01", YB, t(9, 3, 22), [("u", "我表妹叫思思，在读高中。"), ("a", "好的。")],
          [Fact("yb.cousin", "self", "relationship", "表妹 思思", (0,), ("我表妹叫思思",), ("思思",))], tags=("relationship",)),
        C("h02", YB, t(9, 5, 22), [("u", "更正一下，我表妹叫思琪，不是思思。"), ("a", "改好了。")],
          [Fact("yb.cousin2", "self", "relationship", "表妹 思琪", (0,), ("我表妹叫思琪",), ("思琪",), corrects=True, refers="yb.cousin", op="SUPERSEDE",
                relation="CORRECTION")], tags=("correction_multi",)),
        C("h03", YB, t(9, 7, 22), [("u", "我同事叫阿康。"), ("a", "嗯。")],
          [Fact("yb.col1", "self", "relationship", "同事 阿康", (0,), ("我同事叫阿康",), ("阿康",))], tags=("relationship",)),
        C("h04", YB, t(9, 9, 22), [("u", "我还有个同事叫阿伟。"), ("a", "记住了。")],
          [Fact("yb.col2", "self", "relationship", "同事 阿伟", (0,), ("我还有个同事叫阿伟",), ("阿伟",))], tags=("addition_multi",)),
        C("h05", YB, t(9, 12, 22), [("u", "我以前养过一只猫，现在养的是一条金毛。"), ("a", "金毛很温顺。")],
          [Fact("yb.dog", "self", "pet", "金毛", (0,), ("现在养的是一条金毛",), ("一条金毛",))], tags=("change_over_time",)),
        C("h06", YB, t(9, 14, 22), [("u", "我在一家广告公司做设计。"), ("a", "好的。")],
          [Fact("yb.work", "self", "works_at", "一家广告公司", (0,), ("在一家广告公司做设计",), ("广告公司",))], tags=("simple",)),
        # ---------------- 孟川 (YC): 武汉，牙医; likes correction + single-valued correction
        C("k01", YC, t(9, 2, 9), [("u", "我喜欢喝绿茶。"), ("a", "好。")],
          [Fact("yc.tea", "self", "likes", "绿茶", (0,), ("我喜欢喝绿茶",), ())], tags=("preference",)),
        C("k02", YC, t(9, 4, 9), [("u", "我记错了，我喜欢的是红茶，不是绿茶。"), ("a", "已改。")],
          [Fact("yc.tea2", "self", "likes", "红茶", (0,), ("我喜欢的是红茶",), (), corrects=True, refers="yc.tea", op="SUPERSEDE", relation="CORRECTION")],
          tags=("correction_multi",)),
        C("k03", YC, t(9, 6, 9), [("u", "我住在武汉洪山区。"), ("a", "好的。")],
          [Fact("yc.home", "self", "lives_in", "武汉洪山区", (0,), ("我住在武汉洪山区",), ("洪山区",))], tags=("simple",)),
        C("k04", YC, t(9, 8, 9), [("u", "我之前说错了，我住在江夏区，不是洪山区。"), ("a", "已更正。")],
          [Fact("yc.home2", "self", "lives_in", "江夏区", (0,), ("我住在江夏区",), ("武汉江夏区",), corrects=True, refers="yc.home", op="SUPERSEDE",
                relation="CORRECTION")], tags=("correction_single",)),
        C("k05", YC, t(9, 10, 9), [("u", "我也喜欢跑步。"), ("a", "好习惯。")],
          [Fact("yc.run", "self", "likes", "跑步", (0,), ("我也喜欢跑步",), ())], tags=("addition_multi",)),
    ]


def checkpoints() -> List[Checkpoint]:
    return [Checkpoint("ya_parrot_qiuqiu_after_correction", YA, "ya.pet", "pet", t(9, 7), t(9, 7), "球球"),
            Checkpoint("yb_cousin_siqi_after_correction", YB, "yb.cousin", "relationship", t(9, 6), t(9, 6), "思琪"),
            Checkpoint("yc_tea_red_after_correction", YC, "yc.tea", "likes", t(9, 5), t(9, 5), "红茶"),
            Checkpoint("yc_home_jiangxia_after_correction", YC, "yc.home", "lives_in", t(9, 9), t(9, 9), "江夏")]


def observation_expectations() -> List[ObsExpect]:
    return []


def message_id(conv: Conv, i: int) -> str:
    return f"{conv.persona}.{conv.kind}.{conv.cid}.{i}"


def group_id(conv: Conv) -> str:
    if conv.kind == "chat":
        return f"{conv.persona}.{conv.kind}.{conv.cid}"
    contact = next(m.who for m in conv.messages if m.who not in ("u", "a"))
    return f"{conv.persona}.sms.{contact.rsplit('.', 1)[-1]}"


def raw_messages(conv: Conv) -> List[CanonicalMessage]:
    out = []
    for i, m in enumerate(conv.messages):
        speaker = conv.persona if m.who == "u" else BOT[conv.persona] if m.who == "a" else m.who
        loc = f"pos={i}" + ((";type=" + ("发送" if m.who == "u" else "接收")) if conv.kind == "sms" else "")
        out.append(CanonicalMessage(conversation_id=group_id(conv), message_id=message_id(conv, i), target_persona_id=conv.persona, speaker_id=speaker, timestamp=m.at,
                                    content=m.text, source_type="synthetic_chat_round" if conv.kind == "chat" else "synthetic_sms", source_id=CHAT if conv.kind == "chat" else SMS,
                                    locator=loc, recorded_at=m.recorded_at or m.at))
    return out


def sources() -> List[SourceRef]:
    return [SourceRef(source_id=CHAT, kind=SourceKind.CONVERSATION, origin=DataOrigin.SYNTHETIC, description="Batch 2.1 last-fix confirmation dataset: chats."),
            SourceRef(source_id=SMS, kind=SourceKind.CONVERSATION, origin=DataOrigin.SYNTHETIC, description="Batch 2.1 last-fix confirmation dataset: SMS.")]


def load(api) -> None:
    for s in sources():
        api.register_source(s)
    for conv in conversations():
        api.ingest_messages(raw_messages(conv))


def order() -> List[Conv]:
    return sorted(conversations(), key=lambda c: (max(m.recorded_at or m.at for m in c.messages), c.cid))


def manifest() -> dict:
    cs = conversations()
    body = {"version": VERSION, "conversations": [asdict(c) for c in cs], "checkpoints": [asdict(x) for x in checkpoints()]}
    digest = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return {"version": VERSION, "dataset_sha256": digest, "personas": NAMES, "conversations": len(cs), "messages": sum(len(c.messages) for c in cs),
            "gold_facts": sum(len(c.facts) for c in cs), "forbidden": sum(len(c.forbidden) for c in cs), "origin": "synthetic"}
