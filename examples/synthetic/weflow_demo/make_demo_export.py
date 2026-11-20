#!/usr/bin/env python3
"""Generate the public SYNTHETIC WeFlow exports used by the quickstart (invented people and text only; no real chat).

Alice (the contact) is the persona you build; Bob (the export account) is the person talking to her. Over 100 days the chat contains facts
the memory should keep, a plan that changes, and a third person's event that must stay attributed to that person:

* 2025-03-02 Alice adopts a cat named 团子;            * 2025-03-15 she starts a Saturday pottery class (陶艺课);
* 2025-04-05 she says she will move to 杭州 in May;    * 2025-05-10 she has moved (the move is done);
* 2025-05-12 plan: 西湖 on 05-20; 2025-05-18 the plan moves to 05-25 (the older date is superseded);
* her sister 小雨 gets married on 2025-06-08 (小雨's wedding, not Alice's); her favourite dish is 酸菜鱼.

``alice_bob_weflow_update.json`` is a LATER export of the same chat (ten more days): use it to try the incremental import.

    python examples/synthetic/weflow_demo/make_demo_export.py            # rewrites both files (deterministic)
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

TZ = timezone(timedelta(hours=8))
ALICE, BOB = "wxid_demo_alice", "wxid_demo_bob"
NAMES = {ALICE: "Alice", BOB: "Bob"}
START = datetime(2025, 3, 1, 8, 30, tzinfo=TZ)
HERE = Path(__file__).resolve().parent

SMALL = [("今天忙吗？", ["还行，上午开了两个会", "有点忙，刚吃上饭", "不忙，在摸鱼哈哈"]),
         ("晚上吃什么？", ["想吃酸菜鱼", "随便吃点面", "点了个外卖，懒得做了"]),
         ("周末有空吗？", ["周六上午有课，下午可以", "这周末想在家躺着", "看情况，可能要加班"]),
         ("最近睡得怎么样？", ["还不错，就是起不来", "有点失眠，昨晚两点才睡", "挺好的，十一点就睡了"]),
         ("看了那部电影没？", ["还没，等周末看", "看了，结局有点仓促", "排队太长没进去"]),
         ("天气怎么样？", ["下雨了，没带伞", "太阳很大，晒死了", "降温了，记得加衣服"]),
         ("在干嘛呢？", ["在回家的地铁上", "刚到家，准备洗澡", "在写周报"]),
         ("今天心情好吗？", ["挺好的", "一般般吧", "有点累，但还行"])]
CAT = ["团子今天又把杯子推下桌了", "团子在我键盘上睡着了", "给团子买了新猫抓板", "团子今天胃口很好"]
POTTERY = ["陶艺课今天学拉坯，手全是泥", "老师说我捏的杯子有点歪哈哈", "做了一个小碗，下周烧好"]


ASK = ["", "", "诶，", "Alice，", "话说", "在吗？", "嗯，", "对了，", "我刚下班，", "问你个事，"]
OPEN = ["", "嗯，", "哈哈，", "唉，", "其实", "跟你说，", "诶，", "说实话", "怎么说呢，", "还好吧，"]
TAIL = ["", "。", "～", "，你那边呢？", "，晚点再聊", "，累死了", "，不过还挺开心的", "，明天再说吧", "，你别笑我", "，就这样",
        "，等下发你照片", "，周末细说", "，我先去忙了", "，感觉还行", "，你也早点休息"]
DETAIL = ["", "今天地铁特别挤", "公司楼下新开了家奶茶店", "同事请大家吃了蛋糕", "下午下了一场雨", "中午吃了麻辣烫", "晚上想早点睡",
          "刚把衣服洗了", "最近在追一部剧", "明天还要早起", "今天走了一万多步", "又被老板拉去开会"]


def reply(rng, answers) -> str:
    """One varied reply: an opener, the answer, an optional detail and a tail (so replies are not near-duplicates of each other)."""
    d = rng.choice(DETAIL)
    return rng.choice(OPEN) + rng.choice(answers) + ("，" + d if d else "") + rng.choice(TAIL)


def build(days: int, seed: str = "stillhere-demo"):
    rng = random.Random(seed)
    msgs = []

    def add(t, who, content, type_="文本消息", local_type=1, **kw):
        n = len(msgs) + 1
        m = {"localId": n, "createTime": int(t.timestamp()), "formattedTime": t.strftime("%Y-%m-%d %H:%M:%S"), "type": type_, "localType": local_type,
             "content": content, "isSend": 1 if who == BOB else 0, "senderUsername": who, "senderDisplayName": NAMES[who], "source": "",
             "senderAvatarKey": who, "platformMessageId": f"8{n:08d}"}
        m.update(kw)
        msgs.append(m)
        return m

    events = {date(2025, 3, 2): [(BOB, "你最近是不是想养猫？"), (ALICE, "对！今天领养了一只橘猫"), (ALICE, "叫团子，三个月大")],
              date(2025, 3, 15): [(ALICE, "我报了个陶艺课，每周六上午"), (BOB, "好玩吗"), (ALICE, "第一节课挺有意思的")],
              date(2025, 4, 5): [(ALICE, "跟你说个事，我五月要搬去杭州了"), (BOB, "这么突然？"), (ALICE, "新工作在那边，下个月就走")],
              date(2025, 5, 10): [(ALICE, "终于搬完家了，现在在杭州"), (BOB, "辛苦了，房子怎么样"), (ALICE, "离西湖不远，团子也适应了")],
              date(2025, 5, 12): [(BOB, "我5月20号去杭州出差"), (ALICE, "那20号一起去西湖吧")],
              date(2025, 5, 18): [(ALICE, "西湖改到25号吧，20号我要加班"), (BOB, "行，那就25号")],
              date(2025, 5, 28): [(ALICE, "我姐小雨下个月8号结婚"), (BOB, "恭喜她！你当伴娘吗"), (ALICE, "对，我当伴娘，最近在挑礼服")],
              date(2025, 6, 8): [(ALICE, "今天小雨的婚礼，好感动"), (BOB, "替我跟她说声恭喜")]}
    for d in range(days):
        day = START + timedelta(days=d)
        t = day + timedelta(minutes=rng.randrange(0, 90))
        for _ in range(rng.choice([3, 4, 5])):
            q, answers = rng.choice(SMALL)
            add(t, BOB, rng.choice(ASK) + q)
            t += timedelta(minutes=rng.randrange(1, 6))
            add(t, ALICE, reply(rng, answers))
            if rng.random() < 0.3:
                t += timedelta(seconds=40)
                add(t, ALICE, rng.choice(["嗯嗯", "哈哈", "好的", "你呢？"]))
            t += timedelta(minutes=rng.randrange(20, 120))
        if day.date() >= date(2025, 3, 2) and rng.random() < 0.15:
            add(t, ALICE, rng.choice(CAT))
            t += timedelta(minutes=3)
        if day.date() >= date(2025, 3, 15) and day.weekday() == 5 and rng.random() < 0.5:
            add(t, ALICE, rng.choice(POTTERY))
            t += timedelta(minutes=5)
        if rng.random() < 0.08:
            add(t, ALICE, "[图片]", type_="图片消息", local_type=3)
            t += timedelta(minutes=2)
        for who, text in events.get(day.date(), []):
            t += timedelta(minutes=rng.randrange(1, 4))
            add(t, who, text)
    session = {"wxid": ALICE, "nickname": "Alice", "remark": "Alice", "displayName": "Alice", "type": "私聊",
               "lastTimestamp": msgs[-1]["createTime"], "messageCount": len(msgs), "avatar": ""}
    exported = int((START + timedelta(days=days)).timestamp())
    return {"weflow": {"version": "1.0.3", "exportedAt": exported, "generator": "WeFlow"}, "session": session, "messages": msgs,
            "avatars": {ALICE: "", BOB: ""}}


def date(y, m, d):
    return datetime(y, m, d, tzinfo=TZ).date()


def main() -> int:
    for name, days in (("alice_bob_weflow.json", 100), ("alice_bob_weflow_update.json", 110)):
        (HERE / name).write_bytes((json.dumps(build(days), ensure_ascii=False, indent=1) + "\n").encode("utf-8"))   # LF on every OS
        print(name, days, "days")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
