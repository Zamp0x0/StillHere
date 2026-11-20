"""Synthetic WeFlow chat with known facts for the M1 memory tests (no real person). CONTACT is the target persona (本人)."""
import json
import re

import persona_fixture as F

DAY = 86400
# (day offset, seconds, who, text)
SCRIPT = [
    (0, 0, F.SELF, "你之前是不是在广州工作？"), (0, 60, F.CONTACT, "对，我之前在广州上班"),
    (0, 120, F.SELF, "我在澳洲读书呢，最近好忙"), (0, 180, F.CONTACT, "辛苦啦"),
    (2, 0, F.SELF, "你是不是准备去香港？"), (2, 60, F.CONTACT, "还没定呢，可能吧"),
    (2, 120, F.CONTACT, "我妹妹小雨下个月结婚"), (2, 180, F.SELF, "恭喜恭喜"),
    (4, 0, F.CONTACT, "我明天去看牙医"), (4, 60, F.SELF, "好的加油"),
    (5, 0, F.CONTACT, "牙医取消了，改天再去"), (5, 60, F.SELF, "那就好好休息"),
    (6, 0, F.SELF, "现在呢？还在广州吗"), (6, 60, F.CONTACT, "现在换成在深圳上班了"),
    (7, 0, F.CONTACT, "我不太喜欢咖啡"), (7, 60, F.SELF, "那喝茶吧"),
]


def chat(extra=()):
    c = F.Chat()
    for d, s, who, text in SCRIPT:
        c.text(F.T0 + d * DAY + s, who, text)
    c.media(F.T0 + 7 * DAY + 120, F.SELF)
    for d, s, who, text in extra:                      # a later export only APPENDS: earlier message ids stay the same
        c.text(F.T0 + d * DAY + s, who, text)
    return c


def cand(subject, predicate, obj, mid, span, **kw):
    base = {"type": "FACT", "subject": subject, "predicate": predicate, "object": obj, "polarity": "positive", "assertion_type": "stated", "certainty": "certain",
            "event_key": None, "activity": None, "event_status": None, "planned_time": None, "event_time": None, "daypart": None, "valid_from": None, "valid_to": None,
            "temporal_precision": "unknown", "corrects": False, "retracts": False, "refers_to": None, "source_message_ids": [mid], "source_spans": [span]}
    base.update(kw)
    return base


# a REFERENCE extractor (stands in for the model; a test double, never presented as semantic extraction): gold candidates for the
# lines it sees, plus planted WRONG candidates the deterministic verifier must reject
RULES = [
    ("对，我之前在广州上班", lambda m: [cand("self", "works_at", "广州", m, "我之前在广州上班")]),
    ("现在换成在深圳上班了", lambda m: [cand("self", "works_at", "深圳", m, "现在换成在深圳上班了")]),
    ("我在澳洲读书呢", lambda m: [cand("self", "school", "澳洲", m, "我在澳洲读书")]),                          # WRONG: the other person's fact
    ("你是不是准备去香港", lambda m: [cand("self", "event", "去香港", m, "准备去香港", type="EVENT", event_key="trip_hk", activity="旅行",
                                          event_status="COMPLETED")]),                                          # WRONG: a question about the target
    ("还没定呢，可能吧", lambda m: [cand("self", "event", "去香港", m, "可能吧", type="PLAN", event_key="trip_hk2", activity="旅行",
                                       event_status="PLANNED", certainty="uncertain")]),                        # uncertain: held back
    ("我妹妹小雨下个月结婚", lambda m: [cand("self", "relationship", "妹妹 小雨", m, "我妹妹小雨"),
                                        cand("person:小雨", "event", "结婚", m, "小雨下个月结婚", type="EVENT", event_key="wedding_xy", activity="结婚",
                                             event_status="PLANNED")]),
    ("我明天去看牙医", lambda m: [cand("self", "event", "看牙医", m, "明天去看牙医", type="PLAN", event_key="dentist", activity="看牙",
                                     event_status="PLANNED")]),
    ("牙医取消了", lambda m: [cand("self", "event", "看牙医", m, "牙医取消了", type="EVENT", event_key="dentist", activity="看牙", event_status="CANCELLED")]),
    ("我不太喜欢咖啡", lambda m: [cand("self", "dislikes", "咖啡", m, "不太喜欢咖啡")]),
    ("我搬到杭州了", lambda m: [cand("self", "lives_in", "杭州", m, "我搬到杭州了")]),
]
LINE = re.compile(r"^\[(m\d+)\]")


def reference_complete(messages):
    user = messages[-1]["content"]
    out = []
    for line in user.splitlines():
        m = LINE.match(line.strip())
        if not m:
            continue
        for key, fn in RULES:
            if key in line:
                out += fn(m.group(1))
    return {"text": json.dumps({"candidates": out}, ensure_ascii=False), "model": "reference-test-double"}
