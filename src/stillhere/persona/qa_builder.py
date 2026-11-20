"""StillHere Real-History Memory QA (200 questions) and the Writer audit sample (100 sessions), built from the RAW evidence archive.

Questions are built from the archived WeChat messages, not from the Writer's extracted memories (otherwise a fact the Writer missed
could never be asked, and Writer recall would be invisible). Flow:

    raw messages -> evidence selection (deterministic cues per category, seeded) -> question / reference construction by the LOCAL
    model, constrained to the selected lines -> deterministic verification (cited ids, speaker / subject, numbers and wording of every
    reference fact in the cited lines, no answer leak into the question) -> de-duplication -> category balancing -> freeze (sha256)

Unanswerable questions are only those whose basis can be checked over the WHOLE archive: a calendar day with no message at all, or a
private topic none of whose keywords occurs in any text message. Every row is labelled MACHINE_CONSTRUCTED_EVIDENCE_VERIFIED; nothing
here is a human gold label. A category that cannot be filled is reported with its shortfall (never padded with invented answers).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import recall as R

TZ = timezone(timedelta(hours=8))
QA_SCHEMA = "stillhere.persona.real_history_qa/1.0"
QA_PROMPT_VERSION = "stillhere.persona.qa_builder.prompt/1.0"
LABEL = "MACHINE_CONSTRUCTED_EVIDENCE_VERIFIED"
TARGETS = {"PERSONA_SELF_EXPERIENCE": 35, "COUNTERPART_EXPERIENCE": 25, "THIRD_PARTY": 20, "DATE_SUMMARY": 35, "TEMPORAL_CHANGE": 25,
           "NEGATION_PLAN_CANCEL": 20, "CROSS_MESSAGE": 20, "UNANSWERABLE": 20}
FAST_TARGETS = {"PERSONA_SELF_EXPERIENCE": 18, "COUNTERPART_EXPERIENCE": 12, "THIRD_PARTY": 10, "DATE_SUMMARY": 18, "TEMPORAL_CHANGE": 12,
                "NEGATION_PLAN_CANCEL": 10, "CROSS_MESSAGE": 10, "UNANSWERABLE": 10}
MODEL_CATEGORIES = ("PERSONA_SELF_EXPERIENCE", "COUNTERPART_EXPERIENCE", "THIRD_PARTY", "TEMPORAL_CHANGE", "NEGATION_PLAN_CANCEL", "CROSS_MESSAGE")
BACKFILL_ORDER = ("COUNTERPART_EXPERIENCE", "NEGATION_PLAN_CANCEL", "CROSS_MESSAGE", "THIRD_PARTY", "PERSONA_SELF_EXPERIENCE", "TEMPORAL_CHANGE")
OVERSAMPLE = 4                     # model attempts per category at most TARGET x OVERSAMPLE
MIN_CHARS = 4                      # evidence lines this short still carry facts for a person who writes 1-5 characters ("我在上班")
TEMPORAL_DRAWS = 40                # seeded draws of same-topic line sets per (topic, speaker); duplicates are removed later
FIRST_PERSON_FACT = re.compile(r"我(以前|之前|现在|最近|目前|今年|去年|小时候|当时|一直)?.{0,6}(在|是|住|上班|工作|读|学|喜欢|讨厌|不喜欢|爱|养|有|去过|去了|买了|考|做|换|搬|辞|报)")
NEG_PLAN_CANCEL = re.compile(r"不(喜欢|吃|喝|去|想|要|会|打算)|没(去|买|吃|有|考|做)|打算|准备|计划|下周|下个月|明天|后天|周末|取消|不去了|改天|改到|推迟|延期|算了")
TOPICS = {"work": "工作|上班|公司|实习|老板|同事|辞职|入职|加班", "home": "住|搬家|房子|租|宿舍|小区", "study": "学校|上课|考试|老师|毕业|论文|专业|读研|考研",
          "relationship": "男朋友|女朋友|对象|分手|在一起|结婚", "health": "生病|医院|感冒|发烧|看病|牙"}
UNANSWERABLE_TEMPLATES = [
    ("你家的猫叫什么名字？", ["猫"]), ("你养的狗叫什么名字？", ["狗", "汪"]), ("你小学是在哪读的？", ["小学"]), ("你第一次坐飞机是去哪？", ["飞机", "航班", "机票"]),
    ("你考驾照考了几次？", ["驾照", "科目一", "科目二", "科目三"]), ("你高中班主任叫什么？", ["班主任", "高中"]), ("你最喜欢的足球队是哪个？", ["足球", "球队"]),
    ("你学过什么乐器？", ["钢琴", "吉他", "乐器", "小提琴", "古筝"]), ("你去过日本哪个城市？", ["日本", "东京", "大阪", "京都"]), ("你有纹身吗，纹的是什么？", ["纹身"]),
    ("你参加过马拉松吗？", ["马拉松"]), ("你是什么血型？", ["血型"]), ("你会说粤语吗？", ["粤语", "广东话"]), ("你中过彩票吗？", ["彩票"]),
    ("你是在哪家医院出生的？", ["出生"]), ("你第一份工资是多少？", ["第一份工资", "工资"]), ("你学过游泳吗，在哪学的？", ["游泳"]), ("你爷爷以前是做什么的？", ["爷爷"]),
    ("你玩过滑雪吗，去哪滑的？", ["滑雪"]), ("你最怕什么动物？", ["怕蛇", "怕蜘蛛", "怕狗", "怕猫"]), ("你上次献血是什么时候？", ["献血"]),
    ("你小时候住在哪个城市？", ["小时候"]), ("你戴不戴眼镜，多少度？", ["眼镜", "近视", "度数"]), ("你最喜欢哪个歌手？", ["歌手", "演唱会"]),
    ("你玩过哪些乐队？", ["乐队"]), ("你的英文名叫什么？", ["英文名"]), ("你去过西藏吗？", ["西藏", "拉萨"]), ("你买过基金吗，亏了多少？", ["基金", "股票"]),
]

QA_SYSTEM = """你是评测题构建助手。下面是两个人（“本人”和“对方”）的真实微信聊天记录片段，每条有编号（Q01…）。
你要根据这些记录写一道“对方问本人（数字人）”的历史问题：问题里的“你”指本人，“我”指对方（提问的人）。
要求：
1. 问题的答案必须能直接从给出的记录中得到；不要问记录里没有答案的事；不要在问题里泄露答案。
2. reference_answer 写简洁的参考答案（可以很短）；reference_facts 列出 1–4 条答案依据的事实，每条一句话，事实里的人名、地点、数字照抄记录。
3. evidence 只列真正支持答案的记录编号。
4. 写清问题问的是谁的事：subject 取 PERSONA_SELF（本人自己的事）、CONVERSATION_COUNTERPART（对方自己的事）、THIRD_PARTY（第三方，如家人朋友）、JOINT（两人共同的事）。
5. 按给出的【题型要求】出题；如果这些记录不适合出这种题，输出 {"skip": true}。
只输出一个 JSON 对象：{"question": "...", "reference_answer": "...", "reference_facts": ["..."], "evidence": ["Q01"], "subject": "...", "difficulty": "easy|medium|hard"}"""

CATEGORY_ASK = {
    "PERSONA_SELF_EXPERIENCE": "问本人自己说过的个人经历或事实（工作、住处、学习、喜好、做过的事），问题用“你……”来问，答案来自“本人：”开头的记录。",
    "COUNTERPART_EXPERIENCE": "问对方（提问者）自己说过的经历或事实，问题用“我……来着？/ 你还记得我……吗？”来问，答案来自“对方：”开头的记录。",
    "THIRD_PARTY": "问记录里提到的第三方（家人、朋友、同事等）的事，问题里点明是谁（如“你妹妹”“我同事”），答案来自记录。",
    "TEMPORAL_CHANGE": "问一件事随时间的变化或某个时期的状态（例如以前怎样、后来怎样、那时候怎样），答案需要用到不同时间的记录。",
    "NEGATION_PLAN_CANCEL": "问一个否定、计划或取消（例如不喜欢什么、没去成什么、打算做什么、后来取消了什么），答案要准确反映否定或状态。",
    "CROSS_MESSAGE": "出一道必须结合至少两条不同记录才能回答的问题（例如把两条信息联系起来），evidence 至少写两个编号。",
}
CATEGORY_SUBJECT = {"PERSONA_SELF_EXPERIENCE": {R.PERSONA_SELF}, "COUNTERPART_EXPERIENCE": {R.COUNTERPART}, "THIRD_PARTY": {R.THIRD_PARTY}}


# ---------------------------------------------------------------- archive
def load_messages(db_path, persona_id: str) -> List[dict]:
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT message_id, speaker_id, timestamp_utc, content, conversation_id FROM messages WHERE target_persona_id = ? "
                           "ORDER BY timestamp_utc, message_id", (persona_id,)).fetchall()
    finally:
        con.close()
    out = []
    for mid, spk, ts, content, cid in rows:
        t = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(TZ)
        out.append({"message_id": mid, "speaker": "本人" if spk == persona_id else "对方", "time": t.strftime("%Y-%m-%d %H:%M"), "date": t.strftime("%Y-%m-%d"),
                    "text": content or "", "conversation_id": cid, "placeholder": (content or "").startswith("[") and (content or "").endswith("]")})
    return out


def _sha(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _window(msgs: Sequence[dict], idx: int, before: int = 3, after: int = 3) -> List[dict]:
    cid = msgs[idx]["conversation_id"]
    lo, hi = idx, idx
    while lo > 0 and idx - lo < before and msgs[lo - 1]["conversation_id"] == cid:
        lo -= 1
    while hi + 1 < len(msgs) and hi - idx < after and msgs[hi + 1]["conversation_id"] == cid:
        hi += 1
    return [m for m in msgs[lo:hi + 1] if not m["placeholder"]]


# ---------------------------------------------------------------- evidence selection per category
def candidate_windows(msgs: Sequence[dict], seed: int) -> Dict[str, List[List[dict]]]:
    """Deterministic, seeded pools of evidence windows per model-built category."""
    rnd = random.Random(seed)
    text = [i for i, m in enumerate(msgs) if not m["placeholder"] and MIN_CHARS <= len(m["text"]) <= 120]
    pools: Dict[str, List[List[dict]]] = defaultdict(list)
    for i in text:
        m = msgs[i]
        if FIRST_PERSON_FACT.search(m["text"]):
            pools["PERSONA_SELF_EXPERIENCE" if m["speaker"] == "本人" else "COUNTERPART_EXPERIENCE"].append(_window(msgs, i))
        if re.search(R.KIN_RE, m["text"]) and (FIRST_PERSON_FACT.search(m["text"]) or R.DATE_TOPIC.search(m["text"]) or len(m["text"]) >= 10):
            pools["THIRD_PARTY"].append(_window(msgs, i))
        if NEG_PLAN_CANCEL.search(m["text"]):
            pools["NEGATION_PLAN_CANCEL"].append(_window(msgs, i, 2, 4))
    by_conv: Dict[str, List[int]] = defaultdict(list)
    for i in text:
        by_conv[msgs[i]["conversation_id"]].append(i)
    for cid, idx in by_conv.items():
        if len(idx) >= 6:
            a = rnd.randrange(0, max(1, len(idx) - 10))
            pools["CROSS_MESSAGE"].append([msgs[j] for j in idx[a:a + 12]])
    for name, pat in TOPICS.items():                                  # the same topic on >= 2 different dates (by either speaker)
        hits = [i for i in text if re.search(pat, msgs[i]["text"]) and (FIRST_PERSON_FACT.search(msgs[i]["text"]) or NEG_PLAN_CANCEL.search(msgs[i]["text"]))]
        by_speaker: Dict[str, List[int]] = defaultdict(list)
        for i in hits:
            by_speaker[msgs[i]["speaker"]].append(i)
        for spk, idx in by_speaker.items():
            for _ in range(min(TEMPORAL_DRAWS, len(idx))):
                pick = sorted(rnd.sample(idx, min(len(idx), 4)))
                if len({msgs[j]["date"] for j in pick}) >= 2:
                    pools["TEMPORAL_CHANGE"].append([msgs[j] for j in pick])
    for k in pools:
        rnd.shuffle(pools[k])
    return pools


def date_pool(msgs: Sequence[dict], seed: int, n: int) -> List[str]:
    """Days with enough substantive messages, spread over the months of the archive (seeded)."""
    rnd = random.Random(seed)
    per_day: Dict[str, int] = Counter(m["date"] for m in msgs if not m["placeholder"] and not R.TRIVIAL.match(m["text"].strip()))
    days = sorted(d for d, c in per_day.items() if c >= 6)
    by_month: Dict[str, List[str]] = defaultdict(list)
    for d in days:
        by_month[d[:7]].append(d)
    for v in by_month.values():
        rnd.shuffle(v)
    out, months = [], sorted(by_month)
    while len(out) < n * OVERSAMPLE and any(by_month.values()):
        for mth in months:
            if by_month[mth]:
                out.append(by_month[mth].pop())
    return out


def empty_days(msgs: Sequence[dict], seed: int, n: int) -> List[str]:
    """Calendar days inside the archive range on which NO message at all exists (checked over every message)."""
    have = {m["date"] for m in msgs}
    if not msgs:
        return []
    d0 = datetime.strptime(msgs[0]["date"], "%Y-%m-%d")
    d1 = datetime.strptime(msgs[-1]["date"], "%Y-%m-%d")
    days = [(d0 + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((d1 - d0).days + 1)]
    empty = [d for d in days if d not in have]
    random.Random(seed).shuffle(empty)
    return empty[:n]


def absent_topics(msgs: Sequence[dict]) -> List[Tuple[str, List[str]]]:
    texts = [m["text"] for m in msgs if not m["placeholder"]]
    return [(q, kws) for q, kws in UNANSWERABLE_TEMPLATES if not any(k in t for t in texts for k in kws)]


# ---------------------------------------------------------------- model construction + verification
def _bigrams(s: str) -> set:
    return R._bigrams(s)


def _numbers(s: str) -> set:
    return R._numbers(s)


def verify_row(row: Mapping, lines: Mapping[str, dict], category: str) -> Optional[str]:
    """None when the machine-built row is evidence-verified, else the reason it is rejected."""
    q, ans, facts, ev = row.get("question"), row.get("reference_answer"), row.get("reference_facts"), row.get("evidence")
    if not (isinstance(q, str) and 6 <= len(q) <= 80 and isinstance(ans, str) and ans.strip() and isinstance(facts, list) and facts and isinstance(ev, list) and ev):
        return "schema"
    if any(e not in lines for e in ev):
        return "evidence id not in the window"
    if not re.search(r"[？?]|吗|什么|哪|几|谁|多少|怎么|是不是|有没有|来着", q):
        return "not a question"
    cited = [lines[e] for e in ev]
    text = " ".join(c["text"] for c in cited)
    for f in facts:
        if not isinstance(f, str) or not f.strip():
            return "empty fact"
        if _numbers(f) - _numbers(text) - {str(d) for c in cited for d in _numbers(c["time"])}:
            return "number of a fact not in the cited lines"
        fb = _bigrams(f) - _bigrams("本人对方第三方你我他她")
        if fb and len(fb & _bigrams(text)) < min(2, len(fb)):
            return "fact wording not in the cited lines"
    leak = _bigrams(ans) - _bigrams(q) - _bigrams("本人对方你我他她以前之前现在")
    if _bigrams(ans) and not leak:
        return "answer leaked into the question"
    subj = R.resolve_subject(q)
    want = CATEGORY_SUBJECT.get(category)
    if want and subj.kind not in want:
        return f"question subject {subj.kind} does not fit {category}"
    if category == "PERSONA_SELF_EXPERIENCE" and not any(c["speaker"] == "本人" for c in cited):
        return "persona fact not said by the persona"
    if category == "COUNTERPART_EXPERIENCE" and not any(c["speaker"] == "对方" for c in cited):
        return "counterpart fact not said by the counterpart"
    if category == "CROSS_MESSAGE" and len(set(ev)) < 2:
        return "cross-message question cites one line"
    if category == "TEMPORAL_CHANGE" and len({c["date"] for c in cited}) < 2:
        return "temporal question cites one date only"
    return None


def _json(text: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except ValueError:
        return None


def _qrow(qid: str, category: str, question: str, answer: str, facts: List[str], cited: Sequence[dict], subject: str, time_scope, answerability: str,
          difficulty: str, extra: Optional[dict] = None) -> dict:
    ids = [c["message_id"] for c in cited]
    return {"question_id": qid, "category": category, "question": question, "reference_answer": answer, "reference_facts": facts,
            "evidence_message_ids": ids, "conversation_ids": sorted({c["conversation_id"] for c in cited}), "subject": subject, "time_scope": time_scope,
            "answerability": answerability, "difficulty": difficulty, "verification_status": LABEL,
            "source_hash": _sha(*[f"{c['message_id']}:{_sha(c['text'])}" for c in cited]) if cited else _sha(question), **(extra or {})}


def build(db_path, persona_id: str, complete: Callable[[list], dict], *, seed: int = 20261117, targets: Mapping[str, int] = TARGETS,
          log: Callable[[str], None] = print, progress: Optional[Path] = None, backfill: bool = False) -> Tuple[List[dict], dict]:
    """Build the QA set. ``complete`` is the local model (greedy JSON). ``progress``: a JSONL of finished model attempts (resume).
    ``backfill`` (FAST mode): a category shortfall is filled with further VERIFIED rows of other categories in ``BACKFILL_ORDER``; the
    actual distribution and every shortfall are reported. No row is ever invented."""
    msgs = load_messages(db_path, persona_id)
    end = datetime.strptime(msgs[-1]["date"], "%Y-%m-%d").replace(tzinfo=TZ) + timedelta(days=1)
    ask_time = end.isoformat()
    cache: Dict[str, dict] = {}
    if progress is not None and Path(progress).is_file():
        for l in Path(progress).read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                cache[r["key"]] = r
    def ask(key: str, messages) -> dict:
        if key in cache:
            return cache[key]["out"]
        out = {"text": complete(messages).get("text", "")}
        cache[key] = {"key": key, "out": out}
        if progress is not None:
            with open(progress, "a", encoding="utf-8") as f:
                f.write(json.dumps(cache[key], ensure_ascii=False) + "\n")
        return out

    rows: List[dict] = []
    report = {"schema": QA_SCHEMA, "label": LABEL, "prompt_version": QA_PROMPT_VERSION, "summary_version": R.SUMMARY_VERSION, "seed": seed,
              "archive_messages": len(msgs), "ask_time": ask_time, "targets": dict(targets), "categories": {}}
    seen_q: List[set] = []
    seen_ev: set = set()

    def dup(q: str, ev: Sequence[str]) -> bool:
        b = _bigrams(q)
        key = tuple(sorted(ev))
        if key and key in seen_ev:
            return True
        return any(len(b & s) / max(1, len(b | s)) > 0.6 for s in seen_q)

    def add(row):
        rows.append(row)
        seen_q.append(_bigrams(row["question"]))
        if row["evidence_message_ids"]:
            seen_ev.add(tuple(sorted(row["evidence_message_ids"])))

    pools = candidate_windows(msgs, seed)
    state = {c: {"pos": 0, "got": 0, "tried": 0, "reasons": Counter()} for c in MODEL_CATEGORIES}

    def run_cat(cat: str, want: int, max_tries: int) -> int:
        """Continue this category's seeded pool until ``want`` more rows are built or ``max_tries`` more attempts are spent."""
        st, pool = state[cat], pools.get(cat, [])
        built = tried = 0
        while built < want and tried < max_tries and st["pos"] < len(pool):
            win = pool[st["pos"]]
            st["pos"] += 1
            lines = {f"Q{i + 1:02d}": m for i, m in enumerate(win)}
            body = "\n".join(f"[{k}] {m['time']} {m['speaker']}：{m['text']}" for k, m in lines.items())
            key = _sha(cat, *[m["message_id"] for m in win])
            tried += 1
            st["tried"] += 1
            d = _json(ask(key, [{"role": "system", "content": QA_SYSTEM}, {"role": "user", "content": f"【题型要求】{CATEGORY_ASK[cat]}\n\n【聊天记录】\n{body}"}])["text"])
            if not d or d.get("skip"):
                st["reasons"]["model skipped / no JSON"] += 1
                continue
            why = verify_row(d, lines, cat)
            if why is None and dup(d["question"], [lines[e]["message_id"] for e in d["evidence"]]):
                why = "duplicate"
            if why:
                st["reasons"][why] += 1
                continue
            cited = [lines[e] for e in dict.fromkeys(d["evidence"])]
            subj = R.resolve_subject(d["question"]).kind
            st["got"] += 1
            add(_qrow(f"{cat[:4]}-{st['got']:03d}", cat, d["question"].strip(), d["reference_answer"].strip(), [f.strip() for f in d["reference_facts"]], cited,
                      subj, {"evidence_dates": sorted({c["date"] for c in cited}), "ask_time": ask_time}, "ANSWERABLE",
                      d.get("difficulty") if d.get("difficulty") in ("easy", "medium", "hard") else "medium"))
            built += 1
        return built

    for cat in MODEL_CATEGORIES:
        want = targets.get(cat, 0)
        run_cat(cat, want, want * OVERSAMPLE)
        st = state[cat]
        report["categories"][cat] = {"target": want, "built": st["got"], "model_attempts": st["tried"], "pool": len(pools.get(cat, [])), "rejections": dict(st["reasons"]),
                                     "shortfall": max(0, want - st["got"]), "shortfall_reason": None if st["got"] >= want else "evidence pool / verification exhausted"}
        log(f"QA {cat}: {st['got']}/{want}")

    # date summaries: the same grounded window reading as the runtime, but the gold keeps the full day as judge evidence
    want, got, tried, reasons = targets.get("DATE_SUMMARY", 0), 0, 0, Counter()
    for day in date_pool(msgs, seed, want):
        if got >= want or tried >= want * OVERSAMPLE:
            break
        start = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=TZ)
        scope = R.DateScope(start, start + timedelta(days=1), "day", day)
        win = [m for m in msgs if m["date"] == day]
        tried += 1
        key = _sha("DATE_SUMMARY", day)
        summ = R.summarize_window(scope, win, lambda m, k=key: ask(k + _sha(json.dumps(m, ensure_ascii=False)), m))
        if summ.status != "ANSWERED" or len(summ.points) < 2:
            reasons[f"summary {summ.status or 'thin'}"] += 1
            continue
        y, mo, dd = day.split("-")
        q = f"你还记得我们{y}年{int(mo)}月{int(dd)}日那天聊了什么吗？"
        cited_ids = {i for p in summ.points for i in p["message_ids"]}
        cited = [m for m in win if m["message_id"] in cited_ids]
        add(_qrow(f"DATE-{got + 1:03d}", "DATE_SUMMARY", q, summ.answer(), [p["text"] for p in summ.points], cited, R.JOINT,
                  {**scope.to_dict(), "ask_time": ask_time}, "ANSWERABLE", "medium",
                  {"key_topics": [p["text"] for p in summ.points], "day_message_ids": [m["message_id"] for m in win if not m["placeholder"]]}))
        got += 1
    report["categories"]["DATE_SUMMARY"] = {"target": want, "built": got, "model_attempts": tried, "rejections": dict(reasons), "shortfall": max(0, want - got),
                                            "shortfall_reason": None if got >= want else "not enough days with a grounded multi-point summary"}
    log(f"QA DATE_SUMMARY: {got}/{want}")

    # unanswerable: checkable over the whole archive only
    want = targets.get("UNANSWERABLE", 0)
    topics = absent_topics(msgs)
    days = empty_days(msgs, seed, want)
    n_topic = min(len(topics), want - min(len(days), want // 2))
    built = []
    for q, kws in topics[:n_topic]:
        built.append(_qrow(f"UNAN-{len(built) + 1:03d}", "UNANSWERABLE", q, "聊天记录里没有这件事，应表示不知道或不记得，不能编造。", [], [], R.resolve_subject(q).kind, None,
                           "UNANSWERABLE", "medium", {"unanswerable_basis": f"none of {kws} occurs in any of the {sum(not m['placeholder'] for m in msgs)} text messages"}))
    for d in days[:want - len(built)]:
        y, mo, dd = d.split("-")
        q = f"{y}年{int(mo)}月{int(dd)}日那天我们聊了什么？"
        start = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=TZ)
        built.append(_qrow(f"UNAN-{len(built) + 1:03d}", "UNANSWERABLE", q, "那天没有聊天记录，应表示那天没聊或不记得，不能编造。", [], [], R.JOINT,
                           {"start": start.isoformat(), "end": (start + timedelta(days=1)).isoformat(), "granularity": "day", "ask_time": ask_time}, "UNANSWERABLE", "easy",
                           {"unanswerable_basis": f"no message at all on {d} (UTC+8) in the archive"}))
    for b in built:
        add(b)
    report["categories"]["UNANSWERABLE"] = {"target": want, "built": len(built), "absent_topics": n_topic, "empty_days": len(built) - n_topic,
                                            "shortfall": max(0, want - len(built)), "shortfall_reason": None if len(built) >= want else "not enough checkable absent topics / empty days"}
    deficit = sum(targets.values()) - len(rows)
    report["backfill"] = {"enabled": bool(backfill), "deficit_before": max(0, deficit), "order": list(BACKFILL_ORDER), "added": {}}
    if backfill and deficit > 0:                     # pre-registered: real, verified rows of other categories, in a fixed order
        for cat in BACKFILL_ORDER:
            if deficit <= 0:
                break
            n = run_cat(cat, deficit, deficit * OVERSAMPLE)
            if n:
                report["backfill"]["added"][cat] = n
                report["categories"][cat]["built"] += n
                report["categories"][cat]["backfilled"] = n
                report["categories"][cat]["model_attempts"] = state[cat]["tried"]
                report["categories"][cat]["rejections"] = dict(state[cat]["reasons"])
                deficit -= n
        log(f"QA backfill: {report['backfill']['added']} (remaining deficit {max(0, deficit)})")
    report["total"] = len(rows)
    report["total_target"] = sum(targets.values())
    report["actual_distribution"] = dict(Counter(r["category"] for r in rows))
    return rows, report


def freeze(rows: Sequence[Mapping], report: Mapping, out_dir: Path) -> dict:
    """Write REAL_HISTORY_QA_200.jsonl + its sha256 + the build report. A frozen set is never rewritten (answers come later)."""
    out_dir = Path(out_dir)
    path = out_dir / "REAL_HISTORY_QA_200.jsonl"
    data = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode("utf-8")
    if path.is_file():
        raise FileExistsError(f"{path} is frozen; it is never rebuilt inside a run")
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    rep = {**report, "qa_sha256": sha, "frozen_at": datetime.now(timezone.utc).isoformat(), "file": path.name}
    (out_dir / "QA_BUILD_REPORT.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    (out_dir / "QA_FROZEN.sha256").write_text(sha + "  REAL_HISTORY_QA_200.jsonl\n", encoding="utf-8", newline="\n")
    return rep


def load_frozen(out_dir: Path) -> Tuple[List[dict], str]:
    path = Path(out_dir) / "REAL_HISTORY_QA_200.jsonl"
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    want = (Path(out_dir) / "QA_FROZEN.sha256").read_text(encoding="utf-8").split()[0]
    if sha != want:
        raise ValueError("QA_SET_CHANGED: the frozen QA set does not match its sha256 (questions / gold may not change after freezing)")
    return [json.loads(l) for l in data.decode("utf-8").splitlines() if l.strip()], sha


# ---------------------------------------------------------------- writer audit sample (100 sessions, stratified, seeded)
FEATURES = (("third_party", lambda t, n: bool(re.search(R.KIN_RE, t))), ("cancel", lambda t, n: bool(re.search(r"取消|不去了|改天|改到|推迟|延期", t))),
            ("plan", lambda t, n: bool(re.search(r"打算|准备|计划|下周|下个月|明天|后天", t))), ("negation", lambda t, n: bool(re.search(r"不(喜欢|吃|喝|去|想|要)|没(去|有)", t))),
            ("temporal", lambda t, n: bool(re.search(r"以前|之前|现在|后来|去年|今年|最近", t))), ("preference", lambda t, n: bool(re.search(r"喜欢|讨厌|爱吃|不爱", t))),
            ("self_fact", lambda t, n: bool(FIRST_PERSON_FACT.search(t))), ("long_multi_turn", lambda t, n: n >= 30), ("short_replies", lambda t, n: n >= 4),
            ("small_talk", lambda t, n: True))


def audit_sample(db_path, persona_id: str, written: Sequence[str], *, n: int = 100, seed: int = 20261117, min_per_stratum: int = 4) -> List[dict]:
    """Stratify the WRITTEN sessions by their primary feature (first matching of FEATURES), allocate ``n`` proportionally with a
    minimum per stratum, draw with a fixed seed. Returns [{conversation_id, stratum}] (deterministic)."""
    msgs = load_messages(db_path, persona_id)
    by_conv: Dict[str, List[dict]] = defaultdict(list)
    for m in msgs:
        by_conv[m["conversation_id"]].append(m)
    strata: Dict[str, List[str]] = defaultdict(list)
    for cid in sorted(set(written)):
        ms = [m for m in by_conv.get(cid, []) if not m["placeholder"]]
        if len(ms) < 2:
            continue
        t = "\n".join(m["text"] for m in ms)
        if all(len(m["text"]) <= 5 for m in ms):
            strata["short_replies"].append(cid)
            continue
        name = next(nm for nm, f in FEATURES if f(t, len(ms)))
        strata[name].append(cid)
    rnd = random.Random(seed)
    total = sum(len(v) for v in strata.values())
    if total == 0:
        return []
    alloc = {k: min(len(v), max(min_per_stratum, round(n * len(v) / total))) for k, v in strata.items()}
    while sum(alloc.values()) > n:
        k = max(alloc, key=lambda x: alloc[x] - min_per_stratum / 10)
        alloc[k] -= 1
    spare = [k for k in sorted(strata) if alloc[k] < len(strata[k])]
    while sum(alloc.values()) < min(n, total) and spare:
        for k in list(spare):
            if sum(alloc.values()) >= min(n, total):
                break
            if alloc[k] < len(strata[k]):
                alloc[k] += 1
            else:
                spare.remove(k)
    out = []
    for k in sorted(strata):
        pick = sorted(strata[k])
        rnd.shuffle(pick)
        out += [{"conversation_id": c, "stratum": k} for c in sorted(pick[:alloc[k]])]
    return out
