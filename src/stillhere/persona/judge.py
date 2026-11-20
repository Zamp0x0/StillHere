"""DeepSeek LLM-as-Judge for the final StillHere evaluation: continuous 0-100 scores against explicit rubrics (never exact match).

Reuses the repository's verified DeepSeek client (``retrieval.deepseek_client``: thinking off, JSON mode, bounded retries on 429 / 5xx,
every attempt in an append-only cost ledger, key never logged). This module adds: the three rubrics (style / memory / writer), the
prompt + rubric versions, JSON schema validation with one bounded repair turn, score normalisation, config-blind unit ids and a
deterministic order, a checkpointed runner (a unit that already has an OK score is never sent again), a rate limit, a failure queue
(JUDGE_FAILED rows are kept and counted, never scored as 0 and never silently dropped) and a calibration set reported separately.

The judge never sees which configuration produced an answer (no group, prompt or adapter name: opaque ``unit_id`` only).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

RUBRIC_VERSION = "stillhere.judge.rubric/1.0"
PROMPT_VERSION = "stillhere.judge.prompt/1.0"
JUDGE_CEILING_CNY = 100.0
DEFAULT_BUDGET_CNY = 45.0
MAX_UNIT_ATTEMPTS = 3          # a failed unit is retried on later runs at most this many times in total
STYLE_WEIGHTS = {"style_fidelity": 0.35, "contextual_appropriateness": 0.35, "naturalness": 0.20, "non_repetition": 0.10}
MEMORY_WEIGHTS = {"factual_correctness": 0.45, "evidence_grounding": 0.25, "temporal_attribution": 0.20, "appropriate_abstention": 0.10}
WRITER_WEIGHTS = {"fact_correctness": 0.30, "attribution": 0.20, "temporal": 0.15, "coverage": 0.25, "memory_worthiness": 0.10}

ANCHORS = """评分锚点（每个维度 0–5 的整数）：5 完全做到；4 基本做到，只有很小的问题；3 一半做到；2 明显不足；1 几乎没做到；0 完全错误或缺失。"""

STYLE_SYSTEM = """你是严格、公正的对话评审员。任务：评价一条“模型回复”作为某个真实人物在微信聊天中的回复，是否像这个人、是否接得上话。
你会看到：最近的聊天上下文、这个人当时真实的回复（参考，不要求一模一样）、这个人的风格统计（只来自训练集）、模型回复，以及这条模型回复在同一组回答里重复出现的情况。

维度（每个 0–5 整数）：
- style_fidelity（像不像这个人）：长度、语气词、标点、表情、口头禅、分条习惯是否符合这个人的风格统计和参考回复。这个人很多时候只回 1–5 个字：短本身不是缺点；与参考回复语义一致、语气一致的短句（如参考“没呢”、模型“没吃呢”）应给高分。不要因为措辞或字数与参考不同就给低分。
- contextual_appropriateness（接不接得上）：是否回应了对方最新一句的意思。对方问了具体问题（如在哪工作）而模型只回“好”“没呢”这类与问题无关的短句，要扣分；答非所问、自相矛盾要扣分。
- naturalness（自然度）：像真人随手打的字，不像客服或 AI（没有“作为 AI”“希望对你有帮助”这类话，没有生硬的书面语）。
- non_repetition（不条件重复）：如果这条回复在同一组大量不同的问题下反复出现、而且在这里并不合适，扣分；这个人自己常说的短句在合适的场合出现，不算重复。
另外判断 hallucination：回复是否编造了上下文和参考都没有依据的具体私人事实（地点、人名、经历、数字），true / false。

""" + ANCHORS + """
只输出一个 JSON 对象：{"style_fidelity": 0-5, "contextual_appropriateness": 0-5, "naturalness": 0-5, "non_repetition": 0-5, "hallucination": true/false, "reason": "一两句理由"}"""

MEMORY_SYSTEM = """你是严格、公正的记忆问答评审员。任务：评价一个数字人对“历史问题”的回答。提问的是聊天对象（问题里的“你”是数字人本人，“我”是提问的人）。
你会看到：问题、参考答案、参考事实要点、来自真实聊天记录的证据原文、问题的主体与时间范围、是否可回答，以及模型回答。
不要用字面是否相同来判断；语义一致就算对。回答可以很短（这个人说话很简短）：“广州啊”可以完全回答“你以前在哪工作”。

维度（每个 0–5 整数）：
- factual_correctness：回答中的事实是否与参考答案 / 事实要点一致；对日期总结题，看关键事实要点覆盖了多少、有没有错的。答错人、答错时间期间的事实（如问过去却答现在）要扣分。
- evidence_grounding：回答中的每个具体内容是否都能在证据原文里找到依据；编造证据里没有的私人细节要大幅扣分。
- temporal_attribution：人物归属（本人 / 对方 / 第三方）和时间（过去 / 现在 / 哪一天）是否正确。
- appropriate_abstention：可回答的问题，回答“不记得 / 不知道”应给 0–1；不可回答（UNANSWERABLE）的问题，说“不记得 / 不清楚 / 好像没说过”应给 5，编造具体内容给 0。
另外：persona_style（0–5，回答的说话方式像不像一个真实的人在微信里随口回答；只作参考，不计入记忆分）；fabricated_private_fact（是否编造了证据中不存在的具体私人事实，true / false）。

""" + ANCHORS + """
只输出一个 JSON 对象：{"factual_correctness": 0-5, "evidence_grounding": 0-5, "temporal_attribution": 0-5, "appropriate_abstention": 0-5, "persona_style": 0-5, "fabricated_private_fact": true/false, "reason": "一两句理由"}"""

WRITER_SYSTEM = """你是严格、公正的记忆抽取审计员。任务：审计一个“记忆写入器”从一段真实聊天中抽取出的长期记忆候选。
聊天里只有两个人：“本人”（数字人模仿的对象）和“对方”（聊天对象）；其他人都是第三方。每条消息有编号（m1、m2…）。
你会看到：这段聊天的全部消息、写入器给出的候选（编号 C1、C2…，含主体、内容、状态、时间、引用的消息、验证结果：AUTO_VERIFIED 已通过自动验证并写入、REJECTED / AMBIGUOUS 被拦下没有写入）。

请检查：1 是否有错误事实；2 是否归属错误（把对方或第三方的事算成本人的，或反过来）；3 是否有时间错误；4 事件状态（计划 / 完成 / 取消）是否错误；
5 原文中明显值得长期记住、却没有被写入（AUTO_VERIFIED）的重要事实（只列真正重要的：经历、工作、住处、家人、计划、取消、偏好等，寒暄不算）；
6 是否把没有长期价值的闲聊（如“在吃饭”“到家了”）写成了记忆；7 是否有无法从原文追溯的内容。

维度（每个 0–5 整数，只针对 AUTO_VERIFIED 的候选；被拦下的错误候选不扣分，被拦下的正确候选算作遗漏）：
fact_correctness、attribution、temporal、coverage（重要事实的覆盖程度，没有值得记的内容且没有写入时给 5）、memory_worthiness。

""" + ANCHORS + """
只输出一个 JSON 对象：{"fact_correctness": 0-5, "attribution": 0-5, "temporal": 0-5, "coverage": 0-5, "memory_worthiness": 0-5,
"supported": ["C1"], "unsupported": [{"candidate": "C2", "why": "..."}], "missing": [{"fact": "...", "messages": ["m3"]}],
"attribution_errors": [{"candidate": "C2", "why": "..."}], "temporal_errors": [...], "worthiness_errors": [...], "summary": "两三句审计结论"}
supported / unsupported 只列 AUTO_VERIFIED 的候选。"""

KINDS = {"style": (STYLE_SYSTEM, STYLE_WEIGHTS), "memory": (MEMORY_SYSTEM, MEMORY_WEIGHTS), "writer": (WRITER_SYSTEM, WRITER_WEIGHTS)}
BOOLS = {"style": ("hallucination",), "memory": ("fabricated_private_fact",), "writer": ()}
EXTRA_INTS = {"memory": ("persona_style",)}
LISTS = {"writer": ("supported", "unsupported", "missing", "attribution_errors", "temporal_errors", "worthiness_errors")}


class JudgeError(RuntimeError):
    pass


def rubric_sha() -> str:
    return hashlib.sha256((RUBRIC_VERSION + PROMPT_VERSION + STYLE_SYSTEM + MEMORY_SYSTEM + WRITER_SYSTEM + json.dumps(
        [STYLE_WEIGHTS, MEMORY_WEIGHTS, WRITER_WEIGHTS], sort_keys=True)).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- parsing / normalisation
def validate(kind: str, raw: str) -> dict:
    """Parse and check one judge answer; raises JudgeError with the problems (used for the one repair turn)."""
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        raise JudgeError("no JSON object")
    try:
        d = json.loads(m.group(0))
    except ValueError as e:
        raise JudgeError(f"invalid JSON: {e}") from e
    _, weights = KINDS[kind]
    problems = []
    for k in list(weights) + list(EXTRA_INTS.get(kind, ())):
        v = d.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 5 or int(v) != v:
            problems.append(f"{k} must be an integer 0-5")
    for k in BOOLS[kind]:
        if not isinstance(d.get(k), bool):
            problems.append(f"{k} must be true/false")
    for k in LISTS.get(kind, ()):
        if not isinstance(d.get(k, []), list):
            problems.append(f"{k} must be a list")
    if problems:
        raise JudgeError("; ".join(problems))
    return d


def score(kind: str, d: Mapping) -> dict:
    """Each 0-5 dimension -> 0-100 (x20), then the weighted 0-100 score of the kind."""
    _, weights = KINDS[kind]
    dims = {k: float(d[k]) * 20.0 for k in weights}
    return {"dimensions_100": dims, "score_100": round(sum(weights[k] * dims[k] for k in weights), 2)}


# ---------------------------------------------------------------- client
def judge_model(env: Mapping[str, str]) -> str:
    from stillhere.retrieval.deepseek_client import MODEL
    return env.get("STILLHERE_JUDGE_MODEL") or MODEL


def make_client(env: Mapping[str, str], ledger_path, *, transport=None, budget_cny: Optional[float] = None):
    """The repository's DeepSeek client with a judge ledger. Missing key -> a clear error (no silent fallback to another judge).
    A model without a recorded price needs STILLHERE_JUDGE_PRICE_USD_IN / _OUT (USD per million tokens), so the ledger never guesses."""
    from stillhere.retrieval import deepseek_client as DC
    from stillhere.retrieval.alibaba_client import Ledger, _http
    if not env.get("DEEPSEEK_API_KEY"):
        raise JudgeError("DEEPSEEK_API_KEY is not set: add it to Colab Secrets (or the environment); the judge does not fall back to another model")
    model = judge_model(env)
    if model not in DC.PRICES_USD_PER_MILLION:
        pin, pout = env.get("STILLHERE_JUDGE_PRICE_USD_IN"), env.get("STILLHERE_JUDGE_PRICE_USD_OUT")
        if not (pin and pout):
            raise JudgeError(f"no price recorded for judge model {model}: set STILLHERE_JUDGE_PRICE_USD_IN / STILLHERE_JUDGE_PRICE_USD_OUT")
        DC.PRICES_USD_PER_MILLION[model] = {"input_miss": float(pin), "input_hit": float(pin), "output": float(pout)}

    class JudgeLedger(Ledger):
        def __init__(self, path, budget):
            super().__init__(path, budget)
            self.budget = min(float(budget), JUDGE_CEILING_CNY)     # the judge's own ceiling (the reader experiments keep theirs)

    budget = float(budget_cny if budget_cny is not None else env.get("STILLHERE_JUDGE_BUDGET_CNY") or DEFAULT_BUDGET_CNY)
    ledger = JudgeLedger(Path(ledger_path), budget)
    return DC.DeepSeekClient(env, ledger, transport=transport or _http), model


# ---------------------------------------------------------------- units and blinding
def unit_id(kind: str, salt: str, *parts: str) -> str:
    """Opaque, deterministic id: the judge (and the order) never reveal the configuration."""
    return kind[0] + hashlib.sha256("\x1f".join((salt, kind) + tuple(parts)).encode("utf-8")).hexdigest()[:15]


def blind_order(units: Sequence[Mapping], seed: str) -> List[Mapping]:
    """Deterministic shuffle by a keyed hash of the opaque id (configurations interleave; reruns keep the order)."""
    return sorted(units, key=lambda u: hashlib.sha256((seed + u["unit_id"]).encode()).hexdigest())


def _clip(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else "…" + s[-n:]


def style_user(u: Mapping) -> str:
    rep = u.get("repetition") or {}
    return (f"【风格统计（只来自训练集）】\n{u['style_profile']}\n\n【聊天上下文（最后一行是对方最新的话）】\n{_clip(u['context'], 1800)}\n\n"
            f"【这个人当时真实的回复（参考）】\n{u['reference']}\n\n【模型回复】\n{u['response']}\n\n"
            f"【重复情况】这条模型回复在同一组 {rep.get('group_size', '?')} 条回答中出现 {rep.get('count', '?')} 次；"
            f"这组最常见的回复：{rep.get('top', '')}")


def memory_user(u: Mapping) -> str:
    ev = "\n".join(f"[{e.get('id')}] {e.get('time', '')} {e.get('speaker', '')}：{e.get('text', '')}" for e in u.get("evidence", [])) or "（无：这是不可回答题，见构建依据）"
    return (f"【问题】{u['question']}\n【类别】{u['category']}　【主体】{u.get('subject')}　【时间范围】{json.dumps(u.get('time_scope'), ensure_ascii=False)}\n"
            f"【是否可回答】{u['answerability']}{'（依据：' + str(u.get('unanswerable_basis')) + '）' if u.get('unanswerable_basis') else ''}\n"
            f"【参考答案】{u['reference_answer']}\n【参考事实要点】{'；'.join(u.get('reference_facts') or [])}\n\n【证据原文】\n{_clip(ev, 6000)}\n\n"
            f"【模型回答】\n{u['response']}")


def writer_user(u: Mapping) -> str:
    msgs = "\n".join(f"[{m['label']}] {m['time']} {m['speaker']}：{m['text']}" for m in u["messages"])
    cands = "\n".join(f"[C{i + 1}] {c['status']} | 主体 {c['subject']} | {c['predicate']} | {c['object']}"
                      f"{' | 状态 ' + c['event_status'] if c.get('event_status') else ''}{' | 时间 ' + str(c['time']) if c.get('time') else ''}"
                      f" | 引用 {','.join(c['labels'])}{' | 拦下原因 ' + c['reasons'] if c.get('reasons') else ''}"
                      for i, c in enumerate(u["candidates"])) or "（写入器没有给出候选）"
    return f"【聊天消息】\n{_clip(msgs, 9000)}\n\n【写入器候选】\n{cands}"


USERS = {"style": style_user, "memory": memory_user, "writer": writer_user}
MAX_TOKENS = {"style": 300, "memory": 350, "writer": 1200}


def judge_one(client, model: str, kind: str, unit: Mapping) -> dict:
    """One unit: the rubric + the unit text -> validated JSON (one repair turn on a schema problem) -> 0-100 score."""
    system, _ = KINDS[kind]
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": USERS[kind](unit)}]
    usage = {"input_tokens": 0, "output_tokens": 0, "cost_cny": 0.0, "calls": 0}
    last = None
    for attempt in range(2):
        r = client.chat(model, msgs, max_tokens=MAX_TOKENS[kind], temperature=0.0, json_mode=True)
        usage["input_tokens"] += r.input_tokens
        usage["output_tokens"] += r.output_tokens
        usage["cost_cny"] += r.cost_cny
        usage["calls"] += 1
        try:
            d = validate(kind, r.text)
            return {"status": "OK", "judgement": d, **score(kind, d), "resolved_model": r.model, "usage": usage, "repair_turns": attempt}
        except JudgeError as e:
            last = str(e)
            msgs = msgs + [{"role": "assistant", "content": r.text}, {"role": "user", "content": f"格式不符合要求：{last}。请只输出符合要求的 JSON 对象。"}]
    raise JudgeError(f"schema failed after repair: {last}")


# ---------------------------------------------------------------- checkpointed runner
def read_results(path: Path) -> Dict[str, dict]:
    """Latest row per unit (OK rows win; failures keep their attempt count)."""
    out: Dict[str, dict] = {}
    if Path(path).is_file():
        for l in Path(path).read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            prev = out.get(r["unit_id"])
            if prev is not None and prev.get("status") == "OK":
                continue
            if prev is not None and r.get("status") != "OK":
                r["attempts"] = prev.get("attempts", 1) + 1
            out[r["unit_id"]] = r
    return out


class RateLimit:
    def __init__(self, per_second: float):
        self.gap = 1.0 / per_second if per_second > 0 else 0.0
        self.lock = threading.Lock()
        self.next = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.gap
        if t > now:
            time.sleep(t - now)


def run(client, model: str, kind: str, units: Sequence[Mapping], out_path, *, seed: str, concurrency: int = 4, per_second: float = 4.0,
        log: Callable[[str], None] = print) -> dict:
    """Judge every unit not yet scored OK (resume), in the blind order, ``concurrency`` at a time under a rate limit. A failed unit is
    written as JUDGE_FAILED (with its error) and retried on a later run until ``MAX_UNIT_ATTEMPTS``. Returns the counts."""
    from stillhere.retrieval.alibaba_client import BudgetExceeded
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = read_results(out_path)
    todo = [u for u in blind_order(units, seed) if done.get(u["unit_id"], {}).get("status") != "OK" and done.get(u["unit_id"], {}).get("attempts", 0) < MAX_UNIT_ATTEMPTS]
    log(f"judge {kind}: {len(units)} planned, {len(units) - len(todo)} already scored / given up, {len(todo)} to send")
    lock, rl = threading.Lock(), RateLimit(per_second)
    stop = threading.Event()
    counter = {"n": 0}

    def one(u):
        if stop.is_set():
            return
        rl.wait()
        meta = {"unit_id": u["unit_id"], "kind": kind, "judge_model": model, "rubric_version": RUBRIC_VERSION, "prompt_version": PROMPT_VERSION}
        try:
            row = {**meta, **judge_one(client, model, kind, u)}
        except BudgetExceeded as e:
            stop.set()
            row = {**meta, "status": "JUDGE_FAILED", "error": f"BudgetExceeded: {e}"}
        except Exception as e:  # noqa: BLE001 - recorded, never scored
            from stillhere.retrieval.alibaba_client import redact
            row = {**meta, "status": "JUDGE_FAILED", "error": redact(f"{type(e).__name__}: {str(e)[:300]}")}
        row["at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            counter["n"] += 1
            if counter["n"] % 50 == 0:
                log(f"judge {kind}: {counter['n']}/{len(todo)}")

    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as ex:
        list(ex.map(one, todo))
    final = read_results(out_path)
    ids = {u["unit_id"] for u in units}
    ok = sum(1 for i in ids if final.get(i, {}).get("status") == "OK")
    return {"kind": kind, "planned": len(ids), "ok": ok, "failed": len(ids) - ok, "budget_stopped": stop.is_set(),
            "failed_units": sorted(i for i in ids if final.get(i, {}).get("status") != "OK")[:50]}


# ---------------------------------------------------------------- calibration (reported separately; not part of the formal units)
CAL_PROFILE = "回复长度：中位数 3 字，常见 1–6 字；常用短句：没呢、好、哈哈哈、嗯嗯；很少用句号；不用敬语。"
CALIBRATION = [
    {"kind": "style", "name": "semantic_equivalent_short", "expect": ("score_100", ">=", 70),
     "unit": {"context": "对方：你吃饭了没", "reference": "没呢", "response": "没吃呢", "style_profile": CAL_PROFILE, "repetition": {"group_size": 268, "count": 1, "top": "好(9)"}}},
    {"kind": "style", "name": "off_topic_short", "expect": ("contextual_appropriateness", "<=", 2),
     "unit": {"context": "对方：你现在在哪家公司上班呀？", "reference": "还在之前那家", "response": "没呢", "style_profile": CAL_PROFILE,
              "repetition": {"group_size": 268, "count": 40, "top": "没呢(40)"}}},
    {"kind": "style", "name": "assistant_tone", "expect": ("naturalness", "<=", 2),
     "unit": {"context": "对方：今天好累", "reference": "抱抱", "response": "作为你的朋友，我理解你的感受。希望你能好好休息，如果有需要随时告诉我。", "style_profile": CAL_PROFILE,
              "repetition": {"group_size": 268, "count": 1, "top": "好(9)"}}},
    {"kind": "memory", "name": "short_correct", "expect": ("factual_correctness", ">=", 4),
     "unit": {"question": "你以前在哪个城市工作？", "category": "PERSONA_SELF_EXPERIENCE", "subject": "PERSONA_SELF", "time_scope": {"period": "past"},
              "answerability": "ANSWERABLE", "reference_answer": "以前在广州工作（后来换到深圳）。", "reference_facts": ["本人以前在广州工作"],
              "evidence": [{"id": "m1", "time": "2025-01-01 09:01", "speaker": "本人", "text": "对，我之前在广州上班"}], "response": "广州啊"}},
    {"kind": "memory", "name": "wrong_period", "expect": ("factual_correctness", "<=", 2),
     "unit": {"question": "你以前在哪个城市工作？", "category": "TEMPORAL_CHANGE", "subject": "PERSONA_SELF", "time_scope": {"period": "past"},
              "answerability": "ANSWERABLE", "reference_answer": "以前在广州，现在在深圳。", "reference_facts": ["本人以前在广州工作", "现在在深圳工作"],
              "evidence": [{"id": "m1", "time": "2025-01-01", "speaker": "本人", "text": "我之前在广州上班"}, {"id": "m2", "time": "2025-01-05", "speaker": "本人", "text": "现在换成在深圳上班了"}],
              "response": "深圳"}},
    {"kind": "memory", "name": "wrong_person", "expect": ("temporal_attribution", "<=", 2),
     "unit": {"question": "我以前在哪读书来着？", "category": "COUNTERPART_EXPERIENCE", "subject": "CONVERSATION_COUNTERPART", "time_scope": {"period": "past"},
              "answerability": "ANSWERABLE", "reference_answer": "你在澳洲读书。", "reference_facts": ["对方在澳洲读书"],
              "evidence": [{"id": "m2", "time": "2025-01-01", "speaker": "对方", "text": "我在澳洲读书呢"}, {"id": "m3", "time": "2025-01-01", "speaker": "本人", "text": "我在广州上班"}],
              "response": "你在广州上班呀"}},
    {"kind": "memory", "name": "unanswerable_abstain", "expect": ("score_100", ">=", 70),
     "unit": {"question": "你家的猫叫什么名字？", "category": "UNANSWERABLE", "subject": "PERSONA_SELF", "time_scope": None, "answerability": "UNANSWERABLE",
              "unanswerable_basis": "全部聊天记录中没有出现“猫”", "reference_answer": "没有记录，应表示不知道或不记得。", "reference_facts": [], "evidence": [], "response": "不记得了"}},
    {"kind": "memory", "name": "unanswerable_fabricated", "expect": ("evidence_grounding", "<=", 1),
     "unit": {"question": "你家的猫叫什么名字？", "category": "UNANSWERABLE", "subject": "PERSONA_SELF", "time_scope": None, "answerability": "UNANSWERABLE",
              "unanswerable_basis": "全部聊天记录中没有出现“猫”", "reference_answer": "没有记录，应表示不知道或不记得。", "reference_facts": [], "evidence": [],
              "response": "它叫咪咪，三岁了"}},
    {"kind": "memory", "name": "answerable_but_abstained", "expect": ("appropriate_abstention", "<=", 1),
     "unit": {"question": "你以前在哪个城市工作？", "category": "PERSONA_SELF_EXPERIENCE", "subject": "PERSONA_SELF", "time_scope": {"period": "past"},
              "answerability": "ANSWERABLE", "reference_answer": "广州。", "reference_facts": ["本人以前在广州工作"],
              "evidence": [{"id": "m1", "time": "2025-01-01", "speaker": "本人", "text": "我之前在广州上班"}], "response": "不记得了"}},
]


def _check(row: Mapping, expect) -> bool:
    key, op, val = expect
    v = row.get("score_100") if key == "score_100" else (row.get("judgement") or {}).get(key)
    if v is None:
        return False
    return v >= val if op == ">=" else v <= val


def calibrate(client, model: str, out_path, *, log: Callable[[str], None] = print) -> dict:
    """Run the synthetic calibration cases once (checkpointed like the formal units); report which anchors the judge respects."""
    rows = []
    for c in CALIBRATION:
        u = {**c["unit"], "unit_id": unit_id(c["kind"], "calibration", c["name"])}
        res = run(client, model, c["kind"], [u], out_path, seed="calibration", concurrency=1, log=lambda s: None)
        r = read_results(Path(out_path)).get(u["unit_id"], {})
        rows.append({"name": c["name"], "kind": c["kind"], "expect": list(c["expect"]), "status": r.get("status"), "score_100": r.get("score_100"),
                     "passed": r.get("status") == "OK" and _check(r, c["expect"]), "ok": res["ok"]})
    passed = sum(r["passed"] for r in rows)
    rep = {"cases": rows, "passed": passed, "total": len(rows), "status": "PASS" if passed == len(rows) else "PARTIAL",
           "note": "synthetic calibration anchors; NOT part of the 1,772 formal units; reported separately"}
    log(f"judge calibration: {passed}/{len(rows)}")
    return rep


# ---------------------------------------------------------------- statistics
def describe(xs: Sequence[float]) -> dict:
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    n = len(s)
    mean = sum(s) / n
    sd = (sum((x - mean) ** 2 for x in s) / (n - 1)) ** 0.5 if n > 1 else 0.0
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    hist = {f"{b}-{b + 9 if b < 90 else 100}": sum(1 for x in s if b <= x < b + 10 or (b == 90 and x == 100)) for b in range(0, 100, 10)}
    return {"n": n, "mean": round(mean, 2), "median": round(med, 2), "sd": round(sd, 2), "min": round(s[0], 2), "max": round(s[-1], 2), "histogram": hist}


def paired(a: Mapping[str, float], b: Mapping[str, float], *, seed: int = 20261117, boot: int = 2000) -> dict:
    """b - a over the items both have: mean difference, a seeded bootstrap 95% CI, win / tie / loss."""
    keys = sorted(set(a) & set(b))
    d = [b[k] - a[k] for k in keys]
    if not d:
        return {"n": 0}
    rnd = random.Random(seed)
    means = sorted(sum(d[rnd.randrange(len(d))] for _ in d) / len(d) for _ in range(boot))
    return {"n": len(d), "mean_diff": round(sum(d) / len(d), 2), "ci95": [round(means[int(0.025 * boot)], 2), round(means[int(0.975 * boot) - 1], 2)],
            "wins": sum(x > 0 for x in d), "ties": sum(x == 0 for x in d), "losses": sum(x < 0 for x in d)}


def make_ledger_summary(path) -> dict:
    """Requests / tokens / estimated cost of every judge HTTP attempt (from the append-only ledger)."""
    from stillhere.retrieval.alibaba_client import Ledger
    if not Path(path).is_file():
        return {"requests": 0}
    s = Ledger(Path(path), 0.0).summary()
    return {k: v for k, v in s.items() if k not in ("budget_cny", "remaining_cny")}


def run_failures(path) -> List[dict]:
    return [{"unit_id": u, "error": r.get("error"), "attempts": r.get("attempts", 1)} for u, r in read_results(Path(path)).items() if r.get("status") != "OK"]
