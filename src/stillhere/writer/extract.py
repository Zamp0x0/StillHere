"""Candidate extraction: one conversation group in, candidate facts out, through an injected model call.

The model sees the whole group (every message, numbered, with time, sender and recipient) and the persona's current memories,
and answers in JSON object mode. DeepSeek's JSON mode guarantees JSON, not a schema, so ``validate`` checks every field here;
an answer that fails is sent back once with the list of problems, and a second failure yields no candidates (the group is
logged, nothing is written). The answer is cached by the exact request, so a re-run reads the same candidates.
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .contracts import CANDIDATE_TYPES, EVENT_STATES, PRECISIONS, PREDICATES, Candidate

PROMPT_VERSION = "writer-extract-v1"

SYSTEM = """你是长期记忆系统的事实提取器。你只从对话原文中提取明确说出的原子事实，输出 JSON。

规则：
1. 只提取对话里明确说出的内容。不推断、不补充、不总结性格。说话人没说的，不写。
2. 区分说话人：subject 为 "self" 的事实，必须是机主本人（标为“本人”）自己说的。AI 助手或他人说的话不能当作机主的事实；他人关于自己的陈述，subject 写 "person:姓名"。
3. 保留否定：“不太喜欢咖啡”是 dislikes 咖啡，不是 likes 咖啡。
4. 计划不是已发生：“明天打算去日本”是 event_status PLANNED，不是 COMPLETED。改期、取消、完成要分别写出，并以最后的说法为准写出最终状态。
5. 每条事实给出 source_message_ids（如 ["m2"]）和 source_spans：从原文逐字复制的片段，必须一字不差。
6. 日期：只有原文提到日期或“明天、下周三、昨天”这类说法时才填写，按消息时间换算成 YYYY-MM-DD（只知道月份写 YYYY-MM）；没提就填 null，不要猜。
7. 更正和撤回：如果说话人更正了先前的说法（说错了、记错了、其实是……），corrects 为 true；如果收回一件事（那件事不算、我没有……），retracts 为 true。若对应“已有记忆”中的某条，在 refers_to 填它的 id。
8. 不确定的说法（可能、也许、好像）certainty 写 "uncertain"。
9. 同一件事被说了多次，只输出一条。闲聊、问候没有新事实时，输出空列表。

predicate 只能是以下之一：
lives_in 住在哪里 | works_at 在哪工作 | job 职业 | school 学校 | age 年龄 | birthday 生日 | pet 宠物 | owns 拥有的东西 |
relationship 人际关系（object 写“关系 姓名”，如“妹妹 小雨”）| likes 喜欢 | dislikes 不喜欢 |
routine 自述的固定安排（如“早上读书”）| state 状态（object 写“方面:取值”，如“睡眠:差”）| event 一件事（计划、预约、进行中或已完成）

event 还要填：event_key（这件事的简短英文标识，如 "trip_japan"）、activity（核心活动，2–4 个字，如“读书”“游泳”“旅行”）、
event_status（PLANNED 计划、SCHEDULED 已预约或已订好、IN_PROGRESS 正在进行、COMPLETED 已完成、CANCELLED 已取消、UNKNOWN 不明）、
planned_time（计划时间）、event_time（实际发生时间）、daypart（早上/上午/中午/下午/晚上/夜里，原文没说填 null）。

只输出：
{"candidates": [{"type": "FACT|EVENT|PREFERENCE_STATEMENT|PLAN|COMMITMENT|CORRECTION|RETRACTION|UNCERTAINTY", "subject": "self", "predicate": "...", "object": "...",
"polarity": "positive|negative", "assertion_type": "stated|inferred", "certainty": "certain|uncertain", "event_key": null, "activity": null, "event_status": null,
"planned_time": null, "event_time": null, "daypart": null, "valid_from": null, "valid_to": null, "temporal_precision": "exact|day|month|year|unknown",
"corrects": false, "retracts": false, "refers_to": null, "source_message_ids": ["m1"], "source_spans": ["逐字片段"]}]}"""

SYSTEM_V2 = SYSTEM.replace(
    "9. 同一件事被说了多次，只输出一条。闲聊、问候没有新事实时，输出空列表。",
    "9. 同一件事在这段对话里被说了多次，只输出一条。闲聊、问候没有新事实时，输出空列表。\n"
    "10. 职业和工作单位分开写：“在一家医院当护士”是 job 护士，加上 works_at 一家医院。\n"
    "11. 对话再次提到“已有记忆”里的事实（例如又提到同事老王），照样输出这一条，refers_to 填它的 id；系统会去重，并把它记为又一次独立的提及。")
SYSTEM_V3 = SYSTEM_V2.replace("如“妹妹 小雨”", "如“表哥 阿杰”")         # v3: v2 with no name of the evaluation data in any example
# v4 (Batch 2.1): v3 + explicit rules on whose fact it is, what is not a fact, and one event per occurrence
SYSTEM_V4 = SYSTEM_V3.replace(
    "11. 对话再次提到“已有记忆”里的事实（例如又提到同事老王），照样输出这一条，refers_to 填它的 id；系统会去重，并把它记为又一次独立的提及。",
    "11. 对话再次提到“已有记忆”里的事实（例如又提到同事老王），照样输出这一条，refers_to 填它的 id；系统会去重，并把它记为又一次独立的提及。\n"
    "12. 主体归属：别人（朋友、同事、家人）的喜好、宠物、习惯、身体状况不要输出——既不是本人的事实，也不需要记；别人自己的工作、住处、学校、"
    "以及他们自己的计划和经历，subject 写 \"person:姓名\"。别人说本人怎样（“阿雷说我喜欢猫”“人家都说好”）不是本人自己说的，不输出。\n"
    "13. 只提取确定的事：假设（如果、要是……就）、建议或提问（要不……？、医生建议……）、愿望（想去、希望）、猜测（我觉得他可能……）都不输出。"
    "说了“可能、还没定、还没约”的计划，certainty 写 \"uncertain\"。\n"
    "14. 一件事只输出一条 event：原因、结果和细节（“延期了”“脸肿了”“每周三上课”“回不了家”）不要另写成 event 或 routine。"
    "改期、取消、完成、没办成都是同一件事的状态更新：refers_to 填已有记忆里这件事的 id。\n"
    "15. 不再喜欢、戒了、不吃了：写 dislikes，polarity 写 positive。")
# v5 (persona / WeFlow two-person chats; v4 unchanged for LifeBench): who owns a statement in a two-person chat, cancellations that refer to an
# existing event, third-party events, and momentary chat states that are not long-term memory
SYSTEM_V5 = SYSTEM_V4.replace(
    "15. 不再喜欢、戒了、不吃了：写 dislikes，polarity 写 positive。",
    "15. 不再喜欢、戒了、不吃了：写 dislikes，polarity 写 positive。\n"
    "16. 这是两个人的聊天：本人和对方。对方说“我……”说的是对方自己，subject 写 \"person:对方\"，绝不能写成 self；对方向本人提的问题、猜测和建议不是事实，不输出。\n"
    "17. 取消、改期、完成一件已有记忆里的事：object 写这件事的活动（如“看牙医”），event_status 写新状态（CANCELLED / COMPLETED / SCHEDULED），"
    "refers_to 填那件事的 id；不要把 id 写进 object。\n"
    "18. 第三方（家人、朋友、同事）的事也要输出：subject 写 \"person:姓名\"，例如“我妹妹小雨下个月结婚”输出 relationship（妹妹 小雨）和一条 event："
    "subject \"person:小雨\"、activity “结婚”、event_status PLANNED。\n"
    "19. 聊天中即时的动作或状态（“在呢”“去啦”“吃饭去了”“好困”这类当下的临时状态、寒暄、表情互动）不是长期记忆，不输出；"
    "有明确时间、计划、变化或对以后有意义的才输出。")
PROMPTS = {"writer-extract-v1": SYSTEM, "writer-extract-v2": SYSTEM_V2, "writer-extract-v3": SYSTEM_V3, "writer-extract-v4": SYSTEM_V4,
           "writer-extract-v5": SYSTEM_V5}

USER = """机主：{persona_name}（对话中标为“本人”）
已有记忆（id | 主体 | 属性 | 取值）：
{existing}

对话（编号 时间 发送方 → 接收方：原文）：
{messages}"""

REPAIR = "你上一次的输出不符合要求：{problems}。请只输出修正后的完整 JSON，不要解释。"


@dataclass
class GroupMessage:
    label: str
    message_id: str
    speaker_id: str
    sender: str
    recipient: str
    timestamp: object
    recorded_at: object
    content: str
    source_id: str


def render_messages(msgs: Sequence[GroupMessage]) -> str:
    return "\n".join(f"[{m.label}] {m.timestamp.strftime('%Y-%m-%d %H:%M')} {m.sender} → {m.recipient}：{m.content}" for m in msgs)


def render_existing(existing: Sequence[Tuple[str, str, str, str]]) -> str:
    return "\n".join(f"{mid} | {subj} | {attr} | {val}" for mid, subj, attr, val in existing) or "（无）"


def build_messages(persona_name: str, msgs: Sequence[GroupMessage], existing, version: str = PROMPT_VERSION) -> List[Dict[str, str]]:
    return [{"role": "system", "content": PROMPTS[version]}, {"role": "user", "content": USER.format(persona_name=persona_name, existing=render_existing(existing), messages=render_messages(msgs))}]


def prompt_sha256(version: str = PROMPT_VERSION) -> str:
    return hashlib.sha256((PROMPTS[version] + USER + REPAIR).encode("utf-8")).hexdigest()


_STR_OR_NONE = ("event_key", "activity", "event_status", "planned_time", "event_time", "daypart", "valid_from", "valid_to", "refers_to")


def validate(raw: object, labels: Mapping[str, str]) -> Tuple[List[dict], List[str]]:
    """(valid candidate dicts, problems). Every field is checked; a candidate with any problem is not returned."""
    problems: List[str] = []
    if not isinstance(raw, dict) or not isinstance(raw.get("candidates"), list):
        return [], ["top level must be {\"candidates\": [...]}"]
    out = []
    for i, c in enumerate(raw["candidates"]):
        p = []
        if not isinstance(c, dict):
            problems.append(f"candidates[{i}] is not an object")
            continue
        if c.get("type") not in CANDIDATE_TYPES:
            p.append("type")
        if c.get("predicate") not in PREDICATES:
            p.append("predicate not in the list")
        for name in ("subject", "object"):
            if not isinstance(c.get(name), str) or not c.get(name).strip():
                p.append(name)
        if not (c.get("subject") == "self" or str(c.get("subject", "")).startswith("person:")):
            p.append("subject must be self or person:<name>")
        if c.get("polarity", "positive") not in ("positive", "negative") or c.get("assertion_type", "stated") not in ("stated", "inferred") or c.get("certainty", "certain") not in ("certain", "uncertain"):
            p.append("polarity/assertion_type/certainty")
        if c.get("temporal_precision", "unknown") not in PRECISIONS:
            p.append("temporal_precision")
        if c.get("event_status") not in (None, *EVENT_STATES):
            p.append("event_status")
        for name in _STR_OR_NONE:
            if c.get(name) is not None and not isinstance(c.get(name), str):
                p.append(name)
        for name in ("corrects", "retracts"):
            if not isinstance(c.get(name, False), bool):
                p.append(name)
        ids, spans = c.get("source_message_ids"), c.get("source_spans")
        if not isinstance(ids, list) or not ids or any(x not in labels for x in ids):
            p.append("source_message_ids must name messages of the conversation (m1, m2, ...)")
        if not isinstance(spans, list) or not spans or not all(isinstance(x, str) and x.strip() for x in spans):
            p.append("source_spans")
        if c.get("predicate") == "event" and (not c.get("event_key") or not c.get("event_status")):
            p.append("an event needs event_key and event_status")
        if p:
            problems.append(f"candidates[{i}]: {', '.join(p)}")
        else:
            out.append(c)
    return out, problems


class ExtractionCache:
    """Append-only JSONL of model answers keyed by the exact request."""

    def __init__(self, path: Optional[Path]) -> None:
        self.path = path
        self.rows: Dict[str, dict] = {}
        self._lock = threading.Lock()
        if path and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.rows[r["key"]] = r

    @staticmethod
    def key(model: str, messages) -> str:
        return hashlib.sha256(json.dumps({"model": model, "messages": messages}, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[dict]:
        return self.rows.get(key)

    def put(self, row: dict) -> None:
        with self._lock:
            self.rows[row["key"]] = row
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")


Complete = Callable[[List[Dict[str, str]]], dict]      # -> {"text", "input_tokens", "output_tokens", "cost_cny", "latency_ms", "model"}


def extract(complete: Complete, model: str, persona_id: str, persona_name: str, msgs: Sequence[GroupMessage], existing, cache: ExtractionCache,
            version: str = PROMPT_VERSION) -> Tuple[List[Candidate], dict]:
    """Candidates of one group and the extraction record (raw answers, problems, usage). At most two model calls."""
    labels = {m.label: m.message_id for m in msgs}
    request = build_messages(persona_name, msgs, existing, version)
    key = cache.key(model, request)
    hit = cache.get(key)
    if hit is None:
        attempts = []
        conversation = list(request)
        valid: List[dict] = []
        for attempt in range(2):
            reply = complete(conversation)
            try:
                parsed = json.loads(reply["text"])
                valid, problems = validate(parsed, labels)
            except (ValueError, TypeError):
                parsed, valid, problems = None, [], ["not valid JSON"]
            attempts.append({"attempt": attempt, "text": reply["text"], "problems": problems, **{k: reply.get(k) for k in ("input_tokens", "output_tokens", "cost_cny", "latency_ms", "model")}})
            if not problems:
                break
            if attempt == 0:
                conversation = conversation + [{"role": "assistant", "content": reply["text"]}, {"role": "user", "content": REPAIR.format(problems="；".join(problems[:8]))}]
        hit = {"key": key, "model": model, "prompt_version": version, "candidates": valid, "attempts": attempts, "schema_ok": not attempts[-1]["problems"]}
        cache.put(hit)
    out = []
    for i, c in enumerate(hit["candidates"]):
        out.append(Candidate(candidate_id=f"{key[:12]}.{i}", persona_id=persona_id, type=c["type"], subject=c["subject"].strip(), predicate=c["predicate"], object=c["object"].strip(),
                             polarity=c.get("polarity", "positive"), assertion_type=c.get("assertion_type", "stated"), certainty=c.get("certainty", "certain"), event_key=c.get("event_key"),
                             event_status=c.get("event_status"), activity=c.get("activity"), daypart=c.get("daypart"), planned_time=c.get("planned_time"), event_time=c.get("event_time"),
                             valid_from=c.get("valid_from"), valid_to=c.get("valid_to"), temporal_precision=c.get("temporal_precision", "unknown"), corrects=bool(c.get("corrects")),
                             retracts=bool(c.get("retracts")), refers_to=c.get("refers_to"), source_message_ids=[labels[x] for x in c["source_message_ids"]], source_spans=list(c["source_spans"]),
                             extraction_version=version))
    return out, hit
