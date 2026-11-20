"""Prompt renderer ``stillhere.response.grounded_persona/1.0`` (D4). New and versioned: the frozen T2-D3 Contract v2
(``stillhere.style.prompt/2.0``) is NOT modified; this renderer reuses its intro, example note and refusal rules for special tokens and adds
structured Memory Evidence (subject, status, time, disclosure) and a separate conversation block.

Training, every evaluation group (Base and Adapter alike) and the runtime must call ``render_messages`` with the same inputs. The three
sources are kept apart, each in its own labelled block:

    【说话方式】      Style Profile Schema v1 (how to say it)
    【说话方式示例】  K real (or SYNTHETIC_PERSONA) earlier messages (how to say it; never facts)
    【记忆证据】      Memory Evidence items (what may be said) + unknown slots + the conversation date
    【生成要求】      fixed rules
    user message:    the conversation so far (对方 / 我 lines)

Ablations are expressed by inputs only (profile=None, examples=[], evidence=None, a swapped persona, a wrong memory state), never by a
different template, so A/B/C/D and every ablation share one renderer and one hash.
"""
from __future__ import annotations

import hashlib
import re
from typing import List, Mapping, Optional, Sequence

from .evidence import Evidence, UnknownSlot
from .profile_v1 import render as render_profile

RENDERER_VERSION = "stillhere.response.grounded_persona/1.0"

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|<>\s]{1,64}\|>")
_RESERVED_TAGS = ("<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>")

INTRO = ("你要以一位特定说话者的身份回复对方。“说话方式”和“示例”只决定怎么说；"
         "说什么只能来自“记忆证据”和对话本身。")
EXAMPLES_NOTE = "（此人过去在其他场合说的话，只示范说话方式；其中提到的经历和事实不属于本次对话，不能当作事实使用）"
NO_EXAMPLES = "（没有可用的示例）"
NO_PROFILE = "- （没有可用的说话方式描述：用自然、简洁的口吻）"
NO_EVIDENCE = "（没有提供记忆证据：可以对对话内容表达态度、看法或追问，但不要编造具体的个人经历、数字、时间、价格、地点或名称）"
STATUS_LABEL = {"CURRENT": "当前有效", "HISTORICAL": "过去成立，现在已不成立", "CORRECTED": "已更正", "CANCELLED": "已取消",
                "CONFLICTED": "有冲突，未确认", "UPCOMING": "已确定，尚未生效"}
DISCLOSURE_LABEL = {"ALLOWED": "可以说", "PROTECTED": "不可透露", "POLICY_UNCERTAIN": "未确认能否透露，先不要说"}
CONF_LABEL = {"high": "高", "medium": "中", "low": "低"}
RULES = (
    "只说标为“可以说”的证据和对话里出现过的内容；不要新增证据之外的具体经历、身份、数字、时间、地点或名称；",
    "“当前有效”的照实说，不改数字、时间和否定；“过去成立”的不能说成现在；“已更正”的只用更正后的说法；"
    "“已取消”的要说明取消了；“尚未生效”的不要说成已经发生；“有冲突，未确认”的不要替对方选一个，要说明不确定；",
    "证据里写明是别人的事，就说是那个人的，不要说成自己的；",
    "标为“不可透露”或“未确认能否透露”的内容不要说出来（包括其中的部分数字），自然地婉拒或岔开；其他能说的照常回答，不要一律拒绝；",
    "问到没有记录的事，自然地说不知道、还没定或没记，不要猜；",
    "不要照抄示例里的句子，也不要把示例里的经历当成这次的事实；",
    "顺着对方这次说的话回应；回复可以分成几条消息，每条一行。",
)


class RenderError(ValueError):
    pass


def _check(text: str, where: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise RenderError(f"{where} must be a non-empty string")
    if _SPECIAL_TOKEN_RE.search(text) or any(t in text for t in _RESERVED_TAGS):
        raise RenderError(f"{where} contains a special-token or reserved marker")
    return text.strip()


def _time(e: Evidence) -> str:
    if e.valid_from and e.valid_to:
        return f"{e.valid_from} 至 {e.valid_to}"
    if e.valid_from:
        return f"{e.valid_from} 起"
    if e.valid_to:
        return f"到 {e.valid_to} 为止"
    return "未注明"


def render_evidence(evidence: Optional[Sequence[Evidence]], unknown: Sequence[UnknownSlot] = (), now: Optional[str] = None) -> str:
    if evidence is None and not unknown:
        return NO_EVIDENCE
    lines = []
    if now:
        lines.append(f"（对话日期：{_check(now, 'now')}）")
    for e in evidence or ():
        e.validate()
        head = (f"[{e.fact_id}] 关于：{_check(e.subject, 'subject')}｜状态：{STATUS_LABEL[e.status]}｜时间：{_time(e)}｜"
                f"可信度：{CONF_LABEL[e.confidence]}｜{DISCLOSURE_LABEL[e.disclosure_policy]}")
        if e.conflict_group:
            head += f"｜冲突组：{e.conflict_group}"
        body = "    " + _check(e.evidence_text, f"{e.fact_id}.evidence_text")
        if e.status == "CORRECTED":
            body += f"（之前记成“{_check(e.replaces, 'replaces')}”，那是记错了）"
        if e.status == "CANCELLED":
            body += f"（{_check(e.note, 'note')}）"
        lines += [head, body]
    for u in unknown:
        lines.append(f"[未知] {_check(u.label, 'unknown.label')}：没有记录，不要猜")
    if not (evidence or unknown):
        lines.append("（这次没有找到相关记忆证据：不要编造具体经历、数字、时间、地点或名称）")
    return "\n".join(lines)


def render_system(profile: Optional[Mapping], examples: Sequence[str], evidence: Optional[Sequence[Evidence]], unknown: Sequence[UnknownSlot] = (),
                  now: Optional[str] = None, extra_rules: Sequence[str] = ()) -> str:
    ex = [_check(x.replace("\n", " / "), f"example[{i}]") for i, x in enumerate(examples)]
    parts = [INTRO, "【说话方式】", render_profile(profile) if profile else NO_PROFILE, "【说话方式示例】"]
    parts += ([EXAMPLES_NOTE] + [f"例{i + 1}：{x}" for i, x in enumerate(ex)]) if ex else [NO_EXAMPLES]
    parts += ["【记忆证据】", render_evidence(evidence, unknown, now), "【生成要求】"]
    rules = list(RULES) + [_check(r, "extra_rule") for r in extra_rules]
    parts += [f"{i + 1}. {r}" for i, r in enumerate(rules)]
    return "\n".join(parts)


def render_user(conversation: Sequence[Mapping]) -> str:
    if not conversation or conversation[-1]["speaker"] != "partner":
        raise RenderError("the conversation must end with the partner's turn")
    lines = ["【对话】"]
    for i, t in enumerate(conversation):
        who = {"partner": "对方", "self": "我"}[t["speaker"]]
        for b in _check(t["text"], f"turn[{i}]").split("\n"):
            if b.strip():
                lines.append(f"{who}：{b.strip()}")
    return "\n".join(lines)


def render_messages(*, profile, examples, evidence, unknown=(), now=None, conversation, extra_rules=()) -> List[dict]:
    return [{"role": "system", "content": render_system(profile, examples, evidence, unknown, now, extra_rules)},
            {"role": "user", "content": render_user(conversation)}]


def renderer_sha256() -> str:
    """Hash of the renderer's fixed text (version, labels, rules, intro): changes whenever the rendered contract changes."""
    blob = "\x1f".join([RENDERER_VERSION, INTRO, EXAMPLES_NOTE, NO_EXAMPLES, NO_PROFILE, NO_EVIDENCE, repr(sorted(STATUS_LABEL.items())),
                        repr(sorted(DISCLOSURE_LABEL.items())), repr(sorted(CONF_LABEL.items())), *RULES])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
