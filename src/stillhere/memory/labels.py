"""Display labels for message sources and speakers (shared by EF1 retrieval, the dense index text and the fact evidence pack)."""
from __future__ import annotations

from typing import Mapping

SOURCE_LABELS = {"lifebench_sms": "短信", "lifebench_agent_chat": "AI助手对话", "lifebench_call": "通话", "lifebench_calendar": "日程",
                 "lifebench_note": "笔记", "lifebench_photo": "照片", "lifebench_push": "推送", "lifebench_fitness_health": "健康记录"}


def source_label(source_type: str) -> str:
    return SOURCE_LABELS.get(source_type, source_type)


def speaker_label(persona_id: str, speaker_id: str, names: Mapping[str, str] = {}) -> str:
    if speaker_id == persona_id:
        return names.get(persona_id, "本人")
    if speaker_id.endswith(".assistant"):
        return "AI助手"
    return names.get(speaker_id, "其他人")
