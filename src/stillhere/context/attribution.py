"""Who sent a message and who received it, from stored metadata only. Missing metadata stays unknown; nothing is guessed.

``sms_direction`` is the BUILD-V3.4 rule (it moved here from ``evaluation.evidence_render``, which re-exports it): the stored SMS
type says which side the user was on, and the conversation names the other party. ``round_direction`` does the same for a
two-party chat round: each message goes from its speaker to the one other participant; with any other number of
participants the recipient is unknown.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

SELF = "本人"
UNKNOWN = "未知"


@dataclass(frozen=True)
class Direction:
    sender: str
    recipient: str


def locator_field(locator: Optional[str], name: str) -> Optional[str]:
    return next((part.split("=", 1)[1] for part in (locator or "").split(";") if part.startswith(name + "=")), None)


def sms_direction(locator: Optional[str], speaker: Optional[str], counterpart: Optional[str]) -> Optional[Direction]:
    """Both ends of one SMS from its stored type. None when the type or the other party is not known, so nothing is guessed."""
    kind = locator_field(locator, "type")
    if not counterpart:
        return None
    if kind == "发送":
        return Direction(SELF, counterpart)
    if kind == "接收":
        return Direction(speaker or counterpart, SELF)
    return None


def sms_counterpart(conversation_id: str, persona_id: str, speakers: Mapping[str, str]) -> Optional[str]:
    """The other party of an SMS thread ``<prefix>.sms.<suffix>``: the one known speaker of that prefix whose id ends in the suffix."""
    if ".sms." not in conversation_id:
        return None
    prefix, suffix = conversation_id.rsplit(".sms.", 1)
    found = [name for sid, name in speakers.items() if sid.startswith(prefix + ".") and sid.rsplit(".", 1)[-1] == suffix and sid != persona_id]
    return found[0] if len(found) == 1 else None


def name_of(speaker_id: Optional[str], persona_id: str, speakers: Mapping[str, str]) -> str:
    if not speaker_id:
        return UNKNOWN
    return SELF if speaker_id == persona_id else speakers.get(speaker_id, speaker_id)


def round_direction(speaker_id: str, participants: Sequence[str], persona_id: str, speakers: Mapping[str, str]) -> Direction:
    """A two-party round: from the speaker to the other participant. Otherwise the recipient is unknown."""
    others = [p for p in dict.fromkeys(participants) if p != speaker_id]
    recipient = name_of(others[0], persona_id, speakers) if len(set(participants)) == 2 and len(others) == 1 else UNKNOWN
    return Direction(name_of(speaker_id, persona_id, speakers), recipient)
