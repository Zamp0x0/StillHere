"""BUILD-V3.3 reader research: per-provider budgets, reader configurations and their answer keys.

A reader sees the question, the time it was asked and the rendered evidence, nothing else. Official answers, score points and evidence ids never reach it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from stillhere.retrieval.alibaba_client import BudgetExceeded, Ledger

from . import answer_pilot as ap

NEW_WORK_SHARE = 0.85
CLOSING_SHARE = 0.95


class ProviderLedger(Ledger):
    """One provider's own append-only ledger and cap. It never reads or resets another ledger."""

    def __init__(self, path: Path, cap_cny: float, provider: str) -> None:
        super().__init__(path, cap_cny)
        self.budget = float(cap_cny)          # the base class clamps to its own ceiling; this round has its own cap
        self.provider = provider

    def require_room(self, estimate: float, *, closing: bool = False) -> None:
        limit = self.budget * (CLOSING_SHARE if closing else NEW_WORK_SHARE)
        if self.spent >= limit:
            raise BudgetExceeded(f"{self.provider}: spent {self.spent:.4f} CNY, no {'closing' if closing else 'new'} work past {limit:.2f}")
        if self.spent + self.pending + estimate > self.budget:
            raise BudgetExceeded(f"{self.provider}: {estimate:.4f} more could pass {self.budget:.2f}")

    def settle(self, reserved: float, entry: Mapping) -> None:
        super().settle(reserved, {"provider": self.provider, **entry})


@dataclass(frozen=True)
class ReaderConfig:
    name: str
    provider: str
    model: str
    system_prompt: str
    max_tokens: int = ap.MAX_NEW_TOKENS
    temperature: float = 0.0
    thinking: bool = False

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    @property
    def prompt_sha256(self) -> str:
        return ap.sha256_text(self.system_prompt)


def messages(config: ReaderConfig, question: str, ask_time: str, context_text: str) -> List[Dict[str, str]]:
    """The baseline user layout with the configuration's own system prompt."""
    return [{"role": "system", "content": config.system_prompt}, ap.build_messages(question, ask_time, context_text)[1]]


def answer_key(config: ReaderConfig, row: Mapping) -> str:
    """Provider, model, prompt and generation settings are all in the key, so no answer of another reader can ever be reused."""
    return ap.sha256_text(json.dumps({"reader": asdict(config), "qa_id": row["qa_id"], "persona": row["persona_id"], "question": ap.sha256_text(row["question"]), "ask_time": row["ask_time"],
                                      "as_of": row["as_of"], "known_at": row["known_at"], "context": row["context_sha256"]}, ensure_ascii=False, sort_keys=True))


BASELINE = ReaderConfig("deepseek_baseline", "deepseek", "deepseek-flash", ap.SYSTEM_PROMPT)

_RULE5 = "5. 不能只因为两件事先后发生，就说它们有因果关系。"
_RULE6 = "6. 证据不足以回答时，明确表示无法确定，不要勉强作答。"
ANSWERABILITY_RULES = (
    "5. 不能只因为两件事先后发生，就说它们有因果关系。但记录里明确写了原因、依据，或写明某个决定是根据先前的经历、结果或反馈做出的，可以据此说明两者的关系，并指出依据的记录。",
    "6. 先判断能否回答：问题要的具体细节（名称、时间、数字、人物、原话）在证据里找不到时，abstain 写 true，answer 说明缺少哪项信息。"
    "问题让你评价一个计划或安排是否合适时，如果证据显示同一时间已有冲突的安排、身体状况或其他限制，就依据这些记录说明是否合适及原因，不要拒答；证据里没有相关情况时才表示无法确定。")


def with_answerability_rules(prompt: str) -> str:
    """The baseline prompt with rules 5 and 6 replaced, nothing else changed. Raises if the baseline text moved."""
    if prompt.count(_RULE5) != 1 or prompt.count(_RULE6) != 1:
        raise ValueError("the baseline prompt no longer has rules 5 and 6 as expected")
    return prompt.replace(_RULE5, ANSWERABILITY_RULES[0]).replace(_RULE6, ANSWERABILITY_RULES[1])


# Batch 2.1 final (one mechanism, designed on the inspected case c33 "我国庆打算去哪里？": the evidence holds "回扬州老家住几天" with no date and
# the frozen reader abstains): an undated plan whose CONTENT matches a question about a period may answer it, saying the record has no date.
# Appended to rule 6 of the frozen c1 prompt; nothing else changes. The frozen reader (c1) is not modified.
UNDATED_PLAN_RULE = ("问题问某个时间段（如国庆、周末、下个月）的计划或安排，而证据里有一条内容相符、但没有写日期的计划时，可以依据它回答，"
                     "并说明记录里没有写具体时间；如果证据写了日期且与所问时间不符，或内容与问题不相符，仍按无法确定处理。")


def with_undated_plan_rule(prompt: str) -> str:
    """The c1 prompt with one sentence appended to rule 6. Raises if c1's rule 6 moved."""
    if prompt.count(ANSWERABILITY_RULES[1]) != 1:
        raise ValueError("the c1 prompt no longer has its rule 6 as expected")
    return prompt.replace(ANSWERABILITY_RULES[1], ANSWERABILITY_RULES[1] + UNDATED_PLAN_RULE)


# Registered in order (docs/READER_DEEPSEEK_RESEARCH_PROTOCOL.md, amendments). A new entry is a new candidate and counts against the limit of four.
CANDIDATES: Dict[str, ReaderConfig] = {
    "deepseek_c1_answerability": ReaderConfig("deepseek_c1_answerability", "deepseek", "deepseek-flash", with_answerability_rules(ap.SYSTEM_PROMPT)),
    "deepseek_c2_undated_plan": ReaderConfig("deepseek_c2_undated_plan", "deepseek", "deepseek-flash",
                                             with_undated_plan_rule(with_answerability_rules(ap.SYSTEM_PROMPT))),
}


def new_unanswerable_hallucination(champion: Mapping, reference: Mapping) -> bool:
    """BUILD-V3.4 rule, for later runs and sensitivity analysis only; it never rewrites a frozen result.

    An answer on an unanswerable question is a new hallucination of the champion only when the reference reader did not also answer it. A single-vote
    grounding flag cannot decide this alone: two answers with the same claim can get different flags.
    """
    return (not champion["abstain"]) and (not champion["correct"]) and bool(reference["abstain"])


def stratified(rows: Sequence[Mapping], seed: str, *, scorable: int, unanswerable: int) -> List[str]:
    """Types in proportion to what is available, each ordered by a seeded hash. Only ids and types are read."""
    from collections import defaultdict
    from .memory_bench import _largest_remainder
    order = lambda r: hashlib.sha256(f"{seed}:{r['qa_id']}".encode("utf-8")).hexdigest()
    strata: Dict[str, List[Mapping]] = defaultdict(list)
    for r in rows:
        if not r["unanswerable"]:
            strata[r["primary_type"]].append(r)
    shares = _largest_remainder({k: len(v) for k, v in strata.items()}, scorable)
    chosen = [r["qa_id"] for k in sorted(strata) for r in sorted(strata[k], key=order)[:shares.get(k, 0)]]
    chosen += [r["qa_id"] for r in sorted((r for r in rows if r["unanswerable"]), key=order)[:unanswerable]]
    return sorted(chosen)
