"""Shared factories for contract tests. All data is synthetic."""
from datetime import datetime, timedelta, timezone

import pytest

from stillhere.contracts import (AdapterRef, ChatMessage, DataOrigin, MemoryKind, MemoryRecord, ModelRef,
                                 PromptTemplateRef, Provenance, RetrievedMemory, Role)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def at(days: int) -> datetime:
    return T0 + timedelta(days=days)


def make_memory(**overrides) -> MemoryRecord:
    fields = dict(memory_id="mem_1", persona_id="synthetic_p", kind=MemoryKind.FACT, subject="persona",
                  attribute="home_city", value="Alpha Town", statement="The persona lives in Alpha Town.",
                  provenance=(Provenance(source_id="src_1", locator="turn:1"),), origin=DataOrigin.SYNTHETIC,
                  recorded_at=at(0))
    fields.update(overrides)
    return MemoryRecord(**fields)


def make_template() -> PromptTemplateRef:
    return PromptTemplateRef(name="test_template", version="0.1")


def make_adapter(**overrides) -> AdapterRef:
    fields = dict(adapter_id="synthetic_adapter", base_model=ModelRef(model_id="Qwen/Qwen3-8B"),
                  prompt_template=make_template(), origin=DataOrigin.SYNTHETIC)
    fields.update(overrides)
    return AdapterRef(**fields)


def user(*bubbles: str) -> ChatMessage:
    return ChatMessage(role=Role.USER, bubbles=bubbles)


def assistant(*bubbles: str) -> ChatMessage:
    return ChatMessage(role=Role.ASSISTANT, bubbles=bubbles)


def retrieved(record: MemoryRecord, score: float = 0.5) -> RetrievedMemory:
    return RetrievedMemory(record=record, score=score, retriever="test")


@pytest.fixture
def memory() -> MemoryRecord:
    return make_memory()



# ---------------------------------------------------------------- Final repository: legacy research-harness tests
# These exercise research drivers (scripts/evaluate_*.py) tied to the owner's private report folders and left out of this public repository.
# They pass in the source research worktree (StillHere-Context @ 8953fa0); see docs/legacy/LEGACY_TESTS.md. Deselected here, never counted
# as passed.
LEGACY_RESEARCH_TESTS = {
    "tests/test_answer_pilot.py::test_the_script_sends_nothing_without_explicit_permission",
    "tests/test_context_m1.py::test_reader_scripts_keep_the_legacy_path_and_guard_frozen_outputs",
    "tests/test_time_anchor.py::test_the_answer_test_is_skipped_without_room_on_the_gpu",
    "tests/test_time_anchor.py::test_answer_questions_are_chosen_without_results",
    # the fresh-history release repository has no research snapshot tags (they live in the private research repositories)
    "tests/test_context_m1.py::test_frozen_snapshots_did_not_move",
    "tests/test_writer_observation_m2.py::test_frozen_snapshots_did_not_move",
}


def pytest_collection_modifyitems(config, items):
    keep, legacy = [], []
    for it in items:
        (legacy if it.nodeid.replace("\\", "/") in LEGACY_RESEARCH_TESTS else keep).append(it)
    if legacy:
        config.hook.pytest_deselected(items=legacy)
        items[:] = keep
