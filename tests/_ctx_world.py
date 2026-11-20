"""Shared fixtures for the Simple Context v1 tests (public synthetic data, CPU only, fake generators: no model is ever loaded)."""
from datetime import timedelta

from stillhere.evaluation import context_dataset_v1 as cd
from stillhere.evaluation import writer_eval as we
from stillhere.session import SessionRuntime
from stillhere.style import contract as sc

ON = {"GUARD_ENABLED": "1", "GROUNDING_V2_ENABLED": "1"}
V2 = {"GROUNDING_V2_ENABLED": "1"}
HISTORY = [sc.StyleMessage(f"{cd.CA}.h{i}", cd.CA, t, topic=f"t{i % 3}") for i, t in enumerate(
    ["嗯 知道了", "今天有点忙\n晚点说", "好的好的", "哈哈 可以啊", "行 我看看", "明天再说吧", "收到", "没问题\n周末见"])]


def world(path=":memory:"):
    if path == ":memory:":
        return we.run(we.new_world(we.reference_complete(cd.conversations()), "reference", ds=cd, m22=True, m221=True, m222=True), ds=cd)
    raise ValueError


class Clock:
    def __init__(self, start=None):
        self.now = start or cd.t(9, 10, 20)

    def __call__(self):
        self.now = self.now + timedelta(minutes=1)
        return self.now


class Fake:
    """A deterministic generator: returns the given texts in turn and records every prompt it received."""

    def __init__(self, *texts):
        self.texts, self.calls = list(texts) or ["好的"], []

    def __call__(self, messages, config):
        self.calls.append(([dict(m) for m in messages], dict(config)))
        return {"text": self.texts[min(len(self.calls) - 1, len(self.texts) - 1)]}


class Boom:
    def __call__(self, messages, config):
        raise RuntimeError("model unavailable")


def runtime(api=None, **kw):
    api = api or world().api
    return SessionRuntime(api, clock=kw.pop("clock", Clock()), **kw), api
