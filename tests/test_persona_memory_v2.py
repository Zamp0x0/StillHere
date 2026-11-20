"""Persona Memory V2 (P1-P3) on the synthetic two-person chat: router coverage, explicit-date windows, evidence windows, date modes, the
local dense index (with a deterministic FAKE embedder: no torch, no model), the retrieval benchmark metrics and the runtime switch.
No real person, no model, no network."""
import hashlib
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import persona_memory_fixture as MF  # noqa: E402
from stillhere.persona import memory_store as MS  # noqa: E402
from stillhere.persona import memory_v2 as V2  # noqa: E402
from stillhere.persona import retrieval_bench as RB  # noqa: E402
from stillhere.persona import runtime as RT  # noqa: E402
from stillhere.persona import service as svc  # noqa: E402
from stillhere.persona import store as S  # noqa: E402
from stillhere.retrieval import local_embedding as LE  # noqa: E402
from stillhere.retrieval.ef1 import local_dense as LD  # noqa: E402

TZ = timezone(timedelta(hours=8))
AT = datetime(2025, 2, 1, 12, tzinfo=TZ)


@pytest.fixture()
def world(tmp_path):
    path = MF.chat().write(tmp_path / "c.json")
    root = tmp_path / "r"
    pid = svc.init_profile(path, speaker="contact", root=root)["profile_id"]
    MS.build_evidence(root, pid)
    pdir = S.profile_dir(root, pid)
    api = MS.open_api(pdir, S.read_json(pdir / "profile.json"))
    yield {"root": root, "pid": pid, "pdir": pdir, "api": api}
    api.close()


@pytest.mark.parametrize("q,expected", [
    ("在吗", "chat"), ("哈哈哈哈", "chat"), ("今天天气怎么样？", "chat"), ("你吃饭了吗", "chat"),
    ("你以前在哪个城市工作？", "fact"),                              # V1 already
    ("你刚才说要去哪来着？", "fact"), ("你打算什么时候去香港？", "fact"), ("你为什么不喜欢咖啡？", "fact"),
    ("你最喜欢喝什么？", "fact"), ("你跟谁一起去的来着？", "fact"),
    ("在跟谁打电话？", "chat"), ("你在跟谁玩？", "chat"),                # present-tense "who": live chat, not memory (DEV router audit)
    ("我们2025年1月3日聊了什么？", "date"),
])
def test_router_v2_widens_memory_routes_and_keeps_small_talk(q, expected):
    assert V2.route_of_v2(q, AT) == expected
    if RT.route_of(q, AT) != "chat":
        assert V2.route_of_v2(q, AT) == RT.route_of(q, AT)              # every V1 memory decision is kept


def test_explicit_windows_only_for_written_dates():
    assert V2.explicit_windows("你2025年1月3日去了哪里？", AT)
    assert V2.explicit_windows("你一月三号去了哪里？", AT)
    assert not V2.explicit_windows("你今天去了哪里？", AT)
    assert not V2.explicit_windows("你刚才去了哪里？", AT)


def test_evidence_windows_add_neighbours_without_duplicates_or_future(world):
    api, pid = world["api"], world["pid"]
    v1, _ = V2.pack_v2(api, pid, "本人以前在哪里上班？", AT, V2.V2Config(use_dense=False, neighbors=(0, 0)))
    v2, meta = V2.pack_v2(api, pid, "本人以前在哪里上班？", AT, V2.V2Config(use_dense=False, neighbors=(2, 2)))
    ids1, ids2 = V2.retrieved_ids(v1), V2.retrieved_ids(v2)
    assert set(ids1) <= set(ids2) and len(ids2) > len(ids1) and meta["neighbors_added"] == len(ids2) - len(ids1)
    assert len(ids2) == len(set(ids2))                                   # never the same message twice
    early = datetime(2025, 1, 1, 9, 1, 30, tzinfo=TZ)                    # between the 2nd and 3rd message of day 0
    pk, _ = V2.pack_v2(api, pid, "本人以前在哪里上班？", early, V2.V2Config(use_dense=False, neighbors=(3, 3)))
    ids = V2.retrieved_ids(pk)
    times = [r[0] for r in api.store._conn.execute(f"SELECT timestamp FROM messages WHERE message_id IN ({','.join('?' * len(ids))})", ids)]
    assert ids and all(datetime.fromisoformat(t) <= early for t in times)       # nothing after the ask time


def test_context_lines_are_separate_items_with_their_own_speaker_and_source(world):
    from stillhere.persona import recall as R
    api, pid = world["api"], world["pid"]
    pk, meta = V2.pack_v2(api, pid, "本人以前在哪里上班？", AT, V2.V2Config(use_dense=False, neighbors=(2, 2)))
    conn = api.store._conn
    assert meta["neighbors_added"] > 0 and any(i.kind == "context" for i in pk.items)
    for it in pk.items:
        assert len(it.message_ids) == 1                                   # one line per item: citations are exact
        sp, src = conn.execute("SELECT speaker_id, source_id FROM messages WHERE message_id = ?", (it.message_ids[0],)).fetchone()
        assert it.speakers == ["本人" if sp == pid else "对方"] and it.source_id == src
    assert [i.eid for i in pk.items] == [f"E{n:02d}" for n in range(1, len(pk.items) + 1)]
    assert [i.event_time for i in pk.items] == sorted(i.event_time for i in pk.items)
    other = next(i for i in pk.items if i.speakers == ["对方"])
    subj = R.resolve_subject("你以前在哪工作？")
    assert not R.attribution_ok(subj, [other.speakers])                   # a counterpart line can no longer pass as the persona's words
    groups = V2.ranked_unit_messages(pk)
    assert sum(len(g) for g in groups) == len(V2.retrieved_ids(pk)) and len(groups) == len({i.rank for i in pk.items})


def test_date_modes_prefer_keeps_recall_filter_restricts(world):
    api, pid = world["api"], world["pid"]
    q = "本人2025年1月5日说过什么事？"
    prefer, _ = V2.pack_v2(api, pid, q, AT, V2.V2Config(use_dense=False, date_mode="prefer", neighbors=(0, 0)))
    filt, _ = V2.pack_v2(api, pid, q, AT, V2.V2Config(use_dense=False, date_mode="filter", neighbors=(0, 0)))
    days = lambda pk: [i.event_time[:10] for i in sorted(pk.items, key=lambda i: i.rank)]          # noqa: E731
    assert days(prefer) and days(prefer)[0] in ("2025-01-04", "2025-01-05", "2025-01-06")         # ± 1 day slack first
    assert set(days(filt)) <= {"2025-01-04", "2025-01-05", "2025-01-06"} and len(filt.items) <= len(prefer.items)
    none, _ = V2.pack_v2(api, pid, "本人以前在哪里上班？", AT, V2.V2Config(use_dense=False, date_mode="filter", neighbors=(0, 0)))
    assert none.items                                                    # no explicit date: nothing filtered


class FakeEmbedder:
    """Deterministic bag-of-characters vectors (dimension 16): a test double, NOT semantic retrieval."""
    query_task = "fake"

    def __init__(self):
        self.stats = LE.EmbedStats()

    def _v(self, t):
        v = np.zeros(16, dtype=np.float32)
        for ch in t:
            v[int(hashlib.md5(ch.encode()).hexdigest(), 16) % 16] += 1
        return v / (np.linalg.norm(v) or 1)

    def embed_documents(self, texts):
        return np.stack([self._v(t) for t in texts])

    def embed_queries(self, texts):
        return np.stack([self._v(t) for t in texts])


def test_local_dense_index_builds_resumes_and_feeds_ef1(world, tmp_path):
    api, pid = world["api"], world["pid"]
    names = {pid: "本人", MS.other_id(pid): "对方"}
    docs = LD.window_documents(api.store._conn, pid, names, before=1, after=1)
    assert len(docs) == api.store._conn.execute("SELECT count(*) FROM messages").fetchone()[0]
    assert all("▶" in d.text and d.text[:4] == "2025" for d in docs)
    out = tmp_path / "idx"
    part = LD.build_local(api, pid, out, FakeEmbedder(), names=names, model="fake", dimension=16, batch_size=2, time_budget_s=0.0)
    assert part["status"] == "PARTIAL"                                   # stopped by the budget: nothing written yet
    done = LD.build_local(api, pid, out, FakeEmbedder(), names=names, model="fake", dimension=16, batch_size=4)
    assert done["status"] == "COMPLETE" and done["documents"] == len(docs)
    ch, why = LD.open_local(out, model="fake", dimension=16, query_encoder=LE.LocalQueryEncoder(FakeEmbedder()))
    assert ch is not None and why == "OK"
    bad, why2 = LD.open_local(out, model="other", dimension=16)
    assert bad is None and "DENSE_UNAVAILABLE" in why2                   # another model is refused
    pk, meta = V2.pack_v2(api, pid, "本人以前在哪里上班？", AT, V2.V2Config(use_dense=True, neighbors=(1, 1)), dense=ch)
    assert meta["dense"] and pk.items


def test_support_check_accepts_a_time_the_reader_was_shown_in_the_item_header(world):
    from stillhere.reader import fact as RF
    pk, _ = V2.pack_v2(world["api"], world["pid"], "本人以前在哪里上班？", AT, V2.V2Config(use_dense=False, neighbors=(0, 0)))
    it = pk.items[0]
    hhmm = it.event_time[11:16]                                           # e.g. "09:01": shown in "[E01] 2025-01-01T09:01｜…"
    assert hhmm.replace(":", "") not in it.text.replace(":", "")          # not in the message text itself
    assert RF.support_check(f"是{hhmm}说的", pk)["supported"]
    assert not RF.support_check("是23:59说的", pk)["supported"]           # a time nobody was shown is still unsupported


def test_query_vectors_are_cached_per_instruction(world, tmp_path):
    api, pid = world["api"], world["pid"]
    out = tmp_path / "idx"
    LD.build_local(api, pid, out, FakeEmbedder(), names={pid: "本人", MS.other_id(pid): "对方"}, model="fake", dimension=16, batch_size=4)
    a, b = FakeEmbedder(), FakeEmbedder()
    a.query_task, b.query_task = "task one", "task two"
    for e in (a, b):
        ch, why = LD.open_local(out, model="fake", dimension=16, query_encoder=LE.LocalQueryEncoder(e))
        assert why == "OK"
        ch(pid, "本人以前在哪里上班", AT, 4)
    files = sorted(p.name for p in out.glob("query_cache_*.sqlite"))
    assert files == sorted(f"query_cache_{LE.task_tag(t)}.sqlite" for t in ("task one", "task two"))   # never a shared query vector
    assert LE.task_tag("x") != LE.task_tag("")


def test_retrieval_bench_metrics_and_paired_ci():
    s = RB.score([["a", "b"], ["c"], ["d"]], ["c", "x"], k=3)
    assert s["recall"] == 0.5 and s["hit"] == 1 and s["mrr"] == 0.5 and 0 < s["ndcg"] < 1 and s["shown_msgs"] == 4
    assert RB.score([["a"]], ["z"], k=1)["mrr"] == 0.0
    ci = RB.paired_ci({"q1": 0.0, "q2": 0.5, "q3": 0.0}, {"q1": 1.0, "q2": 0.5, "q3": 0.0})
    assert ci["n"] == 3 and ci["win"] == 1 and ci["tie"] == 2 and ci["diff"] == pytest.approx(1 / 3, abs=1e-4)


def test_runtime_memory_pipeline_switch(world):
    from stillhere.persona import card as K
    pid, pdir = world["pid"], world["pdir"]
    card = {"schema": K.CARD_SCHEMA, "persona_id": pid, "display_name": "测试", "ai_identity": "你是 AI", "reply_length": {}, "style_traits": [],
            "examples": [], "evidence": {"split": "train", "dataset_version": "v", "train_sha256": "x", "profiler_version": "p", "example_ids": [],
                                         "method": "m"},
            "review_status": K.AUTO, "authored_prompt": "你是 AI 数字人。" + "说话很短。" * 60 + "没有依据时说不知道，不要编造。"}
    K.set_current(pdir, K._write_version(pdir, card, parent=None)["card_id"])
    MS.build_semantic(world["root"], pid, complete=MF.reference_complete, reviewer="t")
    v1 = RT.PersonaRuntime(world["root"], pid, backend="fake", memory="production")
    v2 = RT.PersonaRuntime(world["root"], pid, backend="fake", memory="production", memory_pipeline="v2",
                           memory_v2=V2.V2Config(use_dense=False))
    q = "你刚才说要去哪来着？"
    assert v1.route(q, AT) == "chat" and v2.route(q, AT) == "fact"
    pk = v2.fact_pack("本人以前在哪里上班？", AT)
    assert pk.items and v2.last_v2_meta["neighbors_added"] >= 0
    with pytest.raises(ValueError):
        RT.PersonaRuntime(world["root"], pid, backend="fake", memory="production", memory_pipeline="v9")
