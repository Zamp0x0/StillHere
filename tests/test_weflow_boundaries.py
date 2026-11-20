"""WeFlow session / turn boundaries (2026-11-19 audit): exact thresholds, hour crossings, alternation, long monologues, and the persona
and memory pipelines cutting the SAME sessions from the same export. Synthetic exports only.

Audit of the real archive (aggregate counts only, kept in docs/PERSONA_IMPORT_AND_TRAINING_GUIDE_CN.md): 1,159 sessions in both
pipelines, 0 sessions split differently; turn length p99 10 events / 330 s, max 53 events / 2,155 s; no turn longer than 60 minutes."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import persona_fixture as F  # noqa: E402
from stillhere.persona import cleaning as C  # noqa: E402
from stillhere.persona import dialogue as G  # noqa: E402
from stillhere.persona import identity as I  # noqa: E402
from stillhere.persona import memory_store as MS  # noqa: E402
from stillhere.persona import weflow as W  # noqa: E402

GAP, MERGE = 120 * 60, 5 * 60


def _load(chat, tmp_path):
    exp = W.load(chat.write(tmp_path / "c.json"))
    cfg = C.CleaningConfig()
    evs, _ = C.events(exp.messages, F.CONTACT, cfg)
    return exp, cfg, evs


def _persona_sessions(evs, cfg):
    return [[i for e in s for i in e.message_ids] for s in G.sessions(evs, cfg.session_gap_minutes)]


def _memory_sessions(exp):
    msgs = [m for m in exp.messages if not I.is_system(m)]
    conv = MS._sessions(msgs)
    out = {}
    for m in msgs:
        out.setdefault(conv[m.platform_id], []).append(m.platform_id)
    return list(out.values())


def test_session_gap_is_inclusive_at_exactly_120_minutes(tmp_path):
    c = F.Chat()
    t = F.T0
    c.text(t, F.SELF, "在吗")
    c.text(t + 60, F.CONTACT, "在")
    c.text(t + 60 + GAP, F.SELF, "刚好两小时")                  # exactly the gap: same session
    c.text(t + 60 + GAP + GAP + 1, F.CONTACT, "两小时零一秒")   # one second over: new session
    exp, cfg, evs = _load(c, tmp_path)
    ps = _persona_sessions(evs, cfg)
    assert [len(s) for s in ps] == [3, 1]
    assert sorted(map(sorted, ps)) == sorted(map(sorted, _memory_sessions(exp)))


def test_turn_merge_is_inclusive_at_exactly_5_minutes_and_crosses_the_hour(tmp_path):
    c = F.Chat()
    t = F.T0 + 3600 - 120                                        # 09:58
    c.text(t, F.SELF, "你好")
    c.text(t + 30, F.CONTACT, "九点五十八")
    c.text(t + 30 + MERGE, F.CONTACT, "十点过三分（正好五分钟）")   # crosses 10:00, exactly 5 minutes: same turn
    c.text(t + 30 + 2 * MERGE + 1, F.CONTACT, "多一秒")          # 5 min + 1 s: a new turn of the same person
    exp, cfg, evs = _load(c, tmp_path)
    s = G.sessions(evs, cfg.session_gap_minutes)
    assert len(s) == 1
    ts = G.turns(s[0], cfg.turn_merge_minutes)
    assert [(x.role, len(x.events)) for x in ts] == [("other", 1), ("target", 2), ("target", 1)]


def test_alternating_speakers_always_cut_turns_and_a_monologue_chains(tmp_path):
    c = F.Chat()
    t = F.T0
    for i in range(6):
        c.text(t + i * 10, F.SELF if i % 2 == 0 else F.CONTACT, f"第{i}句")
    for j in range(30):                                          # 30 messages 4 minutes apart: one chained turn (documented behaviour)
        c.text(t + 100 + j * 240, F.CONTACT, f"独白{j}")
    exp, cfg, evs = _load(c, tmp_path)
    ts = G.turns(G.sessions(evs, cfg.session_gap_minutes)[0], cfg.turn_merge_minutes)
    assert [x.role for x in ts[:5]] == ["other", "target", "other", "target", "other"]
    assert len(ts) == 6 and len(ts[-1].events) == 31 and ts[-1].end - ts[-1].start == 29 * 240 + 50


def test_both_pipelines_cut_the_same_sessions_with_system_messages_and_collapsed_media(tmp_path):
    c = F.Chat()
    t = F.T0
    c.text(t, F.SELF, "早")
    c.media(t + 5, F.CONTACT)
    c.media(t + 6, F.CONTACT)                                    # a repeat collapsed by the persona cleaning
    c.system(t + 3000, F.SELF)                                   # a system notice inside a long silence never bridges it
    c.text(t + 6 + GAP + 60, F.CONTACT, "下午好")
    c.text(t + 6 + GAP + 120, F.SELF, "嗯")
    exp, cfg, evs = _load(c, tmp_path)
    ps = _persona_sessions(evs, cfg)
    ms = _memory_sessions(exp)
    assert len(ps) == len(ms) == 2
    assert [set(s) <= set(m) for s, m in zip(ps, ms)] == [True, True]           # persona drops the collapsed repeat, never moves a message


def test_samples_never_read_a_later_message(tmp_path):
    exp = W.load(F.standard(days=10).write(tmp_path / "c.json"))
    cfg = C.CleaningConfig()
    evs, _ = C.events(exp.messages, F.CONTACT, cfg)
    t_of = {m.platform_id: m.create_time for m in exp.messages}
    samples, _, _ = G.build(evs, cfg, count=lambda s: len(s), quoted_text=lambda pid: None)
    assert samples
    for s in samples:
        assert max(t_of[i] for i in s.context_message_ids) <= min(t_of[i] for i in s.target_message_ids)
