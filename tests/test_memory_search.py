"""Chinese keyword search: segmentation, FTS5 recall on the synthetic persona, ranking, the scan fallback,
and why FTS5 trigram was not used. Keyword matching only; no embeddings, no model calls."""
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from conftest import make_memory
from memory_support import QIAOMO, QIAOMO_NOW, open_store, qiaomo_store
from stillhere.memory import schema as memory_schema
from stillhere.memory.search import index_terms, match_expression, normalize, overlap_score, query_units

CST = timezone(timedelta(hours=8))
FTS5 = memory_schema.fts5_available(sqlite3.connect(":memory:"))
needs_fts5 = pytest.mark.skipif(not FTS5, reason="SQLite built without FTS5; the scan fallback is tested instead")


def _trigram_available():
    try:
        sqlite3.connect(":memory:").execute("CREATE VIRTUAL TABLE t USING fts5(x, tokenize = 'trigram')")
        return True
    except sqlite3.OperationalError:
        return False


# ---------------------------------------------------------------- segmentation

def test_index_terms_split_cjk_into_characters_and_pairs():
    t = index_terms("乔沫的猫叫年糕。")
    assert t.unigrams == ("乔", "沫", "的", "猫", "叫", "年", "糕")
    assert t.bigrams == ("乔沫", "沫的", "的猫", "猫叫", "叫年", "年糕")


def test_non_cjk_words_stay_whole_and_text_is_normalised():
    t = index_terms("Qwen3 和 ＡＢＣ１２３ home_city")
    assert t.bigrams == ("qwen3", "abc123", "home", "city") and t.unigrams == ("和",)
    assert normalize("ＴＥＡ") == "tea"


def test_query_units():
    assert query_units("年糕") == query_units("  年糕？") and query_units("年糕").bigrams == ("年糕",)
    assert query_units("猫").bigrams == () and query_units("猫").unigrams == ("猫",)      # one character still works
    q = query_units("猫的名字")
    assert q.bigrams == ("猫的", "的名", "名字") and q.unigrams == ("猫", "名", "字")    # function characters dropped
    assert not query_units("？！。,,") and not query_units("")


def test_match_expression_quotes_every_term():
    expr = match_expression(query_units('年糕 OR NEAR(a b) "x" *'))
    assert expr == 'bigrams : ("年糕" OR "or" OR "near" OR "a" OR "b" OR "x") OR unigrams : ("年" OR "糕")'


def test_overlap_score_weights_pairs_above_characters():
    terms = index_terms("乔沫的猫叫年糕。")
    assert overlap_score(query_units("年糕"), terms) > overlap_score(query_units("糕点"), terms) > 0
    assert overlap_score(query_units("北川"), terms) == 0


# ---------------------------------------------------------------- recall on the synthetic persona

RECALL = [
    ("年糕", "mem_cat_name_v2"),
    ("猫叫什么名字", "mem_cat_name_v2"),
    ("猫", "mem_cat_name_v2"),                    # a one-character query (trigram could not match it)
    ("香菜", "mem_pref_coriander"),
    ("她吃香菜吗", "mem_pref_coriander"),
    ("攀岩", "mem_hobby_climbing"),
    ("北川", "mem_home_beichuan"),
    ("现在住在哪个城市", "mem_home_beichuan"),
    ("咖啡店长", "mem_job_cafe"),
    ("在哪里工作 店长", "mem_job_cafe"),
    ("ｈｏｂｂｙ", "mem_hobby_baking"),           # full-width Latin matches the attribute word
]


@pytest.mark.parametrize("fts", [pytest.param(None, marks=needs_fts5), False], ids=["fts5", "scan"])
@pytest.mark.parametrize("query,expected", RECALL)
def test_chinese_queries_recall_the_right_memory(fts, query, expected):
    store = qiaomo_store(fts=fts)
    hits = store.search(QIAOMO, query)
    assert hits and hits[0].record.memory_id == expected, [(h.record.memory_id, h.score) for h in hits]
    assert hits[0].retriever == ("fts5_bigram" if fts is None else "keyword_scan")
    assert all(h.record.provenance for h in hits)


@pytest.mark.parametrize("fts", [pytest.param(None, marks=needs_fts5), False], ids=["fts5", "scan"])
def test_search_respects_status_time_persona_and_limit(fts):
    store = qiaomo_store(fts=fts)
    assert store.search(QIAOMO, "汤圆") == []                                   # superseded value
    assert store.search(QIAOMO, "南屿") == []                                   # no longer valid now
    past = dict(as_of=datetime(2025, 1, 1, tzinfo=CST), known_at=QIAOMO_NOW)
    assert [h.record.value for h in store.search(QIAOMO, "南屿", **past)] == ["南屿市"]
    assert [h.record.memory_id for h in store.search(QIAOMO, "甜点师", **past)][:1] == ["mem_job_bakery"]
    assert {h.record.memory_id for h in store.search(QIAOMO, "生日", limit=2)} == {"mem_birthday_oct", "mem_birthday_nov"}
    assert len(store.search(QIAOMO, "乔沫", limit=3)) == 3
    assert store.search(QIAOMO, "乔沫", limit=0) == [] and store.search(QIAOMO, "。。") == []
    assert store.search("synthetic_nobody", "年糕") == []


@pytest.mark.parametrize("fts", [pytest.param(None, marks=needs_fts5), False], ids=["fts5", "scan"])
def test_ranking_is_deterministic(fts):
    store = qiaomo_store(fts=fts)
    first = store.search(QIAOMO, "爱好")
    assert [h.record.memory_id for h in first] == ["mem_hobby_baking", "mem_hobby_climbing"]   # tie: by id
    assert first[0].score == first[1].score
    assert all(store.search(QIAOMO, "爱好") == first for _ in range(3))
    ranked = store.search(QIAOMO, "北川 咖啡")
    assert [h.record.memory_id for h in ranked][:2] == ["mem_job_cafe", "mem_home_beichuan"]
    assert ranked[0].score > ranked[1].score


@needs_fts5
def test_newer_record_wins_a_tie(tmp_path):
    store = open_store(tmp_path / "tie.sqlite3")
    store.add(make_memory(memory_id="a_old", value="年糕", statement="猫叫年糕。", attribute="cat_name"))
    store.add(make_memory(memory_id="b_new", value="年糕", statement="猫叫年糕。", attribute="pet_name",
                          recorded_at=make_memory().recorded_at + timedelta(days=1)))
    assert [h.record.memory_id for h in store.search("synthetic_p", "年糕")] == ["b_new", "a_old"]
    store.close()


@needs_fts5
def test_keyword_index_is_scoped_to_the_persona_at_query_time():
    store = open_store()
    store.add(make_memory(memory_id="p1", persona_id="synthetic_p1", statement="她的猫叫年糕。", value="年糕"))
    store.add(make_memory(memory_id="p2", persona_id="synthetic_p2", statement="她的猫叫年糕。", value="年糕"))
    assert [h.record.memory_id for h in store.search("synthetic_p1", "年糕")] == ["p1"]
    assert [h.record.memory_id for h in store.search("synthetic_p2", "年糕")] == ["p2"]


# ---------------------------------------------------------------- why not trigram

@pytest.mark.skipif(not _trigram_available(), reason="FTS5 trigram tokenizer needs SQLite 3.34+")
def test_trigram_cannot_match_two_character_chinese_queries():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(x, tokenize = 'trigram')")
    conn.execute("INSERT INTO t VALUES ('乔沫的猫叫年糕。')")
    assert conn.execute("SELECT count(*) FROM t WHERE t MATCH '\"叫年糕\"'").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM t WHERE t MATCH '\"年糕\"'").fetchone()[0] == 0   # too short
    assert qiaomo_store().search(QIAOMO, "年糕")[0].record.memory_id == "mem_cat_name_v2"     # bigrams can


@needs_fts5
def test_unicode61_alone_would_index_a_sentence_as_one_token():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(x, tokenize = 'unicode61')")
    conn.execute("INSERT INTO t VALUES ('乔沫的猫叫年糕。')")
    assert conn.execute("SELECT count(*) FROM t WHERE t MATCH '\"年糕\"'").fetchone()[0] == 0
