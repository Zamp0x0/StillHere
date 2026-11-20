"""Style runtime safety: guards, length policy and the bounded retry / repair / fallback loop, with a fake generator (no model is loaded)."""
from stillhere.style import safety as S

ON = {"GUARD_ENABLED": "1"}
G = {"status": "facts", "facts": [
    {"statement": "我上个月买了去成都的往返机票，去程是10月1日早上8点15分起飞。", "role": "corrected", "time": "上个月", "replaces": "去程是10月2日早上9点起飞"},
    {"statement": "我从成都回来订的是10月6日下午的航班。", "role": "current", "time": "10月6日"}], "unknown": ["机票总价"]}
CONV = [{"speaker": "other", "text": "你订的哪个航班？几点起飞？"}]


class Fake:
    """Returns the scripted texts in order and records every config it was called with."""

    def __init__(self, *texts, hit=False):
        self.texts, self.calls, self.hit = list(texts), [], hit

    def __call__(self, messages, config):
        self.calls.append(dict(config))
        return {"text": self.texts[min(len(self.calls) - 1, len(self.texts) - 1)], "hit_token_limit": self.hit}


# ---------------------------------------------------------------- flags
def test_flags_default_off_and_master_switch():
    assert not any(S.flags({}).values())
    f = S.flags(ON)
    assert all(f[k] for k in ("GUARD_ENABLED",) + S.GUARD_SUBFLAGS)
    assert not f["LENGTH_POLICY_ENABLED"] and not f["POSTHOC_TRIM_ENABLED"]            # independent of the guard (Batch 2.1)
    assert S.flags({"LENGTH_POLICY_ENABLED": "1"})["LENGTH_POLICY_ENABLED"] and not S.flags({"LENGTH_POLICY_ENABLED": "1"})["GUARD_ENABLED"]
    assert S.flags({**ON, "CLAIM_GUARD_ENABLED": "0"})["CLAIM_GUARD_ENABLED"] is False
    assert S.flags({"REPETITION_GUARD_ENABLED": "1"})["REPETITION_GUARD_ENABLED"] is False   # a sub-flag never acts without the master


def test_guard_off_is_a_pass_through():
    fake = Fake("毕业典礼，毕业典礼，毕业典礼，毕业典礼")
    r = S.safe_generate(fake, [{"role": "user", "content": "x"}], {"max_new_tokens": 160}, env={})
    assert r.status == "PASS_THROUGH" and r.text == "毕业典礼，毕业典礼，毕业典礼，毕业典礼"
    assert len(fake.calls) == 1 and fake.calls[0] == {"max_new_tokens": 160}


# ---------------------------------------------------------------- repetition
def test_repetition_loops_are_caught():
    for t in ("毕业典礼，毕业典礼，毕业典礼，毕业典礼", "我还没申请，明天就申请了，我还没申请，明天就申请了，我还没申请，明天就申请了", "的的的的的的的", "a\nb\na\na\na"):
        assert S.repetition(t).degenerate, t


def test_emphasis_is_not_a_loop():
    for t in ("哈哈哈哈哈哈哈哈", "对对对对对对", "好的好的好的", "！！！！！！！", "真的真的真的真的", "我觉得可以，看情况吧", "嗯嗯", ""):
        assert not S.repetition(t).degenerate, t
    assert S.repetition("哈" * 40).degenerate          # an emphasis run that never stops is a loop


def test_truncated_tail_loop_needs_the_token_limit():
    t = "这个我知道的，就是那个东西嘛，然后然后然后然后"
    assert not S.repetition("这个我知道，就是那个，然后呢然后呢").degenerate
    assert S.repetition("这个我知道的，那个嘛，就是说就是说就是说", hit_token_limit=True).degenerate or S.repetition(t).degenerate


def test_trim_keeps_the_first_occurrence():
    t = "还没申请，今天是第6天，明天就申请了，明天就申请了，明天就申请了，明天就申请了"
    v = S.repetition(t)
    fixed = S.trim_repetition(t, v)
    assert fixed.startswith("还没申请，今天是第6天，明天就申请了") and not S.repetition(fixed).degenerate


# ---------------------------------------------------------------- must-not
def test_must_not_is_derived_from_the_evidence():
    terms = {d["term"]: d for d in S.forbidden_terms(G, ["经济舱全价"])}
    assert "经济舱全价" in terms and "2日" in terms and "9点" in terms
    assert "10月1日" not in terms and "8点15分" not in terms          # current values are never forbidden


def test_old_value_said_as_old_is_allowed_said_as_current_is_not():
    assert S.must_not("去程是10月2日9点起飞", G)
    assert not S.must_not("之前记错成10月2日了，其实是10月1日8点15分起飞", G)


def test_number_boundaries():
    assert not S.must_not("我12日到", {"status": "facts", "facts": [{"statement": "是3日", "role": "corrected", "replaces": "是2日"}]})
    assert S.must_not("我2日到", {"status": "facts", "facts": [{"statement": "是3日", "role": "corrected", "replaces": "是2日"}]})


def test_past_facts_are_true_and_never_forbidden():
    g = {"status": "facts", "facts": [{"statement": "我上周搬家花了380元", "role": "past"}]}
    assert S.forbidden_terms(g) == [] and not S.must_not("搬家花了380元", g)


# ---------------------------------------------------------------- claims
def test_claim_gate_and_advisories():
    c = S.claims("是10月1日8点15分，票价1200元", G, CONV, ["10月1日"])
    assert any(x.startswith("unsupported_specifics") for x in c)
    adv = S.claims("10月1日的", G, CONV, ["10月6日"])
    assert all(x.startswith(S.ADVISORY) for x in adv) and adv
    assert S.failed({"claims": adv}) == []                 # advisories alone never fail a reply


# ---------------------------------------------------------------- length policy
def test_length_policy_from_own_history_never_the_target():
    hist = ["好", "嗯嗯", "可以啊", "明天再说吧", "我觉得还行", "你看着办", "哈哈哈", "那就这样", "行", "我看看"]
    p = S.length_policy(hist)
    assert p.source == "author_history" and p.max_new_tokens < 160 and p.soft_low <= 3 <= p.soft_high
    p2 = S.length_policy(hist + ["这是一条很长很长很长的目标回复不应该被算进去"], reference_reply="这是一条很长很长很长的目标回复不应该被算进去")
    assert p2.median_chars == p.median_chars


def test_length_policy_fallback_with_little_history():
    p = S.length_policy(["好", "嗯"])
    assert p.source == "fallback" and p.max_new_tokens == 160


def test_trim_to_cuts_at_a_boundary():
    t = "第一句话说完了。第二句话也说完了。第三句话还在继续说个不停没有结束"
    out = S.trim_to(t, 20)
    assert out.endswith("。") and len(out) <= 20


# ---------------------------------------------------------------- the loop
def test_accept_first_attempt():
    fake = Fake("10月1日8点15分起飞，总价我不太清楚")
    r = S.safe_generate(fake, [], {"max_new_tokens": 160}, grounded=G, conversation=CONV, env=ON)
    assert r.status == "ACCEPTED" and len(fake.calls) == 1


def test_repair_a_loop_without_a_retry():
    fake = Fake("10月1日8点15分起飞，10月1日8点15分起飞，10月1日8点15分起飞，10月1日8点15分起飞")
    r = S.safe_generate(fake, [], {}, grounded=G, conversation=CONV, env=ON)
    assert r.status == "REPAIRED" and r.text == "10月1日8点15分起飞" and len(fake.calls) == 1


def test_retry_changes_only_sampling_and_is_bounded():
    fake = Fake("10月2日9点起飞", "10月1日8点15分起飞，总价不太清楚")
    r = S.safe_generate(fake, [{"role": "user", "content": "q"}], {"seed": 7, "max_new_tokens": 100}, grounded=G, conversation=CONV, env=ON, max_retries=1)
    assert r.status == "RETRIED_ACCEPTED" and len(fake.calls) == 2
    assert fake.calls[1]["seed"] != fake.calls[0]["seed"] and fake.calls[1]["repetition_penalty"] > 1.0


def test_fallback_to_the_grounded_plain_answer():
    fake = Fake("10月2日9点起飞")
    r = S.safe_generate(fake, [], {}, grounded=G, conversation=CONV, env=ON, max_retries=2)
    assert r.status == "FALLBACK" and len(fake.calls) == 3 and "10月1日" in r.text and "2日" not in r.text
    assert [a.action for a in r.attempts] == ["retry", "retry", "fail"]


def test_no_fallback_when_disabled():
    r = S.safe_generate(Fake("10月2日9点起飞"), [], {}, grounded=G, conversation=CONV, env={**ON, "FALLBACK_ENABLED": "0"}, max_retries=0)
    assert r.status == "FAILED_NO_FALLBACK"


def test_pre_generation_policy_caps_without_trimming():
    hist = ["好", "嗯嗯", "可以啊", "明天再说吧", "我觉得还行", "你看着办", "哈哈哈", "那就这样", "行", "我看看"]
    long = "好的。" + "我再补充一点细节，" * 10
    fake = Fake(long)
    r = S.safe_generate(fake, [], {"max_new_tokens": 160}, history_texts=hist, env={"LENGTH_POLICY_ENABLED": "1"})
    assert fake.calls[0]["max_new_tokens"] < 160 and r.status == "PASS_THROUGH" and r.text == long     # a cap, never a post-hoc cut
    assert r.length_policy["constraint"] and r.length_policy["target_chars"] >= 5


def test_posthoc_trim_is_separate_and_off_by_default():
    hist = ["好", "嗯嗯", "可以啊", "明天再说吧", "我觉得还行", "你看着办", "哈哈哈", "那就这样", "行", "我看看"]
    long = "好的。" + "我再补充一点细节，" * 10
    r = S.safe_generate(Fake(long), [], {}, history_texts=hist, env={"LENGTH_POLICY_ENABLED": "1", "POSTHOC_TRIM_ENABLED": "1"})
    assert len(r.text) <= r.length_policy["policy"]["hard_cap"] + 1 and r.text != long


def test_pre_generation_plan_content_floor_and_fallback():
    hist = ["好", "嗯嗯", "可以啊", "明天再说吧", "我觉得还行", "你看着办", "哈哈哈", "那就这样", "行", "我看看"]
    plain = S.pre_generation_plan(hist)
    with_facts = S.pre_generation_plan(hist, grounded=G)
    assert plain.target_chars == 5 and not plain.content_floor_applied
    assert with_facts.target_chars >= S.CONTENT_FLOOR_CHARS and with_facts.max_new_tokens >= S.CONTENT_FLOOR_TOKENS and with_facts.content_floor_applied
    assert "{n}" not in with_facts.constraint and str(with_facts.target_chars) in with_facts.constraint
    fb = S.pre_generation_plan(["好", "嗯"])
    assert fb.source == "fallback" and fb.constraint is None and fb.max_new_tokens == 128
    # the target reply is never used: removing it from the history changes nothing
    assert S.pre_generation_plan(hist + ["这是一条很长很长很长的目标回复"], reference_reply="这是一条很长很长很长的目标回复").target_chars == plain.target_chars


def test_same_interface_for_base_and_adapter():
    for backend in ("base", "lora"):
        r = S.safe_generate(Fake("收到"), [], {}, backend=backend, env=ON)
        assert r.backend == backend and r.status == "ACCEPTED" and r.runtime_schema == S.RUNTIME_SCHEMA_VERSION


def test_audit_trace_is_complete():
    r = S.safe_generate(Fake("10月2日9点起飞", "10月1日8点15分"), [], {}, grounded=G, conversation=CONV, env=ON)
    a = r.audit()
    assert a["attempts"][0]["checks"]["must_not"] and a["attempts"][0]["action"] == "retry" and a["flags"]["GUARD_ENABLED"]
