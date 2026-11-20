"""Manual root-cause labels of the 37 wrong StillHere V2D answers of the LifeBench 于晓薇 100-QA pilot (known set; labelled by reading
each question's gold evidence, the retrieved context, the answer and the three judge rationales; see P2_ROOT_CAUSE_AUDIT_CN.md).

Label = (primary layer, subtype, confidence, proposed minimal fix). Layers follow the P2 brief: ROUTER_MISS, RETRIEVAL_MISS,
PARTIAL_EVIDENCE, TEMPORAL_ORDERING, TEMPORAL_STATE_CONFLICT, MULTIHOP_AGGREGATION, READER_REASONING, FALSE_ABSTENTION,
GROUNDING_REJECTION, ATTRIBUTION_ERROR, GOLD_OR_JUDGE_AMBIGUITY, UNRESOLVED. Question ids only (no question text in the repo)."""
OFFSET = ("MULTIHOP_AGGREGATION", "DATE_OFFSET_SECOND_HOP", "high", "resolve the anchor date + N days and retrieve the target day (P2-A)")
LIST = ("MULTIHOP_AGGREGATION", "ENUMERATED_ITEMS", "high", "retrieve each enumerated item separately and pack one hit per item (P2-A)")
LABELS = {
    "lifebench_v5_yuxiaowei_qa15": ("GOLD_OR_JUDGE_AMBIGUITY", "ANSWER_NOT_IN_EVIDENCE", "high",
                                    "none: both gold records were retrieved; the call duration (5 min, ends 11:40) is in no record, the abstention on it was correct"),
    "lifebench_v5_yuxiaowei_qa23": ("PARTIAL_EVIDENCE", "KEY_RECORD_SAME_DAY_MISSED", "medium", "same-day window for an explicit event date (P2-A date requirement)"),
    "lifebench_v5_yuxiaowei_qa35": ("PARTIAL_EVIDENCE", "KEY_RECORD_SAME_DAY_MISSED", "medium", "same-day requirement once the anchor day is found (P2-A)"),
    "lifebench_v5_yuxiaowei_qa348": ("RETRIEVAL_MISS", "RELATIVE_RECENT_WINDOW", "medium", "'最近' -> window of the last 30 days before the ask time (P2-A)"),
    "lifebench_v5_yuxiaowei_qa364": ("PARTIAL_EVIDENCE", "STATE_EVOLUTION_LATER_RECORD", "medium", "temporal evidence view; later same-day record (P2-B)"),
    "lifebench_v5_yuxiaowei_qa357": ("RETRIEVAL_MISS", "LEXICAL_RANKING", "low", "UNRESOLVED ranking cause; no targeted fix"),
    "lifebench_v5_yuxiaowei_qa101": OFFSET, "lifebench_v5_yuxiaowei_qa118": OFFSET, "lifebench_v5_yuxiaowei_qa120": OFFSET,
    "lifebench_v5_yuxiaowei_qa127": OFFSET, "lifebench_v5_yuxiaowei_qa146": OFFSET, "lifebench_v5_yuxiaowei_qa72": OFFSET,
    "lifebench_v5_yuxiaowei_qa85": OFFSET, "lifebench_v5_yuxiaowei_qa87": OFFSET, "lifebench_v5_yuxiaowei_qa96": OFFSET,
    "lifebench_v5_yuxiaowei_qa103": LIST, "lifebench_v5_yuxiaowei_qa107": LIST, "lifebench_v5_yuxiaowei_qa113": LIST, "lifebench_v5_yuxiaowei_qa117": LIST,
    "lifebench_v5_yuxiaowei_qa86": LIST, "lifebench_v5_yuxiaowei_qa91": LIST, "lifebench_v5_yuxiaowei_qa92": LIST, "lifebench_v5_yuxiaowei_qa95": LIST,
    "lifebench_v5_yuxiaowei_qa138": ("MULTIHOP_AGGREGATION", "RELATIVE_CAUSAL_SECOND_HOP", "medium", "relative cue '几天前' -> window before the anchor day (P2-A)"),
    "lifebench_v5_yuxiaowei_qa177": ("MULTIHOP_AGGREGATION", "SPAN_AFTER_ANCHOR", "high", "'X 后…变化/时间线' -> window after the anchor day (P2-A)"),
    "lifebench_v5_yuxiaowei_qa178": ("MULTIHOP_AGGREGATION", "SECOND_EXPLICIT_DATE", "high", "one requirement per explicit date (P2-A)"),
    "lifebench_v5_yuxiaowei_qa180": ("MULTIHOP_AGGREGATION", "MONTH_AGGREGATION", "medium", "month window requirement (P2-A)"),
    "lifebench_v5_yuxiaowei_qa185": ("MULTIHOP_AGGREGATION", "MONTH_AGGREGATION", "medium", "month window requirement (P2-A)"),
    "lifebench_v5_yuxiaowei_qa210": ("MULTIHOP_AGGREGATION", "MONTH_AGGREGATION", "medium", "month window requirement (P2-A)"),
    "lifebench_v5_yuxiaowei_qa171": ("PARTIAL_EVIDENCE", "PATTERN_EXCEPTION_DAY", "medium", "month requirement + temporal view (P2-A/B)"),
    "lifebench_v5_yuxiaowei_qa251": ("PARTIAL_EVIDENCE", "KEY_RECORD_SAME_DAY_MISSED", "medium", "explicit-date requirement (P2-A)"),
    "lifebench_v5_yuxiaowei_qa415": ("PARTIAL_EVIDENCE", "CAUSE_RECORD_MISSED", "medium", "explicit-date requirement (P2-A)"),
    "lifebench_v5_yuxiaowei_qa265": ("TEMPORAL_STATE_CONFLICT", "CONFLICTING_DATES", "medium", "event-time annotation of records (P2-B)"),
    "lifebench_v5_yuxiaowei_qa428": ("TEMPORAL_STATE_CONFLICT", "PLAN_VS_ACTUAL_DATE", "high", "calendar = plan record; event time shown (P2-B)"),
    "lifebench_v5_yuxiaowei_qa370": ("RETRIEVAL_MISS", "HIDDEN_STATE_ADVISORY", "medium", "explicit target date -> window of the 3 weeks before it (P2-A)"),
    "lifebench_v5_yuxiaowei_qa384": ("RETRIEVAL_MISS", "HIDDEN_STATE_ADVISORY", "medium", "explicit target date -> window of the 3 weeks before it (P2-A)"),
    "lifebench_v5_yuxiaowei_qa394": ("RETRIEVAL_MISS", "HIDDEN_STATE_ADVISORY", "medium", "explicit target date -> window of the 3 weeks before it (P2-A)"),
}
