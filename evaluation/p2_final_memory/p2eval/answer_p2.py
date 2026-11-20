"""NEW answer prompt (P2-C): the OLD prompt (lbpilot-answer-v1) plus three evidence-use rules. Same model, temperature and output parsing.

The added rules never relax grounding: they tell the model to answer the parts the records support, to name only the parts that truly
lack records, and to read planned (calendar) times as plans. They do not ask for guesses, and the 'say it cannot be determined when the
records have no basis' rule of the OLD prompt is kept verbatim."""
from __future__ import annotations

from lbpilot import answer as A

PROMPT_VERSION = "lbpilot-answer-p2"
EXTRA = ("补充规则：1）问题有多个部分（例如几件事排序、两个日期、某天之后的变化）时，对记录中有依据的部分直接回答，只对记录里确实找不到的部分说明无法确定，"
         "不要因为个别部分缺失就整题拒答；2）如果记录前有【证据覆盖】说明，先对照它确认每个部分是否找到了记录；3）日程/计划记录只说明当时的安排，"
         "回答“实际做了什么/哪天发生”时以实际发生的记录为准，几条记录冲突时说明依据的是哪一条。")
SYSTEM = A.SYSTEM + EXTRA


def messages(name: str, question: str, ask_time: str, context: str):
    return [{"role": "system", "content": SYSTEM.format(name=name)},
            {"role": "user", "content": A.USER.format(ask_time=ask_time[:10], context=context, question=question)}]
