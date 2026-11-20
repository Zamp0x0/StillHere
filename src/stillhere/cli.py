"""``stillhere`` command line (CPU, public synthetic data, no model download).

    stillhere version
    stillhere demo [--context FULL_CONTEXT|D4_COMPATIBLE] [--loop-guard on|off] [--json]
    stillhere loop-guard "然后又重启，又失败，然后又重启，又失败，然后又重启，又失败，"
    stillhere persona pipeline|init|prepare|review|doctor|package|train|import-adapter|evaluate|chat ...   (see ``stillhere.persona.cli``)
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

import stillhere
from stillhere import api
from stillhere.runtime import loop_guard as lg

DEMO_TURNS = [
    ("你好", None),
    ("我下周去哪出差？", None),
    ("杭州的酒店还订着吗？", None),
    ("那你现在住哪？", "我现在住在杭州西湖区，" * 4 + "离公司挺近的。"),     # a scripted looping reply: shows the Loop Guard
]


def _demo(context: str, guard_on: bool, as_json: bool) -> int:
    rt = api.demo_runtime(context, loop_guard_enabled=guard_on)
    sid = rt.create_session(api.DEMO_PERSONA)
    rows = []
    for i, (text, scripted) in enumerate(DEMO_TURNS, 1):
        gen = api.FakeGenerator([scripted] if scripted else None)
        r = rt.reply_session(api.DEMO_PERSONA, sid, text, f"demo-{i}", generate=gen, backend="fake", env=api.GUARDS_ON)
        calls = (r.trace.get("loop_guard") or {}).get("calls") or []
        rows.append({"turn": i, "user": text, "reply": r.text, "status": r.status, "route": r.trace.get("route"),
                     "loop_guard": calls[-1]["status"] if calls else None, "removed_chars": calls[-1]["removed_char_count"] if calls else 0,
                     "input_tokens": r.trace.get("input_tokens"), "token_mode": r.trace.get("token_mode")})
    if as_json:
        print(json.dumps({"context": context, "loop_guard": guard_on, "turns": rows}, ensure_ascii=False, indent=1))
    else:
        print(f"StillHere {stillhere.__version__} demo | context={context} | loop_guard={'on' if guard_on else 'off'} | generator=FakeGenerator")
        print("(public synthetic persona; no model is loaded; FakeGenerator is NOT Qwen3 inference)")
        for x in rows:
            print(f"\n[{x['turn']}] user : {x['user']}\n    reply: {x['reply']}\n    status={x['status']} route={x['route']} "
                  f"loop_guard={x['loop_guard']} removed={x['removed_chars']} tokens={x['input_tokens']} ({x['token_mode']})")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="stillhere", description="StillHere persona + temporal memory runtime (CPU demo)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("version")
    d = sub.add_parser("demo", help="run a short synthetic multi-turn session with a fake generator")
    d.add_argument("--context", choices=sorted(api.CONTEXTS), default="FULL_CONTEXT")
    d.add_argument("--loop-guard", choices=["on", "off"], default="on")
    d.add_argument("--json", action="store_true")
    g = sub.add_parser("loop-guard", help="apply Loop Guard v1 to a text")
    g.add_argument("text")
    from stillhere.persona import cli as persona_cli
    persona_cli.register(sub)
    a = ap.parse_args(argv)
    if a.cmd == "persona":
        return persona_cli.run(a)
    if a.cmd == "version":
        print(stillhere.__version__)
        return 0
    if a.cmd == "demo":
        return _demo(a.context, a.loop_guard == "on", a.json)
    r = lg.guard(a.text)
    print(json.dumps({k: v for k, v in r.__dict__.items()}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
