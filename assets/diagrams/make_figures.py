#!/usr/bin/env python3
"""README figures (standard library only): the workflow strip and the LifeBench pilot chart, light and dark variants.

    python assets/diagrams/make_figures.py --out <repo root>

Numbers come from docs/benchmark.md (LifeBench 2.0 pilot, 100 questions); edit them only together with that document.
"""
from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI','Helvetica Neue',Helvetica,Arial,sans-serif"
THEMES = {
    "light": {"ink": "#14161A", "muted": "#5B616B", "card": "#F6F7F9", "line": "#D5D8DD", "bar": "#C9CDD3", "hi": "#14161A", "num": "#8A919C"},
    "dark": {"ink": "#E6EDF3", "muted": "#8B949E", "card": "#161B22", "line": "#30363D", "bar": "#484F58", "hi": "#F4F2EE", "num": "#6E7681"},
}
STEPS = [("01", "Import", "Chat export", "WeFlow JSON · cleaning"),
         ("02", "Persona", "Persona card", "style from real replies"),
         ("03", "Memory", "Evidence memory", "SQLite · BM25 · dense · P2"),
         ("04", "Model", "Qwen3 or API", "local GPU · DeepSeek API"),
         ("05", "Reply", "Grounded reply", "reader + attribution checks")]
PILOT = [("StillHere V2D", 63, True), ("Mem0 OSS 2.2.1", 62, False), ("No memory", 18, False)]


def svg(w, h, title, body):
    t = escape(title, quote=True)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{t}">'
            f'<title>{t}</title><g font-family="{FONT}">{body}</g></svg>\n')


def workflow(c):
    cw, ch, gap, x0, y0 = 212, 128, 30, 8, 8
    out = []
    for i, (n, label, title, sub) in enumerate(STEPS):
        x = x0 + i * (cw + gap)
        out.append(f'<rect x="{x}" y="{y0}" width="{cw}" height="{ch}" rx="14" fill="{c["card"]}" stroke="{c["line"]}"/>')
        out.append(f'<text x="{x + 20}" y="{y0 + 34}" font-size="13.5" font-weight="600" fill="{c["num"]}" letter-spacing="1">{n} · {escape(label.upper())}</text>')
        out.append(f'<text x="{x + 20}" y="{y0 + 70}" font-size="20" font-weight="700" fill="{c["ink"]}">{escape(title)}</text>')
        out.append(f'<text x="{x + 20}" y="{y0 + 98}" font-size="14.5" fill="{c["muted"]}">{escape(sub)}</text>')
        if i < len(STEPS) - 1:
            ax = x + cw + 6
            ay = y0 + ch / 2
            out.append(f'<path d="M{ax} {ay} H{ax + gap - 12}" stroke="{c["muted"]}" stroke-width="1.6" fill="none"/>'
                       f'<path d="M{ax + gap - 17} {ay - 5} L{ax + gap - 11} {ay} L{ax + gap - 17} {ay + 5}" stroke="{c["muted"]}" stroke-width="1.6" fill="none" '
                       f'stroke-linecap="round" stroke-linejoin="round"/>')
    w = x0 * 2 + len(STEPS) * cw + (len(STEPS) - 1) * gap
    return svg(w, ch + 2 * y0, "StillHere workflow: import, persona, memory, model, grounded reply", "".join(out))


def pilot_chart(c):
    w, lab, bar_w, top, row = 760, 178, 470, 48, 46
    out = [f'<text x="0" y="22" font-size="16" font-weight="600" fill="{c["ink"]}">LifeBench 2.0 pilot</text>',
           f'<text x="{w}" y="22" font-size="14.5" fill="{c["muted"]}" text-anchor="end">100 questions · accuracy</text>',
           f'<line x1="0" y1="34" x2="{w}" y2="34" stroke="{c["line"]}"/>']
    for i, (name, v, hi) in enumerate(PILOT):
        y = top + i * row
        fill = c["hi"] if hi else c["bar"]
        out.append(f'<text x="0" y="{y + 20}" font-size="17" font-weight="{700 if hi else 500}" fill="{c["ink"] if hi else c["muted"]}">{escape(name)}</text>')
        out.append(f'<rect x="{lab}" y="{y + 4}" width="{bar_w}" height="22" rx="6" fill="{c["card"]}"/>')
        out.append(f'<rect x="{lab}" y="{y + 4}" width="{bar_w * v / 100:.1f}" height="22" rx="6" fill="{fill}"/>')
        out.append(f'<text x="{lab + bar_w * v / 100 + 12:.1f}" y="{y + 21}" font-size="17" font-weight="{700 if hi else 500}" fill="{c["ink"] if hi else c["muted"]}">{v}%</text>')
    h = top + len(PILOT) * row + 4
    return svg(w, h, "LifeBench 2.0 pilot, 100 questions: StillHere V2D 63%, Mem0 OSS 62%, no memory 18%", "".join(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    root = Path(ap.parse_args().out)
    for theme, c in THEMES.items():
        for rel, text in ((f"assets/diagrams/workflow-{theme}.svg", workflow(c)), (f"assets/benchmarks/lifebench-pilot-{theme}.svg", pilot_chart(c))):
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(text.encode("utf-8"))
    print("figures written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
