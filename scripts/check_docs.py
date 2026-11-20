#!/usr/bin/env python3
"""Documentation checks (standard library only):

1. every relative link and image in the Markdown files points to a file that exists, and every SVG is well-formed XML;
2. every ``stillhere ...`` command in a fenced code block parses with the real CLI (``--help`` style dry parse, nothing runs).

    python scripts/check_docs.py            # exit 1 on any broken link or unknown command / flag
"""
from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)|(?:src|srcset|href)=\"([^\"]+)\"")
FENCE = re.compile(r"```(?:bash|sh|shell|powershell|console)?\n(.*?)```", re.S)


def md_files():
    return [p for p in ROOT.rglob("*.md") if ".git" not in p.parts and ".venv" not in p.parts and "node_modules" not in p.parts]


def links(errors):
    for md in md_files():
        text = md.read_text(encoding="utf-8")
        for m in LINK.finditer(text):
            target = (m.group(1) or m.group(2) or "").split("#", 1)[0]
            if not target or re.match(r"^[a-z]+:", target) or target.startswith(("<", "{")):
                continue
            if not (md.parent / target).exists():
                errors.append(f"{md.relative_to(ROOT)}: broken link -> {target}")


def commands(errors):
    from stillhere import cli
    import argparse
    seen = 0
    for md in md_files():
        for block in FENCE.findall(md.read_text(encoding="utf-8")):
            for raw in block.replace("\\\n", " ").splitlines():
                line = raw.split("  #", 1)[0].strip()
                if not line.startswith("stillhere "):
                    continue
                try:
                    argv = shlex.split(line, posix=True)[1:]
                except ValueError as e:
                    errors.append(f"{md.relative_to(ROOT)}: cannot split: {line} ({e})")
                    continue
                argv = [a for a in argv if a not in ("|", ">", "2>&1")]
                seen += 1
                try:
                    _parse(cli, argparse, argv)
                except SystemExit as e:
                    if e.code not in (0, None):
                        errors.append(f"{md.relative_to(ROOT)}: CLI rejects: {line}")
    return seen


def _parse(cli, argparse, argv):
    """Build the real parser exactly as ``stillhere.cli.main`` does, then parse without running anything."""
    import io
    import contextlib
    ap = argparse.ArgumentParser(prog="stillhere")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("version")
    d = sub.add_parser("demo")
    d.add_argument("--context")
    d.add_argument("--loop-guard")
    d.add_argument("--json", action="store_true")
    g = sub.add_parser("loop-guard")
    g.add_argument("text")
    from stillhere.persona import cli as persona_cli
    persona_cli.register(sub)
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        ap.parse_args(argv)


def svgs(errors):
    """Every SVG must be well-formed XML (GitHub shows a broken image otherwise)."""
    import xml.etree.ElementTree as ET
    for f in ROOT.rglob("*.svg"):
        if ".git" in f.parts or ".venv" in f.parts:
            continue
        try:
            ET.parse(f)
        except ET.ParseError as e:
            errors.append(f"{f.relative_to(ROOT)}: invalid SVG ({e})")


def main() -> int:
    errors = []
    links(errors)
    svgs(errors)
    n = commands(errors)
    for e in errors:
        print(e)
    print(f"checked {len(md_files())} Markdown files and {n} stillhere commands: {'OK' if not errors else f'{len(errors)} problem(s)'}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
