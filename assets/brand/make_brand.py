#!/usr/bin/env python3
"""StillHere brand assets ("stillhere."): a rounded tile with a lowercase s cut out of it and an indigo full stop,
and the lowercase wordmark "stillhere." whose full stop carries the same accent.

    python assets/brand/make_brand.py --font <Montserrat-Bold.woff2> --tagline-font <IBMPlexSans-Regular.woff2> --out <repo root> [--browser <msedge|chrome>]

Writes to assets/brand/: logo-light.svg, logo-dark.svg, icon.svg, favicon.svg and, with --browser, logo-light.png, logo-dark.png,
icon.png (512 px) and social-preview.png (1280 x 640). Glyphs are converted to outlines (Montserrat Bold for the mark and wordmark,
IBM Plex Sans for the social-preview tagline; both SIL Open Font License 1.1): no font file is shipped or referenced.
Needs fontTools + brotli (for the .woff2 fonts).
"""
from __future__ import annotations

import argparse
import html
import subprocess
import tempfile
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

INK = {"light": "#111418", "dark": "#F2F3F5"}
ACCENT = {"light": "#5457D6", "dark": "#8F92F5"}
TAGLINE = "Persona and evidence-grounded memory for personalized AI conversations"


class Font:
    def __init__(self, path: Path):
        self.f = TTFont(str(path))
        lic = " ".join(str(n) for n in self.f["name"].names if n.nameID in (13, 14))
        if "Open Font License" not in lic and "OFL" not in lic:
            raise SystemExit(f"{path.name}: not SIL OFL licensed; refusing to use it")
        self.gs, self.cmap, self.upm, self.hmtx = self.f.getGlyphSet(), self.f.getBestCmap(), self.f["head"].unitsPerEm, self.f["hmtx"]

    def path(self, text: str, size: float, x0: float, baseline: float, tracking: float = 0.0):
        pen, s, x = SVGPathPen(self.gs), size / self.upm, x0
        for ch in text:
            g = self.cmap[ord(ch)]
            self.gs[g].draw(TransformPen(pen, (s, 0, 0, -s, x, baseline)))
            x += self.hmtx[g][0] * s + tracking * size
        return pen.getCommands(), x - x0


def svg(w, h, body, title, style=""):
    t = html.escape(title, quote=True)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{t}">'
            f'<title>{t}</title>{style}{body}</svg>\n')


def tile(font: Font, ink: str, accent: str, uid: str, ink_cls: str = "", acc_cls: str = "") -> str:
    """The icon in a 100 x 100 box: rounded tile, the s cut out (transparent), the full stop in the accent colour."""
    d, w = font.path("s", 78, 0, 0)
    x = 50 - (w + 16) / 2
    d, _ = font.path("s", 78, x, 72)
    ic = f' class="{ink_cls}"' if ink_cls else ""
    ac = f' class="{acc_cls}"' if acc_cls else ""
    return (f'<defs><mask id="m{uid}" maskUnits="userSpaceOnUse" x="0" y="0" width="100" height="100">'
            f'<rect width="100" height="100" fill="#fff"/><path d="{d}" fill="#000"/></mask></defs>'
            f'<rect x="4" y="4" width="92" height="92" rx="22" fill="{ink}" mask="url(#m{uid})"{ic}/>'
            f'<circle cx="{x + w + 9:.2f}" cy="66" r="7" fill="{accent}"{ac}/>')


def lockup(font: Font, theme: str):
    """Tile (96 px) + wordmark "stillhere." with an accent full stop; 120 px tall."""
    size, gap = 70, 26
    base = 60 + 0.36 * size
    d_word, w_word = font.path("stillhere", size, 96 + gap, base, -0.015)
    d_dot, w_dot = font.path(".", size, 96 + gap + w_word, base)
    W = int(96 + gap + w_word + w_dot + 6)
    body = (f'<g transform="translate(0 12) scale(0.96)">{tile(font, INK[theme], ACCENT[theme], theme)}</g>'
            f'<path d="{d_word}" fill="{INK[theme]}"/><path d="{d_dot}" fill="{ACCENT[theme]}"/>')
    return svg(W, 120, body, "StillHere"), W


def render(browser: str, page_html: str, out: Path, w: int, h: int, scale: float = 1.0, transparent: bool = True):
    """Screenshot an HTML page with a fresh, private browser profile under %TEMP% (no shared profile, nothing left behind)."""
    with tempfile.TemporaryDirectory(prefix="stillhere-brand-") as td:
        p = Path(td) / "page.html"
        p.write_text(page_html, encoding="utf-8")
        args = [browser, "--headless=new", "--disable-gpu", "--no-first-run", "--hide-scrollbars", f"--user-data-dir={Path(td) / 'profile'}",
                f"--window-size={w},{h}", f"--force-device-scale-factor={scale}", "--allow-file-access-from-files", f"--screenshot={out}", p.as_uri()]
        if transparent:
            args.insert(1, "--default-background-color=00000000")
        subprocess.run(args, capture_output=True, timeout=120)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", required=True, help="Montserrat-Bold.woff2 (SIL OFL)")
    ap.add_argument("--tagline-font", required=True, help="IBMPlexSans-Regular.woff2 (SIL OFL), social preview only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--browser", help="msedge / chrome executable, to render the PNG files")
    a = ap.parse_args()
    font, small = Font(Path(a.font)), Font(Path(a.tagline_font))
    out = (Path(a.out) / "assets" / "brand").resolve()
    out.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for theme in ("light", "dark"):
        text, w = lockup(font, theme)
        (out / f"logo-{theme}.svg").write_bytes(text.encode("utf-8"))
        sizes[theme] = w
    (out / "icon.svg").write_bytes(svg(100, 100, tile(font, INK["light"], ACCENT["light"], "i"), "StillHere").encode("utf-8"))
    style = ('<style>.ink{fill:#111418}.acc{fill:#5457D6}@media (prefers-color-scheme: dark){.ink{fill:#F2F3F5}.acc{fill:#8F92F5}}</style>')
    (out / "favicon.svg").write_bytes(svg(100, 100, tile(font, "#111418", "#5457D6", "f", "ink", "acc"), "StillHere", style).encode("utf-8"))
    if a.browser:
        uri = out.as_uri()
        for theme in ("light", "dark"):
            w = sizes[theme]
            render(a.browser, f'<html><body style="margin:0;background:transparent"><img src="{uri}/logo-{theme}.svg" width="{w}" height="120"></body></html>',
                   out / f"logo-{theme}.png", w, 120, 2)
        render(a.browser, f'<html><body style="margin:0;background:transparent"><img src="{uri}/icon.svg" width="512" height="512"></body></html>',
               out / "icon.png", 512, 512)
        # social preview (GitHub recommends 1280 x 640): logo centred, tagline below, on white
        d_tag, w_tag = small.path(TAGLINE, 34, 0, 0)
        d_tag, _ = small.path(TAGLINE, 34, (1280 - w_tag) / 2, 455)
        lw = sizes["light"] * 1.6
        page = (f'<html><body style="margin:0;background:#fff"><div style="position:relative;width:1280px;height:640px">'
                f'<img src="{uri}/logo-light.svg" style="position:absolute;left:{(1280 - lw) / 2:.0f}px;top:170px" width="{lw:.0f}" height="192">'
                f'<svg style="position:absolute;left:0;top:0" width="1280" height="640"><path d="{d_tag}" fill="#59636E"/></svg></div></body></html>')
        render(a.browser, page, out / "social-preview.png", 1280, 640, 1, transparent=False)
    print("brand assets written to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
