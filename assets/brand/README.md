# StillHere brand assets

The StillHere logo: a rounded tile with a lowercase **s** cut out of it and an indigo full stop, next to the lowercase wordmark
**stillhere.** The full stop belongs to the visual wordmark only. The project, repository and package are named StillHere /
`stillhere`.

| File | Use |
|---|---|
| `logo-light.svg`, `logo-light.png` | logo for light backgrounds (transparent) |
| `logo-dark.svg`, `logo-dark.png` | logo for dark backgrounds (transparent) |
| `icon.svg`, `icon.png` | square icon / avatar (512 px PNG, transparent) |
| `favicon.svg` | favicon; switches colour with `prefers-color-scheme` |
| `social-preview.png` | GitHub social preview, 1280 × 640 |

Colours: ink `#111418` (light) / `#F2F3F5` (dark); accent `#5457D6` (light) / `#8F92F5` (dark).

In Markdown, pick the variant with a `<picture>` element whose dark `<source>` points at `logo-dark.svg` and whose `<img>`
points at `logo-light.svg` (the top of the repository [README](../../README.md) does exactly this).

## Provenance

Original concept generated with ChatGPT; final brand assets refined and redrawn for StillHere.

## Fonts

All lettering is converted to vector outlines; no font file is shipped or referenced.

- **Montserrat Bold** (logo, icon and wordmark), SIL Open Font License 1.1.
- **IBM Plex Sans Regular** (the tagline in `social-preview.png` only), SIL Open Font License 1.1.

## Regenerating

```bash
# needs fontTools + brotli; the PNG files need a headless Edge or Chrome
python assets/brand/make_brand.py --font <Montserrat-Bold.woff2> --tagline-font <IBMPlexSans-Regular.woff2> --out . --browser "<msedge or chrome>"
```
