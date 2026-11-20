"""Privacy guard and secret scan for everything that may leave the private build (code ZIP, Git commit, public safety ZIP, notebook).

privacy_guard(private_build, targets):
    The deny list is built AT RUN TIME from the private build directory and is never written anywhere:
      * real IRC nicks and V2EX author ids from the split manifest (all roles) and the nick audit;
      * text fingerprints: every 8-character window with >= 6 Han and >= 5 distinct characters from real targets, examples and contexts.
    A nick only counts when it is at least 4 characters long and is not an ordinary identifier (it does not occur as a token in the Python
    standard library sources), so words such as "root" or "pointer" are not false alarms. The report contains counts, file names and
    SHA-256 prefixes of the hits only, never the hit text itself.

secret_scan(targets): API keys / tokens / private keys / e-mail addresses / Windows user-profile paths.

Both accept directories and ZIP files and read text-like members only."""
from __future__ import annotations

import hashlib
import json
import re
import sys
import sysconfig
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Set, Tuple

TEXT_EXT = {".py", ".json", ".jsonl", ".md", ".txt", ".yaml", ".yml", ".ipynb", ".csv", ".html", ".cfg", ".toml", ".ini", ".sh", ".ps1", ""}
HAN = re.compile(r"[㐀-鿿]")
TOK = re.compile(r"[A-Za-z0-9_\^\-\|`\\]{4,}")
WIN = 8
SECRET_PATTERNS = {
    "huggingface_token": r"\bhf_[A-Za-z0-9]{30,}\b",
    "openai_style_key": r"\bsk-[A-Za-z0-9_\-]{20,}\b",
    "anthropic_key": r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b",
    "aws_access_key": r"\bAKIA[0-9A-Z]{16}\b",
    "github_token": r"\b(gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})\b",
    "google_api_key": r"\bAIza[0-9A-Za-z_\-]{35}\b",
    "slack_token": r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b",
    "private_key_block": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "password_assignment": r"(?i)\b(password|passwd|secret|api_key|token)\s*[:=]\s*['\"][^'\"\s]{8,}['\"]",
    "email_address": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    "windows_user_profile_path": r"(?i)\b[A-Z]:\\\\?Users\\\\?[^\\\s\"']+",
}
EMAIL_ALLOW = re.compile(r"(?i)@(example\.(com|org)|users\.noreply\.github\.com|localhost)$")


def _members(target: Path) -> Iterator[Tuple[str, str]]:
    target = Path(target)
    if target.is_file() and target.suffix == ".zip":
        with zipfile.ZipFile(target) as z:
            for n in z.namelist():
                if Path(n).suffix.lower() in TEXT_EXT and not n.endswith("/"):
                    yield f"{target.name}!{n}", z.read(n).decode("utf-8", errors="replace")
    elif target.is_file():
        yield target.name, target.read_text(encoding="utf-8", errors="replace")
    else:
        for p in sorted(target.rglob("*")):
            if p.is_file() and p.suffix.lower() in TEXT_EXT and "__pycache__" not in p.parts and ".git" not in p.parts:
                yield p.relative_to(target).as_posix(), p.read_text(encoding="utf-8", errors="replace")


def _generic_tokens() -> Set[str]:
    out: Set[str] = set()
    std = Path(sysconfig.get_paths()["stdlib"])
    for p in std.glob("*.py"):
        try:
            out.update(t.lower() for t in TOK.findall(p.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            pass
    return out


def _windows(text: str) -> Set[str]:
    # low-entropy windows ("哈哈哈哈…") are not fingerprints: at least 5 distinct characters
    return {w for w in (text[i:i + WIN] for i in range(len(text) - WIN + 1)) if len(HAN.findall(w)) >= 6 and len(set(w)) >= 5}


def denylist(private_build: Path) -> Dict[str, Set[str]]:
    b = Path(private_build)
    sm = json.loads((b / "meta" / "split_manifest.json").read_text(encoding="utf-8"))
    names: Set[str] = set()
    for src in ("irc", "v2ex"):
        for v in (sm.get(src, {}).get("authors") or {}).values():
            names.update(str(x).lower() for x in v)
    nd = b / "meta" / "nick_decisions.json"
    if nd.exists():
        d = json.loads(nd.read_text(encoding="utf-8"))
        def walk(x):
            if isinstance(x, str):
                names.add(x.lower())
            elif isinstance(x, dict):
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        walk({k: v for k, v in d.items() if k != "eligible" or not isinstance(v, int)})
    generic = _generic_tokens()
    nicks = {n for n in names if len(n) >= 4 and n not in generic}
    texts: Set[str] = set()
    for f in list((b / "train").glob("*.jsonl")) + list((b / "cases").glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            parts: List[str] = []
            if "conversations" in r:
                parts += [c["value"] for c in r["conversations"]]
            parts += list(r.get("target_bubbles", [])) + list(r.get("style_examples", []))
            parts += [c.get("text", "") for c in r.get("conversation", [])]
            for t in parts:
                texts |= _windows(t)
    return {"nicks": nicks, "text_windows": texts}


def privacy_guard(private_build: Path, targets: Iterable[Path]) -> dict:
    dl = denylist(private_build)
    nick_re_ok = dl["nicks"]
    hits = []
    for t in targets:
        for name, text in _members(Path(t)):
            low = {x.lower() for x in TOK.findall(text)}
            nh = sorted(low & nick_re_ok)
            th = _windows(text) & dl["text_windows"]
            if nh or th:
                hits.append({"file": name, "nick_hits": len(nh), "text_window_hits": len(th),
                             "hit_sha256_prefixes": sorted(hashlib.sha256(x.encode("utf-8")).hexdigest()[:10] for x in list(nh) + list(th))[:20]})
    return {"version": "t2d3-privacy-guard-v1", "denylist_sizes": {k: len(v) for k, v in dl.items()}, "targets": [str(Path(t).name) for t in targets],
            "files_with_hits": hits, "pass": not hits,
            "rule": "real nicks / author ids (len >= 4, not a Python stdlib token) and 8-char Han windows of real text must not occur"}


def secret_scan(targets: Iterable[Path], allow: Iterable[str] = ()) -> dict:
    pats = {k: re.compile(v) for k, v in SECRET_PATTERNS.items()}
    allow = set(allow)
    hits = []
    for t in targets:
        for name, text in _members(Path(t)):
            for k, rx in pats.items():
                for m in rx.finditer(text):
                    s = m.group(0)
                    if k == "email_address" and EMAIL_ALLOW.search(s):
                        continue
                    if s in allow:
                        continue
                    hits.append({"file": name, "kind": k, "sha256_prefix": hashlib.sha256(s.encode("utf-8")).hexdigest()[:10]})
    return {"version": "t2d3-secret-scan-v1", "targets": [str(Path(t).name) for t in targets], "hits": hits, "pass": not hits}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("privacy", "secrets", "both"))
    ap.add_argument("--private-build")
    ap.add_argument("--out")
    ap.add_argument("targets", nargs="+")
    a = ap.parse_args(argv)
    rep = {}
    if a.what in ("privacy", "both"):
        rep["privacy"] = privacy_guard(Path(a.private_build), [Path(t) for t in a.targets])
    if a.what in ("secrets", "both"):
        rep["secrets"] = secret_scan([Path(t) for t in a.targets])
    rep["pass"] = all(v["pass"] for v in rep.values() if isinstance(v, dict))
    s = json.dumps(rep, ensure_ascii=False, indent=1)
    if a.out:
        Path(a.out).write_text(s, encoding="utf-8")
    print(s)
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
