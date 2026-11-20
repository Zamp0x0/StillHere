#!/usr/bin/env python3
"""Fail if files that look private are about to be committed (or exist in the tree with --all).

Checks: blocked extensions (weights, datasets, databases), secret-like file names, secret-like content
patterns, private-data path fragments, and files larger than the size limit. Standard library only.
Usage:  python scripts/privacy_guard.py            # staged files (pre-commit)
        python scripts/privacy_guard.py --all      # every tracked + untracked, non-ignored file
        python scripts/privacy_guard.py PATH...    # explicit files
"""
import os, re, subprocess, sys

BLOCKED_EXT = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".ggml", ".onnx", ".h5",
               ".db", ".sqlite", ".sqlite3", ".parquet", ".arrow", ".pem", ".key", ".pfx", ".p12"}
JSONL_ALLOWED_PREFIX = "examples/synthetic/"
SECRET_NAMES = re.compile(r"(^|/)(\.env(\..+)?|id_rsa.*|.*credentials.*\.json)$", re.I)
PRIVATE_PATH = re.compile(r"(^|/)(runtime_state|raw_data|chat_exports?|persona_private|adapters?|checkpoints?|private)(/|$)", re.I)
SECRET_CONTENT = [re.compile(p) for p in (
    r"sk-[A-Za-z0-9]{20,}", r"AKIA[0-9A-Z]{16}", r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----",
    r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token|service[_-]?role)\s*[:=]\s*['\"]?[A-Za-z0-9_\-\.]{16,}",
    r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")]
MAX_BYTES = 2 * 1024 * 1024

def files_from_git(all_files):
    if all_files:
        out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], capture_output=True).stdout
    else:
        out = subprocess.run(["git", "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"], capture_output=True).stdout
    return [p for p in out.decode("utf-8", "replace").split("\0") if p]

def check(path):
    issues = []
    p = path.replace("\\", "/")
    if os.path.basename(p) == ".env.example":
        pass
    elif SECRET_NAMES.search(p):
        issues.append("secret-like file name")
    ext = os.path.splitext(p)[1].lower()
    if ext in BLOCKED_EXT:
        issues.append(f"blocked extension {ext}")
    if ext == ".jsonl" and not p.startswith(JSONL_ALLOWED_PREFIX):
        issues.append(".jsonl outside examples/synthetic/")
    if PRIVATE_PATH.search(p):
        issues.append("private-data path")
    if os.path.isfile(path):
        if os.path.getsize(path) > MAX_BYTES:
            issues.append(f"file larger than {MAX_BYTES} bytes")
        try:
            text = open(path, encoding="utf-8", errors="ignore").read(MAX_BYTES)
            for rx in SECRET_CONTENT:
                if rx.search(text):
                    issues.append(f"secret-like content ({rx.pattern[:24]}...)")
                    break
        except OSError:
            pass
    return issues

def main(argv):
    if argv and argv[0] == "--all":
        paths = files_from_git(True)
    elif argv:
        paths = argv
    else:
        paths = files_from_git(False)
    bad = {p: i for p in paths if (i := check(p))}
    if p_self := [p for p in bad if p.endswith("privacy_guard.py")]:
        for p in p_self:
            bad.pop(p)  # the guard's own pattern definitions
    if bad:
        print("privacy_guard: BLOCKED")
        for p, i in sorted(bad.items()):
            print(f"  {p}: {', '.join(i)}")
        return 1
    print(f"privacy_guard: OK ({len(paths)} files checked)")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
