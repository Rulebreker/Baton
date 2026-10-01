#!/usr/bin/env python3
"""Secret scanner used by the pre-commit hook and by CI.

Standard library only, on purpose: the hook must work on a fresh clone before
any dependency is installed, otherwise the very first commit is unprotected.

Usage:
    python scripts/secret_scan.py --staged        # what the pre-commit hook runs
    python scripts/secret_scan.py --all           # every tracked file (CI)
    python scripts/secret_scan.py path [path...]  # specific files

Exit code 0 = clean, 1 = findings, 2 = scanner could not run (treated as a
failure by the hook: we fail closed).

To silence a false positive, put the marker `baton:allow-secret` on the same
line. Use it sparingly; it shows up in code review.
"""

from __future__ import annotations

import argparse
import fnmatch
import math
import re
import subprocess
import sys
from collections.abc import Iterable
from typing import NamedTuple

ALLOW_MARKER = "baton:allow-secret"
MAX_SCAN_BYTES = 2 * 1024 * 1024  # larger blobs are flagged rather than scanned

# Files that must never be committed regardless of content.
FORBIDDEN_NAMES = [
    ".env",
    ".env.*",
    "*.vault",
    "*.vault.bak",
    "master.key",
    "*.key",
    "*.pem",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.keystore",
    "id_rsa*",
    "id_ed25519*",
    "state.json",
    "baton.yaml",
    "baton.yml",
    "*.local.yaml",
    "*.log",
    "secrets.*",
    "credentials.*",
]
FORBIDDEN_DIRS = {".baton", "sessions", "checkpoints", "logs"}
ALLOWED_NAMES = {".env.example"}

# (name, regex). Kept deliberately specific to limit false positives.
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Anthropic API key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("OpenAI API key", re.compile(r"sk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{32,}")),
    ("Google API key", re.compile(r"AIza[0-9A-Za-z_\-]{35}")),
    ("Groq API key", re.compile(r"gsk_[A-Za-z0-9]{40,}")),
    ("OpenRouter API key", re.compile(r"sk-or-(?:v1-)?[A-Za-z0-9]{40,}")),
    ("Hugging Face token", re.compile(r"hf_[A-Za-z0-9]{30,}")),
    ("xAI API key", re.compile(r"xai-[A-Za-z0-9]{40,}")),
    ("Baton proxy token", re.compile(r"baton-[A-Za-z0-9_\-]{32,}")),
    ("GitHub token", re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{50,}")),
    ("AWS access key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("Slack token", re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("Private key block", re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----")),
    ("JSON Web Token", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
]

# name = "value" / name: value where the name smells like a credential.
ASSIGNMENT = re.compile(
    r"""(?ix)
    \b[\w\-.]*(?:api[_\-]?key|secret|token|passw(?:or)?d|passphrase|auth)[\w\-.]*
    \s*[:=]\s*
    ["']?(?P<value>[A-Za-z0-9+/_\-=.]{20,})["']?
    """
)

# Personal paths leak usernames; the repo must stay public-ready.
PERSONAL_PATH = re.compile(
    r"""(?ix)
    (?:[a-z]:[\\/]+users[\\/]+|/home/|/Users/)
    (?P<user>[^\\/\s"'<>|:*?$%{}()\[\]]+)
    """
)
GENERIC_USERS = {
    "you", "user", "username", "name", "me", "yourname", "your-name", "your_name",
    "alice", "bob", "runner", "runneradmin", "public", "default", "shared", "example",
}


class Finding(NamedTuple):
    path: str
    line: int
    kind: str
    excerpt: str


def shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts: dict[str, int] = {}
    for ch in value:
        counts[ch] = counts.get(ch, 0) + 1
    total = len(value)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def mask(value: str) -> str:
    """Never echo a suspected secret in full, even in the scanner's own output."""
    value = value.strip()
    if len(value) <= 8:
        return "*" * len(value)
    # ASCII only: hook output may land on a legacy Windows console code page.
    return f"{value[:4]}...{value[-2:]} ({len(value)} chars)"


def forbidden_path(path: str) -> str | None:
    parts = path.replace("\\", "/").split("/")
    name = parts[-1]
    if name in ALLOWED_NAMES:
        return None
    for directory in parts[:-1]:
        if directory in FORBIDDEN_DIRS:
            return f"file inside forbidden directory '{directory}/'"
    for pattern in FORBIDDEN_NAMES:
        if fnmatch.fnmatch(name, pattern):
            return f"forbidden file name (matches '{pattern}')"
    return None


def looks_like_placeholder(value: str) -> bool:
    lowered = value.lower()
    if any(word in lowered for word in ("example", "placeholder", "your", "changeme", "xxxx", "redacted", "dummy")):
        return True
    if len(set(value)) <= 4:  # e.g. "aaaaaaaaaaaaaaaaaaaa"
        return True
    # Lower-case words joined by "." or "_" are code (`self.config.max_tokens`),
    # not credentials: real keys carry digits or mixed case.
    return bool(re.fullmatch(r"[a-z_.]+", value))


def scan_text(path: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if ALLOW_MARKER in line:
            continue
        matched = False
        for kind, pattern in PATTERNS:
            hit = pattern.search(line)
            if hit and not looks_like_placeholder(hit.group(0)):
                findings.append(Finding(path, number, kind, mask(hit.group(0))))
                matched = True
                break
        if not matched:
            hit = ASSIGNMENT.search(line)
            if hit:
                value = hit.group("value")
                # Entropy gate: real keys are random, identifiers and prose are not.
                if not looks_like_placeholder(value) and shannon_entropy(value) >= 4.0:
                    findings.append(Finding(path, number, "High-entropy credential assignment", mask(value)))
        hit = PERSONAL_PATH.search(line)
        if hit and hit.group("user").lower() not in GENERIC_USERS:
            findings.append(Finding(path, number, "Personal filesystem path", hit.group(0)[:12] + "..."))
    return findings


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True, timeout=60).stdout


def staged_files() -> list[str]:
    out = git("diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR")
    return [p for p in out.decode("utf-8", "replace").split("\0") if p]


def tracked_files() -> list[str]:
    out = git("ls-files", "-z")
    return [p for p in out.decode("utf-8", "replace").split("\0") if p]


def read_blob(path: str, staged: bool) -> bytes:
    if staged:
        # Scan the *index* version: that is what will actually be committed,
        # which can differ from the working tree after a partial `git add -p`.
        return git("show", f":{path}")
    with open(path, "rb") as handle:
        return handle.read()


def scan_paths(paths: Iterable[str], staged: bool) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        reason = forbidden_path(path)
        if reason:
            findings.append(Finding(path, 0, "Forbidden file", reason))
            continue
        try:
            blob = read_blob(path, staged)
        except (OSError, subprocess.SubprocessError) as exc:
            findings.append(Finding(path, 0, "Unreadable file", type(exc).__name__))
            continue
        if len(blob) > MAX_SCAN_BYTES:
            findings.append(Finding(path, 0, "File too large to scan", f"{len(blob)} bytes"))
            continue
        if b"\0" in blob[:8192]:
            continue  # binary; name-based rules above already applied
        findings.extend(scan_text(path, blob.decode("utf-8", "replace")))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan for secrets before they reach git history.")
    parser.add_argument("--staged", action="store_true", help="scan staged changes (pre-commit)")
    parser.add_argument("--all", action="store_true", help="scan every tracked file (CI)")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args(argv)

    try:
        if args.staged:
            findings = scan_paths(staged_files(), staged=True)
        elif args.all:
            findings = scan_paths(tracked_files(), staged=False)
        else:
            findings = scan_paths(args.paths, staged=False)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"secret-scan: could not run ({type(exc).__name__}); refusing to continue.", file=sys.stderr)
        return 2

    if not findings:
        return 0

    print("secret-scan: BLOCKED - possible secrets or private data found:\n", file=sys.stderr)
    for item in findings:
        location = f"{item.path}:{item.line}" if item.line else item.path
        print(f"  {location}  [{item.kind}]  {item.excerpt}", file=sys.stderr)
    print(
        "\nRemove the data (and rotate the key if it was real). For a genuine false\n"
        f"positive, add the marker '{ALLOW_MARKER}' to that line.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
