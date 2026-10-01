"""`baton update` / `/update`: bring this installation up to date.

Two install shapes are supported:

* a git checkout installed with `pip install -e .` (how the README installs it):
  fast-forward the checkout, then reinstall so new dependencies arrive too;
* a regular pip install: `pip install --upgrade baton-relay`.

Only fast-forward pulls are made, so local work is never merged over or lost.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .. import __version__

_PACKAGE = "baton-relay"
_TIMEOUT = 300


@dataclass
class UpdateResult:
    ok: bool
    message: str
    changed: bool = False


def checkout_root() -> Path | None:
    """The git checkout this code runs from, if it is one."""
    root = Path(__file__).resolve().parents[3]
    if (root / ".git").exists() and (root / "pyproject.toml").is_file():
        return root
    return None


def _run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=_TIMEOUT, check=False)  # noqa: S603


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _pip(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return _run([sys.executable, "-m", "pip", *args, "--disable-pip-version-check", "-q"], cwd=cwd)


def _update_checkout(root: Path) -> UpdateResult:
    try:
        fetched = _run(["git", "fetch", "--quiet"], cwd=root)
    except FileNotFoundError:
        return UpdateResult(False, "git is not installed or not on PATH, so the checkout cannot be updated.")
    if fetched.returncode != 0:
        return UpdateResult(False, f"git fetch failed: {_last_line(fetched.stderr) or 'no details'}")

    behind = _run(["git", "rev-list", "--count", "HEAD..@{u}"], cwd=root)
    if behind.returncode != 0:
        return UpdateResult(False, "this branch has no upstream to update from (set one with `git branch -u`).")
    count = int(behind.stdout.strip() or 0)
    if count == 0:
        return UpdateResult(True, f"Already up to date (baton {__version__}).")

    pulled = _run(["git", "pull", "--ff-only", "--quiet"], cwd=root)
    if pulled.returncode != 0:
        return UpdateResult(False, "could not fast-forward (you have local commits or edits). "
                                   f"Resolve it with git, then retry. {_last_line(pulled.stderr)}")
    installed = _pip("install", "-e", str(root))
    if installed.returncode != 0:
        return UpdateResult(False, f"code updated, but `pip install` failed: {_last_line(installed.stderr)}", True)
    return UpdateResult(True, f"Updated: pulled {count} new commit(s) and reinstalled.", True)


def _update_package() -> UpdateResult:
    before = _run([sys.executable, "-m", "pip", "show", _PACKAGE])
    installed = _pip("install", "--upgrade", _PACKAGE)
    if installed.returncode != 0:
        return UpdateResult(False, f"pip could not upgrade {_PACKAGE}: {_last_line(installed.stderr)}")
    after = _run([sys.executable, "-m", "pip", "show", _PACKAGE])
    changed = before.stdout != after.stdout
    return UpdateResult(True, "Updated." if changed else f"Already up to date (baton {__version__}).", changed)


def run_update() -> UpdateResult:
    try:
        root = checkout_root()
        return _update_checkout(root) if root else _update_package()
    except subprocess.TimeoutExpired:
        return UpdateResult(False, "the update took too long and was stopped; try again.")
    except OSError as exc:
        return UpdateResult(False, f"could not run the update: {type(exc).__name__}")
