"""`baton update`: the git/pip decisions, with every subprocess faked."""

from __future__ import annotations

import subprocess
from pathlib import Path

from baton.cli import update


def completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def fake_git(monkeypatch, replies: dict[str, subprocess.CompletedProcess[str]]) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(args, cwd=None):
        calls.append(args)
        for needle, reply in replies.items():
            if needle in " ".join(args):
                return reply
        return completed()

    monkeypatch.setattr(update, "_run", run)
    return calls


def test_checkout_already_up_to_date_changes_nothing(monkeypatch):
    calls = fake_git(monkeypatch, {"rev-list": completed(stdout="0\n")})
    result = update._update_checkout(Path("."))
    assert result.ok and not result.changed and "up to date" in result.message
    assert not any("pull" in call or "install" in call for call in calls)


def test_checkout_pulls_fast_forward_only_then_reinstalls(monkeypatch):
    calls = fake_git(monkeypatch, {"rev-list": completed(stdout="3\n")})
    result = update._update_checkout(Path("."))
    assert result.ok and result.changed and "3 new commit" in result.message
    pull = next(call for call in calls if "pull" in call)
    assert "--ff-only" in pull
    assert any(call[1:4] == ["-m", "pip", "install"] for call in calls)


def test_checkout_with_diverged_history_is_reported_not_forced(monkeypatch):
    fake_git(monkeypatch, {"rev-list": completed(stdout="2\n"), "pull": completed(1, stderr="fatal: Not possible to fast-forward")})
    result = update._update_checkout(Path("."))
    assert not result.ok and not result.changed and "fast-forward" in result.message


def test_missing_git_is_a_clear_message(monkeypatch):
    def run(args, cwd=None):
        raise FileNotFoundError("git")

    monkeypatch.setattr(update, "_run", run)
    result = update._update_checkout(Path("."))
    assert not result.ok and "git is not installed" in result.message


def test_branch_without_upstream_is_explained(monkeypatch):
    fake_git(monkeypatch, {"rev-list": completed(128, stderr="fatal: no upstream configured")})
    result = update._update_checkout(Path("."))
    assert not result.ok and "upstream" in result.message


def test_plain_pip_install_upgrades_the_package(monkeypatch):
    shown = iter([completed(stdout="Version: 0.1.0"), completed(stdout="Version: 0.2.0")])

    def run(args, cwd=None):
        return next(shown) if "show" in args else completed()

    monkeypatch.setattr(update, "_run", run)
    monkeypatch.setattr(update, "checkout_root", lambda: None)
    result = update.run_update()
    assert result.ok and result.changed


def test_timeout_is_reported(monkeypatch):
    def run(args, cwd=None):
        raise subprocess.TimeoutExpired(args, 1)

    monkeypatch.setattr(update, "_run", run)
    monkeypatch.setattr(update, "checkout_root", lambda: Path("."))
    assert "too long" in update.run_update().message
