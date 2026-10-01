"""CLI smoke tests: the wizard and the commands a new user runs first."""

from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

from baton.cli.main import app
from baton.config import load_config
from baton.vault import Vault

runner = CliRunner()
CREDENTIAL = "local-test-credential-123456"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("BATON_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("BATON_CONFIG", raising=False)
    monkeypatch.delenv("BATON_PASSPHRASE", raising=False)
    monkeypatch.delenv("BATON_PROXY_TOKEN", raising=False)
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.chdir(tmp_path)
    # Never touch the developer's real OS keychain from tests.
    monkeypatch.setattr("baton.vault._keyring", lambda: None)
    return tmp_path / "home"


def run_init():
    answers = [
        "2",                        # vault protection: keyfile
        "openai_compatible",        # provider
        "",                         # key name (default)
        "http://127.0.0.1:9/v1",    # base URL
        "test-model",               # model
        "", "100", "", "",          # rpm, rpd, tpm, tpd
        CREDENTIAL,                 # API key (stdin, hidden prompt in a real terminal)
        "n",                        # add another?
        "n",                        # test now?
        "y",                        # create proxy token?
    ]
    return runner.invoke(app, ["init"], input="\n".join(answers) + "\n")


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and result.output.startswith("baton ")


def test_init_wizard_creates_vault_config_and_token(home):
    result = run_init()
    assert result.exit_code == 0, result.output
    assert "Setup complete" in result.output
    token = re.search(r"baton-[A-Za-z0-9_\-]{30,}", result.output)
    assert token, "the proxy token must be shown once"

    config = load_config(home / "config.yaml")
    assert [(k.id, k.provider, k.limits.rpd) for k in config.keys] == [("openai-compatible-1", "openai_compatible", 100)]

    # Secrets are in the vault only: not in the config, not echoed by the wizard.
    assert CREDENTIAL not in (home / "config.yaml").read_text(encoding="utf-8")
    assert CREDENTIAL not in (home / "keys.vault").read_text(encoding="utf-8")
    assert CREDENTIAL not in result.output
    vault = Vault.open(home / "keys.vault")
    assert vault.get_key("openai-compatible-1") == CREDENTIAL
    assert vault.verify_token(token.group(0)) == "default"


def test_commands_after_setup(home):
    assert run_init().exit_code == 0

    listed = runner.invoke(app, ["keys", "list"])
    assert listed.exit_code == 0 and "openai-compatible-1" in listed.output and CREDENTIAL not in listed.output

    status = runner.invoke(app, ["status"])
    assert status.exit_code == 0 and "ready" in status.output and CREDENTIAL not in status.output

    assert "default" in runner.invoke(app, ["token", "list"]).output
    created = runner.invoke(app, ["token", "new", "ci"])
    assert created.exit_code == 0 and "baton-" in created.output
    assert runner.invoke(app, ["token", "new", "ci"]).exit_code == 1          # duplicate name
    assert runner.invoke(app, ["token", "revoke", "ci"]).exit_code == 0
    assert runner.invoke(app, ["token", "revoke", "ci"]).exit_code == 1

    assert runner.invoke(app, ["keys", "disable", "openai-compatible-1"]).exit_code == 0
    assert load_config(home / "config.yaml").keys[0].enabled is False
    assert runner.invoke(app, ["keys", "enable", "openai-compatible-1"]).exit_code == 0
    assert runner.invoke(app, ["keys", "enable", "missing"]).exit_code == 1

    doctor = runner.invoke(app, ["doctor"])
    assert "configuration is valid" in doctor.output
    assert "keyfile mode" in doctor.output                                    # honest about the weaker mode

    removed = runner.invoke(app, ["keys", "remove", "openai-compatible-1", "--yes"])
    assert removed.exit_code == 0 and not load_config(home / "config.yaml").keys
    assert Vault.open(home / "keys.vault").get_key("openai-compatible-1") is None


def test_serve_refuses_to_expose_without_both_switches(home):
    assert run_init().exit_code == 0
    refused = runner.invoke(app, ["serve", "--host", "0.0.0.0"])               # noqa: S104
    assert refused.exit_code == 1 and "refusing to listen" in refused.output
    flag_only = runner.invoke(app, ["serve", "--host", "0.0.0.0", "--expose"])  # noqa: S104
    assert flag_only.exit_code == 1 and "allow_remote" in flag_only.output


def test_missing_setup_gives_actionable_errors(home):
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 1 and "baton init" in result.output
    assert runner.invoke(app, ["token", "new"]).exit_code == 1


def test_invalid_config_is_reported_without_a_traceback(home):
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("keys:\n  - id: a\n    provider: openai\n    api_key: oops\n", encoding="utf-8")
    result = runner.invoke(app, ["keys", "list"])
    assert result.exit_code == 1 and "invalid configuration" in result.output
    assert "Traceback" not in result.output
