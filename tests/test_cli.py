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
        "openai_compatible",        # provider (the option's number or its value)
        "",                         # key name (default)
        "http://127.0.0.1:9/v1",    # base URL
        "test-model",               # model
        "", "100", "", "",          # rpm, rpd, tpm, tpd
        CREDENTIAL,                 # API key (stdin, hidden prompt in a real terminal)
        "n",                        # add another?
        "n",                        # test now?
    ]
    result = runner.invoke(app, ["init", "--advanced"], input="\n".join(answers) + "\n")
    assert result.exit_code == 0, result.output
    return result


def run_setup():
    """The wizard, then the proxy token (`init` no longer creates one)."""
    result = run_init()
    created = runner.invoke(app, ["token", "new", "default"])
    assert created.exit_code == 0, created.output
    return result, created.output


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and result.output.startswith("baton ")


def test_init_wizard_creates_vault_config_and_token(home):
    result, token_output = run_setup()
    assert "Setup complete" in result.output
    token = re.search(r"baton-[A-Za-z0-9_\-]{30,}", token_output)
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


def test_simple_wizard_only_asks_for_provider_and_key(home):
    answers = ["1", CREDENTIAL, "n", "n"]        # Gemini (option 1), key, no more keys, no test
    result = runner.invoke(app, ["init"], input="\n".join(answers) + "\n")
    assert result.exit_code == 0, result.output
    assert "Default model" not in result.output and "Requests per" not in result.output
    assert "passphrase" not in result.output.lower()                        # nothing to type, ever
    key = load_config(home / "config.yaml").keys[0]
    assert (key.id, key.provider) == ("gemini-1", "gemini")
    assert key.models[0] == "gemini-3.8-flash"
    assert CREDENTIAL not in result.output


def test_commands_after_setup(home):
    run_setup()

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


def test_slash_commands_work_inside_the_chat(home, tmp_path, monkeypatch):
    """/keys, /doctor, `baton keys list` and a bare `exit` all run in the chat instead of
    being sent to the model; /keys add reloads the pool without a restart."""
    run_setup()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr("baton.cli.main.is_interactive", lambda: True)
    typed = ["/keys", "baton keys list", "/doctor", "/token", "/keys add", "openai_compatible",
             "http://127.0.0.1:9/v1", "other-model", CREDENTIAL, "/keys frobnicate", "/status", "exit"]
    result = runner.invoke(app, ["chat", "--workspace", str(workspace)], input="\n".join(typed) + "\n")
    output = result.output
    assert result.exit_code == 0, output
    assert output.count("openai-compatible-1") >= 3                          # /keys, baton keys list, /status
    assert "configuration is valid" in output                                # /doctor
    assert "keys reloaded: 2 key(s)" in output                               # /keys add took effect live
    assert "openai-compatible-2" in output
    assert "Traceback" not in output and CREDENTIAL not in output


def test_slash_dropdown_matches_only_slash_lines():
    from baton.cli.ui import slash_matches

    assert [name for name, _ in slash_matches("/")][:2] == ["/help", "/model"]
    assert [name for name, _ in slash_matches("/keys a")] == ["/keys add"]
    assert [name for name, _ in slash_matches("/UP")] == ["/update"]
    assert slash_matches("hello") == [] and slash_matches("/zzz") == [] and slash_matches("/exit") == []


def test_update_command_reports_up_to_date(home, monkeypatch):
    from baton.cli import update as update_module

    monkeypatch.setattr(update_module, "run_update", lambda: update_module.UpdateResult(True, "Already up to date."))
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 0 and "Already up to date" in result.output and "Restart" not in result.output

    monkeypatch.setattr(update_module, "run_update", lambda: update_module.UpdateResult(True, "Updated.", True))
    assert "Restart Baton" in runner.invoke(app, ["update"]).output

    monkeypatch.setattr(update_module, "run_update", lambda: update_module.UpdateResult(False, "git fetch failed"))
    failed = runner.invoke(app, ["update"])
    assert failed.exit_code == 1 and "git fetch failed" in failed.output


def test_serve_refuses_to_expose_without_both_switches(home):
    assert run_init().exit_code == 0
    refused = runner.invoke(app, ["serve", "--host", "0.0.0.0"])               # noqa: S104
    assert refused.exit_code == 1 and "refusing to listen" in refused.output
    flag_only = runner.invoke(app, ["serve", "--host", "0.0.0.0", "--expose"])  # noqa: S104
    assert flag_only.exit_code == 1 and "allow_remote" in flag_only.output


def test_interactive_session_survives_slash_commands_and_a_failing_turn(home, tmp_path, monkeypatch):
    """Drive the REPL: every slash command, then a turn whose upstream is
    unreachable. The session must report the failure and keep running."""
    assert run_init().exit_code == 0
    config_file = home / "config.yaml"
    # Fail fast instead of waiting for the unreachable upstream to "recover".
    config_file.write_text(config_file.read_text(encoding="utf-8") + "rotation:\n  max_queue_wait: 0\n  connect_timeout: 2\n",
                           encoding="utf-8")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr("baton.cli.main.is_interactive", lambda: True)
    typed = ["/help", "/status", "/model fast", "/approval auto_edit", "/nonsense", "", "hello there", "/handoff",
             "/clear", "/exit"]
    result = runner.invoke(app, ["chat", "--workspace", str(workspace)], input="\n".join(typed) + "\n")

    assert result.exit_code == 0, result.output
    output = result.output
    assert "Baton" in output and "/help for commands" in output          # banner
    assert "Commands" in output and "openai-compatible-1" in output      # /help and /status
    assert "model: fast" in output and "approval: auto_edit" in output
    assert "unknown command" in output
    assert "exhausted, rate-limited or unhealthy" in output              # the failing turn was reported...
    assert "Your message was not sent" in output
    assert "conversation cleared" in output                              # ...and the session carried on
    assert "Traceback" not in output and CREDENTIAL not in output


def test_one_shot_prompt_exits_nonzero_when_no_key_can_answer(home, tmp_path):
    assert run_init().exit_code == 0
    config_file = home / "config.yaml"
    config_file.write_text(config_file.read_text(encoding="utf-8") + "rotation:\n  max_queue_wait: 0\n  connect_timeout: 2\n",
                           encoding="utf-8")
    result = runner.invoke(app, ["chat", "-p", "hello", "--workspace", str(tmp_path)])
    assert result.exit_code == 1 and "exhausted" in result.output
    assert runner.invoke(app, ["chat", "-p", "x", "--approval", "bogus"]).exit_code == 1


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
