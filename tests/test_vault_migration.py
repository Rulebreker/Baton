"""The passphrase prompt must go away: legacy vaults convert once, keyfile mode works headless."""

from __future__ import annotations

import os
import stat

import pytest
from typer.testing import CliRunner

from baton.cli.main import app
from baton.config import BatonConfig, KeyConfig, load_config, save_config
from baton.errors import VaultError
from baton.vault import Vault, peek_vault_mode, raw_key_location

runner = CliRunner()
PASSPHRASE = "correct horse battery staple"
SECRET = "legacy-secret-" + "q7" * 10
posix_only = pytest.mark.skipif(os.name != "posix", reason="file modes are only meaningful on POSIX")


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A data directory with no OS keychain, like Google Cloud Shell."""
    monkeypatch.setenv("BATON_HOME", str(tmp_path / "home"))
    for name in ("BATON_CONFIG", "BATON_PASSPHRASE", "BATON_PROXY_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setattr("baton.vault._keyring", lambda: None)
    (tmp_path / "home").mkdir()
    return tmp_path / "home"


def legacy_vault(home) -> Vault:
    vault = Vault.create(home / "keys.vault", passphrase=PASSPHRASE, scrypt_n=2**4)
    vault.set_key("gemini-1", SECRET, host="generativelanguage.googleapis.com")
    vault.create_token("default")
    return vault


def open_legacy(home) -> Vault:
    return Vault.open(home / "keys.vault", passphrase_provider=lambda: PASSPHRASE)


def test_migration_converts_in_place_and_erases_the_old_copy(home):
    legacy_vault(home)
    vault = open_legacy(home)
    assert vault.mode == "passphrase" and peek_vault_mode(home / "keys.vault") == "passphrase"

    where = vault.migrate_to_keyfile()
    assert where == "key file" and vault.mode == "keyfile"
    assert peek_vault_mode(home / "keys.vault") == "keyfile"

    reopened = Vault.open(home / "keys.vault")                      # no passphrase provider, no env: nothing to ask
    assert reopened.get_key("gemini-1") == SECRET
    assert reopened.bound_host("gemini-1") == "generativelanguage.googleapis.com"
    assert "default" in reopened.token_names()
    # The passphrase-protected generation must not be left lying around.
    assert [p.name for p in home.iterdir() if p.name.endswith(".bak")] == []
    for path in home.iterdir():
        if path.is_file() and path.name.startswith("keys.vault"):
            assert peek_vault_mode(path) != "passphrase"


def test_migrating_twice_is_refused(home):
    legacy_vault(home)
    vault = open_legacy(home)
    vault.migrate_to_keyfile()
    with pytest.raises(VaultError):
        vault.migrate_to_keyfile()


def test_failed_migration_restores_the_original_vault(home, monkeypatch):
    legacy_vault(home)
    vault = open_legacy(home)
    monkeypatch.setattr("baton.vault.load_raw_key", lambda _home: None)   # the new key cannot be read back

    with pytest.raises(VaultError):
        vault.migrate_to_keyfile()

    assert vault.mode == "passphrase"
    assert peek_vault_mode(home / "keys.vault") == "passphrase"
    assert not (home / "master.key").exists()                             # no orphaned key material
    assert open_legacy(home).get_key("gemini-1") == SECRET                # nothing was lost
    assert not any(p.name.endswith(".bak") for p in home.iterdir())


def test_cli_asks_once_converts_and_never_asks_again(home, monkeypatch):
    legacy_vault(home)
    config = BatonConfig(keys=[KeyConfig(id="gemini-1", provider="gemini",
                                         base_url="https://generativelanguage.googleapis.com/v1beta")])
    save_config(home / "config.yaml", config)

    monkeypatch.setenv("BATON_PASSPHRASE", PASSPHRASE)                    # stands in for typing it once
    first = runner.invoke(app, ["status"])
    assert first.exit_code == 0, first.output
    assert "Vault converted" in first.output and SECRET not in first.output
    assert peek_vault_mode(home / "keys.vault") == "keyfile"

    monkeypatch.delenv("BATON_PASSPHRASE")
    second = runner.invoke(app, ["status"])                               # no env, no stdin: would fail if it prompted
    assert second.exit_code == 0, second.output
    assert "passphrase" not in second.output.lower() and "converted" not in second.output.lower()


def test_doctor_reports_the_vault_mode_and_does_not_convert(home):
    legacy_vault(home)
    save_config(home / "config.yaml", BatonConfig())
    before = runner.invoke(app, ["doctor"])
    assert "vault mode: legacy passphrase" in before.output
    assert peek_vault_mode(home / "keys.vault") == "passphrase"           # doctor only looks

    open_legacy(home).migrate_to_keyfile()
    after = runner.invoke(app, ["doctor"])
    assert "vault mode: keyfile, key stored in the key file" in after.output


def test_keyfile_mode_works_headless_like_cloud_shell(home):
    """No keychain, a persistent home directory: the key lives in a private file and survives restarts."""
    Vault.create(home / "keys.vault").set_key("k", SECRET)
    assert raw_key_location(home) == "key file"
    assert Vault.open(home / "keys.vault").get_key("k") == SECRET


@posix_only
def test_headless_key_file_is_owner_only(home):
    Vault.create(home / "keys.vault")
    for name in ("master.key", "keys.vault"):
        assert stat.S_IMODE((home / name).stat().st_mode) == 0o600, name
    legacy = home / "legacy"
    legacy.mkdir()
    old = Vault.create(legacy / "keys.vault", passphrase=PASSPHRASE, scrypt_n=2**4)
    old.migrate_to_keyfile()
    assert stat.S_IMODE((legacy / "master.key").stat().st_mode) == 0o600  # migration writes it with the same care


def test_a_keychain_that_exists_but_is_broken_falls_back_to_the_file(home, monkeypatch):
    """Cloud Shell can have the `keyring` package installed yet no working backend."""

    class Broken:
        def set_password(self, *args):
            raise RuntimeError("no secret service")

        def get_password(self, *args):
            raise RuntimeError("no secret service")

    monkeypatch.setattr("baton.vault._keyring", lambda: Broken())
    Vault.create(home / "keys.vault").set_key("k", SECRET)
    assert (home / "master.key").is_file()
    assert Vault.open(home / "keys.vault").get_key("k") == SECRET
    assert raw_key_location(home) == "key file"


# --- `baton models` -----------------------------------------------------------------------------

def test_models_command_lists_and_fixes_stale_pins(home, monkeypatch):
    from baton.providers.gemini import GeminiProvider

    async def fake_list(self, *, api_key, base_url):
        return ["gemini-3.8-flash", "gemini-3.8-pro"]

    monkeypatch.setattr(GeminiProvider, "list_models", fake_list)
    Vault.create(home / "keys.vault").set_key("g1", SECRET, host="generativelanguage.googleapis.com")
    config = BatonConfig(keys=[KeyConfig(id="g1", provider="gemini", models=["gemini-2.5-flash"],
                                         base_url="https://generativelanguage.googleapis.com/v1beta")])
    save_config(home / "config.yaml", config)

    listed = runner.invoke(app, ["models"])
    assert listed.exit_code == 0, listed.output
    assert "gemini-2.5-flash" in listed.output and "gemini-3.8-flash" in listed.output and "--fix" in listed.output
    assert load_config(home / "config.yaml").keys[0].models == ["gemini-2.5-flash"]       # looking changes nothing

    fixed = runner.invoke(app, ["models", "--fix"])
    assert fixed.exit_code == 0, fixed.output
    assert load_config(home / "config.yaml").keys[0].models == ["gemini-3.8-flash"]
    assert "Nothing to fix" in runner.invoke(app, ["models"]).output
