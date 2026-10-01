"""Redaction, logging, vault, config validation and the secret scanner."""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path

import pytest

from baton.config import load_config, load_dotenv, parse_config, save_config
from baton.errors import ConfigError, VaultError
from baton.logging_setup import RedactingFilter, setup_logging
from baton.redact import Redactor, get_redactor, mask_secret
from baton.vault import Vault, hash_token

ROOT = Path(__file__).resolve().parents[1]

# Key-shaped strings are assembled at runtime so the repository itself never
# contains anything a secret scanner (ours or GitHub's) would flag.
FAKE_KEYS = {
    "anthropic": "sk-ant-" + "aB3" * 14,
    "openai": "sk-" + "proj-" + "xY7" * 14,
    "google": "AIza" + "Sy9" * 11 + "ab",
    "groq": "gsk_" + "Qw8" * 14,
    "github": "ghp_" + "Zx5" * 12,
    "baton": "baton-" + "Lm4" * 12,
}


# --- redaction --------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(FAKE_KEYS))
def test_known_key_formats_are_redacted(name):
    text = Redactor().scrub(f"calling provider with {FAKE_KEYS[name]} failed")
    assert FAKE_KEYS[name] not in text and "[REDACTED" in text


def test_registered_secret_with_unusual_format_is_redacted():
    redactor = Redactor()
    odd = "corp_gateway_credential_0042"
    assert odd in redactor.scrub(f"token was {odd}")          # unknown format: not caught by patterns
    redactor.register_secret(odd, "key:corp")
    assert redactor.scrub(f"token was {odd}") == "token was [REDACTED:key:corp]"


def test_headers_urls_and_assignments_are_redacted():
    redactor = Redactor()
    value = "v3ryS3cretValue99"
    samples = [
        f"Authorization: Bearer {value}",
        f"x-api-key: {value}",
        f"GET https://host/v1/models?key={value}&alt=sse",
        f"https://user:{value}@host/path",
        f'api_key = "{value}"',
        f"DB_PASSWORD={value}",
    ]
    for sample in samples:
        assert value not in redactor.scrub(sample), sample


def test_pii_redaction_and_luhn_check():
    redactor = Redactor()
    text = redactor.scrub("mail a.user@example.org, card 4111 1111 1111 1111, ssn 123-45-6789, order 1234567890123456")
    assert "a.user@example.org" not in text and "4111" not in text and "123-45-6789" not in text
    assert "1234567890123456" in text                         # fails Luhn: an ordinary number, left alone
    assert "a.user@example.org" in Redactor(pii=False).scrub("mail a.user@example.org")


def test_scrub_obj_handles_nested_structures():
    redactor = Redactor()
    scrubbed = redactor.scrub_obj({"a": [FAKE_KEYS["openai"], {"b": FAKE_KEYS["google"]}], "n": 3})
    assert FAKE_KEYS["openai"] not in json.dumps(scrubbed) and scrubbed["n"] == 3


def test_mask_secret_shows_only_edges():
    masked = mask_secret(FAKE_KEYS["anthropic"])
    assert masked.startswith("sk-a") and len(masked) < 15


def test_log_records_are_scrubbed_including_tracebacks(tmp_path):
    secret = "log-leak-credential-777"
    get_redactor().register_secret(secret, "key:test")
    setup_logging(tmp_path, level="DEBUG")
    try:
        log = logging.getLogger("baton.test")
        log.info("using %s for %s", secret, FAKE_KEYS["openai"])
        try:
            raise RuntimeError(f"upstream said {secret}")
        except RuntimeError:
            log.exception("request failed")
        for handler in logging.getLogger().handlers:
            handler.flush()
        written = (tmp_path / "baton.log").read_text(encoding="utf-8")
    finally:
        setup_logging(None)
    assert secret not in written and FAKE_KEYS["openai"] not in written
    assert "RuntimeError" in written and "[REDACTED:key:test]" in written


def test_redacting_filter_survives_bad_format_strings():
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "broken %s %s", ("only-one",), None)
    assert RedactingFilter().filter(record) is True


# --- vault ------------------------------------------------------------------

def test_vault_roundtrip_with_passphrase(tmp_path, monkeypatch):
    monkeypatch.delenv("BATON_PASSPHRASE", raising=False)
    path = tmp_path / "keys.vault"
    vault = Vault.create(path, passphrase="correct horse battery", scrypt_n=2**10)
    vault.set_key("gemini-a", FAKE_KEYS["google"])

    raw = path.read_text(encoding="utf-8")
    assert FAKE_KEYS["google"] not in raw                     # encrypted at rest
    reopened = Vault.open(path, passphrase_provider=lambda: "correct horse battery")
    assert reopened.get_key("gemini-a") == FAKE_KEYS["google"]

    with pytest.raises(VaultError, match="wrong passphrase"):
        Vault.open(path, passphrase_provider=lambda: "incorrect passphrase")
    with pytest.raises(VaultError, match="passphrase-protected"):
        Vault.open(path)                                      # non-interactive, no env var


def test_vault_detects_tampering(tmp_path):
    path = tmp_path / "keys.vault"
    Vault.create(path, passphrase="correct horse battery", scrypt_n=2**10).set_key("a", "value-123456789")
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["ciphertext"] = envelope["ciphertext"][:-8] + "AAAAAAA="
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(VaultError):
        Vault.open(path, passphrase_provider=lambda: "correct horse battery")


def test_vault_keyfile_mode_and_backup(tmp_path, monkeypatch):
    monkeypatch.setattr("baton.vault._keyring", lambda: None)  # force the file-based key
    path = tmp_path / "keys.vault"
    vault = Vault.create(path)
    vault.set_key("a", "value-123456789")
    assert (tmp_path / "master.key").is_file()
    assert (tmp_path / "keys.vault.bak").is_file()
    assert Vault.open(path).get_key("a") == "value-123456789"
    (tmp_path / "master.key").unlink()
    with pytest.raises(VaultError, match="master key not found"):
        Vault.open(path)


def test_vault_rejects_weak_passphrase_and_bad_keys(tmp_path):
    with pytest.raises(VaultError, match="at least"):
        Vault.create(tmp_path / "v1.vault", passphrase="short")
    vault = Vault.create(tmp_path / "v2.vault", passphrase="long enough passphrase", scrypt_n=2**10)
    for bad in ("", "has space inside", "x" * 5000):
        with pytest.raises(VaultError):
            vault.set_key("a", bad)
    with pytest.raises(VaultError, match="already exists"):
        Vault.create(tmp_path / "v2.vault", passphrase="long enough passphrase")


def test_proxy_tokens_are_stored_hashed_and_verified(tmp_path):
    path = tmp_path / "keys.vault"
    vault = Vault.create(path, passphrase="correct horse battery", scrypt_n=2**10)
    token = vault.create_token("web-app")
    assert token.startswith("baton-") and len(token) > 40
    assert vault.verify_token(token) == "web-app"
    assert vault.verify_token(token + "x") is None and vault.verify_token("") is None
    assert vault._data["tokens"]["web-app"]["hash"] == hash_token(token)
    assert token not in json.dumps(vault._data)               # clear text is never kept
    with pytest.raises(VaultError):
        vault.create_token("web-app")
    with pytest.raises(VaultError):
        vault.create_token("bad name!")
    assert vault.revoke_token("web-app") and vault.verify_token(token) is None


def test_vault_secrets_are_registered_for_redaction(tmp_path):
    odd = "internal-gateway-credential-31337"
    vault = Vault.create(tmp_path / "keys.vault", passphrase="correct horse battery", scrypt_n=2**10)
    vault.set_key("corp", odd)
    assert odd not in get_redactor().scrub(f"error contacting upstream with {odd}")


# --- config -----------------------------------------------------------------

def test_inline_secret_in_config_is_rejected_without_echoing_it():
    with pytest.raises(ConfigError) as caught:
        parse_config({"keys": [{"id": "a", "provider": "openai", "api_key": FAKE_KEYS["openai"]}]})
    assert FAKE_KEYS["openai"] not in str(caught.value)
    with pytest.raises(ConfigError) as caught:
        parse_config({"keys": [{"id": "a", "provider": "openai", "secret": FAKE_KEYS["openai"]}]})
    assert FAKE_KEYS["openai"] not in str(caught.value)


@pytest.mark.parametrize("bad", [
    {"keys": [{"id": "Bad Id", "provider": "openai"}]},
    {"keys": [{"id": "a", "provider": "nope"}]},
    {"keys": [{"id": "a", "provider": "openai"}, {"id": "a", "provider": "gemini"}]},
    {"keys": [{"id": "a", "provider": "openai_compatible"}]},                       # needs base_url + model
    {"keys": [{"id": "a", "provider": "openai", "base_url": "http://example.com/v1"}]},   # clear-text to remote host
    {"keys": [{"id": "a", "provider": "openai", "base_url": "https://u:p@example.com"}]},
    {"keys": [{"id": "a", "provider": "openai", "limits": {"rpm": 0}}]},
    {"rotation": {"handoff_threshold": 0.95, "hard_limit": 0.9}},
    {"proxy": {"cors_origins": ["*"]}},
    {"proxy": {"port": 70000}},
    {"unknown_section": {}},
    ["not", "a", "mapping"],
])
def test_invalid_config_is_rejected(bad):
    with pytest.raises(ConfigError):
        parse_config(bad)


def test_config_defaults_are_safe():
    config = parse_config({"keys": [{"id": "g", "provider": "gemini"}, {
        "id": "local", "provider": "openai_compatible", "base_url": "http://localhost:11434/v1", "models": ["llama3"],
    }]})
    assert config.proxy.host == "127.0.0.1" and config.proxy.allow_remote is False
    assert config.rotation.handoff_threshold == 0.90
    assert config.keys[0].base_url.startswith("https://") and config.keys[0].models
    assert config.keys[1].base_url == "http://localhost:11434/v1"       # loopback http is fine
    assert config.agent.approval == "ask"


def test_config_save_and_load_roundtrip(tmp_path):
    config = parse_config({
        "keys": [{"id": "g", "provider": "gemini", "limits": {"rpm": 15, "rpd": 1500}, "secret": "env:GEMINI_KEY_1"}],
        "proxy": {"port": 9000},
    })
    path = tmp_path / "config.yaml"
    save_config(path, config)
    loaded = load_config(path)
    assert loaded.keys[0].limits.rpd == 1500 and loaded.keys[0].secret == "env:GEMINI_KEY_1"
    assert loaded.proxy.port == 9000
    with pytest.raises(ConfigError, match="baton init"):
        load_config(tmp_path / "missing.yaml")
    (tmp_path / "broken.yaml").write_text("keys: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(tmp_path / "broken.yaml")


def test_dotenv_loader_never_overrides_real_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nexport BATON_T_ONE="from-file"\nBATON_T_TWO=from-file\n\nnot a pair\n', encoding="utf-8")
    monkeypatch.setenv("BATON_T_TWO", "from-env")
    monkeypatch.delenv("BATON_T_ONE", raising=False)
    load_dotenv(env, tmp_path / "does-not-exist")
    import os

    assert os.environ["BATON_T_ONE"] == "from-file" and os.environ["BATON_T_TWO"] == "from-env"
    monkeypatch.delenv("BATON_T_ONE")


# --- secret scanner ---------------------------------------------------------

def load_scanner():
    spec = importlib.util.spec_from_file_location("secret_scan", ROOT / "scripts" / "secret_scan.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", [".env", ".env.local", "keys.vault", "master.key", "server.pem", "state.json",
                                  "baton.yaml", "debug.log", ".baton/config.yaml", "checkpoints/a.json"])
def test_scanner_blocks_forbidden_paths(name):
    assert load_scanner().forbidden_path(name)


@pytest.mark.parametrize("name", [".env.example", "baton.example.yaml", "src/baton/vault.py", "README.md"])
def test_scanner_allows_safe_paths(name):
    assert load_scanner().forbidden_path(name) is None


def test_scanner_finds_keys_and_respects_placeholders_and_marker():
    scanner = load_scanner()
    marker = scanner.ALLOW_MARKER
    text = "\n".join([
        f"A = '{FAKE_KEYS['anthropic']}'",
        f"B = '{FAKE_KEYS['google']}'",
        "OPENAI_API_KEY=sk-your-key-here-replace-me-xxxxxxxxxxxxxxxx",
        f"C = '{FAKE_KEYS['groq']}'  # {marker}",
        "password = 'kJ8#not-matched-because-of-symbols'",
        "api_token = '" + "a8Kd93LmQz71" + "XvBn20RtYp55" + "'",
        "max_tokens = self.config.summary_max_tokens",
    ])
    findings = scanner.scan_text("sample.py", text)
    assert [(f.line, f.kind) for f in findings] == [
        (1, "Anthropic API key"), (2, "Google API key"), (6, "High-entropy credential assignment"),
    ]
    assert all(FAKE_KEYS["anthropic"] not in f.excerpt for f in findings)     # scanner output is masked too


def test_scanner_flags_personal_paths_but_not_generic_ones():
    scanner = load_scanner()
    windows_path = "C:" + "\\Users\\" + "jsmith" + "\\project"
    findings = scanner.scan_text("doc.md", f"open {windows_path}\nopen /home/" + "jsmith/project\nopen /home/you/project")
    assert [f.line for f in findings] == [1, 2]


def test_repository_itself_is_clean():
    """The committed tree must always pass its own scanner."""
    scanner = load_scanner()
    findings = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT).as_posix()
        if not path.is_file() or relative.startswith((".git/", ".venv/")) or "__pycache__" in relative \
                or ".pytest_cache" in relative or ".ruff_cache" in relative or relative.endswith(".egg-info"):
            continue
        if path.suffix in {".py", ".md", ".yaml", ".yml", ".toml", ".sh", ".ps1", ".example", ""}:
            findings += scanner.scan_text(relative, path.read_text(encoding="utf-8", errors="replace"))
    assert not findings, findings
