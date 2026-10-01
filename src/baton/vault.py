"""Encrypted key store.

API keys and proxy-token hashes are kept in one file, encrypted with
AES-256-GCM. The data key comes from one of two places:

* **passphrase mode** - derived with scrypt from a passphrase the user types
  (or supplies via `BATON_PASSPHRASE` for unattended starts). Strongest: the
  vault file is useless without the passphrase.
* **keyfile mode** - a random 32-byte key stored in the OS keychain when the
  optional `keyring` package is installed, otherwise in `master.key` next to
  the vault with owner-only permissions. Convenient for services, but it only
  protects against the vault file leaking *on its own* (backups, accidental
  commits), not against someone who can read the whole data directory.

Decrypted secrets exist only in this process's memory. Python cannot
guarantee zeroing of immutable strings, so "in memory only" is the honest
promise here, not "wiped after use".
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from .errors import VaultError
from .paths import secure_write
from .redact import get_redactor

PASSPHRASE_ENV = "BATON_PASSPHRASE"
VAULT_FILE = "keys.vault"
KEYFILE = "master.key"
TOKEN_PREFIX = "baton-"
MIN_PASSPHRASE_LENGTH = 10

_FORMAT_VERSION = 1
_AAD = b"baton-vault-v1"
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2**16, 8, 1
_KEYRING_SERVICE, _KEYRING_USER = "baton", "vault-master-key"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(data: str) -> bytes:
    return base64.b64decode(data.encode("ascii"))


def _derive(passphrase: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return Scrypt(salt=salt, length=32, n=n, r=r, p=p).derive(passphrase.encode("utf-8"))


def _keyring():
    try:
        import keyring  # type: ignore[import-not-found]
        from keyring.backends import fail  # type: ignore[import-not-found]

        if isinstance(keyring.get_keyring(), fail.Keyring):
            return None  # installed but no usable backend (e.g. headless Linux)
        return keyring
    except Exception:  # noqa: BLE001 - any keyring trouble means "not available"
        return None


def store_raw_key(home: Path, key: bytes) -> str:
    """Persist a keyfile-mode master key. Returns where it went."""
    backend = _keyring()
    if backend is not None:
        try:
            backend.set_password(_KEYRING_SERVICE, _KEYRING_USER, _b64(key))
            return "OS keychain"
        except Exception:  # noqa: BLE001, S110 - fall back to the file
            pass
    secure_write(home / KEYFILE, _b64(key).encode("ascii"))
    return str(home / KEYFILE)


def load_raw_key(home: Path) -> bytes | None:
    backend = _keyring()
    if backend is not None:
        try:
            stored = backend.get_password(_KEYRING_SERVICE, _KEYRING_USER)
            if stored:
                return _unb64(stored)
        except Exception:  # noqa: BLE001, S110
            pass
    path = home / KEYFILE
    if path.is_file():
        try:
            return _unb64(path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            return None
    return None


def hash_token(token: str) -> str:
    # Tokens are 256 bits of randomness, so a fast hash is appropriate; a slow
    # KDF would only add latency to every proxied request.
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class Vault:
    def __init__(self, path: Path, key: bytes, kdf: dict[str, Any], data: dict[str, Any]) -> None:
        self.path = path
        self._key = key
        self._kdf = kdf
        self._data = data
        self._data.setdefault("keys", {})
        self._data.setdefault("tokens", {})
        self._register_secrets()

    # --- construction -------------------------------------------------------

    @classmethod
    def create(cls, path: Path, *, passphrase: str | None = None, scrypt_n: int = _SCRYPT_N) -> Vault:
        """Create a new vault. With no passphrase, keyfile mode is used."""
        if path.exists():
            raise VaultError(f"a vault already exists at {path}")
        if passphrase is not None:
            if len(passphrase) < MIN_PASSPHRASE_LENGTH:
                raise VaultError(f"passphrase must be at least {MIN_PASSPHRASE_LENGTH} characters")
            salt = os.urandom(16)
            kdf = {"name": "scrypt", "salt": _b64(salt), "n": scrypt_n, "r": _SCRYPT_R, "p": _SCRYPT_P}
            key = _derive(passphrase, salt, scrypt_n, _SCRYPT_R, _SCRYPT_P)
        else:
            key = AESGCM.generate_key(bit_length=256)
            store_raw_key(path.parent, key)
            kdf = {"name": "raw"}
        vault = cls(path, key, kdf, {})
        vault.save()
        return vault

    @classmethod
    def open(cls, path: Path, *, passphrase_provider: Callable[[], str] | None = None) -> Vault:
        if not path.is_file():
            raise VaultError(f"no vault found at {path}. Run `baton init` first.")
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            if envelope.get("baton_vault") != _FORMAT_VERSION:
                raise VaultError("unsupported vault format version")
            kdf = envelope["kdf"]
            nonce, ciphertext = _unb64(envelope["nonce"]), _unb64(envelope["ciphertext"])
        except (OSError, ValueError, KeyError, TypeError):
            raise VaultError("vault file is unreadable or corrupted") from None

        if kdf.get("name") == "scrypt":
            passphrase = os.environ.get(PASSPHRASE_ENV)
            if not passphrase and passphrase_provider is not None:
                passphrase = passphrase_provider()
            if not passphrase:
                raise VaultError(f"vault is passphrase-protected; set {PASSPHRASE_ENV} or run interactively")
            key = _derive(passphrase, _unb64(kdf["salt"]), int(kdf["n"]), int(kdf["r"]), int(kdf["p"]))
        elif kdf.get("name") == "raw":
            loaded = load_raw_key(path.parent)
            if loaded is None:
                raise VaultError("vault master key not found (OS keychain entry or master.key is missing)")
            key = loaded
        else:
            raise VaultError("unsupported vault key-derivation method")

        try:
            plaintext = AESGCM(key).decrypt(nonce, ciphertext, _AAD)
            data = json.loads(plaintext)
        except InvalidTag:
            # GCM authenticates: this is either the wrong key or tampering.
            raise VaultError("could not unlock vault: wrong passphrase, or the file was modified") from None
        except ValueError:
            raise VaultError("vault contents are corrupted") from None
        return cls(path, key, kdf, data)

    # --- persistence --------------------------------------------------------

    def save(self) -> None:
        # A fresh random nonce per write: reusing a GCM nonce with the same key
        # would break confidentiality and integrity.
        nonce = os.urandom(12)
        ciphertext = AESGCM(self._key).encrypt(nonce, json.dumps(self._data).encode("utf-8"), _AAD)
        envelope = {
            "baton_vault": _FORMAT_VERSION,
            "kdf": self._kdf,
            "nonce": _b64(nonce),
            "ciphertext": _b64(ciphertext),
        }
        if self.path.is_file():
            # One generation of backup: a failed disk write must never cost a
            # team its whole key pool.
            try:
                shutil.copy2(self.path, self.path.with_name(self.path.name + ".bak"))
            except OSError:
                pass
        secure_write(self.path, json.dumps(envelope).encode("utf-8"))

    def _register_secrets(self) -> None:
        redactor = get_redactor()
        for key_id, secret in self._data["keys"].items():
            redactor.register_secret(secret, f"key:{key_id}")

    # --- provider keys ------------------------------------------------------

    def key_ids(self) -> list[str]:
        return sorted(self._data["keys"])

    def get_key(self, key_id: str) -> str | None:
        return self._data["keys"].get(key_id)

    def set_key(self, key_id: str, secret: str) -> None:
        secret = secret.strip()
        if not secret or any(ch.isspace() for ch in secret) or len(secret) > 4096:
            raise VaultError("API key is empty, contains whitespace, or is implausibly long")
        self._data["keys"][key_id] = secret
        get_redactor().register_secret(secret, f"key:{key_id}")
        self.save()

    def remove_key(self, key_id: str) -> bool:
        removed = self._data["keys"].pop(key_id, None) is not None
        if removed:
            self.save()
        return removed

    # --- proxy client tokens ------------------------------------------------

    def create_token(self, name: str) -> str:
        """Create a proxy token. The clear text is returned once and never stored."""
        if not name or len(name) > 64 or not all(ch.isalnum() or ch in "-_." for ch in name):
            raise VaultError("token name must be 1-64 chars of letters, digits, '-', '_' or '.'")
        if name in self._data["tokens"]:
            raise VaultError(f"a token named '{name}' already exists")
        token = TOKEN_PREFIX + secrets.token_urlsafe(32)
        self._data["tokens"][name] = {"hash": hash_token(token), "created": int(time.time())}
        self.save()
        return token

    def revoke_token(self, name: str) -> bool:
        removed = self._data["tokens"].pop(name, None) is not None
        if removed:
            self.save()
        return removed

    def token_names(self) -> dict[str, int]:
        return {name: int(meta.get("created", 0)) for name, meta in self._data["tokens"].items()}

    def verify_token(self, token: str) -> str | None:
        """Return the client name for a valid token, else None.

        Every stored hash is compared (no early exit) with a constant-time
        comparison, so response timing reveals neither validity nor which
        client a near-miss resembled.
        """
        candidate = hash_token(token)
        matched: str | None = None
        for name, meta in self._data["tokens"].items():
            if hmac.compare_digest(candidate, str(meta.get("hash", ""))):
                matched = name
        return matched
