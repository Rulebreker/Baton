"""Configuration model and loading.

The YAML file holds *settings only*. A key entry says where its secret lives
(`vault`, or `env:SOME_VAR`); the secret itself can never be written here
because every model forbids unknown fields, so a pasted `api_key: sk-...`
is rejected at load time with a clear error instead of silently living in a
file someone might commit.
"""

from __future__ import annotations

import ipaddress
import os
import re
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .errors import ConfigError
from .paths import baton_home, secure_write
from .providers import PROVIDER_DEFAULTS
from .redact import scrub

CONFIG_ENV = "BATON_CONFIG"
MAX_CONFIG_BYTES = 1_000_000
ProviderName = Literal["openai", "anthropic", "gemini", "openai_compatible"]

_ID = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,47}$")
_ENV_REF = re.compile(r"^env:[A-Za-z_][A-Za-z0-9_]*$")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Limits(_Model):
    """Quota ceilings for one key. Leave a field empty when the provider has no such limit."""

    rpm: int | None = Field(default=None, gt=0, description="requests per minute")
    rpd: int | None = Field(default=None, gt=0, description="requests per day")
    tpm: int | None = Field(default=None, gt=0, description="tokens per minute")
    tpd: int | None = Field(default=None, gt=0, description="tokens per day")


class KeyConfig(_Model):
    id: str
    provider: ProviderName
    secret: str = "vault"
    base_url: str | None = None
    models: list[str] = Field(default_factory=list)
    priority: int = Field(default=0, ge=0, le=1000)
    limits: Limits = Field(default_factory=Limits)
    context_window: int | None = Field(default=None, gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    enabled: bool = True
    allow_insecure_http: bool = False

    @field_validator("id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        if not _ID.match(value):
            raise ValueError("must be 1-48 chars of a-z, 0-9, '-' or '_', starting with a letter or digit")
        return value

    @field_validator("secret")
    @classmethod
    def _valid_secret_ref(cls, value: str) -> str:
        if value != "vault" and not _ENV_REF.match(value):
            # Do not echo the value: the likeliest mistake is pasting a real key here.
            raise ValueError("must be 'vault' or 'env:VARIABLE_NAME' (never the key itself)")
        return value

    @model_validator(mode="after")
    def _fill_defaults(self) -> KeyConfig:
        defaults = PROVIDER_DEFAULTS[self.provider]
        if not self.base_url:
            if defaults.base_url is None:
                raise ValueError(f"key '{self.id}': provider 'openai_compatible' requires base_url")
            self.__dict__["base_url"] = defaults.base_url
        if not self.models:
            if not defaults.model:
                raise ValueError(f"key '{self.id}': provider 'openai_compatible' requires at least one model")
            # "auto" = whichever suitable model the provider currently lists (see baton.models),
            # so a config never goes stale when a model is retired.
            self.__dict__["models"] = ["auto"]
        if self.context_window is None:
            self.__dict__["context_window"] = defaults.context_window
        _check_upstream_url(self.id, str(self.base_url), self.allow_insecure_http)
        return self

    @property
    def default_model(self) -> str:
        return self.models[0]


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _check_upstream_url(key_id: str, url: str, allow_insecure: bool) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"key '{key_id}': base_url must be an http(s) URL")
    if parsed.username or parsed.password:
        raise ValueError(f"key '{key_id}': base_url must not contain credentials")
    # Plain HTTP to anything but this machine would send the API key in clear text.
    if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname) and not allow_insecure:
        raise ValueError(
            f"key '{key_id}': refusing plain-http base_url for a remote host "
            "(set allow_insecure_http: true only if you fully trust the network)"
        )


class RotationConfig(_Model):
    # Fraction of a key's quota at which Baton hands the session to the next key.
    handoff_threshold: float = Field(default=0.90, gt=0.1, le=1.0)
    # Fraction past which a key takes no new requests at all (safety reserve
    # for the handoff summary call and for imprecise token estimates).
    hard_limit: float = Field(default=0.98, gt=0.1, le=1.0)
    strategy: Literal["sticky", "least_used", "round_robin"] = "sticky"
    cross_model_fallback: bool = True
    unknown_model: Literal["default", "reject"] = "default"
    max_attempts: int = Field(default=0, ge=0, le=100, description="0 = two passes over the pool")
    request_timeout: float = Field(default=120.0, gt=1, le=3600)
    connect_timeout: float = Field(default=10.0, gt=0.5, le=120)
    stream_idle_timeout: float = Field(default=90.0, gt=1, le=3600)
    max_queue_wait: float = Field(default=15.0, ge=0, le=600)
    # How long to keep waiting when every key is only *temporarily* unavailable (provider
    # overloaded / 503 / timeouts), as opposed to quota windows (`max_queue_wait`).
    overload_max_wait: float = Field(default=90.0, ge=0, le=900)
    backoff_base: float = Field(default=0.5, gt=0, le=60)
    backoff_max: float = Field(default=30.0, gt=0, le=3600)
    breaker_threshold: int = Field(default=3, ge=1, le=50)
    breaker_cooldown: float = Field(default=60.0, gt=0, le=86400)

    @model_validator(mode="after")
    def _ordered(self) -> RotationConfig:
        if self.hard_limit < self.handoff_threshold:
            raise ValueError("rotation.hard_limit must be >= rotation.handoff_threshold")
        return self


class HandoffConfig(_Model):
    enabled: bool = True
    keep_recent_messages: int = Field(default=6, ge=0, le=200)
    min_messages: int = Field(default=8, ge=2, le=1000)
    summary_max_tokens: int = Field(default=1500, ge=200, le=16000)
    # Compact when the transcript would use more than this share of the
    # *receiving* key's context window, even if no quota threshold was hit.
    context_fill_ratio: float = Field(default=0.75, gt=0.1, le=0.95)
    scrub_pii: bool = True
    save_checkpoints: bool = True


class ProxyRateLimit(_Model):
    requests_per_minute: int = Field(default=120, ge=1, le=100_000)
    burst: int = Field(default=30, ge=1, le=100_000)


class ProxyConfig(_Model):
    host: str = "127.0.0.1"
    port: int = Field(default=8787, ge=1, le=65535)
    # Second, deliberate switch required before binding a non-loopback address.
    allow_remote: bool = False
    rate_limit: ProxyRateLimit = Field(default_factory=ProxyRateLimit)
    max_body_bytes: int = Field(default=10_000_000, ge=1024, le=200_000_000)
    max_messages: int = Field(default=2000, ge=1, le=100_000)
    cors_origins: list[str] = Field(default_factory=list)
    compact_on_handoff: bool = False
    auth_fail_limit: int = Field(default=10, ge=1, le=10_000)
    auth_lockout_seconds: float = Field(default=300.0, ge=1, le=86400)

    @field_validator("cors_origins")
    @classmethod
    def _no_wildcard(cls, value: list[str]) -> list[str]:
        if "*" in value:
            raise ValueError("wildcard CORS is not allowed; list explicit origins")
        return value


class AgentConfig(_Model):
    max_tool_iterations: int = Field(default=40, ge=1, le=500)
    command_timeout: float = Field(default=120.0, gt=1, le=3600)
    max_tool_output_chars: int = Field(default=30_000, ge=1000, le=1_000_000)
    approval: Literal["ask", "auto_edit", "auto"] = "ask"
    show_handoffs: bool = True
    system_prompt: str | None = None


class LoggingConfig(_Model):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    redact_pii: bool = True


class BatonConfig(_Model):
    version: int = 1
    keys: list[KeyConfig] = Field(default_factory=list)
    # alias -> {provider name or key id -> model}
    model_aliases: dict[str, dict[str, str]] = Field(default_factory=dict)
    rotation: RotationConfig = Field(default_factory=RotationConfig)
    handoff: HandoffConfig = Field(default_factory=HandoffConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def _unique_ids(self) -> BatonConfig:
        seen: set[str] = set()
        for key in self.keys:
            if key.id in seen:
                raise ValueError(f"duplicate key id '{key.id}'")
            seen.add(key.id)
        return self

    def key(self, key_id: str) -> KeyConfig | None:
        return next((k for k in self.keys if k.id == key_id), None)


def default_config_path() -> Path:
    return baton_home() / "config.yaml"


def find_config(explicit: str | Path | None = None) -> Path:
    """Resolve the config path: --config, $BATON_CONFIG, ~/.baton/config.yaml.

    The current directory is deliberately NOT searched. The agent is run
    inside arbitrary project folders, and a config picked up from a cloned
    repository could point `base_url` at someone else's server and have the
    vault's keys sent there.
    """
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get(CONFIG_ENV)
    if env:
        return Path(env).expanduser()
    return default_config_path()


def _format_validation_error(exc: ValidationError) -> str:
    lines = []
    for error in exc.errors(include_input=False):  # never echo offending input
        location = ".".join(str(part) for part in error["loc"]) or "(root)"
        lines.append(f"  - {location}: {error['msg']}")
    return "\n".join(lines)


def parse_config(data: Any) -> BatonConfig:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError("configuration must be a YAML mapping")
    try:
        return BatonConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError("invalid configuration:\n" + scrub(_format_validation_error(exc))) from None


def load_config(path: Path) -> BatonConfig:
    if not path.is_file():
        raise ConfigError(f"no configuration found at {path}. Run `baton init` to create one.")
    if path.stat().st_size > MAX_CONFIG_BYTES:
        raise ConfigError("configuration file is implausibly large; refusing to parse it")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" (line {mark.line + 1})" if mark else ""
        raise ConfigError(f"configuration is not valid YAML{where}") from None
    except OSError as exc:
        raise ConfigError(f"cannot read configuration: {type(exc).__name__}") from None
    return parse_config(data)


def save_config(path: Path, config: BatonConfig) -> None:
    data = config.model_dump(mode="json", exclude_defaults=True)
    # Always keep the essentials visible even when they equal the defaults.
    data["version"] = config.version
    data["keys"] = [key.model_dump(mode="json", exclude_defaults=True) | {"id": key.id, "provider": key.provider,
                                                                         "models": key.models} for key in config.keys]
    header = (
        "# Baton configuration. Settings only: API keys live in the encrypted vault\n"
        "# (secret: vault) or in environment variables (secret: env:NAME).\n"
    )
    secure_write(path, (header + yaml.safe_dump(data, sort_keys=False)).encode("utf-8"))


def load_dotenv(*paths: Path) -> None:
    """Minimal .env loader. Existing environment variables always win."""
    for path in paths:
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.removeprefix("export ").partition("=")
            name, value = name.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if name and name not in os.environ:
                os.environ[name] = value
