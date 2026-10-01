"""Wires config, vault, pool, router and handoff into one object."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import BatonConfig
from .errors import ConfigError
from .handoff import HandoffManager
from .paths import ensure_private_dir
from .pool import KeyPool
from .providers import build_providers
from .providers.base import ChatRequest
from .redact import get_redactor
from .router import Router
from .session import Relay
from .vault import Vault

log = logging.getLogger("baton.runtime")


def collect_secrets(config: BatonConfig, vault: Vault | None) -> dict[str, str]:
    """Resolve each key's secret from the vault or the environment.

    Every resolved secret is registered with the redactor *before* anything
    else can log, so even an early failure cannot print a key.
    """
    redactor = get_redactor()
    secrets: dict[str, str] = {}
    for key in config.keys:
        if key.secret == "vault":
            value = vault.get_key(key.id) if vault is not None else None
        else:
            value = os.environ.get(key.secret.removeprefix("env:"))
        if value:
            value = value.strip()
            redactor.register_secret(value, f"key:{key.id}")
            secrets[key.id] = value
    return secrets


def build_http_client(config: BatonConfig) -> httpx.AsyncClient:
    rotation = config.rotation
    return httpx.AsyncClient(
        timeout=httpx.Timeout(rotation.request_timeout, connect=rotation.connect_timeout),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        # Never follow redirects: a redirect would replay the API-key header
        # to whatever host the response pointed at.
        follow_redirects=False,
        headers={"User-Agent": "baton"},
    )


@dataclass
class Runtime:
    config: BatonConfig
    home: Path
    pool: KeyPool
    router: Router
    handoff: HandoffManager
    relay: Relay
    client: httpx.AsyncClient
    vault: Vault | None = None

    @classmethod
    def build(
        cls,
        config: BatonConfig,
        vault: Vault | None,
        *,
        home: Path,
        client: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = time.time,
        persist: bool = True,
    ) -> Runtime:
        get_redactor().pii = config.logging.redact_pii
        secrets = collect_secrets(config, vault)
        missing = [key.id for key in config.keys if key.id not in secrets]
        if missing:
            log.warning("no secret found for key(s): %s", ", ".join(missing))
        if config.keys and not secrets:
            raise ConfigError("none of the configured keys has a secret available (vault entry or env variable)")

        if persist:
            ensure_private_dir(home)
        client = client or build_http_client(config)
        pool = KeyPool(config, secrets, state_path=(home / "state.json") if persist else None, clock=clock)
        router = Router(config, pool, build_providers(client))

        async def summarize(request: ChatRequest, pin: str | None, exclude: frozenset[str]) -> str:
            route = await router.complete(request, pin=pin, exclude=exclude)
            return route.result.content or ""

        handoff = HandoffManager(
            config.handoff,
            summarize,
            checkpoint_dir=(home / "checkpoints") if persist else None,
        )
        relay = Relay(config, pool, router, handoff)
        return cls(config, home, pool, router, handoff, relay, client, vault)

    async def aclose(self) -> None:
        self.pool.save(force=True)
        await self.client.aclose()
