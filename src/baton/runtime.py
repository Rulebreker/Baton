"""Wires config, vault, pool, router and handoff into one object."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

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


def collect_secrets(config: BatonConfig, vault: Vault | None, problems: list[str] | None = None) -> dict[str, str]:
    """Resolve each key's secret from the vault or the environment.

    Every resolved secret is registered with the redactor *before* anything
    else can log, so even an early failure cannot print a key. Human-readable
    reasons for unusable keys are appended to `problems`.
    """
    redactor = get_redactor()
    secrets: dict[str, str] = {}
    problems = problems if problems is not None else []
    for key in config.keys:
        if key.secret == "vault":
            value = vault.get_key(key.id) if vault is not None else None
            bound = vault.bound_host(key.id) if vault is not None else None
            actual = (urlparse(str(key.base_url)).hostname or "").lower()
            if value and bound and bound != actual:
                # The config now points this key at a different server than
                # the one it was stored for. Refuse: this is exactly what a
                # tampered config would do to steal the key.
                redactor.register_secret(value, f"key:{key.id}")
                problems.append(
                    f"key '{key.id}' was stored for host {bound} but the config now points to {actual}; refusing "
                    "to use it. If the change is intended, remove the key and add it again."
                )
                continue
        else:
            value = os.environ.get(key.secret.removeprefix("env:"))
        if value:
            value = value.strip()
            redactor.register_secret(value, f"key:{key.id}")
            secrets[key.id] = value
        else:
            source = "the vault" if key.secret == "vault" else f"environment variable {key.secret[4:]}"
            problems.append(f"key '{key.id}' has no secret in {source}")
    for problem in problems:
        log.warning("%s", problem)
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
    # Why some configured keys were left out of the pool (shown by the CLI).
    problems: list[str] = field(default_factory=list)

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
        problems: list[str] = []
        secrets = collect_secrets(config, vault, problems)
        if config.keys and not secrets:
            raise ConfigError("none of the configured keys is usable:\n  " + "\n  ".join(problems))

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
        runtime = cls(config, home, pool, router, handoff, relay, client, vault)
        runtime.problems = problems
        return runtime

    async def aclose(self) -> None:
        self.pool.save(force=True)
        await self.client.aclose()
