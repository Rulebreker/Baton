"""Which model names exist right now, according to the provider.

Model names are retired and renamed all the time, so Baton does not trust any name it
was shipped with or that a config pinned long ago. This module asks each key's provider
for its model list (cached), and uses it to:

* resolve `auto` (and a key configured with `models: [auto]`) to a suitable live model,
* recover when a request fails with "model not found / retired": pick a replacement
  (the one the provider's own error names, if the list confirms it, else the best match),
* audit pinned names at startup so a stale config is reported and worked around.

Everything here degrades quietly: no list endpoint, no network, or a failed call means
"no list available", and callers fall back to the pinned name exactly as before.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .paths import secure_write
from .pool import AUTO_MODELS
from .providers import PROVIDER_DEFAULTS, Provider

if TYPE_CHECKING:
    from .pool import KeyPool, KeyState

log = logging.getLogger("baton.models")

CACHE_TTL = 6 * 3600.0       # a model list changes rarely; six hours is plenty fresh
FAILURE_TTL = 120.0          # after a failed list call, do not hammer the endpoint
LIST_TIMEOUT = 10.0
_CACHE_VERSION = 1

# Names that are not general chat models. Dropped before ranking.
_NOT_CHAT = (
    "embed", "imagen", "veo", "tts", "audio", "image", "live", "transcribe", "realtime", "moderation",
    "whisper", "dall-e", "aqa", "robotics", "computer-use", "search", "instruct", "codex", "deep-research",
    "speech", "omni-moderation",
)
_UNSTABLE = ("preview", "exp", "beta", "thinking")
_PROVIDER_PREFIX = {"gemini": "gemini", "openai": "gpt-", "anthropic": "claude"}
_VERSION = re.compile(r"\d+(?:[.\-]\d+)*")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:\-]*")


def _version(name: str) -> tuple[int, ...]:
    match = _VERSION.search(name)
    numbers = [int(part) for part in re.split(r"[.\-]", match.group())] if match else []
    return tuple((numbers + [0, 0, 0, 0])[:4])


def _family_rank(provider: str, name: str) -> int:
    """Lower is preferred. Fast/cheap tiers first: that is what a pool of shared keys wants."""
    lowered = name.lower()
    if provider == "gemini":
        if "flash" in lowered and "lite" not in lowered:
            return 0
        return 1 if "flash" in lowered else 2 if "pro" in lowered else 3
    if provider == "openai":
        return 0 if "mini" in lowered else 2 if "nano" in lowered else 1
    if provider == "anthropic":
        for rank, family in enumerate(("sonnet", "opus", "haiku")):
            if family in lowered:
                return rank
    return 3


def pick_best(provider: str, models: Collection[str], *, like: str | None = None) -> str | None:
    """The most suitable chat model among `models`, or None when there is no safe choice.

    Newest version wins within a tier, stable names beat previews, and when `like` is the
    name being replaced, the same tier is preferred (a retired "pro" gets a "pro" back).
    No model name is hardcoded: ordering comes from the names themselves.
    """
    prefix = _PROVIDER_PREFIX.get(provider)
    if prefix is None:
        return None  # openai_compatible: a server's list says nothing about which model is right
    candidates = [
        name for name in models
        if name.lower().startswith(prefix) and not any(token in name.lower() for token in _NOT_CHAT)
    ]
    if not candidates:
        return None
    wanted = _family_rank(provider, like) if like else None

    def order(name: str):
        lowered = name.lower()
        return (
            any(token in lowered for token in _UNSTABLE),
            wanted is not None and _family_rank(provider, name) != wanted,
            _family_rank(provider, name),
            tuple(-part for part in _version(name)),
            len(name),
            name,
        )

    return min(candidates, key=order)


def hint_from_message(message: str, bad: str, models: Collection[str]) -> str | None:
    """A replacement named by the provider's own error text, but only if the list confirms it."""
    available = set(models)
    named = [token.rstrip(".-_:") for token in _TOKEN.findall(message or "")]
    for token in reversed(named):
        if token in available and token != bad:
            return token
    return None


@dataclass
class _Entry:
    at: float
    fingerprint: str
    models: list[str]


@dataclass
class StaleModel:
    key_id: str
    model: str
    suggestion: str | None


class ModelCatalog:
    def __init__(
        self,
        providers: dict[str, Provider],
        *,
        clock=time.time,
        cache_path: Path | None = None,
        ttl: float = CACHE_TTL,
        timeout: float = LIST_TIMEOUT,
    ) -> None:
        self.providers = providers
        self.clock = clock
        self.cache_path = cache_path
        self.ttl = ttl
        self.timeout = timeout
        self.swaps: dict[tuple[str, str], str] = {}
        self._lists: dict[str, _Entry] = {}
        self._failed: dict[str, float] = {}
        self._load()

    # --- persistence --------------------------------------------------------

    def _load(self) -> None:
        if self.cache_path is None:
            return
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if data.get("version") != _CACHE_VERSION:
                return
            for key_id, raw in data.get("keys", {}).items():
                self._lists[key_id] = _Entry(float(raw["at"]), str(raw["fingerprint"]), [str(m) for m in raw["models"]])
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            self._lists.clear()  # an unreadable cache is just a cold cache

    def _save(self) -> None:
        if self.cache_path is None:
            return
        data = {"version": _CACHE_VERSION, "keys": {
            key_id: {"at": e.at, "fingerprint": e.fingerprint, "models": e.models} for key_id, e in self._lists.items()
        }}
        try:
            secure_write(self.cache_path, json.dumps(data).encode("utf-8"))
        except OSError:
            log.debug("could not write the model cache")

    # --- lookups ------------------------------------------------------------

    @staticmethod
    def _fingerprint(key: KeyState) -> str:
        return f"{key.config.provider}|{key.config.base_url}"

    def cached_models(self, key_id: str | None = None) -> list[str]:
        """Models learned so far (no network). All keys' models, or one key's."""
        entries = [self._lists[key_id]] if key_id in self._lists else ([] if key_id else list(self._lists.values()))
        return sorted({name for entry in entries for name in entry.models})

    async def available(self, key: KeyState, *, refresh: bool = False) -> list[str] | None:
        """The provider's model list for this key, or None when it cannot be had."""
        now = self.clock()
        fingerprint = self._fingerprint(key)
        entry = self._lists.get(key.id)
        if entry is not None and entry.fingerprint != fingerprint:
            entry = None  # the key now points somewhere else: the old list does not apply
        if not refresh:
            if entry is not None and now - entry.at < self.ttl:
                return entry.models
            if self._failed.get(key.id, 0.0) > now:
                return entry.models if entry else None
        provider = self.providers.get(key.config.provider)
        try:
            if provider is None:
                raise NotImplementedError
            listed = await asyncio.wait_for(
                provider.list_models(api_key=key.secret, base_url=str(key.config.base_url)), timeout=self.timeout
            )
            models = sorted({name.strip() for name in listed if name and name.strip()})
            if not models:
                raise ValueError("empty model list")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - any failure just means "no list right now"
            self._failed[key.id] = now + FAILURE_TTL
            log.debug("model list unavailable for key '%s' (%s)", key.id, type(exc).__name__)
            return entry.models if entry else None  # a stale list still beats guessing
        self._failed.pop(key.id, None)
        self._lists[key.id] = _Entry(now, fingerprint, models)
        self._save()
        return models

    async def best(self, key: KeyState, *, like: str | None = None) -> str | None:
        models = await self.available(key)
        return pick_best(key.config.provider, models, like=like) if models else None

    async def concrete(self, key: KeyState, model: str) -> str:
        """The model name to actually send: swaps applied, `auto` resolved from the live list."""
        model = self.swaps.get((key.id, model), model)
        if model.strip().lower() not in AUTO_MODELS:
            return model
        best = await self.best(key)
        return best or PROVIDER_DEFAULTS[key.config.provider].model or model

    async def replacement(self, key: KeyState, bad: str, message: str) -> str | None:
        """A model to use instead of `bad`, which the provider says does not exist (any more)."""
        models = await self.available(key, refresh=True)
        if not models:
            return None
        hinted = hint_from_message(message, bad, models)
        if hinted:
            return hinted
        return pick_best(key.config.provider, [m for m in models if m != bad], like=bad)

    def remember_swap(self, key_id: str, old: str, new: str) -> None:
        self.swaps[(key_id, old)] = new


async def audit_pinned_models(pool: KeyPool, catalog: ModelCatalog, *, refresh: bool = False) -> list[StaleModel]:
    """Pinned model names the provider no longer lists, each with the best replacement.

    `openai_compatible` servers are skipped: their lists use local naming (`llama3:latest`)
    that does not have to match what a config says, so a mismatch proves nothing.
    """
    stale: list[StaleModel] = []
    for key in pool.keys.values():
        if key.config.provider not in _PROVIDER_PREFIX:
            continue
        listed = await catalog.available(key, refresh=refresh)
        if not listed:
            continue
        have = set(listed)
        for model in key.config.models:
            if model.strip().lower() in AUTO_MODELS or model in have:
                continue
            stale.append(StaleModel(
                key.id, model, pick_best(key.config.provider, [m for m in listed if m != model], like=model)
            ))
    return stale
