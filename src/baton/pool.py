"""The key pool: per-key quota accounting, health and candidate selection.

Concurrency model: every method here is synchronous and is only ever called
from the single asyncio event loop that owns the pool. With no `await` inside
a method, each call is atomic with respect to other coroutines, so no locks
are needed and there is no lock to deadlock on. (Run one Baton process per
data directory; two processes would each keep their own counters.)

Two kinds of "full" are tracked separately, because they call for different
reactions:

* short-window pressure (requests/tokens per *minute*, cooldowns) clears by
  itself in seconds. The router simply routes around it.
* long-window depletion (requests/tokens per *day*, provider says "quota
  exhausted") does not clear soon. That is what triggers a summary handoff.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .config import BatonConfig, KeyConfig
from .errors import ErrorKind, ProviderError
from .paths import secure_write
from .providers.base import RateInfo, Usage
from .redact import mask_secret

log = logging.getLogger("baton.pool")

AUTO_MODELS = {"", "auto", "baton", "default"}
_MINUTE = 60.0
_PROBE_INTERVAL = 3600.0       # how long to park a key whose quota/auth failed with no hint
_HEADER_TTL = 60.0             # provider-reported utilisation is a per-minute signal
_MAX_BREAKER_COOLDOWN = 900.0
_SAVE_INTERVAL = 2.0


class KeyStatus(str, Enum):
    READY = "ready"
    DRAINING = "draining"      # past the handoff threshold, still usable as a last resort
    COOLDOWN = "cooldown"
    EXHAUSTED = "exhausted"
    DISABLED = "disabled"


@dataclass
class KeyState:
    config: KeyConfig
    secret: str = field(repr=False)  # never appears in repr()/logs
    order: int = 0
    # sliding one-minute window of (timestamp, requests, tokens)
    minute: deque[tuple[float, int, int]] = field(default_factory=deque)
    day: str = ""
    day_requests: int = 0
    day_tokens: int = 0
    in_flight: int = 0
    reserved_tokens: int = 0
    header_util: float | None = None
    header_at: float = 0.0
    cooldown_until: float = 0.0
    exhausted_until: float = 0.0
    disabled_until: float = 0.0
    manually_disabled: bool = False
    failures: int = 0
    total_requests: int = 0
    total_tokens: int = 0
    last_used: float = 0.0
    last_error: str = ""

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def masked(self) -> str:
        return mask_secret(self.secret)


def _utc_day(now: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(now))


def _seconds_to_utc_midnight(now: float) -> float:
    return 86400.0 - (now % 86400.0)


class KeyPool:
    def __init__(
        self,
        config: BatonConfig,
        secrets: dict[str, str],
        *,
        state_path: Path | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.rotation = config.rotation
        self.clock = clock
        self.state_path = state_path
        self._last_save = 0.0
        self._dirty = False
        self.keys: dict[str, KeyState] = {}
        for order, key_config in enumerate(config.keys):
            secret = secrets.get(key_config.id)
            if not secret:
                log.warning("key '%s' has no secret available and will be skipped", key_config.id)
                continue
            state = KeyState(config=key_config, secret=secret, order=order)
            state.manually_disabled = not key_config.enabled
            self.keys[key_config.id] = state
        self._load_state()

    # --- model resolution ---------------------------------------------------

    def resolve_model(self, key: KeyState, requested: str) -> tuple[str | None, int]:
        """Map the client's model name to what this key should be asked for.

        Returns (model, tier): tier 0 = the key genuinely offers what was asked
        for, tier 1 = cross-model fallback to the key's default, (None, _) = this
        key must not serve the request.
        """
        requested = (requested or "").strip()
        if requested.lower() in AUTO_MODELS:
            return key.config.default_model, 0
        alias = self.config.model_aliases.get(requested)
        if alias is not None:
            target = alias.get(key.id) or alias.get(key.config.provider)
            return (target, 0) if target else (None, 0)
        if requested in key.config.models:
            return requested, 0
        if self.rotation.unknown_model == "reject":
            return None, 1
        return key.config.default_model, 1

    def model_is_known(self, requested: str) -> bool:
        requested = (requested or "").strip()
        if requested.lower() in AUTO_MODELS or requested in self.config.model_aliases:
            return True
        return any(requested in key.config.models for key in self.keys.values())

    def known_models(self) -> list[str]:
        names = {"auto", *self.config.model_aliases}
        for key in self.keys.values():
            names.update(key.config.models)
        return sorted(names)

    # --- utilisation --------------------------------------------------------

    def _roll(self, key: KeyState, now: float) -> None:
        """Expire old window entries. Called before every read of the counters."""
        while key.minute and now - key.minute[0][0] >= _MINUTE:
            key.minute.popleft()
        today = _utc_day(now)
        if key.day != today:
            key.day, key.day_requests, key.day_tokens = today, 0, 0
            self._dirty = True

    def short_utilization(self, key: KeyState, extra_tokens: int = 0) -> float:
        now = self.clock()
        self._roll(key, now)
        limits = key.config.limits
        ratios = [0.0]
        if limits.rpm:
            ratios.append((sum(e[1] for e in key.minute) + key.in_flight) / limits.rpm)
        if limits.tpm:
            ratios.append((sum(e[2] for e in key.minute) + key.reserved_tokens + extra_tokens) / limits.tpm)
        if key.header_util is not None and now - key.header_at < _HEADER_TTL:
            ratios.append(key.header_util)
        return max(ratios)

    def long_utilization(self, key: KeyState, extra_tokens: int = 0) -> float:
        now = self.clock()
        self._roll(key, now)
        limits = key.config.limits
        ratios = [0.0]
        if limits.rpd:
            ratios.append((key.day_requests + key.in_flight) / limits.rpd)
        if limits.tpd:
            ratios.append((key.day_tokens + key.reserved_tokens + extra_tokens) / limits.tpd)
        return max(ratios)

    def utilization(self, key: KeyState) -> float:
        return max(self.short_utilization(key), self.long_utilization(key))

    def needs_handoff(self, key: KeyState) -> bool:
        """True when a session should plan to leave this key (long-window quota nearly gone)."""
        now = self.clock()
        if key.manually_disabled or key.exhausted_until > now or key.disabled_until > now:
            return True
        return self.long_utilization(key) >= self.rotation.handoff_threshold

    def status(self, key: KeyState) -> KeyStatus:
        now = self.clock()
        if key.manually_disabled or key.disabled_until > now:
            return KeyStatus.DISABLED
        if key.exhausted_until > now or self.long_utilization(key) >= self.rotation.hard_limit:
            return KeyStatus.EXHAUSTED
        if key.cooldown_until > now or self.short_utilization(key) >= self.rotation.hard_limit:
            return KeyStatus.COOLDOWN
        if self.long_utilization(key) >= self.rotation.handoff_threshold:
            return KeyStatus.DRAINING
        return KeyStatus.READY

    def usable(self, key: KeyState, extra_tokens: int = 0) -> bool:
        if self.status(key) not in (KeyStatus.READY, KeyStatus.DRAINING):
            return False
        if extra_tokens:
            # Would this request overshoot? A key with an empty window is
            # always allowed one attempt: if a single request is bigger than
            # our configured limit, the provider is the right judge, not us.
            idle = not key.minute and key.in_flight == 0
            if self.short_utilization(key, extra_tokens) > 1.0 and not idle:
                return False
            if self.long_utilization(key, extra_tokens) > 1.0 and key.day_tokens > 0:
                return False
        return True

    # --- selection ----------------------------------------------------------

    def candidates(
        self,
        model: str,
        *,
        prefer: str | None = None,
        exclude: set[str] | frozenset[str] = frozenset(),
        est_tokens: int = 0,
    ) -> list[tuple[KeyState, str]]:
        """Usable keys for `model`, best first, as (key, model-to-request) pairs."""
        ranked: list[tuple[tuple, KeyState, str]] = []
        for key in self.keys.values():
            if key.id in exclude or not self.usable(key, est_tokens):
                continue
            target, tier = self.resolve_model(key, model)
            if target is None or (tier == 1 and not self.rotation.cross_model_fallback):
                continue
            draining = self.status(key) is KeyStatus.DRAINING
            if self.rotation.strategy == "least_used":
                spread: float = self.utilization(key)
            elif self.rotation.strategy == "round_robin":
                spread = key.last_used
            else:  # sticky: fill keys in config order, one after another
                spread = 0.0
            rank = (
                tier,                       # exact model match beats fallback
                draining,                   # nearly-empty keys are a last resort
                key.id != prefer,           # session affinity
                key.config.priority,
                spread,
                key.order,
            )
            ranked.append((rank, key, target))
        ranked.sort(key=lambda item: item[0])
        return [(key, target) for _rank, key, target in ranked]

    def next_available_in(self) -> float | None:
        """Seconds until some key becomes usable again, or None if none ever will."""
        now = self.clock()
        waits: list[float] = []
        for key in self.keys.values():
            if key.manually_disabled:
                continue
            wait = max(key.cooldown_until, key.exhausted_until, key.disabled_until) - now
            if self.long_utilization(key) >= self.rotation.hard_limit:
                wait = max(wait, _seconds_to_utc_midnight(now))
            elif self.short_utilization(key) >= self.rotation.hard_limit:
                oldest = key.minute[0][0] if key.minute else now
                wait = max(wait, _MINUTE - (now - oldest))
            waits.append(max(wait, 0.0))
        return min(waits) if waits else None

    # --- accounting ---------------------------------------------------------

    def reserve(self, key: KeyState, est_tokens: int) -> None:
        """Claim quota before the call so concurrent requests cannot jointly overshoot."""
        key.in_flight += 1
        key.reserved_tokens += est_tokens
        key.last_used = self.clock()

    def release(self, key: KeyState, est_tokens: int) -> None:
        key.in_flight = max(0, key.in_flight - 1)
        key.reserved_tokens = max(0, key.reserved_tokens - est_tokens)

    def record_success(self, key: KeyState, usage: Usage, rate: RateInfo | None = None) -> None:
        now = self.clock()
        self._roll(key, now)
        tokens = usage.total_tokens
        key.minute.append((now, 1, tokens))
        key.day_requests += 1
        key.day_tokens += tokens
        key.total_requests += 1
        key.total_tokens += tokens
        key.failures = 0
        key.cooldown_until = 0.0
        key.last_error = ""
        if rate is not None:
            reported = rate.utilization()
            if reported is not None:
                key.header_util, key.header_at = reported, now
        self._dirty = True
        self.save()

    def record_failure(self, key: KeyState, error: ProviderError) -> None:
        """Apply the penalty that matches the failure class (see errors.ErrorKind)."""
        now = self.clock()
        rotation = self.rotation
        key.last_error = f"{error.kind.value}: {error.message}"[:300]
        kind = error.kind

        if kind in (ErrorKind.BAD_REQUEST, ErrorKind.CONTEXT_LENGTH):
            return  # the request was at fault, not the key

        if kind is ErrorKind.AUTH:
            # Parked rather than removed for good: a 403 can be a temporary
            # provider-side block. It is re-probed after the interval.
            key.disabled_until = now + _PROBE_INTERVAL
            log.warning("key '%s' rejected by provider (auth); parked for %ds", key.id, _PROBE_INTERVAL)
        elif kind is ErrorKind.QUOTA_EXHAUSTED:
            wait = error.retry_after if error.retry_after and error.retry_after > _MINUTE else _PROBE_INTERVAL
            key.exhausted_until = now + wait
            log.info("key '%s' quota exhausted; parked for %ds", key.id, wait)
        elif kind is ErrorKind.RATE_LIMIT:
            key.failures += 1
            backoff = min(rotation.backoff_max, rotation.backoff_base * 2 ** min(key.failures, 16))
            wait = error.retry_after if error.retry_after is not None else backoff
            if wait > _PROBE_INTERVAL / 2:
                key.exhausted_until = now + wait  # a very long Retry-After is a quota, not a burst
            else:
                key.cooldown_until = now + max(wait, 1.0)
        else:  # TRANSIENT, TIMEOUT, UNKNOWN
            key.failures += 1
            if key.failures >= rotation.breaker_threshold:
                # Circuit breaker open. Each further failure after the cooldown
                # (half-open probe) doubles the wait, up to a cap.
                over = key.failures - rotation.breaker_threshold
                wait = min(_MAX_BREAKER_COOLDOWN, rotation.breaker_cooldown * 2 ** min(over, 8))
                log.warning("key '%s' circuit breaker open for %ds after %d failures", key.id, wait, key.failures)
            else:
                # Jitter keeps a fleet of clients from retrying in lock-step.
                base = rotation.backoff_base * 2 ** (key.failures - 1)
                wait = min(rotation.backoff_max, base) * (0.5 + random.random())  # noqa: S311 - not security
            key.cooldown_until = now + wait
        self._dirty = True
        self.save()

    def set_enabled(self, key_id: str, enabled: bool) -> bool:
        key = self.keys.get(key_id)
        if key is None:
            return False
        key.manually_disabled = not enabled
        if enabled:
            key.disabled_until = key.exhausted_until = key.cooldown_until = 0.0
            key.failures = 0
        self._dirty = True
        self.save(force=True)
        return True

    # --- reporting ----------------------------------------------------------

    def snapshot(self) -> list[dict]:
        """Status rows for the UI and the status endpoint. Contains no secrets."""
        now = self.clock()
        rows = []
        for key in self.keys.values():
            until = max(key.cooldown_until, key.exhausted_until, key.disabled_until)
            rows.append({
                "id": key.id,
                "provider": key.config.provider,
                "model": key.config.default_model,
                "status": self.status(key).value,
                "utilization": round(min(self.utilization(key), 9.99), 3),
                "daily_utilization": round(min(self.long_utilization(key), 9.99), 3),
                "requests_today": key.day_requests,
                "tokens_today": key.day_tokens,
                "retry_in": round(max(0.0, until - now), 1),
                "last_error": key.last_error,
            })
        return rows

    # --- persistence --------------------------------------------------------

    def save(self, *, force: bool = False) -> None:
        """Persist daily counters so a restart does not forget today's usage."""
        if self.state_path is None or not self._dirty:
            return
        now = self.clock()
        if not force and now - self._last_save < _SAVE_INTERVAL:
            return
        data = {
            "version": 1,
            "keys": {
                key.id: {
                    "day": key.day,
                    "day_requests": key.day_requests,
                    "day_tokens": key.day_tokens,
                    "exhausted_until": key.exhausted_until,
                    "disabled_until": key.disabled_until,
                    "total_requests": key.total_requests,
                    "total_tokens": key.total_tokens,
                }
                for key in self.keys.values()
            },
        }
        try:
            secure_write(self.state_path, json.dumps(data).encode("utf-8"))
            self._last_save, self._dirty = now, False
        except OSError as exc:
            # Losing counters is survivable; crashing a request over it is not.
            log.warning("could not persist usage state (%s)", type(exc).__name__)

    def _load_state(self) -> None:
        if self.state_path is None or not self.state_path.is_file():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            stored = data.get("keys", {})
        except (OSError, ValueError, AttributeError):
            log.warning("usage state file is unreadable; starting with fresh counters")
            return
        today = _utc_day(self.clock())
        for key_id, saved in stored.items():
            key = self.keys.get(key_id)
            if key is None or not isinstance(saved, dict):
                continue
            try:
                if saved.get("day") == today:
                    key.day = today
                    key.day_requests = max(0, int(saved.get("day_requests", 0)))
                    key.day_tokens = max(0, int(saved.get("day_tokens", 0)))
                key.exhausted_until = float(saved.get("exhausted_until", 0.0))
                key.disabled_until = float(saved.get("disabled_until", 0.0))
                key.total_requests = max(0, int(saved.get("total_requests", 0)))
                key.total_tokens = max(0, int(saved.get("total_tokens", 0)))
            except (TypeError, ValueError):
                continue
