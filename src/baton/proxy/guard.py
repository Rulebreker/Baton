"""Abuse controls for the proxy: per-client rate limiting and auth lockout.

Both structures are bounded in size so that an attacker cycling through
client addresses cannot grow them without limit.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable

_MAX_TRACKED = 10_000


class TokenBucket:
    """Classic token bucket per client: `rate_per_minute` sustained, `burst` peak."""

    def __init__(self, rate_per_minute: int, burst: int, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.rate = rate_per_minute / 60.0
        self.capacity = float(burst)
        self.clock = clock
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()

    def allow(self, client: str) -> float:
        """Return 0.0 if the request may proceed, else seconds until it may."""
        now = self.clock()
        tokens, stamp = self._buckets.pop(client, (self.capacity, now))
        tokens = min(self.capacity, tokens + (now - stamp) * self.rate)
        wait = 0.0
        if tokens >= 1.0:
            tokens -= 1.0
        else:
            wait = (1.0 - tokens) / self.rate
        self._buckets[client] = (tokens, now)
        while len(self._buckets) > _MAX_TRACKED:
            self._buckets.popitem(last=False)
        return wait


class AuthLockout:
    """Temporarily refuse an address after repeated failed authentications.

    Tokens carry 256 bits of entropy, so guessing is hopeless anyway; the
    lockout exists to keep a guessing loop from consuming CPU and log space.
    """

    def __init__(self, limit: int, lockout_seconds: float, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.limit = limit
        self.lockout = lockout_seconds
        self.clock = clock
        self._failures: OrderedDict[str, tuple[int, float]] = OrderedDict()

    def blocked_for(self, address: str) -> float:
        count, since = self._failures.get(address, (0, 0.0))
        remaining = since + self.lockout - self.clock()
        if remaining <= 0:
            self._failures.pop(address, None)
            return 0.0
        return remaining if count >= self.limit else 0.0

    def record_failure(self, address: str) -> None:
        now = self.clock()
        count, since = self._failures.pop(address, (0, now))
        if now - since > self.lockout:
            count, since = 0, now
        # The window restarts when the limit is hit, so the lockout lasts a
        # full period from the last counted failure.
        self._failures[address] = (count + 1, now if count + 1 >= self.limit else since)
        while len(self._failures) > _MAX_TRACKED:
            self._failures.popitem(last=False)

    def record_success(self, address: str) -> None:
        self._failures.pop(address, None)
