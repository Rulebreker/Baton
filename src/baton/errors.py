"""Exception hierarchy.

The router's whole fallback strategy hangs off `ErrorKind`: every provider
failure is classified into one of these buckets, and the bucket alone decides
whether we rotate, cool the key down, disable it, or give the error back to the
caller. Keeping that decision table in one place is what keeps rotation
predictable.
"""

from __future__ import annotations

from enum import Enum


class BatonError(Exception):
    """Base class for every error Baton raises on purpose."""


class ConfigError(BatonError):
    """The configuration file is missing, malformed or unsafe."""


class VaultError(BatonError):
    """The encrypted key store could not be opened or written."""


class ErrorKind(str, Enum):
    RATE_LIMIT = "rate_limit"            # short-lived 429: cool down, rotate
    QUOTA_EXHAUSTED = "quota_exhausted"  # daily/monthly quota gone: park until reset, rotate
    AUTH = "auth"                        # key revoked or wrong: disable, rotate
    TRANSIENT = "transient"              # 5xx / network: circuit-breaker strike, rotate
    TIMEOUT = "timeout"                  # treated like TRANSIENT
    OVERLOADED = "overloaded"            # 503/529 "high demand": the provider is busy, not the key: back off and wait
    MODEL_NOT_FOUND = "model_not_found"  # 404: the *model name* is wrong or retired; the key itself is fine
    CONTEXT_LENGTH = "context_length"    # prompt too big: compact, do not burn other keys
    BAD_REQUEST = "bad_request"          # caller's fault: never rotate, surface it
    UNKNOWN = "unknown"                  # be conservative: strike, rotate

    @property
    def rotates(self) -> bool:
        """True when trying another key could plausibly succeed."""
        return self in {
            ErrorKind.RATE_LIMIT,
            ErrorKind.QUOTA_EXHAUSTED,
            ErrorKind.AUTH,
            ErrorKind.TRANSIENT,
            ErrorKind.TIMEOUT,
            ErrorKind.UNKNOWN,
            ErrorKind.OVERLOADED,
            ErrorKind.MODEL_NOT_FOUND,
        }

    @property
    def temporary(self) -> bool:
        """True when the problem goes away by itself, so waiting is the right response."""
        return self in {ErrorKind.OVERLOADED, ErrorKind.TRANSIENT, ErrorKind.TIMEOUT}


class ProviderError(BatonError):
    """A classified upstream failure.

    `message` must already be scrubbed by the provider adapter; it may be shown
    to proxy clients and written to logs.
    """

    def __init__(
        self,
        kind: ErrorKind,
        message: str,
        *,
        status: int | None = None,
        retry_after: float | None = None,
        partial: bool = False,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.status = status
        self.retry_after = retry_after
        # True when a stream had already delivered content before failing.
        self.partial = partial


class AllKeysExhaustedError(BatonError):
    """No key can serve the request right now. Baton fails closed."""

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        attempts: list[str] | None = None,
        reasons: list[str] | None = None,
        kinds: dict[str, str] | None = None,
        waited: float = 0.0,
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.attempts = attempts or []
        # key id -> ErrorKind value of that key's most recent failure (the real mix of causes).
        self.kinds = kinds or {}
        self.waited = waited
        # Scrubbed "key: what the provider said" lines. Kept out of `message` so the
        # proxy does not relay upstream error text to its clients; the terminal shows them.
        self.reasons = reasons or []
