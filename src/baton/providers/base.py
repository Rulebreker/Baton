"""Provider-neutral types and the helpers every adapter shares.

Baton's internal wire format is the OpenAI chat-completions shape (messages
with `role`/`content`/`tool_calls`, tools as `{"type": "function", ...}`).
It is the de-facto lingua franca, the proxy speaks it natively, and each
adapter only has to translate to and from it.
"""

from __future__ import annotations

import contextlib
import email.utils
import json
import re
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..errors import ErrorKind, ProviderError
from ..redact import scrub

# How much of an upstream error body we are willing to read and surface.
_MAX_ERROR_BODY = 4096

_CONTEXT_HINTS = (
    "context length", "context_length", "maximum context", "context window", "too many tokens",
    "prompt is too long", "exceeds the maximum number of tokens", "input is too long",
    "request too large", "reduce the length",
)
# Signs that a 429 is a long-lived quota problem rather than a per-minute burst.
_QUOTA_HINTS = (
    "insufficient_quota", "perday", "per day", "daily", "billing", "credit balance",
    "monthly", "exceeded your current quota, please check your plan",
)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class RateInfo:
    """What the provider told us about remaining quota (from response headers)."""

    limit_requests: int | None = None
    remaining_requests: int | None = None
    limit_tokens: int | None = None
    remaining_tokens: int | None = None

    def utilization(self) -> float | None:
        ratios = []
        if self.limit_requests and self.remaining_requests is not None:
            ratios.append(1 - self.remaining_requests / self.limit_requests)
        if self.limit_tokens and self.remaining_tokens is not None:
            ratios.append(1 - self.remaining_tokens / self.limit_tokens)
        return max(0.0, min(1.0, max(ratios))) if ratios else None


@dataclass
class ChatRequest:
    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    response_format: dict[str, Any] | None = None
    # Extra OpenAI parameters (seed, penalties, ...) forwarded only to
    # OpenAI-style upstreams; other adapters ignore them.
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatResult:
    content: str | None
    tool_calls: list[dict[str, Any]]
    finish_reason: str
    usage: Usage
    model: str
    rate: RateInfo | None = None
    usage_estimated: bool = False


@dataclass
class StreamChunk:
    """One increment of a streamed answer, already in OpenAI delta shape."""

    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage | None = None
    rate: RateInfo | None = None


# --- token estimation -------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Cheap, provider-agnostic token estimate (about 4 characters per token).

    Deliberately not exact: it is used to *reserve* quota before a call and as
    a fallback when a provider returns no usage. It errs slightly high, which
    is the safe direction for quota accounting.
    """
    return len(text) // 4 + 1 if text else 0


def message_text(message: dict[str, Any]) -> str:
    """Flatten a message's content (string or content-part list) to text."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                pieces.append(str(part.get("text", "")))
        return "\n".join(pieces)
    return ""


def estimate_message_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int:
    total = 0
    for message in messages:
        total += 4 + estimate_tokens(message_text(message))
        content = message.get("content")
        if isinstance(content, list):
            # Images cost real tokens; a flat allowance beats pretending they are free.
            total += 800 * sum(1 for p in content if isinstance(p, dict) and p.get("type") == "image_url")
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            total += estimate_tokens(str(function.get("name", ""))) + estimate_tokens(str(function.get("arguments", "")))
    if tools:
        total += estimate_tokens(json.dumps(tools))
    return total


# --- error handling ---------------------------------------------------------

_DURATION = re.compile(r"(?:(\d+)h)?(?:(\d+)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+)ms)?$")


def parse_duration(value: str | None) -> float | None:
    """Parse `12`, `1.5s`, `6m0s`, `200ms` or an HTTP date into seconds."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    match = _DURATION.match(value)
    if match and any(match.groups()):
        hours, minutes, seconds, millis = match.groups()
        return int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(seconds or 0) + int(millis or 0) / 1000
    try:
        when = email.utils.parsedate_to_datetime(value)
        return max(0.0, when.timestamp() - time.time())
    except (TypeError, ValueError):
        return None


def parse_rate_headers(headers: httpx.Headers) -> RateInfo | None:
    """Read rate-limit headers in both the OpenAI and the Anthropic spelling.

    OpenAI style:    x-ratelimit-remaining-requests
    Anthropic style: anthropic-ratelimit-requests-remaining
    """
    info = RateInfo()
    found = False
    for name, raw in headers.items():
        lowered = name.lower()
        if "ratelimit-" not in lowered:
            continue
        parts = lowered.split("ratelimit-", 1)[1].split("-")
        kind = "limit" if "limit" in parts else "remaining" if "remaining" in parts else None
        # "input-tokens"/"output-tokens" variants are skipped: the combined
        # "tokens" header is the one comparable to our own counters.
        unit = "requests" if parts in (["limit", "requests"], ["remaining", "requests"], ["requests", "limit"],
                                       ["requests", "remaining"]) else None
        if unit is None and parts in (["limit", "tokens"], ["remaining", "tokens"], ["tokens", "limit"],
                                      ["tokens", "remaining"]):
            unit = "tokens"
        if kind is None or unit is None:
            continue
        try:
            number = int(float(raw))
        except ValueError:
            continue
        setattr(info, f"{kind}_{unit}", number)
        found = True
    return info if found else None


def _extract_error_message(body: str) -> tuple[str, str]:
    """Return (message, machine_code) from a provider error body."""
    try:
        data = json.loads(body)
    except ValueError:
        return body.strip()[:500], ""
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        return str(data)[:500], ""
    error = data.get("error", data)
    if isinstance(error, str):
        return error[:500], ""
    if isinstance(error, dict):
        code = error.get("code") or error.get("type") or error.get("status") or ""
        return str(error.get("message") or error)[:500], str(code)
    return str(error)[:500], ""


def classify_http_error(provider: str, status: int, body: str, headers: httpx.Headers) -> ProviderError:
    """Turn an HTTP failure into a classified, scrubbed `ProviderError`.

    This table is the heart of fallback behaviour:
      401/403         -> AUTH             (key is bad: disable it, rotate)
      402             -> QUOTA_EXHAUSTED  (billing: park the key, rotate)
      429             -> RATE_LIMIT, or QUOTA_EXHAUSTED when the body says the
                         daily/monthly allowance is gone
      408/409/5xx/529 -> TRANSIENT        (provider hiccup: strike, rotate)
      400/413/422     -> BAD_REQUEST or CONTEXT_LENGTH (the *request* is wrong;
                         other keys would fail identically, so do not burn them)
      404             -> UNKNOWN          (usually "model not available to this
                         key": another key may well have it, so rotate)
    """
    message, code = _extract_error_message(body)
    lowered = f"{message} {code} {body[:_MAX_ERROR_BODY]}".lower()
    retry_after = parse_duration(headers.get("retry-after"))
    if retry_after is None:
        # Gemini reports the delay inside the JSON body: "retryDelay": "34s".
        match = re.search(r'"retrydelay"\s*:\s*"([\d.]+)s"', lowered)
        if match:
            retry_after = float(match.group(1))
    if retry_after is None:
        retry_after = parse_duration(headers.get("x-ratelimit-reset-requests"))

    if status in (401, 403):
        kind = ErrorKind.AUTH
    elif status == 402:
        kind = ErrorKind.QUOTA_EXHAUSTED
    elif status == 429:
        kind = ErrorKind.QUOTA_EXHAUSTED if any(h in lowered for h in _QUOTA_HINTS) else ErrorKind.RATE_LIMIT
    elif status in (408, 409) or status >= 500:
        kind = ErrorKind.TRANSIENT
    elif status == 404:
        kind = ErrorKind.UNKNOWN
    elif status in (400, 413, 422):
        if status == 400 and ("api key not valid" in lowered or "api_key_invalid" in lowered):
            kind = ErrorKind.AUTH  # Gemini answers 400, not 401, for a bad key
        elif status == 413 or any(h in lowered for h in _CONTEXT_HINTS):
            kind = ErrorKind.CONTEXT_LENGTH
        else:
            kind = ErrorKind.BAD_REQUEST
    else:
        kind = ErrorKind.UNKNOWN

    text = scrub(f"{provider} returned HTTP {status}: {message or 'no error message'}")
    return ProviderError(kind, text, status=status, retry_after=retry_after)


def classify_transport_error(provider: str, exc: Exception) -> ProviderError:
    """Network-level failures never include response data, only the exception type."""
    if isinstance(exc, httpx.TimeoutException):
        return ProviderError(ErrorKind.TIMEOUT, f"{provider} request timed out ({type(exc).__name__})")
    return ProviderError(ErrorKind.TRANSIENT, f"{provider} connection failed ({type(exc).__name__})")


async def raise_for_status(provider: str, response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    try:
        raw = await response.aread()
        body = raw[:_MAX_ERROR_BODY].decode("utf-8", "replace")
    except httpx.HTTPError:
        body = ""
    raise classify_http_error(provider, response.status_code, body, response.headers)


async def iter_sse(response: httpx.Response) -> AsyncIterator[tuple[str, str]]:
    """Yield (event, data) pairs from a server-sent-events body."""
    event, data_lines = "", []
    # aclosing: when a consumer stops early (e.g. at "[DONE]") the underlying
    # line iterator is closed deterministically instead of by the garbage
    # collector at some later, possibly loop-less, moment.
    async with contextlib.aclosing(response.aiter_lines()) as lines:
        async for line in lines:
            if line == "":
                if data_lines:
                    yield event, "\n".join(data_lines)
                event, data_lines = "", []
            elif line.startswith(":"):
                continue  # comment / keep-alive
            elif line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip(" "))
    if data_lines:
        yield event, "\n".join(data_lines)


def loads_or_error(provider: str, data: str) -> dict[str, Any]:
    try:
        parsed = json.loads(data)
    except ValueError as exc:
        raise ProviderError(ErrorKind.TRANSIENT, f"{provider} sent malformed JSON") from exc
    if not isinstance(parsed, dict):
        raise ProviderError(ErrorKind.TRANSIENT, f"{provider} sent an unexpected payload")
    return parsed


def parse_data_url(url: str) -> tuple[str, str] | None:
    """Split `data:image/png;base64,AAAA` into (media_type, base64_data)."""
    match = re.match(r"data:([\w.+\-/]+);base64,(.+)$", url, re.S)
    return (match.group(1), match.group(2)) if match else None


class Provider(ABC):
    """One adapter instance per provider type; stateless apart from the client.

    The API key is passed per call and only ever placed in a request header
    (never a URL), so it cannot end up in access logs or exception strings.
    """

    name = "provider"

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    @abstractmethod
    async def complete(self, request: ChatRequest, *, api_key: str, base_url: str) -> ChatResult: ...

    @abstractmethod
    def stream(self, request: ChatRequest, *, api_key: str, base_url: str) -> AsyncIterator[StreamChunk]: ...

    async def _post_json(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> httpx.Response:
        try:
            response = await self.client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise classify_transport_error(self.name, exc) from None
        await raise_for_status(self.name, response)
        return response
