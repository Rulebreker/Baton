"""Request routing: pick a key, call it, and fall back when it fails.

Fallback rules (one place, so behaviour is predictable):

* A failure that another key could plausibly avoid (rate limit, exhausted
  quota, bad key, 5xx, timeout) penalises the key in the pool and the request
  moves to the next candidate. The caller sees nothing.
* A failure that is the request's own fault (400) is returned immediately.
  Retrying it elsewhere would only burn quota on every key in the pool.
* When no key is usable, Baton waits for the soonest recovery if that is
  within `rotation.max_queue_wait`, otherwise it fails closed with
  `AllKeysExhaustedError`. It never loops forever and never crashes.
* Streams can fail over only until the first content reaches the caller.
  After that the partial answer cannot be un-sent, so the error is raised
  with `partial=True` and the caller decides (the agent restarts the turn).
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field

from .config import BatonConfig
from .errors import AllKeysExhaustedError, ErrorKind, ProviderError
from .models import ModelCatalog
from .pool import KeyPool, KeyState
from .providers import Provider
from .providers.base import ChatRequest, ChatResult, StreamChunk, Usage, estimate_message_tokens, estimate_tokens
from .redact import scrub

log = logging.getLogger("baton.router")

_CAUSE = {
    ErrorKind.OVERLOADED.value: "provider overloaded (high demand)",
    ErrorKind.MODEL_NOT_FOUND.value: "model not found or retired",
    ErrorKind.AUTH.value: "key rejected",
    ErrorKind.RATE_LIMIT.value: "rate limited",
    ErrorKind.QUOTA_EXHAUSTED.value: "quota used up",
    ErrorKind.TRANSIENT.value: "provider or network error",
    ErrorKind.TIMEOUT.value: "timed out",
    ErrorKind.UNKNOWN.value: "unexpected error",
}

# Reserve for the answer when the caller did not cap it.
_DEFAULT_COMPLETION_RESERVE = 1024
_MAX_COMPLETION_RESERVE = 8192


@dataclass
class Attempt:
    key_id: str
    model: str
    outcome: str  # "ok" or the ErrorKind value
    detail: str = ""


@dataclass
class RouteResult:
    result: ChatResult
    key_id: str
    provider: str
    model: str
    attempts: list[Attempt] = field(default_factory=list)


@dataclass
class RouteInfo:
    """Told to stream consumers once a key has accepted the request."""

    key_id: str
    provider: str
    model: str
    attempts: list[Attempt]


class Router:
    def __init__(
        self,
        config: BatonConfig,
        pool: KeyPool,
        providers: dict[str, Provider],
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        catalog: ModelCatalog | None = None,
        on_notice: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.rotation = config.rotation
        self.pool = pool
        self.providers = providers
        self._sleep = sleep
        self.catalog = catalog or ModelCatalog(providers, clock=pool.clock)
        if pool.discovered is None:
            pool.discovered = self.catalog.cached_models
        # Told about things the user should see mid-request: a model swapped in, "provider busy".
        self.on_notice = on_notice

    def _notify(self, message: str) -> None:
        message = scrub(message)
        log.info("%s", message)
        if self.on_notice is not None:
            try:
                self.on_notice(message)
            except Exception:  # noqa: BLE001 - a broken UI hook must never break a request
                log.debug("notice hook failed", exc_info=True)

    # --- shared pieces ------------------------------------------------------

    def _estimate(self, request: ChatRequest) -> tuple[int, int]:
        prompt = estimate_message_tokens(request.messages, request.tools)
        completion = min(request.max_tokens or _DEFAULT_COMPLETION_RESERVE, _MAX_COMPLETION_RESERVE)
        return prompt, prompt + completion

    def _max_attempts(self) -> int:
        return self.rotation.max_attempts or max(2, 2 * len(self.pool.keys))

    def _budget_left(self, attempts: list[Attempt]) -> bool:
        """Failed attempts are capped, except "provider overloaded" ones: those are bounded by the
        time budget instead (`rotation.overload_max_wait`), since waiting is the point."""
        counted = sum(1 for a in attempts if a.outcome != ErrorKind.OVERLOADED.value)
        return counted < self._max_attempts() and len(attempts) < self._max_attempts() + 60

    async def _recover_model(self, key: KeyState, pool_model: str, model: str, error: ProviderError,
                             done: set[tuple[str, str]]) -> bool:
        """The provider says `model` does not exist (any more). The key is fine: find a live
        replacement once, remember it, and tell the user. True when a retry is worthwhile."""
        tag = (key.id, pool_model)
        if tag in done:
            return False
        done.add(tag)
        replacement = await self.catalog.replacement(key, model, error.message)
        if not replacement or replacement == model:
            return False
        self.catalog.remember_swap(key.id, pool_model, replacement)
        if model != pool_model:
            self.catalog.remember_swap(key.id, model, replacement)
        self._notify(f"model '{model}' is no longer available for key {key.id}; using '{replacement}' instead")
        return True

    def _check_model(self, request: ChatRequest) -> None:
        if self.rotation.unknown_model == "reject" and not self.pool.model_is_known(request.model):
            raise ProviderError(
                ErrorKind.BAD_REQUEST,
                f"model '{scrub(request.model)[:80]}' is not offered by any configured key",
                status=404,
            )

    def _pick(
        self,
        request: ChatRequest,
        prompt_tokens: int,
        est_tokens: int,
        prefer: str | None,
        pin: str | None,
        exclude: set[str],
    ) -> tuple[tuple[KeyState, str] | None, bool]:
        """Return (choice, too_big): `too_big` means usable keys exist but none can hold the prompt."""
        candidates = self.pool.candidates(request.model, prefer=prefer, exclude=exclude, est_tokens=est_tokens)
        if pin is not None:
            candidates = [c for c in candidates if c[0].id == pin]
        for key, model in candidates:
            # Never send a prompt to a key whose context window cannot hold it:
            # that is a guaranteed 400 and a wasted request.
            if key.config.context_window and prompt_tokens > key.config.context_window:
                continue
            return (key, model), False
        return None, bool(candidates)

    def _attempt_request(self, request: ChatRequest, key: KeyState, model: str) -> ChatRequest:
        max_tokens = request.max_tokens
        cap = key.config.max_output_tokens
        if cap and (max_tokens is None or max_tokens > cap):
            max_tokens = cap
        return dataclasses.replace(request, model=model, max_tokens=max_tokens)

    def _exhausted(self, attempts: list[Attempt], context_blocked: bool, waited: float = 0.0) -> Exception:
        if context_blocked and not any(a.outcome != ErrorKind.CONTEXT_LENGTH.value for a in attempts):
            return ProviderError(
                ErrorKind.CONTEXT_LENGTH,
                "the conversation is too long for every available key's context window",
                status=400,
            )
        retry_after = self.pool.next_available_in()
        total = len(self.pool.keys)
        # Each key's most recent failure: the message must describe what really happened to
        # *each* key, not guess one cause for all of them.
        kinds = {key.id: key.last_kind for key in self.pool.keys.values() if key.last_kind}
        present = set(kinds.values())
        covered = total > 0 and len(kinds) == total
        if total == 0:
            message = "No API keys are configured. Run `baton keys add`."
        elif covered and present == {ErrorKind.OVERLOADED.value}:
            message = (f"The provider is overloaded right now (high demand): all {total} key(s) were busy, and Baton "
                       f"stopped retrying after waiting {int(waited)}s. Your keys are fine; try again in a minute.")
        elif covered and present == {ErrorKind.MODEL_NOT_FOUND.value}:
            message = "No key can use the requested model: the provider says it does not exist or has been retired."
        elif present & {ErrorKind.OVERLOADED.value, ErrorKind.MODEL_NOT_FOUND.value}:
            parts = "; ".join(f"{key_id}: {_CAUSE.get(kind, kind)}" for key_id, kind in kinds.items())
            message = f"No key could answer ({parts})."
        else:
            hint = f" Soonest recovery in about {int(retry_after) + 1}s." if retry_after is not None else ""
            message = f"All {total} API key(s) are exhausted, rate-limited or unhealthy.{hint}"
        reasons = list(dict.fromkeys(f"{a.key_id}: {a.detail}" for a in attempts if a.outcome != "ok" and a.detail))
        if not reasons:
            # Nothing was tried in this call: the keys were already benched by earlier failures.
            reasons = [f"{row['id']}: {row['last_error']}" for row in self.pool.snapshot() if row["last_error"]]
        return AllKeysExhaustedError(
            message,
            retry_after=retry_after,
            attempts=[f"{a.key_id}: {a.outcome}" for a in attempts],
            reasons=reasons,
            kinds=kinds,
            waited=waited,
        )

    async def _wait_for_capacity(self, waited: float, pin: str | None, skip: set[str] | frozenset[str] = frozenset()) -> float | None:
        """Sleep until a key recovers, if that is soon enough. Returns the time slept.

        When every key is out only because of a passing problem (provider busy, timeouts) the
        budget is `rotation.overload_max_wait`; for quota windows it stays `max_queue_wait`.
        Keys this request has already ruled out (`skip`) do not count either way."""
        if pin is not None:
            return None  # pinned calls (handoff summaries) must fail fast
        wait = self.pool.next_available_in(skip)
        if wait is None:
            return None
        passing = self.pool.waiting_on_temporary_only(skip)
        budget = self.rotation.overload_max_wait if passing else self.rotation.max_queue_wait
        if waited + wait > budget:
            return None
        wait = max(wait, 0.05) + 0.05
        if passing:
            self._notify(f"provider busy, retrying in {math.ceil(wait)}s ({int(waited)}s of {int(budget)}s used)...")
        await self._sleep(wait)
        return wait

    @staticmethod
    def _as_provider_error(exc: Exception, provider: str) -> ProviderError:
        if isinstance(exc, ProviderError):
            return exc
        if isinstance(exc, asyncio.TimeoutError):
            return ProviderError(ErrorKind.TIMEOUT, f"{provider} request exceeded the time limit")
        # An adapter bug or an unexpected payload shape must degrade to a
        # fallback, never crash the caller. The traceback goes to the
        # (redacted) log for debugging.
        log.exception("unexpected error in %s adapter", provider)
        return ProviderError(ErrorKind.UNKNOWN, f"{provider} adapter failed ({type(exc).__name__})")

    # --- non-streaming ------------------------------------------------------

    async def complete(
        self,
        request: ChatRequest,
        *,
        prefer: str | None = None,
        pin: str | None = None,
        exclude: frozenset[str] | set[str] = frozenset(),
    ) -> RouteResult:
        self._check_model(request)
        prompt_tokens, est_tokens = self._estimate(request)
        attempts: list[Attempt] = []
        skip = set(exclude)
        waited = 0.0
        context_blocked = False
        swapped: set[tuple[str, str]] = set()   # (key, model) pairs already given their one re-resolution
        favourite = prefer

        while self._budget_left(attempts):
            picked, too_big = self._pick(request, prompt_tokens, est_tokens, favourite, pin, skip)
            if picked is None:
                if too_big:
                    context_blocked = True  # keys exist, but none can hold the prompt
                    break
                slept = await self._wait_for_capacity(waited, pin, skip)
                if slept is None:
                    break
                waited += slept
                continue

            key, pool_model = picked
            model = await self.catalog.concrete(key, pool_model)
            provider = self.providers[key.config.provider]
            attempt_request = self._attempt_request(request, key, model)
            favourite = prefer
            self.pool.reserve(key, est_tokens)
            try:
                result = await asyncio.wait_for(
                    provider.complete(attempt_request, api_key=key.secret, base_url=str(key.config.base_url)),
                    timeout=self.rotation.request_timeout,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - classified below
                error = self._as_provider_error(exc, key.config.provider)
                attempts.append(Attempt(key.id, model, error.kind.value, error.message))
                self.pool.record_failure(key, error)
                log.info("key '%s' failed (%s); %s", key.id, error.kind.value,
                         "rotating" if error.kind.rotates else "not retryable")
                if error.kind is ErrorKind.BAD_REQUEST:
                    raise error from None
                if error.kind is ErrorKind.CONTEXT_LENGTH:
                    # Another key may have a larger window; this one is out.
                    context_blocked = True
                    skip.add(key.id)
                elif error.kind is ErrorKind.MODEL_NOT_FOUND:
                    # The model is wrong, the key is fine: retry this key once with a live model.
                    if await self._recover_model(key, pool_model, model, error, swapped):
                        favourite = key.id
                    else:
                        skip.add(key.id)
                continue
            finally:
                self.pool.release(key, est_tokens)

            self.pool.record_success(key, result.usage, result.rate)
            attempts.append(Attempt(key.id, model, "ok"))
            return RouteResult(result, key.id, key.config.provider, model, attempts)

        raise self._exhausted(attempts, context_blocked, waited)

    # --- streaming ----------------------------------------------------------

    async def stream(
        self,
        request: ChatRequest,
        *,
        prefer: str | None = None,
        on_route: Callable[[RouteInfo], None] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        self._check_model(request)
        prompt_tokens, est_tokens = self._estimate(request)
        attempts: list[Attempt] = []
        skip: set[str] = set()
        waited = 0.0
        context_blocked = False
        swapped: set[tuple[str, str]] = set()
        favourite = prefer

        while self._budget_left(attempts):
            picked, too_big = self._pick(request, prompt_tokens, est_tokens, favourite, None, skip)
            if picked is None:
                if too_big:
                    context_blocked = True
                    break
                slept = await self._wait_for_capacity(waited, None, skip)
                if slept is None:
                    break
                waited += slept
                continue

            key, pool_model = picked
            model = await self.catalog.concrete(key, pool_model)
            provider = self.providers[key.config.provider]
            attempt_request = self._attempt_request(request, key, model)
            favourite = prefer
            self.pool.reserve(key, est_tokens)
            source = provider.stream(attempt_request, api_key=key.secret, base_url=str(key.config.base_url))
            committed = False      # has any content reached our caller?
            finished = False
            usage: Usage | None = None
            rate = None
            text_chars = 0
            try:
                while True:
                    try:
                        # Idle timeout per chunk: a stalled stream must not hang
                        # the caller for the whole request timeout.
                        chunk = await asyncio.wait_for(source.__anext__(), timeout=self.rotation.stream_idle_timeout)
                    except StopAsyncIteration:
                        break
                    if chunk.usage is not None:
                        usage = chunk.usage
                    if chunk.rate is not None:
                        rate = chunk.rate
                    text_chars += len(chunk.text) + sum(
                        len(str((call.get("function") or {}).get("arguments", ""))) for call in chunk.tool_calls
                    )
                    if chunk.text or chunk.tool_calls or chunk.finish_reason:
                        if not committed:
                            committed = True
                            if on_route is not None:
                                on_route(RouteInfo(key.id, key.config.provider, model, attempts))
                        yield chunk
                    elif committed and chunk.usage is not None:
                        # OpenAI-style streams send usage in a trailing chunk
                        # with no content; callers still need it.
                        yield chunk
                finished = True
            except asyncio.CancelledError:
                raise
            except GeneratorExit:
                raise
            except Exception as exc:  # noqa: BLE001 - classified below
                error = self._as_provider_error(exc, key.config.provider)
                attempts.append(Attempt(key.id, model, error.kind.value, error.message))
                self.pool.record_failure(key, error)
                if committed:
                    error.partial = True
                    raise error from None
                if error.kind is ErrorKind.BAD_REQUEST:
                    raise error from None
                if error.kind is ErrorKind.CONTEXT_LENGTH:
                    context_blocked = True
                    skip.add(key.id)
                elif error.kind is ErrorKind.MODEL_NOT_FOUND:
                    if await self._recover_model(key, pool_model, model, error, swapped):
                        favourite = key.id
                    else:
                        skip.add(key.id)
                log.info("key '%s' failed before streaming began (%s); rotating", key.id, error.kind.value)
                continue
            finally:
                self.pool.release(key, est_tokens)
                await source.aclose()
                if committed or finished:
                    # Count the usage even when the consumer hung up early or
                    # the stream died midway: the provider billed it either way.
                    if usage is None:
                        usage = Usage(prompt_tokens, estimate_tokens("x" * text_chars))
                    if finished:
                        self.pool.record_success(key, usage, rate)
                    else:
                        key.day_tokens += usage.total_tokens
                        key.day_requests += 1

            attempts.append(Attempt(key.id, model, "ok"))
            if not committed and on_route is not None:
                on_route(RouteInfo(key.id, key.config.provider, model, attempts))  # empty but successful answer
            return

        raise self._exhausted(attempts, context_blocked, waited)
