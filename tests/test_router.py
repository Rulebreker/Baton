"""Router: rotation and graceful fallback. Nothing here may crash or loop forever."""

from __future__ import annotations

import asyncio

import pytest

from baton.errors import AllKeysExhaustedError, ErrorKind, ProviderError
from baton.pool import KeyStatus
from baton.providers.base import ChatRequest, StreamChunk, Usage
from conftest import err, request, secret_for


async def test_first_healthy_key_serves_the_request(harness):
    h = harness(3)
    route = await h.router.complete(request())
    assert route.key_id == "k0" and route.result.content == "ok"
    assert [a.outcome for a in route.attempts] == ["ok"]


@pytest.mark.parametrize("kind", [
    ErrorKind.RATE_LIMIT, ErrorKind.QUOTA_EXHAUSTED, ErrorKind.AUTH,
    ErrorKind.TRANSIENT, ErrorKind.TIMEOUT, ErrorKind.UNKNOWN,
])
async def test_key_failure_falls_back_silently_to_next_key(harness, kind):
    h = harness(3)
    h.provider.script(0, err(kind))
    route = await h.router.complete(request())
    assert route.key_id == "k1"
    assert [a.outcome for a in route.attempts] == [kind.value, "ok"]
    assert h.pool.status(h.key(0)) is not KeyStatus.READY


async def test_bad_request_is_not_retried_on_other_keys(harness):
    h = harness(3)
    h.provider.script(0, err(ErrorKind.BAD_REQUEST, status=400))
    with pytest.raises(ProviderError) as caught:
        await h.router.complete(request())
    assert caught.value.kind is ErrorKind.BAD_REQUEST
    assert len(h.provider.calls) == 1  # no quota burned on k1/k2
    assert h.pool.status(h.key(0)) is KeyStatus.READY


async def test_fails_closed_when_every_key_is_dead(harness):
    h = harness(4)
    for index in range(4):
        h.provider.always(index, err(ErrorKind.QUOTA_EXHAUSTED, retry_after=3600))
    with pytest.raises(AllKeysExhaustedError) as caught:
        await h.router.complete(request())
    assert "All 4 API key(s)" in str(caught.value)
    assert caught.value.retry_after and caught.value.retry_after > 3000
    assert len(h.provider.calls) == 4          # each key tried exactly once
    assert not h.clock.slept                   # an hour away: do not wait, fail now
    # The error names key ids, never secrets.
    assert all(secret_for(i) not in " ".join(caught.value.attempts) for i in range(4))


async def test_attempts_are_bounded_even_if_keys_keep_failing(harness):
    h = harness(3, rotation={"backoff_base": 0.01, "backoff_max": 0.02, "max_queue_wait": 600, "breaker_threshold": 50})
    for index in range(3):
        h.provider.always(index, err(ErrorKind.TRANSIENT))
    with pytest.raises(AllKeysExhaustedError):
        await h.router.complete(request())
    assert len(h.provider.calls) <= 2 * 3      # two passes over the pool, then stop


async def test_waits_for_a_key_that_recovers_soon(harness):
    h = harness(2)
    h.provider.script(0, err(ErrorKind.RATE_LIMIT, retry_after=4))
    h.provider.script(1, err(ErrorKind.RATE_LIMIT, retry_after=3))
    route = await h.router.complete(request())
    assert route.result.content == "ok"
    assert h.clock.slept and 3 <= sum(h.clock.slept) <= h.config.rotation.max_queue_wait


async def test_unexpected_adapter_exception_degrades_to_fallback(harness):
    h = harness(2)
    h.provider.script(0, KeyError("surprising payload shape"))
    route = await h.router.complete(request())
    assert route.key_id == "k1"
    assert route.attempts[0].outcome == ErrorKind.UNKNOWN.value


async def test_hung_provider_is_cut_off_by_timeout(harness):
    h = harness(2, rotation={"request_timeout": 1.5})

    async def hang(*_args, **_kwargs):
        await asyncio.sleep(30)

    original = h.provider.complete

    async def complete(req, *, api_key, base_url):
        if api_key == secret_for(0):
            await hang()
        return await original(req, api_key=api_key, base_url=base_url)

    h.provider.complete = complete  # type: ignore[method-assign]
    route = await asyncio.wait_for(h.router.complete(request()), timeout=10)
    assert route.key_id == "k1"
    assert route.attempts[0].outcome == ErrorKind.TIMEOUT.value


async def test_fifteen_key_pool_survives_fourteen_dead_keys(harness):
    h = harness(15)
    for index in range(14):
        h.provider.always(index, err(ErrorKind.AUTH))
    route = await h.router.complete(request())
    assert route.key_id == "k14"
    assert len(route.attempts) == 15


async def test_rotation_moves_on_as_minute_quota_fills(harness):
    h = harness(3, limits={"rpm": 2})
    served = [(await h.router.complete(request())).key_id for _ in range(6)]
    assert served == ["k0", "k0", "k1", "k1", "k2", "k2"]
    with pytest.raises(AllKeysExhaustedError):  # recovery is ~60s away, beyond max_queue_wait
        await h.router.complete(request())
    h.clock.advance(61)
    assert (await h.router.complete(request())).key_id == "k0"


async def test_prompt_is_routed_to_a_key_whose_context_window_fits(harness):
    h = harness(keys=[
        {"id": "k0", "provider": "openai", "models": ["m"], "context_window": 500},
        {"id": "k1", "provider": "openai", "models": ["m"], "context_window": 100_000},
    ])
    big = request("word " * 2000)
    route = await h.router.complete(big)
    assert route.key_id == "k1" and len(h.provider.calls) == 1  # k0 never received the doomed request


async def test_prompt_too_large_for_every_key_reports_context_length(harness):
    h = harness(keys=[{"id": "k0", "provider": "openai", "models": ["m"], "context_window": 500}])
    with pytest.raises(ProviderError) as caught:
        await h.router.complete(request("word " * 2000))
    assert caught.value.kind is ErrorKind.CONTEXT_LENGTH
    assert not h.provider.calls


async def test_provider_context_error_tries_other_keys_without_penalty(harness):
    h = harness(2)
    h.provider.script(0, err(ErrorKind.CONTEXT_LENGTH, status=400))
    route = await h.router.complete(request())
    assert route.key_id == "k1"
    assert h.pool.status(h.key(0)) is KeyStatus.READY


async def test_max_output_tokens_cap_is_applied_per_key(harness):
    h = harness(keys=[{"id": "k0", "provider": "openai", "models": ["m"], "max_output_tokens": 256}])
    await h.router.complete(ChatRequest(model="auto", messages=[{"role": "user", "content": "hi"}], max_tokens=4000))
    assert h.provider.calls[0][1].max_tokens == 256


async def test_pinned_call_fails_fast_instead_of_waiting(harness):
    h = harness(2)
    h.provider.script(0, err(ErrorKind.RATE_LIMIT, retry_after=5))
    with pytest.raises(AllKeysExhaustedError):
        await h.router.complete(request(), pin="k0")
    assert not h.clock.slept
    assert len(h.provider.calls) == 1  # never leaked onto k1


async def test_no_keys_configured_gives_a_clear_message(harness):
    h = harness(0)
    with pytest.raises(AllKeysExhaustedError, match="No API keys are configured"):
        await h.router.complete(request())


# --- streaming --------------------------------------------------------------

async def collect(stream) -> str:
    return "".join([chunk.text async for chunk in stream])


async def test_stream_fails_over_before_first_chunk(harness):
    h = harness(2)
    h.provider.stream_scripts[secret_for(0)] = [[err(ErrorKind.RATE_LIMIT, retry_after=30)]]
    routes = []
    text = await collect(h.router.stream(request(), on_route=routes.append))
    assert text == "ok"
    assert routes[-1].key_id == "k1"
    assert h.key(1).day_requests == 1


async def test_stream_failure_after_content_is_reported_as_partial(harness):
    h = harness(2)
    h.provider.stream_scripts[secret_for(0)] = [[StreamChunk(text="half an ans"), err(ErrorKind.TRANSIENT)]]
    seen = []
    with pytest.raises(ProviderError) as caught:
        async for chunk in h.router.stream(request()):
            seen.append(chunk.text)
    assert seen == ["half an ans"] and caught.value.partial
    assert h.key(0).failures == 1
    assert h.key(0).day_requests == 1          # the partial call was still billed and is counted
    assert h.key(0).in_flight == 0             # reservation released


async def test_stream_records_usage_and_releases_reservation(harness):
    h = harness(1)
    h.provider.stream_scripts[secret_for(0)] = [[
        StreamChunk(text="a"), StreamChunk(text="b"), StreamChunk(finish_reason="stop", usage=Usage(40, 2)),
    ]]
    assert await collect(h.router.stream(request())) == "ab"
    assert h.key(0).day_tokens == 42 and h.key(0).in_flight == 0 and h.key(0).reserved_tokens == 0


async def test_stream_all_keys_dead_fails_closed(harness):
    h = harness(2)
    for index in range(2):
        h.provider.stream_scripts[secret_for(index)] = [[err(ErrorKind.AUTH)]]
    with pytest.raises(AllKeysExhaustedError):
        await collect(h.router.stream(request()))
