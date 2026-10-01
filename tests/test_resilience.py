"""Retired models, busy providers and the waiting that goes with them.

Each test mirrors something seen on a real Google Cloud Shell run: a pinned
`gemini-2.5-flash` answering 404 "no longer available", a second key answering 503 "high
demand", and Baton giving up with "all keys exhausted" after one second.
"""

from __future__ import annotations

import httpx
import pytest

from baton.cli.main import exhausted_hint
from baton.errors import AllKeysExhaustedError, ErrorKind, ProviderError
from baton.models import ModelCatalog, audit_pinned_models, hint_from_message, pick_best
from baton.pool import KeyStatus
from baton.providers.anthropic import AnthropicProvider
from baton.providers.gemini import GeminiProvider
from baton.providers.openai_compat import OpenAIProvider
from conftest import request, secret_for

RETIRED = ("This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use "
           "models/gemini-3.8-flash for the latest features and improvements.")
BUSY = "gemini returned HTTP 503: This model is currently experiencing high demand. Please try again later."


def gemini_keys(*models: str) -> list[dict]:
    return [{"id": f"k{i}", "provider": "gemini", "models": [model]} for i, model in enumerate(models)]


def retired_error(message: str = RETIRED) -> ProviderError:
    return ProviderError(ErrorKind.MODEL_NOT_FOUND, message, status=404)


def busy_error() -> ProviderError:
    return ProviderError(ErrorKind.OVERLOADED, BUSY, status=503)


def models_sent(h, index: int) -> list[str]:
    return [call.model for call in h.provider.calls_for(index)]


# --- ranking and hints: no model name is hardcoded --------------------------------------------

def test_pick_best_prefers_newest_stable_fast_model():
    listed = ["gemini-2.5-flash", "gemini-3.8-flash", "gemini-3.8-pro", "gemini-3.8-flash-lite",
              "gemini-3.9-flash-preview", "gemini-embedding-001", "imagen-4", "gemini-3.8-flash-image"]
    assert pick_best("gemini", listed) == "gemini-3.8-flash"           # not a preview, lite, pro, embedding or image


def test_pick_best_keeps_the_tier_of_the_model_being_replaced():
    listed = ["gemini-2.5-pro", "gemini-3.8-pro", "gemini-3.8-flash"]
    assert pick_best("gemini", listed, like="gemini-2.5-pro") == "gemini-3.8-pro"


def test_pick_best_for_the_other_providers_and_for_unknowable_ones():
    assert pick_best("openai", ["gpt-4o", "gpt-4o-mini", "gpt-5-mini", "gpt-5-codex", "text-embedding-3-small"]) == "gpt-5-mini"
    assert pick_best("anthropic", ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5-20251001"]) == "claude-sonnet-5-5"
    assert pick_best("openai_compatible", ["llama3:latest", "mistral"]) is None   # a server's list says nothing about quality
    assert pick_best("gemini", []) is None


def test_hint_must_be_confirmed_by_the_model_list():
    assert hint_from_message(RETIRED, "gemini-2.5-flash", ["gemini-3.8-flash", "gemini-2.5-pro"]) == "gemini-3.8-flash"
    assert hint_from_message(RETIRED, "gemini-2.5-flash", ["gemini-2.5-pro"]) is None     # named, but not listed: ignored
    assert hint_from_message(RETIRED, "gemini-3.8-flash", ["gemini-3.8-flash"]) is None   # never "replace" a model by itself


# --- list endpoints (mocked transports) ---------------------------------------------------------

def client_for(handler):
    seen: list[httpx.Request] = []

    def record(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return handler(req)

    return httpx.AsyncClient(transport=httpx.MockTransport(record)), seen


async def test_gemini_list_follows_pages_and_keeps_only_generate_content_models():
    def handler(req: httpx.Request) -> httpx.Response:
        if "pageToken=p2" in str(req.url):
            return httpx.Response(200, json={"models": [
                {"name": "models/gemini-3.8-flash", "supportedGenerationMethods": ["generateContent"]}]})
        return httpx.Response(200, json={"nextPageToken": "p2", "models": [
            {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
            {"name": "models/gemini-3.8-pro", "supportedGenerationMethods": ["generateContent", "countTokens"]}]})

    client, seen = client_for(handler)
    models = await GeminiProvider(client).list_models(api_key="secret-value-123", base_url="https://g.test/v1beta")
    assert models == ["gemini-3.8-pro", "gemini-3.8-flash"]
    assert len(seen) == 2 and all("secret-value-123" not in str(r.url) for r in seen)     # the key travels in a header
    assert seen[0].headers["x-goog-api-key"] == "secret-value-123"


async def test_openai_and_anthropic_lists_are_parsed():
    client, seen = client_for(lambda req: httpx.Response(200, json={"data": [{"id": "gpt-5-mini"}, {"id": "gpt-5"}]}))
    assert await OpenAIProvider(client).list_models(api_key="k" * 12, base_url="https://o.test/v1") == ["gpt-5-mini", "gpt-5"]
    assert str(seen[0].url) == "https://o.test/v1/models"
    client, seen = client_for(lambda req: httpx.Response(200, json={"data": [{"id": "claude-sonnet-5-5"}]}))
    assert await AnthropicProvider(client).list_models(api_key="k" * 12, base_url="https://a.test") == ["claude-sonnet-5-5"]
    assert str(seen[0].url).startswith("https://a.test/v1/models")


async def test_list_endpoint_errors_are_classified_not_swallowed_by_the_adapter():
    client, _ = client_for(lambda req: httpx.Response(403, json={"error": {"message": "denied"}}))
    with pytest.raises(ProviderError) as caught:
        await GeminiProvider(client).list_models(api_key="k" * 12, base_url="https://g.test/v1beta")
    assert caught.value.kind is ErrorKind.AUTH


# --- the catalog: cache, expiry, failure -----------------------------------------------------------

async def test_catalog_caches_expires_and_survives_failures(harness, tmp_path):
    h = harness(keys=gemini_keys("auto"))
    h.provider.model_lists[secret_for(0)] = ["gemini-3.8-flash"]
    catalog = ModelCatalog(h.router.providers, clock=h.clock, cache_path=tmp_path / "models-cache.json", ttl=100)
    key = h.key(0)

    assert await catalog.available(key) == ["gemini-3.8-flash"]
    assert await catalog.available(key) == ["gemini-3.8-flash"]
    assert h.provider.list_calls == 1                                   # cached

    h.clock.advance(101)
    h.provider.model_lists[secret_for(0)] = ["gemini-4-flash"]
    assert await catalog.available(key) == ["gemini-4-flash"]           # expired: asked again
    assert h.provider.list_calls == 2

    h.clock.advance(101)
    h.provider.model_lists[secret_for(0)] = ProviderError(ErrorKind.TRANSIENT, "down")
    assert await catalog.available(key) == ["gemini-4-flash"]           # list endpoint down: stale list beats guessing
    assert await catalog.available(key) == ["gemini-4-flash"]
    assert h.provider.list_calls == 3                                   # and it is not hammered while down

    reloaded = ModelCatalog(h.router.providers, clock=h.clock, cache_path=tmp_path / "models-cache.json", ttl=100)
    assert reloaded.cached_models("k0") == ["gemini-4-flash"]           # survives a restart


async def test_a_list_for_another_endpoint_is_not_reused(harness, tmp_path):
    h = harness(keys=gemini_keys("auto"))
    h.provider.model_lists[secret_for(0)] = ["gemini-3.8-flash"]
    first = ModelCatalog(h.router.providers, clock=h.clock, cache_path=tmp_path / "c.json")
    await first.available(h.key(0))
    h.key(0).config.__dict__["base_url"] = "https://elsewhere.example/v1beta"
    h.provider.model_lists[secret_for(0)] = ProviderError(ErrorKind.TRANSIENT, "down")
    second = ModelCatalog(h.router.providers, clock=h.clock, cache_path=tmp_path / "c.json")
    assert await second.available(h.key(0)) is None


# --- "auto" resolves from the live list --------------------------------------------------------------

async def test_auto_resolves_dynamically_from_the_provider_list(harness):
    h = harness(keys=gemini_keys("auto"))
    h.provider.model_lists[secret_for(0)] = ["gemini-2.5-flash", "gemini-3.8-flash", "gemini-3.8-pro"]
    await h.router.complete(request())
    await h.router.complete(request())
    assert models_sent(h, 0) == ["gemini-3.8-flash", "gemini-3.8-flash"]
    assert h.provider.list_calls == 1                                   # one list call, then cached


async def test_auto_falls_back_to_the_static_default_only_when_no_list_exists(harness):
    h = harness(keys=gemini_keys("auto"))
    await h.router.complete(request())                                  # FakeProvider has no list endpoint
    from baton.providers import PROVIDER_DEFAULTS

    assert models_sent(h, 0) == [PROVIDER_DEFAULTS["gemini"].model]


async def test_a_specific_model_found_in_the_list_is_honoured_for_auto_keys(harness):
    h = harness(keys=gemini_keys("auto"))
    h.provider.model_lists[secret_for(0)] = ["gemini-3.8-flash", "gemini-3.8-pro"]
    await h.router.complete(request())                                  # learns the list
    from baton.providers.base import ChatRequest

    await h.router.complete(ChatRequest(model="gemini-3.8-pro", messages=request().messages))
    assert models_sent(h, 0)[-1] == "gemini-3.8-pro"


# --- retired model: the key is fine, the model is wrong -----------------------------------------------

async def test_retired_model_is_swapped_once_without_penalising_the_key(harness):
    h = harness(keys=gemini_keys("gemini-2.5-flash"))
    h.provider.model_lists[secret_for(0)] = ["gemini-3.8-flash", "gemini-3.8-pro"]
    h.provider.script(0, retired_error())
    notices: list[str] = []
    h.router.on_notice = notices.append

    route = await h.router.complete(request())
    assert route.result.content == "ok" and route.model == "gemini-3.8-flash"
    assert models_sent(h, 0) == ["gemini-2.5-flash", "gemini-3.8-flash"]    # retried on the SAME key
    assert h.pool.status(h.key(0)) is KeyStatus.READY and h.key(0).failures == 0   # never marked unhealthy
    assert any("gemini-2.5-flash" in n and "gemini-3.8-flash" in n for n in notices)   # the user is told what was swapped

    await h.router.complete(request())                                  # remembered: no second 404
    assert models_sent(h, 0)[2:] == ["gemini-3.8-flash"]
    assert len(notices) == 1


async def test_replacement_without_a_hint_is_the_best_listed_model_of_the_same_tier(harness):
    h = harness(keys=gemini_keys("gemini-2.5-pro"))
    h.provider.model_lists[secret_for(0)] = ["gemini-3.8-flash", "gemini-3.8-pro"]
    h.provider.script(0, retired_error("models/gemini-2.5-pro is not found for API version v1beta"))
    route = await h.router.complete(request())
    assert route.model == "gemini-3.8-pro"


async def test_a_hint_the_list_does_not_confirm_is_not_trusted(harness):
    h = harness(keys=gemini_keys("gemini-2.5-flash"))
    h.provider.model_lists[secret_for(0)] = ["gemini-3.8-pro"]          # the named replacement is not offered to this key
    h.provider.script(0, retired_error())
    route = await h.router.complete(request())
    assert route.model == "gemini-3.8-pro"                              # chose from the list instead of the unverified hint


async def test_unrepairable_model_error_fails_once_and_says_so_without_blaming_the_key(harness):
    h = harness(keys=gemini_keys("gemini-2.5-flash"))                   # no list endpoint, so no safe replacement
    h.provider.always(0, retired_error())
    with pytest.raises(AllKeysExhaustedError) as caught:
        await h.router.complete(request())
    assert caught.value.kinds == {"k0": "model_not_found"}
    assert "model" in str(caught.value).lower() and "exhausted" not in str(caught.value)
    assert len(h.provider.calls) == 1                                   # no retry loop on a hopeless model
    assert h.pool.status(h.key(0)) is KeyStatus.READY                   # the key stays in rotation


async def test_retired_model_recovery_also_works_for_streams(harness):
    h = harness(keys=gemini_keys("gemini-2.5-flash"))
    h.provider.model_lists[secret_for(0)] = ["gemini-3.8-flash"]
    h.provider.stream_scripts[secret_for(0)] = [[retired_error()]]
    notices: list[str] = []
    h.router.on_notice = notices.append
    chunks = [chunk async for chunk in h.router.stream(request())]
    assert "".join(c.text for c in chunks) == "ok"
    assert models_sent(h, 0) == ["gemini-2.5-flash", "gemini-3.8-flash"] and notices


async def test_audit_flags_pinned_models_the_provider_no_longer_lists(harness):
    h = harness(keys=gemini_keys("gemini-2.5-flash", "auto", "gemini-3.8-flash"))
    for index in range(3):
        h.provider.model_lists[secret_for(index)] = ["gemini-3.8-flash", "gemini-3.8-pro"]
    stale = await audit_pinned_models(h.pool, h.router.catalog)
    assert [(s.key_id, s.model, s.suggestion) for s in stale] == [("k0", "gemini-2.5-flash", "gemini-3.8-flash")]


async def test_audit_ignores_servers_whose_naming_it_cannot_judge(harness):
    h = harness(keys=[{"id": "k0", "provider": "openai_compatible", "base_url": "http://127.0.0.1:11434/v1",
                       "models": ["llama3"]}])
    h.provider.model_lists[secret_for(0)] = ["llama3:latest"]
    assert await audit_pinned_models(h.pool, h.router.catalog) == []


# --- busy providers: back off, wait, then succeed or fail honestly -------------------------------------

async def test_503_backs_off_with_growing_waits_then_succeeds(harness):
    h = harness(1, rotation={"backoff_base": 1.0})
    h.provider.script(0, busy_error(), busy_error())
    notices: list[str] = []
    h.router.on_notice = notices.append
    route = await h.router.complete(request())
    assert route.result.content == "ok"
    assert len(h.clock.slept) == 2 and h.clock.slept[1] > h.clock.slept[0]      # exponential, not a fixed poll
    assert sum(1 for n in notices if n.startswith("provider busy, retrying in")) == 2
    assert h.key(0).last_kind == "" and h.pool.status(h.key(0)) is KeyStatus.READY   # success wipes the failure


async def test_overload_is_a_short_cooldown_not_quota_or_a_circuit_breaker(harness):
    h = harness(1)
    h.pool.record_failure(h.key(0), busy_error())
    key = h.key(0)
    assert h.pool.status(key) is KeyStatus.COOLDOWN
    assert key.exhausted_until == 0 and key.disabled_until == 0
    assert 0 < key.cooldown_until - h.clock() < 10                       # seconds, not the breaker's minute-scale parking


async def test_every_key_busy_then_waits_and_succeeds(harness):
    h = harness(2)
    h.provider.script(0, busy_error())
    h.provider.script(1, busy_error())
    route = await h.router.complete(request())
    assert route.result.content == "ok"
    assert h.clock.slept                                                 # it waited rather than giving up


async def test_a_long_provider_hiccup_is_waited_out_within_the_overload_budget(harness):
    h = harness(1, rotation={"backoff_base": 4.0, "overload_max_wait": 120, "max_queue_wait": 1})
    h.provider.script(0, *[busy_error() for _ in range(4)])
    route = await h.router.complete(request())
    assert route.result.content == "ok"
    assert sum(h.clock.slept) > 1                                        # well past max_queue_wait: the overload budget applies


async def test_waiting_stops_at_the_cap_and_blames_the_provider_not_the_keys(harness):
    h = harness(2, rotation={"overload_max_wait": 12, "backoff_base": 2.0})
    for index in range(2):
        h.provider.always(index, busy_error())
    with pytest.raises(AllKeysExhaustedError) as caught:
        await h.router.complete(request())
    error = caught.value
    message = str(error)
    assert "overloaded" in message and "exhausted" not in message and "Your keys are fine" in message
    assert error.kinds == {"k0": "overloaded", "k1": "overloaded"}
    assert 0 < sum(h.clock.slept) <= 12 and error.waited <= 12            # stopped at the cap
    assert len(h.provider.calls) < 60                                    # and did not spin


async def test_overload_cap_zero_fails_fast(harness):
    h = harness(1, rotation={"overload_max_wait": 0})
    h.provider.always(0, busy_error())
    with pytest.raises(AllKeysExhaustedError):
        await h.router.complete(request())
    assert not h.clock.slept


async def test_stream_waits_out_a_busy_provider_before_the_first_chunk(harness):
    h = harness(1)
    h.provider.stream_scripts[secret_for(0)] = [[busy_error()], [busy_error()]]
    chunks = [chunk async for chunk in h.router.stream(request())]
    assert "".join(c.text for c in chunks) == "ok" and len(h.clock.slept) == 2


# --- the final message reflects the real mix of causes ----------------------------------------------------

async def test_mixed_causes_are_reported_per_key_and_the_advice_matches(harness):
    h = harness(keys=gemini_keys("gemini-2.5-flash", "gemini-3.8-flash"), rotation={"overload_max_wait": 6})
    h.provider.always(0, retired_error())                                # k0: wrong model, unrepairable (no list)
    h.provider.always(1, busy_error())                                   # k1: provider overloaded
    with pytest.raises(AllKeysExhaustedError) as caught:
        await h.router.complete(request())
    error = caught.value
    assert error.kinds == {"k0": "model_not_found", "k1": "overloaded"}
    assert "k0: model not found or retired" in str(error) and "k1: provider overloaded" in str(error)
    hint = exhausted_hint(error.kinds)
    assert "/models --fix" in hint and "provider is busy" in hint and "rejected" not in hint


def test_advice_only_mentions_causes_that_actually_occurred():
    assert "model" not in exhausted_hint({"k0": "overloaded", "k1": "overloaded"}).lower()
    assert "busy" not in exhausted_hint({"k0": "model_not_found"})
    assert "rejected" in exhausted_hint({"k0": "auth", "k1": "overloaded"})
    assert "quota" in exhausted_hint({"k0": "quota_exhausted"})
