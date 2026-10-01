"""Provider adapters: wire-format translation and error classification.

Upstreams are simulated with httpx.MockTransport; no network is used.
"""

from __future__ import annotations

import json

import httpx
import pytest

from baton.errors import ErrorKind, ProviderError
from baton.providers import build_providers
from baton.providers.anthropic import to_anthropic_messages
from baton.providers.base import (
    ChatRequest,
    classify_http_error,
    estimate_message_tokens,
    parse_duration,
    parse_rate_headers,
)
from baton.providers.gemini import sanitize_schema, to_gemini_contents

KEY = "unit-test-credential-000111"
TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a file",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"],
                   "additionalProperties": False},
}}]
CONVERSATION = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "read a.py"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}},
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": "print('hi')"},
]


def sse(*events: dict | str) -> bytes:
    lines = []
    for event in events:
        lines.append("data: " + (event if isinstance(event, str) else json.dumps(event)) + "\n\n")
    return "".join(lines).encode()


def provider(name: str, handler):
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(wrapped))
    return build_providers(client)[name], seen


async def collect(stream):
    return [chunk async for chunk in stream]


# --- OpenAI / compatible ----------------------------------------------------

async def test_openai_request_and_response():
    def handler(request):
        return httpx.Response(200, headers={"x-ratelimit-limit-requests": "100", "x-ratelimit-remaining-requests": "10"}, json={
            "model": "gpt-x", "choices": [{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        })

    adapter, seen = provider("openai", handler)
    messages = [*CONVERSATION, {"role": "user", "content": "x", "_baton_private": 1}]
    result = await adapter.complete(
        ChatRequest(model="gpt-x", messages=messages, tools=TOOLS, max_tokens=50, extra={"seed": 1, "evil": 2}),
        api_key=KEY, base_url="https://api.openai.com/v1",
    )
    sent = json.loads(seen[0].content)
    assert str(seen[0].url) == "https://api.openai.com/v1/chat/completions"
    assert seen[0].headers["authorization"] == f"Bearer {KEY}" and KEY not in str(seen[0].url)
    assert sent["max_completion_tokens"] == 50 and "max_tokens" not in sent
    assert sent["seed"] == 1 and "evil" not in sent
    assert "_baton_private" not in sent["messages"][-1]
    assert result.content == "done" and result.usage.total_tokens == 15
    assert result.rate.utilization() == pytest.approx(0.9)


async def test_compatible_provider_uses_legacy_max_tokens_and_estimates_missing_usage():
    adapter, seen = provider("openai_compatible", lambda r: httpx.Response(200, json={
        "choices": [{"message": {"content": "hello there"}, "finish_reason": "stop"}],
    }))
    result = await adapter.complete(ChatRequest(model="llama", messages=CONVERSATION[:2], max_tokens=9),
                                    api_key=KEY, base_url="http://localhost:11434/v1/")
    assert json.loads(seen[0].content)["max_tokens"] == 9
    assert result.usage_estimated and result.usage.prompt_tokens > 0


async def test_openai_stream_and_stream_options_retry():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if "stream_options" in json.loads(request.content):
            return httpx.Response(400, json={"error": {"message": "unknown field stream_options"}})
        return httpx.Response(200, content=sse(
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 2}},
            "[DONE]",
        ))

    adapter, _ = provider("openai_compatible", handler)
    chunks = await collect(adapter.stream(ChatRequest(model="m", messages=CONVERSATION[:2]), api_key=KEY, base_url="https://x.test/v1"))
    assert "".join(c.text for c in chunks) == "Hello" and calls["n"] == 2
    assert chunks[-1].usage.total_tokens == 6


# --- Anthropic --------------------------------------------------------------

def test_anthropic_message_translation():
    system, messages = to_anthropic_messages(CONVERSATION)
    assert system == "Be brief."
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"][0] == {"type": "tool_use", "id": "call_1", "name": "read_file", "input": {"path": "a.py"}}
    assert messages[2]["content"][0]["type"] == "tool_result" and messages[2]["content"][0]["tool_use_id"] == "call_1"
    # A history that starts with an assistant turn gets a user turn prepended.
    _, fixed = to_anthropic_messages([{"role": "assistant", "content": "hi"}])
    assert fixed[0]["role"] == "user"
    # Parallel tool results merge into one user turn, results first.
    _, merged = to_anthropic_messages([
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "function": {"name": "f", "arguments": "{}"}}, {"id": "b", "function": {"name": "f", "arguments": "not json"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "1"},
        {"role": "tool", "tool_call_id": "b", "content": ""},
        {"role": "user", "content": "thanks"},
    ])
    assert [block["type"] for block in merged[2]["content"]] == ["tool_result", "tool_result", "text"]
    assert merged[1]["content"][1]["input"] == {}


async def test_anthropic_request_and_response_drop_thinking_and_unsupported_params():
    def handler(request):
        return httpx.Response(200, headers={"anthropic-ratelimit-requests-limit": "50", "anthropic-ratelimit-requests-remaining": "25"}, json={
            "model": "claude-opus-5-5", "stop_reason": "tool_use",
            "content": [
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {"type": "text", "text": "Reading."},
                {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.py"}},
            ],
            "usage": {"input_tokens": 20, "cache_read_input_tokens": 5, "output_tokens": 7},
        })

    adapter, seen = provider("anthropic", handler)
    request = ChatRequest(model="claude-opus-5-5", messages=CONVERSATION, tools=TOOLS, temperature=0.3, tool_choice="required")
    result = await adapter.complete(request, api_key=KEY, base_url="https://api.anthropic.com")
    sent = json.loads(seen[0].content)
    assert seen[0].headers["x-api-key"] == KEY and seen[0].headers["anthropic-version"] == "2023-06-01"
    assert "temperature" not in sent and "tool_choice" not in sent      # current models reject both
    assert sent["max_tokens"] > 0 and sent["system"] == "Be brief."
    assert sent["tools"][0]["input_schema"]["required"] == ["path"]
    assert result.content == "Reading." and result.finish_reason == "tool_calls"
    assert result.tool_calls == [{"id": "toolu_1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}]
    assert result.usage.prompt_tokens == 25 and result.rate.utilization() == pytest.approx(0.5)

    legacy = ChatRequest(model="claude-haiku-4-5", messages=CONVERSATION, tools=TOOLS, temperature=0.3, tool_choice="required")
    await adapter.complete(legacy, api_key=KEY, base_url="https://api.anthropic.com")
    sent = json.loads(seen[1].content)
    assert sent["temperature"] == 0.3 and sent["tool_choice"] == {"type": "any"}


async def test_anthropic_refusal_maps_to_content_filter():
    adapter, _ = provider("anthropic", lambda r: httpx.Response(200, json={"stop_reason": "refusal", "content": [], "usage": {}}))
    result = await adapter.complete(ChatRequest(model="claude-opus-5-5", messages=CONVERSATION[:2]), api_key=KEY, base_url="https://a.test")
    assert result.finish_reason == "content_filter" and result.content is None


async def test_anthropic_stream_translation_and_in_band_error():
    events = sse(
        {"type": "message_start", "message": {"usage": {"input_tokens": 11}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Hi"}},
        {"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "toolu_9", "name": "read_file"}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '{"path":'}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '"a.py"}'}},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}},
        {"type": "message_stop"},
    )
    adapter, _ = provider("anthropic", lambda r: httpx.Response(200, content=events))
    chunks = await collect(adapter.stream(ChatRequest(model="claude-opus-5-5", messages=CONVERSATION[:2]), api_key=KEY, base_url="https://a.test"))
    assert "".join(c.text for c in chunks) == "Hi"                       # thinking text never surfaces
    deltas = [call for c in chunks for call in c.tool_calls]
    assert deltas[0]["index"] == 0 and deltas[0]["id"] == "toolu_9"
    assert "".join(d["function"].get("arguments", "") for d in deltas) == '{"path":"a.py"}'
    assert chunks[-1].finish_reason == "tool_calls" and chunks[-1].usage.total_tokens == 20

    overloaded = sse({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
    adapter, _ = provider("anthropic", lambda r: httpx.Response(200, content=overloaded))
    with pytest.raises(ProviderError) as caught:
        await collect(adapter.stream(ChatRequest(model="claude-opus-5-5", messages=CONVERSATION[:2]), api_key=KEY, base_url="https://a.test"))
    assert caught.value.kind is ErrorKind.OVERLOADED          # a busy provider, not a failing key


# --- Gemini -----------------------------------------------------------------

def test_gemini_translation_and_schema_sanitising():
    system, contents = to_gemini_contents(CONVERSATION, "gemini-2.5-flash")
    assert system == "Be brief." and [c["role"] for c in contents] == ["user", "model", "user"]
    assert contents[1]["parts"][0] == {"functionCall": {"name": "read_file", "args": {"path": "a.py"}}}
    assert contents[2]["parts"][0]["functionResponse"]["name"] == "read_file"      # resolved from the call id
    # Newer models validate thought signatures: keep a real one, placeholder otherwise.
    _, strict = to_gemini_contents(CONVERSATION, "gemini-3-pro")
    assert "thoughtSignature" in strict[1]["parts"][0]
    signed = json.loads(json.dumps(CONVERSATION))
    signed[2]["tool_calls"][0]["extra_content"] = {"google": {"thought_signature": "real-sig"}}
    _, kept = to_gemini_contents(signed, "gemini-3-pro")
    assert kept[1]["parts"][0]["thoughtSignature"] == "real-sig"

    cleaned = sanitize_schema({"type": "object", "additionalProperties": False, "$schema": "x",
                               "properties": {"a": {"type": ["string", "null"], "default": "x"}}})
    assert cleaned == {"type": "object", "properties": {"a": {"type": "string", "nullable": True}}}


async def test_gemini_request_response_and_key_stays_out_of_the_url():
    def handler(request):
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [
                {"text": "thinking...", "thought": True}, {"text": "Reading."},
                {"functionCall": {"name": "read_file", "args": {"path": "a.py"}}, "thoughtSignature": "sig-1"},
            ]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 4, "thoughtsTokenCount": 6},
        })

    adapter, seen = provider("gemini", handler)
    request = ChatRequest(model="gemini-2.5-flash", messages=CONVERSATION, tools=TOOLS, max_tokens=64,
                          response_format={"type": "json_object"}, tool_choice="required")
    result = await adapter.complete(request, api_key=KEY, base_url="https://generativelanguage.googleapis.com/v1beta")
    url = str(seen[0].url)
    assert url.endswith("/v1beta/models/gemini-2.5-flash:generateContent") and KEY not in url
    assert seen[0].headers["x-goog-api-key"] == KEY
    sent = json.loads(seen[0].content)
    assert "additionalProperties" not in json.dumps(sent["tools"])
    assert sent["generationConfig"] == {"maxOutputTokens": 64, "responseMimeType": "application/json"}
    assert sent["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"
    assert result.content == "Reading." and result.finish_reason == "tool_calls"
    assert result.tool_calls[0]["extra_content"]["google"]["thought_signature"] == "sig-1"
    assert result.usage.prompt_tokens == 30 and result.usage.completion_tokens == 10


async def test_gemini_model_name_cannot_escape_the_path():
    adapter, seen = provider("gemini", lambda r: httpx.Response(200, json={"candidates": []}))
    await adapter.complete(ChatRequest(model="../../v1/secrets?x=", messages=CONVERSATION[:2]), api_key=KEY, base_url="https://g.test/v1beta")
    assert "/v1beta/models/....v1secretsx:generateContent" in str(seen[0].url)


async def test_gemini_blocked_prompt_and_stream():
    adapter, _ = provider("gemini", lambda r: httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}}))
    blocked = await adapter.complete(ChatRequest(model="gemini-2.5-flash", messages=CONVERSATION[:2]), api_key=KEY, base_url="https://g.test/v1beta")
    assert blocked.finish_reason == "content_filter"

    events = sse(
        {"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]},
        {"candidates": [{"content": {"parts": [{"text": "lo"}, {"functionCall": {"name": "f", "args": {}}}]}, "finishReason": "STOP"}],
         "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2}},
    )
    adapter, seen = provider("gemini", lambda r: httpx.Response(200, content=events))
    chunks = await collect(adapter.stream(ChatRequest(model="gemini-2.5-flash", messages=CONVERSATION[:2]), api_key=KEY, base_url="https://g.test/v1beta"))
    assert "alt=sse" in str(seen[0].url) and "".join(c.text for c in chunks) == "Hello"
    assert chunks[-1].finish_reason == "tool_calls" and chunks[-1].tool_calls[0]["index"] == 0
    assert chunks[-1].usage.total_tokens == 5


# --- error classification ---------------------------------------------------

@pytest.mark.parametrize(("status", "body", "expected"), [
    (401, '{"error": {"message": "bad key"}}', ErrorKind.AUTH),
    (403, "{}", ErrorKind.AUTH),
    (402, "{}", ErrorKind.QUOTA_EXHAUSTED),
    (429, '{"error": {"message": "Rate limit reached", "code": "rate_limit_exceeded"}}', ErrorKind.RATE_LIMIT),
    (429, '{"error": {"message": "You exceeded your current quota", "code": "insufficient_quota"}}', ErrorKind.QUOTA_EXHAUSTED),
    (429, '{"error": {"status": "RESOURCE_EXHAUSTED", "details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}}', ErrorKind.QUOTA_EXHAUSTED),
    (429, '{"error": {"status": "RESOURCE_EXHAUSTED", "details": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel"}]}}', ErrorKind.RATE_LIMIT),
    (500, "oops", ErrorKind.TRANSIENT),
    (529, '{"error": {"type": "overloaded_error"}}', ErrorKind.OVERLOADED),
    # The two Google errors seen on a real Cloud Shell run:
    (503, '{"error": {"code": 503, "message": "This model is currently experiencing high demand. Spikes in demand '
          'are usually temporary. Please try again later.", "status": "UNAVAILABLE"}}', ErrorKind.OVERLOADED),
    (404, '{"error": {"code": 404, "message": "This model models/gemini-2.5-flash is no longer available to new '
          'users. Please update your code to use models/gemini-3.8-flash for the latest features.", '
          '"status": "NOT_FOUND"}}', ErrorKind.MODEL_NOT_FOUND),
    (503, "", ErrorKind.OVERLOADED),                                        # by status alone, with no message at all
    (502, '{"error": {"message": "backend is overloaded, try again later"}}', ErrorKind.OVERLOADED),
    (502, "bad gateway", ErrorKind.TRANSIENT),                              # a plain gateway error is not "busy"
    (500, '{"error": {"message": "internal error"}}', ErrorKind.TRANSIENT),
    (408, "", ErrorKind.TRANSIENT),
    (404, '{"error": {"message": "model not found"}}', ErrorKind.MODEL_NOT_FOUND),
    (404, '{"error": {"message": "no such route"}}', ErrorKind.UNKNOWN),                 # a 404 that is not about a model
    (400, '{"error": {"message": "x", "code": "model_not_found"}}', ErrorKind.MODEL_NOT_FOUND),
    (400, '{"error": {"message": "This model\'s maximum context length is 8192 tokens"}}', ErrorKind.CONTEXT_LENGTH),
    (413, "", ErrorKind.CONTEXT_LENGTH),
    (400, '{"error": {"message": "API key not valid. Please pass a valid API key.", "status": "INVALID_ARGUMENT"}}', ErrorKind.AUTH),
    (400, '{"error": {"message": "invalid tool schema"}}', ErrorKind.BAD_REQUEST),
    (418, "", ErrorKind.UNKNOWN),
])
def test_http_errors_are_classified(status, body, expected):
    assert classify_http_error("p", status, body, httpx.Headers()).kind is expected


def test_retry_after_is_read_from_header_or_gemini_body():
    assert classify_http_error("p", 429, "{}", httpx.Headers({"retry-after": "17"})).retry_after == 17
    body = '{"error": {"details": [{"@type": "RetryInfo", "retryDelay": "34s"}]}}'
    assert classify_http_error("p", 429, body, httpx.Headers()).retry_after == 34
    assert classify_http_error("p", 429, "{}", httpx.Headers({"x-ratelimit-reset-requests": "6m0s"})).retry_after == 360


def test_error_messages_are_scrubbed():
    leaked = "sk-" + "Ab1" * 12
    error = classify_http_error("p", 401, json.dumps({"error": {"message": f"Incorrect API key provided: {leaked}"}}), httpx.Headers())
    assert leaked not in error.message and "HTTP 401" in error.message


async def test_transport_failures_are_classified_without_leaking_urls():
    def timeout(request):
        raise httpx.ReadTimeout("timed out", request=request)

    def refused(request):
        raise httpx.ConnectError("connection refused", request=request)

    for handler, kind in ((timeout, ErrorKind.TIMEOUT), (refused, ErrorKind.TRANSIENT)):
        adapter, _ = provider("openai", handler)
        with pytest.raises(ProviderError) as caught:
            await adapter.complete(ChatRequest(model="m", messages=CONVERSATION[:2]), api_key=KEY, base_url="https://x.test/v1")
        assert caught.value.kind is kind and KEY not in caught.value.message and "x.test" not in caught.value.message
        with pytest.raises(ProviderError):
            await collect(adapter.stream(ChatRequest(model="m", messages=CONVERSATION[:2]), api_key=KEY, base_url="https://x.test/v1"))


async def test_malformed_upstream_json_is_a_transient_error():
    adapter, _ = provider("openai", lambda r: httpx.Response(200, content=b"<html>gateway</html>"))
    with pytest.raises(ProviderError) as caught:
        await adapter.complete(ChatRequest(model="m", messages=CONVERSATION[:2]), api_key=KEY, base_url="https://x.test/v1")
    assert caught.value.kind is ErrorKind.TRANSIENT


# --- helpers ----------------------------------------------------------------

def test_duration_and_rate_header_parsing():
    assert parse_duration("12") == 12 and parse_duration("1.5s") == 1.5 and parse_duration("200ms") == 0.2
    assert parse_duration("1h2m3s") == 3723 and parse_duration("garbage") is None and parse_duration(None) is None
    info = parse_rate_headers(httpx.Headers({
        "anthropic-ratelimit-tokens-limit": "1000", "anthropic-ratelimit-tokens-remaining": "100",
        "anthropic-ratelimit-input-tokens-remaining": "5", "x-ratelimit-remaining-requests": "oops",
    }))
    assert info.limit_tokens == 1000 and info.remaining_tokens == 100 and info.utilization() == pytest.approx(0.9)
    assert parse_rate_headers(httpx.Headers({"content-type": "x"})) is None


def test_token_estimate_counts_tools_images_and_calls():
    base = estimate_message_tokens([{"role": "user", "content": "hello world"}])
    with_image = estimate_message_tokens([{"role": "user", "content": [
        {"type": "text", "text": "hello world"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}])
    assert with_image > base + 500
    assert estimate_message_tokens(CONVERSATION, TOOLS) > estimate_message_tokens(CONVERSATION)
