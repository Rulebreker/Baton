"""OpenAI and OpenAI-compatible upstreams (Groq, OpenRouter, Mistral, Ollama, ...)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..errors import ErrorKind, ProviderError
from .base import (
    ChatRequest,
    ChatResult,
    Provider,
    StreamChunk,
    Usage,
    classify_transport_error,
    estimate_message_tokens,
    estimate_tokens,
    iter_sse,
    loads_or_error,
    parse_rate_headers,
    raise_for_status,
)

# Parameters we are willing to pass through untouched. An allow-list (rather
# than forwarding everything) stops a proxy client from smuggling fields such
# as `user`-supplied base URLs or provider-specific billing switches upstream.
PASSTHROUGH = {
    "seed", "presence_penalty", "frequency_penalty", "logit_bias", "logprobs", "top_logprobs",
    "parallel_tool_calls", "reasoning_effort", "n", "user",
}

# Private fields Baton attaches to messages; never sent upstream.
_PRIVATE_PREFIX = "_baton"


def clean_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in message.items() if not k.startswith(_PRIVATE_PREFIX)} for message in messages]


def _usage(data: dict[str, Any] | None) -> Usage | None:
    if not isinstance(data, dict):
        return None
    return Usage(int(data.get("prompt_tokens") or 0), int(data.get("completion_tokens") or 0))


class OpenAICompatProvider(Provider):
    name = "openai_compatible"
    # The official API deprecated `max_tokens` in favour of
    # `max_completion_tokens` (reasoning models reject the old name); most
    # compatible servers still only know the old one.
    max_tokens_field = "max_tokens"

    def _headers(self, api_key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def _payload(self, request: ChatRequest, *, stream: bool, include_usage: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": request.model, "messages": clean_messages(request.messages)}
        if request.tools:
            payload["tools"] = request.tools
            if request.tool_choice is not None:
                payload["tool_choice"] = request.tool_choice
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_tokens is not None:
            payload[self.max_tokens_field] = request.max_tokens
        if request.stop:
            payload["stop"] = request.stop
        if request.response_format:
            payload["response_format"] = request.response_format
        for key, value in request.extra.items():
            if key in PASSTHROUGH and value is not None:
                payload[key] = value
        if stream:
            payload["stream"] = True
            if include_usage:
                payload["stream_options"] = {"include_usage": True}
        return payload

    async def complete(self, request: ChatRequest, *, api_key: str, base_url: str) -> ChatResult:
        url = base_url.rstrip("/") + "/chat/completions"
        response = await self._post_json(url, self._headers(api_key), self._payload(request, stream=False))
        data = loads_or_error(self.name, response.text)
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(ErrorKind.TRANSIENT, f"{self.name} returned no choices")
        message = choices[0].get("message") or {}
        usage = _usage(data.get("usage"))
        estimated = usage is None
        if usage is None:
            usage = Usage(
                estimate_message_tokens(request.messages, request.tools),
                estimate_tokens(str(message.get("content") or "")),
            )
        return ChatResult(
            content=message.get("content"),
            tool_calls=list(message.get("tool_calls") or []),
            finish_reason=choices[0].get("finish_reason") or "stop",
            usage=usage,
            model=str(data.get("model") or request.model),
            rate=parse_rate_headers(response.headers),
            usage_estimated=estimated,
        )

    async def stream(self, request: ChatRequest, *, api_key: str, base_url: str) -> AsyncIterator[StreamChunk]:
        url = base_url.rstrip("/") + "/chat/completions"
        include_usage = True
        while True:
            payload = self._payload(request, stream=True, include_usage=include_usage)
            try:
                async with self.client.stream("POST", url, headers=self._headers(api_key), json=payload) as response:
                    try:
                        await raise_for_status(self.name, response)
                    except ProviderError as exc:
                        # Some compatible servers reject `stream_options`.
                        # Retry once without it instead of failing the key.
                        if include_usage and exc.kind is ErrorKind.BAD_REQUEST and "stream_options" in exc.message:
                            include_usage = False
                            continue
                        raise
                    rate = parse_rate_headers(response.headers)
                    first = True
                    async for _event, data in iter_sse(response):
                        if data.strip() == "[DONE]":
                            break
                        parsed = loads_or_error(self.name, data)
                        if "error" in parsed and not parsed.get("choices"):
                            error = parsed["error"]
                            text = error.get("message") if isinstance(error, dict) else str(error)
                            raise ProviderError(ErrorKind.TRANSIENT, f"{self.name} stream error: {str(text)[:300]}")
                        chunk = StreamChunk(usage=_usage(parsed.get("usage")))
                        if first:
                            chunk.rate, first = rate, False
                        choices = parsed.get("choices") or []
                        if choices:
                            delta = choices[0].get("delta") or {}
                            chunk.text = delta.get("content") or ""
                            chunk.tool_calls = list(delta.get("tool_calls") or [])
                            chunk.finish_reason = choices[0].get("finish_reason")
                        yield chunk
                    return
            except httpx.HTTPError as exc:
                raise classify_transport_error(self.name, exc) from None


class OpenAIProvider(OpenAICompatProvider):
    name = "openai"
    max_tokens_field = "max_completion_tokens"
