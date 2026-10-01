"""Anthropic Messages API adapter (raw HTTP, translated to/from OpenAI shape).

Design notes that matter for handoff stability:

* `thinking` / `redacted_thinking` blocks are dropped from responses and never
  replayed. On current Claude models a thinking block is cryptographically
  bound to the exact conversation prefix that produced it; Baton rewrites
  history on handoff (summary + recent turns), which would invalidate every
  retained block and turn the next request into a 400. A history that carries
  no thinking blocks at all is always valid, so that is what we send.
* Sampling parameters and forced `tool_choice` are only forwarded to models
  that still accept them; newer models reject both with a 400.
"""

from __future__ import annotations

import json
import re
import uuid
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
    iter_sse,
    loads_or_error,
    parse_data_url,
    parse_rate_headers,
    raise_for_status,
)

API_VERSION = "2023-06-01"
# Non-streaming default. Generous enough not to truncate, small enough to stay
# inside HTTP timeouts. Callers can override per request or per key.
DEFAULT_MAX_TOKENS = 16000

# Models that still accept temperature/top_p and forced tool_choice. Anything
# not matched is treated as "current" and gets neither.
_LEGACY_MODEL = re.compile(r"claude-(?:3|haiku-4|sonnet-4|opus-4-[0-6](?!\d))")

_STOP_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}

_STREAM_ERROR_KINDS = {
    "overloaded_error": ErrorKind.TRANSIENT,
    "api_error": ErrorKind.TRANSIENT,
    "rate_limit_error": ErrorKind.RATE_LIMIT,
    "authentication_error": ErrorKind.AUTH,
    "permission_error": ErrorKind.AUTH,
    "billing_error": ErrorKind.QUOTA_EXHAUSTED,
    "invalid_request_error": ErrorKind.BAD_REQUEST,
}


def _content_blocks(content: Any) -> list[dict[str, Any]]:
    """OpenAI content (string or parts) -> Anthropic content blocks."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks: list[dict[str, Any]] = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and part.get("text"):
            blocks.append({"type": "text", "text": str(part["text"])})
        elif part.get("type") == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            parsed = parse_data_url(url)
            if parsed:
                blocks.append({"type": "image", "source": {"type": "base64", "media_type": parsed[0], "data": parsed[1]}})
            elif url.startswith("https://"):
                blocks.append({"type": "image", "source": {"type": "url", "url": url}})
    return blocks


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def to_anthropic_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Return (system_prompt, messages) in Anthropic shape."""
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []

    def push(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        # Tool results arrive as separate OpenAI messages but must share one
        # user turn; merging same-role neighbours handles that and keeps the
        # strict user/assistant alternation the API expects.
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})

    for message in messages:
        role = message.get("role")
        if role in ("system", "developer"):
            text = "\n".join(b["text"] for b in _content_blocks(message.get("content")) if b["type"] == "text")
            if text:
                system_parts.append(text)
        elif role == "user":
            push("user", _content_blocks(message.get("content")))
        elif role == "assistant":
            blocks = _content_blocks(message.get("content"))
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                blocks.append({
                    "type": "tool_use",
                    "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                    "name": function.get("name", ""),
                    "input": _parse_arguments(function.get("arguments")),
                })
            push("assistant", blocks)
        elif role == "tool":
            text = "\n".join(b["text"] for b in _content_blocks(message.get("content")) if b["type"] == "text")
            push("user", [{
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id", ""),
                "content": text or "(no output)",
            }])

    if not out or out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": [{"type": "text", "text": "(conversation continues)"}]})
    # tool_result blocks must come first within a user turn.
    for item in out:
        if item["role"] == "user":
            item["content"].sort(key=lambda block: block["type"] != "tool_result")
    return "\n\n".join(system_parts), out


def _usage(data: dict[str, Any] | None) -> Usage:
    data = data or {}
    prompt = sum(int(data.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    return Usage(prompt, int(data.get("output_tokens") or 0))


class AnthropicProvider(Provider):
    name = "anthropic"

    def _headers(self, api_key: str) -> dict[str, str]:
        return {"x-api-key": api_key, "anthropic-version": API_VERSION, "content-type": "application/json"}

    def _payload(self, request: ChatRequest, *, stream: bool) -> dict[str, Any]:
        system, messages = to_anthropic_messages(request.messages)
        legacy = bool(_LEGACY_MODEL.match(request.model))
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens or DEFAULT_MAX_TOKENS,
        }
        if system:
            payload["system"] = system
        if request.stop:
            payload["stop_sequences"] = request.stop
        if legacy:
            # The API accepts only one of the two; temperature wins.
            if request.temperature is not None:
                payload["temperature"] = max(0.0, min(1.0, request.temperature))
            elif request.top_p is not None:
                payload["top_p"] = request.top_p
        if request.tools:
            payload["tools"] = [
                {
                    "name": (tool.get("function") or {}).get("name", ""),
                    "description": (tool.get("function") or {}).get("description", ""),
                    "input_schema": (tool.get("function") or {}).get("parameters") or {"type": "object", "properties": {}},
                }
                for tool in request.tools
                if tool.get("type") == "function"
            ]
            choice = request.tool_choice
            if choice == "none":
                payload["tool_choice"] = {"type": "none"}
            elif legacy and choice == "required":
                payload["tool_choice"] = {"type": "any"}
            elif legacy and isinstance(choice, dict) and (choice.get("function") or {}).get("name"):
                payload["tool_choice"] = {"type": "tool", "name": choice["function"]["name"]}
            # Otherwise leave the default ("auto").
        if stream:
            payload["stream"] = True
        return payload

    async def complete(self, request: ChatRequest, *, api_key: str, base_url: str) -> ChatResult:
        url = base_url.rstrip("/") + "/v1/messages"
        response = await self._post_json(url, self._headers(api_key), self._payload(request, stream=False))
        data = loads_or_error(self.name, response.text)
        texts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in data.get("content") or []:
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block.get("id"),
                    "type": "function",
                    "function": {"name": block.get("name", ""), "arguments": json.dumps(block.get("input") or {})},
                })
            # thinking / redacted_thinking: intentionally dropped, see module docstring.
        return ChatResult(
            content="".join(texts) or None,
            tool_calls=tool_calls,
            finish_reason=_STOP_REASONS.get(data.get("stop_reason") or "end_turn", "stop"),
            usage=_usage(data.get("usage")),
            model=str(data.get("model") or request.model),
            rate=parse_rate_headers(response.headers),
        )

    async def stream(self, request: ChatRequest, *, api_key: str, base_url: str) -> AsyncIterator[StreamChunk]:
        url = base_url.rstrip("/") + "/v1/messages"
        payload = self._payload(request, stream=True)
        usage = Usage()
        # Anthropic indexes every content block; OpenAI indexes tool calls only.
        tool_index: dict[int, int] = {}
        try:
            async with self.client.stream("POST", url, headers=self._headers(api_key), json=payload) as response:
                await raise_for_status(self.name, response)
                yield StreamChunk(rate=parse_rate_headers(response.headers))
                async for _event, data in iter_sse(response):
                    event = loads_or_error(self.name, data)
                    kind = event.get("type")
                    if kind == "message_start":
                        usage.prompt_tokens = _usage((event.get("message") or {}).get("usage")).prompt_tokens
                    elif kind == "content_block_start":
                        block = event.get("content_block") or {}
                        if block.get("type") == "tool_use":
                            index = len(tool_index)
                            tool_index[int(event.get("index", 0))] = index
                            yield StreamChunk(tool_calls=[{
                                "index": index,
                                "id": block.get("id"),
                                "type": "function",
                                "function": {"name": block.get("name", ""), "arguments": ""},
                            }])
                    elif kind == "content_block_delta":
                        delta = event.get("delta") or {}
                        if delta.get("type") == "text_delta":
                            yield StreamChunk(text=delta.get("text", ""))
                        elif delta.get("type") == "input_json_delta":
                            index = tool_index.get(int(event.get("index", 0)))
                            if index is not None:
                                yield StreamChunk(tool_calls=[{
                                    "index": index,
                                    "function": {"arguments": delta.get("partial_json", "")},
                                }])
                    elif kind == "message_delta":
                        usage.completion_tokens = int((event.get("usage") or {}).get("output_tokens") or 0)
                        reason = (event.get("delta") or {}).get("stop_reason")
                        if reason:
                            yield StreamChunk(finish_reason=_STOP_REASONS.get(reason, "stop"), usage=usage)
                    elif kind == "error":
                        error = event.get("error") or {}
                        raise ProviderError(
                            _STREAM_ERROR_KINDS.get(error.get("type", ""), ErrorKind.TRANSIENT),
                            f"anthropic stream error: {str(error.get('message', 'unknown'))[:300]}",
                        )
        except httpx.HTTPError as exc:
            raise classify_transport_error(self.name, exc) from None
