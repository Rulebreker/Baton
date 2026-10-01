"""Google Gemini (Generative Language API) adapter.

The key travels in the `x-goog-api-key` header, never in the `?key=` query
parameter Google's quickstarts use: URLs end up in logs, proxies and exception
messages, headers do not.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import quote

import httpx

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
    raise_for_status,
)

# JSON-Schema keywords the `parameters` field does not understand. Sending
# them is a hard 400, so they are stripped recursively.
_UNSUPPORTED_SCHEMA_KEYS = {
    "$schema", "$id", "$ref", "$defs", "definitions", "additionalProperties", "default", "examples",
    "patternProperties", "unevaluatedProperties", "const", "strict", "title",
}

_FINISH_REASONS = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
}

# Gemini 3+ validates a "thought signature" on replayed function calls. When a
# call came from another provider (cross-provider handoff) there is none, and
# Google documents this placeholder to skip validation for such histories.
_SIGNATURE_PLACEHOLDER = "skip_thought_signature_validator"
_STRICT_SIGNATURE_MODEL = re.compile(r"gemini-(?:[3-9]|\d{2,})")


def sanitize_schema(schema: Any) -> Any:
    if isinstance(schema, dict):
        cleaned = {k: sanitize_schema(v) for k, v in schema.items() if k not in _UNSUPPORTED_SCHEMA_KEYS}
        # `type: ["string", "null"]` -> `type: string, nullable: true`
        if isinstance(cleaned.get("type"), list):
            types = [t for t in cleaned["type"] if t != "null"]
            if len(types) != len(cleaned["type"]):
                cleaned["nullable"] = True
            cleaned["type"] = types[0] if types else "string"
        return cleaned
    if isinstance(schema, list):
        return [sanitize_schema(item) for item in schema]
    return schema


def _parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}] if content else []
    parts: list[dict[str, Any]] = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and part.get("text"):
            parts.append({"text": str(part["text"])})
        elif part.get("type") == "image_url":
            parsed = parse_data_url((part.get("image_url") or {}).get("url", ""))
            if parsed:
                parts.append({"inlineData": {"mimeType": parsed[0], "data": parsed[1]}})
            else:
                parts.append({"text": "[image omitted: this provider only accepts inline image data]"})
    return parts


def _text(content: Any) -> str:
    return "\n".join(p["text"] for p in _parts(content) if "text" in p)


def to_gemini_contents(messages: list[dict[str, Any]], model: str) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    call_names: dict[str, str] = {}  # tool_call_id -> function name
    strict = bool(_STRICT_SIGNATURE_MODEL.match(model))

    def push(role: str, parts: list[dict[str, Any]]) -> None:
        if not parts:
            return
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": role, "parts": parts})

    for message in messages:
        role = message.get("role")
        if role in ("system", "developer"):
            text = _text(message.get("content"))
            if text:
                system_parts.append(text)
        elif role == "user":
            push("user", _parts(message.get("content")))
        elif role == "assistant":
            parts = _parts(message.get("content"))
            first_call = True
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                name = function.get("name", "")
                if call.get("id"):
                    call_names[call["id"]] = name
                try:
                    args = json.loads(function.get("arguments") or "{}")
                except (TypeError, ValueError):
                    args = {}
                part: dict[str, Any] = {"functionCall": {"name": name, "args": args if isinstance(args, dict) else {}}}
                signature = ((call.get("extra_content") or {}).get("google") or {}).get("thought_signature")
                if signature:
                    part["thoughtSignature"] = signature
                elif strict and first_call:
                    part["thoughtSignature"] = _SIGNATURE_PLACEHOLDER
                first_call = False
                parts.append(part)
            push("model", parts)
        elif role == "tool":
            name = message.get("name") or call_names.get(message.get("tool_call_id", ""), "tool")
            push("user", [{"functionResponse": {"name": name, "response": {"result": _text(message.get("content"))}}}])

    if not contents or contents[0]["role"] != "user":
        contents.insert(0, {"role": "user", "parts": [{"text": "(conversation continues)"}]})
    return "\n\n".join(system_parts), contents


def _usage(data: dict[str, Any] | None) -> Usage:
    data = data or {}
    completion = int(data.get("candidatesTokenCount") or 0) + int(data.get("thoughtsTokenCount") or 0)
    return Usage(int(data.get("promptTokenCount") or 0), completion)


def _read_candidate(data: dict[str, Any]) -> tuple[str, list[dict[str, Any]], str | None]:
    """Extract (text, tool_calls, finish_reason) from one response/chunk."""
    candidates = data.get("candidates") or []
    if not candidates:
        # Prompt blocked before generation: report it as a filtered completion
        # rather than an error, mirroring how OpenAI surfaces refusals.
        blocked = (data.get("promptFeedback") or {}).get("blockReason")
        return "", [], "content_filter" if blocked else None
    candidate = candidates[0]
    texts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for part in (candidate.get("content") or {}).get("parts") or []:
        if part.get("thought"):
            continue  # internal reasoning summary, not answer text
        if "text" in part:
            texts.append(part["text"])
        elif "functionCall" in part:
            call = part["functionCall"]
            item: dict[str, Any] = {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": call.get("name", ""), "arguments": json.dumps(call.get("args") or {})},
            }
            if part.get("thoughtSignature"):
                # Same field Google's own OpenAI-compatible endpoint uses, so
                # clients that round-trip tool calls keep the signature.
                item["extra_content"] = {"google": {"thought_signature": part["thoughtSignature"]}}
            tool_calls.append(item)
    reason = candidate.get("finishReason")
    finish = None
    if reason:
        finish = "tool_calls" if tool_calls else _FINISH_REASONS.get(reason, "stop")
    return "".join(texts), tool_calls, finish


class GeminiProvider(Provider):
    name = "gemini"

    def _headers(self, api_key: str) -> dict[str, str]:
        return {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    def _url(self, base_url: str, model: str, *, stream: bool) -> str:
        # The model id comes from config / the client; keep it from escaping the path.
        safe_model = re.sub(r"[^A-Za-z0-9._\-]", "", model.removeprefix("models/"))
        method = "streamGenerateContent?alt=sse" if stream else "generateContent"
        return f"{base_url.rstrip('/')}/models/{safe_model}:{method}"

    def _payload(self, request: ChatRequest) -> dict[str, Any]:
        system, contents = to_gemini_contents(request.messages, request.model)
        payload: dict[str, Any] = {"contents": contents}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        config: dict[str, Any] = {}
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.top_p is not None:
            config["topP"] = request.top_p
        if request.max_tokens is not None:
            config["maxOutputTokens"] = request.max_tokens
        if request.stop:
            config["stopSequences"] = request.stop[:5]
        if (request.response_format or {}).get("type") in ("json_object", "json_schema"):
            config["responseMimeType"] = "application/json"
        if config:
            payload["generationConfig"] = config
        if request.tools:
            declarations = []
            for tool in request.tools:
                function = tool.get("function") or {}
                if tool.get("type") != "function" or not function.get("name"):
                    continue
                declaration: dict[str, Any] = {"name": function["name"], "description": function.get("description", "")}
                parameters = sanitize_schema(function.get("parameters") or {})
                if parameters.get("properties"):
                    declaration["parameters"] = parameters
                declarations.append(declaration)
            if declarations:
                payload["tools"] = [{"functionDeclarations": declarations}]
                choice = request.tool_choice
                mode = "NONE" if choice == "none" else "ANY" if choice == "required" or isinstance(choice, dict) else None
                if mode:
                    calling: dict[str, Any] = {"mode": mode}
                    if isinstance(choice, dict) and (choice.get("function") or {}).get("name"):
                        calling["allowedFunctionNames"] = [choice["function"]["name"]]
                    payload["toolConfig"] = {"functionCallingConfig": calling}
        return payload

    async def list_models(self, *, api_key: str, base_url: str) -> list[str]:
        names: list[str] = []
        token = ""
        for _page in range(10):
            url = f"{base_url.rstrip('/')}/models?pageSize=1000"
            if token:
                url += "&pageToken=" + quote(token, safe="")
            data = await self._get_json(url, self._headers(api_key))
            for model in data.get("models") or []:
                if isinstance(model, dict) and "generateContent" in (model.get("supportedGenerationMethods") or []):
                    names.append(str(model.get("name", "")).removeprefix("models/"))
            token = str(data.get("nextPageToken") or "")
            if not token:
                break
        return [name for name in names if name]

    async def complete(self, request: ChatRequest, *, api_key: str, base_url: str) -> ChatResult:
        url = self._url(base_url, request.model, stream=False)
        response = await self._post_json(url, self._headers(api_key), self._payload(request))
        data = loads_or_error(self.name, response.text)
        text, tool_calls, finish = _read_candidate(data)
        return ChatResult(
            content=text or None,
            tool_calls=tool_calls,
            finish_reason=finish or "stop",
            usage=_usage(data.get("usageMetadata")),
            model=str(data.get("modelVersion") or request.model),
        )

    async def stream(self, request: ChatRequest, *, api_key: str, base_url: str) -> AsyncIterator[StreamChunk]:
        url = self._url(base_url, request.model, stream=True)
        call_count = 0
        try:
            async with self.client.stream("POST", url, headers=self._headers(api_key), json=self._payload(request)) as response:
                await raise_for_status(self.name, response)
                async for _event, data in iter_sse(response):
                    parsed = loads_or_error(self.name, data)
                    text, tool_calls, finish = _read_candidate(parsed)
                    # Gemini delivers each function call whole; give it the
                    # index OpenAI-style clients expect for delta assembly.
                    for call in tool_calls:
                        call["index"] = call_count
                        call_count += 1
                    if finish and call_count:
                        finish = "tool_calls"
                    usage = _usage(parsed["usageMetadata"]) if parsed.get("usageMetadata") else None
                    yield StreamChunk(text=text, tool_calls=tool_calls, finish_reason=finish, usage=usage)
        except httpx.HTTPError as exc:
            raise classify_transport_error(self.name, exc) from None
