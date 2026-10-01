"""OpenAI-compatible HTTP surface.

Point any OpenAI client at `http://127.0.0.1:8787/v1` with a Baton token and
it works unchanged; Baton picks and rotates the real keys behind it.

Every request passes the same gate, in this order (cheapest checks first):

  1. Host header check   - blocks DNS-rebinding attacks on the loopback port
  2. Origin check        - blocks cross-site browser requests
  3. Auth lockout        - refuses addresses with repeated bad tokens
  4. Bearer-token auth   - constant-time comparison against stored hashes
  5. Per-client rate limit
  6. Body size cap and schema validation

Nothing an upstream provider says is relayed verbatim: error text is scrubbed
and re-wrapped, so a provider echoing a key in an error cannot leak it.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from ..errors import AllKeysExhaustedError, BatonError, ErrorKind, ProviderError
from ..providers.base import ChatRequest, StreamChunk
from ..providers.openai_compat import PASSTHROUGH
from ..redact import scrub
from ..router import RouteInfo
from ..runtime import Runtime
from ..session import Session
from ..vault import Vault, hash_token
from .guard import AuthLockout, TokenBucket

log = logging.getLogger("baton.proxy")

PROXY_TOKEN_ENV = "BATON_PROXY_TOKEN"
_ROLES = {"system", "developer", "user", "assistant", "tool"}
_LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1"}
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": "no-store",
    "Referrer-Policy": "no-referrer",
}

TokenVerifier = Callable[[str], str | None]
Endpoint = Callable[[Request, str], Awaitable[Response]]


def is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _host_name(header: str) -> str:
    """Hostname part of a Host header: `localhost:8787`, `[::1]:8787`, `127.0.0.1`."""
    header = header.strip().lower()
    if header.startswith("["):
        return header[1:].split("]", 1)[0]
    return header.rsplit(":", 1)[0] if ":" in header else header


def make_token_verifier(vault: Vault | None) -> TokenVerifier:
    """Accept vault-issued tokens, plus an optional one from the environment
    (for containers and CI where there is no interactive `baton token new`)."""
    env_token = os.environ.get(PROXY_TOKEN_ENV, "").strip()
    env_hash = hash_token(env_token) if len(env_token) >= 24 else None
    if env_token and env_hash is None:
        log.warning("%s is set but shorter than 24 characters; ignoring it", PROXY_TOKEN_ENV)

    def verify(token: str) -> str | None:
        name = vault.verify_token(token) if vault is not None else None
        if env_hash is not None and hmac.compare_digest(hash_token(token), env_hash):
            name = name or "env"
        return name

    return verify


def _error(status: int, message: str, kind: str, code: str | None = None, headers: dict[str, str] | None = None) -> JSONResponse:
    body = {"error": {"message": scrub(message), "type": kind, "param": None, "code": code}}
    return JSONResponse(body, status_code=status, headers={**_SECURITY_HEADERS, **(headers or {})})


def _error_for(exc: Exception) -> JSONResponse:
    if isinstance(exc, AllKeysExhaustedError):
        headers = {"Retry-After": str(int(exc.retry_after or 30) + 1)}
        return _error(503, str(exc), "server_error", "all_keys_exhausted", headers)
    if isinstance(exc, ProviderError):
        if exc.kind is ErrorKind.CONTEXT_LENGTH:
            return _error(400, exc.message, "invalid_request_error", "context_length_exceeded")
        if exc.kind is ErrorKind.BAD_REQUEST:
            status = exc.status if exc.status in (400, 404, 413, 422) else 400
            return _error(status, exc.message, "invalid_request_error")
        return _error(502, exc.message, "upstream_error", exc.kind.value)
    if isinstance(exc, BatonError):
        return _error(500, str(exc), "server_error")
    log.exception("unhandled error in proxy request")
    return _error(500, "internal error", "server_error")


def parse_chat_request(body: Any, *, max_messages: int) -> tuple[ChatRequest, bool, bool]:
    """Validate an OpenAI chat-completions body. Returns (request, stream, include_usage).

    Raises ValueError with a client-safe message on anything malformed.
    """
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("'messages' must be a non-empty array")
    if len(messages) > max_messages:
        raise ValueError(f"too many messages (limit {max_messages})")
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in _ROLES:
            raise ValueError(f"messages[{index}] must be an object with a valid 'role'")
        content = message.get("content")
        if content is not None and not isinstance(content, (str, list)):
            raise ValueError(f"messages[{index}].content must be a string, an array or null")
        if message["role"] == "tool" and not isinstance(message.get("tool_call_id"), str):
            raise ValueError(f"messages[{index}] (role 'tool') requires 'tool_call_id'")
        calls = message.get("tool_calls")
        if calls is not None and not (isinstance(calls, list) and all(isinstance(c, dict) for c in calls)):
            raise ValueError(f"messages[{index}].tool_calls must be an array of objects")

    model = body.get("model", "auto")
    if not isinstance(model, str) or len(model) > 200:
        raise ValueError("'model' must be a string of at most 200 characters")

    tools = body.get("tools")
    if tools is not None:
        if not isinstance(tools, list) or len(tools) > 256 or not all(isinstance(t, dict) for t in tools):
            raise ValueError("'tools' must be an array of at most 256 objects")
        tools = tools or None

    def number(name: str, low: float, high: float) -> float | None:
        value = body.get(name)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise ValueError(f"'{name}' must be a number between {low} and {high}")
        return value

    max_tokens = number("max_completion_tokens", 1, 2_000_000) or number("max_tokens", 1, 2_000_000)
    if body.get("n") not in (None, 1):
        raise ValueError("'n' other than 1 is not supported")
    stop = body.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    if stop is not None and not (isinstance(stop, list) and len(stop) <= 16 and all(isinstance(s, str) for s in stop)):
        raise ValueError("'stop' must be a string or an array of up to 16 strings")
    response_format = body.get("response_format")
    if response_format is not None and not isinstance(response_format, dict):
        raise ValueError("'response_format' must be an object")
    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise ValueError("'stream' must be a boolean")
    options = body.get("stream_options")
    include_usage = bool(isinstance(options, dict) and options.get("include_usage"))

    request = ChatRequest(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice=body.get("tool_choice") if tools else None,
        temperature=number("temperature", 0, 2),
        top_p=number("top_p", 0, 1),
        max_tokens=int(max_tokens) if max_tokens else None,
        stop=stop or None,
        response_format=response_format,
        extra={key: body[key] for key in PASSTHROUGH if key in body and key != "n"},
    )
    return request, stream, include_usage


def create_app(
    runtime: Runtime,
    verify_token: TokenVerifier,
    *,
    bound_host: str | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Starlette:
    config = runtime.config.proxy
    bound_host = bound_host or config.host
    local_only = is_loopback(bound_host)
    limiter = TokenBucket(config.rate_limit.requests_per_minute, config.rate_limit.burst, clock=clock)
    lockout = AuthLockout(config.auth_fail_limit, config.auth_lockout_seconds, clock=clock)
    allowed_origins = set(config.cors_origins)
    # Session affinity: keep a client on one key until it fills (keeps
    # provider-side prompt caches warm and makes usage easy to follow).
    last_key: dict[str, str] = {}

    def cors_headers(request: Request) -> dict[str, str]:
        origin = request.headers.get("origin")
        if origin and origin in allowed_origins:
            return {"Access-Control-Allow-Origin": origin, "Vary": "Origin"}
        return {}

    def guard(endpoint: Endpoint) -> Callable[[Request], Awaitable[Response]]:
        async def wrapped(request: Request) -> Response:
            address = request.client.host if request.client else "unknown"

            # 1. DNS rebinding: a web page can make a browser resolve
            #    evil.example to 127.0.0.1, but the Host header still says
            #    evil.example. Only loopback names are valid on a loopback bind.
            if local_only and _host_name(request.headers.get("host", "")) not in _LOOPBACK_NAMES:
                return _error(403, "invalid Host header", "invalid_request_error", "forbidden_host")

            # 2. Browsers attach Origin to cross-site requests; API clients do not.
            origin = request.headers.get("origin")
            if origin and origin not in allowed_origins:
                return _error(403, "origin not allowed", "invalid_request_error", "forbidden_origin")

            # 3 + 4. Authentication. A valid token is always honoured, even
            #    from a locked-out address: on localhost every client shares
            #    127.0.0.1, and one misconfigured script must not be able to
            #    lock out its neighbours. The lockout only throttles the
            #    *failing* caller (and its log noise).
            header = request.headers.get("authorization", "")
            token = header[7:].strip() if header[:7].lower() == "bearer " else request.headers.get("x-api-key", "").strip()
            client = verify_token(token) if token and len(token) <= 512 else None
            if client is None:
                blocked = lockout.blocked_for(address)
                if blocked > 0:
                    return _error(429, "too many failed authentication attempts", "rate_limit_error",
                                  "auth_lockout", {"Retry-After": str(int(blocked) + 1)})
                lockout.record_failure(address)
                log.warning("rejected request with missing or invalid token from %s", address)
                return _error(401, "missing or invalid Baton token", "authentication_error", "invalid_api_key",
                              {"WWW-Authenticate": "Bearer"})

            # 5. Rate limit per authenticated client.
            wait = limiter.allow(client)
            if wait > 0:
                return _error(429, "rate limit exceeded for this client", "rate_limit_error",
                              "rate_limit_exceeded", {"Retry-After": str(int(wait) + 1)})

            try:
                response = await endpoint(request, client)
            except Exception as exc:  # noqa: BLE001 - the proxy must answer, never crash
                response = _error_for(exc)
            for name, value in {**_SECURITY_HEADERS, **cors_headers(request)}.items():
                response.headers.setdefault(name, value)
            return response

        return wrapped

    async def read_json(request: Request) -> Any:
        """Read the body with a hard size cap (never trust Content-Length alone)."""
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > config.max_body_bytes:
            raise ProviderError(ErrorKind.BAD_REQUEST, "request body too large", status=413)
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > config.max_body_bytes:
                raise ProviderError(ErrorKind.BAD_REQUEST, "request body too large", status=413)
            chunks.append(chunk)
        try:
            return json.loads(b"".join(chunks))
        except (ValueError, UnicodeDecodeError):
            raise ProviderError(ErrorKind.BAD_REQUEST, "request body is not valid JSON", status=400) from None

    async def healthz(_request: Request) -> Response:
        # Unauthenticated on purpose (load balancers, scripts) and therefore
        # reveals nothing beyond "the process is up".
        return JSONResponse({"status": "ok"}, headers=_SECURITY_HEADERS)

    async def models(_request: Request, _client: str) -> Response:
        now = int(time.time())
        data = [{"id": name, "object": "model", "created": now, "owned_by": "baton"} for name in runtime.pool.known_models()]
        return JSONResponse({"object": "list", "data": data})

    async def status(_request: Request, _client: str) -> Response:
        return JSONResponse({"keys": runtime.pool.snapshot()})

    async def chat(request: Request, client: str) -> Response:
        body = await read_json(request)
        try:
            chat_request, stream, include_usage = parse_chat_request(body, max_messages=config.max_messages)
        except ValueError as exc:
            return _error(400, str(exc), "invalid_request_error")

        prefer = last_key.get(client)
        if config.compact_on_handoff:
            # Optional: apply the same planned handoff the agent uses. Off by
            # default because it rewrites what the client sent.
            session = Session(messages=chat_request.messages, model=chat_request.model, active_key=prefer)
            await runtime.relay.prepare(session, chat_request.tools)
            chat_request.messages, prefer = session.messages, session.active_key

        completion_id = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())

        if not stream:
            route = await runtime.router.complete(chat_request, prefer=prefer)
            last_key[client] = route.key_id
            result = route.result
            message: dict[str, Any] = {"role": "assistant", "content": result.content}
            if result.tool_calls:
                message["tool_calls"] = result.tool_calls
            return JSONResponse(
                {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": result.model,
                    "choices": [{"index": 0, "message": message, "finish_reason": result.finish_reason, "logprobs": None}],
                    "usage": {
                        "prompt_tokens": result.usage.prompt_tokens,
                        "completion_tokens": result.usage.completion_tokens,
                        "total_tokens": result.usage.total_tokens,
                    },
                },
                headers={"x-baton-key": route.key_id, "x-baton-attempts": str(len(route.attempts))},
            )

        routes: list[RouteInfo] = []
        source = runtime.router.stream(chat_request, prefer=prefer, on_route=routes.append)
        # Pull the first chunk *before* committing to a 200 response: every
        # failure up to this point (including "all keys exhausted") can still
        # be reported with a proper HTTP status, and the router can still fail
        # over transparently.
        try:
            first: StreamChunk | None = await source.__anext__()
        except StopAsyncIteration:
            first = None
        except Exception:
            await source.aclose()
            raise
        if routes:
            last_key[client] = routes[-1].key_id
        model_name = routes[-1].model if routes else chat_request.model

        def frame(delta: dict[str, Any], finish: str | None = None) -> bytes:
            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n"

        async def events() -> AsyncIterator[bytes]:
            usage = None
            finished = False
            try:
                yield frame({"role": "assistant", "content": ""})
                chunk = first
                while chunk is not None:
                    if chunk.usage is not None:
                        usage = chunk.usage
                    delta: dict[str, Any] = {}
                    if chunk.text:
                        delta["content"] = chunk.text
                    if chunk.tool_calls:
                        delta["tool_calls"] = chunk.tool_calls
                    if delta or chunk.finish_reason:
                        finished = finished or bool(chunk.finish_reason)
                        yield frame(delta, chunk.finish_reason)
                    try:
                        chunk = await source.__anext__()
                    except StopAsyncIteration:
                        chunk = None
                if not finished:
                    yield frame({}, "stop")
                if include_usage and usage is not None:
                    payload = {
                        "id": completion_id, "object": "chat.completion.chunk", "created": created,
                        "model": model_name, "choices": [],
                        "usage": {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
                                  "total_tokens": usage.total_tokens},
                    }
                    yield b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n"
            except Exception as exc:  # noqa: BLE001 - headers are sent; report in-band
                # Content already reached the client, so the status line cannot
                # change. Send an error event in the format OpenAI SDKs raise on.
                message = exc.message if isinstance(exc, ProviderError) else "stream interrupted"
                log.warning("stream to client '%s' failed mid-way: %s", client, message)
                error = {"error": {"message": scrub(message), "type": "upstream_error", "code": "stream_interrupted"}}
                yield b"data: " + json.dumps(error).encode("utf-8") + b"\n\n"
            finally:
                await source.aclose()
            yield b"data: [DONE]\n\n"

        headers = {"x-baton-key": routes[-1].key_id if routes else "", "X-Accel-Buffering": "no"}
        return StreamingResponse(events(), media_type="text/event-stream", headers=headers)

    known_paths = {"/v1/models", "/v1/chat/completions", "/v1/baton/status"}

    async def fallback(request: Request) -> Response:
        """Everything under /v1 that no real route claimed: CORS preflight, 405 or 404."""
        if request.method != "OPTIONS":
            if request.url.path in known_paths:
                return await method_not_allowed(request, None)
            return await not_found(request, None)
        headers = cors_headers(request)
        if not headers:
            return _error(403, "origin not allowed", "invalid_request_error", "forbidden_origin")
        headers.update({
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Authorization, Content-Type, X-Api-Key",
            "Access-Control-Max-Age": "600",
        })
        return Response(status_code=204, headers=headers)

    async def not_found(_request: Request, _exc: Exception | None) -> Response:
        return _error(404, "unknown endpoint; Baton serves /v1/chat/completions and /v1/models",
                      "invalid_request_error", "not_found")

    async def method_not_allowed(_request: Request, _exc: Exception | None) -> Response:
        return _error(405, "method not allowed", "invalid_request_error", "method_not_allowed")

    routes = [
        Route("/healthz", healthz, methods=["GET"]),
        Route("/v1/models", guard(models), methods=["GET"]),
        Route("/v1/chat/completions", guard(chat), methods=["POST"]),
        Route("/v1/baton/status", guard(status), methods=["GET"]),
        Route("/v1/{path:path}", fallback, methods=["OPTIONS", "GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"]),
    ]
    app = Starlette(routes=routes, exception_handlers={404: not_found, 405: method_not_allowed})
    app.state.runtime = runtime
    return app
