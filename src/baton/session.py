"""Session-level orchestration: where rotation and handoff meet.

`Relay` wraps the router for callers that hold a conversation (the terminal
agent, and the proxy when `compact_on_handoff` is on). Before every model call
it decides, at a safe turn boundary, whether the session should move to
another key and whether the transcript should be compacted first.

Three ways a session changes key, in order of preference:

1. **Planned handoff** - the active key's long-window quota crossed
   `handoff_threshold`. The outgoing key writes the summary while it still has
   headroom, then the session moves. This is the normal path.
2. **Rotation** - short-window pressure (per-minute limits, cooldowns). The
   router just uses another key with the full transcript. Lossless, no summary.
3. **Failover** - the active key failed mid-request. The router retries the
   *same* request on the next key with the full transcript, so nothing is
   dropped; the session then stays on the key that answered.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .config import BatonConfig
from .errors import ErrorKind, ProviderError
from .handoff import HandoffManager, HandoffOutcome
from .pool import KeyPool
from .providers.base import ChatRequest, ChatResult, Usage, estimate_message_tokens
from .router import RouteInfo, Router, RouteResult

# How many times a turn is restarted when a stream dies after partial output.
_MAX_STREAM_RESTARTS = 3


@dataclass
class SessionEvent:
    kind: str               # "handoff" | "compaction" | "failover" | "restart"
    from_key: str | None
    to_key: str | None
    detail: str = ""
    at: float = field(default_factory=time.time)


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    messages: list[dict[str, Any]] = field(default_factory=list)
    model: str = "auto"
    active_key: str | None = None
    events: list[SessionEvent] = field(default_factory=list)


def pending_tool_calls(messages: list[dict[str, Any]]) -> set[str]:
    """Ids of tool calls in the last assistant turn that have no result yet."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") == "assistant":
            wanted = {call.get("id") for call in message.get("tool_calls") or [] if call.get("id")}
            answered = {m.get("tool_call_id") for m in messages[index + 1:] if m.get("role") == "tool"}
            return wanted - answered
    return set()


def assemble_tool_calls(deltas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge OpenAI-style streamed tool-call fragments into complete calls."""
    calls: dict[int, dict[str, Any]] = {}
    for delta in deltas:
        index = int(delta.get("index", len(calls)))
        call = calls.setdefault(index, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
        if delta.get("id"):
            call["id"] = delta["id"]
        if delta.get("extra_content"):
            call["extra_content"] = delta["extra_content"]
        function = delta.get("function") or {}
        if function.get("name"):
            call["function"]["name"] += function["name"]
        if function.get("arguments"):
            call["function"]["arguments"] += function["arguments"]
    result = []
    for index in sorted(calls):
        call = calls[index]
        call["id"] = call["id"] or f"call_{uuid.uuid4().hex[:24]}"
        try:
            json.loads(call["function"]["arguments"] or "{}")
        except ValueError:
            # A truncated argument stream must not poison the history with
            # unparseable JSON; an empty object lets the tool report the
            # missing parameters back to the model instead.
            call["function"]["arguments"] = "{}"
        result.append(call)
    return result


class Relay:
    def __init__(self, config: BatonConfig, pool: KeyPool, router: Router, handoff: HandoffManager) -> None:
        self.config = config
        self.pool = pool
        self.router = router
        self.handoff = handoff

    # --- handoff decision ---------------------------------------------------

    async def prepare(self, session: Session, tools: list[dict[str, Any]] | None = None) -> HandoffOutcome | None:
        """Run at the top of every model call. Performs a planned handoff or a
        context compaction when one is due; otherwise does nothing."""
        if not self.config.handoff.enabled:
            return None
        if pending_tool_calls(session.messages):
            # Mid tool round: the model is waiting for results. Never rewrite
            # history here; the check runs again once the round is complete.
            return None

        prompt_tokens = estimate_message_tokens(session.messages, tools)
        active = self.pool.keys.get(session.active_key or "")

        if active is not None and self.pool.needs_handoff(active):
            successors = [
                key for key, _model in self.pool.candidates(session.model, exclude={active.id}, est_tokens=prompt_tokens)
                if not self.pool.needs_handoff(key)
            ]
            if successors:
                target = successors[0]
                outcome = await self.handoff.compact(
                    session.messages,
                    reason="quota",
                    from_key=active.id,
                    to_key=target.id,
                    summary_model=session.model,
                    context_window=active.config.context_window or 32_000,
                    session_id=session.id,
                )
                session.messages = outcome.messages
                session.active_key = target.id
                session.events.append(SessionEvent("handoff", active.id, target.id, f"summary: {outcome.source}"))
                return outcome
            # No healthier key to move to. Stay put: the router will keep
            # using this key until it truly fails, then fail closed.

        window = self._context_window(session, active)
        if self.handoff.should_compact_for_context(session.messages, prompt_tokens, window):
            return await self._compact(session, "context", window)
        return None

    def _context_window(self, session: Session, active) -> int:
        if active is not None and active.config.context_window:
            return active.config.context_window
        candidates = self.pool.candidates(session.model)
        if candidates and candidates[0][0].config.context_window:
            return candidates[0][0].config.context_window
        return 32_000

    async def _compact(self, session: Session, reason: str, window: int, *, force: bool = False) -> HandoffOutcome:
        outcome = await self.handoff.compact(
            session.messages,
            reason=reason,
            from_key=session.active_key,
            to_key=session.active_key,
            summary_model=session.model,
            context_window=window,
            session_id=session.id,
            force=force,
        )
        if outcome.source != "none":
            session.messages = outcome.messages
            session.events.append(SessionEvent("compaction", session.active_key, session.active_key, reason))
        return outcome

    async def force_handoff(self, session: Session) -> HandoffOutcome:
        """User-requested compaction (`/handoff` in the agent)."""
        return await self._compact(session, "manual", self._context_window(session, None), force=True)

    def _note_route(self, session: Session, key_id: str) -> None:
        if session.active_key and session.active_key != key_id:
            session.events.append(SessionEvent("failover", session.active_key, key_id))
        session.active_key = key_id

    def _request(self, session: Session, tools: list[dict[str, Any]] | None, params: dict[str, Any]) -> ChatRequest:
        return ChatRequest(model=session.model, messages=session.messages, tools=tools, **params)

    # --- calls --------------------------------------------------------------

    async def complete(self, session: Session, *, tools: list[dict[str, Any]] | None = None, **params: Any) -> RouteResult:
        await self.prepare(session, tools)
        compacted = False
        while True:
            try:
                route = await self.router.complete(self._request(session, tools, params), prefer=session.active_key)
            except ProviderError as error:
                if error.kind is ErrorKind.CONTEXT_LENGTH and not compacted and self.config.handoff.enabled \
                        and not pending_tool_calls(session.messages):
                    # The provider says the prompt does not fit. Compact once
                    # and retry rather than surfacing an error mid-task.
                    compacted = True
                    outcome = await self._compact(session, "context_overflow", self._context_window(session, None), force=True)
                    if outcome.source != "none":
                        continue
                raise
            self._note_route(session, route.key_id)
            return route

    async def stream(
        self,
        session: Session,
        *,
        tools: list[dict[str, Any]] | None = None,
        on_text: Callable[[str], None] | None = None,
        on_restart: Callable[[], None] | None = None,
        **params: Any,
    ) -> RouteResult:
        """Streamed call that still returns one complete result.

        If the stream dies after partial output, the partial text is discarded
        (`on_restart` lets the UI clear it) and the whole call is retried; the
        router has already penalised the failing key, so the retry lands on a
        different one. The transcript is untouched until a full answer exists.
        """
        await self.prepare(session, tools)
        compacted = False
        restarts = 0
        while True:
            info: list[RouteInfo] = []
            text: list[str] = []
            deltas: list[dict[str, Any]] = []
            finish, usage = "stop", None
            try:
                async for chunk in self.router.stream(
                    self._request(session, tools, params), prefer=session.active_key, on_route=info.append
                ):
                    if chunk.text:
                        text.append(chunk.text)
                        if on_text is not None:
                            on_text(chunk.text)
                    deltas.extend(chunk.tool_calls)
                    if chunk.finish_reason:
                        finish = chunk.finish_reason
                    if chunk.usage is not None:
                        usage = chunk.usage
            except ProviderError as error:
                if error.partial and restarts < _MAX_STREAM_RESTARTS:
                    restarts += 1
                    session.events.append(SessionEvent("restart", session.active_key, None, error.kind.value))
                    if on_restart is not None:
                        on_restart()
                    continue
                if error.kind is ErrorKind.CONTEXT_LENGTH and not compacted and self.config.handoff.enabled \
                        and not pending_tool_calls(session.messages):
                    compacted = True
                    outcome = await self._compact(session, "context_overflow", self._context_window(session, None), force=True)
                    if outcome.source != "none":
                        continue
                raise

            route = info[-1] if info else None
            tool_calls = assemble_tool_calls(deltas)
            if tool_calls and finish == "stop":
                finish = "tool_calls"
            result = ChatResult(
                content="".join(text) or None,
                tool_calls=tool_calls,
                finish_reason=finish,
                usage=usage or Usage(),
                model=route.model if route else session.model,
            )
            key_id = route.key_id if route else (session.active_key or "")
            if key_id:
                self._note_route(session, key_id)
            return RouteResult(result, key_id, route.provider if route else "", result.model, route.attempts if route else [])
