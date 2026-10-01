"""Shared test fixtures: a scripted fake provider, a fake clock, config builders.

No test talks to the network or sleeps for real.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baton.config import BatonConfig, parse_config  # noqa: E402
from baton.errors import ErrorKind, ProviderError  # noqa: E402
from baton.handoff import HandoffManager  # noqa: E402
from baton.pool import KeyPool  # noqa: E402
from baton.providers.base import ChatRequest, ChatResult, Provider, StreamChunk, Usage  # noqa: E402
from baton.redact import Redactor  # noqa: E402
from baton.router import Router  # noqa: E402
from baton.session import Relay  # noqa: E402

GOOD_SUMMARY = (
    "## Goal\nBuild the widget.\n\n## Work done\nCreated widget.py and wrote the parser.\n\n"
    "## Current state\nParser passes its tests; the renderer is half-finished.\n\n"
    "## Next steps\nFinish the renderer.\n\n## Key decisions\nUse a recursive descent parser.\n\n"
    "## Files and context touched\n- widget.py\n"
)


def secret_for(index: int) -> str:
    # Built at runtime so no key-shaped literal ever sits in the repository.
    return "test-secret-" + str(index) + "-" + "z9" * 8


def ok(text: str = "ok", *, prompt: int = 10, completion: int = 5, tool_calls: list | None = None) -> ChatResult:
    return ChatResult(
        content=text,
        tool_calls=tool_calls or [],
        finish_reason="tool_calls" if tool_calls else "stop",
        usage=Usage(prompt, completion),
        model="fake-model",
    )


def err(kind: ErrorKind, *, retry_after: float | None = None, status: int | None = None) -> ProviderError:
    return ProviderError(kind, f"simulated {kind.value}", status=status, retry_after=retry_after)


def is_summary_request(request: ChatRequest) -> bool:
    return "handoff note" in str(request.messages[0].get("content", ""))


class FakeProvider(Provider):
    """Provider whose behaviour is scripted per API key.

    `scripts[secret]` is a queue of results/exceptions/callables consumed one
    per call; when it is empty `default` answers.
    """

    name = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[str, ChatRequest]] = []
        self.scripts: dict[str, list[Any]] = {}
        self.stream_scripts: dict[str, list[list[Any]]] = {}
        self.default = lambda request: ok(GOOD_SUMMARY if is_summary_request(request) else "ok")
        # secret -> model names (or an exception) answered by the list-models endpoint
        self.model_lists: dict[str, Any] = {}
        self.list_calls = 0

    async def list_models(self, *, api_key: str, base_url: str) -> list[str]:
        self.list_calls += 1
        item = self.model_lists.get(api_key)
        if item is None:
            raise NotImplementedError("no model list scripted")
        if isinstance(item, BaseException):
            raise item
        return list(item)

    def script(self, index: int, *items: Any) -> None:
        self.scripts.setdefault(secret_for(index), []).extend(items)

    def always(self, index: int, item: Any) -> None:
        self.scripts[secret_for(index)] = _Forever(item)  # type: ignore[assignment]

    def calls_for(self, index: int) -> list[ChatRequest]:
        return [request for secret, request in self.calls if secret == secret_for(index)]

    async def complete(self, request: ChatRequest, *, api_key: str, base_url: str) -> ChatResult:
        self.calls.append((api_key, request))
        queue = self.scripts.get(api_key)
        item = queue.pop(0) if queue else self.default
        if callable(item):
            item = item(request)
        if isinstance(item, BaseException):
            raise item
        return item

    async def stream(self, request: ChatRequest, *, api_key: str, base_url: str) -> AsyncIterator[StreamChunk]:
        self.calls.append((api_key, request))
        queue = self.stream_scripts.get(api_key)
        items = queue.pop(0) if queue else [StreamChunk(text="ok"), StreamChunk(finish_reason="stop", usage=Usage(10, 5))]
        for item in items:
            if isinstance(item, BaseException):
                raise item
            yield item


class _Forever(list):
    """A script queue that never runs out."""

    def __init__(self, item: Any) -> None:
        super().__init__([item])
        self._item = item

    def pop(self, index: int = -1) -> Any:  # noqa: ARG002
        return self._item

    def __bool__(self) -> bool:
        return True


class Clock:
    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self.now = start
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def make_config(count: int = 3, *, limits: dict | None = None, keys: list[dict] | None = None, **sections: Any) -> BatonConfig:
    if keys is None:
        keys = [
            {"id": f"k{index}", "provider": "openai", "secret": "vault", "models": ["fake-model"], "limits": limits or {}}
            for index in range(count)
        ]
    return parse_config({"keys": keys, **sections})


class Harness:
    """A fully wired Baton core with a fake provider, fake clock and no disk I/O
    unless `home` is given."""

    def __init__(self, config: BatonConfig, *, home: Path | None = None) -> None:
        self.config = config
        self.clock = Clock()
        self.provider = FakeProvider()
        self.redactor = Redactor()
        secrets = {key.id: secret_for(index) for index, key in enumerate(config.keys)}
        for key_id, secret in secrets.items():
            self.redactor.register_secret(secret, f"key:{key_id}")
        self.pool = KeyPool(config, secrets, state_path=(home / "state.json") if home else None, clock=self.clock)
        providers = {name: self.provider for name in ("openai", "anthropic", "gemini", "openai_compatible")}
        self.router = Router(config, self.pool, providers, sleep=self.clock.sleep)

        async def summarize(request: ChatRequest, pin: str | None, exclude: frozenset[str]) -> str:
            route = await self.router.complete(request, pin=pin, exclude=exclude)
            return route.result.content or ""

        self.handoff = HandoffManager(
            config.handoff, summarize, checkpoint_dir=(home / "checkpoints") if home else None, redactor=self.redactor
        )
        self.relay = Relay(config, self.pool, self.router, self.handoff)

    def key(self, index: int):
        return self.pool.keys[f"k{index}"]


@pytest.fixture
def harness():
    def build(count: int = 3, *, home: Path | None = None, **kwargs: Any) -> Harness:
        return Harness(make_config(count, **kwargs), home=home)

    return build


def request(text: str = "hello", **kwargs: Any) -> ChatRequest:
    return ChatRequest(model="auto", messages=[{"role": "user", "content": text}], **kwargs)
