"""Provider adapters and their defaults."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from .anthropic import AnthropicProvider
from .base import ChatRequest, ChatResult, Provider, RateInfo, StreamChunk, Usage
from .gemini import GeminiProvider
from .openai_compat import OpenAICompatProvider, OpenAIProvider


@dataclass(frozen=True)
class ProviderDefaults:
    base_url: str | None
    model: str
    context_window: int


# Suggested defaults used by the setup wizard and when config omits a value.
# Quota limits are deliberately NOT defaulted here: free-tier numbers change
# often, and a wrong built-in limit is worse than learning from 429s/headers.
PROVIDER_DEFAULTS: dict[str, ProviderDefaults] = {
    "openai": ProviderDefaults("https://api.openai.com/v1", "gpt-4o-mini", 128_000),
    "anthropic": ProviderDefaults("https://api.anthropic.com", "claude-opus-5-5", 1_000_000),
    "gemini": ProviderDefaults("https://generativelanguage.googleapis.com/v1beta", "gemini-2.5-flash", 1_000_000),
    "openai_compatible": ProviderDefaults(None, "", 32_000),
}

# Models offered in the `/model` menu once a key for that provider exists. The first
# entry of each list is the provider's default above.
SUGGESTED_MODELS: dict[str, list[str]] = {
    "openai": ["gpt-4o-mini", "gpt-4o"],
    "anthropic": ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5-20251001"],
    "gemini": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"],
    "openai_compatible": [],
}

_ADAPTERS: dict[str, type[Provider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "openai_compatible": OpenAICompatProvider,
}


def build_providers(client: httpx.AsyncClient) -> dict[str, Provider]:
    return {name: adapter(client) for name, adapter in _ADAPTERS.items()}


__all__ = [
    "PROVIDER_DEFAULTS",
    "SUGGESTED_MODELS",
    "ChatRequest",
    "ChatResult",
    "Provider",
    "ProviderDefaults",
    "RateInfo",
    "StreamChunk",
    "Usage",
    "build_providers",
]
