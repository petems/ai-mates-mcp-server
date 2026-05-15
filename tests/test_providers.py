from __future__ import annotations

import pytest

from ai_mates_mcp_server.config import Settings
from ai_mates_mcp_server.providers import ProviderError, ProviderRegistry


def settings(**overrides):
    base = {
        "default_model": "auto",
        "openai_api_key": "openai-key",
        "anthropic_api_key": None,
        "gemini_api_key": None,
        "openai_model": "gpt-4.1",
        "anthropic_model": "claude-sonnet-4-5",
        "gemini_model": "gemini-2.5-pro",
        "conversation_ttl_seconds": 60,
        "max_context_chars": 1000,
    }
    base.update(overrides)
    return Settings(**base)


def test_auto_resolves_first_configured_provider():
    registry = ProviderRegistry(settings())

    provider, model = registry.resolve("auto")

    assert provider.name == "openai"
    assert model == "gpt-4.1"


def test_explicit_unconfigured_provider_errors():
    registry = ProviderRegistry(settings(openai_api_key=None))

    with pytest.raises(ProviderError):
        registry.resolve("gpt-4.1")


def test_no_configured_providers_errors():
    registry = ProviderRegistry(settings(openai_api_key=None))

    with pytest.raises(ProviderError):
        registry.resolve("auto")
