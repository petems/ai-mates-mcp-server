from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ai_mates_mcp_server.config import Settings
from ai_mates_mcp_server.providers import (
    AnthropicProvider,
    OpenAIProvider,
    ProviderError,
    ProviderRegistry,
)
from ai_mates_mcp_server.registry import ModelEntry, ModelRegistry, ModelRegistryError


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
        "models_file": None,
        "model_discovery": "off",
        "allow_deprecated_models": False,
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


def test_packaged_alias_resolves_to_current_model():
    registry = ProviderRegistry(settings(gemini_api_key="gemini-key"))

    provider, model = registry.resolve("pro")

    assert provider.name == "gemini"
    assert model == "gemini-3.1-pro-preview"


def test_provider_alias_uses_env_default_over_packaged_alias():
    registry = ProviderRegistry(settings(openai_model="gpt-4.1"))

    provider, model = registry.resolve("openai")

    assert provider.name == "openai"
    assert model == "gpt-4.1"


def test_local_config_adds_model_without_code_change(tmp_path):
    models_file = tmp_path / "mates-models.json"
    models_file.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "gpt-6-test",
                        "provider": "openai",
                        "aliases": ["future-openai"],
                        "rank": 200,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    registry = ProviderRegistry(settings(models_file=str(models_file)))

    provider, model = registry.resolve("future-openai")

    assert provider.name == "openai"
    assert model == "gpt-6-test"


def test_local_alias_overrides_packaged_alias(tmp_path):
    models_file = tmp_path / "mates-models.json"
    models_file.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "gpt-5.4-mini",
                        "provider": "openai",
                        "aliases": ["sonnet"],
                        "rank": 300,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    registry = ProviderRegistry(settings(models_file=str(models_file)))

    provider, model = registry.resolve("sonnet")

    assert provider.name == "openai"
    assert model == "gpt-5.4-mini"


def test_local_config_rejects_string_aliases(tmp_path):
    models_file = tmp_path / "mates-models.json"
    models_file.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "gpt-6-test",
                        "provider": "openai",
                        "aliases": "future-openai",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ModelRegistryError, match=r"aliases.*array"):
        ProviderRegistry(settings(models_file=str(models_file)))


def test_local_config_rejects_invalid_defaults(tmp_path):
    models_file = tmp_path / "mates-models.json"
    models_file.write_text(
        json.dumps({"defaults": {"openai": 123}, "models": []}),
        encoding="utf-8",
    )

    with pytest.raises(ModelRegistryError, match="Invalid default model id"):
        ProviderRegistry(settings(models_file=str(models_file)))


def test_registry_rejects_unknown_status():
    registry = ModelRegistry(allow_deprecated=True)
    registry.entries["gpt-test"] = ModelEntry(
        id="gpt-test",
        provider="openai",
        status="sunset-soon",
    )

    with pytest.raises(ModelRegistryError, match="unsupported status"):
        registry.resolve("gpt-test")


def test_deprecated_models_error_by_default():
    registry = ProviderRegistry(settings())

    with pytest.raises(ProviderError, match="deprecated"):
        registry.resolve("gpt-4.1-nano")


def test_deprecated_models_can_be_allowed():
    registry = ProviderRegistry(settings(allow_deprecated_models=True))

    provider, model = registry.resolve("gpt-4.1-nano")

    assert provider.name == "openai"
    assert model == "gpt-4.1-nano"


def test_prefix_fallback_accepts_unknown_clearly_routable_model():
    registry = ProviderRegistry(settings())

    provider, model = registry.resolve("gpt-6-new")

    assert provider.name == "openai"
    assert model == "gpt-6-new"


@pytest.mark.asyncio
async def test_live_discovery_augments_listmodels_when_enabled():
    class FakeLiveProvider:
        async def list_model_ids(self):
            return ["gpt-live-model"]

    registry = ProviderRegistry(settings(model_discovery="list"))
    registry.providers["openai"] = FakeLiveProvider()  # type: ignore[assignment]

    result = await registry.list_models()

    live_models = [model for model in result["models"] if model["id"] == "gpt-live-model"]
    assert live_models
    assert live_models[0]["source"] == "live"
    assert live_models[0]["live_discovered"] is True


@pytest.mark.asyncio
async def test_live_discovery_failures_are_reported_without_failing():
    class FailingLiveProvider:
        async def list_model_ids(self):
            raise RuntimeError("nope")

    registry = ProviderRegistry(settings(model_discovery="list"))
    registry.providers["openai"] = FailingLiveProvider()  # type: ignore[assignment]

    result = await registry.list_models()

    assert result["live_errors"]["openai"] == "nope"
    assert result["models"]


@pytest.mark.asyncio
async def test_openai_responses_request_omits_temperature_even_when_provided():
    calls = []

    class FakeResponses:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text="ok", usage={"total_tokens": 1})

    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.api_key = "openai-key"
    provider.default_model = "gpt-5.5"
    provider.client = SimpleNamespace(responses=FakeResponses())

    response = await provider.complete("Review this", temperature=0.1)

    assert response.content == "ok"
    assert calls[0]["model"] == "gpt-5.5"
    assert "temperature" not in calls[0]


@pytest.mark.asyncio
async def test_openai_chat_fallback_request_omits_temperature_even_when_provided():
    calls = []

    class FakeCompletions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(content="ok")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.api_key = "openai-key"
    provider.default_model = "gpt-5.5"
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions()),
    )

    response = await provider.complete("Review this", temperature=0.1)

    assert response.content == "ok"
    assert calls[0]["model"] == "gpt-5.5"
    assert "temperature" not in calls[0]


@pytest.mark.asyncio
async def test_anthropic_request_omits_temperature_even_when_provided():
    calls = []

    class FakeMessages:
        async def create(self, **kwargs):
            calls.append(kwargs)
            block = SimpleNamespace(type="text", text="ok")
            return SimpleNamespace(content=[block], usage=None)

    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.api_key = "anthropic-key"
    provider.default_model = "claude-opus-4-7"
    provider.client = SimpleNamespace(messages=FakeMessages())

    response = await provider.complete("Review this", temperature=0.1)

    assert response.content == "ok"
    assert calls[0]["model"] == "claude-opus-4-7"
    assert "temperature" not in calls[0]
