from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ai_mates_mcp_server.config import Settings
from ai_mates_mcp_server.providers import (
    AnthropicProvider,
    GeminiProvider,
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
        "gemini_model": "gemini-3.1-pro-preview",
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


def test_deprecated_models_block_prefix_fallback():
    registry = ProviderRegistry(settings())

    with pytest.raises(ProviderError, match="deprecated"):
        registry.resolve("gpt-4")


def test_deprecated_models_can_be_allowed():
    registry = ProviderRegistry(settings(allow_deprecated_models=True))

    provider, model = registry.resolve("gpt-4.1-nano")

    assert provider.name == "openai"
    assert model == "gpt-4.1-nano"


def test_local_config_cannot_reactivate_deprecated_model(tmp_path):
    models_file = tmp_path / "mates-models.json"
    models_file.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "gpt-4",
                        "provider": "openai",
                        "aliases": ["old-gpt"],
                        "rank": 999,
                        "status": "active",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    registry = ProviderRegistry(settings(models_file=str(models_file)))

    with pytest.raises(ProviderError, match="deprecated"):
        registry.resolve("gpt-4")


def test_local_alias_cannot_bypass_deprecated_model(tmp_path):
    models_file = tmp_path / "mates-models.json"
    models_file.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "gpt-4",
                        "provider": "openai",
                        "aliases": ["old-gpt"],
                        "rank": 999,
                        "status": "active",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    registry = ProviderRegistry(settings(models_file=str(models_file)))

    with pytest.raises(ProviderError, match="deprecated"):
        registry.resolve("old-gpt")


def test_deprecated_model_metadata_is_listed():
    registry = ProviderRegistry(settings())

    rows = {
        model["id"]: model
        for model in registry.model_registry.list_entries(include_deprecated=True)
    }

    assert rows["gpt-4"]["status"] == "deprecated"
    assert rows["gpt-4"]["shutdown_date"] == "2026-10-23"
    assert rows["gpt-4"]["replacement_models"] == ["gpt-5.5"]


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
async def test_openai_supported_model_keeps_explicit_temperature():
    calls = []

    class FakeResponses:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text="ok", usage=None)

    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.api_key = "openai-key"
    provider.default_model = "gpt-4.1"
    provider.client = SimpleNamespace(responses=FakeResponses())

    await provider.complete("Review this", temperature=0.1)

    assert calls[0]["temperature"] == 0.1


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


@pytest.mark.asyncio
async def test_anthropic_supported_model_keeps_explicit_temperature():
    calls = []

    class FakeMessages:
        async def create(self, **kwargs):
            calls.append(kwargs)
            block = SimpleNamespace(type="text", text="ok")
            return SimpleNamespace(content=[block], usage=None)

    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.api_key = "anthropic-key"
    provider.default_model = "claude-3-5-sonnet-20241022"
    provider.client = SimpleNamespace(messages=FakeMessages())

    await provider.complete("Review this", temperature=0.1)

    assert calls[0]["temperature"] == 0.1


@pytest.mark.asyncio
async def test_gemini_provider_omits_temperature_by_default():
    class FakeModels:
        def __init__(self):
            self.calls = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            return type("Response", (), {"text": "ok", "usage_metadata": {"tokens": 1}})()

    class FakeClient:
        def __init__(self):
            self.models = FakeModels()

    provider = GeminiProvider("key", "gemini-3.1-pro-preview")
    fake_client = FakeClient()
    provider.client = fake_client  # type: ignore[assignment]

    await provider.complete("hello", system_prompt="system")

    config = fake_client.models.calls[0]["config"]
    dumped = config.model_dump(exclude_none=True)
    assert "temperature" not in dumped
    assert dumped["system_instruction"] == "system"


def test_deprecated_provider_default_raises_provider_error():
    with pytest.raises(ProviderError, match="Configured default model for provider 'openai'"):
        ProviderRegistry(settings(openai_model="gpt-4"))


def test_deprecated_provider_default_allowed_when_opted_in():
    registry = ProviderRegistry(settings(openai_model="gpt-4", allow_deprecated_models=True))

    assert registry.providers["openai"].default_model == "gpt-4"


def test_replacement_models_must_be_an_array():
    with pytest.raises(ModelRegistryError, match=r"replacement_models.*array"):
        ModelEntry.from_mapping(
            {"id": "m", "provider": "openai", "replacement_models": "nope"},
            source="local",
        )


def test_replacement_models_preserve_case_and_drop_blanks():
    entry = ModelEntry.from_mapping(
        {
            "id": "m",
            "provider": "openai",
            "aliases": ["  Upper  ", "  "],
            "replacement_models": ["  GPT-5.5  ", "  "],
        },
        source="local",
    )

    assert entry.aliases == ("upper",)
    assert entry.replacement_models == ("GPT-5.5",)


def test_local_deprecated_models_cannot_reactivate_blocked_id(tmp_path):
    models_file = tmp_path / "mates-models.json"
    models_file.write_text(
        json.dumps(
            {"deprecated_models": [{"id": "gpt-4", "provider": "openai", "status": "active"}]}
        ),
        encoding="utf-8",
    )

    with pytest.raises(ModelRegistryError, match="non-deprecated status"):
        ProviderRegistry(settings(models_file=str(models_file)))


def test_list_entries_omits_deprecated_by_default():
    registry = ProviderRegistry(settings())

    default_rows = registry.model_registry.list_entries()
    full_rows = registry.model_registry.list_entries(include_deprecated=True)

    assert all(row["status"] in {"active", "preview"} for row in default_rows)
    assert len(full_rows) > len(default_rows)
    assert "gpt-4" not in {row["id"] for row in default_rows}


@pytest.mark.asyncio
async def test_listmodels_omits_deprecated_registry_by_default():
    from ai_mates_mcp_server.tools import run_listmodels

    registry = ProviderRegistry(settings())

    default_payload = json.loads(await run_listmodels(registry))
    full_payload = json.loads(await run_listmodels(registry, include_deprecated=True))

    assert default_payload["deprecated_models_included"] is False
    assert default_payload["deprecated_model_count"] > 0
    assert "are omitted" in default_payload["note"]
    default_ids = {model["id"] for model in default_payload["models"]}
    full_ids = {model["id"] for model in full_payload["models"]}
    assert "gpt-4" not in default_ids
    assert "gpt-4" in full_ids


def _deprecated_alias_file(tmp_path, aliases):
    models_file = tmp_path / "mates-models.json"
    models_file.write_text(
        json.dumps(
            {
                "deprecated_models": [
                    {
                        "id": "gpt-4",
                        "provider": "openai",
                        "status": "deprecated",
                        "aliases": aliases,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return models_file


def test_deprecated_model_alias_is_blocked(tmp_path):
    models_file = _deprecated_alias_file(tmp_path, ["legacy-gpt"])
    registry = ProviderRegistry(settings(models_file=str(models_file)))

    with pytest.raises(ProviderError, match="deprecated"):
        registry.resolve("legacy-gpt")


def test_deprecated_model_alias_resolves_when_opted_in(tmp_path):
    models_file = _deprecated_alias_file(tmp_path, ["legacy-gpt"])
    registry = ModelRegistry(local_models_file=str(models_file), allow_deprecated=True)

    entry = registry.resolve("legacy-gpt")

    assert entry is not None
    assert entry.id == "gpt-4"


def test_routable_looking_deprecated_alias_does_not_reach_prefix_fallback(tmp_path):
    models_file = _deprecated_alias_file(tmp_path, ["gpt-old"])
    registry = ProviderRegistry(settings(models_file=str(models_file)))

    with pytest.raises(ProviderError, match="deprecated"):
        registry.resolve("gpt-old")


def test_replacing_deprecated_entry_drops_its_stale_aliases():
    registry = ModelRegistry()
    payload = {
        "deprecated_models": [
            {"id": "gpt-4", "provider": "openai", "status": "deprecated", "aliases": ["old-a"]}
        ]
    }
    registry._merge_data(payload, source="local")
    assert registry.aliases["old-a"] == "gpt-4"

    payload["deprecated_models"][0]["aliases"] = ["old-b"]
    registry._merge_data(payload, source="local")

    assert "old-a" not in registry.aliases
    assert registry.aliases["old-b"] == "gpt-4"
