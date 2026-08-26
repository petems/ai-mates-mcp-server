from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from google.auth.credentials import AnonymousCredentials
from google.auth.exceptions import DefaultCredentialsError, RefreshError

from ai_mates_mcp_server.config import Settings
from ai_mates_mcp_server.providers import (
    AnthropicProvider,
    GeminiProvider,
    OpenAIProvider,
    ProviderError,
    ProviderRegistry,
    _normalize_gemini_model_name,
)
from ai_mates_mcp_server.registry import ModelEntry, ModelRegistry, ModelRegistryError


def settings(**overrides):
    base = {
        "default_model": "auto",
        "openai_api_key": "openai-key",
        "anthropic_api_key": None,
        "gemini_api_key": None,
        "gemini_use_gcloud_auth": False,
        "gemini_project": None,
        "gemini_location": None,
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
    assert len(json.dumps(default_payload)) < len(json.dumps(full_payload)) / 5


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


FAKE_ADC_PROJECT = "adc-project"


def fake_adc(credentials=None, project=FAKE_ADC_PROJECT):
    """Patch ADC resolution with credentials that need no network or key material."""
    return patch(
        "ai_mates_mcp_server.providers.google.auth.default",
        return_value=(credentials or AnonymousCredentials(), project),
    )


def test_gemini_api_key_uses_developer_api_client():
    with patch("ai_mates_mcp_server.providers.genai.Client") as mock_client_cls:
        registry = ProviderRegistry(settings(openai_api_key=None, gemini_api_key="my-key"))

    mock_client_cls.assert_called_once_with(api_key="my-key")
    assert registry.providers["gemini"].auth_mode == "api-key"


def test_gemini_gcloud_auth_uses_vertex_client():
    credentials = AnonymousCredentials()
    with fake_adc(credentials), patch("ai_mates_mcp_server.providers.genai.Client") as mock_cls:
        registry = ProviderRegistry(settings(openai_api_key=None, gemini_use_gcloud_auth=True))

    mock_cls.assert_called_once_with(
        vertexai=True, credentials=credentials, project=FAKE_ADC_PROJECT
    )
    assert registry.providers["gemini"].auth_mode == "gcloud-adc"
    assert registry.providers["gemini"].api_key is None


def test_gemini_gcloud_auth_prefers_explicit_project_over_adc_project():
    credentials = AnonymousCredentials()
    with fake_adc(credentials), patch("ai_mates_mcp_server.providers.genai.Client") as mock_cls:
        ProviderRegistry(
            settings(
                openai_api_key=None,
                gemini_use_gcloud_auth=True,
                gemini_project="my-project",
                gemini_location="europe-west1",
            )
        )

    mock_cls.assert_called_once_with(
        vertexai=True,
        credentials=credentials,
        project="my-project",
        location="europe-west1",
    )


def test_gemini_gcloud_auth_survives_adc_without_a_project():
    credentials = AnonymousCredentials()
    with (
        fake_adc(credentials, project=None),
        patch("ai_mates_mcp_server.providers.genai.Client") as mock_cls,
    ):
        ProviderRegistry(settings(openai_api_key=None, gemini_use_gcloud_auth=True))

    mock_cls.assert_called_once_with(vertexai=True, credentials=credentials)


@pytest.mark.parametrize("env_var", ["GEMINI_API_KEY", "GOOGLE_API_KEY"])
def test_ambient_api_key_cannot_hijack_gcloud_auth(monkeypatch, env_var):
    """Regression test for the SDK's env-key fallback.

    Built with only ``vertexai=True``, the real client picks up an ambient key and
    skips ADC entirely, so this exercises the genuine SDK rather than a mock.
    """
    for name in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(env_var, "leftover-key")

    with fake_adc():
        provider = GeminiProvider(None, "gemini-2.5-pro", use_gcloud_auth=True)

    api_client = provider.client._api_client
    assert api_client.api_key is None
    assert api_client.project == FAKE_ADC_PROJECT
    assert api_client._credentials is not None
    assert "x-goog-api-key" not in (api_client._http_options.headers or {})


def test_gemini_gcloud_auth_resolves_provider():
    with fake_adc(), patch("ai_mates_mcp_server.providers.genai.Client"):
        registry = ProviderRegistry(settings(openai_api_key=None, gemini_use_gcloud_auth=True))

    provider, model = registry.resolve("gemini")

    assert provider.name == "gemini"
    assert model == "gemini-3.1-pro-preview"


def test_no_gemini_auth_configured_does_not_add_gemini_provider():
    registry = ProviderRegistry(
        settings(openai_api_key=None, gemini_api_key=None, gemini_use_gcloud_auth=False)
    )

    assert "gemini" not in registry.providers
    assert registry.provider_errors == {}


def test_missing_adc_does_not_break_other_providers():
    with patch(
        "ai_mates_mcp_server.providers.google.auth.default",
        side_effect=DefaultCredentialsError("no ADC"),
    ):
        registry = ProviderRegistry(settings(gemini_use_gcloud_auth=True))

    assert "gemini" not in registry.providers
    assert "gcloud auth application-default login" in registry.provider_errors["gemini"]
    provider, _ = registry.resolve("auto")
    assert provider.name == "openai"


def test_missing_adc_is_reported_when_gemini_is_requested():
    with patch(
        "ai_mates_mcp_server.providers.google.auth.default",
        side_effect=DefaultCredentialsError("no ADC"),
    ):
        registry = ProviderRegistry(settings(gemini_use_gcloud_auth=True))

    with pytest.raises(ProviderError, match="application-default login"):
        registry.resolve("gemini")


def test_missing_adc_is_reported_when_no_provider_is_configured():
    with patch(
        "ai_mates_mcp_server.providers.google.auth.default",
        side_effect=DefaultCredentialsError("no ADC"),
    ):
        registry = ProviderRegistry(settings(openai_api_key=None, gemini_use_gcloud_auth=True))

    with pytest.raises(ProviderError, match="Provider setup errors: gemini"):
        registry.resolve("auto")


def test_vertex_project_misconfiguration_is_reported_as_provider_error():
    with (
        fake_adc(project=None),
        patch(
            "ai_mates_mcp_server.providers.genai.Client",
            side_effect=ValueError("Project or API key must be set when using the Vertex AI API."),
        ),
    ):
        registry = ProviderRegistry(settings(openai_api_key=None, gemini_use_gcloud_auth=True))

    assert "GOOGLE_CLOUD_PROJECT" in registry.provider_errors["gemini"]


def test_gemini_provider_without_key_or_gcloud_auth_raises():
    with pytest.raises(ProviderError, match="needs an API key"):
        GeminiProvider(None, "gemini-2.5-pro")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("models/gemini-2.5-pro", "gemini-2.5-pro"),
        ("publishers/google/models/gemini-2.5-pro", "gemini-2.5-pro"),
        ("gemini-2.5-pro", "gemini-2.5-pro"),
        ("", ""),
    ],
)
def test_normalize_gemini_model_name(raw, expected):
    assert _normalize_gemini_model_name(raw) == expected


async def test_gemini_list_model_ids_strips_vertex_prefixes():
    with fake_adc(), patch("ai_mates_mcp_server.providers.genai.Client") as mock_client_cls:
        client = mock_client_cls.return_value
        client.models.list.return_value = [
            SimpleNamespace(name="publishers/google/models/gemini-2.5-pro"),
            SimpleNamespace(name="publishers/google/models/gemini-2.5-flash"),
        ]
        provider = GeminiProvider(None, "gemini-2.5-pro", use_gcloud_auth=True)

    assert await provider.list_model_ids() == ["gemini-2.5-flash", "gemini-2.5-pro"]


async def test_list_models_reports_auth_mode_and_errors():
    with fake_adc(), patch("ai_mates_mcp_server.providers.genai.Client"):
        registry = ProviderRegistry(settings(openai_api_key=None, gemini_use_gcloud_auth=True))

    result = await registry.list_models()

    assert result["provider_auth"] == {"gemini": "gcloud-adc"}
    assert result["provider_errors"] == {}


async def test_expired_adc_is_translated_into_actionable_error():
    with fake_adc(), patch("ai_mates_mcp_server.providers.genai.Client") as mock_client_cls:
        client = mock_client_cls.return_value
        client.models.generate_content.side_effect = RefreshError("invalid_grant")
        provider = GeminiProvider(None, "gemini-2.5-pro", use_gcloud_auth=True)

    with pytest.raises(ProviderError, match="application-default login"):
        await provider.complete("hello")


async def test_refresh_error_is_not_swallowed_for_api_key_auth():
    with patch("ai_mates_mcp_server.providers.genai.Client") as mock_client_cls:
        client = mock_client_cls.return_value
        client.models.list.side_effect = RefreshError("invalid_grant")
        provider = GeminiProvider("my-key", "gemini-2.5-pro")

    with pytest.raises(RefreshError):
        await provider.list_model_ids()
