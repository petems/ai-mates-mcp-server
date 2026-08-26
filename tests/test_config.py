from __future__ import annotations

import pytest

from ai_mates_mcp_server.config import load_settings

GEMINI_ENV_VARS = (
    "GEMINI_API_KEY",
    "GEMINI_USE_GCLOUD_AUTH",
    "GOOGLE_GENAI_USE_VERTEXAI",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
)


@pytest.fixture(autouse=True)
def clean_gemini_env(monkeypatch):
    """A developer .env is loaded at import time, so start every case from empty."""
    for name in GEMINI_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_gcloud_auth_defaults_to_off():
    settings = load_settings()

    assert settings.gemini_use_gcloud_auth is False
    assert settings.gemini_project is None
    assert settings.gemini_location is None


@pytest.mark.parametrize("raw", ["true", "TRUE", " yes ", "1", "on"])
def test_gcloud_auth_truthy_values(monkeypatch, raw):
    monkeypatch.setenv("GEMINI_USE_GCLOUD_AUTH", raw)

    assert load_settings().gemini_use_gcloud_auth is True


@pytest.mark.parametrize("raw", ["false", "0", "no", "off", ""])
def test_gcloud_auth_falsy_values(monkeypatch, raw):
    monkeypatch.setenv("GEMINI_USE_GCLOUD_AUTH", raw)

    assert load_settings().gemini_use_gcloud_auth is False


def test_sdk_vertex_env_var_also_enables_gcloud_auth(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")

    assert load_settings().gemini_use_gcloud_auth is True


def test_project_and_location_are_read_from_google_cloud_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west1")

    settings = load_settings()

    assert settings.gemini_project == "my-project"
    assert settings.gemini_location == "europe-west1"
