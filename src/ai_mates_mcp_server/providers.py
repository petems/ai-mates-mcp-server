from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import google.auth
from anthropic import AsyncAnthropic
from google import genai
from google.auth import exceptions as google_auth_exceptions
from google.genai import types as genai_types
from openai import AsyncOpenAI

from .config import DEFAULT_MAX_TOKENS, Settings, load_settings
from .models import ProviderName, ProviderResponse
from .registry import ModelRegistry, ModelRegistryError


class ProviderError(RuntimeError):
    pass


UNSUPPORTED_TEMPERATURE_PREFIXES: dict[ProviderName, tuple[str, ...]] = {
    "openai": ("gpt-5", "o1", "o3", "o4"),
    "anthropic": ("claude-opus-4", "claude-sonnet-4", "claude-haiku-4"),
    "gemini": (),
}


class ModelProvider(ABC):
    name: ProviderName
    auth_mode: str = "api-key"

    def __init__(self, api_key: str | None, default_model: str) -> None:
        self.api_key = api_key
        self.default_model = default_model

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> ProviderResponse:
        pass

    @abstractmethod
    async def list_model_ids(self) -> list[str]:
        pass


class OpenAIProvider(ModelProvider):
    name: ProviderName = "openai"

    def __init__(self, api_key: str, default_model: str) -> None:
        super().__init__(api_key, default_model)
        self.client = AsyncOpenAI(api_key=api_key)

    async def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> ProviderResponse:
        model_name = model or self.default_model
        if hasattr(self.client, "responses"):
            kwargs: dict[str, Any] = {
                "model": model_name,
                "input": prompt,
                "instructions": system_prompt,
                "max_output_tokens": max_tokens,
                **_temperature_kwargs(self.name, model_name, temperature),
            }
            response = await self.client.responses.create(**kwargs)
            content = _extract_openai_response_text(response)
            usage = _dump_usage(getattr(response, "usage", None))
        else:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            kwargs: dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                **_temperature_kwargs(self.name, model_name, temperature),
            }
            response = await self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            usage = _dump_usage(getattr(response, "usage", None))

        return ProviderResponse(provider=self.name, model=model_name, content=content, usage=usage)

    async def list_model_ids(self) -> list[str]:
        response = await self.client.models.list()
        return sorted(model.id for model in response.data)


class AnthropicProvider(ModelProvider):
    name: ProviderName = "anthropic"

    def __init__(self, api_key: str, default_model: str) -> None:
        super().__init__(api_key, default_model)
        self.client = AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> ProviderResponse:
        model_name = model or self.default_model
        kwargs: dict[str, Any] = {
            "model": model_name,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}],
            **_temperature_kwargs(self.name, model_name, temperature),
        }
        response = await self.client.messages.create(**kwargs)
        text_parts = [
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ]
        return ProviderResponse(
            provider=self.name,
            model=model_name,
            content="\n".join(text_parts),
            usage=_dump_usage(getattr(response, "usage", None)),
        )

    async def list_model_ids(self) -> list[str]:
        response = await self.client.models.list()
        data = getattr(response, "data", response)
        return sorted(model.id for model in data)


class GeminiProvider(ModelProvider):
    name: ProviderName = "gemini"

    def __init__(
        self,
        api_key: str | None,
        default_model: str,
        *,
        use_gcloud_auth: bool = False,
        project: str | None = None,
        location: str | None = None,
    ) -> None:
        super().__init__(None if use_gcloud_auth else api_key, default_model)
        self._project = project
        self._location = location
        self._reload_lock = asyncio.Lock()
        if use_gcloud_auth:
            self.auth_mode = "gcloud-adc"
            self.client = _gemini_vertex_client(project=project, location=location)
        else:
            if not api_key:
                raise ProviderError(
                    "GeminiProvider needs an API key unless use_gcloud_auth is enabled."
                )
            self.client = genai.Client(api_key=api_key)

    async def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> ProviderResponse:
        model_name = model or self.default_model

        def call() -> Any:
            config = genai_types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                system_instruction=system_prompt,
                **_temperature_kwargs(self.name, model_name, temperature),
            )
            return self.client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )

        response = await self._call(call)
        return ProviderResponse(
            provider=self.name,
            model=model_name,
            content=getattr(response, "text", "") or "",
            usage=_dump_usage(getattr(response, "usage_metadata", None)),
        )

    async def list_model_ids(self) -> list[str]:
        def call() -> list[str]:
            model_ids: list[str] = []
            for model in self.client.models.list():
                name = _normalize_gemini_model_name(getattr(model, "name", ""))
                if name:
                    model_ids.append(name)
            return sorted(set(model_ids))

        return await self._call(call)

    async def _call(self, fn: Callable[[], Any]) -> Any:
        """Run a blocking SDK call, reloading ADC once if the credential is stale.

        Access tokens are refreshed by google-auth transparently, so a RefreshError
        means the underlying grant is gone: revoked, or expired under an org reauth
        policy. Re-running the login command writes a fresh ADC file, but the
        credential object loaded at construction keeps the dead refresh token, so
        the client is rebuilt from disk before giving up. That way logging in again
        is enough, with no server restart.
        """
        try:
            return await asyncio.to_thread(fn)
        except google_auth_exceptions.RefreshError as exc:
            if self.auth_mode != "gcloud-adc":
                raise
            await self._reload_credentials()
            try:
                return await asyncio.to_thread(fn)
            except google_auth_exceptions.RefreshError as retry_exc:
                raise ProviderError(
                    "Gemini gcloud credentials could not be refreshed, and reloading "
                    "Application Default Credentials did not help. Run "
                    f"`gcloud auth application-default login` again ({exc})."
                ) from retry_exc

    async def _reload_credentials(self) -> None:
        """Rebuild the client so a freshly written ADC file is picked up."""
        async with self._reload_lock:
            self.client = await asyncio.to_thread(
                _gemini_vertex_client, project=self._project, location=self._location
            )


class ProviderRegistry:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.model_registry = ModelRegistry(
            local_models_file=self.settings.models_file,
            provider_defaults={
                "openai": self.settings.openai_model,
                "anthropic": self.settings.anthropic_model,
                "gemini": self.settings.gemini_model,
            },
            allow_deprecated=self.settings.allow_deprecated_models,
        )
        self.providers: dict[ProviderName, ModelProvider] = {}
        self.provider_errors: dict[ProviderName, str] = {}
        if self.settings.openai_api_key:
            self.providers["openai"] = OpenAIProvider(
                self.settings.openai_api_key,
                self._provider_default("openai"),
            )
        if self.settings.anthropic_api_key:
            self.providers["anthropic"] = AnthropicProvider(
                self.settings.anthropic_api_key,
                self._provider_default("anthropic"),
            )
        if self.settings.gemini_use_gcloud_auth or self.settings.gemini_api_key:
            self._configure_gemini()

    def _configure_gemini(self) -> None:
        """Build the Gemini provider, preferring gcloud ADC when it is opted into.

        A broken ADC setup must not take the whole server down, so the failure is
        recorded and surfaced when Gemini is actually requested.
        """
        try:
            self.providers["gemini"] = GeminiProvider(
                self.settings.gemini_api_key,
                self._provider_default("gemini"),
                use_gcloud_auth=self.settings.gemini_use_gcloud_auth,
                project=self.settings.gemini_project,
                location=self.settings.gemini_location,
            )
        except ProviderError as exc:
            self.provider_errors["gemini"] = str(exc)

    def available_models(self) -> dict[str, str]:
        return self.model_registry.provider_aliases()

    def resolve(self, model: str | None = None) -> tuple[ModelProvider, str]:
        requested = model or self.settings.default_model
        if not requested or requested == "auto":
            for provider_name in ("openai", "anthropic", "gemini"):
                provider = self.providers.get(provider_name)
                if provider:
                    return provider, provider.default_model
            raise ProviderError(
                self._augment_error(
                    "No AI provider configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, "
                    "GEMINI_API_KEY, or GEMINI_USE_GCLOUD_AUTH=true."
                )
            )

        normalized = requested.lower()
        if normalized in {"openai", "anthropic", "gemini"}:
            provider_name = normalized  # type: ignore[assignment]
            default_model = self._provider_default_or_none(provider_name)
            if not default_model and provider_name in self.providers:
                default_model = self.providers[provider_name].default_model
            if not default_model:
                configured = ", ".join(sorted(self.available_models())) or "none"
                raise ProviderError(
                    f"Cannot route provider alias '{requested}'. Configured models/providers: "
                    f"{configured}"
                )
            return self._resolve_provider_model(provider_name, default_model)

        try:
            registry_entry = self.model_registry.resolve(requested)
        except ModelRegistryError as exc:
            raise ProviderError(str(exc)) from exc
        if registry_entry:
            return self._resolve_provider_model(registry_entry.provider, registry_entry.id)

        if normalized.startswith(("gpt-", "o1", "o3", "o4")):
            return self._resolve_provider_model("openai", requested)
        if normalized.startswith("claude"):
            return self._resolve_provider_model("anthropic", requested)
        if normalized.startswith("gemini"):
            return self._resolve_provider_model("gemini", requested)

        configured = ", ".join(sorted(self.available_models())) or "none"
        raise ProviderError(
            f"Cannot route model '{requested}'. Configured models/providers: {configured}"
        )

    def _resolve_provider_model(
        self,
        provider_name: ProviderName,
        model: str,
    ) -> tuple[ModelProvider, str]:
        provider = self.providers.get(provider_name)
        if not provider:
            detail = self.provider_errors.get(provider_name)
            message = (
                f"Model '{model}' requires {provider_name}, but that provider is not configured."
            )
            raise ProviderError(f"{message} {detail}" if detail else message)
        return provider, model

    def _augment_error(self, message: str) -> str:
        if not self.provider_errors:
            return message
        details = "; ".join(
            f"{name}: {error}" for name, error in sorted(self.provider_errors.items())
        )
        return f"{message} Provider setup errors: {details}"

    async def list_models(self, *, include_deprecated: bool = False) -> dict[str, Any]:
        live_errors: dict[str, str] = {}
        if self.settings.model_discovery == "list":
            for provider_name, provider in self.providers.items():
                try:
                    for model_id in await provider.list_model_ids():
                        self.model_registry.add_live_model(provider_name, model_id)
                except Exception as exc:
                    live_errors[provider_name] = str(exc)

        return {
            "defaults": self.model_registry.provider_aliases(),
            "configured_providers": sorted(self.providers),
            "provider_auth": self._provider_auth_report(),
            "provider_errors": dict(sorted(self.provider_errors.items())),
            "discovery": self.settings.model_discovery,
            "live_errors": live_errors,
            "models": self.model_registry.list_entries(
                set(self.providers),
                include_deprecated=include_deprecated,
            ),
            "deprecated_model_count": len(self.model_registry.deprecated_entries),
            "deprecated_models_included": include_deprecated,
        }

    def _provider_auth_report(self) -> dict[str, dict[str, Any]]:
        return {
            "openai": self._api_key_auth_report("OPENAI_API_KEY", self.settings.openai_api_key),
            "anthropic": self._api_key_auth_report(
                "ANTHROPIC_API_KEY",
                self.settings.anthropic_api_key,
            ),
            "gemini": self._gemini_auth_report(),
        }

    def _api_key_auth_report(self, source: str, api_key: str | None) -> dict[str, Any]:
        return {
            "mode": "api-key",
            "state": "present" if api_key else "missing",
            "source": source if api_key else None,
        }

    def _gemini_auth_report(self) -> dict[str, Any]:
        if not self.settings.gemini_use_gcloud_auth:
            return self._api_key_auth_report("GEMINI_API_KEY", self.settings.gemini_api_key)

        report: dict[str, Any] = {
            "mode": "gcloud-adc",
            "state": "unknown",
            "quota_project": None,
            "credential_type": "unknown",
        }
        try:
            credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            report["quota_project"] = getattr(credentials, "quota_project_id", None)
            report["credential_type"] = self._credential_type(credentials)
            expiry = getattr(credentials, "expiry", None)
            if isinstance(expiry, datetime):
                if expiry.tzinfo is None:
                    expiry = expiry.replace(tzinfo=UTC)
                report["expires_at"] = expiry.astimezone(UTC).isoformat().replace(
                    "+00:00",
                    "Z",
                )
            expired = bool(getattr(credentials, "expired", False))
            if expired:
                report["state"] = "expired"
                report["hint"] = "Run `gcloud auth application-default login` again."
            else:
                report["state"] = "valid"
        except google_auth_exceptions.DefaultCredentialsError:
            report["state"] = "missing"
            report["hint"] = (
                "Run `gcloud auth application-default login` and set a project with "
                "`gcloud config set project <project>` or GOOGLE_CLOUD_PROJECT."
            )
        except Exception:
            report["state"] = "unknown"
            report["hint"] = (
                "Could not inspect gcloud ADC credentials locally. Run "
                "`gcloud auth application-default login` and retry."
            )
        return report

    def _credential_type(self, credentials: Any) -> str:
        if hasattr(credentials, "service_account_email"):
            return "service_account"
        module = getattr(getattr(credentials, "__class__", None), "__module__", "")
        if module.startswith("google.oauth2.credentials"):
            return "authorized_user"
        return "unknown"

    def _provider_default_or_none(self, provider_name: ProviderName) -> str | None:
        try:
            return self.model_registry.default_for_provider(provider_name)
        except ModelRegistryError as exc:
            raise ProviderError(
                f"Configured default model for provider '{provider_name}' is unusable: {exc}"
            ) from exc

    def _provider_default(self, provider_name: ProviderName) -> str:
        return self._provider_default_or_none(provider_name) or getattr(
            self.settings, f"{provider_name}_model"
        )


def _gemini_vertex_client(*, project: str | None, location: str | None) -> genai.Client:
    """Create a Vertex AI client backed by Application Default Credentials.

    Plain ``genai.Client()`` talks to the Gemini Developer API, which only accepts
    API keys. ADC (``gcloud auth application-default login``) is a Vertex AI path,
    so ``vertexai=True`` is required for gcloud credentials to be used at all.

    Credentials and project are resolved here and passed explicitly rather than
    left to the SDK. Given only ``vertexai=True``, the SDK falls back to
    ``GEMINI_API_KEY``/``GOOGLE_API_KEY`` from the environment and never loads ADC
    at all, which would quietly authenticate with the key this mode exists to avoid.
    """
    credentials, adc_project = _load_adc()
    kwargs: dict[str, Any] = {"vertexai": True, "credentials": credentials}
    resolved_project = project or adc_project
    if resolved_project:
        kwargs["project"] = resolved_project
    if location:
        kwargs["location"] = location
    try:
        return genai.Client(**kwargs)
    except ValueError as exc:
        raise ProviderError(
            f"Gemini gcloud auth is enabled but the Vertex AI client could not be built: {exc} "
            "Set GOOGLE_CLOUD_PROJECT (and optionally GOOGLE_CLOUD_LOCATION)."
        ) from exc


def _load_adc() -> tuple[Any, str | None]:
    try:
        return google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    except google_auth_exceptions.DefaultCredentialsError as exc:
        raise ProviderError(
            "Gemini gcloud auth is enabled but no Application Default Credentials were "
            "found. Run `gcloud auth application-default login` (and "
            "`gcloud config set project <project>` or set GOOGLE_CLOUD_PROJECT)."
        ) from exc


def _normalize_gemini_model_name(name: str) -> str:
    """Strip the Developer API (`models/`) and Vertex (`publishers/...`) prefixes."""
    if not name:
        return ""
    if "/models/" in name:
        return name.rsplit("/models/", 1)[1]
    return name.removeprefix("models/")


def _extract_openai_response_text(response: Any) -> str:
    direct = getattr(response, "output_text", None)
    if direct:
        return direct

    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts)


def _temperature_kwargs(
    provider: ProviderName,
    model: str,
    temperature: float | None,
) -> dict[str, float]:
    if temperature is None or not _supports_temperature(provider, model):
        return {}
    return {"temperature": temperature}


def _supports_temperature(provider: ProviderName, model: str) -> bool:
    normalized = model.lower()
    return not normalized.startswith(UNSUPPORTED_TEMPERATURE_PREFIXES[provider])


def _dump_usage(usage: Any) -> dict | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "to_dict"):
        return usage.to_dict()
    if isinstance(usage, dict):
        return usage
    return None
