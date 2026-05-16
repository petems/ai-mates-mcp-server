from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from anthropic import AsyncAnthropic
from google import genai
from google.genai import types as genai_types
from openai import AsyncOpenAI

from .config import DEFAULT_MAX_TOKENS, Settings, load_settings
from .models import ProviderName, ProviderResponse
from .registry import ModelRegistry, ModelRegistryError


class ProviderError(RuntimeError):
    pass


class ModelProvider(ABC):
    name: ProviderName

    def __init__(self, api_key: str, default_model: str) -> None:
        self.api_key = api_key
        self.default_model = default_model

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.2,
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
        temperature: float = 0.2,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> ProviderResponse:
        model_name = model or self.default_model
        try:
            response = await self.client.responses.create(
                model=model_name,
                input=prompt,
                instructions=system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            content = _extract_openai_response_text(response)
            usage = _dump_usage(getattr(response, "usage", None))
        except AttributeError:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
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
        temperature: float = 0.2,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> ProviderResponse:
        model_name = model or self.default_model
        response = await self.client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
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

    def __init__(self, api_key: str, default_model: str) -> None:
        super().__init__(api_key, default_model)
        self.client = genai.Client(api_key=api_key)

    async def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> ProviderResponse:
        model_name = model or self.default_model

        def call() -> Any:
            config = genai_types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_prompt,
            )
            return self.client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )

        response = await asyncio.to_thread(call)
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
                name = getattr(model, "name", "")
                if name.startswith("models/"):
                    name = name.removeprefix("models/")
                if name:
                    model_ids.append(name)
            return sorted(set(model_ids))

        return await asyncio.to_thread(call)


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
        if self.settings.gemini_api_key:
            self.providers["gemini"] = GeminiProvider(
                self.settings.gemini_api_key,
                self._provider_default("gemini"),
            )

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
                "No AI provider configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, "
                "or GEMINI_API_KEY."
            )

        normalized = requested.lower()
        if normalized in {"openai", "anthropic", "gemini"}:
            provider_name = normalized  # type: ignore[assignment]
            default_model = self.model_registry.default_for_provider(provider_name)
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
            raise ProviderError(
                f"Model '{model}' requires {provider_name}, but that provider is not configured."
            )
        return provider, model

    async def list_models(self) -> dict[str, Any]:
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
            "discovery": self.settings.model_discovery,
            "live_errors": live_errors,
            "models": self.model_registry.list_entries(set(self.providers)),
        }

    def _provider_default(self, provider_name: ProviderName) -> str:
        return self.model_registry.default_for_provider(provider_name) or getattr(
            self.settings, f"{provider_name}_model"
        )


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
