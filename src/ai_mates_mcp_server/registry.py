from __future__ import annotations

import json
from dataclasses import dataclass, replace
from importlib.resources import files
from pathlib import Path
from typing import Any

from .models import ProviderName

ACTIVE_STATUSES = {"active", "preview"}
DEPRECATED_STATUSES = {"deprecated", "shutdown", "retired"}


class ModelRegistryError(ValueError):
    pass


@dataclass(frozen=True)
class ModelEntry:
    id: str
    provider: ProviderName
    aliases: tuple[str, ...] = ()
    rank: int = 0
    status: str = "active"
    source: str = "packaged"
    live_discovered: bool = False
    description: str | None = None
    deprecation_date: str | None = None
    shutdown_date: str | None = None
    replacement_models: tuple[str, ...] = ()
    source_url: str | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any], *, source: str) -> ModelEntry:
        try:
            model_id = str(data["id"]).strip()
            provider = str(data["provider"]).strip().lower()
        except KeyError as exc:
            raise ModelRegistryError(f"Model entry missing required field: {exc.args[0]}") from exc

        if provider not in {"openai", "anthropic", "gemini"}:
            raise ModelRegistryError(f"Unsupported provider for model '{model_id}': {provider}")
        if not model_id:
            raise ModelRegistryError("Model entry id cannot be empty")

        aliases = _string_list(data.get("aliases"), field="aliases", model_id=model_id)
        replacement_models = _string_list(
            data.get("replacement_models"),
            field="replacement_models",
            model_id=model_id,
            normalize=False,
        )

        return cls(
            id=model_id,
            provider=provider,  # type: ignore[arg-type]
            aliases=tuple(aliases),
            rank=int(data.get("rank", 0)),
            status=str(data.get("status", "active")).strip().lower(),
            source=source,
            description=data.get("description"),
            deprecation_date=data.get("deprecation_date"),
            shutdown_date=data.get("shutdown_date"),
            replacement_models=tuple(replacement_models),
            source_url=data.get("source_url"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "aliases": list(self.aliases),
            "rank": self.rank,
            "status": self.status,
            "source": self.source,
            "live_discovered": self.live_discovered,
            "description": self.description,
            "deprecation_date": self.deprecation_date,
            "shutdown_date": self.shutdown_date,
            "replacement_models": list(self.replacement_models),
            "source_url": self.source_url,
        }


class ModelRegistry:
    def __init__(
        self,
        *,
        local_models_file: str | None = None,
        provider_defaults: dict[ProviderName, str] | None = None,
        allow_deprecated: bool = False,
    ) -> None:
        self.allow_deprecated = allow_deprecated
        self.entries: dict[str, ModelEntry] = {}
        self.deprecated_entries: dict[str, ModelEntry] = {}
        self.aliases: dict[str, str] = {}
        self.defaults: dict[ProviderName, str] = {}
        self._load_packaged()
        if local_models_file:
            self._load_file(Path(local_models_file).expanduser(), source="local")
        if provider_defaults:
            for provider, model_id in provider_defaults.items():
                if model_id:
                    self.defaults[provider] = _validate_default_model_id(
                        provider,
                        model_id,
                        source="settings",
                    )

    def resolve(self, model: str) -> ModelEntry | None:
        key = _normalize_key(model)
        entry = self.deprecated_entries.get(key)
        if entry:
            self._validate_status(entry)
            return entry

        entry = self.entries.get(key)
        if not entry:
            target = self.aliases.get(key)
            if target:
                entry = self.deprecated_entries.get(target)
                if entry:
                    self._validate_status(entry)
                    return entry
                entry = self.entries.get(target)
        if entry:
            self._validate_status(entry)
        return entry

    def default_for_provider(self, provider: ProviderName) -> str | None:
        model_id = self.defaults.get(provider)
        if not model_id:
            return None
        entry = self.resolve(model_id)
        return entry.id if entry else model_id

    def provider_aliases(self) -> dict[ProviderName, str]:
        return dict(self.defaults)

    def add_live_model(self, provider: ProviderName, model_id: str) -> None:
        key = _normalize_key(model_id)
        if key in self.deprecated_entries:
            return
        if key in self.entries:
            self.entries[key] = replace(self.entries[key], live_discovered=True)
            return
        self.entries[key] = ModelEntry(
            id=model_id,
            provider=provider,
            rank=0,
            status="active",
            source="live",
            live_discovered=True,
        )

    def list_entries(
        self,
        configured_providers: set[ProviderName] | None = None,
    ) -> list[dict[str, Any]]:
        rows = []
        for entry in self.entries.values():
            if _normalize_key(entry.id) in self.deprecated_entries:
                continue
            row = entry.to_dict()
            row["configured"] = (
                entry.provider in configured_providers if configured_providers is not None else None
            )
            row["is_default"] = self.defaults.get(entry.provider) == entry.id
            rows.append(row)
        for entry in self.deprecated_entries.values():
            row = entry.to_dict()
            row["configured"] = (
                entry.provider in configured_providers if configured_providers is not None else None
            )
            row["is_default"] = False
            rows.append(row)
        rows.sort(key=lambda item: (item["provider"], -item["rank"], item["id"]))
        return rows

    def _load_packaged(self) -> None:
        data_path = files("ai_mates_mcp_server").joinpath("data/models.json")
        data = json.loads(data_path.read_text(encoding="utf-8"))
        self._merge_data(data, source="packaged")

    def _load_file(self, path: Path, *, source: str) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ModelRegistryError(f"Could not read model registry file '{path}': {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ModelRegistryError(
                f"Invalid JSON in model registry file '{path}': {exc}"
            ) from exc
        self._merge_data(data, source=source)

    def _merge_data(self, data: dict[str, Any], *, source: str) -> None:
        for provider, model_id in data.get("defaults", {}).items():
            normalized_provider = provider.strip().lower()
            if normalized_provider in {"openai", "anthropic", "gemini"}:
                self.defaults[normalized_provider] = _validate_default_model_id(
                    normalized_provider,
                    model_id,
                    source=source,
                )

        for raw_entry in data.get("models", []):
            entry = ModelEntry.from_mapping(raw_entry, source=source)
            key = _normalize_key(entry.id)
            old_entry = self.entries.get(key)
            if old_entry:
                for alias in old_entry.aliases:
                    if self.aliases.get(alias) == key:
                        del self.aliases[alias]
            self.entries[key] = entry
            self.aliases[key] = key
            for alias in entry.aliases:
                self.aliases[alias] = key

        for raw_entry in data.get("deprecated_models", []):
            entry = ModelEntry.from_mapping(raw_entry, source=source)
            key = _normalize_key(entry.id)
            self.deprecated_entries[key] = entry

    def _validate_status(self, entry: ModelEntry) -> None:
        if entry.status in ACTIVE_STATUSES:
            return
        if entry.status in DEPRECATED_STATUSES and self.allow_deprecated:
            return
        if entry.status in DEPRECATED_STATUSES:
            raise ModelRegistryError(
                f"Model '{entry.id}' is marked {entry.status}. "
                "Set MATES_ALLOW_DEPRECATED_MODELS=true to use it anyway."
            )
        allowed_statuses = sorted(ACTIVE_STATUSES | DEPRECATED_STATUSES)
        raise ModelRegistryError(
            f"Model '{entry.id}' has unsupported status '{entry.status}'. "
            f"Allowed statuses: {', '.join(allowed_statuses)}."
        )


def _normalize_key(value: str) -> str:
    return value.strip().lower()


def _string_list(
    raw: Any,
    *,
    field: str,
    model_id: str,
    normalize: bool = True,
) -> list[str]:
    """Parse a model entry's list-of-strings field, dropping blank values."""
    if raw is None:
        raw = []
    if not isinstance(raw, (list, tuple)):
        raise ModelRegistryError(f"Model entry {field} for '{model_id}' must be an array")
    values = []
    for value in raw:
        if not isinstance(value, str):
            raise ModelRegistryError(f"Model entry {field} for '{model_id}' must be strings")
        if value.strip():
            values.append(_normalize_key(value) if normalize else value.strip())
    return values


def _validate_default_model_id(provider: str, model_id: Any, *, source: str) -> str:
    if not isinstance(model_id, str) or not model_id.strip():
        raise ModelRegistryError(
            f"Invalid default model id for provider '{provider}' in {source} data: {model_id!r}"
        )
    return model_id.strip()
