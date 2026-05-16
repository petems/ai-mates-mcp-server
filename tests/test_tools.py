from __future__ import annotations

import json

import pytest

from ai_mates_mcp_server.models import ProviderResponse
from ai_mates_mcp_server.tools import run_codereview, run_consensus, run_listmodels, run_planner


class FakeProvider:
    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.calls = []

    async def complete(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return ProviderResponse(
            provider=self.provider,
            model=kwargs["model"],
            content=f"{self.provider} response",
            usage={"input_tokens": 1, "output_tokens": 2},
        )


class FakeRegistry:
    def __init__(self) -> None:
        self.openai = FakeProvider("openai")
        self.anthropic = FakeProvider("anthropic")
        self.gemini = FakeProvider("gemini")

    def resolve(self, model=None):
        if model in (None, "auto", "openai", "gpt-4.1"):
            return self.openai, "gpt-4.1"
        if model in ("anthropic", "claude-sonnet-4-5"):
            return self.anthropic, "claude-sonnet-4-5"
        if model in ("gemini", "gemini-2.5-pro"):
            return self.gemini, "gemini-2.5-pro"
        raise AssertionError(f"unexpected model: {model}")

    def available_models(self):
        return {
            "openai": "gpt-4.1",
            "anthropic": "claude-sonnet-4-5",
            "gemini": "gemini-2.5-pro",
        }

    async def list_models(self):
        return {
            "defaults": self.available_models(),
            "configured_providers": ["openai", "anthropic", "gemini"],
            "discovery": "off",
            "live_errors": {},
            "models": [
                {
                    "id": "gpt-4.1",
                    "provider": "openai",
                    "aliases": ["openai"],
                    "configured": True,
                    "is_default": True,
                }
            ],
        }


@pytest.mark.asyncio
async def test_planner_completes_without_provider():
    result = json.loads(
        await run_planner(
            step="Ship a minimal MCP package.",
            step_number=1,
            total_steps=1,
            next_step_required=False,
        )
    )

    assert result["status"] == "planning_complete"
    assert result["continuation_id"]
    assert result["data"]["planning_complete"] is True


@pytest.mark.asyncio
async def test_consensus_consults_multiple_models():
    registry = FakeRegistry()
    result = json.loads(
        await run_consensus(
            proposal="Should we keep the scope small?",
            models=[
                {"model": "openai", "stance": "for"},
                {"model": "anthropic", "stance": "against"},
            ],
            registry=registry,
        )
    )

    assert result["status"] == "consensus_complete"
    assert [response["provider"] for response in result["data"]["responses"]] == [
        "openai",
        "anthropic",
    ]
    assert len(registry.openai.calls) == 1
    assert len(registry.anthropic.calls) == 1


@pytest.mark.asyncio
async def test_codereview_can_run_internal_only():
    result = json.loads(
        await run_codereview(
            step="Review these changes.",
            findings="No issues found.",
            relevant_files=[],
            use_assistant_model=False,
        )
    )

    assert result["status"] == "code_review_complete"
    assert result["data"]["assistant_validation"] is None


@pytest.mark.asyncio
async def test_codereview_calls_assistant_model_when_enabled():
    registry = FakeRegistry()
    result = json.loads(
        await run_codereview(
            step="Review these changes.",
            findings="Potential issue in auth.",
            model="gemini",
            registry=registry,
        )
    )

    assert result["data"]["assistant_validation"]["provider"] == "gemini"
    assert registry.gemini.calls[0]["model"] == "gemini-2.5-pro"


@pytest.mark.asyncio
async def test_listmodels_uses_registry():
    result = json.loads(await run_listmodels(FakeRegistry()))

    assert result["defaults"]["openai"] == "gpt-4.1"
    assert result["models"][0]["aliases"] == ["openai"]
