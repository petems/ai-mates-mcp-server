from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

ProviderName = Literal["openai", "anthropic", "gemini"]
Stance = Literal["for", "against", "neutral"]
Severity = Literal["critical", "high", "medium", "low", "all"]


class ModelConfig(BaseModel):
    model: str = Field(..., description="Model name or provider alias to consult.")
    stance: Stance = Field("neutral", description="Perspective the model should take.")
    stance_prompt: str | None = Field(None, description="Custom stance instructions.")


class Issue(BaseModel):
    severity: Severity = "medium"
    description: str
    file: str | None = None
    line: int | None = None
    recommendation: str | None = None


class ProviderResponse(BaseModel):
    provider: ProviderName
    model: str
    content: str
    usage: dict | None = None


class ToolEnvelope(BaseModel):
    status: str
    continuation_id: str | None = None
    next_steps: str | None = None
    data: dict


class PlannerRequest(BaseModel):
    step: str
    step_number: int = Field(..., ge=1)
    total_steps: int = Field(..., ge=1)
    next_step_required: bool
    continuation_id: str | None = None
    is_step_revision: bool = False
    revises_step_number: int | None = Field(None, ge=1)
    is_branch_point: bool = False
    branch_from_step: int | None = Field(None, ge=1)
    branch_id: str | None = None
    more_steps_needed: bool = False


class ConsensusRequest(BaseModel):
    proposal: str = Field(..., description="Exact question or proposal for all models.")
    models: list[ModelConfig]
    initial_analysis: str | None = Field(None, description="Your own neutral analysis.")
    relevant_files: list[str] = Field(default_factory=list)
    workspace_root: str | None = Field(
        None,
        description="Workspace directory that bounds relevant_files access.",
    )
    continuation_id: str | None = None

    @field_validator("models")
    @classmethod
    def require_two_models(cls, value: list[ModelConfig]) -> list[ModelConfig]:
        if len(value) < 2:
            raise ValueError("consensus requires at least two model entries")
        seen = set()
        for model in value:
            key = (model.model, model.stance)
            if key in seen:
                raise ValueError(f"duplicate model and stance: {model.model}:{model.stance}")
            seen.add(key)
        return value


class CodeReviewRequest(BaseModel):
    step: str = Field(..., description="Review request or current review narrative.")
    findings: str = Field("", description="Findings gathered so far.")
    relevant_files: list[str] = Field(default_factory=list)
    workspace_root: str | None = Field(
        None,
        description="Workspace directory that bounds relevant_files access.",
    )
    files_checked: list[str] = Field(default_factory=list)
    relevant_context: list[str] = Field(default_factory=list)
    issues_found: list[Issue] = Field(default_factory=list)
    review_type: Literal["full", "security", "performance", "quick"] = "full"
    focus_on: str | None = None
    standards: str | None = None
    severity_filter: Severity = "all"
    model: str | None = None
    use_assistant_model: bool = True
    continuation_id: str | None = None
