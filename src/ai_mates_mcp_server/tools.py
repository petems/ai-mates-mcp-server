from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import ValidationError

from .context import build_context
from .memory import store
from .models import (
    CodeReviewRequest,
    ConsensusRequest,
    ModelConfig,
    PlannerRequest,
    ToolEnvelope,
)
from .providers import ProviderError, ProviderRegistry

PLANNER_NEXT_STEPS = (
    "Continue the plan if next_step_required is true. When complete, present a concise plan with "
    "clear phases, dependencies, risks, and first implementation steps."
)

CONSENSUS_SYSTEM_PROMPT = """You are participating in a multi-model consensus workflow.
Respond independently. Do not assume other models agree with you.
Be specific, weigh trade-offs, and end with a clear recommendation."""

CODEREVIEW_SYSTEM_PROMPT = """You are a senior code reviewer.
Prioritize correctness, security, maintainability, performance, and architectural fit.
Lead with concrete findings by severity.
Include file and line references when evidence supports them.
Avoid style-only comments unless they create real maintenance risk."""


async def run_planner(
    *,
    step: str,
    step_number: int,
    total_steps: int,
    next_step_required: bool,
    continuation_id: str | None = None,
    is_step_revision: bool = False,
    revises_step_number: int | None = None,
    is_branch_point: bool = False,
    branch_from_step: int | None = None,
    branch_id: str | None = None,
    more_steps_needed: bool = False,
) -> str:
    request = PlannerRequest(
        step=step,
        step_number=step_number,
        total_steps=total_steps,
        next_step_required=next_step_required,
        continuation_id=continuation_id,
        is_step_revision=is_step_revision,
        revises_step_number=revises_step_number,
        is_branch_point=is_branch_point,
        branch_from_step=branch_from_step,
        branch_id=branch_id,
        more_steps_needed=more_steps_needed,
    )
    thread = store.create_or_get("planner", request.continuation_id)
    data = {
        "step_number": request.step_number,
        "total_steps": request.total_steps,
        "step_content": request.step,
        "next_step_required": request.next_step_required,
        "planning_complete": not request.next_step_required,
        "revision": {
            "is_step_revision": request.is_step_revision,
            "revises_step_number": request.revises_step_number,
        },
        "branch": {
            "is_branch_point": request.is_branch_point,
            "branch_from_step": request.branch_from_step,
            "branch_id": request.branch_id,
        },
        "more_steps_needed": request.more_steps_needed,
    }
    store.add_turn(thread.id, "planner", data)
    envelope = ToolEnvelope(
        status="planning_complete" if not request.next_step_required else "planning_in_progress",
        continuation_id=thread.id,
        next_steps=PLANNER_NEXT_STEPS,
        data=data,
    )
    return envelope.model_dump_json(indent=2)


async def run_consensus(
    *,
    proposal: str,
    models: list[dict[str, Any]],
    initial_analysis: str | None = None,
    relevant_files: list[str] | None = None,
    continuation_id: str | None = None,
    registry: ProviderRegistry | None = None,
) -> str:
    try:
        request = ConsensusRequest(
            proposal=proposal,
            models=[ModelConfig.model_validate(model) for model in models],
            initial_analysis=initial_analysis,
            relevant_files=relevant_files or [],
            continuation_id=continuation_id,
        )
    except ValidationError as exc:
        raise ValueError(exc.errors()) from exc

    thread = store.create_or_get("consensus", request.continuation_id)
    provider_registry = registry or ProviderRegistry()
    context = build_context(request.relevant_files, store.context(thread.id))
    prompt = _join_prompt(
        request.proposal,
        ("Initial neutral analysis:\n" + request.initial_analysis)
        if request.initial_analysis
        else "",
        context,
    )

    async def consult(model_config: ModelConfig) -> dict[str, Any]:
        provider, routed_model = provider_registry.resolve(model_config.model)
        system_prompt = _stance_prompt(model_config)
        response = await provider.complete(
            prompt,
            model=routed_model,
            system_prompt=system_prompt,
            temperature=0.2,
        )
        return {
            "requested_model": model_config.model,
            "stance": model_config.stance,
            "provider": response.provider,
            "model": response.model,
            "content": response.content,
            "usage": response.usage,
        }

    responses = await asyncio.gather(*(consult(model_config) for model_config in request.models))
    data = {
        "proposal": request.proposal,
        "initial_analysis": request.initial_analysis,
        "responses": responses,
        "synthesis_instructions": [
            "Identify agreements across models.",
            "Identify disagreements and the assumptions behind them.",
            "Give a final recommendation with concrete next steps.",
            "Call out risks that remain unresolved.",
        ],
    }
    store.add_turn(thread.id, "consensus", data)
    envelope = ToolEnvelope(
        status="consensus_complete",
        continuation_id=thread.id,
        next_steps="Synthesize the returned model perspectives into a single recommendation.",
        data=data,
    )
    return envelope.model_dump_json(indent=2)


async def run_codereview(
    *,
    step: str,
    findings: str = "",
    relevant_files: list[str] | None = None,
    files_checked: list[str] | None = None,
    relevant_context: list[str] | None = None,
    issues_found: list[dict[str, Any]] | None = None,
    review_type: str = "full",
    focus_on: str | None = None,
    standards: str | None = None,
    severity_filter: str = "all",
    model: str | None = None,
    use_assistant_model: bool = True,
    continuation_id: str | None = None,
    registry: ProviderRegistry | None = None,
) -> str:
    try:
        request = CodeReviewRequest(
            step=step,
            findings=findings,
            relevant_files=relevant_files or [],
            files_checked=files_checked or [],
            relevant_context=relevant_context or [],
            issues_found=issues_found or [],
            review_type=review_type,
            focus_on=focus_on,
            standards=standards,
            severity_filter=severity_filter,
            model=model,
            use_assistant_model=use_assistant_model,
            continuation_id=continuation_id,
        )
    except ValidationError as exc:
        raise ValueError(exc.errors()) from exc

    thread = store.create_or_get("codereview", request.continuation_id)
    data: dict[str, Any] = {
        "review_type": request.review_type,
        "focus_on": request.focus_on,
        "standards": request.standards,
        "severity_filter": request.severity_filter,
        "files_checked": request.files_checked,
        "relevant_files": request.relevant_files,
        "relevant_context": request.relevant_context,
        "issues_found": [issue.model_dump() for issue in request.issues_found],
        "findings": request.findings,
        "assistant_validation": None,
    }

    if request.use_assistant_model:
        provider_registry = registry or ProviderRegistry()
        provider, routed_model = provider_registry.resolve(request.model)
        context = build_context(request.relevant_files, store.context(thread.id))
        prompt = _join_prompt(
            f"Review request:\n{request.step}",
            f"Review type: {request.review_type}",
            f"Focus: {request.focus_on}" if request.focus_on else "",
            f"Standards: {request.standards}" if request.standards else "",
            f"Severity filter: {request.severity_filter}",
            f"Agent findings so far:\n{request.findings}" if request.findings else "",
            _format_issues(request.issues_found),
            context,
        )
        response = await provider.complete(
            prompt,
            model=routed_model,
            system_prompt=CODEREVIEW_SYSTEM_PROMPT,
            temperature=0.1,
        )
        data["assistant_validation"] = response.model_dump()

    store.add_turn(thread.id, "codereview", data)
    envelope = ToolEnvelope(
        status="code_review_complete",
        continuation_id=thread.id,
        next_steps=(
            "Present findings by severity, include evidence-backed file references, "
            "and prioritize the highest-impact fixes."
        ),
        data=data,
    )
    return envelope.model_dump_json(indent=2)


def run_listmodels(registry: ProviderRegistry | None = None) -> str:
    provider_registry = registry or ProviderRegistry()
    data = {
        "available": provider_registry.available_models(),
        "note": "Use provider aliases openai, anthropic, gemini or an explicit model name.",
    }
    return json.dumps(data, indent=2)


def _stance_prompt(model_config: ModelConfig) -> str:
    if model_config.stance_prompt:
        stance = model_config.stance_prompt
    elif model_config.stance == "for":
        stance = "Argue for the proposal, while still identifying serious risks."
    elif model_config.stance == "against":
        stance = "Argue against the proposal, while still acknowledging strong benefits."
    else:
        stance = "Give a balanced analysis without forcing artificial neutrality."
    return f"{CONSENSUS_SYSTEM_PROMPT}\n\nSTANCE:\n{stance}"


def _join_prompt(*parts: str) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def _format_issues(issues: list[Any]) -> str:
    if not issues:
        return ""
    lines = ["Issues already identified:"]
    for issue in issues:
        if hasattr(issue, "model_dump"):
            item = issue.model_dump()
        else:
            item = dict(issue)
        location = ""
        if item.get("file"):
            location = item["file"]
            if item.get("line"):
                location = f"{location}:{item['line']}"
        description = item.get("description", "")
        lines.append(f"- [{item.get('severity', 'medium')}] {location} {description}")
        if item.get("recommendation"):
            lines.append(f"  Recommendation: {item['recommendation']}")
    return "\n".join(lines)


def tool_error(exc: Exception) -> str:
    if isinstance(exc, ProviderError):
        message = str(exc)
    else:
        message = f"{type(exc).__name__}: {exc}"
    return json.dumps({"status": "error", "error": message}, indent=2)
