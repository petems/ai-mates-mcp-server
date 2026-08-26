from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from .tools import run_codereview, run_consensus, run_listmodels, run_planner, tool_error

mcp = FastMCP("AI Mates")


@mcp.tool(
    annotations=ToolAnnotations(
        title="Planner",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    )
)
async def planner(
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
    """Break complex work into sequential planning steps with continuation support."""
    try:
        return await run_planner(
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
    except Exception as exc:
        return tool_error(exc)


@mcp.tool(
    annotations=ToolAnnotations(
        title="Consensus",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    )
)
async def consensus(
    proposal: str,
    models: list[dict[str, Any]],
    initial_analysis: str | None = None,
    relevant_files: list[str] | None = None,
    workspace_root: str | None = None,
    continuation_id: str | None = None,
) -> str:
    """Consult multiple OpenAI, Anthropic, or Gemini models and return their perspectives."""
    try:
        return await run_consensus(
            proposal=proposal,
            models=models,
            initial_analysis=initial_analysis,
            relevant_files=relevant_files,
            workspace_root=workspace_root,
            continuation_id=continuation_id,
        )
    except Exception as exc:
        return tool_error(exc)


@mcp.tool(
    annotations=ToolAnnotations(
        title="Code Review",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    )
)
async def codereview(
    step: str,
    findings: str = "",
    relevant_files: list[str] | None = None,
    workspace_root: str | None = None,
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
) -> str:
    """Run a focused code review workflow with optional assistant-model validation."""
    try:
        return await run_codereview(
            step=step,
            findings=findings,
            relevant_files=relevant_files,
            workspace_root=workspace_root,
            files_checked=files_checked,
            relevant_context=relevant_context,
            issues_found=issues_found,
            review_type=review_type,
            focus_on=focus_on,
            standards=standards,
            severity_filter=severity_filter,
            model=model,
            use_assistant_model=use_assistant_model,
            continuation_id=continuation_id,
        )
    except Exception as exc:
        return tool_error(exc)


@mcp.tool(
    annotations=ToolAnnotations(
        title="List Models",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    )
)
async def listmodels(include_deprecated: bool = False) -> str:
    """List configured model aliases, defaults, statuses, and optional live discovery data.

    Set include_deprecated=True to also return the blocked/deprecated model registry, which is
    large and omitted by default.
    """
    try:
        return await run_listmodels(include_deprecated=include_deprecated)
    except Exception as exc:
        return tool_error(exc)


def main() -> None:
    mcp.run(transport="stdio")
