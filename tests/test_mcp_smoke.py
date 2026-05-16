from __future__ import annotations

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.mark.asyncio
async def test_mcp_lists_core_tools():
    params = StdioServerParameters(
        command="uv",
        args=["run", "ai-mates-mcp-server"],
        env={},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()

    names = {tool.name for tool in tools.tools}
    assert names == {"planner", "consensus", "codereview", "listmodels"}

    annotations = {
        tool.name: tool.annotations.model_dump(exclude_none=True)
        for tool in tools.tools
        if tool.annotations
    }
    assert annotations == {
        "planner": {
            "title": "Planner",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
        "consensus": {
            "title": "Consensus",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        "codereview": {
            "title": "Code Review",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        "listmodels": {
            "title": "List Models",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    }
