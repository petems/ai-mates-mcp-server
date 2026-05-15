# AI Mates MCP Server

AI Mates is a slim, British-flavoured take on the PAL MCP idea: a small Model
Context Protocol server for getting useful second opinions from a few trusted AI
vendors.

It intentionally supports only:

- Providers: OpenAI, Anthropic, Gemini
- Tools: `planner`, `consensus`, `codereview`

The goal is simple installation, small configuration, and predictable tool
schemas.

## Install With uvx

From GitHub:

```bash
uvx --from git+https://github.com/<owner>/ai-mates-mcp-server.git ai-mates-mcp-server
```

From PyPI, once published:

```bash
uvx ai-mates-mcp-server
```

## MCP Configuration

Add this to a client config such as `.mcp.json`, Claude Desktop, Claude Code, or
Codex:

```json
{
  "mcpServers": {
    "mates": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/<owner>/ai-mates-mcp-server.git",
        "ai-mates-mcp-server"
      ],
      "env": {
        "OPENAI_API_KEY": "your-openai-key",
        "ANTHROPIC_API_KEY": "your-anthropic-key",
        "GEMINI_API_KEY": "your-gemini-key",
        "DEFAULT_MODEL": "auto"
      }
    }
  }
}
```

Set at least one API key. You can also override defaults:

```bash
OPENAI_MODEL=gpt-4.1
ANTHROPIC_MODEL=claude-sonnet-4-5
GEMINI_MODEL=gemini-2.5-pro
DEFAULT_MODEL=auto
```

## Tools

### `planner`

Self-contained sequential planning. It does not call an external model.

Use it to break down migrations, architecture changes, feature plans, and
implementation work.

### `consensus`

Consults at least two model entries independently and returns their responses for
the calling agent to synthesize.

Each model entry supports:

```json
{ "model": "openai", "stance": "for" }
```

Valid stances are `for`, `against`, and `neutral`. You can also provide a
`stance_prompt`.

### `codereview`

Runs a structured code-review workflow. It can use local findings only or call a
configured assistant model for validation.

Set `use_assistant_model` to `false` when you only want the MCP tool to package
and track your own review findings.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv build
```

## Attribution

AI Mates is inspired by
[`BeehiveInnovations/pal-mcp-server`](https://github.com/BeehiveInnovations/pal-mcp-server),
which is licensed under Apache-2.0. This project is a fresh, smaller
implementation focused on three providers and three workflows.
