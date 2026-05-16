# AI Mates MCP Server

AI Mates is a slim, British-flavoured take on the PAL MCP idea: a small Model
Context Protocol server for getting useful second opinions from a few trusted AI
vendors.

It intentionally supports only:

- Providers: OpenAI, Anthropic, Gemini
- Core tools: `planner`, `consensus`, `codereview`
- Utility tools: `listmodels`

The goal is simple installation, small configuration, and predictable tool
schemas.

## Install With uvx

From GitHub:

```bash
uvx --from git+https://github.com/petems/ai-mates-mcp-server.git ai-mates-mcp-server
```

From PyPI, once published:

```bash
uvx ai-mates-mcp-server
```

## MCP Configuration

Set at least one API key in your client config. Leave unused providers out or
replace their placeholders before starting the client.

### Claude Code

Add this to your user-level Claude Code MCP config:

```json
{
  "mcpServers": {
    "mates": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/petems/ai-mates-mcp-server.git",
        "ai-mates-mcp-server"
      ],
      "env": {
        "OPENAI_API_KEY": "<OPENAI_API_KEY>",
        "ANTHROPIC_API_KEY": "<ANTHROPIC_API_KEY>",
        "GEMINI_API_KEY": "<GEMINI_API_KEY>"
      }
    }
  }
}
```

Copy-and-paste prompt:

```text
Add the AI Mates MCP server to my user-level Claude Code MCP config.

Use this config, replacing only the API key placeholders with my real keys. Keep
any provider I leave blank out of the final env block.

{
  "mcpServers": {
    "mates": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/petems/ai-mates-mcp-server.git",
        "ai-mates-mcp-server"
      ],
      "env": {
        "OPENAI_API_KEY": "<OPENAI_API_KEY>",
        "ANTHROPIC_API_KEY": "<ANTHROPIC_API_KEY>",
        "GEMINI_API_KEY": "<GEMINI_API_KEY>"
      }
    }
  }
}
```

### Codex CLI

Codex CLI uses TOML in `~/.codex/config.toml`, not `mcp.json`:

```toml
[mcp_servers.mates]
command = "uvx"
args = [
  "--from",
  "git+https://github.com/petems/ai-mates-mcp-server.git",
  "ai-mates-mcp-server",
]

[mcp_servers.mates.env]
OPENAI_API_KEY = "<OPENAI_API_KEY>"
ANTHROPIC_API_KEY = "<ANTHROPIC_API_KEY>"
GEMINI_API_KEY = "<GEMINI_API_KEY>"
```

Copy-and-paste prompt:

```text
Add the AI Mates MCP server to my user-level Codex CLI config at
~/.codex/config.toml.

Use this TOML config, replacing only the API key placeholders with my real keys.
Keep any provider I leave blank out of the final env block.

[mcp_servers.mates]
command = "uvx"
args = [
  "--from",
  "git+https://github.com/petems/ai-mates-mcp-server.git",
  "ai-mates-mcp-server",
]

[mcp_servers.mates.env]
OPENAI_API_KEY = "<OPENAI_API_KEY>"
ANTHROPIC_API_KEY = "<ANTHROPIC_API_KEY>"
GEMINI_API_KEY = "<GEMINI_API_KEY>"
```

### Gemini CLI

Add this to your user-level Gemini CLI settings at `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "mates": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/petems/ai-mates-mcp-server.git",
        "ai-mates-mcp-server"
      ],
      "env": {
        "OPENAI_API_KEY": "<OPENAI_API_KEY>",
        "ANTHROPIC_API_KEY": "<ANTHROPIC_API_KEY>",
        "GEMINI_API_KEY": "<GEMINI_API_KEY>"
      }
    }
  }
}
```

Copy-and-paste prompt:

```text
Add the AI Mates MCP server to my user-level Gemini CLI settings at
~/.gemini/settings.json.

Use this config, replacing only the API key placeholders with my real keys. Keep
any provider I leave blank out of the final env block.

{
  "mcpServers": {
    "mates": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/petems/ai-mates-mcp-server.git",
        "ai-mates-mcp-server"
      ],
      "env": {
        "OPENAI_API_KEY": "<OPENAI_API_KEY>",
        "ANTHROPIC_API_KEY": "<ANTHROPIC_API_KEY>",
        "GEMINI_API_KEY": "<GEMINI_API_KEY>"
      }
    }
  }
}
```

You can also override defaults:

```bash
export OPENAI_MODEL=gpt-5.5
export ANTHROPIC_MODEL=claude-sonnet-4-6
export GEMINI_MODEL=gemini-3.1-pro-preview
export DEFAULT_MODEL=auto
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

Model names can be exact API IDs or aliases from the registry. Common aliases
include `openai`, `gpt`, `sonnet`, `opus`, `haiku`, `gemini`, `pro`, `flash`,
`mini`, and `nano`.

### `codereview`

Runs a structured code-review workflow. It can use local findings only or call a
configured assistant model for validation.

Set `use_assistant_model` to `false` when you only want the MCP tool to package
and track your own review findings.

### `listmodels`

Lists model IDs, aliases, provider defaults, status, and whether each provider is
configured. By default this uses the packaged and local registry only.

Set `MATES_MODEL_DISCOVERY=list` to make `listmodels` augment the registry with
live provider model-list API results when API keys are configured.

## Local Model Registry

AI Mates ships with a small current model catalogue, but you can add or override
models locally without opening a PR:

```bash
MATES_MODELS_FILE=$HOME/.config/ai-mates/models.json
```

```json
{
  "defaults": {
    "openai": "gpt-5.5"
  },
  "models": [
    {
      "id": "gpt-new-example",
      "provider": "openai",
      "aliases": ["new-openai"],
      "rank": 120,
      "status": "active"
    }
  ]
}
```

Local entries replace packaged entries with the same `id`, and local aliases win
when there is a collision. Deprecated, retired, or shut-down models are blocked
unless `MATES_ALLOW_DEPRECATED_MODELS=true`.

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
