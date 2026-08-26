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
replace their placeholders before starting the client. Gemini can also run
without an API key via gcloud Application Default Credentials, see
[Gemini without an API key (gcloud ADC)](#gemini-without-an-api-key-gcloud-adc).

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

## Gemini without an API key (gcloud ADC)

Google's guidance for local development is to avoid long-lived API keys and
service account keys, and use Application Default Credentials instead: tokens
auto-refresh and expire after an hour.

Set it up once:

```bash
gcloud auth application-default login
gcloud config set project <your-project>
gcloud services enable aiplatform.googleapis.com --project <your-project>
```

Then run the server with:

```bash
export GEMINI_USE_GCLOUD_AUTH=true
export GOOGLE_CLOUD_PROJECT=<your-project>   # optional if gcloud already has one
export GOOGLE_CLOUD_LOCATION=global          # optional, defaults to global
```

Or in an MCP client config:

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
        "GEMINI_USE_GCLOUD_AUTH": "true",
        "GOOGLE_CLOUD_PROJECT": "<your-project>"
      }
    }
  }
}
```

Things worth knowing:

- ADC only works through **Vertex AI**, not the Gemini Developer API, so this
  flag makes the server build a Vertex AI client (`vertexai=True`). You need a
  GCP project with the Vertex AI API enabled and the `roles/aiplatform.user`
  role.
- `GEMINI_USE_GCLOUD_AUTH=true` wins over `GEMINI_API_KEY` and `GOOGLE_API_KEY`.
  The explicit opt-in is treated as the stronger signal, so an API key left over
  in your shell does not silently take over. Credentials are resolved up front
  and passed to the SDK explicitly to keep that promise: given only
  `vertexai=True`, the SDK would fall back to either of those environment keys
  and never load ADC.
- `GOOGLE_GENAI_USE_VERTEXAI=true` (the variable the google-genai SDK reads)
  works as an alias if you already export it.
- Model IDs are the same, but Vertex only serves the models available in your
  project and region. `listmodels` reports the auth mode per provider under
  `provider_auth`.
- ADC access tokens last an hour. If one expires while the server is running,
  the Gemini tool call fails with "Run `gcloud auth application-default login`
  again" rather than a raw OAuth `invalid_grant` error.
- If ADC is missing or the project cannot be resolved, the rest of the server
  still starts. The Gemini failure is reported under `provider_errors` in
  `listmodels` and in the error returned when you ask for a Gemini model:
  "Run `gcloud auth application-default login`".

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

Lists model IDs, aliases, provider defaults, status, the auth mode used per
provider (`api-key` or `gcloud-adc`), any provider setup errors, and whether each
provider is configured. By default this uses the packaged and local registry only.

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

Packaged model data separates reachable defaults from blocked model IDs:

- `models`: current, reachable, recommended model IDs and aliases.
- `deprecated_models`: OpenAI, Anthropic, and Google/Gemini model IDs sourced
  from the deprecations.info feed, including deprecation dates, shutdown dates,
  replacement models, and source URLs.

Exact requests for models in `deprecated_models` are blocked before provider
prefix fallback can route them. Keep deprecated and near-shutdown models out of
the active `models` list.

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
