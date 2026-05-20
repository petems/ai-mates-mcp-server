## AI Model Registry Policy

- Follow `AGENTS.md` for the full project instructions.
- Keep deprecated OpenAI, Anthropic, and Gemini model IDs out of the active
  `models` list in `src/ai_mates_mcp_server/data/models.json`.
- Put deprecated or near-shutdown model IDs in `deprecated_models` with their
  feed metadata so the registry can block exact requests before provider prefix
  fallback routes them.
- Use `https://deprecations.info/v1/feed.json` for model deprecation updates.
