## AI Model Registry Policy

- Keep `src/ai_mates_mcp_server/data/models.json` active `models` limited to
  reachable, recommended OpenAI, Anthropic, and Gemini models.
- Do not add deprecated, retired, shutdown, legacy snapshot, or near-shutdown
  models to the active `models` list.
- Track blocked model IDs in the separate `deprecated_models` list instead.
  Include provider, status, deprecation date, shutdown date, replacements, and
  source URL when available.
- Use `https://deprecations.info/v1/feed.json` as the current deprecation feed.
  The unversioned `/feed.json` path may not be available.
- Map feed providers into this server's supported providers as follows:
  `OpenAI` -> `openai`, `Anthropic` -> `anthropic`, and Google/Gemini entries
  -> `gemini`. Only include Google Vertex entries when they can block a model
  ID this server could otherwise route, such as `claude-*` or `gemini-*`.
- If a model appears in the deprecation feed, remove it from active `models`
  even if its shutdown date is still in the future.
- Keep defaults and aliases pointed at non-deprecated replacements.

## Pre-commit Hook Installation Notes

- If `pre-commit install` is blocked by a global `core.hooksPath` (for example
  `/usr/local/dd/global_hooks`), install hooks directly into the repo's
  `.git/hooks` via pre-commit's install API.
- The standard `pre-commit install` CLI does not expose a `--hooks-dir` flag for
  a custom target path.
- Work around a global `core.hooksPath` by overriding it for a single command:

```bash
git -c core.hooksPath=.git/hooks pre-commit install
```

- Alternative two-step local override flow:

```bash
# 1. Temporarily override hooksPath locally
git config --local core.hooksPath .git/hooks

# 2. Install pre-commit hooks (targets .git/hooks)
pre-commit install

# 3. Unset local override so global hooksPath resumes
git config --local --unset core.hooksPath
```

- After either approach, verify hook scripts exist and are executable:

```bash
ls -la .git/hooks/pre-commit .git/hooks/pre-push
```

- Ensure the global hook runner (for example
  `/usr/local/dd/global_hooks/pre-commit`) chains into repo-level hooks:

```bash
# Inside /usr/local/dd/global_hooks/pre-commit
if [ -x ".git/hooks/pre-commit" ]; then
    .git/hooks/pre-commit "$@"
fi
```
