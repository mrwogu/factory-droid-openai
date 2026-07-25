# Contributing

Contributions are welcome when they improve OpenAI compatibility, Factory Droid
integration, reliability, security, or developer experience.

## Development setup

Requirements:

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- Factory Droid CLI only for manual end-to-end testing

Clone and install all locked dependencies:

```bash
git clone https://github.com/mrwogu/factory-droid-openai.git
cd factory-droid-openai
uv sync --all-extras
```

Unit and HTTP integration tests use fake SDK clients. They do not start Droid,
access Factory services, or require authentication.

## Validation

Run every required check before opening a pull request:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run python scripts/generate_openapi.py
uv run openapi-spec-validator openapi.json
uv run pytest --cov=factory_droid_openai --cov-report=term-missing
uv build
```

Apply formatting with:

```bash
uv run ruff format .
```

CI runs tests on every supported Python version.

## Change guidelines

- Keep the bridge local-first and OpenAI-compatible.
- Preserve streaming and non-streaming response contracts.
- Never allow Factory-native tools to bypass client-owned tool execution.
- Add behavior tests for every user-visible change.
- Keep dependencies bounded and regenerate `uv.lock`.
- Avoid live network calls in tests.
- Never commit credentials, private prompts, generated packages, or local logs.

## Commit messages

Use Conventional Commits with a subject no longer than 70 characters:

```text
feat: add response field compatibility
fix: interrupt Droid after client disconnect
test: cover split tool-call markers
docs: document bearer authentication
```

## Pull requests

Include:

- Problem and intended behavior
- Compatibility impact
- Security impact
- Tests added or updated
- Manual validation, when applicable

Keep pull requests focused. Separate unrelated refactors from behavior changes.

## Reporting bugs

Use the bug report form and include a minimal sanitized request. Remove API
keys, access tokens, private filesystem paths, and sensitive prompt content.

Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).
