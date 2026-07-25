## Summary

Describe the change and why it is needed.

## Validation

- [ ] `uv run ruff check .`
- [ ] `uv run ruff format --check .`
- [ ] `uv run mypy`
- [ ] `uv run pytest --cov=factory_droid_openai --cov-report=term-missing`
- [ ] `uv build`

## Security and compatibility

- [ ] No credentials, private prompt data, or generated artifacts committed.
- [ ] OpenAI response shapes remain compatible in streaming and non-streaming modes.
- [ ] Factory-native tools remain blocked.
- [ ] Documentation and tests cover user-visible behavior changes.
