# Factory Droid OpenAI Bridge

Unofficial local OpenAI-compatible HTTP facade for
[`droid-sdk-python`](https://github.com/Factory-AI/droid-sdk-python). It lets
OpenAI clients, including Hermes Agent, use Factory Droid through
`/v1/chat/completions`.

## Requirements

- Python 3.11+
- Authenticated Factory `droid` CLI

## Install and run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/factory-droid-openai
```

The server listens on `127.0.0.1:8787` by default.

```bash
curl http://127.0.0.1:8787/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "factory-droid",
    "messages": [{"role": "user", "content": "Reply with hello."}]
  }'
```

Streaming uses standard OpenAI server-sent events:

```bash
curl -N http://127.0.0.1:8787/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "factory-droid",
    "stream": true,
    "messages": [{"role": "user", "content": "Count to three."}]
  }'
```

## Hermes Agent

Set the main model in `~/.hermes/config.yaml`:

```yaml
model:
  provider: custom
  default: factory-droid
  base_url: http://127.0.0.1:8787/v1
  api_key: none
  api_mode: chat_completions
```

Hermes keeps ownership of tool execution. The bridge serializes the complete
OpenAI transcript and tool schemas into the Droid prompt, parses strict
`<hermes_tool_call>` responses, and returns OpenAI `tool_calls`.

## Configuration

| Environment variable | Default | Purpose |
|---|---:|---|
| `FACTORY_DROID_OPENAI_HOST` | `127.0.0.1` | Listen address |
| `FACTORY_DROID_OPENAI_PORT` | `8787` | Listen port |
| `FACTORY_DROID_OPENAI_API_KEY` | unset | Optional bearer token |
| `FACTORY_DROID_PATH` | `droid` | Droid executable path |
| `FACTORY_DROID_OPENAI_WORKDIR` | current directory | Droid working directory |
| `FACTORY_DROID_OPENAI_TIMEOUT_SECONDS` | `600` | Request timeout |
| `FACTORY_DROID_OPENAI_MAX_CONCURRENCY` | `1` | Concurrent Droid processes |
| `FACTORY_DROID_OPENAI_MODEL_ALIAS` | `factory-droid` | Alias using Droid's default model |

If `FACTORY_DROID_OPENAI_API_KEY` is set, use the same value as
`model.api_key` in Hermes.

## Compatibility limits

The Factory SDK controls a full Droid agent rather than exposing raw model
inference. It cannot receive arbitrary native chat history, external tool
schemas, or external tool-result continuation. This bridge therefore uses a
strict text protocol. Tool-call fidelity and prompt caching are not equivalent
to a native OpenAI inference endpoint.

Factory-native tools and interactive questions are cancelled. Only tools sent
by the OpenAI client are allowed.
