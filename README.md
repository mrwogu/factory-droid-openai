# Factory Droid OpenAI Bridge

[![CI](https://github.com/mrwogu/factory-droid-openai/actions/workflows/ci.yml/badge.svg)](https://github.com/mrwogu/factory-droid-openai/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

Unofficial, local-first OpenAI-compatible HTTP bridge for
[Factory Droid](https://www.factory.ai/) using
[`droid-sdk-python`](https://github.com/Factory-AI/droid-sdk-python).

Connect OpenAI-compatible clients, frameworks, and agents to a locally
authenticated Factory Droid CLI through `/v1/chat/completions`.

This project is not affiliated with, endorsed by, or maintained by Factory.

## Quick start

Requires Python 3.11+, an installed `droid` CLI, and an authenticated Factory
session.

```bash
git clone https://github.com/mrwogu/factory-droid-openai.git
cd factory-droid-openai
uv sync
FACTORY_DROID_OPENAI_WORKDIR="/path/to/your/project" \
  uv run factory-droid-openai
```

From another terminal, send an explicit Droid model ID:

```bash
curl --fail http://127.0.0.1:8787/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-5.4",
    "messages": [
      {"role": "user", "content": "Reply with hello."}
    ]
  }'
```

Use an ID listed under `Available Models` by `droid exec --help`. The bridge
forwards every model value except its `factory-droid` alias directly to Droid.
The alias requests the Droid CLI's configured default instead of selecting a
model explicitly. Examples below use explicit `gpt-5.4` and
`gemini-3.1-pro-preview` model IDs.

## API compatibility

✅ Supported · ⚠️ Partial · ❌ Unsupported

| OpenAI capability | Status | Bridge behavior |
|---|---|---|
| Text chat completions | ✅ | Returns one assistant choice |
| System and developer messages | ✅ | Serialized with the complete transcript |
| Non-streaming responses | ✅ | OpenAI-compatible JSON completion |
| Streaming responses | ✅ | SSE chunks followed by `[DONE]` |
| Function tool schemas | ✅ | Serialized into the strict Droid prompt |
| Tool choice | ✅ | `auto`, `none`, `required`, or one named function |
| Tool-result continuation | ✅ | Client resends the complete transcript on the next request |
| Parallel tool calls | ❌ | At most one external tool call per Droid turn |
| Reasoning output | ✅ | Emitted as `reasoning` and `reasoning_content` |
| Token usage | ✅ | Includes cache read and write token details |
| Model selection | ✅ | Alias uses the Droid default; other IDs are forwarded |
| Multimodal content | ⚠️ | Content structures are serialized as JSON, not SDK attachments |
| Structured outputs | ❌ | `response_format` and JSON schema enforcement are ignored |
| Sampling controls | ❌ | `temperature`, `top_p`, penalties, and `seed` are ignored |
| Multiple choices | ❌ | `n` is ignored and one choice is returned |
| Log probabilities | ❌ | `logprobs` and `top_logprobs` are ignored |
| Output token limits | ❌ | `max_tokens` and `max_completion_tokens` are ignored |
| Stored completions | ❌ | `store`, `metadata`, listing, retrieval, and deletion are unavailable |
| Prompt cache controls | ❌ | OpenAI cache keys and retention settings are ignored |
| Built-in web search | ❌ | `web_search_options` is ignored |
| Audio output | ❌ | Audio modalities and audio response fields are ignored |

### Unsupported API families

| API family | Status |
|---|---|
| Responses API | ❌ |
| Embeddings | ❌ |
| Images | ❌ |
| Audio | ❌ |
| Files and batches | ❌ |
| Fine-tuning | ❌ |
| Moderations | ❌ |
| Realtime | ❌ |
| Vector stores and uploads | ❌ |

## Features

- OpenAI-compatible `POST /v1/chat/completions`
- OpenAI-compatible `GET /v1/models`
- Non-streaming JSON responses
- Streaming server-sent events with `[DONE]` termination
- System, developer, user, assistant, and tool message mapping
- Function tool schemas and validated tool-call responses
- Reasoning delta and token usage mapping
- Optional constant-time bearer authentication
- Bounded request timeout and process concurrency
- Client disconnect cancellation
- Factory-native tool blocking
- Versioned and externally validated OpenAPI 3.1 contract
- Strict typing, locked dependencies, and 99% test coverage

## How it works

```text
OpenAI client
    |
    | POST /v1/chat/completions
    v
Factory Droid OpenAI Bridge
    |
    | OpenAI transcript + tool schemas encoded as JSON
    v
droid-sdk-python
    |
    | JSON-RPC over a local droid exec subprocess
    v
Factory Droid
```

Factory's SDK controls a complete Droid agent rather than exposing a raw model
inference endpoint. The bridge creates a new Droid session for each completion,
serializes the complete OpenAI transcript into one prompt, and maps Droid stream
events back to OpenAI response objects.

External tools use a strict text protocol. Droid emits:

```text
<tool_call>{"name":"weather","arguments":{"city":"Gdansk"}}</tool_call>
```

The bridge validates the tool name and JSON arguments, then returns a standard
OpenAI `tool_calls` object. The OpenAI client executes the tool and sends its
result in the next request.

## Requirements

- Python 3.11 or newer
- Installed Factory `droid` CLI
- Authenticated Droid session

Verify the CLI before starting the bridge:

```bash
droid --version
```

Run Droid normally once if authentication or first-time setup is required.

## Installation

### uv

```bash
git clone https://github.com/mrwogu/factory-droid-openai.git
cd factory-droid-openai
uv sync
uv run factory-droid-openai
```

### pip

```bash
git clone https://github.com/mrwogu/factory-droid-openai.git
cd factory-droid-openai
python3 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/factory-droid-openai
```

The default address is `http://127.0.0.1:8787`.

Check health and inspect the configured alias:

```bash
curl --fail http://127.0.0.1:8787/health
curl --fail http://127.0.0.1:8787/v1/models
```

Stream a completion:

```bash
curl --no-buffer --fail http://127.0.0.1:8787/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemini-3.1-pro-preview",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Count from one to three."}
    ]
  }'
```

## OpenAI Python client

Install the official client in your application:

```bash
python -m pip install "openai>=2,<3"
```

### Non-streaming

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8787/v1",
    api_key="none",
)

response = client.chat.completions.create(
    model="gpt-5.4",
    messages=[
        {"role": "system", "content": "Answer concisely."},
        {"role": "user", "content": "Explain JSON-RPC in one sentence."},
    ],
)

print(response.choices[0].message.content)
```

### Streaming

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8787/v1",
    api_key="none",
)

stream = client.chat.completions.create(
    model="gemini-3.1-pro-preview",
    messages=[{"role": "user", "content": "Count from one to three."}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
```

### Function tools

```python
import json

from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8787/v1",
    api_key="none",
)

messages = [{"role": "user", "content": "What is the weather in Gdansk?"}]
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Read current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }
]

first = client.chat.completions.create(
    model="gpt-5.4",
    messages=messages,
    tools=tools,
)
assistant = first.choices[0].message
messages.append(assistant.model_dump(exclude_none=True))

for tool_call in assistant.tool_calls or []:
    arguments = json.loads(tool_call.function.arguments)
    result = {"city": arguments["city"], "temperature_c": 20}
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result),
        }
    )

final = client.chat.completions.create(
    model="gpt-5.4",
    messages=messages,
    tools=tools,
)
print(final.choices[0].message.content)
```

## Hermes Agent

Configure Hermes as a custom OpenAI-compatible provider in
`~/.hermes/config.yaml`:

```yaml
model:
  provider: custom
  default: gpt-5.4
  base_url: http://127.0.0.1:8787/v1
  api_key: none
  api_mode: chat_completions
```

`model.default` is forwarded to Droid as an explicit model ID. Replace it with
another ID from `droid exec --help` when needed.

Hermes retains ownership of tool execution. The bridge converts Droid's strict
tool protocol into OpenAI `tool_calls`, and Hermes returns tool results in the
next serialized transcript.

When bearer authentication is enabled, set the same token as `model.api_key`.

## OpenClaw

Configure OpenClaw's `openai-completions` adapter in
`~/.openclaw/openclaw.json`, following the
[custom provider model](https://docs.openclaw.ai/concepts/model-providers).
Set a bridge token in the environment used by both processes:

```bash
export FACTORY_DROID_OPENAI_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

Add this JSON5 configuration:

```json5
{
  agents: {
    defaults: {
      model: { primary: "factory-droid/gpt-5.4" },
      models: {
        "factory-droid/gpt-5.4": {
          alias: "GPT-5.4 via Factory Droid",
        },
      },
    },
  },
  models: {
    mode: "merge",
    providers: {
      "factory-droid": {
        baseUrl: "http://127.0.0.1:8787/v1",
        apiKey: "${FACTORY_DROID_OPENAI_API_KEY}",
        api: "openai-completions",
        timeoutSeconds: 600,
        models: [
          {
            id: "gpt-5.4",
            name: "GPT-5.4 via Factory Droid",
            reasoning: true,
            input: ["text"],
            compat: {
              supportsDeveloperRole: true,
              supportsReasoningEffort: true,
              supportsUsageInStreaming: true,
              supportsTools: true,
              supportsStrictMode: false,
            },
          },
        ],
      },
    },
  },
}
```

Here, the first `factory-droid` segment names the OpenClaw provider. The
`gpt-5.4` segment and model `id` are the explicit Droid model ID.

Start the bridge from the working directory Droid should access, then restart
the OpenClaw gateway. Verify the configuration with:

```bash
openclaw doctor
openclaw models list
```

OpenClaw retains ownership of tool execution. Tool calling, tool-result
continuation, streaming usage, developer messages, and reasoning effort are
enabled explicitly. Strict tool schemas, temperature control, multimodal SDK
attachments, and parallel tool calls remain unsupported by this bridge.

## API reference

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Process health check |
| `GET` | `/v1/models` | Configured bridge model alias |
| `POST` | `/v1/chat/completions` | Chat completions and streaming |
| `GET` | `/openapi.json` | OpenAPI 3.1 contract |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/redoc` | Interactive ReDoc reference |

### OpenAPI contract

The versioned [OpenAPI 3.1 document](openapi.json) describes the supported
HTTP contract. Regenerate and validate it after changing routes or models:

```bash
uv run python scripts/generate_openapi.py
uv run openapi-spec-validator openapi.json
```

Tests compare the committed document with FastAPI's generated schema. CI also
validates it independently with `openapi-spec-validator`, so malformed or
stale API contracts fail automatically. The contract covers the implemented
OpenAI-compatible subset, not the complete OpenAI platform. Separate contract
tests exercise models, non-streaming chat, streaming chat, usage, and function
tool calls through the official `openai` Python client.

### Request fields

| Field | Support | Behavior |
|---|---|---|
| `model` | Yes | Alias uses Droid default; other values become Droid model IDs |
| `messages` | Yes | Complete transcript serialized in original order |
| `tools` | Yes | OpenAI function tools |
| `tool_choice` | Yes | `auto`, `none`, `required`, or one named function |
| `stream` | Yes | OpenAI-compatible SSE chunks |
| `stream_options.include_usage` | Yes | Emits null usage on normal chunks and one final usage-only chunk |
| `stream_options.include_obfuscation` | No | Accepted but ignored |
| `reasoning_effort` | Yes | Mapped to Droid reasoning effort |
| `factory_droid_reasoning_effort` | Yes | Bridge-specific override |
| `timeout` | Yes | Per-request value capped by server timeout |
| `temperature`, `top_p`, penalties, `seed` | No | Accepted but ignored |
| `max_tokens`, `max_completion_tokens`, `stop` | No | Accepted but ignored |
| `response_format` | No | Accepted but not enforced |
| `parallel_tool_calls` | No | Accepted but ignored |
| `functions`, `function_call` | No | Legacy function-calling fields are ignored |
| `modalities`, `audio` | No | Accepted but ignored |
| `logprobs`, `top_logprobs`, `logit_bias` | No | Accepted but ignored |
| `store`, `metadata`, `user`, `safety_identifier` | No | Accepted but ignored |
| `prompt_cache_key`, cache options and retention | No | Accepted but ignored |
| `prediction`, `service_tier`, `verbosity` | No | Accepted but ignored |
| `web_search_options` | No | Accepted but ignored |
| Unknown fields | Accepted | Ignored by the bridge |

The bridge currently returns one choice with index `0`.

### Model selection

`factory-droid` is a server-side alias. It passes no explicit model ID and lets
Droid use its configured default model.

Any other `model` value is forwarded as `model_id` during Droid session
initialization:

```json
{
  "model": "gpt-5.4",
  "messages": [{"role": "user", "content": "Hello"}]
}
```

For the official OpenAI client, the same explicit selection is:

```python
client.chat.completions.create(
    model="gemini-3.1-pro-preview",
    messages=[{"role": "user", "content": "Hello"}],
)
```

Run `droid exec --help` to list model IDs available in the installed Droid CLI.
Availability also depends on the authenticated Factory account.

### Reasoning

Supported values follow `droid-sdk-python`:

- `none`
- `dynamic`
- `off`
- `minimal`
- `low`
- `medium`
- `high`
- `xhigh`
- `max`

Reasoning text appears as both `reasoning` and `reasoning_content` for broad
client compatibility.

### Errors

Non-streaming failures use the OpenAI error object:

```json
{
  "error": {
    "message": "Factory Droid executable was not found: droid",
    "type": "factory_droid_unavailable",
    "param": null,
    "code": null
  }
}
```

| HTTP status | Meaning |
|---:|---|
| `400` | Invalid request or unsupported option |
| `401` | Missing or invalid bearer token |
| `502` | Droid SDK, process, or bridge protocol failure |
| `503` | Droid executable unavailable |
| `504` | Request timeout |

After streaming headers are sent, errors arrive as an SSE `error` object
followed by `[DONE]`.

## Configuration

| Environment variable | Default | Description |
|---|---:|---|
| `FACTORY_DROID_OPENAI_HOST` | `127.0.0.1` | Listen address |
| `FACTORY_DROID_OPENAI_PORT` | `8787` | Listen port |
| `FACTORY_DROID_OPENAI_API_KEY` | unset | Optional bearer token |
| `FACTORY_DROID_PATH` | `droid` | Droid executable path |
| `FACTORY_DROID_OPENAI_WORKDIR` | current directory | Droid working directory |
| `FACTORY_DROID_OPENAI_TIMEOUT_SECONDS` | `600` | Maximum request duration |
| `FACTORY_DROID_OPENAI_MAX_CONCURRENCY` | `1` | Concurrent Droid subprocesses |
| `FACTORY_DROID_OPENAI_MODEL_ALIAS` | `factory-droid` | Alias using Droid default model |

Command-line options:

```text
usage: factory-droid-openai [-h] [--host HOST] [--port PORT]
                            [--log-level {critical,error,warning,info,debug,trace}]
```

Environment variables define defaults. `--host`, `--port`, and `--log-level`
override server options.

## Authentication

Local loopback use requires no bridge token by default. OpenAI clients still
expect an API key value, so use a placeholder such as `none`.

Enable bearer authentication:

```bash
export FACTORY_DROID_OPENAI_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
factory-droid-openai
```

Send the token:

```bash
curl --fail http://127.0.0.1:8787/v1/models \
  -H "Authorization: Bearer $FACTORY_DROID_OPENAI_API_KEY"
```

Never expose an unauthenticated bridge on a non-loopback interface. See
[SECURITY.md](SECURITY.md) before remote deployment.

## Tool execution safety

The OpenAI client, not Factory Droid, owns tool execution.

For every Droid session, the bridge:

1. Sets Droid autonomy to `off`.
2. Cancels permission requests.
3. Cancels interactive questions.
4. Passes no additional enabled tool IDs.
5. Rejects any Factory-native tool event.
6. Validates generated tool names against the request schema.
7. Requires tool arguments to be a JSON object with unique keys.

Set `FACTORY_DROID_OPENAI_WORKDIR` to the smallest directory required by your
workflow.

## Bridge architecture limits

This is a compatibility bridge, not a native OpenAI inference implementation.

| Limitation | Effect |
|---|---|
| No native SDK chat-history input | History is serialized into one prompt |
| No native SDK external-tool schemas | Tool definitions are serialized into the prompt |
| No native SDK tool-result continuation | Each request creates a new Droid session |
| Session-per-request execution | Prompt caching differs from a native inference endpoint |
| Strict tool marker protocol | Invalid generated tool payloads fail closed |
| One external call per Droid turn | Parallel tool calls are rejected |

## Troubleshooting

### Droid executable not found

```text
factory_droid_unavailable
```

Set the executable path:

```bash
export FACTORY_DROID_PATH="$HOME/.local/bin/droid"
```

### Authentication failure

An HTTP `401` means `FACTORY_DROID_OPENAI_API_KEY` is configured and the
request token is missing or incorrect. This is separate from Factory account
authentication used by the Droid CLI.

### Factory-native tool blocked

```text
factory_native_tool_blocked
```

Droid attempted to use one of its own tools instead of returning bridge
protocol output. Retry with a clearer request or inspect the supplied system
and user messages for conflicting instructions.

### Incomplete response

```text
factory_incomplete_response
```

The Droid stream ended without a turn-complete event. Check Droid CLI health,
request timeout, and bridge logs.

## Development

Install locked development dependencies:

```bash
uv sync --all-extras
```

Run required validation:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run python scripts/generate_openapi.py
uv run openapi-spec-validator openapi.json
uv run pytest --cov=factory_droid_openai --cov-report=term-missing
uv build
```

### Automated test layers

| Layer | CI status | Coverage |
|---|---|---|
| Unit tests | Automated | Configuration, protocol parsing, runner lifecycle, CLI, and response mapping |
| HTTP integration | Automated | Authentication, JSON, SSE, errors, timeouts, and disconnect cleanup through ASGI |
| OpenAI client contract | Automated | Official `openai` client models, chat, streaming, usage, and function tools |
| OpenAPI contract | Automated | Generated-schema drift plus independent OpenAPI 3.1 validation |
| Package smoke test | Automated | Build, isolated wheel install, CLI, live process health, models, schema, and validation error |
| Live Factory inference | Manual | Requires an authenticated local Droid CLI and can consume account quota |

The automated suite does not access Factory services or require credentials.
Public CI intentionally excludes live inference because it requires a Factory
account. The local curl and client examples above provide the credentialed
end-to-end smoke path before deployment.

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution requirements and
[SECURITY.md](SECURITY.md) for private vulnerability reporting. Release history
is maintained in [CHANGELOG.md](CHANGELOG.md).

## License

Licensed under the [Apache License 2.0](LICENSE).
