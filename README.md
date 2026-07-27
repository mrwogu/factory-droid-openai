# Factory Droid OpenAI Bridge

[![CI](https://github.com/mrwogu/factory-droid-openai/actions/workflows/ci.yml/badge.svg)](https://github.com/mrwogu/factory-droid-openai/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/factory-droid-openai.svg)](https://pypi.org/project/factory-droid-openai/)
[![Docker](https://img.shields.io/badge/docker-ghcr.io-blue.svg)](https://github.com/mrwogu/factory-droid-openai/pkgs/container/factory-droid-openai)
[![Codecov](https://codecov.io/github/mrwogu/factory-droid-openai/graph/badge.svg)](https://codecov.io/github/mrwogu/factory-droid-openai)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

Use your [Factory Droid](https://www.factory.ai/) models in VS Code, Cursor,
Continue, Aider, Zed - any tool that speaks the OpenAI Chat Completions API.

This bridge runs locally on your machine. It translates between OpenAI's
API format and Factory Droid, so your tools talk to `http://127.0.0.1:8787/v1`
like it's OpenAI. Models like GPT-5.4 and Gemini 3.1 Pro become available
anywhere you can set a custom `base_url`. The bridge never runs tools and
never reads your credentials.

Not affiliated with, endorsed by, or maintained by Factory.

## Contents

- [Quick start](#quick-start)
- [Client guides](#client-guides)
  - [VS Code (Copilot BYOK)](#vs-code)
  - [OpenClaw](#openclaw)
  - [Hermes Agent](#hermes-agent)
  - [OpenAI Python client](#openai-python-client)
- [API compatibility](#api-compatibility)
- [API reference](#api-reference)
- [Installation](#installation)
  - [Docker](#docker)
- [Configuration](#configuration)
- [Features](#features)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Limits and metrics](#limits-and-metrics)
- [Authentication](#authentication)
- [Tool execution safety](#tool-execution-safety)
- [Bridge architecture limits](#bridge-architecture-limits)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [License](#license)

## Quick start

Requires Python 3.11+, an installed `droid` CLI, and an authenticated Factory
session. Verify the CLI once with `droid --version`.

Install the bridge as a tool, then run it from the project directory Droid
should operate in. The working directory defaults to the current directory, so
no environment variable is required for local use.

```bash
uv tool install factory-droid-openai
cd /path/to/your/project
factory-droid-openai
```

`pipx install factory-droid-openai` works the same way if you prefer pipx.

No `droid` CLI on the host? Use [Docker](#docker) - it bundles Droid inside
the container, you only need `FACTORY_API_KEY`.

```bash
docker run -d --name droid-bridge \
  -p 127.0.0.1:8787:8787 \
  -e FACTORY_API_KEY="$FACTORY_API_KEY" \
  ghcr.io/mrwogu/factory-droid-openai:latest
```

From another terminal, send a request:

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

Use any model ID from `GET /v1/models`, or the `factory-droid` alias to pick
Droid's configured default.

## API compatibility

✅ Supported · ⚠️ Partial · ❌ Unsupported

| OpenAI capability | Status | Bridge behavior |
|---|---|---|
| Text chat completions | ✅ | Returns one or more assistant choices |
| System and developer messages | ✅ | Serialized with the complete transcript |
| Non-streaming responses | ✅ | OpenAI-compatible JSON completion |
| Streaming responses | ✅ | SSE chunks followed by `[DONE]` |
| Function tool schemas | ✅ | Serialized into the strict Droid prompt |
| Tool choice | ✅ | `auto`, `none`, `required`, or one named function |
| Tool-result continuation | ✅ | Client resends the complete transcript on the next request |
| Parallel tool calls | ✅ | Several sequential tool calls per turn, bounded by a configured cap |
| Stop sequences | ✅ | `stop` truncates the reply and interrupts the Droid turn |
| Reasoning output | ✅ | Emitted as `reasoning` and `reasoning_content` |
| Token usage | ✅ | Includes cache read and write token details |
| Model selection | ✅ | `/v1/models` discovers the authenticated account's current catalog |
| Multimodal content | ⚠️ | Inline image and file data URIs use SDK attachments; remote URLs are rejected |
| Structured outputs | ✅ | `json_schema` and `json_object` are enforced by Droid and validated by the bridge |
| Sampling controls | ❌ | `temperature`, `top_p`, penalties, and `seed` are ignored |
| Multiple choices | ✅ | `n` runs that many Droid turns sequentially and returns `n` choices |
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
- Dynamic OpenAI-compatible `GET /v1/models`
- Non-streaming JSON responses
- Streaming server-sent events with `[DONE]` termination
- System, developer, user, assistant, and tool message mapping
- Function tool schemas and validated tool-call responses
- Reasoning delta and token usage mapping
- Optional constant-time bearer authentication
- Bounded request timeout, payload size, and process concurrency
- Queue-based admission control and Prometheus-style metrics
- Inline image and document attachments over the native SDK channel
- Stop sequences, `n` choices, and multiple tool calls per turn
- Optional Droid session continuity across requests
- Guarded context, compaction, fork, rename, and close session extensions
- Client disconnect cancellation
- Discoverable, verified Factory-native tool disabling
- Versioned and externally validated OpenAPI 3.1 contract
- Strict typing, locked dependencies, and at least 95% branch coverage

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
- Authenticated Droid session or `FACTORY_API_KEY`

Verify the CLI before starting the bridge:

```bash
droid --version
```

Run Droid normally once if authentication or first-time setup is required.
For service-account execution, export `FACTORY_API_KEY`; the Droid subprocess
inherits it directly. The bridge does not rename or copy that credential.

### Python

[Python](https://www.python.org/downloads/) 3.11 or newer (3.12, 3.13, and
3.14 are also tested). On Windows, the
[official installer](https://www.python.org/downloads/windows/) works; ensure
Python is on `PATH` or invoke it via `py -3`.

### Factory Droid CLI

Install the [Factory `droid` CLI](https://docs.factory.ai/reference/cli-reference#installation):

| OS | Command |
|---|---|
| macOS / Linux | `curl -fsSL https://app.factory.ai/cli \| sh` |
| Windows (PowerShell) | `irm https://app.factory.ai/cli/windows \| iex` |
| Any (npm) | `npm install -g droid` |

Authenticate interactively once, or set a `FACTORY_API_KEY` service-account
key generated at
[app.factory.ai/settings/api-keys](https://app.factory.ai/settings/api-keys).

### Docker (optional)

[Docker](https://docs.docker.com/get-docker/) or
[Docker Desktop](https://docs.docker.com/desktop/) is optional and only needed
to run the bridge as a long-lived container from the prebuilt image on
[GHCR](https://github.com/mrwogu/factory-droid-openai/pkgs/container/factory-droid-openai).
See the [Docker](#docker) section. The bridge itself has no Docker runtime
dependency.

## Installation

Released versions are published to
[PyPI](https://pypi.org/project/factory-droid-openai/).

### uv

```bash
uv tool install factory-droid-openai
factory-droid-openai
```

### pipx

```bash
pipx install factory-droid-openai
factory-droid-openai
```

### pip

```bash
python3 -m venv .venv
.venv/bin/python -m pip install factory-droid-openai
.venv/bin/factory-droid-openai
```

### From source

```bash
git clone https://github.com/mrwogu/factory-droid-openai.git
cd factory-droid-openai
uv sync
uv run factory-droid-openai
```

### Docker

A prebuilt multi-arch image is published to GitHub Container Registry for
each release:

```bash
docker pull ghcr.io/mrwogu/factory-droid-openai:latest
```

The image bundles a pinned Droid CLI with auto-updates disabled, so the
bridge and CLI versions are reproducible. Droid runs as a non-root user
(`uid 1000`) and the bridge listens on `0.0.0.0:8787` inside the container.

Run it with a Factory service-account key and a bridge bearer token. Bind
the port to loopback on the host so the bridge is not exposed to the
network:

```bash
docker run -d --name droid-bridge \
  -p 127.0.0.1:8787:8787 \
  -e FACTORY_API_KEY="$FACTORY_API_KEY" \
  -e FACTORY_DROID_OPENAI_API_KEY="$FACTORY_DROID_OPENAI_API_KEY" \
  -v "$PWD":/work:ro \
  ghcr.io/mrwogu/factory-droid-openai:latest
```

`docker compose up -d` with the included `docker-compose.yml` does the same.
Copy the compose file, set `FACTORY_API_KEY` and
`FACTORY_DROID_OPENAI_API_KEY` in your environment or a `.env` file, and run:

```bash
export FACTORY_API_KEY=fk-...
export FACTORY_DROID_OPENAI_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
docker compose up -d
```

The `/work` volume is optional. The bridge disables all Factory-native
tools, so Droid never writes to the working directory; a read-only mount
is safe and sufficient when you want Droid to see project context. Remove
the `-v` flag or the `volumes` block to run without a workdir.

To build the image locally instead of pulling it:

```bash
docker build -t factory-droid-openai .
# or, pinning the Droid CLI version:
docker build --build-arg DROID_VERSION=0.174.0 -t factory-droid-openai .
```

Authentication inside a container uses `FACTORY_API_KEY` only. Interactive
`droid` login does not work in a container, so generate a key at
[app.factory.ai/settings/api-keys](https://app.factory.ai/settings/api-keys)
and pass it as an environment variable. The bridge never reads or copies
that credential; the Droid subprocess inherits it directly.

Never publish the bridge port to a non-loopback interface without a
configured `FACTORY_DROID_OPENAI_API_KEY`. See
[SECURITY.md](SECURITY.md) before any remote deployment.

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

### Structured output

Use OpenAI `response_format` with either `json_schema` or `json_object`:

```python
response = client.chat.completions.create(
    model="gpt-5.4",
    messages=[{"role": "user", "content": "Pick an integer from 1 to 10."}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "integer"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    },
)
```

The bridge sends the schema through Droid's native `outputFormat` RPC field,
then independently parses and validates the completed JSON. Invalid JSON,
duplicate keys, non-finite numbers, schema violations, and trailing output
fail closed. Remote JSON Schema references are rejected. `response_format`
cannot be combined with bridge text tools or `stop`. Structured streaming
buffers the JSON until it validates, then emits it in one content chunk, so no
content chunks arrive while the model is generating. Buffered output is capped
by `FACTORY_DROID_OPENAI_MAX_STRUCTURED_OUTPUT_BYTES`, and a schema larger than
`FACTORY_DROID_OPENAI_MAX_TOOL_SCHEMA_BYTES` is rejected with `413`.

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

## VS Code

VS Code's
[Bring Your Own Key](https://code.visualstudio.com/docs/agent-customization/language-models#_bring-your-own-language-model-key)
Custom Endpoint provider speaks Chat Completions, so it can drive the bridge
without a Copilot plan or a GitHub sign-in.

Open the model picker, select **Manage Language Models**, then **Add Models**
and **Custom Endpoint**. Pick **Chat Completions** as the API type. VS Code
opens `chatLanguageModels.json`; add this provider entry:

```json
[
  {
    "name": "Factory Droid",
    "vendor": "customendpoint",
    "apiKey": "none",
    "apiType": "chat-completions",
    "models": [
      {
        "id": "gpt-5.4",
        "name": "GPT-5.4 via Factory Droid",
        "url": "http://127.0.0.1:8787/v1/chat/completions",
        "toolCalling": true,
        "thinking": true,
        "streaming": true,
        "supportsReasoningEffort": ["minimal", "low", "medium", "high"],
        "maxInputTokens": 180000,
        "maxOutputTokens": 20000
      }
    ]
  }
]
```

`id` is the explicit Droid model ID. Replace it with another ID from
`GET /v1/models` or `droid exec --help`, such as `gemini-3.1-pro-preview` or
the `factory-droid` alias. Each ID needs its own entry in `models`.

When bearer authentication is enabled, set the same token as `apiKey`. The
placeholder `none` works only on a loopback bridge without a configured token.

`toolCalling: true` is required for the model to appear in agent sessions.
`thinking` and `supportsReasoningEffort` surface the reasoning effort picker;
the bridge maps those levels to Droid reasoning effort.

VS Code retains ownership of tool execution. The bridge converts Droid's
strict tool protocol into OpenAI `tool_calls`, and VS Code returns tool
results in the next serialized transcript.

BYOK covers chat and utility tasks only. Inline suggestions (code
completions), semantic search, and embedding-dependent features still require
a Copilot plan and a GitHub account. Without a GitHub sign-in, set
`chat.utilityModel` and `chat.utilitySmallModel` to a configured model so
title generation, commit messages, and intent detection keep working.

The bridge defaults to one concurrent Droid subprocess
(`FACTORY_DROID_OPENAI_MAX_CONCURRENCY=1`), so a chat turn blocks utility
tasks until it finishes. Raise that limit, or point utility tasks at a
separate lightweight model, to avoid serial stalls.

### Coverage with VS Code BYOK

What works through the Custom Endpoint provider:

- Chat and agent sessions, including tool calling
- Streaming responses
- Reasoning effort picker (`thinking`, `supportsReasoningEffort`)
- Structured output via `response_format`
- Inline image and PDF attachments sent as `data:` URIs

What VS Code BYOK does not cover, regardless of the bridge:

- Inline suggestions (code completions)
- Semantic search and embedding-dependent features
- Any feature that requires a Copilot plan or GitHub sign-in

What the bridge accepts but ignores when VS Code sends it:
`temperature`, `top_p`, `seed`, `max_tokens`, `max_completion_tokens`,
`logprobs`, and `top_logprobs`. These are documented in the
[API compatibility](#api-compatibility) table. Remote `http(s)` image URLs are
rejected with `400`; send vision inputs as base64 `data:` URIs instead.
VSCode does not auto-discover models from a Custom Endpoint, so every Droid
model ID you want in the picker needs its own entry in `chatLanguageModels.json`.

## API reference

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Process health check |
| `GET` | `/v1/models` | Bridge alias and dynamically discovered Droid models |
| `POST` | `/v1/chat/completions` | Chat completions and streaming |
| `GET` | `/v1/factory/sessions/{id}/context` | Context statistics and breakdown |
| `POST` | `/v1/factory/sessions/{id}/compact` | Compact history into a new session |
| `POST` | `/v1/factory/sessions/{id}/fork` | Fork a session with context preserved |
| `PATCH` | `/v1/factory/sessions/{id}` | Rename a session |
| `DELETE` | `/v1/factory/sessions/{id}` | Close a session |
| `GET` | `/metrics` | Prometheus-style bridge metrics |
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
| `n` | Yes | Runs `n` Droid turns sequentially, capped by the server |
| `stop` | Yes | String or list; truncates output and interrupts the turn |
| `parallel_tool_calls` | Yes | `true` allows several tool calls per turn, `false` allows one |
| `factory_droid_session_id` | Yes | Bridge-specific session continuation, disabled by default |
| `factory_droid_status` | Yes | Bridge-specific Droid working-state SSE events |
| `reasoning_effort` | Yes | Mapped to Droid reasoning effort |
| `factory_droid_reasoning_effort` | Yes | Bridge-specific override |
| `timeout` | Yes | Per-request value capped by server timeout |
| `temperature`, `top_p`, penalties, `seed` | No | Accepted but ignored |
| `max_tokens`, `max_completion_tokens` | No | Accepted but ignored |
| `response_format` | Yes | `json_schema` and `json_object`; validated again before completion |
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

Call `GET /v1/models` to list the bridge alias and the models currently
available to the authenticated Droid CLI. Factory-specific metadata includes
the display name, reasoning efforts, and image and PDF support.

Discovery starts a short-lived Droid session, so the catalog is cached for
`FACTORY_DROID_OPENAI_MODEL_CACHE_SECONDS` and concurrent callers share a
single discovery attempt. If discovery cannot start or authenticate Droid, the
endpoint preserves compatibility by serving the last known catalog, or the
bridge alias alone, and marks the response with the
`x-factory-droid-model-discovery: degraded` header while incrementing
`factory_droid_openai_model_discovery_failures_total`.

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

### Multimodal attachments

Image and file content parts are delivered over the Droid SDK's native
attachment channel instead of being inlined into the prompt text. The
transcript keeps a short placeholder in their place, so positional context
survives without sending the payload twice.

```python
client.chat.completions.create(
    model="gpt-5.4",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this screenshot?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,<base64>"},
                },
            ],
        }
    ],
)
```

Supported inline media types are `image/jpeg`, `image/png`, `image/gif`,
`image/webp`, `application/pdf`, and `text/plain`. Remote `http(s)` URLs and
`file_id` references are rejected with HTTP `400`: the bridge never fetches
remote content on a client's behalf. Model support for images and PDFs still
depends on the selected Droid model.

### Session continuity

By default every request creates a new Droid session and serializes the whole
transcript. Setting `FACTORY_DROID_OPENAI_SESSION_CONTINUITY=true` lets a
client continue an existing session instead, so only the new turn is sent:

```python
first = client.chat.completions.create(model="gpt-5.4", messages=messages)
session_id = first.model_extra["factory_droid_session_id"]

second = client.chat.completions.create(
    model="gpt-5.4",
    messages=[*messages, assistant_reply, next_question],
    extra_body={"factory_droid_session_id": session_id},
)
```

The session ID is also returned in the `x-factory-droid-session-id` response
header, and announced in the first streaming chunk.

Only sessions this bridge process created can be continued. Session IDs
stored by the local Droid CLI - including your own interactive sessions - are
rejected with HTTP `404`, so a client cannot read back conversations it does
not own. Restarting the bridge clears the set of continuable sessions.

The same guard applies to the Factory session extension endpoints. They are
available only when continuity is enabled and only for IDs created by the
current bridge process:

```bash
curl http://127.0.0.1:8787/v1/factory/sessions/$SESSION_ID/context

curl -X POST \
  http://127.0.0.1:8787/v1/factory/sessions/$SESSION_ID/compact \
  -H 'Content-Type: application/json' \
  -d '{"custom_instructions":"Keep API decisions."}'

curl -X POST \
  http://127.0.0.1:8787/v1/factory/sessions/$SESSION_ID/fork

curl -X PATCH \
  http://127.0.0.1:8787/v1/factory/sessions/$SESSION_ID \
  -H 'Content-Type: application/json' \
  -d '{"title":"API design"}'

curl -X DELETE \
  http://127.0.0.1:8787/v1/factory/sessions/$SESSION_ID
```

Compaction and fork return a new `session_id`, which the bridge registers for
later continuation. Context responses contain aggregate token categories, not
message or prompt content. Context, fork, rename, and close run no model turn,
so they leave the session's tool settings untouched; compaction runs a turn and
therefore disables the tool catalog first.

### Droid status events

`factory_droid_status: true` adds Droid working-state transitions to a
streaming response as an extra `factory_droid_status` key inside the chunk
delta:

```json
{"choices":[{"index":0,"delta":{"factory_droid_status":"executing_tool"}}]}
```

States follow the SDK: `idle`, `streaming_assistant_message`,
`waiting_for_tool_confirmation`, `executing_tool`, and
`compacting_conversation`. The field is absent unless explicitly requested, so
strict OpenAI clients are unaffected.

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
| `404` | Unknown Factory Droid session |
| `413` | Request body or serialized transcript over the configured limit |
| `429` | Droid request queue full, with a `Retry-After` header |
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
| `FACTORY_DROID_OPENAI_BODY_TIMEOUT_SECONDS` | `30` | Maximum time to receive a request body |
| `FACTORY_DROID_OPENAI_MAX_CONCURRENCY` | `1` | Concurrent Droid subprocesses |
| `FACTORY_DROID_OPENAI_MAX_QUEUE_SIZE` | `8` | Requests waiting for a Droid slot |
| `FACTORY_DROID_OPENAI_RETRY_AFTER_SECONDS` | `1` | `Retry-After` value sent with `429` |
| `FACTORY_DROID_OPENAI_MAX_REQUEST_BYTES` | `4194304` | Request body size limit |
| `FACTORY_DROID_OPENAI_MAX_MESSAGES` | `512` | Messages accepted per request |
| `FACTORY_DROID_OPENAI_MAX_TOOLS` | `128` | Tool schemas accepted per request |
| `FACTORY_DROID_OPENAI_MAX_TRANSCRIPT_BYTES` | `4194304` | Serialized transcript size limit |
| `FACTORY_DROID_OPENAI_MAX_TOOL_SCHEMA_BYTES` | `1048576` | Serialized tool schema size limit |
| `FACTORY_DROID_OPENAI_MAX_STRUCTURED_OUTPUT_BYTES` | `1048576` | Buffered structured output size limit |
| `FACTORY_DROID_OPENAI_MAX_JSON_DEPTH` | `32` | Request JSON nesting limit |
| `FACTORY_DROID_OPENAI_PROCESS_GRACE_SECONDS` | `1` | Wait before killing a Droid process |
| `FACTORY_DROID_OPENAI_CLEANUP_TIMEOUT_SECONDS` | `4` | Total budget for session cleanup |
| `FACTORY_DROID_OPENAI_UVICORN_LIMIT_CONCURRENCY` | `64` | Uvicorn connection limit |
| `FACTORY_DROID_OPENAI_UVICORN_BACKLOG` | `128` | Uvicorn listen backlog |
| `FACTORY_DROID_OPENAI_MAX_TOOL_CALLS` | `8` | Tool calls accepted per Droid turn |
| `FACTORY_DROID_OPENAI_MAX_ATTACHMENTS` | `16` | Inline attachments per request |
| `FACTORY_DROID_OPENAI_MAX_ATTACHMENT_BYTES` | `8388608` | Total encoded attachment bytes |
| `FACTORY_DROID_OPENAI_MAX_CHOICES` | `4` | Upper bound for `n` |
| `FACTORY_DROID_OPENAI_MAX_STOP_SEQUENCES` | `4` | Upper bound for `stop` entries |
| `FACTORY_DROID_OPENAI_SESSION_CONTINUITY` | `false` | Allow continuing bridge-created sessions |
| `FACTORY_DROID_OPENAI_MAX_TRACKED_SESSIONS` | `256` | Continuable sessions kept in memory |
| `FACTORY_DROID_OPENAI_WORKTREE` | unset | Run Droid in a git worktree |
| `FACTORY_DROID_OPENAI_APPEND_SYSTEM_PROMPT_FILE` | unset | File appended to the Droid system prompt |
| `FACTORY_DROID_OPENAI_MODEL_ALIAS` | `factory-droid` | Alias using Droid default model |
| `FACTORY_DROID_OPENAI_MODEL_CACHE_SECONDS` | `300` | `GET /v1/models` catalog cache lifetime |
| `FACTORY_DROID_OPENAI_MCP_SETTLE_SECONDS` | `0` | MCP initialization window before tool discovery |

Command-line options:

```text
usage: factory-droid-openai [-h] [--host HOST] [--port PORT]
                            [--limit-concurrency LIMIT_CONCURRENCY]
                            [--backlog BACKLOG]
                            [--log-level {critical,error,warning,info,debug,trace}]
```

Environment variables define defaults. The command-line options override the
matching server settings.

## Limits and metrics

Every request passes two gates before a Droid process is involved.

Size limits are enforced while the body is still arriving, so an oversized
payload is rejected with `413` instead of being buffered in full. Body bytes,
message count, tool count, serialized transcript size, tool schema size, and
JSON nesting depth each have their own bound. Body size, nesting depth, the
body timeout, and request metrics cover every HTTP route, including
`GET /v1/models` and the `/v1/factory/sessions` endpoints.

Admission control then bounds how many requests reach Droid. Requests beyond
`FACTORY_DROID_OPENAI_MAX_CONCURRENCY` wait in a queue of at most
`FACTORY_DROID_OPENAI_MAX_QUEUE_SIZE`; once that is full the bridge answers
`429` with a `Retry-After` header rather than queueing without limit.

`GET /metrics` renders Prometheus-style text. It is excluded from the OpenAPI
contract because it is an operational endpoint, not part of the OpenAI API
surface.

| Metric | Meaning |
|---|---|
| `factory_droid_openai_requests_total` | Requests by outcome |
| `factory_droid_openai_request_duration_seconds` | Request duration sum and count |
| `factory_droid_openai_queue_wait_seconds` | Time spent waiting for admission |
| `factory_droid_openai_droid_startup_seconds` | Droid session startup time |
| `factory_droid_openai_ttft_seconds` | Time to first token |
| `factory_droid_openai_active_sessions` | Droid sessions running now |
| `factory_droid_openai_queued_requests` | Requests waiting for admission |
| `factory_droid_openai_overload_rejections_total` | Requests rejected with `429` |
| `factory_droid_openai_payload_rejections_total` | Requests rejected with `413` |
| `factory_droid_openai_forced_kills_total` | Droid processes that needed a kill |
| `factory_droid_openai_model_discovery_failures_total` | Failed Droid model discovery attempts |

`forced_kills_total` counts any process that did not exit within its grace
period, so treat it as an upper bound on genuinely stuck processes.

Serve `/metrics` on a loopback interface or behind your own access control;
it carries no prompt content, but it does expose traffic shape.

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

1. Selects Droid's `auto` interaction mode with autonomy `off`.
2. Requests no additional MCP servers from the SDK.
3. Cancels permission requests and interactive questions.
4. Optionally waits `FACTORY_DROID_OPENAI_MCP_SETTLE_SECONDS` for configured MCP
   servers, which only widens the verification snapshot and is off by default.
5. Discovers the native and MCP tool catalog known at that moment through
   `droid.list_tools`.
6. Disables every discovered tool ID and verifies no unexpected tool remains.
   Tools that appear after this point stay denied, because Droid defaults to
   deny once a disabled set has been sent.
7. Rejects any Factory-native tool event as a second line of defense.
8. Validates generated tool names against the request schema.
9. Requires tool arguments to be a JSON object with unique keys.
10. Caps the number of tool calls accepted in one turn.
11. Rejects any non-whitespace output between or after tool calls.
12. Refuses to fetch remote attachment URLs.
13. Only continues Droid sessions this bridge process created.

Set `FACTORY_DROID_OPENAI_WORKDIR` to the smallest directory required by your
workflow.

## Bridge architecture limits

This is a compatibility bridge, not a native OpenAI inference implementation.

| Limitation | Effect |
|---|---|
| No native SDK chat-history input | History is serialized into one prompt unless session continuity is enabled |
| No native SDK external-tool schemas | Tool definitions are serialized into the prompt |
| No native SDK tool-result continuation | Each request creates a new Droid session |
| Python SDK protocol drift | A small isolated compatibility shim supplies structured output, tool controls, and session RPCs |
| Session-per-request execution | Prompt caching differs from a native inference endpoint |
| Strict tool marker protocol | Invalid generated tool payloads fail closed |
| Sequential tool calls only | Calls arrive one after another, never concurrently |

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
is published on the
[releases page](https://github.com/mrwogu/factory-droid-openai/releases) and
generated from the commit history by Release Please.

## License

Licensed under the [Apache License 2.0](LICENSE).
