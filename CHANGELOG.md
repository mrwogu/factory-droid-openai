# Changelog

All notable changes are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases follow [Semantic Versioning](https://semver.org/).

## Unreleased

## 1.0.0 (2026-07-26)


### chore

* cut the first release as 1.0.0 ([0b71f3c](https://github.com/mrwogu/factory-droid-openai/commit/0b71f3c96dfcfb80d040f28cf481d0cc6bc9748d))


### Features

* add attachments, multi tool calls and session events ([945cb48](https://github.com/mrwogu/factory-droid-openai/commit/945cb485f20e140f810585de5c8da4e2f0f7fd74))
* add OpenAI-compatible Factory Droid bridge ([5b657d5](https://github.com/mrwogu/factory-droid-openai/commit/5b657d50a655ff4363b409987ed699ca53e9e080))
* add PromptScript targets and release 1.0.0 ([b9227bc](https://github.com/mrwogu/factory-droid-openai/commit/b9227bcdb4ca125f60920b5bd2c2093f55e36e93))
* add request limits, admission control and metrics ([9b79773](https://github.com/mrwogu/factory-droid-openai/commit/9b79773d76f58ed2be1b2257778f1fa72fa89f50))
* add security, CI, and package safeguards ([0bb592f](https://github.com/mrwogu/factory-droid-openai/commit/0bb592f7f68332c8a941f664859ee29f68cad341))
* add verified OpenAI contract and OpenClaw guide ([32a6773](https://github.com/mrwogu/factory-droid-openai/commit/32a67731e30752ab26a83380fde2ac1083fd0e1b))
* align streaming usage with OpenAI ([a0fa506](https://github.com/mrwogu/factory-droid-openai/commit/a0fa506b2b75a71e50765a8ae0201ca2fe7fc425))
* wire stop, n, sessions and status into the API ([16208c6](https://github.com/mrwogu/factory-droid-openai/commit/16208c695ba963268f9e698b1d9389f4fda9c4a8))


### Performance Improvements

* scan only structural bytes for the JSON depth guard ([65ea02e](https://github.com/mrwogu/factory-droid-openai/commit/65ea02e090898a71e7ac254ddc5e5bc7f7e8f4c1))

## 1.0.0 - 2026-07-26

### Bridge

- OpenAI-compatible `/v1/chat/completions` and `/v1/models` endpoints
- Non-streaming JSON and streaming SSE responses
- Reasoning and token usage mapping
- Model alias and explicit Droid model selection
- Request timeout, concurrency limits, cancellation, and disconnect cleanup
- Inline image and file attachments over the native Droid SDK channel
- Stop sequences applied during streaming
- Multiple choices per request through `n`
- Optional session continuity across requests
- Optional Droid working-state events on streamed chunks

### Tool calling

- Function tool schemas and tool choice support
- Strict tool-call marker protocol with name, payload, and duplicate-key validation
- Several tool calls per assistant turn
- Client-owned tool execution and tool-result continuation
- Factory-native tool, permission, and interactive-question blocking

### Operations

- Request body, message, tool, transcript, schema, and JSON depth limits
- Queue-based admission control answering `429` with `Retry-After` when full
- Prometheus-style `/metrics` endpoint
- Graceful Droid subprocess shutdown with a forced-reap fallback
- Uvicorn concurrency and backlog tuning on the command line
- Linear-time tool-call parsing and structural-only JSON depth scanning

### API contract

- Versioned OpenAPI 3.1 document
- External OpenAPI validation and generated-schema drift checks
- Official OpenAI Python client contract tests
- OpenAI-compatible streaming usage chunks
- Support and limitation matrices for Chat Completions and other API families

### Integrations

- Hermes Agent provider configuration
- OpenClaw custom `openai-completions` provider configuration
- PromptScript source instructions for Claude Code and Factory Droid

### Security and validation

- Optional constant-time bearer authentication
- Loopback-only default binding
- SHA-pinned GitHub Actions with Python 3.11-3.14 coverage
- Unit, ASGI integration, package, process, and authenticated Droid smoke tests
- Locked dependencies, strict typing, linting, formatting, and 95 percent coverage gate
- Remote attachment URLs and `file_id` references rejected instead of fetched
- Session continuation limited to sessions this bridge process created
