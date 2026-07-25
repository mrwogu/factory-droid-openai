# Changelog

All notable changes are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases follow [Semantic Versioning](https://semver.org/).

## Unreleased

## 1.0.0 - 2026-07-25

### Bridge

- OpenAI-compatible `/v1/chat/completions` and `/v1/models` endpoints
- Non-streaming JSON and streaming SSE responses
- Reasoning and token usage mapping
- Model alias and explicit Droid model selection
- Request timeout, concurrency limits, cancellation, and disconnect cleanup

### Tool calling

- Function tool schemas and tool choice support
- Strict tool-call marker protocol with name, payload, and duplicate-key validation
- Client-owned tool execution and tool-result continuation
- Factory-native tool, permission, and interactive-question blocking

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
