from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import httpx
import pytest

from factory_droid_openai.app import create_app
from factory_droid_openai.config import Settings
from factory_droid_openai.protocol import TOOL_CALL_CLOSE, TOOL_CALL_OPEN
from factory_droid_openai.runner import (
    ReasoningDelta,
    RunComplete,
    RunEvent,
    RunnerError,
    RunRequest,
    TextDelta,
    Usage,
    UsageUpdate,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path
    from typing import Any

    from factory_droid_openai.app import RunnerFactory


class FakeRunner:
    def __init__(
        self,
        events: list[RunEvent],
        *,
        error: RunnerError | None = None,
    ) -> None:
        self.events = events
        self.error = error
        self.requests: list[RunRequest] = []

    async def run(self, request: RunRequest) -> AsyncIterator[RunEvent]:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        for event in self.events:
            yield event


def _settings(tmp_path: Path, *, api_key: str | None = None) -> Settings:
    return Settings(
        api_key=api_key,
        droid_path="droid",
        workdir=tmp_path,
        timeout_seconds=30.0,
    )


def _app(
    tmp_path: Path,
    runner: FakeRunner,
    *,
    api_key: str | None = None,
) -> Any:
    return create_app(
        _settings(tmp_path, api_key=api_key),
        runner_factory=cast("RunnerFactory", lambda: runner),
    )


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": "factory-droid",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    payload.update(overrides)
    return payload


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


@pytest.mark.asyncio
async def test_health_and_models(tmp_path: Path) -> None:
    runner = FakeRunner([])
    async with _client(_app(tmp_path, runner)) as client:
        health = await client.get("/health")
        models = await client.get("/v1/models")

    assert health.json() == {"status": "ok"}
    assert models.json()["data"][0]["id"] == "factory-droid"


@pytest.mark.asyncio
async def test_non_streaming_chat_completion(tmp_path: Path) -> None:
    usage = Usage(10, 4, 2, 1)
    runner = FakeRunner(
        [
            ReasoningDelta("brief thought"),
            TextDelta("Hello"),
            TextDelta(" world"),
            UsageUpdate(usage),
            RunComplete(usage),
        ]
    )
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post("/v1/chat/completions", json=_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "Hello world"
    assert body["choices"][0]["message"]["reasoning"] == "brief thought"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] == 14
    assert body["usage"]["prompt_tokens_details"]["cached_tokens"] == 2
    assert runner.requests[0].model == "factory-droid"
    assert "OPENAI_TRANSCRIPT_JSON" in runner.requests[0].prompt


@pytest.mark.asyncio
async def test_non_streaming_tool_call(tmp_path: Path) -> None:
    runner = FakeRunner(
        [
            TextDelta(f'{TOOL_CALL_OPEN}{{"name":"weather",'),
            TextDelta(f'"arguments":{{"city":"Gdansk"}}}}{TOOL_CALL_CLOSE}'),
            RunComplete(Usage()),
        ]
    )
    payload = _payload(
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "parameters": {"type": "object"},
                },
            }
        ]
    )
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    tool_call = choice["message"]["tool_calls"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert tool_call["function"]["name"] == "weather"
    assert json.loads(tool_call["function"]["arguments"]) == {"city": "Gdansk"}


@pytest.mark.asyncio
async def test_streaming_chat_completion_uses_openai_sse(tmp_path: Path) -> None:
    usage = Usage(8, 3, 1, 0)
    runner = FakeRunner(
        [
            ReasoningDelta("think"),
            TextDelta("Hi"),
            RunComplete(usage),
        ]
    )
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_payload(stream=True),
        )

    assert response.status_code == 200
    events = [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(event) for event in events[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[1]["choices"][0]["delta"]["reasoning"] == "think"
    assert chunks[2]["choices"][0]["delta"]["content"] == "Hi"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"]["total_tokens"] == 11


@pytest.mark.asyncio
async def test_streaming_tool_call_handles_split_markers(tmp_path: Path) -> None:
    runner = FakeRunner(
        [
            TextDelta("<hermes_tool_"),
            TextDelta('call>{"name":"weather","arguments":{'),
            TextDelta('"city":"Gdansk"}}</hermes_tool_call>'),
            RunComplete(Usage()),
        ]
    )
    payload = _payload(
        stream=True,
        tools=[
            {
                "type": "function",
                "function": {"name": "weather", "parameters": {}},
            }
        ],
    )
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    data = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    tool_delta = next(
        chunk
        for chunk in data
        if chunk.get("choices") and chunk["choices"][0]["delta"].get("tool_calls")
    )
    assert tool_delta["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "weather"
    assert data[-1]["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_optional_bearer_auth(tmp_path: Path) -> None:
    runner = FakeRunner([])
    app = _app(tmp_path, runner, api_key="secret")
    async with _client(app) as client:
        missing = await client.get("/v1/models")
        invalid = await client.get(
            "/v1/models",
            headers={"Authorization": "Bearer wrong"},
        )
        valid = await client.get(
            "/v1/models",
            headers={"Authorization": "Bearer secret"},
        )

    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert valid.status_code == 200


@pytest.mark.asyncio
async def test_runner_error_maps_to_openai_error(tmp_path: Path) -> None:
    runner = FakeRunner(
        [],
        error=RunnerError(
            "Droid missing",
            status_code=503,
            error_type="factory_droid_unavailable",
        ),
    )
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post("/v1/chat/completions", json=_payload())

    assert response.status_code == 503
    assert response.json()["error"]["type"] == "factory_droid_unavailable"


@pytest.mark.asyncio
async def test_invalid_request_uses_openai_error_shape(tmp_path: Path) -> None:
    runner = FakeRunner([])
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_payload(messages=[]),
        )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
