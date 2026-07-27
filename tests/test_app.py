from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import httpx
import pytest
from fastapi.exceptions import RequestValidationError

from factory_droid_openai.app import (
    AdmissionController,
    AdmissionLease,
    FinalizingStreamingResponse,
    RequestSizeLimitMiddleware,
    SessionRegistry,
    _collect_completion_or_disconnect,
    _content_length,
    _finalize_stream,
    _JsonDepthTracker,
    _stream_completion,
    _validation_message,
    create_app,
)
from factory_droid_openai.config import Settings
from factory_droid_openai.droid_rpc import (
    CompactionResult,
    ContextBreakdown,
    ContextCategory,
    ContextStats,
)
from factory_droid_openai.metrics import BridgeMetrics
from factory_droid_openai.pool import BackgroundReaper
from factory_droid_openai.protocol import (
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    ToolCallStreamParser,
)
from factory_droid_openai.runner import (
    DroidModel,
    ReasoningDelta,
    RunComplete,
    RunEvent,
    RunnerError,
    RunRequest,
    SessionKey,
    SessionStarted,
    StatusUpdate,
    TextDelta,
    Usage,
    UsageUpdate,
    WarmSession,
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
        self.closed = False
        self.session_operations: list[tuple[str, str, object | None]] = []

    async def run(self, request: RunRequest) -> AsyncIterator[RunEvent]:
        self.requests.append(request)
        try:
            if self.error is not None:
                raise self.error
            for event in self.events:
                yield event
        finally:
            self.closed = True

    async def list_models(self, *, timeout_seconds: float) -> tuple[DroidModel, ...]:
        del timeout_seconds
        return (
            DroidModel(
                id="gpt-5.4",
                display_name="GPT-5.4",
                provider="openai",
                supported_reasoning_efforts=("low", "high"),
                default_reasoning_effort="high",
                supports_images=True,
                supports_pdfs=True,
            ),
        )

    async def get_context(
        self,
        session_id: str,
        *,
        timeout_seconds: float,
    ) -> tuple[ContextStats, ContextBreakdown]:
        del timeout_seconds
        self.session_operations.append(("context", session_id, None))
        return (
            ContextStats(100, 900, 1000, "exact", "2026-07-27T00:00:00Z"),
            ContextBreakdown(
                "gpt-5.4",
                "GPT-5.4",
                1000,
                100,
                900,
                (ContextCategory("messages", 100, "blue"),),
            ),
        )

    async def compact_session(
        self,
        session_id: str,
        *,
        custom_instructions: str | None,
        timeout_seconds: float,
    ) -> CompactionResult:
        del timeout_seconds
        self.session_operations.append(("compact", session_id, custom_instructions))
        return CompactionResult("session-compact", 4)

    async def fork_session(self, session_id: str, *, timeout_seconds: float) -> str:
        del timeout_seconds
        self.session_operations.append(("fork", session_id, None))
        return "session-fork"

    async def rename_session(
        self,
        session_id: str,
        *,
        title: str,
        timeout_seconds: float,
    ) -> None:
        del timeout_seconds
        self.session_operations.append(("rename", session_id, title))

    async def close_session(self, session_id: str, *, timeout_seconds: float) -> None:
        del timeout_seconds
        self.session_operations.append(("close", session_id, None))


class BlockingRunner(FakeRunner):
    def __init__(self) -> None:
        super().__init__([])
        self.started = asyncio.Event()

    async def run(self, request: RunRequest) -> AsyncIterator[RunEvent]:
        self.requests.append(request)
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.closed = True
        if False:
            yield RunComplete(Usage())


class GateRunner(FakeRunner):
    def __init__(self) -> None:
        super().__init__([])
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self, request: RunRequest) -> AsyncIterator[RunEvent]:
        self.requests.append(request)
        self.started.set()
        try:
            await self.release.wait()
            yield RunComplete(Usage())
        finally:
            self.closed = True


class BlockingAdmission:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.proceed = asyncio.Event()
        self.releases = 0

    async def release(self) -> None:
        self.started.set()
        await self.proceed.wait()
        self.releases += 1


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
    alias = models.json()["data"][0]
    assert alias["id"] == "factory-droid"
    assert alias["created"] > 0
    assert models.json()["data"][1] == {
        "id": "gpt-5.4",
        "object": "model",
        "created": alias["created"],
        "owned_by": "openai",
        "factory_droid_display_name": "GPT-5.4",
        "factory_droid_supported_reasoning_efforts": ["low", "high"],
        "factory_droid_default_reasoning_effort": "high",
        "factory_droid_supports_images": True,
        "factory_droid_supports_pdfs": True,
    }


@pytest.mark.asyncio
async def test_models_falls_back_to_alias_when_droid_is_unavailable(
    tmp_path: Path,
) -> None:
    class UnavailableRunner(FakeRunner):
        async def list_models(self, *, timeout_seconds: float) -> tuple[DroidModel, ...]:
            del timeout_seconds
            raise RunnerError(
                "Droid missing",
                status_code=503,
                error_type="factory_droid_unavailable",
            )

    runner = UnavailableRunner([])
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.get("/v1/models")
        metrics = await client.get("/metrics")

    assert response.status_code == 200
    assert [model["id"] for model in response.json()["data"]] == ["factory-droid"]
    assert response.headers["x-factory-droid-model-discovery"] == "degraded"
    assert "factory_droid_openai_model_discovery_failures_total 1" in metrics.text
    assert 'outcome="success",status="200"' in metrics.text


@pytest.mark.asyncio
async def test_model_discovery_is_cached_and_shared_between_callers(
    tmp_path: Path,
) -> None:
    class CountingRunner(FakeRunner):
        calls = 0

        async def list_models(self, *, timeout_seconds: float) -> tuple[DroidModel, ...]:
            type(self).calls += 1
            await asyncio.sleep(0)
            return await super().list_models(timeout_seconds=timeout_seconds)

    runner = CountingRunner([])
    async with _client(_app(tmp_path, runner)) as client:
        first, second = await asyncio.gather(
            client.get("/v1/models"),
            client.get("/v1/models"),
        )
        third = await client.get("/v1/models")

    assert CountingRunner.calls == 1
    assert [response.status_code for response in (first, second, third)] == [200, 200, 200]
    assert "x-factory-droid-model-discovery" not in third.headers


@pytest.mark.asyncio
async def test_model_discovery_serves_the_last_known_catalog_when_droid_breaks(
    tmp_path: Path,
) -> None:
    class FlakyRunner(FakeRunner):
        fail = False

        async def list_models(self, *, timeout_seconds: float) -> tuple[DroidModel, ...]:
            if type(self).fail:
                raise RunnerError(
                    "Droid crashed",
                    status_code=502,
                    error_type="factory_droid_error",
                )
            return await super().list_models(timeout_seconds=timeout_seconds)

    runner = FlakyRunner([])
    async with _client(_feature_app(tmp_path, runner, model_cache_seconds=0.0)) as client:
        fresh = await client.get("/v1/models")
        FlakyRunner.fail = True
        stale = await client.get("/v1/models")

    assert [model["id"] for model in fresh.json()["data"]] == ["factory-droid", "gpt-5.4"]
    assert [model["id"] for model in stale.json()["data"]] == ["factory-droid", "gpt-5.4"]
    assert stale.headers["x-factory-droid-model-discovery"] == "degraded"


@pytest.mark.asyncio
async def test_model_discovery_hides_an_alias_collision(tmp_path: Path) -> None:
    class AliasRunner(FakeRunner):
        async def list_models(self, *, timeout_seconds: float) -> tuple[DroidModel, ...]:
            del timeout_seconds
            return (
                DroidModel(
                    id="factory-droid",
                    display_name="Alias collision",
                    provider="factory",
                    supported_reasoning_efforts=("low",),
                    default_reasoning_effort="low",
                    supports_images=False,
                    supports_pdfs=False,
                ),
            )

    runner = AliasRunner([])
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.get("/v1/models")

    assert [model["id"] for model in response.json()["data"]] == ["factory-droid"]
    assert response.json()["data"][0]["owned_by"] == "factory"


@pytest.mark.asyncio
async def test_overloaded_bridge_rejects_model_and_session_operations(
    tmp_path: Path,
) -> None:
    runner = GateRunner()
    app = _feature_app(
        tmp_path,
        runner,
        max_concurrency=1,
        max_queue_size=0,
        session_continuity=True,
        retry_after_seconds=3,
    )
    app.state.sessions.remember("session-9")
    async with _client(app) as client:
        active = asyncio.create_task(client.post("/v1/chat/completions", json=_payload()))
        await runner.started.wait()

        models = await client.get("/v1/models")
        context = await client.get("/v1/factory/sessions/session-9/context")

        runner.release.set()
        assert (await active).status_code == 200

    assert models.status_code == 429
    assert models.headers["retry-after"] == "3"
    assert context.status_code == 429
    assert context.json()["error"]["type"] == "rate_limit_error"
    assert runner.session_operations == []


@pytest.mark.asyncio
async def test_session_operations_time_out_when_the_queue_eats_the_budget(
    tmp_path: Path,
) -> None:
    runner = GateRunner()
    app = _feature_app(
        tmp_path,
        runner,
        timeout_seconds=0.2,
        max_concurrency=1,
        max_queue_size=1,
        session_continuity=True,
    )
    app.state.sessions.remember("session-9")
    async with _client(app) as client:
        active = asyncio.create_task(client.post("/v1/chat/completions", json=_payload()))
        await runner.started.wait()

        response = await client.delete("/v1/factory/sessions/session-9")

        runner.release.set()
        await active

    assert response.status_code == 504
    assert response.json()["error"]["type"] == "factory_droid_timeout"
    assert runner.session_operations == []


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
            json=_payload(
                stream=True,
                stream_options={"include_usage": True},
            ),
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
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert all(chunk["usage"] is None for chunk in chunks[:-1])
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["total_tokens"] == 11


@pytest.mark.asyncio
async def test_streaming_tool_call_handles_split_markers(tmp_path: Path) -> None:
    runner = FakeRunner(
        [
            TextDelta("<tool_"),
            TextDelta('call>{"name":"weather","arguments":{'),
            TextDelta('"city":"Gdansk"}}</tool_call>'),
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
    assert all("usage" not in chunk for chunk in data)


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


@pytest.mark.asyncio
async def test_tool_choice_validation_uses_openai_error_shape(tmp_path: Path) -> None:
    runner = FakeRunner([])
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_payload(tools=[], tool_choice="required"),
        )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "message": "tool_choice='required' needs at least one tool",
        "type": "invalid_request_error",
        "param": None,
        "code": None,
    }


@pytest.mark.asyncio
async def test_non_streaming_protocol_failure_returns_bad_gateway(
    tmp_path: Path,
) -> None:
    runner = FakeRunner(
        [
            TextDelta(f"{TOOL_CALL_OPEN}invalid{TOOL_CALL_CLOSE}"),
            RunComplete(Usage()),
        ]
    )
    payload = _payload(
        tools=[
            {
                "type": "function",
                "function": {"name": "weather", "parameters": {}},
            }
        ]
    )
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "factory_protocol_error"
    assert runner.closed is True


@pytest.mark.asyncio
async def test_non_streaming_incomplete_response_returns_bad_gateway(
    tmp_path: Path,
) -> None:
    runner = FakeRunner([TextDelta("partial")])
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post("/v1/chat/completions", json=_payload())

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "factory_incomplete_response"
    assert runner.closed is True


@pytest.mark.asyncio
async def test_request_options_map_to_runner_request(tmp_path: Path) -> None:
    runner = FakeRunner([RunComplete(Usage())])
    payload = _payload(
        model="claude-sonnet-4",
        timeout=120,
        reasoning_effort="low",
        factory_droid_reasoning_effort="high",
    )
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert runner.requests[0].model == "claude-sonnet-4"
    assert runner.requests[0].timeout_seconds == 30.0
    assert runner.requests[0].reasoning_effort == "high"


@pytest.mark.asyncio
async def test_bounded_queue_rejects_overload_before_stream_headers(
    tmp_path: Path,
) -> None:
    runner = GateRunner()
    app = create_app(
        Settings(
            workdir=tmp_path,
            timeout_seconds=30,
            max_concurrency=1,
            max_queue_size=1,
            retry_after_seconds=2,
        ),
        runner_factory=cast("RunnerFactory", lambda: runner),
    )
    async with _client(app) as client:
        active = asyncio.create_task(client.post("/v1/chat/completions", json=_payload()))
        await runner.started.wait()
        queued = asyncio.create_task(
            client.post("/v1/chat/completions", json=_payload(stream=True))
        )
        await _wait_for_metric(app, "factory_droid_openai_queued_requests 1")

        rejected = await client.post(
            "/v1/chat/completions",
            json=_payload(stream=True),
        )

        assert rejected.status_code == 429
        assert rejected.headers["retry-after"] == "2"
        assert rejected.json()["error"]["type"] == "rate_limit_error"
        runner.release.set()
        assert (await active).status_code == 200
        assert (await queued).status_code == 200


@pytest.mark.asyncio
async def test_queue_wait_consumes_end_to_end_timeout(tmp_path: Path) -> None:
    runner = GateRunner()
    app = create_app(
        Settings(
            workdir=tmp_path,
            timeout_seconds=30,
            max_concurrency=1,
            max_queue_size=1,
        ),
        runner_factory=cast("RunnerFactory", lambda: runner),
    )
    async with _client(app) as client:
        active = asyncio.create_task(client.post("/v1/chat/completions", json=_payload()))
        await runner.started.wait()

        timed_out = await client.post(
            "/v1/chat/completions",
            json=_payload(timeout=0.02),
        )

        assert timed_out.status_code == 504
        assert timed_out.json()["error"]["type"] == "factory_droid_timeout"
        assert len(runner.requests) == 1
        runner.release.set()
        assert (await active).status_code == 200


@pytest.mark.asyncio
async def test_request_body_limit_applies_before_json_parsing(tmp_path: Path) -> None:
    runner = FakeRunner([RunComplete(Usage())])
    raw = json.dumps(_payload(), separators=(",", ":")).encode()
    app = create_app(
        Settings(workdir=tmp_path, max_request_bytes=len(raw)),
        runner_factory=cast("RunnerFactory", lambda: runner),
    )
    async with _client(app) as client:
        accepted = await client.post(
            "/v1/chat/completions",
            content=raw,
            headers={"Content-Type": "application/json"},
        )
        rejected = await client.post(
            "/v1/chat/completions",
            content=raw + b" ",
            headers={"Content-Type": "application/json"},
        )

    assert accepted.status_code == 200
    assert rejected.status_code == 413
    assert rejected.json()["error"]["type"] == "invalid_request_error"
    assert len(runner.requests) == 1


@pytest.mark.asyncio
async def test_chunked_request_body_stops_at_size_limit(tmp_path: Path) -> None:
    runner = FakeRunner([RunComplete(Usage())])
    app = create_app(
        Settings(workdir=tmp_path, max_request_bytes=32),
        runner_factory=cast("RunnerFactory", lambda: runner),
    )

    async def chunks() -> AsyncIterator[bytes]:
        yield b'{"model":"factory-droid",'
        yield b'"messages":[{"role":"user","content":"large"}]}'

    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=chunks(),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 413
    assert not runner.requests


@pytest.mark.asyncio
async def test_request_body_read_has_bounded_timeout(tmp_path: Path) -> None:
    runner = FakeRunner([RunComplete(Usage())])
    app = create_app(
        Settings(workdir=tmp_path, body_timeout_seconds=0.01),
        runner_factory=cast("RunnerFactory", lambda: runner),
    )

    async def slow_chunks() -> AsyncIterator[bytes]:
        yield b'{"model":"factory-droid",'
        await asyncio.Event().wait()

    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=slow_chunks(),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 408
    assert response.json()["error"]["type"] == "request_timeout"
    assert not runner.requests


@pytest.mark.asyncio
async def test_client_disconnect_during_body_does_not_reach_application() -> None:
    called = False
    metrics = BridgeMetrics()

    async def downstream(_scope: Any, _receive: Any, _send: Any) -> None:
        nonlocal called
        called = True

    middleware = RequestSizeLimitMiddleware(
        downstream,
        max_request_bytes=1024,
        max_json_depth=8,
        body_timeout_seconds=1,
        metrics=metrics,
    )
    messages = [
        {
            "type": "http.request",
            "body": b'{"model":"factory-droid"',
            "more_body": True,
        },
        {"type": "http.disconnect"},
    ]

    async def receive() -> Any:
        return messages.pop(0)

    async def send(_message: Any) -> None:
        pytest.fail("disconnect must not send a response")

    await middleware(
        cast(
            "Any",
            {
                "type": "http",
                "path": "/v1/chat/completions",
                "headers": [],
            },
        ),
        receive,
        send,
    )

    assert called is False
    assert 'outcome="cancelled",status="499"} 1' in metrics.render()


@pytest.mark.asyncio
async def test_json_depth_and_transcript_limits_return_413(tmp_path: Path) -> None:
    runner = FakeRunner([RunComplete(Usage())])
    app = create_app(
        Settings(
            workdir=tmp_path,
            max_json_depth=5,
            max_messages=1,
        ),
        runner_factory=cast("RunnerFactory", lambda: runner),
    )
    deep_payload = _payload(metadata={"a": {"b": {"c": {"d": {}}}}})
    async with _client(app) as client:
        depth = await client.post("/v1/chat/completions", json=deep_payload)
        messages = await client.post(
            "/v1/chat/completions",
            json=_payload(
                messages=[
                    {"role": "user", "content": "one"},
                    {"role": "user", "content": "two"},
                ]
            ),
        )

    assert depth.status_code == 413
    assert messages.status_code == 413
    assert not runner.requests


@pytest.mark.asyncio
async def test_invalid_reasoning_is_rejected_before_admission_and_runner(
    tmp_path: Path,
) -> None:
    factory_calls = 0

    def factory() -> FakeRunner:
        nonlocal factory_calls
        factory_calls += 1
        return FakeRunner([])

    app = create_app(
        Settings(workdir=tmp_path),
        runner_factory=cast("RunnerFactory", factory),
    )
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_payload(reasoning_effort="extreme"),
        )
        metrics = await client.get("/metrics")

    assert response.status_code == 400
    assert factory_calls == 0
    assert "factory_droid_openai_active_sessions 0" in metrics.text
    assert "factory_droid_openai_queued_requests 0" in metrics.text


@pytest.mark.asyncio
async def test_metrics_cover_latency_ttft_overload_and_payload_rejections(
    tmp_path: Path,
) -> None:
    runner = FakeRunner([TextDelta("ok"), RunComplete(Usage())])
    raw = json.dumps(_payload(), separators=(",", ":")).encode()
    app = create_app(
        Settings(workdir=tmp_path, max_request_bytes=len(raw)),
        runner_factory=cast("RunnerFactory", lambda: runner),
    )
    async with _client(app) as client:
        assert (
            await client.post(
                "/v1/chat/completions",
                content=raw,
                headers={"Content-Type": "application/json"},
            )
        ).status_code == 200
        assert (
            await client.post(
                "/v1/chat/completions",
                content=raw + b" ",
                headers={"Content-Type": "application/json"},
            )
        ).status_code == 413
        metrics = await client.get("/metrics")

    assert 'outcome="success",status="200"} 1' in metrics.text
    assert 'outcome="payload_too_large",status="413"} 1' in metrics.text
    assert "factory_droid_openai_request_duration_seconds_count 2" in metrics.text
    assert "factory_droid_openai_queue_wait_seconds_count 1" in metrics.text
    assert "factory_droid_openai_ttft_seconds_count 1" in metrics.text
    assert "factory_droid_openai_payload_rejections_total 1" in metrics.text


@pytest.mark.asyncio
async def test_streaming_runner_error_uses_sse_error_shape(tmp_path: Path) -> None:
    runner = FakeRunner(
        [],
        error=RunnerError(
            "Droid failed",
            error_type="factory_droid_sdk_error",
        ),
    )
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_payload(stream=True),
        )
        metrics = await client.get("/metrics")

    events = [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events[-1] == "[DONE]"
    assert json.loads(events[-2])["error"]["type"] == "factory_droid_sdk_error"
    assert 'outcome="error",status="200"} 1' in metrics.text


@pytest.mark.asyncio
async def test_streaming_protocol_error_uses_sse_error_shape(tmp_path: Path) -> None:
    runner = FakeRunner(
        [
            TextDelta(f"{TOOL_CALL_OPEN}invalid{TOOL_CALL_CLOSE}"),
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

    error_event = next(
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith('data: {"error"')
    )
    assert error_event["error"]["type"] == "factory_protocol_error"


@pytest.mark.asyncio
async def test_chat_endpoint_requires_configured_bearer_token(tmp_path: Path) -> None:
    runner = FakeRunner([RunComplete(Usage())])
    app = _app(tmp_path, runner, api_key="secret")
    async with _client(app) as client:
        denied = await client.post("/v1/chat/completions", json=_payload())
        allowed = await client.post(
            "/v1/chat/completions",
            json=_payload(),
            headers={"Authorization": "Bearer secret"},
        )

    assert denied.status_code == 401
    assert allowed.status_code == 200


def test_validation_message_handles_empty_error_list() -> None:
    assert _validation_message(RequestValidationError([])) == "Invalid request."


@pytest.mark.asyncio
async def test_stream_generator_maps_every_event_type() -> None:
    usage = Usage(6, 2, 1, 0)
    runner = FakeRunner(
        [
            TextDelta("Hello"),
            ReasoningDelta("Think"),
            UsageUpdate(usage),
            RunComplete(usage),
        ]
    )

    events = await _collect_stream(runner, include_usage=True)

    assert any('"content":"Hello"' in event for event in events)
    assert any('"reasoning":"Think"' in event for event in events)
    assert any('"total_tokens":8' in event for event in events)
    assert events[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_stream_generator_emits_tool_call_from_finish_buffer() -> None:
    runner = FakeRunner(
        [
            TextDelta(f'{TOOL_CALL_OPEN}{{"name":"weather","arguments":{{}}}}{TOOL_CALL_CLOSE}'),
            RunComplete(Usage()),
        ]
    )

    events = await _collect_stream(
        runner,
        allowed_tool_names=frozenset({"weather"}),
    )

    assert any('"tool_calls"' in event for event in events)
    assert any('"finish_reason":"tool_calls"' in event for event in events)


@pytest.mark.asyncio
async def test_stream_generator_flushes_text_left_in_the_finish_buffer() -> None:
    runner = FakeRunner(
        [
            TextDelta("answer <too"),
            RunComplete(Usage()),
        ]
    )

    events = await _collect_stream(runner)

    assert any('"content":"answer "' in event for event in events)
    assert any('"content":"<too"' in event for event in events)
    assert any('"finish_reason":"stop"' in event for event in events)


@pytest.mark.asyncio
async def test_stream_generator_flushes_a_partial_stop_sequence() -> None:
    runner = FakeRunner(
        [
            TextDelta("keep HA"),
            RunComplete(Usage()),
        ]
    )

    events = await _collect_stream(runner, stop_sequences=("HALT",))

    assert any('"content":"keep "' in event for event in events)
    assert any('"content":"HA"' in event for event in events)


@pytest.mark.asyncio
async def test_stream_generator_maps_incomplete_response_to_sse_error() -> None:
    events = await _collect_stream(FakeRunner([TextDelta("partial")]))

    assert any('"type":"factory_incomplete_response"' in event for event in events)
    assert events[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_stream_generator_cancellation_releases_runner_and_admission() -> None:
    runner = BlockingRunner()
    metrics = BridgeMetrics()
    admission = AdmissionController(max_concurrency=1, max_queue_size=1, metrics=metrics)
    lease = await admission.acquire(asyncio.get_running_loop().time() + 30)
    stream = _make_stream(runner, lease=lease)

    first = await anext(stream)
    pending: asyncio.Future[str] = asyncio.ensure_future(anext(stream))
    await runner.started.wait()
    pending.cancel()

    with pytest.raises(asyncio.CancelledError):
        await pending

    assert '"role":"assistant"' in first
    assert runner.closed is True
    assert "factory_droid_openai_active_sessions 0" in metrics.render()


@pytest.mark.asyncio
async def test_admission_release_completes_under_cancellation() -> None:
    admission = BlockingAdmission()
    lease = AdmissionLease(cast("Any", admission))
    task = asyncio.create_task(lease.release())
    await admission.started.wait()
    task.cancel()
    admission.proceed.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    await lease.release()

    assert admission.releases == 1


@pytest.mark.asyncio
async def test_stream_finalizer_releases_never_started_lease() -> None:
    runner = FakeRunner([RunComplete(Usage())])
    metrics = BridgeMetrics()
    admission = AdmissionController(max_concurrency=1, max_queue_size=1, metrics=metrics)
    lease = await admission.acquire(asyncio.get_running_loop().time() + 30)
    stream = _make_stream(runner, lease=lease)
    request = SimpleNamespace(state=SimpleNamespace(stream_outcome="pending"))

    await _finalize_stream(stream, lease, cast("Any", request))

    assert runner.requests == []
    assert request.state.stream_outcome == "cancelled"
    assert "factory_droid_openai_active_sessions 0" in metrics.render()


@pytest.mark.asyncio
async def test_finalizing_streaming_response_cleans_up_after_send_failure() -> None:
    finalized = False

    async def content() -> AsyncIterator[str]:
        yield "data"

    async def finalizer() -> None:
        nonlocal finalized
        finalized = True

    response = FinalizingStreamingResponse(
        content(),
        finalizer=finalizer,
        media_type="text/event-stream",
        headers={},
    )

    async def receive() -> Any:
        return {"type": "http.disconnect"}

    async def send(_message: Any) -> None:
        raise RuntimeError("send failed")

    with pytest.raises(RuntimeError, match="send failed"):
        await response(
            cast(
                "Any",
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/v1/chat/completions",
                    "headers": [],
                    "asgi": {"spec_version": "2.4"},
                },
            ),
            receive,
            send,
        )

    assert finalized is True


@pytest.mark.asyncio
async def test_non_stream_disconnect_cancels_runner_cleanup() -> None:
    runner = BlockingRunner()

    async def receive() -> Any:
        await runner.started.wait()
        return {"type": "http.disconnect"}

    result = await _collect_completion_or_disconnect(
        request=cast("Any", SimpleNamespace(receive=receive)),
        runner=cast("Any", runner),
        run_request=RunRequest(
            prompt="prompt",
            model="factory-droid",
            model_alias="factory-droid",
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        parser=ToolCallStreamParser(frozenset()),
    )

    assert result is None
    assert runner.closed is True


async def _collect_stream(
    runner: FakeRunner,
    *,
    allowed_tool_names: frozenset[str] = frozenset(),
    include_usage: bool = False,
    stop_sequences: tuple[str, ...] = (),
) -> list[str]:
    metrics = BridgeMetrics()
    admission = AdmissionController(max_concurrency=1, max_queue_size=1, metrics=metrics)
    lease = await admission.acquire(asyncio.get_running_loop().time() + 30)
    return [
        event
        async for event in _stream_completion(
            request_id="chatcmpl-test",
            created=1,
            model="factory-droid",
            parser=ToolCallStreamParser(allowed_tool_names),
            runner=cast("Any", runner),
            run_request=RunRequest(
                prompt="prompt",
                model="factory-droid",
                model_alias="factory-droid",
                reasoning_effort=None,
                timeout_seconds=30,
            ),
            lease=lease,
            include_usage=include_usage,
            stop_sequences=stop_sequences,
        )
    ]


def _make_stream(
    runner: FakeRunner,
    *,
    lease: Any,
) -> AsyncIterator[str]:
    return _stream_completion(
        request_id="chatcmpl-test",
        created=1,
        model="factory-droid",
        parser=ToolCallStreamParser(frozenset()),
        runner=cast("Any", runner),
        run_request=RunRequest(
            prompt="prompt",
            model="factory-droid",
            model_alias="factory-droid",
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        lease=lease,
    )


async def _wait_for_metric(app: Any, expected: str) -> None:
    for _ in range(100):
        if expected in app.state.metrics.render():
            return
        await asyncio.sleep(0)
    pytest.fail(f"metric did not reach expected value: {expected}")


def _feature_app(
    tmp_path: Path,
    runner: FakeRunner,
    **settings_overrides: object,
) -> Any:
    return create_app(
        Settings(workdir=tmp_path, **cast("Any", settings_overrides)),
        runner_factory=cast("RunnerFactory", lambda: runner),
    )


@pytest.mark.asyncio
async def test_stop_sequence_truncates_non_streaming_content(tmp_path: Path) -> None:
    runner = FakeRunner(
        [
            TextDelta("keep this "),
            TextDelta("HALT drop this"),
            RunComplete(Usage()),
        ]
    )
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_payload(stop="HALT"),
        )

    body = response.json()
    assert body["choices"][0]["message"]["content"] == "keep this "
    assert body["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_partial_stop_sequence_is_flushed_into_the_final_content(
    tmp_path: Path,
) -> None:
    runner = FakeRunner([TextDelta("keep HA"), RunComplete(Usage())])
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_payload(stop=["HALT"]),
        )

    assert response.json()["choices"][0]["message"]["content"] == "keep HA"


@pytest.mark.asyncio
async def test_stop_sequence_split_across_deltas_is_still_detected(
    tmp_path: Path,
) -> None:
    runner = FakeRunner(
        [
            TextDelta("alpha HA"),
            TextDelta("LT omega"),
            RunComplete(Usage()),
        ]
    )
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_payload(stop=["HALT"]),
        )

    assert response.json()["choices"][0]["message"]["content"] == "alpha "


@pytest.mark.asyncio
async def test_stop_sequence_truncates_streaming_content(tmp_path: Path) -> None:
    runner = FakeRunner(
        [
            TextDelta("visible "),
            TextDelta("HALT hidden"),
            RunComplete(Usage()),
        ]
    )
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_payload(stream=True, stop="HALT"),
        )

    assert "visible " in response.text
    assert "hidden" not in response.text
    assert '"finish_reason":"stop"' in response.text


@pytest.mark.asyncio
async def test_multiple_choices_run_sequentially_and_sum_usage(
    tmp_path: Path,
) -> None:
    runner = FakeRunner([TextDelta("answer"), RunComplete(Usage(3, 2, 1, 0))])
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post("/v1/chat/completions", json=_payload(n=3))

    body = response.json()
    assert [choice["index"] for choice in body["choices"]] == [0, 1, 2]
    assert all(choice["message"]["content"] == "answer" for choice in body["choices"])
    assert body["usage"]["prompt_tokens"] == 9
    assert body["usage"]["completion_tokens"] == 6
    assert len(runner.requests) == 3


@pytest.mark.asyncio
async def test_choice_count_above_the_configured_cap_is_rejected(
    tmp_path: Path,
) -> None:
    runner = FakeRunner([RunComplete(Usage())])
    async with _client(_feature_app(tmp_path, runner, max_choices=2)) as client:
        response = await client.post("/v1/chat/completions", json=_payload(n=3))

    assert response.status_code == 400
    assert "at most 2" in response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_multiple_choices_are_rejected_for_streaming(tmp_path: Path) -> None:
    runner = FakeRunner([RunComplete(Usage())])
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_payload(n=2, stream=True),
        )

    assert response.status_code == 400
    assert "stream=true" in response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_too_many_stop_sequences_are_rejected(tmp_path: Path) -> None:
    runner = FakeRunner([RunComplete(Usage())])
    async with _client(_feature_app(tmp_path, runner, max_stop_sequences=1)) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_payload(stop=["a", "b"]),
        )

    assert response.status_code == 400
    assert "at most 1 sequences" in response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_session_continuation_is_rejected_when_disabled(tmp_path: Path) -> None:
    runner = FakeRunner([RunComplete(Usage())])
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_payload(factory_droid_session_id="session-1"),
        )

    assert response.status_code == 400
    assert "disabled" in response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_unknown_session_is_rejected_even_when_continuity_is_enabled(
    tmp_path: Path,
) -> None:
    runner = FakeRunner([RunComplete(Usage())])
    app = _feature_app(tmp_path, runner, session_continuity=True)
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_payload(factory_droid_session_id="not-ours"),
        )

    assert response.status_code == 404
    assert response.json()["error"]["type"] == "session_not_found"


@pytest.mark.asyncio
async def test_session_id_round_trips_and_drives_a_continuation(
    tmp_path: Path,
) -> None:
    runner = FakeRunner(
        [
            SessionStarted("session-77"),
            TextDelta("first reply"),
            RunComplete(Usage()),
        ]
    )
    app = _feature_app(tmp_path, runner, session_continuity=True)
    async with _client(app) as client:
        first = await client.post("/v1/chat/completions", json=_payload())
        second = await client.post(
            "/v1/chat/completions",
            json=_payload(
                factory_droid_session_id="session-77",
                messages=[
                    {"role": "user", "content": "first question"},
                    {"role": "assistant", "content": "first reply"},
                    {"role": "user", "content": "second question"},
                ],
            ),
        )

    assert first.json()["factory_droid_session_id"] == "session-77"
    assert first.headers["x-factory-droid-session-id"] == "session-77"
    assert second.status_code == 200
    assert runner.requests[1].session_id == "session-77"
    assert "second question" in runner.requests[1].prompt
    assert "first question" not in runner.requests[1].prompt


@pytest.mark.asyncio
async def test_multiple_choices_cannot_continue_a_session(tmp_path: Path) -> None:
    runner = FakeRunner([SessionStarted("session-9"), RunComplete(Usage())])
    app = _feature_app(tmp_path, runner, session_continuity=True)
    async with _client(app) as client:
        await client.post("/v1/chat/completions", json=_payload())
        response = await client.post(
            "/v1/chat/completions",
            json=_payload(n=2, factory_droid_session_id="session-9"),
        )

    assert response.status_code == 400
    assert "cannot continue" in response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_session_id_is_hidden_when_continuity_is_disabled(
    tmp_path: Path,
) -> None:
    runner = FakeRunner([SessionStarted("session-3"), RunComplete(Usage())])
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post("/v1/chat/completions", json=_payload())

    assert "factory_droid_session_id" not in response.json()
    assert "x-factory-droid-session-id" not in response.headers


@pytest.mark.asyncio
async def test_status_events_are_opt_in_for_streaming(tmp_path: Path) -> None:
    events: list[RunEvent] = [
        StatusUpdate("executing_tool"),
        TextDelta("done"),
        RunComplete(Usage()),
    ]
    async with _client(_app(tmp_path, FakeRunner(list(events)))) as client:
        without = await client.post(
            "/v1/chat/completions",
            json=_payload(stream=True),
        )
    async with _client(_app(tmp_path, FakeRunner(list(events)))) as client:
        with_status = await client.post(
            "/v1/chat/completions",
            json=_payload(stream=True, factory_droid_status=True),
        )

    assert "factory_droid_status" not in without.text
    assert '"factory_droid_status":"executing_tool"' in with_status.text


@pytest.mark.asyncio
async def test_streaming_session_id_is_announced_when_continuity_is_enabled(
    tmp_path: Path,
) -> None:
    runner = FakeRunner([SessionStarted("session-5"), RunComplete(Usage())])
    app = _feature_app(tmp_path, runner, session_continuity=True)
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_payload(stream=True),
        )

    assert '"factory_droid_session_id":"session-5"' in response.text


@pytest.mark.asyncio
async def test_streaming_emits_indexed_parallel_tool_calls(tmp_path: Path) -> None:
    first = f'{TOOL_CALL_OPEN}{{"name":"weather","arguments":{{"city":"A"}}}}{TOOL_CALL_CLOSE}'
    second = f'{TOOL_CALL_OPEN}{{"name":"weather","arguments":{{"city":"B"}}}}{TOOL_CALL_CLOSE}'
    runner = FakeRunner([TextDelta(first + second), RunComplete(Usage())])
    payload = _payload(
        stream=True,
        tools=[
            {
                "type": "function",
                "function": {"name": "weather", "parameters": {"type": "object"}},
            }
        ],
    )
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert '"index":0' in response.text
    assert '"index":1' in response.text
    assert '"finish_reason":"tool_calls"' in response.text


@pytest.mark.asyncio
async def test_parallel_tool_calls_false_keeps_the_single_call_contract(
    tmp_path: Path,
) -> None:
    first = f'{TOOL_CALL_OPEN}{{"name":"weather","arguments":{{"city":"A"}}}}{TOOL_CALL_CLOSE}'
    second = f'{TOOL_CALL_OPEN}{{"name":"weather","arguments":{{"city":"B"}}}}{TOOL_CALL_CLOSE}'
    runner = FakeRunner([TextDelta(first + second), RunComplete(Usage())])
    payload = _payload(
        parallel_tool_calls=False,
        tools=[
            {
                "type": "function",
                "function": {"name": "weather", "parameters": {"type": "object"}},
            }
        ],
    )
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "factory_protocol_error"


@pytest.mark.asyncio
async def test_image_attachments_reach_the_runner_and_leave_the_prompt(
    tmp_path: Path,
) -> None:
    runner = FakeRunner([TextDelta("described"), RunComplete(Usage())])
    payload = _payload(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,QUJD"},
                    },
                ],
            }
        ]
    )
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert runner.requests[0].images == (
        {"type": "base64", "mediaType": "image/png", "data": "QUJD"},
    )
    assert "QUJD" not in runner.requests[0].prompt


@pytest.mark.asyncio
async def test_rejected_attachment_returns_a_client_error(tmp_path: Path) -> None:
    runner = FakeRunner([RunComplete(Usage())])
    payload = _payload(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/cat.png"},
                    }
                ],
            }
        ]
    )
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 400
    assert "remote image URLs" in response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_json_schema_response_format_is_enforced_and_forwarded(
    tmp_path: Path,
) -> None:
    runner = FakeRunner([TextDelta('{"answer":7}'), RunComplete(Usage())])
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    payload = _payload(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": True,
                "schema": schema,
            },
        }
    )
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == '{"answer":7}'
    assert runner.requests[0].output_format == {
        "type": "json_schema",
        "schema": schema,
    }


@pytest.mark.asyncio
async def test_json_object_response_format_maps_to_a_generic_schema(
    tmp_path: Path,
) -> None:
    runner = FakeRunner([TextDelta('{"ok":true}'), RunComplete(Usage())])
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_payload(response_format={"type": "json_object"}),
        )

    assert response.status_code == 200
    assert runner.requests[0].output_format == {
        "type": "json_schema",
        "schema": {"type": "object", "additionalProperties": True},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_format",
    [
        {
            "type": "json_schema",
            "json_schema": {
                "name": "not_object",
                "schema": {"type": "array"},
            },
        },
        {
            "type": "json_schema",
            "json_schema": {
                "name": "invalid",
                "schema": {"type": "object", "required": "not-an-array"},
            },
        },
        {
            "type": "json_schema",
            "json_schema": {
                "name": "remote",
                "schema": {
                    "type": "object",
                    "properties": {"value": {"$ref": "https://example.com/schema"}},
                },
            },
        },
    ],
)
async def test_invalid_response_format_is_rejected_before_the_runner(
    tmp_path: Path,
    response_format: dict[str, object],
) -> None:
    runner = FakeRunner([RunComplete(Usage())])
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_payload(response_format=response_format),
        )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert runner.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        '{"answer":"wrong"}',
        '{"answer":1,"answer":2}',
        "NaN",
        '{"answer":1e999}',
        '{"answer":1} trailing',
    ],
)
async def test_invalid_structured_output_fails_closed(
    tmp_path: Path,
    text: str,
) -> None:
    runner = FakeRunner([TextDelta(text), RunComplete(Usage())])
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
    }
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_payload(
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "answer", "schema": schema},
                }
            ),
        )

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "factory_protocol_error"


@pytest.mark.asyncio
async def test_streaming_structured_output_is_checked_before_finish(
    tmp_path: Path,
) -> None:
    runner = FakeRunner([TextDelta('{"answer":7}'), RunComplete(Usage())])
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_payload(
                stream=True,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "integer"}},
                            "required": ["answer"],
                        },
                    },
                },
            ),
        )

    assert '"content":"{\\"answer\\":7}"' in response.text
    assert '"finish_reason":"stop"' in response.text


@pytest.mark.asyncio
async def test_invalid_streaming_structured_output_is_not_emitted(
    tmp_path: Path,
) -> None:
    runner = FakeRunner([TextDelta('{"answer":"wrong"}'), RunComplete(Usage())])
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_payload(
                stream=True,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "integer"}},
                            "required": ["answer"],
                        },
                    },
                },
            ),
        )

    chunks = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    content = [
        choice["delta"].get("content")
        for chunk in chunks
        for choice in chunk.get("choices", [])
        if choice["delta"].get("content")
    ]
    assert content == []
    assert chunks[-1]["error"]["type"] == "factory_protocol_error"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [
        {"stop": "HALT"},
        {
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {}},
                }
            ]
        },
    ],
)
async def test_response_format_rejects_incompatible_bridge_options(
    tmp_path: Path,
    extra: dict[str, object],
) -> None:
    runner = FakeRunner([RunComplete(Usage())])
    payload = _payload(response_format={"type": "json_object"}, **extra)
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 400
    assert runner.requests == []


@pytest.mark.asyncio
async def test_guarded_factory_session_operations(tmp_path: Path) -> None:
    runner = FakeRunner(
        [
            SessionStarted("session-77"),
            TextDelta("created"),
            RunComplete(Usage()),
        ]
    )
    app = _feature_app(tmp_path, runner, session_continuity=True)
    async with _client(app) as client:
        assert (await client.post("/v1/chat/completions", json=_payload())).status_code == 200
        context = await client.get("/v1/factory/sessions/session-77/context")
        compact = await client.post(
            "/v1/factory/sessions/session-77/compact",
            json={"custom_instructions": "Keep decisions."},
        )
        fork = await client.post("/v1/factory/sessions/session-77/fork")
        rename = await client.patch(
            "/v1/factory/sessions/session-77",
            json={"title": "New title"},
        )
        close = await client.delete("/v1/factory/sessions/session-77")
        missing = await client.get("/v1/factory/sessions/session-77/context")

    assert context.json()["stats"]["used"] == 100
    assert context.json()["breakdown"]["categories"][0]["name"] == "messages"
    assert compact.json() == {"session_id": "session-compact", "removed_count": 4}
    assert fork.json() == {"session_id": "session-fork"}
    assert rename.json()["status"] == "renamed"
    assert close.json()["status"] == "closed"
    assert missing.status_code == 404
    assert runner.session_operations == [
        ("context", "session-77", None),
        ("compact", "session-77", "Keep decisions."),
        ("fork", "session-77", None),
        ("rename", "session-77", "New title"),
        ("close", "session-77", None),
    ]


@pytest.mark.asyncio
async def test_response_format_schema_over_the_size_cap_is_rejected(tmp_path: Path) -> None:
    runner = FakeRunner([RunComplete(Usage())])
    schema = {
        "type": "object",
        "properties": {f"field_{index}": {"type": "string"} for index in range(64)},
    }
    app = _feature_app(tmp_path, runner, max_tool_schema_bytes=128)
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_payload(
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "wide", "schema": schema},
                }
            ),
        )
        metrics = await client.get("/metrics")

    assert response.status_code == 413
    assert "schema exceeds maximum" in response.json()["error"]["message"]
    assert runner.requests == []
    assert "factory_droid_openai_payload_rejections_total 1" in metrics.text


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_structured_output_over_the_byte_cap_fails_closed(
    tmp_path: Path,
    stream: bool,
) -> None:
    runner = FakeRunner(
        [
            TextDelta(json.dumps({"answer": "x" * 256})),
            RunComplete(Usage()),
        ]
    )
    app = _feature_app(tmp_path, runner, max_structured_output_bytes=64)
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_payload(
                stream=stream,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                        },
                    },
                },
            ),
        )

    if not stream:
        assert response.status_code == 502
        assert "exceeds maximum" in response.json()["error"]["message"]
        return
    chunks = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    assert chunks[-1]["error"]["type"] == "factory_protocol_error"
    assert "exceeds maximum" in chunks[-1]["error"]["message"]
    assert not [
        choice
        for chunk in chunks
        for choice in chunk.get("choices", [])
        if choice["delta"].get("content")
    ]


@pytest.mark.asyncio
async def test_request_limits_cover_the_factory_session_endpoints(tmp_path: Path) -> None:
    runner = FakeRunner([])
    app = _feature_app(
        tmp_path,
        runner,
        session_continuity=True,
        max_request_bytes=256,
    )
    app.state.sessions.remember("session-9")
    async with _client(app) as client:
        oversized = await client.post(
            "/v1/factory/sessions/session-9/compact",
            json={"custom_instructions": "x" * 512},
        )
        deep = await client.patch(
            "/v1/factory/sessions/session-9",
            content=b'{"title":' + b"[" * 64 + b"]" * 64 + b"}",
            headers={"content-type": "application/json"},
        )
        metrics = await client.get("/metrics")

    assert oversized.status_code == 413
    assert deep.status_code == 413
    assert runner.session_operations == []
    assert 'outcome="payload_too_large",status="413"' in metrics.text


@pytest.mark.asyncio
async def test_factory_session_operations_require_continuity(tmp_path: Path) -> None:
    runner = FakeRunner([])
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.get("/v1/factory/sessions/session-1/context")

    assert response.status_code == 400
    assert "disabled" in response.json()["error"]["message"]


def test_session_registry_evicts_the_oldest_entries() -> None:
    registry = SessionRegistry(2)

    registry.remember("a")
    registry.remember("b")
    assert registry.knows("a") is True

    registry.remember("c")

    assert registry.knows("b") is False
    assert registry.knows("a") is True
    assert registry.knows("c") is True


def test_session_registry_ignores_duplicate_entries() -> None:
    registry = SessionRegistry(2)

    registry.remember("a")
    registry.remember("a")
    registry.remember("b")

    assert registry.knows("a") is True
    assert registry.knows("b") is True


@pytest.mark.asyncio
async def test_request_limit_middleware_passes_non_http_scopes_through() -> None:
    seen: list[str] = []

    async def downstream(scope: Any, _receive: Any, _send: Any) -> None:
        seen.append(scope["type"])

    middleware = RequestSizeLimitMiddleware(
        cast("Any", downstream),
        max_request_bytes=16,
        max_json_depth=4,
        body_timeout_seconds=1.0,
        metrics=BridgeMetrics(),
    )

    async def receive() -> Any:  # pragma: no cover - never awaited
        raise AssertionError("lifespan scopes must not read a body")

    async def send(_message: Any) -> None:  # pragma: no cover - never awaited
        raise AssertionError("lifespan scopes must not send a response")

    await middleware(cast("Any", {"type": "lifespan"}), receive, send)

    assert seen == ["lifespan"]


@pytest.mark.parametrize("value", [b"not-a-number", b"-1"])
def test_content_length_ignores_unusable_headers(value: bytes) -> None:
    assert _content_length({"headers": [(b"content-length", value)]}) is None


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 4096])
def test_json_depth_tracker_survives_escapes_split_across_chunks(chunk_size: int) -> None:
    """Brackets inside strings must stay invisible however the body is chunked.

    A backslash landing on a chunk boundary is the case that decides whether
    the next byte is escaped, so the tracker has to carry that state over.
    """
    # The braces sit inside a string, so depth never rises above the outer
    # object; the escaped quote must not be read as closing that string.
    body = rb'{"a":"\"{{{{{{{{{{{{{{{{{{{{{{{{","b":"c\\","d":[1]}'
    tracker = _JsonDepthTracker(3)

    for index in range(0, len(body), chunk_size):
        tracker.feed(body[index : index + chunk_size])

    assert json.loads(body) == {"a": '"{{{{{{{{{{{{{{{{{{{{{{{{', "b": "c\\", "d": [1]}


def test_json_depth_tracker_rejects_nesting_over_the_limit() -> None:
    tracker = _JsonDepthTracker(3)
    tracker.feed(b"[[[")

    with pytest.raises(Exception, match="depth limit"):
        tracker.feed(b"[")


class WarmingRunner(FakeRunner):
    def __init__(self) -> None:
        super().__init__([RunComplete(Usage())])
        self.warmed: list[SessionKey] = []
        self.discarded: list[WarmSession] = []

    async def warm(self, key: SessionKey, *, timeout_seconds: float) -> WarmSession:
        del timeout_seconds
        self.warmed.append(key)
        return WarmSession(
            key=key,
            client=cast("Any", object()),
            transport=None,
            session_id="session-1",
            created_at=asyncio.get_running_loop().time(),
        )

    async def discard(self, session: WarmSession) -> None:
        self.discarded.append(session)


def _warm_session(key: SessionKey) -> WarmSession:
    return WarmSession(
        key=key,
        client=cast("Any", object()),
        transport=None,
        session_id="session-1",
        created_at=asyncio.get_running_loop().time(),
    )


@pytest.mark.asyncio
async def test_chat_completion_uses_a_warm_session(tmp_path: Path) -> None:
    runner = FakeRunner([TextDelta("hi"), RunComplete(Usage())])
    app = _app(tmp_path, runner)
    key = SessionKey(model_id=None, reasoning_effort=None)
    app.state.pool.note(key)
    app.state.pool.offer(_warm_session(key))

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=_payload())

    assert response.status_code == 200
    assert runner.requests[0].warm_session is not None
    assert runner.requests[0].warm_session.session_id == "session-1"
    assert "factory_droid_openai_warm_session_hits_total 1" in app.state.metrics.render()


@pytest.mark.asyncio
async def test_extra_choices_do_not_reuse_the_warm_session(tmp_path: Path) -> None:
    runner = FakeRunner([TextDelta("hi"), RunComplete(Usage())])
    app = _app(tmp_path, runner)
    key = SessionKey(model_id=None, reasoning_effort=None)
    app.state.pool.note(key)
    app.state.pool.offer(_warm_session(key))

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=_payload(n=2))

    assert response.status_code == 200
    assert runner.requests[0].warm_session is not None
    assert runner.requests[1].warm_session is None


@pytest.mark.asyncio
async def test_continuation_requests_skip_the_warm_pool(tmp_path: Path) -> None:
    runner = FakeRunner([SessionStarted("session-9"), RunComplete(Usage())])
    settings = Settings(
        droid_path="droid",
        workdir=tmp_path,
        timeout_seconds=30.0,
        session_continuity=True,
    )
    app = create_app(settings, runner_factory=cast("RunnerFactory", lambda: runner))
    key = SessionKey(model_id=None, reasoning_effort=None)
    app.state.pool.note(key)
    app.state.pool.offer(_warm_session(key))

    async with _client(app) as client:
        first = await client.post("/v1/chat/completions", json=_payload())
        session_id = first.json()["factory_droid_session_id"]
        second = await client.post(
            "/v1/chat/completions",
            json=_payload(factory_droid_session_id=session_id),
        )

    assert second.status_code == 200
    assert runner.requests[1].session_id == "session-9"
    assert runner.requests[1].warm_session is None


@pytest.mark.asyncio
async def test_stream_finalizer_discards_an_unused_warm_session() -> None:
    runner = WarmingRunner()
    metrics = BridgeMetrics()
    admission = AdmissionController(max_concurrency=1, max_queue_size=1, metrics=metrics)
    lease = await admission.acquire(asyncio.get_running_loop().time() + 30)
    stream = _make_stream(runner, lease=lease)
    request = SimpleNamespace(state=SimpleNamespace(stream_outcome="pending"))
    warm = _warm_session(SessionKey(model_id=None, reasoning_effort=None))
    reaper = BackgroundReaper()

    await _finalize_stream(
        stream,
        lease,
        cast("Any", request),
        warm_session=warm,
        reaper=reaper,
        runner_factory=cast("RunnerFactory", lambda: runner),
    )
    await reaper.drain()

    assert runner.discarded == [warm]


@pytest.mark.asyncio
async def test_lifespan_prewarms_and_drains_the_pool(tmp_path: Path) -> None:
    runner = WarmingRunner()
    settings = Settings(
        droid_path="droid",
        workdir=tmp_path,
        timeout_seconds=30.0,
        max_concurrency=1,
    )
    app = create_app(settings, runner_factory=cast("RunnerFactory", lambda: runner))

    async with app.router.lifespan_context(app):
        await asyncio.sleep(0.05)
        assert runner.warmed == [SessionKey(model_id=None, reasoning_effort=None)]
        assert "factory_droid_openai_warm_sessions 1" in app.state.metrics.render()

    assert len(runner.discarded) == 1
    assert "factory_droid_openai_warm_sessions 0" in app.state.metrics.render()


@pytest.mark.asyncio
async def test_detached_cleanup_can_be_disabled(tmp_path: Path) -> None:
    settings = Settings(
        droid_path="droid",
        workdir=tmp_path,
        timeout_seconds=30.0,
        warm_sessions=0,
        detached_cleanup=False,
    )
    app = create_app(settings)

    assert app.state.pool.enabled is False
    async with app.router.lifespan_context(app):
        assert "factory_droid_openai_warm_sessions 0" in app.state.metrics.render()
