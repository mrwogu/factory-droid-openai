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
    _collect_completion_or_disconnect,
    _finalize_stream,
    _stream_completion,
    _validation_message,
    create_app,
)
from factory_droid_openai.config import Settings
from factory_droid_openai.metrics import BridgeMetrics
from factory_droid_openai.protocol import (
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    ToolCallStreamParser,
)
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
        self.closed = False

    async def run(self, request: RunRequest) -> AsyncIterator[RunEvent]:
        self.requests.append(request)
        try:
            if self.error is not None:
                raise self.error
            for event in self.events:
                yield event
        finally:
            self.closed = True


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
