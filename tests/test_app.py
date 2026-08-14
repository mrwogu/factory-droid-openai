from __future__ import annotations

import asyncio
import importlib.metadata
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import httpx
import pytest
from fastapi.exceptions import RequestValidationError
from jsonschema.validators import validator_for

from factory_droid_openai import telemetry as telemetry_module
from factory_droid_openai.app import (
    AdmissionController,
    AdmissionLease,
    FinalizingStreamingResponse,
    RequestSizeLimitMiddleware,
    SessionRegistry,
    StructuredOutput,
    _collect_completion,
    _collect_completion_or_disconnect,
    _content_length,
    _finalize_stream,
    _JsonDepthTracker,
    _latency_bucket,
    _payload_size_bucket,
    _request_mode,
    _request_outcome,
    _request_route,
    _request_telemetry_features,
    _RequestPayloadLimitError,
    _set_request_state,
    _stream_completion,
    _telemetry_error_category,
    _validation_message,
    _wait_for_disconnect,
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
    ProtocolEmission,
    TextEmission,
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


class ToolCallBlockingRunner(FakeRunner):
    """Emits one complete tool call, then trailing events, then never completes."""

    def __init__(self, events: list[RunEvent]) -> None:
        super().__init__(events)
        self.started = asyncio.Event()

    async def run(self, request: RunRequest) -> AsyncIterator[RunEvent]:
        self.requests.append(request)
        self.started.set()
        try:
            yield TextDelta(
                f'{TOOL_CALL_OPEN}{{"name":"weather","arguments":{{"city":"Hel"}}}}'
                f"{TOOL_CALL_CLOSE}"
            )
            for event in self.events:
                yield event
            await asyncio.Event().wait()
        finally:
            self.closed = True


class ToolCallFailingRunner(FakeRunner):
    """Emits one complete tool call, then fails the turn."""

    def __init__(self, failure: RunnerError) -> None:
        super().__init__([])
        self.failure = failure

    async def run(self, request: RunRequest) -> AsyncIterator[RunEvent]:
        self.requests.append(request)
        try:
            yield TextDelta(
                f'{TOOL_CALL_OPEN}{{"name":"weather","arguments":{{"city":"Hel"}}}}'
                f"{TOOL_CALL_CLOSE}"
            )
            raise self.failure
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
async def test_apps_keep_payload_tracing_configuration_isolated(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    disabled_runner = FakeRunner([RunComplete(Usage())])
    enabled_runner = FakeRunner([RunComplete(Usage())])
    disabled_app = create_app(
        Settings(workdir=tmp_path),
        runner_factory=cast("RunnerFactory", lambda: disabled_runner),
    )
    enabled_app = create_app(
        Settings(
            workdir=tmp_path,
            trace_payloads="full",
            trace_payload_file=trace_path,
        ),
        runner_factory=cast("RunnerFactory", lambda: enabled_runner),
    )

    async with _client(disabled_app) as client:
        disabled_response = await client.post(
            "/v1/chat/completions",
            json=_payload(messages=[{"role": "user", "content": "do not trace"}]),
        )
    async with _client(enabled_app) as client:
        enabled_response = await client.post(
            "/v1/chat/completions",
            json=_payload(messages=[{"role": "user", "content": "trace this"}]),
        )

    assert disabled_response.status_code == 200
    assert enabled_response.status_code == 200
    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    prompts = [record for record in records if record["event"] == "chat.prompt"]
    assert len(prompts) == 1
    assert "trace this" in prompts[0]["payload"]
    assert "do not trace" not in prompts[0]["payload"]


@pytest.mark.asyncio
async def test_payload_tracing_records_sdk_events_for_replay(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    runner = FakeRunner(
        [
            SessionStarted("session-secret"),
            StatusUpdate("executing_tool"),
            ReasoningDelta("thinking"),
            TextDelta("hello"),
            UsageUpdate(Usage(input_tokens=1, output_tokens=1)),
            RunComplete(Usage(input_tokens=1, output_tokens=2)),
        ]
    )
    app = create_app(
        Settings(
            workdir=tmp_path,
            trace_payloads="full",
            trace_payload_file=trace_path,
        ),
        runner_factory=cast("RunnerFactory", lambda: runner),
    )

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=_payload())

    assert response.status_code == 200
    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    events = [
        json.loads(record["payload"]) for record in records if record["event"] == "droid.event"
    ]
    assert [event["kind"] for event in events] == [
        "session_started",
        "status",
        "reasoning_delta",
        "text_delta",
        "usage",
        "run_complete",
    ]
    assert events[3]["text"] == "hello"
    assert events[5]["usage"]["completion_tokens"] == 2
    assert "session-secret" not in trace_path.read_text(encoding="utf-8")


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
async def test_version_reports_the_bridge_package_version(tmp_path: Path) -> None:
    runner = FakeRunner([])
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.get("/version")

    assert response.status_code == 200
    assert response.json() == {
        "version": importlib.metadata.version("factory-droid-openai"),
    }


@pytest.mark.asyncio
async def test_retrieve_model_matches_the_list_entry(tmp_path: Path) -> None:
    runner = FakeRunner([])
    async with _client(_app(tmp_path, runner)) as client:
        listed = await client.get("/v1/models")
        alias = await client.get("/v1/models/factory-droid")
        discovered = await client.get("/v1/models/gpt-5.4")

    entries = {model["id"]: model for model in listed.json()["data"]}
    assert alias.status_code == 200
    assert alias.json() == entries["factory-droid"]
    assert discovered.status_code == 200
    assert discovered.json() == entries["gpt-5.4"]


@pytest.mark.asyncio
async def test_retrieve_model_rejects_an_unknown_model(tmp_path: Path) -> None:
    runner = FakeRunner([])
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.get("/v1/models/kimi-k3")

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "message": "Model 'kimi-k3' is not available on this bridge.",
            "type": "model_not_found",
            "param": None,
            "code": None,
        }
    }


@pytest.mark.asyncio
async def test_retrieve_model_serves_the_alias_when_droid_is_unavailable(
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
        alias = await client.get("/v1/models/factory-droid")
        missing = await client.get("/v1/models/gpt-5.4")

    assert alias.status_code == 200
    assert alias.json()["id"] == "factory-droid"
    assert alias.headers["x-factory-droid-model-discovery"] == "degraded"
    assert missing.status_code == 404


@pytest.mark.asyncio
async def test_retrieve_model_withholds_a_quarantined_model(tmp_path: Path) -> None:
    runner = FakeRunner(
        [],
        error=RunnerError(
            "Model 'gpt-5.4' is not available for this Factory account",
            status_code=404,
            error_type="model_not_found",
        ),
    )
    async with _client(_app(tmp_path, runner)) as client:
        rejected = await client.post("/v1/chat/completions", json=_payload(model="gpt-5.4"))
        retrieved = await client.get("/v1/models/gpt-5.4")
        alias = await client.get("/v1/models/factory-droid")

    assert rejected.status_code == 404
    assert retrieved.status_code == 404
    assert retrieved.json()["error"]["type"] == "model_not_found"
    assert alias.status_code == 200


@pytest.mark.asyncio
async def test_chat_quarantines_a_model_droid_refuses(tmp_path: Path) -> None:
    runner = FakeRunner(
        [],
        error=RunnerError(
            "Model 'gpt-5.4' is not available for this Factory account: "
            "Model not allowed by organization policy",
            status_code=404,
            error_type="model_not_found",
        ),
    )
    app = _app(tmp_path, runner)

    async with _client(app) as client:
        first = await client.post("/v1/chat/completions", json=_payload(model="gpt-5.4"))
        models = await client.get("/v1/models")
        second = await client.post("/v1/chat/completions", json=_payload(model="gpt-5.4"))
        metrics = await client.get("/metrics")

    assert first.status_code == 404
    assert first.json()["error"]["type"] == "model_not_found"
    assert [model["id"] for model in models.json()["data"]] == ["factory-droid"]
    assert models.headers["x-factory-droid-models-quarantined"] == "1"
    assert second.status_code == 404
    # The second request is refused before a Droid session is started.
    assert len(runner.requests) == 1
    assert "factory_droid_openai_model_quarantines_total 1" in metrics.text


@pytest.mark.asyncio
async def test_streamed_chat_quarantines_a_model_droid_refuses(tmp_path: Path) -> None:
    runner = FakeRunner(
        [],
        error=RunnerError(
            "Model 'gpt-5.4' is not available for this Factory account",
            status_code=404,
            error_type="model_not_found",
        ),
    )
    app = _app(tmp_path, runner)

    async with _client(app) as client:
        payload = _payload(model="gpt-5.4", stream=True)
        stream = await client.post("/v1/chat/completions", json=payload)
        blocked = await client.post("/v1/chat/completions", json=payload)

    assert "model_not_found" in stream.text
    assert blocked.status_code == 404
    assert len(runner.requests) == 1


@pytest.mark.asyncio
async def test_chat_keeps_serving_models_droid_did_not_refuse(tmp_path: Path) -> None:
    runner = FakeRunner(
        [],
        error=RunnerError("Factory Droid SDK failed: boom", error_type="factory_droid_sdk_error"),
    )
    app = _app(tmp_path, runner)

    async with _client(app) as client:
        first = await client.post("/v1/chat/completions", json=_payload(model="gpt-5.4"))
        second = await client.post("/v1/chat/completions", json=_payload(model="gpt-5.4"))
        models = await client.get("/v1/models")

    assert first.status_code == 502
    assert second.status_code == 502
    assert len(runner.requests) == 2
    assert [model["id"] for model in models.json()["data"]] == ["factory-droid", "gpt-5.4"]
    assert "x-factory-droid-models-quarantined" not in models.headers


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
    app.state.sessions.remember("session-9", SessionKey(None, None))
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
    app.state.sessions.remember("session-9", SessionKey(None, None))
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
async def test_non_streaming_tool_call_does_not_need_turn_complete(tmp_path: Path) -> None:
    runner = FakeRunner(
        [
            TextDelta(
                f'{TOOL_CALL_OPEN}{{"name":"weather","arguments":{{"city":"Hel"}}}}'
                f"{TOOL_CALL_CLOSE}"
            ),
        ]
    )
    payload = _payload(
        tools=[
            {
                "type": "function",
                "function": {"name": "weather", "parameters": {}},
            }
        ],
        tool_choice="required",
    )

    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == "tool_calls"
    assert runner.closed is True


@pytest.mark.asyncio
async def test_streaming_recovers_tool_call_without_close_marker(tmp_path: Path) -> None:
    runner = FakeRunner(
        [
            TextDelta(f'{TOOL_CALL_OPEN}{{"name":"weather","arguments":{{"city":"Hel"}}}}'),
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

    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ") and not line.endswith("[DONE]")
    ]
    tool_calls = [
        call for event in events for call in event["choices"][0]["delta"].get("tool_calls", [])
    ]
    assert len(tool_calls) == 1
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"city": "Hel"}
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("close_prefix_length", range(len(TOOL_CALL_CLOSE)))
async def test_every_partial_close_marker_recovers_tool_call(
    tmp_path: Path,
    stream: bool,
    close_prefix_length: int,
) -> None:
    runner = FakeRunner(
        [
            ReasoningDelta("planning"),
            TextDelta(
                f'{TOOL_CALL_OPEN}{{"name":"weather","arguments":{{"city":"Hel"}}}}'
                f"{TOOL_CALL_CLOSE[:close_prefix_length]}"
            ),
            RunComplete(Usage()),
        ]
    )
    payload = _payload(
        stream=stream,
        tools=[
            {
                "type": "function",
                "function": {"name": "weather", "parameters": {}},
            }
        ],
    )

    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    if stream:
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ") and not line.endswith("[DONE]")
        ]
        choice = events[-1]["choices"][0]
        tool_calls = [
            call for event in events for call in event["choices"][0]["delta"].get("tool_calls", [])
        ]
        reasoning = "".join(
            event["choices"][0]["delta"].get("reasoning", "")
            for event in events
            if event["choices"]
        )
    else:
        choice = response.json()["choices"][0]
        tool_calls = choice["message"]["tool_calls"]
        reasoning = choice["message"]["reasoning"]
    assert choice["finish_reason"] == "tool_calls"
    assert len(tool_calls) == 1
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"city": "Hel"}
    assert reasoning == "planning"


@pytest.mark.asyncio
async def test_streaming_tool_call_does_not_need_turn_complete(tmp_path: Path) -> None:
    runner = FakeRunner(
        [
            TextDelta(
                f'{TOOL_CALL_OPEN}{{"name":"weather","arguments":{{"city":"Hel"}}}}'
                f"{TOOL_CALL_CLOSE}"
            ),
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
        tool_choice="required",
    )

    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ") and not line.endswith("[DONE]")
    ]
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert response.text.endswith("data: [DONE]\n\n")
    assert runner.closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_tool_call_timeout_without_turn_complete_returns_completion(
    tmp_path: Path,
    stream: bool,
) -> None:
    runner = ToolCallBlockingRunner([])
    payload = _payload(
        stream=stream,
        tools=[
            {
                "type": "function",
                "function": {"name": "weather", "parameters": {}},
            }
        ],
        tool_choice="required",
    )

    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    if stream:
        assert response.text.endswith("data: [DONE]\n\n")
    else:
        assert response.json()["choices"][0]["finish_reason"] == "tool_calls"
    assert runner.closed is True


@pytest.mark.asyncio
async def test_tool_call_drain_keeps_usage_reported_by_the_sdk(tmp_path: Path) -> None:
    runner = ToolCallBlockingRunner([UsageUpdate(Usage(11, 4, 1, 0))])
    payload = _payload(
        tools=[
            {
                "type": "function",
                "function": {"name": "weather", "parameters": {}},
            }
        ],
        tool_choice="required",
    )

    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert response.json()["usage"]["total_tokens"] == 15


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_tool_call_drain_keeps_delayed_parallel_calls(
    tmp_path: Path,
    stream: bool,
) -> None:
    class DelayedParallelRunner(FakeRunner):
        async def run(self, request: RunRequest) -> AsyncIterator[RunEvent]:
            self.requests.append(request)
            try:
                for city in ("Gdansk", "Sopot"):
                    call = json.dumps(
                        {"name": "weather", "arguments": {"city": city}},
                        separators=(",", ":"),
                    )
                    yield TextDelta(f"{TOOL_CALL_OPEN}{call}{TOOL_CALL_CLOSE}")
                    await asyncio.sleep(0.01)
                await asyncio.Event().wait()
            finally:
                self.closed = True

    runner = DelayedParallelRunner([])
    app = _feature_app(tmp_path, runner, tool_call_drain_seconds=0.05)
    payload = _payload(
        stream=stream,
        tools=[
            {
                "type": "function",
                "function": {"name": "weather", "parameters": {}},
            }
        ],
    )

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    if stream:
        calls = [
            call
            for line in response.text.splitlines()
            if line.startswith("data: {")
            for choice in json.loads(line.removeprefix("data: "))["choices"]
            for call in choice["delta"].get("tool_calls", [])
        ]
    else:
        calls = response.json()["choices"][0]["message"]["tool_calls"]
    assert [json.loads(call["function"]["arguments"])["city"] for call in calls] == [
        "Gdansk",
        "Sopot",
    ]
    assert runner.closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_backend_timeout_after_tool_call_returns_the_call(
    tmp_path: Path,
    stream: bool,
) -> None:
    # Cancelling the drained event races the runner deadline, which reports
    # the cancellation as a backend timeout even though the call is complete.
    runner = ToolCallFailingRunner(
        RunnerError(
            "Factory Droid timed out after 30.0 seconds.",
            status_code=504,
            error_type="factory_droid_timeout",
        )
    )
    payload = _payload(
        stream=stream,
        tools=[
            {
                "type": "function",
                "function": {"name": "weather", "parameters": {}},
            }
        ],
        tool_choice="required",
    )

    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    if stream:
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ") and not line.endswith("[DONE]")
        ]
        assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"
    else:
        assert response.json()["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_other_failures_after_a_tool_call_still_surface(tmp_path: Path) -> None:
    runner = ToolCallFailingRunner(
        RunnerError("Factory Droid SDK failed: boom", error_type="factory_droid_sdk_error")
    )
    payload = _payload(
        tools=[
            {
                "type": "function",
                "function": {"name": "weather", "parameters": {}},
            }
        ],
        tool_choice="required",
    )

    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "factory_droid_sdk_error"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_truncated_payload_after_a_tool_call_keeps_tool_calls_finish(
    tmp_path: Path,
    stream: bool,
) -> None:
    # "length" would make the client ask the model to continue instead of
    # running the call it already received.
    runner = FakeRunner(
        [
            TextDelta(
                f'{TOOL_CALL_OPEN}{{"name":"weather","arguments":{{"city":"Hel"}}}}'
                f"{TOOL_CALL_CLOSE}"
            ),
            TextDelta(f'{TOOL_CALL_OPEN}{{"name":"weather","arguments":{{"city":"Kr'),
            RunComplete(Usage()),
        ]
    )
    payload = _payload(
        stream=stream,
        tools=[
            {
                "type": "function",
                "function": {"name": "weather", "parameters": {}},
            }
        ],
    )

    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    if stream:
        chunks = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ") and not line.endswith("[DONE]")
        ]
        finish_reasons = [
            choice["finish_reason"]
            for chunk in chunks
            for choice in chunk["choices"]
            if choice["finish_reason"] is not None
        ]
        assert finish_reasons == ["tool_calls"]
    else:
        choice = response.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert len(choice["message"]["tool_calls"]) == 1


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
        retrieve_missing = await client.get("/v1/models/factory-droid")
        retrieve_invalid = await client.get(
            "/v1/models/factory-droid",
            headers={"Authorization": "Bearer wrong"},
        )
        retrieve_valid = await client.get(
            "/v1/models/factory-droid",
            headers={"Authorization": "Bearer secret"},
        )
        version = await client.get("/version")

    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert valid.status_code == 200
    assert retrieve_missing.status_code == 401
    assert retrieve_invalid.status_code == 401
    assert retrieve_valid.status_code == 200
    assert version.status_code == 200


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
            TextDelta(
                f'{TOOL_CALL_OPEN}{{"name":"weather","arguments":{{}}}}{TOOL_CALL_CLOSE}trailing'
            ),
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
async def test_configured_reasoning_effort_overrides_request(tmp_path: Path) -> None:
    runner = FakeRunner([RunComplete(Usage())])
    settings = Settings(
        droid_path="droid",
        workdir=tmp_path,
        timeout_seconds=30.0,
        reasoning_effort="low",
    )
    app = create_app(settings, runner_factory=cast("RunnerFactory", lambda: runner))
    payload = _payload(reasoning_effort="high", factory_droid_reasoning_effort="medium")
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert runner.requests[0].reasoning_effort == "low"


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
            session_continuity=True,
        ),
        runner_factory=cast("RunnerFactory", lambda: runner),
    )
    app.state.sessions.remember("session-rejected", SessionKey(None, None))
    async with _client(app) as client:
        active = asyncio.create_task(client.post("/v1/chat/completions", json=_payload()))
        await runner.started.wait()
        queued = asyncio.create_task(
            client.post("/v1/chat/completions", json=_payload(stream=True))
        )
        await _wait_for_metric(app, "factory_droid_openai_queued_requests 1")

        rejected = await client.post(
            "/v1/chat/completions",
            json=_payload(
                stream=True,
                factory_droid_session_id="session-rejected",
            ),
        )

        assert rejected.status_code == 429
        assert rejected.headers["retry-after"] == "2"
        assert rejected.json()["error"]["type"] == "rate_limit_error"
        runner.release.set()
        assert (await active).status_code == 200
        assert (await queued).status_code == 200
        retried = await client.post(
            "/v1/chat/completions",
            json=_payload(factory_droid_session_id="session-rejected"),
        )

    assert retried.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("use_session", [False, True])
async def test_queue_wait_consumes_end_to_end_timeout(
    tmp_path: Path,
    use_session: bool,
) -> None:
    runner = GateRunner()
    app = create_app(
        Settings(
            workdir=tmp_path,
            timeout_seconds=30,
            max_concurrency=1,
            max_queue_size=1,
            session_continuity=True,
        ),
        runner_factory=cast("RunnerFactory", lambda: runner),
    )
    timed_payload = _payload(timeout=0.02)
    if use_session:
        app.state.sessions.remember("session-timeout", SessionKey(None, None))
        timed_payload = _payload(
            timeout=0.02,
            factory_droid_session_id="session-timeout",
        )
    async with _client(app) as client:
        active = asyncio.create_task(client.post("/v1/chat/completions", json=_payload()))
        await runner.started.wait()

        timed_out = await client.post(
            "/v1/chat/completions",
            json=timed_payload,
        )

        assert timed_out.status_code == 504
        assert timed_out.json()["error"]["type"] == "factory_droid_timeout"
        assert len(runner.requests) == 1
        runner.release.set()
        assert (await active).status_code == 200
        retried = await client.post(
            "/v1/chat/completions",
            json=timed_payload,
        )

    assert retried.status_code == 200


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
async def test_duplicate_request_keys_are_rejected(tmp_path: Path) -> None:
    runner = FakeRunner([RunComplete(Usage())])
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=(
                b'{"model":"factory-droid","model":"gpt-5.4",'
                b'"messages":[{"role":"user","content":"hi"}]}'
            ),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "duplicate key" in response.json()["error"]["message"]
    assert not runner.requests


@pytest.mark.asyncio
async def test_invalid_tool_name_is_rejected_before_runner(tmp_path: Path) -> None:
    runner = FakeRunner([RunComplete(Usage())])
    payload = _payload(
        tools=[
            {
                "type": "function",
                "function": {"name": "bad.name", "parameters": {}},
            }
        ]
    )

    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert runner.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
async def test_token_limit_fields_are_accepted_and_ignored(
    tmp_path: Path,
    field: str,
) -> None:
    # Copilot, litellm and LangChain send these by default; rejecting them
    # would break every one of those clients.
    runner = FakeRunner([TextDelta("Hi"), RunComplete(Usage())])
    payload = _payload(**{field: 5})

    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == "stop"
    assert len(runner.requests) == 1


@pytest.mark.asyncio
async def test_duplicate_nested_keys_are_rejected(tmp_path: Path) -> None:
    runner = FakeRunner([RunComplete(Usage())])
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=(
                b'{"model":"factory-droid",'
                b'"messages":[{"role":"user","content":"hi","content":"dup"}]}'
            ),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 400
    assert "duplicate key" in response.json()["error"]["message"]
    assert not runner.requests


@pytest.mark.asyncio
async def test_unique_keys_still_pass_the_size_middleware(tmp_path: Path) -> None:
    runner = FakeRunner([RunComplete(Usage())])
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=(b'{"model":"factory-droid","messages":[{"role":"user","content":"hi"}]}'),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 200
    assert runner.requests


@pytest.mark.asyncio
async def test_malformed_json_falls_through_to_fastapi(tmp_path: Path) -> None:
    # The duplicate-key check must not swallow a body that is not JSON at all;
    # FastAPI still owns the malformed-JSON error.
    runner = FakeRunner([RunComplete(Usage())])
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=b"{not json}",
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
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
            TextDelta(
                f'{TOOL_CALL_OPEN}{{"name":"weather","arguments":{{}}}}{TOOL_CALL_CLOSE}trailing'
            ),
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
@pytest.mark.parametrize("stream", [False, True])
async def test_model_tool_json_over_the_depth_limit_never_returns_500(
    tmp_path: Path,
    stream: bool,
) -> None:
    nested = "[" * 8 + "0" + "]" * 8
    runner = FakeRunner(
        [
            TextDelta(
                f'{TOOL_CALL_OPEN}{{"name":"weather","arguments":{{"value":{nested}}}}}'
                f"{TOOL_CALL_CLOSE}"
            ),
            RunComplete(Usage()),
        ]
    )
    payload = _payload(
        stream=stream,
        tools=[
            {
                "type": "function",
                "function": {"name": "weather", "parameters": {}},
            }
        ],
    )
    async with _client(_feature_app(tmp_path, runner, max_json_depth=8)) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code != 500
    if not stream:
        assert response.status_code == 502
        assert response.json()["error"]["type"] == "factory_protocol_error"
        return
    events = [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert json.loads(events[-2])["error"]["type"] == "factory_protocol_error"
    assert events[-1] == "[DONE]"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_model_lone_surrogate_never_returns_500(
    tmp_path: Path,
    stream: bool,
) -> None:
    runner = FakeRunner(
        [
            TextDelta("invalid\ud800text"),
            RunComplete(Usage()),
        ]
    )
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_payload(stream=stream),
        )

    assert response.status_code != 500
    if not stream:
        assert response.status_code == 502
        assert response.json()["error"]["type"] == "factory_protocol_error"
        return
    events = [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert json.loads(events[-2])["error"]["type"] == "factory_protocol_error"
    assert events[-1] == "[DONE]"


@pytest.mark.asyncio
async def test_streaming_malformed_tool_call_stops_with_note(tmp_path: Path) -> None:
    runner = FakeRunner(
        [
            TextDelta("checking "),
            TextDelta(f'{TOOL_CALL_OPEN}weather","city":"Krakow"}}{TOOL_CALL_CLOSE}'),
            RunComplete(Usage()),
        ]
    )
    payload = _payload(
        stream=True,
        stream_options={"include_usage": True},
        tools=[
            {
                "type": "function",
                "function": {"name": "weather", "parameters": {}},
            }
        ],
    )
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post("/v1/chat/completions", json=payload)
        metrics = await client.get("/metrics")

    events = [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    chunks = [json.loads(event) for event in events if event.startswith("{")]
    assert not any("error" in chunk for chunk in chunks)
    assert events[-1] == "[DONE]"
    finish_chunk = next(
        chunk
        for chunk in chunks
        if chunk["choices"] and chunk["choices"][0]["finish_reason"] is not None
    )
    assert finish_chunk["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["choices"] == []
    content = "".join(
        choice["delta"].get("content") or ""
        for chunk in chunks
        for choice in chunk.get("choices", [])
    )
    assert content.startswith("checking \n\n[bridge notice: dropped a malformed tool call")
    assert 'for tool "weather"' in content
    assert "Retry the request" in content
    assert "<tool_call>" not in content
    assert 'outcome="malformed",status="200"} 1' in metrics.text


@pytest.mark.asyncio
async def test_streaming_mangled_openai_tool_calls_do_not_leak_json(tmp_path: Path) -> None:
    runner = FakeRunner(
        [
            TextDelta(
                'checking (reset|", "tool calls":"id":"call_1","type":"function",'
                '"function":("name":"weather","arguments":("city":"Gdansk"))'
            ),
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

    chunks = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith('data: {"')
    ]
    content = "".join(
        choice["delta"].get("content") or ""
        for chunk in chunks
        for choice in chunk.get("choices", [])
    )
    assert content.startswith('checking (reset|", \n\n[bridge notice:')
    assert '"tool calls"' not in content
    assert '"function"' not in content
    assert '"arguments"' not in content
    assert "<tool_call>" not in content
    finish_chunk = next(
        chunk
        for chunk in chunks
        if chunk["choices"] and chunk["choices"][0]["finish_reason"] is not None
    )
    assert finish_chunk["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_non_streaming_malformed_tool_call_stops_with_note(tmp_path: Path) -> None:
    runner = FakeRunner(
        [
            TextDelta("partial answer"),
            TextDelta(f'{TOOL_CALL_OPEN}weather","city":"Krakow"}}{TOOL_CALL_CLOSE}'),
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
        metrics = await client.get("/metrics")

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    content = choice["message"]["content"]
    assert content.startswith("partial answer\n\n[bridge notice: dropped a malformed tool call")
    assert 'for tool "weather"' in content
    assert "tool_calls" not in choice["message"]
    assert runner.closed is True
    assert 'outcome="malformed",status="200"} 1' in metrics.text


@pytest.mark.asyncio
async def test_repaired_tool_call_counts_a_bounded_telemetry_feature(tmp_path: Path) -> None:
    runner = FakeRunner(
        [
            TextDelta(f'{TOOL_CALL_OPEN}weather{{"city":"Gdansk"}}{TOOL_CALL_CLOSE}'),
            RunComplete(Usage()),
        ]
    )
    payload = _payload(
        tools=[
            {
                "type": "function",
                "function": {"name": "weather", "parameters": {}},
            }
        ],
    )
    app = _app(tmp_path, runner)
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    features = dict(app.state.metrics.telemetry_snapshot().features)
    assert features["tool_repair:native:bare_call"] == 1
    # The label must carry no tool name, model name or payload text.
    assert not any("weather" in feature or "Gdansk" in feature for feature in features)


@pytest.mark.asyncio
async def test_unparsed_tool_call_counts_a_bounded_telemetry_feature(tmp_path: Path) -> None:
    runner = FakeRunner(
        [
            TextDelta(f"{TOOL_CALL_OPEN}not json{TOOL_CALL_CLOSE}"),
            RunComplete(Usage()),
        ]
    )
    payload = _payload(
        tools=[
            {
                "type": "function",
                "function": {"name": "weather", "parameters": {}},
            }
        ],
    )
    app = _app(tmp_path, runner)
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    features = dict(app.state.metrics.telemetry_snapshot().features)
    assert features["tool_unparsed:native"] == 1


@pytest.mark.asyncio
async def test_non_streaming_turn_over_the_tool_call_limit_stops_with_note(tmp_path: Path) -> None:
    runner = FakeRunner(
        [
            TextDelta(
                f"{TOOL_CALL_OPEN}"
                '{"name":"weather","arguments":{}}{"name":"weather","arguments":{}}'
                f"{TOOL_CALL_CLOSE}"
            ),
            RunComplete(Usage()),
        ]
    )
    payload = _payload(
        parallel_tool_calls=False,
        tools=[
            {
                "type": "function",
                "function": {"name": "weather", "parameters": {}},
            }
        ],
    )
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert 'for tool "weather"' in choice["message"]["content"]
    assert "tool_calls" not in choice["message"]


@pytest.mark.asyncio
async def test_non_streaming_incomplete_message_json_stops_with_note(tmp_path: Path) -> None:
    runner = FakeRunner(
        [
            TextDelta('{"role":"assistant","tool_calls":'),
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

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"].startswith("[bridge notice: dropped a malformed tool call")
    assert "tool_calls" not in choice["message"]


@pytest.mark.asyncio
async def test_streaming_malformed_tool_call_without_prior_text(tmp_path: Path) -> None:
    runner = FakeRunner(
        [
            TextDelta(f"{TOOL_CALL_OPEN}not json{TOOL_CALL_CLOSE}"),
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

    chunks = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith('data: {"')
    ]
    content = "".join(
        choice["delta"].get("content") or ""
        for chunk in chunks
        for choice in chunk.get("choices", [])
    )
    assert content.startswith("[bridge notice: dropped a malformed tool call.")
    assert "for tool" not in content
    finish_chunk = next(
        chunk
        for chunk in chunks
        if chunk["choices"] and chunk["choices"][0]["finish_reason"] is not None
    )
    assert finish_chunk["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_streaming_malformed_tool_call_flushes_held_text(tmp_path: Path) -> None:
    runner = FakeRunner(
        [
            TextDelta("working on EN"),
            TextDelta(f"{TOOL_CALL_OPEN}not json{TOOL_CALL_CLOSE}"),
            RunComplete(Usage()),
        ]
    )
    payload = _payload(
        stream=True,
        stop="END",
        tools=[
            {
                "type": "function",
                "function": {"name": "weather", "parameters": {}},
            }
        ],
    )
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    chunks = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith('data: {"')
    ]
    content = "".join(
        choice["delta"].get("content") or ""
        for chunk in chunks
        for choice in chunk.get("choices", [])
    )
    assert content.startswith("working on EN\n\n[bridge notice:")


@pytest.mark.asyncio
async def test_stream_generator_drops_note_for_structured_output() -> None:
    schema = {"type": "object"}
    validator_class = validator_for(schema)
    structured = StructuredOutput(
        payload={"type": "json_schema", "schema": schema},
        validator=validator_class(schema),
        max_bytes=1024,
    )
    runner = FakeRunner(
        [
            TextDelta(f"{TOOL_CALL_OPEN}not json{TOOL_CALL_CLOSE}"),
            RunComplete(Usage()),
        ]
    )

    events = await _collect_stream(
        runner,
        allowed_tool_names=frozenset({"weather"}),
        structured=structured,
    )

    assert any('"finish_reason":"stop"' in event for event in events)
    assert not any("bridge notice" in event for event in events)


@pytest.mark.asyncio
async def test_stream_generator_drops_held_text_for_structured_output() -> None:
    schema = {"type": "object"}
    validator_class = validator_for(schema)
    structured = StructuredOutput(
        payload={"type": "json_schema", "schema": schema},
        validator=validator_class(schema),
        max_bytes=1024,
    )
    runner = FakeRunner(
        [
            TextDelta("working on EN"),
            TextDelta(f"{TOOL_CALL_OPEN}not json{TOOL_CALL_CLOSE}"),
            RunComplete(Usage()),
        ]
    )

    events = await _collect_stream(
        runner,
        allowed_tool_names=frozenset({"weather"}),
        stop_sequences=("END",),
        structured=structured,
    )

    assert any('"finish_reason":"stop"' in event for event in events)
    assert not any("working on" in event for event in events)
    assert not any("bridge notice" in event for event in events)


@pytest.mark.asyncio
async def test_streaming_truncated_tool_call_finishes_with_length(tmp_path: Path) -> None:
    runner = FakeRunner(
        [
            TextDelta("working on EN"),
            TextDelta(f'{TOOL_CALL_OPEN}{{"name":"weather","arguments":{{"city":"Kr'),
            RunComplete(Usage()),
        ]
    )
    payload = _payload(
        stream=True,
        stream_options={"include_usage": True},
        stop="END",
        tools=[
            {
                "type": "function",
                "function": {"name": "weather", "parameters": {}},
            }
        ],
    )
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post("/v1/chat/completions", json=payload)
        metrics = await client.get("/metrics")

    events = [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    chunks = [json.loads(event) for event in events if event.startswith("{")]
    assert not any("error" in chunk for chunk in chunks)
    assert events[-1] == "[DONE]"
    finish_chunk = next(
        chunk
        for chunk in chunks
        if chunk["choices"] and chunk["choices"][0]["finish_reason"] is not None
    )
    assert finish_chunk["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["total_tokens"] == 0
    content = "".join(
        choice["delta"].get("content") or ""
        for chunk in chunks
        for choice in chunk.get("choices", [])
    )
    assert content == "working on EN"
    assert 'outcome="truncated",status="200"} 1' in metrics.text


@pytest.mark.asyncio
async def test_non_streaming_truncated_tool_call_returns_length(tmp_path: Path) -> None:
    runner = FakeRunner(
        [
            TextDelta("partial answer"),
            TextDelta(f'{TOOL_CALL_OPEN}{{"name":"weather","arguments":{{"city":"Kr'),
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
        metrics = await client.get("/metrics")

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "length"
    assert choice["message"]["content"] == "partial answer"
    assert "tool_calls" not in choice["message"]
    assert runner.closed is True
    assert 'outcome="truncated",status="200"} 1' in metrics.text


@pytest.mark.asyncio
async def test_lost_prefix_repair_recovers_tool_call_when_enabled(tmp_path: Path) -> None:
    runner = FakeRunner(
        [
            TextDelta(f'{TOOL_CALL_OPEN}weather","city":"Krakow"}}{TOOL_CALL_CLOSE}'),
            RunComplete(Usage()),
        ]
    )
    app = create_app(
        Settings(workdir=tmp_path, repair_lost_prefix=True),
        runner_factory=cast("RunnerFactory", lambda: runner),
    )
    payload = _payload(
        tools=[
            {
                "type": "function",
                "function": {"name": "weather", "parameters": {}},
            }
        ]
    )
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "weather"
    assert json.loads(tool_call["function"]["arguments"]) == {"city": "Krakow"}


@pytest.mark.asyncio
async def test_non_streaming_truncation_keeps_completed_tool_calls(tmp_path: Path) -> None:
    complete = (
        f'{TOOL_CALL_OPEN}{{"name":"weather","arguments":{{"city":"Gdansk"}}}}{TOOL_CALL_CLOSE}'
    )
    truncated = f'{TOOL_CALL_OPEN}{{"name":"weather","arguments":{{"city":"Kr'
    runner = FakeRunner([TextDelta(complete + truncated), RunComplete(Usage())])
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

    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert len(choice["message"]["tool_calls"]) == 1
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "weather"


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"stream_outcome": "truncated"}, "truncated"),
        ({"stream_outcome": "malformed"}, "malformed"),
        ({}, "success"),
    ],
)
def test_request_outcome_handles_truncation(state: dict[str, str], expected: str) -> None:
    assert _request_outcome(200, cast("Any", {"state": state})) == expected


def test_request_telemetry_classifies_routes_and_modes() -> None:
    assert _request_route(cast("Any", {"path": "/health"})) == "health"
    assert _request_route(cast("Any", {"path": "/v1/models/gpt-test"})) == "models"
    assert _request_route(cast("Any", {"path": "/unknown"})) == "other"
    assert (
        _request_mode(cast("Any", {"state": {"telemetry_mode": "unexpected"}})) == "not_applicable"
    )


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.099, "lt_100ms"),
        (0.1, "100ms_500ms"),
        (0.5, "500ms_1s"),
        (1.0, "1s_5s"),
        (5.0, "5s_30s"),
        (30.0, "gt_30s"),
    ],
)
def test_latency_bucket_is_bounded(seconds: float, expected: str) -> None:
    assert _latency_bucket(seconds) == expected


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (0, "empty"),
        (10_239, "lt_10kb"),
        (10_240, "10kb_100kb"),
        (102_400, "100kb_1mb"),
        (1_048_576, "gt_1mb"),
    ],
)
def test_payload_size_bucket_is_bounded(size: int, expected: str) -> None:
    assert _payload_size_bucket(size) == expected


def test_set_request_state_merges_into_existing_state_dict() -> None:
    scope = cast("Any", {"state": {"existing": 1}})

    _set_request_state(scope, added=2)

    assert scope["state"] == {"existing": 1, "added": 2}


def test_request_telemetry_features_use_only_coarse_dimensions() -> None:
    scope = cast(
        "Any",
        {
            "path": "/v1/chat/completions",
            "state": {
                "telemetry_model_family": "gpt",
                "payload_size_bucket": "10kb_100kb",
            },
        },
    )

    features = _request_telemetry_features(
        scope,
        status_code=504,
        seconds=5.0,
        outcome="timeout",
    )

    assert features == (
        "request_latency:chat_completions:5s_30s",
        "request_payload:chat_completions:10kb_100kb",
        "model_family:gpt",
        "request_error:chat_completions:timeout",
    )


@pytest.mark.parametrize(
    ("status_code", "outcome", "expected"),
    [
        (429, "error", "rate_limited"),
        (401, "error", "authentication"),
        (404, "error", "not_found"),
        (200, "success", "other"),
        (499, "cancelled", "cancelled"),
        (200, "timeout", "timeout"),
        (504, "error", "timeout"),
        (200, "malformed", "protocol"),
        (200, "truncated", "protocol"),
        (200, "error", "stream_error"),
    ],
)
def test_telemetry_error_category_covers_status_fallbacks(
    status_code: int,
    outcome: str,
    expected: str,
) -> None:
    assert (
        _telemetry_error_category(
            cast("Any", {"state": {}}),
            status_code=status_code,
            outcome=outcome,
        )
        == expected
    )


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
async def test_stream_generator_handles_truncation_without_held_text() -> None:
    runner = FakeRunner(
        [
            TextDelta(f'{TOOL_CALL_OPEN}{{"name":"weather","arguments":{{'),
            RunComplete(Usage()),
        ]
    )

    events = await _collect_stream(
        runner,
        allowed_tool_names=frozenset({"weather"}),
    )

    assert any('"finish_reason":"length"' in event for event in events)


@pytest.mark.asyncio
async def test_stream_generator_drops_buffered_structured_output_on_truncation() -> None:
    schema = {"type": "object"}
    validator_class = validator_for(schema)
    structured = StructuredOutput(
        payload={"type": "json_schema", "schema": schema},
        validator=validator_class(schema),
        max_bytes=1024,
    )
    runner = FakeRunner(
        [
            TextDelta("partial EN"),
            TextDelta(f'{TOOL_CALL_OPEN}{{"name":"weather","arguments":{{'),
            RunComplete(Usage()),
        ]
    )

    events = await _collect_stream(
        runner,
        allowed_tool_names=frozenset({"weather"}),
        stop_sequences=("END",),
        structured=structured,
    )

    assert any('"finish_reason":"length"' in event for event in events)
    assert not any('"content":"partial EN"' in event for event in events)


@pytest.mark.asyncio
async def test_stream_generator_emits_validated_structured_output() -> None:
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    validator_class = validator_for(schema)
    structured = StructuredOutput(
        payload={"type": "json_schema", "schema": schema},
        validator=validator_class(schema),
        max_bytes=1024,
    )
    events = await _collect_stream(
        FakeRunner([TextDelta('{"ok":true}'), RunComplete(Usage())]),
        structured=structured,
    )

    assert any('"content":"{\\"ok\\":true}"' in event for event in events)


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


@pytest.mark.asyncio
async def test_collect_completion_applies_finish_emissions() -> None:
    class FinishTextParser:
        def feed(self, _chunk: str) -> list[ProtocolEmission]:
            return []

        def finish(self) -> list[ProtocolEmission]:
            return [TextEmission("finished")]

    result = await _collect_completion(
        runner=cast("Any", FakeRunner([RunComplete(Usage())])),
        run_request=RunRequest(
            prompt="prompt",
            model="factory-droid",
            model_alias="factory-droid",
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        parser=cast("Any", FinishTextParser()),
    )

    assert result.text == "finished"


@pytest.mark.asyncio
async def test_wait_for_disconnect_ignores_request_messages() -> None:
    messages = iter(
        [
            {"type": "http.request", "body": b"", "more_body": False},
            {"type": "http.disconnect"},
        ]
    )

    async def receive() -> Any:
        return next(messages)

    await _wait_for_disconnect(cast("Any", SimpleNamespace(receive=receive)))


def test_request_payload_limit_error_preserves_message() -> None:
    error = _RequestPayloadLimitError("too large")

    assert error.message == "too large"
    assert str(error) == "too large"


async def _collect_stream(
    runner: FakeRunner,
    *,
    allowed_tool_names: frozenset[str] = frozenset(),
    include_usage: bool = False,
    stop_sequences: tuple[str, ...] = (),
    structured: StructuredOutput | None = None,
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
            structured=structured,
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
async def test_concurrent_session_uses_return_conflict_and_release_the_lease(
    tmp_path: Path,
) -> None:
    runner = GateRunner()
    app = _feature_app(
        tmp_path,
        runner,
        session_continuity=True,
        max_concurrency=3,
        max_queue_size=0,
    )
    app.state.sessions.remember("session-77", SessionKey(None, None))
    continuation = _payload(factory_droid_session_id="session-77")

    async with _client(app) as client:
        active = asyncio.create_task(client.post("/v1/chat/completions", json=continuation))
        await runner.started.wait()

        duplicate = await client.post("/v1/chat/completions", json=continuation)
        context = await client.get("/v1/factory/sessions/session-77/context")

        runner.release.set()
        completed = await active
        retried = await client.post("/v1/chat/completions", json=continuation)

    assert duplicate.status_code == 409
    assert context.status_code == 409
    assert duplicate.json()["error"]["type"] == "invalid_request_error"
    assert context.json()["error"]["type"] == "invalid_request_error"
    assert completed.status_code == 200
    assert retried.status_code == 200
    assert len(runner.requests) == 2
    assert runner.session_operations == []


@pytest.mark.asyncio
@pytest.mark.parametrize("use_session", [False, True])
async def test_cancelled_admission_wait_releases_the_session_lease(
    tmp_path: Path,
    use_session: bool,
) -> None:
    runner = GateRunner()
    app = _feature_app(
        tmp_path,
        runner,
        session_continuity=True,
        max_concurrency=1,
        max_queue_size=1,
    )
    app.state.sessions.remember("session-active", SessionKey(None, None))
    queued_payload = _payload()
    if use_session:
        app.state.sessions.remember("session-queued", SessionKey(None, None))
        queued_payload = _payload(factory_droid_session_id="session-queued")

    async with _client(app) as client:
        active = asyncio.create_task(
            client.post(
                "/v1/chat/completions",
                json=_payload(factory_droid_session_id="session-active"),
            )
        )
        await runner.started.wait()
        queued = asyncio.create_task(
            client.post(
                "/v1/chat/completions",
                json=queued_payload,
            )
        )
        await _wait_for_metric(app, "factory_droid_openai_queued_requests 1")

        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued

        runner.release.set()
        assert (await active).status_code == 200
        retried = await client.post(
            "/v1/chat/completions",
            json=queued_payload,
        )

    assert retried.status_code == 200


@pytest.mark.asyncio
async def test_session_continuation_rejects_settings_switches(tmp_path: Path) -> None:
    runner = FakeRunner(
        [
            SessionStarted("session-77"),
            TextDelta("first reply"),
            RunComplete(Usage()),
        ]
    )
    app = _feature_app(tmp_path, runner, session_continuity=True)
    async with _client(app) as client:
        first = await client.post(
            "/v1/chat/completions",
            json=_payload(model="model-a"),
        )
        switched = await client.post(
            "/v1/chat/completions",
            json=_payload(
                model="model-b",
                factory_droid_session_id="session-77",
            ),
        )
        effort_switched = await client.post(
            "/v1/chat/completions",
            json=_payload(
                model="model-a",
                reasoning_effort="low",
                factory_droid_session_id="session-77",
            ),
        )

    assert first.status_code == 200
    for rejected in (switched, effort_switched):
        assert rejected.status_code == 400
        assert rejected.json()["error"]["type"] == "invalid_request_error"
        assert "switch model or reasoning effort" in rejected.json()["error"]["message"]
    assert len(runner.requests) == 1


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
    assert app.state.sessions.key("session-5") == SessionKey(None, None)


@pytest.mark.asyncio
async def test_failed_stream_does_not_register_its_started_session(
    tmp_path: Path,
) -> None:
    runner = FakeRunner([SessionStarted("session-failed"), TextDelta("partial")])
    app = _feature_app(tmp_path, runner, session_continuity=True)

    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_payload(stream=True),
        )

    assert '"type":"factory_incomplete_response"' in response.text
    assert app.state.sessions.key("session-failed") is None


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
@pytest.mark.parametrize("stream", [False, True])
async def test_response_format_preserves_message_shaped_json(
    tmp_path: Path,
    stream: bool,
) -> None:
    message = '{"role":"assistant","content":"done"}'
    runner = FakeRunner(
        [
            TextDelta(message[:12]),
            TextDelta(message[12:]),
            RunComplete(Usage()),
        ]
    )
    schema = {
        "type": "object",
        "properties": {
            "role": {"const": "assistant"},
            "content": {"type": "string"},
        },
        "required": ["role", "content"],
        "additionalProperties": False,
    }
    payload = _payload(
        stream=stream,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "message", "schema": schema},
        },
    )

    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    if stream:
        chunks = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith('data: {"')
        ]
        content = "".join(
            choice["delta"].get("content") or ""
            for chunk in chunks
            for choice in chunk.get("choices", [])
        )
    else:
        content = response.json()["choices"][0]["message"]["content"]
    assert content == message


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
        assert (
            await client.post(
                "/v1/chat/completions",
                json=_payload(model="model-a"),
            )
        ).status_code == 200
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
    expected_key = SessionKey("model-a", None)
    assert app.state.sessions.key("session-compact") == expected_key
    assert app.state.sessions.key("session-fork") == expected_key
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
@pytest.mark.parametrize("stream", [False, True])
async def test_structured_output_over_the_depth_limit_fails_closed(
    tmp_path: Path,
    stream: bool,
) -> None:
    nested: object = 0
    for _ in range(8):
        nested = {"value": nested}
    runner = FakeRunner(
        [
            TextDelta(json.dumps({"result": nested})),
            RunComplete(Usage()),
        ]
    )
    app = _feature_app(tmp_path, runner, max_json_depth=8)
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_payload(
                stream=stream,
                response_format={"type": "json_object"},
            ),
        )

    if not stream:
        assert response.status_code == 502
        assert "maximum JSON depth" in response.json()["error"]["message"]
        return
    chunks = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    assert chunks[-1]["error"]["type"] == "factory_protocol_error"
    assert "maximum JSON depth" in chunks[-1]["error"]["message"]
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
    app.state.sessions.remember("session-9", SessionKey(None, None))
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

    key_a = SessionKey("model-a", None)
    key_b = SessionKey("model-b", None)
    registry.remember("a", key_a)
    registry.remember("b", key_b)
    assert registry.key("a") == key_a

    key_c = SessionKey("model-c", None)
    registry.remember("c", key_c)

    assert registry.key("b") is None
    assert registry.key("a") == key_a
    assert registry.key("c") == key_c


def test_session_registry_ignores_duplicate_entries() -> None:
    registry = SessionRegistry(2)

    key_a = SessionKey("model-a", None)
    key_b = SessionKey("model-b", None)
    registry.remember("a", key_a)
    registry.remember("a", key_b)
    registry.remember("b", key_b)

    assert registry.key("a") == key_b
    assert registry.key("b") == key_b
    assert registry.key("missing") is None


def test_session_registry_does_not_evict_a_session_in_use() -> None:
    registry = SessionRegistry(2)
    key_a = SessionKey("model-a", None)
    key_b = SessionKey("model-b", None)
    key_c = SessionKey("model-c", None)
    registry.remember("a", key_a)
    registry.remember("b", key_b)
    lease = registry.acquire("a")
    assert lease is not None

    registry.remember("c", key_c)

    assert registry.key("a") == key_a
    assert registry.key("b") is None
    assert registry.key("c") == key_c
    lease.release()


def test_session_registry_defers_eviction_until_a_busy_session_releases() -> None:
    registry = SessionRegistry(2)
    keys = {session_id: SessionKey(f"model-{session_id}", None) for session_id in ("a", "b", "c")}
    registry.remember("a", keys["a"])
    registry.remember("b", keys["b"])
    first = registry.acquire("a")
    second = registry.acquire("b")
    assert first is not None
    assert second is not None

    registry.remember("c", keys["c"])
    assert registry.key("a") == keys["a"]
    assert registry.key("b") == keys["b"]
    assert registry.key("c") == keys["c"]

    first.release()
    assert registry.key("a") is None
    assert registry.key("b") == keys["b"]
    assert registry.key("c") == keys["c"]
    second.release()


@pytest.mark.asyncio
async def test_session_registry_lease_is_exclusive_and_idempotent() -> None:
    registry = SessionRegistry(2)
    registry.remember("a", SessionKey("model-a", None))

    first = registry.acquire("a")
    assert first is not None
    assert registry.acquire("a") is None

    first.release()
    first.release()
    second = registry.acquire("a")
    assert second is not None
    async with second:
        assert registry.acquire("a") is None

    assert registry.acquire("a") is not None


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
async def test_warm_session_key_follows_the_configured_reasoning_effort(tmp_path: Path) -> None:
    runner = FakeRunner([TextDelta("hi"), RunComplete(Usage())])
    settings = Settings(
        droid_path="droid",
        workdir=tmp_path,
        timeout_seconds=30.0,
        reasoning_effort="low",
    )
    app = create_app(settings, runner_factory=cast("RunnerFactory", lambda: runner))
    key = SessionKey(model_id=None, reasoning_effort="low")
    app.state.pool.note(key)
    app.state.pool.offer(_warm_session(key))

    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_payload(reasoning_effort="high"),
        )

    assert response.status_code == 200
    assert runner.requests[0].warm_session is not None
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
        telemetry=False,
    )
    app = create_app(settings, runner_factory=cast("RunnerFactory", lambda: runner))

    async with app.router.lifespan_context(app):
        await asyncio.sleep(0.05)
        # One session per concurrency slot plus a spare for the refill window.
        assert runner.warmed == [SessionKey(model_id=None, reasoning_effort=None)] * 2
        assert "factory_droid_openai_warm_sessions 2" in app.state.metrics.render()

    assert len(runner.discarded) == 2
    assert "factory_droid_openai_warm_sessions 0" in app.state.metrics.render()


@pytest.mark.asyncio
async def test_detached_cleanup_can_be_disabled(tmp_path: Path) -> None:
    settings = Settings(
        droid_path="droid",
        workdir=tmp_path,
        timeout_seconds=30.0,
        warm_sessions=0,
        detached_cleanup=False,
        telemetry=False,
    )
    app = create_app(settings)

    assert app.state.pool.enabled is False
    async with app.router.lifespan_context(app):
        assert "factory_droid_openai_warm_sessions 0" in app.state.metrics.render()


@pytest.mark.asyncio
async def test_lifespan_flushes_anonymous_telemetry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payloads: list[dict[str, object]] = []

    def post(_endpoint: str, body: bytes, _timeout: float) -> bool:
        payloads.append(json.loads(body))
        return True

    monkeypatch.setattr(telemetry_module, "_post", post)
    runner = FakeRunner([RunComplete(Usage())])
    app = create_app(
        Settings(
            droid_path="droid",
            workdir=tmp_path,
            timeout_seconds=30.0,
            warm_sessions=0,
        ),
        runner_factory=cast("RunnerFactory", lambda: runner),
    )

    async with app.router.lifespan_context(app):
        async with _client(app) as client:
            response = await client.post(
                "/v1/chat/completions",
                json=_payload(
                    reasoning_effort="low",
                    n=2,
                    tools=[
                        {
                            "type": "function",
                            "function": {"name": "weather", "parameters": {}},
                        }
                    ],
                ),
            )
        assert response.status_code == 200

    events = [
        event
        for payload in payloads
        for event in cast("list[dict[str, object]]", payload["events"])
    ]
    assert {"name": "bridge_started", "count": 1} in events
    request_events = [event for event in events if event["name"] == "request"]
    assert len(request_events) == 1
    assert request_events[0] | {"duration_ms_sum": 0} == {
        "name": "request",
        "route": "chat_completions",
        "outcome": "success",
        "mode": "non_stream",
        "count": 1,
        "duration_ms_sum": 0,
    }
    assert isinstance(request_events[0]["duration_ms_sum"], int)
    assert {"name": "feature", "feature": "tools", "count": 1} in events
    assert {"name": "feature", "feature": "reasoning_effort", "count": 1} in events
    assert {"name": "feature", "feature": "multiple_choices", "count": 1} in events
    assert {
        "name": "feature",
        "feature": "model_family:factory_default",
        "count": 1,
    } in events
    assert any(
        event["name"] == "feature"
        and str(event["feature"]).startswith("request_latency:chat_completions:")
        for event in events
    )
    assert any(
        event["name"] == "feature"
        and str(event["feature"]).startswith("request_payload:chat_completions:")
        for event in events
    )
    assert "Hello" not in json.dumps(payloads)
    assert "factory-droid" not in json.dumps(payloads)


@pytest.mark.asyncio
async def test_admission_release_joins_an_in_flight_release() -> None:
    admission = BlockingAdmission()
    lease = AdmissionLease(cast("Any", admission))

    first = asyncio.create_task(lease.release())
    await admission.started.wait()
    second = asyncio.create_task(lease.release())
    await asyncio.sleep(0)
    admission.proceed.set()
    await asyncio.gather(first, second)

    assert admission.releases == 1


@pytest.mark.asyncio
async def test_admission_refuses_an_already_expired_deadline() -> None:
    admission = AdmissionController(
        max_concurrency=1,
        max_queue_size=1,
        metrics=BridgeMetrics(),
    )

    with pytest.raises(TimeoutError):
        await admission.acquire(asyncio.get_running_loop().time() - 1)


def test_json_depth_tracker_frees_depth_when_containers_close() -> None:
    tracker = _JsonDepthTracker(2)

    tracker.feed(b"[[]][[]]")
    tracker.feed(b"[[")

    with pytest.raises(_RequestPayloadLimitError, match="depth limit"):
        tracker.feed(b"[")


def test_json_depth_tracker_ignores_backslashes_outside_strings() -> None:
    """A stray escape outside a string is not JSON, but it must not shift depth."""
    tracker = _JsonDepthTracker(1)

    tracker.feed(rb"\ [\]")
    tracker.feed(b"[")

    with pytest.raises(_RequestPayloadLimitError, match="depth limit"):
        tracker.feed(b"[")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [_RequestPayloadLimitError("body too deep"), TimeoutError()],
)
async def test_request_limit_middleware_reraises_failures_after_the_response_started(
    failure: Exception,
) -> None:
    """Once headers are on the wire the middleware cannot answer with an error."""

    async def downstream(_scope: Any, _receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise failure

    middleware = RequestSizeLimitMiddleware(
        cast("Any", downstream),
        max_request_bytes=64,
        max_json_depth=4,
        body_timeout_seconds=1.0,
        metrics=BridgeMetrics(),
    )

    async def receive() -> Any:
        return {"type": "http.request", "body": b"{}", "more_body": False}

    sent: list[Any] = []

    async def send(message: Any) -> None:
        sent.append(message)

    with pytest.raises(type(failure)):
        await middleware(
            cast(
                "Any",
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/v1/chat/completions",
                    "headers": [],
                },
            ),
            receive,
            send,
        )

    assert [message["type"] for message in sent] == ["http.response.start"]


@pytest.mark.asyncio
async def test_finalizing_streaming_response_finishes_a_cancelled_finalizer() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    finalized = False

    async def content() -> AsyncIterator[str]:
        yield "data"

    async def finalizer() -> None:
        nonlocal finalized
        started.set()
        await release.wait()
        finalized = True

    response = FinalizingStreamingResponse(
        content(),
        finalizer=finalizer,
        media_type="text/event-stream",
        headers={},
    )

    async def receive() -> Any:
        await asyncio.Event().wait()
        return {"type": "http.disconnect"}

    async def send(_message: Any) -> None:
        return

    task = asyncio.create_task(
        response(
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
    )
    await started.wait()
    task.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert finalized is True


@pytest.mark.asyncio
async def test_model_failures_are_not_quarantined_when_the_ttl_is_zero(tmp_path: Path) -> None:
    runner = FakeRunner(
        [],
        error=RunnerError("Model is gone.", status_code=404, error_type="model_not_found"),
    )
    app = _feature_app(tmp_path, runner, model_quarantine_seconds=0.0)

    async with _client(app) as client:
        first = await client.post("/v1/chat/completions", json=_payload())
        second = await client.post("/v1/chat/completions", json=_payload())

    assert first.status_code == 404
    assert second.status_code == 404
    assert "factory_droid_openai_model_quarantines_total 0" in app.state.metrics.render()


@pytest.mark.asyncio
async def test_runner_factory_failure_releases_the_admission_slot(tmp_path: Path) -> None:
    attempts = 0

    def factory() -> Any:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("no droid runner")

    app = create_app(
        Settings(workdir=tmp_path, max_concurrency=1, max_queue_size=0),
        runner_factory=cast("RunnerFactory", factory),
    )

    async with _client(app) as client:
        for _ in range(2):
            with pytest.raises(RuntimeError, match="no droid runner"):
                await client.post("/v1/chat/completions", json=_payload())

    assert attempts == 2


@pytest.mark.asyncio
async def test_runner_factory_failure_releases_the_session_lease(tmp_path: Path) -> None:
    attempts = 0

    def factory() -> Any:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("no droid runner")

    app = create_app(
        Settings(
            workdir=tmp_path,
            session_continuity=True,
            max_concurrency=1,
            max_queue_size=0,
        ),
        runner_factory=cast("RunnerFactory", factory),
    )
    app.state.sessions.remember("session-1", SessionKey(None, None))
    payload = _payload(factory_droid_session_id="session-1")

    async with _client(app) as client:
        for _ in range(2):
            with pytest.raises(RuntimeError, match="no droid runner"):
                await client.post("/v1/chat/completions", json=payload)

    assert attempts == 2


@pytest.mark.asyncio
async def test_non_streaming_request_answers_499_when_the_client_disconnects(
    tmp_path: Path,
) -> None:
    runner = BlockingRunner()
    app = _app(tmp_path, runner)
    body = json.dumps(_payload()).encode()
    delivered = False

    async def receive() -> Any:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        await runner.started.wait()
        return {"type": "http.disconnect"}

    sent: list[Any] = []

    async def send(message: Any) -> None:
        sent.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"host", b"test"),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
            "client": ("127.0.0.1", 4242),
            "server": ("test", 80),
        },
        receive,
        send,
    )

    assert sent[0]["status"] == 499
    assert runner.closed is True


@pytest.mark.asyncio
async def test_collect_completion_ignores_unmapped_runner_events() -> None:
    runner = FakeRunner(
        [cast("RunEvent", SimpleNamespace(kind="unknown")), RunComplete(Usage())],
    )

    result = await _collect_completion(
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

    assert result.text == ""
    assert result.completed is True


@pytest.mark.asyncio
async def test_streaming_keeps_the_session_id_private_without_a_callback() -> None:
    runner = FakeRunner([SessionStarted("session-1"), TextDelta("hi"), RunComplete(Usage())])

    events = await _collect_stream(runner)

    assert "factory_droid_session_id" not in "".join(events)
    assert '"content":"hi"' in "".join(events)


@pytest.mark.asyncio
async def test_streaming_ignores_unmapped_runner_events() -> None:
    runner = FakeRunner(
        [cast("RunEvent", SimpleNamespace(kind="unknown")), RunComplete(Usage())],
    )

    events = await _collect_stream(runner)

    assert '"finish_reason":"stop"' in "".join(events)


@pytest.mark.asyncio
async def test_streaming_structured_output_absorbs_finish_and_held_text() -> None:
    """Structured output is buffered, so held stop-sequence text is not streamed."""

    class FinishParser:
        def feed(self, _chunk: str) -> list[ProtocolEmission]:
            return []

        def finish(self) -> list[ProtocolEmission]:
            return [TextEmission(""), TextEmission('{"a":1}')]

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "required": ["a"],
    }
    structured = StructuredOutput(
        payload={"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}},
        validator=validator_for(schema)(schema),
        max_bytes=1024,
    )
    metrics = BridgeMetrics()
    admission = AdmissionController(max_concurrency=1, max_queue_size=1, metrics=metrics)
    lease = await admission.acquire(asyncio.get_running_loop().time() + 30)

    events = [
        event
        async for event in _stream_completion(
            request_id="chatcmpl-test",
            created=1,
            model="factory-droid",
            parser=cast("Any", FinishParser()),
            runner=cast("Any", FakeRunner([RunComplete(Usage())])),
            run_request=RunRequest(
                prompt="prompt",
                model="factory-droid",
                model_alias="factory-droid",
                reasoning_effort=None,
                timeout_seconds=30,
            ),
            lease=lease,
            stop_sequences=("}}",),
            structured=structured,
        )
    ]

    body = "".join(events)
    assert body.count('{\\"a\\":1}') == 1
    assert '"finish_reason":"stop"' in body


@pytest.mark.asyncio
async def test_streaming_reports_a_cancelled_outcome() -> None:
    runner = BlockingRunner()
    outcomes: list[str] = []
    metrics = BridgeMetrics()
    admission = AdmissionController(max_concurrency=1, max_queue_size=1, metrics=metrics)
    lease = await admission.acquire(asyncio.get_running_loop().time() + 30)
    stream = _stream_completion(
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
        outcome_callback=outcomes.append,
    )

    first = await anext(stream)
    pending: asyncio.Future[str] = asyncio.ensure_future(anext(stream))
    await runner.started.wait()
    pending.cancel()

    with pytest.raises(asyncio.CancelledError):
        await pending

    assert '"role":"assistant"' in first
    assert outcomes == ["cancelled"]


@pytest.mark.asyncio
async def test_stream_finalizer_accepts_a_stream_without_aclose() -> None:
    metrics = BridgeMetrics()
    admission = AdmissionController(max_concurrency=1, max_queue_size=1, metrics=metrics)
    lease = await admission.acquire(asyncio.get_running_loop().time() + 30)
    registry = SessionRegistry(1)
    registry.remember("session-1", SessionKey(None, None))
    session_use = registry.acquire("session-1")
    assert session_use is not None
    assert registry.acquire("session-1") is None
    request = SimpleNamespace(state=SimpleNamespace(stream_outcome="pending"))

    await _finalize_stream(
        cast("Any", SimpleNamespace(aclose=None)),
        lease,
        cast("Any", request),
        session_use=session_use,
    )

    assert request.state.stream_outcome == "cancelled"
    assert "factory_droid_openai_active_sessions 0" in metrics.render()
    assert registry.acquire("session-1") is not None


def test_request_outcome_falls_back_to_the_status_code() -> None:
    scope = cast("Any", {"state": {"stream_outcome": "pending"}})

    assert _request_outcome(200, scope) == "success"
