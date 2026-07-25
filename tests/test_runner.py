from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

import pytest
from droid_sdk import (
    AssistantTextDelta,
    DroidClient,
    DroidClientError,
    ErrorEvent,
    ThinkingTextDelta,
    TokenUsageUpdate,
    ToolUse,
    TurnComplete,
)
from droid_sdk import TimeoutError as DroidTimeoutError
from droid_sdk.schemas.enums import AutonomyLevel, ReasoningEffort

from factory_droid_openai.runner import (
    DroidRunner,
    ReasoningDelta,
    RunComplete,
    RunnerError,
    RunRequest,
    TextDelta,
    Usage,
    UsageUpdate,
    _create_client,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path
    from typing import Any


class FakeClient:
    def __init__(self, events: list[object]) -> None:
        self.events = events
        self.connected = False
        self.closed = False
        self.interrupted = False
        self.prompt = ""
        self.init_kwargs: dict[str, Any] = {}
        self.permission_handler: Any = None
        self.ask_user_handler: Any = None

    def set_permission_handler(self, handler: Any) -> None:
        self.permission_handler = handler

    def set_ask_user_handler(self, handler: Any) -> None:
        self.ask_user_handler = handler

    async def connect(self) -> None:
        self.connected = True

    async def initialize_session(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs

    async def add_user_message(self, *, text: str) -> None:
        self.prompt = text

    async def receive_response(self) -> AsyncIterator[object]:
        for event in self.events:
            yield event

    async def interrupt_session(self) -> None:
        self.interrupted = True

    async def close(self) -> None:
        self.closed = True


def _request(**overrides: object) -> RunRequest:
    values: dict[str, Any] = {
        "prompt": "prompt",
        "model": "factory-droid",
        "model_alias": "factory-droid",
        "reasoning_effort": "high",
        "timeout_seconds": 1.0,
    }
    values.update(overrides)
    return RunRequest(**values)


@pytest.mark.asyncio
async def test_runner_maps_sdk_stream_and_usage(tmp_path: Path) -> None:
    usage = TokenUsageUpdate(
        input_tokens=12,
        output_tokens=5,
        cache_read_tokens=3,
        cache_write_tokens=2,
    )
    client = FakeClient(
        [
            ThinkingTextDelta("think"),
            AssistantTextDelta("hello"),
            usage,
            TurnComplete(usage),
        ]
    )
    runner = DroidRunner(
        droid_path="droid",
        workdir=tmp_path,
        client_factory=cast("Any", lambda _path, _cwd: client),
    )

    events = [event async for event in runner.run(_request())]

    assert events == [
        ReasoningDelta("think"),
        TextDelta("hello"),
        UsageUpdate(usage=Usage(12, 5, 3, 2)),
        RunComplete(usage=Usage(12, 5, 3, 2)),
    ]
    assert isinstance(events[2], UsageUpdate)
    assert events[2].usage.cache_read_tokens == 3
    assert client.init_kwargs["model_id"] is None
    assert client.init_kwargs["reasoning_effort"] is ReasoningEffort.High
    assert client.init_kwargs["autonomy_level"] is AutonomyLevel.Off
    assert client.permission_handler({}) == "cancel"
    assert client.ask_user_handler({}) == {"cancelled": True, "answers": []}
    assert client.closed is True
    assert client.interrupted is False


@pytest.mark.asyncio
async def test_runner_passes_explicit_model_id(tmp_path: Path) -> None:
    client = FakeClient([TurnComplete()])
    runner = DroidRunner(
        droid_path="droid",
        workdir=tmp_path,
        client_factory=cast("Any", lambda _path, _cwd: client),
    )

    events = [
        event
        async for event in runner.run(_request(model="claude-sonnet-4", reasoning_effort=None))
    ]

    assert isinstance(events[-1], RunComplete)
    assert client.init_kwargs["model_id"] == "claude-sonnet-4"
    assert client.init_kwargs["reasoning_effort"] is None


@pytest.mark.asyncio
async def test_runner_blocks_factory_native_tools(tmp_path: Path) -> None:
    client = FakeClient(
        [
            ToolUse(
                tool_name="terminal",
                tool_input={"command": "pwd"},
                tool_use_id="native-call",
            )
        ]
    )
    runner = DroidRunner(
        droid_path="droid",
        workdir=tmp_path,
        client_factory=cast("Any", lambda _path, _cwd: client),
    )

    with pytest.raises(RunnerError, match="native tool"):
        _ = [event async for event in runner.run(_request())]

    assert client.interrupted is True
    assert client.closed is True


@pytest.mark.asyncio
async def test_runner_maps_sdk_error_event(tmp_path: Path) -> None:
    client = FakeClient([ErrorEvent("service failed", "ServiceError")])
    runner = DroidRunner(
        droid_path="droid",
        workdir=tmp_path,
        client_factory=cast("Any", lambda _path, _cwd: client),
    )

    with pytest.raises(RunnerError, match="service failed") as error:
        _ = [event async for event in runner.run(_request())]

    assert error.value.error_type == "ServiceError"


class BlockingClient(FakeClient):
    def __init__(self) -> None:
        super().__init__([])
        self.started = asyncio.Event()

    async def receive_response(self) -> AsyncIterator[object]:
        self.started.set()
        await asyncio.Event().wait()
        if False:
            yield None


@pytest.mark.asyncio
async def test_runner_interrupts_on_cancellation(tmp_path: Path) -> None:
    client = BlockingClient()
    runner = DroidRunner(
        droid_path="droid",
        workdir=tmp_path,
        client_factory=cast("Any", lambda _path, _cwd: client),
    )

    task = asyncio.create_task(_collect(runner, _request(timeout_seconds=10.0)))
    await client.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert client.interrupted is True
    assert client.closed is True


@pytest.mark.asyncio
async def test_runner_times_out(tmp_path: Path) -> None:
    client = BlockingClient()
    runner = DroidRunner(
        droid_path="droid",
        workdir=tmp_path,
        client_factory=cast("Any", lambda _path, _cwd: client),
    )

    with pytest.raises(RunnerError, match="timed out") as error:
        await _collect(runner, _request(timeout_seconds=0.01))

    assert error.value.status_code == 504
    assert client.interrupted is True
    assert client.closed is True


class SdkTimeoutClient(FakeClient):
    async def connect(self) -> None:
        raise DroidTimeoutError("SDK request timed out")


@pytest.mark.asyncio
async def test_runner_maps_sdk_timeout_to_gateway_timeout(tmp_path: Path) -> None:
    client = SdkTimeoutClient([])
    runner = DroidRunner(
        droid_path="droid",
        workdir=tmp_path,
        client_factory=cast("Any", lambda _path, _cwd: client),
    )

    with pytest.raises(RunnerError) as error:
        await _collect(runner, _request())

    assert error.value.status_code == 504
    assert error.value.error_type == "factory_droid_timeout"
    assert client.closed is True


class MissingExecutableClient(FakeClient):
    async def connect(self) -> None:
        raise FileNotFoundError


@pytest.mark.asyncio
async def test_runner_maps_missing_executable_to_service_unavailable(
    tmp_path: Path,
) -> None:
    client = MissingExecutableClient([])
    runner = DroidRunner(
        droid_path="/missing/droid",
        workdir=tmp_path,
        client_factory=cast("Any", lambda _path, _cwd: client),
    )

    with pytest.raises(RunnerError, match="/missing/droid") as error:
        await _collect(runner, _request())

    assert error.value.status_code == 503
    assert error.value.error_type == "factory_droid_unavailable"


class FailingSdkClient(FakeClient):
    async def connect(self) -> None:
        raise DroidClientError("connection failed")


@pytest.mark.asyncio
async def test_runner_maps_generic_sdk_failure(tmp_path: Path) -> None:
    client = FailingSdkClient([])
    runner = DroidRunner(
        droid_path="droid",
        workdir=tmp_path,
        client_factory=cast("Any", lambda _path, _cwd: client),
    )

    with pytest.raises(RunnerError, match="SDK failed") as error:
        await _collect(runner, _request())

    assert error.value.status_code == 502
    assert error.value.error_type == "factory_droid_sdk_error"


@pytest.mark.asyncio
async def test_runner_rejects_invalid_reasoning_effort(tmp_path: Path) -> None:
    client = FakeClient([])
    runner = DroidRunner(
        droid_path="droid",
        workdir=tmp_path,
        client_factory=cast("Any", lambda _path, _cwd: client),
    )

    with pytest.raises(RunnerError, match="Unsupported reasoning_effort") as error:
        await _collect(runner, _request(reasoning_effort="extreme"))

    assert error.value.status_code == 400
    assert error.value.error_type == "invalid_request_error"
    assert client.closed is True


@pytest.mark.asyncio
async def test_runner_clamps_negative_usage_to_zero(tmp_path: Path) -> None:
    usage = TokenUsageUpdate(
        input_tokens=-1,
        output_tokens=-2,
        cache_read_tokens=-3,
        cache_write_tokens=-4,
    )
    client = FakeClient([usage, TurnComplete(usage)])
    runner = DroidRunner(
        droid_path="droid",
        workdir=tmp_path,
        client_factory=cast("Any", lambda _path, _cwd: client),
    )

    events = await _collect(runner, _request())

    assert events[-1] == RunComplete(Usage())


def test_default_client_factory_configures_droid_process(tmp_path: Path) -> None:
    client = _create_client("/usr/local/bin/droid", tmp_path)

    assert isinstance(client, DroidClient)
    assert client.is_connected is False


async def _collect(runner: DroidRunner, request: RunRequest) -> list[object]:
    return [event async for event in runner.run(request)]
