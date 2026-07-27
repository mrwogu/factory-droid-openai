from __future__ import annotations

import asyncio
import os
import sys
import textwrap
import time
from types import SimpleNamespace
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
    WorkingStateChanged,
)
from droid_sdk import TimeoutError as DroidTimeoutError
from droid_sdk.errors import SessionNotFoundError
from droid_sdk.schemas.enums import (
    AutonomyLevel,
    DroidInteractionMode,
    DroidWorkingState,
    ReasoningEffort,
)

from factory_droid_openai.metrics import BridgeMetrics
from factory_droid_openai.runner import (
    DroidRunner,
    ReasoningDelta,
    RunComplete,
    RunnerError,
    RunRequest,
    SessionStarted,
    StatusUpdate,
    TextDelta,
    Usage,
    UsageUpdate,
    _build_exec_args,
    _create_client,
    _run_until,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path
    from typing import Any


class FakeClient:
    def __init__(self, events: list[object], *, session_id: str | None = "session-1") -> None:
        self.events = events
        self.connected = False
        self.closed = False
        self.interrupted = False
        self.prompt = ""
        self.session_id = session_id
        self.loaded_session_id: str | None = None
        self.images: Any = None
        self.files: Any = None
        self.init_kwargs: dict[str, Any] = {}
        self.permission_handler: Any = None
        self.ask_user_handler: Any = None
        self.rpc_requests: list[tuple[str, dict[str, Any], float | None]] = []
        self.disabled_tool_ids: set[str] = set()
        self.output_format: dict[str, Any] | None = None
        self._protocol = self

    def set_permission_handler(self, handler: Any) -> None:
        self.permission_handler = handler

    def set_ask_user_handler(self, handler: Any) -> None:
        self.ask_user_handler = handler

    async def connect(self) -> None:
        self.connected = True

    async def initialize_session(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs

    async def load_session(
        self,
        *,
        session_id: str,
        mcp_servers: list[dict[str, Any]] | None = None,
    ) -> None:
        del mcp_servers
        self.loaded_session_id = session_id
        self.session_id = session_id

    async def add_user_message(
        self,
        *,
        text: str,
        images: Any = None,
        files: Any = None,
    ) -> None:
        self.prompt = text
        self.images = images
        self.files = files

    async def send_request(
        self,
        method: str,
        params: dict[str, Any],
        timeout: float | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        del request_id
        self.rpc_requests.append((method, params, timeout))
        if method == "droid.list_mcp_servers":
            return {"result": {"servers": []}}
        if method == "droid.list_tools":
            return {
                "result": {
                    "tools": [
                        {
                            "id": "read-cli",
                            "currentlyAllowed": "read-cli" not in self.disabled_tool_ids,
                        },
                        {
                            "id": "exit-spec-mode",
                            "currentlyAllowed": True,
                        },
                    ]
                }
            }
        if method == "droid.update_session_settings":
            self.disabled_tool_ids = set(params["disabledToolIds"])
            return {"result": {}}
        if method == "droid.add_user_message":
            self.prompt = params["text"]
            self.images = params.get("images")
            self.files = params.get("files")
            self.output_format = params.get("outputFormat")
            return {"result": {}}
        if method in {"droid.close_session", "droid.rename_session"}:
            return {"result": {"success": True}}
        raise AssertionError(f"unexpected RPC method: {method}")

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
        SessionStarted("session-1"),
        ReasoningDelta("think"),
        TextDelta("hello"),
        UsageUpdate(usage=Usage(12, 5, 3, 2)),
        RunComplete(usage=Usage(12, 5, 3, 2)),
    ]
    assert isinstance(events[3], UsageUpdate)
    assert events[3].usage.cache_read_tokens == 3
    assert client.init_kwargs["model_id"] is None
    assert client.init_kwargs["reasoning_effort"] is ReasoningEffort.High
    assert client.init_kwargs["interaction_mode"] is DroidInteractionMode.Auto
    assert client.init_kwargs["autonomy_level"] is AutonomyLevel.Off
    assert client.init_kwargs["mcp_servers"] == []
    assert client.disabled_tool_ids == {"read-cli", "exit-spec-mode"}
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
    assert client.connected is False
    assert client.closed is False


@pytest.mark.asyncio
async def test_runner_rejects_expired_deadline_before_client_factory(
    tmp_path: Path,
) -> None:
    factory_calls = 0

    def factory(_path: str, _workdir: Path) -> FakeClient:
        nonlocal factory_calls
        factory_calls += 1
        return FakeClient([])

    runner = DroidRunner(
        droid_path="droid",
        workdir=tmp_path,
        client_factory=cast("Any", factory),
    )

    with pytest.raises(RunnerError, match="timed out") as error:
        await _collect(
            runner,
            _request(deadline=asyncio.get_running_loop().time() - 1),
        )

    assert error.value.status_code == 504
    assert factory_calls == 0


class HangingCleanupClient(BlockingClient):
    async def interrupt_session(self) -> None:
        await asyncio.Event().wait()

    async def close(self) -> None:
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_runner_cleanup_uses_one_bounded_deadline(tmp_path: Path) -> None:
    client = HangingCleanupClient()
    runner = DroidRunner(
        droid_path="droid",
        workdir=tmp_path,
        client_factory=cast("Any", lambda _path, _cwd: client),
        cleanup_timeout_seconds=0.12,
    )
    task = asyncio.create_task(_collect(runner, _request(timeout_seconds=10)))
    await client.started.wait()
    started = time.perf_counter()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert time.perf_counter() - started < 0.5


def _sigterm_ignoring_droid(tmp_path: Path) -> Path:
    executable = tmp_path / "fake-droid"
    pid_file = tmp_path / "child.pid"
    executable.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import os
            import signal
            import time
            from pathlib import Path

            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            Path({str(pid_file)!r}).write_text(str(os.getpid()), encoding="utf-8")
            while True:
                time.sleep(1)
            """
        ),
        encoding="utf-8",
    )
    executable.chmod(0o700)
    return executable


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal behavior")
@pytest.mark.asyncio
async def test_runner_kills_and_reaps_process_ignoring_sigterm(
    tmp_path: Path,
) -> None:
    executable = _sigterm_ignoring_droid(tmp_path)
    pid_file = tmp_path / "child.pid"
    metrics = BridgeMetrics()
    runner = DroidRunner(
        droid_path=str(executable),
        workdir=tmp_path,
        process_grace_seconds=0.05,
        cleanup_timeout_seconds=0.5,
        metrics=metrics,
    )

    task = asyncio.create_task(_collect(runner, _request(timeout_seconds=10)))
    for _ in range(100):
        if pid_file.exists():
            break
        await asyncio.sleep(0.01)
    assert pid_file.exists()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    pid = int(pid_file.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert "factory_droid_openai_forced_kills_total 1" in metrics.render()


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


@pytest.mark.asyncio
async def test_runner_loads_existing_session_instead_of_initializing(
    tmp_path: Path,
) -> None:
    usage = TokenUsageUpdate(
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )
    client = FakeClient([TurnComplete(usage)])
    runner = DroidRunner(
        droid_path="droid",
        workdir=tmp_path,
        client_factory=cast("Any", lambda _path, _cwd: client),
    )

    events = await _collect(runner, _request(session_id="session-42"))

    assert client.loaded_session_id == "session-42"
    assert client.init_kwargs == {}
    assert events[0] == SessionStarted("session-42")


@pytest.mark.asyncio
async def test_runner_reports_unknown_session_as_not_found(tmp_path: Path) -> None:
    class MissingSessionClient(FakeClient):
        async def load_session(
            self,
            *,
            session_id: str,
            mcp_servers: list[dict[str, Any]] | None = None,
        ) -> None:
            del mcp_servers
            raise SessionNotFoundError(session_id)

    client = MissingSessionClient([])
    runner = DroidRunner(
        droid_path="droid",
        workdir=tmp_path,
        client_factory=cast("Any", lambda _path, _cwd: client),
    )

    with pytest.raises(RunnerError) as error:
        await _collect(runner, _request(session_id="ghost"))

    assert error.value.status_code == 404
    assert error.value.error_type == "session_not_found"


@pytest.mark.asyncio
async def test_runner_forwards_attachments_to_the_sdk(tmp_path: Path) -> None:
    usage = TokenUsageUpdate(
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )
    client = FakeClient([TurnComplete(usage)])
    runner = DroidRunner(
        droid_path="droid",
        workdir=tmp_path,
        client_factory=cast("Any", lambda _path, _cwd: client),
    )
    image = {"type": "base64", "mediaType": "image/png", "data": "QUJD"}
    document = {"type": "text", "mediaType": "text/plain", "data": "hi"}

    await _collect(runner, _request(images=(image,), documents=(document,)))

    assert client.images == [image]
    assert client.files == [document]


@pytest.mark.asyncio
async def test_runner_forwards_structured_output_through_raw_rpc(
    tmp_path: Path,
) -> None:
    client = FakeClient([AssistantTextDelta('{"answer":7}'), TurnComplete()])
    runner = DroidRunner(
        droid_path="droid",
        workdir=tmp_path,
        client_factory=cast("Any", lambda _path, _cwd: client),
    )
    output_format = {
        "type": "json_schema",
        "schema": {"type": "object", "properties": {"answer": {"type": "integer"}}},
    }

    events = await _collect(runner, _request(output_format=output_format))

    assert TextDelta('{"answer":7}') in events
    assert client.output_format == output_format
    assert client.prompt == "prompt"


@pytest.mark.asyncio
async def test_runner_fails_before_prompt_if_native_tools_remain_enabled(
    tmp_path: Path,
) -> None:
    class UnsafeClient(FakeClient):
        async def send_request(
            self,
            method: str,
            params: dict[str, Any],
            timeout: float | None = None,
            request_id: str | None = None,
        ) -> dict[str, Any]:
            if method == "droid.list_tools":
                return {
                    "result": {
                        "tools": [
                            {"id": "execute-cli", "currentlyAllowed": True},
                        ]
                    }
                }
            return await super().send_request(method, params, timeout, request_id)

    client = UnsafeClient([TurnComplete()])
    runner = DroidRunner(
        droid_path="droid",
        workdir=tmp_path,
        client_factory=cast("Any", lambda _path, _cwd: client),
    )

    with pytest.raises(RunnerError, match="Failed to disable native Droid tools"):
        await _collect(runner, _request())

    assert client.prompt == ""
    assert client.interrupted is True


@pytest.mark.asyncio
async def test_runner_omits_empty_attachment_lists(tmp_path: Path) -> None:
    usage = TokenUsageUpdate(
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )
    client = FakeClient([TurnComplete(usage)])
    runner = DroidRunner(
        droid_path="droid",
        workdir=tmp_path,
        client_factory=cast("Any", lambda _path, _cwd: client),
    )

    await _collect(runner, _request())

    assert client.images is None
    assert client.files is None


@pytest.mark.asyncio
async def test_runner_maps_working_state_changes_to_status_events(
    tmp_path: Path,
) -> None:
    usage = TokenUsageUpdate(
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )
    client = FakeClient(
        [
            WorkingStateChanged(state=DroidWorkingState.ExecutingTool),
            TurnComplete(usage),
        ]
    )
    runner = DroidRunner(
        droid_path="droid",
        workdir=tmp_path,
        client_factory=cast("Any", lambda _path, _cwd: client),
    )

    events = await _collect(runner, _request())

    assert StatusUpdate("executing_tool") in events


def test_exec_args_stay_default_without_cli_only_options() -> None:
    assert _build_exec_args(worktree=None, append_system_prompt_file=None) is None


def test_exec_args_append_cli_only_flags_to_the_jsonrpc_defaults(tmp_path: Path) -> None:
    prompt_file = tmp_path / "system.md"

    args = _build_exec_args(worktree="wt", append_system_prompt_file=prompt_file)

    assert args is not None
    assert args[:5] == [
        "exec",
        "--input-format",
        "stream-jsonrpc",
        "--output-format",
        "stream-jsonrpc",
    ]
    assert args[5:] == [
        "--worktree",
        "wt",
        "--append-system-prompt-file",
        str(prompt_file),
    ]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal behavior")
@pytest.mark.asyncio
async def test_cleanup_force_reaps_when_client_close_never_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback that exists for a hung close() must actually reap.

    The SDK's own close escalates to SIGKILL by itself, so a plain
    SIGTERM-ignoring process never reaches this path. Hanging close() is the
    only way to reach the fallback - and it also exercises the abandoned-task
    handoff, where the cancelled close and force_kill_and_reap act on the same
    process.
    """
    executable = _sigterm_ignoring_droid(tmp_path)
    pid_file = tmp_path / "child.pid"

    async def never_returns(_self: DroidClient) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(DroidClient, "close", never_returns)
    metrics = BridgeMetrics()
    runner = DroidRunner(
        droid_path=str(executable),
        workdir=tmp_path,
        process_grace_seconds=0.05,
        cleanup_timeout_seconds=0.5,
        metrics=metrics,
    )

    task = asyncio.create_task(_collect(runner, _request(timeout_seconds=10)))
    for _ in range(100):
        if pid_file.exists():
            break
        await asyncio.sleep(0.01)
    assert pid_file.exists()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    pid = int(pid_file.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert "factory_droid_openai_forced_kills_total 1" in metrics.render()


@pytest.mark.asyncio
async def test_run_until_reports_failure_for_expired_and_raising_operations() -> None:
    loop = asyncio.get_running_loop()

    async def unreached() -> None:  # pragma: no cover - must never be awaited
        raise AssertionError("operation ran despite an expired deadline")

    async def boom() -> None:
        raise RuntimeError("close failed")

    assert await _run_until(unreached, loop.time() - 1) is False
    assert await _run_until(boom, loop.time() + 5) is False


@pytest.mark.asyncio
async def test_cleanup_completes_when_the_caller_is_cancelled_again(
    tmp_path: Path,
) -> None:
    """A second cancellation must not abandon an in-flight cleanup.

    The first cancel unwinds run() into its finally block; the second lands
    while the shielded cleanup is still running, which is the disconnect
    shape where a client goes away mid-request.
    """
    closing = asyncio.Event()
    release = asyncio.Event()

    class SlowClosingClient(FakeClient):
        async def receive_response(self) -> AsyncIterator[object]:
            await asyncio.Event().wait()
            if False:
                yield None

        async def close(self) -> None:
            closing.set()
            await release.wait()
            self.closed = True

    client = SlowClosingClient([])
    runner = DroidRunner(
        droid_path="droid",
        workdir=tmp_path,
        client_factory=lambda *_: cast("DroidClient", client),
        cleanup_timeout_seconds=5.0,
    )

    task = asyncio.create_task(_collect(runner, _request(timeout_seconds=10)))
    await asyncio.sleep(0)
    task.cancel()
    await closing.wait()
    task.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.closed is True


@pytest.mark.asyncio
async def test_runner_discovers_models_from_session_initialization(
    tmp_path: Path,
) -> None:
    class ModelClient(FakeClient):
        async def initialize_session(self, **kwargs: Any) -> Any:
            await super().initialize_session(**kwargs)
            return SimpleNamespace(
                available_models=[
                    SimpleNamespace(
                        id="gpt-5.4",
                        display_name="GPT-5.4",
                        model_provider=SimpleNamespace(value="openai"),
                        supported_reasoning_efforts=[
                            ReasoningEffort.Low,
                            ReasoningEffort.High,
                        ],
                        default_reasoning_effort=ReasoningEffort.High,
                        no_image_support=False,
                        supports_pdfs=True,
                    )
                ]
            )

    client = ModelClient([])
    runner = DroidRunner(
        droid_path="droid",
        workdir=tmp_path,
        client_factory=cast("Any", lambda _path, _cwd: client),
    )

    models = await runner.list_models(timeout_seconds=1)

    assert models[0].id == "gpt-5.4"
    assert models[0].provider == "openai"
    assert models[0].supported_reasoning_efforts == ("low", "high")
    assert models[0].supports_images is True
    assert models[0].supports_pdfs is True
    assert ("droid.close_session", {"reason": "clear"}, 30.0) in client.rpc_requests


@pytest.mark.asyncio
async def test_runner_exposes_guarded_session_rpc_operations(tmp_path: Path) -> None:
    class SessionOperationClient(FakeClient):
        async def send_request(
            self,
            method: str,
            params: dict[str, Any],
            timeout: float | None = None,
            request_id: str | None = None,
        ) -> dict[str, Any]:
            if method == "droid.get_context_stats":
                return {
                    "result": {
                        "used": 10,
                        "remaining": 90,
                        "limit": 100,
                        "accuracy": "exact",
                        "updatedAt": "now",
                    }
                }
            if method == "droid.get_context_breakdown":
                return {
                    "result": {
                        "modelId": "gpt-5.4",
                        "modelDisplayName": "GPT-5.4",
                        "contextBudget": 100,
                        "usedTokens": 10,
                        "freeTokens": 90,
                        "categories": [{"name": "messages", "tokens": 10, "colorKey": "blue"}],
                    }
                }
            if method == "droid.compact_session":
                return {"result": {"newSessionId": "compact", "removedCount": 3}}
            if method == "droid.fork_session":
                return {"result": {"newSessionId": "fork"}}
            return await super().send_request(method, params, timeout, request_id)

    client = SessionOperationClient([])
    runner = DroidRunner(
        droid_path="droid",
        workdir=tmp_path,
        client_factory=cast("Any", lambda _path, _cwd: client),
    )

    stats, breakdown = await runner.get_context("session", timeout_seconds=1)
    compacted = await runner.compact_session(
        "session",
        custom_instructions="Keep decisions.",
        timeout_seconds=1,
    )
    forked = await runner.fork_session("session", timeout_seconds=1)
    await runner.rename_session(
        "session",
        title="Title",
        timeout_seconds=1,
    )
    await runner.close_session("session", timeout_seconds=1)

    assert stats.used == 10
    assert breakdown.categories[0].name == "messages"
    assert compacted.new_session_id == "compact"
    assert compacted.removed_count == 3
    assert forked == "fork"
    assert client.loaded_session_id == "session"
    assert any(method == "droid.rename_session" for method, _, _ in client.rpc_requests)
    assert any(method == "droid.close_session" for method, _, _ in client.rpc_requests)
    # Metadata operations run no model turn, so they must not touch the tool
    # catalog or rewrite session settings. Compaction does run a turn.
    assert [method for method, _, _ in client.rpc_requests].count(
        "droid.update_session_settings"
    ) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "status_code", "error_type"),
    [
        (SessionNotFoundError("session"), 404, "session_not_found"),
        (FileNotFoundError("droid"), 503, "factory_droid_unavailable"),
        (DroidClientError("broken"), 502, "factory_droid_sdk_error"),
        (DroidTimeoutError("slow"), 504, "factory_droid_timeout"),
    ],
)
async def test_session_operations_map_sdk_failures(
    tmp_path: Path,
    failure: Exception,
    status_code: int,
    error_type: str,
) -> None:
    class FailingClient(FakeClient):
        async def load_session(
            self,
            *,
            session_id: str,
            mcp_servers: list[dict[str, Any]] | None = None,
        ) -> None:
            del session_id, mcp_servers
            raise failure

    client = FailingClient([])
    runner = DroidRunner(
        droid_path="droid",
        workdir=tmp_path,
        client_factory=cast("Any", lambda _path, _cwd: client),
    )

    with pytest.raises(RunnerError) as excinfo:
        await runner.rename_session("session", title="Title", timeout_seconds=1)

    assert excinfo.value.status_code == status_code
    assert excinfo.value.error_type == error_type
    assert client.closed is True


@pytest.mark.asyncio
async def test_session_operations_time_out_on_a_slow_droid(tmp_path: Path) -> None:
    class SlowClient(FakeClient):
        async def load_session(
            self,
            *,
            session_id: str,
            mcp_servers: list[dict[str, Any]] | None = None,
        ) -> None:
            del session_id, mcp_servers
            await asyncio.sleep(1)

    client = SlowClient([])
    runner = DroidRunner(
        droid_path="droid",
        workdir=tmp_path,
        client_factory=cast("Any", lambda _path, _cwd: client),
    )

    with pytest.raises(RunnerError, match="timed out"):
        await runner.close_session("session", timeout_seconds=0.01)

    assert client.closed is True
