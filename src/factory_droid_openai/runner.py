from __future__ import annotations

import asyncio
import contextlib
import signal
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from droid_sdk import (
    AssistantTextDelta,
    DroidClient,
    DroidClientError,
    ErrorEvent,
    ProcessTransport,
    ThinkingTextDelta,
    TokenUsageUpdate,
    ToolProgress,
    ToolResult,
    ToolUse,
    TurnComplete,
)
from droid_sdk import TimeoutError as DroidTimeoutError
from droid_sdk.schemas.enums import AutonomyLevel, ReasoningEffort


class RunnerError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 502,
        error_type: str = "factory_droid_error",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type


@dataclass(frozen=True, slots=True)
class RunRequest:
    prompt: str
    model: str
    model_alias: str
    reasoning_effort: str | None
    timeout_seconds: float
    deadline: float | None = None


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    text: str


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass(frozen=True, slots=True)
class UsageUpdate:
    usage: Usage


@dataclass(frozen=True, slots=True)
class RunComplete:
    usage: Usage


RunEvent = TextDelta | ReasoningDelta | UsageUpdate | RunComplete
ClientFactory = Callable[[str, Path], DroidClient]


class RunnerMetrics(Protocol):
    def observe_droid_startup(self, seconds: float) -> None: ...

    def increment_forced_kills(self) -> None: ...


class _ManagedProcessTransport(ProcessTransport):
    def __init__(
        self,
        *,
        exec_path: str,
        cwd: str,
        grace_period: float,
    ) -> None:
        super().__init__(
            exec_path=exec_path,
            cwd=cwd,
            grace_period=grace_period,
        )
        self._owned_process: asyncio.subprocess.Process | None = None
        self._forced_kill = False
        self._grace_period_seconds = grace_period

    async def connect(self) -> None:
        await super().connect()
        self._owned_process = self._process

    async def close(self) -> None:
        process = self._owned_process
        was_running = process is not None and process.returncode is None
        started = asyncio.get_running_loop().time()
        await super().close()
        elapsed = asyncio.get_running_loop().time() - started
        if was_running and process is not None and process.returncode is not None:
            sigkill = getattr(signal, "SIGKILL", None)
            self._forced_kill = (
                sigkill is not None and process.returncode == -sigkill
            ) or elapsed >= self._grace_period_seconds

    async def force_kill_and_reap(self, timeout: float) -> bool:
        process = self._owned_process
        if process is None or process.returncode is not None or timeout <= 0:
            return False
        with contextlib.suppress(ProcessLookupError, OSError):
            process.kill()
        await asyncio.wait_for(process.wait(), timeout=timeout)
        self._forced_kill = True
        return True

    def is_reaped(self) -> bool:
        return self._owned_process is None or self._owned_process.returncode is not None

    def consumed_forced_kill(self) -> bool:
        forced = self._forced_kill
        self._forced_kill = False
        return forced


class DroidRunner:
    def __init__(
        self,
        *,
        droid_path: str,
        workdir: Path,
        client_factory: ClientFactory | None = None,
        process_grace_seconds: float = 1.0,
        cleanup_timeout_seconds: float = 4.0,
        metrics: RunnerMetrics | None = None,
    ) -> None:
        self._droid_path = droid_path
        self._workdir = workdir
        self._client_factory = client_factory
        self._process_grace_seconds = process_grace_seconds
        self._cleanup_timeout_seconds = cleanup_timeout_seconds
        self._metrics = metrics

    async def run(self, request: RunRequest) -> AsyncGenerator[RunEvent, None]:
        reasoning_effort = _resolve_reasoning_effort(request.reasoning_effort)
        loop = asyncio.get_running_loop()
        deadline = request.deadline or loop.time() + request.timeout_seconds
        if deadline <= loop.time():
            raise _timeout_error(request)

        client, transport = self._new_client()
        initialized = False
        completed = False
        usage = Usage()
        client.set_permission_handler(lambda _params: "cancel")
        client.set_ask_user_handler(
            lambda _params: {"cancelled": True, "answers": []},
        )

        try:
            async with asyncio.timeout_at(deadline):
                startup_started = time.perf_counter()
                try:
                    await client.connect()
                finally:
                    if self._metrics is not None:
                        self._metrics.observe_droid_startup(time.perf_counter() - startup_started)
                await client.initialize_session(
                    machine_id="factory-droid-openai",
                    cwd=str(self._workdir),
                    model_id=_resolve_model_id(request.model, request.model_alias),
                    reasoning_effort=reasoning_effort,
                    autonomy_level=AutonomyLevel.Off,
                    skip_permissions_unsafe=False,
                    enabled_tool_ids=[],
                )
                initialized = True
                await client.add_user_message(text=request.prompt)

                async for event in client.receive_response():
                    if isinstance(event, AssistantTextDelta):
                        yield TextDelta(event.text)
                    elif isinstance(event, ThinkingTextDelta):
                        yield ReasoningDelta(event.text)
                    elif isinstance(event, TokenUsageUpdate):
                        usage = _map_usage(event)
                        yield UsageUpdate(usage)
                    elif isinstance(event, TurnComplete):
                        if event.token_usage is not None:
                            usage = _map_usage(event.token_usage)
                        completed = True
                        yield RunComplete(usage)
                    elif isinstance(event, (ToolUse, ToolResult, ToolProgress)):
                        raise RunnerError(
                            "Factory Droid attempted to use a native tool. "
                            "The bridge only permits tools supplied by the OpenAI client.",
                            error_type="factory_native_tool_blocked",
                        )
                    elif isinstance(event, ErrorEvent):
                        raise RunnerError(
                            event.message or "Factory Droid returned an error.",
                            error_type=event.error_type or "factory_droid_error",
                        )
        except (TimeoutError, DroidTimeoutError) as exc:
            raise _timeout_error(request) from exc
        except FileNotFoundError as exc:
            raise RunnerError(
                f"Factory Droid executable was not found: {self._droid_path}",
                status_code=503,
                error_type="factory_droid_unavailable",
            ) from exc
        except DroidClientError as exc:
            raise RunnerError(
                f"Factory Droid SDK failed: {exc}",
                error_type="factory_droid_sdk_error",
            ) from exc
        finally:
            cleanup_task = asyncio.create_task(
                self._cleanup(
                    client,
                    transport,
                    interrupt=initialized and not completed,
                )
            )
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                await cleanup_task
                raise

    def _new_client(self) -> tuple[DroidClient, _ManagedProcessTransport | None]:
        if self._client_factory is not None:
            return self._client_factory(self._droid_path, self._workdir), None
        transport = _ManagedProcessTransport(
            exec_path=self._droid_path,
            cwd=str(self._workdir),
            grace_period=self._process_grace_seconds,
        )
        return DroidClient(transport=transport), transport

    async def _cleanup(
        self,
        client: DroidClient,
        transport: _ManagedProcessTransport | None,
        *,
        interrupt: bool,
    ) -> None:
        loop = asyncio.get_running_loop()
        cleanup_deadline = loop.time() + self._cleanup_timeout_seconds
        force_reap_budget = min(1.0, self._cleanup_timeout_seconds / 3)

        if interrupt:
            interrupt_deadline = min(
                cleanup_deadline - force_reap_budget,
                loop.time() + min(0.5, self._cleanup_timeout_seconds / 4),
            )
            await _run_until(client.interrupt_session, interrupt_deadline)

        close_deadline = cleanup_deadline - force_reap_budget
        closed = await _run_until(client.close, close_deadline)
        if transport is None:
            return

        forced = False
        if not closed or not transport.is_reaped():
            remaining = cleanup_deadline - loop.time()
            if remaining > 0:
                with contextlib.suppress(
                    TimeoutError,
                    ProcessLookupError,
                    OSError,
                ):
                    forced = await transport.force_kill_and_reap(remaining)
        if (forced or transport.consumed_forced_kill()) and self._metrics is not None:
            self._metrics.increment_forced_kills()


def _create_client(
    droid_path: str,
    workdir: Path,
    *,
    grace_period: float = 1.0,
) -> DroidClient:
    transport = ProcessTransport(
        exec_path=droid_path,
        cwd=str(workdir),
        grace_period=grace_period,
    )
    return DroidClient(transport=transport)


def _resolve_model_id(model: str, model_alias: str) -> str | None:
    return None if model == model_alias else model


def _resolve_reasoning_effort(value: str | None) -> ReasoningEffort | None:
    if value is None or not value.strip():
        return None
    normalized = value.strip().lower()
    try:
        return ReasoningEffort(normalized)
    except ValueError as exc:
        valid = ", ".join(item.value for item in ReasoningEffort)
        raise RunnerError(
            f"Unsupported reasoning_effort '{value}'. Expected one of: {valid}.",
            status_code=400,
            error_type="invalid_request_error",
        ) from exc


def normalize_reasoning_effort(value: str | None) -> str | None:
    resolved = _resolve_reasoning_effort(value)
    return resolved.value if resolved is not None else None


def _map_usage(event: TokenUsageUpdate) -> Usage:
    return Usage(
        input_tokens=max(0, event.input_tokens),
        output_tokens=max(0, event.output_tokens),
        cache_read_tokens=max(0, event.cache_read_tokens),
        cache_write_tokens=max(0, event.cache_write_tokens),
    )


async def _run_until(
    operation: Callable[[], Awaitable[object]],
    deadline: float,
) -> bool:
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        return False
    task: asyncio.Future[object] = asyncio.ensure_future(operation())
    done, _ = await asyncio.wait({task}, timeout=remaining)
    if not done:
        task.cancel()
        task.add_done_callback(_consume_future_result)
        return False
    try:
        await task
    except (Exception, asyncio.CancelledError):
        return False
    return True


def _consume_future_result(future: asyncio.Future[object]) -> None:
    with contextlib.suppress(Exception, asyncio.CancelledError):
        future.result()


def _timeout_error(request: RunRequest) -> RunnerError:
    return RunnerError(
        f"Factory Droid timed out after {request.timeout_seconds:.1f} seconds.",
        status_code=504,
        error_type="factory_droid_timeout",
    )
