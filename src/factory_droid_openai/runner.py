from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from droid_sdk import (
    AssistantTextDelta,
    DroidClient,
    DroidClientError,
    ErrorEvent,
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


class DroidRunner:
    def __init__(
        self,
        *,
        droid_path: str,
        workdir: Path,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self._droid_path = droid_path
        self._workdir = workdir
        self._client_factory = client_factory or _create_client

    async def run(self, request: RunRequest) -> AsyncGenerator[RunEvent, None]:
        client = self._client_factory(self._droid_path, self._workdir)
        initialized = False
        completed = False
        usage = Usage()
        client.set_permission_handler(lambda _params: "cancel")
        client.set_ask_user_handler(
            lambda _params: {"cancelled": True, "answers": []},
        )

        try:
            async with asyncio.timeout(request.timeout_seconds):
                await client.connect()
                await client.initialize_session(
                    machine_id="factory-droid-openai",
                    cwd=str(self._workdir),
                    model_id=_resolve_model_id(request.model, request.model_alias),
                    reasoning_effort=_resolve_reasoning_effort(request.reasoning_effort),
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
            raise RunnerError(
                f"Factory Droid timed out after {request.timeout_seconds:.1f} seconds.",
                status_code=504,
                error_type="factory_droid_timeout",
            ) from exc
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
            if initialized and not completed:
                await _best_effort(client.interrupt_session)
            await _best_effort(client.close)


def _create_client(droid_path: str, workdir: Path) -> DroidClient:
    return DroidClient(exec_path=droid_path, cwd=str(workdir))


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


def _map_usage(event: TokenUsageUpdate) -> Usage:
    return Usage(
        input_tokens=max(0, event.input_tokens),
        output_tokens=max(0, event.output_tokens),
        cache_read_tokens=max(0, event.cache_read_tokens),
        cache_write_tokens=max(0, event.cache_write_tokens),
    )


async def _best_effort(operation: Callable[[], Awaitable[object]]) -> None:
    with contextlib.suppress(Exception, asyncio.CancelledError):
        await asyncio.wait_for(operation(), timeout=2.0)
