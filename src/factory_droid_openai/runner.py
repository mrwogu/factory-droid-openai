from __future__ import annotations

import asyncio
import contextlib
import json
import re
import signal
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

from droid_sdk import (
    AssistantTextDelta,
    DroidClient,
    DroidClientError,
    ErrorEvent,
    ProcessTransport,
    SessionNotFoundError,
    ThinkingTextDelta,
    TokenUsageUpdate,
    ToolProgress,
    ToolResult,
    ToolUse,
    TurnComplete,
    WorkingStateChanged,
)
from droid_sdk import TimeoutError as DroidTimeoutError
from droid_sdk.schemas.enums import (
    AutonomyLevel,
    DroidInteractionMode,
    ReasoningEffort,
)

from factory_droid_openai.config import (
    DEFAULT_MODEL_ALIAS,
    DEFAULT_SESSION_INIT_TIMEOUT_SECONDS,
)
from factory_droid_openai.dialects import TOOL_CALL_CLOSE, TOOL_CALL_OPEN
from factory_droid_openai.droid_rpc import (
    CompactionResult,
    ContextBreakdown,
    ContextStats,
    DroidRpcExtension,
    NativeToolUnavailableError,
)
from factory_droid_openai.logs import TRACE as _TRACE_LEVEL
from factory_droid_openai.logs import current_timeline
from factory_droid_openai.logs import debug as log_debug
from factory_droid_openai.logs import enabled as log_enabled
from factory_droid_openai.logs import millis as _millis
from factory_droid_openai.logs import trace as log_trace
from factory_droid_openai.logs import warning as log_warning
from factory_droid_openai.mcp_tools import (
    MCP_TOOL_ID_PREFIX,
    NativeToolBinding,
)

# droid exec flags the SDK's ProcessTransport always passes. Overriding
# exec_args replaces this list wholesale, so extra flags must be appended
# to a copy of it rather than passed on their own.
_BASE_EXEC_ARGS = (
    "exec",
    "--input-format",
    "stream-jsonrpc",
    "--output-format",
    "stream-jsonrpc",
)
# Droid refuses a model an organization policy blocks with a JSON-RPC internal
# error, so the wording is the only signal that the model, not the bridge, is
# at fault.
_MODEL_DENIED_PATTERN = re.compile(
    r"(?:model (?:is |was )?not (?:allowed|available|permitted|enabled|found|deployed)"
    r"|model (?:is |was )?inaccessible"
    r"|invalid model id)",
    re.IGNORECASE,
)
# Droid emits this generic message for a transient upstream connection blip.
_TRANSIENT_CONNECTION_PATTERN = re.compile(r"connection error\.?", re.IGNORECASE)
# Droid keeps two of its own meta tools callable whatever a session disables:
# exit-spec-mode, and the loader that would fetch a deferred tool. Neither
# touches the machine, and Droid refuses to run them in a session with every
# tool disabled, so the model only wastes its own turn reaching for them.
# Weaker models do exactly that before answering, which is worth tolerating;
# every other native tool still fails the turn closed.
_IGNORED_NATIVE_TOOLS = frozenset({"exitspecmode", "toolsearch"})
_MODEL_FAMILY_PREFIXES = (
    (("gpt-", "o1", "o3", "o4"), "gpt"),
    (("gemini",), "gemini"),
    (("claude",), "claude"),
    (("qwen",), "qwen"),
    (("kimi",), "kimi"),
    (("deepseek",), "deepseek"),
)


def _is_ignorable_native_tool(tool_name: str) -> bool:
    """Report whether an event names a Droid meta tool the bridge tolerates."""
    return tool_name.replace("-", "").replace("_", "").casefold() in _IGNORED_NATIVE_TOOLS


def _native_tool_ids(binding: NativeToolBinding | None) -> frozenset[str] | None:
    """Tool ids Droid has to report for ``binding``, or ``None`` for text tools.

    Both a warm session and a per-request session verify against this, so a
    session that never published the request's catalog is never used to serve
    a native turn.
    """
    if binding is None:
        return None
    return frozenset(f"{MCP_TOOL_ID_PREFIX}{name}" for name in binding.names)


def _native_tool_marker(name: str, arguments: Any) -> str:
    """Render a structured tool call in the bridge's own marker form."""
    payload = json.dumps(
        {"name": name, "arguments": arguments if isinstance(arguments, dict) else {}},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    # An argument value may legitimately contain marker text. Escaping every
    # "<" keeps the markers the only ones in the stream and decodes back to the
    # value the model sent.
    escaped = payload.replace("<", "\\u003c")
    return f"{TOOL_CALL_OPEN}{escaped}{TOOL_CALL_CLOSE}"


def model_family(model: str) -> str:
    """Groups a requested model or resolved Droid model id into one family.

    Telemetry labels group traffic by family so a model id never becomes a
    metric label of its own.
    """
    normalized = model.strip().lower()
    if normalized == DEFAULT_MODEL_ALIAS:
        return "factory_default"
    for prefixes, family in _MODEL_FAMILY_PREFIXES:
        if normalized.startswith(prefixes):
            return family
    return "other"


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
class SessionKey:
    """Droid session settings a warm session was initialized with."""

    model_id: str | None
    reasoning_effort: str | None

    @classmethod
    def from_request(
        cls,
        *,
        model: str,
        model_alias: str,
        reasoning_effort: str | None,
    ) -> SessionKey:
        return cls(
            model_id=_resolve_model_id(model, model_alias),
            reasoning_effort=reasoning_effort,
        )

    def can_retune_from(self, other: SessionKey) -> bool:
        """Whether a session warmed for ``other`` can be repointed at ``self``.

        Only the reasoning effort is retunable. A retune swaps the model id
        while Droid keeps the tool-call template the session was initialized
        with, and that template belongs to the model rather than to its
        family: Kimi K2 frames calls in ``<|tool_calls_section_begin|>`` where
        K3 uses ``<|open|>tools<|sep|>``. Any model change therefore waits for
        a fresh or exact-match session. ``None`` means "whatever Droid
        defaults to", which cannot be restored on a session that already
        carries an explicit value.
        """
        if self.model_id is None or self.model_id != other.model_id:
            return False
        return not (self.reasoning_effort is None and other.reasoning_effort is not None)


@dataclass(slots=True)
class WarmSession:
    key: SessionKey
    client: DroidClient
    transport: _ManagedProcessTransport | None
    session_id: str | None
    created_at: float
    native_binding: NativeToolBinding | None = None
    consumed: bool = False

    def is_alive(self) -> bool:
        return self.transport is None or not self.transport.is_reaped()


@dataclass(frozen=True, slots=True)
class RunRequest:
    prompt: str
    model: str
    model_alias: str
    reasoning_effort: str | None
    timeout_seconds: float
    deadline: float | None = None
    images: tuple[dict[str, Any], ...] = ()
    documents: tuple[dict[str, Any], ...] = ()
    session_id: str | None = None
    output_format: dict[str, Any] | None = None
    warm_session: WarmSession | None = None
    # Set when the request's tools are published to Droid over MCP instead of
    # being described in the prompt.
    native_tools: NativeToolBinding | None = None

    def session_key(self) -> SessionKey:
        return SessionKey.from_request(
            model=self.model,
            model_alias=self.model_alias,
            reasoning_effort=self.reasoning_effort,
        )


@dataclass(frozen=True, slots=True)
class DroidModel:
    id: str
    display_name: str
    provider: str
    supported_reasoning_efforts: tuple[str, ...]
    default_reasoning_effort: str
    supports_images: bool
    supports_pdfs: bool


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


@dataclass(frozen=True, slots=True)
class SessionStarted:
    session_id: str


@dataclass(frozen=True, slots=True)
class StatusUpdate:
    state: str


RunEvent = TextDelta | ReasoningDelta | UsageUpdate | RunComplete | SessionStarted | StatusUpdate
ClientFactory = Callable[[str, Path], DroidClient]
OperationResult = TypeVar("OperationResult")


class RunnerMetrics(Protocol):
    def observe_droid_startup(self, seconds: float) -> None: ...

    def increment_forced_kills(self) -> None: ...


class SessionReaper(Protocol):
    def submit(self, coroutine: Coroutine[Any, Any, None]) -> None: ...


class _ManagedProcessTransport(ProcessTransport):
    def __init__(
        self,
        *,
        exec_path: str,
        cwd: str,
        grace_period: float,
        exec_args: list[str] | None = None,
    ) -> None:
        super().__init__(
            exec_path=exec_path,
            cwd=cwd,
            grace_period=grace_period,
            exec_args=exec_args,
        )
        self._owned_process: asyncio.subprocess.Process | None = None
        self._forced_kill = False

    async def connect(self) -> None:
        await super().connect()
        self._owned_process = self._process

    async def close(self) -> None:
        process = self._owned_process
        was_running = process is not None and process.returncode is None
        await super().close()
        if was_running and process is not None and process.returncode is not None:
            # Only SIGKILL proves a forced kill. The SDK's own close() sends
            # SIGTERM as its first shutdown step, so a droid build that does
            # not trap SIGTERM dies with -SIGTERM during a fully graceful
            # close and must not be counted.
            self._forced_kill = process.returncode == -signal.SIGKILL

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
        process_grace_seconds: float = 2.0,
        cleanup_timeout_seconds: float = 4.0,
        session_init_timeout_seconds: float = DEFAULT_SESSION_INIT_TIMEOUT_SECONDS,
        metrics: RunnerMetrics | None = None,
        worktree: str | None = None,
        append_system_prompt_file: Path | None = None,
        rpc_extension: DroidRpcExtension | None = None,
        reaper: SessionReaper | None = None,
    ) -> None:
        self._droid_path = droid_path
        self._workdir = workdir
        self._client_factory = client_factory
        self._process_grace_seconds = process_grace_seconds
        self._cleanup_timeout_seconds = cleanup_timeout_seconds
        self._session_init_timeout_seconds = session_init_timeout_seconds
        self._metrics = metrics
        self._rpc = rpc_extension or DroidRpcExtension()
        self._reaper = reaper
        self._exec_args = _build_exec_args(
            worktree=worktree,
            append_system_prompt_file=append_system_prompt_file,
        )

    async def warm(
        self,
        key: SessionKey,
        *,
        timeout_seconds: float,
        native_tools: NativeToolBinding | None = None,
    ) -> WarmSession:
        """Start a Droid session that is initialized but has no turn yet."""
        client, transport = self._new_client()
        client.set_permission_handler(lambda _params: "cancel")
        client.set_ask_user_handler(
            lambda _params: {"cancelled": True, "answers": []},
        )
        started = time.perf_counter()
        try:

            async def initialize() -> None:
                await client.connect()
                await client.initialize_session(
                    machine_id="factory-droid-openai",
                    cwd=str(self._workdir),
                    mcp_servers=([] if native_tools is None else [native_tools.server_config()]),
                    model_id=key.model_id,
                    reasoning_effort=_resolve_reasoning_effort(key.reasoning_effort),
                    interaction_mode=DroidInteractionMode.Auto,
                    autonomy_level=AutonomyLevel.Off,
                    skip_permissions_unsafe=False,
                    enabled_tool_ids=[],
                )
                await self._rpc.disable_native_tools(
                    client,
                    keep_tool_prefix=(None if native_tools is None else MCP_TOOL_ID_PREFIX),
                    expected_tool_ids=_native_tool_ids(native_tools),
                    native_server_url=(None if native_tools is None else native_tools.url),
                )

            await self._run_session_init(
                asyncio.get_running_loop().time() + timeout_seconds,
                initialize,
            )
        except BaseException:
            await self.discard(
                WarmSession(
                    key,
                    client,
                    transport,
                    None,
                    started,
                    native_binding=native_tools,
                )
            )
            raise
        if self._metrics is not None:
            self._metrics.observe_droid_startup(time.perf_counter() - started)
        return WarmSession(
            key=key,
            client=client,
            transport=transport,
            session_id=client.session_id,
            created_at=asyncio.get_running_loop().time(),
            native_binding=native_tools,
        )

    async def _run_session_init(
        self,
        deadline: float,
        operation: Callable[[], Awaitable[OperationResult]],
        *,
        request_timeout: asyncio.Timeout | None = None,
    ) -> OperationResult:
        """Run session setup under its own deadline inside the request budget."""
        started = asyncio.get_running_loop().time()
        phase_deadline = min(deadline, started + self._session_init_timeout_seconds)
        try:
            async with asyncio.timeout_at(phase_deadline):
                return await operation()
        except (TimeoutError, DroidTimeoutError) as exc:
            raise self._session_init_timeout_error(started) from exc
        except asyncio.CancelledError as exc:
            # A request deadline that lands on the same instant as the phase
            # deadline cancels this scope before the phase timer converts it,
            # and the phase still names where the budget went. A cancel from
            # anywhere else, such as a client disconnect, has to stay one.
            if request_timeout is None or not request_timeout.expired():
                raise
            raise self._session_init_timeout_error(started) from exc

    def _session_init_timeout_error(self, started: float) -> RunnerError:
        elapsed = asyncio.get_running_loop().time() - started
        log_warning(
            "droid.timeout",
            phase="session_init",
            elapsed_s=round(elapsed, 1),
        )
        return RunnerError(
            f"Factory Droid session initialization timed out after {elapsed:.1f} seconds.",
            status_code=504,
            error_type="factory_droid_timeout",
        )

    async def discard(self, session: WarmSession) -> None:
        """Tear down a warm session that will not serve a turn."""
        await self._cleanup(session.client, session.transport, interrupt=False)

    async def _retune(self, warm: WarmSession, key: SessionKey) -> None:
        """Repoint a warm session at the settings this turn asked for."""
        if key == warm.key:
            return
        model_id = key.model_id
        effort = _resolve_reasoning_effort(key.reasoning_effort)
        if model_id is None or effort is None or not key.can_retune_from(warm.key):
            raise RunnerError(
                "Warm Droid session cannot be repointed at the requested settings.",
            )
        started = time.perf_counter()
        await self._rpc.retune_session(
            warm.client,
            model_id=model_id,
            reasoning_effort=effort.value,
        )
        previous = warm.key
        warm.key = key
        timeline = current_timeline()
        retune_seconds = time.perf_counter() - started
        log_debug(
            "droid.session_retuned",
            model=key.model_id,
            previous_model=previous.model_id,
            reasoning_effort=key.reasoning_effort,
            retune_ms=(
                timeline.observe("retune_ms", retune_seconds)
                if timeline is not None
                else _millis(retune_seconds)
            ),
        )

    async def run(self, request: RunRequest) -> AsyncGenerator[RunEvent, None]:
        reasoning_effort = _resolve_reasoning_effort(request.reasoning_effort)
        loop = asyncio.get_running_loop()
        deadline = (
            request.deadline
            if request.deadline is not None
            else loop.time() + request.timeout_seconds
        )
        if deadline <= loop.time():
            raise _timeout_error(request)

        # The deadline above is the request's total budget (queue wait plus the
        # run). From here on the message reports how long the Droid run itself
        # took before it timed out, so a 60 s model-backend timeout no longer
        # reads as the configured 600 s ceiling.
        started = loop.time()
        warm = request.warm_session
        if warm is not None:
            # A binding always carries its catalog, so ``None`` here means the
            # side publishes no tools at all, and one comparison covers both a
            # catalog mismatch and a native session offered to a text turn.
            warm_catalog = None if warm.native_binding is None else warm.native_binding.catalog
            requested_catalog = (
                None if request.native_tools is None else request.native_tools.catalog
            )
            if warm_catalog != requested_catalog:
                raise RunnerError(
                    "A warm Droid session cannot serve a different native tool catalog.",
                )
            warm.consumed = True
            client, transport = warm.client, warm.transport
        else:
            client, transport = self._new_client()
            client.set_permission_handler(lambda _params: "cancel")
            client.set_ask_user_handler(
                lambda _params: {"cancelled": True, "answers": []},
            )
        mcp_servers = [] if request.native_tools is None else [request.native_tools.server_config()]
        initialized = warm is not None
        completed = False
        ignored_native_tool = False
        saw_assistant_text = False
        native_tool_calls = 0
        unmapped_event_kind: str | None = None
        usage = Usage()

        try:
            async with asyncio.timeout_at(deadline) as request_timeout:
                if warm is None:

                    async def initialize() -> None:
                        nonlocal initialized
                        startup_started = time.perf_counter()
                        try:
                            await client.connect()
                        finally:
                            startup_seconds = time.perf_counter() - startup_started
                            if self._metrics is not None:
                                self._metrics.observe_droid_startup(startup_seconds)
                            timeline = current_timeline()
                            startup_ms = (
                                timeline.observe("droid_startup_ms", startup_seconds)
                                if timeline is not None
                                else None
                            )
                            log_debug(
                                "droid.connected",
                                droid=self._droid_path,
                                droid_startup_ms=startup_ms,
                            )
                        if request.session_id is not None:
                            # Continuation: reuse the stored Droid session so only
                            # the new turn is sent instead of the full transcript.
                            await client.load_session(
                                session_id=request.session_id,
                                mcp_servers=mcp_servers,
                            )
                        else:
                            await client.initialize_session(
                                machine_id="factory-droid-openai",
                                cwd=str(self._workdir),
                                mcp_servers=mcp_servers,
                                model_id=_resolve_model_id(request.model, request.model_alias),
                                reasoning_effort=reasoning_effort,
                                interaction_mode=DroidInteractionMode.Auto,
                                autonomy_level=AutonomyLevel.Off,
                                skip_permissions_unsafe=False,
                                enabled_tool_ids=[],
                            )
                        initialized = True
                        native_binding = request.native_tools
                        await self._rpc.disable_native_tools(
                            client,
                            keep_tool_prefix=(
                                None if native_binding is None else MCP_TOOL_ID_PREFIX
                            ),
                            expected_tool_ids=_native_tool_ids(native_binding),
                            native_server_url=(
                                None if native_binding is None else native_binding.url
                            ),
                        )

                    await self._run_session_init(
                        deadline,
                        initialize,
                        request_timeout=request_timeout,
                    )
                else:
                    await self._retune(warm, request.session_key())
                session_timeline = current_timeline()
                log_debug(
                    "droid.session_ready",
                    session_id=client.session_id,
                    model=_resolve_model_id(request.model, request.model_alias),
                    reasoning_effort=_state_value(reasoning_effort) if reasoning_effort else None,
                    continuation=request.session_id is not None,
                    warm=warm is not None,
                    session_ready_ms=(
                        session_timeline.since_start("session_ready_ms")
                        if session_timeline is not None
                        else None
                    ),
                )
                if client.session_id is not None:
                    yield SessionStarted(client.session_id)
                if request.output_format is None:
                    await client.add_user_message(
                        text=request.prompt,
                        images=list(request.images) or None,
                        files=list(request.documents) or None,
                    )
                else:
                    await self._rpc.add_user_message(
                        client,
                        text=request.prompt,
                        images=list(request.images) or None,
                        files=list(request.documents) or None,
                        output_format=request.output_format,
                    )

                prompt_timeline = current_timeline()
                log_debug(
                    "droid.prompt_sent",
                    prompt_bytes=len(request.prompt.encode("utf-8")),
                    images=len(request.images),
                    documents=len(request.documents),
                    structured=request.output_format is not None,
                    prompt_sent_ms=(
                        prompt_timeline.since_start("prompt_sent_ms")
                        if prompt_timeline is not None
                        else None
                    ),
                )

                async for event in client.receive_response():
                    if log_enabled(_TRACE_LEVEL):
                        log_trace("droid.event", kind=type(event).__name__)
                    if isinstance(event, AssistantTextDelta):
                        saw_assistant_text = saw_assistant_text or bool(event.text)
                        yield TextDelta(event.text)
                    elif isinstance(event, ThinkingTextDelta):
                        yield ReasoningDelta(event.text)
                    elif isinstance(event, WorkingStateChanged):
                        yield StatusUpdate(_state_value(event.state))
                    elif isinstance(event, TokenUsageUpdate):
                        # The SDK forwards session-cumulative counters, and its
                        # own TurnComplete carries the newest update as the turn
                        # total. Summing them would count the same tokens once
                        # per event.
                        usage = _map_usage(event)
                        yield UsageUpdate(usage)
                    elif isinstance(event, TurnComplete):
                        if (
                            not saw_assistant_text
                            and not native_tool_calls
                            and (ignored_native_tool or unmapped_event_kind is not None)
                        ):
                            reason = (
                                "a disabled Droid meta-tool attempt"
                                if ignored_native_tool
                                else f"unmapped SDK event {unmapped_event_kind!r}"
                            )
                            raise RunnerError(
                                f"Factory Droid completed without assistant output after {reason}.",
                                error_type="factory_incomplete_response",
                            )
                        if event.token_usage is not None:
                            usage = _map_usage(event.token_usage)
                        completed = True
                        turn_timeline = current_timeline()
                        log_debug(
                            "droid.turn_complete",
                            input_tokens=usage.input_tokens,
                            output_tokens=usage.output_tokens,
                            cache_read_tokens=usage.cache_read_tokens,
                            turn_ms=(
                                turn_timeline.since_start("turn_ms")
                                if turn_timeline is not None
                                else None
                            ),
                        )
                        yield RunComplete(usage)
                    elif isinstance(event, (ToolUse, ToolResult, ToolProgress)):
                        if request.native_tools is not None:
                            published = request.native_tools.resolve(event.tool_name)
                            if published is not None:
                                if isinstance(event, ToolUse):
                                    native_tool_calls += 1
                                    log_debug("droid.native_tool_call", tool=published)
                                    # Reusing the marker form runs the call
                                    # through the same validation, cap and
                                    # de-duplication path as a text answer.
                                    yield TextDelta(
                                        _native_tool_marker(published, event.tool_input)
                                    )
                                continue
                            if native_tool_calls and not event.tool_name:
                                # The bridge refuses the call so the OpenAI
                                # client can run it, and Droid reports that
                                # refusal as a nameless result.
                                continue
                        if _is_ignorable_native_tool(event.tool_name):
                            ignored_native_tool = True
                            log_debug("droid.native_tool_ignored", tool=event.tool_name)
                            continue
                        # Droid answers a tolerated call with a result that
                        # carries no tool name, so provenance is what decides:
                        # nameless events belong to the call just ignored, and
                        # anything else is a native tool this bridge refuses.
                        if not event.tool_name and ignored_native_tool:
                            continue
                        # Naming the tool is what tells an operator whether the
                        # model reached for Droid's own toolset or answered a
                        # client tool in Droid's dialect instead of the markers.
                        raise RunnerError(
                            f"Factory Droid attempted to use native tool {event.tool_name!r}. "
                            "The bridge only permits tools supplied by the OpenAI client.",
                            error_type="factory_native_tool_blocked",
                        )
                    elif isinstance(event, ErrorEvent):
                        raise _error_event_failure(
                            event,
                            model=_resolve_model_id(request.model, request.model_alias),
                        )
                    else:
                        unmapped_event_kind = type(event).__name__
                        log_debug("droid.event_unmapped", kind=unmapped_event_kind)
        except (TimeoutError, DroidTimeoutError) as exc:
            raise RunnerError(
                f"Factory Droid timed out after {loop.time() - started:.1f} seconds.",
                status_code=504,
                error_type="factory_droid_timeout",
            ) from exc
        except FileNotFoundError as exc:
            raise RunnerError(
                f"Factory Droid executable was not found: {self._droid_path}",
                status_code=503,
                error_type="factory_droid_unavailable",
            ) from exc
        except SessionNotFoundError as exc:
            raise RunnerError(
                f"Factory Droid session '{request.session_id}' was not found.",
                status_code=404,
                error_type="session_not_found",
            ) from exc
        except NativeToolUnavailableError as exc:
            raise RunnerError(
                str(exc),
                status_code=503,
                error_type="factory_native_tool_unavailable",
            ) from exc
        except DroidClientError as exc:
            model_id = _resolve_model_id(request.model, request.model_alias)
            raise sdk_error(exc, model=model_id) from exc
        finally:
            interrupt = initialized and not completed
            if self._reaper is not None:
                # Interrupting and reaping a Droid process costs about a second
                # of grace period, which the client would otherwise wait for
                # after the last token.
                self._reaper.submit(self._cleanup(client, transport, interrupt=interrupt))
            else:
                cleanup_task = asyncio.create_task(
                    self._cleanup(client, transport, interrupt=interrupt)
                )
                try:
                    await asyncio.shield(cleanup_task)
                except asyncio.CancelledError:
                    await cleanup_task
                    raise

    async def list_models(self, *, timeout_seconds: float) -> tuple[DroidModel, ...]:
        models: list[Any] = []

        async def initialize(client: DroidClient) -> None:
            result = await client.initialize_session(
                machine_id="factory-droid-openai-models",
                cwd=str(self._workdir),
                mcp_servers=[],
                interaction_mode=DroidInteractionMode.Auto,
                autonomy_level=AutonomyLevel.Off,
                skip_permissions_unsafe=False,
                enabled_tool_ids=[],
            )

            models.extend(result.available_models or [])

        async def operation(client: DroidClient) -> tuple[DroidModel, ...]:
            await self._rpc.close_session(client, reason="clear")
            return tuple(
                DroidModel(
                    id=model.id,
                    display_name=model.display_name,
                    provider=_state_value(model.model_provider),
                    supported_reasoning_efforts=tuple(
                        _state_value(effort) for effort in model.supported_reasoning_efforts
                    ),
                    default_reasoning_effort=_state_value(model.default_reasoning_effort),
                    supports_images=not bool(model.no_image_support),
                    supports_pdfs=bool(model.supports_pdfs),
                )
                for model in models
            )

        return await self._session_operation(
            operation,
            timeout_seconds=timeout_seconds,
            init_operation=initialize,
        )

    async def get_context(
        self,
        session_id: str,
        *,
        timeout_seconds: float,
    ) -> tuple[ContextStats, ContextBreakdown]:
        async def operation(
            client: DroidClient,
        ) -> tuple[ContextStats, ContextBreakdown]:
            stats, breakdown = await asyncio.gather(
                self._rpc.get_context_stats(client),
                self._rpc.get_context_breakdown(client),
            )
            return stats, breakdown

        return await self._loaded_session_operation(
            session_id,
            operation,
            timeout_seconds=timeout_seconds,
            disable_tools=False,
        )

    async def compact_session(
        self,
        session_id: str,
        *,
        custom_instructions: str | None,
        timeout_seconds: float,
    ) -> CompactionResult:
        async def operation(client: DroidClient) -> CompactionResult:
            return await self._rpc.compact_session(
                client,
                custom_instructions=custom_instructions,
            )

        return await self._loaded_session_operation(
            session_id,
            operation,
            timeout_seconds=timeout_seconds,
        )

    async def fork_session(self, session_id: str, *, timeout_seconds: float) -> str:
        return await self._loaded_session_operation(
            session_id,
            self._rpc.fork_session,
            timeout_seconds=timeout_seconds,
            disable_tools=False,
        )

    async def rename_session(
        self,
        session_id: str,
        *,
        title: str,
        timeout_seconds: float,
    ) -> None:
        async def operation(client: DroidClient) -> None:
            await self._rpc.rename_session(client, title=title)

        await self._loaded_session_operation(
            session_id,
            operation,
            timeout_seconds=timeout_seconds,
            disable_tools=False,
        )

    async def close_session(self, session_id: str, *, timeout_seconds: float) -> None:
        async def operation(client: DroidClient) -> None:
            await self._rpc.close_session(client)

        await self._loaded_session_operation(
            session_id,
            operation,
            timeout_seconds=timeout_seconds,
            disable_tools=False,
        )

    async def _loaded_session_operation(
        self,
        session_id: str,
        operation: Callable[[DroidClient], Awaitable[OperationResult]],
        *,
        timeout_seconds: float,
        disable_tools: bool = True,
    ) -> OperationResult:
        async def initialize(client: DroidClient) -> None:
            await client.load_session(session_id=session_id, mcp_servers=[])
            # Metadata-only operations never run a model turn, so they neither
            # need the tool guard nor should they rewrite session settings.
            if disable_tools:
                await self._rpc.disable_native_tools(client)

        return await self._session_operation(
            operation,
            timeout_seconds=timeout_seconds,
            session_id=session_id,
            init_operation=initialize,
        )

    async def _session_operation(
        self,
        operation: Callable[[DroidClient], Awaitable[OperationResult]],
        *,
        timeout_seconds: float,
        session_id: str | None = None,
        init_operation: Callable[[DroidClient], Awaitable[None]] | None = None,
    ) -> OperationResult:
        client, transport = self._new_client()
        client.set_permission_handler(lambda _params: "cancel")
        client.set_ask_user_handler(
            lambda _params: {"cancelled": True, "answers": []},
        )
        loop = asyncio.get_running_loop()
        started = loop.time()
        deadline = started + timeout_seconds
        try:
            async with asyncio.timeout_at(deadline) as request_timeout:

                async def initialize() -> None:
                    await client.connect()
                    if init_operation is not None:
                        await init_operation(client)

                await self._run_session_init(
                    deadline,
                    initialize,
                    request_timeout=request_timeout,
                )
                return await operation(client)
        except (TimeoutError, DroidTimeoutError) as exc:
            raise RunnerError(
                f"Factory Droid timed out after"
                f" {asyncio.get_running_loop().time() - started:.1f} seconds.",
                status_code=504,
                error_type="factory_droid_timeout",
            ) from exc
        except FileNotFoundError as exc:
            raise RunnerError(
                f"Factory Droid executable was not found: {self._droid_path}",
                status_code=503,
                error_type="factory_droid_unavailable",
            ) from exc
        except SessionNotFoundError as exc:
            raise RunnerError(
                f"Factory Droid session '{session_id}' was not found.",
                status_code=404,
                error_type="session_not_found",
            ) from exc
        except DroidClientError as exc:
            raise RunnerError(
                f"Factory Droid SDK failed: {exc}",
                error_type="factory_droid_sdk_error",
            ) from exc
        finally:
            cleanup_task = asyncio.create_task(self._cleanup(client, transport, interrupt=False))
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
            exec_args=self._exec_args,
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
        cleanup_started = loop.time()
        cleanup_deadline = cleanup_started + self._cleanup_timeout_seconds
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
        if forced or transport.consumed_forced_kill():
            log_warning("droid.forced_kill", cleanup_ms=_millis(loop.time() - cleanup_started))
            if self._metrics is not None:
                self._metrics.increment_forced_kills()
            return
        log_debug("droid.cleanup", cleanup_ms=_millis(loop.time() - cleanup_started))


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


def _build_exec_args(
    *,
    worktree: str | None,
    append_system_prompt_file: Path | None,
) -> list[str] | None:
    if worktree is None and append_system_prompt_file is None:
        return None
    args = list(_BASE_EXEC_ARGS)
    if worktree is not None:
        args.extend(["--worktree", worktree])
    if append_system_prompt_file is not None:
        args.extend(["--append-system-prompt-file", str(append_system_prompt_file)])
    return args


def _resolve_model_id(model: str, model_alias: str) -> str | None:
    return None if model == model_alias else model


def sdk_error(exc: DroidClientError, *, model: str | None = None) -> RunnerError:
    """Map a Droid SDK failure onto the closest OpenAI-compatible error.

    Droid lists every model its CLI knows about and only refuses the ones an
    organization policy blocks when a session is initialized, so that refusal
    has to read as an unavailable model rather than a bridge failure.
    """
    message = str(exc)
    denied = _model_denied_error(message, model=model)
    if denied is not None:
        return denied
    return RunnerError(
        f"Factory Droid SDK failed: {message}",
        error_type="factory_droid_sdk_error",
    )


def _error_event_failure(event: ErrorEvent, *, model: str | None) -> RunnerError:
    """Map a Droid error event onto the closest OpenAI-compatible error.

    Droid reports an unusable model id mid-stream instead of failing the
    session call, so the same denial wording decides the status code here.
    """
    message = event.message or "Factory Droid returned an error."
    denied = _model_denied_error(message, model=model)
    if denied is not None:
        return denied
    if _TRANSIENT_CONNECTION_PATTERN.fullmatch(message.strip()):
        return RunnerError(
            message,
            status_code=503,
            error_type="factory_droid_unavailable",
        )
    # The SDK labels error events with its own class names, such as "Error".
    # Forwarding those as OpenAI error types would put undocumented values in
    # the public contract, so the detail stays in the message instead.
    return RunnerError(message, error_type="factory_droid_error")


def _model_denied_error(message: str, *, model: str | None) -> RunnerError | None:
    if not _MODEL_DENIED_PATTERN.search(message):
        return None
    subject = f"Model '{model}'" if model else "The requested model"
    return RunnerError(
        f"{subject} is not available for this Factory account: {message}",
        status_code=404,
        error_type="model_not_found",
    )


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


def _state_value(state: object) -> str:
    return str(getattr(state, "value", state))


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
        # Deliberately fire-and-forget: the caller's remaining budget is spent
        # on force_kill_and_reap instead of waiting for a close() that already
        # blew its deadline to unwind. Both paths then act on the same process,
        # which is safe because force_kill_and_reap suppresses
        # ProcessLookupError/OSError and Process.wait() tolerates several
        # waiters. Awaiting the cancellation here would reintroduce the hang
        # this timeout exists to break.
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
