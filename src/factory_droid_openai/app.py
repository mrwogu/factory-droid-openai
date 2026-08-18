from __future__ import annotations

import asyncio
import contextlib
import json
import re
import secrets
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Annotated, Any, TypeVar

from fastapi import Depends, FastAPI, Request, Response, Security
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for

from factory_droid_openai.availability import ModelQuarantine
from factory_droid_openai.config import DEFAULT_TOOL_CALL_DRAIN_SECONDS, Settings
from factory_droid_openai.droid_rpc import DroidRpcExtension
from factory_droid_openai.logs import bind_request, current_timeline
from factory_droid_openai.logs import debug as log_debug
from factory_droid_openai.logs import info as log_info
from factory_droid_openai.logs import warning as log_warning
from factory_droid_openai.mcp_tools import (
    NativeToolBinding,
    NativeToolRegistry,
    build_router,
    to_mcp_tools,
)
from factory_droid_openai.metrics import BridgeMetrics
from factory_droid_openai.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompactSessionRequest,
    CompactSessionResponse,
    ContextBreakdownResponse,
    ContextCategoryResponse,
    ContextStatsResponse,
    ErrorResponse,
    ForkSessionResponse,
    HealthResponse,
    ModelInfo,
    ModelListResponse,
    RenameSessionRequest,
    SessionContextResponse,
    SessionOperationResponse,
    VersionResponse,
)
from factory_droid_openai.payloadlog import configure_payload_tracing
from factory_droid_openai.pool import BackgroundReaper, WarmSessionPool
from factory_droid_openai.protocol import (
    IncompleteToolCallError,
    MalformedToolCallError,
    ProtocolEmission,
    ProtocolError,
    RequestTooLargeError,
    StopSequenceBuffer,
    TextEmission,
    ToolCallEmission,
    ToolCallStreamParser,
    build_prompt,
    parse_strict_json,
)
from factory_droid_openai.runner import (
    DroidModel,
    DroidRunner,
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
    model_family,
    normalize_reasoning_effort,
)
from factory_droid_openai.strictjson import (
    DuplicateKeyError,
    check_no_duplicate_keys,
    json_depth_exceeds,
)
from factory_droid_openai.telemetry import DEFAULT_TELEMETRY_ENDPOINT, TelemetryReporter

if TYPE_CHECKING:
    from jsonschema.protocols import Validator
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

RunnerFactory = Callable[[], DroidRunner]
BearerCredentials = Annotated[
    HTTPAuthorizationCredentials | None,
    Security(HTTPBearer(auto_error=False)),
]

# Warming happens off the request path, so it never needs the full request
# timeout budget.
_WARM_TIMEOUT_CEILING = 120.0
_BRIDGE_VERSION = "1.6.12"  # x-release-please-version
_LATENCY_BUCKETS = (
    (0.1, "lt_100ms"),
    (0.5, "100ms_500ms"),
    (1.0, "500ms_1s"),
    (5.0, "1s_5s"),
    (30.0, "5s_30s"),
)
_PAYLOAD_SIZE_BUCKETS = (
    (10_240, "lt_10kb"),
    (102_400, "10kb_100kb"),
    (1_048_576, "100kb_1mb"),
)

CHAT_COMPLETION_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {
        "model": ChatCompletionResponse,
        "description": "A JSON completion or an SSE completion stream.",
        "content": {
            "text/event-stream": {
                "schema": {"type": "string"},
                "example": 'data: {"object":"chat.completion.chunk",...}\n\ndata: [DONE]\n\n',
            }
        },
    },
    "4XX": {
        "model": ErrorResponse,
        "description": "Invalid request or bearer authentication failure.",
    },
    413: {
        "model": ErrorResponse,
        "description": "Request body or serialized transcript exceeds configured limits.",
    },
    429: {
        "model": ErrorResponse,
        "description": "The bounded Droid request queue is full.",
        "headers": {
            "Retry-After": {
                "description": "Seconds to wait before retrying.",
                "schema": {"type": "integer"},
            }
        },
    },
    404: {
        "model": ErrorResponse,
        "description": "The requested Factory Droid session is unknown to this bridge.",
    },
    409: {
        "model": ErrorResponse,
        "description": "The requested Factory Droid session is already in use.",
    },
    502: {
        "model": ErrorResponse,
        "description": "Droid SDK, process, or bridge protocol failure.",
    },
    503: {
        "model": ErrorResponse,
        "description": "Droid executable unavailable.",
    },
    504: {
        "model": ErrorResponse,
        "description": "Droid request timeout.",
    },
}

FACTORY_OPERATION_RESPONSES: dict[int | str, dict[str, Any]] = {
    "4XX": {
        "model": ErrorResponse,
        "description": "Invalid request or bearer authentication failure.",
    },
    404: {
        "model": ErrorResponse,
        "description": "The requested Factory Droid session is unknown to this bridge.",
    },
    409: CHAT_COMPLETION_RESPONSES[409],
    429: CHAT_COMPLETION_RESPONSES[429],
    502: CHAT_COMPLETION_RESPONSES[502],
    503: CHAT_COMPLETION_RESPONSES[503],
    504: CHAT_COMPLETION_RESPONSES[504],
}
MODEL_LIST_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {
        "model": ModelListResponse,
        "description": "The bridge alias and the discovered Droid model catalog.",
        "headers": {
            "x-factory-droid-model-discovery": {
                "description": (
                    "Set to 'degraded' when discovery failed and the bridge served the "
                    "last known catalog, or the alias alone."
                ),
                "schema": {"type": "string"},
            }
        },
    },
    **{
        status: response
        for status, response in FACTORY_OPERATION_RESPONSES.items()
        if status != 404
    },
}
MODEL_RETRIEVE_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {
        "model": ModelInfo,
        "description": "The bridge alias or one discovered Droid model.",
        "headers": MODEL_LIST_RESPONSES[200]["headers"],
    },
    **{
        status: response
        for status, response in FACTORY_OPERATION_RESPONSES.items()
        if status != 404
    },
    404: {
        "model": ErrorResponse,
        "description": "The requested model is unknown to this bridge.",
    },
}


FactoryOperationResult = TypeVar("FactoryOperationResult")


class AdmissionRejectedError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class StructuredOutput:
    """A validated ``response_format`` request, compiled once per completion."""

    payload: dict[str, Any]
    validator: Validator
    max_bytes: int
    max_depth: int = 32


class StructuredOutputBuffer:
    """Collects structured completion text until it can be validated.

    Structured output is only useful once the whole JSON document is present,
    so the bridge holds it back instead of streaming partial JSON. The byte
    cap keeps a runaway generation from growing the buffer without bound.
    """

    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max_bytes
        self._parts: list[str] = []
        self._bytes = 0

    def append(self, text: str) -> None:
        self._bytes += len(text.encode("utf-8"))
        if self._bytes > self._max_bytes:
            raise ProtocolError(
                f"Factory Droid structured output exceeds maximum of {self._max_bytes} bytes"
            )
        self._parts.append(text)

    def text(self) -> str:
        return "".join(self._parts)


class ModelCatalog:
    """Caches discovered Droid models behind a single-flight lock.

    Model discovery starts a Droid process and takes an admission slot, so an
    uncached endpoint would let routine ``GET /v1/models`` polling stall chat
    traffic. Concurrent callers share one discovery; a failed discovery serves
    the last known catalog instead of replacing it.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float,
        load: Callable[[], Awaitable[tuple[DroidModel, ...]]],
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._load = load
        self._lock = asyncio.Lock()
        self._models: tuple[DroidModel, ...] | None = None
        self._expires_at = 0.0

    async def get(self) -> tuple[tuple[DroidModel, ...], bool]:
        async with self._lock:
            loop = asyncio.get_running_loop()
            cached = self._models
            if cached is not None and loop.time() < self._expires_at:
                return cached, False
            try:
                models = await self._load()
            except BridgeHTTPError as exc:
                if exc.status_code not in {502, 503, 504}:
                    raise
                return cached or (), True
            self._models = models
            self._expires_at = loop.time() + self._ttl_seconds
            return models, False


class SessionRegistry:
    """Tracks the Droid sessions this bridge process created.

    Continuation only ever resumes a session the bridge itself started.
    Accepting arbitrary IDs would let a client load unrelated sessions the
    local Droid CLI stored - including the operator's own interactive work -
    and read their contents back out through the completion.
    """

    def __init__(self, max_entries: int) -> None:
        self._max_entries = max_entries
        self._sessions: OrderedDict[str, SessionKey] = OrderedDict()
        self._in_use: set[str] = set()

    def remember(self, session_id: str, key: SessionKey) -> None:
        self._sessions.pop(session_id, None)
        self._sessions[session_id] = key
        self._evict_excess(protected=session_id)

    def _evict_excess(self, *, protected: str | None = None) -> None:
        while len(self._sessions) > self._max_entries:
            candidate = next(
                (
                    existing_id
                    for existing_id in self._sessions
                    if existing_id not in self._in_use and existing_id != protected
                ),
                None,
            )
            if candidate is None:
                return
            self._sessions.pop(candidate)

    def key(self, session_id: str) -> SessionKey | None:
        key = self._sessions.get(session_id)
        if key is None:
            return None
        self._sessions.move_to_end(session_id)
        return key

    def forget(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def acquire(self, session_id: str) -> SessionUseLease | None:
        if session_id in self._in_use:
            return None
        self._in_use.add(session_id)
        return SessionUseLease(self, session_id)

    def release(self, session_id: str) -> None:
        self._in_use.discard(session_id)
        self._evict_excess()


class SessionUseLease:
    """Keeps one bridge-created Droid session on one operation at a time."""

    def __init__(self, registry: SessionRegistry, session_id: str) -> None:
        self._registry = registry
        self._session_id = session_id
        self._released = False

    async def __aenter__(self) -> SessionUseLease:
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: object,
    ) -> None:
        self.release()

    def release(self) -> None:
        if self._released:
            return
        self._registry.release(self._session_id)
        self._released = True


class AdmissionLease:
    """A held admission slot, releasable more than once.

    Streaming releases the lease from two places: the generator's own
    ``async with``, and the response finalizer. Only the finalizer runs when a
    generator is closed before it ever starts, so both are needed; the
    idempotency guard in ``release`` is what makes the overlap safe. Removing
    either site leaks a slot.
    """

    def __init__(self, admission: AdmissionController) -> None:
        self._admission = admission
        self._released = False
        self._release_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> AdmissionLease:
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: object,
    ) -> None:
        await self.release()

    async def release(self) -> None:
        if self._released:
            return
        if self._release_task is None:
            self._release_task = asyncio.create_task(self._admission.release())
        cancelled: asyncio.CancelledError | None = None
        while not self._release_task.done():
            try:
                await asyncio.shield(self._release_task)
            except asyncio.CancelledError as exc:
                cancelled = exc
        await self._release_task
        self._released = True
        if cancelled is not None:
            raise cancelled


class AdmissionController:
    def __init__(
        self,
        *,
        max_concurrency: int,
        max_queue_size: int,
        metrics: BridgeMetrics,
    ) -> None:
        self._max_concurrency = max_concurrency
        self._max_queue_size = max_queue_size
        self._metrics = metrics
        self._condition = asyncio.Condition()
        self._active = 0
        self._waiters: deque[object] = deque()

    async def acquire(self, deadline: float) -> AdmissionLease:
        loop = asyncio.get_running_loop()
        started = loop.time()
        ticket: object | None = None
        async with self._condition:
            if deadline <= loop.time():
                raise TimeoutError
            if self._active >= self._max_concurrency or self._waiters:
                if len(self._waiters) >= self._max_queue_size:
                    raise AdmissionRejectedError
                ticket = object()
                self._waiters.append(ticket)
                self._publish()
                try:
                    async with asyncio.timeout_at(deadline):
                        while (
                            self._active >= self._max_concurrency or self._waiters[0] is not ticket
                        ):
                            await self._condition.wait()
                except (TimeoutError, asyncio.CancelledError):
                    self._waiters.remove(ticket)
                    self._publish()
                    self._condition.notify_all()
                    raise
                self._waiters.popleft()

            self._active += 1
            self._publish()

        self._metrics.observe_queue_wait(loop.time() - started)
        return AdmissionLease(self)

    async def release(self) -> None:
        async with self._condition:
            self._active -= 1
            self._publish()
            self._condition.notify_all()

    def _publish(self) -> None:
        self._metrics.set_admission(
            active=self._active,
            queued=len(self._waiters),
        )


class _RequestPayloadLimitError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class _ClientDisconnectedError(Exception):
    pass


_JSON_STRUCTURAL = re.compile(rb'["\\{}\[\]]')
_JSON_QUOTE = ord('"')
_JSON_BACKSLASH = ord("\\")
_JSON_OPENERS = frozenset({ord("{"), ord("[")})
_JSON_CLOSERS = frozenset({ord("}"), ord("]")})


class _JsonDepthTracker:
    """Bounds JSON nesting depth while the body is still being received.

    Only quotes, escapes and brackets can change the state, so the search for
    them runs in the regex engine rather than as one interpreter step per
    byte. A large but flat body - a single multi-megabyte message string is
    the common shape - costs a native scan instead of blocking the event loop
    for tens of milliseconds.
    """

    def __init__(self, max_depth: int) -> None:
        self._max_depth = max_depth
        self._depth = 0
        self._in_string = False
        # Index of the byte neutralized by a backslash carried over from the
        # previous chunk; -1 when no escape is pending.
        self._escaped_index = -1

    def feed(self, data: bytes) -> None:
        escaped_index = self._escaped_index
        self._escaped_index = -1
        for match in _JSON_STRUCTURAL.finditer(data):
            index = match.start()
            if index == escaped_index:
                continue
            value = data[index]
            if self._in_string:
                if value == _JSON_BACKSLASH:
                    escaped_index = index + 1
                elif value == _JSON_QUOTE:
                    self._in_string = False
                continue
            if value == _JSON_QUOTE:
                self._in_string = True
            elif value in _JSON_OPENERS:
                self._depth += 1
                if self._depth > self._max_depth:
                    raise _RequestPayloadLimitError("Request JSON exceeds configured depth limit.")
            elif value in _JSON_CLOSERS:
                self._depth = max(0, self._depth - 1)
        if escaped_index == len(data):
            self._escaped_index = 0


class RequestSizeLimitMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        max_request_bytes: int,
        max_json_depth: int,
        body_timeout_seconds: float,
        metrics: BridgeMetrics,
    ) -> None:
        self._app = app
        self._max_request_bytes = max_request_bytes
        self._max_json_depth = max_json_depth
        self._body_timeout_seconds = body_timeout_seconds
        self._metrics = metrics

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        # Every HTTP route is covered, not only chat completions: the session
        # extensions accept request bodies too, and FastAPI buffers a body
        # before authentication runs.
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        started = time.perf_counter()
        status_code = 500
        response_started = False
        content_length = _content_length(scope)

        async def tracked_send(message: Message) -> None:
            nonlocal response_started, status_code
            if message["type"] == "http.response.start":
                response_started = True
                status_code = int(message["status"])
            await send(message)

        try:
            if content_length is not None and content_length > self._max_request_bytes:
                _set_request_state(
                    scope,
                    payload_size_bucket=_payload_size_bucket(content_length),
                )
                status_code = 413
                self._metrics.increment_payload_rejections()
                await _send_payload_too_large(
                    scope,
                    receive,
                    send,
                    "Request body exceeds configured size limit.",
                )
                return
            async with asyncio.timeout(self._body_timeout_seconds):
                body = await _read_limited_body(
                    receive,
                    max_request_bytes=self._max_request_bytes,
                    max_json_depth=self._max_json_depth,
                )
            _set_request_state(scope, payload_size_bucket=_payload_size_bucket(len(body)))
            # FastAPI binds the body with Pydantic, which keeps the last value for
            # a repeated key instead of rejecting it. A duplicate key in the raw
            # request body is almost always a client bug, so the bridge fails
            # closed here before the handler can act on a silently-wrong value.
            if body and _json_content_type(scope):
                try:
                    check_no_duplicate_keys(body.decode("utf-8"))
                except DuplicateKeyError as exc:
                    status_code = 400
                    await _send_asgi_error(
                        scope,
                        receive,
                        send,
                        message=str(exc),
                        status_code=400,
                        error_type="invalid_request_error",
                    )
                    return
                except (UnicodeDecodeError, json.JSONDecodeError):
                    # Malformed JSON or encoding is left for FastAPI to report.
                    pass
            replayed = False

            async def replay_receive() -> Message:
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {
                        "type": "http.request",
                        "body": body,
                        "more_body": False,
                    }
                return await receive()

            await self._app(scope, replay_receive, tracked_send)
        except _RequestPayloadLimitError as exc:
            if response_started:
                raise
            status_code = 413
            self._metrics.increment_payload_rejections()
            await _send_payload_too_large(scope, receive, send, exc.message)
        except TimeoutError:
            if response_started:
                raise
            status_code = 408
            await _send_asgi_error(
                scope,
                receive,
                send,
                message="Request body timed out.",
                status_code=408,
                error_type="request_timeout",
            )
        except _ClientDisconnectedError:
            status_code = 499
        finally:
            elapsed = time.perf_counter() - started
            outcome = _request_outcome(status_code, scope)
            self._metrics.record_request(
                outcome,
                status_code,
                elapsed,
                route=_request_route(scope),
                mode=_request_mode(scope),
                features=_request_telemetry_features(
                    scope,
                    status_code=status_code,
                    seconds=elapsed,
                    outcome=outcome,
                ),
            )


class FinalizingStreamingResponse(StreamingResponse):
    def __init__(
        self,
        content: AsyncIterator[str],
        *,
        finalizer: Callable[[], Awaitable[None]],
        media_type: str,
        headers: dict[str, str],
    ) -> None:
        super().__init__(content, media_type=media_type, headers=headers)
        self._finalizer = finalizer

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            task: asyncio.Future[None] = asyncio.ensure_future(self._finalizer())
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                raise


def create_app(
    settings: Settings | None = None,
    *,
    runner_factory: RunnerFactory | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    payload_tracer = configure_payload_tracing(
        mode=resolved_settings.trace_payloads,
        path=resolved_settings.trace_payload_file,
    )
    metrics = BridgeMetrics()
    reaper = BackgroundReaper(metrics=metrics)
    resolved_runner_factory = runner_factory or (
        lambda: DroidRunner(
            droid_path=resolved_settings.droid_path,
            workdir=resolved_settings.workdir,
            process_grace_seconds=resolved_settings.process_grace_seconds,
            cleanup_timeout_seconds=resolved_settings.cleanup_timeout_seconds,
            session_init_timeout_seconds=resolved_settings.session_init_timeout_seconds,
            metrics=metrics,
            worktree=resolved_settings.worktree,
            append_system_prompt_file=resolved_settings.append_system_prompt_file,
            rpc_extension=DroidRpcExtension(
                mcp_settle_seconds=resolved_settings.mcp_settle_seconds,
            ),
            reaper=reaper if resolved_settings.detached_cleanup else None,
        )
    )
    bridge_created = int(time.time())
    sessions = SessionRegistry(resolved_settings.max_tracked_sessions)
    quarantine = ModelQuarantine(ttl_seconds=resolved_settings.model_quarantine_seconds)
    admission = AdmissionController(
        max_concurrency=resolved_settings.max_concurrency,
        max_queue_size=resolved_settings.max_queue_size,
        metrics=metrics,
    )
    native_tools = NativeToolRegistry(
        base_url=resolved_settings.native_tool_call_base_url(),
        max_sessions=resolved_settings.max_tracked_sessions,
    )
    pool = WarmSessionPool(
        runner_factory=resolved_runner_factory,
        reaper=reaper,
        size=resolved_settings.warm_session_count(),
        warm_timeout_seconds=min(resolved_settings.timeout_seconds, _WARM_TIMEOUT_CEILING),
        ttl_seconds=resolved_settings.warm_session_ttl_seconds,
        metrics=metrics,
        native_registry=native_tools,
    )
    telemetry = TelemetryReporter(
        metrics=metrics,
        app_version=_BRIDGE_VERSION,
        endpoint=DEFAULT_TELEMETRY_ENDPOINT,
        enabled=resolved_settings.telemetry,
    )

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        pool.start(
            # Warm with the configured effort override so clamped requests
            # reuse the session instead of paying for a retune.
            initial_key=SessionKey(
                model_id=None,
                reasoning_effort=resolved_settings.reasoning_effort,
            ),
        )
        telemetry.start()
        log_info(
            "bridge.started",
            warm_sessions=resolved_settings.warm_session_count(),
            max_concurrency=resolved_settings.max_concurrency,
            detached_cleanup=resolved_settings.detached_cleanup,
            telemetry=telemetry.enabled,
            reasoning_effort=resolved_settings.reasoning_effort,
            trace_payloads=resolved_settings.trace_payloads,
            trace_payload_file=(
                str(resolved_settings.trace_payload_file)
                if resolved_settings.trace_payload_file is not None
                else None
            ),
        )
        try:
            yield
        finally:
            await pool.aclose()
            await reaper.drain()
            await telemetry.close()

    application = FastAPI(
        lifespan=lifespan,
        title="Factory Droid OpenAI Bridge",
        summary="OpenAI-compatible access to Factory Droid.",
        description=(
            "An unofficial compatibility bridge for OpenAI Chat Completions clients. "
            "The bridge runs one isolated Factory Droid session per request."
        ),
        version=_BRIDGE_VERSION,
        license_info={
            "name": "Apache License 2.0",
            "identifier": "Apache-2.0",
        },
        openapi_tags=[
            {
                "name": "Service",
                "description": "Bridge health and service metadata.",
            },
            {
                "name": "OpenAI compatibility",
                "description": "OpenAI-compatible model and chat endpoints.",
            },
            {
                "name": "Factory extensions",
                "description": "Guarded operations for bridge-created Droid sessions.",
            },
        ],
    )
    application.add_middleware(
        RequestSizeLimitMiddleware,
        max_request_bytes=resolved_settings.max_request_bytes,
        max_json_depth=resolved_settings.max_json_depth,
        body_timeout_seconds=resolved_settings.body_timeout_seconds,
        metrics=metrics,
    )
    application.state.admission = admission
    application.state.metrics = metrics
    application.state.sessions = sessions
    application.state.pool = pool
    application.state.reaper = reaper
    application.state.quarantine = quarantine
    application.state.telemetry = telemetry
    application.state.native_tools = native_tools
    if resolved_settings.native_tool_calls:
        # The token in each path is the credential, so the endpoint carries no
        # API-key dependency: only the Droid process this bridge started is
        # ever told the URL.
        application.include_router(build_router(native_tools))

    async def require_auth(credentials: BearerCredentials) -> None:
        expected = resolved_settings.api_key
        if expected is None:
            return
        if credentials is None or not secrets.compare_digest(
            credentials.credentials,
            expected,
        ):
            raise BridgeHTTPError(
                "Invalid API key.",
                status_code=401,
                error_type="authentication_error",
            )

    @application.exception_handler(BridgeHTTPError)
    async def handle_bridge_error(request: Request, exc: BridgeHTTPError) -> JSONResponse:
        request.state.telemetry_error_type = exc.error_type
        return _error_response(
            exc.message,
            exc.status_code,
            exc.error_type,
            headers=exc.headers,
        )

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        request.state.telemetry_error_type = "invalid_request_error"
        return _error_response(
            _validation_message(exc),
            400,
            "invalid_request_error",
        )

    async def run_factory_operation(
        operation: Callable[[DroidRunner, float], Awaitable[FactoryOperationResult]],
        *,
        timeout_seconds: float | None = None,
    ) -> FactoryOperationResult:
        timeout = min(
            timeout_seconds or resolved_settings.timeout_seconds,
            resolved_settings.timeout_seconds,
        )
        deadline = asyncio.get_running_loop().time() + timeout
        try:
            lease = await admission.acquire(deadline)
        except AdmissionRejectedError as exc:
            metrics.increment_overload_rejections()
            raise BridgeHTTPError(
                "Factory Droid request queue is full.",
                status_code=429,
                error_type="rate_limit_error",
                headers={"Retry-After": str(resolved_settings.retry_after_seconds)},
            ) from exc
        except TimeoutError as exc:
            raise BridgeHTTPError(
                f"Factory Droid timed out after {timeout:.1f} seconds.",
                status_code=504,
                error_type="factory_droid_timeout",
            ) from exc

        async with lease:
            try:
                # A budget already spent in the queue is forwarded as-is; the
                # runner's own timeout turns it into a 504.
                remaining = deadline - asyncio.get_running_loop().time()
                runner = resolved_runner_factory()
                return await operation(runner, remaining)
            except RunnerError as exc:
                raise BridgeHTTPError(
                    str(exc),
                    status_code=exc.status_code,
                    error_type=exc.error_type,
                ) from exc

    model_catalog = ModelCatalog(
        ttl_seconds=resolved_settings.model_cache_seconds,
        load=lambda: run_factory_operation(
            lambda runner, remaining: runner.list_models(timeout_seconds=remaining),
            timeout_seconds=min(30.0, resolved_settings.timeout_seconds),
        ),
    )

    def note_runner_failure(model: str, exc: RunnerError) -> None:
        if exc.error_type != "model_not_found":
            return
        if quarantine.record(model, str(exc)):
            metrics.increment_model_quarantines()
            log_warning(
                "models.quarantined",
                model=model,
                ttl_seconds=resolved_settings.model_quarantine_seconds,
            )

    def require_known_session(session_id: str) -> SessionKey:
        if not resolved_settings.session_continuity:
            log_warning("session.rejected", status=400, phase="session_continuity")
            raise BridgeHTTPError(
                "Session continuity is disabled on this bridge.",
                status_code=400,
                error_type="invalid_request_error",
            )
        key = sessions.key(session_id)
        if key is None:
            log_warning("session.rejected", status=404, phase="session_unknown")
            raise BridgeHTTPError(
                "Unknown Factory Droid session. Only sessions created by this "
                "bridge process can be used.",
                status_code=404,
                error_type="session_not_found",
            )
        return key

    def acquire_session_use(session_id: str) -> SessionUseLease:
        lease = sessions.acquire(session_id)
        if lease is None:
            log_warning("session.rejected", status=409, phase="session_busy")
            raise BridgeHTTPError(
                "Factory Droid session is already serving another request.",
                status_code=409,
                error_type="invalid_request_error",
            )
        return lease

    @application.get(
        "/health",
        response_model=HealthResponse,
        tags=["Service"],
        summary="Check bridge health",
    )
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @application.get(
        "/version",
        response_model=VersionResponse,
        tags=["Service"],
        summary="Read the bridge version",
    )
    async def version() -> VersionResponse:
        return VersionResponse(version=application.version)

    @application.get("/metrics", include_in_schema=False)
    async def service_metrics() -> PlainTextResponse:
        return PlainTextResponse(
            metrics.render(),
            media_type="text/plain; version=0.0.4",
        )

    @application.get(
        "/v1/models",
        response_model=ModelListResponse,
        response_model_exclude_none=True,
        dependencies=[Depends(require_auth)],
        responses=MODEL_LIST_RESPONSES,
        tags=["OpenAI compatibility"],
        summary="List available Factory Droid models",
    )
    async def models(response: Response) -> ModelListResponse:
        discovered, degraded = await model_catalog.get()
        if degraded:
            metrics.increment_model_discovery_failures()
            log_warning("models.discovery_degraded", cached=len(discovered))
            response.headers["x-factory-droid-model-discovery"] = "degraded"
        # The Droid catalog also lists models an organization policy blocks, so
        # anything already known to be refused is withheld from clients that
        # build their model picker from this response.
        available = tuple(model for model in discovered if quarantine.allows(model.id))
        withheld = len(discovered) - len(available)
        if not degraded:
            log_debug("models.listed", discovered=len(available), quarantined=withheld or None)
        if withheld:
            response.headers["x-factory-droid-models-quarantined"] = str(withheld)
        return ModelListResponse(
            data=_model_list(
                resolved_settings.model_alias,
                available,
                created=bridge_created,
            )
        )

    @application.get(
        "/v1/models/{model_id}",
        response_model=ModelInfo,
        response_model_exclude_none=True,
        dependencies=[Depends(require_auth)],
        responses=MODEL_RETRIEVE_RESPONSES,
        tags=["OpenAI compatibility"],
        summary="Retrieve one available Factory Droid model",
    )
    async def retrieve_model(model_id: str, response: Response) -> ModelInfo:
        discovered, degraded = await model_catalog.get()
        if degraded:
            metrics.increment_model_discovery_failures()
            log_warning("models.discovery_degraded", cached=len(discovered))
            response.headers["x-factory-droid-model-discovery"] = "degraded"
        available = tuple(model for model in discovered if quarantine.allows(model.id))
        catalog = _model_list(
            resolved_settings.model_alias,
            available,
            created=bridge_created,
        )
        for model in catalog:
            if model.id == model_id:
                return model
        raise BridgeHTTPError(
            f"Model '{model_id}' is not available on this bridge.",
            status_code=404,
            error_type="model_not_found",
        )

    @application.get(
        "/v1/factory/sessions/{session_id}/context",
        response_model=SessionContextResponse,
        dependencies=[Depends(require_auth)],
        responses=FACTORY_OPERATION_RESPONSES,
        tags=["Factory extensions"],
        summary="Read Factory Droid context utilization",
    )
    async def session_context(session_id: str) -> SessionContextResponse:
        require_known_session(session_id)
        async with acquire_session_use(session_id):
            stats, breakdown = await run_factory_operation(
                lambda runner, remaining: runner.get_context(
                    session_id,
                    timeout_seconds=remaining,
                )
            )
        return SessionContextResponse(
            session_id=session_id,
            stats=ContextStatsResponse(
                used=stats.used,
                remaining=stats.remaining,
                limit=stats.limit,
                accuracy=stats.accuracy,
                updated_at=stats.updated_at,
            ),
            breakdown=ContextBreakdownResponse(
                model_id=breakdown.model_id,
                model_display_name=breakdown.model_display_name,
                context_budget=breakdown.context_budget,
                used_tokens=breakdown.used_tokens,
                free_tokens=breakdown.free_tokens,
                categories=[
                    ContextCategoryResponse(
                        name=category.name,
                        tokens=category.tokens,
                        color_key=category.color_key,
                    )
                    for category in breakdown.categories
                ],
            ),
        )

    @application.post(
        "/v1/factory/sessions/{session_id}/compact",
        response_model=CompactSessionResponse,
        dependencies=[Depends(require_auth)],
        responses=FACTORY_OPERATION_RESPONSES,
        tags=["Factory extensions"],
        summary="Compact a Factory Droid session",
    )
    async def compact_session(
        session_id: str,
        payload: CompactSessionRequest,
    ) -> CompactSessionResponse:
        key = require_known_session(session_id)
        async with acquire_session_use(session_id):
            result = await run_factory_operation(
                lambda runner, remaining: runner.compact_session(
                    session_id,
                    custom_instructions=payload.custom_instructions,
                    timeout_seconds=remaining,
                )
            )
        sessions.remember(result.new_session_id, key)
        return CompactSessionResponse(
            session_id=result.new_session_id,
            removed_count=result.removed_count,
        )

    @application.post(
        "/v1/factory/sessions/{session_id}/fork",
        response_model=ForkSessionResponse,
        dependencies=[Depends(require_auth)],
        responses=FACTORY_OPERATION_RESPONSES,
        tags=["Factory extensions"],
        summary="Fork a Factory Droid session",
    )
    async def fork_session(session_id: str) -> ForkSessionResponse:
        key = require_known_session(session_id)
        async with acquire_session_use(session_id):
            new_session_id = await run_factory_operation(
                lambda runner, remaining: runner.fork_session(
                    session_id,
                    timeout_seconds=remaining,
                )
            )
        sessions.remember(new_session_id, key)
        return ForkSessionResponse(session_id=new_session_id)

    @application.patch(
        "/v1/factory/sessions/{session_id}",
        response_model=SessionOperationResponse,
        dependencies=[Depends(require_auth)],
        responses=FACTORY_OPERATION_RESPONSES,
        tags=["Factory extensions"],
        summary="Rename a Factory Droid session",
    )
    async def rename_session(
        session_id: str,
        payload: RenameSessionRequest,
    ) -> SessionOperationResponse:
        require_known_session(session_id)
        async with acquire_session_use(session_id):
            await run_factory_operation(
                lambda runner, remaining: runner.rename_session(
                    session_id,
                    title=payload.title,
                    timeout_seconds=remaining,
                )
            )
        return SessionOperationResponse(session_id=session_id, status="renamed")

    @application.delete(
        "/v1/factory/sessions/{session_id}",
        response_model=SessionOperationResponse,
        dependencies=[Depends(require_auth)],
        responses=FACTORY_OPERATION_RESPONSES,
        tags=["Factory extensions"],
        summary="Close a Factory Droid session",
    )
    async def close_session(session_id: str) -> SessionOperationResponse:
        require_known_session(session_id)
        async with acquire_session_use(session_id):
            await run_factory_operation(
                lambda runner, remaining: runner.close_session(
                    session_id,
                    timeout_seconds=remaining,
                )
            )
        sessions.forget(session_id)
        return SessionOperationResponse(session_id=session_id, status="closed")

    @application.post(
        "/v1/chat/completions",
        dependencies=[Depends(require_auth)],
        response_model=None,
        responses=CHAT_COMPLETION_RESPONSES,
        tags=["OpenAI compatibility"],
        summary="Create a chat completion",
    )
    async def chat_completions(
        payload: ChatCompletionRequest,
        request: Request,
    ) -> JSONResponse | StreamingResponse:
        # Requests rejected before the handler (validation, auth, request size)
        # never reach this line, so they stay reported as "not_applicable".
        request.state.telemetry_mode = "stream" if payload.stream else "non_stream"
        request.state.telemetry_model_family = model_family(payload.model)
        timeout_seconds = min(
            payload.timeout or resolved_settings.timeout_seconds,
            resolved_settings.timeout_seconds,
        )
        request_started_at = asyncio.get_running_loop().time()
        deadline = request_started_at + timeout_seconds
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        timeline = bind_request(request_id)
        log_debug(
            "chat.received",
            model=payload.model,
            stream=bool(payload.stream),
            choices=payload.n,
            messages=len(payload.messages),
            tools=len(payload.tools or ()),
            timeout_s=round(timeout_seconds, 1),
            reasoning_effort=_effective_reasoning_effort(payload, resolved_settings),
            reasoning_effort_overridden=resolved_settings.reasoning_effort is not None,
            continuation=payload.factory_droid_session_id is not None,
        )

        rejection = _validate_options(payload, resolved_settings)
        if rejection is not None:
            request.state.telemetry_error_type = "invalid_request_error"
            log_warning("chat.rejected", status=rejection.status_code, phase="options")
            return rejection

        reasoning_effort = _effective_reasoning_effort(payload, resolved_settings)
        try:
            reasoning_effort = normalize_reasoning_effort(reasoning_effort)
        except RunnerError as exc:
            request.state.telemetry_error_type = exc.error_type
            log_warning("chat.rejected", status=exc.status_code, phase="reasoning_effort")
            return _error_response(str(exc), exc.status_code, exc.error_type)

        requested_key = SessionKey.from_request(
            model=payload.model,
            model_alias=resolved_settings.model_alias,
            reasoning_effort=reasoning_effort,
        )

        # Settled before the transcript is serialized so a mismatch cannot be
        # masked by a payload-size rejection and costs no prompt work.
        session_id: str | None = None
        if payload.factory_droid_session_id is not None:
            session_id = payload.factory_droid_session_id
            if require_known_session(session_id) != requested_key:
                request.state.telemetry_error_type = "invalid_request_error"
                log_warning("chat.rejected", status=400, phase="session_settings")
                return _error_response(
                    "Factory Droid session settings do not match the request. "
                    "Start a new session to switch model or reasoning effort.",
                    400,
                    "invalid_request_error",
                )

        try:
            structured = _prepare_output_format(
                payload,
                max_schema_bytes=resolved_settings.max_tool_schema_bytes,
                max_output_bytes=resolved_settings.max_structured_output_bytes,
                max_json_depth=resolved_settings.max_json_depth,
            )
            plan = build_prompt(
                payload,
                max_messages=resolved_settings.max_messages,
                max_tools=resolved_settings.max_tools,
                max_transcript_bytes=resolved_settings.max_transcript_bytes,
                max_tool_schema_bytes=resolved_settings.max_tool_schema_bytes,
                max_json_depth=resolved_settings.max_json_depth,
                max_tool_calls=(
                    resolved_settings.max_tool_calls if payload.parallel_tool_calls else 1
                ),
                max_attachments=resolved_settings.max_attachments,
                max_attachment_bytes=resolved_settings.max_attachment_bytes,
                continuation=session_id is not None,
                native_tools=resolved_settings.native_tool_calls,
            )
        except RequestTooLargeError as exc:
            metrics.increment_payload_rejections()
            request.state.telemetry_error_type = "invalid_request_error"
            log_warning("chat.rejected", status=413, phase="prompt", reason=str(exc))
            return _error_response(str(exc), 413, "invalid_request_error")
        except ProtocolError as exc:
            request.state.telemetry_error_type = "invalid_request_error"
            log_warning("chat.rejected", status=400, phase="prompt", reason=str(exc))
            return _error_response(str(exc), 400, "invalid_request_error")

        log_debug(
            "chat.prompt_built",
            prompt_bytes=len(plan.prompt.encode("utf-8")),
            allowed_tools=len(plan.allowed_tool_names),
            require_tool_call=plan.require_tool_call,
            images=len(plan.attachments.images),
            documents=len(plan.attachments.documents),
            prompt_ms=timeline.mark("prompt_ms"),
        )
        payload_tracer.trace("chat.prompt", plan.prompt, model=payload.model)

        quarantined = quarantine.reason(payload.model)
        if quarantined is not None:
            request.state.telemetry_error_type = "model_not_found"
            log_warning("chat.rejected", status=404, phase="model", model=payload.model)
            return _error_response(quarantined, 404, "model_not_found")

        features: list[str] = []
        if payload.tools:
            features.append("tools")
        if structured is not None:
            features.append("structured_output")
        if plan.attachments:
            features.append("attachments")
        if reasoning_effort is not None:
            features.append("reasoning_effort")
        if session_id is not None:
            features.append("session_continuity")
        if payload.n > 1:
            features.append("multiple_choices")
        metrics.record_features(tuple(features))

        created = int(time.time())
        run_request = RunRequest(
            prompt=plan.prompt,
            model=payload.model,
            model_alias=resolved_settings.model_alias,
            reasoning_effort=reasoning_effort,
            timeout_seconds=timeout_seconds,
            deadline=deadline,
            images=tuple(image.to_sdk() for image in plan.attachments.images),
            documents=tuple(document.to_sdk() for document in plan.attachments.documents),
            session_id=session_id,
            output_format=structured.payload if structured is not None else None,
        )
        session_use = acquire_session_use(session_id) if session_id is not None else None

        try:
            lease = await admission.acquire(deadline)
        except AdmissionRejectedError:
            if session_use is not None:
                session_use.release()
            metrics.increment_overload_rejections()
            request.state.telemetry_error_type = "rate_limit_error"
            log_warning("chat.rejected", status=429, phase="queue")
            return _error_response(
                "Factory Droid request queue is full.",
                429,
                "rate_limit_error",
                headers={"Retry-After": str(resolved_settings.retry_after_seconds)},
            )
        except TimeoutError:
            if session_use is not None:
                session_use.release()
            request.state.telemetry_error_type = "factory_droid_timeout"
            log_warning(
                "chat.rejected",
                status=504,
                phase="queue",
                queue_ms=timeline.mark("queue_ms"),
            )
            return _error_response(
                f"Factory Droid timed out after {timeout_seconds:.1f} seconds.",
                504,
                "factory_droid_timeout",
            )
        except BaseException:
            if session_use is not None:
                session_use.release()
            raise

        log_debug("chat.admitted", queue_ms=timeline.mark("queue_ms"))

        try:
            runner = resolved_runner_factory()
        except BaseException:
            try:
                await lease.release()
            finally:
                if session_use is not None:
                    session_use.release()
            raise

        native_binding: NativeToolBinding | None = None
        native_catalog = None
        if plan.native_tools:
            native_catalog = native_tools.catalog_identity(to_mcp_tools(plan.native_tools))
            metrics.record_features(("native_tool_calls",))

        warm_session: WarmSession | None = None
        if session_id is None:
            warm_session = pool.acquire(requested_key, native_catalog)
            if warm_session is not None:
                metrics.record_features(("warm_session",))
                run_request = replace(run_request, warm_session=warm_session)

        def release_native_tools() -> None:
            if native_binding is not None:
                native_tools.close(native_binding.token)

        def discard_unused_warm_session() -> None:
            if warm_session is not None and not warm_session.consumed:
                reaper.submit(resolved_runner_factory().discard(warm_session))

        if native_catalog is not None:
            try:
                if warm_session is not None:
                    native_binding = warm_session.native_binding
                    if native_binding is None:
                        raise RuntimeError("native warm session has no catalog binding")
                else:
                    native_binding = native_tools.open_catalog(native_catalog)
            except BaseException:
                discard_unused_warm_session()
                try:
                    await lease.release()
                finally:
                    try:
                        if session_use is not None:
                            session_use.release()
                    finally:
                        release_native_tools()
                raise
            run_request = replace(run_request, native_tools=native_binding)
            log_debug(
                "chat.native_tools_published",
                tools=len(plan.native_tools or ()),
                open_catalogs=len(native_tools),
            )

        def record_repair(
            event: str,
            *,
            dialect: str,
            variant: str | None = None,
        ) -> None:
            metrics.record_features(
                (_tool_call_repair_feature(event, dialect=dialect, variant=variant),)
            )

        def message_parser() -> ToolCallStreamParser:
            return ToolCallStreamParser(
                plan.allowed_tool_names,
                require_tool_call=plan.require_tool_call,
                max_tool_calls=(
                    resolved_settings.max_tool_calls if payload.parallel_tool_calls else 1
                ),
                max_json_depth=resolved_settings.max_json_depth,
                repair_lost_prefix=resolved_settings.repair_lost_prefix,
                parse_message_json=structured is None,
                trace_payload=payload_tracer.trace,
                record_repair=record_repair,
            )

        if payload.stream:
            request.state.stream_outcome = "pending"
            started_stream_session: str | None = None

            def record_started_session(started_id: str) -> None:
                nonlocal started_stream_session
                started_stream_session = started_id

            def record_stream_outcome(outcome: str) -> None:
                request.state.stream_outcome = outcome
                if started_stream_session is not None and outcome in {
                    "success",
                    "truncated",
                    "malformed",
                }:
                    sessions.remember(started_stream_session, requested_key)

            event_stream = _stream_completion(
                request_id=request_id,
                created=created,
                model=payload.model,
                parser=message_parser(),
                runner=runner,
                run_request=run_request,
                lease=lease,
                metrics=metrics,
                request_started_at=request_started_at,
                outcome_callback=record_stream_outcome,
                include_usage=bool(payload.stream_options and payload.stream_options.include_usage),
                stop_sequences=payload.stop_sequences,
                emit_status=payload.factory_droid_status,
                session_callback=record_started_session,
                expose_session=resolved_settings.session_continuity,
                structured=structured,
                failure_callback=note_runner_failure,
                drain_seconds=resolved_settings.tool_call_drain_seconds,
                trace_event=payload_tracer.trace,
            )
            return FinalizingStreamingResponse(
                event_stream,
                media_type="text/event-stream",
                finalizer=lambda: _finalize_stream(
                    event_stream,
                    lease,
                    request,
                    warm_session=run_request.warm_session,
                    reaper=reaper,
                    runner_factory=resolved_runner_factory,
                    session_use=session_use,
                    release_native_tools=release_native_tools,
                ),
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "x-request-id": request_id,
                },
            )

        choices: list[dict[str, Any]] = []
        total_usage = Usage()
        started_session: str | None = None
        try:
            async with lease:
                # Choices run one after another so n completions never exceed the
                # configured Droid process concurrency.
                for index in range(payload.n):
                    choice_runner = runner if index == 0 else resolved_runner_factory()
                    # A warm session serves a single turn, so extra choices run
                    # on their own cold sessions.
                    choice_request = (
                        run_request if index == 0 else replace(run_request, warm_session=None)
                    )
                    result = await _collect_completion_or_disconnect(
                        request=request,
                        runner=choice_runner,
                        run_request=choice_request,
                        parser=message_parser(),
                        metrics=metrics,
                        request_started_at=request_started_at,
                        stop_sequences=payload.stop_sequences,
                        drain_seconds=resolved_settings.tool_call_drain_seconds,
                        trace_event=payload_tracer.trace,
                    )
                    if result is None:
                        request.state.telemetry_error_type = "client_disconnected"
                        return _error_response(
                            "Client disconnected.",
                            499,
                            "client_disconnected",
                        )
                    if result.truncated:
                        request.state.stream_outcome = "truncated"
                    elif result.malformed_note is not None:
                        request.state.stream_outcome = "malformed"
                    if result.session_id is not None:
                        sessions.remember(result.session_id, requested_key)
                        started_session = result.session_id
                    if structured is not None:
                        _validate_structured_output(result.text, structured)
                    total_usage = _add_usage(total_usage, result.usage)
                    choices.append(_choice_dict(result, index))
        except ProtocolError as exc:
            request.state.telemetry_error_type = "factory_protocol_error"
            log_warning(
                "chat.failed",
                status=502,
                error_type="factory_protocol_error",
                reason=str(exc),
            )
            return _error_response(str(exc), 502, "factory_protocol_error")
        except RunnerError as exc:
            request.state.telemetry_error_type = exc.error_type
            log_warning("chat.failed", status=exc.status_code, error_type=exc.error_type)
            note_runner_failure(payload.model, exc)
            return _error_response(str(exc), exc.status_code, exc.error_type)
        finally:
            try:
                discard_unused_warm_session()
            finally:
                try:
                    if session_use is not None:
                        session_use.release()
                finally:
                    release_native_tools()

        log_info(
            "chat.completed",
            status=200,
            model=payload.model,
            stream=False,
            choices=len(choices),
            tool_calls=sum(len(choice["message"].get("tool_calls") or ()) for choice in choices),
            input_tokens=total_usage.input_tokens,
            output_tokens=total_usage.output_tokens,
            **timeline.fields(),
        )

        body: dict[str, Any] = {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": payload.model,
            "choices": choices,
            "usage": _usage_dict(total_usage),
        }
        headers = {"x-request-id": request_id}
        if resolved_settings.session_continuity and started_session is not None:
            body["factory_droid_session_id"] = started_session
            headers["x-factory-droid-session-id"] = started_session

        return JSONResponse(body, headers=headers)

    return application


class BridgeHTTPError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        error_type: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.headers = headers


class CollectedCompletion:
    def __init__(self) -> None:
        self.text_parts: list[str] = []
        self.reasoning_parts: list[str] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.usage = Usage()
        self.completed = False
        self.session_id: str | None = None
        self.stopped = False
        self.truncated = False
        self.malformed_note: str | None = None

    @property
    def text(self) -> str:
        return "".join(self.text_parts)

    @property
    def reasoning(self) -> str:
        return "".join(self.reasoning_parts)


def _event_record(event: RunEvent) -> dict[str, Any]:
    """Describe one SDK event as the JSON the replay fixtures are built from."""
    if isinstance(event, TextDelta):
        return {"kind": "text_delta", "text": event.text}
    if isinstance(event, ReasoningDelta):
        return {"kind": "reasoning_delta", "text": event.text}
    if isinstance(event, UsageUpdate):
        return {"kind": "usage", "usage": _usage_dict(event.usage)}
    if isinstance(event, RunComplete):
        return {"kind": "run_complete", "usage": _usage_dict(event.usage)}
    if isinstance(event, StatusUpdate):
        return {"kind": "status", "state": event.state}
    # The session id identifies a Factory account session and replay never
    # needs it, so SessionStarted is recorded without its payload.
    return {"kind": "session_started"}


async def _run_events(
    runner: DroidRunner,
    run_request: RunRequest,
    *,
    drain_seconds: float,
    saw_tool_call: Callable[[], bool],
    trace_event: Callable[..., None] | None = None,
) -> AsyncGenerator[RunEvent, None]:
    """Yield runner events, bounding the wait once a tool call is complete.

    Droid does not always emit TurnComplete after a client-side tool call, so
    an unbounded ``async for`` stalls the client for the whole backend timeout.
    A complete tool call is already a complete assistant turn: the bridge keeps
    reading only while events arrive within ``drain_seconds``, which leaves
    room for parallel calls the SDK has already queued.
    """
    async with contextlib.aclosing(runner.run(run_request)) as events:
        iterator = events.__aiter__()
        while True:
            try:
                if saw_tool_call():
                    event = await asyncio.wait_for(
                        iterator.__anext__(),
                        timeout=drain_seconds,
                    )
                else:
                    event = await iterator.__anext__()
            except StopAsyncIteration:
                return
            except TimeoutError:
                return
            except RunnerError as exc:
                if saw_tool_call() and exc.error_type == "factory_droid_timeout":
                    # Cancelling the pending event can race the runner's own
                    # deadline, which reports the cancellation as a backend
                    # timeout. The turn already produced a tool call, so the
                    # answer stands instead of turning into a 504.
                    return
                raise
            if trace_event is not None:
                trace_event(
                    "droid.event",
                    json.dumps(_event_record(event), ensure_ascii=False),
                    model=run_request.model,
                )
            yield event


async def _collect_completion(
    *,
    runner: DroidRunner,
    run_request: RunRequest,
    parser: ToolCallStreamParser,
    metrics: BridgeMetrics | None = None,
    request_started_at: float | None = None,
    stop_sequences: tuple[str, ...] = (),
    drain_seconds: float = DEFAULT_TOOL_CALL_DRAIN_SECONDS,
    trace_event: Callable[..., None] | None = None,
) -> CollectedCompletion:
    result = CollectedCompletion()
    stop_buffer = StopSequenceBuffer(stop_sequences)
    observed_ttft = False

    def record_malformed(exc: MalformedToolCallError) -> None:
        result.malformed_note = _malformed_tool_call_note(exc)
        result.completed = True
        log_warning(
            "chat.malformed",
            stream=False,
            error_type="factory_protocol_error",
            reason=str(exc),
            tool_name=exc.tool_name,
            payload_bytes=exc.payload_bytes,
        )

    async with contextlib.aclosing(
        _run_events(
            runner,
            run_request,
            drain_seconds=drain_seconds,
            saw_tool_call=lambda: bool(result.tool_calls),
            trace_event=trace_event,
        )
    ) as events:
        async for event in events:
            if isinstance(event, TextDelta):
                observed_ttft = _observe_ttft(
                    observed_ttft,
                    metrics,
                    request_started_at,
                )
                try:
                    emissions = parser.feed(event.text)
                    _apply_emissions(result, emissions, stop_buffer)
                except MalformedToolCallError as exc:
                    # The parser found a malformed call or transcript fragment,
                    # so continuing would only expose or repeat garbage. A
                    # plain-text note plus finish_reason="stop" prevents client
                    # continuation loops. The call is dropped, never executed.
                    record_malformed(exc)
                    break
                if stop_buffer.triggered:
                    # Leaving the loop closes the runner generator, which
                    # interrupts the Droid turn instead of draining it.
                    result.stopped = True
                    result.completed = True
                    break
            elif isinstance(event, ReasoningDelta):
                observed_ttft = _observe_ttft(
                    observed_ttft,
                    metrics,
                    request_started_at,
                )
                result.reasoning_parts.append(event.text)
            elif isinstance(event, SessionStarted):
                result.session_id = event.session_id
            elif isinstance(event, UsageUpdate):
                result.usage = event.usage
            elif isinstance(event, RunComplete):
                result.usage = event.usage
                result.completed = True
                break
    if result.tool_calls:
        # A complete client tool call settles the turn even when the drain
        # ended before Droid sent a completion event.
        result.completed = True
    if not result.completed:
        raise RunnerError(
            "Factory Droid ended without a completion event.",
            error_type="factory_incomplete_response",
        )
    if not result.stopped:
        if not result.truncated and result.malformed_note is None:
            # A failed parse already settled the turn; asking the parser to
            # finish would re-raise on the same garbage.
            try:
                _apply_emissions(result, parser.finish(), stop_buffer)
            except MalformedToolCallError as exc:
                record_malformed(exc)
            except IncompleteToolCallError as exc:
                # Same contract as the streaming path: a turn that died inside
                # a tool-call payload returns completed output with
                # finish_reason="length" instead of a 502. The partial call is
                # dropped, never executed.
                result.truncated = True
                log_warning(
                    "chat.truncated",
                    stream=False,
                    error_type="factory_protocol_error",
                    reason=str(exc),
                    tool_name=exc.tool_name,
                    payload_bytes=exc.payload_bytes,
                )
        held = stop_buffer.flush()
        if held:
            result.text_parts.append(held)
    return result


async def _collect_completion_or_disconnect(
    *,
    request: Request,
    runner: DroidRunner,
    run_request: RunRequest,
    parser: ToolCallStreamParser,
    metrics: BridgeMetrics | None = None,
    request_started_at: float | None = None,
    stop_sequences: tuple[str, ...] = (),
    drain_seconds: float = DEFAULT_TOOL_CALL_DRAIN_SECONDS,
    trace_event: Callable[..., None] | None = None,
) -> CollectedCompletion | None:
    completion_task = asyncio.create_task(
        _collect_completion(
            runner=runner,
            run_request=run_request,
            parser=parser,
            metrics=metrics,
            request_started_at=request_started_at,
            stop_sequences=stop_sequences,
            drain_seconds=drain_seconds,
            trace_event=trace_event,
        )
    )
    disconnect_task = asyncio.create_task(_wait_for_disconnect(request))
    tasks = (completion_task, disconnect_task)
    try:
        done, _ = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if completion_task in done:
            return await completion_task
        return None
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _wait_for_disconnect(request: Request) -> None:
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return


async def _stream_completion(
    *,
    request_id: str,
    created: int,
    model: str,
    parser: ToolCallStreamParser,
    runner: DroidRunner,
    run_request: RunRequest,
    lease: AdmissionLease,
    metrics: BridgeMetrics | None = None,
    request_started_at: float | None = None,
    outcome_callback: Callable[[str], None] | None = None,
    include_usage: bool = False,
    stop_sequences: tuple[str, ...] = (),
    emit_status: bool = False,
    session_callback: Callable[[str], None] | None = None,
    expose_session: bool = False,
    structured: StructuredOutput | None = None,
    failure_callback: Callable[[str, RunnerError], None] | None = None,
    drain_seconds: float = DEFAULT_TOOL_CALL_DRAIN_SECONDS,
    trace_event: Callable[..., None] | None = None,
) -> AsyncIterator[str]:
    usage = Usage()
    completed = False
    saw_tool_call = False
    saw_text = False
    observed_ttft = False
    outcome = "success"
    stop_buffer = StopSequenceBuffer(stop_sequences)
    tool_call_index = 0
    structured_buffer = (
        StructuredOutputBuffer(structured.max_bytes) if structured is not None else None
    )

    def text_chunk(text: str) -> str | None:
        if structured_buffer is not None:
            structured_buffer.append(text)
            return None
        return _sse(
            _chunk_for_emission(
                request_id,
                created,
                model,
                TextEmission(text),
                include_usage=include_usage,
            )
        )

    async with lease:
        yield _sse(
            _chunk(
                request_id,
                created,
                model,
                delta={"role": "assistant", "content": ""},
                include_usage=include_usage,
            )
        )
        try:
            async with contextlib.aclosing(
                _run_events(
                    runner,
                    run_request,
                    drain_seconds=drain_seconds,
                    saw_tool_call=lambda: saw_tool_call,
                    trace_event=trace_event,
                )
            ) as events:
                async for event in events:
                    if isinstance(event, TextDelta):
                        observed_ttft = _observe_ttft(
                            observed_ttft,
                            metrics,
                            request_started_at,
                        )
                        for emission in parser.feed(event.text):
                            if isinstance(emission, ToolCallEmission):
                                saw_tool_call = True
                                chunk = _chunk_for_emission(
                                    request_id,
                                    created,
                                    model,
                                    emission,
                                    include_usage=include_usage,
                                    tool_call_index=tool_call_index,
                                )
                                tool_call_index += 1
                                yield _sse(chunk)
                                continue
                            text = stop_buffer.feed(emission.text)
                            if text:
                                chunk_payload = text_chunk(text)
                                if chunk_payload is not None:
                                    saw_text = True
                                    yield chunk_payload
                        if stop_buffer.triggered:
                            # Closing the runner generator interrupts the Droid
                            # turn instead of draining output nobody will read.
                            completed = True
                            break
                    elif isinstance(event, SessionStarted):
                        if session_callback is not None:
                            session_callback(event.session_id)
                        if expose_session:
                            yield _sse(
                                _chunk(
                                    request_id,
                                    created,
                                    model,
                                    delta={"factory_droid_session_id": event.session_id},
                                    include_usage=include_usage,
                                )
                            )
                    elif isinstance(event, StatusUpdate):
                        if emit_status:
                            yield _sse(
                                _chunk(
                                    request_id,
                                    created,
                                    model,
                                    delta={"factory_droid_status": event.state},
                                    include_usage=include_usage,
                                )
                            )
                    elif isinstance(event, ReasoningDelta):
                        observed_ttft = _observe_ttft(
                            observed_ttft,
                            metrics,
                            request_started_at,
                        )
                        yield _sse(
                            _chunk(
                                request_id,
                                created,
                                model,
                                delta={
                                    "reasoning": event.text,
                                    "reasoning_content": event.text,
                                },
                                include_usage=include_usage,
                            )
                        )
                    elif isinstance(event, UsageUpdate):
                        usage = event.usage
                    elif isinstance(event, RunComplete):
                        usage = event.usage
                        completed = True
                        break

            if saw_tool_call:
                # A complete client tool call settles the turn even when the
                # drain ended before Droid sent a completion event.
                completed = True
            if not completed:
                raise RunnerError(
                    "Factory Droid ended without a completion event.",
                    error_type="factory_incomplete_response",
                )
            if not stop_buffer.triggered:
                for emission in parser.finish():
                    if isinstance(emission, ToolCallEmission):
                        saw_tool_call = True
                        yield _sse(
                            _chunk_for_emission(
                                request_id,
                                created,
                                model,
                                emission,
                                include_usage=include_usage,
                                tool_call_index=tool_call_index,
                            )
                        )
                        tool_call_index += 1
                        continue
                    text = stop_buffer.feed(emission.text)
                    if text:
                        chunk_payload = text_chunk(text)
                        if chunk_payload is not None:
                            saw_text = True
                            yield chunk_payload
                held = stop_buffer.flush()
                if held:
                    chunk_payload = text_chunk(held)
                    if chunk_payload is not None:
                        saw_text = True
                        yield chunk_payload
            if structured is not None and structured_buffer is not None:
                structured_text = structured_buffer.text()
                _validate_structured_output(structured_text, structured)
                yield _sse(
                    _chunk_for_emission(
                        request_id,
                        created,
                        model,
                        TextEmission(structured_text),
                        include_usage=include_usage,
                    )
                )
            yield _sse(
                _chunk(
                    request_id,
                    created,
                    model,
                    delta={},
                    finish_reason="tool_calls" if saw_tool_call else "stop",
                    include_usage=include_usage,
                )
            )
            if include_usage:
                yield _sse(_usage_chunk(request_id, created, model, usage))
        except IncompleteToolCallError as exc:
            # The turn died inside a tool-call payload. Surface it the way
            # OpenAI surfaces a length cutoff - completed output plus
            # finish_reason="length" - so clients continue instead of
            # blind-retrying an unrecoverable request. The partial call is
            # dropped, never executed.
            outcome = "truncated"
            log_warning(
                "chat.truncated",
                stream=True,
                error_type="factory_protocol_error",
                reason=str(exc),
                tool_name=exc.tool_name,
                payload_bytes=exc.payload_bytes,
            )
            held = stop_buffer.flush()
            if held:
                chunk_payload = text_chunk(held)
                if chunk_payload is not None:
                    yield chunk_payload
            yield _sse(
                _chunk(
                    request_id,
                    created,
                    model,
                    delta={},
                    # A call already on the wire outranks the truncated payload
                    # that followed it, so the client runs it instead of asking
                    # the model to continue.
                    finish_reason="tool_calls" if saw_tool_call else "length",
                    include_usage=include_usage,
                )
            )
            if include_usage:
                yield _sse(_usage_chunk(request_id, created, model, usage))
        except MalformedToolCallError as exc:
            # The parser found a malformed call or transcript fragment. A
            # length finish would make the client ask the model to continue,
            # exposing or repeating garbage. Return a plain-text note with
            # finish_reason="stop"; the call is dropped, never executed.
            outcome = "malformed"
            log_warning(
                "chat.malformed",
                stream=True,
                error_type="factory_protocol_error",
                reason=str(exc),
                tool_name=exc.tool_name,
                payload_bytes=exc.payload_bytes,
            )
            held = stop_buffer.flush()
            if held:
                chunk_payload = text_chunk(held)
                if chunk_payload is not None:
                    saw_text = True
                    yield chunk_payload
            note = _malformed_tool_call_note(exc)
            note_chunk = text_chunk(f"\n\n{note}" if saw_text else note)
            if note_chunk is not None:
                yield note_chunk
            yield _sse(
                _chunk(
                    request_id,
                    created,
                    model,
                    delta={},
                    finish_reason="tool_calls" if saw_tool_call else "stop",
                    include_usage=include_usage,
                )
            )
            if include_usage:
                yield _sse(_usage_chunk(request_id, created, model, usage))
        except ProtocolError as exc:
            outcome = "error"
            log_warning(
                "chat.failed",
                stream=True,
                error_type="factory_protocol_error",
                reason=str(exc),
            )
            yield _sse(_error_body(str(exc), "factory_protocol_error"))
        except RunnerError as exc:
            outcome = "timeout" if exc.error_type == "factory_droid_timeout" else "error"
            log_warning("chat.failed", stream=True, error_type=exc.error_type)
            if failure_callback is not None:
                failure_callback(model, exc)
            yield _sse(_error_body(str(exc), exc.error_type))
        except asyncio.CancelledError:
            log_warning("chat.cancelled", stream=True)
            if outcome_callback is not None:
                outcome_callback("cancelled")
            raise
        if outcome_callback is not None:
            outcome_callback(outcome)
        _log_stream_outcome(outcome, model=model, usage=usage, tool_calls=tool_call_index)
        yield "data: [DONE]\n\n"


def _log_stream_outcome(
    outcome: str,
    *,
    model: str,
    usage: Usage,
    tool_calls: int,
) -> None:
    timeline = current_timeline()
    log_info(
        "chat.completed",
        outcome=outcome,
        model=model,
        stream=True,
        tool_calls=tool_calls,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        **(timeline.fields() if timeline is not None else {}),
    )


async def _finalize_stream(
    stream: AsyncIterator[str],
    lease: AdmissionLease,
    request: Request,
    *,
    warm_session: WarmSession | None = None,
    reaper: BackgroundReaper | None = None,
    runner_factory: RunnerFactory | None = None,
    session_use: SessionUseLease | None = None,
    release_native_tools: Callable[[], None] | None = None,
) -> None:
    close = getattr(stream, "aclose", None)
    try:
        if callable(close):
            await close()
    finally:
        if getattr(request.state, "stream_outcome", None) == "pending":
            request.state.stream_outcome = "cancelled"
        if (
            warm_session is not None
            and not warm_session.consumed
            and reaper is not None
            and runner_factory is not None
        ):
            # The stream was discarded before the run started, so nothing else
            # will tear this session down.
            reaper.submit(runner_factory().discard(warm_session))
        try:
            await lease.release()
        finally:
            try:
                if session_use is not None:
                    session_use.release()
            finally:
                if release_native_tools is not None:
                    release_native_tools()


def _apply_emissions(
    result: CollectedCompletion,
    emissions: Sequence[ProtocolEmission],
    stop_buffer: StopSequenceBuffer | None = None,
) -> None:
    for emission in emissions:
        if isinstance(emission, TextEmission):
            text = emission.text if stop_buffer is None else stop_buffer.feed(emission.text)
            if text:
                result.text_parts.append(text)
        else:
            result.tool_calls.append(_tool_call_dict(emission))


def _chunk_for_emission(
    request_id: str,
    created: int,
    model: str,
    emission: ProtocolEmission,
    *,
    include_usage: bool = False,
    tool_call_index: int = 0,
) -> dict[str, Any]:
    if isinstance(emission, TextEmission):
        return _chunk(
            request_id,
            created,
            model,
            delta={"content": emission.text},
            include_usage=include_usage,
        )
    return _chunk(
        request_id,
        created,
        model,
        delta={"tool_calls": [{"index": tool_call_index, **_tool_call_dict(emission)}]},
        include_usage=include_usage,
    )


def _tool_call_dict(emission: ToolCallEmission) -> dict[str, Any]:
    return {
        "id": emission.id,
        "type": "function",
        "function": {
            "name": emission.name,
            "arguments": emission.arguments,
        },
    }


def _chunk(
    request_id: str,
    created: int,
    model: str,
    *,
    delta: dict[str, Any],
    finish_reason: str | None = None,
    include_usage: bool = False,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    if include_usage:
        body["usage"] = None
    return body


def _usage_chunk(
    request_id: str,
    created: int,
    model: str,
    usage: Usage,
) -> dict[str, Any]:
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": _usage_dict(usage),
    }


def _effective_reasoning_effort(
    payload: ChatCompletionRequest,
    settings: Settings,
) -> str | None:
    """A configured effort overrides whatever effort the request carries.

    Agent frameworks pin their own value (often ``medium``), so the operator
    setting has to win for the bridge to hold a chosen effort.
    """
    return (
        settings.reasoning_effort
        or payload.factory_droid_reasoning_effort
        or payload.reasoning_effort
    )


def _validate_options(
    payload: ChatCompletionRequest,
    settings: Settings,
) -> JSONResponse | None:
    if payload.n > settings.max_choices:
        return _error_response(
            f"n must be at most {settings.max_choices} on this bridge.",
            400,
            "invalid_request_error",
        )
    if payload.n > 1 and payload.stream:
        return _error_response(
            "n greater than 1 is not supported together with stream=true.",
            400,
            "invalid_request_error",
        )
    if payload.n > 1 and payload.factory_droid_session_id is not None:
        return _error_response(
            "n greater than 1 cannot continue an existing Factory Droid session.",
            400,
            "invalid_request_error",
        )
    if len(payload.stop_sequences) > settings.max_stop_sequences:
        return _error_response(
            f"stop accepts at most {settings.max_stop_sequences} sequences.",
            400,
            "invalid_request_error",
        )
    if payload.response_format is not None and payload.stop_sequences:
        return _error_response(
            "stop is not supported together with response_format.",
            400,
            "invalid_request_error",
        )
    if payload.response_format is not None and payload.tools and payload.tool_choice != "none":
        return _error_response(
            "tools are not supported together with response_format on this bridge.",
            400,
            "invalid_request_error",
        )
    return None


def _model_list(
    alias: str,
    discovered: tuple[DroidModel, ...],
    *,
    created: int,
) -> list[ModelInfo]:
    models = [
        ModelInfo(
            id=alias,
            created=created,
            owned_by="factory",
        )
    ]
    seen = {alias}
    for model in discovered:
        if model.id in seen:
            continue
        seen.add(model.id)
        models.append(
            ModelInfo(
                id=model.id,
                created=created,
                owned_by=model.provider,
                factory_droid_display_name=model.display_name,
                factory_droid_supported_reasoning_efforts=list(model.supported_reasoning_efforts),
                factory_droid_default_reasoning_effort=model.default_reasoning_effort,
                factory_droid_supports_images=model.supports_images,
                factory_droid_supports_pdfs=model.supports_pdfs,
            )
        )
    return models


def _prepare_output_format(
    payload: ChatCompletionRequest,
    *,
    max_schema_bytes: int,
    max_output_bytes: int,
    max_json_depth: int,
) -> StructuredOutput | None:
    response_format = payload.response_format
    if response_format is None:
        return None
    if response_format.type == "json_object":
        schema: dict[str, Any] = {
            "type": "object",
            "additionalProperties": True,
        }
    else:
        schema = response_format.json_schema.schema_
        if schema.get("type") != "object":
            raise ProtocolError("response_format JSON schema must have type 'object'")
    serialized = json.dumps(
        schema,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    if len(serialized.encode("utf-8")) > max_schema_bytes:
        raise RequestTooLargeError(
            f"response_format schema exceeds maximum of {max_schema_bytes} bytes"
        )
    _reject_remote_schema_refs(schema)
    validator_class = validator_for(schema)
    try:
        validator_class.check_schema(schema)
    except SchemaError as exc:
        raise ProtocolError(
            f"response_format contains an invalid JSON schema: {exc.message}"
        ) from exc
    return StructuredOutput(
        payload={"type": "json_schema", "schema": schema},
        validator=validator_class(schema),
        max_bytes=max_output_bytes,
        max_depth=max_json_depth,
    )


def _validate_structured_output(text: str, structured: StructuredOutput) -> None:
    if len(text.encode("utf-8")) > structured.max_bytes:
        raise ProtocolError(
            f"Factory Droid structured output exceeds maximum of {structured.max_bytes} bytes"
        )
    try:
        value = parse_strict_json(text)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ProtocolError("Factory Droid returned invalid structured JSON output") from exc
    if json_depth_exceeds(value, structured.max_depth):
        raise ProtocolError(
            f"Factory Droid structured output exceeds maximum JSON depth of {structured.max_depth}"
        )
    try:
        structured.validator.validate(value)
    except JsonSchemaValidationError as exc:
        raise ProtocolError(
            f"Factory Droid structured output violated the requested schema: {exc.message}"
        ) from exc


def _reject_remote_schema_refs(value: Any) -> None:
    if isinstance(value, dict):
        for keyword in ("$ref", "$dynamicRef", "$recursiveRef"):
            reference = value.get(keyword)
            if isinstance(reference, str) and not reference.startswith("#"):
                raise ProtocolError("response_format does not allow remote JSON schema references")
        for item in value.values():
            _reject_remote_schema_refs(item)
    elif isinstance(value, list):
        for item in value:
            _reject_remote_schema_refs(item)


def _malformed_tool_call_note(exc: MalformedToolCallError) -> str:
    """Client-readable explanation for a dropped malformed call.

    The note becomes assistant content, so it must not contain tool-call JSON
    that clients may render or models may copy into the next turn.
    """
    tool = f' for tool "{exc.tool_name}"' if exc.tool_name else ""
    return (
        f"[bridge notice: dropped a malformed tool call{tool}. "
        "Retry the request if a tool call is still needed.]"
    )


def _choice_dict(result: CollectedCompletion, index: int) -> dict[str, Any]:
    content = result.text
    if result.malformed_note is not None:
        content = f"{content}\n\n{result.malformed_note}" if content else result.malformed_note
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content or None,
    }
    if result.reasoning:
        message["reasoning"] = result.reasoning
        message["reasoning_content"] = result.reasoning
    if result.tool_calls:
        message["tool_calls"] = result.tool_calls
    finish_reason = "stop"
    if result.tool_calls:
        # A captured call outranks a truncated payload that follows it: with
        # "length" the client asks the model to continue instead of running
        # the call it already received.
        finish_reason = "tool_calls"
    elif result.truncated:
        finish_reason = "length"
    return {
        "index": index,
        "message": message,
        "finish_reason": finish_reason,
    }


def _add_usage(left: Usage, right: Usage) -> Usage:
    return Usage(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        cache_read_tokens=left.cache_read_tokens + right.cache_read_tokens,
        cache_write_tokens=left.cache_write_tokens + right.cache_write_tokens,
    )


def _usage_dict(usage: Usage) -> dict[str, Any]:
    return {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens + usage.output_tokens,
        "prompt_tokens_details": {
            "cached_tokens": usage.cache_read_tokens,
        },
    }


def _sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _error_body(message: str, error_type: str) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": None,
        }
    }


def _error_response(
    message: str,
    status_code: int,
    error_type: str,
    *,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        _error_body(message, error_type),
        status_code=status_code,
        headers=headers,
    )


def _content_length(scope: Scope) -> int | None:
    for name, value in scope.get("headers", []):
        if name.lower() != b"content-length":
            continue
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    return None


def _set_request_state(scope: Scope, **values: object) -> None:
    state = scope.get("state")
    if not isinstance(state, dict):
        state = {}
        scope["state"] = state
    state.update(values)


def _request_state(scope: Scope) -> dict[str, object]:
    state = scope.get("state")
    return state if isinstance(state, dict) else {}


def _payload_size_bucket(size: int) -> str:
    if size <= 0:
        return "empty"
    for upper, bucket in _PAYLOAD_SIZE_BUCKETS:
        if size < upper:
            return bucket
    return "gt_1mb"


def _latency_bucket(seconds: float) -> str:
    for upper, bucket in _LATENCY_BUCKETS:
        if seconds < upper:
            return bucket
    return "gt_30s"


def _json_content_type(scope: Scope) -> bool:
    for name, value in scope.get("headers", []):
        if name.lower() != b"content-type":
            continue
        return bool(value.lower().strip().startswith(b"application/json"))
    return False


async def _read_limited_body(
    receive: Receive,
    *,
    max_request_bytes: int,
    max_json_depth: int,
) -> bytes:
    body_buffer = bytearray()
    depth_tracker = _JsonDepthTracker(max_json_depth)
    more_body = True
    while more_body:
        message = await receive()
        if message["type"] == "http.disconnect":
            raise _ClientDisconnectedError
        body = message.get("body", b"")
        if len(body_buffer) + len(body) > max_request_bytes:
            raise _RequestPayloadLimitError("Request body exceeds configured size limit.")
        depth_tracker.feed(body)
        body_buffer.extend(body)
        more_body = bool(message.get("more_body", False))
    return bytes(body_buffer)


async def _send_payload_too_large(
    scope: Scope,
    receive: Receive,
    send: Send,
    message: str,
) -> None:
    await _send_asgi_error(
        scope,
        receive,
        send,
        message=message,
        status_code=413,
        error_type="invalid_request_error",
    )


async def _send_asgi_error(
    scope: Scope,
    receive: Receive,
    send: Send,
    *,
    message: str,
    status_code: int,
    error_type: str,
) -> None:
    response = _error_response(
        message,
        status_code,
        error_type,
    )
    await response(scope, receive, send)


def _request_outcome(status_code: int, scope: Scope) -> str:
    state = scope.get("state")
    if isinstance(state, dict):
        stream_outcome = state.get("stream_outcome")
        if stream_outcome in {"success", "error", "timeout", "cancelled", "truncated", "malformed"}:
            return str(stream_outcome)
    if status_code < 400:
        return "success"
    if status_code == 499:
        return "cancelled"
    if status_code == 413:
        return "payload_too_large"
    if status_code == 429:
        return "overloaded"
    if status_code in {408, 504}:
        return "timeout"
    return "error"


def _request_route(scope: Scope) -> str:
    path = str(scope.get("path", ""))
    if path == "/health":
        return "health"
    if path == "/version":
        return "version"
    if path == "/metrics":
        return "metrics"
    if path == "/v1/models" or path.startswith("/v1/models/"):
        return "models"
    if path == "/v1/chat/completions":
        return "chat_completions"
    if path.startswith("/v1/factory/sessions/"):
        return "session_operation"
    return "other"


def _request_mode(scope: Scope) -> str:
    mode = _request_state(scope).get("telemetry_mode")
    if mode in {"stream", "non_stream"}:
        return str(mode)
    return "not_applicable"


def _telemetry_error_category(
    scope: Scope,
    *,
    status_code: int,
    outcome: str,
) -> str:
    error_type = _request_state(scope).get("telemetry_error_type")
    if isinstance(error_type, str):
        categories = {
            "authentication_error": "authentication",
            "client_disconnected": "cancelled",
            "factory_droid_timeout": "timeout",
            "factory_incomplete_response": "incomplete_response",
            "factory_protocol_error": "protocol",
            "factory_droid_sdk_error": "sdk",
            "invalid_request_error": "invalid_request",
            "model_not_found": "model_not_found",
            "rate_limit_error": "rate_limited",
            "session_not_found": "session_not_found",
        }
        return categories.get(error_type, "other")
    if outcome == "cancelled":
        return "cancelled"
    if outcome in {"malformed", "truncated"}:
        return "protocol"
    if outcome == "timeout":
        return "timeout"
    if outcome == "error" and status_code < 400:
        return "stream_error"
    if status_code == 413:
        return "payload_too_large"
    if status_code == 429:
        return "rate_limited"
    if status_code in {408, 504}:
        return "timeout"
    if status_code == 401:
        return "authentication"
    if status_code == 404:
        return "not_found"
    if 400 <= status_code < 500:
        return "invalid_request"
    if status_code >= 500:
        return "server_error"
    return "other"


def _request_telemetry_features(
    scope: Scope,
    *,
    status_code: int,
    seconds: float,
    outcome: str,
) -> tuple[str, ...]:
    route = _request_route(scope)
    state = _request_state(scope)
    features = [
        f"request_latency:{route}:{_latency_bucket(max(0.0, seconds))}",
        f"request_payload:{route}:{state.get('payload_size_bucket', 'unknown')}",
    ]
    model_family = state.get("telemetry_model_family")
    if route == "chat_completions" and isinstance(model_family, str):
        features.append(f"model_family:{model_family}")
    if outcome != "success":
        error_category = _telemetry_error_category(
            scope,
            status_code=status_code,
            outcome=outcome,
        )
        features.append(f"request_error:{route}:{error_category}")
    return tuple(features)


_TOOL_CALL_REPAIR_FEATURES = {
    "tool_call.repaired": "tool_repair",
    "tool_call.unparsed": "tool_unparsed",
    "tool_call.over_limit": "tool_over_limit",
}


def _tool_call_repair_feature(
    event: str,
    *,
    dialect: str,
    variant: str | None = None,
) -> str:
    """Names one tool-call repair outcome as a telemetry feature.

    Both halves come from marker-dialect and decoder constants, so the label
    space stays closed and carries no payload text, tool name or model name.
    """
    prefix = _TOOL_CALL_REPAIR_FEATURES.get(event, event.replace(".", "_"))
    return f"{prefix}:{dialect}" if variant is None else f"{prefix}:{dialect}:{variant}"


def _observe_ttft(
    observed: bool,
    metrics: BridgeMetrics | None,
    request_started_at: float | None,
) -> bool:
    if observed:
        return True
    if metrics is not None and request_started_at is not None:
        metrics.observe_ttft(asyncio.get_running_loop().time() - request_started_at)
    timeline = current_timeline()
    if timeline is not None:
        log_debug("chat.first_token", ttft_ms=timeline.since_start("ttft_ms"))
    return True


def _validation_message(exc: RequestValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "Invalid request."
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", ()))
    message = str(first.get("msg", "Invalid request."))
    return f"{location}: {message}" if location else message


app = create_app()
