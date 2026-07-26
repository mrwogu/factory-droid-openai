from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import Depends, FastAPI, Request, Security
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from factory_droid_openai.config import Settings
from factory_droid_openai.metrics import BridgeMetrics
from factory_droid_openai.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    ModelListResponse,
)
from factory_droid_openai.protocol import (
    ProtocolEmission,
    ProtocolError,
    RequestTooLargeError,
    TextEmission,
    ToolCallEmission,
    ToolCallStreamParser,
    build_prompt,
)
from factory_droid_openai.runner import (
    DroidRunner,
    ReasoningDelta,
    RunComplete,
    RunnerError,
    RunRequest,
    TextDelta,
    Usage,
    UsageUpdate,
    normalize_reasoning_effort,
)

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

RunnerFactory = Callable[[], DroidRunner]
BearerCredentials = Annotated[
    HTTPAuthorizationCredentials | None,
    Security(HTTPBearer(auto_error=False)),
]

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


class AdmissionRejectedError(RuntimeError):
    pass


class AdmissionLease:
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


class _JsonDepthTracker:
    def __init__(self, max_depth: int) -> None:
        self._max_depth = max_depth
        self._depth = 0
        self._in_string = False
        self._escaped = False

    def feed(self, data: bytes) -> None:
        for value in data:
            if self._in_string:
                if self._escaped:
                    self._escaped = False
                elif value == ord("\\"):
                    self._escaped = True
                elif value == ord('"'):
                    self._in_string = False
                continue
            if value == ord('"'):
                self._in_string = True
            elif value in (ord("{"), ord("[")):
                self._depth += 1
                if self._depth > self._max_depth:
                    raise _RequestPayloadLimitError("Request JSON exceeds configured depth limit.")
            elif value in (ord("}"), ord("]")):
                self._depth = max(0, self._depth - 1)


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
        if scope["type"] != "http" or scope.get("path") != "/v1/chat/completions":
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
            self._metrics.record_request(
                _request_outcome(status_code, scope),
                status_code,
                time.perf_counter() - started,
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
    metrics = BridgeMetrics()
    resolved_runner_factory = runner_factory or (
        lambda: DroidRunner(
            droid_path=resolved_settings.droid_path,
            workdir=resolved_settings.workdir,
            process_grace_seconds=resolved_settings.process_grace_seconds,
            cleanup_timeout_seconds=resolved_settings.cleanup_timeout_seconds,
            metrics=metrics,
        )
    )
    admission = AdmissionController(
        max_concurrency=resolved_settings.max_concurrency,
        max_queue_size=resolved_settings.max_queue_size,
        metrics=metrics,
    )
    application = FastAPI(
        title="Factory Droid OpenAI Bridge",
        summary="OpenAI-compatible access to Factory Droid.",
        description=(
            "An unofficial compatibility bridge for OpenAI Chat Completions clients. "
            "The bridge runs one isolated Factory Droid session per request."
        ),
        version="1.0.0",  # x-release-please-version
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
    async def handle_bridge_error(_request: Request, exc: BridgeHTTPError) -> JSONResponse:
        return _error_response(exc.message, exc.status_code, exc.error_type)

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return _error_response(
            _validation_message(exc),
            400,
            "invalid_request_error",
        )

    @application.get(
        "/health",
        response_model=HealthResponse,
        tags=["Service"],
        summary="Check bridge health",
    )
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @application.get("/metrics", include_in_schema=False)
    async def service_metrics() -> PlainTextResponse:
        return PlainTextResponse(
            metrics.render(),
            media_type="text/plain; version=0.0.4",
        )

    @application.get(
        "/v1/models",
        response_model=ModelListResponse,
        dependencies=[Depends(require_auth)],
        responses={
            "4XX": {
                "model": ErrorResponse,
                "description": "Bearer authentication failure.",
            }
        },
        tags=["OpenAI compatibility"],
        summary="List the bridge model alias",
    )
    async def models() -> ModelListResponse:
        return ModelListResponse(
            data=[
                ModelInfo(
                    id=resolved_settings.model_alias,
                    created=0,
                    owned_by="factory",
                )
            ]
        )

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
        timeout_seconds = min(
            payload.timeout or resolved_settings.timeout_seconds,
            resolved_settings.timeout_seconds,
        )
        request_started_at = asyncio.get_running_loop().time()
        deadline = request_started_at + timeout_seconds
        try:
            plan = build_prompt(
                payload,
                max_messages=resolved_settings.max_messages,
                max_tools=resolved_settings.max_tools,
                max_transcript_bytes=resolved_settings.max_transcript_bytes,
                max_tool_schema_bytes=resolved_settings.max_tool_schema_bytes,
                max_json_depth=resolved_settings.max_json_depth,
            )
        except RequestTooLargeError as exc:
            metrics.increment_payload_rejections()
            return _error_response(str(exc), 413, "invalid_request_error")
        except ProtocolError as exc:
            return _error_response(str(exc), 400, "invalid_request_error")

        reasoning_effort = payload.factory_droid_reasoning_effort or payload.reasoning_effort
        try:
            reasoning_effort = normalize_reasoning_effort(reasoning_effort)
        except RunnerError as exc:
            return _error_response(str(exc), exc.status_code, exc.error_type)

        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        run_request = RunRequest(
            prompt=plan.prompt,
            model=payload.model,
            model_alias=resolved_settings.model_alias,
            reasoning_effort=reasoning_effort,
            timeout_seconds=timeout_seconds,
            deadline=deadline,
        )

        try:
            lease = await admission.acquire(deadline)
        except AdmissionRejectedError:
            metrics.increment_overload_rejections()
            return _error_response(
                "Factory Droid request queue is full.",
                429,
                "rate_limit_error",
                headers={"Retry-After": str(resolved_settings.retry_after_seconds)},
            )
        except TimeoutError:
            return _error_response(
                f"Factory Droid timed out after {timeout_seconds:.1f} seconds.",
                504,
                "factory_droid_timeout",
            )

        try:
            runner = resolved_runner_factory()
        except BaseException:
            await lease.release()
            raise

        if payload.stream:
            request.state.stream_outcome = "pending"
            event_stream = _stream_completion(
                request_id=request_id,
                created=created,
                model=payload.model,
                parser=ToolCallStreamParser(
                    plan.allowed_tool_names,
                    require_tool_call=plan.require_tool_call,
                ),
                runner=runner,
                run_request=run_request,
                lease=lease,
                metrics=metrics,
                request_started_at=request_started_at,
                outcome_callback=lambda outcome: setattr(
                    request.state,
                    "stream_outcome",
                    outcome,
                ),
                include_usage=bool(payload.stream_options and payload.stream_options.include_usage),
            )
            return FinalizingStreamingResponse(
                event_stream,
                media_type="text/event-stream",
                finalizer=lambda: _finalize_stream(
                    event_stream,
                    lease,
                    request,
                ),
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "x-request-id": request_id,
                },
            )

        try:
            async with lease:
                result = await _collect_completion_or_disconnect(
                    request=request,
                    runner=runner,
                    run_request=run_request,
                    parser=ToolCallStreamParser(
                        plan.allowed_tool_names,
                        require_tool_call=plan.require_tool_call,
                    ),
                    metrics=metrics,
                    request_started_at=request_started_at,
                )
                if result is None:
                    return _error_response(
                        "Client disconnected.",
                        499,
                        "client_disconnected",
                    )
        except ProtocolError as exc:
            return _error_response(str(exc), 502, "factory_protocol_error")
        except RunnerError as exc:
            return _error_response(str(exc), exc.status_code, exc.error_type)

        message: dict[str, Any] = {
            "role": "assistant",
            "content": result.text or None,
        }
        if result.reasoning:
            message["reasoning"] = result.reasoning
            message["reasoning_content"] = result.reasoning
        if result.tool_calls:
            message["tool_calls"] = result.tool_calls

        return JSONResponse(
            {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": payload.model,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": ("tool_calls" if result.tool_calls else "stop"),
                    }
                ],
                "usage": _usage_dict(result.usage),
            },
            headers={"x-request-id": request_id},
        )

    return application


class BridgeHTTPError(Exception):
    def __init__(self, message: str, *, status_code: int, error_type: str) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type


class CollectedCompletion:
    def __init__(self) -> None:
        self.text_parts: list[str] = []
        self.reasoning_parts: list[str] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.usage = Usage()
        self.completed = False

    @property
    def text(self) -> str:
        return "".join(self.text_parts)

    @property
    def reasoning(self) -> str:
        return "".join(self.reasoning_parts)


async def _collect_completion(
    *,
    runner: DroidRunner,
    run_request: RunRequest,
    parser: ToolCallStreamParser,
    metrics: BridgeMetrics | None = None,
    request_started_at: float | None = None,
) -> CollectedCompletion:
    result = CollectedCompletion()
    observed_ttft = False
    async with contextlib.aclosing(runner.run(run_request)) as events:
        async for event in events:
            if isinstance(event, TextDelta):
                observed_ttft = _observe_ttft(
                    observed_ttft,
                    metrics,
                    request_started_at,
                )
                _apply_emissions(result, parser.feed(event.text))
            elif isinstance(event, ReasoningDelta):
                observed_ttft = _observe_ttft(
                    observed_ttft,
                    metrics,
                    request_started_at,
                )
                result.reasoning_parts.append(event.text)
            elif isinstance(event, UsageUpdate):
                result.usage = event.usage
            elif isinstance(event, RunComplete):
                result.usage = event.usage
                result.completed = True
    if not result.completed:
        raise RunnerError(
            "Factory Droid ended without a completion event.",
            error_type="factory_incomplete_response",
        )
    _apply_emissions(result, parser.finish())
    return result


async def _collect_completion_or_disconnect(
    *,
    request: Request,
    runner: DroidRunner,
    run_request: RunRequest,
    parser: ToolCallStreamParser,
    metrics: BridgeMetrics | None = None,
    request_started_at: float | None = None,
) -> CollectedCompletion | None:
    completion_task = asyncio.create_task(
        _collect_completion(
            runner=runner,
            run_request=run_request,
            parser=parser,
            metrics=metrics,
            request_started_at=request_started_at,
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
) -> AsyncIterator[str]:
    usage = Usage()
    completed = False
    saw_tool_call = False
    observed_ttft = False
    outcome = "success"

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
            async with contextlib.aclosing(runner.run(run_request)) as events:
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
                            yield _sse(
                                _chunk_for_emission(
                                    request_id,
                                    created,
                                    model,
                                    emission,
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

            if not completed:
                raise RunnerError(
                    "Factory Droid ended without a completion event.",
                    error_type="factory_incomplete_response",
                )
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
        except ProtocolError as exc:
            outcome = "error"
            yield _sse(_error_body(str(exc), "factory_protocol_error"))
        except RunnerError as exc:
            outcome = "timeout" if exc.error_type == "factory_droid_timeout" else "error"
            yield _sse(_error_body(str(exc), exc.error_type))
        except asyncio.CancelledError:
            if outcome_callback is not None:
                outcome_callback("cancelled")
            raise
        if outcome_callback is not None:
            outcome_callback(outcome)
        yield "data: [DONE]\n\n"


async def _finalize_stream(
    stream: AsyncIterator[str],
    lease: AdmissionLease,
    request: Request,
) -> None:
    close = getattr(stream, "aclose", None)
    try:
        if callable(close):
            await close()
    finally:
        if getattr(request.state, "stream_outcome", None) == "pending":
            request.state.stream_outcome = "cancelled"
        await lease.release()


def _apply_emissions(
    result: CollectedCompletion,
    emissions: list[ProtocolEmission],
) -> None:
    for emission in emissions:
        if isinstance(emission, TextEmission):
            result.text_parts.append(emission.text)
        else:
            result.tool_calls.append(_tool_call_dict(emission))


def _chunk_for_emission(
    request_id: str,
    created: int,
    model: str,
    emission: ProtocolEmission,
    *,
    include_usage: bool = False,
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
        delta={"tool_calls": [{"index": 0, **_tool_call_dict(emission)}]},
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
        if stream_outcome in {"success", "error", "timeout", "cancelled"}:
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


def _observe_ttft(
    observed: bool,
    metrics: BridgeMetrics | None,
    request_started_at: float | None,
) -> bool:
    if observed:
        return True
    if metrics is not None and request_started_at is not None:
        metrics.observe_ttft(asyncio.get_running_loop().time() - request_started_at)
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
