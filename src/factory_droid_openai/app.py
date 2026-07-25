from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Request, Security
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from factory_droid_openai.config import Settings
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
)

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


def create_app(
    settings: Settings | None = None,
    *,
    runner_factory: RunnerFactory | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    resolved_runner_factory = runner_factory or (
        lambda: DroidRunner(
            droid_path=resolved_settings.droid_path,
            workdir=resolved_settings.workdir,
        )
    )
    concurrency = asyncio.Semaphore(resolved_settings.max_concurrency)
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
        try:
            plan = build_prompt(payload)
        except ProtocolError as exc:
            return _error_response(str(exc), 400, "invalid_request_error")

        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        run_request = RunRequest(
            prompt=plan.prompt,
            model=payload.model,
            model_alias=resolved_settings.model_alias,
            reasoning_effort=(payload.factory_droid_reasoning_effort or payload.reasoning_effort),
            timeout_seconds=min(
                payload.timeout or resolved_settings.timeout_seconds,
                resolved_settings.timeout_seconds,
            ),
        )

        if payload.stream:
            event_stream = _stream_completion(
                request=request,
                request_id=request_id,
                created=created,
                model=payload.model,
                parser=ToolCallStreamParser(
                    plan.allowed_tool_names,
                    require_tool_call=plan.require_tool_call,
                ),
                runner=resolved_runner_factory(),
                run_request=run_request,
                concurrency=concurrency,
                include_usage=bool(payload.stream_options and payload.stream_options.include_usage),
            )
            return StreamingResponse(
                event_stream,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "x-request-id": request_id,
                },
            )

        try:
            async with concurrency:
                result = await _collect_completion(
                    runner=resolved_runner_factory(),
                    run_request=run_request,
                    parser=ToolCallStreamParser(
                        plan.allowed_tool_names,
                        require_tool_call=plan.require_tool_call,
                    ),
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
        self.text = ""
        self.reasoning = ""
        self.tool_calls: list[dict[str, Any]] = []
        self.usage = Usage()
        self.completed = False


async def _collect_completion(
    *,
    runner: DroidRunner,
    run_request: RunRequest,
    parser: ToolCallStreamParser,
) -> CollectedCompletion:
    result = CollectedCompletion()
    async with contextlib.aclosing(runner.run(run_request)) as events:
        async for event in events:
            if isinstance(event, TextDelta):
                _apply_emissions(result, parser.feed(event.text))
            elif isinstance(event, ReasoningDelta):
                result.reasoning += event.text
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


async def _stream_completion(
    *,
    request: Request,
    request_id: str,
    created: int,
    model: str,
    parser: ToolCallStreamParser,
    runner: DroidRunner,
    run_request: RunRequest,
    concurrency: asyncio.Semaphore,
    include_usage: bool = False,
) -> AsyncIterator[str]:
    yield _sse(
        _chunk(
            request_id,
            created,
            model,
            delta={"role": "assistant", "content": ""},
            include_usage=include_usage,
        )
    )
    usage = Usage()
    completed = False
    saw_tool_call = False

    try:
        async with concurrency, contextlib.aclosing(runner.run(run_request)) as events:
            async for event in events:
                if await request.is_disconnected():
                    return
                if isinstance(event, TextDelta):
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
        yield _sse(_error_body(str(exc), "factory_protocol_error"))
    except RunnerError as exc:
        yield _sse(_error_body(str(exc), exc.error_type))
    yield "data: [DONE]\n\n"


def _apply_emissions(
    result: CollectedCompletion,
    emissions: list[ProtocolEmission],
) -> None:
    for emission in emissions:
        if isinstance(emission, TextEmission):
            result.text += emission.text
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


def _error_response(message: str, status_code: int, error_type: str) -> JSONResponse:
    return JSONResponse(
        _error_body(message, error_type),
        status_code=status_code,
    )


def _validation_message(exc: RequestValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "Invalid request."
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", ()))
    message = str(first.get("msg", "Invalid request."))
    return f"{location}: {message}" if location else message


app = create_app()
