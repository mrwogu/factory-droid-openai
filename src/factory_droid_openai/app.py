from __future__ import annotations

import asyncio
import json
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from factory_droid_openai.config import Settings
from factory_droid_openai.models import ChatCompletionRequest  # noqa: TC001
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
        version="0.1.0",
    )

    async def require_auth(request: Request) -> None:
        expected = resolved_settings.api_key
        if expected is None:
            return
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(token, expected):
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

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/v1/models", dependencies=[Depends(require_auth)])
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": resolved_settings.model_alias,
                    "object": "model",
                    "created": 0,
                    "owned_by": "factory",
                }
            ],
        }

    @application.post(
        "/v1/chat/completions",
        dependencies=[Depends(require_auth)],
        response_model=None,
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
    async for event in runner.run(run_request):
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
) -> AsyncIterator[str]:
    yield _sse(
        _chunk(
            request_id,
            created,
            model,
            delta={"role": "assistant", "content": ""},
        )
    )
    usage = Usage()
    completed = False
    saw_tool_call = False

    try:
        async with concurrency:
            async for event in runner.run(run_request):
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
                )
            )
        yield _sse(
            _chunk(
                request_id,
                created,
                model,
                delta={},
                finish_reason="tool_calls" if saw_tool_call else "stop",
                usage=usage,
            )
        )
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
) -> dict[str, Any]:
    if isinstance(emission, TextEmission):
        return _chunk(
            request_id,
            created,
            model,
            delta={"content": emission.text},
        )
    return _chunk(
        request_id,
        created,
        model,
        delta={"tool_calls": [{"index": 0, **_tool_call_dict(emission)}]},
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
    usage: Usage | None = None,
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
    if usage is not None:
        body["usage"] = _usage_dict(usage)
    return body


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
