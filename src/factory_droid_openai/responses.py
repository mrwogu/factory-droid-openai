from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from factory_droid_openai.models import (
    ChatCompletionRequest,
    ResponsesRequest,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable


class ResponsesConversionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ResponsesPlan:
    chat_request: ChatCompletionRequest
    response_id: str
    reasoning_item_id: str
    message_item_id: str
    continuation_references: tuple[tuple[str, str], ...]


def build_responses_plan(payload: ResponsesRequest) -> ResponsesPlan:
    response_id = f"resp_{uuid.uuid4().hex}"
    messages, references = _messages(payload)
    tools = [_function_tool(tool) for tool in payload.tools or ()]
    response_format = _response_format(payload.text)
    reasoning_effort = payload.reasoning.effort if payload.reasoning is not None else None
    chat_request = ChatCompletionRequest(
        model=payload.model,
        messages=messages,
        tools=tools or None,
        tool_choice=_tool_choice(payload.tool_choice),
        stream=payload.stream,
        stream_options={"include_usage": payload.stream},
        reasoning_effort=reasoning_effort,
        factory_droid_reasoning_effort=payload.factory_droid_reasoning_effort,
        timeout=payload.timeout,
        max_completion_tokens=payload.max_output_tokens,
        parallel_tool_calls=payload.parallel_tool_calls,
        response_format=response_format,
    )
    return ResponsesPlan(
        chat_request=chat_request,
        response_id=response_id,
        reasoning_item_id=f"rs_{response_id[5:]}",
        message_item_id=f"msg_{response_id[5:]}",
        continuation_references=tuple(references),
    )


def response_from_chat(
    payload: ResponsesRequest,
    plan: ResponsesPlan,
    chat: dict[str, Any],
) -> dict[str, Any]:
    choices = chat.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ResponsesConversionError("chat completion returned an invalid choice set")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ResponsesConversionError("chat completion returned an invalid choice")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ResponsesConversionError("chat completion returned an invalid message")
    finish_reason = choice.get("finish_reason")
    output = _response_output(plan, message, finish_reason)
    usage = _responses_usage(chat.get("usage"))
    now = float(chat.get("created", 0))
    status = "incomplete" if finish_reason == "length" else "completed"
    return {
        "id": plan.response_id,
        "object": "response",
        "created_at": now,
        "completed_at": now,
        "status": status,
        "error": None,
        "incomplete_details": ({"reason": "max_output_tokens"} if status == "incomplete" else None),
        "instructions": payload.instructions,
        "model": payload.model,
        "output": output,
        "parallel_tool_calls": payload.parallel_tool_calls,
        "tool_choice": payload.tool_choice or "auto",
        "tools": payload.tools or [],
        "previous_response_id": payload.previous_response_id,
        "reasoning": (
            payload.reasoning.model_dump(exclude_none=True)
            if payload.reasoning is not None
            else None
        ),
        "store": payload.store,
        "usage": usage,
    }


async def response_stream_from_chat(
    payload: ResponsesRequest,
    plan: ResponsesPlan,
    chat_stream: AsyncIterator[str],
    *,
    created_at: float,
) -> AsyncIterator[str]:
    state = _ResponsesStreamState(payload, plan, created_at)
    yield state.event("response.created", response=state.response("in_progress"))
    yield state.event("response.in_progress", response=state.response("in_progress"))
    try:
        async for raw in chat_stream:
            data = _sse_data(raw)
            if data is None or data == "[DONE]":
                continue
            if not isinstance(data, dict):
                continue
            error = data.get("error")
            if isinstance(error, dict):
                yield state.event(
                    "error",
                    code=_optional_string(error.get("code") or error.get("type")),
                    message=str(error.get("message") or "Factory Droid failed."),
                    param=_optional_string(error.get("param")),
                )
                async for _ in chat_stream:
                    pass
                return
            usage = data.get("usage")
            if isinstance(usage, dict):
                state.usage = _responses_usage(usage)
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict):
                async for event in state.delta_events(delta):
                    yield event
            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str):
                state.finish_reason = finish_reason
        async for event in state.done_events():
            yield event
    finally:
        close = getattr(chat_stream, "aclose", None)
        if callable(close):
            await close()


def continuation_references_from_chat(
    payload: ChatCompletionRequest,
) -> tuple[tuple[str, str], ...]:
    references: list[tuple[str, str]] = []
    for message in reversed(payload.messages):
        if message.role == "tool" and message.tool_call_id:
            references.append(("tool_call", message.tool_call_id))
        if message.role != "assistant":
            continue
        if message.reasoning_details:
            digest = reasoning_details_digest(message.reasoning_details)
            if digest is not None:
                references.append(("reasoning", digest))
        break
    return tuple(references)


def reasoning_details_digest(details: Iterable[dict[str, Any]]) -> str | None:
    protected = False
    normalized: list[dict[str, Any]] = []
    for detail in details:
        if not isinstance(detail, dict):
            return None
        signature = detail.get("signature")
        data = detail.get("data")
        protected = protected or bool(signature) or bool(data)
        normalized.append(detail)
    if not protected:
        return None
    raw = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _messages(payload: ResponsesRequest) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    messages: list[dict[str, Any]] = []
    references: list[tuple[str, str]] = []
    if payload.instructions:
        messages.append({"role": "system", "content": payload.instructions})
    if isinstance(payload.input, str):
        messages.append({"role": "user", "content": payload.input})
        return messages, references
    pending_tool_calls: list[dict[str, Any]] = []
    for item in payload.input:
        item_type = item.get("type")
        if item_type in (None, "message") and isinstance(item.get("role"), str):
            _flush_tool_calls(messages, pending_tool_calls)
            messages.append(_input_message(item))
            continue
        if item_type == "function_call":
            function_call = _function_call(item)
            pending_tool_calls.append(function_call)
            references.append(("tool_call", function_call["id"]))
            continue
        _flush_tool_calls(messages, pending_tool_calls)
        if item_type == "function_call_output":
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ResponsesConversionError("function_call_output.call_id is required")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": _function_output(item.get("output")),
                }
            )
            references.append(("tool_call", call_id))
            continue
        if item_type == "reasoning":
            item_id = item.get("id")
            if isinstance(item_id, str) and item_id:
                references.append(("reasoning_item", item_id))
            continue
        raise ResponsesConversionError(f"unsupported Responses input item type: {item_type!r}")
    _flush_tool_calls(messages, pending_tool_calls)
    if not messages:
        raise ResponsesConversionError("input must contain at least one message or tool output")
    return messages, references


def _input_message(item: dict[str, Any]) -> dict[str, Any]:
    role = item.get("role")
    if role not in ("system", "developer", "user", "assistant"):
        raise ResponsesConversionError(f"unsupported Responses message role: {role!r}")
    content = item.get("content")
    if isinstance(content, str):
        return {"role": role, "content": content}
    if not isinstance(content, list):
        raise ResponsesConversionError("Responses message content must be text or a list")
    parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            raise ResponsesConversionError("Responses content parts must be objects")
        part_type = part.get("type")
        if part_type in ("input_text", "output_text"):
            text = part.get("text")
            if not isinstance(text, str):
                raise ResponsesConversionError(f"{part_type}.text is required")
            parts.append({"type": "text", "text": text})
        elif part_type == "input_image":
            image_url = part.get("image_url")
            if not isinstance(image_url, str):
                raise ResponsesConversionError(
                    "input_image.image_url is required; file_id is unsupported"
                )
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "detail": part.get("detail", "auto"),
                    },
                }
            )
        elif part_type == "input_file":
            file_data = part.get("file_data")
            if not isinstance(file_data, str):
                raise ResponsesConversionError(
                    "input_file.file_data is required; file_id and file_url are unsupported"
                )
            parts.append(
                {
                    "type": "file",
                    "file": {
                        "file_data": file_data,
                        "filename": part.get("filename"),
                    },
                }
            )
        else:
            raise ResponsesConversionError(f"unsupported Responses content type: {part_type!r}")
    return {"role": role, "content": parts}


def _function_call(item: dict[str, Any]) -> dict[str, Any]:
    call_id = item.get("call_id")
    name = item.get("name")
    arguments = item.get("arguments")
    if not all(isinstance(value, str) and value for value in (call_id, name)):
        raise ResponsesConversionError("function_call.call_id and name are required")
    if not isinstance(arguments, str):
        raise ResponsesConversionError("function_call.arguments must be a JSON string")
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _flush_tool_calls(
    messages: list[dict[str, Any]],
    pending: list[dict[str, Any]],
) -> None:
    if pending:
        messages.append({"role": "assistant", "content": None, "tool_calls": list(pending)})
        pending.clear()


def _function_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        texts: list[str] = []
        for part in output:
            if not isinstance(part, dict) or part.get("type") not in (
                "input_text",
                "output_text",
            ):
                continue
            text = part.get("text")
            if isinstance(text, str):
                texts.append(text)
        if len(texts) == len(output):
            return "\n".join(texts)
    raise ResponsesConversionError("function_call_output.output must be text")


def _function_tool(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") != "function":
        raise ResponsesConversionError("only function tools are supported by this bridge")
    name = tool.get("name")
    parameters = tool.get("parameters")
    if not isinstance(name, str) or not name:
        raise ResponsesConversionError("function tool name is required")
    if parameters is not None and not isinstance(parameters, dict):
        raise ResponsesConversionError("function tool parameters must be an object")
    function: dict[str, Any] = {
        "name": name,
        "description": tool.get("description") or "",
        "parameters": parameters or {},
    }
    if "strict" in tool:
        function["strict"] = tool["strict"]
    return {"type": "function", "function": function}


def _tool_choice(choice: Any) -> Any:
    if not isinstance(choice, dict):
        return choice
    if choice.get("type") != "function":
        raise ResponsesConversionError("only function tool_choice objects are supported")
    name = choice.get("name")
    if not isinstance(name, str) or not name:
        raise ResponsesConversionError("function tool_choice name is required")
    return {"type": "function", "function": {"name": name}}


def _response_format(text: dict[str, Any] | None) -> dict[str, Any] | None:
    if text is None:
        return None
    value = text.get("format")
    if value is None or value == {"type": "text"}:
        return None
    if not isinstance(value, dict):
        raise ResponsesConversionError("text.format must be an object")
    format_type = value.get("type")
    if format_type == "json_object":
        return {"type": "json_object"}
    if format_type == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": value.get("name"),
                "description": value.get("description"),
                "schema": value.get("schema"),
                "strict": value.get("strict"),
            },
        }
    raise ResponsesConversionError(f"unsupported Responses text format: {format_type!r}")


def _response_output(
    plan: ResponsesPlan,
    message: dict[str, Any],
    finish_reason: Any,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    details = message.get("reasoning_details")
    if isinstance(reasoning, str) or isinstance(details, list):
        output.append(_reasoning_item(plan.reasoning_item_id, reasoning, details))
    content = message.get("content")
    if isinstance(content, str) and content:
        output.append(
            {
                "id": plan.message_item_id,
                "type": "message",
                "role": "assistant",
                "status": "incomplete" if finish_reason == "length" else "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                        "annotations": [],
                    }
                ],
            }
        )
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            call_id = str(call.get("id") or f"call_{uuid.uuid4().hex[:24]}")
            output.append(
                {
                    "id": f"fc_{call_id.removeprefix('call_')}",
                    "call_id": call_id,
                    "type": "function_call",
                    "name": function.get("name"),
                    "arguments": function.get("arguments", "{}"),
                    "status": "completed",
                }
            )
    return output


def _reasoning_item(item_id: str, reasoning: Any, details: Any) -> dict[str, Any]:
    summaries: list[dict[str, str]] = []
    encrypted_content: str | None = None
    if isinstance(details, list):
        for detail in details:
            if not isinstance(detail, dict):
                continue
            if detail.get("type") == "reasoning.summary" and isinstance(detail.get("summary"), str):
                summaries.append({"type": "summary_text", "text": detail["summary"]})
            if (
                detail.get("type") == "reasoning.encrypted"
                and detail.get("format") == "openai-responses-v1"
                and isinstance(detail.get("data"), str)
            ):
                encrypted_content = detail["data"]
    item: dict[str, Any] = {
        "id": item_id,
        "type": "reasoning",
        "status": "completed",
        "summary": summaries,
    }
    if isinstance(reasoning, str) and reasoning:
        item["content"] = [{"type": "reasoning_text", "text": reasoning}]
    if encrypted_content is not None:
        item["encrypted_content"] = encrypted_content
    if isinstance(details, list) and details:
        item["reasoning_details"] = details
    return item


def _responses_usage(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    prompt = int(value.get("prompt_tokens", 0))
    completion = int(value.get("completion_tokens", 0))
    prompt_details = value.get("prompt_tokens_details")
    completion_details = value.get("completion_tokens_details")
    return {
        "input_tokens": prompt,
        "input_tokens_details": {
            "cached_tokens": _detail_int(prompt_details, "cached_tokens"),
            "cache_write_tokens": _detail_int(prompt_details, "cache_write_tokens"),
        },
        "output_tokens": completion,
        "output_tokens_details": {
            "reasoning_tokens": _detail_int(completion_details, "reasoning_tokens"),
        },
        "total_tokens": prompt + completion,
    }


def _detail_int(value: Any, key: str) -> int:
    if not isinstance(value, dict):
        return 0
    raw = value.get(key, 0)
    return int(raw) if isinstance(raw, int | float) else 0


def _sse_data(raw: str) -> Any:
    for line in raw.splitlines():
        if line.startswith("data: "):
            value = line[6:]
            if value == "[DONE]":
                return value
            return json.loads(value)
    return None


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


class _ResponsesStreamState:
    def __init__(
        self,
        payload: ResponsesRequest,
        plan: ResponsesPlan,
        created_at: float,
    ) -> None:
        self.payload = payload
        self.plan = plan
        self.created_at = created_at
        self.sequence = 0
        self.finish_reason = "stop"
        self.usage: dict[str, Any] | None = None
        self.reasoning_chunks: list[str] = []
        self.reasoning_details: list[dict[str, Any]] = []
        self.text_chunks: list[str] = []
        self.reasoning_index: int | None = None
        self.message_index: int | None = None
        self.reasoning_content_started = False
        self.message_content_started = False
        self.tool_items: list[dict[str, Any]] = []
        self.output_order: list[tuple[str, int | None]] = []

    @property
    def reasoning_text(self) -> str:
        return "".join(self.reasoning_chunks)

    @property
    def text(self) -> str:
        return "".join(self.text_chunks)

    def event(self, event_type: str, **fields: Any) -> str:
        payload = {"type": event_type, "sequence_number": self.sequence, **fields}
        self.sequence += 1
        return _sse(payload)

    def response(self, status: str) -> dict[str, Any]:
        output: list[dict[str, Any]] = []
        for kind, tool_index in self.output_order:
            if kind == "reasoning":
                output.append(
                    _reasoning_item(
                        self.plan.reasoning_item_id,
                        self.reasoning_text,
                        self.reasoning_details,
                    )
                )
            elif kind == "message":
                output.append(
                    {
                        "id": self.plan.message_item_id,
                        "type": "message",
                        "role": "assistant",
                        "status": status,
                        "content": [
                            {
                                "type": "output_text",
                                "text": self.text,
                                "annotations": [],
                            }
                        ],
                    }
                )
            else:
                output.append(self.tool_items[cast("int", tool_index)])
        completed = status in {"completed", "incomplete"}
        return {
            "id": self.plan.response_id,
            "object": "response",
            "created_at": self.created_at,
            "completed_at": self.created_at if completed else None,
            "status": status,
            "error": None,
            "incomplete_details": (
                {"reason": "max_output_tokens"} if status == "incomplete" else None
            ),
            "instructions": self.payload.instructions,
            "model": self.payload.model,
            "output": output,
            "parallel_tool_calls": self.payload.parallel_tool_calls,
            "tool_choice": self.payload.tool_choice or "auto",
            "tools": self.payload.tools or [],
            "previous_response_id": self.payload.previous_response_id,
            "reasoning": (
                self.payload.reasoning.model_dump(exclude_none=True)
                if self.payload.reasoning is not None
                else None
            ),
            "store": self.payload.store,
            "usage": self.usage,
        }

    async def delta_events(self, delta: dict[str, Any]) -> AsyncIterator[str]:
        reasoning = delta.get("reasoning") or delta.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            if self.reasoning_index is None:
                self.reasoning_index = self._next_output_index()
                self.output_order.append(("reasoning", None))
                added_item = _reasoning_item(
                    self.plan.reasoning_item_id,
                    "",
                    [],
                )
                added_item["status"] = "in_progress"
                yield self.event(
                    "response.output_item.added",
                    output_index=self.reasoning_index,
                    item=added_item,
                )
            if not self.reasoning_content_started:
                self.reasoning_content_started = True
                yield self.event(
                    "response.content_part.added",
                    item_id=self.plan.reasoning_item_id,
                    output_index=self.reasoning_index,
                    content_index=0,
                    part={"type": "reasoning_text", "text": ""},
                )
            self.reasoning_chunks.append(reasoning)
            yield self.event(
                "response.reasoning_text.delta",
                item_id=self.plan.reasoning_item_id,
                output_index=self.reasoning_index,
                content_index=0,
                delta=reasoning,
            )
        details = delta.get("reasoning_details")
        if isinstance(details, list):
            self.reasoning_details.extend(detail for detail in details if isinstance(detail, dict))
            if self.reasoning_index is None:
                self.reasoning_index = self._next_output_index()
                self.output_order.append(("reasoning", None))
                added_item = _reasoning_item(
                    self.plan.reasoning_item_id,
                    "",
                    [],
                )
                added_item["status"] = "in_progress"
                yield self.event(
                    "response.output_item.added",
                    output_index=self.reasoning_index,
                    item=added_item,
                )
        content = delta.get("content")
        if isinstance(content, str) and content:
            if self.message_index is None:
                self.message_index = self._next_output_index()
                self.output_order.append(("message", None))
                yield self.event(
                    "response.output_item.added",
                    output_index=self.message_index,
                    item={
                        "id": self.plan.message_item_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    },
                )
            if not self.message_content_started:
                self.message_content_started = True
                yield self.event(
                    "response.content_part.added",
                    item_id=self.plan.message_item_id,
                    output_index=self.message_index,
                    content_index=0,
                    part={
                        "type": "output_text",
                        "text": "",
                        "annotations": [],
                        "logprobs": [],
                    },
                )
            self.text_chunks.append(content)
            yield self.event(
                "response.output_text.delta",
                item_id=self.plan.message_item_id,
                output_index=self.message_index,
                content_index=0,
                delta=content,
                logprobs=[],
            )
        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if not isinstance(function, dict):
                    continue
                call_id = str(call.get("id") or f"call_{uuid.uuid4().hex[:24]}")
                item = {
                    "id": f"fc_{call_id.removeprefix('call_')}",
                    "call_id": call_id,
                    "type": "function_call",
                    "name": function.get("name"),
                    "arguments": function.get("arguments", "{}"),
                    "status": "completed",
                }
                output_index = self._next_output_index()
                self.tool_items.append(item)
                self.output_order.append(("tool", len(self.tool_items) - 1))
                added_item = item | {
                    "arguments": "",
                    "status": "in_progress",
                }
                yield self.event(
                    "response.output_item.added",
                    output_index=output_index,
                    item=added_item,
                )
                yield self.event(
                    "response.function_call_arguments.delta",
                    item_id=item["id"],
                    output_index=output_index,
                    delta=item["arguments"],
                )
                yield self.event(
                    "response.function_call_arguments.done",
                    item_id=item["id"],
                    output_index=output_index,
                    name=item["name"],
                    arguments=item["arguments"],
                )
                yield self.event(
                    "response.output_item.done",
                    output_index=output_index,
                    item=item,
                )

    async def done_events(self) -> AsyncIterator[str]:
        status = "incomplete" if self.finish_reason == "length" else "completed"
        if self.reasoning_index is not None:
            item = _reasoning_item(
                self.plan.reasoning_item_id,
                self.reasoning_text,
                self.reasoning_details,
            )
            if self.reasoning_text:
                yield self.event(
                    "response.reasoning_text.done",
                    item_id=self.plan.reasoning_item_id,
                    output_index=self.reasoning_index,
                    content_index=0,
                    text=self.reasoning_text,
                )
                yield self.event(
                    "response.content_part.done",
                    item_id=self.plan.reasoning_item_id,
                    output_index=self.reasoning_index,
                    content_index=0,
                    part={
                        "type": "reasoning_text",
                        "text": self.reasoning_text,
                    },
                )
            yield self.event(
                "response.output_item.done",
                output_index=self.reasoning_index,
                item=item,
            )
        if self.message_index is not None:
            item = {
                "id": self.plan.message_item_id,
                "type": "message",
                "role": "assistant",
                "status": status,
                "content": [
                    {
                        "type": "output_text",
                        "text": self.text,
                        "annotations": [],
                    }
                ],
            }
            yield self.event(
                "response.output_text.done",
                item_id=self.plan.message_item_id,
                output_index=self.message_index,
                content_index=0,
                text=self.text,
                logprobs=[],
            )
            yield self.event(
                "response.content_part.done",
                item_id=self.plan.message_item_id,
                output_index=self.message_index,
                content_index=0,
                part={
                    "type": "output_text",
                    "text": self.text,
                    "annotations": [],
                    "logprobs": [],
                },
            )
            yield self.event(
                "response.output_item.done",
                output_index=self.message_index,
                item=item,
            )
        yield self.event(
            "response.completed" if status == "completed" else "response.incomplete",
            response=self.response(status),
        )

    def _next_output_index(self) -> int:
        return len(self.output_order)
