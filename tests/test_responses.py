from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import pytest

from factory_droid_openai.models import (
    ChatCompletionRequest,
    ResponsesRequest,
)
from factory_droid_openai.responses import (
    ResponsesConversionError,
    _detail_int,
    _drain_stream,
    _function_output,
    _reasoning_item,
    _response_output,
    _responses_usage,
    _stream_error_fields,
    build_responses_plan,
    continuation_references_from_chat,
    reasoning_details_digest,
    response_from_chat,
    response_stream_from_chat,
)
from factory_droid_openai.sse import sse
from factory_droid_openai.sse import sse_data as _sse_data

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _request(**overrides: Any) -> ResponsesRequest:
    values: dict[str, Any] = {
        "model": "factory-droid",
        "input": "hello",
    }
    values.update(overrides)
    return ResponsesRequest(**values)


def _chat(
    message: dict[str, Any],
    *,
    finish_reason: str = "stop",
    usage: Any = None,
) -> dict[str, Any]:
    return {
        "created": 10,
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": usage,
    }


def _events(values: list[str]) -> AsyncIterator[str]:
    async def stream() -> AsyncIterator[str]:
        for value in values:
            yield value

    return stream()


async def _collect_stream(
    payload: ResponsesRequest,
    values: AsyncIterator[str],
) -> list[dict[str, Any]]:
    plan = build_responses_plan(payload)
    return [
        json.loads(event.removeprefix("data: "))
        async for event in response_stream_from_chat(
            payload,
            plan,
            values,
            created_at=10,
        )
    ]


@pytest.mark.parametrize(
    ("chat", "message"),
    [
        ({"choices": []}, "choice set"),
        ({"choices": [7]}, "invalid choice"),
        ({"choices": [{"message": 7}]}, "invalid message"),
    ],
)
def test_response_from_chat_rejects_invalid_chat_shapes(
    chat: dict[str, Any],
    message: str,
) -> None:
    payload = _request()
    plan = build_responses_plan(payload)

    with pytest.raises(ResponsesConversionError, match=message):
        response_from_chat(payload, plan, chat)


def test_response_from_chat_maps_length_and_empty_usage() -> None:
    payload = _request(
        instructions="Be short.",
        previous_response_id="resp_old",
        store=False,
        reasoning={"effort": "low"},
    )

    response = response_from_chat(
        payload,
        build_responses_plan(payload),
        _chat({"content": "partial"}, finish_reason="length"),
    )

    assert response["status"] == "incomplete"
    assert response["incomplete_details"] == {"reason": "max_output_tokens"}
    assert response["completed_at"] is None
    assert response["usage"] is None
    assert response["reasoning"] == {"effort": "low"}
    assert response["tool_choice"] == "auto"


def test_response_from_chat_records_completion_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # responses.py imports the stdlib time module, so patching time.time
    # covers the completed_at call under test deterministically.
    monkeypatch.setattr("time.time", lambda: 1234.5)

    payload = _request()
    response = response_from_chat(
        payload,
        build_responses_plan(payload),
        _chat({"content": "done"}),
    )

    assert response["status"] == "completed"
    assert response["created_at"] == 10.0
    assert response["completed_at"] == 1234.5


def test_build_responses_plan_maps_all_supported_inputs() -> None:
    payload = _request(
        instructions="System",
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Look"},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,QUJD",
                    },
                    {
                        "type": "input_file",
                        "file_data": "data:text/plain;base64,QUJD",
                        "filename": "a.txt",
                    },
                ],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "weather",
                "arguments": "{}",
            },
            {"type": "message", "role": "assistant", "content": "Calling."},
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "weather",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_2",
                "output": [
                    {"type": "input_text", "text": "sunny"},
                    {"type": "output_text", "text": "warm"},
                ],
            },
            {"type": "reasoning", "id": "rs_old"},
        ],
        tools=[
            {
                "type": "function",
                "name": "weather",
                "parameters": {},
            }
        ],
        tool_choice={"type": "function", "name": "weather"},
        text={"format": {"type": "json_object"}},
        reasoning={"effort": "high"},
        factory_droid_reasoning_effort="low",
        max_output_tokens=100,
        stream=True,
    )

    plan = build_responses_plan(payload)
    chat = plan.chat_request

    assert [message.role for message in chat.messages] == [
        "system",
        "user",
        "assistant",
        "assistant",
        "assistant",
        "tool",
    ]
    assert chat.messages[1].content == [
        {"type": "text", "text": "Look"},
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64,QUJD",
                "detail": "auto",
            },
        },
        {
            "type": "file",
            "file": {
                "file_data": "data:text/plain;base64,QUJD",
                "filename": "a.txt",
            },
        },
    ]
    assert chat.tool_choice == {
        "type": "function",
        "function": {"name": "weather"},
    }
    assert chat.response_format is not None
    assert chat.response_format.type == "json_object"
    assert chat.factory_droid_reasoning_effort == "low"
    assert plan.continuation_references == (
        ("tool_call", "call_1"),
        ("tool_call", "call_2"),
        ("tool_call", "call_2"),
        ("reasoning_item", "rs_old"),
    )


@pytest.mark.parametrize(
    ("input_value", "message"),
    [
        ([{"type": "function_call_output", "call_id": "", "output": "x"}], "call_id"),
        ([], "at least one"),
        ([{"type": "reasoning", "id": None}], "reasoning.id"),
        ([{"type": "reasoning", "id": ""}], "reasoning.id"),
        ([{"type": "reasoning", "id": 7}], "reasoning.id"),
        ([{"type": "unknown"}], "unsupported Responses input"),
        ([{"type": [], "role": "user", "content": "x"}], "unsupported Responses input"),
        ([{"type": "message", "role": [], "content": "x"}], "unsupported Responses input"),
        ([{"type": "message", "role": "tool", "content": "x"}], "message role"),
        ([{"type": "message", "role": "user", "content": 7}], "content must"),
        (
            [{"type": "message", "role": "user", "content": [7]}],
            "content parts",
        ),
        (
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text"}],
                }
            ],
            "input_text.text",
        ),
        (
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_image", "file_id": "file_1"}],
                }
            ],
            "input_image.image_url",
        ),
        (
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_file", "file_id": "file_1"}],
                }
            ],
            "input_file.file_data",
        ),
        (
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": []}],
                }
            ],
            "unsupported Responses content",
        ),
        (
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "audio"}],
                }
            ],
            "unsupported Responses content",
        ),
        (
            [
                {
                    "type": "function_call",
                    "call_id": "",
                    "name": "weather",
                    "arguments": "{}",
                }
            ],
            "call_id and name",
        ),
        (
            [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "",
                    "arguments": "{}",
                }
            ],
            "call_id and name",
        ),
        (
            [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "weather",
                    "arguments": {},
                }
            ],
            "arguments must",
        ),
    ],
)
def test_build_responses_plan_rejects_invalid_inputs(
    input_value: list[dict[str, Any]],
    message: str,
) -> None:
    payload = _request(input=input_value)

    with pytest.raises(ResponsesConversionError, match=message):
        build_responses_plan(payload)


@pytest.mark.parametrize(
    ("tool", "message"),
    [
        ({"type": "web_search"}, "only function tools"),
        ({"type": "function", "name": ""}, "name is required"),
        (
            {"type": "function", "name": "weather", "parameters": []},
            "parameters must",
        ),
    ],
)
def test_build_responses_plan_rejects_invalid_tools(
    tool: dict[str, Any],
    message: str,
) -> None:
    payload = _request(tools=[tool])

    with pytest.raises(ResponsesConversionError, match=message):
        build_responses_plan(payload)


@pytest.mark.parametrize(
    ("choice", "message"),
    [
        ({"type": "custom", "name": "x"}, "only function tool_choice"),
        ({"type": "function", "name": ""}, "name is required"),
    ],
)
def test_build_responses_plan_rejects_invalid_tool_choice(
    choice: dict[str, Any],
    message: str,
) -> None:
    payload = _request(tool_choice=choice)

    with pytest.raises(ResponsesConversionError, match=message):
        build_responses_plan(payload)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ({"format": "json"}, "must be an object"),
        ({"format": {"type": "yaml"}}, "unsupported Responses text format"),
    ],
)
def test_build_responses_plan_rejects_invalid_text_format(
    text: dict[str, Any],
    message: str,
) -> None:
    payload = _request(text=text)

    with pytest.raises(ResponsesConversionError, match=message):
        build_responses_plan(payload)


@pytest.mark.parametrize(
    "text",
    [
        {},
        {"format": {"type": "text"}},
    ],
)
def test_build_responses_plan_accepts_plain_text_formats(
    text: dict[str, Any],
) -> None:
    assert build_responses_plan(_request(text=text)).chat_request.response_format is None


def test_build_responses_plan_maps_json_schema_format() -> None:
    plan = build_responses_plan(
        _request(
            text={
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "description": "Answer schema.",
                    "schema": {"type": "object"},
                    "strict": True,
                }
            }
        )
    )

    assert plan.chat_request.response_format is not None
    assert plan.chat_request.response_format.type == "json_schema"


@pytest.mark.parametrize(
    "output",
    [
        7,
        [7],
        [{"type": [], "text": "x"}],
        [{"type": "other", "text": "x"}],
        [{"type": "input_text", "text": 7}],
    ],
)
def test_function_output_rejects_non_text(output: Any) -> None:
    with pytest.raises(ResponsesConversionError, match="output must be text"):
        _function_output(output)


def test_continuation_references_skip_unsigned_and_unrelated_messages() -> None:
    request = ChatCompletionRequest(
        model="factory-droid",
        messages=[
            {"role": "system", "content": "system"},
            {
                "role": "assistant",
                "content": "old",
                "reasoning_details": [
                    {
                        "type": "reasoning.text",
                        "text": "plain",
                        "signature": None,
                    }
                ],
            },
            {"role": "user", "content": "next"},
        ],
    )

    assert continuation_references_from_chat(request) == ()
    assert reasoning_details_digest(cast("Any", [None])) is None


def test_continuation_references_include_signed_reasoning() -> None:
    details = [
        {
            "type": "reasoning.text",
            "text": "thought",
            "signature": "signed",
        }
    ]
    request = ChatCompletionRequest(
        model="factory-droid",
        messages=[
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_details": details,
            }
        ],
    )

    assert continuation_references_from_chat(request) == (
        ("reasoning", cast("str", reasoning_details_digest(details))),
    )


def test_continuation_references_stop_at_latest_assistant_message() -> None:
    request = ChatCompletionRequest(
        model="factory-droid",
        messages=[
            {
                "role": "assistant",
                "content": "old",
                "reasoning_details": [
                    {
                        "type": "reasoning.text",
                        "text": "old thought",
                        "signature": "old signature",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_old", "content": "old"},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": "current"},
            {"role": "tool", "tool_call_id": "call_current", "content": "current"},
        ],
    )

    assert continuation_references_from_chat(request) == (("tool_call", "call_current"),)


def test_response_output_skips_invalid_calls_and_generates_missing_id() -> None:
    plan = build_responses_plan(_request())
    output = _response_output(
        plan,
        {
            "content": None,
            "tool_calls": [
                7,
                {"function": 7},
                {"function": {"name": ""}},
                {"function": {"arguments": "{}"}},
                {"function": {"name": "weather"}},
            ],
        },
        "tool_calls",
    )

    assert len(output) == 1
    assert output[0]["name"] == "weather"
    assert output[0]["call_id"].startswith("call_")
    assert output[0]["arguments"] == "{}"


def test_reasoning_item_filters_invalid_provider_details() -> None:
    details: list[Any] = [
        7,
        {"type": "reasoning.summary", "summary": 7},
        {"type": "reasoning.summary", "summary": "summary"},
        {
            "type": "reasoning.encrypted",
            "format": "anthropic-claude-v1",
            "data": "other",
        },
        {
            "type": "reasoning.encrypted",
            "format": "openai-responses-v1",
            "data": 7,
        },
    ]

    item = _reasoning_item("rs_1", None, details)

    assert item["summary"] == [{"type": "summary_text", "text": "summary"}]
    assert "content" not in item
    assert "encrypted_content" not in item
    assert item["reasoning_details"] == details
    assert _reasoning_item("rs_2", "", None) == {
        "id": "rs_2",
        "type": "reasoning",
        "status": "completed",
        "summary": [],
    }


def test_usage_helpers_handle_missing_and_non_numeric_details() -> None:
    assert _responses_usage(None) is None
    assert _detail_int(None, "tokens") == 0
    assert _detail_int({"tokens": "many"}, "tokens") == 0
    assert _detail_int({"tokens": 2.5}, "tokens") == 2
    assert _responses_usage(
        {
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "prompt_tokens_details": None,
            "completion_tokens_details": None,
        }
    ) == {
        "input_tokens": 2,
        "input_tokens_details": {
            "cached_tokens": 0,
            "cache_write_tokens": 0,
        },
        "output_tokens": 3,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 5,
    }


def test_sse_parser_handles_done_comments_and_multiline_input() -> None:
    assert _sse_data("event: ping\n\n") is None
    assert _sse_data("event: message\ndata: [DONE]\n\n") == "[DONE]"
    assert _sse_data('event: message\ndata: {"ok":true}\n\n') == {"ok": True}


@pytest.mark.parametrize("character", ["\u0085", "\u2028", "\u2029"])
def test_sse_roundtrip_preserves_unicode_line_separators(character: str) -> None:
    # splitlines() would treat these as line breaks and truncate the JSON
    # payload mid-string; the shared parser must split on "\n" only.
    raw = sse({"choices": [{"delta": {"content": f"a{character}b"}}]})

    assert _sse_data(raw) == {"choices": [{"delta": {"content": f"a{character}b"}}]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "values",
    [
        [],
        ["event: ping\n\n", "data: [DONE]\n\n"],
    ],
)
async def test_drain_stream_handles_empty_and_pending_events(values: list[str]) -> None:
    await _drain_stream(_events(values))


@pytest.mark.asyncio
async def test_response_stream_maps_errors_and_ignores_non_choice_data() -> None:
    payload = _request(stream=True)
    drained = False

    async def error_stream() -> AsyncIterator[str]:
        nonlocal drained
        for value in [
            "event: ping\n\n",
            "data: [DONE]\n\n",
            "data: []\n\n",
            "data: null\n\n",
            'data: {"usage":{"prompt_tokens":1,"completion_tokens":2}}\n\n',
            'data: {"choices":[7]}\n\n',
            'data: {"error":{"type":"backend","message":"","param":7}}\n\n',
        ]:
            yield value
        drained = True
        yield "data: [DONE]\n\n"

    events = await _collect_stream(
        payload,
        error_stream(),
    )

    assert drained is True
    assert events[0]["type"] == "response.created"
    assert events[-2] == {
        "type": "error",
        "sequence_number": 2,
        "code": "backend",
        "message": "Factory Droid failed.",
        "param": None,
    }
    failed = events[-1]
    assert failed["type"] == "response.failed"
    assert failed["sequence_number"] == 3
    assert failed["response"]["status"] == "failed"
    assert failed["response"]["completed_at"] is None
    assert failed["response"]["error"] == {
        "code": "backend",
        "message": "Factory Droid failed.",
    }


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            {"type": "backend", "message": "", "param": 7},
            {
                "code": "backend",
                "message": "Factory Droid failed.",
                "param": None,
            },
        ),
        (
            {"code": "overloaded", "message": "busy", "param": "model"},
            {"code": "overloaded", "message": "busy", "param": "model"},
        ),
    ],
)
def test_stream_error_fields_normalize_backend_errors(
    error: dict[str, object],
    expected: dict[str, str | None],
) -> None:
    assert _stream_error_fields(error) == expected


@pytest.mark.asyncio
async def test_response_stream_maps_details_tools_and_incomplete_output() -> None:
    payload = _request(stream=True)
    values = [
        'data: {"choices":[{"delta":"ignored"}]}\n\n',
        (
            'data: {"choices":[{"delta":{"reasoning_details":'
            '[7,{"type":"reasoning.encrypted","data":"state",'
            '"format":"openai-responses-v1"}]}}]}\n\n'
        ),
        'data: {"choices":[{"delta":{"content":"a"}}]}\n\n',
        'data: {"choices":[{"delta":{"content":"b"}}]}\n\n',
        (
            'data: {"choices":[{"delta":{"tool_calls":'
            '[7,{"function":7},{"function":{"name":"weather"}}]}}]}\n\n'
        ),
        'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n',
        (
            'data: {"choices":[],"usage":{"prompt_tokens":2,'
            '"completion_tokens":3,"prompt_tokens_details":{},'
            '"completion_tokens_details":{}}}\n\n'
        ),
    ]

    events = await _collect_stream(payload, _events(values))

    assert events[-1]["type"] == "response.incomplete"
    response = events[-1]["response"]
    assert response["status"] == "incomplete"
    assert response["completed_at"] is None
    assert [item["type"] for item in response["output"]] == [
        "reasoning",
        "message",
        "function_call",
    ]
    assert response["usage"]["total_tokens"] == 5
    assert not any(event["type"] == "response.reasoning_text.done" for event in events)
    assert {
        "response.in_progress",
        "response.content_part.added",
        "response.content_part.done",
        "response.output_text.done",
    }.issubset({event["type"] for event in events})


@pytest.mark.asyncio
async def test_response_stream_appends_reasoning_and_late_details() -> None:
    events = await _collect_stream(
        _request(stream=True),
        _events(
            [
                'data: {"choices":[{"delta":{"reasoning":"one"}}]}\n\n',
                'data: {"choices":[{"delta":{"reasoning_content":"two"}}]}\n\n',
                (
                    'data: {"choices":[{"delta":{"reasoning_details":'
                    '[{"type":"reasoning.text","text":"onetwo","signature":"signed"}]}}]}\n\n'
                ),
            ]
        ),
    )

    assert events[-1]["response"]["output"][0]["content"] == [
        {"type": "reasoning_text", "text": "onetwo"}
    ]
    assert events[-1]["response"]["output"][0]["reasoning_details"][0]["signature"] == "signed"


@pytest.mark.asyncio
async def test_response_stream_keeps_late_encrypted_reasoning_after_text() -> None:
    events = await _collect_stream(
        _request(stream=True),
        _events(
            [
                'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n',
                (
                    'data: {"choices":[{"delta":{"reasoning_details":'
                    '[{"type":"reasoning.encrypted","data":"state",'
                    '"format":"openai-responses-v1"}]}}]}\n\n'
                ),
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
            ]
        ),
    )

    # Encrypted reasoning arrives only with the terminal chunk, after message
    # text already took index 0; allocation order keeps every output_index
    # equal to its position in the terminal output array.
    assert events[-1]["type"] == "response.completed"
    assert events[-1]["response"]["completed_at"] is not None
    response_output = events[-1]["response"]["output"]
    assert [item["type"] for item in response_output] == ["message", "reasoning"]
    assert response_output[1]["encrypted_content"] == "state"
    done_items = [
        event["item"]["type"] for event in events if event["type"] == "response.output_item.done"
    ]
    assert done_items == ["message", "reasoning"]
    assert [event["sequence_number"] for event in events] == list(range(len(events)))


@pytest.mark.asyncio
async def test_response_stream_handles_tool_only_output_without_close() -> None:
    class StreamWithoutClose:
        def __init__(self) -> None:
            self._values = iter(
                [
                    (
                        'data: {"choices":[{"delta":{"tool_calls":'
                        '[{"id":"call_1","function":{"name":"weather",'
                        '"arguments":"{}"}}]}}]}\n\n'
                    )
                ]
            )

        def __aiter__(self) -> StreamWithoutClose:
            return self

        async def __anext__(self) -> str:
            try:
                return next(self._values)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    events = await _collect_stream(
        _request(stream=True),
        cast("AsyncIterator[str]", StreamWithoutClose()),
    )

    assert events[-1]["type"] == "response.completed"
    assert [item["type"] for item in events[-1]["response"]["output"]] == ["function_call"]
