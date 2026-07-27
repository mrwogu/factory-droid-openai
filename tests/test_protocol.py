from __future__ import annotations

import json

import pytest

from factory_droid_openai.models import ChatCompletionRequest
from factory_droid_openai.protocol import (
    _MAX_TOOL_PAYLOAD_BYTES,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    ProtocolError,
    StopSequenceBuffer,
    TextEmission,
    ToolCallEmission,
    ToolCallStreamParser,
    build_prompt,
    parse_strict_json,
)


def _request(**overrides: object) -> ChatCompletionRequest:
    payload: dict[str, object] = {
        "model": "factory-droid",
        "messages": [
            {"role": "user", "content": "What is the weather?"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Read the weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    }
    payload.update(overrides)
    return ChatCompletionRequest.model_validate(payload)


def test_build_prompt_preserves_openai_transcript_and_tools() -> None:
    request = _request(
        messages=[
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Weather in Gdansk?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_old",
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "arguments": '{"city":"Gdansk"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_old",
                "content": '{"temperature":20}',
            },
        ]
    )

    plan = build_prompt(request)
    serialized = plan.prompt.split("OPENAI_TRANSCRIPT_JSON\n", 1)[1].split(
        "\nEND_OPENAI_TRANSCRIPT_JSON",
        1,
    )[0]
    transcript = json.loads(serialized)

    assert transcript["messages"][2]["tool_calls"][0]["id"] == "call_old"
    assert transcript["messages"][3]["tool_call_id"] == "call_old"
    assert transcript["tools"][0]["function"]["name"] == "weather"
    assert plan.allowed_tool_names == frozenset({"weather"})


def test_tool_choice_limits_the_available_tool() -> None:
    request = _request(
        tools=[
            {
                "type": "function",
                "function": {"name": "weather", "parameters": {}},
            },
            {
                "type": "function",
                "function": {"name": "calendar", "parameters": {}},
            },
        ],
        tool_choice={
            "type": "function",
            "function": {"name": "calendar"},
        },
    )

    plan = build_prompt(request)

    assert plan.allowed_tool_names == frozenset({"calendar"})
    assert plan.require_tool_call is True
    assert '"name":"weather"' not in plan.prompt


def test_tool_choice_none_removes_tools() -> None:
    plan = build_prompt(_request(tool_choice="none"))

    assert plan.allowed_tool_names == frozenset()
    assert '"tools":[]' in plan.prompt


def test_tool_choice_required_keeps_all_tools() -> None:
    plan = build_prompt(_request(tool_choice="required"))

    assert plan.allowed_tool_names == frozenset({"weather"})
    assert plan.require_tool_call is True


@pytest.mark.parametrize(
    ("chat_request", "message"),
    [
        (_request(tools=[], tool_choice="required"), "needs at least one tool"),
        (
            _request(tool_choice={"type": "function", "function": {}}),
            "function name is missing",
        ),
        (
            _request(
                tool_choice={
                    "type": "function",
                    "function": {"name": "missing"},
                }
            ),
            "unknown tool",
        ),
        (_request(tool_choice=42), "unsupported tool_choice"),
    ],
)
def test_build_prompt_rejects_invalid_tool_choice(
    chat_request: ChatCompletionRequest,
    message: str,
) -> None:
    with pytest.raises(ProtocolError, match=message):
        build_prompt(chat_request)


def test_build_prompt_enforces_message_count_before_serializing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = [{"role": "user", "content": f"message {index}"} for index in range(3)]

    build_prompt(_request(messages=messages[:2]), max_messages=2)
    request = _request(messages=messages)
    message_type = type(request.messages[0])
    monkeypatch.setattr(
        message_type,
        "model_dump",
        lambda *_args, **_kwargs: pytest.fail("message serialized before count check"),
    )

    with pytest.raises(ProtocolError, match="maximum of 2 messages"):
        build_prompt(request, max_messages=2)


def test_build_prompt_enforces_tool_count_boundary() -> None:
    tools = [
        {
            "type": "function",
            "function": {"name": f"tool_{index}", "parameters": {}},
        }
        for index in range(3)
    ]

    build_prompt(_request(tools=tools[:2]), max_tools=2)

    with pytest.raises(ProtocolError, match="maximum of 2 tools"):
        build_prompt(_request(tools=tools), max_tools=2)


def test_build_prompt_enforces_utf8_tool_schema_byte_boundary() -> None:
    request = _request(
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Zażółć",
                    "parameters": {},
                },
            }
        ]
    )
    assert request.tools is not None
    serialized = json.dumps(
        request.tools[0].model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    schema_bytes = len(serialized.encode("utf-8"))

    build_prompt(request, max_tool_schema_bytes=schema_bytes)

    with pytest.raises(ProtocolError, match=f"maximum of {schema_bytes - 1} bytes"):
        build_prompt(request, max_tool_schema_bytes=schema_bytes - 1)


def test_build_prompt_enforces_utf8_transcript_byte_boundary() -> None:
    request = _request(
        messages=[{"role": "user", "content": "Zażółć gęślą jaźń"}],
        tools=[],
    )
    plan = build_prompt(request)
    transcript = plan.prompt.split("OPENAI_TRANSCRIPT_JSON\n", 1)[1].split(
        "\nEND_OPENAI_TRANSCRIPT_JSON",
        1,
    )[0]
    transcript_bytes = len(transcript.encode("utf-8"))

    build_prompt(request, max_transcript_bytes=transcript_bytes)

    with pytest.raises(ProtocolError, match=f"maximum of {transcript_bytes - 1} bytes"):
        build_prompt(request, max_transcript_bytes=transcript_bytes - 1)


def test_build_prompt_enforces_tool_schema_depth_boundary() -> None:
    max_depth = 6

    def nested_parameters(wrappers: int) -> dict[str, object]:
        value: dict[str, object] = {}
        for _ in range(wrappers):
            value = {"nested": value}
        return value

    build_prompt(
        _request(
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "parameters": nested_parameters(max_depth - 3),
                    },
                }
            ]
        ),
        max_json_depth=max_depth,
    )

    with pytest.raises(ProtocolError, match="maximum JSON depth of 6"):
        build_prompt(
            _request(
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "parameters": nested_parameters(max_depth - 2),
                        },
                    }
                ]
            ),
            max_json_depth=max_depth,
        )


def test_stream_parser_handles_every_marker_split() -> None:
    value = (
        "Before "
        f"{TOOL_CALL_OPEN}"
        '{"name":"weather","arguments":{"city":"Gdańsk"}}'
        f"{TOOL_CALL_CLOSE}"
    )

    for split in range(1, len(value)):
        parser = ToolCallStreamParser(frozenset({"weather"}))
        emissions = parser.feed(value[:split])
        emissions.extend(parser.feed(value[split:]))
        emissions.extend(parser.finish())

        text = "".join(
            emission.text for emission in emissions if isinstance(emission, TextEmission)
        )
        tool_calls = [emission for emission in emissions if isinstance(emission, ToolCallEmission)]
        assert text == "Before "
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "weather"
        assert json.loads(tool_calls[0].arguments) == {"city": "Gdańsk"}


@pytest.mark.parametrize("chunk_size", [1, 7])
def test_stream_parser_handles_one_character_and_multibyte_chunks(
    chunk_size: int,
) -> None:
    value = (
        "Przed "
        f"{TOOL_CALL_OPEN}"
        '{"name":"weather","arguments":{"city":"Gdańsk"}}'
        f"{TOOL_CALL_CLOSE}"
    )
    parser = ToolCallStreamParser(frozenset({"weather"}))
    emissions: list[TextEmission | ToolCallEmission] = []

    for index in range(0, len(value), chunk_size):
        emissions.extend(parser.feed(value[index : index + chunk_size]))
    emissions.extend(parser.finish())

    assert (
        "".join(emission.text for emission in emissions if isinstance(emission, TextEmission))
        == "Przed "
    )
    tool_calls = [emission for emission in emissions if isinstance(emission, ToolCallEmission)]
    assert len(tool_calls) == 1
    assert json.loads(tool_calls[0].arguments) == {"city": "Gdańsk"}


def test_stream_parser_handles_every_close_marker_split() -> None:
    payload = '{"name":"weather","arguments":{"city":"Gdańsk"}}'

    for split in range(1, len(TOOL_CALL_CLOSE)):
        parser = ToolCallStreamParser(frozenset({"weather"}))
        emissions = parser.feed(f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE[:split]}")
        emissions.extend(parser.feed(TOOL_CALL_CLOSE[split:]))
        emissions.extend(parser.finish())

        assert len(emissions) == 1
        assert isinstance(emissions[0], ToolCallEmission)
        assert json.loads(emissions[0].arguments) == {"city": "Gdańsk"}


def test_stream_parser_rejects_unknown_tool() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    with pytest.raises(ProtocolError, match="not available"):
        parser.feed(f'{TOOL_CALL_OPEN}{{"name":"shell","arguments":{{}}}}{TOOL_CALL_CLOSE}')


def test_stream_parser_rejects_duplicate_keys() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    with pytest.raises(ProtocolError, match="duplicate key"):
        parser.feed(
            f'{TOOL_CALL_OPEN}{{"name":"weather","name":"shell","arguments":{{}}}}{TOOL_CALL_CLOSE}'
        )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("1e999", "non-finite number"),
        ("-1e999", "non-finite number"),
        ("NaN", "non-JSON numeric constant"),
        ("Infinity", "non-JSON numeric constant"),
    ],
)
def test_strict_json_rejects_values_outside_the_json_grammar(
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_strict_json(f'{{"amount":{value}}}')


def test_strict_json_keeps_finite_numbers() -> None:
    assert parse_strict_json('{"amount":1.5,"count":2}') == {"amount": 1.5, "count": 2}


def test_stream_parser_rejects_non_finite_tool_arguments() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    with pytest.raises(ProtocolError, match="non-finite number"):
        parser.feed(
            f'{TOOL_CALL_OPEN}{{"name":"weather","arguments":{{"lat":1e999}}}}{TOOL_CALL_CLOSE}'
        )


def test_stream_parser_requires_object_arguments() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    with pytest.raises(ProtocolError, match="arguments must be a JSON object"):
        parser.feed(
            f"{TOOL_CALL_OPEN}"
            '{"name":"weather","arguments":"{\\"city\\":\\"Gdansk\\"}"}'
            f"{TOOL_CALL_CLOSE}"
        )


def test_required_tool_call_rejects_plain_text() -> None:
    parser = ToolCallStreamParser(
        frozenset({"weather"}),
        require_tool_call=True,
    )
    parser.feed("No tool needed.")

    with pytest.raises(ProtocolError, match="required tool call"):
        parser.finish()


def test_stream_parser_accepts_whitespace_after_tool_call() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    assert parser.feed("") == []
    emissions = parser.feed(
        f'{TOOL_CALL_OPEN}{{"name":"weather","arguments":{{}}}}{TOOL_CALL_CLOSE}\n'
    )
    emissions.extend(parser.feed(" \t"))
    emissions.extend(parser.finish())

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert emissions[0].id.startswith("call_")

    with pytest.raises(ProtocolError, match="unexpected text after tool call"):
        parser.feed("trailing")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (f"{TOOL_CALL_OPEN}{{", "incomplete tool-call marker"),
        (
            f'{TOOL_CALL_OPEN}{{"name":"weather","arguments":{{}}}}{TOOL_CALL_CLOSE}trailing',
            "unexpected text after tool call",
        ),
        (
            f'{TOOL_CALL_OPEN}{{"name":"weather","arguments":{{}}}}{TOOL_CALL_CLOSE}',
            "none are available",
        ),
    ],
)
def test_stream_parser_rejects_invalid_marker_sequences(
    value: str,
    message: str,
) -> None:
    parser = ToolCallStreamParser(
        frozenset() if "none are available" in message else frozenset({"weather"})
    )

    if "incomplete" in message:
        parser.feed(value)
        with pytest.raises(ProtocolError, match=message):
            parser.finish()
    else:
        with pytest.raises(ProtocolError, match=message):
            parser.feed(value)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("not-json", "invalid tool-call JSON"),
        ("[]", "must be a JSON object"),
        ('{"name":"","arguments":{}}', "non-empty string"),
    ],
)
def test_stream_parser_rejects_invalid_payloads(
    payload: str,
    message: str,
) -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    with pytest.raises(ProtocolError, match=message):
        parser.feed(f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE}")


def test_stream_parser_rejects_oversized_payload() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    with pytest.raises(ProtocolError, match="too large"):
        parser.feed(f"{TOOL_CALL_OPEN}{'x' * 1_000_001}")


def test_stream_parser_enforces_unicode_payload_byte_boundary() -> None:
    prefix = '{"name":"weather","arguments":{"value":"'
    suffix = '"}}'
    fixed_bytes = len((prefix + suffix).encode("utf-8"))
    filler = "é" + ("x" * (_MAX_TOOL_PAYLOAD_BYTES - fixed_bytes - 2))
    payload = f"{prefix}{filler}{suffix}"
    assert len(payload.encode("utf-8")) == _MAX_TOOL_PAYLOAD_BYTES

    parser = ToolCallStreamParser(frozenset({"weather"}))
    partial_close = TOOL_CALL_CLOSE[:-1]
    assert parser.feed(f"{TOOL_CALL_OPEN}{payload}{partial_close}") == []
    emissions = parser.feed(TOOL_CALL_CLOSE[-1:])
    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert json.loads(emissions[0].arguments) == {"value": filler}

    parser = ToolCallStreamParser(frozenset({"weather"}))
    assert parser.feed(f"{TOOL_CALL_OPEN}{payload}") == []
    with pytest.raises(ProtocolError, match="too large"):
        parser.feed("x")


def test_stream_parser_handles_large_chunked_payload() -> None:
    value = "ż" * 100_000
    payload = json.dumps(
        {"name": "weather", "arguments": {"value": value}},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    stream = f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE}"
    parser = ToolCallStreamParser(frozenset({"weather"}))
    emissions: list[TextEmission | ToolCallEmission] = []

    for index in range(0, len(stream), 31):
        emissions.extend(parser.feed(stream[index : index + 31]))
    emissions.extend(parser.finish())

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert json.loads(emissions[0].arguments) == {"value": value}


def test_stream_parser_preserves_partial_marker_as_plain_text() -> None:
    parser = ToolCallStreamParser(frozenset())

    emissions = parser.feed("literal <tool_")
    emissions.extend(parser.finish())

    assert (
        "".join(emission.text for emission in emissions if isinstance(emission, TextEmission))
        == "literal <tool_"
    )


def _tool_call(name: str, arguments: str) -> str:
    return f'{TOOL_CALL_OPEN}{{"name":"{name}","arguments":{arguments}}}{TOOL_CALL_CLOSE}'


def test_parser_accepts_several_tool_calls_when_allowed() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}), max_tool_calls=3)

    emissions = parser.feed(
        _tool_call("weather", '{"city":"Gdansk"}')
        + "\n  "
        + _tool_call("weather", '{"city":"Sopot"}')
    )
    emissions.extend(parser.finish())

    calls = [item for item in emissions if isinstance(item, ToolCallEmission)]
    assert [json.loads(call.arguments)["city"] for call in calls] == ["Gdansk", "Sopot"]
    assert all(isinstance(item, ToolCallEmission) for item in emissions)


def test_parser_stops_accepting_calls_past_the_configured_cap() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}), max_tool_calls=1)

    with pytest.raises(ProtocolError, match="unexpected text after tool call"):
        parser.feed(
            _tool_call("weather", '{"city":"Gdansk"}') + _tool_call("weather", '{"city":"Sopot"}')
        )


def test_parser_rejects_prose_between_tool_calls() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}), max_tool_calls=3)
    parser.feed(_tool_call("weather", '{"city":"Gdansk"}'))

    with pytest.raises(ProtocolError, match="unexpected text after tool call"):
        parser.feed("and now some prose")


def test_parser_rejects_trailing_prose_at_finish_after_tool_call() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}), max_tool_calls=3)
    parser.feed(_tool_call("weather", '{"city":"Gdansk"}'))
    # Held back as a possible opening marker, so it only fails at finish().
    parser.feed("  <tool")

    with pytest.raises(ProtocolError, match="unexpected text after tool call"):
        parser.finish()


def test_parser_drops_whitespace_after_final_tool_call() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}), max_tool_calls=3)
    parser.feed(_tool_call("weather", '{"city":"Gdansk"}'))
    parser.feed("   ")

    assert parser.finish() == []


def test_stop_buffer_truncates_at_first_sequence() -> None:
    buffer = StopSequenceBuffer(("STOP",))

    assert buffer.feed("hello ") == "hello "
    assert buffer.feed("world STOP tail") == "world "
    assert buffer.triggered is True
    assert buffer.feed("more") == ""
    assert buffer.flush() == ""


def test_stop_buffer_holds_back_partial_sequences() -> None:
    buffer = StopSequenceBuffer(("STOP",))

    assert buffer.feed("hello ST") == "hello "
    assert buffer.feed("OP after") == ""
    assert buffer.triggered is True


def test_stop_buffer_releases_held_text_that_is_not_a_stop() -> None:
    buffer = StopSequenceBuffer(("STOP",))

    assert buffer.feed("done ST") == "done "
    assert buffer.flush() == "ST"
    assert buffer.triggered is False


def test_stop_buffer_picks_the_earliest_of_several_sequences() -> None:
    buffer = StopSequenceBuffer(("END", "STOP"))

    assert buffer.feed("a STOP b END") == "a "


def test_stop_buffer_without_sequences_passes_text_through() -> None:
    buffer = StopSequenceBuffer(())

    assert buffer.feed("anything") == "anything"
    assert buffer.flush() == ""


def test_multi_tool_prompt_mentions_the_cap() -> None:
    plan = build_prompt(_request(), max_tool_calls=4)

    assert "up to 4 tool requests" in plan.prompt


def test_continuation_prompt_sends_only_messages_after_last_assistant() -> None:
    request = _request(
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
    )

    plan = build_prompt(request, continuation=True)

    assert "second" in plan.prompt
    assert "first" not in plan.prompt
    assert "session already holds the earlier turns" in plan.prompt


def test_continuation_falls_back_to_full_transcript_without_assistant_turn() -> None:
    request = _request(
        messages=[
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
    )

    plan = build_prompt(request, continuation=True)

    assert "first" in plan.prompt
    assert "second" in plan.prompt


def test_continuation_falls_back_when_assistant_message_is_last() -> None:
    request = _request(
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
        ]
    )

    plan = build_prompt(request, continuation=True)

    assert "first" in plan.prompt


def test_build_prompt_moves_images_out_of_the_transcript() -> None:
    request = _request(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,QUJD"},
                    },
                ],
            }
        ]
    )

    plan = build_prompt(request)

    assert "QUJD" not in plan.prompt
    assert "[attached image #1 (image/png)]" in plan.prompt
    assert plan.attachments.images[0].data == "QUJD"


def test_streaming_parser_keeps_a_bounded_tail_for_large_payloads() -> None:
    """Guards the invariant that keeps tool-call parsing linear.

    The parser must only ever re-scan the incoming chunk plus a tail shorter
    than the closing marker. Accumulating the payload and re-scanning it on
    every feed is what made this O(n^2), and nothing else in the suite would
    notice that coming back.
    """
    payload = json.dumps({"name": "lookup", "arguments": {"q": "x" * 200_000}})
    stream = f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE}"
    parser = ToolCallStreamParser(frozenset({"lookup"}))
    limit = len(TOOL_CALL_CLOSE) - 1
    widest_tail = 0

    emissions: list[object] = []
    for index in range(0, len(stream), 16):
        emissions.extend(parser.feed(stream[index : index + 16]))
        widest_tail = max(widest_tail, len(parser._close_tail))

    assert widest_tail <= limit
    emissions.extend(parser.finish())
    assert len(emissions) == 1
    call = emissions[0]
    assert isinstance(call, ToolCallEmission)
    assert json.loads(call.arguments)["q"] == "x" * 200_000
