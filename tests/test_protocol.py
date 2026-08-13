from __future__ import annotations

import io
import json
from typing import Any

import pytest

from factory_droid_openai import logs, protocol
from factory_droid_openai.dialects import (
    _MAX_PACKED_CALLS,
    MARKER_DIALECTS,
    MarkerDialect,
)
from factory_droid_openai.models import ChatCompletionRequest
from factory_droid_openai.protocol import (
    _MAX_TOOL_PAYLOAD_BYTES,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    IncompleteToolCallError,
    MalformedToolCallError,
    ProtocolError,
    RequestTooLargeError,
    StopSequenceBuffer,
    TextEmission,
    ToolCallEmission,
    ToolCallStreamParser,
    build_prompt,
    parse_strict_json,
)
from factory_droid_openai.strictjson import (
    DuplicateKeyError,
    check_no_duplicate_keys,
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


def _feed_parser_to_finish(
    parser: ToolCallStreamParser,
    emissions: list[object],
    *chunks: str,
) -> None:
    for chunk in chunks:
        emissions.extend(parser.feed(chunk))
    emissions.extend(parser.finish())


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


@pytest.mark.parametrize("arguments", ['{"city":"A","city":"B"}', "[]"])
def test_build_prompt_rejects_malformed_assistant_tool_arguments(arguments: str) -> None:
    request = _request(
        messages=[
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_old",
                        "type": "function",
                        "function": {"name": "weather", "arguments": arguments},
                    }
                ],
            }
        ]
    )

    with pytest.raises(ProtocolError, match="assistant tool-call arguments"):
        build_prompt(request)


@pytest.mark.parametrize("arguments", ["", "   "])
def test_build_prompt_normalizes_blank_assistant_tool_arguments(arguments: str) -> None:
    request = _request(
        messages=[
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_old",
                        "type": "function",
                        "function": {"name": "weather", "arguments": arguments},
                    }
                ],
            }
        ]
    )

    plan = build_prompt(request)
    serialized = plan.prompt.split("OPENAI_TRANSCRIPT_JSON\n", 1)[1].split(
        "\nEND_OPENAI_TRANSCRIPT_JSON",
        1,
    )[0]
    transcript = json.loads(serialized)

    assert transcript["messages"][0]["tool_calls"][0]["function"]["arguments"] == "{}"


def test_build_prompt_shows_a_concrete_tool_call_example() -> None:
    request = _request(
        tools=[
            {
                "type": "function",
                "function": {"name": "weather", "parameters": {}},
            }
        ]
    )

    plan = build_prompt(request)

    assert (
        'Example: <tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>.'
    ) in plan.prompt
    assert "python-style name(...) or name{...} call syntax" in plan.prompt


def test_build_prompt_names_the_callable_tools() -> None:
    request = _request(
        tools=[
            {"type": "function", "function": {"name": "weather", "parameters": {}}},
            {"type": "function", "function": {"name": "clock", "parameters": {}}},
        ]
    )

    plan = build_prompt(request)

    assert "The client provides these callable tools: clock, weather." in plan.prompt


def test_build_prompt_forbids_prose_when_a_tool_call_is_required() -> None:
    request = _request(
        tools=[{"type": "function", "function": {"name": "weather", "parameters": {}}}],
        tool_choice="required",
    )

    plan = build_prompt(request)

    assert "reply with the tool request markers only and no prose" in plan.prompt


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


def test_build_prompt_enforces_message_depth() -> None:
    request = _request(
        tools=[],
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    )

    with pytest.raises(RequestTooLargeError, match="message exceeds maximum JSON depth of 2"):
        build_prompt(request, max_json_depth=2)


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
        emissions: list[object] = []
        _feed_parser_to_finish(parser, emissions, value[:split], value[split:])

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
        emissions: list[object] = []
        _feed_parser_to_finish(
            parser,
            emissions,
            f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE[:split]}",
            TOOL_CALL_CLOSE[split:],
        )

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


def test_check_no_duplicate_keys_rejects_repeated_keys() -> None:
    with pytest.raises(DuplicateKeyError, match="duplicate key 'model'"):
        check_no_duplicate_keys('{"model":"a","model":"b"}')


def test_check_no_duplicate_keys_rejects_nested_repeats() -> None:
    with pytest.raises(DuplicateKeyError, match="duplicate key 'content'"):
        check_no_duplicate_keys('{"messages":[{"content":"a","content":"b"}]}')


def test_check_no_duplicate_keys_accepts_unique_keys() -> None:
    # Must not raise for a well-formed document, including nested objects.
    check_no_duplicate_keys('{"model":"a","messages":[{"role":"user","content":"hi"}]}')


def test_check_no_duplicate_keys_keeps_standard_number_handling() -> None:
    # Unlike parse_strict_json, this check must not reject values the standard
    # JSON parser accepts, so it only adds the duplicate-key rejection.
    check_no_duplicate_keys('{"big":1e400}')


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
        ("7", "must be a JSON object"),
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


def test_stream_parser_repairs_arg_key_value_payload() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    emissions = parser.feed(
        f"{TOOL_CALL_OPEN}weather\n"
        f"{'<arg_key>'}city{'</arg_key>'}\n"
        f"{'<arg_value>'}Gdańsk{'</arg_value>'}\n"
        f"{'<arg_key>'}days{'</arg_key>'}\n"
        f"{'<arg_value>'}3{'</arg_value>'}\n"
        f"{TOOL_CALL_CLOSE}"
    )
    emissions.extend(parser.finish())

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert json.loads(emissions[0].arguments) == {"city": "Gdańsk", "days": 3}


@pytest.mark.parametrize(
    "payload",
    [
        "weather<arg_key>city</arg_key><arg_value>Gdansk</arg_value>unexpected",
        "weather<arg_key>city</arg_key>unexpected<arg_value>Gdansk</arg_value>",
        (
            "weather<arg_key>city</arg_key><arg_value>Gdansk</arg_value>"
            "unexpected<arg_key>days</arg_key><arg_value>2</arg_value>"
        ),
    ],
)
def test_stream_parser_rejects_residue_inside_arg_key_value_payload(payload: str) -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    with pytest.raises(MalformedToolCallError):
        parser.feed(f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE}")


def test_stream_parser_keeps_separate_tagged_arg_key_calls() -> None:
    parser = ToolCallStreamParser(
        frozenset({"weather", "calendar"}),
        max_tool_calls=2,
    )
    payload = (
        "weather<arg_key>city</arg_key><arg_value>Gdansk</arg_value>"
        "<tool_call>"
        "calendar<arg_key>day</arg_key><arg_value>2</arg_value>"
    )

    emissions = parser.feed(f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE}")

    assert [emission.name for emission in emissions if isinstance(emission, ToolCallEmission)] == [
        "weather",
        "calendar",
    ]
    assert [
        json.loads(emission.arguments)
        for emission in emissions
        if isinstance(emission, ToolCallEmission)
    ] == [{"city": "Gdansk"}, {"day": 2}]


def test_stream_parser_uses_injected_payload_tracer() -> None:
    traces: list[tuple[str, str, dict[str, Any]]] = []

    def trace(event: str, payload: str, **fields: Any) -> None:
        traces.append((event, payload, fields))

    parser = ToolCallStreamParser(
        frozenset({"weather"}),
        max_tool_calls=2,
        trace_payload=trace,
    )
    payload = (
        '{"name":"weather","arguments":{"city":"Gdansk"}}'
        '{"name":"weather","arguments":{"city":"Sopot"}}'
    )

    parser.feed(f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE}")

    assert traces == [("tool_call.repaired", payload, {"variant": "packed_objects"})]


def test_stream_parser_traces_unparsed_payload() -> None:
    traces: list[tuple[str, str, dict[str, Any]]] = []

    def trace(event: str, payload: str, **fields: Any) -> None:
        traces.append((event, payload, fields))

    parser = ToolCallStreamParser(frozenset({"weather"}), trace_payload=trace)

    with pytest.raises(ProtocolError, match="invalid tool-call JSON"):
        parser.feed(f"{TOOL_CALL_OPEN}not json{TOOL_CALL_CLOSE}")

    assert traces == [("tool_call.unparsed", "not json", {})]


def test_stream_parser_malformed_error_carries_name_and_size() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))
    body = 'weather","city":"Krakow"}'

    with pytest.raises(MalformedToolCallError) as excinfo:
        parser.feed(f"{TOOL_CALL_OPEN}{body}{TOOL_CALL_CLOSE}")

    assert isinstance(excinfo.value, ProtocolError)
    assert excinfo.value.tool_name == "weather"
    assert excinfo.value.payload_bytes == len(body.encode("utf-8"))
    assert "invalid tool-call JSON" in str(excinfo.value)


def test_stream_parser_malformed_error_without_recognized_name() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    with pytest.raises(MalformedToolCallError) as excinfo:
        parser.feed(f"{TOOL_CALL_OPEN}not json{TOOL_CALL_CLOSE}")

    assert excinfo.value.tool_name is None


def test_stream_parser_finish_after_malformed_does_not_reraise() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    with pytest.raises(MalformedToolCallError):
        parser.feed(f"{TOOL_CALL_OPEN}not json{TOOL_CALL_CLOSE}")

    assert parser.finish() == []


def test_stream_parser_logs_unparsed_payload_details() -> None:
    stream = io.StringIO()
    logs.configure_logging(level="trace", log_format="json", stream=stream)
    parser = ToolCallStreamParser(frozenset({"weather"}))
    body = "x" * 80

    with pytest.raises(MalformedToolCallError):
        parser.feed(f"{TOOL_CALL_OPEN}{body}{TOOL_CALL_CLOSE}")

    records = [json.loads(line) for line in stream.getvalue().splitlines() if line]
    event = next(record for record in records if record["event"] == "tool_call.unparsed")
    assert event["head"] == body[:64]
    assert event["tail"] == body[-64:]
    assert event["payload_bytes"] == 80
    assert event["dialect"] == "native"
    assert "tool_name" not in event


def test_stream_parser_omits_tail_for_short_unparsed_payloads() -> None:
    stream = io.StringIO()
    logs.configure_logging(level="trace", log_format="json", stream=stream)
    parser = ToolCallStreamParser(frozenset({"weather"}))

    with pytest.raises(MalformedToolCallError):
        parser.feed(f"{TOOL_CALL_OPEN}not json{TOOL_CALL_CLOSE}")

    records = [json.loads(line) for line in stream.getvalue().splitlines() if line]
    event = next(record for record in records if record["event"] == "tool_call.unparsed")
    assert event["head"] == "not json"
    assert "tail" not in event


def test_stream_parser_repairs_lost_prefix_only_when_enabled() -> None:
    body = 'weather","city":"Krakow","days":3}'

    strict = ToolCallStreamParser(frozenset({"weather"}))
    with pytest.raises(MalformedToolCallError):
        strict.feed(f"{TOOL_CALL_OPEN}{body}{TOOL_CALL_CLOSE}")

    repairing = ToolCallStreamParser(frozenset({"weather"}), repair_lost_prefix=True)
    emissions = repairing.feed(f"{TOOL_CALL_OPEN}{body}{TOOL_CALL_CLOSE}")

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert emissions[0].name == "weather"
    assert json.loads(emissions[0].arguments) == {"city": "Krakow", "days": 3}


def test_stream_parser_translates_openai_assistant_message_json() -> None:
    parser = ToolCallStreamParser(frozenset({"run_in_terminal"}))
    message = json.dumps(
        {
            "role": "assistant",
            "content": "I will check.",
            "tool_calls": [
                {
                    "id": "call_model",
                    "type": "function",
                    "function": {
                        "name": "run_in_terminal",
                        "arguments": '{"command":"pwd"}',
                    },
                }
            ],
        },
        separators=(",", ":"),
    )

    emissions: list[object] = []
    emissions.extend(parser.feed("Status: " + message[:8]))
    for start in range(8, len(message), 5):
        emissions.extend(parser.feed(message[start : start + 5]))
    emissions.extend(parser.finish())

    assert [type(emission) for emission in emissions] == [
        TextEmission,
        TextEmission,
        ToolCallEmission,
    ]
    assert isinstance(emissions[0], TextEmission)
    assert emissions[0].text == "Status: "
    assert isinstance(emissions[1], TextEmission)
    assert emissions[1].text == "I will check."
    assert isinstance(emissions[2], ToolCallEmission)
    assert emissions[2].name == "run_in_terminal"
    assert json.loads(emissions[2].arguments) == {"command": "pwd"}


def test_stream_parser_ignores_transcript_tail_after_message_json() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))
    message = '{"role":"assistant","tool_calls":[{"function":{"name":"weather","arguments":"{}"}}]}'

    assert len(parser.feed(message)) == 1
    assert parser.feed('{"role":"tool","content":"copied"}') == []


def test_stream_parser_rejects_incomplete_transcript_message() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    parser.feed('{"role":"assistant","tool_calls":')

    with pytest.raises(MalformedToolCallError, match="echoed an OpenAI transcript"):
        parser.finish()


def test_stream_parser_rejects_an_oversized_transcript_message() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    with pytest.raises(MalformedToolCallError, match="echoed an OpenAI transcript"):
        parser.feed('{"role":"assistant","content":"' + ("x" * 1_000_001))
    assert parser.finish() == []


def test_stream_parser_preserves_assistant_message_content_without_tool_calls() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))
    message = '{"role":"assistant","content":"done"}'

    emissions = parser.feed(message)

    assert emissions == [TextEmission(message)]


@pytest.mark.parametrize(
    "message",
    [
        (
            '{ "role" : "assistant", "tool_calls": '
            '[{"function":{"name":"weather","arguments":"{}"}}]}'
        ),
        (
            '{"content":null,"tool_calls":'
            '[{"function":{"name":"weather","arguments":"{}"}}],"role":"assistant"}'
        ),
    ],
)
def test_stream_parser_recognizes_message_json_across_layouts(message: str) -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))
    emissions: list[object] = []

    for char in message:
        emissions.extend(parser.feed(char))
    emissions.extend(parser.finish())

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert emissions[0].name == "weather"


def test_stream_parser_preserves_prose_that_quotes_an_assistant_message() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))
    text = 'Example: {"role":"assistant","content":"done"}'

    emissions = parser.feed(text)
    emissions.extend(parser.finish())

    assert "".join(item.text for item in emissions if isinstance(item, TextEmission)) == text


def test_stream_parser_rejects_incomplete_reordered_message_json() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    assert parser.feed('{ "content":') == []
    with pytest.raises(MalformedToolCallError, match="echoed an OpenAI transcript"):
        parser.finish()


@pytest.mark.parametrize("separator", [" ", "  ", "\t\n", "_"])
def test_stream_parser_stops_mangled_openai_tool_calls_before_they_leak(
    separator: str,
) -> None:
    prefix = 'safe answer (reset|", '
    value = (
        prefix + f'"tool{separator}calls":"id":"call_1","type":"function","function":'
        '("name":"weather","arguments":("city":"Gdansk"))'
    )

    for split in range(1, len(value)):
        parser = ToolCallStreamParser(frozenset({"weather"}))
        emissions: list[object] = []

        with pytest.raises(MalformedToolCallError, match="echoed an OpenAI transcript"):
            _feed_parser_to_finish(parser, emissions, value[:split], value[split:])

        assert "".join(item.text for item in emissions if isinstance(item, TextEmission)) == prefix


@pytest.mark.parametrize(
    "mangled",
    [
        '"tool_calls":"call_1"',
        '"tool_calls": "id" : "call_1"',
        '"TOOL_CALLS":"ID":"CALL_1"',
        '"Tool Calls":"call_1"',
    ],
)
def test_stream_parser_stops_mangled_tool_calls_without_an_id_or_lower_case(
    mangled: str,
) -> None:
    prefix = "safe answer "
    value = f'{prefix}{mangled},"function":("name":"weather")'

    for split in range(1, len(value)):
        parser = ToolCallStreamParser(frozenset({"weather"}))
        emissions: list[object] = []

        with pytest.raises(MalformedToolCallError, match="echoed an OpenAI transcript"):
            _feed_parser_to_finish(parser, emissions, value[:split], value[split:])

        assert "".join(item.text for item in emissions if isinstance(item, TextEmission)) == prefix


@pytest.mark.parametrize("separator", ["_", " "])
def test_stream_parser_detects_mangled_tool_calls_after_plain_json(separator: str) -> None:
    # A non-tool-call assistant message followed by a mangled tool_calls echo
    # in the same chunk: _emit_plain_json disables message-JSON re-detection
    # in the trailing but mangled detection must stay active.
    plain = '{"role":"assistant","content":"hi"}'
    mangled = (
        f'"tool{separator}calls":"call_1","type":"function","function":'
        '("name":"weather","arguments":("city":"Gdansk"))'
    )
    value = f"{plain}{mangled}"

    for split in range(1, len(value)):
        parser = ToolCallStreamParser(frozenset({"weather"}))
        emissions: list[object] = []

        with pytest.raises(MalformedToolCallError, match="echoed an OpenAI transcript"):
            _feed_parser_to_finish(parser, emissions, value[:split], value[split:])

        assert "".join(item.text for item in emissions if isinstance(item, TextEmission)) == plain


def test_stream_parser_skips_mangled_detection_when_message_json_disabled() -> None:
    # Structured-output mode disables message-JSON detection entirely, which
    # also disables mangled tool-call detection.
    parser = ToolCallStreamParser(frozenset({"weather"}), parse_message_json=False)
    text = '"tool_calls":"call_1","function":("name":"weather")'

    emissions: list[object] = []
    for char in text:
        emissions.extend(parser.feed(char))
    emissions.extend(parser.finish())

    assert "".join(item.text for item in emissions if isinstance(item, TextEmission)) == text


def test_stream_parser_preserves_tool_calls_phrase_without_json_separator() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))
    text = 'The label "tool calls": remains ordinary prose.'

    emissions: list[object] = []
    for char in text:
        emissions.extend(parser.feed(char))
    emissions.extend(parser.finish())

    assert "".join(item.text for item in emissions if isinstance(item, TextEmission)) == text


@pytest.mark.parametrize(
    "value",
    [
        '"toolX',
        '"tool_bad',
        '"tool calls":"id":"call_x',
    ],
)
def test_mangled_tool_calls_prefix_rejects_non_prefixes(value: str) -> None:
    assert protocol._is_mangled_tool_calls_prefix(value) is False


@pytest.mark.parametrize(
    "text",
    [
        '{ "answer": invalid}',
        '{"role":"assistant","content":"done"} trailing',
    ],
)
def test_stream_parser_preserves_plain_json_candidates(text: str) -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    emissions = parser.feed(text)
    emissions.extend(parser.finish())

    assert "".join(item.text for item in emissions if isinstance(item, TextEmission)) == text


def test_stream_parser_preserves_nested_message_fields() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))
    text = '{"content":"ok","metadata":{"role":"tool","tool_calls":[]}}'

    assert parser.feed(text) == [TextEmission(text)]


def test_stream_parser_handles_many_adjacent_json_objects_iteratively() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))
    text = '{"role":"assistant","content":"ok"}' + '{"x":1}' * 1_500

    emissions = parser.feed(text)
    emissions.extend(parser.finish())

    assert "".join(item.text for item in emissions if isinstance(item, TextEmission)) == text


def test_stream_parser_keeps_a_bounded_partial_message_prefix() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))
    emissions: list[object] = []
    widest_tail = 0

    for _ in range(1_000):
        emissions.extend(parser.feed("{ "))
        widest_tail = max(widest_tail, len(parser._text_tail))
    emissions.extend(parser.finish())

    assert widest_tail <= 64
    assert (
        "".join(item.text for item in emissions if isinstance(item, TextEmission)) == "{ " * 1_000
    )


def test_stream_parser_releases_non_message_object_prefix() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    assert parser.feed("{x") == [TextEmission("{x")]


def test_stream_parser_decodes_message_json_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = protocol.parse_strict_json

    def counted_parse(value: str) -> Any:
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(protocol, "parse_strict_json", counted_parse)
    parser = ToolCallStreamParser(frozenset({"weather"}))
    message = (
        '{"content":null,"tool_calls":'
        '[{"function":{"name":"weather","arguments":{}}}],"role":"assistant"}'
    )

    for char in message:
        parser.feed(char)

    assert calls == 1


def test_stream_parser_drops_copied_tool_messages_after_assistant_json() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))
    assistant = (
        '{"role":"assistant","tool_calls":[{"id":"call_model","type":"function",'
        '"function":{"name":"weather","arguments":"{\\"city\\":\\"Gdansk\\"}"}}]}'
    )
    tool = '{"role":"tool","content":"Gdansk","tool_call_id":"call_model"}'

    emissions = parser.feed(assistant + tool + assistant)

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert emissions[0].name == "weather"
    assert json.loads(emissions[0].arguments) == {"city": "Gdansk"}
    assert parser.finish() == []


@pytest.mark.parametrize(
    "message",
    [
        '{"role":"tool","content":"copied","tool_call_id":"call_model"}',
        '{"role":"assistant","tool_calls":[{"function":{"name":"weather","arguments":"[]"}}]}',
        '{"role":"assistant","tool_calls":"not a list"}',
        '{"role":"assistant","tool_calls":[7]}',
        '{"role":"assistant","tool_calls":[{"function":"bad"}]}',
        '{"role":"assistant","tool_calls":[{"function":{"name":7}}]}',
        '{"role":"assistant","tool_calls":[{"function":{"name":"weather","arguments":7}}]}',
        '{"role":"assistant","tool_calls":[{"function":{"name":"weather",'
        '"arguments":"not json"}}]}',
        '{"role":"user","content":"copied"}',
    ],
)
def test_stream_parser_rejects_malformed_openai_message_json(message: str) -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    with pytest.raises(MalformedToolCallError, match="echoed an OpenAI transcript"):
        parser.feed(message)


def test_stream_parser_normalizes_blank_assistant_message_arguments() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    emissions = parser.feed(
        '{"role":"assistant","tool_calls":[{"function":{"name":"weather","arguments":""}}]}'
    )

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert json.loads(emissions[0].arguments) == {}


def test_stream_parser_accepts_object_assistant_message_arguments() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    emissions = parser.feed(
        '{"role":"assistant","tool_calls":[{"function":{"name":"weather","arguments":{}}}]}'
    )

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert json.loads(emissions[0].arguments) == {}


def test_stream_parser_rejects_duplicate_keys_in_assistant_message() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    with pytest.raises(MalformedToolCallError, match="echoed an OpenAI transcript"):
        parser.feed('{"role":"assistant","role":"assistant"}')


@pytest.mark.parametrize(
    "payload",
    [
        '{"role":"assistant","tool_calls":[{"function":{"name":"weather",'
        '"arguments":"{\\"value\\":NaN}"}}]}',
        '{"role":"assistant","tool_calls":[{"function":{"name":"weather",'
        '"arguments":"{\\"value\\":1e999}"}}]}',
    ],
)
def test_stream_parser_rejects_non_finite_assistant_arguments(payload: str) -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    with pytest.raises(MalformedToolCallError, match="echoed an OpenAI transcript"):
        parser.feed(payload)


@pytest.mark.parametrize("payload", ['{"role":NaN}', '{"role":1e999}'])
def test_stream_parser_rejects_non_finite_message_values(payload: str) -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    with pytest.raises(MalformedToolCallError, match="echoed an OpenAI transcript"):
        parser.feed(payload)


def test_stream_parser_preserves_non_message_json() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))
    payload = '{"role":1.5}'

    assert parser.feed(payload) == [TextEmission(payload)]


def test_stream_parser_repairs_lost_prefix_wrapped_form() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}), repair_lost_prefix=True)

    emissions = parser.feed(
        f'{TOOL_CALL_OPEN}weather","arguments":{{"city":"Gdansk"}}}}{TOOL_CALL_CLOSE}'
    )

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert json.loads(emissions[0].arguments) == {"city": "Gdansk"}


def test_stream_parser_repairs_lost_prefix_quoted_name() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}), repair_lost_prefix=True)

    emissions = parser.feed(f'{TOOL_CALL_OPEN}"weather","city":"Sopot"}}{TOOL_CALL_CLOSE}')

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert emissions[0].name == "weather"
    assert json.loads(emissions[0].arguments) == {"city": "Sopot"}


def test_stream_parser_lost_prefix_prefers_longest_tool_name() -> None:
    parser = ToolCallStreamParser(
        frozenset({"run", "run_in_terminal"}),
        repair_lost_prefix=True,
    )

    emissions = parser.feed(
        f'{TOOL_CALL_OPEN}run_in_terminal","command":"cd /app"}}{TOOL_CALL_CLOSE}'
    )

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert emissions[0].name == "run_in_terminal"
    assert json.loads(emissions[0].arguments) == {"command": "cd /app"}


def test_stream_parser_lost_prefix_keeps_scalar_wrapper_key() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}), repair_lost_prefix=True)

    emissions = parser.feed(f'{TOOL_CALL_OPEN}weather","arguments":5}}{TOOL_CALL_CLOSE}')

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert json.loads(emissions[0].arguments) == {"arguments": 5}


@pytest.mark.parametrize(
    "body",
    [
        'unknown","city":"Gdansk"}',
        'weather","city":"Gdansk"',
        'weather","city":"A","city":"B"}',
    ],
)
def test_stream_parser_lost_prefix_stays_fail_closed(body: str) -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}), repair_lost_prefix=True)

    with pytest.raises(MalformedToolCallError, match="invalid tool-call JSON"):
        parser.feed(f"{TOOL_CALL_OPEN}{body}{TOOL_CALL_CLOSE}")


@pytest.mark.parametrize(
    "payload",
    [
        "weather<arg_key>city",
        "weather<arg_key>city</arg_key>",
        "weather<arg_key></arg_key><arg_value>Gdansk</arg_value>",
        "weather<arg_key>city</arg_key><arg_value>A</arg_value>"
        "<arg_key>city</arg_key><arg_value>B</arg_value>",
        "<arg_key>city</arg_key><arg_value>Gdansk</arg_value>",
    ],
)
def test_stream_parser_rejects_unrepairable_arg_key_payloads(payload: str) -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    with pytest.raises(ProtocolError, match="invalid tool-call JSON"):
        parser.feed(f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE}")


def test_stream_parser_repairs_arg_key_value_missing_close_tag() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    emissions = parser.feed(
        f"{TOOL_CALL_OPEN}weather<arg_key>city</arg_key><arg_value>Gdansk{TOOL_CALL_CLOSE}"
    )

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert json.loads(emissions[0].arguments) == {"city": "Gdansk"}


def test_stream_parser_repairs_python_call_payload() -> None:
    parser = ToolCallStreamParser(frozenset({"skill_view"}))

    emissions = parser.feed(
        f'{TOOL_CALL_OPEN}skill_view({{"name":"apple-notes"}}){TOOL_CALL_CLOSE}'
    )

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert emissions[0].name == "skill_view"
    assert json.loads(emissions[0].arguments) == {"name": "apple-notes"}


def test_stream_parser_repairs_python_call_with_repeated_close_residue() -> None:
    parser = ToolCallStreamParser(frozenset({"skill_view"}))

    emissions = parser.feed(
        TOOL_CALL_OPEN + 'skill_view({"name":"apple-notes"})})' + TOOL_CALL_CLOSE
    )

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert emissions[0].name == "skill_view"
    assert json.loads(emissions[0].arguments) == {"name": "apple-notes"}


@pytest.mark.parametrize(
    "payload",
    [
        'unknown_view({"name":"apple-notes"})',
        'skill_view({"name":"apple-notes")',
        'skill_view({"name":})',
        'skill_view(["apple-notes"])',
        "skill_view()",
        'print("hello")',
        'skill_view({"name":"apple-notes"}) and then I read it',
        'skill_view({"name":"apple-notes"}',
        'skill_view({"name":"apple-notes"})skill_view({"name":',
        '})skill_view({"name":"apple-notes"})',
        f'skill_view({{"name":"apple-notes"}}){TOOL_CALL_OPEN}',
    ],
)
def test_stream_parser_rejects_unrepairable_python_call_payloads(payload: str) -> None:
    parser = ToolCallStreamParser(frozenset({"skill_view"}))

    with pytest.raises(ProtocolError, match="invalid tool-call JSON"):
        parser.feed(f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE}")


def test_stream_parser_repairs_bare_call_payload() -> None:
    parser = ToolCallStreamParser(frozenset({"skill_view"}))

    emissions = parser.feed(f'{TOOL_CALL_OPEN}skill_view{{"name":"apple-notes"}}{TOOL_CALL_CLOSE}')

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert emissions[0].name == "skill_view"
    assert json.loads(emissions[0].arguments) == {"name": "apple-notes"}


@pytest.mark.parametrize(
    "payload",
    [
        'unknown_view{"name":"apple-notes"}',
        'skill_view{"name":"apple-notes"',
        'skill_view{"name":"a","name":"b"}',
    ],
)
def test_stream_parser_rejects_unrepairable_bare_call_payloads(payload: str) -> None:
    parser = ToolCallStreamParser(frozenset({"skill_view"}))

    with pytest.raises(ProtocolError, match="invalid tool-call JSON"):
        parser.feed(f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE}")


def test_stream_parser_repairs_arg_key_json_residue_payload() -> None:
    parser = ToolCallStreamParser(frozenset({"skill_view"}))

    emissions = parser.feed(
        f'{TOOL_CALL_OPEN}skill_view<arg_key>name":"apple-notes"}}{TOOL_CALL_CLOSE}'
    )

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert emissions[0].name == "skill_view"
    assert json.loads(emissions[0].arguments) == {"name": "apple-notes"}


def test_stream_parser_repairs_arg_key_payload_with_leading_separator() -> None:
    parser = ToolCallStreamParser(frozenset({"skill_view"}))

    emissions = parser.feed(
        f'{TOOL_CALL_OPEN}{TOOL_CALL_OPEN}skill_view<arg_key>name":"apple-notes"}}{TOOL_CALL_CLOSE}'
    )

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert emissions[0].name == "skill_view"
    assert json.loads(emissions[0].arguments) == {"name": "apple-notes"}


def test_stream_parser_repairs_arg_key_mangled_two_call_payload() -> None:
    parser = ToolCallStreamParser(
        frozenset({"skill_view", "terminal"}),
        max_tool_calls=2,
    )
    payload = (
        'skill_view<arg_key>name":"apple-notes"}'
        "<tool_call>"
        'terminal<arg_key>command":"which memo && memo --help 2>&1 | head -40</arg_value>'
    )

    emissions = parser.feed(f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE}")

    calls = [emission for emission in emissions if isinstance(emission, ToolCallEmission)]
    assert len(calls) == 2
    assert calls[0].name == "skill_view"
    assert json.loads(calls[0].arguments) == {"name": "apple-notes"}
    assert calls[1].name == "terminal"
    assert json.loads(calls[1].arguments) == {
        "command": "which memo && memo --help 2>&1 | head -40"
    }


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ('skill_view<arg_key>count":"3}', {"count": 3}),
        ('skill_view<arg_key>name":"abc"', {"name": "abc"}),
    ],
)
def test_stream_parser_repairs_arg_key_value_endings(
    payload: str,
    expected: dict[str, Any],
) -> None:
    parser = ToolCallStreamParser(frozenset({"skill_view"}))

    emissions = parser.feed(f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE}")

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert json.loads(emissions[0].arguments) == expected


@pytest.mark.parametrize(
    "payload",
    [
        'unknown<arg_key>name":"apple-notes"}',
        'skill_view<arg_key>na me":"apple-notes"}',
        'skill_view<arg_key>name":"a"}<arg_key>name":"b"}',
        'skill_view<arg_key>name":"a"}</arg_value> junk',
        'skill_view<arg_key>name":"a"}<tool_call>unknown<arg_key>k":"v"}',
    ],
)
def test_stream_parser_rejects_unrepairable_arg_key_repair_payloads(payload: str) -> None:
    parser = ToolCallStreamParser(frozenset({"skill_view"}))

    with pytest.raises(ProtocolError, match="invalid tool-call JSON"):
        parser.feed(f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE}")


def test_stream_parser_repairs_fenced_payload() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    emissions = parser.feed(
        f"{TOOL_CALL_OPEN}```json\n"
        '{"name":"weather","arguments":{"city":"Sopot"}}\n'
        f"```{TOOL_CALL_CLOSE}"
    )

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert json.loads(emissions[0].arguments) == {"city": "Sopot"}


def test_stream_parser_repairs_unterminated_fence() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    emissions = parser.feed(
        f'{TOOL_CALL_OPEN}```\n{{"name":"weather","arguments":{{"city":"Sopot"}}}}{TOOL_CALL_CLOSE}'
    )

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)


def test_stream_parser_keeps_single_line_fence_unrepaired() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    with pytest.raises(ProtocolError, match="invalid tool-call JSON"):
        parser.feed(f"{TOOL_CALL_OPEN}```json{TOOL_CALL_CLOSE}")


def test_stream_parser_accepts_objects_packed_in_one_marker_pair() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}), max_tool_calls=2)

    emissions = parser.feed(
        f"{TOOL_CALL_OPEN}"
        '{"name":"weather","arguments":{"city":"Gdansk"}}\n '
        '{"name":"weather","arguments":{"city":"Sopot"}}'
        f"{TOOL_CALL_CLOSE}"
    )

    calls = [emission for emission in emissions if isinstance(emission, ToolCallEmission)]
    assert [json.loads(call.arguments)["city"] for call in calls] == ["Gdansk", "Sopot"]


def test_stream_parser_rejects_packed_objects_past_the_cap() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}), max_tool_calls=1)
    payload = '{"name":"weather","arguments":{}}{"name":"weather","arguments":{}}'

    with pytest.raises(MalformedToolCallError) as excinfo:
        parser.feed(f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE}")

    assert str(excinfo.value) == "more tool calls than the configured maximum"
    assert excinfo.value.tool_name == "weather"
    assert excinfo.value.payload_bytes == len(payload.encode("utf-8"))


def _repair_recorder() -> tuple[list[tuple[str, str, str | None]], Any]:
    reported: list[tuple[str, str, str | None]] = []

    def record(event: str, *, dialect: str, variant: str | None = None) -> None:
        reported.append((event, dialect, variant))

    return reported, record


def _feed_parts(parser: ToolCallStreamParser, *parts: str) -> None:
    for part in parts:
        parser.feed(part)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ('weather{"city":"Gdansk"}', [("tool_call.repaired", "native", "bare_call")]),
        (
            '{"name":"weather","arguments":{}}{"name":"weather","arguments":{}}',
            [("tool_call.repaired", "native", "packed_objects")],
        ),
        (
            '[{"name":"weather","arguments":{}}]',
            [("tool_call.repaired", "native", "json_array")],
        ),
    ],
)
def test_stream_parser_reports_the_repair_variant_it_used(
    payload: str,
    expected: list[tuple[str, str, str | None]],
) -> None:
    reported, record = _repair_recorder()
    parser = ToolCallStreamParser(
        frozenset({"weather"}),
        max_tool_calls=2,
        record_repair=record,
    )

    parser.feed(f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE}")

    assert reported == expected


def test_stream_parser_reports_an_unparsed_payload() -> None:
    reported, record = _repair_recorder()
    parser = ToolCallStreamParser(frozenset({"weather"}), record_repair=record)

    with pytest.raises(MalformedToolCallError):
        parser.feed(f"{TOOL_CALL_OPEN}not json{TOOL_CALL_CLOSE}")

    assert reported == [("tool_call.unparsed", "native", None)]


def test_stream_parser_reports_the_repair_before_the_call_limit() -> None:
    reported, record = _repair_recorder()
    parser = ToolCallStreamParser(
        frozenset({"weather"}),
        max_tool_calls=1,
        record_repair=record,
    )

    with pytest.raises(MalformedToolCallError):
        parser.feed(
            f"{TOOL_CALL_OPEN}"
            '{"name":"weather","arguments":{}}{"name":"weather","arguments":{}}'
            f"{TOOL_CALL_CLOSE}"
        )

    assert reported == [
        ("tool_call.repaired", "native", "packed_objects"),
        ("tool_call.over_limit", "native", None),
    ]


def test_stream_parser_reports_a_same_feed_trailing_call_over_limit() -> None:
    reported, record = _repair_recorder()
    parser = ToolCallStreamParser(
        frozenset({"weather"}),
        max_tool_calls=1,
        record_repair=record,
    )

    with pytest.raises(ProtocolError, match="unexpected text after tool call"):
        parser.feed(_tool_call("weather", "{}") + _tool_call("weather", "{}"))

    assert reported == [("tool_call.over_limit", "native", None)]


@pytest.mark.parametrize("dialect", MARKER_DIALECTS, ids=lambda dialect: dialect.name)
def test_stream_parser_reports_the_trailing_call_dialect_over_limit(
    dialect: MarkerDialect,
) -> None:
    reported, record = _repair_recorder()
    parser = ToolCallStreamParser(
        frozenset({"weather"}),
        max_tool_calls=1,
        record_repair=record,
    )
    parser.feed(_tool_call("weather", "{}"))

    with pytest.raises(
        ProtocolError,
        match=rf"unexpected text after tool call \({dialect.name} dialect\)",
    ):
        parser.feed(dialect.open_marker)

    assert reported == [("tool_call.over_limit", dialect.name, None)]


def test_stream_parser_reports_a_split_trailing_call_over_limit() -> None:
    second = _tool_call("weather", "{}")

    for split in range(1, len(second)):
        reported, record = _repair_recorder()
        parser = ToolCallStreamParser(
            frozenset({"weather"}),
            max_tool_calls=1,
            record_repair=record,
        )
        parser.feed(_tool_call("weather", "{}"))

        with pytest.raises(ProtocolError, match="unexpected text after tool call"):
            _feed_parts(parser, second[:split], second[split:])

        assert reported == [("tool_call.over_limit", "native", None)]


def test_stream_parser_rejects_an_incomplete_trailing_marker_after_limit() -> None:
    reported, record = _repair_recorder()
    parser = ToolCallStreamParser(
        frozenset({"weather"}),
        max_tool_calls=1,
        record_repair=record,
    )
    parser.feed(_tool_call("weather", "{}"))
    parser.feed(_tool_call("weather", "{}")[: len(TOOL_CALL_OPEN) - 1])

    with pytest.raises(ProtocolError, match="unexpected text after tool call"):
        parser.finish()

    assert reported == []


def test_stream_parser_does_not_report_prose_before_a_trailing_call() -> None:
    second = _tool_call("weather", "{}")

    for trailing in (f"prose {second}", f"prose {second[: len(TOOL_CALL_OPEN) - 1]}"):
        reported, record = _repair_recorder()
        parser = ToolCallStreamParser(
            frozenset({"weather"}),
            max_tool_calls=1,
            record_repair=record,
        )
        parser.feed(_tool_call("weather", "{}"))

        with pytest.raises(ProtocolError, match="unexpected text after tool call"):
            parser.feed(trailing)

        assert reported == []


def test_stream_parser_keeps_diagnostics_when_an_unclosed_payload_packs_too_many_calls() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}), max_tool_calls=1)
    payload = f'weather{{"city":"Gdansk"}}{TOOL_CALL_OPEN}weather{{"city":"Sopot"}}'
    parser.feed(f"{TOOL_CALL_OPEN}{payload}")

    with pytest.raises(MalformedToolCallError) as excinfo:
        parser.finish()

    assert excinfo.value.tool_name == "weather"
    assert excinfo.value.payload_bytes == len(payload.encode("utf-8"))


def test_stream_parser_keeps_diagnostics_when_a_transcript_echo_packs_too_many_calls() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}), max_tool_calls=1)
    calls = [
        {"id": "call_1", "type": "function", "function": {"name": "weather", "arguments": "{}"}},
        {"id": "call_2", "type": "function", "function": {"name": "weather", "arguments": "{}"}},
    ]

    with pytest.raises(MalformedToolCallError) as excinfo:
        parser.feed(json.dumps({"role": "assistant", "tool_calls": calls}))

    assert str(excinfo.value) == "more tool calls than the configured maximum"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("weather", {}),
        ('weather\n{"city":"Gdansk"}', {"city": "Gdansk"}),
    ],
)
def test_stream_parser_repairs_name_outside_the_json_object(
    payload: str,
    expected: dict[str, str],
) -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    emissions = parser.feed(f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE}")

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert json.loads(emissions[0].arguments) == expected


@pytest.mark.parametrize(
    "payload",
    [
        "shell",
        'shell\n{"cmd":"ls"}',
        "weather\nnot-json",
        "weather\n[]",
    ],
)
def test_stream_parser_refuses_to_repair_unnamed_payloads(payload: str) -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    with pytest.raises(ProtocolError, match="invalid tool-call JSON"):
        parser.feed(f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE}")


_PIPE_TAG_OPEN = "<|open|>tools<|sep|>"
_PIPE_TAG_CLOSE = "<|close|>tools<|sep|>"
_PIPE_TAG_CALL = (
    '<|open|>call tool="weather" index="1"<|sep|>'
    '<|open|>argument key="city" type="string"<|sep|>Gdańsk<|close|>argument<|sep|>'
    "<|close|>call<|sep|>"
)


def test_stream_parser_translates_pipe_tag_template_tokens() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    emissions = parser.feed(f"{_PIPE_TAG_OPEN}{_PIPE_TAG_CALL}{_PIPE_TAG_CLOSE}")
    emissions.extend(parser.finish())

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert emissions[0].name == "weather"
    assert json.loads(emissions[0].arguments) == {"city": "Gdańsk"}


def test_stream_parser_translates_pipe_tag_tokens_inside_markers() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    emissions = parser.feed(f"{TOOL_CALL_OPEN}{_PIPE_TAG_CALL}{TOOL_CALL_CLOSE}")
    emissions.extend(parser.finish())

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert json.loads(emissions[0].arguments) == {"city": "Gdańsk"}


def test_stream_parser_keeps_text_before_pipe_tag_tokens() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    emissions = parser.feed(f"Checking now. {_PIPE_TAG_OPEN}{_PIPE_TAG_CALL}{_PIPE_TAG_CLOSE}")
    emissions.extend(parser.finish())

    assert [type(emission) for emission in emissions] == [TextEmission, ToolCallEmission]
    assert isinstance(emissions[0], TextEmission)
    assert emissions[0].text == "Checking now. "


def test_stream_parser_translates_pipe_tag_tokens_across_chunks() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))
    stream = f"{_PIPE_TAG_OPEN}{_PIPE_TAG_CALL}{_PIPE_TAG_CLOSE}"

    emissions: list[object] = []
    for start in range(0, len(stream), 5):
        emissions.extend(parser.feed(stream[start : start + 5]))
    emissions.extend(parser.finish())

    assert [type(emission) for emission in emissions] == [ToolCallEmission]


def test_stream_parser_translates_multiple_pipe_tag_calls() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}), max_tool_calls=2)

    emissions = parser.feed(
        f"{_PIPE_TAG_OPEN}"
        '<|open|>call tool="weather" index="1"<|sep|>'
        '<|open|>argument key="city" type="string"<|sep|>Gdansk<|close|>argument<|sep|>'
        "<|close|>call<|sep|>"
        '<|open|>call tool="weather" index="2"<|sep|>'
        '<|open|>argument key="city" type="string"<|sep|>Sopot<|close|>argument<|sep|>'
        "<|close|>call<|sep|>"
        f"{_PIPE_TAG_CLOSE}"
    )
    emissions.extend(parser.finish())

    calls = [emission for emission in emissions if isinstance(emission, ToolCallEmission)]
    assert [json.loads(call.arguments)["city"] for call in calls] == ["Gdansk", "Sopot"]


def test_stream_parser_recovers_pipe_tag_call_without_close_marker() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    parser.feed(f"{_PIPE_TAG_OPEN}{_PIPE_TAG_CALL}")
    emissions = parser.finish()

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)


@pytest.mark.parametrize(
    ("declared_type", "raw", "expected"),
    [
        ('key="city" type="string"', "123", "123"),
        ('key="days" type="number"', "3", 3),
        ('key="metric" type="boolean"', "true", True),
        ('key="filter" type="object"', '{"unit":"c"}', {"unit": "c"}),
        ('key="days"', "7", 7),
    ],
)
def test_stream_parser_coerces_pipe_tag_argument_values(
    declared_type: str,
    raw: str,
    expected: object,
) -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    emissions = parser.feed(
        f"{_PIPE_TAG_OPEN}"
        '<|open|>call tool="weather" index="1"<|sep|>'
        f"<|open|>argument {declared_type}<|sep|>{raw}<|close|>argument<|sep|>"
        "<|close|>call<|sep|>"
        f"{_PIPE_TAG_CLOSE}"
    )

    assert isinstance(emissions[0], ToolCallEmission)
    assert list(json.loads(emissions[0].arguments).values()) == [expected]


def test_stream_parser_preserves_pipe_tag_string_whitespace() -> None:
    parser = ToolCallStreamParser(frozenset({"write_file"}))

    emissions = parser.feed(
        f"{_PIPE_TAG_OPEN}"
        '<|open|>call tool="write_file" index="1"<|sep|>'
        '<|open|>argument key="file content" type="string"<|sep|>'
        "  indented  "
        "<|close|>argument<|sep|><|close|>call<|sep|>"
        f"{_PIPE_TAG_CLOSE}"
    )

    assert isinstance(emissions[0], ToolCallEmission)
    assert json.loads(emissions[0].arguments) == {"file content": "  indented  "}


def test_stream_parser_accepts_pipe_tag_call_without_arguments() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    emissions = parser.feed(
        f"{_PIPE_TAG_OPEN}"
        '<|open|>call tool="weather" index="1"<|sep|><|close|>call<|sep|>'
        f"{_PIPE_TAG_CLOSE}"
    )

    assert isinstance(emissions[0], ToolCallEmission)
    assert json.loads(emissions[0].arguments) == {}


@pytest.mark.parametrize(
    "call",
    [
        '<|open|>call tool="weather" index="1"',
        '<|open|>call index="1"<|sep|><|close|>call',
        '<|open|>call tool=""<|sep|><|close|>call',
        '<|open|>call tool="weather<|sep|><|close|>call',
        '<|open|>call tool="weather"<|sep|><|open|>argument key="city" type="string"',
        '<|open|>call tool="weather"<|sep|><|open|>argument key="city" type="string"<|sep|>Gdansk',
        '<|open|>call tool="weather"<|sep|><|open|>argument key=""<|sep|>Gdansk<|close|>argument',
        '<|open|>call tool="weather"<|sep|>'
        '<|open|>argument key="city"<|sep|>A<|close|>argument'
        '<|open|>argument key="city"<|sep|>B<|close|>argument',
        'garbage<|open|>call tool="weather"<|sep|><|close|>call',
        '<|open|>call nottool="weather"<|sep|><|close|>call',
        '<|open|>call tool="weather" tool="weather"<|sep|><|close|>call',
        '<|open|>call tool="weather" mode="fast"<|sep|><|close|>call',
        '<|open|>call tool="weather" garbage index="1"<|sep|><|close|>call',
        '<|open|>call tool="weather"<|sep|>',
        '<|open|>call tool="weather"<|sep|>garbage<|close|>call',
        '<|open|>call tool="weather"<|sep|><|close|>call<|sep|>garbage',
        '<|open|>call tool="weather"<|sep|>garbage'
        '<|open|>argument key="city"<|sep|>Gdansk<|close|>argument'
        "<|close|>call",
        '<|open|>call tool="weather"<|sep|>'
        '<|open|>argument key="city"<|close|>argument<|close|>call',
        '<|open|>call tool="weather"<|sep|>'
        '<|open|>argument bogus="city"<|sep|>Gdansk<|close|>argument'
        "<|close|>call",
        '<|open|>call tool="weather"<|sep|>'
        '<|open|>argument type="string"<|sep|>Gdansk<|close|>argument'
        "<|close|>call",
        '<|open|>call tool="weather"<|sep|>'
        '<|open|>argument key="city"<|sep|>Gdansk<|close|>argument'
        "garbage<|close|>call",
    ],
)
def test_stream_parser_rejects_unrepairable_pipe_tag_payloads(call: str) -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    with pytest.raises(ProtocolError, match="invalid tool-call JSON"):
        parser.feed(f"{TOOL_CALL_OPEN}{call}{TOOL_CALL_CLOSE}")


def test_stream_parser_rejects_pipe_tag_call_to_unknown_tool() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    with pytest.raises(ProtocolError, match="tool 'shell' is not available"):
        parser.feed(
            f"{_PIPE_TAG_OPEN}"
            '<|open|>call tool="shell" index="1"<|sep|><|close|>call<|sep|>'
            f"{_PIPE_TAG_CLOSE}"
        )


def test_stream_parser_rejects_prose_after_pipe_tag_call() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    with pytest.raises(ProtocolError, match="unexpected text after tool call"):
        parser.feed(f"{_PIPE_TAG_OPEN}{_PIPE_TAG_CALL}{_PIPE_TAG_CLOSE} and done")


def test_stream_parser_names_the_dialect_in_trailing_text_error() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    with pytest.raises(
        ProtocolError,
        match=r"unexpected text after tool call \(pipe_tag dialect\)",
    ):
        parser.feed(f"{_PIPE_TAG_OPEN}{_PIPE_TAG_CALL}{_PIPE_TAG_CLOSE} and done")


def test_every_trailing_text_error_names_the_native_dialect() -> None:
    expected = r"unexpected text after tool call \(native dialect\)"

    capped = ToolCallStreamParser(frozenset({"weather"}), max_tool_calls=1)
    capped.feed(_tool_call("weather", "{}"))
    with pytest.raises(ProtocolError, match=expected):
        capped.feed("trailing")

    over_cap = ToolCallStreamParser(frozenset({"weather"}), max_tool_calls=1)
    with pytest.raises(ProtocolError, match=expected):
        over_cap.feed(_tool_call("weather", "{}") + "trailing")

    between = ToolCallStreamParser(frozenset({"weather"}), max_tool_calls=3)
    between.feed(_tool_call("weather", "{}"))
    with pytest.raises(ProtocolError, match=expected):
        between.feed("and now some prose")

    at_finish = ToolCallStreamParser(frozenset({"weather"}), max_tool_calls=3)
    at_finish.feed(_tool_call("weather", "{}"))
    at_finish.feed("  <tool")
    with pytest.raises(ProtocolError, match=expected):
        at_finish.finish()


def test_stream_parser_ignores_pipe_tag_control_tokens_after_the_call() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    emissions = parser.feed(f"{_PIPE_TAG_OPEN}{_PIPE_TAG_CALL}{_PIPE_TAG_CLOSE}")
    emissions.extend(parser.feed("<|close|><|sep|>\n"))
    emissions.extend(parser.finish())

    assert [type(emission) for emission in emissions] == [ToolCallEmission]


def test_stream_parser_flushes_a_held_control_token_tail() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}), max_tool_calls=2)

    emissions = parser.feed(f"{_PIPE_TAG_OPEN}{_PIPE_TAG_CALL}{_PIPE_TAG_CLOSE}")
    # The trailing "<|" could still open another marker, so it stays buffered
    # until finish decides it was only scaffolding.
    emissions.extend(parser.feed("<|sep|>\n<|"))
    emissions.extend(parser.finish())

    assert [type(emission) for emission in emissions] == [ToolCallEmission]


def test_stream_parser_buffers_a_split_control_token_between_calls() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}), max_tool_calls=2)

    emissions = parser.feed(f"{_PIPE_TAG_OPEN}{_PIPE_TAG_CALL}{_PIPE_TAG_CLOSE}<|se")
    emissions.extend(parser.feed("p|>"))
    emissions.extend(parser.finish())

    assert [type(emission) for emission in emissions] == [ToolCallEmission]


_KIMI_SECTION_OPEN = "<|tool_calls_section_begin|>"
_KIMI_SECTION_CLOSE = "<|tool_calls_section_end|>"
_KIMI_CALL = (
    '<|tool_call_begin|>functions.weather:0<|tool_call_argument_begin|>{"city":"Gdansk"}'
    "<|tool_call_end|>"
)
_HARMONY_OPEN = "<|channel|>commentary to="


def test_stream_parser_translates_kimi_sections() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    emissions = parser.feed(f"{_KIMI_SECTION_OPEN}{_KIMI_CALL}{_KIMI_SECTION_CLOSE}")
    emissions.extend(parser.finish())

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert emissions[0].name == "weather"
    assert json.loads(emissions[0].arguments) == {"city": "Gdansk"}


def test_stream_parser_recovers_kimi_call_with_partial_section_close() -> None:
    for split in range(1, len(_KIMI_SECTION_CLOSE)):
        parser = ToolCallStreamParser(frozenset({"weather"}))

        assert (
            parser.feed(
                f"{_KIMI_SECTION_OPEN}{_KIMI_CALL}{_KIMI_SECTION_CLOSE[:split]}",
            )
            == []
        )
        emissions = parser.finish()

        assert len(emissions) == 1
        assert isinstance(emissions[0], ToolCallEmission)
        assert json.loads(emissions[0].arguments) == {"city": "Gdansk"}


def test_stream_parser_translates_multiple_kimi_calls() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}), max_tool_calls=2)

    emissions = parser.feed(
        f"{_KIMI_SECTION_OPEN}{_KIMI_CALL}"
        '<|tool_call_begin|>functions.weather:1<|tool_call_argument_begin|>{"city":"Sopot"}'
        f"<|tool_call_end|>{_KIMI_SECTION_CLOSE}"
    )
    emissions.extend(parser.finish())

    calls = [emission for emission in emissions if isinstance(emission, ToolCallEmission)]
    assert [json.loads(call.arguments)["city"] for call in calls] == ["Gdansk", "Sopot"]


def test_stream_parser_translates_kimi_sections_across_chunks() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))
    stream = f"{_KIMI_SECTION_OPEN}{_KIMI_CALL}{_KIMI_SECTION_CLOSE}<|im_end|>"

    emissions: list[object] = []
    for start in range(0, len(stream), 7):
        emissions.extend(parser.feed(stream[start : start + 7]))
    emissions.extend(parser.finish())

    assert [type(emission) for emission in emissions] == [ToolCallEmission]


@pytest.mark.parametrize(
    "call",
    [
        "<|tool_call_begin|>functions.weather:0",
        '<|tool_call_begin|>functions.weather:0<|tool_call_argument_begin|>{"city":"X"}',
        "<|tool_call_begin|>functions.:0<|tool_call_argument_begin|>{}<|tool_call_end|>",
        "<|tool_call_begin|>functions.weather:0<|tool_call_argument_begin|>[]<|tool_call_end|>",
        "<|tool_call_begin|>functions.weather:0<|tool_call_argument_begin|>{oops}<|tool_call_end|>",
        f"garbage{_KIMI_CALL}",
        f"{_KIMI_CALL}garbage",
    ],
)
def test_stream_parser_rejects_unrepairable_kimi_payloads(call: str) -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    with pytest.raises(ProtocolError, match="invalid tool-call JSON"):
        parser.feed(f"{TOOL_CALL_OPEN}{call}{TOOL_CALL_CLOSE}")


def test_stream_parser_accepts_kimi_call_without_namespace_or_arguments() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    emissions = parser.feed(
        f"{_KIMI_SECTION_OPEN}"
        "<|tool_call_begin|>weather<|tool_call_argument_begin|><|tool_call_end|>"
        f"{_KIMI_SECTION_CLOSE}"
    )

    assert isinstance(emissions[0], ToolCallEmission)
    assert emissions[0].name == "weather"
    assert json.loads(emissions[0].arguments) == {}


def test_stream_parser_translates_harmony_commentary() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    emissions = parser.feed(
        f'{_HARMONY_OPEN}functions.weather <|constrain|>json<|message|>{{"city":"Hel"}}<|call|>'
    )
    emissions.extend(parser.finish())

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert emissions[0].name == "weather"
    assert json.loads(emissions[0].arguments) == {"city": "Hel"}


def test_stream_parser_translates_harmony_without_constrain_tag() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    emissions = parser.feed(f"{_HARMONY_OPEN}functions.weather<|message|>{{}}<|call|>")
    emissions.extend(parser.feed("<|end|>"))
    emissions.extend(parser.finish())

    assert [type(emission) for emission in emissions] == [ToolCallEmission]
    assert isinstance(emissions[0], ToolCallEmission)
    assert json.loads(emissions[0].arguments) == {}


@pytest.mark.parametrize(
    "payload",
    [
        "functions.weather",
        'functions.<|message|>{"city":"X"}',
        "functions.weather<|message|>[1]",
        "functions.weather<|message|>{oops}",
    ],
)
def test_stream_parser_rejects_unrepairable_harmony_payloads(payload: str) -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    with pytest.raises(ProtocolError, match="invalid tool-call JSON"):
        parser.feed(f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE}")


# DeepSeek spells these with U+FF5C fullwidth vertical line and U+2581 lower one
# eighth block, never ASCII "|" or "_". Samples below are verbatim from
# vllm/tests/tool_parsers/test_deepseekv3_tool_parser.py and
# test_deepseekv31_tool_parser.py at vLLM bb3b61f2fd2333ab165ebaba13f133db4210b9f2,
# cross-checked against the chat_template of deepseek-ai/DeepSeek-V3 (e815299)
# and DeepSeek-V3.1 (c0781d0).
_DS_BAR = "\uff5c"
_DS_BLOCK = "\u2581"
_DEEPSEEK_OPEN = f"<{_DS_BAR}tool{_DS_BLOCK}calls{_DS_BLOCK}begin{_DS_BAR}>"
_DEEPSEEK_CLOSE = f"<{_DS_BAR}tool{_DS_BLOCK}calls{_DS_BLOCK}end{_DS_BAR}>"
_DEEPSEEK_CALL_OPEN = f"<{_DS_BAR}tool{_DS_BLOCK}call{_DS_BLOCK}begin{_DS_BAR}>"
_DEEPSEEK_CALL_CLOSE = f"<{_DS_BAR}tool{_DS_BLOCK}call{_DS_BLOCK}end{_DS_BAR}>"
_DEEPSEEK_SEP = f"<{_DS_BAR}tool{_DS_BLOCK}sep{_DS_BAR}>"
_DEEPSEEK_V3_CALL = (
    f"{_DEEPSEEK_CALL_OPEN}function{_DEEPSEEK_SEP}get_weather\n"
    '```json\n{"city": "Tokyo", "unit": "celsius"}\n```'
    f"{_DEEPSEEK_CALL_CLOSE}"
)
_DEEPSEEK_V31_CALL = (
    f'{_DEEPSEEK_CALL_OPEN}get_weather{_DEEPSEEK_SEP}{{"city": "Tokyo"}}{_DEEPSEEK_CALL_CLOSE}'
)


def test_stream_parser_translates_a_deepseek_v3_fenced_call() -> None:
    parser = ToolCallStreamParser(frozenset({"get_weather"}))

    emissions = parser.feed(f"{_DEEPSEEK_OPEN}{_DEEPSEEK_V3_CALL}{_DEEPSEEK_CLOSE}")
    emissions.extend(parser.finish())

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert emissions[0].name == "get_weather"
    assert json.loads(emissions[0].arguments) == {"city": "Tokyo", "unit": "celsius"}


def test_stream_parser_translates_a_deepseek_v31_call_after_prose() -> None:
    parser = ToolCallStreamParser(frozenset({"get_weather"}))

    emissions = parser.feed(f"normal text{_DEEPSEEK_OPEN}{_DEEPSEEK_V31_CALL}{_DEEPSEEK_CLOSE}")
    emissions.extend(parser.finish())

    assert [type(emission) for emission in emissions] == [TextEmission, ToolCallEmission]
    assert isinstance(emissions[1], ToolCallEmission)
    assert emissions[1].name == "get_weather"
    assert json.loads(emissions[1].arguments) == {"city": "Tokyo"}


@pytest.mark.parametrize("separator", ["", "\n"])
def test_stream_parser_translates_parallel_deepseek_calls(separator: str) -> None:
    parser = ToolCallStreamParser(frozenset({"get_weather", "search_hotels"}), max_tool_calls=2)
    second = (
        f"{_DEEPSEEK_CALL_OPEN}function{_DEEPSEEK_SEP}search_hotels\n"
        '```json\n{"location": "Tokyo"}\n```'
        f"{_DEEPSEEK_CALL_CLOSE}"
    )

    emissions = parser.feed(
        f"{_DEEPSEEK_OPEN}{_DEEPSEEK_V3_CALL}{separator}{second}{_DEEPSEEK_CLOSE}"
    )
    emissions.extend(parser.finish())

    calls = [emission for emission in emissions if isinstance(emission, ToolCallEmission)]
    assert [call.name for call in calls] == ["get_weather", "search_hotels"]


def test_stream_parser_translates_a_deepseek_call_with_empty_arguments() -> None:
    parser = ToolCallStreamParser(frozenset({"get_current_time"}))

    emissions = parser.feed(
        f"{_DEEPSEEK_OPEN}{_DEEPSEEK_CALL_OPEN}function{_DEEPSEEK_SEP}get_current_time\n"
        "```json\n{}\n```"
        f"{_DEEPSEEK_CALL_CLOSE}{_DEEPSEEK_CLOSE}"
    )

    assert isinstance(emissions[0], ToolCallEmission)
    assert json.loads(emissions[0].arguments) == {}


def test_stream_parser_recovers_a_deepseek_section_without_its_close() -> None:
    # The V3 template only emits the section close for parallel calls, so a
    # single call ends at EOS instead.
    parser = ToolCallStreamParser(frozenset({"get_weather"}))

    emissions = parser.feed(f"{_DEEPSEEK_OPEN}{_DEEPSEEK_V3_CALL}")
    emissions.extend(parser.finish())

    assert [type(emission) for emission in emissions] == [ToolCallEmission]


def test_stream_parser_translates_deepseek_calls_across_chunks() -> None:
    parser = ToolCallStreamParser(frozenset({"get_weather"}))
    stream = (
        f"{_DEEPSEEK_OPEN}{_DEEPSEEK_V31_CALL}{_DEEPSEEK_CLOSE}"
        f"<{_DS_BAR}end{_DS_BLOCK}of{_DS_BLOCK}sentence{_DS_BAR}>"
    )

    emissions: list[object] = []
    for start in range(0, len(stream), 5):
        emissions.extend(parser.feed(stream[start : start + 5]))
    emissions.extend(parser.finish())

    assert [type(emission) for emission in emissions] == [ToolCallEmission]


def test_stream_parser_rejects_prose_after_a_deepseek_section() -> None:
    parser = ToolCallStreamParser(frozenset({"get_weather"}))

    with pytest.raises(ProtocolError, match="unexpected text after tool call"):
        parser.feed(
            f"{_DEEPSEEK_OPEN}{_DEEPSEEK_V31_CALL}{_DEEPSEEK_CLOSE} some suffix text",
        )


@pytest.mark.parametrize(
    "payload",
    [
        f"{_DEEPSEEK_CALL_OPEN}function{_DEEPSEEK_SEP}get_weather\n"
        '```json\n{"city": "Tokyo"\n```'
        f"{_DEEPSEEK_CALL_CLOSE}",
        f'function{_DEEPSEEK_SEP}get_weather\n```json\n{{"city": "Tokyo"}}\n```',
        f'{_DEEPSEEK_CALL_OPEN}get_weather{_DEEPSEEK_SEP}{{"city": "Tokyo"}}',
        f'{_DEEPSEEK_CALL_OPEN}get_weather{{"city": "Tokyo"}}{_DEEPSEEK_CALL_CLOSE}',
        f'{_DEEPSEEK_CALL_OPEN}{_DEEPSEEK_SEP}{{"city": "Tokyo"}}{_DEEPSEEK_CALL_CLOSE}',
        f"{_DEEPSEEK_CALL_OPEN}get_weather{_DEEPSEEK_SEP}[1]{_DEEPSEEK_CALL_CLOSE}",
        f"garbage{_DEEPSEEK_V31_CALL}",
        f"{_DEEPSEEK_V31_CALL}garbage",
    ],
)
def test_stream_parser_rejects_unrepairable_deepseek_payloads(payload: str) -> None:
    parser = ToolCallStreamParser(frozenset({"get_weather"}))

    with pytest.raises(ProtocolError, match="invalid tool-call JSON"):
        parser.feed(f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE}")


def test_stream_parser_rejects_a_deepseek_call_to_an_unknown_tool() -> None:
    parser = ToolCallStreamParser(frozenset({"get_weather"}))

    with pytest.raises(ProtocolError, match="tool 'shell' is not available"):
        parser.feed(
            f"{_DEEPSEEK_OPEN}{_DEEPSEEK_CALL_OPEN}shell{_DEEPSEEK_SEP}{{}}"
            f"{_DEEPSEEK_CALL_CLOSE}{_DEEPSEEK_CLOSE}"
        )


# Verbatim from vllm/tests/tool_parsers/test_qwen3coder_tool_parser.py at
# bb3b61f, matching the chat_template of Qwen/Qwen3-Coder-480B-A35B-Instruct
# (9d90cf8), which documents this exact skeleton.
_QWEN_CALL = (
    "\n<function=get_current_weather>\n"
    "<parameter=city>\nDallas\n</parameter>\n"
    "<parameter=state>\nTX\n</parameter>\n"
    "<parameter=unit>\nfahrenheit\n</parameter>\n"
    "</function>\n"
)


def test_stream_parser_translates_qwen_parameter_tags() -> None:
    parser = ToolCallStreamParser(frozenset({"get_current_weather"}))

    emissions = parser.feed(f"{TOOL_CALL_OPEN}{_QWEN_CALL}{TOOL_CALL_CLOSE}")
    emissions.extend(parser.finish())

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert emissions[0].name == "get_current_weather"
    assert json.loads(emissions[0].arguments) == {
        "city": "Dallas",
        "state": "TX",
        "unit": "fahrenheit",
    }


def test_stream_parser_translates_two_qwen_blocks() -> None:
    parser = ToolCallStreamParser(frozenset({"get_current_weather"}), max_tool_calls=2)

    emissions = parser.feed(
        f"{TOOL_CALL_OPEN}{_QWEN_CALL}{TOOL_CALL_CLOSE}\n"
        f"{TOOL_CALL_OPEN}"
        "\n<function=get_current_weather>\n<parameter=city>\nOrlando\n</parameter>\n</function>\n"
        f"{TOOL_CALL_CLOSE}"
    )
    emissions.extend(parser.finish())

    calls = [emission for emission in emissions if isinstance(emission, ToolCallEmission)]
    assert [json.loads(call.arguments)["city"] for call in calls] == ["Dallas", "Orlando"]


def test_stream_parser_translates_two_qwen_blocks_inside_one_marker() -> None:
    parser = ToolCallStreamParser(frozenset({"get_current_weather"}), max_tool_calls=2)

    emissions = parser.feed(
        f"{TOOL_CALL_OPEN}{_QWEN_CALL}"
        "\n<function=get_current_weather>\n<parameter=city>\nOrlando\n</parameter>\n</function>\n"
        f"{TOOL_CALL_CLOSE}"
    )

    calls = [emission for emission in emissions if isinstance(emission, ToolCallEmission)]
    assert [json.loads(call.arguments)["city"] for call in calls] == ["Dallas", "Orlando"]


def test_stream_parser_decodes_only_structured_qwen_values() -> None:
    # A scalar carries no type on the wire, so "2" stays text; the template
    # serializes mappings and sequences as JSON, so those are recovered.
    parser = ToolCallStreamParser(frozenset({"calculate_area"}))

    emissions = parser.feed(
        f"{TOOL_CALL_OPEN}\n<function=calculate_area>\n"
        "<parameter=shape>\nrectangle\n</parameter>\n"
        '<parameter=dimensions>\n{"width": 10, \n "height": 20}\n</parameter>\n'
        "<parameter=precision>\n2\n</parameter>\n"
        f"</function>\n{TOOL_CALL_CLOSE}"
    )

    assert isinstance(emissions[0], ToolCallEmission)
    assert json.loads(emissions[0].arguments) == {
        "shape": "rectangle",
        "dimensions": {"width": 10, "height": 20},
        "precision": "2",
    }


def test_stream_parser_preserves_qwen_parameter_whitespace() -> None:
    parser = ToolCallStreamParser(frozenset({"write_file"}))

    emissions = parser.feed(
        f"{TOOL_CALL_OPEN}\n<function=write_file>\n"
        "<parameter=content>\n    def foo():\n        return 1\n\n</parameter>\n"
        f"</function>\n{TOOL_CALL_CLOSE}"
    )

    assert isinstance(emissions[0], ToolCallEmission)
    assert json.loads(emissions[0].arguments) == {"content": "    def foo():\n        return 1\n"}


def test_stream_parser_keeps_a_qwen_value_that_only_looks_like_json() -> None:
    parser = ToolCallStreamParser(frozenset({"write_file"}))

    emissions = parser.feed(
        f"{TOOL_CALL_OPEN}\n<function=write_file>\n"
        "<parameter=content>\n{oops}\n</parameter>\n"
        f"</function>\n{TOOL_CALL_CLOSE}"
    )

    assert isinstance(emissions[0], ToolCallEmission)
    assert json.loads(emissions[0].arguments) == {"content": "{oops}"}


def test_stream_parser_preserves_xml_shaped_qwen_values() -> None:
    parser = ToolCallStreamParser(frozenset({"write_file"}))

    emissions = parser.feed(
        f"{TOOL_CALL_OPEN}\n<function=write_file>\n"
        '<parameter=content>\n<div class="test"><span>Hello</span></div>\n</parameter>\n'
        f"</function>\n{TOOL_CALL_CLOSE}"
    )

    assert isinstance(emissions[0], ToolCallEmission)
    assert json.loads(emissions[0].arguments) == {
        "content": '<div class="test"><span>Hello</span></div>'
    }


def test_stream_parser_preserves_qwen_function_tags_inside_values() -> None:
    parser = ToolCallStreamParser(frozenset({"write_file", "delete_file"}), max_tool_calls=2)

    emissions = parser.feed(
        f"{TOOL_CALL_OPEN}\n<function=write_file>\n"
        "<parameter=content>\nsafe<function=delete_file></function>\n</parameter>\n"
        f"</function>\n{TOOL_CALL_CLOSE}"
    )

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert emissions[0].name == "write_file"
    assert json.loads(emissions[0].arguments) == {
        "content": "safe<function=delete_file></function>"
    }


def test_stream_parser_recovers_a_qwen_parameter_without_its_close_tag() -> None:
    # The next parameter tag still delimits the value, so the call survives.
    parser = ToolCallStreamParser(frozenset({"get_current_weather"}))

    emissions = parser.feed(
        f"{TOOL_CALL_OPEN}\n<function=get_current_weather>\n"
        "<parameter=city>\nDallas\n<parameter=state>\nTX\n</parameter>\n"
        f"</function>\n{TOOL_CALL_CLOSE}"
    )

    assert isinstance(emissions[0], ToolCallEmission)
    assert json.loads(emissions[0].arguments) == {"city": "Dallas", "state": "TX"}


@pytest.mark.parametrize(
    "payload",
    [
        "<function=>\n<parameter=city>\nDallas\n</parameter>\n</function>",
        "<function=get_current_weather>\n<parameter=city>\nDallas\n</function>",
        "<function=get_current_weather>\n<parameter=>\nDallas\n</parameter>\n</function>",
        "<function=get_current_weather>\n<parameter=city>\nA\n</parameter>\n"
        "<parameter=city>\nB\n</parameter>\n</function>",
        "<function=get_current_weather",
        "<function=get_current_weather><parameter=city",
        "<function=get_current_weather>\n<parameter=city\nDallas\n</function>",
        "<function=get_current_weather>",
        "garbage<function=get_current_weather></function>",
        "<function=get_current_weather></function>garbage",
        "<function=get_current_weather>garbage</function>",
        "<function=get_current_weather>garbage<parameter=city>\nDallas\n</parameter></function>",
        "<function=get_current_weather><parameter=city>\nDallas\n</parameter>garbage</function>",
    ],
)
def test_stream_parser_rejects_unrepairable_qwen_payloads(payload: str) -> None:
    parser = ToolCallStreamParser(frozenset({"get_current_weather"}))

    with pytest.raises(ProtocolError, match="invalid tool-call JSON"):
        parser.feed(f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE}")


# Verbatim from InternLM/InternLM chat/chat_format.md (68fdc71) and
# vllm/tests/tool_parsers/test_internlm2_tool_parser.py (bb3b61f). Token names
# match the added_tokens_decoder of internlm/internlm2-chat-7b (c2ba644).
_INTERNLM_OPEN = "<|action_start|><|plugin|>"
_INTERNLM_CLOSE = "<|action_end|>"


def test_stream_parser_translates_an_internlm2_plugin_call() -> None:
    parser = ToolCallStreamParser(frozenset({"get_current_weather"}))

    emissions = parser.feed(
        "Sure, I will search for the weather of Shanghai."
        f'{_INTERNLM_OPEN}\n{{"name": "get_current_weather", '
        f'"parameters": {{"location": "Shanghai"}}}}{_INTERNLM_CLOSE}<|im_end|>'
    )
    emissions.extend(parser.finish())

    assert [type(emission) for emission in emissions] == [TextEmission, ToolCallEmission]
    assert isinstance(emissions[1], ToolCallEmission)
    assert emissions[1].name == "get_current_weather"
    assert json.loads(emissions[1].arguments) == {"location": "Shanghai"}


def test_stream_parser_translates_two_internlm2_plugin_calls() -> None:
    # The reference parser splits on the opening token and crashes on a second
    # block; marker framing handles repeats.
    parser = ToolCallStreamParser(frozenset({"get_weather"}), max_tool_calls=2)

    emissions = parser.feed(
        f'{_INTERNLM_OPEN}{{"name": "get_weather", "parameters": {{"city": "Tokyo"}}}}'
        f"{_INTERNLM_CLOSE}"
        f'{_INTERNLM_OPEN}{{"name": "get_weather", "parameters": {{"city": "Osaka"}}}}'
        f"{_INTERNLM_CLOSE}"
    )
    emissions.extend(parser.finish())

    calls = [emission for emission in emissions if isinstance(emission, ToolCallEmission)]
    assert [json.loads(call.arguments)["city"] for call in calls] == ["Tokyo", "Osaka"]


def test_stream_parser_rejects_internlm2_arguments_that_are_not_an_object() -> None:
    parser = ToolCallStreamParser(frozenset({"func"}))

    with pytest.raises(ProtocolError, match="tool-call arguments must be a JSON object"):
        parser.feed(
            f'{_INTERNLM_OPEN}{{"name": "func", "parameters": "not a dict"}}{_INTERNLM_CLOSE}'
        )


@pytest.mark.parametrize(
    "payload",
    [
        '{"name": "func", "parameters": {',
        "not json",
        "",
    ],
)
def test_stream_parser_rejects_unrepairable_internlm2_payloads(payload: str) -> None:
    parser = ToolCallStreamParser(frozenset({"func"}))

    with pytest.raises(ProtocolError, match="invalid tool-call JSON"):
        parser.feed(f"{_INTERNLM_OPEN}{payload}{_INTERNLM_CLOSE}")


def test_stream_parser_accepts_a_json_array_of_calls() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}), max_tool_calls=2)

    emissions = parser.feed(
        f"{TOOL_CALL_OPEN}"
        '[{"name":"weather","arguments":{"city":"Gdansk"}},'
        '{"name":"weather","arguments":{"city":"Sopot"}}]'
        f"{TOOL_CALL_CLOSE}"
    )

    calls = [emission for emission in emissions if isinstance(emission, ToolCallEmission)]
    assert [json.loads(call.arguments)["city"] for call in calls] == ["Gdansk", "Sopot"]


@pytest.mark.parametrize("payload", ["[]", '[{"name":"weather","arguments":{}},7]'])
def test_stream_parser_rejects_invalid_json_arrays(payload: str) -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    with pytest.raises(ProtocolError, match="must be a JSON object"):
        parser.feed(f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE}")


def test_stream_parser_rejects_empty_payload() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    with pytest.raises(ProtocolError, match="the payload is empty"):
        parser.feed(f"{TOOL_CALL_OPEN}   {TOOL_CALL_CLOSE}")


def test_stream_parser_reads_argument_key_aliases() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    emissions = parser.feed(
        f'{TOOL_CALL_OPEN}{{"name":"weather","parameters":{{"city":"Hel"}}}}{TOOL_CALL_CLOSE}'
    )

    assert isinstance(emissions[0], ToolCallEmission)
    assert json.loads(emissions[0].arguments) == {"city": "Hel"}


def test_stream_parser_requires_an_arguments_key() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    with pytest.raises(ProtocolError, match="arguments must be a JSON object"):
        parser.feed(f'{TOOL_CALL_OPEN}{{"name":"weather"}}{TOOL_CALL_CLOSE}')


def test_stream_parser_recovers_tool_call_without_close_marker() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))

    assert parser.feed(f'{TOOL_CALL_OPEN}{{"name":"weather","arguments":{{"city":"Hel"}}}}') == []
    emissions = parser.finish()

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert json.loads(emissions[0].arguments) == {"city": "Hel"}


def test_stream_parser_recovers_tool_call_with_partial_close_marker() -> None:
    payload = '{"name":"weather","arguments":{"city":"Hel"}}'

    for split in range(1, len(TOOL_CALL_CLOSE)):
        parser = ToolCallStreamParser(frozenset({"weather"}))

        assert (
            parser.feed(
                f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE[:split]}",
            )
            == []
        )
        emissions = parser.finish()

        assert len(emissions) == 1
        assert isinstance(emissions[0], ToolCallEmission)
        assert json.loads(emissions[0].arguments) == {"city": "Hel"}


@pytest.mark.parametrize("close_marker", ["", TOOL_CALL_CLOSE])
def test_stream_parser_recovers_json_with_glm_value_close(close_marker: str) -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))
    emissions = parser.feed(
        f'{TOOL_CALL_OPEN}{{"name":"weather","arguments":{{"city":"Gdansk"}}}}'
        f"</arg_value>{close_marker}"
    )

    emissions.extend(parser.finish())

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert json.loads(emissions[0].arguments) == {"city": "Gdansk"}


@pytest.mark.parametrize("close_marker", ["", TOOL_CALL_CLOSE])
def test_stream_parser_recovers_packed_json_with_glm_value_closes(close_marker: str) -> None:
    payload = (
        '{"name":"weather","arguments":{"city":"Gdansk"}}</arg_value>'
        '<tool_call>{"name":"clock","arguments":{"city":"Gdansk"}}</arg_value>'
    )
    stream = f"{TOOL_CALL_OPEN}{payload}{close_marker}"

    for boundary in range(len(stream) + 1):
        parser = ToolCallStreamParser(
            frozenset({"weather", "clock"}),
            max_tool_calls=2,
        )
        emissions = parser.feed(stream[:boundary])
        emissions.extend(parser.feed(stream[boundary:]))
        emissions.extend(parser.finish())

        calls = [emission for emission in emissions if isinstance(emission, ToolCallEmission)]
        assert [call.name for call in calls] == ["weather", "clock"]
        assert [json.loads(call.arguments) for call in calls] == [
            {"city": "Gdansk"},
            {"city": "Gdansk"},
        ]


def test_stream_parser_recovers_packed_glm_bare_read_file_calls() -> None:
    base = "/workspace/example/monorepo/service/api/src/ciapi/service/runtime_x/"
    paths = [
        base + "README.md",
        base + "pyproject.toml",
        base + "main.py",
        base + "config.py",
    ]
    calls = [
        "read_file"
        + json.dumps(
            {"filePath": path, "startLine": 1, "endLine": 300},
            separators=(",", ":"),
        )
        for path in paths
    ]
    payload = TOOL_CALL_OPEN.join(calls)

    stream = f"{TOOL_CALL_OPEN}{payload}"
    for chunk_size in (1, 17, len(stream)):
        parser = ToolCallStreamParser(frozenset({"read_file"}), max_tool_calls=4)
        emissions: list[object] = []
        for index in range(0, len(stream), chunk_size):
            emissions.extend(parser.feed(stream[index : index + chunk_size]))
        emissions.extend(parser.finish())

        parsed_calls = [
            emission for emission in emissions if isinstance(emission, ToolCallEmission)
        ]
        assert [call.name for call in parsed_calls] == ["read_file"] * 4
        assert [json.loads(call.arguments) for call in parsed_calls] == [
            {"filePath": path, "startLine": 1, "endLine": 300} for path in paths
        ]


def test_stream_parser_recovers_packed_bare_calls_with_glm_value_closes() -> None:
    payload = (
        'read_file{"filePath":"a"}</arg_value>'
        f"{TOOL_CALL_OPEN}"
        'read_file{"filePath":"b"}</arg_value>'
    )
    parser = ToolCallStreamParser(frozenset({"read_file"}), max_tool_calls=2)

    parser.feed(f"{TOOL_CALL_OPEN}{payload}")
    emissions = parser.finish()

    calls = [emission for emission in emissions if isinstance(emission, ToolCallEmission)]
    assert [json.loads(call.arguments) for call in calls] == [
        {"filePath": "a"},
        {"filePath": "b"},
    ]


@pytest.mark.parametrize(
    "payload",
    [
        'read_file{"filePath":"a"}<tool_call>unknown{"filePath":"b"}',
        'read_file{"filePath":"a"}<tool_call>read_file{"filePath":"b"',
        'read_file{"filePath":"a"}junk<tool_call>read_file{"filePath":"b"}',
        'read_file{"filePath":"a"}<tool_call>read_file{"filePath":"b","filePath":"c"}',
        'read_file{"filePath":"a"}<tool_call><tool_call>read_file{"filePath":"b"}',
        'read_file{"filePath":"a"}<tool_call>',
        'read_file{"filePath":"a"}</arg_value></arg_value><tool_call>read_file{"filePath":"b"}',
    ],
)
def test_stream_parser_rejects_invalid_packed_bare_calls(payload: str) -> None:
    parser = ToolCallStreamParser(frozenset({"read_file"}), max_tool_calls=3)

    emissions = parser.feed(f"{TOOL_CALL_OPEN}{payload}")

    # A rejected payload must not have leaked its valid leading segment first.
    assert [emission for emission in emissions if isinstance(emission, ToolCallEmission)] == []
    with pytest.raises(IncompleteToolCallError):
        parser.finish()


def test_stream_parser_rejects_packed_bare_calls_beyond_the_repair_cap() -> None:
    segments = [f'read_file{{"filePath":"f{index}"}}' for index in range(_MAX_PACKED_CALLS + 1)]
    parser = ToolCallStreamParser(
        frozenset({"read_file"}),
        max_tool_calls=_MAX_PACKED_CALLS + 1,
    )
    parser.feed(TOOL_CALL_OPEN + TOOL_CALL_OPEN.join(segments))

    with pytest.raises(IncompleteToolCallError):
        parser.finish()


def test_stream_parser_recovers_packed_glm_python_read_file_calls() -> None:
    base = "/home/user/.config/Code/User/workspaceStorage/9f1c8d/tasks/1786532954597/"
    paths = [base + "content.txt", base + "api_conversation_history.json"]
    calls = [
        "read_file("
        + json.dumps(
            {"filePath": path, "startLine": 1, "endLine": 300},
            separators=(",", ":"),
        )
        + ")"
        for path in paths
    ]
    # Observed on a GLM turn: packed python calls, the last call's closing
    # punctuation repeated, and the turn dying two bytes into the close marker.
    stream = TOOL_CALL_OPEN + "".join(calls) + "})" + TOOL_CALL_CLOSE[:2]

    for chunk_size in (1, 17, len(stream)):
        parser = ToolCallStreamParser(frozenset({"read_file"}), max_tool_calls=2)
        emissions: list[object] = []
        for index in range(0, len(stream), chunk_size):
            emissions.extend(parser.feed(stream[index : index + chunk_size]))
        emissions.extend(parser.finish())

        parsed_calls = [
            emission for emission in emissions if isinstance(emission, ToolCallEmission)
        ]
        assert [call.name for call in parsed_calls] == ["read_file"] * 2
        assert [json.loads(call.arguments) for call in parsed_calls] == [
            {"filePath": path, "startLine": 1, "endLine": 300} for path in paths
        ]


def test_stream_parser_rejects_packed_python_calls_beyond_the_repair_cap() -> None:
    segments = [f'read_file({{"filePath":"f{index}"}})' for index in range(_MAX_PACKED_CALLS + 1)]
    parser = ToolCallStreamParser(
        frozenset({"read_file"}),
        max_tool_calls=_MAX_PACKED_CALLS + 1,
    )
    parser.feed(TOOL_CALL_OPEN + "".join(segments))

    with pytest.raises(IncompleteToolCallError):
        parser.finish()


def test_stream_parser_recovers_packed_bare_calls_separated_by_whitespace() -> None:
    payload = 'read_file{"filePath":"a"}\n\t<tool_call>\n read_file{"filePath":"b"}'
    parser = ToolCallStreamParser(frozenset({"read_file"}), max_tool_calls=2)
    parser.feed(f"{TOOL_CALL_OPEN}{payload}")

    calls = parser.finish()

    assert [json.loads(call.arguments) for call in calls if isinstance(call, ToolCallEmission)] == [
        {"filePath": "a"},
        {"filePath": "b"},
    ]


def test_stream_parser_keeps_open_marker_inside_packed_bare_call_arguments() -> None:
    payload = (
        'write_file{"content":"literal <tool_call> marker"}'
        '<tool_call>read_file{"filePath":"README.md"}'
    )
    parser = ToolCallStreamParser(
        frozenset({"write_file", "read_file"}),
        max_tool_calls=2,
    )
    parser.feed(f"{TOOL_CALL_OPEN}{payload}")

    calls = parser.finish()

    assert [call.name for call in calls if isinstance(call, ToolCallEmission)] == [
        "write_file",
        "read_file",
    ]
    assert [json.loads(call.arguments) for call in calls if isinstance(call, ToolCallEmission)] == [
        {"content": "literal <tool_call> marker"},
        {"filePath": "README.md"},
    ]


@pytest.mark.parametrize(
    "payload",
    [
        (
            '{"name":"weather","arguments":{"city":"Gdansk"}}</arg_value>'
            '<tool_call>{"name":"unknown","arguments":{"city":"Gdansk"}}</arg_value>'
        ),
        (
            '{"name":"weather","arguments":{"city":"Gdansk"}}</arg_value>junk'
            '<tool_call>{"name":"clock","arguments":{"city":"Gdansk"}}</arg_value>'
        ),
        (
            '{"name":"weather","arguments":{"city":"Gdansk"}}</arg_value>'
            '<tool_call>{"name":"clock","name":"clock","arguments":{}}</arg_value>'
        ),
    ],
)
def test_stream_parser_rejects_invalid_packed_json_with_glm_value_closes(
    payload: str,
) -> None:
    parser = ToolCallStreamParser(
        frozenset({"weather", "clock"}),
        max_tool_calls=2,
    )
    parser.feed(f"{TOOL_CALL_OPEN}{payload}")

    with pytest.raises(IncompleteToolCallError):
        parser.finish()


@pytest.mark.parametrize(
    "suffix",
    [
        "</arg_key>",
        "</arg_value>junk",
        "junk</arg_value>",
        "</arg_value></arg_value>",
    ],
)
def test_stream_parser_rejects_other_json_close_residue(suffix: str) -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))
    parser.feed(f'{TOOL_CALL_OPEN}{{"name":"weather","arguments":{{"city":"Gdansk"}}}}{suffix}')

    with pytest.raises(IncompleteToolCallError):
        parser.finish()


@pytest.mark.parametrize(
    "payload",
    [
        "[]</arg_value>",
        '{"name":"unknown","arguments":{}}</arg_value>',
    ],
)
def test_stream_parser_rejects_invalid_glm_value_close_payloads(payload: str) -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))
    parser.feed(f"{TOOL_CALL_OPEN}{payload}")

    with pytest.raises(IncompleteToolCallError):
        parser.finish()


def test_stream_parser_counts_partial_close_marker_in_incomplete_payload() -> None:
    payload = '{"name":"weather","arguments":{"city":"Hel"'
    partial_close = TOOL_CALL_CLOSE[:-1]
    parser = ToolCallStreamParser(frozenset({"weather"}))

    parser.feed(f"{TOOL_CALL_OPEN}{payload}{partial_close}")

    with pytest.raises(IncompleteToolCallError) as excinfo:
        parser.finish()

    assert excinfo.value.payload_bytes == len((payload + partial_close).encode("utf-8"))


def test_recovered_tool_call_satisfies_a_required_tool_call() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}), require_tool_call=True)
    parser.feed(f'{TOOL_CALL_OPEN}weather\n{{"city":"Hel"}}')

    emissions = parser.finish()

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)


def test_unclosed_marker_without_tools_still_fails() -> None:
    parser = ToolCallStreamParser(frozenset())
    parser.feed(f'{TOOL_CALL_OPEN}{{"name":"weather","arguments":{{}}}}')

    with pytest.raises(ProtocolError, match="incomplete tool-call marker"):
        parser.finish()


def test_truncated_tool_call_reports_name_and_size() -> None:
    payload = '{"name":"weather","arguments":{"city":"Kr'
    parser = ToolCallStreamParser(frozenset({"weather"}))
    parser.feed(f"{TOOL_CALL_OPEN}{payload}")

    with pytest.raises(IncompleteToolCallError) as excinfo:
        parser.finish()

    assert excinfo.value.tool_name == "weather"
    assert excinfo.value.payload_bytes == len(payload.encode("utf-8"))
    assert "weather" in str(excinfo.value)
    assert str(len(payload.encode("utf-8"))) in str(excinfo.value)
    # Stays a ProtocolError so existing fail-closed handlers keep working.
    assert isinstance(excinfo.value, ProtocolError)


def test_truncated_tool_call_guesses_bare_name_form() -> None:
    parser = ToolCallStreamParser(frozenset({"execute_code"}))
    parser.feed(f'{TOOL_CALL_OPEN}execute_code{{"code":"from hermes')

    with pytest.raises(IncompleteToolCallError) as excinfo:
        parser.finish()

    assert excinfo.value.tool_name == "execute_code"


def test_truncated_tool_call_without_recognized_name_reports_none() -> None:
    parser = ToolCallStreamParser(frozenset({"weather"}))
    parser.feed(f"{TOOL_CALL_OPEN}{{")

    with pytest.raises(IncompleteToolCallError) as excinfo:
        parser.finish()

    assert excinfo.value.tool_name is None
    assert excinfo.value.payload_bytes == 1


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


def test_stream_parser_excludes_multibyte_close_prefix_from_payload_limit() -> None:
    prefix = f'{_DEEPSEEK_CALL_OPEN}get_weather{_DEEPSEEK_SEP}{{"value":"'
    suffix = f'"}}{_DEEPSEEK_CALL_CLOSE}'
    fixed_bytes = len((prefix + suffix).encode("utf-8"))
    filler = "x" * (_MAX_TOOL_PAYLOAD_BYTES - fixed_bytes)
    payload = f"{prefix}{filler}{suffix}"
    assert len(payload.encode("utf-8")) == _MAX_TOOL_PAYLOAD_BYTES

    parser = ToolCallStreamParser(frozenset({"get_weather"}))
    assert parser.feed(f"{_DEEPSEEK_OPEN}{payload}{_DEEPSEEK_CLOSE[:2]}") == []
    emissions = parser.feed(_DEEPSEEK_CLOSE[2:])

    assert len(emissions) == 1
    assert isinstance(emissions[0], ToolCallEmission)
    assert json.loads(emissions[0].arguments) == {"value": filler}


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


def test_build_prompt_forbids_transcript_reproduction() -> None:
    plan = build_prompt(_request())

    assert "Never copy, quote, serialize, or reproduce the transcript" in plan.prompt


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
