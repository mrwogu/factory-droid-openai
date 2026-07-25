from __future__ import annotations

import json

import pytest

from factory_droid_openai.models import ChatCompletionRequest
from factory_droid_openai.protocol import (
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    ProtocolError,
    TextEmission,
    ToolCallEmission,
    ToolCallStreamParser,
    build_prompt,
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
