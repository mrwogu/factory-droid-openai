from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from factory_droid_openai.dialects import MARKER_DIALECTS, PAYLOAD_DECODERS, MarkerDialect
from factory_droid_openai.protocol import (
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    IncompleteToolCallError,
    ProtocolError,
    TextEmission,
    ToolCallEmission,
    ToolCallStreamParser,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_DS_BAR = "\uff5c"
_DS_BLOCK = "\u2581"
_DEEPSEEK_CALL_OPEN = f"<{_DS_BAR}tool{_DS_BLOCK}call{_DS_BLOCK}begin{_DS_BAR}>"
_DEEPSEEK_CALL_CLOSE = f"<{_DS_BAR}tool{_DS_BLOCK}call{_DS_BLOCK}end{_DS_BAR}>"
_DEEPSEEK_SEP = f"<{_DS_BAR}tool{_DS_BLOCK}sep{_DS_BAR}>"


def _pipe_tag_call(city: str, index: int) -> str:
    return (
        f'<|open|>call tool="weather" index="{index}"<|sep|>'
        f'<|open|>argument key="city" type="string"<|sep|>{city}<|close|>argument<|sep|>'
        "<|close|>call<|sep|>"
    )


def _kimi_call(city: str, index: int) -> str:
    return (
        f"<|tool_call_begin|>functions.weather:{index}"
        f'<|tool_call_argument_begin|>{{"city":"{city}"}}<|tool_call_end|>'
    )


def _deepseek_call(city: str) -> str:
    return f'{_DEEPSEEK_CALL_OPEN}weather{_DEEPSEEK_SEP}{{"city":"{city}"}}{_DEEPSEEK_CALL_CLOSE}'


def _qwen_call(city: str) -> str:
    return f"<function=weather><parameter=city>{city}</parameter></function>"


def _native_call(city: str) -> str:
    return f'{{"name":"weather","arguments":{{"city":"{city}"}}}}'


_PIPE_TAG_CALL = _pipe_tag_call("Gdansk", 1)
_KIMI_CALL = _kimi_call("Gdansk", 0)
_HARMONY_CALL = 'functions.weather <|constrain|>json<|message|>{"city":"Gdansk"}'
_DEEPSEEK_CALL = _deepseek_call("Gdansk")
_INTERNLM_CALL = '{"name":"weather","parameters":{"city":"Gdansk"}}'
_NATIVE_CALL = _native_call("Gdansk")
_CHUNK_SIZES = (1, 2, 5, 13, 64)


@dataclass(frozen=True, slots=True)
class RecoveryFixture:
    name: str
    payload: str
    expected_arguments: dict[str, Any]
    diagnostic_tool_name: str | None = None


@dataclass(frozen=True, slots=True)
class PackedRecoveryFixture:
    """One payload holding several calls the decoder must return together."""

    name: str
    payload: str
    expected_arguments: tuple[dict[str, Any], ...]


_MARKER_FIXTURES = {
    fixture.name: fixture
    for fixture in (
        RecoveryFixture(
            "native",
            _NATIVE_CALL,
            {"city": "Gdansk"},
            diagnostic_tool_name="weather",
        ),
        RecoveryFixture("pipe_tag", _PIPE_TAG_CALL, {"city": "Gdansk"}),
        RecoveryFixture("kimi_k2", _KIMI_CALL, {"city": "Gdansk"}),
        RecoveryFixture("harmony", _HARMONY_CALL, {"city": "Gdansk"}),
        RecoveryFixture("deepseek", _DEEPSEEK_CALL, {"city": "Gdansk"}),
        RecoveryFixture(
            "internlm2",
            _INTERNLM_CALL,
            {"city": "Gdansk"},
            diagnostic_tool_name="weather",
        ),
    )
}

_DECODER_FIXTURES = (
    RecoveryFixture("pipe_tag_tokens", _PIPE_TAG_CALL, {"city": "Gdansk"}),
    RecoveryFixture("kimi_sections", _KIMI_CALL, {"city": "Gdansk"}),
    RecoveryFixture("harmony_commentary", _HARMONY_CALL, {"city": "Gdansk"}),
    RecoveryFixture("deepseek_calls", _DEEPSEEK_CALL, {"city": "Gdansk"}),
    RecoveryFixture(
        "function_parameter_tags",
        "<function=weather><parameter=city>Gdansk</parameter></function>",
        {"city": "Gdansk"},
    ),
    RecoveryFixture(
        "json_arg_value_close",
        _NATIVE_CALL + "</arg_value>",
        {"city": "Gdansk"},
    ),
    RecoveryFixture(
        "arg_key_value",
        "weather<arg_key>city</arg_key><arg_value>Gdansk</arg_value>",
        {"city": "Gdansk"},
    ),
    RecoveryFixture("python_call", 'weather({"city":"Gdansk"})', {"city": "Gdansk"}),
    RecoveryFixture(
        "arg_key_value_repair",
        'weather<arg_key>city":"Gdansk"}',
        {"city": "Gdansk"},
    ),
    RecoveryFixture("bare_name", 'weather\n{"city":"Gdansk"}', {"city": "Gdansk"}),
    RecoveryFixture("bare_call", 'weather{"city":"Gdansk"}', {"city": "Gdansk"}),
)

_PACKED_ARGUMENTS = ({"city": "Gdansk"}, {"city": "Sopot"})
_PACKED_DECODER_FIXTURES = (
    PackedRecoveryFixture(
        "pipe_tag_tokens",
        _pipe_tag_call("Gdansk", 1) + _pipe_tag_call("Sopot", 2),
        _PACKED_ARGUMENTS,
    ),
    PackedRecoveryFixture(
        "kimi_sections",
        _kimi_call("Gdansk", 0) + _kimi_call("Sopot", 1),
        _PACKED_ARGUMENTS,
    ),
    PackedRecoveryFixture(
        "deepseek_calls",
        _deepseek_call("Gdansk") + "\n" + _deepseek_call("Sopot"),
        _PACKED_ARGUMENTS,
    ),
    PackedRecoveryFixture(
        "function_parameter_tags",
        _qwen_call("Gdansk") + _qwen_call("Sopot"),
        _PACKED_ARGUMENTS,
    ),
    PackedRecoveryFixture(
        "json_arg_value_close",
        f"{_native_call('Gdansk')}</arg_value>{TOOL_CALL_OPEN}{_native_call('Sopot')}</arg_value>",
        _PACKED_ARGUMENTS,
    ),
    PackedRecoveryFixture(
        "arg_key_value_repair",
        f'weather<arg_key>city":"Gdansk"}}{TOOL_CALL_OPEN}weather<arg_key>city":"Sopot"}}',
        _PACKED_ARGUMENTS,
    ),
    PackedRecoveryFixture(
        "bare_call",
        f'weather{{"city":"Gdansk"}}{TOOL_CALL_OPEN}weather{{"city":"Sopot"}}',
        _PACKED_ARGUMENTS,
    ),
    PackedRecoveryFixture(
        "bare_call",
        f'weather{{"city":"Gdansk"}}</arg_value>{TOOL_CALL_OPEN}'
        'weather{"city":"Sopot"}</arg_value>',
        _PACKED_ARGUMENTS,
    ),
    PackedRecoveryFixture(
        "python_call",
        f'weather({{"city":"Gdansk"}}){TOOL_CALL_OPEN}weather({{"city":"Sopot"}})',
        _PACKED_ARGUMENTS,
    ),
)
# These shapes carry exactly one call per marker pair, so packing them is not a
# form the wire can produce. A new decoder has to land in one list or the other.
_SINGLE_CALL_DECODERS = frozenset(
    {
        "harmony_commentary",
        "arg_key_value",
        "bare_name",
    }
)


def _assert_tool_call(
    emissions: Sequence[object],
    expected_arguments: dict[str, Any],
) -> None:
    assert len(emissions) == 1
    emission = emissions[0]
    assert isinstance(emission, ToolCallEmission)
    assert emission.name == "weather"
    assert json.loads(emission.arguments) == expected_arguments


def _assert_tool_calls(
    emissions: Sequence[object],
    expected_arguments: tuple[dict[str, Any], ...],
) -> None:
    calls = [emission for emission in emissions if isinstance(emission, ToolCallEmission)]
    assert len(calls) == len(emissions)
    assert [call.name for call in calls] == ["weather"] * len(expected_arguments)
    assert [json.loads(call.arguments) for call in calls] == list(expected_arguments)


def _variant_tracer(variants: list[str]) -> Callable[..., None]:
    def trace(_event: str, _payload: str, **fields: Any) -> None:
        variant = fields.get("variant")
        if isinstance(variant, str):
            variants.append(variant)

    return trace


def _feed_in_chunks(
    parser: ToolCallStreamParser,
    stream: str,
    chunk_size: int,
) -> list[object]:
    emissions: list[object] = []
    for index in range(0, len(stream), chunk_size):
        emissions.extend(parser.feed(stream[index : index + chunk_size]))
    return emissions


def test_marker_recovery_matrix_matches_every_registered_dialect() -> None:
    assert set(_MARKER_FIXTURES) == {dialect.name for dialect in MARKER_DIALECTS}


@pytest.mark.parametrize("dialect", MARKER_DIALECTS, ids=lambda dialect: dialect.name)
def test_every_dialect_recovers_from_every_partial_close_marker(
    dialect: MarkerDialect,
) -> None:
    fixture = _MARKER_FIXTURES[dialect.name]

    for prefix_length in range(len(dialect.close_marker)):
        stream = dialect.open_marker + fixture.payload + dialect.close_marker[:prefix_length]
        for chunk_size in _CHUNK_SIZES:
            parser = ToolCallStreamParser(frozenset({"weather"}))
            assert _feed_in_chunks(parser, stream, chunk_size) == []

            _assert_tool_call(parser.finish(), fixture.expected_arguments)


@pytest.mark.parametrize("dialect", MARKER_DIALECTS, ids=lambda dialect: dialect.name)
def test_every_dialect_handles_every_two_chunk_boundary(
    dialect: MarkerDialect,
) -> None:
    fixture = _MARKER_FIXTURES[dialect.name]
    stream = dialect.open_marker + fixture.payload + dialect.close_marker

    for boundary in range(len(stream) + 1):
        parser = ToolCallStreamParser(frozenset({"weather"}))
        emissions = parser.feed(stream[:boundary])
        emissions.extend(parser.feed(stream[boundary:]))
        emissions.extend(parser.finish())

        _assert_tool_call(emissions, fixture.expected_arguments)


@pytest.mark.parametrize("dialect", MARKER_DIALECTS, ids=lambda dialect: dialect.name)
def test_every_partial_open_marker_remains_plain_text(
    dialect: MarkerDialect,
) -> None:
    for prefix_length in range(1, len(dialect.open_marker)):
        text = "literal " + dialect.open_marker[:prefix_length]
        parser = ToolCallStreamParser(frozenset())
        emissions = parser.feed(text)
        emissions.extend(parser.finish())

        assert (
            "".join(emission.text for emission in emissions if isinstance(emission, TextEmission))
            == text
        )


@pytest.mark.parametrize("dialect", MARKER_DIALECTS, ids=lambda dialect: dialect.name)
def test_every_dialect_rejects_text_after_a_completed_call(
    dialect: MarkerDialect,
) -> None:
    fixture = _MARKER_FIXTURES[dialect.name]
    parser = ToolCallStreamParser(frozenset({"weather"}))

    with pytest.raises(ProtocolError, match="unexpected text after tool call"):
        parser.feed(dialect.open_marker + fixture.payload + dialect.close_marker + "unexpected")


@pytest.mark.parametrize("dialect", MARKER_DIALECTS, ids=lambda dialect: dialect.name)
def test_every_dialect_keeps_truncated_payload_diagnostics(
    dialect: MarkerDialect,
) -> None:
    fixture = _MARKER_FIXTURES[dialect.name]
    truncated_payload = fixture.payload[:-1]

    for prefix_length in range(len(dialect.close_marker)):
        close_prefix = dialect.close_marker[:prefix_length]
        parser = ToolCallStreamParser(frozenset({"weather"}))
        parser.feed(dialect.open_marker + truncated_payload + close_prefix)

        with pytest.raises(IncompleteToolCallError) as excinfo:
            parser.finish()

        captured = truncated_payload + close_prefix
        assert excinfo.value.tool_name == fixture.diagnostic_tool_name
        assert excinfo.value.payload_bytes == len(captured.encode("utf-8"))


def test_decoder_recovery_matrix_matches_every_registered_decoder() -> None:
    assert {fixture.name for fixture in _DECODER_FIXTURES} == {
        decoder.name for decoder in PAYLOAD_DECODERS
    }


@pytest.mark.parametrize("fixture", _DECODER_FIXTURES, ids=lambda fixture: fixture.name)
def test_every_decoder_recovers_from_every_partial_native_close_marker(
    fixture: RecoveryFixture,
) -> None:
    for prefix_length in range(len(TOOL_CALL_CLOSE)):
        stream = TOOL_CALL_OPEN + fixture.payload + TOOL_CALL_CLOSE[:prefix_length]
        for chunk_size in _CHUNK_SIZES:
            variants: list[str] = []
            parser = ToolCallStreamParser(
                frozenset({"weather"}),
                trace_payload=_variant_tracer(variants),
            )
            assert _feed_in_chunks(parser, stream, chunk_size) == []

            _assert_tool_call(parser.finish(), fixture.expected_arguments)
            assert variants == [fixture.name]


def test_packed_decoder_matrix_accounts_for_every_registered_decoder() -> None:
    packed = {fixture.name for fixture in _PACKED_DECODER_FIXTURES}
    assert not packed & _SINGLE_CALL_DECODERS
    assert packed | _SINGLE_CALL_DECODERS == {decoder.name for decoder in PAYLOAD_DECODERS}


@pytest.mark.parametrize("fixture", _PACKED_DECODER_FIXTURES, ids=lambda fixture: fixture.name)
def test_every_packing_decoder_recovers_every_call_from_every_partial_close_marker(
    fixture: PackedRecoveryFixture,
) -> None:
    for prefix_length in range(len(TOOL_CALL_CLOSE)):
        stream = TOOL_CALL_OPEN + fixture.payload + TOOL_CALL_CLOSE[:prefix_length]
        for chunk_size in _CHUNK_SIZES:
            variants: list[str] = []
            parser = ToolCallStreamParser(
                frozenset({"weather"}),
                max_tool_calls=len(fixture.expected_arguments),
                trace_payload=_variant_tracer(variants),
            )
            assert _feed_in_chunks(parser, stream, chunk_size) == []

            _assert_tool_calls(parser.finish(), fixture.expected_arguments)
            # The decoder repair is traced first, the packing it returned second.
            assert variants == [fixture.name, "packed_objects"]


@pytest.mark.parametrize("fixture", _PACKED_DECODER_FIXTURES, ids=lambda fixture: fixture.name)
def test_every_packing_decoder_handles_every_two_chunk_boundary(
    fixture: PackedRecoveryFixture,
) -> None:
    stream = TOOL_CALL_OPEN + fixture.payload + TOOL_CALL_CLOSE

    for boundary in range(len(stream) + 1):
        parser = ToolCallStreamParser(
            frozenset({"weather"}),
            max_tool_calls=len(fixture.expected_arguments),
        )
        emissions = parser.feed(stream[:boundary])
        emissions.extend(parser.feed(stream[boundary:]))
        emissions.extend(parser.finish())

        _assert_tool_calls(emissions, fixture.expected_arguments)


@pytest.mark.parametrize("fixture", _PACKED_DECODER_FIXTURES, ids=lambda fixture: fixture.name)
def test_every_packing_decoder_rejects_more_calls_than_the_turn_allows(
    fixture: PackedRecoveryFixture,
) -> None:
    parser = ToolCallStreamParser(
        frozenset({"weather"}),
        max_tool_calls=len(fixture.expected_arguments) - 1,
    )

    with pytest.raises(ProtocolError, match="more tool calls than the configured maximum"):
        parser.feed(TOOL_CALL_OPEN + fixture.payload + TOOL_CALL_CLOSE)
