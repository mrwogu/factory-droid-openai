"""The tool-call formats the bridge accepts from a model.

The bridge asks for one shape, ``<tool_call>{"name":...,"arguments":...}
</tool_call>``, but a model served through Droid sometimes answers in the chat
template it was trained on instead. Translating those forms is the bridge's
job: a client cannot repair what it never asked for, and forwarding raw
template tokens as assistant text is worse than either error or repair.

Every accepted form lives in one of the two tables below, so adding a format
never touches the stream parser:

* :data:`MARKER_DIALECTS` frames a payload in the stream.
* :data:`PAYLOAD_DECODERS` rebuilds ``{"name", "arguments"}`` objects from a
  payload strict JSON could not parse.

Decoders return ``None`` when they do not recognise a payload, and the parser
fails closed once every decoder declines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from factory_droid_openai.strictjson import parse_strict_json

if TYPE_CHECKING:
    from collections.abc import Callable

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"

_ARG_KEY_OPEN = "<arg_key>"
_ARG_KEY_CLOSE = "</arg_key>"
_ARG_VALUE_OPEN = "<arg_value>"
_ARG_VALUE_CLOSE = "</arg_value>"

_PIPE_TAG_TOOLS_OPEN = "<|open|>tools<|sep|>"
_PIPE_TAG_TOOLS_CLOSE = "<|close|>tools"
_PIPE_TAG_CALL_OPEN = "<|open|>call "
_PIPE_TAG_ARG_OPEN = "<|open|>argument "
_PIPE_TAG_ARG_CLOSE = "<|close|>argument"
_PIPE_TAG_SEP = "<|sep|>"
# Every block closes with a <|close|> tag and a truncated tag is the common
# tail of these turns. The parser only treats these as ignorable once a call
# has already been emitted in this dialect, so prose still fails closed.
_PIPE_TAG_CONTROL_TOKENS = (
    _PIPE_TAG_TOOLS_CLOSE,
    "<|close|>call",
    _PIPE_TAG_ARG_CLOSE,
    "<|close|>",
    _PIPE_TAG_SEP,
)

_KIMI_SECTION_OPEN = "<|tool_calls_section_begin|>"
_KIMI_SECTION_CLOSE = "<|tool_calls_section_end|>"
_KIMI_CALL_OPEN = "<|tool_call_begin|>"
_KIMI_ARG_OPEN = "<|tool_call_argument_begin|>"
_KIMI_CALL_CLOSE = "<|tool_call_end|>"
_KIMI_CONTROL_TOKENS = (
    _KIMI_SECTION_CLOSE,
    _KIMI_CALL_CLOSE,
    "<|im_end|>",
)

_HARMONY_COMMENTARY_OPEN = "<|channel|>commentary to="
_HARMONY_CALL_CLOSE = "<|call|>"
_HARMONY_MESSAGE = "<|message|>"
_HARMONY_CONSTRAIN = "<|constrain|>"
_HARMONY_CONTROL_TOKENS = (
    _HARMONY_CALL_CLOSE,
    "<|start|>assistant",
    "<|return|>",
    "<|end|>",
)
# Both formats namespace a tool as functions.<name>; Kimi also appends the
# call index as functions.<name>:<index>.
_FUNCTIONS_PREFIX = "functions."

__all__ = [
    "MARKER_DIALECTS",
    "PAYLOAD_DECODERS",
    "TOOL_CALL_CLOSE",
    "TOOL_CALL_OPEN",
    "MarkerDialect",
    "PayloadDecoder",
    "find_open_marker",
    "strip_code_fence",
]


@dataclass(frozen=True, slots=True)
class MarkerDialect:
    """A marker pair the parser accepts around a tool-call payload."""

    name: str
    open_marker: str
    close_marker: str
    control_tokens: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PayloadDecoder:
    """Rebuilds tool-call objects from a payload strict JSON rejected."""

    name: str
    decode: Callable[[str, frozenset[str]], list[dict[str, Any]] | None]


NATIVE_DIALECT = MarkerDialect("native", TOOL_CALL_OPEN, TOOL_CALL_CLOSE)
# Observed on Kimi K3 turns served through Droid. Named after the token shape
# rather than the model, because the same scaffolding shows up whenever a model
# answers in its own template.
PIPE_TAG_DIALECT = MarkerDialect(
    "pipe_tag",
    _PIPE_TAG_TOOLS_OPEN,
    _PIPE_TAG_TOOLS_CLOSE,
    _PIPE_TAG_CONTROL_TOKENS,
)
KIMI_K2_DIALECT = MarkerDialect(
    "kimi_k2",
    _KIMI_SECTION_OPEN,
    _KIMI_SECTION_CLOSE,
    _KIMI_CONTROL_TOKENS,
)
HARMONY_DIALECT = MarkerDialect(
    "harmony",
    _HARMONY_COMMENTARY_OPEN,
    _HARMONY_CALL_CLOSE,
    _HARMONY_CONTROL_TOKENS,
)
MARKER_DIALECTS: tuple[MarkerDialect, ...] = (
    NATIVE_DIALECT,
    PIPE_TAG_DIALECT,
    KIMI_K2_DIALECT,
    HARMONY_DIALECT,
)


def find_open_marker(value: str) -> tuple[int, MarkerDialect] | None:
    """Returns the earliest open marker in ``value`` and its dialect."""
    found: tuple[int, MarkerDialect] | None = None
    for dialect in MARKER_DIALECTS:
        index = value.find(dialect.open_marker)
        if index >= 0 and (found is None or index < found[0]):
            found = (index, dialect)
    return found


def strip_code_fence(payload: str) -> str:
    if not payload.startswith("```"):
        return payload
    newline = payload.find("\n")
    if newline < 0:
        return payload
    body = payload[newline + 1 :]
    closing = body.rfind("```")
    return body[:closing].strip() if closing >= 0 else body.strip()


def _decode_pipe_tag_tokens(
    body: str,
    allowed_tool_names: frozenset[str],  # noqa: ARG001 - uniform decoder signature
) -> list[dict[str, Any]] | None:
    """Rebuilds calls a model emitted with pipe-tag chat-template tokens.

    Shape per call: ``<|open|>call tool="name" index="1"<|sep|><|open|>argument
    key="k" type="string"<|sep|>value<|close|>argument<|sep|><|close|>call``.
    """
    if _PIPE_TAG_CALL_OPEN not in body:
        return None
    calls: list[dict[str, Any]] = []
    index = body.find(_PIPE_TAG_CALL_OPEN)
    while index >= 0:
        header_end = body.find(_PIPE_TAG_SEP, index)
        if header_end < 0:
            return None
        name = _pipe_tag_attribute(body[index + len(_PIPE_TAG_CALL_OPEN) : header_end], "tool")
        if name is None:
            return None
        next_index = body.find(_PIPE_TAG_CALL_OPEN, header_end)
        block_end = next_index if next_index >= 0 else len(body)
        arguments = _pipe_tag_arguments(body[header_end:block_end])
        if arguments is None:
            return None
        calls.append({"name": name, "arguments": arguments})
        index = next_index
    return calls


def _pipe_tag_arguments(block: str) -> dict[str, Any] | None:
    arguments: dict[str, Any] = {}
    index = block.find(_PIPE_TAG_ARG_OPEN)
    while index >= 0:
        header_end = block.find(_PIPE_TAG_SEP, index)
        if header_end < 0:
            return None
        header = block[index + len(_PIPE_TAG_ARG_OPEN) : header_end]
        key = _pipe_tag_attribute(header, "key")
        value_end = block.find(_PIPE_TAG_ARG_CLOSE, header_end)
        if key is None or key in arguments or value_end < 0:
            # A truncated value would silently drop data, so fail closed.
            return None
        raw = block[header_end + len(_PIPE_TAG_SEP) : value_end]
        arguments[key] = _coerce_pipe_tag_value(raw, _pipe_tag_attribute(header, "type"))
        index = block.find(_PIPE_TAG_ARG_OPEN, value_end)
    return arguments


def _pipe_tag_attribute(header: str, name: str) -> str | None:
    prefix = f'{name}="'
    start = header.find(prefix)
    if start < 0:
        return None
    start += len(prefix)
    end = header.find('"', start)
    if end < 0:
        return None
    return header[start:end].strip() or None


def _coerce_pipe_tag_value(raw: str, kind: str | None) -> Any:
    # A declared string type is authoritative: coercing "1" to a number there
    # would change the argument the client receives.
    if kind == "string":
        return raw.strip()
    return _coerce_arg_value(raw.strip())


def _decode_kimi_sections(
    body: str,
    allowed_tool_names: frozenset[str],  # noqa: ARG001 - uniform decoder signature
) -> list[dict[str, Any]] | None:
    """Rebuilds Kimi K2's tool-call section.

    Shape per call: ``<|tool_call_begin|>functions.name:0
    <|tool_call_argument_begin|>{"k":"v"}<|tool_call_end|>``.
    """
    if _KIMI_CALL_OPEN not in body:
        return None
    calls: list[dict[str, Any]] = []
    index = body.find(_KIMI_CALL_OPEN)
    while index >= 0:
        arguments_start = body.find(_KIMI_ARG_OPEN, index)
        if arguments_start < 0:
            return None
        name = _strip_function_namespace(body[index + len(_KIMI_CALL_OPEN) : arguments_start])
        call_end = body.find(_KIMI_CALL_CLOSE, arguments_start)
        if name is None or call_end < 0:
            # A truncated call would drop arguments, so fail closed.
            return None
        raw = body[arguments_start + len(_KIMI_ARG_OPEN) : call_end]
        arguments = _decode_json_object(raw)
        if arguments is None:
            return None
        calls.append({"name": name, "arguments": arguments})
        index = body.find(_KIMI_CALL_OPEN, call_end)
    return calls


def _decode_harmony_commentary(
    body: str,
    allowed_tool_names: frozenset[str],  # noqa: ARG001 - uniform decoder signature
) -> list[dict[str, Any]] | None:
    """Rebuilds a gpt-oss Harmony commentary call.

    Payload between ``<|channel|>commentary to=`` and ``<|call|>`` reads
    ``functions.name <|constrain|>json<|message|>{"k":"v"}``.
    """
    message_start = body.find(_HARMONY_MESSAGE)
    if message_start < 0:
        return None
    header = body[:message_start]
    constrain = header.find(_HARMONY_CONSTRAIN)
    if constrain >= 0:
        header = header[:constrain]
    name = _strip_function_namespace(header)
    if name is None:
        return None
    arguments = _decode_json_object(body[message_start + len(_HARMONY_MESSAGE) :])
    if arguments is None:
        return None
    return [{"name": name, "arguments": arguments}]


def _strip_function_namespace(raw: str) -> str | None:
    name = raw.strip()
    if name.startswith(_FUNCTIONS_PREFIX):
        name = name[len(_FUNCTIONS_PREFIX) :]
    head, separator, index = name.rpartition(":")
    if separator and index.isdigit():
        name = head
    return name.strip() or None


def _decode_json_object(raw: str) -> dict[str, Any] | None:
    body = strip_code_fence(raw.strip())
    if not body:
        return {}
    try:
        parsed = parse_strict_json(body)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _decode_arg_key_value(
    body: str,
    allowed_tool_names: frozenset[str],  # noqa: ARG001 - uniform decoder signature
) -> list[dict[str, Any]] | None:
    """Rebuilds GLM's ``name<arg_key>k</arg_key><arg_value>v</arg_value>`` form."""
    if _ARG_KEY_OPEN not in body:
        return None
    index = body.find(_ARG_KEY_OPEN)
    name = body[:index].strip()
    if not name:
        return None
    arguments: dict[str, Any] = {}
    while index >= 0:
        key_end = body.find(_ARG_KEY_CLOSE, index)
        if key_end < 0:
            return None
        key = body[index + len(_ARG_KEY_OPEN) : key_end].strip()
        value_start = body.find(_ARG_VALUE_OPEN, key_end)
        if not key or key in arguments or value_start < 0:
            return None
        value_end = body.find(_ARG_VALUE_CLOSE, value_start)
        if value_end < 0:
            # A truncated value would silently drop data, so fail closed.
            return None
        raw = body[value_start + len(_ARG_VALUE_OPEN) : value_end].strip()
        arguments[key] = _coerce_arg_value(raw)
        index = body.find(_ARG_KEY_OPEN, value_end)
    return [{"name": name, "arguments": arguments}]


def _decode_bare_name(
    body: str,
    allowed_tool_names: frozenset[str],
) -> list[dict[str, Any]] | None:
    """Rebuilds a payload that names the tool outside the JSON object."""
    head, _, tail = body.partition("\n")
    name = head.strip()
    if name not in allowed_tool_names:
        return None
    remainder = tail.strip()
    if not remainder:
        return [{"name": name, "arguments": {}}]
    try:
        arguments = parse_strict_json(remainder)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(arguments, dict):
        return None
    return [{"name": name, "arguments": arguments}]


def _coerce_arg_value(raw: str) -> Any:
    try:
        parsed = parse_strict_json(raw)
    except (json.JSONDecodeError, ValueError):
        return raw
    return parsed


PAYLOAD_DECODERS: tuple[PayloadDecoder, ...] = (
    PayloadDecoder("pipe_tag_tokens", _decode_pipe_tag_tokens),
    PayloadDecoder("kimi_sections", _decode_kimi_sections),
    PayloadDecoder("harmony_commentary", _decode_harmony_commentary),
    PayloadDecoder("arg_key_value", _decode_arg_key_value),
    PayloadDecoder("bare_name", _decode_bare_name),
)
