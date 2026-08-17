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
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from factory_droid_openai.strictjson import (
    JsonNestingError,
    parse_strict_json,
    raw_decode_strict,
)

if TYPE_CHECKING:
    from collections.abc import Callable

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"

_ARG_KEY_OPEN = "<arg_key>"
_ARG_KEY_CLOSE = "</arg_key>"
_ARG_VALUE_OPEN = "<arg_value>"
_ARG_VALUE_CLOSE = "</arg_value>"
_ARG_CONTROL_PREFIXES = ("<arg_", "</arg_", "<tool_call", "</tool_call")

_PIPE_TAG_TOOLS_OPEN = "<|open|>tools<|sep|>"
_PIPE_TAG_TOOLS_CLOSE = "<|close|>tools"
_PIPE_TAG_CALL_OPEN = "<|open|>call "
_PIPE_TAG_CALL_CLOSE = "<|close|>call"
_PIPE_TAG_ARG_OPEN = "<|open|>argument "
_PIPE_TAG_ARG_CLOSE = "<|close|>argument"
_PIPE_TAG_SEP = "<|sep|>"
# Every block closes with a <|close|> tag and a truncated tag is the common
# tail of these turns. The parser only treats these as ignorable once a call
# has already been emitted in this dialect, so prose still fails closed.
_PIPE_TAG_CONTROL_TOKENS = (
    _PIPE_TAG_TOOLS_CLOSE,
    _PIPE_TAG_CALL_CLOSE,
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

_INTERNLM_ACTION_OPEN = "<|action_start|><|plugin|>"
_INTERNLM_ACTION_CLOSE = "<|action_end|>"
_INTERNLM_CONTROL_TOKENS = (
    _INTERNLM_ACTION_CLOSE,
    "<|im_end|>",
)

# DeepSeek builds these tokens from a fullwidth vertical line (U+FF5C) and a
# lower one eighth block (U+2581), not from ASCII "|" and "_". Spelling them as
# escapes keeps the exact codepoints visible and unambiguous in review.
_FULLWIDTH_BAR = "\uff5c"
_LOWER_BLOCK = "\u2581"
_DEEPSEEK_SECTION_OPEN = (
    f"<{_FULLWIDTH_BAR}tool{_LOWER_BLOCK}calls{_LOWER_BLOCK}begin{_FULLWIDTH_BAR}>"
)
_DEEPSEEK_SECTION_CLOSE = (
    f"<{_FULLWIDTH_BAR}tool{_LOWER_BLOCK}calls{_LOWER_BLOCK}end{_FULLWIDTH_BAR}>"
)
_DEEPSEEK_CALL_OPEN = f"<{_FULLWIDTH_BAR}tool{_LOWER_BLOCK}call{_LOWER_BLOCK}begin{_FULLWIDTH_BAR}>"
_DEEPSEEK_CALL_CLOSE = f"<{_FULLWIDTH_BAR}tool{_LOWER_BLOCK}call{_LOWER_BLOCK}end{_FULLWIDTH_BAR}>"
_DEEPSEEK_SEP = f"<{_FULLWIDTH_BAR}tool{_LOWER_BLOCK}sep{_FULLWIDTH_BAR}>"
_DEEPSEEK_EOS = f"<{_FULLWIDTH_BAR}end{_LOWER_BLOCK}of{_LOWER_BLOCK}sentence{_FULLWIDTH_BAR}>"
_DEEPSEEK_CONTROL_TOKENS = (
    _DEEPSEEK_SECTION_CLOSE,
    _DEEPSEEK_CALL_CLOSE,
    _DEEPSEEK_EOS,
)

_QWEN_FUNCTION_OPEN = "<function="
_QWEN_FUNCTION_CLOSE = "</function>"
_QWEN_PARAMETER_OPEN = "<parameter="
_QWEN_PARAMETER_CLOSE = "</parameter>"
_QWEN_PARAMETER_TOKEN = re.compile(r"</parameter>")

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
    "LOST_PREFIX_DECODER",
    "MARKER_DIALECTS",
    "MAX_PACKED_CALLS",
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
    json_quoted_close: bool = False


@dataclass(frozen=True, slots=True)
class PayloadDecoder:
    """Rebuilds tool-call objects from a payload strict JSON rejected."""

    name: str
    decode: Callable[[str, frozenset[str]], list[dict[str, Any]] | None]


# Past this many segments a malformed payload can only exceed every supported
# operator limit, so every packing decoder stops before doing unbounded work.
MAX_PACKED_CALLS = 64


def _append_packed_call(
    calls: list[dict[str, Any]],
    call: dict[str, Any],
) -> bool:
    if len(calls) >= MAX_PACKED_CALLS:
        return False
    calls.append(call)
    return True


NATIVE_DIALECT = MarkerDialect(
    "native",
    TOOL_CALL_OPEN,
    TOOL_CALL_CLOSE,
    json_quoted_close=True,
)
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
    json_quoted_close=True,
)
HARMONY_DIALECT = MarkerDialect(
    "harmony",
    _HARMONY_COMMENTARY_OPEN,
    _HARMONY_CALL_CLOSE,
    _HARMONY_CONTROL_TOKENS,
    json_quoted_close=True,
)
DEEPSEEK_DIALECT = MarkerDialect(
    "deepseek",
    _DEEPSEEK_SECTION_OPEN,
    _DEEPSEEK_SECTION_CLOSE,
    _DEEPSEEK_CONTROL_TOKENS,
    json_quoted_close=True,
)
# The InternLM2 template selects a target right after the opening token, so the
# tool form is the two-token sequence rather than <|action_start|> alone; the
# <|interpreter|> target is a different feature and stays unsupported.
INTERNLM2_DIALECT = MarkerDialect(
    "internlm2",
    _INTERNLM_ACTION_OPEN,
    _INTERNLM_ACTION_CLOSE,
    _INTERNLM_CONTROL_TOKENS,
    json_quoted_close=True,
)
MARKER_DIALECTS: tuple[MarkerDialect, ...] = (
    NATIVE_DIALECT,
    PIPE_TAG_DIALECT,
    KIMI_K2_DIALECT,
    HARMONY_DIALECT,
    DEEPSEEK_DIALECT,
    INTERNLM2_DIALECT,
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
    if closing < 0:
        return body.strip()
    if body[closing + 3 :].strip():
        return payload
    return body[:closing].strip()


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
    if body[:index].strip():
        return None
    while index >= 0:
        header_end = body.find(_PIPE_TAG_SEP, index)
        if header_end < 0:
            return None
        attributes = _pipe_tag_attributes(
            body[index + len(_PIPE_TAG_CALL_OPEN) : header_end],
            frozenset({"tool", "index"}),
        )
        name = attributes.get("tool") if attributes is not None else None
        if name is None:
            return None
        call_end = body.find(_PIPE_TAG_CALL_CLOSE, header_end)
        if call_end < 0:
            return None
        next_index = body.find(_PIPE_TAG_CALL_OPEN, call_end + len(_PIPE_TAG_CALL_CLOSE))
        block_end = next_index if next_index >= 0 else len(body)
        arguments = _pipe_tag_arguments(body[header_end:call_end])
        trailing = body[call_end + len(_PIPE_TAG_CALL_CLOSE) : block_end]
        if arguments is None or not _pipe_tag_noise_only(trailing):
            return None
        if not _append_packed_call(calls, {"name": name, "arguments": arguments}):
            return None
        index = next_index
    return calls


def _pipe_tag_arguments(block: str) -> dict[str, Any] | None:
    arguments: dict[str, Any] = {}
    index = block.find(_PIPE_TAG_ARG_OPEN)
    if index < 0:
        return arguments if _pipe_tag_noise_only(block) else None
    if not _pipe_tag_noise_only(block[:index]):
        return None
    while index >= 0:
        header_end = block.find(_PIPE_TAG_SEP, index)
        if header_end < 0:
            return None
        header = block[index + len(_PIPE_TAG_ARG_OPEN) : header_end]
        attributes = _pipe_tag_attributes(header, frozenset({"key", "type"}))
        if attributes is None:
            return None
        key = attributes.get("key")
        value_end = block.find(_PIPE_TAG_ARG_CLOSE, header_end)
        if key is None or key in arguments or value_end < 0:
            # A truncated value would silently drop data, so fail closed.
            return None
        raw = block[header_end + len(_PIPE_TAG_SEP) : value_end]
        valid, decoded = _decode_pipe_tag_value(raw, attributes.get("type"))
        if not valid:
            return None
        arguments[key] = decoded
        next_index = block.find(_PIPE_TAG_ARG_OPEN, value_end + len(_PIPE_TAG_ARG_CLOSE))
        trailing_end = next_index if next_index >= 0 else len(block)
        if not _pipe_tag_noise_only(block[value_end + len(_PIPE_TAG_ARG_CLOSE) : trailing_end]):
            return None
        index = next_index
    return arguments


def _pipe_tag_noise_only(value: str) -> bool:
    return not value.replace(_PIPE_TAG_SEP, "").strip()


def _pipe_tag_attributes(
    header: str,
    allowed: frozenset[str],
) -> dict[str, str] | None:
    attributes: dict[str, str] = {}
    cursor = 0
    for match in re.finditer(r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"', header):
        name, raw = match.groups()
        value = raw.strip()
        if header[cursor : match.start()].strip():
            return None
        if name not in allowed or name in attributes or not value:
            return None
        attributes[name] = value
        cursor = match.end()
    return attributes if not header[cursor:].strip() else None


def _decode_pipe_tag_value(raw: str, kind: str | None) -> tuple[bool, Any]:
    # A declared string type is authoritative: coercing "1" to a number there
    # would change the argument the client receives.
    if kind == "string":
        return True, raw
    candidate = raw.strip()
    try:
        parsed = parse_strict_json(candidate)
    except JsonNestingError:
        raise
    except (json.JSONDecodeError, ValueError):
        return (True, candidate) if kind is None else (False, None)
    if kind is None:
        return True, parsed
    matches = {
        "array": isinstance(parsed, list),
        "boolean": isinstance(parsed, bool),
        "integer": isinstance(parsed, int) and not isinstance(parsed, bool),
        "null": parsed is None,
        "number": isinstance(parsed, (int, float)) and not isinstance(parsed, bool),
        "object": isinstance(parsed, dict),
    }
    return matches.get(kind, False), parsed


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
    if body[:index].strip():
        return None
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
        if not _append_packed_call(calls, {"name": name, "arguments": arguments}):
            return None
        call_end += len(_KIMI_CALL_CLOSE)
        next_index = body.find(_KIMI_CALL_OPEN, call_end)
        trailing_end = next_index if next_index >= 0 else len(body)
        if body[call_end:trailing_end].strip():
            return None
        index = next_index
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


def _decode_deepseek_calls(
    body: str,
    allowed_tool_names: frozenset[str],  # noqa: ARG001 - uniform decoder signature
) -> list[dict[str, Any]] | None:
    """Rebuilds DeepSeek's tool-call section.

    V3 shape per call: ``<..call begin..>function<..sep..>name`` then the
    arguments in a ```json fence. V3.1 drops both the type field and the fence:
    ``<..call begin..>name<..sep..>{"k":"v"}``. Calls follow each other with no
    separator, optionally with a newline between them.
    """
    if _DEEPSEEK_CALL_OPEN not in body:
        return None
    calls: list[dict[str, Any]] = []
    index = body.find(_DEEPSEEK_CALL_OPEN)
    if body[:index].strip():
        return None
    while index >= 0:
        start = index + len(_DEEPSEEK_CALL_OPEN)
        call_end = body.find(_DEEPSEEK_CALL_CLOSE, start)
        separator = body.find(_DEEPSEEK_SEP, start)
        if call_end < 0 or separator < 0 or separator > call_end:
            # A call without its own closing token is truncated, and dropping
            # its arguments silently would change what the client executes.
            return None
        name, raw = _split_deepseek_call(
            body[start:separator],
            body[separator + len(_DEEPSEEK_SEP) : call_end],
        )
        arguments = _decode_json_object(raw) if name else None
        if name is None or arguments is None:
            return None
        if not _append_packed_call(calls, {"name": name, "arguments": arguments}):
            return None
        call_end += len(_DEEPSEEK_CALL_CLOSE)
        next_index = body.find(_DEEPSEEK_CALL_OPEN, call_end)
        trailing_end = next_index if next_index >= 0 else len(body)
        if body[call_end:trailing_end].strip():
            return None
        index = next_index
    return calls


def _split_deepseek_call(head: str, rest: str) -> tuple[str | None, str]:
    name, newline, arguments = rest.partition("\n")
    named_first_line = bool(newline) and arguments.lstrip().startswith(("```", "{"))
    if named_first_line and not rest.lstrip().startswith("{"):
        # V3 spends the field before the separator on the tool type and puts the
        # name on the first line after it.
        return name.strip() or None, arguments
    return head.strip() or None, rest


def _decode_function_parameter_tags(
    body: str,
    allowed_tool_names: frozenset[str],  # noqa: ARG001 - uniform decoder signature
) -> list[dict[str, Any]] | None:
    """Rebuilds Qwen3's XML-shaped call.

    Shape: ``<function=name><parameter=key>value</parameter></function>``, with
    one newline of markup around each value.

    The value body is never fed to an XML parser, because a parameter may
    legally carry XML-looking text that must survive unchanged.
    """
    if _QWEN_FUNCTION_OPEN not in body:
        return None
    calls: list[dict[str, Any]] = []
    index = body.find(_QWEN_FUNCTION_OPEN)
    if body[:index].strip():
        return None
    while index >= 0:
        name_end = body.find(">", index)
        if name_end < 0:
            return None
        parsed = _qwen_call(body, name_end)
        if parsed is None:
            return None
        close, arguments = parsed
        name = body[index + len(_QWEN_FUNCTION_OPEN) : name_end].strip()
        next_index = body.find(_QWEN_FUNCTION_OPEN, close + len(_QWEN_FUNCTION_CLOSE))
        trailing_end = next_index if next_index >= 0 else len(body)
        if not name:
            return None
        if body[close + len(_QWEN_FUNCTION_CLOSE) : trailing_end].strip():
            return None
        if not _append_packed_call(calls, {"name": name, "arguments": arguments}):
            return None
        index = next_index
    return calls


def _qwen_call(body: str, name_end: int) -> tuple[int, dict[str, Any]] | None:
    arguments: dict[str, Any] = {}
    cursor = name_end + 1
    while True:
        close = body.find(_QWEN_FUNCTION_CLOSE, cursor)
        parameter_index = body.find(_QWEN_PARAMETER_OPEN, cursor)
        if close >= 0 and (parameter_index < 0 or close < parameter_index):
            return (close, arguments) if not body[cursor:close].strip() else None
        if parameter_index < 0 or body[cursor:parameter_index].strip():
            return None
        key_end = body.find(">", parameter_index)
        if key_end < 0:
            return None
        key = body[parameter_index + len(_QWEN_PARAMETER_OPEN) : key_end].strip()
        parameter_token = _QWEN_PARAMETER_TOKEN.search(body, key_end + 1)
        if parameter_token is None or not key or key in arguments:
            return None
        raw_value = body[key_end + 1 : parameter_token.start()]
        if _QWEN_PARAMETER_OPEN in raw_value:
            return None
        arguments[key] = _qwen_value(raw_value)
        cursor = parameter_token.end()


def _qwen_value(raw: str) -> Any:
    value = raw.removeprefix("\n").removesuffix("\n")
    candidate = value.strip()
    if candidate.startswith(("{", "[")):
        # The template serializes mappings and sequences as JSON, so those are
        # recoverable. A scalar carries no type on the wire: only the tool
        # schema knows whether 2 meant a number, so it stays the text the model
        # wrote instead of being guessed into another type.
        try:
            return parse_strict_json(candidate)
        except JsonNestingError:
            raise
        except (json.JSONDecodeError, ValueError):
            return value
    return value


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


def _decode_json_arg_value_close(
    body: str,
    allowed_tool_names: frozenset[str],
) -> list[dict[str, Any]] | None:
    """Repairs strict JSON calls wrapped in GLM's leftover template markers.

    GLM ends a call with a mismatched ``</arg_value>`` and packs further calls
    by repeating ``<tool_call>`` without the matching closes. The two residues
    are independent, so ``{...}</arg_value>``, ``{...}<tool_call>{...}`` and
    ``{...}</arg_value><tool_call>{...}</arg_value>`` all arrive on the wire.
    Every segment must strict-parse to an object naming an allowed tool, which
    also keeps a segment left empty by a doubled marker failing closed instead
    of dropping the call whose payload never arrived.
    """
    stripped = body.strip()
    calls: list[dict[str, Any]] = []
    index = 0
    while True:
        try:
            parsed, end = raw_decode_strict(stripped, index)
        except (json.JSONDecodeError, ValueError):
            return None
        if (
            not isinstance(parsed, dict)
            or not isinstance(parsed.get("name"), str)
            or parsed["name"] not in allowed_tool_names
        ):
            return None
        if not _append_packed_call(calls, parsed):
            return None
        index = _skip_whitespace(stripped, end)
        if stripped.startswith(_ARG_VALUE_CLOSE, index):
            index = _skip_whitespace(stripped, index + len(_ARG_VALUE_CLOSE))
        if index == len(stripped):
            return calls
        if not stripped.startswith(TOOL_CALL_OPEN, index):
            return None
        index = _skip_whitespace(stripped, index + len(TOOL_CALL_OPEN))
        if index == len(stripped):
            return None


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
        if not key or _has_arg_control_markup(key) or key in arguments or value_start < 0:
            return None
        key_trailing = body[key_end + len(_ARG_KEY_CLOSE) : value_start]
        if key_trailing.strip():
            return None
        value_end = body.find(_ARG_VALUE_CLOSE, value_start)
        if value_end < 0:
            # A truncated value would silently drop data, so fail closed.
            return None
        raw = body[value_start + len(_ARG_VALUE_OPEN) : value_end].strip()
        arguments[key] = _coerce_arg_value(raw)
        value_end += len(_ARG_VALUE_CLOSE)
        index = body.find(_ARG_KEY_OPEN, value_end)
        trailing_end = index if index >= 0 else len(body)
        if body[value_end:trailing_end].strip():
            return None
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
    except JsonNestingError:
        raise
    except (json.JSONDecodeError, ValueError):
        return raw
    return parsed


def _has_arg_control_markup(value: str) -> bool:
    return any(prefix in value for prefix in _ARG_CONTROL_PREFIXES)


_PYTHON_CALL_PATTERN = re.compile(r"([A-Za-z_][\w.-]*)\(\s*(?=\{)")
# Closing punctuation a template repeats around a call. It carries no argument
# data, so skipping it between and after calls cannot change what the client
# executes, while prose still fails to parse as the next segment.
_CALL_RESIDUE = ")}"


def _decode_python_call(
    body: str,
    allowed_tool_names: frozenset[str],
) -> list[dict[str, Any]] | None:
    """Rebuilds one or more python-call segments: ``name({"key":"value"})``.

    Observed on GLM-family turns: the model answers with the function call it
    would write in code instead of the requested JSON object, packs several of
    them into one marker pair, and repeats the closing ``})`` after the last
    one. Every segment must name an allowed tool and carry one strict JSON
    object, so prose containing parentheses stays rejected.
    """
    stripped = body.strip()
    calls: list[dict[str, Any]] = []
    index = 0
    while len(calls) < MAX_PACKED_CALLS:
        parsed = _decode_python_call_segment(stripped, index, allowed_tool_names)
        if parsed is None:
            return None
        call, index = parsed
        calls.append(call)
        index = _skip_call_residue(stripped, index)
        if index == len(stripped):
            return calls
        if stripped.startswith(TOOL_CALL_OPEN, index):
            index = _skip_call_residue(stripped, index + len(TOOL_CALL_OPEN))
            if index == len(stripped):
                return None
    return None


def _decode_python_call_segment(
    segment: str,
    index: int,
    allowed_tool_names: frozenset[str],
) -> tuple[dict[str, Any], int] | None:
    match = _PYTHON_CALL_PATTERN.match(segment, index)
    if match is None:
        return None
    name = match.group(1)
    if name not in allowed_tool_names:
        return None
    try:
        arguments, end = raw_decode_strict(segment, match.end())
    except (json.JSONDecodeError, ValueError):
        return None
    end = _skip_whitespace(segment, end)
    if not segment.startswith(")", end):
        return None
    return {"name": name, "arguments": arguments}, end + 1


def _skip_call_residue(value: str, index: int) -> int:
    index = _skip_whitespace(value, index)
    while index < len(value) and value[index] in _CALL_RESIDUE:
        index = _skip_whitespace(value, index + 1)
    return index


_BARE_CALL_PATTERN = re.compile(r"([A-Za-z_][\w.-]*)\s*(?=\{)")


def _decode_bare_call(
    body: str,
    allowed_tool_names: frozenset[str],
) -> list[dict[str, Any]] | None:
    """Rebuilds one or more ``name{"key":"value"}`` calls.

    GLM may pack calls by repeating the opening marker without emitting the
    matching closes. Every segment must name an allowed tool and carry one
    strict JSON object. A single GLM value-close residue may separate segments
    or terminate the payload; other residue and partial calls remain rejected.
    Segments are walked by offset instead of re-sliced, keeping a payload at
    the byte cap linear work.
    """
    stripped = body.strip()
    calls: list[dict[str, Any]] = []
    index = 0
    while len(calls) < MAX_PACKED_CALLS:
        parsed = _decode_bare_call_segment(stripped, index, allowed_tool_names)
        if parsed is None:
            return None
        call, index = parsed
        calls.append(call)
        index = _skip_whitespace(stripped, index)
        if stripped.startswith(_ARG_VALUE_CLOSE, index):
            index = _skip_whitespace(stripped, index + len(_ARG_VALUE_CLOSE))
        if index == len(stripped):
            return calls
        if not stripped.startswith(TOOL_CALL_OPEN, index):
            return None
        index = _skip_whitespace(stripped, index + len(TOOL_CALL_OPEN))
        if index == len(stripped):
            return None
    return None


def _decode_bare_call_segment(
    segment: str,
    index: int,
    allowed_tool_names: frozenset[str],
) -> tuple[dict[str, Any], int] | None:
    match = _BARE_CALL_PATTERN.match(segment, index)
    if match is None:
        return None
    name = match.group(1)
    if name not in allowed_tool_names:
        return None
    try:
        arguments, end = raw_decode_strict(segment, match.end())
    except (json.JSONDecodeError, ValueError):
        return None
    return {"name": name, "arguments": arguments}, end


def _skip_whitespace(value: str, index: int) -> int:
    while index < len(value) and value[index].isspace():
        index += 1
    return index


# GLM-5.2 drops both the ``{"name":"`` opening and the ``","arguments":{"``
# infix, so the first argument key fuses onto the tool name. The fused key is
# still a plain identifier followed by the JSON ``":"`` separator, the only
# shape observed on the wire: a first argument that is not a string stays
# rejected rather than guessed at.
_FUSED_FIRST_KEY_PATTERN = re.compile(r'[A-Za-z_][A-Za-z0-9_]*":"')
_FUSED_OBJECT_OPEN = '{"'


def _decode_lost_prefix_fused(
    body: str,
    allowed_tool_names: frozenset[str],
) -> list[dict[str, Any]] | None:
    """Rebuilds GLM calls whose tool name fuses with the first argument key.

    Observed on GLM-5.2 turns: the payload drops the ``{"name":"`` opening and
    the ``","arguments":{"`` infix, so ``run_in_terminalmode":"sync",...}``
    repeats per call, packed by a bare ``<tool_call>`` without any close. A
    single GLM value-close residue may separate segments or terminate the
    payload; other residue and partial calls remain rejected.

    The split between name and key must be the single allowed tool whose
    remainder strict-parses as a JSON object; zero or several parses fail
    closed, so an alias collision or plain garbage stays rejected. Segments are
    walked from the end offset the strict decoder reports instead of split on
    the marker, so an argument value holding a literal ``<tool_call>`` survives
    intact.

    This form loses more bytes than the opt-in lost-prefix repair yet stays
    always-on, because it anchors on an exact allowed tool name, a complete
    strict JSON object and a single surviving reading, where that repair only
    has the name and the ``","`` the wire kept.
    """
    stripped = body.strip()
    calls: list[dict[str, Any]] = []
    index = 0
    while len(calls) < MAX_PACKED_CALLS:
        parsed = _decode_fused_segment(stripped, index, allowed_tool_names)
        if parsed is None:
            return None
        call, index = parsed
        calls.append(call)
        index = _skip_whitespace(stripped, index)
        if stripped.startswith(_ARG_VALUE_CLOSE, index):
            index = _skip_whitespace(stripped, index + len(_ARG_VALUE_CLOSE))
        if index == len(stripped):
            return calls
        if not stripped.startswith(TOOL_CALL_OPEN, index):
            return None
        index = _skip_whitespace(stripped, index + len(TOOL_CALL_OPEN))
        if index == len(stripped):
            return None
    return None


def _decode_fused_segment(
    segment: str,
    index: int,
    allowed_tool_names: frozenset[str],
) -> tuple[dict[str, Any], int] | None:
    candidates: list[tuple[str, dict[str, Any], int, bool]] = []
    for name in allowed_tool_names:
        if not segment.startswith(name, index):
            continue
        start = index + len(name)
        try:
            # The forced ``{"`` opening means a successful parse is always a
            # dict, and its end offset maps back by the two forced bytes.
            arguments, end = raw_decode_strict(_FUSED_OBJECT_OPEN + segment[start:], 0)
        except JsonNestingError:
            raise
        except (json.JSONDecodeError, ValueError):
            continue
        candidates.append(
            (
                name,
                arguments,
                start + end - len(_FUSED_OBJECT_OPEN),
                _FUSED_FIRST_KEY_PATTERN.match(segment, start) is not None,
            )
        )
    if len(candidates) != 1:
        # Ambiguity is counted before the key shape is judged: a split the key
        # pattern would reject still proves the payload has two readings, and
        # keeping the surviving one would dispatch a different tool than the
        # model asked for.
        return None
    name, arguments, end, fused_key = candidates[0]
    if not fused_key:
        return None
    return {"name": name, "arguments": arguments}, end


_ARG_KEY_REPAIR_TERMINATORS = ("<arg_key>", _ARG_VALUE_CLOSE)


def _decode_arg_key_value_repair(
    body: str,
    allowed_tool_names: frozenset[str],
) -> list[dict[str, Any]] | None:
    """Rebuilds GLM's mangled arg_key form observed in the wild.

    Shape per segment: ``name<arg_key>key":"value"}`` where the closing tags
    and braces are missing or half-JSON (``":"`` separators, a trailing
    ``"}``, a bare ``</arg_value>``). Several calls may be packed into one
    payload separated by a literal ``<tool_call>``. Only the quote/brace
    residue the template leaves behind is stripped; the value text itself is
    never re-guessed. Every segment's name must match an allowed tool, so
    prose that merely mentions ``<arg_key>`` stays rejected.
    """
    if _ARG_KEY_OPEN not in body:
        return None
    segments = _arg_key_repair_segments(body)
    if segments is None:
        return None
    calls: list[dict[str, Any]] = []
    for segment in segments:
        parsed = _arg_key_repair_segment(segment, allowed_tool_names)
        if parsed is None:
            return None
        calls.append(parsed)
    return calls or None


def _arg_key_repair_segments(body: str) -> list[str] | None:
    value = body.replace(TOOL_CALL_CLOSE, TOOL_CALL_OPEN).strip()
    start = 0
    if value.startswith(TOOL_CALL_OPEN):
        start = _skip_whitespace(value, len(TOOL_CALL_OPEN))
    if start >= len(value) or value.startswith(TOOL_CALL_OPEN, start):
        return None
    segments: list[str] = []
    while True:
        marker = _find_arg_key_separator(value, start)
        if marker < 0:
            segment = value[start:].strip()
            if not segment or len(segments) >= MAX_PACKED_CALLS:
                return None
            segments.append(segment)
            return segments
        segment = value[start:marker].strip()
        valid_end = (
            segment.endswith(_ARG_VALUE_CLOSE)
            if _ARG_VALUE_OPEN in segment
            else segment.endswith("}")
        )
        if not segment or not valid_end or len(segments) >= MAX_PACKED_CALLS:
            return None
        segments.append(segment)
        start = _skip_whitespace(value, marker + len(TOOL_CALL_OPEN))
        if start >= len(value) or value.startswith(TOOL_CALL_OPEN, start):
            return None


def _find_arg_key_separator(value: str, start: int) -> int:
    in_string = False
    escaped = False
    previous_significant: str | None = None
    index = 0
    while index < len(value):
        char = value[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if index >= start and value.startswith(TOOL_CALL_OPEN, index):
            return index
        if char == '"' and previous_significant in {None, "{", "[", ":", ","}:
            in_string = True
        if not char.isspace():
            previous_significant = char
        index += 1
    return -1


def _arg_key_repair_segment(
    segment: str,
    allowed_tool_names: frozenset[str],
) -> dict[str, Any] | None:
    if TOOL_CALL_OPEN in segment or TOOL_CALL_CLOSE in segment:
        return None
    index = segment.find(_ARG_KEY_OPEN)
    if index < 0:
        return None
    name = segment[:index].strip()
    if not name or name not in allowed_tool_names:
        return None
    arguments: dict[str, Any] = {}
    rest = segment[index:]
    while rest.startswith(_ARG_KEY_OPEN):
        rest = rest[len(_ARG_KEY_OPEN) :]
        entry = _arg_key_repair_entry(rest)
        if entry is None:
            return None
        key, value, rest = entry
        if key in arguments:
            return None
        arguments[key] = value
        rest = rest.lstrip()
    # Anything left over after the last value is markup residue only.
    if rest.strip("}\"'` \t\r\n"):
        return None
    return {"name": name, "arguments": arguments}


def _arg_key_repair_entry(rest: str) -> tuple[str, Any, str] | None:
    key_close = rest.find(_ARG_KEY_CLOSE)
    json_sep = rest.find('":"')
    if key_close >= 0 and (json_sep < 0 or key_close < json_sep):
        # Proper ``key</arg_key><arg_value>value`` form with a missing or
        # present closing </arg_value>.
        key = rest[:key_close].strip()
        tail = rest[key_close + len(_ARG_KEY_CLOSE) :]
        if not tail.startswith(_ARG_VALUE_OPEN):
            return None
        tail = tail[len(_ARG_VALUE_OPEN) :]
        value, remainder = _arg_key_repair_value(tail, quoted=False)
    elif json_sep > 0:
        # Mangled JSON form: ``key":"value`` with the opening quote spent on
        # the separator.
        key = rest[:json_sep].strip()
        value, remainder = _arg_key_repair_value(rest[json_sep + 3 :], quoted=True)
    else:
        return None
    if not key or _has_arg_control_markup(key):
        return None
    return key, value, remainder


def _arg_key_repair_value(text: str, *, quoted: bool) -> tuple[Any, str]:
    end = len(text)
    for terminator in _ARG_KEY_REPAIR_TERMINATORS:
        index = text.find(terminator)
        if index >= 0 and index < end:
            end = index
    raw = text[:end]
    remainder = text[end:]
    if remainder.startswith(_ARG_VALUE_CLOSE):
        remainder = remainder[len(_ARG_VALUE_CLOSE) :]
    raw = raw.strip()
    if raw.endswith('"}'):
        raw = raw[:-2]
    elif raw.endswith("}"):
        raw = raw[:-1]
    if quoted and raw.endswith('"'):
        raw = raw[:-1]
    return _coerce_arg_value(raw.strip()), remainder


_WRAPPER_KEYS = ("arguments", "parameters", "args", "input")


def _decode_lost_prefix(
    body: str,
    allowed_tool_names: frozenset[str],
) -> list[dict[str, Any]] | None:
    """Rebuilds a call whose payload lost its opening ``{"name":"`` bytes.

    Observed on GLM-family turns: the payload starts mid-object at
    ``name","key":"value"...}``. The name must match an allowed tool exactly
    and the remainder must strict-parse as a JSON object, so a payload that
    is merely garbage stays rejected. Longest names match first so a tool
    whose name prefixes another tool's name cannot shadow it.
    """
    for name in sorted(allowed_tool_names, key=len, reverse=True):
        for prefix in (f'{name}","', f'"{name}","'):
            if not body.startswith(prefix):
                continue
            # The forced ``{"`` opening means the remainder either parses as a
            # JSON object or does not parse at all.
            try:
                parsed = parse_strict_json('{"' + body[len(prefix) :])
            except (json.JSONDecodeError, ValueError):
                return None
            arguments: Any = parsed
            wrapper_keys = [key for key in _WRAPPER_KEYS if key in parsed]
            if len(wrapper_keys) > 1:
                # Two argument aliases in one payload is ambiguous, so the
                # repair fails closed like the strict path does.
                return None
            if len(parsed) == 1 and wrapper_keys and isinstance(parsed[wrapper_keys[0]], dict):
                # The wrapped form survived as ``"arguments":{...}``;
                # a single-key mapping unwraps back to the arguments.
                arguments = parsed[wrapper_keys[0]]
            return [{"name": name, "arguments": arguments}]
    return None


PAYLOAD_DECODERS: tuple[PayloadDecoder, ...] = (
    PayloadDecoder("pipe_tag_tokens", _decode_pipe_tag_tokens),
    PayloadDecoder("kimi_sections", _decode_kimi_sections),
    PayloadDecoder("harmony_commentary", _decode_harmony_commentary),
    PayloadDecoder("deepseek_calls", _decode_deepseek_calls),
    PayloadDecoder("function_parameter_tags", _decode_function_parameter_tags),
    PayloadDecoder("json_arg_value_close", _decode_json_arg_value_close),
    PayloadDecoder("arg_key_value", _decode_arg_key_value),
    PayloadDecoder("python_call", _decode_python_call),
    PayloadDecoder("arg_key_value_repair", _decode_arg_key_value_repair),
    PayloadDecoder("lost_prefix_fused", _decode_lost_prefix_fused),
    PayloadDecoder("bare_name", _decode_bare_name),
    PayloadDecoder("bare_call", _decode_bare_call),
)

# Opt-in only: the payload lost bytes, so the repair trusts less of the wire
# than the always-on decoders do. The stream parser appends this decoder only
# when the operator enables lost-prefix repair.
LOST_PREFIX_DECODER = PayloadDecoder("lost_prefix", _decode_lost_prefix)
