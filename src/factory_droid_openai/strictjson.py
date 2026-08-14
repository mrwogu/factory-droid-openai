from __future__ import annotations

import json
import math
from typing import Any

__all__ = [
    "DuplicateKeyError",
    "JsonNestingError",
    "check_no_duplicate_keys",
    "decode_json_values",
    "json_depth_exceeds",
    "parse_strict_json",
    "raw_decode_strict",
]


class DuplicateKeyError(ValueError):
    """Raised when a JSON object repeats a key the bridge must not accept."""


class JsonNestingError(ValueError):
    """Raised when Python's JSON decoder cannot safely traverse the input."""


def check_no_duplicate_keys(text: str) -> None:
    """Rejects a JSON document that repeats any object key.

    Unlike :func:`parse_strict_json` this keeps the standard JSON number
    handling, so it only adds the duplicate-key rejection the request-body
    parser needs, without changing what FastAPI would otherwise accept.
    """
    json.loads(text, object_pairs_hook=_reject_duplicate_keys)


def parse_strict_json(text: str) -> Any:
    """Parses model-generated JSON, rejecting everything outside RFC 8259.

    Duplicate keys, ``NaN``/``Infinity`` literals and numbers that overflow to
    a non-finite float are all refused, so a caller never forwards a value a
    strict JSON parser on the client side would reject.
    """
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_json_constant,
            parse_float=_parse_finite_float,
        )
    except RecursionError as exc:
        raise JsonNestingError("JSON nesting exceeds the parser limit") from exc
    _check_utf8_strings(parsed)
    return parsed


def decode_json_values(text: str, *, max_values: int | None = None) -> list[Any]:
    """Decodes the JSON values a payload holds, back to back.

    A model that packs several tool calls into one marker pair produces
    ``{...}{...}``, which ``json.loads`` rejects as trailing data. When a
    caller supplies a limit, one extra value is enough to prove overflow.
    """
    values: list[Any] = []
    index = 0
    while True:
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            return values
        value, index = raw_decode_strict(text, index)
        values.append(value)
        if max_values is not None and len(values) > max_values:
            return values


def raw_decode_strict(text: str, index: int) -> tuple[Any, int]:
    """Decodes the strict JSON value at ``index`` and reports where it ends.

    The end offset is what separates a packed payload's calls from the residue
    that follows them, so the boundary is taken from the same strict decoder
    that validates the value instead of a permissive second pass.
    """
    try:
        parsed, end = _STRICT_DECODER.raw_decode(text, index)
    except RecursionError as exc:
        raise JsonNestingError("JSON nesting exceeds the parser limit") from exc
    _check_utf8_strings(parsed)
    return parsed, end


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate key '{key}'")
        result[key] = value
    return result


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"non-JSON numeric constant '{value}'")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite number '{value}'")
    return parsed


def _check_utf8_strings(value: Any) -> None:
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, str):
            try:
                current.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError("JSON string contains an invalid Unicode surrogate") from exc
        elif isinstance(current, dict):
            stack.extend(current.keys())
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)


def json_depth_exceeds(value: Any, max_depth: int) -> bool:
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        if not isinstance(current, (dict, list)):
            continue
        if depth > max_depth:
            return True
        children = current.values() if isinstance(current, dict) else current
        stack.extend((child, depth + 1) for child in children)
    return False


_STRICT_DECODER = json.JSONDecoder(
    object_pairs_hook=_reject_duplicate_keys,
    parse_constant=_reject_non_json_constant,
    parse_float=_parse_finite_float,
)
