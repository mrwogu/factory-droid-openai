from __future__ import annotations

import json
import math
from typing import Any

__all__ = [
    "DuplicateKeyError",
    "check_no_duplicate_keys",
    "decode_json_values",
    "parse_strict_json",
    "raw_decode_strict",
]


class DuplicateKeyError(ValueError):
    """Raised when a JSON object repeats a key the bridge must not accept."""


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
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_non_json_constant,
        parse_float=_parse_finite_float,
    )


def decode_json_values(text: str) -> list[Any]:
    """Decodes the JSON values a payload holds, back to back.

    A model that packs several tool calls into one marker pair produces
    ``{...}{...}``, which ``json.loads`` rejects as trailing data.
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


def raw_decode_strict(text: str, index: int) -> tuple[Any, int]:
    """Decodes the strict JSON value at ``index`` and reports where it ends.

    The end offset is what separates a packed payload's calls from the residue
    that follows them, so the boundary is taken from the same strict decoder
    that validates the value instead of a permissive second pass.
    """
    return _STRICT_DECODER.raw_decode(text, index)


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


_STRICT_DECODER = json.JSONDecoder(
    object_pairs_hook=_reject_duplicate_keys,
    parse_constant=_reject_non_json_constant,
    parse_float=_parse_finite_float,
)
