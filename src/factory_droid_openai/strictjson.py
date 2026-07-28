from __future__ import annotations

import json
import math
from typing import Any

__all__ = [
    "decode_json_values",
    "parse_strict_json",
]


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
    decoder = json.JSONDecoder(
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_non_json_constant,
        parse_float=_parse_finite_float,
    )
    values: list[Any] = []
    index = 0
    while True:
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            return values
        value, index = decoder.raw_decode(text, index)
        values.append(value)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key '{key}'")
        result[key] = value
    return result


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"non-JSON numeric constant '{value}'")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite number '{value}'")
    return parsed
