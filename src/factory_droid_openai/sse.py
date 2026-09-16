"""Shared SSE framing for the Chat Completions and Responses streams."""

from __future__ import annotations

import json
from typing import Any

__all__ = ["sse", "sse_data"]


def sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def sse_data(raw: str) -> Any:
    # Split on "\n" only. U+0085/U+2028/U+2029 are valid inside JSON strings,
    # but str.splitlines() treats them as line breaks and would truncate the
    # payload mid-string before json.loads sees it.
    for line in raw.split("\n"):
        if line.startswith("data: "):
            value = line[6:]
            if value == "[DONE]":
                return value
            return json.loads(value)
    return None
