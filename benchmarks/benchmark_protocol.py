from __future__ import annotations

import json
import time
import tracemalloc
from typing import TYPE_CHECKING, Any

from factory_droid_openai.app import CollectedCompletion, _apply_emissions
from factory_droid_openai.models import ChatCompletionRequest
from factory_droid_openai.protocol import (
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    TextEmission,
    ToolCallStreamParser,
    build_prompt,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _measure(operation: Callable[[], Any]) -> dict[str, float | int]:
    tracemalloc.start()
    started = time.perf_counter()
    operation()
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"seconds": elapsed, "peak_bytes": peak}


def _parser(chunk_size: int) -> None:
    payload = json.dumps(
        {
            "name": "benchmark",
            "arguments": {"value": "x" * 900_000},
        },
        separators=(",", ":"),
    )
    stream = f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE}"
    parser = ToolCallStreamParser(frozenset({"benchmark"}))
    emissions = []
    for offset in range(0, len(stream), chunk_size):
        emissions.extend(parser.feed(stream[offset : offset + chunk_size]))
    emissions.extend(parser.finish())
    if len(emissions) != 1:
        raise RuntimeError("parser benchmark produced invalid output")


def _collector() -> None:
    result = CollectedCompletion()
    for _ in range(65_536):
        _apply_emissions(result, [TextEmission("0123456789abcdef")])
    if len(result.text) != 1_048_576:
        raise RuntimeError("collector benchmark produced invalid output")


def _prompt() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "model": "factory-droid",
            "messages": [{"role": "user", "content": "x" * 4_000} for _ in range(512)],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": f"tool_{index}",
                        "parameters": {
                            "type": "object",
                            "properties": {"value": {"type": "string"}},
                        },
                    },
                }
                for index in range(128)
            ],
        }
    )
    plan = build_prompt(request)
    if not plan.prompt:
        raise RuntimeError("prompt benchmark produced invalid output")


def main() -> None:
    results = {
        "parser_chunk_16": _measure(lambda: _parser(16)),
        "parser_chunk_8192": _measure(lambda: _parser(8192)),
        "collector_1mib": _measure(_collector),
        "prompt_limits": _measure(_prompt),
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
