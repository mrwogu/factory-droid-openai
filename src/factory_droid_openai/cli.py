from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import uvicorn

from factory_droid_openai.config import Settings

if TYPE_CHECKING:
    from collections.abc import Sequence


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _at_least_two(value: str) -> int:
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("must be at least 2")
    return parsed


def main(argv: Sequence[str] | None = None) -> None:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(
        description="Run the Factory Droid OpenAI-compatible bridge.",
    )
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument(
        "--limit-concurrency",
        type=_at_least_two,
        default=settings.server_limit_concurrency,
    )
    parser.add_argument("--backlog", type=_positive_int, default=settings.server_backlog)
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default="info",
    )
    args = parser.parse_args(argv)
    uvicorn.run(
        "factory_droid_openai.app:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        limit_concurrency=args.limit_concurrency,
        backlog=args.backlog,
    )


if __name__ == "__main__":
    main()
