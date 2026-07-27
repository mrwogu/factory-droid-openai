from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import uvicorn

from factory_droid_openai.config import Settings
from factory_droid_openai.logs import LOG_FORMATS, LOG_LEVELS, configure_logging

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
        choices=LOG_LEVELS,
        default=settings.log_level,
    )
    parser.add_argument(
        "--log-format",
        choices=LOG_FORMATS,
        default=settings.log_format,
    )
    parser.add_argument(
        "--access-log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Uvicorn per-request access log lines.",
    )
    args = parser.parse_args(argv)

    configure_logging(level=args.log_level, log_format=args.log_format)

    if not settings.api_key:
        import sys

        print(
            "  WARNING  FACTORY_DROID_OPENAI_API_KEY is not set.\n"
            "           The bridge accepts unauthenticated requests.\n"
            "           Generate a token and export it to secure the bridge:\n"
            '             export FACTORY_DROID_OPENAI_API_KEY="$(python -c \\\n'
            "               'import secrets; print(secrets.token_urlsafe(32))')\"\n",
            file=sys.stderr,
        )

    uvicorn.run(
        "factory_droid_openai.app:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=args.access_log,
        limit_concurrency=args.limit_concurrency,
        backlog=args.backlog,
    )


if __name__ == "__main__":
    main()
