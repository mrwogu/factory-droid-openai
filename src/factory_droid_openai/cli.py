from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import uvicorn

from factory_droid_openai.config import Settings

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(
        description="Run the Factory Droid OpenAI-compatible bridge.",
    )
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
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
    )


if __name__ == "__main__":
    main()
