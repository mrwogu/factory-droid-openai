from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8787
    api_key: str | None = None
    droid_path: str = "droid"
    workdir: Path = field(default_factory=Path.cwd)
    timeout_seconds: float = 600.0
    body_timeout_seconds: float = 30.0
    max_concurrency: int = 1
    max_queue_size: int = 8
    max_request_bytes: int = 4_194_304
    max_messages: int = 512
    max_tools: int = 128
    max_transcript_bytes: int = 4_194_304
    max_tool_schema_bytes: int = 1_048_576
    max_json_depth: int = 32
    retry_after_seconds: int = 1
    process_grace_seconds: float = 1.0
    cleanup_timeout_seconds: float = 4.0
    server_limit_concurrency: int = 64
    server_backlog: int = 128
    model_alias: str = "factory-droid"

    @classmethod
    def from_env(cls) -> Settings:
        configured_workdir = os.getenv("FACTORY_DROID_OPENAI_WORKDIR")
        workdir = Path(configured_workdir).expanduser() if configured_workdir else Path.cwd()
        if not workdir.is_dir():
            raise ValueError(f"Factory Droid workdir does not exist: {workdir}")
        timeout_seconds = _positive_float(
            "FACTORY_DROID_OPENAI_TIMEOUT_SECONDS",
            default=600.0,
        )
        body_timeout_seconds = _positive_float(
            "FACTORY_DROID_OPENAI_BODY_TIMEOUT_SECONDS",
            default=30.0,
        )
        max_concurrency = _positive_int(
            "FACTORY_DROID_OPENAI_MAX_CONCURRENCY",
            default=1,
        )
        max_queue_size = _non_negative_int(
            "FACTORY_DROID_OPENAI_MAX_QUEUE_SIZE",
            default=8,
        )
        max_request_bytes = _positive_int(
            "FACTORY_DROID_OPENAI_MAX_REQUEST_BYTES",
            default=4_194_304,
        )
        max_messages = _positive_int(
            "FACTORY_DROID_OPENAI_MAX_MESSAGES",
            default=512,
        )
        max_tools = _positive_int(
            "FACTORY_DROID_OPENAI_MAX_TOOLS",
            default=128,
        )
        max_transcript_bytes = _positive_int(
            "FACTORY_DROID_OPENAI_MAX_TRANSCRIPT_BYTES",
            default=4_194_304,
        )
        max_tool_schema_bytes = _positive_int(
            "FACTORY_DROID_OPENAI_MAX_TOOL_SCHEMA_BYTES",
            default=1_048_576,
        )
        max_json_depth = _positive_int(
            "FACTORY_DROID_OPENAI_MAX_JSON_DEPTH",
            default=32,
        )
        retry_after_seconds = _positive_int(
            "FACTORY_DROID_OPENAI_RETRY_AFTER_SECONDS",
            default=1,
        )
        process_grace_seconds = _positive_float(
            "FACTORY_DROID_OPENAI_PROCESS_GRACE_SECONDS",
            default=1.0,
        )
        cleanup_timeout_seconds = _positive_float(
            "FACTORY_DROID_OPENAI_CLEANUP_TIMEOUT_SECONDS",
            default=4.0,
        )
        if cleanup_timeout_seconds <= process_grace_seconds:
            raise ValueError(
                "FACTORY_DROID_OPENAI_CLEANUP_TIMEOUT_SECONDS must be greater than "
                "FACTORY_DROID_OPENAI_PROCESS_GRACE_SECONDS"
            )
        server_limit_concurrency = _positive_int(
            "FACTORY_DROID_OPENAI_UVICORN_LIMIT_CONCURRENCY",
            default=64,
        )
        if server_limit_concurrency < 2:
            raise ValueError("FACTORY_DROID_OPENAI_UVICORN_LIMIT_CONCURRENCY must be at least 2")
        server_backlog = _positive_int(
            "FACTORY_DROID_OPENAI_UVICORN_BACKLOG",
            default=128,
        )
        port = _positive_int("FACTORY_DROID_OPENAI_PORT", default=8787)
        if port > 65535:
            raise ValueError("FACTORY_DROID_OPENAI_PORT must be at most 65535")
        return cls(
            host=os.getenv("FACTORY_DROID_OPENAI_HOST", "127.0.0.1"),
            port=port,
            api_key=os.getenv("FACTORY_DROID_OPENAI_API_KEY") or None,
            droid_path=os.getenv("FACTORY_DROID_PATH", "droid"),
            workdir=workdir.resolve(),
            timeout_seconds=timeout_seconds,
            body_timeout_seconds=body_timeout_seconds,
            max_concurrency=max_concurrency,
            max_queue_size=max_queue_size,
            max_request_bytes=max_request_bytes,
            max_messages=max_messages,
            max_tools=max_tools,
            max_transcript_bytes=max_transcript_bytes,
            max_tool_schema_bytes=max_tool_schema_bytes,
            max_json_depth=max_json_depth,
            retry_after_seconds=retry_after_seconds,
            process_grace_seconds=process_grace_seconds,
            cleanup_timeout_seconds=cleanup_timeout_seconds,
            server_limit_concurrency=server_limit_concurrency,
            server_backlog=server_backlog,
            model_alias=os.getenv("FACTORY_DROID_OPENAI_MODEL_ALIAS", "factory-droid"),
        )


def _positive_float(name: str, *, default: float) -> float:
    raw = os.getenv(name)
    try:
        value = float(raw) if raw is not None else default
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be greater than zero and finite")
    return value


def _positive_int(name: str, *, default: int) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None else default
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _non_negative_int(name: str, *, default: int) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None else default
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be zero or greater")
    return value
