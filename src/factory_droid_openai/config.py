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
    max_tool_calls: int = 8
    max_attachments: int = 16
    max_attachment_bytes: int = 8_388_608
    max_choices: int = 4
    max_stop_sequences: int = 4
    session_continuity: bool = False
    max_tracked_sessions: int = 256
    worktree: str | None = None
    append_system_prompt_file: Path | None = None
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
        max_tool_calls = _positive_int(
            "FACTORY_DROID_OPENAI_MAX_TOOL_CALLS",
            default=8,
        )
        max_attachments = _non_negative_int(
            "FACTORY_DROID_OPENAI_MAX_ATTACHMENTS",
            default=16,
        )
        max_attachment_bytes = _positive_int(
            "FACTORY_DROID_OPENAI_MAX_ATTACHMENT_BYTES",
            default=8_388_608,
        )
        max_choices = _positive_int(
            "FACTORY_DROID_OPENAI_MAX_CHOICES",
            default=4,
        )
        max_stop_sequences = _non_negative_int(
            "FACTORY_DROID_OPENAI_MAX_STOP_SEQUENCES",
            default=4,
        )
        session_continuity = _boolean(
            "FACTORY_DROID_OPENAI_SESSION_CONTINUITY",
            default=False,
        )
        max_tracked_sessions = _positive_int(
            "FACTORY_DROID_OPENAI_MAX_TRACKED_SESSIONS",
            default=256,
        )
        worktree = os.getenv("FACTORY_DROID_OPENAI_WORKTREE") or None
        append_system_prompt_file = _optional_file(
            "FACTORY_DROID_OPENAI_APPEND_SYSTEM_PROMPT_FILE",
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
            max_tool_calls=max_tool_calls,
            max_attachments=max_attachments,
            max_attachment_bytes=max_attachment_bytes,
            max_choices=max_choices,
            max_stop_sequences=max_stop_sequences,
            session_continuity=session_continuity,
            max_tracked_sessions=max_tracked_sessions,
            worktree=worktree,
            append_system_prompt_file=append_system_prompt_file,
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


def _boolean(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _optional_file(name: str) -> Path | None:
    raw = os.getenv(name)
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_file():
        raise ValueError(f"{name} does not point at a readable file: {path}")
    return path.resolve()


def _non_negative_int(name: str, *, default: int) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None else default
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be zero or greater")
    return value
