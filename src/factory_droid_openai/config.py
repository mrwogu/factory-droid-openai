from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from droid_sdk.schemas.enums import ReasoningEffort

from factory_droid_openai.dialects import MAX_PACKED_CALLS
from factory_droid_openai.logs import LOG_FORMATS, LOG_LEVELS
from factory_droid_openai.payloadlog import PAYLOAD_TRACE_MODES

_REASONING_EFFORTS: tuple[str, ...] = tuple(effort.value for effort in ReasoningEffort)

# The model name that means "whatever Droid defaults to" instead of a concrete
# Droid model id.
DEFAULT_MODEL_ALIAS = "factory-droid"

# Droid does not always emit a completion event after a client-side tool call,
# so the bridge stops waiting this long after one arrives. Raise it when a
# model spreads parallel calls over slower gaps than this.
DEFAULT_TOOL_CALL_DRAIN_SECONDS = 0.5
DEFAULT_SESSION_INIT_TIMEOUT_SECONDS = 60.0
MAX_SESSION_INIT_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8787
    api_key: str | None = None
    droid_path: str = "droid"
    workdir: Path = field(default_factory=Path.cwd)
    timeout_seconds: float = 600.0
    session_init_timeout_seconds: float = DEFAULT_SESSION_INIT_TIMEOUT_SECONDS
    body_timeout_seconds: float = 30.0
    max_concurrency: int = 2
    max_queue_size: int = 8
    max_request_bytes: int = 4_194_304
    max_messages: int = 512
    max_tools: int = 128
    max_transcript_bytes: int = 4_194_304
    max_tool_schema_bytes: int = 1_048_576
    max_structured_output_bytes: int = 1_048_576
    max_json_depth: int = 32
    retry_after_seconds: int = 1
    process_grace_seconds: float = 5.0
    cleanup_timeout_seconds: float = 10.0
    server_limit_concurrency: int = 64
    server_backlog: int = 128
    # Defaults to the parser's own packing cap: a model that answers with a
    # burst of calls is asking the client for work it can run, and dropping the
    # turn over the count alone only costs a retry.
    max_tool_calls: int = MAX_PACKED_CALLS
    tool_call_drain_seconds: float = DEFAULT_TOOL_CALL_DRAIN_SECONDS
    max_attachments: int = 16
    max_attachment_bytes: int = 8_388_608
    max_choices: int = 4
    max_stop_sequences: int = 4
    session_continuity: bool = False
    max_tracked_sessions: int = 256
    mcp_settle_seconds: float = 0.0
    model_cache_seconds: float = 300.0
    model_quarantine_seconds: float = 900.0
    worktree: str | None = None
    append_system_prompt_file: Path | None = None
    model_alias: str = DEFAULT_MODEL_ALIAS
    # Overrides any reasoning_effort a request carries when set.
    reasoning_effort: str | None = None
    log_level: str = "info"
    log_format: str = "text"
    warm_sessions: int = -1
    warm_session_ttl_seconds: float = 600.0
    detached_cleanup: bool = True
    telemetry: bool = True
    # Off by default: the payload lost bytes, so the repair trusts less of the
    # wire than the always-on decoders do.
    repair_lost_prefix: bool = False
    # Off by default: publishing tools over MCP moves tool calling off the text
    # channel, which changes how a turn ends and costs a warm session per
    # tool-bearing request.
    native_tool_calls: bool = False
    # Where the Droid process reaches this bridge back. The bind host can be a
    # wildcard, which is not a usable target, so the loopback address is the
    # default and an override exists for a bridge behind a different port.
    native_tool_call_url: str | None = None
    # Off by default: prompts and tool payloads are private user content.
    trace_payloads: str = "off"
    trace_payload_file: Path | None = None

    def __post_init__(self) -> None:
        if self.max_tool_calls > MAX_PACKED_CALLS:
            raise ValueError(f"max_tool_calls must be at most {MAX_PACKED_CALLS}")
        if not math.isfinite(self.session_init_timeout_seconds) or (
            self.session_init_timeout_seconds <= 0
        ):
            raise ValueError("session_init_timeout_seconds must be greater than zero and finite")
        if self.session_init_timeout_seconds > MAX_SESSION_INIT_TIMEOUT_SECONDS:
            raise ValueError(
                "session_init_timeout_seconds must be at most "
                f"{MAX_SESSION_INIT_TIMEOUT_SECONDS:g} seconds"
            )
        # The warm pool keys sessions by this value, so it has to match the
        # normalized effort the request path sends to Droid, and a bad value
        # has to fail at construction instead of on every request.
        if self.reasoning_effort is None:
            return
        value = self.reasoning_effort.strip().lower()
        if not value:
            object.__setattr__(self, "reasoning_effort", None)
            return
        if value not in _REASONING_EFFORTS:
            raise ValueError(f"reasoning_effort must be one of: {', '.join(_REASONING_EFFORTS)}")
        object.__setattr__(self, "reasoning_effort", value)

    def warm_session_count(self) -> int:
        """Warm sessions to keep ready; ``-1`` keeps one spare per concurrency slot.

        A session serves a single turn, and refilling one costs seconds, so the
        spare covers back-to-back requests while the pool refills.
        """
        if self.warm_sessions < 0:
            return self.max_concurrency + 1
        return self.warm_sessions

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
        session_init_timeout_seconds = _positive_float(
            "FACTORY_DROID_OPENAI_SESSION_INIT_TIMEOUT_SECONDS",
            default=DEFAULT_SESSION_INIT_TIMEOUT_SECONDS,
        )
        if session_init_timeout_seconds > MAX_SESSION_INIT_TIMEOUT_SECONDS:
            raise ValueError(
                "FACTORY_DROID_OPENAI_SESSION_INIT_TIMEOUT_SECONDS must be at most "
                f"{MAX_SESSION_INIT_TIMEOUT_SECONDS:g} seconds for droid-sdk==0.1.2"
            )
        body_timeout_seconds = _positive_float(
            "FACTORY_DROID_OPENAI_BODY_TIMEOUT_SECONDS",
            default=30.0,
        )
        max_concurrency = _positive_int(
            "FACTORY_DROID_OPENAI_MAX_CONCURRENCY",
            default=2,
        )
        warm_sessions = _optional_non_negative_int("FACTORY_DROID_OPENAI_WARM_SESSIONS")
        warm_session_ttl_seconds = _positive_float(
            "FACTORY_DROID_OPENAI_WARM_SESSION_TTL_SECONDS",
            default=600.0,
        )
        detached_cleanup = _boolean(
            "FACTORY_DROID_OPENAI_DETACHED_CLEANUP",
            default=True,
        )
        repair_lost_prefix = _boolean(
            "FACTORY_DROID_OPENAI_REPAIR_LOST_PREFIX",
            default=False,
        )
        native_tool_calls = _boolean(
            "FACTORY_DROID_OPENAI_NATIVE_TOOL_CALLS",
            default=False,
        )
        native_tool_call_url = os.getenv("FACTORY_DROID_OPENAI_NATIVE_TOOL_CALL_URL") or None
        telemetry = _telemetry_enabled()
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
        max_structured_output_bytes = _positive_int(
            "FACTORY_DROID_OPENAI_MAX_STRUCTURED_OUTPUT_BYTES",
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
            # Teardown after a full turn keeps droid busy for seconds past
            # SIGTERM, so a shorter grace period turns every cleanup into a
            # SIGKILL.
            default=5.0,
        )
        cleanup_timeout_seconds = _positive_float(
            "FACTORY_DROID_OPENAI_CLEANUP_TIMEOUT_SECONDS",
            default=10.0,
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
            default=MAX_PACKED_CALLS,
        )
        tool_call_drain_seconds = _positive_float(
            "FACTORY_DROID_OPENAI_TOOL_CALL_DRAIN_SECONDS",
            default=DEFAULT_TOOL_CALL_DRAIN_SECONDS,
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
        mcp_settle_seconds = _non_negative_float(
            "FACTORY_DROID_OPENAI_MCP_SETTLE_SECONDS",
            default=0.0,
        )
        model_cache_seconds = _non_negative_float(
            "FACTORY_DROID_OPENAI_MODEL_CACHE_SECONDS",
            default=300.0,
        )
        model_quarantine_seconds = _non_negative_float(
            "FACTORY_DROID_OPENAI_MODEL_QUARANTINE_SECONDS",
            default=900.0,
        )
        worktree = os.getenv("FACTORY_DROID_OPENAI_WORKTREE") or None
        append_system_prompt_file = _optional_file(
            "FACTORY_DROID_OPENAI_APPEND_SYSTEM_PROMPT_FILE",
        )
        reasoning_effort = _optional_choice(
            "FACTORY_DROID_OPENAI_REASONING_EFFORT",
            allowed=_REASONING_EFFORTS,
        )
        log_level = _choice(
            "FACTORY_DROID_OPENAI_LOG_LEVEL",
            default="info",
            allowed=LOG_LEVELS,
        )
        log_format = _choice(
            "FACTORY_DROID_OPENAI_LOG_FORMAT",
            default="text",
            allowed=LOG_FORMATS,
        )
        trace_payloads = _choice(
            "FACTORY_DROID_OPENAI_TRACE_PAYLOADS",
            default="off",
            allowed=PAYLOAD_TRACE_MODES,
        )
        trace_payload_file = _optional_path("FACTORY_DROID_OPENAI_TRACE_FILE")
        if trace_payloads != "off" and trace_payload_file is None:
            # No implicit default: the Droid session runs with workdir as its cwd,
            # so a trace file placed there would expose every prompt to the agent.
            raise ValueError(
                "FACTORY_DROID_OPENAI_TRACE_FILE is required when "
                "FACTORY_DROID_OPENAI_TRACE_PAYLOADS is not 'off'",
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
            session_init_timeout_seconds=session_init_timeout_seconds,
            body_timeout_seconds=body_timeout_seconds,
            max_concurrency=max_concurrency,
            max_queue_size=max_queue_size,
            max_request_bytes=max_request_bytes,
            max_messages=max_messages,
            max_tools=max_tools,
            max_transcript_bytes=max_transcript_bytes,
            max_tool_schema_bytes=max_tool_schema_bytes,
            max_structured_output_bytes=max_structured_output_bytes,
            max_json_depth=max_json_depth,
            retry_after_seconds=retry_after_seconds,
            process_grace_seconds=process_grace_seconds,
            cleanup_timeout_seconds=cleanup_timeout_seconds,
            server_limit_concurrency=server_limit_concurrency,
            server_backlog=server_backlog,
            max_tool_calls=max_tool_calls,
            tool_call_drain_seconds=tool_call_drain_seconds,
            max_attachments=max_attachments,
            max_attachment_bytes=max_attachment_bytes,
            max_choices=max_choices,
            max_stop_sequences=max_stop_sequences,
            session_continuity=session_continuity,
            max_tracked_sessions=max_tracked_sessions,
            mcp_settle_seconds=mcp_settle_seconds,
            model_cache_seconds=model_cache_seconds,
            model_quarantine_seconds=model_quarantine_seconds,
            worktree=worktree,
            append_system_prompt_file=append_system_prompt_file,
            model_alias=os.getenv("FACTORY_DROID_OPENAI_MODEL_ALIAS", DEFAULT_MODEL_ALIAS),
            reasoning_effort=reasoning_effort,
            log_level=log_level,
            log_format=log_format,
            warm_sessions=warm_sessions,
            warm_session_ttl_seconds=warm_session_ttl_seconds,
            detached_cleanup=detached_cleanup,
            telemetry=telemetry,
            repair_lost_prefix=repair_lost_prefix,
            native_tool_calls=native_tool_calls,
            native_tool_call_url=native_tool_call_url,
            trace_payloads=trace_payloads,
            trace_payload_file=trace_payload_file,
        )

    def native_tool_call_base_url(self) -> str:
        """Base URL the Droid process uses to reach the bridge's MCP endpoint."""
        if self.native_tool_call_url is not None:
            return self.native_tool_call_url
        return f"http://127.0.0.1:{self.port}"


def _positive_float(name: str, *, default: float) -> float:
    raw = os.getenv(name)
    try:
        value = float(raw) if raw is not None else default
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be greater than zero and finite")
    return value


def _non_negative_float(name: str, *, default: float) -> float:
    raw = os.getenv(name)
    try:
        value = float(raw) if raw is not None else default
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be zero or greater and finite")
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


def _telemetry_enabled() -> bool:
    # The DO_NOT_TRACK convention treats any value other than "0" as opt-out.
    do_not_track = os.getenv("DO_NOT_TRACK", "").strip()
    if do_not_track not in {"", "0"}:
        return False
    return _boolean("FACTORY_DROID_OPENAI_TELEMETRY", default=True)


def _choice(name: str, *, default: str, allowed: tuple[str, ...]) -> str:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value not in allowed:
        raise ValueError(f"{name} must be one of: {', '.join(allowed)}")
    return value


def _optional_choice(name: str, *, allowed: tuple[str, ...]) -> str | None:
    """Return the configured value, or ``None`` when the variable is unset."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    value = raw.strip().lower()
    if value not in allowed:
        raise ValueError(f"{name} must be one of: {', '.join(allowed)}")
    return value


def _optional_file(name: str) -> Path | None:
    raw = os.getenv(name)
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_file():
        raise ValueError(f"{name} does not point at a readable file: {path}")
    return path.resolve()


def _optional_path(name: str) -> Path | None:
    """Return an appendable file path; the parent must exist, the file need not."""
    raw = os.getenv(name)
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.parent.is_dir():
        raise ValueError(f"{name} parent directory does not exist: {path.parent}")
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink: {path}")
    if path.exists() and not path.is_file():
        raise ValueError(f"{name} must be a regular file: {path}")
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


def _optional_non_negative_int(name: str) -> int:
    """Return the configured count, or ``-1`` when the variable is unset."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return -1
    return _non_negative_int(name, default=-1)
