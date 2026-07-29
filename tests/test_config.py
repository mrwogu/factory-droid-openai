from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from factory_droid_openai.config import Settings

if TYPE_CHECKING:
    from pathlib import Path

_ENVIRONMENT_KEYS = (
    "FACTORY_DROID_OPENAI_HOST",
    "FACTORY_DROID_OPENAI_PORT",
    "FACTORY_DROID_OPENAI_API_KEY",
    "FACTORY_DROID_PATH",
    "FACTORY_DROID_OPENAI_WORKDIR",
    "FACTORY_DROID_OPENAI_TIMEOUT_SECONDS",
    "FACTORY_DROID_OPENAI_BODY_TIMEOUT_SECONDS",
    "FACTORY_DROID_OPENAI_MAX_CONCURRENCY",
    "FACTORY_DROID_OPENAI_MAX_QUEUE_SIZE",
    "FACTORY_DROID_OPENAI_MAX_REQUEST_BYTES",
    "FACTORY_DROID_OPENAI_MAX_MESSAGES",
    "FACTORY_DROID_OPENAI_MAX_TOOLS",
    "FACTORY_DROID_OPENAI_MAX_TRANSCRIPT_BYTES",
    "FACTORY_DROID_OPENAI_MAX_TOOL_SCHEMA_BYTES",
    "FACTORY_DROID_OPENAI_MAX_STRUCTURED_OUTPUT_BYTES",
    "FACTORY_DROID_OPENAI_MAX_JSON_DEPTH",
    "FACTORY_DROID_OPENAI_MCP_SETTLE_SECONDS",
    "FACTORY_DROID_OPENAI_MODEL_CACHE_SECONDS",
    "FACTORY_DROID_OPENAI_MODEL_QUARANTINE_SECONDS",
    "FACTORY_DROID_OPENAI_RETRY_AFTER_SECONDS",
    "FACTORY_DROID_OPENAI_PROCESS_GRACE_SECONDS",
    "FACTORY_DROID_OPENAI_CLEANUP_TIMEOUT_SECONDS",
    "FACTORY_DROID_OPENAI_MAX_TOOL_CALLS",
    "FACTORY_DROID_OPENAI_MAX_ATTACHMENTS",
    "FACTORY_DROID_OPENAI_MAX_ATTACHMENT_BYTES",
    "FACTORY_DROID_OPENAI_MAX_CHOICES",
    "FACTORY_DROID_OPENAI_MAX_STOP_SEQUENCES",
    "FACTORY_DROID_OPENAI_SESSION_CONTINUITY",
    "FACTORY_DROID_OPENAI_MAX_TRACKED_SESSIONS",
    "FACTORY_DROID_OPENAI_WORKTREE",
    "FACTORY_DROID_OPENAI_APPEND_SYSTEM_PROMPT_FILE",
    "FACTORY_DROID_OPENAI_UVICORN_LIMIT_CONCURRENCY",
    "FACTORY_DROID_OPENAI_UVICORN_BACKLOG",
    "FACTORY_DROID_OPENAI_MODEL_ALIAS",
    "FACTORY_DROID_OPENAI_REASONING_EFFORT",
    "FACTORY_DROID_OPENAI_LOG_LEVEL",
    "FACTORY_DROID_OPENAI_LOG_FORMAT",
    "FACTORY_DROID_OPENAI_WARM_SESSIONS",
    "FACTORY_DROID_OPENAI_WARM_SESSION_TTL_SECONDS",
    "FACTORY_DROID_OPENAI_DETACHED_CLEANUP",
    "FACTORY_DROID_OPENAI_TRACE_PAYLOADS",
    "FACTORY_DROID_OPENAI_TRACE_FILE",
)


def _clear_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_settings_from_env_uses_safe_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_environment(monkeypatch)
    monkeypatch.chdir(tmp_path)

    settings = Settings.from_env()

    assert settings == Settings(workdir=tmp_path)


def test_settings_from_env_reads_log_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_environment(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FACTORY_DROID_OPENAI_LOG_LEVEL", " DEBUG ")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_LOG_FORMAT", "json")

    settings = Settings.from_env()

    assert settings.log_level == "debug"
    assert settings.log_format == "json"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("FACTORY_DROID_OPENAI_LOG_LEVEL", "verbose"),
        ("FACTORY_DROID_OPENAI_LOG_FORMAT", "logfmt"),
    ],
)
def test_settings_from_env_rejects_unknown_log_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str,
    value: str,
) -> None:
    _clear_environment(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match="must be one of"):
        Settings.from_env()


def test_settings_from_env_reads_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_environment(monkeypatch)
    monkeypatch.chdir(tmp_path)

    assert Settings.from_env().reasoning_effort is None

    monkeypatch.setenv("FACTORY_DROID_OPENAI_REASONING_EFFORT", " LOW ")
    assert Settings.from_env().reasoning_effort == "low"

    monkeypatch.setenv("FACTORY_DROID_OPENAI_REASONING_EFFORT", "extreme")
    with pytest.raises(ValueError, match="must be one of"):
        Settings.from_env()


def test_settings_normalize_reasoning_effort_on_construction() -> None:
    assert Settings(reasoning_effort=" HIGH ").reasoning_effort == "high"
    assert Settings(reasoning_effort="   ").reasoning_effort is None
    assert Settings().reasoning_effort is None

    with pytest.raises(ValueError, match="reasoning_effort must be one of"):
        Settings(reasoning_effort="extreme")


def test_warm_session_count_tracks_max_concurrency_unless_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_environment(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FACTORY_DROID_OPENAI_MAX_CONCURRENCY", "3")

    tracking = Settings.from_env()
    assert tracking.warm_sessions == -1
    assert tracking.warm_session_count() == 4

    monkeypatch.setenv("FACTORY_DROID_OPENAI_WARM_SESSIONS", " ")
    assert Settings.from_env().warm_session_count() == 4

    monkeypatch.setenv("FACTORY_DROID_OPENAI_WARM_SESSIONS", "0")
    disabled = Settings.from_env()
    assert disabled.warm_sessions == 0
    assert disabled.warm_session_count() == 0

    monkeypatch.setenv("FACTORY_DROID_OPENAI_WARM_SESSIONS", "5")
    assert Settings.from_env().warm_session_count() == 5


def test_settings_from_env_reads_pool_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_environment(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FACTORY_DROID_OPENAI_WARM_SESSION_TTL_SECONDS", "90")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_DETACHED_CLEANUP", "off")

    settings = Settings.from_env()

    assert settings.warm_session_ttl_seconds == 90.0
    assert settings.detached_cleanup is False


def test_settings_from_env_rejects_negative_warm_sessions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_environment(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FACTORY_DROID_OPENAI_WARM_SESSIONS", "-2")

    with pytest.raises(ValueError, match="zero or greater"):
        Settings.from_env()


def test_settings_from_env_reads_all_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_environment(monkeypatch)
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    monkeypatch.setenv("FACTORY_DROID_OPENAI_HOST", "0.0.0.0")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_PORT", "9000")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("FACTORY_DROID_PATH", "/usr/local/bin/droid")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_WORKDIR", str(workdir))
    monkeypatch.setenv("FACTORY_DROID_OPENAI_TIMEOUT_SECONDS", "45.5")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_BODY_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_MAX_CONCURRENCY", "3")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_MAX_QUEUE_SIZE", "12")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_MAX_REQUEST_BYTES", "1024")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_MAX_MESSAGES", "64")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_MAX_TOOLS", "16")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_MAX_TRANSCRIPT_BYTES", "2048")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_MAX_TOOL_SCHEMA_BYTES", "512")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_MAX_STRUCTURED_OUTPUT_BYTES", "4096")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_MAX_JSON_DEPTH", "8")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_MCP_SETTLE_SECONDS", "1.5")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_MODEL_CACHE_SECONDS", "0")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_MODEL_QUARANTINE_SECONDS", "120")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_RETRY_AFTER_SECONDS", "3")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_PROCESS_GRACE_SECONDS", "2.5")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_CLEANUP_TIMEOUT_SECONDS", "6.5")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_UVICORN_LIMIT_CONCURRENCY", "24")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_UVICORN_BACKLOG", "96")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_MODEL_ALIAS", "droid-default")

    settings = Settings.from_env()

    assert settings == Settings(
        host="0.0.0.0",
        port=9000,
        api_key="test-key",
        droid_path="/usr/local/bin/droid",
        workdir=workdir,
        timeout_seconds=45.5,
        body_timeout_seconds=12.5,
        max_concurrency=3,
        max_queue_size=12,
        max_request_bytes=1024,
        max_messages=64,
        max_tools=16,
        max_transcript_bytes=2048,
        max_tool_schema_bytes=512,
        max_structured_output_bytes=4096,
        max_json_depth=8,
        mcp_settle_seconds=1.5,
        model_cache_seconds=0.0,
        model_quarantine_seconds=120.0,
        retry_after_seconds=3,
        process_grace_seconds=2.5,
        cleanup_timeout_seconds=6.5,
        server_limit_concurrency=24,
        server_backlog=96,
        model_alias="droid-default",
    )


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("FACTORY_DROID_OPENAI_TIMEOUT_SECONDS", "invalid", "must be a number"),
        ("FACTORY_DROID_OPENAI_TIMEOUT_SECONDS", "0", "must be greater than zero"),
        ("FACTORY_DROID_OPENAI_TIMEOUT_SECONDS", "inf", "must be greater than zero"),
        ("FACTORY_DROID_OPENAI_BODY_TIMEOUT_SECONDS", "nan", "must be greater than zero"),
        ("FACTORY_DROID_OPENAI_MAX_CONCURRENCY", "invalid", "must be an integer"),
        ("FACTORY_DROID_OPENAI_MAX_CONCURRENCY", "0", "must be greater than zero"),
        ("FACTORY_DROID_OPENAI_MAX_QUEUE_SIZE", "invalid", "must be an integer"),
        ("FACTORY_DROID_OPENAI_MAX_QUEUE_SIZE", "-1", "must be zero or greater"),
        ("FACTORY_DROID_OPENAI_MAX_REQUEST_BYTES", "0", "must be greater than zero"),
        ("FACTORY_DROID_OPENAI_MAX_MESSAGES", "0", "must be greater than zero"),
        ("FACTORY_DROID_OPENAI_MAX_TOOLS", "0", "must be greater than zero"),
        (
            "FACTORY_DROID_OPENAI_MAX_TRANSCRIPT_BYTES",
            "0",
            "must be greater than zero",
        ),
        (
            "FACTORY_DROID_OPENAI_MAX_TOOL_SCHEMA_BYTES",
            "0",
            "must be greater than zero",
        ),
        (
            "FACTORY_DROID_OPENAI_MAX_STRUCTURED_OUTPUT_BYTES",
            "0",
            "must be greater than zero",
        ),
        ("FACTORY_DROID_OPENAI_MAX_JSON_DEPTH", "0", "must be greater than zero"),
        ("FACTORY_DROID_OPENAI_MCP_SETTLE_SECONDS", "invalid", "must be a number"),
        ("FACTORY_DROID_OPENAI_MCP_SETTLE_SECONDS", "-1", "must be zero or greater"),
        ("FACTORY_DROID_OPENAI_MODEL_CACHE_SECONDS", "inf", "must be zero or greater"),
        ("FACTORY_DROID_OPENAI_MODEL_QUARANTINE_SECONDS", "-1", "must be zero or greater"),
        (
            "FACTORY_DROID_OPENAI_RETRY_AFTER_SECONDS",
            "0",
            "must be greater than zero",
        ),
        (
            "FACTORY_DROID_OPENAI_PROCESS_GRACE_SECONDS",
            "0",
            "must be greater than zero",
        ),
        (
            "FACTORY_DROID_OPENAI_CLEANUP_TIMEOUT_SECONDS",
            "0",
            "must be greater than zero",
        ),
        (
            "FACTORY_DROID_OPENAI_UVICORN_LIMIT_CONCURRENCY",
            "0",
            "must be greater than zero",
        ),
        (
            "FACTORY_DROID_OPENAI_UVICORN_LIMIT_CONCURRENCY",
            "1",
            "must be at least 2",
        ),
        (
            "FACTORY_DROID_OPENAI_UVICORN_BACKLOG",
            "0",
            "must be greater than zero",
        ),
        ("FACTORY_DROID_OPENAI_PORT", "70000", "must be at most 65535"),
    ],
)
def test_settings_from_env_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str,
    value: str,
    message: str,
) -> None:
    _clear_environment(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        Settings.from_env()


def test_settings_from_env_accepts_zero_queue_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_environment(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FACTORY_DROID_OPENAI_MAX_QUEUE_SIZE", "0")

    settings = Settings.from_env()

    assert settings.max_queue_size == 0


@pytest.mark.parametrize(
    ("process_grace_seconds", "cleanup_timeout_seconds"),
    [("1", "1"), ("2", "1")],
)
def test_settings_from_env_rejects_cleanup_timeout_not_greater_than_grace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    process_grace_seconds: str,
    cleanup_timeout_seconds: str,
) -> None:
    _clear_environment(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "FACTORY_DROID_OPENAI_PROCESS_GRACE_SECONDS",
        process_grace_seconds,
    )
    monkeypatch.setenv(
        "FACTORY_DROID_OPENAI_CLEANUP_TIMEOUT_SECONDS",
        cleanup_timeout_seconds,
    )

    with pytest.raises(
        ValueError,
        match="CLEANUP_TIMEOUT_SECONDS must be greater than "
        "FACTORY_DROID_OPENAI_PROCESS_GRACE_SECONDS",
    ):
        Settings.from_env()


def test_settings_from_env_rejects_missing_workdir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_environment(monkeypatch)
    missing = tmp_path / "missing"
    monkeypatch.setenv("FACTORY_DROID_OPENAI_WORKDIR", str(missing))

    with pytest.raises(ValueError, match="workdir does not exist"):
        Settings.from_env()


def test_settings_read_feature_limits_from_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prompt_file = tmp_path / "system.md"
    prompt_file.write_text("extra instructions", encoding="utf-8")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_WORKDIR", str(tmp_path))
    monkeypatch.setenv("FACTORY_DROID_OPENAI_MAX_TOOL_CALLS", "3")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_MAX_ATTACHMENTS", "0")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_MAX_ATTACHMENT_BYTES", "1024")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_MAX_CHOICES", "2")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_MAX_STOP_SEQUENCES", "0")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_SESSION_CONTINUITY", "yes")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_MAX_TRACKED_SESSIONS", "5")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_WORKTREE", "feature-branch")
    monkeypatch.setenv(
        "FACTORY_DROID_OPENAI_APPEND_SYSTEM_PROMPT_FILE",
        str(prompt_file),
    )

    settings = Settings.from_env()

    assert settings.max_tool_calls == 3
    assert settings.max_attachments == 0
    assert settings.max_attachment_bytes == 1024
    assert settings.max_choices == 2
    assert settings.max_stop_sequences == 0
    assert settings.session_continuity is True
    assert settings.max_tracked_sessions == 5
    assert settings.worktree == "feature-branch"
    assert settings.append_system_prompt_file == prompt_file.resolve()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1", True), ("true", True), ("on", True), ("0", False), ("off", False)],
)
def test_settings_parse_boolean_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("FACTORY_DROID_OPENAI_WORKDIR", str(tmp_path))
    monkeypatch.setenv("FACTORY_DROID_OPENAI_SESSION_CONTINUITY", raw)

    assert Settings.from_env().session_continuity is expected


def test_settings_reject_non_boolean_continuity_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FACTORY_DROID_OPENAI_WORKDIR", str(tmp_path))
    monkeypatch.setenv("FACTORY_DROID_OPENAI_SESSION_CONTINUITY", "maybe")

    with pytest.raises(ValueError, match="must be a boolean value"):
        Settings.from_env()


def test_settings_reject_missing_system_prompt_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FACTORY_DROID_OPENAI_WORKDIR", str(tmp_path))
    monkeypatch.setenv(
        "FACTORY_DROID_OPENAI_APPEND_SYSTEM_PROMPT_FILE",
        str(tmp_path / "missing.md"),
    )

    with pytest.raises(ValueError, match="does not point at a readable file"):
        Settings.from_env()


def test_settings_ignore_blank_optional_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FACTORY_DROID_OPENAI_WORKDIR", str(tmp_path))
    monkeypatch.setenv("FACTORY_DROID_OPENAI_WORKTREE", "")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_APPEND_SYSTEM_PROMPT_FILE", "")

    settings = Settings.from_env()

    assert settings.worktree is None
    assert settings.append_system_prompt_file is None


def test_settings_trace_payloads_defaults_to_off(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_environment(monkeypatch)
    monkeypatch.setenv("FACTORY_DROID_OPENAI_WORKDIR", str(tmp_path))

    settings = Settings.from_env()

    assert settings.trace_payloads == "off"
    assert settings.trace_payload_file is None


def test_settings_trace_payloads_defaults_file_to_workdir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_environment(monkeypatch)
    monkeypatch.setenv("FACTORY_DROID_OPENAI_WORKDIR", str(tmp_path))
    monkeypatch.setenv("FACTORY_DROID_OPENAI_TRACE_PAYLOADS", " FULL ")

    settings = Settings.from_env()

    assert settings.trace_payloads == "full"
    assert settings.trace_payload_file == (tmp_path / "trace.jsonl").resolve()


def test_settings_trace_payloads_reads_explicit_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_environment(monkeypatch)
    monkeypatch.setenv("FACTORY_DROID_OPENAI_WORKDIR", str(tmp_path))
    monkeypatch.setenv("FACTORY_DROID_OPENAI_TRACE_PAYLOADS", "heads")
    target = tmp_path / "custom.jsonl"
    monkeypatch.setenv("FACTORY_DROID_OPENAI_TRACE_FILE", str(target))

    settings = Settings.from_env()

    assert settings.trace_payloads == "heads"
    assert settings.trace_payload_file == target.resolve()


def test_settings_trace_payloads_rejects_unknown_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_environment(monkeypatch)
    monkeypatch.setenv("FACTORY_DROID_OPENAI_WORKDIR", str(tmp_path))
    monkeypatch.setenv("FACTORY_DROID_OPENAI_TRACE_PAYLOADS", "everything")

    with pytest.raises(ValueError, match="must be one of"):
        Settings.from_env()


def test_settings_trace_file_rejects_missing_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_environment(monkeypatch)
    monkeypatch.setenv("FACTORY_DROID_OPENAI_WORKDIR", str(tmp_path))
    monkeypatch.setenv("FACTORY_DROID_OPENAI_TRACE_FILE", str(tmp_path / "missing" / "t.jsonl"))

    with pytest.raises(ValueError, match="parent directory does not exist"):
        Settings.from_env()
