from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from factory_droid_openai import payloadlog

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _reset_tracing() -> None:
    payloadlog.reset_payload_tracing()


def _read_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_tracing_off_writes_nothing(tmp_path: Path) -> None:
    payloadlog.configure_payload_tracing(mode="off", path=tmp_path / "trace.jsonl")
    payloadlog.trace_payload("chat.prompt", "hello")
    assert not (tmp_path / "trace.jsonl").exists()


def test_full_mode_writes_redacted_payload(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    payloadlog.configure_payload_tracing(mode="full", path=path)
    payloadlog.trace_payload("chat.prompt", "use key sk-abcdef123456 please")
    (record,) = _read_records(path)
    assert record["event"] == "chat.prompt"
    assert record["mode"] == "full"
    assert record["payload"] == "use key [REDACTED] please"
    assert record["payload_sha256"] == hashlib.sha256(b"use key sk-abcdef123456 please").hexdigest()
    assert record["payload_bytes"] == len(b"use key sk-abcdef123456 please")


def test_heads_mode_truncates_and_hashes(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    payloadlog.configure_payload_tracing(mode="heads", path=path)
    payload = "xy " * 2000
    payloadlog.trace_payload("chat.prompt", payload)
    (record,) = _read_records(path)
    assert "payload" not in record
    assert record["payload_head"] == payload[:2048]
    assert record["payload_tail"] == payload[-1024:]
    assert record["payload_sha256"] == hashlib.sha256(payload.encode()).hexdigest()


def test_heads_mode_short_payload_has_no_tail(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    payloadlog.configure_payload_tracing(mode="heads", path=path)
    payloadlog.trace_payload("chat.prompt", "short")
    (record,) = _read_records(path)
    assert record["payload_head"] == "short"
    assert "payload_tail" not in record


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('"api_key": "supersecretvalue"', '"api_key": "[REDACTED]"'),
        ("Authorization: Bearer abcdef1234567890", "Authorization: [REDACTED]"),
        ("token=ghp_abcdef123456", "token=[REDACTED]"),
        ("A" * 130, "[REDACTED]"),
    ],
)
def test_redact_patterns(raw: str, expected: str) -> None:
    assert payloadlog.redact(raw) == expected


def test_extra_fields_pass_through(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    payloadlog.configure_payload_tracing(mode="heads", path=path)
    payloadlog.trace_payload("tool_call.repaired", "{}", variant="json_array", skipped=None)
    (record,) = _read_records(path)
    assert record["variant"] == "json_array"
    assert "skipped" not in record


def test_configure_rejects_unknown_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported payload trace mode"):
        payloadlog.configure_payload_tracing(mode="verbose", path=tmp_path / "t.jsonl")


def test_configure_requires_path_when_enabled() -> None:
    with pytest.raises(ValueError, match="requires a destination file"):
        payloadlog.configure_payload_tracing(mode="full", path=None)


def test_configure_rejects_missing_parent(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="directory does not exist"):
        payloadlog.configure_payload_tracing(
            mode="full",
            path=tmp_path / "missing" / "trace.jsonl",
        )


def test_configure_replaces_previous_setup(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    payloadlog.configure_payload_tracing(mode="full", path=first)
    payloadlog.configure_payload_tracing(mode="off", path=None)
    payloadlog.trace_payload("chat.prompt", "hello")
    assert not first.exists()
