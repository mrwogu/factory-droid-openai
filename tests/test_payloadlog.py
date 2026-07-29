from __future__ import annotations

import hashlib
import json
import os
import stat
from typing import TYPE_CHECKING

import pytest

from factory_droid_openai import payloadlog

if TYPE_CHECKING:
    from pathlib import Path


def _read_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_tracing_off_writes_nothing(tmp_path: Path) -> None:
    tracer = payloadlog.configure_payload_tracing(mode="off", path=tmp_path / "trace.jsonl")
    tracer.trace("chat.prompt", "hello")
    assert not (tmp_path / "trace.jsonl").exists()


def test_full_mode_writes_redacted_payload(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    tracer = payloadlog.configure_payload_tracing(mode="full", path=path)
    raw = b"use key sk-abcdef123456 please"
    tracer.trace("chat.prompt", raw.decode())
    (record,) = _read_records(path)
    assert record["event"] == "chat.prompt"
    assert record["mode"] == "full"
    assert record["payload"] == "use key [REDACTED] please"
    assert record["redacted_sha256"] == hashlib.sha256(b"use key [REDACTED] please").hexdigest()
    assert record["redacted_sha256"] != hashlib.sha256(raw).hexdigest()
    assert record["payload_bytes"] == len(raw)
    assert "payload_truncated" not in record


def test_full_mode_caps_oversized_payload(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    tracer = payloadlog.configure_payload_tracing(mode="full", path=path)
    payload = "zy " * 400_000

    tracer.trace("chat.prompt", payload)

    (record,) = _read_records(path)
    assert len(payload) > 1_048_576
    assert record["payload"] == payload[:1_048_576]
    assert record["payload_truncated"] is True
    assert record["payload_bytes"] == len(payload)


def test_heads_mode_truncates_and_hashes(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    tracer = payloadlog.configure_payload_tracing(mode="heads", path=path)
    payload = "xy " * 2000
    tracer.trace("chat.prompt", payload)
    (record,) = _read_records(path)
    assert "payload" not in record
    assert record["payload_head"] == payload[:2048]
    assert record["payload_tail"] == payload[-1024:]
    assert record["redacted_sha256"] == hashlib.sha256(payload.encode()).hexdigest()


def test_heads_mode_short_payload_has_no_tail(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    tracer = payloadlog.configure_payload_tracing(mode="heads", path=path)
    tracer.trace("chat.prompt", "short")
    (record,) = _read_records(path)
    assert record["payload_head"] == "short"
    assert "payload_tail" not in record


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('"api_key": "supersecretvalue"', '"api_key": "[REDACTED]"'),
        ("Authorization: Bearer abcdef1234567890", "Authorization: [REDACTED]"),
        ("Authorization: ApiKey abcdef1234567890", "Authorization: [REDACTED]"),
        ("password: correct horse battery staple", "password: [REDACTED]"),
        ("Authorization: Basic dXNlcjpwYXNzd29yZA==", "Authorization: [REDACTED]"),
        ("token=ghp_abcdef123456", "token=[REDACTED]"),
        (
            '{"content":"{\\"password\\":\\"supersecretvalue\\"}"}',
            '{"content":"{\\"password\\":\\"[REDACTED]\\"}"}',
        ),
        ('"access_token":"supersecretvalue"', '"access_token":"[REDACTED]"'),
        ('"client_secret":"supersecretvalue"', '"client_secret":"[REDACTED]"'),
        ("A" * 130, "[REDACTED]"),
    ],
)
def test_redact_patterns(raw: str, expected: str) -> None:
    assert payloadlog.redact(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            '{"headers":"Authorization: Bearer abc","model":"gpt-5.4"}',
            '{"headers":"Authorization: [REDACTED]","model":"gpt-5.4"}',
        ),
        (
            '{"note":"password: hunter2","model":"gpt-5.4","keep":"this"}',
            '{"note":"password: [REDACTED]","model":"gpt-5.4","keep":"this"}',
        ),
        ("?api_key=abcdef&model=gpt-5.4", "?api_key=[REDACTED]&model=gpt-5.4"),
        ("token: abc; model: gpt-5.4", "token: [REDACTED]; model: gpt-5.4"),
        ("{secret: abc}", "{secret: [REDACTED]}"),
        ("[token: abc]", "[token: [REDACTED]]"),
    ],
)
def test_redact_stops_at_delimiters(raw: str, expected: str) -> None:
    assert payloadlog.redact(raw) == expected
    assert payloadlog.redact(expected) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "Authorization: Bearer abcdef1234567890",
        "Authorization: Basic QUJDREVGZ2hpamtsZa==",
        "token=abcdef1234567890",
        "password: correct horse battery staple",
        '"api_key": "supersecretvalue"',
    ],
)
def test_redact_is_idempotent(raw: str) -> None:
    once = payloadlog.redact(raw)

    assert payloadlog.redact(once) == once


def test_redact_keeps_trailing_fields_of_minified_prompt() -> None:
    payload = json.dumps(
        {
            "messages": [{"role": "user", "content": "set authorization: Bearer abc header"}],
            "model": "gpt-5.4",
            "stream": True,
        },
    )

    redacted = payloadlog.redact(payload)

    assert "abc" not in redacted
    assert json.loads(redacted)["model"] == "gpt-5.4"
    assert json.loads(redacted)["stream"] is True


def test_sequence_fields_are_redacted(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    tracer = payloadlog.configure_payload_tracing(mode="heads", path=path)

    tracer.trace(
        "tool_call.repaired",
        "{}",
        variants=["sk-abcdef123456", {"password": "metadata secret"}],
        tried=("Bearer abcdef1234567890",),
    )

    (record,) = _read_records(path)
    assert record["variants"] == ["[REDACTED]", {"password": "[REDACTED]"}]
    assert record["tried"] == ["[REDACTED]"]


def test_redact_nested_json_at_any_escape_depth() -> None:
    raw = r'{"password":"prefix\"supersecretvalue"}'

    for _ in range(5):
        assert "supersecretvalue" not in payloadlog.redact(raw)
        raw = json.dumps(raw)


def test_extra_fields_pass_through(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    tracer = payloadlog.configure_payload_tracing(mode="heads", path=path)
    tracer.trace(
        "tool_call.repaired",
        "{}",
        variant="json_array",
        attempt=3,
        recovered=True,
        skipped=None,
    )
    (record,) = _read_records(path)
    assert record["variant"] == "json_array"
    assert record["attempt"] == 3
    assert record["recovered"] is True
    assert "skipped" not in record


def test_extra_fields_are_redacted_and_ascii_safe(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    tracer = payloadlog.configure_payload_tracing(mode="full", path=path)

    tracer.trace(
        "chat.prompt",
        "hello",
        model="sk-abcdef123456",
        password="ordinary metadata secret",
        nested={
            "authorization": "Bearer abcdef1234567890",
            "client_secret": "another metadata secret",
        },
        unusual="\ud800",
    )

    (record,) = _read_records(path)
    assert record["model"] == "[REDACTED]"
    assert record["password"] == "[REDACTED]"
    assert record["nested"] == {
        "authorization": "[REDACTED]",
        "client_secret": "[REDACTED]",
    }
    assert record["unusual"] == "\ud800"


def test_unencodable_payload_does_not_escape(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    tracer = payloadlog.configure_payload_tracing(mode="full", path=path)

    tracer.trace("chat.prompt", "\ud800")

    assert not path.exists()


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


def test_tracers_keep_independent_configuration(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    enabled = payloadlog.configure_payload_tracing(mode="full", path=first)
    disabled = payloadlog.configure_payload_tracing(mode="off", path=None)
    disabled.trace("chat.prompt", "ignored")
    enabled.trace("chat.prompt", "hello")
    (record,) = _read_records(first)
    assert record["payload"] == "hello"


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_trace_file_permissions_are_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text("", encoding="utf-8")
    path.chmod(0o644)
    tracer = payloadlog.configure_payload_tracing(mode="full", path=path)

    tracer.trace("chat.prompt", "private")

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_configure_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be a regular file"):
        payloadlog.configure_payload_tracing(mode="full", path=tmp_path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO unavailable")
def test_configure_rejects_fifo(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    os.mkfifo(path)

    with pytest.raises(ValueError, match="must be a regular file"):
        payloadlog.configure_payload_tracing(mode="full", path=path)


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks")
def test_configure_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("protected", encoding="utf-8")
    link = tmp_path / "trace.jsonl"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="must not be a symlink"):
        payloadlog.configure_payload_tracing(mode="full", path=link)

    assert target.read_text(encoding="utf-8") == "protected"


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory permissions")
def test_trace_io_failure_does_not_escape(tmp_path: Path) -> None:
    destination = tmp_path / "traces"
    destination.mkdir()
    path = destination / "trace.jsonl"
    tracer = payloadlog.configure_payload_tracing(mode="full", path=path)
    destination.chmod(0o500)

    try:
        tracer.trace("chat.prompt", "private")
        assert not path.exists()
    finally:
        destination.chmod(0o700)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO unavailable")
def test_trace_rejects_destination_swapped_after_validation(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text("", encoding="utf-8")
    tracer = payloadlog.configure_payload_tracing(mode="full", path=path)
    path.unlink()
    os.mkfifo(path)
    reader = os.open(path, os.O_RDONLY | os.O_NONBLOCK)

    try:
        tracer.trace("chat.prompt", "private")
        assert os.read(reader, 4096) == b""
    finally:
        os.close(reader)
