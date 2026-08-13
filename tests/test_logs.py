from __future__ import annotations

import io
import json
import logging
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest

from factory_droid_openai import logs
from factory_droid_openai.app import create_app
from factory_droid_openai.config import Settings
from factory_droid_openai.runner import (
    DroidModel,
    RunComplete,
    RunnerError,
    TextDelta,
    Usage,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from factory_droid_openai.app import RunnerFactory
    from factory_droid_openai.runner import RunEvent, RunRequest


class FakeRunner:
    def __init__(self, events: list[RunEvent], *, error: RunnerError | None = None) -> None:
        self.events = events
        self.error = error

    async def run(self, request: RunRequest) -> AsyncIterator[RunEvent]:
        del request
        if self.error is not None:
            raise self.error
        for event in self.events:
            yield event

    async def list_models(self, *, timeout_seconds: float) -> tuple[DroidModel, ...]:
        del timeout_seconds
        return (
            DroidModel(
                id="gpt-5.4",
                display_name="GPT-5.4",
                provider="openai",
                supported_reasoning_efforts=("low", "high"),
                default_reasoning_effort="high",
                supports_images=True,
                supports_pdfs=True,
            ),
        )


@pytest.fixture
def log_stream() -> io.StringIO:
    stream = io.StringIO()
    logs.configure_logging(level="trace", log_format="text", stream=stream)
    return stream


def _lines(stream: io.StringIO) -> list[str]:
    return [line for line in stream.getvalue().splitlines() if line]


def _events(stream: io.StringIO) -> list[str]:
    events = []
    for line in _lines(stream):
        for field in line.split(" "):
            if field.startswith("event="):
                events.append(field.removeprefix("event="))
                break
    return events


def _fields(line: str) -> dict[str, str]:
    parts = line.split(" ")
    return dict(
        part.split("=", 1) for part in parts if "=" in part and not part.startswith("event=")
    )


def test_level_number_supports_trace_and_rejects_unknown() -> None:
    assert logs.level_number("trace") == logs.TRACE
    assert logs.level_number("Warning") == logging.WARNING

    with pytest.raises(ValueError, match="Unsupported log level"):
        logs.level_number("verbose")


def test_configure_logging_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match="Unsupported log format"):
        logs.configure_logging(log_format="logfmt")


def test_configure_logging_replaces_previous_handler() -> None:
    first = io.StringIO()
    second = io.StringIO()
    logs.configure_logging(level="info", stream=first)
    logs.configure_logging(level="info", stream=second)

    logs.info("probe")

    assert first.getvalue() == ""
    assert len(_lines(second)) == 1
    assert "event=probe" in second.getvalue()


def test_text_format_renders_fields_and_request_id(log_stream: io.StringIO) -> None:
    logs.bind_request("chatcmpl-test")
    logs.info(
        "chat.completed",
        model="gpt-5.4",
        stream=False,
        note="two words",
        dropped=None,
        rate=12.25,
        zero=0.0,
    )

    line = _lines(log_stream)[0]
    fields = _fields(line)
    assert line.startswith("INFO    ")
    assert fields["request_id"] == "chatcmpl-test"
    assert fields["model"] == "gpt-5.4"
    assert fields["stream"] == "false"
    assert fields["note"] == '"two'
    assert fields["rate"] == "12.2"
    assert fields["zero"] == "0.0"
    assert "dropped" not in fields
    assert "elapsed_ms" in fields


def test_text_format_emits_each_record_and_field_once(log_stream: io.StringIO) -> None:
    logs.bind_request("chatcmpl-once")
    logs.info("chat.completed", model="gpt-5.4", status=200)

    lines = _lines(log_stream)

    assert len(lines) == 1
    assert lines[0].count("event=chat.completed") == 1
    assert lines[0].count("request_id=chatcmpl-once") == 1
    assert lines[0].count("model=gpt-5.4") == 1
    assert lines[0].count("status=200") == 1


def test_elapsed_is_omitted_when_phase_timings_are_present(log_stream: io.StringIO) -> None:
    logs.bind_request("chatcmpl-phase")

    logs.debug("chat.admitted", queue_ms=1.5)

    fields = _fields(_lines(log_stream)[0])
    assert fields["queue_ms"] == "1.5"
    assert "elapsed_ms" not in fields


def test_text_format_falls_back_when_unbound() -> None:
    stream = io.StringIO()
    logs.configure_logging(level="debug", stream=stream)
    logs._request_id.set("-")
    logs._timeline.set(None)

    logs.debug("plain", empty="")

    fields = _fields(_lines(stream)[0])
    assert fields["request_id"] == "-"
    assert fields["empty"] == "-"
    assert "elapsed_ms" not in fields


def test_text_format_renders_none_fields_from_direct_logger_use() -> None:
    stream = io.StringIO()
    logs.configure_logging(level="info", stream=stream)

    logging.getLogger(logs.LOGGER_NAME).info(
        "raw.event",
        extra={"factory_fields": {"missing": None}},
    )

    assert _fields(_lines(stream)[0])["missing"] == "-"


def test_json_format_emits_one_object_per_line() -> None:
    stream = io.StringIO()
    logs.configure_logging(level="debug", log_format="json", stream=stream)
    logs.bind_request("chatcmpl-json")

    logs.warning("chat.rejected", status=429)

    payload = json.loads(_lines(stream)[0])
    assert payload["event"] == "chat.rejected"
    assert payload["level"] == "WARNING"
    assert payload["request_id"] == "chatcmpl-json"
    assert payload["status"] == 429


def test_json_format_emits_each_record_and_key_once() -> None:
    stream = io.StringIO()
    logs.configure_logging(level="info", log_format="json", stream=stream)
    logs.bind_request("chatcmpl-json-once")

    logs.info("chat.completed", model="gpt-5.4", status=200)

    lines = _lines(stream)
    assert len(lines) == 1
    assert lines[0].count('"event"') == 1
    assert lines[0].count('"request_id"') == 1
    assert lines[0].count('"model"') == 1
    assert lines[0].count('"status"') == 1
    payload = json.loads(lines[0])
    assert payload["event"] == "chat.completed"
    assert payload["request_id"] == "chatcmpl-json-once"
    assert payload["model"] == "gpt-5.4"
    assert payload["status"] == 200


def test_json_format_includes_exception_text() -> None:
    stream = io.StringIO()
    logs.configure_logging(level="debug", log_format="json", stream=stream)
    logger = logging.getLogger(logs.LOGGER_NAME)

    try:
        raise RuntimeError("boom")
    except RuntimeError:
        logger.exception("droid.failed")

    payload = json.loads(_lines(stream)[0])
    assert "RuntimeError: boom" in payload["error"]


def test_text_format_includes_exception_text() -> None:
    stream = io.StringIO()
    logs.configure_logging(level="debug", stream=stream)
    logger = logging.getLogger(logs.LOGGER_NAME)

    try:
        raise RuntimeError("boom")
    except RuntimeError:
        logger.exception("droid.failed")

    assert "RuntimeError: boom" in stream.getvalue()


def test_levels_below_threshold_are_dropped() -> None:
    stream = io.StringIO()
    logs.configure_logging(level="warning", stream=stream)

    logs.trace("dropped.trace")
    logs.debug("dropped.debug")
    logs.info("dropped.info")
    logs.warning("kept.warning")

    assert _events(stream) == ["kept.warning"]
    assert logs.enabled(logging.WARNING)
    assert not logs.enabled(logging.INFO)


def test_timeline_tracks_phases_and_totals() -> None:
    timeline = logs.bind_request("chatcmpl-timeline")

    assert logs.current_request_id() == "chatcmpl-timeline"
    assert logs.current_timeline() is timeline
    timeline.mark("prompt_ms")
    timeline.observe("droid_startup_ms", 0.25)
    timeline.since_start("ttft_ms")

    fields = timeline.fields()
    assert fields["droid_startup_ms"] == 250.0
    assert fields["prompt_ms"] >= 0.0
    assert fields["total_ms"] >= fields["ttft_ms"]
    assert logs.millis(-1.0) == 0.0


def _app(tmp_path: Path, runner: FakeRunner) -> Any:
    return create_app(
        Settings(droid_path="droid", workdir=tmp_path, timeout_seconds=30.0),
        runner_factory=cast("RunnerFactory", lambda: runner),
    )


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_chat_completion_logs_phase_timings(
    tmp_path: Path,
    log_stream: io.StringIO,
) -> None:
    runner = FakeRunner([TextDelta("Hi"), RunComplete(Usage(input_tokens=7, output_tokens=3))])
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "factory-droid", "messages": [{"role": "user", "content": "Hello"}]},
        )

    assert response.status_code == 200
    events = _events(log_stream)
    assert events[:5] == [
        "chat.received",
        "chat.prompt_built",
        "chat.admitted",
        "pool.miss",
        "chat.first_token",
    ]
    assert events[-1] == "chat.completed"
    summary = _fields(_lines(log_stream)[-1])
    assert summary["request_id"].startswith("chatcmpl-")
    assert summary["input_tokens"] == "7"
    assert summary["output_tokens"] == "3"
    assert summary["stream"] == "false"
    assert float(summary["ttft_ms"]) >= 0.0
    assert float(summary["queue_ms"]) >= 0.0
    assert float(summary["prompt_ms"]) >= 0.0
    assert float(summary["total_ms"]) >= 0.0


@pytest.mark.asyncio
async def test_streaming_chat_logs_outcome(tmp_path: Path, log_stream: io.StringIO) -> None:
    runner = FakeRunner([TextDelta("Hi"), RunComplete(Usage(output_tokens=2))])
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "factory-droid",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    summary = _fields(_lines(log_stream)[-1])
    assert _events(log_stream)[-1] == "chat.completed"
    assert summary["stream"] == "true"
    assert summary["outcome"] == "success"


@pytest.mark.asyncio
async def test_failed_chat_logs_warning(tmp_path: Path, log_stream: io.StringIO) -> None:
    runner = FakeRunner([], error=RunnerError("nope", status_code=504))
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "factory-droid", "messages": [{"role": "user", "content": "Hello"}]},
        )

    assert response.status_code == 504
    assert _events(log_stream)[-1] == "chat.failed"
    assert _fields(_lines(log_stream)[-1])["status"] == "504"


@pytest.mark.asyncio
async def test_rejected_chat_logs_warning(tmp_path: Path, log_stream: io.StringIO) -> None:
    runner = FakeRunner([])
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "factory-droid",
                "messages": [{"role": "user", "content": "Hello"}],
                "n": 99,
            },
        )

    assert response.status_code == 400
    assert _events(log_stream)[-1] == "chat.rejected"
    assert _fields(_lines(log_stream)[-1])["phase"] == "options"


@pytest.mark.asyncio
async def test_model_listing_logs_discovery(tmp_path: Path, log_stream: io.StringIO) -> None:
    runner = FakeRunner([])
    async with _client(_app(tmp_path, runner)) as client:
        response = await client.get("/v1/models")

    assert response.status_code == 200
    assert "models.listed" in _events(log_stream)
