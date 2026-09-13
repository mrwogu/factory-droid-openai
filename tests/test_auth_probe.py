from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from factory_droid_openai import logs
from factory_droid_openai.auth_probe import AuthProbe
from factory_droid_openai.metrics import BridgeMetrics
from factory_droid_openai.runner import (
    ReasoningDelta,
    RunComplete,
    RunEvent,
    RunnerError,
    RunRequest,
    SessionStarted,
    StatusUpdate,
    TextDelta,
    Usage,
    UsageUpdate,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from factory_droid_openai.runner import DroidRunner


def _recorded_usage(payload: dict[str, Any]) -> Usage:
    details = payload.get("prompt_tokens_details") or {}
    return Usage(
        input_tokens=int(payload.get("prompt_tokens", 0)),
        output_tokens=int(payload.get("completion_tokens", 0)),
        cache_read_tokens=int(details.get("cached_tokens", 0)),
    )


def _recorded_event(record: dict[str, Any]) -> RunEvent:
    kind = record["kind"]
    if kind == "text_delta":
        return TextDelta(record["text"])
    if kind == "reasoning_delta":
        return ReasoningDelta(record["text"])
    if kind == "usage":
        return UsageUpdate(_recorded_usage(record["usage"]))
    if kind == "run_complete":
        return RunComplete(_recorded_usage(record["usage"]))
    if kind == "status":
        return StatusUpdate(record["state"])
    if kind == "session_started":
        output_tokens = record.get("output_tokens", 0)
        return SessionStarted(
            "replay-session",
            None if output_tokens is None else int(output_tokens),
        )
    raise AssertionError(f"unknown recorded event kind: {kind}")


def _recorded_events() -> list[RunEvent]:
    path = Path(__file__).parent / "fixtures" / "events" / "hello--gpt-5-4-mini.jsonl"
    records = [
        cast("dict[str, Any]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    return [_recorded_event(record) for record in records if record["kind"] != "meta"]


class _RecordedRunner:
    """Replays one captured Droid event stream in its original order."""

    def __init__(
        self,
        *,
        events: list[RunEvent] | None = None,
        error: RunnerError | None = None,
    ) -> None:
        self.events = _recorded_events() if events is None else events
        self.error = error
        self.requests: list[RunRequest] = []
        self.started: asyncio.Queue[None] = asyncio.Queue()

    async def run(self, request: RunRequest) -> AsyncIterator[RunEvent]:
        self.requests.append(request)
        self.started.put_nowait(None)
        if self.error is not None:
            raise self.error
        for event in self.events:
            yield event


class _ControlledRunner(_RecordedRunner):
    def __init__(self) -> None:
        super().__init__()
        self.release: asyncio.Queue[None] = asyncio.Queue()

    async def run(self, request: RunRequest) -> AsyncIterator[RunEvent]:
        self.requests.append(request)
        self.started.put_nowait(None)
        await self.release.get()
        for event in self.events:
            yield event


class _HangingRunner:
    """Never yields; cancellation must end the probe."""

    def __init__(self) -> None:
        self.requests: list[RunRequest] = []
        self.started = asyncio.Event()

    async def run(self, request: RunRequest) -> AsyncIterator[RunEvent]:
        self.requests.append(request)
        self.started.set()
        await asyncio.Event().wait()
        yield TextDelta("never")


class _CrashingRunner(_RecordedRunner):
    async def run(self, request: RunRequest) -> AsyncIterator[RunEvent]:
        self.requests.append(request)
        self.started.put_nowait(None)
        for event in self.events:
            yield event
            if isinstance(event, TextDelta):
                raise RuntimeError("kaboom")


def _probe(
    runner: Any,
    *,
    interval_seconds: float = 60.0,
    timeout_seconds: float = 30.0,
    failure_threshold: int = 3,
    metrics: BridgeMetrics | None = None,
) -> AuthProbe:
    return AuthProbe(
        runner_factory=cast("Callable[[], DroidRunner]", lambda: runner),
        model_alias="factory-droid",
        timeout_seconds=timeout_seconds,
        interval_seconds=interval_seconds,
        failure_threshold=failure_threshold,
        metrics=metrics,
    )


def _events(stream: io.StringIO) -> list[dict[str, Any]]:
    return [
        cast("dict[str, Any]", json.loads(line)) for line in stream.getvalue().splitlines() if line
    ]


@pytest.mark.asyncio
async def test_successful_probe_reports_ok() -> None:
    stream = io.StringIO()
    logs.configure_logging(level="debug", log_format="json", stream=stream)
    metrics = BridgeMetrics()
    runner = _RecordedRunner()
    probe = _probe(runner, metrics=metrics)

    await probe._probe_once()

    assert probe.status == "ok"
    assert len(runner.requests) == 1
    request = runner.requests[0]
    assert request.model == "factory-droid"
    assert request.model_alias == "factory-droid"
    assert "factory_droid_openai_auth_probe_successes_total 1" in metrics.render()
    assert "factory_droid_openai_auth_probe_failures_total 0" in metrics.render()
    assert any(entry["event"] == "auth.probe_ok" for entry in _events(stream))


@pytest.mark.asyncio
async def test_probe_timeout_is_capped_at_sixty_seconds() -> None:
    runner = _RecordedRunner()
    probe = _probe(runner, timeout_seconds=600.0)

    await probe._probe_once()

    assert probe.status == "ok"
    assert runner.requests[0].timeout_seconds == 60.0


@pytest.mark.asyncio
async def test_runner_error_counts_as_probe_failure() -> None:
    stream = io.StringIO()
    logs.configure_logging(level="warning", log_format="json", stream=stream)
    metrics = BridgeMetrics()
    runner = _RecordedRunner(error=RunnerError("Factory rejected the bridge's API key: revoked"))
    probe = _probe(runner, metrics=metrics)

    await probe._probe_once()

    assert probe.status == "degraded"
    assert probe.consecutive_failures == 1
    assert probe.failing is False
    assert "factory_droid_openai_auth_probe_failures_total 1" in metrics.render()
    failures = [entry for entry in _events(stream) if entry["event"] == "auth.probe_failed"]
    assert len(failures) == 1
    assert failures[0]["reason"] == "Factory rejected the bridge's API key: revoked"
    assert failures[0]["consecutive"] == 1


@pytest.mark.asyncio
async def test_empty_answer_counts_as_probe_failure() -> None:
    stream = io.StringIO()
    logs.configure_logging(level="warning", log_format="json", stream=stream)
    runner = _RecordedRunner(
        events=[event for event in _recorded_events() if not isinstance(event, TextDelta)]
    )
    probe = _probe(runner)

    await probe._probe_once()

    assert probe.status == "degraded"
    failures = [entry for entry in _events(stream) if entry["event"] == "auth.probe_failed"]
    assert failures[0]["reason"] == "Auth probe completed without any assistant text."


@pytest.mark.asyncio
async def test_incomplete_answer_counts_as_probe_failure() -> None:
    runner = _RecordedRunner(
        events=[event for event in _recorded_events() if not isinstance(event, RunComplete)]
    )
    probe = _probe(runner)

    await probe._probe_once()

    assert probe.status == "degraded"
    assert probe.consecutive_failures == 1


@pytest.mark.asyncio
async def test_hanging_runner_times_out() -> None:
    runner = _HangingRunner()
    probe = _probe(runner, timeout_seconds=0.0)

    await probe._probe_once()

    assert probe.status == "degraded"
    assert probe.consecutive_failures == 1


@pytest.mark.asyncio
async def test_probe_crash_is_reported_as_failure() -> None:
    runner = _CrashingRunner()
    probe = _probe(runner)

    await probe._probe_once()

    assert probe.status == "degraded"
    assert probe.consecutive_failures == 1


@pytest.mark.asyncio
async def test_gate_opens_at_the_failure_threshold() -> None:
    runner = _RecordedRunner(error=RunnerError("unauthorized"))
    probe = _probe(runner, failure_threshold=2)

    await probe._probe_once()
    await probe._probe_once()

    assert probe.status == "degraded"
    assert probe.consecutive_failures == 2
    assert probe.failing is True

    probe.record_success()
    assert probe.failing is False
    assert probe.status == "ok"


@pytest.mark.asyncio
async def test_suspect_triggers_an_immediate_probe() -> None:
    runner = _ControlledRunner()
    probe = _probe(runner)

    probe.start()
    await runner.started.get()
    probe.suspect()
    runner.release.put_nowait(None)
    await runner.started.get()

    await probe.aclose()
    assert len(runner.requests) == 2


@pytest.mark.asyncio
async def test_periodic_interval_reprobes_without_suspect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _RecordedRunner()
    probe = _probe(runner)

    async def expire_wait(awaitable: Awaitable[bool], *, timeout: float) -> bool:
        del timeout
        cast("Any", awaitable).close()
        await asyncio.sleep(0)
        raise TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", expire_wait)
    probe.start()
    await runner.started.get()
    await runner.started.get()

    await probe.aclose()
    assert len(runner.requests) >= 2


@pytest.mark.asyncio
async def test_zero_interval_disables_the_loop() -> None:
    runner = _RecordedRunner()
    probe = _probe(runner, interval_seconds=0.0)

    probe.start()

    assert probe.status == "unknown"
    assert runner.requests == []
    assert probe._task is None
    await probe.aclose()


@pytest.mark.asyncio
async def test_start_is_idempotent() -> None:
    runner = _HangingRunner()
    probe = _probe(runner)

    probe.start()
    await runner.started.wait()
    task = probe._task
    assert task is not None
    probe.start()
    assert probe._task is task

    await probe.aclose()
    assert probe._task is None


@pytest.mark.asyncio
async def test_aclose_before_start_is_a_noop() -> None:
    runner = _RecordedRunner()
    probe = _probe(runner)

    await probe.aclose()

    assert probe.status == "unknown"


def test_new_counters_render_at_zero() -> None:
    text = BridgeMetrics().render()

    assert "factory_droid_openai_empty_completions_total 0" in text
    assert "factory_droid_openai_auth_probe_successes_total 0" in text
    assert "factory_droid_openai_auth_probe_failures_total 0" in text
