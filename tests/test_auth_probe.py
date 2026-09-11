from __future__ import annotations

import asyncio
import io
import json
import time
from typing import TYPE_CHECKING, Any, cast

import pytest

from factory_droid_openai import logs
from factory_droid_openai.auth_probe import AuthProbe
from factory_droid_openai.metrics import BridgeMetrics
from factory_droid_openai.runner import RunnerError, RunRequest, TextDelta

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from factory_droid_openai.runner import DroidRunner


class _StubRunner:
    """DroidRunner double whose run() only needs to work for one turn."""

    def __init__(
        self,
        *,
        events: list[Any] | None = None,
        error: RunnerError | None = None,
    ) -> None:
        self.events = [TextDelta("ok")] if events is None else events
        self.error = error
        self.requests: list[RunRequest] = []

    async def run(self, request: RunRequest) -> AsyncIterator[Any]:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        for event in self.events:
            yield event


class _HangingRunner:
    """Never yields; only an outer timeout can end the probe."""

    def __init__(self) -> None:
        self.requests: list[RunRequest] = []

    async def run(self, request: RunRequest) -> AsyncIterator[Any]:
        self.requests.append(request)
        await asyncio.Event().wait()
        yield TextDelta("never")


class _CrashingRunner(_StubRunner):
    async def run(self, request: RunRequest) -> AsyncIterator[Any]:
        self.requests.append(request)
        yield TextDelta("half")
        raise RuntimeError("kaboom")


async def _wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    pytest.fail("condition not reached in time")


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
    runner = _StubRunner()
    probe = _probe(runner, metrics=metrics)

    probe.start()
    await _wait_until(lambda: probe.status == "ok")

    await probe.aclose()
    assert len(runner.requests) == 1
    request = runner.requests[0]
    assert request.model == "factory-droid"
    assert request.model_alias == "factory-droid"
    assert "factory_droid_openai_auth_probe_successes_total 1" in metrics.render()
    assert "factory_droid_openai_auth_probe_failures_total 0" in metrics.render()
    assert any(entry["event"] == "auth.probe_ok" for entry in _events(stream))


@pytest.mark.asyncio
async def test_probe_timeout_is_capped_at_sixty_seconds() -> None:
    runner = _StubRunner()
    probe = _probe(runner, timeout_seconds=600.0)

    probe.start()
    await _wait_until(lambda: probe.status == "ok")

    await probe.aclose()
    assert runner.requests[0].timeout_seconds == 60.0


@pytest.mark.asyncio
async def test_runner_error_counts_as_probe_failure() -> None:
    stream = io.StringIO()
    logs.configure_logging(level="warning", log_format="json", stream=stream)
    metrics = BridgeMetrics()
    runner = _StubRunner(error=RunnerError("Factory rejected the bridge's API key: revoked"))
    probe = _probe(runner, metrics=metrics)

    probe.start()
    await _wait_until(lambda: probe.status == "degraded")

    await probe.aclose()
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
    runner = _StubRunner(events=[])
    probe = _probe(runner)

    probe.start()
    await _wait_until(lambda: probe.status == "degraded")

    await probe.aclose()
    failures = [entry for entry in _events(stream) if entry["event"] == "auth.probe_failed"]
    assert failures[0]["reason"] == "Auth probe completed without any assistant text."


@pytest.mark.asyncio
async def test_hanging_runner_times_out() -> None:
    runner = _HangingRunner()
    probe = _probe(runner, timeout_seconds=0.05)

    probe.start()
    await _wait_until(lambda: probe.status == "degraded")

    await probe.aclose()
    assert probe.consecutive_failures == 1


@pytest.mark.asyncio
async def test_probe_crash_is_reported_as_failure() -> None:
    runner = _CrashingRunner()
    probe = _probe(runner)

    probe.start()
    await _wait_until(lambda: probe.status == "degraded")

    await probe.aclose()
    assert probe.consecutive_failures == 1


@pytest.mark.asyncio
async def test_gate_opens_at_the_failure_threshold() -> None:
    runner = _StubRunner(error=RunnerError("unauthorized"))
    probe = _probe(runner, failure_threshold=2)

    probe.start()
    await _wait_until(lambda: len(runner.requests) == 1)
    probe._consecutive_failures = 1
    probe.suspect()
    await _wait_until(lambda: len(runner.requests) == 2)

    assert probe.status == "degraded"
    assert probe.consecutive_failures == 2
    assert probe.failing is True

    probe.record_success()
    assert probe.failing is False
    assert probe.status == "ok"

    await probe.aclose()


@pytest.mark.asyncio
async def test_suspect_triggers_an_immediate_probe() -> None:
    runner = _StubRunner(error=RunnerError("unauthorized"))
    probe = _probe(runner, interval_seconds=60.0)

    probe.start()
    await _wait_until(lambda: len(runner.requests) == 1)

    runner.error = None
    probe.suspect()
    await _wait_until(lambda: len(runner.requests) == 2)
    await _wait_until(lambda: probe.status == "ok")

    await probe.aclose()
    assert probe.consecutive_failures == 0


@pytest.mark.asyncio
async def test_periodic_interval_reprobes_without_suspect() -> None:
    runner = _StubRunner()
    probe = _probe(runner, interval_seconds=0.05)

    probe.start()
    await _wait_until(lambda: len(runner.requests) >= 2)

    await probe.aclose()


@pytest.mark.asyncio
async def test_zero_interval_disables_the_loop() -> None:
    runner = _StubRunner()
    probe = _probe(runner, interval_seconds=0.0)

    probe.start()
    await asyncio.sleep(0.05)

    assert probe.status == "unknown"
    assert runner.requests == []
    await probe.aclose()


@pytest.mark.asyncio
async def test_start_is_idempotent() -> None:
    runner = _StubRunner()
    probe = _probe(runner)

    probe.start()
    task = probe._task
    assert task is not None
    probe.start()
    assert probe._task is task

    await probe.aclose()
    assert probe._task is None


@pytest.mark.asyncio
async def test_aclose_before_start_is_a_noop() -> None:
    runner = _StubRunner()
    probe = _probe(runner)

    await probe.aclose()

    assert probe.status == "unknown"


def test_new_counters_render_at_zero() -> None:
    text = BridgeMetrics().render()

    assert "factory_droid_openai_empty_completions_total 0" in text
    assert "factory_droid_openai_auth_probe_successes_total 0" in text
    assert "factory_droid_openai_auth_probe_failures_total 0" in text
