from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from typing import Any, cast

import pytest

from factory_droid_openai import telemetry as telemetry_module
from factory_droid_openai.metrics import BridgeMetrics, MetricsSnapshot, RequestMetric
from factory_droid_openai.telemetry import (
    DEFAULT_TELEMETRY_TIMEOUT_SECONDS,
    TelemetryReporter,
    _event_batches,
    _events_since,
    _NoRedirectHandler,
    _post,
    _runtime_metadata,
)


def test_metrics_snapshot_contains_only_telemetry_dimensions() -> None:
    metrics = BridgeMetrics()
    metrics.record_request(
        "success",
        200,
        1.25,
        route="chat_completions",
        mode="stream",
        features=(
            "model_family:gpt",
            "request_latency:chat_completions:1s_5s",
        ),
    )
    metrics.record_request(
        "error",
        500,
        -1,
        route="models",
        mode="not_applicable",
    )
    for _ in range(2):
        metrics.record_request(
            "success",
            200,
            0.0004,
            route="health",
            mode="not_applicable",
        )
    metrics.record_features(("tools", "attachments", "tools"))
    metrics.increment_forced_kills()
    metrics.increment_model_discovery_failures()
    metrics.increment_model_quarantines()
    metrics.increment_warm_hits()
    metrics.increment_warm_misses()
    metrics.increment_warm_retunes("effort")
    metrics.increment_warm_failures()

    snapshot = metrics.telemetry_snapshot()

    assert snapshot.requests == (
        RequestMetric("chat_completions", "success", "stream", 1, 1250),
        RequestMetric("health", "success", "not_applicable", 2, 1),
        RequestMetric("models", "error", "not_applicable", 1, 0),
    )
    assert snapshot.features == (
        ("attachments", 1),
        ("model_family:gpt", 1),
        ("request_latency:chat_completions:1s_5s", 1),
        ("tools", 2),
    )
    assert snapshot.internal == (
        ("forced_kill", 1),
        ("model_discovery_failure", 1),
        ("model_quarantine", 1),
        ("warm_failure", 1),
        ("warm_hit", 1),
        ("warm_miss", 1),
        ("warm_retune", 1),
    )


@pytest.mark.asyncio
async def test_reporter_sends_startup_and_metric_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("factory_droid_openai.telemetry.platform.system", lambda: "Darwin")
    monkeypatch.setattr("factory_droid_openai.telemetry.platform.machine", lambda: "arm64")
    metrics = BridgeMetrics()
    sent: list[tuple[str, dict[str, object], float]] = []

    def post(endpoint: str, body: bytes, timeout: float) -> bool:
        sent.append((endpoint, json.loads(body), timeout))
        return True

    reporter = TelemetryReporter(
        metrics=metrics,
        app_version="1.5.0",
        endpoint="https://telemetry.example/v1/events",
        enabled=True,
        post=post,
    )

    assert reporter.enabled is True
    assert await reporter.flush() is True
    metrics.record_request(
        "success",
        200,
        0.25,
        route="chat_completions",
        mode="non_stream",
    )
    metrics.record_features(("tools",))
    metrics.increment_warm_hits()
    assert await reporter.flush() is True
    assert await reporter.flush() is True

    assert len(sent) == 2
    endpoint, startup, timeout = sent[0]
    assert endpoint == "https://telemetry.example/v1/events"
    assert 0 < timeout <= DEFAULT_TELEMETRY_TIMEOUT_SECONDS
    assert startup == {
        "schema": 1,
        "app": "factory_droid_openai",
        "event_schema": 1,
        "app_version": "1.5.0",
        "runtime": "python",
        "runtime_version": _runtime_metadata("unused")["runtime_version"],
        "os": "darwin",
        "arch": "arm64",
        "events": [{"name": "bridge_started", "count": 1}],
    }
    assert sent[1][1]["events"] == [
        {
            "name": "request",
            "route": "chat_completions",
            "outcome": "success",
            "mode": "non_stream",
            "count": 1,
            "duration_ms_sum": 250,
        },
        {"name": "feature", "feature": "tools", "count": 1},
        {"name": "internal", "metric": "warm_hit", "count": 1},
    ]


@pytest.mark.asyncio
async def test_reporter_retries_failed_batch_without_advancing_baseline() -> None:
    attempts: list[dict[str, object]] = []

    def post(_endpoint: str, body: bytes, _timeout: float) -> bool:
        attempts.append(json.loads(body))
        return len(attempts) > 1

    reporter = TelemetryReporter(
        metrics=BridgeMetrics(),
        app_version="1.5.0",
        endpoint="https://telemetry.example/v1/events",
        enabled=True,
        post=post,
    )

    assert await reporter.flush() is False
    assert await reporter.flush() is True

    assert [attempt["events"] for attempt in attempts] == [
        [{"name": "bridge_started", "count": 1}],
        [{"name": "bridge_started", "count": 1}],
    ]


@pytest.mark.asyncio
async def test_reporter_swallows_sender_exceptions() -> None:
    def post(_endpoint: str, _body: bytes, _timeout: float) -> bool:
        raise RuntimeError("sender failed")

    reporter = TelemetryReporter(
        metrics=BridgeMetrics(),
        app_version="1.5.0",
        endpoint="https://telemetry.example/v1/events",
        enabled=True,
        post=post,
    )

    assert await reporter.flush() is False


@pytest.mark.asyncio
async def test_reporter_bounds_sender_deadline() -> None:
    release = threading.Event()
    bodies: list[dict[str, object]] = []

    def post(_endpoint: str, body: bytes, _timeout: float) -> bool:
        bodies.append(json.loads(body))
        release.wait()
        return True

    reporter = TelemetryReporter(
        metrics=BridgeMetrics(),
        app_version="1.5.0",
        endpoint="https://telemetry.example/v1/events",
        enabled=True,
        timeout_seconds=0.01,
        post=post,
    )
    started_at = asyncio.get_running_loop().time()

    assert await reporter.flush() is False
    assert asyncio.get_running_loop().time() - started_at < 0.1
    release.set()
    await asyncio.sleep(0)
    # A send that outran the deadline may or may not have landed, so aggregate
    # counters move on while the startup event is retried until it is confirmed.
    assert await reporter.flush() is True
    assert [body["events"] for body in bodies] == [
        [{"name": "bridge_started", "count": 1}],
        [{"name": "bridge_started", "count": 1}],
    ]
    assert await reporter.flush() is True
    assert len(bodies) == 2


@pytest.mark.asyncio
async def test_reporter_applies_one_deadline_to_all_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = BridgeMetrics()
    monkeypatch.setattr(
        metrics,
        "telemetry_snapshot",
        lambda: MetricsSnapshot(
            requests=(),
            features=(("tools", 2_000_001),),
            internal=(),
        ),
    )
    timeouts: list[float] = []

    def post(_endpoint: str, _body: bytes, timeout: float) -> bool:
        timeouts.append(timeout)
        time.sleep(0.03)
        return True

    reporter = TelemetryReporter(
        metrics=metrics,
        app_version="1.5.0",
        endpoint="https://telemetry.example/v1/events",
        enabled=True,
        timeout_seconds=0.05,
        post=post,
    )
    started_at = asyncio.get_running_loop().time()

    assert await reporter.flush() is False
    assert asyncio.get_running_loop().time() - started_at < 0.1
    assert 1 <= len(timeouts) <= 2
    assert all(timeout > 0 for timeout in timeouts)
    assert all(timeouts[index] < timeouts[index - 1] for index in range(1, len(timeouts)))


@pytest.mark.asyncio
async def test_reporter_swallows_thread_start_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_start(_thread: threading.Thread) -> None:
        raise RuntimeError("thread start failed")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    reporter = TelemetryReporter(
        metrics=BridgeMetrics(),
        app_version="1.5.0",
        endpoint="https://telemetry.example/v1/events",
        enabled=True,
    )

    assert await reporter.flush() is False


@pytest.mark.asyncio
async def test_reporter_skips_send_after_expired_deadline() -> None:
    calls = 0

    def post(_endpoint: str, _body: bytes, _timeout: float) -> bool:
        nonlocal calls
        calls += 1
        return True

    reporter = TelemetryReporter(
        metrics=BridgeMetrics(),
        app_version="1.5.0",
        endpoint="https://telemetry.example/v1/events",
        enabled=True,
        timeout_seconds=0,
        post=post,
    )

    assert await reporter.flush() is False
    assert calls == 0


@pytest.mark.asyncio
async def test_reporter_chunks_batches_and_keeps_failed_chunk_pending() -> None:
    metrics = BridgeMetrics()
    routes = (
        "health",
        "version",
        "metrics",
        "models",
        "chat_completions",
        "session_operation",
        "other",
    )
    outcomes = ("success", "error", "timeout", "cancelled")
    for index in range(26):
        metrics.record_request(
            outcomes[index % len(outcomes)],
            200,
            0.001,
            route=routes[index % len(routes)],
            mode=f"mode-{index}",
        )

    payloads: list[dict[str, object]] = []

    def post(_endpoint: str, body: bytes, _timeout: float) -> bool:
        payloads.append(json.loads(body))
        return len(payloads) != 2

    reporter = TelemetryReporter(
        metrics=metrics,
        app_version="1.5.0",
        endpoint="https://telemetry.example/v1/events",
        enabled=True,
        post=post,
    )

    assert await reporter.flush() is False
    assert [len(cast("list[object]", payload["events"])) for payload in payloads] == [25, 2]

    payloads.clear()
    assert await reporter.flush() is True
    assert [len(cast("list[object]", payload["events"])) for payload in payloads] == [2]
    assert all(
        cast("dict[str, object]", event)["name"] != "bridge_started"
        for event in cast("list[object]", payloads[0]["events"])
    )

    payloads.clear()
    assert await reporter.flush() is True
    assert payloads == []


@pytest.mark.asyncio
async def test_reporter_splits_event_counts_above_collector_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = BridgeMetrics()
    monkeypatch.setattr(
        metrics,
        "telemetry_snapshot",
        lambda: MetricsSnapshot(
            requests=(),
            features=(("tools", 2_000_001),),
            internal=(),
        ),
    )
    payloads: list[dict[str, object]] = []

    def post(_endpoint: str, body: bytes, _timeout: float) -> bool:
        payloads.append(json.loads(body))
        return True

    reporter = TelemetryReporter(
        metrics=metrics,
        app_version="1.5.0",
        endpoint="https://telemetry.example/v1/events",
        enabled=True,
        post=post,
    )

    assert await reporter.flush() is True
    feature_counts = [
        cast("int", event["count"])
        for payload in payloads
        for event in cast("list[dict[str, object]]", payload["events"])
        if event["name"] == "feature"
    ]
    assert feature_counts == [1_000_000, 1_000_000, 1]
    assert await reporter.flush() is True
    assert len(payloads) == 3


def test_event_batches_split_request_duration_proportionally() -> None:
    batches = _event_batches(
        [
            {
                "name": "request",
                "route": "health",
                "outcome": "success",
                "mode": "not_applicable",
                "count": 2_000_001,
                "duration_ms_sum": 5,
            }
        ]
    )

    assert [(batch[0]["count"], batch[0]["duration_ms_sum"]) for batch in batches] == [
        (1_000_000, 2),
        (1_000_000, 2),
        (1, 1),
    ]

    duration_batches = _event_batches(
        [
            {
                "name": "request",
                "route": "health",
                "outcome": "success",
                "mode": "not_applicable",
                "count": 3,
                "duration_ms_sum": 3_000_000_000_001,
            }
        ]
    )
    assert [(batch[0]["count"], batch[0]["duration_ms_sum"]) for batch in duration_batches] == [
        (1, 1_000_000_000_000),
        (1, 1_000_000_000_000),
        (1, 1_000_000_000_000),
    ]


@pytest.mark.asyncio
async def test_reporter_start_close_and_disabled_paths() -> None:
    sent: list[bytes] = []

    def post(_endpoint: str, body: bytes, _timeout: float) -> bool:
        sent.append(body)
        return True

    enabled = TelemetryReporter(
        metrics=BridgeMetrics(),
        app_version="1.5.0",
        endpoint="https://telemetry.example/v1/events",
        enabled=True,
        interval_seconds=0.01,
        post=post,
    )
    enabled.start()
    enabled.start()
    await asyncio.sleep(0.03)
    await enabled.close()

    unstarted = TelemetryReporter(
        metrics=BridgeMetrics(),
        app_version="1.5.0",
        endpoint="https://telemetry.example/v1/events",
        enabled=True,
        post=post,
    )
    await unstarted.close()

    disabled = TelemetryReporter(
        metrics=BridgeMetrics(),
        app_version="1.5.0",
        endpoint="https://telemetry.example/v1/events",
        enabled=False,
        post=post,
    )
    disabled.start()
    assert disabled.enabled is False
    assert await disabled.flush() is True
    await disabled.close()

    assert len(sent) == 2


@pytest.mark.asyncio
async def test_reporter_survives_a_failing_flush(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics = BridgeMetrics()
    snapshots = 0

    def telemetry_snapshot() -> MetricsSnapshot:
        nonlocal snapshots
        snapshots += 1
        if snapshots == 1:
            raise RuntimeError("snapshot failed")
        return MetricsSnapshot(requests=(), features=(), internal=())

    monkeypatch.setattr(metrics, "telemetry_snapshot", telemetry_snapshot)
    sent: list[bytes] = []

    def post(_endpoint: str, body: bytes, _timeout: float) -> bool:
        sent.append(body)
        return True

    reporter = TelemetryReporter(
        metrics=metrics,
        app_version="1.5.0",
        endpoint="https://telemetry.example/v1/events",
        enabled=True,
        interval_seconds=0.01,
        post=post,
    )
    reporter.start()
    for _ in range(50):
        await asyncio.sleep(0.01)
        if sent:
            break
    await reporter.close()

    assert len(sent) == 1


@pytest.mark.asyncio
async def test_close_bounds_an_in_flight_periodic_batch() -> None:
    release = threading.Event()
    started = threading.Event()

    def post(_endpoint: str, _body: bytes, _timeout: float) -> bool:
        started.set()
        release.wait()
        return True

    reporter = TelemetryReporter(
        metrics=BridgeMetrics(),
        app_version="1.5.0",
        endpoint="https://telemetry.example/v1/events",
        enabled=True,
        interval_seconds=0.01,
        timeout_seconds=30.0,
        shutdown_timeout_seconds=0.05,
        post=post,
    )
    reporter.start()
    for _ in range(50):
        await asyncio.sleep(0.01)
        if started.is_set():
            break
    started_at = asyncio.get_running_loop().time()

    await reporter.close()

    assert started.is_set()
    assert asyncio.get_running_loop().time() - started_at < 1.0
    release.set()


@pytest.mark.asyncio
async def test_close_applies_the_shutdown_deadline() -> None:
    timeouts: list[float] = []

    def post(_endpoint: str, _body: bytes, timeout: float) -> bool:
        timeouts.append(timeout)
        return True

    reporter = TelemetryReporter(
        metrics=BridgeMetrics(),
        app_version="1.5.0",
        endpoint="https://telemetry.example/v1/events",
        enabled=True,
        timeout_seconds=30.0,
        shutdown_timeout_seconds=0.25,
        post=post,
    )

    await reporter.close()

    assert timeouts
    assert all(0 < timeout <= 0.25 for timeout in timeouts)


def test_events_since_ignores_non_positive_deltas() -> None:
    baseline = MetricsSnapshot(
        requests=(RequestMetric("health", "success", "not_applicable", 1, 100),),
        features=(("tools", 2),),
        internal=(("warm_hit", 2),),
    )
    unchanged = MetricsSnapshot(
        requests=(RequestMetric("health", "success", "not_applicable", 1, 50),),
        features=(("tools", 1),),
        internal=(("warm_hit", 2),),
    )
    increased = MetricsSnapshot(
        requests=(RequestMetric("health", "success", "not_applicable", 2, 50),),
        features=(("tools", 3),),
        internal=(("warm_hit", 3),),
    )

    assert _events_since(baseline, unchanged, include_startup=False) == []
    assert _events_since(baseline, increased, include_startup=False) == [
        {
            "name": "request",
            "route": "health",
            "outcome": "success",
            "mode": "not_applicable",
            "count": 1,
            "duration_ms_sum": 0,
        },
        {"name": "feature", "feature": "tools", "count": 1},
        {"name": "internal", "metric": "warm_hit", "count": 1},
    ]


@pytest.mark.parametrize(
    ("system", "machine", "expected_os", "expected_arch"),
    [
        ("Linux", "x86_64", "linux", "x86_64"),
        ("Windows", "AMD64", "windows", "x86_64"),
        ("Unknown", "riscv64", "other", "other"),
    ],
)
def test_runtime_metadata_normalizes_platform(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    machine: str,
    expected_os: str,
    expected_arch: str,
) -> None:
    monkeypatch.setattr("factory_droid_openai.telemetry.platform.system", lambda: system)
    monkeypatch.setattr("factory_droid_openai.telemetry.platform.machine", lambda: machine)

    metadata = _runtime_metadata("1.5.0")

    assert metadata["schema"] == 1
    assert metadata["app"] == "factory_droid_openai"
    assert metadata["event_schema"] == 1
    assert metadata["app_version"] == "1.5.0"
    assert metadata["runtime"] == "python"
    assert metadata["runtime_version"] == f"{sys.version_info.major}.{sys.version_info.minor}"
    assert metadata["os"] == expected_os
    assert metadata["arch"] == expected_arch


class _Response:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class _Opener:
    def __init__(self, result: _Response | Exception) -> None:
        self._result = result
        self.captured: dict[str, Any] = {}

    def open(self, request: Any, *, timeout: float) -> _Response:
        self.captured["request"] = request
        self.captured["timeout"] = timeout
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


@pytest.mark.parametrize(
    ("status", "expected"),
    [(200, True), (204, True), (302, False), (500, False)],
)
def test_post_accepts_success_responses_only(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    expected: bool,
) -> None:
    opener = _Opener(_Response(status))
    monkeypatch.setattr(telemetry_module, "_OPENER", opener)

    assert _post("https://telemetry.example/v1/events", b"{}", 0.5) is expected
    assert opener.captured["request"].full_url == "https://telemetry.example/v1/events"
    assert opener.captured["request"].data == b"{}"
    assert opener.captured["request"].get_header("Content-type") == "application/json"
    assert opener.captured["timeout"] == 0.5


def test_post_refuses_redirects() -> None:
    handler = _NoRedirectHandler()

    assert (
        handler.redirect_request(
            cast("Any", None),
            cast("Any", None),
            302,
            "Found",
            cast("Any", None),
            "http://telemetry.example/v1/events",
        )
        is None
    )


@pytest.mark.parametrize("error", [OSError("offline"), ValueError("invalid URL")])
def test_post_swallows_network_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    monkeypatch.setattr(telemetry_module, "_OPENER", _Opener(error))

    assert _post("https://telemetry.example/v1/events", b"{}", 0.5) is False


@pytest.mark.parametrize("endpoint", ["http://telemetry.example/v1/events", "https://[invalid"])
def test_post_rejects_non_https_and_invalid_urls(endpoint: str) -> None:
    assert _post(endpoint, b"{}", 0.5) is False
