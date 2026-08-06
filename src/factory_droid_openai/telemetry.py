from __future__ import annotations

import asyncio
import contextlib
import json
import platform
import sys
import threading
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import IO, TYPE_CHECKING, Literal, cast

from factory_droid_openai.metrics import BridgeMetrics, MetricsSnapshot, RequestMetric

if TYPE_CHECKING:
    from http.client import HTTPMessage

DEFAULT_TELEMETRY_ENDPOINT = "https://telemetry.guziak.net/v1/events"
DEFAULT_TELEMETRY_INTERVAL_SECONDS = 900.0
# Reporting runs off the request path, so the budget only has to stay well below
# the reporting interval. A tighter budget mostly discards cold DNS and TLS
# handshakes, which silently drops the batch.
DEFAULT_TELEMETRY_TIMEOUT_SECONDS = 5.0
# Shutdown blocks on the final batch, so it gets a much smaller budget.
DEFAULT_TELEMETRY_SHUTDOWN_TIMEOUT_SECONDS = 1.0
_MAX_EVENTS_PER_REQUEST = 25
_MAX_EVENT_COUNT = 1_000_000
_MAX_EVENT_DURATION_SUM_MS = 1_000_000_000_000

TelemetryPost = Callable[[str, bytes, float], bool]
SendResult = Literal["sent", "failed", "unknown"]


class TelemetryReporter:
    def __init__(
        self,
        *,
        metrics: BridgeMetrics,
        app_version: str,
        endpoint: str,
        enabled: bool,
        interval_seconds: float = DEFAULT_TELEMETRY_INTERVAL_SECONDS,
        timeout_seconds: float = DEFAULT_TELEMETRY_TIMEOUT_SECONDS,
        shutdown_timeout_seconds: float = DEFAULT_TELEMETRY_SHUTDOWN_TIMEOUT_SECONDS,
        post: TelemetryPost | None = None,
    ) -> None:
        self._metrics = metrics
        self._app_version = app_version
        self._endpoint = endpoint
        self._enabled = enabled
        self._interval_seconds = interval_seconds
        self._timeout_seconds = timeout_seconds
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._post = post or _post
        self._baseline = MetricsSnapshot(requests=(), features=(), internal=())
        self._startup_pending = True
        self._stop = asyncio.Event()
        self._flush_lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start(self) -> None:
        if not self._enabled or self._task is not None:
            return
        self._task = asyncio.create_task(self._run())

    async def close(self) -> None:
        if not self._enabled:
            return
        task = self._task
        self._task = None
        if task is not None:
            self._stop.set()
            # A periodic batch may be mid-flight, and shutdown must not inherit
            # the much larger periodic budget.
            done, _pending = await asyncio.wait({task}, timeout=self._shutdown_timeout_seconds)
            if not done:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        await self.flush(timeout_seconds=self._shutdown_timeout_seconds)

    async def flush(self, *, timeout_seconds: float | None = None) -> bool:
        if not self._enabled:
            return True
        async with self._flush_lock:
            snapshot = self._metrics.telemetry_snapshot()
            events = _events_since(
                self._baseline,
                snapshot,
                include_startup=self._startup_pending,
            )
            if not events:
                return True
            metadata = _runtime_metadata(self._app_version)
            loop = asyncio.get_running_loop()
            budget = self._timeout_seconds if timeout_seconds is None else timeout_seconds
            deadline = loop.time() + budget
            for batch in _event_batches(events):
                remaining = deadline - loop.time()
                if remaining <= 0:
                    return False
                payload = {
                    **metadata,
                    "events": batch,
                }
                body = json.dumps(
                    payload,
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode()
                sent = await _post_with_deadline(
                    self._post,
                    self._endpoint,
                    body,
                    remaining,
                )
                if sent != "failed":
                    # A timed-out send may still land, so counters advance to keep
                    # aggregates from double counting.
                    self._baseline = _advance_snapshot(self._baseline, batch)
                if sent == "sent" and _contains_startup(batch):
                    # Startup is the one event that must arrive, so an uncertain
                    # send keeps it queued for the next batch.
                    self._startup_pending = False
                if sent != "sent":
                    return False
            return True

    async def _run(self) -> None:
        while not self._stop.is_set():
            # Reporting is best effort and shutdown awaits this task, so a
            # reporting failure must not escape.
            with contextlib.suppress(Exception):
                await self.flush()
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=self._interval_seconds,
                )
            except TimeoutError:
                continue


def _events_since(
    baseline: MetricsSnapshot,
    current: MetricsSnapshot,
    *,
    include_startup: bool,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    if include_startup:
        events.append({"name": "bridge_started", "count": 1})

    previous_requests = {
        (metric.route, metric.outcome, metric.mode): metric for metric in baseline.requests
    }
    for metric in current.requests:
        key = (metric.route, metric.outcome, metric.mode)
        previous = previous_requests.get(key)
        previous_count = previous.count if previous is not None else 0
        count = metric.count - previous_count
        if count <= 0:
            continue
        previous_duration = previous.duration_ms_sum if previous is not None else 0
        events.append(
            {
                "name": "request",
                "route": metric.route,
                "outcome": metric.outcome,
                "mode": metric.mode,
                "count": count,
                "duration_ms_sum": max(0, metric.duration_ms_sum - previous_duration),
            }
        )

    previous_features = dict(baseline.features)
    for feature, total in current.features:
        count = total - previous_features.get(feature, 0)
        if count > 0:
            events.append({"name": "feature", "feature": feature, "count": count})

    previous_internal = dict(baseline.internal)
    for metric_name, total in current.internal:
        count = total - previous_internal.get(metric_name, 0)
        if count > 0:
            events.append({"name": "internal", "metric": metric_name, "count": count})
    return events


def _contains_startup(batch: list[dict[str, object]]) -> bool:
    return any(event["name"] == "bridge_started" for event in batch)


def _event_batches(events: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    batches: list[list[dict[str, object]]] = []
    batch_keys: list[set[tuple[object, ...]]] = []
    for event in events:
        key = _event_key(event)
        for part in _split_event(event):
            for batch, keys in zip(batches, batch_keys, strict=True):
                if len(batch) < _MAX_EVENTS_PER_REQUEST and key not in keys:
                    batch.append(part)
                    keys.add(key)
                    break
            else:
                batches.append([part])
                batch_keys.append({key})
    return batches


def _split_event(event: dict[str, object]) -> list[dict[str, object]]:
    count = cast("int", event["count"])
    duration = cast("int", event.get("duration_ms_sum", 0))
    if count <= _MAX_EVENT_COUNT and duration <= _MAX_EVENT_DURATION_SUM_MS:
        return [event]
    duration = min(duration, count * _MAX_EVENT_DURATION_SUM_MS)
    part_total = max(
        (count + _MAX_EVENT_COUNT - 1) // _MAX_EVENT_COUNT,
        (duration + _MAX_EVENT_DURATION_SUM_MS - 1) // _MAX_EVENT_DURATION_SUM_MS,
    )
    parts: list[dict[str, object]] = []
    remaining_count = count
    remaining_duration = duration
    for remaining_parts in range(part_total, 0, -1):
        part_count = min(
            _MAX_EVENT_COUNT,
            remaining_count - remaining_parts + 1,
        )
        part = {**event, "count": part_count}
        if event["name"] == "request":
            part_duration = min(
                _MAX_EVENT_DURATION_SUM_MS,
                (remaining_duration + remaining_parts - 1) // remaining_parts,
            )
            part["duration_ms_sum"] = part_duration
            remaining_duration -= part_duration
        parts.append(part)
        remaining_count -= part_count
    return parts


def _event_key(event: dict[str, object]) -> tuple[object, ...]:
    name = event["name"]
    if name == "request":
        return name, event["route"], event["outcome"], event["mode"]
    if name == "feature":
        return name, event["feature"]
    if name == "internal":
        return name, event["metric"]
    return (name,)


def _advance_snapshot(
    baseline: MetricsSnapshot,
    delivered: list[dict[str, object]],
) -> MetricsSnapshot:
    requests = {(metric.route, metric.outcome, metric.mode): metric for metric in baseline.requests}
    features = dict(baseline.features)
    internal = dict(baseline.internal)
    for event in delivered:
        name = event["name"]
        count = cast("int", event["count"])
        if name == "request":
            route = cast("str", event["route"])
            outcome = cast("str", event["outcome"])
            mode = cast("str", event["mode"])
            key = route, outcome, mode
            previous = requests.get(key)
            requests[key] = RequestMetric(
                route=route,
                outcome=outcome,
                mode=mode,
                count=(previous.count if previous is not None else 0) + count,
                duration_ms_sum=(previous.duration_ms_sum if previous is not None else 0)
                + cast("int", event["duration_ms_sum"]),
            )
        elif name == "feature":
            feature = cast("str", event["feature"])
            features[feature] = features.get(feature, 0) + count
        elif name == "internal":
            metric = cast("str", event["metric"])
            internal[metric] = internal.get(metric, 0) + count
    return MetricsSnapshot(
        requests=tuple(
            sorted(
                requests.values(),
                key=lambda metric: (metric.route, metric.outcome, metric.mode),
            )
        ),
        features=tuple(sorted(features.items())),
        internal=tuple(sorted(internal.items())),
    )


async def _post_with_deadline(
    post: TelemetryPost,
    endpoint: str,
    body: bytes,
    timeout_seconds: float,
) -> SendResult:
    loop = asyncio.get_running_loop()
    delivered: asyncio.Future[bool] = loop.create_future()

    def send() -> None:
        try:
            sent = post(endpoint, body, timeout_seconds)
        except Exception:
            sent = False
        # The loop can already be closed once a timed-out send returns.
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(delivered.set_result, sent)

    thread = threading.Thread(
        target=send,
        name="factory-droid-openai-telemetry",
        daemon=True,
    )
    try:
        thread.start()
    except Exception:
        return "failed"
    done, _pending = await asyncio.wait({delivered}, timeout=timeout_seconds)
    if not done:
        return "unknown"
    return "sent" if delivered.result() else "failed"


def _runtime_metadata(app_version: str) -> dict[str, object]:
    operating_system = {
        "Darwin": "darwin",
        "Linux": "linux",
        "Windows": "windows",
    }.get(platform.system(), "other")
    architecture = {
        "aarch64": "arm64",
        "amd64": "x86_64",
        "arm64": "arm64",
        "x86_64": "x86_64",
    }.get(platform.machine().lower(), "other")
    return {
        "schema": 1,
        "app": "factory_droid_openai",
        "event_schema": 1,
        "app_version": app_version,
        "runtime": "python",
        "runtime_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "os": operating_system,
        "arch": architecture,
    }


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuses redirects so a redirect can never downgrade the transport."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        del req, fp, code, msg, headers, newurl
        return None


_OPENER = urllib.request.build_opener(_NoRedirectHandler())


def _post(endpoint: str, body: bytes, timeout_seconds: float) -> bool:
    try:
        if urllib.parse.urlsplit(endpoint).scheme != "https":
            return False
        request = urllib.request.Request(  # noqa: S310 - HTTPS is required above.
            endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "factory-droid-openai-telemetry/1",
            },
            method="POST",
        )
        with _OPENER.open(request, timeout=timeout_seconds) as response:
            return 200 <= int(response.status) < 300
    except (OSError, ValueError):
        return False
