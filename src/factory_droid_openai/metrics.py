from __future__ import annotations

import threading
from collections import Counter


class BridgeMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request_totals: Counter[tuple[str, int]] = Counter()
        self._request_duration_sum = 0.0
        self._request_duration_count = 0
        self._queue_wait_sum = 0.0
        self._queue_wait_count = 0
        self._startup_sum = 0.0
        self._startup_count = 0
        self._ttft_sum = 0.0
        self._ttft_count = 0
        self._active_sessions = 0
        self._queued_requests = 0
        self._overload_rejections = 0
        self._payload_rejections = 0
        self._forced_kills = 0
        self._model_discovery_failures = 0
        self._warm_sessions = 0
        self._warm_hits = 0
        self._warm_misses = 0
        self._warm_failures = 0
        self._pending_reaps = 0

    def record_request(self, outcome: str, status_code: int, seconds: float) -> None:
        with self._lock:
            self._request_totals[(outcome, status_code)] += 1
            self._request_duration_sum += max(0.0, seconds)
            self._request_duration_count += 1

    def observe_queue_wait(self, seconds: float) -> None:
        with self._lock:
            self._queue_wait_sum += max(0.0, seconds)
            self._queue_wait_count += 1

    def observe_droid_startup(self, seconds: float) -> None:
        with self._lock:
            self._startup_sum += max(0.0, seconds)
            self._startup_count += 1

    def observe_ttft(self, seconds: float) -> None:
        with self._lock:
            self._ttft_sum += max(0.0, seconds)
            self._ttft_count += 1

    def set_admission(self, *, active: int, queued: int) -> None:
        with self._lock:
            self._active_sessions = active
            self._queued_requests = queued

    def increment_overload_rejections(self) -> None:
        with self._lock:
            self._overload_rejections += 1

    def increment_payload_rejections(self) -> None:
        with self._lock:
            self._payload_rejections += 1

    def increment_forced_kills(self) -> None:
        with self._lock:
            self._forced_kills += 1

    def increment_model_discovery_failures(self) -> None:
        with self._lock:
            self._model_discovery_failures += 1

    def set_warm_sessions(self, count: int) -> None:
        with self._lock:
            self._warm_sessions = count

    def increment_warm_hits(self) -> None:
        with self._lock:
            self._warm_hits += 1

    def increment_warm_misses(self) -> None:
        with self._lock:
            self._warm_misses += 1

    def increment_warm_failures(self) -> None:
        with self._lock:
            self._warm_failures += 1

    def set_pending_reaps(self, count: int) -> None:
        with self._lock:
            self._pending_reaps = count

    def render(self) -> str:
        with self._lock:
            request_totals = sorted(self._request_totals.items())
            values = {
                "request_duration_sum": self._request_duration_sum,
                "request_duration_count": self._request_duration_count,
                "queue_wait_sum": self._queue_wait_sum,
                "queue_wait_count": self._queue_wait_count,
                "startup_sum": self._startup_sum,
                "startup_count": self._startup_count,
                "ttft_sum": self._ttft_sum,
                "ttft_count": self._ttft_count,
                "active_sessions": self._active_sessions,
                "queued_requests": self._queued_requests,
                "overload_rejections": self._overload_rejections,
                "payload_rejections": self._payload_rejections,
                "forced_kills": self._forced_kills,
                "model_discovery_failures": self._model_discovery_failures,
                "warm_sessions": self._warm_sessions,
                "warm_hits": self._warm_hits,
                "warm_misses": self._warm_misses,
                "warm_failures": self._warm_failures,
                "pending_reaps": self._pending_reaps,
            }

        lines = [
            "# TYPE factory_droid_openai_requests_total counter",
            *[
                (
                    "factory_droid_openai_requests_total"
                    f'{{outcome="{outcome}",status="{status_code}"}} {count}'
                )
                for (outcome, status_code), count in request_totals
            ],
            "# TYPE factory_droid_openai_request_duration_seconds summary",
            (f"factory_droid_openai_request_duration_seconds_sum {values['request_duration_sum']}"),
            (
                "factory_droid_openai_request_duration_seconds_count "
                f"{values['request_duration_count']}"
            ),
            "# TYPE factory_droid_openai_queue_wait_seconds summary",
            f"factory_droid_openai_queue_wait_seconds_sum {values['queue_wait_sum']}",
            f"factory_droid_openai_queue_wait_seconds_count {values['queue_wait_count']}",
            "# TYPE factory_droid_openai_droid_startup_seconds summary",
            f"factory_droid_openai_droid_startup_seconds_sum {values['startup_sum']}",
            f"factory_droid_openai_droid_startup_seconds_count {values['startup_count']}",
            "# TYPE factory_droid_openai_ttft_seconds summary",
            f"factory_droid_openai_ttft_seconds_sum {values['ttft_sum']}",
            f"factory_droid_openai_ttft_seconds_count {values['ttft_count']}",
            "# TYPE factory_droid_openai_active_sessions gauge",
            f"factory_droid_openai_active_sessions {values['active_sessions']}",
            "# TYPE factory_droid_openai_queued_requests gauge",
            f"factory_droid_openai_queued_requests {values['queued_requests']}",
            "# TYPE factory_droid_openai_overload_rejections_total counter",
            (f"factory_droid_openai_overload_rejections_total {values['overload_rejections']}"),
            "# TYPE factory_droid_openai_payload_rejections_total counter",
            (f"factory_droid_openai_payload_rejections_total {values['payload_rejections']}"),
            "# TYPE factory_droid_openai_forced_kills_total counter",
            f"factory_droid_openai_forced_kills_total {values['forced_kills']}",
            "# TYPE factory_droid_openai_model_discovery_failures_total counter",
            (
                "factory_droid_openai_model_discovery_failures_total "
                f"{values['model_discovery_failures']}"
            ),
            "# TYPE factory_droid_openai_warm_sessions gauge",
            f"factory_droid_openai_warm_sessions {values['warm_sessions']}",
            "# TYPE factory_droid_openai_warm_session_hits_total counter",
            f"factory_droid_openai_warm_session_hits_total {values['warm_hits']}",
            "# TYPE factory_droid_openai_warm_session_misses_total counter",
            f"factory_droid_openai_warm_session_misses_total {values['warm_misses']}",
            "# TYPE factory_droid_openai_warm_session_failures_total counter",
            f"factory_droid_openai_warm_session_failures_total {values['warm_failures']}",
            "# TYPE factory_droid_openai_pending_reaps gauge",
            f"factory_droid_openai_pending_reaps {values['pending_reaps']}",
        ]
        return "\n".join(lines) + "\n"
