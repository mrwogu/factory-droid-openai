from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import IO, Any

LOGGER_NAME = "factory_droid_openai"
TRACE = 5
LOG_LEVELS: tuple[str, ...] = (
    "critical",
    "error",
    "warning",
    "info",
    "debug",
    "trace",
)
LOG_FORMATS: tuple[str, ...] = ("text", "json")
_FIELDS_KEY = "factory_fields"
_UNSET = "-"

_logger = logging.getLogger(LOGGER_NAME)
_request_id: ContextVar[str] = ContextVar("factory_droid_openai_request_id", default=_UNSET)
_timeline: ContextVar[RequestTimeline | None] = ContextVar(
    "factory_droid_openai_timeline",
    default=None,
)


@dataclass(slots=True)
class RequestTimeline:
    """Per-request phase stopwatch shared by the HTTP layer and the runner."""

    request_id: str = _UNSET
    started: float = field(default_factory=time.perf_counter)
    phases: dict[str, float] = field(default_factory=dict)
    _last: float = field(default_factory=time.perf_counter)

    def mark(self, name: str) -> float:
        now = time.perf_counter()
        elapsed = millis(now - self._last)
        self._last = now
        self.phases[name] = elapsed
        return elapsed

    def observe(self, name: str, seconds: float) -> float:
        elapsed = millis(seconds)
        self.phases[name] = elapsed
        return elapsed

    def since_start(self, name: str) -> float:
        elapsed = millis(time.perf_counter() - self.started)
        self.phases[name] = elapsed
        return elapsed

    @property
    def total_ms(self) -> float:
        return millis(time.perf_counter() - self.started)

    def fields(self) -> dict[str, float]:
        return {**self.phases, "total_ms": self.total_ms}


def millis(seconds: float) -> float:
    return round(max(0.0, seconds) * 1000, 1)


def bind_request(request_id: str) -> RequestTimeline:
    """Bind a request id and a fresh timeline to the current context."""
    _request_id.set(request_id)
    timeline = RequestTimeline(request_id=request_id)
    _timeline.set(timeline)
    return timeline


def current_timeline() -> RequestTimeline | None:
    return _timeline.get()


def current_request_id() -> str:
    return _request_id.get()


def level_number(level: str) -> int:
    normalized = level.strip().lower()
    if normalized == "trace":
        return TRACE
    if normalized not in LOG_LEVELS:
        raise ValueError(f"Unsupported log level: {level}")
    return int(getattr(logging, normalized.upper()))


class KeyValueFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        parts = [
            f"{record.levelname:<8}",
            self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            f"event={record.getMessage()}",
            f"request_id={_render(_field_request_id(record))}",
        ]
        parts.extend(f"{key}={_render(value)}" for key, value in _fields(record).items())
        line = " ".join(parts)
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "event": record.getMessage(),
            "request_id": _field_request_id(record),
            **_fields(record),
        }
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, sort_keys=True)


def _fields(record: logging.LogRecord) -> dict[str, Any]:
    fields = getattr(record, _FIELDS_KEY, None)
    return dict(fields) if isinstance(fields, dict) else {}


def _field_request_id(record: logging.LogRecord) -> str:
    value = getattr(record, "factory_request_id", None)
    return str(value) if value else _UNSET


def _render(value: Any) -> str:
    if value is None:
        return _UNSET
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.1f}" if value else "0.0"
    text = str(value)
    if not text:
        return _UNSET
    if any(char.isspace() for char in text):
        return json.dumps(text)
    return text


def configure_logging(
    *,
    level: str = "info",
    log_format: str = "text",
    stream: IO[str] | None = None,
) -> logging.Logger:
    """Install the bridge log handler, replacing any previous one."""
    if log_format not in LOG_FORMATS:
        raise ValueError(f"Unsupported log format: {log_format}")
    logging.addLevelName(TRACE, "TRACE")
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(JsonFormatter() if log_format == "json" else KeyValueFormatter())
    for existing in list(_logger.handlers):
        _logger.removeHandler(existing)
    _logger.addHandler(handler)
    _logger.setLevel(level_number(level))
    _logger.propagate = False
    return _logger


def enabled(level: int) -> bool:
    return _logger.isEnabledFor(level)


def log_event(level: int, event: str, **fields: Any) -> None:
    if not _logger.isEnabledFor(level):
        return
    timeline = _timeline.get()
    if timeline is not None and not any(key.endswith("_ms") for key in fields):
        fields["elapsed_ms"] = timeline.total_ms
    _logger.log(
        level,
        event,
        extra={
            _FIELDS_KEY: {key: value for key, value in fields.items() if value is not None},
            "factory_request_id": _request_id.get(),
        },
    )


def trace(event: str, **fields: Any) -> None:
    log_event(TRACE, event, **fields)


def debug(event: str, **fields: Any) -> None:
    log_event(logging.DEBUG, event, **fields)


def info(event: str, **fields: Any) -> None:
    log_event(logging.INFO, event, **fields)


def warning(event: str, **fields: Any) -> None:
    log_event(logging.WARNING, event, **fields)
