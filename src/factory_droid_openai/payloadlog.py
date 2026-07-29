from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from factory_droid_openai.logs import current_request_id
from factory_droid_openai.logs import warning as log_warning

PAYLOAD_TRACE_MODES: tuple[str, ...] = ("off", "heads", "full")
_HEAD_CHARS = 2048
_TAIL_CHARS = 1024
_MAX_FULL_CHARS = 1 << 20
_REDACTED = "[REDACTED]"
_SENSITIVE_KEY = r"(?:api_?key|(?:access_)?token|(?:client_)?secret|password|authorization)"
# Payloads are minified single-line JSON, so an unquoted `key: value` run has to
# stop at structural delimiters; matching to end of line would let one redaction
# swallow every remaining field of the record.
_SENSITIVE_VALUE = r"[^\r\n\"',;&}\]]+"
# Prompts and tool payloads can carry credentials pasted by users or tool
# output, so every line written to the trace file passes through redaction
# regardless of mode.
_REDACTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"sk-[A-Za-z0-9_-]{8,}"), _REDACTED),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{8,}"), _REDACTED),
    (re.compile(r"(?i)basic\s+[A-Za-z0-9+/]{4,}={0,2}"), _REDACTED),
    (
        re.compile(
            rf'(?i)(?P<prefix>(?<!\\)(?P<escape>\\*)"{_SENSITIVE_KEY}(?P=escape)"'
            rf'\s*:\s*(?P=escape)").*?(?P<suffix>(?<!\\)(?P=escape)")'
        ),
        rf"\g<prefix>{_REDACTED}\g<suffix>",
    ),
    (
        # The separator run is possessive so the marker lookahead cannot be
        # bypassed by giving whitespace back, which would re-redact a redaction.
        re.compile(
            rf"(?im)({_SENSITIVE_KEY}\s*(?:=|:)\s*+)(?!{re.escape(_REDACTED)}){_SENSITIVE_VALUE}",
        ),
        rf"\g<1>{_REDACTED}",
    ),
    (re.compile(r"[A-Za-z0-9+/]{120,}={0,2}"), _REDACTED),
)


def redact(text: str) -> str:
    for pattern, replacement in _REDACTION_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            rendered_key = str(key)
            if re.fullmatch(_SENSITIVE_KEY, rendered_key, re.IGNORECASE):
                redacted[rendered_key] = _REDACTED
            else:
                redacted[redact(rendered_key)] = _redact_value(item)
        return redacted
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item) for item in value)
    return value


def validate_trace_path(path: Path) -> None:
    """Reject destinations that could never be appended to as a private regular file."""
    if not path.parent.is_dir():
        raise ValueError(f"Payload trace directory does not exist: {path.parent}")
    if path.is_symlink():
        raise ValueError(f"Payload trace file must not be a symlink: {path}")
    if path.exists() and not path.is_file():
        raise ValueError(f"Payload trace file must be a regular file: {path}")


@dataclass(frozen=True, slots=True)
class PayloadTracer:
    mode: str
    path: Path | None
    _lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.mode not in PAYLOAD_TRACE_MODES:
            raise ValueError(f"Unsupported payload trace mode: {self.mode}")
        if self.mode != "off" and self.path is None:
            raise ValueError("Payload tracing requires a destination file")
        if self.path is not None:
            validate_trace_path(self.path)

    def trace(self, event: str, payload: str, **fields: Any) -> None:
        """Write one redacted payload event; logging failures never fail requests."""
        path = self.path
        if self.mode == "off" or path is None:
            return
        try:
            encoded = payload.encode("utf-8")
            redacted = redact(payload)
            record: dict[str, Any] = {
                **_redact_value({key: value for key, value in fields.items() if value is not None}),
                "ts": datetime.now(UTC).isoformat(),
                "event": redact(event),
                "request_id": redact(current_request_id()),
                "mode": self.mode,
                "payload_bytes": len(encoded),
                "redacted_sha256": hashlib.sha256(redacted.encode("utf-8")).hexdigest(),
            }
            if self.mode == "full":
                record["payload"] = redacted[:_MAX_FULL_CHARS]
                if len(redacted) > _MAX_FULL_CHARS:
                    record["payload_truncated"] = True
            else:
                record["payload_head"] = redacted[:_HEAD_CHARS]
                if len(redacted) > _HEAD_CHARS:
                    record["payload_tail"] = redacted[-_TAIL_CHARS:]
            line = json.dumps(
                record,
                ensure_ascii=True,
                default=lambda value: redact(str(value)),
            )
            self._write_line(path, line)
        except Exception as exc:
            # Deliberately broad: a diagnostics writer must never fail a request.
            log_warning("payload_trace.write_failed", error_type=type(exc).__name__)

    def _write_line(self, path: Path, line: str) -> None:
        # Written synchronously: the payload cap above bounds the cost, and an
        # async writer would drop pending records when a request is cancelled.
        flags = (
            os.O_WRONLY
            | os.O_APPEND
            | os.O_CREAT
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        with self._lock:
            descriptor = os.open(path, flags, stat.S_IRUSR | stat.S_IWUSR)
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise OSError("Payload trace destination must be a regular file")
                if os.name != "nt":  # pragma: no branch - fchmod is POSIX only
                    os.fchmod(descriptor, stat.S_IRUSR | stat.S_IWUSR)
                with os.fdopen(
                    descriptor,
                    "a",
                    encoding="utf-8",
                    closefd=False,
                ) as handle:
                    handle.write(f"{line}\n")
            finally:
                os.close(descriptor)


def configure_payload_tracing(*, mode: str, path: Path | None) -> PayloadTracer:
    """Build an independent payload tracer for one application instance."""
    return PayloadTracer(mode=mode, path=path)


NULL_PAYLOAD_TRACER = PayloadTracer(mode="off", path=None)
