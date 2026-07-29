from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from factory_droid_openai.logs import current_request_id

PAYLOAD_TRACE_MODES: tuple[str, ...] = ("off", "heads", "full")
_HEAD_CHARS = 2048
_TAIL_CHARS = 1024
_REDACTED = "[REDACTED]"
# Prompts and tool payloads can carry credentials pasted by users or tool
# output, so every line written to the trace file passes through redaction
# regardless of mode.
_REDACTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)(\"(?:api_?key|token|secret|password|authorization)\"\s*:\s*\")[^\"]*(\")"),
    re.compile(r"(?i)((?:api_?key|token|secret|password)=)\S+"),
    re.compile(r"[A-Za-z0-9+/]{120,}={0,2}"),
)

_lock = threading.Lock()
_mode: str = "off"
_path: Path | None = None


def configure_payload_tracing(*, mode: str, path: Path | None) -> None:
    """Enable payload tracing to a JSONL file, replacing any previous setup."""
    global _mode, _path
    if mode not in PAYLOAD_TRACE_MODES:
        raise ValueError(f"Unsupported payload trace mode: {mode}")
    if mode != "off" and path is None:
        raise ValueError("Payload tracing requires a destination file")
    if path is not None and not path.parent.is_dir():
        raise ValueError(f"Payload trace directory does not exist: {path.parent}")
    with _lock:
        _mode = mode
        _path = path


def reset_payload_tracing() -> None:
    """Disable tracing; used by tests to isolate module state."""
    configure_payload_tracing(mode="off", path=None)


def payload_tracing_enabled() -> bool:
    return _mode != "off"


def redact(text: str) -> str:
    for pattern in _REDACTION_PATTERNS:
        if pattern.groups == 2:
            text = pattern.sub(rf"\g<1>{_REDACTED}\g<2>", text)
        elif pattern.groups == 1:
            text = pattern.sub(rf"\g<1>{_REDACTED}", text)
        else:
            text = pattern.sub(_REDACTED, text)
    return text


def trace_payload(event: str, payload: str, **fields: Any) -> None:
    """Write one redacted payload event to the trace file; no-op when off."""
    with _lock:
        mode = _mode
        path = _path
    if mode == "off" or path is None:
        return
    encoded = payload.encode("utf-8")
    record: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(),
        "event": event,
        "request_id": current_request_id(),
        "mode": mode,
        "payload_bytes": len(encoded),
        "payload_sha256": hashlib.sha256(encoded).hexdigest(),
        **{key: value for key, value in fields.items() if value is not None},
    }
    redacted = redact(payload)
    if mode == "full":
        record["payload"] = redacted
    else:
        record["payload_head"] = redacted[:_HEAD_CHARS]
        if len(redacted) > _HEAD_CHARS:
            record["payload_tail"] = redacted[-_TAIL_CHARS:]
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _lock, path.open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")
