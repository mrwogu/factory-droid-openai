from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from hashlib import _Hash

    from factory_droid_openai.attachments import AttachmentSet
    from factory_droid_openai.models import ToolDefinition

# Bumped whenever a fingerprint component changes meaning, so entries written
# by an older bridge can never be hit after an upgrade.
CACHE_SCHEMA_VERSION = "response-cache-v1"


@dataclass(frozen=True, slots=True)
class CachedResponse:
    payload: bytes
    expires_at: float
    size_bytes: int


class ResponseCache:
    """Process-local exact-match cache for completed responses.

    LRU order covers both the byte and the entry cap. The TTL is absolute:
    a hit never extends an entry, so every replayed answer is at most
    ``ttl_seconds`` old.
    """

    def __init__(
        self,
        *,
        max_bytes: int,
        max_entries: int,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._entries: OrderedDict[str, CachedResponse] = OrderedDict()
        self._lock = threading.Lock()
        self._max_bytes = max_bytes
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._bytes = 0

    def get(self, key: str) -> bytes | None:
        now = self._clock()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if now >= entry.expires_at:
                del self._entries[key]
                self._bytes -= entry.size_bytes
                return None
            self._entries.move_to_end(key)
            return entry.payload

    def put(self, key: str, payload: bytes) -> bool:
        now = self._clock()
        # The key is part of the resident footprint, so it counts toward the cap.
        size = len(key.encode("utf-8")) + len(payload)
        if size > self._max_bytes:
            return False
        with self._lock:
            # Expired entries are dead weight even before their key is read
            # again, so a store reclaims them instead of evicting live ones.
            expired = [
                stale_key for stale_key, entry in self._entries.items() if now >= entry.expires_at
            ]
            for stale_key in expired:
                entry = self._entries.pop(stale_key)
                self._bytes -= entry.size_bytes
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes -= previous.size_bytes
            self._entries[key] = CachedResponse(
                payload=payload,
                expires_at=now + self._ttl_seconds,
                size_bytes=size,
            )
            self._bytes += size
            while len(self._entries) > self._max_entries or self._bytes > self._max_bytes:
                _, evicted = self._entries.popitem(last=False)
                self._bytes -= evicted.size_bytes
        return True

    @property
    def byte_size(self) -> int:
        with self._lock:
            return self._bytes

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._entries)


def response_cache_key(
    *,
    model: str,
    reasoning_effort: str | None,
    prompt: str,
    output_format: dict[str, Any] | None,
    output_token_limit: int | None,
    stop_sequences: tuple[str, ...],
    max_tool_calls: int,
    attachments: AttachmentSet,
    native_tools: tuple[ToolDefinition, ...],
) -> str:
    """Fingerprint every input that can change the generated response.

    The prompt covers messages, tool schemas, and tool_choice; the remaining
    components cover the execution knobs the prompt cannot see. Any change to
    the component list bumps ``CACHE_SCHEMA_VERSION`` so old keys can never
    collide with new meanings.
    """
    hasher = hashlib.sha256()
    _feed(hasher, CACHE_SCHEMA_VERSION)
    _feed(hasher, model)
    _feed(hasher, reasoning_effort or "")
    _feed(hasher, prompt)
    if output_format is not None:
        _feed(hasher, _canonical_json(output_format))
    _feed(hasher, str(output_token_limit) if output_token_limit is not None else "none")
    for sequence in stop_sequences:
        _feed(hasher, sequence)
    _feed(hasher, str(max_tool_calls))
    for image in attachments.images:
        _feed(hasher, "image")
        _feed(hasher, image.media_type)
        _feed(hasher, _digest(image.data))
    for document in attachments.documents:
        _feed(hasher, "document")
        _feed(hasher, document.media_type)
        _feed(hasher, document.source_type)
        _feed(hasher, document.name or "")
        _feed(hasher, _digest(document.data))
    for tool in native_tools:
        _feed(hasher, _canonical_json(tool.model_dump(mode="json")))
    return hasher.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _feed(hasher: _Hash, value: str) -> None:
    # Length-prefixing keeps adjacent components from ever merging into one.
    encoded = value.encode("utf-8")
    hasher.update(f"{len(encoded)}:".encode("ascii"))
    hasher.update(encoded)
