from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


@dataclass(frozen=True, slots=True)
class QuarantinedModel:
    model_id: str
    reason: str
    until: float


class ModelQuarantine:
    """Remembers models Droid refused, so the bridge stops offering them.

    The Droid catalog lists every model the CLI knows about, including ones an
    organization policy blocks, and the refusal only surfaces when a session is
    initialized. Recording the refusal keeps blocked models out of
    ``GET /v1/models`` and turns later requests into an immediate error instead
    of another Droid startup.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl_seconds = max(0.0, ttl_seconds)
        self._clock = clock
        self._entries: dict[str, QuarantinedModel] = {}

    @property
    def enabled(self) -> bool:
        return self._ttl_seconds > 0

    def record(self, model_id: str, reason: str) -> bool:
        """Quarantine ``model_id``; returns whether it was newly recorded."""
        if not self.enabled or not model_id:
            return False
        known = self.reason(model_id) is not None
        self._entries[model_id] = QuarantinedModel(
            model_id=model_id,
            reason=reason,
            until=self._clock() + self._ttl_seconds,
        )
        return not known

    def reason(self, model_id: str) -> str | None:
        """Why ``model_id`` is unavailable, or ``None`` when it may be used."""
        entry = self._entries.get(model_id)
        if entry is None:
            return None
        if entry.until <= self._clock():
            del self._entries[model_id]
            return None
        return entry.reason

    def allows(self, model_id: str) -> bool:
        return self.reason(model_id) is None

    def blocked(self) -> tuple[str, ...]:
        return tuple(sorted(model_id for model_id in self._entries if not self.allows(model_id)))

    def filter(self, model_ids: Iterable[str]) -> tuple[str, ...]:
        return tuple(model_id for model_id in model_ids if self.allows(model_id))
