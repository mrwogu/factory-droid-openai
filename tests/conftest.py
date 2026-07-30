from collections.abc import Iterator
from typing import Any

import pytest

from factory_droid_openai import telemetry


@pytest.fixture(autouse=True)
def block_telemetry_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fails tests that reach the collector instead of disabling or stubbing telemetry."""
    attempts: list[str] = []

    class _RefusingOpener:
        def open(self, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            attempts.append("opener")
            raise AssertionError("telemetry opened a connection to the collector")

    def _refusing_post(endpoint: str, body: bytes, timeout_seconds: float) -> bool:
        del endpoint, body, timeout_seconds
        attempts.append("post")
        return False

    monkeypatch.setattr(telemetry, "_OPENER", _RefusingOpener())
    monkeypatch.setattr(telemetry, "_post", _refusing_post)
    yield
    assert not attempts, (
        "telemetry used the real sender: disable telemetry in the settings or "
        f"pass a stub sender (reached: {sorted(set(attempts))})"
    )
