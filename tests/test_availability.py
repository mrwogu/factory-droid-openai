from __future__ import annotations

from factory_droid_openai.availability import ModelQuarantine


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_quarantine_hides_a_model_until_the_ttl_expires() -> None:
    clock = FakeClock()
    quarantine = ModelQuarantine(ttl_seconds=60.0, clock=clock)

    assert quarantine.enabled is True
    assert quarantine.record("claude-fable-5", "not allowed") is True
    assert quarantine.reason("claude-fable-5") == "not allowed"
    assert quarantine.allows("claude-fable-5") is False
    assert quarantine.allows("gpt-5.4") is True
    assert quarantine.blocked() == ("claude-fable-5",)
    assert quarantine.filter(["gpt-5.4", "claude-fable-5"]) == ("gpt-5.4",)

    clock.now = 60.0

    assert quarantine.reason("claude-fable-5") is None
    assert quarantine.blocked() == ()
    assert quarantine.filter(["gpt-5.4", "claude-fable-5"]) == ("gpt-5.4", "claude-fable-5")


def test_quarantine_reports_only_the_first_record_as_new() -> None:
    clock = FakeClock()
    quarantine = ModelQuarantine(ttl_seconds=60.0, clock=clock)

    assert quarantine.record("gpt-5.4", "first") is True
    assert quarantine.record("gpt-5.4", "second") is False
    assert quarantine.reason("gpt-5.4") == "second"

    clock.now = 120.0

    assert quarantine.record("gpt-5.4", "third") is True


def test_quarantine_can_be_disabled_and_ignores_empty_model_ids() -> None:
    quarantine = ModelQuarantine(ttl_seconds=0.0)

    assert quarantine.enabled is False
    assert quarantine.record("gpt-5.4", "not allowed") is False
    assert quarantine.allows("gpt-5.4") is True

    enabled = ModelQuarantine(ttl_seconds=60.0)

    assert enabled.record("", "not allowed") is False
