from __future__ import annotations

from typing import Any, cast

from factory_droid_openai.attachments import (
    AttachmentSet,
    DocumentAttachment,
    ImageAttachment,
)
from factory_droid_openai.cache import ResponseCache, response_cache_key
from factory_droid_openai.models import ToolDefinition, ToolFunction


class FakeClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _cache(
    *,
    max_bytes: int = 1024,
    max_entries: int = 4,
    ttl_seconds: float = 60.0,
) -> tuple[ResponseCache, FakeClock]:
    clock = FakeClock()
    cache = ResponseCache(
        max_bytes=max_bytes,
        max_entries=max_entries,
        ttl_seconds=ttl_seconds,
        clock=clock,
    )
    return cache, clock


def _key(**overrides: object) -> str:
    components: dict[str, object] = {
        "model": "factory-droid",
        "reasoning_effort": None,
        "prompt": "prompt",
        "output_format": None,
        "output_token_limit": None,
        "stop_sequences": (),
        "max_tool_calls": 1,
        "attachments": AttachmentSet(),
        "native_tools": (),
    }
    components.update(overrides)
    return response_cache_key(**cast("Any", components))


def _tool(name: str, description: str = "Read the weather.") -> ToolDefinition:
    return ToolDefinition(
        type="function",
        function=ToolFunction(name=name, description=description),
    )


def test_get_misses_when_the_cache_is_empty() -> None:
    cache, _clock = _cache()

    assert cache.get("missing") is None
    assert cache.byte_size == 0
    assert cache.entry_count == 0


def test_put_then_get_hits() -> None:
    cache, _clock = _cache()

    assert cache.put("key", b"payload") is True
    assert cache.get("key") == b"payload"
    assert cache.byte_size == len("key") + len(b"payload")
    assert cache.entry_count == 1


def test_expired_entry_is_a_miss() -> None:
    cache, clock = _cache(ttl_seconds=60.0)
    cache.put("key", b"payload")

    clock.now += 60.1

    assert cache.get("key") is None
    assert cache.byte_size == 0
    assert cache.entry_count == 0


def test_expiry_boundary_is_inclusive() -> None:
    cache, clock = _cache(ttl_seconds=60.0)
    cache.put("key", b"payload")

    clock.now = 159.999
    assert cache.get("key") == b"payload"

    clock.now = 160.0
    assert cache.get("key") is None


def test_hit_does_not_extend_ttl() -> None:
    cache, clock = _cache(ttl_seconds=60.0)
    cache.put("key", b"payload")

    clock.now = 150.0
    assert cache.get("key") == b"payload"

    # The hit did not slide the deadline: 100 + 60 stays the expiry.
    clock.now = 161.0
    assert cache.get("key") is None


def test_lru_lookup_refreshes_recency() -> None:
    cache, _clock = _cache(max_entries=2)
    cache.put("a", b"1")
    cache.put("b", b"2")

    assert cache.get("a") == b"1"
    cache.put("c", b"3")

    # "b" was the least recently used, so the entry cap evicts it.
    assert cache.get("b") is None
    assert cache.get("a") == b"1"
    assert cache.get("c") == b"3"


def test_byte_cap_evicts_least_recently_used() -> None:
    cache, _clock = _cache(max_bytes=8, max_entries=10)
    first = b"1234"
    second = b"5678"

    cache.put("a", first)
    cache.put("b", second)

    assert cache.get("a") is None
    assert cache.get("b") == second


def test_replacement_replaces_size_and_payload() -> None:
    cache, _clock = _cache()
    cache.put("key", b"longer-payload")
    cache.put("key", b"short")

    assert cache.get("key") == b"short"
    assert cache.entry_count == 1
    assert cache.byte_size == len("key") + len(b"short")


def test_oversized_payload_is_rejected() -> None:
    cache, _clock = _cache(max_bytes=8)

    assert cache.put("key", b"payload-longer-than-the-cap") is False
    assert cache.entry_count == 0
    assert cache.byte_size == 0


def test_put_prunes_expired_entries() -> None:
    cache, clock = _cache()
    cache.put("a", b"1")
    cache.put("b", b"2")

    clock.now += 60.1
    cache.put("c", b"3")

    assert cache.entry_count == 1
    assert cache.byte_size == len("c") + len(b"3")
    assert cache.get("c") == b"3"


def test_keys_are_stable_for_identical_inputs() -> None:
    first = _key()
    second = _key()

    assert first == second


def test_keys_separate_each_execution_input() -> None:
    baseline = _key()

    assert _key(model="gpt-5.4") != baseline
    assert _key(reasoning_effort="low") != baseline
    assert _key(prompt="other") != baseline
    assert _key(output_token_limit=32) != baseline
    assert _key(stop_sequences=("STOP",)) != baseline
    assert _key(max_tool_calls=2) != baseline


def test_keys_separate_output_token_limit_values() -> None:
    assert _key(output_token_limit=32) != _key(output_token_limit=64)


def test_keys_separate_stop_sequence_sets() -> None:
    assert _key(stop_sequences=("STOP",)) != _key(stop_sequences=("STOP", "END"))


def test_canonical_json_collapses_key_order_only_differences() -> None:
    assert _key(output_format={"b": 1, "a": 2}) == _key(output_format={"a": 2, "b": 1})
    assert _key(output_format={"a": 2}) != _key(output_format={"a": 3})


def test_keys_separate_attachment_bytes_behind_identical_placeholders() -> None:
    first = AttachmentSet()
    first.images.append(ImageAttachment(media_type="image/png", data="aGk="))
    second = AttachmentSet()
    second.images.append(ImageAttachment(media_type="image/png", data="Ynll"))

    assert _key(attachments=first) != _key(attachments=second)


def test_keys_separate_attachment_metadata() -> None:
    first = AttachmentSet()
    first.images.append(ImageAttachment(media_type="image/png", data="aGk="))
    second = AttachmentSet()
    second.images.append(ImageAttachment(media_type="image/jpeg", data="aGk="))

    named = AttachmentSet()
    named.documents.append(
        DocumentAttachment(media_type="text/plain", data="hi", source_type="text", name="a.txt")
    )
    unnamed = AttachmentSet()
    unnamed.documents.append(
        DocumentAttachment(media_type="text/plain", data="hi", source_type="text")
    )

    assert _key(attachments=first) != _key(attachments=second)
    assert _key(attachments=named) != _key(attachments=unnamed)


def test_keys_separate_native_tool_catalogs() -> None:
    baseline = _key(native_tools=(_tool("weather"),))

    assert _key(native_tools=()) != baseline
    assert _key(native_tools=(_tool("calendar"),)) != baseline
    assert _key(native_tools=(_tool("weather", description="Read the calendar."),)) != baseline
    assert _key(native_tools=(_tool("weather"),)) == baseline
