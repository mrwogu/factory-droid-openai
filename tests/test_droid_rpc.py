from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from droid_sdk.errors import DroidClientError

from factory_droid_openai.droid_rpc import DroidRpcExtension, _ProtocolEngine

if TYPE_CHECKING:
    from collections.abc import Callable

    from droid_sdk import DroidClient


class FakeProtocol:
    def __init__(
        self,
        handler: Callable[[str, dict[str, Any]], dict[str, Any]],
    ) -> None:
        self.handler = handler
        self.calls: list[tuple[str, dict[str, Any], float | None]] = []

    async def send_request(
        self,
        method: str,
        params: dict[str, Any],
        timeout: float | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        del request_id
        self.calls.append((method, params, timeout))
        return self.handler(method, params)


def _client(protocol: object | None) -> DroidClient:
    return cast("DroidClient", SimpleNamespace(_protocol=protocol))


@pytest.mark.asyncio
async def test_add_user_message_forwards_all_structured_fields() -> None:
    protocol = FakeProtocol(lambda _method, _params: {"result": {}})
    extension = DroidRpcExtension()
    output_format = {"type": "json_schema", "schema": {"type": "object"}}

    await extension.add_user_message(
        _client(protocol),
        text="prompt",
        images=[{"type": "base64"}],
        files=[{"type": "text"}],
        output_format=output_format,
    )
    await extension.add_user_message(
        _client(protocol),
        text="plain",
        images=None,
        files=None,
        output_format=None,
    )

    assert protocol.calls[0][1] == {
        "text": "prompt",
        "images": [{"type": "base64"}],
        "files": [{"type": "text"}],
        "outputFormat": output_format,
    }
    assert protocol.calls[1][1] == {"text": "plain"}


@pytest.mark.asyncio
async def test_retune_session_pins_modes_and_omits_a_default_effort() -> None:
    protocol = FakeProtocol(lambda _method, _params: {"result": {}})
    extension = DroidRpcExtension()

    await extension.retune_session(
        _client(protocol),
        model_id="gpt-5.4",
        reasoning_effort="low",
    )
    await extension.retune_session(
        _client(protocol),
        model_id="glm-5.2",
        reasoning_effort=None,
    )

    assert protocol.calls[0][0] == "droid.update_session_settings"
    assert protocol.calls[0][1] == {
        "modelId": "gpt-5.4",
        "interactionMode": "auto",
        "autonomyLevel": "off",
        "reasoningEffort": "low",
    }
    assert protocol.calls[1][1] == {
        "modelId": "glm-5.2",
        "interactionMode": "auto",
        "autonomyLevel": "off",
    }


@pytest.mark.asyncio
async def test_disable_native_tools_verifies_exec_and_mcp_catalogs() -> None:
    disabled: set[str] = set()

    def handler(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "droid.list_mcp_servers":
            return {"result": {"servers": []}}
        if method == "droid.list_tools":
            return {
                "result": {
                    "tools": [
                        {
                            "id": "read-cli",
                            "currentlyAllowed": "read-cli" not in disabled,
                        },
                        {"id": "exit-spec-mode", "currentlyAllowed": True},
                    ]
                }
            }
        if method == "droid.update_session_settings":
            disabled.update(params["disabledToolIds"])
            return {"result": {}}
        raise AssertionError(method)

    protocol = FakeProtocol(handler)

    await DroidRpcExtension().disable_native_tools(_client(protocol))

    update = next(params for method, params, _ in protocol.calls if "update" in method)
    assert update["interactionMode"] == "auto"
    assert update["disabledToolIds"] == ["exit-spec-mode", "read-cli"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tools", "message"),
    [
        ([], "empty native tool catalog"),
        ("not-a-list", "'tools' is malformed"),
        ([{"id": 3, "currentlyAllowed": True}], "'id' is malformed"),
        ([{"id": "read-cli", "currentlyAllowed": "yes"}], "'currentlyAllowed' is malformed"),
    ],
)
async def test_disable_native_tools_rejects_malformed_catalogs(
    tools: object,
    message: str,
) -> None:
    def handler(method: str, _params: dict[str, Any]) -> dict[str, Any]:
        if method == "droid.list_mcp_servers":
            return {"result": {"servers": []}}
        if method == "droid.list_tools":
            return {"result": {"tools": tools}}
        return {"result": {}}

    protocol = FakeProtocol(handler)

    with pytest.raises(DroidClientError, match=message):
        await DroidRpcExtension().disable_native_tools(_client(protocol))


@pytest.mark.asyncio
async def test_disable_native_tools_skips_the_mcp_probe_by_default() -> None:
    protocol = FakeProtocol(_settle_handler())

    await DroidRpcExtension().disable_native_tools(_client(protocol))

    assert [method for method, _, _ in protocol.calls] == [
        "droid.list_tools",
        "droid.update_session_settings",
        "droid.list_tools",
    ]


@pytest.mark.asyncio
async def test_disable_native_tools_bounds_mcp_catalog_wait() -> None:
    protocol = FakeProtocol(_settle_handler())

    await DroidRpcExtension(mcp_settle_seconds=0.15).disable_native_tools(_client(protocol))

    probes = [method for method, _, _ in protocol.calls if method == "droid.list_mcp_servers"]
    assert len(probes) > 1
    assert any(method == "droid.update_session_settings" for method, _, _ in protocol.calls)


@pytest.mark.asyncio
async def test_disable_native_tools_tolerates_mcp_servers_without_status() -> None:
    handler = _settle_handler(servers=[{"name": "local"}])
    protocol = FakeProtocol(handler)

    await DroidRpcExtension(mcp_settle_seconds=5.0).disable_native_tools(_client(protocol))

    probes = [method for method, _, _ in protocol.calls if method == "droid.list_mcp_servers"]
    assert probes == ["droid.list_mcp_servers"]


def _settle_handler(
    servers: list[dict[str, Any]] | None = None,
) -> Callable[[str, dict[str, Any]], dict[str, Any]]:
    resolved_servers = [{"status": "connecting"}] if servers is None else servers
    disabled = False

    def handler(method: str, params: dict[str, Any]) -> dict[str, Any]:
        nonlocal disabled
        if method == "droid.list_mcp_servers":
            return {"result": {"servers": resolved_servers}}
        if method == "droid.list_tools":
            return {
                "result": {
                    "tools": [
                        {"id": "mcp_tool", "currentlyAllowed": not disabled},
                    ]
                }
            }
        if method == "droid.update_session_settings":
            disabled = params["disabledToolIds"] == ["mcp_tool"]
            return {"result": {}}
        raise AssertionError(method)

    return handler


@pytest.mark.asyncio
async def test_context_and_session_operations_parse_rpc_results() -> None:
    def handler(method: str, params: dict[str, Any]) -> dict[str, Any]:
        results: dict[str, dict[str, Any]] = {
            "droid.get_context_stats": {
                "used": 1,
                "remaining": 9,
                "limit": 10,
                "accuracy": "estimated",
                "updatedAt": "now",
            },
            "droid.get_context_breakdown": {
                "modelId": "model",
                "modelDisplayName": "Model",
                "contextBudget": 10,
                "usedTokens": 1,
                "freeTokens": 9,
                "categories": [{"name": "system", "tokens": 1, "colorKey": "gray"}],
            },
            "droid.compact_session": {
                "newSessionId": "compact",
                "removedCount": 2,
            },
            "droid.fork_session": {"newSessionId": "fork"},
            "droid.rename_session": {},
            "droid.close_session": {},
        }
        assert params == (
            {"customInstructions": "Keep facts."}
            if method == "droid.compact_session"
            else {"title": "Title"}
            if method == "droid.rename_session"
            else {"reason": "clear"}
            if method == "droid.close_session"
            else {}
        )
        return {"result": results[method]}

    protocol = FakeProtocol(handler)
    client = _client(protocol)
    extension = DroidRpcExtension()

    stats = await extension.get_context_stats(client)
    breakdown = await extension.get_context_breakdown(client)
    compacted = await extension.compact_session(
        client,
        custom_instructions="Keep facts.",
    )
    forked = await extension.fork_session(client)
    await extension.rename_session(client, title="Title")
    await extension.close_session(client, reason="clear")

    assert stats.remaining == 9
    assert breakdown.categories[0].tokens == 1
    assert compacted.removed_count == 2
    assert forked == "fork"


@pytest.mark.asyncio
async def test_compaction_omits_absent_custom_instructions() -> None:
    protocol = FakeProtocol(
        lambda _method, params: {"result": {"newSessionId": "compact", "removedCount": len(params)}}
    )

    result = await DroidRpcExtension().compact_session(
        _client(protocol),
        custom_instructions=None,
    )

    assert result.removed_count == 0
    assert protocol.calls[0][2] == 300.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        None,
        {"used": True},
        {"used": 1, "remaining": 2, "limit": 3, "accuracy": 4, "updatedAt": "now"},
    ],
)
async def test_rpc_extension_rejects_malformed_results(result: object) -> None:
    protocol = FakeProtocol(lambda _method, _params: {"result": result})

    with pytest.raises(DroidClientError, match="malformed"):
        await DroidRpcExtension().get_context_stats(_client(protocol))


@pytest.mark.asyncio
async def test_rpc_extension_requires_an_sdk_protocol_engine() -> None:
    with pytest.raises(DroidClientError, match="protocol engine"):
        await DroidRpcExtension().fork_session(_client(None))


@pytest.mark.asyncio
async def test_protocol_engine_contract_carries_no_default_behavior() -> None:
    engine: _ProtocolEngine = FakeProtocol(lambda _method, _params: {"result": {}})

    assert not await _ProtocolEngine.send_request(engine, "droid.list_tools", {})
