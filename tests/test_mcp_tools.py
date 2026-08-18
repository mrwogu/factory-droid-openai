from __future__ import annotations

from typing import Any, cast

import httpx
import pytest
from fastapi import FastAPI

import factory_droid_openai.mcp_tools as mcp_tools
from factory_droid_openai.mcp_tools import (
    MCP_PROTOCOL_VERSION,
    MCP_ROUTE_PREFIX,
    MCP_SERVER_NAME,
    NativeCatalogUnavailableError,
    NativeToolCatalog,
    NativeToolRegistry,
    build_router,
    handle_rpc,
    to_mcp_tools,
)
from factory_droid_openai.models import ToolDefinition


def _tool(**overrides: Any) -> ToolDefinition:
    function: dict[str, Any] = {
        "name": "get_weather",
        "description": "Return the weather.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    }
    function.update(overrides)
    return ToolDefinition(type="function", function=cast("Any", function))


def _registry(**overrides: Any) -> NativeToolRegistry:
    options: dict[str, Any] = {"base_url": "http://127.0.0.1:8787"}
    options.update(overrides)
    return NativeToolRegistry(**options)


def _catalog() -> tuple[dict[str, Any], ...]:
    return to_mcp_tools([_tool()])


def _app(registry: NativeToolRegistry) -> FastAPI:
    application = FastAPI()
    application.include_router(build_router(registry))
    return application


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


def test_registry_publishes_a_catalog_under_a_single_use_token() -> None:
    registry = _registry()

    binding = registry.open(_catalog())

    assert binding.url == f"http://127.0.0.1:8787{MCP_ROUTE_PREFIX}/{binding.token}"
    assert registry.catalog(binding.token) == _catalog()
    assert len(registry) == 1
    assert binding.server_config() == {
        "name": MCP_SERVER_NAME,
        "type": "http",
        "url": binding.url,
    }


def test_registry_trims_a_trailing_slash_from_the_base_url() -> None:
    registry = _registry(base_url="http://127.0.0.1:8787/")

    binding = registry.open(())

    assert binding.url == f"http://127.0.0.1:8787{MCP_ROUTE_PREFIX}/{binding.token}"


def test_registry_forgets_a_closed_catalog() -> None:
    registry = _registry()
    binding = registry.open(_catalog())

    registry.pin("missing")
    registry.close(binding.token)
    registry.close(binding.token)

    assert registry.catalog(binding.token) is None
    assert len(registry) == 0


def test_registry_evicts_the_oldest_catalog_at_the_cap() -> None:
    registry = _registry(max_sessions=1)
    first = registry.open(_catalog())

    second = registry.open(_catalog())

    assert registry.catalog(first.token) is None
    assert registry.catalog(second.token) == _catalog()


def test_registry_copies_the_published_tools() -> None:
    registry = _registry()
    tool: dict[str, Any] = {"name": "get_weather", "inputSchema": {"type": "object"}}

    binding = registry.open([tool])
    tool["name"] = "mutated"

    catalog = registry.catalog(binding.token)
    assert catalog is not None
    assert catalog[0]["name"] == "get_weather"


def test_catalog_identity_includes_order_and_every_descriptor_field() -> None:
    registry = _registry()
    first = registry.catalog_identity(
        (
            {"name": "a", "description": "one", "inputSchema": {"type": "object"}},
            {"name": "b", "inputSchema": {"type": "string"}},
        )
    )
    same = registry.catalog_identity(
        (
            {"name": "a", "description": "one", "inputSchema": {"type": "object"}},
            {"name": "b", "inputSchema": {"type": "string"}},
        )
    )
    changed = registry.catalog_identity(
        (
            {"name": "b", "inputSchema": {"type": "string"}},
            {"name": "a", "description": "two", "inputSchema": {"type": "object"}},
        )
    )

    assert isinstance(first, NativeToolCatalog)
    assert first == same
    assert hash(first) == hash(same)
    assert first != changed
    assert first.names == frozenset({"a", "b"})


def test_catalog_serves_the_tools_in_the_order_the_client_sent_them() -> None:
    """The serialization is what the model reads, so it must not be re-keyed."""
    registry = _registry()
    schema = {"type": "object", "properties": {"zone": {"type": "string"}, "city": {}}}

    binding = registry.open(({"inputSchema": schema, "name": "get_weather"},))

    served = registry.catalog(binding.token)
    assert served is not None
    assert list(served[0]) == ["inputSchema", "name"]
    assert list(served[0]["inputSchema"]["properties"]) == ["zone", "city"]


def test_catalog_tools_are_fresh_copies() -> None:
    catalog = _registry().catalog_identity(_catalog())

    first = catalog.tools
    first[0]["name"] = "mutated"

    assert catalog.tools[0]["name"] == "get_weather"


def test_catalog_rejects_malformed_serialized_tools() -> None:
    catalog = NativeToolCatalog(serialized="{}", names=frozenset())

    with pytest.raises(NativeCatalogUnavailableError, match="serialization is malformed"):
        _ = catalog.tools


def test_catalog_identity_rejects_malformed_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = cast("Any", mcp_tools)
    monkeypatch.setattr(module.__dict__["json"], "loads", lambda _serialized: {})

    with pytest.raises(NativeCatalogUnavailableError, match="serialization is malformed"):
        _registry().catalog_identity(())


def test_pinned_catalogs_survive_eviction_until_closed() -> None:
    registry = _registry(max_sessions=2)
    first = registry.open(_catalog())
    second = registry.open(())
    registry.pin(first.token)

    third = registry.open(({"name": "third", "inputSchema": {}},))

    assert registry.catalog(first.token) == _catalog()
    assert registry.catalog(second.token) is None
    assert registry.catalog(third.token) is not None

    registry.close(first.token)
    fourth = registry.open(({"name": "fourth", "inputSchema": {}},))
    assert registry.catalog(first.token) is None
    assert registry.catalog(fourth.token) is not None


def test_registry_rejects_opening_when_all_catalogs_are_pinned() -> None:
    registry = _registry(max_sessions=1)
    first = registry.open(_catalog())
    registry.pin(first.token)

    with pytest.raises(NativeCatalogUnavailableError, match="capacity is exhausted"):
        registry.open(())


def test_openai_tools_become_mcp_descriptors() -> None:
    published = to_mcp_tools([_tool(), _tool(name="ping", description="", parameters={})])

    assert published == (
        {
            "name": "get_weather",
            "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}},
            "description": "Return the weather.",
        },
        {"name": "ping", "inputSchema": {"type": "object"}},
    )


@pytest.mark.parametrize(
    ("tool_name", "expected"),
    [
        # Every spelling Droid uses for a published tool resolves to the name
        # the OpenAI client knows.
        (f"{MCP_SERVER_NAME}___get_weather", "get_weather"),
        (f"mcp_{MCP_SERVER_NAME}_get_weather", "get_weather"),
        ("get_weather", "get_weather"),
        (f"{MCP_SERVER_NAME}___", None),
        (f"{MCP_SERVER_NAME}___read-cli", None),
        ("other-server___get_weather", None),
        ("read-cli", None),
    ],
)
def test_a_binding_resolves_only_the_tools_it_published(
    tool_name: str, expected: str | None
) -> None:
    binding = NativeToolRegistry(base_url="http://127.0.0.1:8787").open(_catalog())

    assert binding.resolve(tool_name) == expected


def test_initialize_answers_with_the_negotiated_protocol() -> None:
    status, body = handle_rpc(_catalog(), {"jsonrpc": "2.0", "id": 0, "method": "initialize"})

    assert status == 200
    assert body is not None
    assert body["result"]["protocolVersion"] == MCP_PROTOCOL_VERSION
    assert body["result"]["serverInfo"]["name"] == MCP_SERVER_NAME
    assert body["result"]["capabilities"] == {"tools": {"listChanged": False}}


def test_tools_list_serves_the_request_catalog() -> None:
    status, body = handle_rpc(_catalog(), {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})

    assert status == 200
    assert body is not None
    assert body["result"] == {"tools": list(_catalog())}


def test_ping_is_answered_with_an_empty_result() -> None:
    status, body = handle_rpc((), {"jsonrpc": "2.0", "id": 2, "method": "ping"})

    assert (status, body) == (200, {"jsonrpc": "2.0", "id": 2, "result": {}})


def test_a_notification_is_acknowledged_without_a_body() -> None:
    status, body = handle_rpc((), {"jsonrpc": "2.0", "method": "notifications/initialized"})

    assert (status, body) == (202, None)


def test_a_tool_call_is_refused_because_the_client_runs_the_tool() -> None:
    status, body = handle_rpc(
        _catalog(),
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "get_weather", "arguments": {"city": "Gdansk"}},
        },
    )

    assert status == 200
    assert body is not None
    assert body["result"]["isError"] is True
    assert "OpenAI client" in body["result"]["content"][0]["text"]


def test_an_unsupported_method_is_reported_as_such() -> None:
    status, body = handle_rpc((), {"jsonrpc": "2.0", "id": 4, "method": "resources/list"})

    assert status == 200
    assert body is not None
    assert body["error"]["code"] == -32601


@pytest.mark.parametrize("message", [["not", "an", "object"], {"jsonrpc": "2.0", "id": 5}])
def test_a_malformed_request_is_rejected(message: Any) -> None:
    status, body = handle_rpc((), message)

    assert status == 400
    assert body is not None
    assert body["error"]["code"] == -32600


@pytest.mark.asyncio
async def test_endpoint_serves_the_catalog_of_an_open_request() -> None:
    registry = _registry()
    binding = registry.open(_catalog())

    async with _client(_app(registry)) as client:
        response = await client.post(
            f"{MCP_ROUTE_PREFIX}/{binding.token}",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        )

    assert response.status_code == 200
    assert response.headers["mcp-session-id"] == binding.token
    assert response.json()["result"]["tools"][0]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_endpoint_answers_a_notification_with_an_empty_body() -> None:
    registry = _registry()
    binding = registry.open(())

    async with _client(_app(registry)) as client:
        response = await client.post(
            f"{MCP_ROUTE_PREFIX}/{binding.token}",
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )

    assert response.status_code == 202
    assert response.content == b""


@pytest.mark.asyncio
async def test_endpoint_rejects_an_unknown_token() -> None:
    registry = _registry()

    async with _client(_app(registry)) as client:
        response = await client.post(
            f"{MCP_ROUTE_PREFIX}/nope",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        )

    assert response.status_code == 404
    assert response.json() == {"error": "unknown MCP session"}


@pytest.mark.asyncio
async def test_endpoint_rejects_a_body_that_is_not_json() -> None:
    registry = _registry()
    binding = registry.open(())

    async with _client(_app(registry)) as client:
        response = await client.post(
            f"{MCP_ROUTE_PREFIX}/{binding.token}",
            content=b"{not json",
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == -32600


@pytest.mark.asyncio
async def test_endpoint_refuses_the_event_stream_and_accepts_teardown() -> None:
    registry = _registry()
    binding = registry.open(())

    async with _client(_app(registry)) as client:
        stream = await client.get(f"{MCP_ROUTE_PREFIX}/{binding.token}")
        teardown = await client.delete(f"{MCP_ROUTE_PREFIX}/{binding.token}")

    assert stream.status_code == 405
    assert teardown.status_code == 204
