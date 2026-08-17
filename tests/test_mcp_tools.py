from __future__ import annotations

from typing import Any, cast

import httpx
import pytest
from fastapi import FastAPI

from factory_droid_openai.mcp_tools import (
    MCP_PROTOCOL_VERSION,
    MCP_ROUTE_PREFIX,
    MCP_SERVER_NAME,
    NativeToolRegistry,
    build_router,
    handle_rpc,
    is_bridge_tool_id,
    strip_tool_prefix,
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
        (f"{MCP_SERVER_NAME}___get_weather", "get_weather"),
        (f"{MCP_SERVER_NAME}___", None),
        ("get_weather", None),
        ("other-server___get_weather", None),
    ],
)
def test_only_the_bridge_prefix_names_a_published_tool(
    tool_name: str, expected: str | None
) -> None:
    assert strip_tool_prefix(tool_name) == expected


def test_only_the_bridge_prefix_names_a_published_tool_id() -> None:
    assert is_bridge_tool_id(f"mcp_{MCP_SERVER_NAME}_get_weather")
    assert not is_bridge_tool_id("read-cli")


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
