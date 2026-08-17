"""Publish client tools to Droid natively over an in-process MCP endpoint.

By default the bridge asks the model to write tool calls as text and recovers
them from the answer, which means tolerating every dialect a model invents.
This module offers the other route: the tools a request carries are served to
Droid as an MCP server, so Droid renders them through the model's own tool slot
and reports each call as a structured event. Nothing has to be parsed out of
free text.

The endpoint holds no conversation state. A single-use token in the URL binds
one MCP server to one chat completion, and the tool catalog is all the endpoint
serves: the OpenAI client, never the bridge, executes the tools, so a call is
answered with a refusal and the bridge reads the call itself off the session's
events instead.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from factory_droid_openai.logs import debug as log_debug

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

# Droid derives both public names of an MCP tool from the server name: the
# model sees "<server>___<tool>" and session settings key the same tool as
# "mcp_<server>_<tool>".
MCP_SERVER_NAME: Final = "openai-bridge"
MCP_TOOL_PREFIX: Final = f"{MCP_SERVER_NAME}___"
MCP_TOOL_ID_PREFIX: Final = f"mcp_{MCP_SERVER_NAME}_"
# Latest revision Droid's client negotiates down to without complaint. Droid
# offers a newer one and accepts this answer.
MCP_PROTOCOL_VERSION: Final = "2025-06-18"
MCP_ROUTE_PREFIX: Final = "/factory/mcp"
# Answered to a tool call that reaches the endpoint anyway, which only happens
# when a Droid setting pre-approves MCP tools and skips the permission gate.
_CALL_REFUSAL: Final = (
    "This tool is executed by the OpenAI client, not by Factory Droid. "
    "The call has been reported to the client."
)
_JSONRPC_VERSION: Final = "2.0"
_METHOD_NOT_FOUND: Final = -32601
_INVALID_REQUEST: Final = -32600


@dataclass(frozen=True, slots=True)
class NativeToolBinding:
    """What one request needs to reach its own MCP endpoint."""

    token: str
    url: str
    names: frozenset[str]

    def server_config(self) -> dict[str, Any]:
        """Render the MCP server entry Droid's session initializer expects."""
        return {"name": MCP_SERVER_NAME, "type": "http", "url": self.url}

    def resolve(self, tool_name: str) -> str | None:
        """Return the published tool an event names, or ``None`` for a foreign one.

        Droid spells one MCP tool three ways: the model calls it
        ``<server>___<tool>``, session settings key it ``mcp_<server>_<tool>``,
        and a tool reached through the deferred-tool loader is reported under
        its own bare name. Membership in this request's catalog is what makes
        the bare spelling safe to honour.
        """
        for prefix in (MCP_TOOL_PREFIX, MCP_TOOL_ID_PREFIX):
            if tool_name.startswith(prefix):
                tool_name = tool_name[len(prefix) :]
                break
        return tool_name if tool_name in self.names else None


class NativeToolRegistry:
    """Tool catalogs of in-flight requests, keyed by single-use token."""

    def __init__(self, *, base_url: str, max_sessions: int = 256) -> None:
        self._base_url = base_url.rstrip("/")
        self._max_sessions = max_sessions
        self._catalogs: dict[str, tuple[dict[str, Any], ...]] = {}

    def open(self, tools: Sequence[Mapping[str, Any]]) -> NativeToolBinding:
        """Publish ``tools`` under a fresh token and return how to reach them."""
        token = secrets.token_urlsafe(24)
        while len(self._catalogs) >= self._max_sessions:
            # A request that never closed its catalog must not pin memory for
            # the life of the process. The oldest entry belongs to the oldest
            # request, whose turn is over by the time the cap is reached.
            self._catalogs.pop(next(iter(self._catalogs)))
        catalog = tuple(dict(tool) for tool in tools)
        self._catalogs[token] = catalog
        return NativeToolBinding(
            token=token,
            url=f"{self._base_url}{MCP_ROUTE_PREFIX}/{token}",
            names=frozenset(str(tool["name"]) for tool in catalog),
        )

    def close(self, token: str) -> None:
        self._catalogs.pop(token, None)

    def catalog(self, token: str) -> tuple[dict[str, Any], ...] | None:
        return self._catalogs.get(token)

    def __len__(self) -> int:
        return len(self._catalogs)


def to_mcp_tools(tools: Iterable[Any]) -> tuple[dict[str, Any], ...]:
    """Convert OpenAI tool definitions into MCP tool descriptors."""
    published: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.function
        # MCP requires an object schema; OpenAI allows a tool with none, which
        # means "no arguments".
        schema = dict(function.parameters) if function.parameters else {"type": "object"}
        entry: dict[str, Any] = {"name": function.name, "inputSchema": schema}
        if function.description:
            entry["description"] = function.description
        published.append(entry)
    return tuple(published)


def handle_rpc(
    catalog: tuple[dict[str, Any], ...],
    message: Any,
) -> tuple[int, dict[str, Any] | None]:
    """Answer one JSON-RPC message, as ``(status_code, body)``.

    A body of ``None`` means the message was a notification, which the MCP
    transport acknowledges with an empty 202.
    """
    if not isinstance(message, dict):
        return 400, _error(None, _INVALID_REQUEST, "request must be a JSON-RPC object")
    method = message.get("method")
    call_id = message.get("id")
    if not isinstance(method, str):
        return 400, _error(call_id, _INVALID_REQUEST, "request is missing a method")
    if method.startswith("notifications/"):
        return 202, None
    if method == "initialize":
        return 200, _result(
            call_id,
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": MCP_SERVER_NAME, "version": "1"},
            },
        )
    if method == "ping":
        return 200, _result(call_id, {})
    if method == "tools/list":
        return 200, _result(call_id, {"tools": list(catalog)})
    if method == "tools/call":
        return 200, _result(
            call_id,
            {"content": [{"type": "text", "text": _CALL_REFUSAL}], "isError": True},
        )
    return 200, _error(call_id, _METHOD_NOT_FOUND, f"unsupported method {method}")


def build_router(registry: NativeToolRegistry) -> APIRouter:
    """Serve every open tool catalog under its own token."""
    router = APIRouter(include_in_schema=False)
    path = f"{MCP_ROUTE_PREFIX}/{{token}}"

    @router.post(path)
    async def rpc(token: str, request: Request) -> Response:
        catalog = registry.catalog(token)
        if catalog is None:
            return JSONResponse({"error": "unknown MCP session"}, status_code=404)
        try:
            message = await request.json()
        except ValueError:
            return JSONResponse(
                _error(None, _INVALID_REQUEST, "request body is not JSON"),
                status_code=400,
            )
        status, body = handle_rpc(catalog, message)
        log_debug(
            "mcp.request",
            method=message.get("method") if isinstance(message, dict) else None,
            status=status,
            tools=len(catalog),
        )
        if body is None:
            return Response(status_code=status)
        # Droid keys follow-up requests by this header, and a stateless
        # endpoint can hand back the token it already routed on.
        return JSONResponse(body, status_code=status, headers={"Mcp-Session-Id": token})

    @router.get(path)
    async def stream() -> Response:
        # The bridge never pushes server-initiated messages, and the MCP spec
        # lets such a server refuse the event stream. Droid carries on.
        return Response(status_code=405)

    @router.delete(path)
    async def terminate() -> Response:
        # The request that opened the catalog closes it, so a client-side
        # teardown has nothing left to release.
        return Response(status_code=204)

    return router


def _result(call_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": _JSONRPC_VERSION, "id": call_id, "result": result}


def _error(call_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": _JSONRPC_VERSION, "id": call_id, "error": {"code": code, "message": message}}
