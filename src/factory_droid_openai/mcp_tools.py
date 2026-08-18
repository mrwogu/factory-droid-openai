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

import hashlib
import json
import secrets
from collections import OrderedDict
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
class NativeToolCatalog:
    """Immutable identity and payload for one ordered MCP catalog."""

    serialized: str
    fingerprint: str
    names: frozenset[str]

    @property
    def tools(self) -> tuple[dict[str, Any], ...]:
        """Return a fresh catalog copy for an SDK or HTTP response."""
        decoded = json.loads(self.serialized)
        if not isinstance(decoded, list) or not all(isinstance(item, dict) for item in decoded):
            raise RuntimeError("native tool catalog serialization is malformed")
        return tuple(decoded)


@dataclass(frozen=True, slots=True)
class NativeToolBinding:
    """What one request needs to reach its own MCP endpoint."""

    token: str
    url: str
    names: frozenset[str]
    catalog: NativeToolCatalog | None = None

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
        self._catalogs: OrderedDict[str, NativeToolCatalog] = OrderedDict()
        self._pinned: set[str] = set()

    def open(self, tools: Sequence[Mapping[str, Any]]) -> NativeToolBinding:
        """Publish ``tools`` under a fresh token and return how to reach them."""
        return self.open_catalog(self.catalog_identity(tools))

    def catalog_identity(self, tools: Sequence[Mapping[str, Any]]) -> NativeToolCatalog:
        """Canonicalize a request catalog without allocating a credential."""
        return _catalog(tools)

    def open_catalog(self, catalog: NativeToolCatalog) -> NativeToolBinding:
        """Publish an existing catalog under a fresh token."""
        token = secrets.token_urlsafe(24)
        self._evict_unpinned()
        if len(self._catalogs) >= self._max_sessions:
            raise RuntimeError("native tool catalog capacity is exhausted")
        self._catalogs[token] = catalog
        return NativeToolBinding(
            token=token,
            url=f"{self._base_url}{MCP_ROUTE_PREFIX}/{token}",
            names=catalog.names,
            catalog=catalog,
        )

    def close(self, token: str) -> None:
        self._catalogs.pop(token, None)
        self._pinned.discard(token)

    def pin(self, token: str) -> None:
        if token in self._catalogs:
            self._pinned.add(token)

    def unpin(self, token: str) -> None:
        self._pinned.discard(token)

    def catalog(self, token: str) -> tuple[dict[str, Any], ...] | None:
        catalog = self._catalogs.get(token)
        return None if catalog is None else catalog.tools

    def __len__(self) -> int:
        return len(self._catalogs)

    def _evict_unpinned(self) -> None:
        while len(self._catalogs) >= self._max_sessions:
            token = next((token for token in self._catalogs if token not in self._pinned), None)
            if token is None:
                return
            self._catalogs.pop(token)


def _catalog(tools: Sequence[Mapping[str, Any]]) -> NativeToolCatalog:
    serialized = json.dumps(
        [dict(tool) for tool in tools],
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    decoded = json.loads(serialized)
    if not isinstance(decoded, list) or not all(isinstance(item, dict) for item in decoded):
        raise RuntimeError("native tool catalog serialization is malformed")
    names = frozenset(str(tool["name"]) for tool in decoded)
    return NativeToolCatalog(
        serialized=serialized,
        fingerprint=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        names=names,
    )


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
