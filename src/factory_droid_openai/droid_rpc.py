from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast
from urllib.parse import urlsplit

from droid_sdk.errors import DroidClientError
from droid_sdk.schemas.enums import (
    AutonomyLevel,
    DroidInteractionMode,
    DroidServerMethod,
)

from factory_droid_openai.mcp_tools import MCP_SERVER_NAME, MCP_TOOL_ID_PREFIX

if TYPE_CHECKING:
    from droid_sdk import DroidClient

_RPC_TIMEOUT_SECONDS = 30.0
# Compaction runs a full model turn, so it needs a far larger budget than the
# metadata RPCs. The caller's own timeout still bounds the operation.
_COMPACTION_TIMEOUT_SECONDS = 300.0
_MCP_POLL_SECONDS = 0.1
_NATIVE_MCP_WAIT_SECONDS = 5.0
_TOOL_DISABLE_RETRIES = 3
_TOOL_DISABLE_RETRY_SECONDS = 0.1
_UNAVOIDABLE_TOOL_IDS = frozenset({"exit-spec-mode"})
# Droid keeps the deferred-tool loader callable in any session that has a tool
# left to load, which is every session that publishes tools over MCP. It only
# fetches a schema Droid already holds, and the model needs it to reach those
# tools at all.
_DEFERRED_TOOL_LOADER_IDS = frozenset({"tool-search-cli"})
_CONNECTED_MCP_STATUS = "connected"
_FAILED_MCP_STATUSES = frozenset({"failed", "disconnected", "disabled"})
_MCP_URL_FIELDS = ("url", "uri")
_DEFAULT_PORTS = {"http": 80, "https": 443}
_MCP_POLICY_PATTERN = re.compile(
    r"(?:mcp\s*policy|allowlist|allow\s+list|organization\s+policy)",
    re.IGNORECASE,
)


class NativeToolUnavailableError(DroidClientError):
    """The bridge's MCP server or exact native tool catalog is unavailable."""


class _ProtocolEngine(Protocol):
    async def send_request(
        self,
        method: str,
        params: dict[str, Any],
        timeout: float | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ContextStats:
    used: int
    remaining: int
    limit: int
    accuracy: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class ContextCategory:
    name: str
    tokens: int
    color_key: str


@dataclass(frozen=True, slots=True)
class ContextBreakdown:
    model_id: str
    model_display_name: str
    context_budget: int
    used_tokens: int
    free_tokens: int
    categories: tuple[ContextCategory, ...]


@dataclass(frozen=True, slots=True)
class CompactionResult:
    new_session_id: str
    removed_count: int


class DroidRpcExtension:
    """Compatibility shim for RPCs not yet exposed by droid-sdk-python."""

    def __init__(self, *, mcp_settle_seconds: float = 0.0) -> None:
        self._mcp_settle_seconds = max(0.0, mcp_settle_seconds)

    async def add_user_message(
        self,
        client: DroidClient,
        *,
        text: str,
        images: list[dict[str, Any]] | None,
        files: list[dict[str, Any]] | None,
        output_format: dict[str, Any] | None,
    ) -> None:
        params: dict[str, Any] = {"text": text}
        if images is not None:
            params["images"] = images
        if files is not None:
            params["files"] = files
        if output_format is not None:
            params["outputFormat"] = output_format
        await self._request(client, DroidServerMethod.ADD_USER_MESSAGE.value, params)

    async def retune_session(
        self,
        client: DroidClient,
        *,
        model_id: str,
        reasoning_effort: str | None,
    ) -> None:
        """Apply a model id and optional reasoning effort to a session.

        Interaction mode and autonomy level are resent so bridge invariants
        stay pinned; the disabled native tool list is preserved by Droid.
        """
        params: dict[str, Any] = {
            "modelId": model_id,
            "interactionMode": DroidInteractionMode.Auto.value,
            "autonomyLevel": AutonomyLevel.Off.value,
        }
        if reasoning_effort is not None:
            params["reasoningEffort"] = reasoning_effort
        await self._request(
            client,
            DroidServerMethod.UPDATE_SESSION_SETTINGS.value,
            params,
        )

    async def disable_native_tools(
        self,
        client: DroidClient,
        *,
        keep_tool_prefix: str | None = None,
        expected_tool_ids: frozenset[str] | None = None,
        native_server_url: str | None = None,
    ) -> None:
        """Disable every Droid tool, optionally sparing one id prefix.

        ``keep_tool_prefix`` leaves the tools the bridge itself published over
        MCP callable. Everything Droid owns still goes away, so a turn can only
        reach tools the OpenAI client asked for.
        """
        if expected_tool_ids is None:
            await self._wait_for_mcp_catalog(client)
        else:
            await self._wait_for_native_mcp_server(client, native_server_url)
        tools = await self._list_tools(client)
        if not tools:
            raise DroidClientError("Droid returned an empty native tool catalog")
        tool_ids = {_required_str(tool, "id") for tool in tools}
        if expected_tool_ids is not None:
            self._verify_native_tool_ids(tool_ids, expected_tool_ids)
        tolerated = _UNAVOIDABLE_TOOL_IDS
        if keep_tool_prefix is not None:
            tolerated = tolerated | _DEFERRED_TOOL_LOADER_IDS
        unexpected: set[str] = set()
        missing_expected: set[str] = set()
        for attempt in range(_TOOL_DISABLE_RETRIES):
            kept = (
                set(expected_tool_ids)
                if expected_tool_ids is not None
                else _matching(tool_ids, keep_tool_prefix)
            )
            await self._request(
                client,
                DroidServerMethod.UPDATE_SESSION_SETTINGS.value,
                {
                    "interactionMode": DroidInteractionMode.Auto.value,
                    "autonomyLevel": AutonomyLevel.Off.value,
                    "enabledToolIds": sorted(kept),
                    "disabledToolIds": sorted(tool_ids - kept),
                },
            )
            remaining: set[str] = set()
            for tool in await self._list_tools(client):
                tool_id = _required_str(tool, "id")
                tool_ids.add(tool_id)
                if _required_bool(tool, "currentlyAllowed"):
                    remaining.add(tool_id)
            if expected_tool_ids is not None:
                self._verify_native_tool_ids(tool_ids, expected_tool_ids)
                missing_expected = set(expected_tool_ids - remaining)
            unexpected = remaining - tolerated - _matching(remaining, keep_tool_prefix)
            if not unexpected and not missing_expected:
                return
            if attempt + 1 < _TOOL_DISABLE_RETRIES:
                await asyncio.sleep(_TOOL_DISABLE_RETRY_SECONDS)
        if missing_expected:
            raise NativeToolUnavailableError(
                "Droid did not keep every bridge tool enabled: "
                + ", ".join(sorted(missing_expected))
            )
        rendered = ", ".join(sorted(unexpected))
        raise DroidClientError(f"Failed to disable native Droid tools: {rendered}")

    async def _wait_for_native_mcp_server(
        self,
        client: DroidClient,
        server_url: str | None,
    ) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _NATIVE_MCP_WAIT_SECONDS
        last_status: str | None = None
        while True:
            timeout = min(_RPC_TIMEOUT_SECONDS, max(0.1, deadline - loop.time()))
            result = await self._request(
                client,
                DroidServerMethod.LIST_MCP_SERVERS.value,
                {},
                timeout=timeout,
            )
            servers = _required_list(result, "servers")
            target = [server for server in servers if _native_server_matches(server, server_url)]
            if len(target) > 1:
                raise NativeToolUnavailableError(
                    f"Droid reported multiple MCP servers named '{MCP_SERVER_NAME}': "
                    + ", ".join(sorted(_reported_endpoint(server) for server in target))
                )
            if target:
                server = target[0]
                last_status = _state_value(server.get("status"))
                if last_status == _CONNECTED_MCP_STATUS:
                    return
                if last_status in _FAILED_MCP_STATUSES:
                    raise _native_server_error(server, server_url)
                # Every other status, including one a newer Droid introduces,
                # is a reason to keep polling: only the deadline decides, so a
                # renamed "connecting" cannot fail an otherwise healthy turn.
            if loop.time() >= deadline:
                hostname = _server_hostname(server_url)
                seen = "" if last_status is None else f", last status {last_status!r}"
                raise NativeToolUnavailableError(
                    f"Droid MCP server '{MCP_SERVER_NAME}' did not connect at {hostname} "
                    f"within {_NATIVE_MCP_WAIT_SECONDS:.1f} seconds{seen}"
                )
            await asyncio.sleep(_MCP_POLL_SECONDS)

    @staticmethod
    def _verify_native_tool_ids(
        tool_ids: set[str],
        expected_tool_ids: frozenset[str],
    ) -> None:
        missing = expected_tool_ids - tool_ids
        unexpected = _matching(tool_ids, MCP_TOOL_ID_PREFIX) - expected_tool_ids
        if not missing and not unexpected:
            return
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unexpected:
            details.append("unexpected " + ", ".join(sorted(unexpected)))
        raise NativeToolUnavailableError(
            "Droid MCP tool catalog does not match bridge tools: " + "; ".join(details)
        )

    async def _wait_for_mcp_catalog(self, client: DroidClient) -> None:
        # Off by default: MCP servers that never leave "connecting" would
        # otherwise burn the whole window on every turn, and the catalog can
        # still grow afterwards. Droid registers tools disallowed by default,
        # so waiting buys a larger verification snapshot, not the guarantee.
        if self._mcp_settle_seconds <= 0:
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._mcp_settle_seconds
        while True:
            result = await self._request(client, DroidServerMethod.LIST_MCP_SERVERS.value, {})
            servers = _required_list(result, "servers")
            if not any(server.get("status") == "connecting" for server in servers):
                return
            if loop.time() >= deadline:
                return
            await asyncio.sleep(_MCP_POLL_SECONDS)

    async def get_context_stats(self, client: DroidClient) -> ContextStats:
        result = await self._request(client, "droid.get_context_stats", {})
        return ContextStats(
            used=_required_int(result, "used"),
            remaining=_required_int(result, "remaining"),
            limit=_required_int(result, "limit"),
            accuracy=_required_str(result, "accuracy"),
            updated_at=_required_str(result, "updatedAt"),
        )

    async def get_context_breakdown(self, client: DroidClient) -> ContextBreakdown:
        result = await self._request(client, "droid.get_context_breakdown", {})
        categories = tuple(
            ContextCategory(
                name=_required_str(item, "name"),
                tokens=_required_int(item, "tokens"),
                color_key=_required_str(item, "colorKey"),
            )
            for item in _required_list(result, "categories")
        )
        return ContextBreakdown(
            model_id=_required_str(result, "modelId"),
            model_display_name=_required_str(result, "modelDisplayName"),
            context_budget=_required_int(result, "contextBudget"),
            used_tokens=_required_int(result, "usedTokens"),
            free_tokens=_required_int(result, "freeTokens"),
            categories=categories,
        )

    async def compact_session(
        self,
        client: DroidClient,
        *,
        custom_instructions: str | None,
    ) -> CompactionResult:
        params = (
            {"customInstructions": custom_instructions} if custom_instructions is not None else {}
        )
        result = await self._request(
            client,
            "droid.compact_session",
            params,
            timeout=_COMPACTION_TIMEOUT_SECONDS,
        )
        return CompactionResult(
            new_session_id=_required_str(result, "newSessionId"),
            removed_count=_required_int(result, "removedCount"),
        )

    async def fork_session(self, client: DroidClient) -> str:
        result = await self._request(client, "droid.fork_session", {})
        return _required_str(result, "newSessionId")

    async def rename_session(self, client: DroidClient, *, title: str) -> None:
        await self._request(client, "droid.rename_session", {"title": title})

    async def close_session(self, client: DroidClient, *, reason: str = "other") -> None:
        await self._request(client, "droid.close_session", {"reason": reason})

    async def _list_tools(self, client: DroidClient) -> list[dict[str, Any]]:
        result = await self._request(client, "droid.list_tools", {})
        return _required_list(result, "tools")

    async def _request(
        self,
        client: DroidClient,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float = _RPC_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        response = await _protocol(client).send_request(
            method=method,
            params=params,
            timeout=timeout,
        )
        result = response.get("result")
        if not isinstance(result, dict):
            raise DroidClientError(f"{method} returned a malformed result")
        return result


def _matching(tool_ids: set[str], prefix: str | None) -> set[str]:
    if prefix is None:
        return set()
    return {tool_id for tool_id in tool_ids if tool_id.startswith(prefix)}


def _native_server_matches(server: dict[str, Any], server_url: str | None) -> bool:
    if server.get("name") != MCP_SERVER_NAME:
        return False
    reported = [server[field] for field in _MCP_URL_FIELDS if field in server]
    if not reported:
        # Droid may list a server without echoing its URL, and the name is then
        # all there is to match on.
        return True
    return all(_same_endpoint(value, server_url) for value in reported)


def _reported_endpoint(server: dict[str, Any]) -> str:
    for field in _MCP_URL_FIELDS:
        value = server.get(field)
        if isinstance(value, str) and value:
            return value
    return "no URL reported"


def _same_endpoint(reported: object, configured: str | None) -> bool:
    """Compare two URLs by endpoint identity rather than by spelling."""
    if configured is None or not isinstance(reported, str):
        return False
    left = _endpoint_identity(reported)
    right = _endpoint_identity(configured)
    return left is not None and left == right


def _endpoint_identity(url: str) -> tuple[str, str, int | None, str] | None:
    parts = urlsplit(url)
    try:
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    return (
        scheme,
        (parts.hostname or "").lower(),
        port if port is not None else _DEFAULT_PORTS.get(scheme),
        parts.path.rstrip("/"),
    )


def _native_server_error(
    server: dict[str, Any],
    server_url: str | None,
) -> NativeToolUnavailableError:
    detail = server.get("error")
    message = detail if isinstance(detail, str) and detail else "connection failed"
    hostname = _server_hostname(server_url)
    if _MCP_POLICY_PATTERN.search(message):
        return NativeToolUnavailableError(
            f"Factory Droid MCP server '{MCP_SERVER_NAME}' was rejected by mcpPolicy. "
            f"Allow hostname '{hostname}' in the managed MCP allowlist: {message}"
        )
    return NativeToolUnavailableError(
        f"Factory Droid MCP server '{MCP_SERVER_NAME}' failed at {hostname}: {message}"
    )


def _server_hostname(server_url: str | None) -> str:
    if server_url is None:
        return "the configured native tool URL"
    return urlsplit(server_url).hostname or "the configured native tool URL"


def _state_value(value: object) -> str:
    return str(getattr(value, "value", value))


def _protocol(client: DroidClient) -> _ProtocolEngine:
    protocol = getattr(client, "_protocol", None)
    if protocol is None or not callable(getattr(protocol, "send_request", None)):
        raise DroidClientError("Droid SDK protocol engine is unavailable")
    return cast("_ProtocolEngine", protocol)


def _required_list(payload: dict[str, Any], field: str) -> list[dict[str, Any]]:
    value = payload.get(field)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise DroidClientError(f"Droid RPC field '{field}' is malformed")
    return cast("list[dict[str, Any]]", value)


def _required_str(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise DroidClientError(f"Droid RPC field '{field}' is malformed")
    return value


def _required_int(payload: dict[str, Any], field: str) -> int:
    value = payload.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise DroidClientError(f"Droid RPC field '{field}' is malformed")
    return value


def _required_bool(payload: dict[str, Any], field: str) -> bool:
    value = payload.get(field)
    if not isinstance(value, bool):
        raise DroidClientError(f"Droid RPC field '{field}' is malformed")
    return value
