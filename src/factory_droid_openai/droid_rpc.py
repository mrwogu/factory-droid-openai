from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from droid_sdk.errors import DroidClientError
from droid_sdk.schemas.enums import (
    AutonomyLevel,
    DroidInteractionMode,
    DroidServerMethod,
)

if TYPE_CHECKING:
    from droid_sdk import DroidClient

_RPC_TIMEOUT_SECONDS = 30.0
# Compaction runs a full model turn, so it needs a far larger budget than the
# metadata RPCs. The caller's own timeout still bounds the operation.
_COMPACTION_TIMEOUT_SECONDS = 300.0
_MCP_POLL_SECONDS = 0.1
_UNAVOIDABLE_TOOL_IDS = frozenset({"exit-spec-mode"})


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
        """Repoint an initialized session at another model and effort.

        Droid applies this without restarting the process, so a warm session
        can serve a model it was not initialized with. Interaction mode and
        autonomy level are resent so the bridge invariants stay pinned; the
        disabled native tool list is preserved by Droid.
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

    async def disable_native_tools(self, client: DroidClient) -> None:
        await self._wait_for_mcp_catalog(client)
        tools = await self._list_tools(client)
        if not tools:
            raise DroidClientError("Droid returned an empty native tool catalog")
        tool_ids = sorted({_required_str(tool, "id") for tool in tools})
        await self._request(
            client,
            DroidServerMethod.UPDATE_SESSION_SETTINGS.value,
            {
                "interactionMode": DroidInteractionMode.Auto.value,
                "autonomyLevel": AutonomyLevel.Off.value,
                "enabledToolIds": [],
                "disabledToolIds": tool_ids,
            },
        )
        remaining = {
            _required_str(tool, "id")
            for tool in await self._list_tools(client)
            if _required_bool(tool, "currentlyAllowed")
        }
        unexpected = remaining - _UNAVOIDABLE_TOOL_IDS
        if unexpected:
            rendered = ", ".join(sorted(unexpected))
            raise DroidClientError(f"Failed to disable native Droid tools: {rendered}")

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
