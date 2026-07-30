from __future__ import annotations


class ProtocolError(ValueError):
    pass


class RequestTooLargeError(ProtocolError):
    pass


class IncompleteToolCallError(ProtocolError):
    """The turn ended inside a tool-call payload that never reached its close marker.

    Carries the guessed tool name and captured size so logs and clients can
    tell what was truncated without re-parsing the raw payload.
    """

    def __init__(self, tool_name: str | None, payload_bytes: int) -> None:
        self.tool_name = tool_name
        self.payload_bytes = payload_bytes
        detail = f"tool '{tool_name}'" if tool_name else "unknown tool"
        super().__init__(
            f"incomplete tool-call marker ({detail}, "
            f"{payload_bytes} bytes captured before the stream ended)"
        )
