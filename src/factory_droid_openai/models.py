from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FunctionCall(BaseModel):
    name: str
    arguments: str = "{}"


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None
    reasoning: str | None = None
    reasoning_content: str | None = None


class ToolFunction(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    type: Literal["function"]
    function: ToolFunction


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    tools: list[ToolDefinition] | None = None
    tool_choice: Any = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    reasoning_effort: str | None = None
    factory_droid_reasoning_effort: str | None = None
    timeout: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_messages(self) -> ChatCompletionRequest:
        if not self.messages:
            raise ValueError("messages must contain at least one message")
        return self
