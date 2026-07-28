from __future__ import annotations

from typing import Annotated, Any, Literal

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


class HealthResponse(BaseModel):
    status: Literal["ok"]


class VersionResponse(BaseModel):
    version: str


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str
    factory_droid_display_name: str | None = None
    factory_droid_supported_reasoning_efforts: list[str] | None = None
    factory_droid_default_reasoning_effort: str | None = None
    factory_droid_supports_images: bool | None = None
    factory_droid_supports_pdfs: bool | None = None


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


class UsageResponse(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: dict[str, int]


class AssistantMessageResponse(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None
    reasoning: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: AssistantMessageResponse
    finish_reason: Literal["stop", "tool_calls"]


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageResponse
    factory_droid_session_id: str | None = None


class ErrorDetail(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class JsonSchemaDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    description: str | None = None
    schema_: dict[str, Any] = Field(alias="schema")
    strict: bool | None = None


class JsonSchemaResponseFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["json_schema"]
    json_schema: JsonSchemaDefinition


class JsonObjectResponseFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["json_object"]


ResponseFormat = Annotated[
    JsonSchemaResponseFormat | JsonObjectResponseFormat,
    Field(discriminator="type"),
]


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
    n: int = Field(default=1, ge=1)
    stop: str | list[str] | None = None
    parallel_tool_calls: bool = True
    factory_droid_session_id: str | None = None
    factory_droid_status: bool = False
    response_format: ResponseFormat | None = None

    @model_validator(mode="after")
    def validate_messages(self) -> ChatCompletionRequest:
        if not self.messages:
            raise ValueError("messages must contain at least one message")
        return self

    @property
    def stop_sequences(self) -> tuple[str, ...]:
        if self.stop is None:
            return ()
        values = [self.stop] if isinstance(self.stop, str) else self.stop
        return tuple(value for value in values if value)


class ContextStatsResponse(BaseModel):
    used: int
    remaining: int
    limit: int
    accuracy: str
    updated_at: str


class ContextCategoryResponse(BaseModel):
    name: str
    tokens: int
    color_key: str


class ContextBreakdownResponse(BaseModel):
    model_id: str
    model_display_name: str
    context_budget: int
    used_tokens: int
    free_tokens: int
    categories: list[ContextCategoryResponse]


class SessionContextResponse(BaseModel):
    session_id: str
    stats: ContextStatsResponse
    breakdown: ContextBreakdownResponse


class CompactSessionRequest(BaseModel):
    custom_instructions: str | None = Field(default=None, max_length=16_384)


class CompactSessionResponse(BaseModel):
    session_id: str
    removed_count: int


class ForkSessionResponse(BaseModel):
    session_id: str


class RenameSessionRequest(BaseModel):
    title: str = Field(min_length=1, max_length=256)


class SessionOperationResponse(BaseModel):
    session_id: str
    status: Literal["renamed", "closed"]
