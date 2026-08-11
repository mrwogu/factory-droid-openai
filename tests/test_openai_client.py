from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

import httpx
import pytest
from openai import APIStatusError, AsyncOpenAI

from factory_droid_openai.app import create_app
from factory_droid_openai.config import Settings
from factory_droid_openai.protocol import TOOL_CALL_CLOSE, TOOL_CALL_OPEN
from factory_droid_openai.runner import (
    DroidModel,
    ReasoningDelta,
    RunComplete,
    RunEvent,
    RunRequest,
    TextDelta,
    Usage,
    UsageUpdate,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from factory_droid_openai.app import RunnerFactory


class ScriptedRunner:
    def __init__(self, events: list[RunEvent]) -> None:
        self.events = events
        self.requests: list[RunRequest] = []

    async def run(self, request: RunRequest) -> AsyncIterator[RunEvent]:
        self.requests.append(request)
        for event in self.events:
            yield event

    async def list_models(self, *, timeout_seconds: float) -> tuple[DroidModel, ...]:
        del timeout_seconds
        return (
            DroidModel(
                id="gpt-5.4",
                display_name="GPT-5.4",
                provider="openai",
                supported_reasoning_efforts=("low", "high"),
                default_reasoning_effort="high",
                supports_images=True,
                supports_pdfs=True,
            ),
        )


class BlockingRunner(ScriptedRunner):
    def __init__(self) -> None:
        super().__init__([])
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self, request: RunRequest) -> AsyncIterator[RunEvent]:
        self.requests.append(request)
        self.started.set()
        await self.release.wait()
        yield RunComplete(Usage())


def _sdk_client(
    tmp_path: Path,
    runner: ScriptedRunner,
    *,
    settings: Settings | None = None,
) -> tuple[AsyncOpenAI, httpx.AsyncClient]:
    app = create_app(
        settings or Settings(api_key="sdk-token", workdir=tmp_path),
        runner_factory=cast("RunnerFactory", lambda: runner),
    )
    http_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://bridge",
    )
    client = AsyncOpenAI(
        base_url="http://bridge/v1",
        api_key="sdk-token",
        http_client=http_client,
        max_retries=0,
    )
    return client, http_client


@pytest.mark.asyncio
async def test_official_openai_client_parses_completion_and_models(
    tmp_path: Path,
) -> None:
    usage = Usage(
        input_tokens=7,
        output_tokens=3,
        cache_read_tokens=2,
        cache_write_tokens=1,
    )
    runner = ScriptedRunner(
        [
            ReasoningDelta("brief reasoning"),
            TextDelta("Hello from Droid"),
            UsageUpdate(usage),
            RunComplete(usage),
        ]
    )
    client, http_client = _sdk_client(tmp_path, runner)
    async with http_client:
        models = await client.models.list()
        model = await client.models.retrieve("gpt-5.4")
        completion = await client.chat.completions.create(
            model="factory-droid",
            messages=[{"role": "user", "content": "Hello"}],
        )

    assert models.data[0].id == "factory-droid"
    assert model.id == "gpt-5.4"
    assert model.object == "model"
    assert completion.object == "chat.completion"
    assert completion.choices[0].finish_reason == "stop"
    assert completion.choices[0].message.content == "Hello from Droid"
    assert completion.usage is not None
    assert completion.usage.prompt_tokens == 7
    assert completion.usage.completion_tokens == 3
    assert runner.requests[0].model == "factory-droid"


@pytest.mark.asyncio
async def test_official_openai_client_parses_stream(
    tmp_path: Path,
) -> None:
    usage = Usage(input_tokens=4, output_tokens=2)
    runner = ScriptedRunner(
        [
            TextDelta("Hello"),
            TextDelta(" stream"),
            UsageUpdate(usage),
            RunComplete(usage),
        ]
    )
    client, http_client = _sdk_client(tmp_path, runner)
    async with http_client:
        stream = await client.chat.completions.create(
            model="factory-droid",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks = [chunk async for chunk in stream]

    content = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
    assert content == "Hello stream"
    assert chunks[-2].choices[0].finish_reason == "stop"
    assert all(chunk.usage is None for chunk in chunks[:-1])
    assert chunks[-1].choices == []
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.total_tokens == 6


@pytest.mark.asyncio
async def test_official_openai_client_sends_structured_output_format(
    tmp_path: Path,
) -> None:
    runner = ScriptedRunner([TextDelta('{"answer":7}'), RunComplete(Usage())])
    client, http_client = _sdk_client(tmp_path, runner)
    async with http_client:
        completion = await client.chat.completions.create(
            model="gpt-5.4",
            messages=[{"role": "user", "content": "Pick a number."}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "integer"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                },
            },
        )

    assert completion.choices[0].message.content == '{"answer":7}'
    assert runner.requests[0].output_format is not None
    assert runner.requests[0].output_format["type"] == "json_schema"


@pytest.mark.asyncio
async def test_official_openai_client_parses_function_tool_call(
    tmp_path: Path,
) -> None:
    marker = (
        f'{TOOL_CALL_OPEN}{{"name":"get_weather","arguments":{{"city":"Gdansk"}}}}{TOOL_CALL_CLOSE}'
    )
    runner = ScriptedRunner([TextDelta(marker), RunComplete(Usage())])
    client, http_client = _sdk_client(tmp_path, runner)
    async with http_client:
        completion = await client.chat.completions.create(
            model="factory-droid",
            messages=[{"role": "user", "content": "Check the weather"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Read weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
        )

    message = completion.choices[0].message
    assert completion.choices[0].finish_reason == "tool_calls"
    assert message.tool_calls is not None
    tool_call = message.tool_calls[0]
    assert tool_call.type == "function"
    assert tool_call.function.name == "get_weather"
    assert tool_call.function.arguments == '{"city":"Gdansk"}'


@pytest.mark.asyncio
async def test_official_openai_client_recovers_partial_tool_close(
    tmp_path: Path,
) -> None:
    marker = (
        f'{TOOL_CALL_OPEN}{{"name":"get_weather","arguments":{{"city":"Gdansk"}}}}'
        f"{TOOL_CALL_CLOSE[:-1]}"
    )
    runner = ScriptedRunner([TextDelta(marker), RunComplete(Usage())])
    client, http_client = _sdk_client(tmp_path, runner)
    async with http_client:
        completion = await client.chat.completions.create(
            model="factory-droid",
            messages=[{"role": "user", "content": "Check the weather"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        )

    message = completion.choices[0].message
    assert completion.choices[0].finish_reason == "tool_calls"
    assert message.content is None
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 1
    tool_call = message.tool_calls[0]
    assert tool_call.type == "function"
    assert tool_call.function.name == "get_weather"
    assert tool_call.function.arguments == '{"city":"Gdansk"}'


@pytest.mark.asyncio
async def test_official_openai_stream_recovers_partial_tool_close(
    tmp_path: Path,
) -> None:
    marker = (
        f'{TOOL_CALL_OPEN}{{"name":"get_weather","arguments":{{"city":"Gdansk"}}}}'
        f"{TOOL_CALL_CLOSE[:-1]}"
    )
    runner = ScriptedRunner([TextDelta(marker), RunComplete(Usage())])
    client, http_client = _sdk_client(tmp_path, runner)
    async with http_client:
        stream = await client.chat.completions.create(
            model="factory-droid",
            messages=[{"role": "user", "content": "Check the weather"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
            stream=True,
        )
        chunks = [chunk async for chunk in stream]

    tool_calls = [
        call
        for chunk in chunks
        for choice in chunk.choices
        for call in choice.delta.tool_calls or ()
    ]
    finish_reasons = [
        choice.finish_reason
        for chunk in chunks
        for choice in chunk.choices
        if choice.finish_reason is not None
    ]
    assert finish_reasons == ["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0].function is not None
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[0].function.arguments == '{"city":"Gdansk"}'


@pytest.mark.asyncio
async def test_official_openai_client_parses_payload_limit_error(
    tmp_path: Path,
) -> None:
    runner = ScriptedRunner([])
    client, http_client = _sdk_client(
        tmp_path,
        runner,
        settings=Settings(
            api_key="sdk-token",
            workdir=tmp_path,
            max_request_bytes=100,
        ),
    )
    async with http_client:
        with pytest.raises(APIStatusError) as error:
            await client.chat.completions.create(
                model="factory-droid",
                messages=[{"role": "user", "content": "x" * 200}],
            )

    assert error.value.status_code == 413
    assert isinstance(error.value.body, dict)
    assert error.value.body["type"] == "invalid_request_error"


@pytest.mark.asyncio
async def test_official_openai_client_parses_queue_overload_and_retry_after(
    tmp_path: Path,
) -> None:
    runner = BlockingRunner()
    client, http_client = _sdk_client(
        tmp_path,
        runner,
        settings=Settings(
            api_key="sdk-token",
            workdir=tmp_path,
            max_concurrency=1,
            max_queue_size=0,
            retry_after_seconds=3,
        ),
    )
    async with http_client:
        active = asyncio.create_task(
            http_client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sdk-token"},
                json={
                    "model": "factory-droid",
                    "messages": [{"role": "user", "content": "first"}],
                },
            )
        )
        await runner.started.wait()
        with pytest.raises(APIStatusError) as error:
            await client.chat.completions.create(
                model="factory-droid",
                messages=[{"role": "user", "content": "second"}],
            )
        runner.release.set()
        assert (await active).status_code == 200

    assert error.value.status_code == 429
    assert error.value.response.headers["retry-after"] == "3"
    assert isinstance(error.value.body, dict)
    assert error.value.body["type"] == "rate_limit_error"


@pytest.mark.asyncio
async def test_official_openai_client_applies_stop_sequences(tmp_path: Path) -> None:
    runner = ScriptedRunner(
        [
            TextDelta("kept "),
            TextDelta("HALT dropped"),
            RunComplete(Usage()),
        ]
    )
    client, http_client = _sdk_client(tmp_path, runner)
    async with http_client:
        completion = await client.chat.completions.create(
            model="factory-droid",
            messages=[{"role": "user", "content": "Hello"}],
            stop="HALT",
        )

    assert completion.choices[0].message.content == "kept "
    assert completion.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_official_openai_client_receives_multiple_choices(tmp_path: Path) -> None:
    runner = ScriptedRunner([TextDelta("answer"), RunComplete(Usage(2, 1, 0, 0))])
    client, http_client = _sdk_client(tmp_path, runner)
    async with http_client:
        completion = await client.chat.completions.create(
            model="factory-droid",
            messages=[{"role": "user", "content": "Hello"}],
            n=2,
        )

    assert [choice.index for choice in completion.choices] == [0, 1]
    assert completion.usage is not None
    assert completion.usage.prompt_tokens == 4


@pytest.mark.asyncio
async def test_official_openai_client_parses_parallel_tool_calls(tmp_path: Path) -> None:
    body = '{{"name":"get_weather","arguments":{{"city":"{city}"}}}}'
    first = f"{TOOL_CALL_OPEN}{body.format(city='Gdansk')}{TOOL_CALL_CLOSE}"
    second = f"{TOOL_CALL_OPEN}{body.format(city='Sopot')}{TOOL_CALL_CLOSE}"
    runner = ScriptedRunner([TextDelta(first + second), RunComplete(Usage())])
    client, http_client = _sdk_client(tmp_path, runner)
    async with http_client:
        completion = await client.chat.completions.create(
            model="factory-droid",
            messages=[{"role": "user", "content": "Weather?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        )

    tool_calls = completion.choices[0].message.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 2
    arguments = []
    for call in tool_calls:
        assert call.type == "function"
        arguments.append(call.function.arguments)
    assert arguments == ['{"city":"Gdansk"}', '{"city":"Sopot"}']
    assert completion.choices[0].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_official_openai_client_sends_image_parts(tmp_path: Path) -> None:
    runner = ScriptedRunner([TextDelta("a cat"), RunComplete(Usage())])
    client, http_client = _sdk_client(tmp_path, runner)
    async with http_client:
        completion = await client.chat.completions.create(
            model="factory-droid",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,QUJD"},
                        },
                    ],
                }
            ],
        )

    assert completion.choices[0].message.content == "a cat"
    assert runner.requests[0].images == (
        {"type": "base64", "mediaType": "image/png", "data": "QUJD"},
    )
