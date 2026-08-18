from __future__ import annotations

import importlib.util
import json
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from factory_droid_openai.protocol import _MANGLED_TOOL_CALLS_PATTERN

if TYPE_CHECKING:
    from types import ModuleType

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "e2e_matrix.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("e2e_matrix", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve string annotations through sys.modules, so a module
    # loaded straight from a path has to register before it executes.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def e2e() -> ModuleType:
    return _load_module()


def _observation(e2e: ModuleType, **overrides: Any) -> Any:
    defaults: dict[str, Any] = {
        "status": 200,
        "finish_reason": "stop",
        "content_chars": 12,
    }
    defaults.update(overrides)
    return e2e.Observation(**defaults)


def _scenario(e2e: ModuleType, **overrides: Any) -> Any:
    defaults: dict[str, Any] = {
        "name": "hello",
        "body": {"messages": [{"role": "user", "content": "hi"}]},
    }
    defaults.update(overrides)
    return e2e.Scenario(**defaults)


def test_scenarios_cover_both_transports_and_run_hostile_cases_once(e2e: ModuleType) -> None:
    plan = e2e.scenarios()
    switch_plan = e2e.switch_scenarios()
    continuation_plan = e2e.continuation_switch_scenarios()

    streamed = {scenario.name for scenario in plan if scenario.stream}
    hostile = [scenario for scenario in plan if not scenario.per_model]

    assert "tool_required" in streamed
    assert hostile, "hostile scenarios must be part of the default plan"
    assert all(scenario.expect_status != (200,) for scenario in hostile)
    assert all(not scenario.stream for scenario in hostile)
    assert len(e2e.scenarios(streaming=False)) < len(plan)
    assert {scenario.stream for scenario in switch_plan} == {False, True}
    assert len(e2e.switch_scenarios(streaming=False)) == 1
    assert {scenario.stream for scenario in continuation_plan} == {False, True}
    assert len(e2e.continuation_switch_scenarios(streaming=False)) == 1


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("   ", "must not be empty"),
        ("line\nbreak", "must not contain newlines"),
        ("line\u2028break", "must not contain newlines"),
        ("bad\x00label", "must not contain control characters"),
        ("x" * 129, "at most 128"),
    ],
)
def test_matrix_labels_are_trimmed_and_validated(
    e2e: ModuleType,
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        e2e.normalize_label(value)

    assert e2e.normalize_label("  native Ż  ") == "native Ż"


def test_blank_argument_scenarios_accept_answer_or_tool_retry(e2e: ModuleType) -> None:
    scenarios = [
        scenario for scenario in e2e.scenarios() if scenario.name == "tool_blank_arguments"
    ]

    assert {scenario.stream for scenario in scenarios} == {False, True}
    for scenario in scenarios:
        answered = _observation(
            e2e,
            finish_reason="stop",
            content_chars=12,
            stream_done=scenario.stream,
        )
        retried = _observation(
            e2e,
            finish_reason="tool_calls",
            tool_calls=1,
            content_chars=0,
            stream_done=scenario.stream,
        )

        assert e2e.classify(scenario, answered)[0] == e2e.SUCCESS
        assert e2e.classify(scenario, retried)[0] == e2e.SUCCESS


def test_contract_satisfied_is_a_pass(e2e: ModuleType) -> None:
    verdict, _ = e2e.classify(_scenario(e2e), _observation(e2e))

    assert verdict == e2e.SUCCESS


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (
            '{"messages":[{"role":"user","content":"hi"}],"tools":[]}',
            "assistant reproduced the OpenAI transcript",
        ),
        (
            '{"role":"assistant","content":"copied"}',
            "assistant reproduced an OpenAI assistant message",
        ),
        (
            '{"role":"tool","content":"copied","tool_call_id":"call_1"}',
            "assistant reproduced an OpenAI tool message",
        ),
        (
            '{"messages":[],"tools":[]} continuation garbage',
            "assistant reproduced the OpenAI transcript",
        ),
        (
            '```json\n{"messages":[{"role":"user","content":"hi"}],"tools":[]}\n```',
            "assistant reproduced the OpenAI transcript",
        ),
        (
            '```\n{"role":"assistant","content":"copied"}\n```',
            "assistant reproduced an OpenAI assistant message",
        ),
        (
            '"tool calls": "id": "call_123"',
            "assistant exposed malformed OpenAI tool-call output",
        ),
        (
            '"tool_calls":"call_123"',
            "assistant exposed malformed OpenAI tool-call output",
        ),
        ("[1,2,3]", None),
        ("ordinary answer", None),
    ],
)
def test_content_semantics_detect_transcript_reproduction(
    e2e: ModuleType,
    content: str,
    expected: str | None,
) -> None:
    assert e2e._content_semantic_error(content) == expected


def test_transcript_reproduction_is_a_bridge_defect(e2e: ModuleType) -> None:
    observation = _observation(
        e2e,
        content_semantic_error="assistant reproduced the OpenAI transcript",
    )

    verdict, detail = e2e.classify(_scenario(e2e), observation)

    assert verdict == e2e.BRIDGE_DEFECT
    assert "transcript" in detail


def test_expected_rejection_is_a_pass(e2e: ModuleType) -> None:
    scenario = _scenario(
        e2e,
        expect_status=(400,),
        expect_error_type="invalid_request_error",
        expect_error_contains="settings do not match",
    )

    verdict, detail = e2e.classify(
        scenario,
        _observation(
            e2e,
            status=400,
            error_type="invalid_request_error",
            error_message="session settings do not match",
            finish_reason=None,
        ),
    )

    assert verdict == e2e.SUCCESS
    assert "expected" in detail


@pytest.mark.parametrize(
    ("overrides", "detail"),
    [
        ({"error_type": "other"}, "error type"),
        ({"error_message": "other"}, "message marker"),
    ],
)
def test_expected_rejection_checks_error_contract(
    e2e: ModuleType,
    overrides: dict[str, str],
    detail: str,
) -> None:
    scenario = _scenario(
        e2e,
        expect_status=(400,),
        expect_error_type="invalid_request_error",
        expect_error_contains="settings do not match",
    )
    fields: dict[str, Any] = {
        "status": 400,
        "error_type": "invalid_request_error",
        "error_message": "session settings do not match",
        "finish_reason": None,
    }
    fields.update(overrides)
    observation = _observation(e2e, **fields)

    verdict, reason = e2e.classify(scenario, observation)

    assert verdict == e2e.BRIDGE_DEFECT
    assert detail in reason


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"timed_out": True}, "harness_timeout"),
        ({"transport_error": "ReadError"}, "bridge_defect"),
        ({"status": 404, "error_type": "model_not_found"}, "account_policy"),
        ({"status": 502, "error_type": "factory_native_tool_blocked"}, "model_behavior"),
        ({"status": 429, "error_type": "rate_limit_error"}, "capacity"),
        ({"status": 503, "error_type": "factory_droid_unavailable"}, "provider_unavailable"),
        ({"status": 504, "error_type": "factory_droid_timeout"}, "backend_timeout"),
        ({"status": 502, "error_type": "factory_protocol_error"}, "bridge_defect"),
        (
            {
                "status": 502,
                "error_type": "factory_protocol_error",
                "error_message": "the model did not produce the required tool call",
            },
            "model_behavior",
        ),
        (
            {
                "status": 502,
                "error_type": "factory_protocol_error",
                "error_message": "tool 'assistant' is not available",
            },
            "model_behavior",
        ),
        (
            {
                "status": 502,
                "error_type": "factory_protocol_error",
                "error_message": "unexpected text after tool call (native dialect)",
            },
            "model_behavior",
        ),
        ({"finish_reason": "length"}, "bridge_defect"),
        ({"finish_reason": "tool_calls"}, "model_behavior"),
        ({"content_chars": 0}, "model_behavior"),
    ],
)
def test_failures_land_in_their_own_bucket(
    e2e: ModuleType,
    overrides: dict[str, Any],
    expected: str,
) -> None:
    verdict, detail = e2e.classify(_scenario(e2e), _observation(e2e, **overrides))

    assert verdict == expected
    assert detail


@pytest.mark.parametrize(
    ("finish_reason", "tool_calls", "expected"),
    [
        ("tool_calls", 0, "model_behavior"),
        ("stop", 0, "model_behavior"),
        ("length", 0, "model_behavior"),
        ("length", 1, "bridge_defect"),
    ],
)
def test_tool_call_outcomes_are_classified_by_model_responsibility(
    e2e: ModuleType,
    finish_reason: str,
    tool_calls: int,
    expected: str,
) -> None:
    scenario = _scenario(
        e2e,
        expect_finish=("tool_calls",),
        expect_tool_call=True,
        expect_content=False,
    )

    verdict, _ = e2e.classify(
        scenario,
        _observation(
            e2e,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            content_chars=0,
        ),
    )

    assert verdict == expected


def test_a_stream_that_fails_after_its_headers_is_still_judged(e2e: ModuleType) -> None:
    scenario = _scenario(e2e, stream=True, expect_finish=("tool_calls",), expect_content=False)
    ignored_tool_choice = _observation(
        e2e,
        stream_done=True,
        finish_reason=None,
        error_type="factory_protocol_error",
        error_message="the model did not produce the required tool call",
    )
    broken = _observation(
        e2e,
        stream_done=True,
        finish_reason=None,
        error_type="factory_protocol_error",
        error_message="tool call payload never closed",
    )

    assert e2e.classify(scenario, ignored_tool_choice)[0] == "model_behavior"
    assert e2e.classify(scenario, broken)[0] == "bridge_defect"


def test_a_mid_stream_backend_timeout_is_not_a_bridge_defect(e2e: ModuleType) -> None:
    scenario = _scenario(e2e, stream=True)

    verdict, _ = e2e.classify(
        scenario,
        _observation(
            e2e,
            stream_done=True,
            finish_reason=None,
            error_type="factory_droid_timeout",
        ),
    )

    assert verdict == "backend_timeout"


def test_stream_without_done_is_a_bridge_defect(e2e: ModuleType) -> None:
    scenario = _scenario(e2e, stream=True)

    verdict, detail = e2e.classify(scenario, _observation(e2e, stream_done=False))

    assert verdict == "bridge_defect"
    assert "[DONE]" in detail


def _bridge(e2e: ModuleType, handler: Any) -> Any:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return e2e.Bridge(base_url="http://bridge", api_key="secret-key", client=client)


@pytest.mark.asyncio
async def test_bridge_reads_a_json_completion(e2e: ModuleType) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer secret-key"
        return httpx.Response(
            200,
            json={
                "factory_droid_session_id": "session-json",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "hello",
                            "tool_calls": [{"id": "call_1"}],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
            headers={"x-request-id": "req-1"},
        )

    bridge = _bridge(e2e, handler)
    async with bridge.client:
        observation = await bridge.run("m", _scenario(e2e))

    assert observation.status == 200
    assert observation.finish_reason == "tool_calls"
    assert observation.tool_calls == 1
    assert observation.content_chars == 5
    assert observation.request_id == "req-1"
    assert observation.session_id == "session-json"


@pytest.mark.asyncio
async def test_bridge_reads_a_streamed_completion(e2e: ModuleType) -> None:
    chunks = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "he",
                        "factory_droid_session_id": "session-stream",
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "id": "call_1"}]},
                    "finish_reason": None,
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            text=body,
            headers={"x-request-id": "req-2", "content-type": "text/event-stream"},
        )

    bridge = _bridge(e2e, handler)
    async with bridge.client:
        observation = await bridge.run("m", _scenario(e2e, stream=True))

    assert observation.stream_done is True
    assert observation.finish_reason == "tool_calls"
    assert observation.tool_calls == 1
    assert observation.content_chars == 2
    assert observation.ttft_ms is not None
    assert observation.session_id == "session-stream"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_bridge_flags_transcript_reproduction_in_final_content(
    e2e: ModuleType,
    stream: bool,
) -> None:
    transcript = '{"messages":[{"role":"user","content":"hi"}],"tools":[]}'

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        if stream:
            chunk = {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": transcript},
                        "finish_reason": "stop",
                    }
                ]
            }
            body = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"
            return httpx.Response(
                200,
                text=body,
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": transcript},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    bridge = _bridge(e2e, handler)
    async with bridge.client:
        observation = await bridge.run("m", _scenario(e2e, stream=stream))

    assert observation.content_semantic_error == "assistant reproduced the OpenAI transcript"
    assert e2e.classify(_scenario(e2e, stream=stream), observation)[0] == e2e.BRIDGE_DEFECT


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_bridge_reports_transport_failures_without_raising(
    e2e: ModuleType,
    stream: bool,
) -> None:
    def timing_out(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    def failing(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    bridge = _bridge(e2e, timing_out)
    async with bridge.client:
        timed_out = await bridge.run("m", _scenario(e2e, stream=stream))
    broken_bridge = _bridge(e2e, failing)
    async with broken_bridge.client:
        broken = await broken_bridge.run("m", _scenario(e2e, stream=stream))

    assert timed_out.timed_out is True
    assert broken.transport_error == "ConnectError"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("not json at all", "response was not JSON"),
        ('"a string"', "response was not a JSON object"),
    ],
)
async def test_bridge_flags_responses_that_are_not_completion_objects(
    e2e: ModuleType,
    payload: str,
    expected: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, text=payload)

    bridge = _bridge(e2e, handler)
    async with bridge.client:
        observation = await bridge.run("m", _scenario(e2e))

    assert observation.transport_error == expected


@pytest.mark.asyncio
async def test_a_stream_request_rejected_before_the_stream_reads_the_json_error(
    e2e: ModuleType,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            404,
            json={"error": {"type": "model_not_found", "message": "not for this account"}},
        )

    bridge = _bridge(e2e, handler)
    async with bridge.client:
        observation = await bridge.run("m", _scenario(e2e, stream=True))

    assert observation.status == 404
    assert observation.error_type == "model_not_found"
    assert observation.error_message == "not for this account"


@pytest.mark.asyncio
async def test_bridge_flags_invalid_stream_json(e2e: ModuleType) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            text="data: {oops\n\n",
            headers={"content-type": "text/event-stream"},
        )

    bridge = _bridge(e2e, handler)
    async with bridge.client:
        observation = await bridge.run("m", _scenario(e2e, stream=True))

    assert observation.transport_error == "stream carried invalid JSON"


@pytest.mark.asyncio
async def test_bridge_lists_models(e2e: ModuleType) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "a"}, {"id": "b"}, "junk"]})

    bridge = _bridge(e2e, handler)
    async with bridge.client:
        models = await bridge.models()

    assert models == ["a", "b"]


@pytest.mark.asyncio
async def test_matrix_runs_every_pair_and_streams_rows_out(e2e: ModuleType) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    plan = [_scenario(e2e), _scenario(e2e, name="hostile", per_model=False, expect_status=(400,))]
    written: list[dict[str, Any]] = []
    bridge = _bridge(e2e, handler)
    async with bridge.client:
        rows = await e2e.run_matrix(
            bridge,
            ["m1", "m2"],
            plan,
            concurrency=1,
            on_row=written.append,
            label="native",
        )

    assert len(rows) == 3
    assert written == rows
    assert all(record["label"] == "native" for record in rows)
    assert {row["model"] for row in rows} == {"m1", "m2"}
    assert sum(1 for row in rows if row["scenario"] == "hostile") == 1


@pytest.mark.asyncio
async def test_matrix_without_models_still_runs_hostile_cases(e2e: ModuleType) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(400, json={"error": {"type": "invalid_request_error"}})

    plan = [_scenario(e2e, name="hostile", per_model=False, expect_status=(400,))]
    bridge = _bridge(e2e, handler)
    async with bridge.client:
        rows = await e2e.run_matrix(bridge, [], plan, concurrency=2)

    assert [row["model"] for row in rows] == ["factory-droid"]
    assert rows[0]["verdict"] == e2e.SUCCESS


@pytest.mark.asyncio
async def test_switch_matrix_runs_a_serial_model_ring(e2e: ModuleType) -> None:
    seen_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_models.append(json.loads(request.content)["model"])
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    written: list[dict[str, Any]] = []
    bridge = _bridge(e2e, handler)
    plan = [_scenario(e2e, name="model_switch")]
    async with bridge.client:
        rows = await e2e.run_switch_matrix(
            bridge,
            ["m1", "m2", "m3"],
            plan,
            settle_seconds=0,
            on_row=written.append,
            label="native",
        )

    assert seen_models == ["m1", "m2", "m2", "m3", "m3", "m1"]
    assert [(row["source_model"], row["model"]) for row in rows] == [
        ("m1", "m2"),
        ("m2", "m3"),
        ("m3", "m1"),
    ]
    assert all(row["prime_verdict"] == e2e.SUCCESS for row in rows)
    assert all(record["label"] == "native" for record in rows)
    assert written == rows


@pytest.mark.asyncio
async def test_switch_matrix_skips_one_model_and_supports_no_callback(e2e: ModuleType) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    bridge = _bridge(e2e, handler)
    plan = [_scenario(e2e, name="model_switch")]
    async with bridge.client:
        assert (
            await e2e.run_switch_matrix(
                bridge,
                ["m1"],
                plan,
                settle_seconds=0,
            )
            == []
        )
        rows = await e2e.run_switch_matrix(
            bridge,
            ["m1", "m2"],
            plan,
            settle_seconds=0,
        )

    assert len(rows) == 2


def test_transition_row_propagates_a_failed_prime(e2e: ModuleType) -> None:
    record = e2e._transition_row(
        "source",
        "target",
        _scenario(e2e),
        _observation(e2e),
        _observation(
            e2e,
            status=404,
            error_type="model_not_found",
            finish_reason=None,
        ),
    )

    assert record["verdict"] == e2e.ACCOUNT_POLICY
    assert record["prime_verdict"] == e2e.ACCOUNT_POLICY
    assert "prime failed" in record["detail"]


@pytest.mark.asyncio
async def test_continuation_switch_matrix_rejects_a_model_ring(e2e: ModuleType) -> None:
    seen: list[tuple[str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        model = body["model"]
        session_id = body.get("factory_droid_session_id")
        seen.append((model, session_id))
        if session_id is not None:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "type": "invalid_request_error",
                        "message": "session settings do not match",
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "factory_droid_session_id": f"session-{model}",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ready"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    written: list[dict[str, Any]] = []
    bridge = _bridge(e2e, handler)
    plan = e2e.continuation_switch_scenarios()
    async with bridge.client:
        rows = await e2e.run_continuation_switch_matrix(
            bridge,
            ["m1", "m2"],
            plan,
            on_row=written.append,
            label="native",
        )

    assert seen == [
        ("m1", None),
        ("m2", "session-m1"),
        ("m2", "session-m1"),
        ("m2", None),
        ("m1", "session-m2"),
        ("m1", "session-m2"),
    ]
    assert all(row["verdict"] == e2e.SUCCESS for row in rows)
    assert all(record["label"] == "native" for record in rows)
    assert written == rows


@pytest.mark.asyncio
async def test_continuation_switch_matrix_reports_missing_session_ids(
    e2e: ModuleType,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ready"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    bridge = _bridge(e2e, handler)
    plan = e2e.continuation_switch_scenarios(streaming=False)
    async with bridge.client:
        assert (
            await e2e.run_continuation_switch_matrix(
                bridge,
                ["m1"],
                plan,
            )
            == []
        )
        rows = await e2e.run_continuation_switch_matrix(
            bridge,
            ["m1", "m2"],
            plan,
        )

    assert len(rows) == 2
    assert all(row["verdict"] == e2e.BRIDGE_DEFECT for row in rows)
    assert all("no session id" in row["detail"] for row in rows)


@pytest.mark.asyncio
async def test_continuation_switch_matrix_skips_an_opted_out_plan(e2e: ModuleType) -> None:
    """An empty plan is the default, so it must not prime a single model."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["model"])
        return httpx.Response(200, json={"choices": []})

    bridge = _bridge(e2e, handler)
    async with bridge.client:
        rows = await e2e.run_continuation_switch_matrix(bridge, ["m1", "m2", "m3"], [])

    assert rows == []
    assert seen == []


@pytest.mark.parametrize(
    "mangled",
    [
        '"tool_calls":"id":"call_1"',
        '"tool_calls":"call_1"',
        '"tool calls": "id" : "call_1"',
        '"TOOL_CALLS":"ID":"CALL_1"',
    ],
)
def test_mangled_detector_agrees_with_the_bridge_parser(e2e: ModuleType, mangled: str) -> None:
    """A shape the harness reports must be one the bridge can actually contain."""
    assert e2e._MANGLED_TOOL_CALL_OUTPUT.search(mangled) is not None
    assert _MANGLED_TOOL_CALLS_PATTERN.search(mangled) is not None


def _rows(e2e: ModuleType) -> list[dict[str, Any]]:
    return [
        e2e.row("m1", _scenario(e2e), _observation(e2e)),
        e2e.row(
            "m2",
            _scenario(e2e),
            _observation(e2e, status=404, error_type="model_not_found"),
        ),
        e2e.row(
            "m3",
            _scenario(e2e, stream=True),
            _observation(e2e, stream_done=False),
        ),
    ]


def test_summary_separates_blocking_findings(e2e: ModuleType) -> None:
    summary = e2e.summarize(_rows(e2e))

    assert summary["total"] == 3
    assert summary["counts"]["account_policy"] == 1
    assert len(summary["blocking"]) == 1
    assert summary["median_ms"] is not None


def test_summary_of_an_empty_run_is_not_a_division_by_zero(e2e: ModuleType) -> None:
    summary = e2e.summarize([])

    assert summary["pass_rate"] == 0.0
    assert summary["median_ms"] is None


def test_report_names_the_blocking_cases(e2e: ModuleType) -> None:
    report = e2e.render_report(_rows(e2e))

    assert "bridge_defect" in report
    assert "`m3`" in report
    assert "## Scenarios" in report
    assert "| Model | Pass | Total | Tool calls |" in report


def test_report_displays_labels_and_legacy_artifacts(e2e: ModuleType) -> None:
    labeled = e2e.render_report([e2e.row("m1", _scenario(e2e), _observation(e2e), label="native")])
    legacy = e2e.render_report(_rows(e2e))

    assert "- Label: native" in labeled
    assert "- Label: unlabeled legacy artifact" in legacy


def test_report_says_so_when_nothing_blocks(e2e: ModuleType) -> None:
    report = e2e.render_report([e2e.row("m1", _scenario(e2e), _observation(e2e))])

    assert "No result classified as a bridge defect" in report


def test_compare_reports_regressions_and_fixes(e2e: ModuleType) -> None:
    before = [
        e2e.row("m1", _scenario(e2e), _observation(e2e)),
        e2e.row("m2", _scenario(e2e), _observation(e2e, finish_reason="length")),
        e2e.row("gone", _scenario(e2e), _observation(e2e)),
    ]
    after = [
        e2e.row("m1", _scenario(e2e), _observation(e2e, finish_reason="length")),
        e2e.row("m2", _scenario(e2e), _observation(e2e)),
        e2e.row("new", _scenario(e2e), _observation(e2e)),
    ]

    result = e2e.compare(before, after)
    rendered = e2e.render_comparison(result)

    assert result["shared"] == 2
    assert [entry["key"] for entry in result["regressions"]] == [("", "m1", "hello", False)]
    assert [entry["key"] for entry in result["fixes"]] == [("", "m2", "hello", False)]
    assert result["only_before"]
    assert result["only_after"]
    assert "## Regressions" in rendered
    assert "## Fixes" in rendered


def test_comparison_without_changes_stays_quiet(e2e: ModuleType) -> None:
    rows = [e2e.row("m1", _scenario(e2e), _observation(e2e))]

    rendered = e2e.render_comparison(e2e.compare(rows, rows))

    assert "## Regressions" not in rendered
    assert "## Fixes" not in rendered


def test_comparison_keeps_transition_sources_distinct(e2e: ModuleType) -> None:
    first = e2e.row("target", _scenario(e2e), _observation(e2e))
    first["source_model"] = "source-a"
    second = e2e.row("target", _scenario(e2e), _observation(e2e))
    second["source_model"] = "source-b"

    assert e2e.compare([first, second], [first, second])["shared"] == 2


def test_artifact_labels_reject_mixed_rows_and_round_trip_unicode(
    e2e: ModuleType,
    tmp_path: Path,
) -> None:
    rows = [e2e.row("m1", _scenario(e2e), _observation(e2e), label=" native Ż ")]
    path = tmp_path / "unicode.jsonl"
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in rows) + "\n",
        encoding="utf-8",
    )

    loaded = e2e.load_rows(path)

    assert loaded[0]["label"] == "native Ż"
    assert e2e.artifact_label(loaded) == "native Ż"
    e2e._validate_append_target(path, "native Ż")
    empty = tmp_path / "empty.jsonl"
    empty.write_text(" \n\n", encoding="utf-8")
    e2e._validate_append_target(empty, "native")
    with pytest.raises(ValueError, match="mixed labels") as error:
        e2e.artifact_label([*loaded, e2e.row("m2", _scenario(e2e), _observation(e2e))])
    assert "native Ż" in str(error.value)
    assert "unlabeled legacy artifact" in str(error.value)


def test_artifact_labels_report_distinct_labels(e2e: ModuleType) -> None:
    rows = [
        e2e.row("m1", _scenario(e2e), _observation(e2e), label="text"),
        e2e.row("m2", _scenario(e2e), _observation(e2e), label="native"),
    ]

    with pytest.raises(ValueError, match="mixed labels") as error:
        e2e.artifact_label(rows)

    assert "native" in str(error.value)
    assert "text" in str(error.value)


def test_locked_output_waits_for_posix_lock(e2e: ModuleType, tmp_path: Path) -> None:
    if e2e._fcntl is None:
        pytest.skip("POSIX locking is unavailable")
    path = tmp_path / "run.jsonl"
    started = threading.Event()
    acquired = threading.Event()
    errors: list[BaseException] = []

    def contend() -> None:
        started.set()
        try:
            with e2e._locked_output(path, None):
                acquired.set()
        except BaseException as exc:
            errors.append(exc)

    with e2e._locked_output(path, None):
        thread = threading.Thread(target=contend)
        thread.start()
        assert started.wait(1)
        assert not acquired.wait(0.05)
    thread.join(1)

    assert not thread.is_alive()
    assert not errors
    assert acquired.is_set()


def test_locked_output_retries_windows_lock(
    e2e: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMsvcrt:
        LK_NBLCK = 1
        LK_UNLCK = 2

        def __init__(self) -> None:
            self.calls: list[tuple[int, int, int]] = []
            self.attempts = 0

        def locking(self, fd: int, mode: int, size: int) -> None:
            self.calls.append((fd, mode, size))
            if mode == self.LK_NBLCK and self.attempts == 0:
                self.attempts += 1
                raise OSError("locked")

    fake_msvcrt = FakeMsvcrt()
    sleeps: list[float] = []
    monkeypatch.setattr(e2e, "_fcntl", None)
    monkeypatch.setattr(e2e, "_msvcrt", fake_msvcrt)
    monkeypatch.setattr(e2e.time, "sleep", sleeps.append)
    path = tmp_path / "run.jsonl"

    with e2e._locked_output(path, None):
        pass

    assert [mode for _, mode, _ in fake_msvcrt.calls] == [
        fake_msvcrt.LK_NBLCK,
        fake_msvcrt.LK_NBLCK,
        fake_msvcrt.LK_UNLCK,
    ]
    assert sleeps == [0.1]


def test_compare_renders_distinct_labels_and_allows_legacy_inputs(e2e: ModuleType) -> None:
    before = [e2e.row("m1", _scenario(e2e), _observation(e2e), label="text")]
    after = [e2e.row("m1", _scenario(e2e), _observation(e2e), label="native")]

    result = e2e.compare(before, after)
    rendered = e2e.render_comparison(result)
    legacy_rendered = e2e.render_comparison(e2e.compare(_rows(e2e), after))

    assert result["before_label"] == "text"
    assert result["after_label"] == "native"
    assert "- Before label: text" in rendered
    assert "- After label: native" in rendered
    assert "- Label transition: text -> native" in rendered
    assert "identity is unavailable" in legacy_rendered


def test_compare_rejects_same_label_unless_overridden(e2e: ModuleType) -> None:
    rows = [e2e.row("m1", _scenario(e2e), _observation(e2e), label="native")]

    with pytest.raises(ValueError, match="cannot compare"):
        e2e.compare(rows, rows)

    assert e2e.compare(rows, rows, allow_same_label=True)["shared"] == 1


def test_compare_command_allows_same_label_with_flag(
    e2e: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    row = e2e.row("m1", _scenario(e2e), _observation(e2e), label="native")
    before = tmp_path / "before.jsonl"
    after = tmp_path / "after.jsonl"
    for path in (before, after):
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    code = e2e.main(["compare", str(before), str(after), "--allow-same-label"])

    assert code == 0
    assert "Before label: native" in capsys.readouterr().out


def test_report_command_reads_a_jsonl_run(
    e2e: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "run.jsonl"
    rows = _rows(e2e)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n\n",
        encoding="utf-8",
    )

    code = e2e.main(["report", str(path)])

    assert code == 0
    assert "Bridge e2e run" in capsys.readouterr().out


def test_compare_command_fails_on_regressions(
    e2e: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    before = tmp_path / "before.jsonl"
    after = tmp_path / "after.jsonl"
    before.write_text(
        json.dumps(e2e.row("m1", _scenario(e2e), _observation(e2e))) + "\n",
        encoding="utf-8",
    )
    after.write_text(
        json.dumps(
            e2e.row("m1", _scenario(e2e), _observation(e2e, finish_reason="length")),
        )
        + "\n",
        encoding="utf-8",
    )

    code = e2e.main(["compare", str(before), str(after)])

    assert code == 1
    assert "Regressions" in capsys.readouterr().out


@pytest.mark.parametrize("test_continuity", [False, True])
def test_run_command_writes_rows_and_reports_blocking_findings(
    e2e: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    test_continuity: bool,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "m1"}]})
        return httpx.Response(200, json={"choices": []})

    class MockClient(httpx.AsyncClient):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(e2e.httpx, "AsyncClient", MockClient)
    monkeypatch.delenv("FACTORY_DROID_OPENAI_API_KEY", raising=False)
    out = tmp_path / "nested" / "run.jsonl"
    args = ["run", "--no-stream", "--concurrency", "4", "--out", str(out)]
    args.extend(["--label", "native"])
    if test_continuity:
        args.append("--test-session-continuity")

    code = e2e.main(args)

    rows = e2e.load_rows(out)
    assert code == 1
    assert rows
    assert {row["label"] for row in rows} == {"native"}
    assert all(row["model"] == "m1" for row in rows if row["scenario"] != "hostile_unknown_model")
    assert "Bridge e2e run" in capsys.readouterr().out


@pytest.mark.parametrize("label_args", [["--label", "native"], []])
def test_run_rejects_mismatched_append_before_network(
    e2e: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    label_args: list[str],
) -> None:
    out = tmp_path / "run.jsonl"
    out.write_text(
        json.dumps(e2e.row("m1", _scenario(e2e), _observation(e2e), label="text")) + "\n",
        encoding="utf-8",
    )

    class NoNetworkClient:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            raise AssertionError("network must not start")

    monkeypatch.setattr(e2e.httpx, "AsyncClient", NoNetworkClient)

    code = e2e.main(["run", *label_args, "--out", str(out)])

    assert code == 2
    assert "error: cannot append" in capsys.readouterr().err
