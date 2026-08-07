from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest

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

    streamed = {scenario.name for scenario in plan if scenario.stream}
    hostile = [scenario for scenario in plan if not scenario.per_model]

    assert "tool_required" in streamed
    assert hostile, "hostile scenarios must be part of the default plan"
    assert all(scenario.expect_status != (200,) for scenario in hostile)
    assert all(not scenario.stream for scenario in hostile)
    assert len(e2e.scenarios(streaming=False)) < len(plan)


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
    scenario = _scenario(e2e, expect_status=(400,))

    verdict, detail = e2e.classify(scenario, _observation(e2e, status=400, finish_reason=None))

    assert verdict == e2e.SUCCESS
    assert "expected" in detail


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


def test_missing_tool_call_is_model_behavior_not_a_bridge_defect(e2e: ModuleType) -> None:
    scenario = _scenario(
        e2e,
        expect_finish=("tool_calls",),
        expect_tool_call=True,
        expect_content=False,
    )

    verdict, _ = e2e.classify(
        scenario,
        _observation(e2e, finish_reason="tool_calls", tool_calls=0, content_chars=0),
    )

    assert verdict == "model_behavior"


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
                ]
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


@pytest.mark.asyncio
async def test_bridge_reads_a_streamed_completion(e2e: ModuleType) -> None:
    chunks = [
        {"choices": [{"index": 0, "delta": {"content": "he"}, "finish_reason": None}]},
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
        )

    assert len(rows) == 3
    assert written == rows
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
    assert [entry["key"] for entry in result["regressions"]] == [("m1", "hello", False)]
    assert [entry["key"] for entry in result["fixes"]] == [("m2", "hello", False)]
    assert result["only_before"]
    assert result["only_after"]
    assert "## Regressions" in rendered
    assert "## Fixes" in rendered


def test_comparison_without_changes_stays_quiet(e2e: ModuleType) -> None:
    rows = [e2e.row("m1", _scenario(e2e), _observation(e2e))]

    rendered = e2e.render_comparison(e2e.compare(rows, rows))

    assert "## Regressions" not in rendered
    assert "## Fixes" not in rendered


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


def test_run_command_writes_rows_and_reports_blocking_findings(
    e2e: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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

    code = e2e.main(["run", "--no-stream", "--concurrency", "4", "--out", str(out)])

    rows = e2e.load_rows(out)
    assert code == 1
    assert rows
    assert all(row["model"] == "m1" for row in rows if row["scenario"] != "hostile_unknown_model")
    assert "Bridge e2e run" in capsys.readouterr().out
