from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from types import ModuleType

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "e2e_fixtures.py"


@pytest.fixture(scope="module")
def fixtures_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("e2e_fixtures", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _trace_line(request_id: str, payload: dict[str, Any], *, event: str = "droid.event") -> str:
    return json.dumps(
        {
            "event": event,
            "request_id": request_id,
            "mode": "full",
            "payload": json.dumps(payload),
        }
    )


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": "gpt-5.4",
        "scenario": "tool_required",
        "stream": False,
        "request_id": "req-1",
        "verdict": "pass",
    }
    row.update(overrides)
    return row


def test_events_are_grouped_per_request_in_arrival_order(fixtures_script: ModuleType) -> None:
    trace = [
        json.loads(_trace_line("req-1", {"kind": "text_delta", "text": "a"})),
        json.loads(_trace_line("req-2", {"kind": "text_delta", "text": "b"})),
        json.loads(_trace_line("req-1", {"kind": "text_delta", "text": "c"})),
        json.loads(_trace_line("req-1", {"kind": "text_delta", "text": "x"}, event="chat.prompt")),
        {"event": "droid.event", "request_id": "req-3", "payload_head": "truncated"},
        {"event": "droid.event", "payload": "{}"},
    ]

    grouped = fixtures_script.group_events(trace)

    assert [event["text"] for event in grouped["req-1"]] == ["a", "c"]
    assert set(grouped) == {"req-1", "req-2"}


def test_fixture_names_stay_filesystem_safe(fixtures_script: ModuleType) -> None:
    name = fixtures_script.fixture_name(_row(model="Claude Sonnet 4.5", stream=True))

    assert name == "tool-required--claude-sonnet-4-5-stream.jsonl"


def test_only_selected_rows_with_recorded_events_become_fixtures(
    fixtures_script: ModuleType,
) -> None:
    rows = [
        _row(),
        _row(request_id="req-2", verdict="model_behavior", scenario="hello"),
        _row(request_id="missing"),
        _row(request_id=None),
        _row(request_id="req-4", scenario="unicode"),
    ]
    events = {
        "req-1": [{"kind": "text_delta", "text": "a"}],
        "req-2": [{"kind": "text_delta", "text": "b"}],
        "req-4": [{"kind": "text_delta", "text": "c"}],
    }

    built = fixtures_script.build_fixtures(rows, events, scenarios=("tool_required", "hello"))

    assert list(built) == ["tool-required--gpt-5-4.jsonl"]
    assert built["tool-required--gpt-5-4.jsonl"][0] == {
        "kind": "meta",
        "scenario": "tool_required",
        "stream": False,
        "recorded_model": "gpt-5.4",
        "expect_verdict": "pass",
    }


def test_fixtures_can_be_limited_to_one_model(fixtures_script: ModuleType) -> None:
    rows = [_row(), _row(request_id="req-2", model="kimi-k3")]
    events = {
        "req-1": [{"kind": "text_delta", "text": "a"}],
        "req-2": [{"kind": "text_delta", "text": "b"}],
    }

    built = fixtures_script.build_fixtures(rows, events, models=("kimi-k3",))

    assert list(built) == ["tool-required--kimi-k3.jsonl"]


def test_reasoning_is_dropped_unless_it_is_asked_for(fixtures_script: ModuleType) -> None:
    rows = [_row()]
    events = {
        "req-1": [
            {"kind": "reasoning_delta", "text": "operator instructions quoted back"},
            {"kind": "text_delta", "text": "a"},
        ]
    }

    stripped = fixtures_script.build_fixtures(rows, events)
    kept = fixtures_script.build_fixtures(rows, events, keep_reasoning=True)

    assert [line["kind"] for line in stripped["tool-required--gpt-5-4.jsonl"]] == [
        "meta",
        "text_delta",
    ]
    assert [line["kind"] for line in kept["tool-required--gpt-5-4.jsonl"]] == [
        "meta",
        "reasoning_delta",
        "text_delta",
    ]


def test_other_verdicts_can_be_exported_on_purpose(fixtures_script: ModuleType) -> None:
    rows = [_row(verdict="model_behavior")]
    events = {"req-1": [{"kind": "text_delta", "text": "a"}]}

    built = fixtures_script.build_fixtures(rows, events, verdicts=("model_behavior",))

    assert built["tool-required--gpt-5-4.jsonl"][0]["expect_verdict"] == "model_behavior"


def test_build_command_writes_one_file_per_case(
    fixtures_script: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace = tmp_path / "trace.jsonl"
    run = tmp_path / "run.jsonl"
    out = tmp_path / "fixtures"
    trace.write_text(
        _trace_line("req-1", {"kind": "text_delta", "text": "a"}) + "\n\n",
        encoding="utf-8",
    )
    run.write_text(json.dumps(_row()) + "\n", encoding="utf-8")

    code = fixtures_script.main(
        ["build", "--trace", str(trace), "--run", str(run), "--out", str(out)]
    )

    written = list(out.glob("*.jsonl"))
    assert code == 0
    assert len(written) == 1
    lines = [json.loads(line) for line in written[0].read_text(encoding="utf-8").splitlines()]
    assert [line["kind"] for line in lines] == ["meta", "text_delta"]
    assert str(written[0]) in capsys.readouterr().out


def test_build_command_reports_when_nothing_matched(
    fixtures_script: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace = tmp_path / "trace.jsonl"
    run = tmp_path / "run.jsonl"
    trace.write_text("", encoding="utf-8")
    run.write_text(json.dumps(_row()) + "\n", encoding="utf-8")

    code = fixtures_script.main(
        [
            "build",
            "--trace",
            str(trace),
            "--run",
            str(run),
            "--out",
            str(tmp_path / "fixtures"),
            "--verdicts",
            "pass",
            "--scenarios",
            "",
        ]
    )

    assert code == 1
    assert "no fixture matched" in capsys.readouterr().out
