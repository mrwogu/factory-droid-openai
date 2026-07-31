"""Repeatable end-to-end matrix for a running bridge.

Runs a fixed scenario set against every model the bridge lists, classifies each
result, and writes one JSONL row per request. Results stay out of the
repository: they carry model output and account-specific denials.

    uv run python scripts/e2e_matrix.py run --out traces/run-a.jsonl
    uv run python scripts/e2e_matrix.py report traces/run-a.jsonl
    uv run python scripts/e2e_matrix.py compare traces/run-a.jsonl traces/run-b.jsonl

The compare subcommand is the A/B tool: run the matrix, change the bridge (a
prompt variant, a parser fix), run it again, and read the transitions instead
of eyeballing two reports.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8787"
_WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "weather",
        "description": "Current weather for one city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
_CLOCK_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "clock",
        "description": "Current time in one city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

# Verdicts that do not indicate a bridge defect still fail a scenario; only
# BLOCKING ones mean the bridge itself has to change.
SUCCESS = "pass"
BRIDGE_DEFECT = "bridge_defect"
MODEL_BEHAVIOR = "model_behavior"
ACCOUNT_POLICY = "account_policy"
PROVIDER_UNAVAILABLE = "provider_unavailable"
CAPACITY = "capacity"
BACKEND_TIMEOUT = "backend_timeout"
HARNESS_TIMEOUT = "harness_timeout"
VERDICTS: tuple[str, ...] = (
    SUCCESS,
    BRIDGE_DEFECT,
    MODEL_BEHAVIOR,
    ACCOUNT_POLICY,
    PROVIDER_UNAVAILABLE,
    CAPACITY,
    BACKEND_TIMEOUT,
    HARNESS_TIMEOUT,
)


@dataclass(frozen=True, slots=True)
class Scenario:
    """One request shape plus the contract it has to satisfy."""

    name: str
    body: dict[str, Any]
    stream: bool = False
    per_model: bool = True
    expect_status: tuple[int, ...] = (200,)
    expect_finish: tuple[str, ...] = ("stop",)
    expect_tool_call: bool = False
    expect_content: bool = True
    timeout_seconds: float = 120.0


def _baseline_scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="hello",
            body={"messages": [{"role": "user", "content": "Reply with exactly: OK"}]},
        ),
        Scenario(
            name="unicode",
            body={
                "messages": [
                    {"role": "system", "content": "Odpowiadaj po polsku. Używaj ogonków."},
                    {"role": "user", "content": "Napisz jedno zdanie o Gdańsku."},
                ]
            },
        ),
        Scenario(
            name="stop_sequence",
            body={
                "messages": [{"role": "user", "content": "Count: one two THREE four"}],
                "stop": "THREE",
            },
        ),
        Scenario(
            name="ignored_token_limit",
            body={
                "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
                "max_tokens": 5,
            },
        ),
    ]


def _tool_scenarios() -> list[Scenario]:
    prompt = "What is the weather in Gdansk? Use the tool."
    return [
        Scenario(
            name="tool_auto",
            body={
                "messages": [{"role": "user", "content": prompt}],
                "tools": [_WEATHER_TOOL],
            },
            expect_finish=("tool_calls", "stop"),
            expect_content=False,
        ),
        Scenario(
            name="tool_required",
            body={
                "messages": [{"role": "user", "content": prompt}],
                "tools": [_WEATHER_TOOL],
                "tool_choice": "required",
            },
            expect_finish=("tool_calls",),
            expect_tool_call=True,
            expect_content=False,
        ),
        Scenario(
            name="tool_parallel",
            body={
                "messages": [
                    {
                        "role": "user",
                        "content": "Weather and local time in Gdansk. Call both tools.",
                    }
                ],
                "tools": [_WEATHER_TOOL, _CLOCK_TOOL],
                "tool_choice": "required",
            },
            expect_finish=("tool_calls",),
            expect_tool_call=True,
            expect_content=False,
        ),
        Scenario(
            name="tool_result_followup",
            body={
                "messages": [
                    {"role": "user", "content": prompt},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "weather",
                                    "arguments": '{"city":"Gdansk"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": '{"temperature_c":20}',
                    },
                ],
                "tools": [_WEATHER_TOOL],
            },
        ),
        Scenario(
            name="tool_blank_arguments",
            body={
                "messages": [
                    {"role": "user", "content": "What time is it in Gdansk?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "clock", "arguments": ""},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "12:00"},
                ],
                "tools": [_CLOCK_TOOL],
            },
        ),
    ]


def _hostile_scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="hostile_bad_tool_name",
            body={
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "bad.name", "parameters": {}},
                    }
                ],
            },
            per_model=False,
            expect_status=(400,),
            timeout_seconds=30.0,
        ),
        Scenario(
            name="hostile_malformed_tool_arguments",
            body={
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "weather", "arguments": "[]"},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "x"},
                ],
                "tools": [_WEATHER_TOOL],
            },
            per_model=False,
            expect_status=(400,),
            timeout_seconds=30.0,
        ),
        Scenario(
            name="hostile_unknown_model",
            body={
                "messages": [{"role": "user", "content": "hi"}],
                "model": "definitely-not-a-model",
            },
            per_model=False,
            expect_status=(404,),
            timeout_seconds=60.0,
        ),
        Scenario(
            name="hostile_empty_messages",
            body={"messages": []},
            per_model=False,
            expect_status=(400,),
            timeout_seconds=30.0,
        ),
        Scenario(
            name="hostile_too_many_choices",
            body={"messages": [{"role": "user", "content": "hi"}], "n": 99},
            per_model=False,
            expect_status=(400,),
            timeout_seconds=30.0,
        ),
    ]


def scenarios(*, streaming: bool = True) -> list[Scenario]:
    """Every scenario, each non-hostile one in both transport modes."""
    base = _baseline_scenarios() + _tool_scenarios()
    built: list[Scenario] = []
    for scenario in base:
        built.append(scenario)
        if streaming:
            built.append(
                Scenario(
                    name=scenario.name,
                    body=scenario.body,
                    stream=True,
                    per_model=scenario.per_model,
                    expect_status=scenario.expect_status,
                    expect_finish=scenario.expect_finish,
                    expect_tool_call=scenario.expect_tool_call,
                    expect_content=scenario.expect_content,
                    timeout_seconds=scenario.timeout_seconds,
                )
            )
    built.extend(_hostile_scenarios())
    return built


@dataclass(frozen=True, slots=True)
class Observation:
    """What one request produced, before it is judged."""

    status: int
    error_type: str | None = None
    finish_reason: str | None = None
    tool_calls: int = 0
    content_chars: int = 0
    request_id: str | None = None
    duration_ms: float = 0.0
    ttft_ms: float | None = None
    stream_done: bool = False
    timed_out: bool = False
    transport_error: str | None = None


def classify(scenario: Scenario, observation: Observation) -> tuple[str, str]:
    """Return the verdict bucket and a short reason for one observation."""
    if observation.timed_out:
        return HARNESS_TIMEOUT, f"no response within {scenario.timeout_seconds:.0f}s"
    if observation.transport_error is not None:
        return BRIDGE_DEFECT, f"transport error: {observation.transport_error}"
    error_type = observation.error_type
    if observation.status not in scenario.expect_status:
        if error_type == "model_not_found":
            return ACCOUNT_POLICY, "model unavailable for this account"
        if error_type == "factory_native_tool_blocked":
            return MODEL_BEHAVIOR, "model attempted a Factory-native tool"
        if observation.status == 429:
            return CAPACITY, "bridge queue is full"
        if observation.status == 503:
            return PROVIDER_UNAVAILABLE, "Droid executable or provider unavailable"
        if observation.status == 504:
            return BACKEND_TIMEOUT, "backend timed out"
        return BRIDGE_DEFECT, f"unexpected status {observation.status} ({error_type})"
    if observation.status != 200:
        return SUCCESS, f"rejected with {observation.status} as expected"
    if scenario.stream and not observation.stream_done:
        return BRIDGE_DEFECT, "stream ended without [DONE]"
    if observation.finish_reason not in scenario.expect_finish:
        if observation.finish_reason == "tool_calls":
            return MODEL_BEHAVIOR, "model called a tool when none was required"
        return BRIDGE_DEFECT, f"unexpected finish_reason {observation.finish_reason}"
    if scenario.expect_tool_call and observation.tool_calls == 0:
        return MODEL_BEHAVIOR, "model produced no client tool call"
    if scenario.expect_content and observation.content_chars == 0:
        return MODEL_BEHAVIOR, "model produced no content"
    return SUCCESS, "contract satisfied"


def _error_type(body: dict[str, Any]) -> str | None:
    error = body.get("error")
    if not isinstance(error, dict):
        return None
    value = error.get("type")
    return value if isinstance(value, str) else None


def _read_json_observation(
    response: httpx.Response,
    *,
    duration_ms: float,
) -> Observation:
    try:
        body = response.json()
    except ValueError:
        return Observation(
            status=response.status_code,
            duration_ms=duration_ms,
            transport_error="response was not JSON",
        )
    if not isinstance(body, dict):
        return Observation(
            status=response.status_code,
            duration_ms=duration_ms,
            transport_error="response was not a JSON object",
        )
    choices = body.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    tool_calls = message.get("tool_calls") or []
    content = message.get("content") or ""
    return Observation(
        status=response.status_code,
        error_type=_error_type(body),
        finish_reason=choice.get("finish_reason") if isinstance(choice, dict) else None,
        tool_calls=len(tool_calls) if isinstance(tool_calls, list) else 0,
        content_chars=len(content) if isinstance(content, str) else 0,
        request_id=response.headers.get("x-request-id"),
        duration_ms=duration_ms,
    )


def _stream_observation(
    status: int,
    headers: httpx.Headers,
    events: list[tuple[float, str]],
    *,
    duration_ms: float,
) -> Observation:
    finish_reason: str | None = None
    tool_calls = 0
    content_chars = 0
    error_type: str | None = None
    stream_done = False
    ttft_ms: float | None = None
    for elapsed_ms, raw in events:
        if raw == "[DONE]":
            stream_done = True
            continue
        try:
            chunk = json.loads(raw)
        except ValueError:
            return Observation(
                status=status,
                duration_ms=duration_ms,
                transport_error="stream carried invalid JSON",
            )
        if not isinstance(chunk, dict):
            continue
        error_type = error_type or _error_type(chunk)
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            text = delta.get("content") or ""
            if text:
                content_chars += len(text)
                ttft_ms = elapsed_ms if ttft_ms is None else ttft_ms
            calls = delta.get("tool_calls") or []
            if calls:
                tool_calls += len(calls)
                ttft_ms = elapsed_ms if ttft_ms is None else ttft_ms
            finish_reason = choice.get("finish_reason") or finish_reason
    return Observation(
        status=status,
        error_type=error_type,
        finish_reason=finish_reason,
        tool_calls=tool_calls,
        content_chars=content_chars,
        request_id=headers.get("x-request-id"),
        duration_ms=duration_ms,
        ttft_ms=ttft_ms,
        stream_done=stream_done,
    )


@dataclass(frozen=True, slots=True)
class Bridge:
    """Thin client for the bridge under test."""

    base_url: str
    api_key: str | None
    client: httpx.AsyncClient

    @property
    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def models(self) -> list[str]:
        response = await self.client.get(
            f"{self.base_url}/v1/models",
            headers=self.headers,
            timeout=60.0,
        )
        response.raise_for_status()
        body = response.json()
        return [entry["id"] for entry in body.get("data", []) if isinstance(entry, dict)]

    async def run(self, model: str, scenario: Scenario) -> Observation:
        body: dict[str, Any] = {"model": model, **scenario.body}
        if scenario.stream:
            body["stream"] = True
        started = time.perf_counter()
        try:
            if scenario.stream:
                return await self._run_stream(body, scenario, started)
            response = await self.client.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json=body,
                timeout=scenario.timeout_seconds,
            )
        except httpx.TimeoutException:
            return Observation(
                status=0,
                duration_ms=(time.perf_counter() - started) * 1000,
                timed_out=True,
            )
        except httpx.HTTPError as exc:
            return Observation(
                status=0,
                duration_ms=(time.perf_counter() - started) * 1000,
                transport_error=type(exc).__name__,
            )
        return _read_json_observation(
            response,
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    async def _run_stream(
        self,
        body: dict[str, Any],
        scenario: Scenario,
        started: float,
    ) -> Observation:
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json=body,
                timeout=scenario.timeout_seconds,
            ) as response:
                events = [
                    ((time.perf_counter() - started) * 1000, line.removeprefix("data: "))
                    async for line in response.aiter_lines()
                    if line.startswith("data: ")
                ]
                status = response.status_code
                headers = response.headers
        except httpx.TimeoutException:
            return Observation(
                status=0,
                duration_ms=(time.perf_counter() - started) * 1000,
                timed_out=True,
            )
        except httpx.HTTPError as exc:
            return Observation(
                status=0,
                duration_ms=(time.perf_counter() - started) * 1000,
                transport_error=type(exc).__name__,
            )
        return _stream_observation(
            status,
            headers,
            events,
            duration_ms=(time.perf_counter() - started) * 1000,
        )


def row(model: str, scenario: Scenario, observation: Observation) -> dict[str, Any]:
    """One JSONL record: the request, what came back, and the verdict."""
    verdict, detail = classify(scenario, observation)
    return {
        "ts": datetime.now(UTC).isoformat(),
        "model": model,
        "scenario": scenario.name,
        "stream": scenario.stream,
        "status": observation.status,
        "error_type": observation.error_type,
        "finish_reason": observation.finish_reason,
        "tool_calls": observation.tool_calls,
        "content_chars": observation.content_chars,
        "request_id": observation.request_id,
        "duration_ms": round(observation.duration_ms, 1),
        "ttft_ms": None if observation.ttft_ms is None else round(observation.ttft_ms, 1),
        "verdict": verdict,
        "detail": detail,
    }


async def run_matrix(
    bridge: Bridge,
    models: list[str],
    plan: list[Scenario],
    *,
    concurrency: int,
    on_row: Any = None,
) -> list[dict[str, Any]]:
    """Run every applicable (model, scenario) pair and return the rows."""
    jobs: list[tuple[str, Scenario]] = []
    for scenario in plan:
        if scenario.per_model:
            jobs.extend((model, scenario) for model in models)
        else:
            jobs.append((models[0] if models else "factory-droid", scenario))
    semaphore = asyncio.Semaphore(concurrency)
    rows: list[dict[str, Any]] = []

    async def execute(model: str, scenario: Scenario) -> None:
        async with semaphore:
            observation = await bridge.run(model, scenario)
        record = row(model, scenario, observation)
        rows.append(record)
        if on_row is not None:
            on_row(record)

    await asyncio.gather(*(execute(model, scenario) for model, scenario in jobs))
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate verdicts, blocking failures and latency for a run."""
    counts = dict.fromkeys(VERDICTS, 0)
    for record in rows:
        verdict = str(record.get("verdict"))
        counts[verdict] = counts.get(verdict, 0) + 1
    durations = [float(record["duration_ms"]) for record in rows if record.get("status") == 200]
    blocking = [record for record in rows if record.get("verdict") == BRIDGE_DEFECT]
    return {
        "total": len(rows),
        "counts": counts,
        "pass_rate": (counts[SUCCESS] / len(rows)) if rows else 0.0,
        "median_ms": round(statistics.median(durations), 1) if durations else None,
        "slowest_ms": round(max(durations), 1) if durations else None,
        "blocking": blocking,
    }


def render_report(rows: list[dict[str, Any]]) -> str:
    """Markdown summary, generated so no report is ever written by hand."""
    summary = summarize(rows)
    lines = [
        "# Bridge e2e run",
        "",
        f"- Requests: {summary['total']}",
        f"- Pass rate: {summary['pass_rate']:.1%}",
        f"- Median completed latency: {summary['median_ms']} ms",
        f"- Slowest completed: {summary['slowest_ms']} ms",
        "",
        "| Verdict | Count |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| `{verdict}` | {count} |"
        for verdict, count in summary["counts"].items()
        if count or verdict in {SUCCESS, BRIDGE_DEFECT}
    )
    lines.extend(["", "## Blocking findings", ""])
    if not summary["blocking"]:
        lines.append("None. No result classified as a bridge defect.")
    else:
        lines.extend(["| Model | Scenario | Stream | Detail |", "| --- | --- | --- | --- |"])
        lines.extend(
            f"| `{record['model']}` | `{record['scenario']}` | {record['stream']} "
            f"| {record['detail']} |"
            for record in summary["blocking"]
        )
    lines.extend(["", "## Scenarios", "", "| Scenario | Pass | Total |", "| --- | ---: | ---: |"])
    lines.extend(
        f"| `{name}` | {sum(1 for record in group if record['verdict'] == SUCCESS)} "
        f"| {len(group)} |"
        for name, group in sorted(_group_by(rows, "scenario").items())
    )
    # Tool-call compliance is the number a prompt change is judged on, so it
    # gets its own column instead of hiding inside the pass rate.
    lines.extend(
        [
            "",
            "## Models",
            "",
            "| Model | Pass | Total | Tool calls |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| `{name}` | {sum(1 for record in group if record['verdict'] == SUCCESS)} "
        f"| {len(group)} | {sum(int(record['tool_calls']) for record in group)} |"
        for name, group in sorted(_group_by(rows, "model").items())
    )
    return "\n".join(lines) + "\n"


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in rows:
        grouped.setdefault(str(record[key]), []).append(record)
    return grouped


def _key(record: dict[str, Any]) -> tuple[str, str, bool]:
    return (str(record["model"]), str(record["scenario"]), bool(record["stream"]))


def compare(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, Any]:
    """Verdict transitions and latency delta between two runs."""
    left = {_key(record): record for record in before}
    right = {_key(record): record for record in after}
    regressions = []
    fixes = []
    for key, new in right.items():
        old = left.get(key)
        if old is None:
            continue
        if old["verdict"] == SUCCESS and new["verdict"] != SUCCESS:
            regressions.append({"key": key, "to": new["verdict"], "detail": new["detail"]})
        elif old["verdict"] != SUCCESS and new["verdict"] == SUCCESS:
            fixes.append({"key": key, "from": old["verdict"]})
    return {
        "shared": len(set(left) & set(right)),
        "only_before": sorted(str(key) for key in set(left) - set(right)),
        "only_after": sorted(str(key) for key in set(right) - set(left)),
        "regressions": regressions,
        "fixes": fixes,
        "pass_rate_before": summarize(before)["pass_rate"],
        "pass_rate_after": summarize(after)["pass_rate"],
    }


def render_comparison(result: dict[str, Any]) -> str:
    lines = [
        "# Bridge e2e comparison",
        "",
        f"- Shared cases: {result['shared']}",
        f"- Pass rate: {result['pass_rate_before']:.1%} -> {result['pass_rate_after']:.1%}",
        f"- Regressions: {len(result['regressions'])}",
        f"- Fixes: {len(result['fixes'])}",
        "",
    ]
    if result["regressions"]:
        lines.extend(["## Regressions", "", "| Case | Now | Detail |", "| --- | --- | --- |"])
        lines.extend(
            f"| `{entry['key']}` | `{entry['to']}` | {entry['detail']} |"
            for entry in result["regressions"]
        )
        lines.append("")
    if result["fixes"]:
        lines.extend(["## Fixes", "", "| Case | Was |", "| --- | --- |"])
        lines.extend(f"| `{entry['key']}` | `{entry['from']}` |" for entry in result["fixes"])
        lines.append("")
    return "\n".join(lines)


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


@dataclass(slots=True)
class RunOptions:
    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = None
    models: list[str] = field(default_factory=list)
    concurrency: int = 2
    streaming: bool = True
    out: Path | None = None


async def _run(options: RunOptions) -> int:
    out = options.out or Path("traces") / f"e2e-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient() as client:
        bridge = Bridge(base_url=options.base_url, api_key=options.api_key, client=client)
        models = options.models or await bridge.models()
        plan = scenarios(streaming=options.streaming)
        print(f"{len(models)} models x {len(plan)} scenarios -> {out}", file=sys.stderr)
        with out.open("a", encoding="utf-8") as handle:

            def write(record: dict[str, Any]) -> None:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

            rows = await run_matrix(
                bridge,
                models,
                plan,
                concurrency=options.concurrency,
                on_row=write,
            )
    print(render_report(rows))
    return 1 if summarize(rows)["blocking"] else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="run the matrix against a live bridge")
    run_parser.add_argument("--base-url", default=os.getenv("BRIDGE_URL", DEFAULT_BASE_URL))
    run_parser.add_argument("--models", default="", help="comma separated; default is discovery")
    run_parser.add_argument("--concurrency", type=int, default=2)
    run_parser.add_argument("--no-stream", action="store_true")
    run_parser.add_argument("--out", type=Path, default=None)

    report_parser = sub.add_parser("report", help="render a markdown report from a JSONL run")
    report_parser.add_argument("path", type=Path)

    compare_parser = sub.add_parser("compare", help="diff two JSONL runs")
    compare_parser.add_argument("before", type=Path)
    compare_parser.add_argument("after", type=Path)

    args = parser.parse_args(argv)
    if args.command == "report":
        print(render_report(load_rows(args.path)), end="")
        return 0
    if args.command == "compare":
        result = compare(load_rows(args.before), load_rows(args.after))
        print(render_comparison(result), end="")
        return 1 if result["regressions"] else 0
    options = RunOptions(
        base_url=args.base_url,
        api_key=os.getenv("FACTORY_DROID_OPENAI_API_KEY"),
        models=[value for value in args.models.split(",") if value],
        concurrency=args.concurrency,
        streaming=not args.no_stream,
        out=args.out,
    )
    return asyncio.run(_run(options))


if __name__ == "__main__":
    raise SystemExit(main())
