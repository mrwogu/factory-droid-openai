"""Replay recorded Droid event streams against the bridge, offline.

Each fixture under ``tests/fixtures/events`` is a real (or hand-seeded) model
dialect plus the contract its live run satisfied. Replaying them keeps every
finding from a live matrix run as a free regression test.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest

from factory_droid_openai.app import create_app
from factory_droid_openai.config import Settings
from factory_droid_openai.runner import (
    ReasoningDelta,
    RunComplete,
    RunEvent,
    RunRequest,
    SessionStarted,
    StatusUpdate,
    TextDelta,
    Usage,
    UsageUpdate,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import ModuleType

    from factory_droid_openai.app import RunnerFactory

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "events"
FIXTURES = sorted(FIXTURE_DIR.glob("*.jsonl"))
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "e2e_matrix.py"


@pytest.fixture(scope="module")
def e2e() -> ModuleType:
    spec = importlib.util.spec_from_file_location("e2e_matrix", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve string annotations through sys.modules, so a module
    # loaded straight from a path has to register before it executes.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ReplayRunner:
    """Feeds recorded events back into the bridge in the order Droid sent them."""

    def __init__(self, events: list[RunEvent]) -> None:
        self.events = events
        self.closed = False

    async def run(self, request: RunRequest) -> AsyncIterator[RunEvent]:
        del request
        try:
            for event in self.events:
                yield event
        finally:
            self.closed = True


def _usage(payload: dict[str, Any]) -> Usage:
    details = payload.get("prompt_tokens_details") or {}
    return Usage(
        input_tokens=int(payload.get("prompt_tokens", 0)),
        output_tokens=int(payload.get("completion_tokens", 0)),
        cache_read_tokens=int(details.get("cached_tokens", 0)),
    )


def _event(record: dict[str, Any]) -> RunEvent:
    kind = record["kind"]
    if kind == "text_delta":
        return TextDelta(record["text"])
    if kind == "reasoning_delta":
        return ReasoningDelta(record["text"])
    if kind == "usage":
        return UsageUpdate(_usage(record["usage"]))
    if kind == "run_complete":
        return RunComplete(_usage(record["usage"]))
    if kind == "status":
        return StatusUpdate(record["state"])
    if kind == "session_started":
        return SessionStarted("replay-session")
    raise AssertionError(f"unknown recorded event kind: {kind}")


def _scenario(e2e: ModuleType, meta: dict[str, Any]) -> Any:
    for scenario in e2e.scenarios():
        if scenario.name == meta["scenario"] and scenario.stream == meta["stream"]:
            return scenario
    raise AssertionError(f"fixture references an unknown scenario: {meta['scenario']}")


def test_the_fixture_directory_is_not_empty() -> None:
    assert FIXTURES, "replay fixtures are the offline safety net; do not delete them all"


def test_replay_fixture_identities_are_unique() -> None:
    identities: dict[tuple[str, str, bool], Path] = {}
    for path in FIXTURES:
        first = path.read_text(encoding="utf-8").splitlines()[0]
        meta = json.loads(first)
        assert meta["kind"] == "meta"
        identity = (
            meta["scenario"],
            meta["recorded_model"],
            meta["stream"],
        )
        assert identity not in identities, (
            f"{path.name} duplicates replay identity from {identities[identity].name}"
        )
        identities[identity] = path


@pytest.mark.asyncio
@pytest.mark.parametrize("path", FIXTURES, ids=lambda path: path.stem)
async def test_recorded_events_keep_satisfying_the_contract(
    e2e: ModuleType,
    tmp_path: Path,
    path: Path,
) -> None:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    meta, *events = records
    scenario = _scenario(e2e, meta)
    runner = ReplayRunner([_event(record) for record in events])
    app = create_app(
        Settings(workdir=tmp_path),
        runner_factory=cast("RunnerFactory", lambda: runner),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://replay",
    ) as client:
        bridge = e2e.Bridge(base_url="http://replay", api_key=None, client=client)
        observation = await bridge.run("factory-droid", scenario)

    verdict, detail = e2e.classify(scenario, observation)
    assert verdict == meta["expect_verdict"], f"{path.name}: {detail}"
