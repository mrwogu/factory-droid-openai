"""Turn a recorded bridge session into offline replay fixtures.

Run the bridge with payload tracing on, run the matrix, then join the two
files: the matrix rows say what each request was and how it was judged, the
trace says which Droid events produced it.

    FACTORY_DROID_OPENAI_TRACE_PAYLOADS=full \\
    FACTORY_DROID_OPENAI_TRACE_PAYLOAD_FILE=traces/trace.jsonl \\
    uv run factory-droid-openai
    uv run python scripts/e2e_matrix.py run --out traces/run.jsonl
    uv run python scripts/e2e_fixtures.py build \\
        --trace traces/trace.jsonl --run traces/run.jsonl

Every fixture is a real model dialect frozen in time, so tests/test_replay.py
re-checks the whole contract offline instead of spending live requests.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "events"
TRACE_EVENT = "droid.event"
# Fixtures are regression evidence, so only outcomes that a replay can assert
# without failing the suite on purpose are exported by default.
DEFAULT_VERDICTS: tuple[str, ...] = ("pass",)
_SLUG = re.compile(r"[^a-z0-9]+")


def slug(value: str) -> str:
    return _SLUG.sub("-", value.lower()).strip("-")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if isinstance(record, dict):
            records.append(record)
    return records


def group_events(trace: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Collect the SDK events of each request, in the order they arrived."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in trace:
        if record.get("event") != TRACE_EVENT:
            continue
        payload = record.get("payload")
        request_id = record.get("request_id")
        if not isinstance(payload, str) or not isinstance(request_id, str):
            # Head mode truncates payloads, so only full traces can be replayed.
            continue
        grouped.setdefault(request_id, []).append(json.loads(payload))
    return grouped


def fixture_name(row: dict[str, Any]) -> str:
    suffix = "-stream" if row.get("stream") else ""
    return f"{slug(str(row['scenario']))}--{slug(str(row['model']))}{suffix}.jsonl"


def build_fixtures(
    rows: list[dict[str, Any]],
    events: dict[str, list[dict[str, Any]]],
    *,
    verdicts: tuple[str, ...] = DEFAULT_VERDICTS,
    scenarios: tuple[str, ...] = (),
) -> dict[str, list[dict[str, Any]]]:
    """Pair matrix rows with their recorded events and stamp each with its contract."""
    fixtures: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        request_id = row.get("request_id")
        recorded = events.get(request_id) if isinstance(request_id, str) else None
        if not recorded:
            continue
        if row.get("verdict") not in verdicts:
            continue
        if scenarios and str(row["scenario"]) not in scenarios:
            continue
        meta = {
            "kind": "meta",
            "scenario": row["scenario"],
            "stream": bool(row["stream"]),
            "recorded_model": row["model"],
            "expect_verdict": row["verdict"],
        }
        fixtures[fixture_name(row)] = [meta, *recorded]
    return fixtures


def write_fixtures(fixtures: dict[str, list[dict[str, Any]]], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, lines in sorted(fixtures.items()):
        path = out_dir / name
        path.write_text(
            "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines),
            encoding="utf-8",
        )
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="write replay fixtures from a recorded run")
    build.add_argument("--trace", type=Path, required=True)
    build.add_argument("--run", type=Path, required=True)
    build.add_argument("--out", type=Path, default=FIXTURE_DIR)
    build.add_argument("--verdicts", default=",".join(DEFAULT_VERDICTS))
    build.add_argument("--scenarios", default="", help="comma separated; default is every one")

    args = parser.parse_args(argv)
    fixtures = build_fixtures(
        load_jsonl(args.run),
        group_events(load_jsonl(args.trace)),
        verdicts=tuple(value for value in args.verdicts.split(",") if value),
        scenarios=tuple(value for value in args.scenarios.split(",") if value),
    )
    written = write_fixtures(fixtures, args.out)
    for path in written:
        print(path)
    if not written:
        print("no fixture matched: check the trace mode and the verdict filter")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
