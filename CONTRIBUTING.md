# Contributing

Contributions are welcome when they improve OpenAI compatibility, Factory Droid
integration, reliability, security, or developer experience.

## Development setup

Requirements:

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- Factory Droid CLI only for manual end-to-end testing

Clone and install all locked dependencies:

```bash
git clone https://github.com/mrwogu/factory-droid-openai.git
cd factory-droid-openai
uv sync --all-extras
```

Unit and HTTP integration tests use fake SDK clients. They do not start Droid,
access Factory services, or require authentication.

## Validation

Run every required check before opening a pull request:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run python scripts/generate_openapi.py
uv run openapi-spec-validator openapi.json
uv run pytest --cov=factory_droid_openai --cov-report=term-missing
uv build
prs validate
prs diff --all --no-color
prs check
```

Apply formatting with:

```bash
uv run ruff format .
```

CI runs tests on every supported Python version.

## End-to-end matrix

The default suite never talks to Factory. Behavior that only shows up with a
real model is covered by a separate, manual loop: run the matrix, read the
verdicts, fix, run again, compare.

Start a bridge with payload tracing on, then run every scenario against every
model the bridge lists:

```bash
export FACTORY_DROID_OPENAI_TRACE_PAYLOADS=full
export FACTORY_DROID_OPENAI_TRACE_FILE="$PWD/traces/trace.jsonl"
env -u FACTORY_APPEND_SYSTEM_PROMPT uv run factory-droid-openai &
uv run python scripts/e2e_matrix.py run --out traces/run-a.jsonl
```

Run it from a plain shell. Inside a Droid session,
`FACTORY_APPEND_SYSTEM_PROMPT` is exported and the bridge inherits it, so every
recorded turn answers a different prompt than a user of the bridge would send.
`~/.factory/system-prompt.md` and enabled skills reach the model the same way,
and no CLI flag turns them off: a run measures this machine's Droid profile,
not a default one. For recordings that have to be profile-neutral, point
`FACTORY_HOME_OVERRIDE` at a directory with its own `droid login` and no custom
system prompt.

Each request becomes one JSONL row with a verdict. Only `bridge_defect` fails
the run; `model_behavior`, `account_policy`, `capacity`,
`provider_unavailable`, and `backend_timeout` describe the environment, not the
code. Re-render a report or diff two runs after a change:

After the parallel per-model cases, the harness runs a serialized model-switch
ring against the same bridge process. Each pair primes the warm pool with one
model, waits five seconds for refill, then sends a required tool-call request
to the next model in both transport modes. Use
`--switch-settle-seconds SECONDS` when the bridge needs a different refill
window. Switch rows include `source_model` and prime-request diagnostics.

To verify explicit session continuity as well, start the bridge with
`FACTORY_DROID_OPENAI_SESSION_CONTINUITY=true` and add
`--test-session-continuity` to the matrix command. That phase creates a session
with one model and confirms that continuing its ID with different settings is
rejected instead of silently running under a stale model or reasoning effort.

```bash
uv run python scripts/e2e_matrix.py report traces/run-a.jsonl
uv run python scripts/e2e_matrix.py compare traces/run-a.jsonl traces/run-b.jsonl
```

`compare` exits non-zero on any pass to non-pass transition, which makes it the
A/B tool for prompt and parser changes.

The matrix talks to whatever bridge answers on `--base-url`, so comparing the
two tool-calling paths means running it twice against two bridges that differ
only in the flag:

```bash
export FACTORY_HOME_OVERRIDE="$PWD/traces/profile"
env -u FACTORY_APPEND_SYSTEM_PROMPT uv run factory-droid-openai --port 8798 &
FACTORY_DROID_OPENAI_NATIVE_TOOL_CALLS=true \
  env -u FACTORY_APPEND_SYSTEM_PROMPT uv run factory-droid-openai --port 8799 &
uv run python scripts/e2e_matrix.py run --base-url http://127.0.0.1:8798 --out traces/text.jsonl
uv run python scripts/e2e_matrix.py run --base-url http://127.0.0.1:8799 --out traces/native.jsonl
uv run python scripts/e2e_matrix.py compare traces/text.jsonl traces/native.jsonl
```

Native runs need the profile override. A tool-bearing request publishes an MCP
server to Droid, which then waits for every other MCP server the profile
configures, and one stuck on a login prompt exhausts the session-init timeout.
Both bridges must run under the same profile, or the comparison measures the
profile instead of the flag.

Read the transitions, not the pass rates: a model that answers a tool contract
in one path and not the other is the finding. `model_behavior` on the native
path still means the model chose that, but a native run that reports a client
tool as a Factory-native tool is a bridge defect in the prefix handling.

Turn what a live run found into offline regression tests:

```bash
uv run python scripts/e2e_fixtures.py build \
  --trace traces/trace.jsonl --run traces/run-a.jsonl \
  --models kimi-k3 --scenarios tool_auto,tool_parallel
```

Each fixture in `tests/fixtures/events/` holds one recorded Droid event stream
plus the contract it satisfied, and `tests/test_replay.py` replays all of them
through the ASGI app on every test run. Live runs stay in `traces/`, which is
gitignored: rows and traces carry model output and account-specific denials.
Every recovery path justified by live output should include its sanitized
full-trace fixture. Use `seed` as the recorded model for hand-seeded fixtures;
a seed can extend negative coverage but does not replace a production capture.
Keep one fixture per scenario, model, and stream mode so replay IDs stay unique.

Reasoning deltas are dropped from fixtures, because models quote the operator's
instructions back inside them; `--keep-reasoning` overrides that for local
debugging only. Fixtures are committed, so read them before adding them and
delete anything that describes your machine rather than the bridge.

## Change guidelines

- Keep the bridge local-first and OpenAI-compatible.
- Preserve streaming and non-streaming response contracts.
- Never allow Factory-native tools to bypass client-owned tool execution.
- Add behavior tests for every user-visible change.
- Keep dependencies bounded and regenerate `uv.lock`.
- Avoid live network calls in tests.
- Never commit credentials, private prompts, generated packages, or local logs.

## Commit messages

Use Conventional Commits with a subject no longer than 70 characters:

```text
feat: add response field compatibility
fix: interrupt Droid after client disconnect
test: cover split tool-call markers
docs: document bearer authentication
```

## Pull requests

Include:

- Problem and intended behavior
- Compatibility impact
- Security impact
- Tests added or updated
- Manual validation, when applicable

Keep pull requests focused. Separate unrelated refactors from behavior changes.

## Reporting bugs

Use the bug report form and include a minimal sanitized request. Remove API
keys, access tokens, private filesystem paths, and sensitive prompt content.

Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).
