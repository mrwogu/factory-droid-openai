from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Literal

from factory_droid_openai.logs import debug as log_debug
from factory_droid_openai.logs import millis
from factory_droid_openai.logs import warning as log_warning
from factory_droid_openai.runner import RunComplete, RunnerError, RunRequest, TextDelta

if TYPE_CHECKING:
    from collections.abc import Callable

    from factory_droid_openai.metrics import BridgeMetrics
    from factory_droid_openai.runner import DroidRunner

AuthProbeStatus = Literal["ok", "degraded", "unknown"]

# A probe turn only needs one short answer, so a healthy exec has no reason to
# approach the request timeout ceiling. Hanging spawns fail here instead of
# delaying the next probe by minutes.
_PROBE_TIMEOUT_CEILING_SECONDS = 60.0

_PROBE_PROMPT = "Reply with exactly: ok"


class AuthProbe:
    """Periodic exec-based check that the bridge's Factory key still works.

    Model catalog discovery stays healthy while key-authenticated exec fails
    (issue #122), so the probe has to run a real one-turn Droid exec. An empty
    answer counts as a failure: a dead key surfaces as empty 200 completions
    before it surfaces as an error.
    """

    def __init__(
        self,
        *,
        runner_factory: Callable[[], DroidRunner],
        model_alias: str,
        timeout_seconds: float,
        interval_seconds: float,
        failure_threshold: int,
        metrics: BridgeMetrics | None = None,
    ) -> None:
        self._runner_factory = runner_factory
        self._model_alias = model_alias
        self._timeout_seconds = timeout_seconds
        self._interval_seconds = interval_seconds
        self._failure_threshold = failure_threshold
        self._metrics = metrics
        self._trigger = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._consecutive_failures = 0
        self._status: AuthProbeStatus = "unknown"

    @property
    def status(self) -> AuthProbeStatus:
        """Last known state of the Factory key, for the health endpoint."""
        return self._status

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def failing(self) -> bool:
        """Whether the gate should fail fast instead of spawning doomed sessions."""
        return self._consecutive_failures >= self._failure_threshold

    def start(self) -> None:
        if self._task is not None or self._interval_seconds <= 0:
            return
        self._task = asyncio.create_task(self._loop(), name="factory-droid-openai-auth-probe")

    def record_success(self) -> None:
        """A completion that carried content proves the key works."""
        self._consecutive_failures = 0
        self._status = "ok"

    def suspect(self) -> None:
        """Ask for an immediate probe after an auth-shaped or empty completion."""
        self._trigger.set()

    async def aclose(self) -> None:
        task = self._task
        if task is None:
            return
        self._task = None
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _loop(self) -> None:
        while True:
            await self._probe_once()
            # Clearing after the wait keeps a suspect() raised while the probe
            # was running from being swallowed by the clear.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._trigger.wait(), timeout=self._interval_seconds)
            self._trigger.clear()

    async def _probe_once(self) -> None:
        started = time.perf_counter()
        try:
            reason = await self._run_probe()
        except Exception as exc:  # the probe must never kill the app
            reason = f"Auth probe crashed: {exc}"
        elapsed_ms = millis(time.perf_counter() - started)
        if reason is None:
            self._consecutive_failures = 0
            self._status = "ok"
            if self._metrics is not None:
                self._metrics.increment_auth_probe_successes()
            log_debug("auth.probe_ok", elapsed_ms=elapsed_ms)
            return
        self._consecutive_failures += 1
        self._status = "degraded"
        if self._metrics is not None:
            self._metrics.increment_auth_probe_failures()
        log_warning(
            "auth.probe_failed",
            reason=reason,
            consecutive=self._consecutive_failures,
            threshold=self._failure_threshold,
            elapsed_ms=elapsed_ms,
        )

    async def _run_probe(self) -> str | None:
        """Run one minimal Droid exec; ``None`` means the key answered."""
        runner = self._runner_factory()
        budget = min(self._timeout_seconds, _PROBE_TIMEOUT_CEILING_SECONDS)
        request = RunRequest(
            prompt=_PROBE_PROMPT,
            model=self._model_alias,
            model_alias=self._model_alias,
            reasoning_effort=None,
            timeout_seconds=budget,
        )
        saw_text = False
        completed = False
        try:
            # The outer bound catches a runner whose own deadline never fires,
            # so one stuck exec cannot stall the probe loop for good.
            async with asyncio.timeout(budget):
                async for event in runner.run(request):
                    if isinstance(event, TextDelta):
                        saw_text = saw_text or bool(event.text)
                    elif isinstance(event, RunComplete):
                        completed = True
        except RunnerError as exc:
            return str(exc)
        except TimeoutError:
            return "Auth probe timed out."
        if not saw_text:
            return "Auth probe completed without any assistant text."
        if not completed:
            return "Auth probe ended without a completion event."
        return None
