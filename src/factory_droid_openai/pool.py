import asyncio
import contextlib
from collections import defaultdict, deque
from collections.abc import Callable, Coroutine
from typing import Any, Protocol

from factory_droid_openai.logs import debug as log_debug
from factory_droid_openai.logs import warning as log_warning
from factory_droid_openai.metrics import WARM_RETUNE_EFFORT
from factory_droid_openai.runner import DroidRunner, SessionKey, WarmSession

RunnerFactory = Callable[[], DroidRunner]

# Upper bound on how long the refill loop sleeps between stale sweeps.
_MAX_IDLE_SWEEP_SECONDS = 30.0


class PoolMetrics(Protocol):
    def set_warm_sessions(self, count: int) -> None: ...

    def increment_warm_hits(self) -> None: ...

    def increment_warm_retunes(self, reason: str) -> None: ...

    def increment_warm_misses(self) -> None: ...

    def increment_warm_failures(self) -> None: ...

    def set_pending_reaps(self, count: int) -> None: ...


class BackgroundReaper:
    """Runs Droid teardown after the client already got its last token.

    Interrupting and killing a Droid process takes roughly the configured
    grace period, so keeping it on the response path adds that latency to
    every completion.
    """

    def __init__(self, *, metrics: PoolMetrics | None = None, max_pending: int = 32) -> None:
        self._metrics = metrics
        self._max_pending = max_pending
        self._tasks: set[asyncio.Task[None]] = set()

    def submit(self, coroutine: Coroutine[Any, Any, None]) -> None:
        if len(self._tasks) >= self._max_pending:
            log_warning("reaper.saturated", pending=len(self._tasks))
        task = asyncio.create_task(self._guard(coroutine))
        self._tasks.add(task)
        task.add_done_callback(self._forget)
        self._publish()

    async def drain(self, *, timeout: float = 10.0) -> None:
        while self._tasks:
            pending = tuple(self._tasks)
            done, _ = await asyncio.wait(pending, timeout=timeout)
            if not done:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                break

    async def _guard(self, coroutine: Coroutine[Any, Any, None]) -> None:
        try:
            await coroutine
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # teardown failures must not escape
            log_warning("droid.reap_failed", error=type(exc).__name__)

    def _forget(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        self._publish()

    def _publish(self) -> None:
        if self._metrics is not None:
            self._metrics.set_pending_reaps(len(self._tasks))


class WarmSessionPool:
    """Keeps initialized Droid sessions ready so requests skip session startup.

    Sessions are keyed by the settings they were initialized with. An exact
    key match serves a turn with no extra round trip; otherwise a session
    warmed for the same model can be repointed at another explicit reasoning
    effort. A model change waits for a fresh or exact-match session, because
    the tool-call template Droid installed at session start survives a retune.
    A session serves at most one turn, which keeps the bridge stateless.
    """

    def __init__(
        self,
        *,
        runner_factory: RunnerFactory,
        reaper: BackgroundReaper,
        size: int,
        warm_timeout_seconds: float,
        ttl_seconds: float = 600.0,
        max_keys: int = 2,
        retry_seconds: float = 5.0,
        metrics: PoolMetrics | None = None,
    ) -> None:
        self._runner_factory = runner_factory
        self._reaper = reaper
        self._size = max(0, size)
        self._warm_timeout_seconds = warm_timeout_seconds
        self._ttl_seconds = ttl_seconds
        self._max_keys = max(1, max_keys)
        self._retry_seconds = retry_seconds
        self._metrics = metrics
        self._sessions: dict[SessionKey, deque[WarmSession]] = defaultdict(deque)
        self._wanted: list[SessionKey] = []
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    @property
    def enabled(self) -> bool:
        return self._size > 0

    def start(self, *, initial_key: SessionKey | None = None) -> None:
        if not self.enabled or self._task is not None:
            return
        if initial_key is not None:
            self.note(initial_key)
        self._task = asyncio.create_task(self._loop())

    async def aclose(self) -> None:
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        sessions = [session for queue in self._sessions.values() for session in queue]
        self._sessions.clear()
        self._publish()
        for session in sessions:
            self._reaper.submit(self._runner_factory().discard(session))

    def note(self, key: SessionKey) -> None:
        if key in self._wanted:
            self._wanted.remove(key)
        self._wanted.insert(0, key)
        for stale in self._wanted[self._max_keys :]:
            self._drop(stale)
        del self._wanted[self._max_keys :]
        self._wake.set()

    def offer(self, session: WarmSession) -> None:
        """Register an already warmed session, or discard it when unwanted."""
        if len(self._sessions.get(session.key, ())) >= self._targets().get(session.key, 0):
            self._discard(session)
            return
        self._sessions[session.key].append(session)
        self._publish()
        log_debug("pool.warmed", model=session.key.model_id, warm_sessions=self._total())

    def acquire(self, key: SessionKey) -> WarmSession | None:
        """Take a ready session for ``key``, or ``None`` when none is warm yet."""
        if not self.enabled:
            return None
        self.note(key)
        session = self._take(key)
        if session is not None:
            self._publish()
            if self._metrics is not None:
                self._metrics.increment_warm_hits()
            log_debug("pool.hit", model=key.model_id, warm_sessions=self._total())
            return session
        session = self._take_retunable(key)
        if session is not None:
            self._publish()
            if self._metrics is not None:
                self._metrics.increment_warm_hits()
                self._metrics.increment_warm_retunes(WARM_RETUNE_EFFORT)
            log_debug(
                "pool.retune",
                model=key.model_id,
                reason=WARM_RETUNE_EFFORT,
                warmed_for_effort=session.key.reasoning_effort,
                warm_sessions=self._total(),
            )
            return session
        self._publish()
        if self._metrics is not None:
            self._metrics.increment_warm_misses()
        log_debug("pool.miss", model=key.model_id, warm_sessions=self._total())
        return None

    def _take(self, key: SessionKey) -> WarmSession | None:
        queue = self._sessions.get(key)
        while queue:
            session = queue.popleft()
            if self._usable(session):
                return session
            self._discard(session)
        return None

    def _take_retunable(self, key: SessionKey) -> WarmSession | None:
        """Borrow a session the runner can safely repoint."""
        for warmed_key, queue in self._sessions.items():
            if warmed_key == key or not key.can_retune_from(warmed_key):
                continue
            while queue:
                session = queue.popleft()
                if self._usable(session):
                    return session
                self._discard(session)
        return None

    async def _loop(self) -> None:
        while True:
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(self._refill_interval()):
                    await self._wake.wait()
            self._wake.clear()
            await self._refill()

    def _refill_interval(self) -> float:
        return min(_MAX_IDLE_SWEEP_SECONDS, max(1.0, self._ttl_seconds / 2))

    async def _refill(self) -> None:
        self._drop_stale()
        self._rebalance()
        while True:
            deficits = self._deficits()
            if not deficits:
                return
            # Droid startup is seconds of mostly idle waiting, so a burst that
            # drained the pool refills in one startup instead of N.
            warmed = await asyncio.gather(*(self._warm(key) for key in deficits))
            for session in warmed:
                if session is not None:
                    self.offer(session)
            if any(session is None for session in warmed):
                await asyncio.sleep(self._retry_seconds)
                return

    async def _warm(self, key: SessionKey) -> WarmSession | None:
        runner = self._runner_factory()
        try:
            return await runner.warm(key, timeout_seconds=self._warm_timeout_seconds)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._metrics is not None:
                self._metrics.increment_warm_failures()
            log_warning("pool.warm_failed", model=key.model_id, error=type(exc).__name__)
            # Settings Droid rejects, such as a model an organization policy
            # blocks, would otherwise be retried for as long as they stay in
            # the wanted list. Traffic that still works re-adds the key.
            self._forget(key)
            return None

    def _forget(self, key: SessionKey) -> None:
        if key in self._wanted:
            self._wanted.remove(key)
        self._drop(key)

    def _targets(self) -> dict[SessionKey, int]:
        """Split the pool size across the keys traffic is currently using."""
        if not self._wanted:
            return {}
        base, extra = divmod(self._size, len(self._wanted))
        return {key: base + (1 if index < extra else 0) for index, key in enumerate(self._wanted)}

    def _deficits(self) -> list[SessionKey]:
        deficits: list[SessionKey] = []
        for key, target in self._targets().items():
            missing = target - len(self._sessions.get(key, ()))
            deficits.extend([key] * max(0, missing))
        return deficits

    def _rebalance(self) -> None:
        """Free capacity held for keys that traffic has moved away from."""
        targets = self._targets()
        for key, target in reversed(list(targets.items())):
            queue = self._sessions.get(key)
            while queue is not None and len(queue) > target:
                self._discard(queue.pop())
        self._publish()

    def _usable(self, session: WarmSession) -> bool:
        if not session.is_alive():
            return False
        age = asyncio.get_running_loop().time() - session.created_at
        return age < self._ttl_seconds

    def _drop_stale(self) -> None:
        for queue in self._sessions.values():
            for session in tuple(queue):
                if not self._usable(session):
                    queue.remove(session)
                    self._discard(session)
        self._publish()

    def _drop(self, key: SessionKey) -> None:
        for session in self._sessions.pop(key, ()):
            self._discard(session)
        self._publish()

    def _discard(self, session: WarmSession) -> None:
        self._reaper.submit(self._runner_factory().discard(session))

    def _total(self) -> int:
        return sum(len(queue) for queue in self._sessions.values())

    def _publish(self) -> None:
        if self._metrics is not None:
            self._metrics.set_warm_sessions(self._total())
