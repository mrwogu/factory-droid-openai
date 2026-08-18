import asyncio
import contextlib
from collections import defaultdict, deque
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from factory_droid_openai.logs import debug as log_debug
from factory_droid_openai.logs import warning as log_warning
from factory_droid_openai.mcp_tools import (
    NativeToolBinding,
    NativeToolCatalog,
    NativeToolRegistry,
)
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


@dataclass(frozen=True, slots=True)
class _WarmDemand:
    key: SessionKey
    catalog: NativeToolCatalog | None = None


def _share(demands: Sequence[_WarmDemand], size: int) -> dict[_WarmDemand, int]:
    """Hand ``size`` sessions to ``demands``, most recent first."""
    if not demands:
        return {}
    base, extra = divmod(size, len(demands))
    return {demand: base + (1 if index < extra else 0) for index, demand in enumerate(demands)}


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
        native_registry: NativeToolRegistry | None = None,
    ) -> None:
        self._runner_factory = runner_factory
        self._reaper = reaper
        self._size = max(0, size)
        self._warm_timeout_seconds = warm_timeout_seconds
        self._ttl_seconds = ttl_seconds
        self._max_keys = max(1, max_keys)
        self._retry_seconds = retry_seconds
        self._metrics = metrics
        self._native_registry = native_registry
        self._sessions: dict[_WarmDemand, deque[WarmSession]] = defaultdict(deque)
        self._native_sessions: dict[_WarmDemand, deque[WarmSession]] = defaultdict(deque)
        self._wanted: list[SessionKey] = []
        self._native_wanted: list[_WarmDemand] = []
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
        sessions = [
            session
            for queues in (self._sessions, self._native_sessions)
            for queue in queues.values()
            for session in queue
        ]
        self._sessions.clear()
        self._native_sessions.clear()
        self._wanted.clear()
        self._native_wanted.clear()
        self._publish()
        for session in sessions:
            self._reaper.submit(self._discard_session(session))

    def note(self, key: SessionKey, catalog: NativeToolCatalog | None = None) -> None:
        demand = _WarmDemand(key, catalog)
        if catalog is None:
            if key in self._wanted:
                self._wanted.remove(key)
            self._wanted.insert(0, key)
            for stale_key in self._wanted[self._max_keys :]:
                self._drop_demand(_WarmDemand(stale_key))
            del self._wanted[self._max_keys :]
        else:
            if demand in self._native_wanted:
                self._native_wanted.remove(demand)
            self._native_wanted.insert(0, demand)
            for stale_demand in self._native_wanted[self._max_keys :]:
                self._drop_demand(stale_demand)
            del self._native_wanted[self._max_keys :]
        self._wake.set()

    def offer(self, session: WarmSession) -> None:
        """Register an already warmed session, or discard it when unwanted."""
        demand = _WarmDemand(
            session.key,
            session.native_binding.catalog if session.native_binding is not None else None,
        )
        queues = self._native_sessions if demand.catalog is not None else self._sessions
        if len(queues.get(demand, ())) >= self._targets().get(demand, 0):
            self._discard(session)
            return
        queues[demand].append(session)
        self._publish()
        log_debug("pool.warmed", model=session.key.model_id, warm_sessions=self._total())

    def acquire(
        self,
        key: SessionKey,
        catalog: NativeToolCatalog | None = None,
    ) -> WarmSession | None:
        """Take a ready session for ``key``, or ``None`` when none is warm yet."""
        if not self.enabled:
            return None
        demand = _WarmDemand(key, catalog)
        self.note(key, catalog)
        session = self._take_demand(demand)
        if session is not None:
            self._publish()
            if self._metrics is not None:
                self._metrics.increment_warm_hits()
            log_debug("pool.hit", model=key.model_id, warm_sessions=self._total())
            return session
        session = self._take_retunable(demand)
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

    def _take_demand(self, demand: _WarmDemand) -> WarmSession | None:
        queues = self._native_sessions if demand.catalog is not None else self._sessions
        queue = queues.get(demand)
        while queue:
            session = queue.popleft()
            if self._usable(session):
                return session
            self._discard(session)
        return None

    def _take_retunable(self, demand: _WarmDemand) -> WarmSession | None:
        """Borrow a session the runner can safely repoint."""
        queues = self._native_sessions if demand.catalog is not None else self._sessions
        for warmed_demand, queue in queues.items():
            if (
                warmed_demand.catalog != demand.catalog
                or warmed_demand.key == demand.key
                or not demand.key.can_retune_from(warmed_demand.key)
            ):
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
            warmed = await asyncio.gather(*(self._warm(demand) for demand in deficits))
            for session in warmed:
                if session is not None:
                    self.offer(session)
            if any(session is None for session in warmed):
                await asyncio.sleep(self._retry_seconds)
                return

    async def _warm(self, demand: _WarmDemand) -> WarmSession | None:
        runner = self._runner_factory()
        binding: NativeToolBinding | None = None
        try:
            if demand.catalog is not None:
                if self._native_registry is None:
                    raise RuntimeError("native warm pool has no catalog registry")
                binding = self._native_registry.open_catalog(demand.catalog)
                self._native_registry.pin(binding.token)
                return await runner.warm(
                    demand.key,
                    timeout_seconds=self._warm_timeout_seconds,
                    native_tools=binding,
                )
            return await runner.warm(demand.key, timeout_seconds=self._warm_timeout_seconds)
        except asyncio.CancelledError:
            if binding is not None and self._native_registry is not None:
                self._native_registry.close(binding.token)
            raise
        except Exception as exc:
            if binding is not None and self._native_registry is not None:
                self._native_registry.close(binding.token)
            if self._metrics is not None:
                self._metrics.increment_warm_failures()
            log_warning("pool.warm_failed", model=demand.key.model_id, error=type(exc).__name__)
            # Settings Droid rejects, such as a model an organization policy
            # blocks, would otherwise be retried for as long as they stay in
            # the wanted list. Traffic that still works re-adds the key.
            self._forget_demand(demand)
            return None

    def _forget(self, key: SessionKey) -> None:
        self._forget_demand(_WarmDemand(key))

    def _forget_demand(self, demand: _WarmDemand) -> None:
        if demand.catalog is None:
            if demand.key in self._wanted:
                self._wanted.remove(demand.key)
        elif demand in self._native_wanted:
            self._native_wanted.remove(demand)
        self._drop_demand(demand)

    def _targets(self) -> dict[_WarmDemand, int]:
        """Split the pool size across the demands traffic is currently using.

        Text and native demands draw on separate halves of the pool. Sharing
        one budget let a client that rotates tool catalogs multiply the demand
        count until the plain text keys landed on a target of zero, and the
        next rebalance then discarded a live session the same traffic needed.
        """
        text = [_WarmDemand(key) for key in self._wanted]
        native = list(self._native_wanted)
        if not text or not native:
            return _share(text or native, self._size)
        native_size = self._size // 2
        return _share(text, self._size - native_size) | _share(native, native_size)

    def _deficits(self) -> list[_WarmDemand]:
        deficits: list[_WarmDemand] = []
        targets = self._targets()
        for demand, target in targets.items():
            queues = self._native_sessions if demand.catalog is not None else self._sessions
            missing = target - len(queues.get(demand, ()))
            deficits.extend([demand] * max(0, missing))
        return deficits

    def _rebalance(self) -> None:
        """Free capacity held for keys that traffic has moved away from."""
        targets = self._targets()
        for demand, target in reversed(list(targets.items())):
            queues = self._native_sessions if demand.catalog is not None else self._sessions
            queue = queues.get(demand)
            while queue is not None and len(queue) > target:
                self._discard(queue.pop())
        self._publish()

    def _usable(self, session: WarmSession) -> bool:
        if not session.is_alive():
            return False
        age = asyncio.get_running_loop().time() - session.created_at
        return age < self._ttl_seconds

    def _drop_stale(self) -> None:
        for queues in (self._sessions, self._native_sessions):
            for queue in queues.values():
                for session in tuple(queue):
                    if not self._usable(session):
                        queue.remove(session)
                        self._discard(session)
        self._publish()

    def _drop_demand(self, demand: _WarmDemand) -> None:
        queues = self._native_sessions if demand.catalog is not None else self._sessions
        for session in queues.pop(demand, ()):
            self._discard(session)
        self._publish()

    def _drop(self, key: SessionKey) -> None:
        self._drop_demand(_WarmDemand(key))

    def _discard(self, session: WarmSession) -> None:
        self._reaper.submit(self._discard_session(session))

    async def _discard_session(self, session: WarmSession) -> None:
        try:
            await self._runner_factory().discard(session)
        finally:
            if session.native_binding is not None and self._native_registry is not None:
                self._native_registry.close(session.native_binding.token)

    def _total(self) -> int:
        return sum(
            len(queue)
            for queues in (self._sessions, self._native_sessions)
            for queue in queues.values()
        )

    def _publish(self) -> None:
        if self._metrics is not None:
            self._metrics.set_warm_sessions(self._total())
