from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

import pytest

from factory_droid_openai.mcp_tools import NativeToolBinding, NativeToolRegistry
from factory_droid_openai.metrics import BridgeMetrics
from factory_droid_openai.pool import (
    BackgroundReaper,
    PoolMetrics,
    WarmSessionPool,
    _WarmDemand,
)
from factory_droid_openai.runner import RunnerError, SessionKey, WarmSession

if TYPE_CHECKING:
    from factory_droid_openai.pool import RunnerFactory

KEY = SessionKey(model_id="model-a", reasoning_effort="high")
OTHER_KEY = SessionKey(model_id="model-b", reasoning_effort=None)
RETUNE_KEY = SessionKey(model_id="model-a", reasoning_effort="low")
CROSS_MODEL_KEY = SessionKey(model_id="model-b", reasoning_effort="high")


class FakeTransport:
    def __init__(self, *, reaped: bool = False) -> None:
        self.reaped = reaped

    def is_reaped(self) -> bool:
        return self.reaped


class FakeRunner:
    def __init__(self, *, fail: bool = False, log: list[str] | None = None) -> None:
        self.fail = fail
        self.log = log if log is not None else []
        self.warmed = 0
        self.native_bindings: list[NativeToolBinding | None] = []

    async def warm(
        self,
        key: SessionKey,
        *,
        timeout_seconds: float,
        native_tools: NativeToolBinding | None = None,
    ) -> WarmSession:
        del timeout_seconds
        if self.fail:
            raise RuntimeError("warm failed")
        self.warmed += 1
        self.native_bindings.append(native_tools)
        self.log.append(f"warm:{key.model_id}")
        return _session(
            key,
            created_at=asyncio.get_running_loop().time(),
            native_binding=native_tools,
        )

    async def discard(self, session: WarmSession) -> None:
        self.log.append(f"discard:{session.key.model_id}")


def _session(
    key: SessionKey = KEY,
    *,
    created_at: float = 0.0,
    reaped: bool = False,
    native_binding: NativeToolBinding | None = None,
) -> WarmSession:
    return WarmSession(
        key=key,
        client=cast("Any", object()),
        transport=cast("Any", FakeTransport(reaped=reaped)),
        session_id="session-1",
        created_at=created_at,
        native_binding=native_binding,
    )


def _pool(
    runner: FakeRunner,
    *,
    size: int = 1,
    ttl_seconds: float = 600.0,
    max_keys: int = 2,
    retry_seconds: float = 0.0,
    metrics: BridgeMetrics | None = None,
    native_registry: NativeToolRegistry | None = None,
) -> WarmSessionPool:
    return WarmSessionPool(
        runner_factory=cast("RunnerFactory", lambda: runner),
        reaper=BackgroundReaper(),
        size=size,
        warm_timeout_seconds=5.0,
        ttl_seconds=ttl_seconds,
        max_keys=max_keys,
        retry_seconds=retry_seconds,
        metrics=metrics,
        native_registry=native_registry,
    )


def _native_catalog(registry: NativeToolRegistry, name: str = "weather") -> Any:
    return registry.catalog_identity(({"name": name, "inputSchema": {"type": "object"}},))


@pytest.mark.asyncio
async def test_reaper_runs_submitted_teardown() -> None:
    metrics = BridgeMetrics()
    reaper = BackgroundReaper(metrics=metrics)
    done = asyncio.Event()

    async def teardown() -> None:
        done.set()

    reaper.submit(teardown())
    await reaper.drain()

    assert done.is_set()
    assert "factory_droid_openai_pending_reaps 0" in metrics.render()


@pytest.mark.asyncio
async def test_reaper_swallows_teardown_failures_and_warns_when_saturated() -> None:
    reaper = BackgroundReaper(max_pending=1)
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked() -> None:
        started.set()
        await release.wait()

    async def failing() -> None:
        raise RuntimeError("teardown exploded")

    reaper.submit(blocked())
    await started.wait()
    reaper.submit(failing())
    release.set()
    await reaper.drain()


@pytest.mark.asyncio
async def test_reaper_drain_cancels_stuck_teardown() -> None:
    reaper = BackgroundReaper()
    started = asyncio.Event()

    async def stuck() -> None:
        started.set()
        await asyncio.sleep(60)

    reaper.submit(stuck())
    await started.wait()
    await reaper.drain(timeout=0.01)


@pytest.mark.asyncio
async def test_disabled_pool_never_hands_out_sessions() -> None:
    runner = FakeRunner()
    pool = _pool(runner, size=0)

    pool.start(initial_key=KEY)

    assert pool.enabled is False
    assert pool.acquire(KEY) is None
    await pool.aclose()
    assert runner.warmed == 0


@pytest.mark.asyncio
async def test_pool_warms_requested_key_and_serves_it() -> None:
    metrics = BridgeMetrics()
    runner = FakeRunner()
    pool = _pool(runner, metrics=metrics)

    pool.start(initial_key=KEY)
    pool.start(initial_key=KEY)
    await asyncio.sleep(0.05)

    session = pool.acquire(KEY)
    assert session is not None
    assert session.key == KEY
    rendered = metrics.render()
    assert "factory_droid_openai_warm_session_hits_total 1" in rendered

    await pool.aclose()


@pytest.mark.asyncio
async def test_pool_reports_miss_for_unwarmed_key() -> None:
    metrics = BridgeMetrics()
    pool = _pool(FakeRunner(), metrics=metrics)

    assert pool.acquire(KEY) is None
    assert "factory_droid_openai_warm_session_misses_total 1" in metrics.render()


@pytest.mark.asyncio
async def test_pool_discards_dead_and_expired_sessions() -> None:
    log: list[str] = []
    runner = FakeRunner(log=log)
    pool = _pool(runner, size=4, ttl_seconds=1.0)
    pool.note(KEY)
    pool.offer(_session(reaped=True))
    pool.offer(_session(created_at=asyncio.get_running_loop().time() - 10.0))
    pool.offer(_session(created_at=asyncio.get_running_loop().time()))

    session = pool.acquire(KEY)

    assert session is not None
    await asyncio.sleep(0)
    assert log.count("discard:model-a") == 2
    await pool.aclose()


@pytest.mark.asyncio
async def test_pool_drops_sessions_for_keys_that_fell_out_of_use() -> None:
    log: list[str] = []
    pool = _pool(FakeRunner(log=log), size=4, max_keys=1)
    pool.note(KEY)
    pool.offer(_session(created_at=asyncio.get_running_loop().time()))

    pool.note(OTHER_KEY)
    await asyncio.sleep(0)

    assert log == ["discard:model-a"]
    assert pool.acquire(KEY) is None


@pytest.mark.asyncio
async def test_pool_discards_sessions_offered_when_full_or_unwanted() -> None:
    log: list[str] = []
    pool = _pool(FakeRunner(log=log), size=1)
    pool.note(KEY)
    pool.offer(_session(created_at=asyncio.get_running_loop().time()))
    pool.offer(_session(created_at=asyncio.get_running_loop().time()))
    pool.offer(_session(OTHER_KEY, created_at=asyncio.get_running_loop().time()))

    await asyncio.sleep(0)

    assert log == ["discard:model-a", "discard:model-b"]


@pytest.mark.asyncio
async def test_pool_warms_and_reuses_an_exact_native_catalog() -> None:
    log: list[str] = []
    reaper = BackgroundReaper()
    registry = NativeToolRegistry(base_url="http://127.0.0.1:8787")
    runner = FakeRunner(log=log)
    pool = WarmSessionPool(
        runner_factory=cast("RunnerFactory", lambda: runner),
        reaper=reaper,
        size=1,
        warm_timeout_seconds=5.0,
        native_registry=registry,
    )
    catalog = _native_catalog(registry)
    pool.start()
    pool.note(KEY, catalog)
    await asyncio.sleep(0.05)

    session = pool.acquire(KEY, catalog)

    assert session is not None
    assert session.native_binding is not None
    assert session.native_binding.catalog == catalog
    assert runner.native_bindings == [session.native_binding]
    assert len(registry) == 1
    await pool.aclose()
    await reaper.drain()
    assert len(registry) == 1
    registry.close(session.native_binding.token)
    assert len(registry) == 0


@pytest.mark.asyncio
async def test_pool_misses_when_native_catalog_changes() -> None:
    reaper = BackgroundReaper()
    registry = NativeToolRegistry(base_url="http://127.0.0.1:8787")
    pool = WarmSessionPool(
        runner_factory=cast("RunnerFactory", lambda: FakeRunner()),
        reaper=reaper,
        size=1,
        warm_timeout_seconds=5.0,
        native_registry=registry,
    )
    first_catalog = _native_catalog(registry, "first")
    second_catalog = _native_catalog(registry, "second")
    binding = registry.open_catalog(first_catalog)
    pool.note(KEY, first_catalog)
    pool.offer(_session(created_at=asyncio.get_running_loop().time(), native_binding=binding))

    assert pool.acquire(KEY, second_catalog) is None
    await pool.aclose()
    await reaper.drain()


@pytest.mark.asyncio
async def test_pool_drops_stale_native_demand() -> None:
    log: list[str] = []
    registry = NativeToolRegistry(base_url="http://127.0.0.1:8787")
    pool = _pool(FakeRunner(log=log), max_keys=1, native_registry=registry)
    first_catalog = _native_catalog(registry, "first")
    second_catalog = _native_catalog(registry, "second")
    first_binding = registry.open_catalog(first_catalog)
    pool.note(KEY, first_catalog)
    pool.offer(_session(created_at=asyncio.get_running_loop().time(), native_binding=first_binding))

    pool.note(OTHER_KEY, second_catalog)
    await asyncio.sleep(0)

    assert log == ["discard:model-a"]
    assert len(registry) == 0
    await pool.aclose()


@pytest.mark.asyncio
async def test_pool_reports_missing_native_registry() -> None:
    registry = NativeToolRegistry(base_url="http://127.0.0.1:8787")
    pool = _pool(FakeRunner(), native_registry=None)
    catalog = _native_catalog(registry)

    result = await pool._warm(_WarmDemand(KEY, catalog))

    assert result is None


@pytest.mark.asyncio
async def test_pool_closes_a_native_catalog_after_warm_failure() -> None:
    registry = NativeToolRegistry(base_url="http://127.0.0.1:8787")
    runner = FakeRunner(fail=True)
    pool = _pool(runner, native_registry=registry)
    catalog = _native_catalog(registry)

    pool.start()
    pool.note(KEY, catalog)
    await asyncio.sleep(0.05)

    assert len(registry) == 0
    assert pool.acquire(KEY, catalog) is None
    await pool.aclose()


def test_pool_drop_wrapper_removes_plain_demand() -> None:
    pool = _pool(FakeRunner())

    pool._drop(KEY)

    assert pool.acquire(KEY) is None


@pytest.mark.asyncio
async def test_pool_closes_a_native_catalog_when_warmup_is_cancelled() -> None:
    class SlowNativeRunner(FakeRunner):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()

        async def warm(
            self,
            key: SessionKey,
            *,
            timeout_seconds: float,
            native_tools: NativeToolBinding | None = None,
        ) -> WarmSession:
            del key, timeout_seconds, native_tools
            self.started.set()
            await asyncio.sleep(30)
            raise AssertionError("warmup should have been cancelled")

    registry = NativeToolRegistry(base_url="http://127.0.0.1:8787")
    runner = SlowNativeRunner()
    pool = _pool(runner, native_registry=registry)
    catalog = _native_catalog(registry)
    pool.start()
    pool.note(KEY, catalog)
    await runner.started.wait()

    await pool.aclose()

    assert len(registry) == 0


@pytest.mark.asyncio
async def test_pool_records_warm_failures_and_keeps_running() -> None:
    metrics = BridgeMetrics()
    runner = FakeRunner(fail=True)
    pool = _pool(runner, metrics=metrics)

    pool.start(initial_key=KEY)
    await asyncio.sleep(0.05)

    assert "factory_droid_openai_warm_session_failures_total 1" in metrics.render()
    await pool.aclose()


@pytest.mark.asyncio
async def test_pool_drops_a_key_after_session_init_timeout() -> None:
    class InitTimeoutRunner(FakeRunner):
        def __init__(self) -> None:
            super().__init__()
            self.attempts = 0

        async def warm(self, key: SessionKey, *, timeout_seconds: float) -> WarmSession:
            del key, timeout_seconds
            self.attempts += 1
            raise RunnerError(
                "Factory Droid session initialization timed out after 0.1 seconds.",
                status_code=504,
                error_type="factory_droid_timeout",
            )

    runner = InitTimeoutRunner()
    pool = _pool(runner, retry_seconds=0.01)
    pool.start(initial_key=KEY)
    await asyncio.sleep(0.05)

    assert runner.attempts == 1
    assert pool.acquire(KEY) is None
    await pool.aclose()


@pytest.mark.asyncio
async def test_pool_stops_warming_a_key_droid_rejects() -> None:
    class CountingRunner(FakeRunner):
        def __init__(self) -> None:
            super().__init__(fail=True)
            self.attempts = 0

        async def warm(
            self,
            key: SessionKey,
            *,
            timeout_seconds: float,
            native_tools: NativeToolBinding | None = None,
        ) -> WarmSession:
            self.attempts += 1
            return await super().warm(
                key,
                timeout_seconds=timeout_seconds,
                native_tools=native_tools,
            )

    runner = CountingRunner()
    pool = _pool(runner, retry_seconds=0.01)

    pool.start(initial_key=KEY)
    await asyncio.sleep(0.1)

    assert runner.attempts == 1

    pool.note(KEY)
    await asyncio.sleep(0.05)

    assert runner.attempts == 2
    await pool.aclose()


@pytest.mark.asyncio
async def test_pool_spreads_warm_sessions_across_keys() -> None:
    log: list[str] = []
    runner = FakeRunner(log=log)
    pool = _pool(runner, size=2, max_keys=2)
    pool.note(KEY)
    pool.note(OTHER_KEY)

    pool.start()
    await asyncio.sleep(0.05)

    assert sorted(log) == ["warm:model-a", "warm:model-b"]
    await pool.aclose()
    await asyncio.sleep(0)
    assert sorted(entry for entry in log if entry.startswith("discard")) == [
        "discard:model-a",
        "discard:model-b",
    ]


@pytest.mark.asyncio
async def test_pool_refills_after_a_session_is_taken() -> None:
    runner = FakeRunner()
    pool = _pool(runner, size=1)

    pool.start(initial_key=KEY)
    await asyncio.sleep(0.05)
    first = pool.acquire(KEY)
    await asyncio.sleep(0.05)
    second = pool.acquire(KEY)

    assert first is not None
    assert second is not None
    assert runner.warmed == 2
    await pool.aclose()


@pytest.mark.asyncio
async def test_pool_sweeps_expired_sessions_in_the_background() -> None:
    log: list[str] = []
    runner = FakeRunner(log=log)
    pool = _pool(runner, size=1, ttl_seconds=1.0)
    pool.note(KEY)
    pool.offer(_session(created_at=asyncio.get_running_loop().time() - 10.0))

    pool.start()
    await asyncio.sleep(0.05)

    assert sorted(log) == ["discard:model-a", "warm:model-a"]
    await pool.aclose()


@pytest.mark.asyncio
async def test_pool_tolerates_warm_failures_without_metrics() -> None:
    pool = _pool(FakeRunner(fail=True))

    pool.start(initial_key=KEY)
    await asyncio.sleep(0.05)

    assert pool.acquire(KEY) is None
    await pool.aclose()


@pytest.mark.asyncio
async def test_pool_close_cancels_an_in_flight_warmup() -> None:
    class SlowRunner(FakeRunner):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()

        async def warm(
            self,
            key: SessionKey,
            *,
            timeout_seconds: float,
            native_tools: NativeToolBinding | None = None,
        ) -> WarmSession:
            del key, timeout_seconds, native_tools
            self.started.set()
            await asyncio.sleep(30)
            raise AssertionError("warmup should have been cancelled")

    runner = SlowRunner()
    pool = _pool(runner)

    pool.start(initial_key=KEY)
    await runner.started.wait()
    await pool.aclose()

    assert pool.acquire(KEY) is None


@pytest.mark.asyncio
async def test_pool_moves_capacity_to_the_key_traffic_switched_to() -> None:
    log: list[str] = []
    runner = FakeRunner(log=log)
    pool = _pool(runner, size=1)

    pool.start(initial_key=KEY)
    await asyncio.sleep(0.05)
    assert log == ["warm:model-a"]

    assert pool.acquire(OTHER_KEY) is None
    await asyncio.sleep(0.05)

    assert sorted(log) == ["discard:model-a", "warm:model-a", "warm:model-b"]
    session = pool.acquire(OTHER_KEY)
    assert session is not None
    await pool.aclose()


@pytest.mark.asyncio
async def test_pool_hands_over_a_session_the_runner_can_retune() -> None:
    metrics = BridgeMetrics()
    pool = _pool(FakeRunner(), size=2, metrics=metrics)
    pool.note(KEY)
    pool.offer(_session(created_at=asyncio.get_running_loop().time()))

    session = pool.acquire(RETUNE_KEY)

    assert session is not None
    assert session.key == KEY
    rendered = metrics.render()
    assert 'factory_droid_openai_warm_session_retunes_total{reason="effort"} 1' in rendered
    assert "factory_droid_openai_warm_session_hits_total 1" in rendered
    assert "factory_droid_openai_warm_session_misses_total 0" in rendered


@pytest.mark.asyncio
async def test_pool_hands_over_a_retunable_session_without_metrics() -> None:
    pool = _pool(FakeRunner(), size=2)
    pool.note(KEY)
    pool.offer(_session(created_at=asyncio.get_running_loop().time()))

    assert pool.acquire(RETUNE_KEY) is not None


@pytest.mark.asyncio
async def test_pool_does_not_retune_a_session_across_models() -> None:
    metrics = BridgeMetrics()
    pool = _pool(FakeRunner(), size=2, metrics=metrics)
    pool.note(KEY)
    pool.offer(_session(created_at=asyncio.get_running_loop().time()))

    assert pool.acquire(CROSS_MODEL_KEY) is None
    rendered = metrics.render()
    assert 'factory_droid_openai_warm_session_retunes_total{reason="effort"} 0' in rendered
    assert "factory_droid_openai_warm_session_misses_total 1" in rendered


@pytest.mark.asyncio
async def test_pool_does_not_hand_over_dead_sessions_for_retuning() -> None:
    log: list[str] = []
    metrics = BridgeMetrics()
    pool = _pool(FakeRunner(log=log), size=2, metrics=metrics)
    pool.note(KEY)
    pool.offer(_session(reaped=True))

    assert pool.acquire(RETUNE_KEY) is None
    await asyncio.sleep(0)

    assert log == ["discard:model-a"]
    assert "factory_droid_openai_warm_session_misses_total 1" in metrics.render()


@pytest.mark.asyncio
async def test_pool_refills_every_missing_session_in_one_pass() -> None:
    class SlowRunner(FakeRunner):
        def __init__(self) -> None:
            super().__init__()
            self.concurrent = 0
            self.peak = 0

        async def warm(
            self,
            key: SessionKey,
            *,
            timeout_seconds: float,
            native_tools: NativeToolBinding | None = None,
        ) -> WarmSession:
            self.concurrent += 1
            self.peak = max(self.peak, self.concurrent)
            try:
                await asyncio.sleep(0.02)
                return await super().warm(
                    key,
                    timeout_seconds=timeout_seconds,
                    native_tools=native_tools,
                )
            finally:
                self.concurrent -= 1

    runner = SlowRunner()
    pool = _pool(runner, size=3)

    pool.start(initial_key=KEY)
    await asyncio.sleep(0.05)

    assert runner.warmed == 3
    assert runner.peak == 3
    await pool.aclose()


@pytest.mark.asyncio
async def test_pool_discards_sessions_for_keys_it_never_saw() -> None:
    log: list[str] = []
    pool = _pool(FakeRunner(log=log), size=2)

    pool.offer(_session(created_at=asyncio.get_running_loop().time()))
    await asyncio.sleep(0)

    assert log == ["discard:model-a"]


def test_pool_metrics_contract_carries_no_default_behavior() -> None:
    metrics = BridgeMetrics()

    PoolMetrics.set_warm_sessions(metrics, 3)
    PoolMetrics.increment_warm_hits(metrics)
    PoolMetrics.increment_warm_retunes(metrics, "effort")
    PoolMetrics.increment_warm_misses(metrics)
    PoolMetrics.increment_warm_failures(metrics)
    PoolMetrics.set_pending_reaps(metrics, 2)

    rendered = metrics.render()
    assert "factory_droid_openai_warm_sessions 0" in rendered
    assert "factory_droid_openai_warm_session_hits_total 0" in rendered
    assert "factory_droid_openai_pending_reaps 0" in rendered


def test_forgetting_an_already_evicted_key_is_a_no_op() -> None:
    pool = _pool(FakeRunner(), max_keys=1)
    pool.note(KEY)
    pool.note(OTHER_KEY)

    pool._forget(KEY)

    assert pool._wanted == [OTHER_KEY]
