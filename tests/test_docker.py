from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_installs_tini() -> None:
    # tini must be installed before it can serve as the entrypoint, and the
    # Debian package is what the ENTRYPOINT line points at.
    lines = (_ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()

    assert any("apt-get install" in line and " tini" in line for line in lines), (
        "tini is no longer installed; the ENTRYPOINT below would fail to start"
    )


def test_dockerfile_entrypoint_is_tini() -> None:
    # The bridge is not a reaping init: orphaned git helpers reparent to PID 1
    # and accumulate as zombies until the container can no longer fork (#138).
    # tini as PID 1 reaps every orphan regardless of which path spawned it
    # (#143), so dropping this entrypoint reintroduces the leak for every
    # deployment. A Python-side waitpid reaper is not a substitute: it races
    # with asyncio's ThreadedChildWatcher and steals Droid exit statuses.
    lines = (_ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()

    assert 'ENTRYPOINT ["/usr/bin/tini", "--"]' in lines
