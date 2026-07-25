from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8787
    api_key: str | None = None
    droid_path: str = "droid"
    workdir: Path = field(default_factory=Path.cwd)
    timeout_seconds: float = 600.0
    max_concurrency: int = 1
    model_alias: str = "factory-droid"

    @classmethod
    def from_env(cls) -> Settings:
        configured_workdir = os.getenv("FACTORY_DROID_OPENAI_WORKDIR")
        workdir = Path(configured_workdir).expanduser() if configured_workdir else Path.cwd()
        if not workdir.is_dir():
            raise ValueError(f"Factory Droid workdir does not exist: {workdir}")
        timeout_seconds = _positive_float(
            "FACTORY_DROID_OPENAI_TIMEOUT_SECONDS",
            default=600.0,
        )
        max_concurrency = _positive_int(
            "FACTORY_DROID_OPENAI_MAX_CONCURRENCY",
            default=1,
        )
        port = _positive_int("FACTORY_DROID_OPENAI_PORT", default=8787)
        if port > 65535:
            raise ValueError("FACTORY_DROID_OPENAI_PORT must be at most 65535")
        return cls(
            host=os.getenv("FACTORY_DROID_OPENAI_HOST", "127.0.0.1"),
            port=port,
            api_key=os.getenv("FACTORY_DROID_OPENAI_API_KEY") or None,
            droid_path=os.getenv("FACTORY_DROID_PATH", "droid"),
            workdir=workdir.resolve(),
            timeout_seconds=timeout_seconds,
            max_concurrency=max_concurrency,
            model_alias=os.getenv("FACTORY_DROID_OPENAI_MODEL_ALIAS", "factory-droid"),
        )


def _positive_float(name: str, *, default: float) -> float:
    raw = os.getenv(name)
    try:
        value = float(raw) if raw is not None else default
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _positive_int(name: str, *, default: int) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None else default
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value
