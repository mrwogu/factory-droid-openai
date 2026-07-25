from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from factory_droid_openai.config import Settings

if TYPE_CHECKING:
    from pathlib import Path

_ENVIRONMENT_KEYS = (
    "FACTORY_DROID_OPENAI_HOST",
    "FACTORY_DROID_OPENAI_PORT",
    "FACTORY_DROID_OPENAI_API_KEY",
    "FACTORY_DROID_PATH",
    "FACTORY_DROID_OPENAI_WORKDIR",
    "FACTORY_DROID_OPENAI_TIMEOUT_SECONDS",
    "FACTORY_DROID_OPENAI_MAX_CONCURRENCY",
    "FACTORY_DROID_OPENAI_MODEL_ALIAS",
)


def _clear_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_settings_from_env_uses_safe_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_environment(monkeypatch)
    monkeypatch.chdir(tmp_path)

    settings = Settings.from_env()

    assert settings == Settings(workdir=tmp_path)


def test_settings_from_env_reads_all_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_environment(monkeypatch)
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    monkeypatch.setenv("FACTORY_DROID_OPENAI_HOST", "0.0.0.0")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_PORT", "9000")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("FACTORY_DROID_PATH", "/usr/local/bin/droid")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_WORKDIR", str(workdir))
    monkeypatch.setenv("FACTORY_DROID_OPENAI_TIMEOUT_SECONDS", "45.5")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_MAX_CONCURRENCY", "3")
    monkeypatch.setenv("FACTORY_DROID_OPENAI_MODEL_ALIAS", "droid-default")

    settings = Settings.from_env()

    assert settings == Settings(
        host="0.0.0.0",
        port=9000,
        api_key="test-key",
        droid_path="/usr/local/bin/droid",
        workdir=workdir,
        timeout_seconds=45.5,
        max_concurrency=3,
        model_alias="droid-default",
    )


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("FACTORY_DROID_OPENAI_TIMEOUT_SECONDS", "invalid", "must be a number"),
        ("FACTORY_DROID_OPENAI_TIMEOUT_SECONDS", "0", "must be greater than zero"),
        ("FACTORY_DROID_OPENAI_MAX_CONCURRENCY", "invalid", "must be an integer"),
        ("FACTORY_DROID_OPENAI_MAX_CONCURRENCY", "0", "must be greater than zero"),
        ("FACTORY_DROID_OPENAI_PORT", "70000", "must be at most 65535"),
    ],
)
def test_settings_from_env_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str,
    value: str,
    message: str,
) -> None:
    _clear_environment(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        Settings.from_env()


def test_settings_from_env_rejects_missing_workdir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_environment(monkeypatch)
    missing = tmp_path / "missing"
    monkeypatch.setenv("FACTORY_DROID_OPENAI_WORKDIR", str(missing))

    with pytest.raises(ValueError, match="workdir does not exist"):
        Settings.from_env()
