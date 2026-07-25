from __future__ import annotations

from typing import Any

import pytest
import uvicorn

from factory_droid_openai import cli
from factory_droid_openai.config import Settings


def test_cli_runs_uvicorn_with_environment_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(host="127.0.0.2", port=9123)
    calls: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        Settings,
        "from_env",
        classmethod(lambda _cls: settings),
    )
    monkeypatch.setattr(
        uvicorn,
        "run",
        lambda app, **kwargs: calls.append((app, kwargs)),
    )

    cli.main([])

    assert calls == [
        (
            "factory_droid_openai.app:app",
            {
                "host": "127.0.0.2",
                "port": 9123,
                "log_level": "info",
            },
        )
    ]


def test_cli_arguments_override_server_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        Settings,
        "from_env",
        classmethod(lambda _cls: Settings()),
    )
    monkeypatch.setattr(
        uvicorn,
        "run",
        lambda app, **kwargs: calls.append((app, kwargs)),
    )

    cli.main(
        [
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--log-level",
            "debug",
        ]
    )

    assert calls[0][1] == {
        "host": "0.0.0.0",
        "port": 9000,
        "log_level": "debug",
    }


def test_cli_rejects_unknown_log_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Settings,
        "from_env",
        classmethod(lambda _cls: Settings()),
    )

    with pytest.raises(SystemExit) as error:
        cli.main(["--log-level", "verbose"])

    assert error.value.code == 2
