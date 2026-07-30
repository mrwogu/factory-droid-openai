from __future__ import annotations

from typing import Any

import pytest
import uvicorn

from factory_droid_openai import cli
from factory_droid_openai.config import Settings


def test_cli_runs_uvicorn_with_environment_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        host="127.0.0.2",
        port=9123,
        server_limit_concurrency=48,
        server_backlog=144,
    )
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
                "access_log": True,
                "limit_concurrency": 48,
                "backlog": 144,
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
            "--log-format",
            "json",
            "--no-access-log",
            "--limit-concurrency",
            "32",
            "--backlog",
            "96",
        ]
    )

    assert calls[0][1] == {
        "host": "0.0.0.0",
        "port": 9000,
        "log_level": "debug",
        "access_log": False,
        "limit_concurrency": 32,
        "backlog": 96,
    }


def test_cli_stays_silent_when_the_api_key_is_configured(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        Settings,
        "from_env",
        classmethod(lambda _cls: Settings(api_key="secret")),
    )
    monkeypatch.setattr(uvicorn, "run", lambda _app, **_kwargs: None)

    cli.main([])

    assert "WARNING" not in capsys.readouterr().err


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


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--limit-concurrency", "0"),
        ("--limit-concurrency", "1"),
        ("--backlog", "0"),
    ],
)
def test_cli_rejects_unsafe_server_options(
    monkeypatch: pytest.MonkeyPatch,
    option: str,
    value: str,
) -> None:
    monkeypatch.setattr(
        Settings,
        "from_env",
        classmethod(lambda _cls: Settings()),
    )

    with pytest.raises(SystemExit) as error:
        cli.main([option, value])

    assert error.value.code == 2
