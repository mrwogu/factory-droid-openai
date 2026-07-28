from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from types import ModuleType

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_vscode_models.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generate_vscode_models", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen() -> ModuleType:
    return _load_module()


def _catalog() -> list[dict[str, Any]]:
    return [
        {
            "id": "factory-droid",
            "factory_droid_display_name": None,
            "factory_droid_supported_reasoning_efforts": ["low", "high"],
            "factory_droid_supports_images": True,
        },
        {
            "id": "gpt-5.4",
            "factory_droid_display_name": "GPT-5.4",
            "factory_droid_supported_reasoning_efforts": ["off", "low", "high"],
            "factory_droid_supports_images": False,
        },
        {
            "id": "glm-5.2",
            "factory_droid_display_name": "GLM 5.2",
            "factory_droid_supported_reasoning_efforts": ["none"],
            "factory_droid_supports_images": True,
        },
    ]


def test_model_entry_maps_display_name_and_efforts(gen: ModuleType) -> None:
    entry = gen._model_entry(
        _catalog()[1],
        chat_url="http://127.0.0.1:8787/v1/chat/completions",
        max_input_tokens=180000,
        max_output_tokens=20000,
    )

    assert entry["id"] == "gpt-5.4"
    assert entry["name"] == "GPT-5.4 via Factory Droid"
    assert entry["vision"] is False
    assert entry["thinking"] is True
    # Non-thinking levels are hidden from the picker but kept in the list.
    assert entry["supportsReasoningEffort"] == ["off", "low", "high"]
    assert entry["reasoningEffortFormat"] == "chat-completions"


def test_model_entry_uses_alias_label_without_display_name(gen: ModuleType) -> None:
    entry = gen._model_entry(
        _catalog()[0],
        chat_url="http://127.0.0.1:8787/v1/chat/completions",
        max_input_tokens=180000,
        max_output_tokens=20000,
    )

    assert entry["name"] == gen.ALIAS_LABEL


def test_model_entry_without_thinking_efforts_omits_reasoning_keys(gen: ModuleType) -> None:
    entry = gen._model_entry(
        _catalog()[2],
        chat_url="http://127.0.0.1:8787/v1/chat/completions",
        max_input_tokens=180000,
        max_output_tokens=20000,
    )

    assert entry["thinking"] is False
    assert "supportsReasoningEffort" not in entry
    assert "reasoningEffortFormat" not in entry


def test_build_config_wraps_models_in_provider(gen: ModuleType) -> None:
    config = gen._build_config(
        _catalog(),
        provider_name="Factory Droid",
        chat_url="http://127.0.0.1:8787/v1/chat/completions",
        api_key="none",
        max_input_tokens=180000,
        max_output_tokens=20000,
    )

    assert len(config) == 1
    provider = config[0]
    assert provider["vendor"] == "customendpoint"
    assert provider["apiType"] == "chat-completions"
    assert provider["apiKey"] == "none"
    assert [model["id"] for model in provider["models"]] == [
        "factory-droid",
        "gpt-5.4",
        "glm-5.2",
    ]


def test_fetch_models_rejects_non_http_scheme(gen: ModuleType) -> None:
    with pytest.raises(SystemExit, match="Unsupported URL scheme"):
        gen._fetch_models("ftp://example.test/v1", None)


def test_fetch_models_sends_bearer_header(gen: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _Response:
        def __enter__(self) -> io.BytesIO:
            return io.BytesIO(json.dumps({"data": _catalog()}).encode())

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(request: Any) -> _Response:
        captured["url"] = request.full_url
        captured["auth"] = request.get_header("Authorization")
        return _Response()

    monkeypatch.setattr(gen.urllib.request, "urlopen", fake_urlopen)

    models = gen._fetch_models("http://127.0.0.1:8787/v1/", "secret")

    assert captured["url"] == "http://127.0.0.1:8787/v1/models"
    assert captured["auth"] == "Bearer secret"
    assert [model["id"] for model in models] == ["factory-droid", "gpt-5.4", "glm-5.2"]


def test_probe_model_returns_refusal_reason(
    gen: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from factory_droid_openai import runner as runner_module

    class _RejectingRunner:
        def __init__(self, *, droid_path: str, workdir: Path) -> None:
            pass

        async def warm(self, _session_key: object, *, timeout_seconds: float) -> object:  # noqa: ARG002
            raise RuntimeError("model blocked by organization policy")

        async def discard(self, _session: object) -> None:
            raise AssertionError("discard must not run after a refused warm")

    monkeypatch.setattr(runner_module, "DroidRunner", _RejectingRunner)

    reason = asyncio.run(
        gen._probe_model(
            "gpt-5.4",
            droid_path="droid",
            workdir=tmp_path,
            timeout_seconds=1.0,
            gate=asyncio.Semaphore(1),
        )
    )

    assert reason == "model blocked by organization policy"


def test_probe_model_returns_none_and_discards_on_success(
    gen: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from factory_droid_openai import runner as runner_module

    discarded: list[object] = []

    class _AcceptingRunner:
        def __init__(self, *, droid_path: str, workdir: Path) -> None:
            pass

        async def warm(self, _session_key: object, *, timeout_seconds: float) -> object:  # noqa: ARG002
            return object()

        async def discard(self, _session: object) -> None:
            discarded.append(_session)

    monkeypatch.setattr(runner_module, "DroidRunner", _AcceptingRunner)

    reason = asyncio.run(
        gen._probe_model(
            "gpt-5.4",
            droid_path="droid",
            workdir=tmp_path,
            timeout_seconds=1.0,
            gate=asyncio.Semaphore(1),
        )
    )

    assert reason is None
    assert len(discarded) == 1


def test_verify_models_reports_progress_and_refusals(
    gen: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    async def fake_probe(
        model_id: str,
        *,
        droid_path: str,  # noqa: ARG001
        workdir: Path,  # noqa: ARG001
        timeout_seconds: float,  # noqa: ARG001
        gate: asyncio.Semaphore,  # noqa: ARG001
    ) -> str | None:
        return "blocked" if model_id == "glm-5.2" else None

    monkeypatch.setattr(gen, "_probe_model", fake_probe)

    refused = gen._verify_models(
        ["factory-droid", "gpt-5.4", "glm-5.2"],
        droid_path="droid",
        workdir=tmp_path,
        timeout_seconds=1.0,
        concurrency=2,
    )

    assert refused == {"glm-5.2": "blocked"}
    stderr = capsys.readouterr().err
    assert "[3/3] glm-5.2: refused: blocked" in stderr
    assert "factory-droid: ok" in stderr
    assert "gpt-5.4: ok" in stderr


def test_main_generates_config_with_verified_models(
    gen: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "chatLanguageModels.json"
    monkeypatch.setattr(gen, "_fetch_models", lambda *_args, **_kwargs: _catalog())
    monkeypatch.setattr(gen, "_verify_models", lambda *_args, **_kwargs: {"glm-5.2": "blocked"})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_vscode_models.py",
            "--all-models",
            "--verify",
            "--output",
            str(output),
        ],
    )

    gen.main()

    config = json.loads(output.read_text(encoding="utf-8"))
    ids = [model["id"] for model in config[0]["models"]]
    # The alias is never probed, and the refused model is dropped.
    assert ids == ["factory-droid", "gpt-5.4"]
    assert "verified 1/2 models" in capsys.readouterr().err


def test_main_rejects_unknown_model_ids(
    gen: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(gen, "_fetch_models", lambda *_args, **_kwargs: _catalog())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_vscode_models.py",
            "--model",
            "no-such-model",
            "--output",
            str(tmp_path / "out.json"),
        ],
    )

    with pytest.raises(SystemExit, match="no-such-model"):
        gen.main()
