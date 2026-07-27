from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from droid_sdk.schemas.enums import ReasoningEffort

REQUIRED_MODEL_KEYS = {
    "id",
    "name",
    "url",
    "toolCalling",
    "vision",
    "maxInputTokens",
    "maxOutputTokens",
}


def _load_config() -> list[dict[str, Any]]:
    root = Path(__file__).resolve().parents[1]
    path = root / "examples" / "vscode" / "chatLanguageModels.json"
    config: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return config


def test_vscode_example_declares_one_custom_endpoint_provider() -> None:
    config = _load_config()

    assert len(config) == 1
    provider = config[0]
    assert provider["vendor"] == "customendpoint"
    assert provider["apiType"] == "chat-completions"
    assert provider["apiKey"] == "none"
    assert provider["models"]


def test_vscode_example_models_carry_required_properties() -> None:
    models = _load_config()[0]["models"]
    valid_efforts = {effort.value for effort in ReasoningEffort}

    assert len({model["id"] for model in models}) == len(models)
    for model in models:
        assert set(model) >= REQUIRED_MODEL_KEYS, model["id"]
        assert model["url"].endswith("/v1/chat/completions")
        assert model["toolCalling"] is True
        assert isinstance(model["vision"], bool)
        assert model["maxInputTokens"] > 0
        assert model["maxOutputTokens"] > 0
        efforts = model.get("supportsReasoningEffort", [])
        assert set(efforts) <= valid_efforts, model["id"]
        assert model["thinking"] is bool(efforts)
        if efforts:
            assert model["reasoningEffortFormat"] == "chat-completions"
