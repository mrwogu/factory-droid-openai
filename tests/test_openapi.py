from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openapi_spec_validator import validate

from factory_droid_openai.app import create_app
from factory_droid_openai.config import Settings


def test_committed_openapi_contract_is_valid_and_current(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    committed = json.loads((root / "openapi.json").read_text(encoding="utf-8"))
    generated = create_app(Settings(workdir=tmp_path)).openapi()

    validate(committed)
    assert committed == generated


def test_openapi_contract_documents_compatibility_surface(tmp_path: Path) -> None:
    schema: dict[str, Any] = create_app(Settings(workdir=tmp_path)).openapi()
    paths = schema["paths"]

    assert set(paths) == {"/health", "/v1/models", "/v1/chat/completions"}
    assert "security" not in paths["/health"]["get"]
    assert paths["/v1/models"]["get"]["security"] == [{"HTTPBearer": []}]

    chat = paths["/v1/chat/completions"]["post"]
    assert chat["security"] == [{"HTTPBearer": []}]
    assert set(chat["responses"]) == {"200", "4XX", "502", "503", "504"}
    assert set(chat["responses"]["200"]["content"]) == {
        "application/json",
        "text/event-stream",
    }
    assert schema["components"]["securitySchemes"]["HTTPBearer"]["scheme"].lower() == "bearer"
