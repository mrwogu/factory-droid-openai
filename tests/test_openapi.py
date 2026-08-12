from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openapi_spec_validator import validate

from factory_droid_openai.app import create_app
from factory_droid_openai.config import Settings

if TYPE_CHECKING:
    from types import ModuleType

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _ROOT / "scripts" / "generate_openapi.py"


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generate_openapi", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_openapi_contract_is_valid_and_current() -> None:
    committed = json.loads((_ROOT / "openapi.json").read_text(encoding="utf-8"))
    generated = json.loads(_load_generator().render(_ROOT))

    validate(committed)
    # Dumping both sides makes the comparison type-sensitive: a parsed 0 equals
    # a parsed 0.0, which is exactly the drift that kept rewriting the file.
    assert json.dumps(committed, sort_keys=True) == json.dumps(generated, sort_keys=True)


def test_openapi_generator_normalizes_whole_floats() -> None:
    normalized = _load_generator().normalize_numbers(
        {"minimum": 0.0, "values": [1.0, 1.5, True, "1.0", None], "nested": {"timeout": 2.0}}
    )

    assert json.dumps(normalized, sort_keys=True) == (
        '{"minimum": 0, "nested": {"timeout": 2}, "values": [1, 1.5, true, "1.0", null]}'
    )


def test_openapi_contract_documents_compatibility_surface(tmp_path: Path) -> None:
    schema: dict[str, Any] = create_app(Settings(workdir=tmp_path)).openapi()
    paths = schema["paths"]

    assert set(paths) == {
        "/health",
        "/version",
        "/v1/models",
        "/v1/models/{model_id}",
        "/v1/chat/completions",
        "/v1/factory/sessions/{session_id}",
        "/v1/factory/sessions/{session_id}/compact",
        "/v1/factory/sessions/{session_id}/context",
        "/v1/factory/sessions/{session_id}/fork",
    }
    assert "security" not in paths["/health"]["get"]
    assert "security" not in paths["/version"]["get"]
    assert paths["/v1/models"]["get"]["security"] == [{"HTTPBearer": []}]
    assert paths["/v1/models/{model_id}"]["get"]["security"] == [{"HTTPBearer": []}]

    chat = paths["/v1/chat/completions"]["post"]
    assert chat["security"] == [{"HTTPBearer": []}]
    assert set(chat["responses"]) == {
        "200",
        "4XX",
        "404",
        "409",
        "413",
        "429",
        "502",
        "503",
        "504",
    }
    assert "Retry-After" in chat["responses"]["429"]["headers"]
    assert set(chat["responses"]["200"]["content"]) == {
        "application/json",
        "text/event-stream",
    }
    assert schema["components"]["securitySchemes"]["HTTPBearer"]["scheme"].lower() == "bearer"
    assert paths["/v1/factory/sessions/{session_id}/context"]["get"]["security"] == [
        {"HTTPBearer": []}
    ]
    request_schema = schema["components"]["schemas"]["ChatCompletionRequest"]
    assert "response_format" in request_schema["properties"]
    finish_reasons = schema["components"]["schemas"]["ChatCompletionChoice"]["properties"][
        "finish_reason"
    ]["enum"]
    assert set(finish_reasons) == {"stop", "tool_calls", "length"}
