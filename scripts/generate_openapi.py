from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def normalize_numbers(value: Any) -> Any:
    """Rewrites whole floats as ints so the document is serializer-independent.

    Pydantic renders a numeric constraint as ``0`` or ``0.0`` depending on the
    release, and both parse to the same JSON number, so a parsed comparison
    never notices while every regeneration rewrites the committed file.
    """
    if isinstance(value, dict):
        return {key: normalize_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_numbers(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def render(root: Path) -> str:
    """Returns the committed contract's exact text for ``root``."""
    from factory_droid_openai.app import create_app
    from factory_droid_openai.config import Settings

    schema = normalize_numbers(create_app(Settings(workdir=root)).openapi())
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    (root / "openapi.json").write_text(render(root), encoding="utf-8")


if __name__ == "__main__":
    main()
