from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    from factory_droid_openai.app import create_app
    from factory_droid_openai.config import Settings

    root = Path(__file__).resolve().parents[1]
    schema = create_app(Settings(workdir=root)).openapi()
    output = root / "openapi.json"
    output.write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
