from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

DEFAULT_BASE_URL = "http://127.0.0.1:8787/v1"
DEFAULT_PROVIDER_NAME = "Factory Droid"
ALIAS_LABEL = "Factory Droid (server default model)"
DEFAULT_MAX_INPUT_TOKENS = 180000
DEFAULT_MAX_OUTPUT_TOKENS = 20000
NON_THINKING_EFFORTS = frozenset({"off", "none"})
CURATED_MODELS = (
    "factory-droid",
    "claude-opus-5",
    "claude-sonnet-5",
    "gpt-5.6-sol",
    "gpt-5.4",
    "gemini-3.1-pro-preview",
    "glm-5.2",
)


def _fetch_models(base_url: str, api_key: str | None) -> list[dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/models"
    if urlsplit(url).scheme not in {"http", "https"}:
        message = f"Unsupported URL scheme in {url!r}"
        raise SystemExit(message)
    request = urllib.request.Request(url)  # noqa: S310
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(request) as response:  # noqa: S310
        payload = json.load(response)
    models: list[dict[str, Any]] = payload["data"]
    return models


def _model_entry(
    model: dict[str, Any],
    *,
    chat_url: str,
    max_input_tokens: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    model_id = str(model["id"])
    display_name = model.get("factory_droid_display_name")
    raw_efforts = model.get("factory_droid_supported_reasoning_efforts") or []
    efforts = [str(effort) for effort in raw_efforts]
    thinking_efforts = [effort for effort in efforts if effort not in NON_THINKING_EFFORTS]
    entry: dict[str, Any] = {
        "id": model_id,
        "name": f"{display_name} via Factory Droid" if display_name else ALIAS_LABEL,
        "url": chat_url,
        "toolCalling": True,
        "vision": bool(model.get("factory_droid_supports_images", True)),
        "streaming": True,
        "thinking": bool(thinking_efforts),
        "maxInputTokens": max_input_tokens,
        "maxOutputTokens": max_output_tokens,
    }
    if thinking_efforts:
        entry["supportsReasoningEffort"] = efforts
        entry["reasoningEffortFormat"] = "chat-completions"
    return entry


def _build_config(
    models: list[dict[str, Any]],
    *,
    provider_name: str,
    chat_url: str,
    api_key: str,
    max_input_tokens: int,
    max_output_tokens: int,
) -> list[dict[str, Any]]:
    return [
        {
            "name": provider_name,
            "vendor": "customendpoint",
            "apiKey": api_key,
            "apiType": "chat-completions",
            "models": [
                _model_entry(
                    model,
                    chat_url=chat_url,
                    max_input_tokens=max_input_tokens,
                    max_output_tokens=max_output_tokens,
                )
                for model in models
            ],
        }
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a VS Code chatLanguageModels.json from a running bridge.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--chat-url", default=None)
    parser.add_argument("--provider-name", default=DEFAULT_PROVIDER_NAME)
    parser.add_argument("--api-key-placeholder", default="none")
    parser.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model ID to include; repeatable. Defaults to a curated set.",
    )
    parser.add_argument("--all-models", action="store_true", help="Include every discovered model.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_url = str(args.base_url).rstrip("/")
    chat_url = str(args.chat_url or f"{base_url}/chat/completions")
    discovered = _fetch_models(base_url, os.getenv("FACTORY_DROID_OPENAI_API_KEY"))
    by_id = {str(model["id"]): model for model in discovered}

    if args.all_models:
        selected = discovered
    else:
        wanted = args.models or list(CURATED_MODELS)
        missing = [model_id for model_id in wanted if model_id not in by_id]
        if missing:
            message = f"Models not available on the bridge: {', '.join(missing)}"
            raise SystemExit(message)
        selected = [by_id[model_id] for model_id in wanted]

    config = _build_config(
        selected,
        provider_name=str(args.provider_name),
        chat_url=chat_url,
        api_key=str(args.api_key_placeholder),
        max_input_tokens=int(args.max_input_tokens),
        max_output_tokens=int(args.max_output_tokens),
    )
    document = json.dumps(config, indent=2) + "\n"
    output: Path | None = args.output
    if output is None:
        output = Path(__file__).resolve().parents[1] / "examples/vscode/chatLanguageModels.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")


if __name__ == "__main__":
    main()
