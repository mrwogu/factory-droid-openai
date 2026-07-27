from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
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
ALIAS_MODEL_ID = "factory-droid"
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


async def _probe_model(
    model_id: str,
    *,
    droid_path: str,
    workdir: Path,
    timeout_seconds: float,
    gate: asyncio.Semaphore,
) -> str | None:
    """Return why ``model_id`` is unusable, or ``None`` when it works.

    Only a session is initialized, which is where Droid refuses models an
    organization policy blocks. No model turn runs, so probing the whole
    catalog costs no tokens.
    """
    from factory_droid_openai.runner import DroidRunner, SessionKey

    runner = DroidRunner(droid_path=droid_path, workdir=workdir)
    async with gate:
        try:
            session = await runner.warm(
                SessionKey(model_id=model_id, reasoning_effort=None),
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            # Any refusal disqualifies the model, whatever its shape.
            return str(exc) or type(exc).__name__
    await runner.discard(session)
    return None


def _verify_models(
    model_ids: list[str],
    *,
    droid_path: str,
    workdir: Path,
    timeout_seconds: float,
    concurrency: int,
) -> dict[str, str]:
    """Probe every model and map the unusable ones to their refusal."""

    async def run() -> dict[str, str]:
        gate = asyncio.Semaphore(max(1, concurrency))
        results = await asyncio.gather(
            *(
                _probe_model(
                    model_id,
                    droid_path=droid_path,
                    workdir=workdir,
                    timeout_seconds=timeout_seconds,
                    gate=gate,
                )
                for model_id in model_ids
            )
        )
        return {
            model_id: reason
            for model_id, reason in zip(model_ids, results, strict=True)
            if reason is not None
        }

    return asyncio.run(run())


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
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Drop models Droid refuses to start a session with, such as ones "
        "blocked by an organization policy.",
    )
    parser.add_argument("--verify-concurrency", type=int, default=4)
    parser.add_argument("--verify-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--droid-path", default=os.getenv("FACTORY_DROID_PATH", "droid"))
    parser.add_argument("--workdir", type=Path, default=None)
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

    if args.verify:
        alias = ALIAS_MODEL_ID
        probed = [str(model["id"]) for model in selected if str(model["id"]) != alias]
        refused = _verify_models(
            probed,
            droid_path=str(args.droid_path),
            workdir=Path(args.workdir) if args.workdir else Path.cwd(),
            timeout_seconds=float(args.verify_timeout_seconds),
            concurrency=int(args.verify_concurrency),
        )
        for model_id, reason in sorted(refused.items()):
            print(f"skipping {model_id}: {reason}", file=sys.stderr)
        print(
            f"verified {len(probed) - len(refused)}/{len(probed)} models",
            file=sys.stderr,
        )
        selected = [model for model in selected if str(model["id"]) not in refused]

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
