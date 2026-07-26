from __future__ import annotations

import base64
from typing import Any

import pytest

from factory_droid_openai.attachments import (
    AttachmentError,
    AttachmentSet,
    extract_attachments,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-image-body"
PNG_B64 = base64.b64encode(PNG_BYTES).decode("ascii")
PDF_BYTES = b"%PDF-1.7 fake"
PDF_B64 = base64.b64encode(PDF_BYTES).decode("ascii")


def _extract(
    payload: dict[str, Any],
    **overrides: int,
) -> tuple[dict[str, Any], AttachmentSet]:
    collected = AttachmentSet()
    limits: dict[str, int] = {"max_attachments": 8, "max_attachment_bytes": 1_000_000}
    limits.update(overrides)
    return extract_attachments(payload, collected, **limits), collected


def _image_message(url: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": url}},
        ],
    }


def test_string_content_is_returned_unchanged() -> None:
    payload = {"role": "user", "content": "plain text"}

    result, collected = _extract(payload)

    assert result is payload
    assert bool(collected) is False
    assert collected.count == 0


def test_content_without_attachments_is_returned_unchanged() -> None:
    payload = {"role": "user", "content": [{"type": "text", "text": "hi"}, "raw"]}

    result, collected = _extract(payload)

    assert result is payload
    assert bool(collected) is False


def test_image_data_uri_becomes_sdk_attachment_and_placeholder() -> None:
    payload = _image_message(f"data:image/png;base64,{PNG_B64}")

    result, collected = _extract(payload)

    assert result["content"] == [
        {"type": "text", "text": "look"},
        {"type": "text", "text": "[attached image #1 (image/png)]"},
    ]
    assert len(collected.images) == 1
    assert collected.images[0].to_sdk() == {
        "type": "base64",
        "mediaType": "image/png",
        "data": PNG_B64,
    }
    assert bool(collected) is True


def test_image_url_accepts_bare_string_form() -> None:
    payload = {
        "role": "user",
        "content": [{"type": "image_url", "image_url": f"data:image/webp;base64,{PNG_B64}"}],
    }

    _, collected = _extract(payload)

    assert collected.images[0].media_type == "image/webp"


def test_remote_image_urls_are_rejected() -> None:
    payload = _image_message("https://example.com/cat.png")

    with pytest.raises(AttachmentError, match="remote image URLs are not supported"):
        _extract(payload)


def test_missing_image_url_is_rejected() -> None:
    payload = {"role": "user", "content": [{"type": "image_url", "image_url": {}}]}

    with pytest.raises(AttachmentError, match="must contain a URL"):
        _extract(payload)


def test_unsupported_image_media_type_is_rejected() -> None:
    payload = _image_message(f"data:image/tiff;base64,{PNG_B64}")

    with pytest.raises(AttachmentError, match="unsupported image media type"):
        _extract(payload)


def test_invalid_base64_is_rejected() -> None:
    payload = _image_message("data:image/png;base64,not!valid!base64")

    with pytest.raises(AttachmentError, match="not valid base64"):
        _extract(payload)


def test_non_base64_image_data_uri_is_rejected() -> None:
    payload = _image_message("data:image/png,raw-bytes")

    with pytest.raises(AttachmentError, match="must be base64 encoded"):
        _extract(payload)


def test_pdf_file_part_becomes_document_attachment() -> None:
    payload = {
        "role": "user",
        "content": [
            {
                "type": "file",
                "file": {
                    "filename": "spec.pdf",
                    "file_data": f"data:application/pdf;base64,{PDF_B64}",
                },
            }
        ],
    }

    result, collected = _extract(payload)

    assert result["content"] == [
        {"type": "text", "text": "[attached file #1 (application/pdf) named spec.pdf]"}
    ]
    assert collected.documents[0].to_sdk() == {
        "type": "base64",
        "mediaType": "application/pdf",
        "data": PDF_B64,
        "name": "spec.pdf",
    }


def test_plain_text_file_part_keeps_inline_text_source() -> None:
    payload = {
        "role": "user",
        "content": [
            {"type": "file", "file": {"file_data": "data:text/plain,hello world"}},
        ],
    }

    result, collected = _extract(payload)

    assert result["content"] == [{"type": "text", "text": "[attached file #1 (text/plain)]"}]
    assert collected.documents[0].to_sdk() == {
        "type": "text",
        "mediaType": "text/plain",
        "data": "hello world",
    }


def test_file_part_requires_file_object() -> None:
    payload = {"role": "user", "content": [{"type": "file", "file": "spec.pdf"}]}

    with pytest.raises(AttachmentError, match="must contain a 'file' object"):
        _extract(payload)


def test_file_id_references_are_rejected() -> None:
    payload = {"role": "user", "content": [{"type": "file", "file": {"file_id": "f-1"}}]}

    with pytest.raises(AttachmentError, match="file_id references are not supported"):
        _extract(payload)


def test_file_part_requires_inline_data() -> None:
    payload = {"role": "user", "content": [{"type": "file", "file": {"filename": "a.pdf"}}]}

    with pytest.raises(AttachmentError, match="must contain inline 'file_data'"):
        _extract(payload)


def test_audio_parts_are_rejected() -> None:
    payload = {"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}

    with pytest.raises(AttachmentError, match="audio content parts are not supported"):
        _extract(payload)


def test_attachment_count_limit_is_enforced() -> None:
    payload = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_B64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_B64}"}},
        ],
    }

    with pytest.raises(AttachmentError, match="maximum of 1 attachments"):
        _extract(payload, max_attachments=1)


def test_attachment_byte_limit_is_enforced() -> None:
    payload = _image_message(f"data:image/png;base64,{PNG_B64}")

    with pytest.raises(AttachmentError, match="exceed maximum of 4 bytes"):
        _extract(payload, max_attachment_bytes=4)
