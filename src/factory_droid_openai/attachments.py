from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass, field
from typing import Any

from factory_droid_openai.errors import ProtocolError

IMAGE_MEDIA_TYPES = frozenset(
    {
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
    }
)
DOCUMENT_MEDIA_TYPES = frozenset(
    {
        "application/pdf",
        "text/plain",
    }
)

_DATA_URI = re.compile(r"^data:([\w.+-]+/[\w.+-]+)((?:;[^,]*)*),(.*)$", re.DOTALL)


class AttachmentError(ProtocolError):
    pass


@dataclass(frozen=True, slots=True)
class ImageAttachment:
    media_type: str
    data: str

    def to_sdk(self) -> dict[str, Any]:
        return {"type": "base64", "mediaType": self.media_type, "data": self.data}


@dataclass(frozen=True, slots=True)
class DocumentAttachment:
    media_type: str
    data: str
    source_type: str
    name: str | None = None

    def to_sdk(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.source_type,
            "mediaType": self.media_type,
            "data": self.data,
        }
        if self.name is not None:
            payload["name"] = self.name
        return payload


@dataclass(slots=True)
class AttachmentSet:
    images: list[ImageAttachment] = field(default_factory=list)
    documents: list[DocumentAttachment] = field(default_factory=list)
    decoded_bytes: int = 0

    def __bool__(self) -> bool:
        return bool(self.images or self.documents)

    @property
    def count(self) -> int:
        return len(self.images) + len(self.documents)


def extract_attachments(
    payload: dict[str, Any],
    collected: AttachmentSet,
    *,
    max_attachments: int,
    max_attachment_bytes: int,
) -> dict[str, Any]:
    """Pull binary content parts out of one serialized message.

    Returns the message payload with every attachment part replaced by a short
    text placeholder, so the transcript keeps positional context without
    carrying the base64 blob a second time. Extracted parts are appended to
    ``collected`` for delivery over the SDK's native attachment channel.
    """
    content = payload.get("content")
    if not isinstance(content, list):
        return payload

    rewritten: list[Any] = []
    changed = False
    for part in content:
        if not isinstance(part, dict):
            rewritten.append(part)
            continue
        placeholder = _extract_part(
            part,
            collected,
            max_attachments=max_attachments,
            max_attachment_bytes=max_attachment_bytes,
        )
        if placeholder is None:
            rewritten.append(part)
            continue
        rewritten.append(placeholder)
        changed = True

    if not changed:
        return payload
    return {**payload, "content": rewritten}


def _extract_part(
    part: dict[str, Any],
    collected: AttachmentSet,
    *,
    max_attachments: int,
    max_attachment_bytes: int,
) -> dict[str, Any] | None:
    part_type = part.get("type")
    if part_type == "image_url":
        url = _image_url(part)
        media_type, data = _decode_data_uri(url, IMAGE_MEDIA_TYPES, "image")
        _account(collected, data, max_attachments, max_attachment_bytes)
        collected.images.append(ImageAttachment(media_type=media_type, data=data))
        return _placeholder(f"image #{len(collected.images)} ({media_type})")
    if part_type == "file":
        return _extract_file_part(
            part,
            collected,
            max_attachments=max_attachments,
            max_attachment_bytes=max_attachment_bytes,
        )
    if part_type in {"input_audio", "audio"}:
        raise AttachmentError("audio content parts are not supported by the bridge")
    return None


def _extract_file_part(
    part: dict[str, Any],
    collected: AttachmentSet,
    *,
    max_attachments: int,
    max_attachment_bytes: int,
) -> dict[str, Any]:
    descriptor = part.get("file")
    if not isinstance(descriptor, dict):
        raise AttachmentError("file content part must contain a 'file' object")
    if descriptor.get("file_id") is not None and descriptor.get("file_data") is None:
        raise AttachmentError("file_id references are not supported; inline file_data is required")
    file_data = descriptor.get("file_data")
    if not isinstance(file_data, str) or not file_data:
        raise AttachmentError("file content part must contain inline 'file_data'")

    media_type, data = _decode_data_uri(file_data, DOCUMENT_MEDIA_TYPES, "file")
    _account(collected, data, max_attachments, max_attachment_bytes)
    name = descriptor.get("filename")
    collected.documents.append(
        DocumentAttachment(
            media_type=media_type,
            data=data,
            # The SDK distinguishes base64 blobs from inline plain text.
            source_type="text" if media_type == "text/plain" else "base64",
            name=name if isinstance(name, str) and name else None,
        )
    )
    label = f"file #{len(collected.documents)} ({media_type})"
    if isinstance(name, str) and name:
        label = f"{label} named {name}"
    return _placeholder(label)


def _image_url(part: dict[str, Any]) -> str:
    image_url = part.get("image_url")
    url = image_url.get("url") if isinstance(image_url, dict) else image_url
    if not isinstance(url, str) or not url:
        raise AttachmentError("image_url content part must contain a URL")
    return url


def _decode_data_uri(
    value: str,
    allowed: frozenset[str],
    label: str,
) -> tuple[str, str]:
    match = _DATA_URI.match(value)
    if match is None:
        raise AttachmentError(
            f"remote {label} URLs are not supported; inline base64 data URIs are required"
        )
    media_type = match.group(1).lower()
    parameters = match.group(2) or ""
    payload = match.group(3)
    if media_type not in allowed:
        supported = ", ".join(sorted(allowed))
        raise AttachmentError(
            f"unsupported {label} media type '{media_type}'. Expected one of: {supported}."
        )

    if ";base64" in parameters.lower():
        try:
            decoded = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise AttachmentError(f"{label} data is not valid base64: {exc}") from exc
        return media_type, base64.b64encode(decoded).decode("ascii")

    if media_type != "text/plain":
        raise AttachmentError(f"{label} data URI must be base64 encoded")
    return media_type, payload


def _account(
    collected: AttachmentSet,
    data: str,
    max_attachments: int,
    max_attachment_bytes: int,
) -> None:
    if collected.count >= max_attachments:
        raise AttachmentError(f"request exceeds maximum of {max_attachments} attachments")
    collected.decoded_bytes += len(data.encode("utf-8"))
    if collected.decoded_bytes > max_attachment_bytes:
        raise AttachmentError(f"attachments exceed maximum of {max_attachment_bytes} bytes")


def _placeholder(label: str) -> dict[str, Any]:
    return {"type": "text", "text": f"[attached {label}]"}
