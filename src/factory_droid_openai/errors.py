from __future__ import annotations


class ProtocolError(ValueError):
    pass


class RequestTooLargeError(ProtocolError):
    pass
