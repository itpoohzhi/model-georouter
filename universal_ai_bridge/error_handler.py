"""Слой 4: безопасное преобразование сетевых/прокси ошибок в JSON-ответы (никогда не 403)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .errors import (
    AllProxiesPenalizedError,
    BadRequestError,
    BodyTooLargeError,
    ClientTimeoutError,
    EgressConnectError,
    EgressTimeoutError,
    NotFoundError,
    SlotsExhaustedError,
    UpstreamProtocolError,
    UpstreamTimeoutError,
)
from .logging_utils import mask_secrets

REASONS = {
    200: "OK",
    400: "Bad Request",
    404: "Not Found",
    405: "Method Not Allowed",
    408: "Request Timeout",
    413: "Payload Too Large",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
}
RETRY_AFTER_SECONDS = 5


@dataclass(frozen=True)
class ErrorResponse:
    status: int
    payload: dict
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def body(self) -> bytes:
        return json.dumps(self.payload, separators=(",", ":")).encode("utf-8")

    def to_bytes(self) -> bytes:
        return json_response_bytes(self.status, self.payload, self.headers)


def json_response_bytes(status: int, payload: dict, headers: dict[str, str] | None = None) -> bytes:
    """Полный HTTP/1.1-ответ с JSON-телом и `Connection: close`."""
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    lines = [
        f"HTTP/1.1 {status} {REASONS.get(status, 'Status')}",
        "Content-Type: application/json",
        f"Content-Length: {len(body)}",
        "Connection: close",
    ]
    lines.extend(f"{key}: {value}" for key, value in (headers or {}).items())
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body


class ErrorHandler:
    """Каждая ошибка транспорта → 502/504 с `retryable: true`; текст очищается от секретов."""

    def build(
        self,
        status: int,
        etype: str,
        message: str,
        *,
        retryable: bool,
        headers: dict[str, str] | None = None,
        extra: dict | None = None,
    ) -> ErrorResponse:
        if status == 403:  # инвариант: сбой транспорта не маскируется под отказ в доступе
            status = 502
        error = {"type": etype, "message": mask_secrets(message), "retryable": retryable}
        error.update(extra or {})
        return ErrorResponse(status, {"error": error}, dict(headers or {}))

    def convert(self, exc: BaseException) -> ErrorResponse:
        if isinstance(exc, AllProxiesPenalizedError):
            return self.build(502, "proxy_unavailable", str(exc), retryable=True)
        if isinstance(exc, EgressTimeoutError):
            return self.build(504, "proxy_gateway_timeout", str(exc), retryable=True)
        if isinstance(exc, EgressConnectError):
            return self.build(502, "proxy_connect_failed", str(exc), retryable=True)
        if isinstance(exc, UpstreamTimeoutError):
            return self.build(504, "upstream_timeout", str(exc) or "upstream sent no response headers in time", retryable=True)
        if isinstance(exc, UpstreamProtocolError):
            return self.build(502, "upstream_unreachable", str(exc), retryable=True)
        if isinstance(exc, SlotsExhaustedError):
            return self.build(
                503,
                "bridge_overloaded",
                str(exc),
                retryable=True,
                headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
                extra={"retry_after_seconds": RETRY_AFTER_SECONDS},
            )
        if isinstance(exc, BodyTooLargeError):
            return self.build(413, "payload_too_large", str(exc), retryable=False, extra={"limit_bytes": exc.limit})
        if isinstance(exc, ClientTimeoutError):
            return self.build(408, "request_timeout", str(exc), retryable=False)
        if isinstance(exc, NotFoundError):
            return self.build(404, "not_found", str(exc), retryable=False)
        if isinstance(exc, BadRequestError):
            return self.build(400, "bad_request", str(exc), retryable=False)
        return self.build(502, "bridge_internal_error", "internal bridge error", retryable=False)
