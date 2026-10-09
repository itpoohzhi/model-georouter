"""Admin-адаптер: `/health`, `/metrics`, `/cache/flush` (обслуживаются мостом локально, без upstream)."""

from __future__ import annotations

from typing import Protocol

from ..errors import BadRequestError
from .base import clean_path


class AdminProvider(Protocol):
    def health_payload(self) -> dict: ...

    def metrics_payload(self) -> dict: ...

    def flush_cache(self) -> int: ...


ADMIN_METHODS = {
    "/health": ("GET", "HEAD"),
    "/metrics": ("GET", "HEAD"),
    "/cache/flush": ("POST",),
}


class AdminAdapter:
    name = "admin"

    def matches(self, path: str) -> bool:
        try:
            return clean_path(path) in ADMIN_METHODS
        except BadRequestError:  # некорректный путь не может быть админским
            return False

    def handle(self, method: str, path: str, provider: AdminProvider) -> tuple[int, dict, dict[str, str]]:
        """Вернуть (status, JSON-payload, extra headers)."""
        path = clean_path(path)
        allowed = ADMIN_METHODS[path]
        if method not in allowed:
            error = {"type": "method_not_allowed", "message": f"use {', '.join(allowed)}", "retryable": False}
            return 405, {"error": error}, {"Allow": ", ".join(allowed)}
        if path == "/health":
            return 200, provider.health_payload(), {}
        if path == "/metrics":
            return 200, provider.metrics_payload(), {}
        return 200, {"flushed": provider.flush_cache()}, {}
