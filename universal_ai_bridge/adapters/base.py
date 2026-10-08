"""Базовые типы клиентских адаптеров: распознавание префиксов и нормализация путей."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import unquote

from ..errors import BadRequestError


@dataclass(frozen=True)
class AdapterRoute:
    """Результат разбора входящего пути: адаптер, имя upstream и нормализованный путь для upstream."""

    adapter: str
    upstream: str
    path: str


def clean_path(path: str) -> str:
    """Схлопнуть `//`, `.` и хвостовой `/`; `..` (в том числе %2e%2e) и управляющие символы — 400."""
    if not path.startswith("/"):
        raise BadRequestError("request path must start with '/'")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in path):
        raise BadRequestError("request path contains control characters")
    segments = []
    for segment in path.split("/"):
        if segment in ("", "."):
            continue
        if unquote(segment) in ("..", ".") or "\\" in unquote(segment):
            raise BadRequestError("request path contains forbidden segments")
        segments.append(segment)
    return "/" + "/".join(segments)


def prefix_match(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + "/")


class ClientAdapter:
    """Адаптер клиента: владеет набором префиксов и знает, как перевести путь в путь upstream."""

    name = "adapter"

    def __init__(self, prefixes: tuple[str, ...], upstream: str):
        self.prefixes = prefixes
        self.upstream = upstream

    def match_length(self, path: str) -> int:
        """Длина самого длинного подходящего префикса (0 — путь не наш)."""
        return max((len(p) for p in self.prefixes if prefix_match(path, p)), default=0)

    def matches(self, path: str) -> bool:
        return self.match_length(clean_path(path)) > 0

    def normalize(self, path: str) -> str:
        return clean_path(path)

    def route(self, path: str) -> AdapterRoute | None:
        cleaned = clean_path(path)
        if not self.match_length(cleaned):
            return None
        return AdapterRoute(self.name, self.upstream, self.normalize(cleaned))
