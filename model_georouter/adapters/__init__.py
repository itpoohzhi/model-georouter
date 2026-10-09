"""Слой 1: клиентские адаптеры и реестр разрешения путей."""

from __future__ import annotations

from collections.abc import Sequence

from ..config import BridgeConfig
from .admin import AdminAdapter
from .base import AdapterRoute, ClientAdapter, clean_path
from .cordis import CordisAdapter
from .generic import GenericAdapter
from .opencode import OpenCodeAdapter

ADAPTER_CLASSES: dict[str, type[ClientAdapter]] = {
    "opencode": OpenCodeAdapter,
    "cordis": CordisAdapter,
    "generic": GenericAdapter,
}


class AdapterRegistry:
    """Выбирает адаптер по самому длинному совпавшему префиксу пути."""

    def __init__(self, adapters: Sequence[ClientAdapter]):
        self.adapters = tuple(adapters)

    @classmethod
    def from_config(cls, config: BridgeConfig) -> AdapterRegistry:
        return cls(
            [
                ADAPTER_CLASSES[name](adapter.prefixes, adapter.upstream)
                for name, adapter in config.adapters.items()
                if name in ADAPTER_CLASSES
            ]
        )

    def resolve(self, path: str) -> AdapterRoute | None:
        cleaned = clean_path(path)
        best: ClientAdapter | None = None
        best_length = 0
        for adapter in self.adapters:
            length = adapter.match_length(cleaned)
            if length > best_length:
                best, best_length = adapter, length
        return best.route(cleaned) if best else None


__all__ = [
    "ADAPTER_CLASSES",
    "AdapterRegistry",
    "AdapterRoute",
    "AdminAdapter",
    "ClientAdapter",
    "CordisAdapter",
    "GenericAdapter",
    "OpenCodeAdapter",
    "clean_path",
]
