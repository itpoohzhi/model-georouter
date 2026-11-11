"""Слой 1: клиентские адаптеры и реестр разрешения путей."""

from __future__ import annotations

from collections.abc import Sequence

from ..config import AdapterConfig, BridgeConfig
from .admin import AdminAdapter
from .base import AdapterRoute, ClientAdapter, PassthroughAdapter, clean_path
from .cloudcode import CloudCodeAdapter
from .cordis import CordisAdapter
from .generic import GenericAdapter
from .opencode import OpenCodeAdapter

ADAPTER_CLASSES: dict[str, type[ClientAdapter]] = {
    "opencode": OpenCodeAdapter,
    "cordis": CordisAdapter,
    "generic": GenericAdapter,
    "cloudcode": CloudCodeAdapter,
}


def _build_adapter(name: str, adapter: AdapterConfig) -> ClientAdapter:
    """Встроенное имя — свой класс; любое новое из конфига — passthrough со снятием префикса."""
    factory = ADAPTER_CLASSES.get(name)
    if factory is not None:
        return factory(adapter.prefixes, adapter.upstream, pool=adapter.pool)
    return PassthroughAdapter(adapter.prefixes, adapter.upstream, pool=adapter.pool, name=name)


class AdapterRegistry:
    """Выбирает адаптер по самому длинному совпавшему префиксу пути."""

    def __init__(self, adapters: Sequence[ClientAdapter]):
        self.adapters = tuple(adapters)

    @classmethod
    def from_config(cls, config: BridgeConfig) -> AdapterRegistry:
        return cls([_build_adapter(name, adapter) for name, adapter in config.adapters.items()])

    def get(self, name: str) -> ClientAdapter | None:
        """Получить адаптер по его имени."""
        for adapter in self.adapters:
            if adapter.name == name:
                return adapter
        return None

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
    "CloudCodeAdapter",
    "CordisAdapter",
    "GenericAdapter",
    "OpenCodeAdapter",
    "PassthroughAdapter",
    "clean_path",
]
