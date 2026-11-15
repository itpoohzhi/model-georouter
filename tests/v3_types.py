"""Типовые контракты общих фикстур и изменяемых тестовых конфигураций v3."""

from __future__ import annotations

from typing import NotRequired, Protocol, TypedDict, TypeVar

from tests.helpers import BridgeHarness

_Resource = TypeVar("_Resource")


class Track(Protocol):
    def __call__(self, item: _Resource) -> _Resource: ...


class StartBridge(Protocol):
    def __call__(self, config: object, **kwargs: object) -> BridgeHarness: ...


class ConfigData(TypedDict):
    server: dict[str, object]
    pools: dict[str, dict[str, object]]
    rules: list[dict[str, object]]
    upstreams: NotRequired[dict[str, object]]
