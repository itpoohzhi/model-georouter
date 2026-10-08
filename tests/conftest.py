from __future__ import annotations

import pytest

from tests.helpers import BridgeHarness


@pytest.fixture
def resources():
    """Список закрываемых ресурсов (фейковые серверы, мосты); закрываются в обратном порядке."""
    items: list = []
    yield items
    for item in reversed(items):
        try:
            item.close()
        except Exception:  # noqa: BLE001
            pass


@pytest.fixture
def track(resources):
    def _track(item):
        resources.append(item)
        return item

    return _track


@pytest.fixture
def start_bridge(track):
    def _start(config: dict, **kwargs) -> BridgeHarness:
        return track(BridgeHarness(config, **kwargs))

    return _start
