from __future__ import annotations

import logging

import pytest

from tests.helpers import BridgeHarness

LOGGER = logging.getLogger(__name__)


@pytest.fixture
def resources():
    """Список закрываемых ресурсов (фейковые серверы, мосты); закрываются в обратном порядке."""
    items: list = []
    yield items
    for item in reversed(items):
        try:
            item.close()
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("teardown close failed: %r", exc)


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
