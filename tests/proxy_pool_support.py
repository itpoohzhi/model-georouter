"""Небольшие типизированные заготовки для тестов retry/backoff прокси-пула."""

from __future__ import annotations

import socket

from model_georouter.proxy_pool import EgressSettings
from tests.fake_server import FakeServer


class StallingFakeServer(FakeServer):
    """Принимает соединение и ждёт закрытия тестового ресурса."""

    def __init__(self) -> None:
        super().__init__(self._stall)

    def _stall(self, _conn: socket.socket) -> None:
        _ = self.stop.wait(3)


def backoff_settings(retries: int, **overrides: float) -> EgressSettings:
    values = {
        "retry_sleep_seconds": 0.5,
        "retry_backoff_factor": 2.0,
        "retry_max_sleep_seconds": 30.0,
        **overrides,
    }
    return EgressSettings(
        connect_timeout=2.0,
        total_deadline=60.0,
        retries=retries,
        penalty_seconds=30,
        retry_sleep_seconds=values["retry_sleep_seconds"],
        retry_backoff_factor=values["retry_backoff_factor"],
        retry_max_sleep_seconds=values["retry_max_sleep_seconds"],
    )
