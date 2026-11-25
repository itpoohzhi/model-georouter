"""Unit tests for egress_policy_helpers (Wave 7)."""

from __future__ import annotations

from model_georouter.egress_policy_helpers import (
    bounded_max_attempts,
    bounded_setup_timeout,
    build_route_breaker,
    floor_breaker_params,
    should_reject_breaker_floor,
)
from model_georouter.transport_failover import RouteCircuitBreaker


def test_floor_breaker_params_and_route_breaker_reuse() -> None:
    assert floor_breaker_params(99, 2) == (5, 10)
    assert floor_breaker_params(2, 15) == (2, 15)
    brk = build_route_breaker("up-1", failure_threshold=20, cooldown_ticks=1)
    assert isinstance(brk, RouteCircuitBreaker)
    assert brk.failure_threshold == 5
    assert brk.cooldown_ticks == 10


def test_bounded_setup_and_attempts_with_rollback(monkeypatch) -> None:
    monkeypatch.delenv("GEOROUTER_BREAKER_FLOOR_ENABLED", raising=False)
    assert bounded_setup_timeout(10.0, 5.0) == 2.5
    assert bounded_max_attempts(5) == 1  # retries=1 -> total attempts <= 2

    monkeypatch.setenv("GEOROUTER_BREAKER_FLOOR_ENABLED", "0")
    assert bounded_setup_timeout(10.0, 5.0) == 5.0
    assert bounded_max_attempts(5) == 5


def test_should_reject_breaker_floor_closed_vs_open(monkeypatch) -> None:
    monkeypatch.delenv("GEOROUTER_BREAKER_FLOOR_ENABLED", raising=False)
    primary = build_route_breaker("p", failure_threshold=2, cooldown_ticks=10)
    fallback = build_route_breaker("fb", failure_threshold=2, cooldown_ticks=10)
    # CLOSED: try_acquire returns True -> acquired_primary=True -> healthy traffic allowed!
    assert primary.try_acquire(100) is True
    assert should_reject_breaker_floor(True, "p", "p", primary, 100) is False
    # OPEN: try_acquire returns False -> acquired_primary=False -> 503 fail-fast unless healthy fallback
    primary.record_failure(100); primary.record_failure(100)
    assert primary.try_acquire(101) is False
    assert should_reject_breaker_floor(False, "p", "p", primary, 101) is True
    assert should_reject_breaker_floor(False, "fb", "p", fallback, 101) is False
    fallback.record_failure(101); fallback.record_failure(101)
    assert should_reject_breaker_floor(False, "fb", "p", fallback, 101) is True
