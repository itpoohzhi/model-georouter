"""Хелперы ограничения попыток и переиспользования RouteCircuitBreaker без правки transport_failover.py."""

from __future__ import annotations

import os

from .transport_failover import RouteCircuitBreaker

FLAG_BREAKER_FLOOR = "GEOROUTER_BREAKER_FLOOR_ENABLED"
DEFAULT_MAX_ATTEMPTS_PER_OP = 2
DEFAULT_SETUP_CAP_S = 2.5
MIN_COOLDOWN_TICKS = 10
MAX_FAILURE_THRESHOLD = 5


def breaker_floor_enabled() -> bool:
    """True если включено принудительное соблюдение floor брейкера и лимита попыток."""
    raw = os.environ.get(FLAG_BREAKER_FLOOR, "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def floor_breaker_params(failure_threshold: int = 3, cooldown_ticks: int = 10) -> tuple[int, int]:
    """Возвращает (failure_threshold <= 5, cooldown_ticks >= 10) для RouteCircuitBreaker."""
    thresh = max(1, min(MAX_FAILURE_THRESHOLD, int(failure_threshold)))
    cooldown = max(MIN_COOLDOWN_TICKS, int(cooldown_ticks))
    return thresh, cooldown


def build_route_breaker(route_id: str, failure_threshold: int = 3, cooldown_ticks: int = 10) -> RouteCircuitBreaker:
    """Создаёт штатный RouteCircuitBreaker с соблюдением floor-порогов."""
    thresh, cooldown = floor_breaker_params(failure_threshold, cooldown_ticks)
    return RouteCircuitBreaker(route_id, failure_threshold=thresh, cooldown_ticks=cooldown)


def bounded_setup_timeout(remaining_s: float, connect_timeout_s: float, *, max_setup_s: float = DEFAULT_SETUP_CAP_S) -> float:
    """Ограничивает T_setup одной попытки <= 2.5s (с учётом оставшегося дедлайна)."""
    if not breaker_floor_enabled():
        return max(0.01, min(remaining_s, connect_timeout_s))
    return max(0.01, min(remaining_s, connect_timeout_s, max_setup_s))


def bounded_max_attempts(configured_retries: int, *, cap: int = DEFAULT_MAX_ATTEMPTS_PER_OP) -> int:
    """Возвращает число retry-итераций (retries) при общем потолке attempts/op <= cap."""
    if not breaker_floor_enabled():
        return max(0, int(configured_retries))
    return min(max(0, int(configured_retries)), max(0, cap - 1))


def should_reject_breaker_floor(
    acquired_primary: bool,
    pool_name: str,
    decision_pool: str,
    selected_breaker: RouteCircuitBreaker,
    now_tick: int,
) -> bool:
    """True -> fail-fast 503; acquired_primary True покрывает CLOSED и 1-й HALF_OPEN probe."""
    if not breaker_floor_enabled() or acquired_primary:
        return False
    return pool_name == decision_pool or not selected_breaker.allows(now_tick)
