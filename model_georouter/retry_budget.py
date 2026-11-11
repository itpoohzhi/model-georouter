"""Общий monotonic-дедлайн egress и контракт retry-бюджета F-08."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

__all__ = ["AttemptEvent", "AttemptState", "EgressDeadline", "RetryBudgetSpec", "transition"]

_F08_MAX_TOTAL_EGRESS_SECONDS = 40.0
_F08_MAX_CONNECT_SECONDS = 3.0
_F08_MAX_PRE_SEND_RETRIES = 2
_F08_MAX_PRIMARY_HEADER_SECONDS = 5.0


def _require_finite_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _require_finite_non_negative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def _require_non_negative_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class EgressDeadline:
    """Абсолютный monotonic-дедлайн, общий для всех фаз одного egress-запроса."""

    absolute_deadline: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.absolute_deadline):
            raise ValueError("absolute_deadline must be finite")

    @classmethod
    def from_timeout(
        cls,
        total_seconds: float,
        *,
        started_at: float | None = None,
    ) -> EgressDeadline:
        """Создать дедлайн один раз в начале egress-запроса."""
        _require_finite_positive(total_seconds, "total_seconds")
        started = time.monotonic() if started_at is None else started_at
        if not math.isfinite(started):
            raise ValueError("started_at must be finite")
        return cls(absolute_deadline=started + total_seconds)

    def remaining(self, *, now: float | None = None) -> float:
        """Вернуть неотрицательный остаток общего бюджета в секундах."""
        current = time.monotonic() if now is None else now
        if not math.isfinite(current):
            raise ValueError("now must be finite")
        return max(0.0, self.absolute_deadline - current)

    def expired(self, *, now: float | None = None) -> bool:
        """Проверить, исчерпан ли общий egress-бюджет."""
        return self.remaining(now=now) <= 0.0

    def clamp(self, requested_timeout: float, *, now: float | None = None) -> float:
        """Ограничить таймаут фазы остатком общего дедлайна."""
        _require_finite_positive(requested_timeout, "requested_timeout")
        remaining = self.remaining(now=now)
        if remaining <= 0.0:
            raise TimeoutError("egress deadline expired")
        return min(requested_timeout, remaining)


@dataclass(frozen=True, slots=True)
class RetryBudgetSpec:
    """Параметры retry/failover, удовлетворяющие инварианту F-08."""

    total_egress_deadline: float = 40.0
    connect_timeout: float = 3.0
    pre_send_retries: int = 2
    max_backoff_seconds: float = 2.0
    single_pool_max_retries: int = 0
    primary_first_header_timeout: float = 5.0
    penalize_on_header_timeout: bool = True

    @property
    def multi_proxy_attempts(self) -> int:
        """Число попыток: первая плюс разрешённые pre-send retries."""
        return self.pre_send_retries + 1

    @property
    def worst_case_pre_send_seconds(self) -> float:
        """Верхняя граница connect-попыток вместе с суммарным backoff."""
        return self.multi_proxy_attempts * self.connect_timeout + self.max_backoff_seconds

    def validate_f08(self) -> None:
        """Проверить математические и failover-ограничения F-08."""
        _require_finite_positive(self.total_egress_deadline, "total_egress_deadline")
        _require_finite_positive(self.connect_timeout, "connect_timeout")
        _require_non_negative_integer(self.pre_send_retries, "pre_send_retries")
        _require_finite_non_negative(self.max_backoff_seconds, "max_backoff_seconds")
        _require_non_negative_integer(self.single_pool_max_retries, "single_pool_max_retries")
        _require_finite_positive(self.primary_first_header_timeout, "primary_first_header_timeout")

        if self.worst_case_pre_send_seconds > self.total_egress_deadline:
            raise ValueError("pre-send budget must fit within total_egress_deadline")
        if self.total_egress_deadline > _F08_MAX_TOTAL_EGRESS_SECONDS:
            raise ValueError("total_egress_deadline must not exceed 40 seconds")
        if self.connect_timeout > _F08_MAX_CONNECT_SECONDS:
            raise ValueError("connect_timeout must not exceed 3 seconds")
        if self.pre_send_retries > _F08_MAX_PRE_SEND_RETRIES:
            raise ValueError("pre_send_retries must not exceed 2")
        if self.single_pool_max_retries != 0:
            raise ValueError("single_pool_max_retries must be 0 when failover is configured")
        if self.primary_first_header_timeout > _F08_MAX_PRIMARY_HEADER_SECONDS:
            raise ValueError("primary_first_header_timeout must not exceed 5 seconds")
        if not self.penalize_on_header_timeout:
            raise ValueError("penalize_on_header_timeout must be enabled")


_FAILURE_EVENTS = frozenset({"penalized", "connect_error", "header_timeout", "protocol_error"})


@dataclass(frozen=True, slots=True)
class AttemptState:
    """Неизменяемое состояние попытки маршрутизации и failover (F-FSM-I1)."""

    pool: str
    fallback_pool: str = ""
    single_proxy: bool = False
    committed: bool = False
    penalize_on_header_timeout: bool = True
    status: str = "pending"
    should_penalize: bool = False
    next_pool: str = ""


@dataclass(frozen=True, slots=True)
class AttemptEvent:
    """Неизменяемое событие перехода состояния попытки маршрутизации."""

    kind: str
    deadline_expired: bool = False
    has_endpoint: bool = False


def _should_penalize(state: AttemptState, event: AttemptEvent) -> bool:
    if not event.has_endpoint:
        return False
    if event.kind == "protocol_error":
        return state.single_proxy
    if event.kind == "header_timeout":
        return state.single_proxy or state.penalize_on_header_timeout
    return False


def transition(state: AttemptState, event: AttemptEvent) -> AttemptState:
    """Чистая функция перехода `transition(state, event) -> new_state` над неизменяемым состоянием."""
    if event.kind in ("committed", "completed"):
        return replace(state, committed=True, status=event.kind, should_penalize=False, next_pool="")
    if event.kind not in _FAILURE_EVENTS:
        raise ValueError(f"unsupported attempt event: {event.kind}")
    penalize = _should_penalize(state, event)
    if state.single_proxy and not state.committed and not event.deadline_expired and state.fallback_pool:
        return replace(
            state,
            pool=state.fallback_pool,
            single_proxy=False,
            status="failover",
            should_penalize=penalize,
            next_pool=state.fallback_pool,
        )
    return replace(state, status="failed", should_penalize=penalize, next_pool="")
