"""Общий monotonic-дедлайн egress и контракт retry-бюджета F-08."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TypeGuard

from .errors import ConfigError
from .transport_failover import (
    MAX_RETRY_RATIO,
    MAX_TOTAL_ATTEMPTS,
    NO_RETRY_HEADER,
    STATELESS_GENERATION_ENDPOINTS,
    BreakerState,
    RetryReservation,
    RouteCandidate,
    RouteCircuitBreaker,
    TokenBucketRetryBudget,
    evaluate_replay,
    panic_open_candidates,
    run_failover,
    select_next_route,
)

__all__ = [
    "BRIDGE_OWNER",
    "F08_V2_CLIENT_MARGIN",
    "F08_V2_NON_STREAM_DEADLINE",
    "F08_V2_POLICY",
    "F08_V2_RISK_ACK",
    "F08_V2_STREAM_MARKER",
    "F08_V2_T_IDLE",
    "F08_V2_T_PRE",
    "GEOROUTER_OWNER",
    "MAX_RETRY_RATIO",
    "MAX_TOTAL_ATTEMPTS",
    "NO_OWNER",
    "NO_RETRY_HEADER",
    "STATELESS_GENERATION_ENDPOINTS",
    "AttemptEvent",
    "AttemptState",
    "BreakerState",
    "EgressDeadline",
    "F08V2Budget",
    "RetryBudgetSpec",
    "RetryReservation",
    "RouteCandidate",
    "RouteCircuitBreaker",
    "TokenBucketRetryBudget",
    "bridge_replay_allowed",
    "consume_live_retry",
    "effective_attempt_cap",
    "evaluate_replay",
    "evaluate_replay_for_endpoint",
    "no_retry_requested",
    "normalize_endpoint",
    "panic_open_candidates",
    "parse_control_headers",
    "resolve_retry_owner",
    "run_failover",
    "select_next_route",
    "transition",
    "validate_f08_v2",
]

_F08_MAX_TOTAL_EGRESS_SECONDS = 40.0
_F08_MAX_CONNECT_SECONDS = 3.0
_F08_MAX_PRE_SEND_RETRIES = 2
_F08_MAX_PRIMARY_HEADER_SECONDS = 5.0
F08_V2_POLICY, F08_V2_STREAM_MARKER, F08_V2_RISK_ACK = "v2", "streamGenerateContent", "AD-003/2026-11-14"
F08_V2_T_PRE, F08_V2_T_IDLE, F08_V2_NON_STREAM_DEADLINE, F08_V2_CLIENT_MARGIN = 19.0, 150.0, 40.0, 11.0


def _require_finite_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0: raise ValueError(f"{name} must be finite and positive")  # fmt: skip


def _require_finite_non_negative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0: raise ValueError(f"{name} must be finite and non-negative")  # fmt: skip


def _require_non_negative_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0: raise ValueError(f"{name} must be a non-negative integer")  # fmt: skip


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

    def slide(self, idle_seconds: float = F08_V2_T_IDLE, *, now: float | None = None) -> EgressDeadline:
        """Сбросить скользящий дедлайн чтения при получении очередного чанка."""
        return EgressDeadline.from_timeout(idle_seconds, started_at=now)


@dataclass(frozen=True, slots=True)
class F08V2Budget:
    """Бюджет политики F-08v2 (AD-003, вариант B: 19s до первого байта + 150s скользящий stream)."""

    t_pre: float = F08_V2_T_PRE
    t_idle: float = F08_V2_T_IDLE
    non_stream_deadline: float = F08_V2_NON_STREAM_DEADLINE
    scope: str = F08_V2_STREAM_MARKER
    client_timeout_s: float | None = None
    f08_v2_risk_ack: str | None = None
    T_pre: float = F08_V2_T_PRE
    T_idle: float = F08_V2_T_IDLE

    def resolve_for_path(self, path: str, *, deadline: EgressDeadline | None = None, started_at: float | None = None, is_stream: bool = False) -> tuple[EgressDeadline, float, float | None]:  # fmt: skip
        """Вычислить `(deadline, t_pre, sliding_idle_s)` для HTTP-пути и признака потока запроса."""
        streaming = self.scope in path or is_stream
        dl = deadline or EgressDeadline.from_timeout(self.t_idle if streaming else self.non_stream_deadline, started_at=started_at)  # fmt: skip
        return dl, self.t_pre, (self.t_idle if streaming else None)


def _is_finite_num(val: object) -> TypeGuard[int | float]:
    return not isinstance(val, bool) and isinstance(val, int | float) and math.isfinite(val)


def validate_f08_v2(t_pre: object = F08_V2_T_PRE, t_idle: object = F08_V2_T_IDLE, client_timeout_s: object = None, f08_v2_risk_ack: object = None, scope: object = F08_V2_STREAM_MARKER, *, T_pre: object = None, T_idle: object = None, non_stream_deadline: object = F08_V2_NON_STREAM_DEADLINE) -> F08V2Budget:  # fmt: skip
    """Проверить инварианты политики F-08v2 (fail-closed) и вернуть подтверждённый `F08V2Budget`."""
    pre, idle = (t_pre if T_pre is None else T_pre), (t_idle if T_idle is None else T_idle)
    if (
        not _is_finite_num(pre)
        or float(pre) != F08_V2_T_PRE
        or not _is_finite_num(idle)
        or float(idle) != F08_V2_T_IDLE
    ):
        raise ConfigError(f"f08_v2: T_pre must be 19s and T_idle must be 150s, got T_pre={pre!r}, T_idle={idle!r}")
    if not _is_finite_num(non_stream_deadline) or float(non_stream_deadline) != F08_V2_NON_STREAM_DEADLINE:
        raise ConfigError(f"f08_v2.non_stream_deadline: must be 40s, got {non_stream_deadline!r}")
    if not isinstance(scope, str) or scope != F08_V2_STREAM_MARKER:
        raise ConfigError(f"f08_v2.scope: must be {F08_V2_STREAM_MARKER!r}, got {scope!r}")
    if client_timeout_s is not None:
        if (
            not _is_finite_num(client_timeout_s)
            or F08_V2_T_PRE + F08_V2_T_IDLE > float(client_timeout_s) - F08_V2_CLIENT_MARGIN
        ):
            raise ConfigError(
                f"f08_v2.client_timeout_s: T_pre + T_idle must be <= client_timeout_s - 11, got {client_timeout_s!r}"
            )
        ct_val: float | None = float(client_timeout_s)
    elif f08_v2_risk_ack != F08_V2_RISK_ACK:
        raise ConfigError(f"f08_v2.f08_v2_risk_ack: must be {F08_V2_RISK_ACK!r} when client_timeout_s is None")
    else:
        ct_val = None
    return F08V2Budget(F08_V2_T_PRE, F08_V2_T_IDLE, F08_V2_NON_STREAM_DEADLINE, scope, ct_val, f08_v2_risk_ack if isinstance(f08_v2_risk_ack, str) else None)  # fmt: skip


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
    has_alternative: bool = True
    penalize_without_alternative: bool = True
    status: str = "pending"
    should_penalize: bool = False
    next_pool: str = ""
    is_long_header: bool = False


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
        if state.single_proxy:
            return True
        if state.is_long_header:
            return False
        if not state.has_alternative and not state.penalize_without_alternative:
            return False
        return state.penalize_on_header_timeout
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


# --- Фасад W3-политики (`transport_failover`) для bridge / geo-router ---

BRIDGE_OWNER, GEOROUTER_OWNER, NO_OWNER = "BRIDGE_OWNER", "GEOROUTER_OWNER", "NONE"
_OWNER_BY_LABEL = {"cli-agent-bridge": BRIDGE_OWNER, "model-georouter": GEOROUTER_OWNER}


def parse_control_headers(headers: Mapping[str, object]) -> dict[str, str]:
    """Разобрать управляющие заголовки запроса ОДИН раз на запрос (lower-case, strip)."""
    return {str(key).strip().lower(): str(value).strip() for key, value in headers.items()}


def no_retry_requested(control: Mapping[str, str]) -> bool:
    """Доверенный `X-Georouter-No-Retry` может запрещать повторы, но не разрешать их."""
    return control.get(NO_RETRY_HEADER.lower(), "").lower() in {"1", "true", "yes"}


def resolve_retry_owner(control: Mapping[str, str], declared: str = "") -> str:
    """Владелец retry; расхождение поля и заголовка трактуется как `NO_OWNER` (fail-closed)."""
    field, published = declared.strip().upper(), _OWNER_BY_LABEL.get(control.get("x-failover-owner", ""), "")
    if field and published and field != published:
        return NO_OWNER
    if no_retry_requested(control):
        return BRIDGE_OWNER
    resolved = published or field
    return resolved if resolved in (BRIDGE_OWNER, GEOROUTER_OWNER) else NO_OWNER


def effective_attempt_cap(control: Mapping[str, str], owner: str, configured: int = MAX_TOTAL_ATTEMPTS) -> int:
    """Число физических попыток; клиентские заголовки никогда не увеличивают бюджет."""
    if owner != GEOROUTER_OWNER or no_retry_requested(control):
        return 1
    return max(1, min(MAX_TOTAL_ATTEMPTS, configured))


def normalize_endpoint(path: str) -> str:
    """F-113: stateless-проверка идёт по нормализованному path без query-string."""
    return path.split("?", 1)[0].strip() or "/"


def is_stateless_endpoint(path: str) -> bool:
    """F-113: stateless-эндпоинт определяется по нормализованному path."""
    return normalize_endpoint(path) in STATELESS_GENERATION_ENDPOINTS


def evaluate_replay_for_endpoint(facts: Mapping[str, object]) -> dict[str, object]:
    """Replay-контракт W3, но с F-113-нормализацией endpoint перед проверкой."""
    return evaluate_replay({**facts, "endpoint": normalize_endpoint(str(facts.get("endpoint") or "/"))})


def consume_live_retry(headers: Mapping[str, object], budget: TokenBucketRetryBudget, *, enabled: bool, declared: str = "", configured: int = MAX_TOTAL_ATTEMPTS) -> bool:  # fmt: skip
    """F-004/F-112: primary не блокируется пустым bucket; replay требует потреблённого токена."""
    if not enabled:
        return False
    control = parse_control_headers(headers)
    cap = effective_attempt_cap(control, resolve_retry_owner(control, declared), configured)
    reservation = budget.reserve_upfront(max(1, cap), cap) if cap > 1 else None
    if reservation is None or not reservation.consume_one():
        return False
    _ = reservation.release_unused()
    return True


def bridge_replay_allowed(headers: Mapping[str, object]) -> bool:
    """Фасад geo-replay: заголовки клиента могут только запретить replay, но не разрешить его."""
    control = parse_control_headers(headers)
    return not no_retry_requested(control) and resolve_retry_owner(control) != BRIDGE_OWNER
