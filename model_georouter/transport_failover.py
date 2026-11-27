"""Чистая доменная политика failover: budget, circuit breaker и replay FSM."""

from __future__ import annotations

import inspect
import math
import random
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Literal, cast, final

__all__ = ["MAX_RETRY_RATIO", "MAX_TOTAL_ATTEMPTS", "NO_RETRY_HEADER", "STATELESS_GENERATION_ENDPOINTS", "BreakerState", "RetryReservation", "RouteCandidate", "RouteCircuitBreaker", "TokenBucketRetryBudget", "evaluate_replay", "panic_open_candidates", "run_failover", "select_next_route"]  # fmt: skip

MAX_TOTAL_ATTEMPTS = 3
MAX_RETRY_RATIO = 0.20
NO_RETRY_HEADER = "X-Georouter-No-Retry"
STATELESS_GENERATION_ENDPOINTS = ("/v1/chat/completions", "/cloudcode/v1internal:streamGenerateContent")
HealthScope = Literal["origin", "node", "credential_scope"]
_SCOPES: tuple[HealthScope, ...] = ("origin", "node", "credential_scope")
_OWNERS = frozenset({"BRIDGE_OWNER", "GEOROUTER_OWNER", "NONE"})
_FAILURE_FLAGS = ("timed_out", "truncated", "protocol_error", "connect_error")


def _as_bool(value: object, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def _as_str(value: object, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _as_int(value: object, default: int = 0) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _headers(facts: Mapping[str, object]) -> dict[str, str]:
    raw = facts.get("control_headers")
    return (
        {str(key).lower(): str(val).strip() for key, val in cast("Mapping[object, object]", raw).items()}
        if isinstance(raw, Mapping)
        else {}
    )


def _status_known(facts: Mapping[str, object]) -> bool:
    status = facts.get("status_code")
    return isinstance(status, int) and not isinstance(status, bool)


def _ok_status(facts: Mapping[str, object]) -> bool:
    return _status_known(facts) and 200 <= _as_int(facts.get("status_code")) < 300


def _has_failure_flag(facts: Mapping[str, object]) -> bool:
    return any(_as_bool(facts.get(name)) for name in _FAILURE_FLAGS)


def _is_success(facts: Mapping[str, object]) -> bool:
    return _ok_status(facts) and not _has_failure_flag(facts)


def _no_retry(facts: Mapping[str, object], headers: Mapping[str, str]) -> bool:
    return _as_bool(facts.get("no_retry")) or headers.get(NO_RETRY_HEADER.lower(), "").lower() in {"1", "true", "yes"}


def _resolve_owner(facts: Mapping[str, object], headers: Mapping[str, str]) -> str:
    field_owner = _as_str(facts.get("retry_owner")).upper()
    by_header = {"cli-agent-bridge": "BRIDGE_OWNER", "model-georouter": "GEOROUTER_OWNER"}
    header_owner = by_header.get(headers.get("x-failover-owner", ""), "")
    if field_owner and header_owner and field_owner != header_owner:
        return "NONE"
    resolved = header_owner or field_owner
    if _no_retry(facts, headers):
        return "BRIDGE_OWNER"
    return resolved if resolved in _OWNERS else "NONE"


def _effective_cap(facts: Mapping[str, object], owner: str, no_retry: bool) -> int:
    if no_retry or owner != "GEOROUTER_OWNER":
        return 1
    return min(MAX_TOTAL_ATTEMPTS, max(1, _as_int(facts.get("max_total_attempts"), MAX_TOTAL_ATTEMPTS)))


def _normalized_endpoint(facts: Mapping[str, object]) -> str:
    return _as_str(facts.get("endpoint")).split("?", 1)[0].strip() or "/"


def _stateless(endpoint: str) -> bool:
    google = any(f"/v{ver}/models/" in endpoint for ver in ("1", "1beta")) and endpoint.endswith(
        (":generateContent", ":streamGenerateContent")
    )
    return endpoint in STATELESS_GENERATION_ENDPOINTS or google


@dataclass(frozen=True, slots=True)
class RouteCandidate:
    """Замороженный маршрут одного config epoch: identity, effort и credentials уже запинены."""

    route_id: str
    origin: str = ""
    node: str = ""
    credential_scope: str = ""
    priority: int = 0
    weight: float = 1.0
    canonical_model_id: str = ""
    effort: str = ""
    credential_valid: bool = True
    enabled: bool = True
    identity_compatible: bool = True

    def __post_init__(self) -> None:
        if not self.route_id or not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("route_id must be non-empty and weight finite positive")


class BreakerState(Enum):
    """Состояние per-route circuit breaker."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass(slots=True)
class _ScopeState:
    failures: int = 0
    opened_at: int | None = None


@final
class RouteCircuitBreaker:
    """Per-route breaker с раздельным здоровьем origin/node/credential_scope (thread-safe)."""

    __slots__ = ("_half_open_inflight", "_lock", "_scopes", "cooldown_ticks", "failure_threshold", "route_id")

    def __init__(self, route_id: str, failure_threshold: int = 3, cooldown_ticks: int = 10) -> None:
        if not route_id or failure_threshold < 1 or cooldown_ticks < 0:
            raise ValueError("route_id required, threshold>=1, cooldown>=0")
        self._lock: threading.Lock = threading.Lock()
        self._scopes: dict[HealthScope, _ScopeState] = {scope: _ScopeState() for scope in _SCOPES}
        self._half_open_inflight, self.route_id = 0, route_id
        self.failure_threshold, self.cooldown_ticks = failure_threshold, cooldown_ticks

    @property
    def half_open_inflight(self) -> int:
        return self._half_open_inflight

    def scope(self, scope: HealthScope) -> tuple[int, int | None]:
        state = self._scopes[scope]
        return state.failures, state.opened_at

    def _opened(self) -> list[int]:
        return [item.opened_at for item in self._scopes.values() if item.opened_at is not None]

    def oldest_opened_at(self) -> int | None:
        return min(self._opened(), default=None)

    def state_at(self, now_tick: int = 0) -> BreakerState:
        opened = self._opened()
        if not opened:
            return BreakerState.CLOSED
        return (
            BreakerState.OPEN
            if any(now_tick - item < self.cooldown_ticks for item in opened)
            else BreakerState.HALF_OPEN
        )

    def allows(self, now_tick: int = 0) -> bool:
        return self.state_at(now_tick) is not BreakerState.OPEN

    def try_acquire(self, now_tick: int = 0) -> bool:
        with self._lock:
            state = self.state_at(now_tick)
            if state is BreakerState.OPEN or (state is BreakerState.HALF_OPEN and self._half_open_inflight):
                return False
            self._half_open_inflight = int(state is BreakerState.HALF_OPEN)
            return True

    def release_probe(self, success: bool, now_tick: int = 0) -> None:
        with self._lock:
            self._half_open_inflight = 0
            for state in self._scopes.values():
                if success:
                    state.failures, state.opened_at = 0, None
                elif state.opened_at is not None:
                    state.opened_at = now_tick

    def record_failure(
        self, now_tick: int = 0, scope: HealthScope = "origin", *, client_validation: bool = False
    ) -> None:
        if client_validation:
            return
        with self._lock:
            state = self._scopes[scope]
            state.failures += 1
            if scope == "credential_scope" or state.failures >= self.failure_threshold:
                state.opened_at = now_tick

    def record_success(self, scope: HealthScope | None = None) -> None:
        with self._lock:
            for state in self._scopes.values() if scope is None else (self._scopes[scope],):
                state.failures, state.opened_at = 0, None
            if scope is None:
                self._half_open_inflight = 0

    def panic_eligible(self, now_tick: int = 0) -> bool:
        return self._scopes["credential_scope"].opened_at is None and self.state_at(now_tick) is BreakerState.OPEN


@final
class RetryReservation:
    """Upfront-резерв retry-токенов; `release_unused` срабатывает однократно."""

    __slots__ = ("_budget", "_consumed", "_released", "reserved")

    def __init__(self, budget: TokenBucketRetryBudget, reserved: int) -> None:
        self._budget, self.reserved, self._consumed, self._released = budget, reserved, 0, False

    def consume_one(self) -> bool:
        if self._released or self._consumed >= self.reserved:
            return False
        self._consumed += 1
        self._budget.note_consumed()
        return True

    def release_unused(self) -> int:
        if self._released:
            return 0
        self._released, unused = True, self.reserved - self._consumed
        self._budget.release_tokens(unused)
        return unused


def _fraction(value: object, name: str) -> Fraction:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return Fraction(str(value))


@final
class TokenBucketRetryBudget:
    """Token bucket: пустой на старте, ratio<=0.20, reload/restart токены не досыпает (thread-safe)."""

    __slots__ = ("_capacity", "_lock", "_ratio", "_tokens", "admitted_primary", "consumed_total")

    def __init__(self, retry_ratio: float = MAX_RETRY_RATIO, capacity: float = 10.0) -> None:
        ratio, cap = _fraction(retry_ratio, "retry_ratio"), _fraction(capacity, "capacity")
        if ratio < 0 or ratio > Fraction(str(MAX_RETRY_RATIO)) or cap < 0:
            raise ValueError("retry_ratio must be within [0, 0.20] and capacity>=0")
        self._lock = threading.Lock()
        self._ratio, self._capacity, self._tokens = ratio, cap, Fraction(0)
        self.admitted_primary, self.consumed_total = 0, 0

    @property
    def tokens(self) -> float:
        return float(self._tokens)

    @property
    def available_tokens(self) -> int:
        return self._tokens.numerator // self._tokens.denominator

    def mint_on_admitted_primary(self, count: object = 1) -> float:
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("count must be a non-negative integer")
        with self._lock:
            self.admitted_primary += count
            self._tokens = min(self._capacity, self._tokens + self._ratio * count)
            return float(self._tokens)

    def reload_config(self, retry_ratio: float | None = None, capacity: float | None = None) -> None:
        ratio = None if retry_ratio is None else _fraction(retry_ratio, "retry_ratio")
        if ratio is not None and (ratio < 0 or ratio > Fraction(str(MAX_RETRY_RATIO))):
            raise ValueError("retry_ratio must be within [0, 0.20]")
        with self._lock:
            self._ratio = ratio if ratio is not None else self._ratio
            if capacity is not None:
                cap = _fraction(capacity, "capacity")
                if cap < 0:
                    raise ValueError("capacity>=0")
                self._capacity, self._tokens = cap, min(self._tokens, cap)

    def reserve_upfront(self, eligible_unvisited_routes: int, configured_max_total_attempts: int) -> RetryReservation:
        if eligible_unvisited_routes < 0 or configured_max_total_attempts < 1:
            raise ValueError("eligible routes>=0 and max attempts>=1")
        with self._lock:
            reserved = min(
                self.available_tokens, max(0, eligible_unvisited_routes - 1), configured_max_total_attempts - 1, 2
            )
            self._tokens -= reserved
            return RetryReservation(self, reserved)

    def note_consumed(self) -> None:
        with self._lock:
            self.consumed_total += 1

    def release_tokens(self, amount: int) -> None:
        with self._lock:
            self._tokens = min(self._capacity, self._tokens + max(0, amount))


def _candidates(facts: Mapping[str, object]) -> list[RouteCandidate]:
    raw = facts.get("routes")
    given = (
        [item for item in raw if isinstance(item, RouteCandidate)]
        if isinstance(raw, Sequence) and not isinstance(raw, str | bytes)
        else []
    )
    canonical, effort = _as_str(facts.get("canonical_model_id")), _as_str(facts.get("effort"))
    if not given:
        return [RouteCandidate(route_id=canonical or "primary", canonical_model_id=canonical, effort=effort)]
    return [route for route in given if route.canonical_model_id == canonical and route.effort == effort]


def _usable(
    route: RouteCandidate,
    breaker: RouteCircuitBreaker | None,
    tried: frozenset[str],
    now_tick: int = 0,
    *,
    panic: bool = False,
) -> bool:
    if route.route_id in tried or not (route.enabled and route.credential_valid and route.identity_compatible):
        return False
    if breaker is None:
        return True
    return breaker.panic_eligible(now_tick) if panic else breaker.allows(now_tick)


def _ordered(routes: Sequence[RouteCandidate], rng: random.Random) -> list[RouteCandidate]:
    remaining: list[RouteCandidate] = list(routes)
    ordered: list[RouteCandidate] = []
    while remaining:
        best = min(item.priority for item in remaining)
        pool = [item for item in remaining if item.priority == best]
        chosen = rng.choices(pool, weights=[item.weight for item in pool], k=1)[0] if len(pool) > 1 else pool[0]
        ordered.append(chosen)
        remaining.remove(chosen)
    return ordered


def select_next_route(
    routes: Sequence[RouteCandidate],
    breakers: Mapping[str, RouteCircuitBreaker],
    tried: Sequence[str] = (),
    now_tick: int = 0,
    rng: random.Random | None = None,
) -> RouteCandidate | None:
    pool = [route for route in routes if _usable(route, breakers.get(route.route_id), frozenset(tried), now_tick)]
    return _ordered(pool, rng or random.Random())[0] if pool else None


def _oldest(breaker: RouteCircuitBreaker | None) -> int:
    return (breaker.oldest_opened_at() if breaker is not None else None) or 0


def panic_open_candidates(
    routes: Sequence[RouteCandidate],
    breakers: Mapping[str, RouteCircuitBreaker],
    tried: Sequence[str] = (),
    now_tick: int = 0,
) -> list[RouteCandidate]:
    safe = [
        route
        for route in routes
        if _usable(route, breakers.get(route.route_id), frozenset(tried), now_tick, panic=True)
    ]
    return sorted(safe, key=lambda route: (_oldest(breakers.get(route.route_id)), route.priority, route.route_id))


def _breakers(facts: Mapping[str, object], routes: Sequence[RouteCandidate]) -> dict[str, RouteCircuitBreaker]:
    raw = facts.get("breakers")
    given = (
        {
            str(key): item
            for key, item in cast("Mapping[object, object]", raw).items()
            if isinstance(item, RouteCircuitBreaker)
        }
        if isinstance(raw, Mapping)
        else {}
    )
    return {route.route_id: given.get(route.route_id) or RouteCircuitBreaker(route.route_id) for route in routes}


def evaluate_replay(facts: Mapping[str, object]) -> dict[str, object]:
    headers = _headers(facts)
    owner, no_retry = _resolve_owner(facts, headers), _no_retry(facts, headers)
    endpoint = _normalized_endpoint(facts)
    reasons = [
        name
        for name, blocked in (
            ("already_succeeded", _is_success(facts)),
            ("status_unknown", not _has_failure_flag(facts) and not _status_known(facts)),
            ("identity_invalid", not (_as_str(facts.get("canonical_model_id")) and _as_str(facts.get("effort")))),
            ("downstream_committed", facts.get("downstream_committed") is not False),
            ("tool_side_effects_unknown", facts.get("tool_side_effects") is not False),
            ("endpoint_not_stateless", not _stateless(endpoint)),
            ("no_retry_requested", no_retry),
            ("retry_owner_not_georouter", owner != "GEOROUTER_OWNER"),
        )
        if blocked
    ]
    return {
        "evaluator_invoked": True,
        "replay_allowed": not reasons,
        "reasons": tuple(reasons),
        "retry_owner": owner,
        "no_retry": no_retry,
        "endpoint": endpoint,
    }


def _invoke(
    attempt: Callable[..., Mapping[str, object]], route: RouteCandidate, canonical: str, effort: str
) -> Mapping[str, object]:
    if len(inspect.signature(attempt).parameters) >= 3:
        return attempt(canonical, effort, route.route_id)
    return attempt(canonical, effort)


def run_failover(facts: Mapping[str, object], attempt: Callable[..., Mapping[str, object]]) -> dict[str, object]:
    headers = _headers(facts)
    owner, no_retry = _resolve_owner(facts, headers), _no_retry(facts, headers)
    cap = _effective_cap(facts, owner, no_retry)
    routes = _candidates(facts)
    breakers = _breakers(facts, routes)
    budget = facts.get("budget")
    now_tick, canonical, effort = (
        _as_int(facts.get("now_tick")),
        _as_str(facts.get("canonical_model_id")),
        _as_str(facts.get("effort")),
    )
    eligible = len([route for route in routes if _usable(route, breakers.get(route.route_id), frozenset(), now_tick)])
    reservation = (
        budget.reserve_upfront(eligible, cap) if isinstance(budget, TokenBucketRetryBudget) and cap > 1 else None
    )
    tried: set[str] = set()
    decision: Mapping[str, object] = {"replay_allowed": False}
    physical = 0
    try:
        while physical < cap:
            route = select_next_route(routes, breakers, tuple(tried), now_tick)
            if route is None:
                break
            tried.add(route.route_id)
            breaker = breakers.get(route.route_id)
            if breaker is not None and not breaker.try_acquire(now_tick):
                continue
            if physical > 0 and (reservation is None or not reservation.consume_one()):
                if breaker is not None:
                    breaker.release_probe(False, now_tick)
                break
            try:
                outcome = _invoke(attempt, route, canonical, effort)
            except BaseException:
                if breaker is not None:
                    breaker.release_probe(False, now_tick)
                raise
            physical += 1
            success = _is_success(outcome)
            if breaker is not None:
                breaker.release_probe(success, now_tick)
                if not success:
                    breaker.record_failure(now_tick)
            decision = evaluate_replay({**facts, **outcome, "retry_owner": owner})
            if not decision["replay_allowed"]:
                break
    finally:
        if reservation is not None:
            _ = reservation.release_unused()
    return {
        "physical_attempts": physical,
        "replay_allowed": bool(decision["replay_allowed"]),
        "max_total_attempts": cap,
        "canonical_model_id": canonical,
        "effort": effort,
        "tried_routes": tuple(sorted(tried)),
        "reasons": decision.get("reasons", ()),
    }
