"""Wave-3: retry budget, universal failover и per-route circuit breaker."""

from __future__ import annotations

import random
import threading
from collections.abc import Callable, Mapping
from typing import cast

import pytest

from model_georouter.retry_budget import consume_live_retry
from model_georouter.transport_failover import (
    MAX_RETRY_RATIO,
    STATELESS_GENERATION_ENDPOINTS,
    BreakerState,
    RouteCandidate,
    RouteCircuitBreaker,
    TokenBucketRetryBudget,
    evaluate_replay,
    panic_open_candidates,
    run_failover,
    select_next_route,
)

CANONICAL = "claude-sonnet-4-5"
ENDPOINT = "/v1/chat/completions"
STATELESS_STREAM = "/cloudcode/v1internal:streamGenerateContent"


def _facts(**changes: object) -> dict[str, object]:
    facts: dict[str, object] = {"status_code": 503, "endpoint": ENDPOINT, "tool_side_effects": False, "downstream_committed": False, "retry_owner": "GEOROUTER_OWNER", "control_headers": {}, "max_total_attempts": 3, "canonical_model_id": CANONICAL, "effort": "high"}  # fmt: skip
    facts.update(changes)
    return facts


def _routes(count: int = 3) -> list[RouteCandidate]:
    return [RouteCandidate(route_id=f"r{index}", origin=f"origin-{index}", node=f"node-{index}", credential_scope=f"cred-{index}", priority=index, weight=1.0, canonical_model_id=CANONICAL, effort="high") for index in range(count)]  # fmt: skip


def _budget(tokens: int) -> TokenBucketRetryBudget:
    budget = TokenBucketRetryBudget()
    while budget.available_tokens < tokens: _ = budget.mint_on_admitted_primary(5)
    return budget


def _failing(seen: list[tuple[str, str]], outcome: Mapping[str, object]) -> Callable[[str, str], Mapping[str, object]]:
    def attempt(model: str, effort: str) -> Mapping[str, object]: seen.append((model, effort)); return dict(outcome)

    return attempt


def test_bridge_owned_no_retry_request_gets_exactly_one_attempt() -> None:
    seen: list[tuple[str, str]] = []
    facts = _facts(
        retry_owner="BRIDGE_OWNER",
        control_headers={"X-Georouter-No-Retry": "1"},
        routes=_routes(4),
        budget=_budget(3),
    )
    result = run_failover(facts, _failing(seen, {"status_code": 503}))
    assert result["physical_attempts"] == result["max_total_attempts"] == 1
    assert result["tried_routes"] == ("r0",) and seen == [(CANONICAL, "high")]


@pytest.mark.parametrize(
    "changes",
    [
        {"control_headers": {"x-georouter-no-retry": "1"}},
        {"control_headers": {"X-Georouter-No-Retry": "true"}},
        {"retry_owner": "NONE"},
        {"retry_owner": "BRIDGE_OWNER"},
    ],
)
def test_retry_suppression_forces_single_attempt(changes: dict[str, object]) -> None:
    seen: list[tuple[str, str]] = []
    facts = _facts(routes=_routes(3), budget=_budget(2), **changes)
    result = run_failover(facts, _failing(seen, {"status_code": 503}))
    assert result["physical_attempts"] == 1 and len(seen) == 1


def test_budget_starts_empty_and_caps_ratio_at_twenty_percent() -> None:
    budget = TokenBucketRetryBudget()
    assert budget.tokens == 0.0 and budget.available_tokens == 0
    with pytest.raises(ValueError):
        _ = TokenBucketRetryBudget(retry_ratio=0.21)
    _ = budget.mint_on_admitted_primary(1)
    assert budget.tokens == pytest.approx(MAX_RETRY_RATIO) and budget.available_tokens == 0


def test_reload_and_restart_mint_no_retry_tokens() -> None:
    budget = _budget(1)
    before = budget.tokens
    budget.reload_config(retry_ratio=0.1, capacity=5.0)
    assert budget.tokens == before
    with pytest.raises(ValueError):
        budget.reload_config(retry_ratio=0.5)
    assert budget.tokens == before


def test_reservation_is_upfront_and_releases_unused_once() -> None:
    budget = _budget(2)
    reservation = budget.reserve_upfront(4, 3)
    assert reservation.reserved == 2 and budget.available_tokens == 0
    assert reservation.consume_one() is True
    assert reservation.release_unused() == 1 and reservation.release_unused() == 0
    assert budget.available_tokens == 1
    assert budget.consumed_total == 1


def test_reservation_never_reserves_more_than_attempts_allow() -> None:
    budget = _budget(5)
    reservation = budget.reserve_upfront(9, 3)
    assert reservation.reserved == 2
    assert reservation.consume_one() is True and reservation.consume_one() is True
    assert reservation.consume_one() is False


def test_empty_budget_yields_no_retry_traffic() -> None:
    seen: list[tuple[str, str]] = []
    facts = _facts(routes=_routes(3), budget=TokenBucketRetryBudget())
    result = run_failover(facts, _failing(seen, {"status_code": 503}))
    assert result["physical_attempts"] == 1 and len(seen) == 1

def _failing3(seen: list[tuple[str, str, str]], outcome: Mapping[str, object]) -> Callable[[str, str, str], Mapping[str, object]]:
    def attempt(model: str, effort: str, route_id: str) -> Mapping[str, object]: seen.append((model, effort, route_id)); return dict(outcome)

    return attempt

@pytest.mark.parametrize(
    "outcome",
    [
        {"status_code": 500},
        {"status_code": 502},
        {"status_code": 504},
        {"status_code": 429},
        {"status_code": 200, "timed_out": True},
        {"status_code": 200, "truncated": True},
        {"status_code": 0, "connect_error": True},
        {"status_code": 0, "protocol_error": True},
    ],
)
def test_any_unsuccessful_outcome_fails_over_without_status_exceptions(outcome: dict[str, object]) -> None:
    seen: list[tuple[str, str, str]] = []
    result = run_failover(_facts(routes=_routes(3), budget=_budget(2)), _failing3(seen, outcome))
    assert result["physical_attempts"] == 3 and result["tried_routes"] == ("r0", "r1", "r2")
    assert [item[2] for item in seen] == ["r0", "r1", "r2"]

def test_failover_keeps_canonical_model_and_effort_without_downgrade() -> None:
    seen: list[tuple[str, str, str]] = []
    routes = [
        RouteCandidate(route_id=f"alias-{index}", priority=index, canonical_model_id=CANONICAL, effort="high")
        for index in range(3)
    ]
    result = run_failover(_facts(routes=routes, budget=_budget(2)), _failing3(seen, {"status_code": 503}))
    assert result["physical_attempts"] == len(seen) == 3
    assert {(model, effort) for model, effort, _ in seen} == {(CANONICAL, "high")}
    assert result["canonical_model_id"] == CANONICAL and result["effort"] == "high"

@pytest.mark.parametrize(
    "changes",
    [
        {"downstream_committed": True},
        {"tool_side_effects": True},
        {"tool_side_effects": None},
        {"endpoint": "/v1/messages"},
        {"endpoint": f"{STATELESS_STREAM}/extra"},
        {"control_headers": {"X-Georouter-No-Retry": "1"}},
        {"retry_owner": "NONE"},
        {"max_total_attempts": 1},
    ],
)
def test_replay_fences_stop_after_first_attempt(changes: dict[str, object]) -> None:
    seen: list[tuple[str, str]] = []
    facts = _facts(routes=_routes(3), budget=_budget(3), **changes)
    result = run_failover(facts, _failing(seen, {"status_code": 503}))
    assert result["physical_attempts"] == 1 and len(seen) == 1

@pytest.mark.parametrize("endpoint", STATELESS_GENERATION_ENDPOINTS)
def test_replay_allowed_only_for_exact_stateless_endpoints(endpoint: str) -> None:
    decision = evaluate_replay(_facts(endpoint=endpoint))
    assert decision["evaluator_invoked"] is True and decision["replay_allowed"] is True and decision["reasons"] == ()

@pytest.mark.parametrize(
    "endpoint",
    ["/v1/messages", "/v1/chat/completions/", f"{STATELESS_STREAM}/"],
)
def test_replay_rejects_inexact_endpoints(endpoint: str) -> None:
    decision = evaluate_replay(_facts(endpoint=endpoint))
    reasons = decision["reasons"]
    assert decision["replay_allowed"] is False and isinstance(reasons, tuple) and "endpoint_not_stateless" in reasons

def test_replay_fence_reports_every_reason() -> None:
    facts = _facts(
        downstream_committed=True,
        tool_side_effects=None,
        endpoint="/v1/messages",
        control_headers={"X-Georouter-No-Retry": "1"},
        retry_owner="BRIDGE_OWNER",
    )
    decision = evaluate_replay(facts)
    reasons = decision["reasons"]
    assert decision["evaluator_invoked"] is True and decision["replay_allowed"] is False
    assert isinstance(reasons, tuple) and set(cast("tuple[str, ...]", reasons)) == {"downstream_committed", "tool_side_effects_unknown", "endpoint_not_stateless", "no_retry_requested", "retry_owner_not_georouter"}

def test_query_string_does_not_change_stateless_classification() -> None:
    """F-113: query-string нормализуется внутри ядра `evaluate_replay`."""
    decision = evaluate_replay(_facts(endpoint=f"{ENDPOINT}?beta=1"))
    assert decision["replay_allowed"] is True and decision["endpoint"] == ENDPOINT

def test_failover_owner_header_cannot_bypass_no_retry() -> None:
    """F-004: `X-Failover-Owner` не отменяет `X-Georouter-No-Retry` и не растит лимит."""
    seen: list[tuple[str, str]] = []
    facts = _facts(routes=_routes(3), budget=_budget(3), retry_owner="BRIDGE_OWNER", control_headers={"X-Georouter-No-Retry": "1", "X-Failover-Owner": "model-georouter"})
    result = run_failover(facts, _failing(seen, {"status_code": 503}))
    assert result["physical_attempts"] == result["max_total_attempts"] == 1 and len(seen) == 1

def test_breaker_half_open_admits_one_probe() -> None:
    breaker = RouteCircuitBreaker("r0", failure_threshold=2, cooldown_ticks=5)
    for _ in range(2):
        breaker.record_failure(now_tick=0, scope="origin")
    assert breaker.state_at(0) is BreakerState.OPEN and breaker.try_acquire(0) is False
    assert breaker.state_at(5) is BreakerState.HALF_OPEN
    assert breaker.try_acquire(5) is True and breaker.try_acquire(5) is False and breaker.half_open_inflight == 1
    breaker.release_probe(success=True, now_tick=6)
    assert breaker.state_at(6) is BreakerState.CLOSED

def test_breaker_scopes_are_independent_and_ignore_client_validation() -> None:
    breaker = RouteCircuitBreaker("r0", failure_threshold=2, cooldown_ticks=4)
    breaker.record_failure(now_tick=1, scope="node", client_validation=True)
    assert breaker.scope("node") == (0, None)
    breaker.record_failure(now_tick=2, scope="node")
    assert breaker.scope("node") == (1, None) and breaker.scope("origin") == (0, None)
    breaker.record_failure(now_tick=3, scope="node")
    assert breaker.scope("node") == (2, 3) and breaker.scope("credential_scope") == (0, None)
    breaker.record_failure(now_tick=4, scope="credential_scope")
    breaker.record_success("node")
    assert breaker.scope("node") == (0, None) and breaker.scope("credential_scope") == (1, 4)

def test_panic_open_selects_safe_untried_routes_by_oldest_opened() -> None:
    routes = [RouteCandidate(route_id=name) for name in ("newer", "older", "cred", "closed", "tried")]
    routes.append(RouteCandidate(route_id="bad", credential_valid=False))
    newer, older = RouteCircuitBreaker("newer", failure_threshold=1), RouteCircuitBreaker("older", failure_threshold=1)
    newer.record_failure(now_tick=5, scope="origin")
    older.record_failure(now_tick=1, scope="origin")
    cred = RouteCircuitBreaker("cred", failure_threshold=1)
    cred.record_failure(now_tick=2, scope="credential_scope")
    breakers = {"newer": newer, "older": older, "cred": cred, "closed": RouteCircuitBreaker("closed")}
    assert older.panic_eligible(0) is True and cred.panic_eligible(0) is False
    picked = panic_open_candidates(routes, breakers, tried=("tried",), now_tick=0)
    assert [route.route_id for route in picked] == ["older", "newer"]

def test_select_next_route_priority_weight_and_open_skip() -> None:
    routes = [RouteCandidate(route_id="a", weight=5.0), RouteCandidate(route_id="b"), RouteCandidate(route_id="c", priority=1, weight=9.0)]
    picks = {route.route_id for seed in range(120) if (route := select_next_route(routes, {}, now_tick=0, rng=random.Random(seed))) is not None}
    assert picks == {"a", "b"}
    assert select_next_route(routes, {}, tried=("a",), now_tick=0) == routes[1]
    assert select_next_route(routes, {}, tried=("a", "b"), now_tick=0) == routes[2]
    breaker = RouteCircuitBreaker("a", failure_threshold=1, cooldown_ticks=5)
    breaker.record_failure(now_tick=0, scope="origin")
    assert select_next_route(routes, {"a": breaker}, now_tick=0) == routes[1]
    reopened = select_next_route(routes, {"a": breaker}, now_tick=5)
    assert reopened is not None and reopened.priority == 0

def test_half_open_admits_exactly_one_probe_under_threads() -> None:
    """F-009: гонка не пускает в HALF_OPEN больше одного пробного запроса."""
    breaker = RouteCircuitBreaker("r0", failure_threshold=1, cooldown_ticks=1)
    breaker.record_failure(now_tick=0)
    barrier = threading.Barrier(8)
    results: list[bool] = []

    def probe() -> None:
        _ = barrier.wait()
        results.append(breaker.try_acquire(5))

    threads = [threading.Thread(target=probe) for _ in range(8)]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]
    assert results.count(True) == 1 and breaker.half_open_inflight == 1

def test_live_retry_is_fail_closed_when_disabled() -> None:
    """F-003: kill-switch `enabled=False` не выдаёт токен даже при полном бюджете."""
    assert consume_live_retry({}, _budget(2), enabled=False) is False

@pytest.mark.parametrize(("changes", "reason"), [({"effort": ""}, "identity_invalid"), ({"status_code": True}, "status_unknown")])
def test_replay_identity_and_status_fences(changes: dict[str, object], reason: str) -> None:
    """F-007: пустая identity или bool-статус блокируют replay с точной причиной."""
    assert reason in cast("tuple[str, ...]", evaluate_replay(_facts(**changes))["reasons"])

def test_attempt_exception_restores_reserved_tokens() -> None:
    """F-008: исключение внутри attempt возвращает зарезервированные токены."""
    budget = _budget(2)
    def boom(model: str, effort: str) -> Mapping[str, object]: raise RuntimeError(f"boom {model} {effort}")
    with pytest.raises(RuntimeError):
        _ = run_failover(_facts(routes=_routes(3), budget=budget), boom)
    assert budget.available_tokens == 2
