"""W0 contract tests for georouter replay policy and retry suppression."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Protocol, cast

import pytest

from model_georouter.bridge_server import _breaker
from model_georouter.transport_failover import BreakerState
from tests.helpers import FakeConnectProxy, HttpUpstream, http_request, make_config, respond
from tests.test_retry_budget_and_routes import BODY, PATH, _dual_pool_config

_EXPECT_RED = os.getenv("FAILOVER_EXPECT_RED") == "1"
_MODULE = "model_georouter.transport_failover"
_STATELESS_ENDPOINTS = (
    "/v1/chat/completions",
    "/cloudcode/v1internal:streamGenerateContent",
)


class _EvaluateReplay(Protocol):
    def __call__(self, facts: Mapping[str, object], /) -> Mapping[str, object]: ...


class _RunFailover(Protocol):
    def __call__(
        self,
        facts: Mapping[str, object],
        attempt: Callable[[str, str], Mapping[str, object]],
        /,
    ) -> Mapping[str, object]: ...


def _load_api() -> tuple[_EvaluateReplay, _RunFailover] | None:
    try:
        module = import_module(_MODULE)
    except ModuleNotFoundError:
        if _EXPECT_RED:
            return None
        raise AssertionError(f"W0 contract module missing: {_MODULE}")
    namespace = vars(module)
    required = ("evaluate_replay", "run_failover")
    missing = [name for name in required if name not in namespace]
    if missing:
        if _EXPECT_RED:
            return None
        raise AssertionError(
            f"W0 contract API missing from {_MODULE}: {', '.join(missing)}"
        )
    return (
        cast(_EvaluateReplay, namespace["evaluate_replay"]),
        cast(_RunFailover, namespace["run_failover"]),
    )


def _facts(**changes: object) -> dict[str, object]:
    facts: dict[str, object] = {
        "status_code": 503,
        "timed_out": False,
        "truncated": False,
        "wire_send_started": True,
        "endpoint": "/v1/chat/completions",
        "tool_side_effects": False,
        "downstream_committed": False,
        "retry_owner": "GEOROUTER_OWNER",
        "control_headers": {},
        "max_total_attempts": 3,
        "canonical_model_id": "claude-sonnet-4-5",
        "effort": "high",
    }
    facts.update(changes)
    return facts


def _failure(_model: str, _effort: str) -> Mapping[str, object]:
    return {"status_code": 503, "timed_out": False, "truncated": False}


@pytest.mark.parametrize("status_code", [100, 201, 301, 400, 418, 500, 599])
def test_every_non_200_status_reaches_replay_evaluator(status_code: int) -> None:
    """VER-012: every non-200 outcome reaches policy evaluation."""
    api = _load_api()
    if api is None:
        return
    decision = api[0](_facts(status_code=status_code))
    assert decision["evaluator_invoked"] is True


@pytest.mark.parametrize(
    "changes",
    [
        {"status_code": None, "timed_out": True},
        {"status_code": None, "truncated": True},
    ],
)
def test_timeout_and_precommit_truncation_reach_evaluator(
    changes: dict[str, object],
) -> None:
    """VER-012: transport timeout and pre-commit truncation are policy inputs."""
    api = _load_api()
    if api is None:
        return
    decision = api[0](_facts(**changes))
    assert decision["evaluator_invoked"] is True


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        (*_STATELESS_ENDPOINTS[:1], True),
        (*_STATELESS_ENDPOINTS[1:], True),
        ("/v1/responses", False),
        ("/v1/chat/completions/", False),
        ("/v1/chat/completions?stream=true", True),
        ("/tools/execute", False),
    ],
)
def test_wire_send_replay_has_exact_stateless_endpoint_allowlist(
    endpoint: str,
    expected: bool,
) -> None:
    """VER-013: wire-send replay is limited to exact stateless paths."""
    api = _load_api()
    if api is None:
        return
    decision = api[0](_facts(endpoint=endpoint))
    assert decision["replay_allowed"] is expected


@pytest.mark.parametrize(
    "changes",
    [
        {"tool_side_effects": True},
        {"tool_side_effects": None},
        {"downstream_committed": True},
        {"downstream_committed": 1},
    ],
)
def test_replay_fences_leave_exactly_one_attempt(changes: dict[str, object]) -> None:
    """VER-013: side effects, unknown ledger, and downstream commit are final."""
    api = _load_api()
    if api is None:
        return
    result = api[1](_facts(**changes), _failure)
    assert result["physical_attempts"] == 1
    assert result["replay_allowed"] is False


def test_bridge_no_retry_control_header_forces_one_attempt() -> None:
    """VER-014: bridge-owned requests cannot recurse inside georouter."""
    api = _load_api()
    if api is None:
        return
    facts = _facts(
        retry_owner="BRIDGE_OWNER",
        control_headers={"X-Georouter-No-Retry": "1"},
    )
    result = api[1](facts, _failure)
    assert result["physical_attempts"] == 1


def test_total_attempt_cap_preserves_model_and_effort() -> None:
    """VER-014: no route chain exceeds three or mutates canonical identity."""
    api = _load_api()
    if api is None:
        return
    seen: list[tuple[str, str]] = []

    def record(model: str, effort: str) -> Mapping[str, object]:
        seen.append((model, effort))
        return _failure(model, effort)

    result = api[1](_facts(max_total_attempts=99), record)
    assert cast(int, result["physical_attempts"]) <= 3
    assert cast(int, result["max_total_attempts"]) <= 3
    assert seen and set(seen) == {("claude-sonnet-4-5", "high")}
    assert result["canonical_model_id"] == "claude-sonnet-4-5"
    assert result["effort"] == "high"


def test_201_success_is_final_and_never_replays() -> None:
    """F-002: 2xx-статус = успех, replay не нужен."""
    api = _load_api()
    if api is None:
        return
    decision = api[0](_facts(status_code=201))
    assert decision["replay_allowed"] is False
    assert "already_succeeded" in cast(tuple[str, ...], decision["reasons"])


def test_unknown_status_without_failure_flags_is_final() -> None:
    """F-002: status_code=None без failure-флагов не разрешает replay."""
    api = _load_api()
    if api is None:
        return
    decision = api[0](_facts(status_code=None))
    assert decision["replay_allowed"] is False
    assert "status_unknown" in cast(tuple[str, ...], decision["reasons"])


def test_run_failover_skips_foreign_identity_candidates() -> None:
    """F-006: маршрут с чужой identity/effort не вызывается с подменой модели."""
    api = _load_api()
    if api is None:
        return
    namespace = vars(import_module(_MODULE))
    candidate = cast(Callable[..., object], namespace["RouteCandidate"])
    calls: list[tuple[str, str]] = []

    def record(model: str, effort: str) -> Mapping[str, object]:
        calls.append((model, effort))
        return _failure(model, effort)

    facts = _facts(routes=[candidate("r0", canonical_model_id="claude-opus-4-1", effort="high")])
    result = api[1](facts, record)
    assert result["physical_attempts"] == 0
    assert calls == []
    assert result["canonical_model_id"] == "claude-sonnet-4-5"


def test_panic_open_candidates_tolerates_missing_breakers() -> None:
    """F-008: отсутствующий breaker не даёт KeyError в sort-key."""
    api = _load_api()
    if api is None:
        return
    namespace = vars(import_module(_MODULE))
    candidate = cast(Callable[..., object], namespace["RouteCandidate"])
    panic = cast(Callable[..., list[object]], namespace["panic_open_candidates"])
    assert len(panic([candidate("r0")], {}, now_tick=0)) == 1


def _breaker_state(bridge, pool: str = "direct") -> BreakerState:
    route = bridge.server.runtime_for(bridge.server.config_manager.get()).registry.resolve(PATH)
    return _breaker(route.upstream, pool).state_at(int(time.monotonic()))


def test_single_pool_failover_requires_live_replay_token(track, start_bridge) -> None:
    """F-001: одиночный прокси не уходит в geo-fallback без replay-токена."""
    dead = track(FakeConnectProxy(status=502))
    up = track(HttpUpstream(lambda conn, _req: respond(conn, 200, b"fallback")))
    proxy = track(FakeConnectProxy(target=("127.0.0.1", up.port)))
    bridge = start_bridge(_dual_pool_config(up.port, dead.port, proxy.port))

    assert http_request(bridge.port, "POST", PATH, BODY).status == 502
    bridge.prime_replay_tokens()
    response = http_request(bridge.port, "POST", PATH, BODY)
    assert response.status == 200 and response.body == b"fallback"


def test_breaker_keys_are_pool_scoped() -> None:
    """F-003: пулы не делят breaker, пустой pool сохраняет legacy-ключ."""
    primary, secondary, legacy = _breaker("up", "p1"), _breaker("up", "p2"), _breaker("up")
    assert primary is not secondary and primary is not legacy and secondary is not legacy
    assert _breaker("up", "p1") is primary


@pytest.mark.parametrize(("status", "opened"), [(400, False), (500, True)])
def test_client_errors_do_not_open_breaker(track, start_bridge, status, opened) -> None:
    """F-005: 4xx (кроме 403/429) — вина клиента, breaker копит только серверные отказы."""
    up = track(HttpUpstream(lambda conn, _req: respond(conn, status, b"x")))
    bridge = start_bridge(make_config(up.port))
    for _ in range(3):
        _ = http_request(bridge.port, "POST", PATH, BODY)
    assert (_breaker_state(bridge) is BreakerState.OPEN) is opened


def test_failure_recorded_without_replay_token(track, start_bridge) -> None:
    """F-004: отказ учитывается в breaker даже когда replay запрещён."""
    up = track(HttpUpstream(lambda conn, _req: respond(conn, 503, b"x")))
    bridge = start_bridge(make_config(up.port))
    for _ in range(3):
        _ = http_request(bridge.port, "POST", PATH, BODY)
    assert _breaker_state(bridge) is BreakerState.OPEN


def test_failed_half_open_probe_is_released(track, start_bridge) -> None:
    """F-002: проба half-open освобождается, следующий кулдаун снова берёт токен."""
    up = track(HttpUpstream(lambda conn, _req: respond(conn, 500, b"x")))
    bridge = start_bridge(make_config(up.port))
    route = bridge.server.runtime_for(bridge.server.config_manager.get()).registry.resolve(PATH)
    breaker, past = _breaker(route.upstream, "direct"), int(time.monotonic()) - 20
    for _ in range(3):
        breaker.record_failure(now_tick=past)
    assert breaker.state_at(int(time.monotonic())) is BreakerState.HALF_OPEN

    assert http_request(bridge.port, "POST", PATH, BODY).status == 500
    assert breaker.try_acquire(int(time.monotonic()) + 100) is True

