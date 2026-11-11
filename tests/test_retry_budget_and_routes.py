"""Контрактные тесты общего дедлайна egress и retry-бюджета F-08."""

from __future__ import annotations

import time

import pytest

from model_georouter.retry_budget import EgressDeadline, RetryBudgetSpec
from tests.helpers import FakeConnectProxy, FakeServer, HttpUpstream, http_request, make_config, recv_head, respond

PATH = "/v1/chat/completions"
BODY = b'{"model":"claude-3-opus","messages":[{"role":"user","content":"ping"}],"stream":true}'


def _dual_pool_config(upstream_port: int, de_port: int, nl_port: int, **server_overrides: object) -> dict[str, object]:
    server: dict[str, object] = {
        "connect_timeout": 0.25,
        "pre_send_retries": 2,
        "total_egress_deadline": 40.0,
        "headers_timeout": 120.0,
        **server_overrides,
    }
    return make_config(
        upstream_port,
        pools={
            "route-de": {"type": "http_connect", "proxies": [f"http://127.0.0.1:{de_port}"]},
            "route-nl": {"type": "http_connect", "proxies": [f"http://127.0.0.1:{nl_port}"]},
        },
        rules=[{"match_prefix": ["claude-"], "pool": "route-de"}],
        geo_fallback_pool="route-nl",
        server=server,
    )


def test_f08_retry_budget_contract() -> None:
    budget = RetryBudgetSpec()

    budget.validate_f08()

    assert budget.total_egress_deadline == 40.0
    assert budget.connect_timeout == 3.0
    assert budget.pre_send_retries == 2
    assert budget.multi_proxy_attempts == 3
    assert budget.max_backoff_seconds == 2.0
    assert budget.worst_case_pre_send_seconds == 11.0
    assert budget.single_pool_max_retries == 0
    assert budget.primary_first_header_timeout == 5.0
    assert budget.penalize_on_header_timeout is True


def test_f08_rejects_budget_that_can_outlive_total_deadline() -> None:
    budget = RetryBudgetSpec(
        connect_timeout=13.0,
        total_egress_deadline=40.0,
        pre_send_retries=2,
        max_backoff_seconds=2.0,
    )

    with pytest.raises(ValueError, match="pre-send budget"):
        budget.validate_f08()


@pytest.mark.parametrize(
    ("budget", "message"),
    [
        (RetryBudgetSpec(total_egress_deadline=40.1), "total_egress_deadline"),
        (RetryBudgetSpec(connect_timeout=3.1), "connect_timeout"),
        (RetryBudgetSpec(pre_send_retries=3), "pre_send_retries"),
        (RetryBudgetSpec(single_pool_max_retries=1), "single_pool_max_retries"),
        (RetryBudgetSpec(primary_first_header_timeout=5.1), "primary_first_header_timeout"),
        (RetryBudgetSpec(penalize_on_header_timeout=False), "penalize_on_header_timeout"),
    ],
)
def test_f08_rejects_values_outside_failover_contract(
    budget: RetryBudgetSpec,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        budget.validate_f08()


def test_egress_deadline_clamps_every_phase_to_one_absolute_deadline() -> None:
    deadline = EgressDeadline.from_timeout(40.0, started_at=100.0)

    assert deadline.absolute_deadline == 140.0
    assert deadline.remaining(now=112.0) == 28.0
    assert deadline.clamp(3.0, now=112.0) == 3.0
    assert deadline.clamp(30.0, now=112.0) == 28.0
    assert deadline.remaining(now=141.0) == 0.0
    assert deadline.expired(now=141.0) is True


def test_egress_deadline_refuses_to_start_expired_phase() -> None:
    deadline = EgressDeadline(absolute_deadline=140.0)

    with pytest.raises(TimeoutError, match="egress deadline expired"):
        _ = deadline.clamp(5.0, now=140.0)


def test_single_pool_connect_stall_fails_over_to_geo_fallback_under_7s(track, start_bridge) -> None:
    de_attempts: list[int] = []
    de_hang = track(FakeServer(lambda _conn: (de_attempts.append(1), de_hang.stop.wait(5))))
    nl_up = track(HttpUpstream(lambda conn, _req: respond(conn, 200, b"via-nl")))
    nl_proxy = track(FakeConnectProxy(target=("127.0.0.1", nl_up.port)))
    bridge = start_bridge(_dual_pool_config(nl_up.port, de_hang.port, nl_proxy.port))

    started = time.monotonic()
    first = http_request(bridge.port, "POST", PATH, BODY)
    elapsed = time.monotonic() - started

    assert first.status == 200 and first.body == b"via-nl"
    assert elapsed < 7.0 and len(de_attempts) == 1
    assert bridge.health()["pools"]["route-de"]["proxies_penalized"] == 1
    second = http_request(bridge.port, "POST", PATH, BODY)
    assert second.status == 200 and second.body == b"via-nl" and len(de_attempts) == 1


def test_single_pool_post_connect_header_stall_penalizes_and_fails_over_under_7s(track, start_bridge) -> None:
    de_connects: list[int] = []

    def stalled_after_connect(conn) -> None:
        recv_head(conn)
        de_connects.append(1)
        conn.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
        recv_head(conn)
        de_proxy.stop.wait(5)

    de_proxy = track(FakeServer(stalled_after_connect))
    nl_up = track(HttpUpstream(lambda conn, _req: respond(conn, 200, b"nl-after-header-stall")))
    nl_proxy = track(FakeConnectProxy(target=("127.0.0.1", nl_up.port)))
    bridge = start_bridge(_dual_pool_config(nl_up.port, de_proxy.port, nl_proxy.port, headers_timeout=120.0))
    bridge.server.retry_budget = RetryBudgetSpec(primary_first_header_timeout=0.3)

    started = time.monotonic()
    resp = http_request(bridge.port, "POST", PATH, BODY)
    elapsed = time.monotonic() - started

    assert resp.status == 200 and resp.body == b"nl-after-header-stall"
    assert elapsed < 7.0 and len(de_connects) == 1
    assert bridge.health()["pools"]["route-de"]["proxies_penalized"] == 1


def test_shared_monotonic_egress_deadline_clamps_connect_send_and_headers(track, start_bridge) -> None:
    de_hang = track(FakeServer(lambda _conn: de_hang.stop.wait(5)))
    nl_stalled_up = track(HttpUpstream(lambda _conn, _req: nl_stalled_up.stop.wait(5)))
    nl_proxy = track(FakeConnectProxy(target=("127.0.0.1", nl_stalled_up.port)))
    bridge = start_bridge(
        _dual_pool_config(
            nl_stalled_up.port,
            de_hang.port,
            nl_proxy.port,
            connect_timeout=0.2,
            total_egress_deadline=0.45,
            headers_timeout=30.0,
        )
    )

    started = time.monotonic()
    resp = http_request(bridge.port, "POST", PATH, BODY)
    elapsed = time.monotonic() - started

    assert resp.status == 504 and resp.json["error"]["type"] == "upstream_timeout"
    assert elapsed < 1.5
