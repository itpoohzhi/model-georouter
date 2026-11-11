"""Контрактные тесты общего дедлайна egress и retry-бюджета F-08."""

from __future__ import annotations

import socket
import time
from types import SimpleNamespace

import pytest

from model_georouter.adapters import AdapterRoute
from model_georouter.bridge_server import (
    Metrics,
    _Handler,
)
from model_georouter.http_wire import ResponseHead, parse_request_head
from model_georouter.model_router import BodyInspector
from model_georouter.proxy_pool import ProxyEndpoint, ProxyPoolManager
from model_georouter.retry_budget import EgressDeadline, RetryBudgetSpec
from tests.helpers import (
    FakeConnectProxy,
    FakeServer,
    HttpUpstream,
    http_request,
    make_config,
    recv_head,
    respond,
)

PATH = "/v1/chat/completions"
BODY = b'{"model":"claude-3-opus","messages":[{"role":"user","content":"ping"}],"stream":true}'
REGION_BODY = b'{"error":{"type":"RegionError","message":"not available in your region"}}'


class _RecordingProxyPoolManager(ProxyPoolManager):
    def __init__(self) -> None:
        super().__init__()
        self.penalties: list[ProxyEndpoint] = []

    def penalize(self, endpoint: ProxyEndpoint, seconds: float) -> None:
        self.penalties.append(endpoint)
        super().penalize(endpoint, seconds)


class _ExplodingEmptyInspector:
    size = 0

    @property
    def body(self) -> bytes:
        raise AssertionError("empty request body must not be inspected")

    def replace_body(self, _data: bytes) -> None:
        raise AssertionError("empty request body must not be replaced")


def _sanitize(method: str, path: str, body: bytes, model: str | None) -> tuple[bytes, list[tuple[str, str]]]:
    handler = object.__new__(_Handler)
    handler.model = model
    req = parse_request_head(f"{method} / HTTP/1.1\r\nHost: h\r\nAnthropic-Beta: old, , value\r\n\r\n".encode())
    inspector = BodyInspector(65536)
    inspector.feed(body)
    handler._sanitize_request_body(req, AdapterRoute("test", "up", path), inspector)
    return inspector.body, req.headers


def _diagnostic_metrics(body: bytes) -> dict[str, int]:
    client, peer = socket.socketpair()
    upstream, upstream_peer = socket.socketpair()
    try:
        handler = object.__new__(_Handler)
        object.__setattr__(handler, "server", SimpleNamespace(metrics=Metrics()))
        handler.status, handler.committed, handler.outcome, handler.bytes_sent = 0, False, "-", 0
        snapshot = SimpleNamespace(server=SimpleNamespace(body_timeout=0.2, inactivity_timeout=0.2))
        registry = SimpleNamespace(get=lambda _name: None)
        classifier = SimpleNamespace(classify=lambda *_args: SimpleNamespace(is_region_error=False))
        runtime = SimpleNamespace(registry=registry, classifier=classifier)
        req = parse_request_head(b"GET / HTTP/1.1\r\nHost: h\r\n\r\n")
        route = AdapterRoute("none", "up", "/")
        object.__setattr__(handler, "_ctx", (client, snapshot, runtime, req, route, "", None, None))
        object.__setattr__(handler, "_relay", lambda *_args: None)
        head = ResponseHead("HTTP/1.1", 403, "Forbidden", [("Content-Length", str(len(body)))])
        handler._finish_upstream_response(upstream, head, body, True)
        return handler.server.metrics.snapshot()
    finally:
        client.close()
        peer.close()
        upstream.close()
        upstream_peer.close()


def _transformed_head_response(method: str, body: bytes) -> bytes:
    client, peer = socket.socketpair()
    try:
        handler = object.__new__(_Handler)
        object.__setattr__(handler, "server", SimpleNamespace(metrics=Metrics()))
        handler.status, handler.committed, handler.outcome, handler.bytes_sent = 0, False, "-", 0
        snapshot = SimpleNamespace(server=SimpleNamespace(body_timeout=0.2))
        req = parse_request_head(f"{method} / HTTP/1.1\r\nHost: h\r\n\r\n".encode())
        object.__setattr__(handler, "_ctx", (client, snapshot, None, req, None, "", None, None))
        head = ResponseHead(
            "HTTP/1.1",
            200,
            "OK",
            [("Transfer-Encoding", "chunked"), ("Content-Length", "1"), ("X-Test", "yes")],
        )
        handler._send_transformed_response(head, body)
        peer.settimeout(0.2)
        return peer.recv(65536)
    finally:
        client.close()
        peer.close()


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

    def hang_on_connect(_conn: socket.socket) -> None:
        de_attempts.append(1)
        _ = de_hang.stop.wait(5)

    de_hang = track(FakeServer(hang_on_connect))
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
