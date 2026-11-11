from __future__ import annotations

import dataclasses
import socket
import threading
import time

import pytest

from model_georouter.bridge_server import _Handler
from model_georouter.config import ConfigError, PoolConfig, parse_config
from model_georouter.errors import (
    AllProxiesPenalizedError,
    EgressConnectError,
    EgressTimeoutError,
    UpstreamProtocolError,
    UpstreamTimeoutError,
)
from model_georouter.http_wire import ChunkedFraming, EofFraming, LengthFraming
from model_georouter.proxy_pool import (
    _DNS_SEMAPHORE,
    CONNECT_RESPONSE_LIMIT,
    EgressSettings,
    ProxyPoolManager,
    _read_connect_head,
    _resolve,
    parse_proxy_url,
)
from model_georouter.retry_budget import AttemptEvent, AttemptState, EgressDeadline, RetryBudgetSpec, transition
from tests.helpers import FakeConnectProxy, FakeServer, HttpUpstream, http_request, make_config, respond


def test_egress_deadline_validation_math_and_clamp():
    for bad_abs in (float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite"):
            EgressDeadline(bad_abs)
    assert EgressDeadline(0.0).absolute_deadline == 0.0
    for bad in (0.0, -0.5, float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError, match="total_seconds"):
            EgressDeadline.from_timeout(bad, started_at=10.0)
    for bad_start in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError, match="started_at"):
            EgressDeadline.from_timeout(1.0, started_at=bad_start)
    dl = EgressDeadline.from_timeout(2.5, started_at=10.0)
    assert dl.absolute_deadline == 12.5
    assert (dl.remaining(now=10.0), dl.remaining(now=12.5), dl.remaining(now=15.0)) == (2.5, 0.0, 0.0)
    assert (dl.expired(now=12.4), dl.expired(now=12.5), dl.expired(now=12.6)) == (False, True, True)
    with pytest.raises(ValueError, match="now"):
        dl.remaining(now=float("nan"))
    live = EgressDeadline.from_timeout(5.0)
    assert 0.0 < live.remaining() <= 5.0 and not live.expired()
    c_dl = EgressDeadline(absolute_deadline=100.0)
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="requested_timeout"):
            c_dl.clamp(bad, now=90.0)
    assert (c_dl.clamp(2.0, now=95.0), c_dl.clamp(10.0, now=95.0), c_dl.clamp(5.0, now=95.0)) == (2.0, 5.0, 5.0)
    for exp_now in (100.0, 101.0):
        with pytest.raises(TimeoutError, match="expired"):
            c_dl.clamp(1.0, now=exp_now)


def test_retry_budget_spec_defaults_and_valid_boundaries():
    spec = RetryBudgetSpec()
    assert (spec.total_egress_deadline, spec.connect_timeout, spec.pre_send_retries) == (40.0, 3.0, 2)
    assert (spec.max_backoff_seconds, spec.single_pool_max_retries, spec.primary_first_header_timeout) == (2.0, 0, 5.0)
    assert (spec.penalize_on_header_timeout, spec.multi_proxy_attempts) == (True, 3)
    assert spec.worst_case_pre_send_seconds == 11.0
    spec.validate_f08()
    exact = RetryBudgetSpec(40.0, 3.0, 2, 31.0, 0, 5.0, True)
    assert exact.worst_case_pre_send_seconds == 40.0
    exact.validate_f08()
    zero_retries = RetryBudgetSpec(pre_send_retries=0, max_backoff_seconds=0.0, connect_timeout=0.5)
    assert (zero_retries.multi_proxy_attempts, zero_retries.worst_case_pre_send_seconds) == (1, 0.5)
    zero_retries.validate_f08()


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"total_egress_deadline": 0.0}, "total_egress_deadline"),
        ({"total_egress_deadline": float("inf")}, "total_egress_deadline"),
        ({"total_egress_deadline": 40.1}, "40 seconds"),
        ({"connect_timeout": 0.0}, "connect_timeout"),
        ({"connect_timeout": 3.1}, "3 seconds"),
        ({"pre_send_retries": -1}, "pre_send_retries"),
        ({"pre_send_retries": True}, "pre_send_retries"),
        ({"pre_send_retries": 1.5}, "pre_send_retries"),
        ({"pre_send_retries": 3}, "exceed 2"),
        ({"max_backoff_seconds": -0.1}, "max_backoff_seconds"),
        ({"max_backoff_seconds": float("nan")}, "max_backoff_seconds"),
        ({"total_egress_deadline": 10.0}, "pre-send budget"),
        ({"single_pool_max_retries": 1}, "single_pool_max_retries"),
        ({"single_pool_max_retries": -1}, "single_pool_max_retries"),
        ({"single_pool_max_retries": False}, "single_pool_max_retries"),
        ({"primary_first_header_timeout": 0.0}, "primary_first_header_timeout"),
        ({"primary_first_header_timeout": 5.1}, "5 seconds"),
        ({"penalize_on_header_timeout": False}, "penalize_on_header_timeout"),
    ],
)
def test_retry_budget_spec_rejects_invalid_configs(overrides, match):
    spec = dataclasses.replace(RetryBudgetSpec(), **overrides)
    with pytest.raises(ValueError, match=match):
        spec.validate_f08()


def test_pure_transition_immutability_and_matrix():
    initial = AttemptState(pool="route-primary", fallback_pool="route-de", single_proxy=True)
    event = AttemptEvent(kind="protocol_error", deadline_expired=False, has_endpoint=True)
    ht = AttemptEvent("header_timeout", has_endpoint=True)
    for obj, attr in ((initial, "pool"), (event, "kind")):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, attr, "mutated")
    nxt = transition(initial, event)
    assert initial.status == "pending" and initial.pool == "route-primary" and initial.single_proxy is True
    assert (nxt.status, nxt.pool, nxt.next_pool) == ("failover", "route-de", "route-de")
    assert nxt.should_penalize is True and nxt.single_proxy is False
    assert transition(initial, AttemptEvent("protocol_error", has_endpoint=False)).should_penalize is False
    assert transition(initial, AttemptEvent("connect_error", has_endpoint=True)).should_penalize is False
    assert transition(initial, AttemptEvent("penalized", has_endpoint=True)).should_penalize is False
    assert transition(initial, ht).should_penalize is True
    assert transition(dataclasses.replace(initial, penalize_on_header_timeout=False), ht).should_penalize is True
    assert transition(dataclasses.replace(initial, fallback_pool=""), event).status == "failed"
    multi = AttemptState(pool="route-multi", fallback_pool="", single_proxy=False, penalize_on_header_timeout=True)
    assert transition(multi, event).should_penalize is False and transition(multi, ht).should_penalize is True
    assert transition(dataclasses.replace(multi, penalize_on_header_timeout=False), ht).should_penalize is False
    expired = transition(initial, AttemptEvent("connect_error", deadline_expired=True, has_endpoint=True))
    assert (expired.status, expired.next_pool, expired.pool) == ("failed", "", "route-primary")
    committed, done = transition(initial, AttemptEvent("committed")), transition(initial, AttemptEvent("completed"))
    assert (committed.committed, committed.status, committed.next_pool) == (True, "committed", "")
    after_commit = transition(committed, event)
    assert (after_commit.status, after_commit.next_pool, after_commit.should_penalize) == ("failed", "", True)
    assert done.committed is True and done.status == "completed"
    with pytest.raises(ValueError, match="unsupported"):
        transition(initial, AttemptEvent("unknown"))


def test_config_single_pool_max_retries_validation():
    base = {"server": {"listen": "127.0.0.1", "port": 10830}, "pools": {}, "rules": []}
    assert parse_config(base).server.single_pool_max_retries == 0
    cfg0 = {**base, "server": {**base["server"], "single_pool_max_retries": 0}}
    assert parse_config(cfg0).server.single_pool_max_retries == 0
    for bad in (-1, 1, True):
        with pytest.raises(ConfigError, match="single_pool_max_retries"):
            parse_config({**base, "server": {**base["server"], "single_pool_max_retries": bad}})


def test_single_pool_max_retries_zero_caps_single_proxy_but_not_multi_proxy():
    p1, p2 = "http://127.0.0.1:4001", "http://127.0.0.1:4002"
    single_pool = PoolConfig("s1", "http_connect", "first_available", True, (p1,))
    multi_pool = PoolConfig("m2", "http_connect", "first_available", True, (p1, p2))
    settings = EgressSettings(1.0, 5.0, retries=2, penalty_seconds=30, single_pool_max_retries=0)
    calls: list[str] = []

    def fake_open(_pool, endpoint, _host, _port, _deadline):
        assert endpoint is not None
        calls.append(endpoint.raw)
        if endpoint.port == 4001:
            raise EgressConnectError("dead proxy", phase="tcp")
        a, b = socket.socketpair()
        b.sendall(b"ok")
        b.close()
        return a

    for s_cfg in (settings, EgressSettings(1.0, 5.0, retries=0, penalty_seconds=30, single_pool_max_retries=1)):
        calls.clear()
        mgr = ProxyPoolManager()
        mgr._open = fake_open  # type: ignore[method-assign]
        with pytest.raises(EgressConnectError, match="dead proxy"):
            mgr.connect(single_pool, "h", 1, s_cfg)
        assert calls == [p1] and mgr.is_penalized(mgr.endpoints(single_pool)[0])
    calls.clear()
    mgr2 = ProxyPoolManager()
    mgr2._open = fake_open  # type: ignore[method-assign]
    with mgr2.connect(multi_pool, "h", 1, settings) as sock:
        sock.settimeout(1.0)
        assert sock.recv(2) == b"ok"
    eps = mgr2.endpoints(multi_pool)
    assert calls == [p1, p2] and mgr2.is_penalized(eps[0]) and not mgr2.is_penalized(eps[1])


def _buffered_pair() -> tuple[socket.socket, socket.socket]:
    a, b = socket.socketpair()
    for s in (a, b):
        s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        s.settimeout(1.0)
    return a, b


def test_read_connect_head_exact_limit_and_split_crlf():
    ep, prefix = parse_proxy_url("http://127.0.0.1:3128"), b"HTTP/1.1 200 OK\r\nX: "
    exact_pad = b"a" * (CONNECT_RESPONSE_LIMIT - len(prefix) - 4)
    a, b = _buffered_pair()
    with a, b:
        b.sendall(prefix + exact_pad + b"\r\n\r\nTAIL")
        head = _read_connect_head(a, ep, time.monotonic() + 2.0)
        assert len(head) == CONNECT_RESPONSE_LIMIT and a.recv(10) == b"TAIL"
    over_a, over_b = _buffered_pair()
    with over_a, over_b:
        over_b.sendall(b"a" * (CONNECT_RESPONSE_LIMIT + 1))
        with pytest.raises(EgressConnectError, match="oversized CONNECT response") as exc_info:
            _read_connect_head(over_a, ep, time.monotonic() + 2.0)
        assert exc_info.value.phase == "connect_read"
    c, d = _buffered_pair()

    class _SplitSock:
        sent, settimeout = False, c.settimeout

        def recv(self, size: int, flags: int = 0) -> bytes:
            data = c.recv(size, flags)
            if not flags and not self.sent:
                self.sent = True
                d.sendall(b"\nAFTER")
            return data

    with c, d:
        d.sendall(b"HTTP/1.1 200 OK\r\n\r")
        head2 = _read_connect_head(_SplitSock(), ep, time.monotonic() + 2.0)  # type: ignore[arg-type]
        assert head2.endswith(b"\r\n\r\n") and c.recv(10) == b"AFTER"


def test_resolve_semaphore_limit_and_expired_deadline(monkeypatch):
    gate, started = threading.BoundedSemaphore(1), time.monotonic()
    assert gate.acquire(timeout=0.0)
    monkeypatch.setattr("model_georouter.proxy_pool._DNS_SEMAPHORE", gate)
    with pytest.raises(EgressTimeoutError, match="concurrency limit"):
        _resolve("example.com", 443, time.monotonic() - 0.01)
    assert time.monotonic() - started < 0.2
    gate.release()
    unblock, done, t0 = threading.Event(), threading.Event(), time.monotonic()
    workers: list[threading.Thread] = []

    def slow_resolve(*_a, **_kw):
        workers.append(threading.current_thread())
        unblock.wait(1.0)
        done.set()
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]

    monkeypatch.setattr("model_georouter.proxy_pool._DNS_SEMAPHORE", _DNS_SEMAPHORE)
    monkeypatch.setattr(socket, "getaddrinfo", slow_resolve)
    try:
        with pytest.raises(TimeoutError, match="DNS resolution"):
            _resolve("slow.example.com", 443, time.monotonic() - 0.01)
        assert time.monotonic() - t0 < 0.2
    finally:
        unblock.set()
        assert done.wait(1.0) and workers[0].join(1.0) is None and not workers[0].is_alive()


def test_bridge_single_proxy_failover_on_penalized_and_header_timeout(track, start_bridge):
    silent = track(FakeServer(lambda conn: (conn.recv(1024), silent.stop.wait(1.0))))
    good_up = track(HttpUpstream(lambda conn, req: respond(conn, 200, b"fallback-ok")))
    bad_proxy = track(FakeConnectProxy(target=("127.0.0.1", silent.port)))
    good_proxy = track(FakeConnectProxy(target=("127.0.0.1", good_up.port)))
    p_pool = {"type": "http_connect", "proxies": [f"http://127.0.0.1:{bad_proxy.port}"]}
    fb_pool = {"type": "http_connect", "proxies": [f"http://127.0.0.1:{good_proxy.port}"]}
    pools, rules = {"primary": p_pool, "route-de": fb_pool}, [{"match_prefix": ["m"], "pool": "primary"}]
    cfg = make_config(good_up.port, headers_timeout=0.1, pools=pools, rules=rules, geo_fallback_pool="route-de")
    bridge = start_bridge(cfg)
    resp1 = http_request(bridge.port, "POST", "/v1/chat/completions", b'{"model":"m"}')
    assert resp1.status == 200 and resp1.body == b"fallback-ok"
    ep = bridge.server.proxy_manager.endpoints(bridge.server.config_manager.get().pools["primary"])[0]
    assert bridge.server.proxy_manager.is_penalized(ep)
    resp2 = http_request(bridge.port, "POST", "/v1/chat/completions", b'{"model":"m"}')
    assert resp2.status == 200 and resp2.body == b"fallback-ok"
    assert len(bad_proxy.connects) == 1 and len(good_proxy.connects) == 2


def test_handler_helpers_and_upstream_head():
    active = EgressDeadline.from_timeout(60.0)
    ep = parse_proxy_url("http://127.0.0.1:8080")
    ev = _Handler._attempt_event
    assert ev(UpstreamTimeoutError("t"), active, ep).kind == "header_timeout"
    assert ev(UpstreamProtocolError("p"), active, ep).kind == "protocol_error"
    assert ev(AllProxiesPenalizedError("x"), active, None) == AttemptEvent("penalized", False, False)
    assert ev(EgressConnectError("c"), EgressDeadline(0.0), ep) == AttemptEvent("connect_error", True, True)
    out, eof_out = bytearray(), bytearray()
    a, b = socket.socketpair()
    with a, b:
        b.sendall(b"world")
        _Handler._read_buffered_body(a, LengthFraming(10, out.extend), b"hello", 1.0)
        assert bytes(out) == b"helloworld"
        b.close()
        _Handler._read_buffered_body(a, EofFraming(eof_out.extend), b"init", 1.0)
        assert bytes(eof_out) == b"init"
        with pytest.raises(UpstreamProtocolError, match="truncated"):
            _Handler._read_buffered_body(a, ChunkedFraming(), b"", 1.0)
        with pytest.raises(UpstreamTimeoutError, match="body timeout"):
            _Handler._read_buffered_body(a, LengthFraming(5), b"", 0.0)
    c, d = socket.socketpair()
    with c, d:
        d.sendall(b"HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhi")
        head, rest = _Handler._read_upstream_head(c, 2.0, EgressDeadline.from_timeout(5.0))
        assert (head.status, rest) == (200, b"hi")
        d.sendall(b"HTTP/1.1 101 Switching Protocols\r\n\r\nraw")
        head_101, rest_101 = _Handler._read_upstream_head(c, 2.0)
        assert (head_101.status, rest_101) == (101, b"raw")
        with pytest.raises(UpstreamTimeoutError, match="within 2s"):
            _Handler._read_upstream_head(c, 2.0, EgressDeadline(0.0))
