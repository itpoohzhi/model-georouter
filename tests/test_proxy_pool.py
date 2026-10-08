from __future__ import annotations

import random
import time

import pytest

from tests.helpers import FakeConnectProxy, FakeServer, FakeSocks5Proxy, basic_token, echo_handler, free_port
from universal_ai_bridge.config import PoolConfig
from universal_ai_bridge.errors import AllProxiesPenalizedError, EgressConnectError, EgressTimeoutError
from universal_ai_bridge.proxy_pool import EgressSettings, ProxyPoolManager, parse_proxy_url

SETTINGS = EgressSettings(connect_timeout=2.0, total_deadline=5.0, retries=1, penalty_seconds=30)


def http_pool(*urls, strategy="first_available") -> PoolConfig:
    return PoolConfig("p", "http_connect", strategy, True, tuple(urls))


def socks_pool(*urls, remote_dns=True) -> PoolConfig:
    return PoolConfig("s", "socks5", "first_available", remote_dns, tuple(urls))


def roundtrip(sock, payload=b"ping") -> bytes:
    sock.settimeout(5)
    sock.sendall(payload)
    return sock.recv(100)


def test_parse_proxy_url():
    endpoint = parse_proxy_url("http://us%40er:p%3Ass@proxy.local:3128")
    assert (endpoint.host, endpoint.port, endpoint.username, endpoint.password) == ("proxy.local", 3128, "us@er", "p:ss")
    assert "p:ss" not in repr(endpoint) and "p:ss" not in endpoint.label
    assert parse_proxy_url("socks5h://h").port == 1080
    with pytest.raises(ValueError):
        parse_proxy_url("ftp://h:1")
    with pytest.raises(ValueError):
        parse_proxy_url("http://")


def test_direct_connect(track):
    upstream = track(FakeServer(echo_handler))
    sock = ProxyPoolManager().connect(PoolConfig("direct", "direct"), "127.0.0.1", upstream.port, SETTINGS)
    with sock:
        assert roundtrip(sock) == b"ping"


def test_http_connect_with_basic_auth(track):
    upstream = track(FakeServer(echo_handler))
    proxy = track(FakeConnectProxy(target=("127.0.0.1", upstream.port)))
    pool = http_pool(f"http://user:p%40ss@127.0.0.1:{proxy.port}")
    with ProxyPoolManager().connect(pool, "api.example.com", 443, SETTINGS) as sock:
        assert roundtrip(sock) == b"ping"  # данные идут через туннель
    connect = proxy.connects[0]
    assert connect["authority"] == "api.example.com:443"
    assert connect["headers"]["proxy-authorization"] == "Basic " + basic_token("user", "p@ss")


def test_http_connect_without_credentials_sends_no_auth(track):
    upstream = track(FakeServer(echo_handler))
    proxy = track(FakeConnectProxy(target=("127.0.0.1", upstream.port)))
    ProxyPoolManager().connect(http_pool(f"http://127.0.0.1:{proxy.port}"), "h", 1, SETTINGS).close()
    assert "proxy-authorization" not in proxy.connects[0]["headers"]


def test_http_connect_refusal_penalizes_proxy(track):
    proxy = track(FakeConnectProxy(status=407))
    manager = ProxyPoolManager()
    pool = http_pool(f"http://127.0.0.1:{proxy.port}")
    with pytest.raises(EgressConnectError, match="HTTP 407"):
        manager.connect(pool, "h", 443, SETTINGS)
    assert manager.is_penalized(manager.endpoints(pool)[0])


def test_socks5h_uses_remote_dns_domain_atyp(track):
    upstream = track(FakeServer(echo_handler))
    proxy = track(FakeSocks5Proxy(target=("127.0.0.1", upstream.port)))
    pool = socks_pool(f"socks5h://127.0.0.1:{proxy.port}")
    # имя не резолвится локально — успех доказывает, что DNS делал прокси
    with ProxyPoolManager().connect(pool, "only-the-proxy-knows.invalid", 8443, SETTINGS) as sock:
        assert roundtrip(sock) == b"ping"
    request = proxy.requests[0]
    assert request["atyp"] == 0x03
    assert (request["host"], request["port"]) == ("only-the-proxy-knows.invalid", 8443)


def test_socks5_remote_dns_flag_applies_to_plain_socks5_scheme(track):
    upstream = track(FakeServer(echo_handler))
    proxy = track(FakeSocks5Proxy(target=("127.0.0.1", upstream.port)))
    pool = socks_pool(f"socks5://127.0.0.1:{proxy.port}", remote_dns=True)
    ProxyPoolManager().connect(pool, "remote.invalid", 1, SETTINGS).close()
    assert proxy.requests[0]["atyp"] == 0x03


def test_socks5_local_dns_when_disabled(track):
    upstream = track(FakeServer(echo_handler))
    proxy = track(FakeSocks5Proxy(target=("127.0.0.1", upstream.port)))
    pool = socks_pool(f"socks5://127.0.0.1:{proxy.port}", remote_dns=False)
    ProxyPoolManager().connect(pool, "localhost", 80, SETTINGS).close()
    assert proxy.requests[0]["atyp"] in (0x01, 0x04)


def test_socks5_ip_literal_uses_ipv4_atyp(track):
    upstream = track(FakeServer(echo_handler))
    proxy = track(FakeSocks5Proxy(target=("127.0.0.1", upstream.port)))
    pool = socks_pool(f"socks5h://127.0.0.1:{proxy.port}")
    ProxyPoolManager().connect(pool, "10.1.2.3", 80, SETTINGS).close()
    assert (proxy.requests[0]["atyp"], proxy.requests[0]["host"]) == (0x01, "10.1.2.3")


def test_socks5_username_password_auth(track):
    upstream = track(FakeServer(echo_handler))
    proxy = track(FakeSocks5Proxy(target=("127.0.0.1", upstream.port), require_auth=True))
    pool = socks_pool(f"socks5h://bob:s3cret@127.0.0.1:{proxy.port}")
    ProxyPoolManager().connect(pool, "h.invalid", 1, SETTINGS).close()
    assert proxy.requests[0]["auth"] == ("bob", "s3cret")
    assert proxy.requests[0]["methods"] == [0x00, 0x02]


def test_socks5_failure_reply_is_connect_error(track):
    proxy = track(FakeSocks5Proxy(reply=5))
    pool = socks_pool(f"socks5h://127.0.0.1:{proxy.port}")
    with pytest.raises(EgressConnectError, match="connection refused"):
        ProxyPoolManager().connect(pool, "h.invalid", 1, SETTINGS)


def test_failed_proxy_gets_penalty_until_expiry():
    clock = [1000.0]
    manager = ProxyPoolManager(clock=lambda: clock[0])
    pool = http_pool(f"http://127.0.0.1:{free_port()}")
    endpoint = manager.endpoints(pool)[0]
    with pytest.raises(EgressConnectError):
        manager.connect(pool, "h", 1, SETTINGS)
    assert manager.is_penalized(endpoint)
    assert manager._penalties[endpoint.raw] == 1030.0  # time.time() + penalty_seconds
    clock[0] = 1029.9
    assert manager.is_penalized(endpoint)
    clock[0] = 1030.1
    assert not manager.is_penalized(endpoint)
    assert manager.available(pool) == [endpoint]


def test_retry_moves_to_next_proxy_and_dead_one_stays_penalized(track):
    upstream = track(FakeServer(echo_handler))
    good = track(FakeConnectProxy(target=("127.0.0.1", upstream.port)))
    dead_url = f"http://127.0.0.1:{free_port()}"
    pool = http_pool(dead_url, f"http://127.0.0.1:{good.port}")
    manager = ProxyPoolManager()
    with manager.connect(pool, "h", 1, SETTINGS) as sock:
        assert roundtrip(sock) == b"ping"
    assert manager.is_penalized(manager.endpoints(pool)[0])
    assert len(good.connects) == 1
    with manager.connect(pool, "h", 1, SETTINGS):  # мёртвая прокси больше не пробуется
        pass
    assert len(good.connects) == 2


def test_pre_send_retries_zero_does_not_try_second_proxy(track):
    good = track(FakeConnectProxy(target=("127.0.0.1", 1)))
    pool = http_pool(f"http://127.0.0.1:{free_port()}", f"http://127.0.0.1:{good.port}")
    settings = EgressSettings(2.0, 5.0, 0, 30)
    with pytest.raises(EgressConnectError):
        ProxyPoolManager().connect(pool, "h", 1, settings)
    assert good.connects == []


def test_direct_pool_retries_without_penalties():
    pool = PoolConfig("direct", "direct")
    manager = ProxyPoolManager()
    with pytest.raises(EgressConnectError):
        manager.connect(pool, "127.0.0.1", free_port(), SETTINGS)
    manager.ensure_available(pool)  # direct не штрафуется


def test_first_available_prefers_first_and_random_covers_all():
    urls = [f"http://127.0.0.1:{4000 + i}" for i in range(3)]
    manager = ProxyPoolManager(rng=random.Random(7))
    first = http_pool(*urls)
    assert all(manager.select(first).port == 4000 for _ in range(10))
    rand = http_pool(*urls, strategy="random")
    seen = {manager.select(rand).port for _ in range(60)}
    assert seen == {4000, 4001, 4002}
    manager.penalize(manager.endpoints(rand)[1], 30)
    assert {manager.select(rand).port for _ in range(60)} == {4000, 4002}


def test_circuit_breaker_fails_fast_when_all_penalized():
    manager = ProxyPoolManager()
    pool = http_pool("http://127.0.0.1:4001", "http://127.0.0.1:4002")
    for endpoint in manager.endpoints(pool):
        manager.penalize(endpoint, 30)
    timings = []
    for _ in range(5):
        started = time.perf_counter()
        with pytest.raises(AllProxiesPenalizedError):
            manager.ensure_available(pool)
        timings.append(time.perf_counter() - started)
    assert min(timings) < 0.001  # мгновенно: сеть не задействована
    started = time.perf_counter()
    with pytest.raises(AllProxiesPenalizedError):
        manager.connect(pool, "h", 1, SETTINGS)
    assert time.perf_counter() - started < 0.05


def test_circuit_breaker_closes_after_penalty_expires():
    clock = [0.0]
    manager = ProxyPoolManager(clock=lambda: clock[0])
    pool = http_pool("http://127.0.0.1:4001")
    manager.penalize(manager.endpoints(pool)[0], 30)
    with pytest.raises(AllProxiesPenalizedError):
        manager.ensure_available(pool)
    clock[0] = 31
    manager.ensure_available(pool)


def test_connect_timeout_gives_egress_timeout_error(track):
    hang = track(FakeServer(lambda conn: time.sleep(3)))  # принимает, но молчит на CONNECT
    pool = http_pool(f"http://127.0.0.1:{hang.port}")
    settings = EgressSettings(connect_timeout=0.3, total_deadline=5.0, retries=0, penalty_seconds=30)
    started = time.monotonic()
    with pytest.raises(EgressTimeoutError):
        ProxyPoolManager().connect(pool, "h", 1, settings)
    assert time.monotonic() - started < 2


def test_total_egress_deadline_bounds_all_attempts(track):
    hang = track(FakeServer(lambda conn: time.sleep(3)))
    pool = http_pool(f"http://127.0.0.1:{hang.port}", f"http://127.0.0.1:{hang.port}/x")
    settings = EgressSettings(connect_timeout=5.0, total_deadline=0.4, retries=1, penalty_seconds=30)
    started = time.monotonic()
    with pytest.raises(EgressTimeoutError):
        ProxyPoolManager().connect(pool, "h", 1, settings)
    assert time.monotonic() - started < 1.5


def test_stats_report_penalized_proxies():
    manager = ProxyPoolManager()
    pool = http_pool("http://127.0.0.1:4001", "http://127.0.0.1:4002")
    manager.penalize(manager.endpoints(pool)[0], 30)
    stats = manager.stats([pool, PoolConfig("direct", "direct")])
    assert stats["p"] == {"type": "http_connect", "proxies_total": 2, "proxies_penalized": 1}
    assert stats["direct"]["proxies_total"] == 0
