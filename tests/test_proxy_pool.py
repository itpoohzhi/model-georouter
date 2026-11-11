from __future__ import annotations

import logging
import random
import socket
import ssl
import threading
import time

import pytest

from model_georouter.config import PoolConfig
from model_georouter.errors import AllProxiesPenalizedError, EgressConnectError, EgressTimeoutError
from model_georouter.proxy_pool import (
    CONNECT_RESPONSE_LIMIT,
    EgressSettings,
    ProxyPoolManager,
    open_http_connect,
    parse_proxy_url,
)
from tests.helpers import (
    FakeConnectProxy,
    FakeServer,
    FakeSocks5Proxy,
    basic_token,
    echo_handler,
    free_port,
    recv_head,
)

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
    assert (endpoint.host, endpoint.port, endpoint.username, endpoint.password) == (
        "proxy.local",
        3128,
        "us@er",
        "p:ss",
    )
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
    hang: FakeServer | None = None
    hang = track(FakeServer(lambda conn: hang.stop.wait(3) if hang else None))  # принимает, но молчит на CONNECT
    pool = http_pool(f"http://127.0.0.1:{hang.port}")
    settings = EgressSettings(connect_timeout=0.3, total_deadline=5.0, retries=0, penalty_seconds=30)
    started = time.monotonic()
    with pytest.raises(EgressTimeoutError):
        ProxyPoolManager().connect(pool, "h", 1, settings)
    assert time.monotonic() - started < 2


def test_total_egress_deadline_bounds_all_attempts(track):
    hang: FakeServer | None = None
    hang = track(FakeServer(lambda conn: hang.stop.wait(3) if hang else None))
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


# ───────────────────────────── Rework Cycle 1 ─────────────────────────────


def flaky_proxy(track, good: FakeConnectProxy, fail_first: int) -> tuple[FakeServer, list[int]]:
    attempts: list[int] = []

    def handler(conn):
        attempts.append(1)
        if len(attempts) <= fail_first:
            return  # EOF посреди handshake
        good._serve(conn)

    return track(FakeServer(handler)), attempts


def test_single_proxy_pool_retries_same_proxy_on_transient_failure(track):  # RW-005
    upstream = track(FakeServer(echo_handler))
    good = FakeConnectProxy(target=("127.0.0.1", upstream.port))
    proxy, attempts = flaky_proxy(track, good, fail_first=1)
    pool = http_pool(f"http://127.0.0.1:{proxy.port}")
    manager = ProxyPoolManager()
    with manager.connect(pool, "h", 1, SETTINGS) as sock:
        assert roundtrip(sock) == b"ping"
    assert len(attempts) == 2
    assert not manager.is_penalized(manager.endpoints(pool)[0])


def test_single_proxy_pool_uses_all_retries_then_penalizes(track):  # RW-005
    proxy, attempts = flaky_proxy(track, FakeConnectProxy(), fail_first=99)
    pool = http_pool(f"http://127.0.0.1:{proxy.port}")
    manager = ProxyPoolManager()
    settings = EgressSettings(connect_timeout=2.0, total_deadline=5.0, retries=2, penalty_seconds=30)
    with pytest.raises(EgressConnectError):
        manager.connect(pool, "h", 1, settings)
    assert len(attempts) == 3
    assert manager.is_penalized(manager.endpoints(pool)[0])


class _Handshake:
    """`wrap`-заглушка: первые `fail_first` вызовов падают TLS-ошибкой, остальные возвращают сокет."""

    def __init__(self, fail_first: int, error: Exception | None = None):
        self.fail_first = fail_first
        self.error = error or ssl.SSLError("handshake failure")
        self.raw: list[socket.socket] = []

    def __call__(self, raw: socket.socket) -> socket.socket:
        self.raw.append(raw)
        if len(self.raw) <= self.fail_first:
            raise self.error
        return raw


def test_tls_handshake_failure_closes_socket_and_retries_origin(track):  # RW-004
    upstream = track(FakeServer(echo_handler))
    pool = PoolConfig("direct", "direct")
    wrap = _Handshake(fail_first=1)
    with ProxyPoolManager().connect(pool, "127.0.0.1", upstream.port, SETTINGS, wrap) as sock:
        assert roundtrip(sock) == b"ping"
    assert len(wrap.raw) == 2 and wrap.raw[0].fileno() == -1  # первый сокет закрыт, FD не утёк


def test_tls_handshake_failure_through_proxy_penalizes_proxy_and_fails_over(track):  # RW-004
    upstream = track(FakeServer(echo_handler))
    first = track(FakeConnectProxy(target=("127.0.0.1", upstream.port)))
    second = track(FakeConnectProxy(target=("127.0.0.1", upstream.port)))
    pool = http_pool(f"http://127.0.0.1:{first.port}", f"http://127.0.0.1:{second.port}")
    manager = ProxyPoolManager()
    wrap = _Handshake(fail_first=1)
    with manager.connect(pool, "h", 1, SETTINGS, wrap) as sock:
        assert roundtrip(sock) == b"ping"
    assert manager.is_penalized(manager.endpoints(pool)[0]) and not manager.is_penalized(manager.endpoints(pool)[1])
    assert len(first.connects) == 1 and len(second.connects) == 1


def test_persistent_tls_failure_exhausts_retries_with_egress_error(track):  # RW-004
    upstream = track(FakeServer(echo_handler))
    proxy = track(FakeConnectProxy(target=("127.0.0.1", upstream.port)))
    pool = http_pool(f"http://127.0.0.1:{proxy.port}")
    manager = ProxyPoolManager()
    wrap = _Handshake(fail_first=99, error=ssl.SSLCertVerificationError("bad cert"))
    with pytest.raises(EgressConnectError, match="TLS handshake"):
        manager.connect(pool, "h", 1, SETTINGS, wrap)
    assert len(wrap.raw) == 2 and all(raw.fileno() == -1 for raw in wrap.raw)
    assert manager.is_penalized(manager.endpoints(pool)[0])


def test_connect_failures_back_off_exponentially_up_to_cap(monkeypatch):
    pauses: list[float] = []
    monkeypatch.setattr("model_georouter.proxy_pool.time.sleep", pauses.append)
    settings = EgressSettings(2.0, 30.0, 3, 30, 0.5, 2.0, 1.5)
    with pytest.raises(EgressConnectError):
        ProxyPoolManager().connect(PoolConfig("direct", "direct"), "127.0.0.1", free_port(), settings)
    assert pauses == [0.5, 1.0, 1.5]  # после последней попытки паузы нет


def test_backoff_is_skipped_when_it_would_exceed_total_deadline(monkeypatch):
    pauses: list[float] = []
    monkeypatch.setattr("model_georouter.proxy_pool.time.sleep", pauses.append)
    settings = EgressSettings(2.0, 1.0, 2, 30, 5.0, 2.0, 5.0)
    with pytest.raises(EgressConnectError):
        ProxyPoolManager().connect(PoolConfig("direct", "direct"), "127.0.0.1", free_port(), settings)
    assert pauses == []


def _backoff_settings(retries: int, **overrides) -> EgressSettings:
    values = {"retry_sleep_seconds": 0.5, "retry_backoff_factor": 2.0, "retry_max_sleep_seconds": 30.0}
    values.update(overrides)
    return EgressSettings(connect_timeout=2.0, total_deadline=60.0, retries=retries, penalty_seconds=30, **values)


@pytest.fixture
def pauses(monkeypatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr("model_georouter.proxy_pool.time.sleep", recorded.append)
    return recorded


def test_tls_handshake_failure_backs_off_before_retrying_same_target(track, pauses):  # F-002
    upstream = track(FakeServer(echo_handler))
    wrap = _Handshake(fail_first=2)
    pool = PoolConfig("direct", "direct")
    with ProxyPoolManager().connect(pool, "127.0.0.1", upstream.port, _backoff_settings(retries=2), wrap) as sock:
        assert roundtrip(sock) == b"ping"
    assert len(wrap.raw) == 3
    assert pauses == [0.5, 1.0]


def test_tls_handshake_failure_through_single_proxy_backs_off(track, pauses):  # F-002
    upstream = track(FakeServer(echo_handler))
    proxy = track(FakeConnectProxy(target=("127.0.0.1", upstream.port)))
    wrap = _Handshake(fail_first=99)
    with pytest.raises(EgressConnectError, match="TLS handshake"):
        ProxyPoolManager().connect(
            http_pool(f"http://127.0.0.1:{proxy.port}"), "h", 1, _backoff_settings(retries=2), wrap
        )
    assert pauses == [0.5, 1.0]  # после третьей (последней) попытки паузы нет


def test_failover_to_next_proxy_in_pool_does_not_sleep(track, pauses):  # F-005
    upstream = track(FakeServer(echo_handler))
    good = track(FakeConnectProxy(target=("127.0.0.1", upstream.port)))
    pool = http_pool(f"http://127.0.0.1:{free_port()}", f"http://127.0.0.1:{good.port}")
    with ProxyPoolManager().connect(pool, "h", 1, _backoff_settings(retries=2)) as sock:
        assert roundtrip(sock) == b"ping"
    assert pauses == []


def test_tls_failover_to_next_proxy_in_pool_does_not_sleep(track, pauses):  # F-002 + F-005
    upstream = track(FakeServer(echo_handler))
    first = track(FakeConnectProxy(target=("127.0.0.1", upstream.port)))
    second = track(FakeConnectProxy(target=("127.0.0.1", upstream.port)))
    pool = http_pool(f"http://127.0.0.1:{first.port}", f"http://127.0.0.1:{second.port}")
    with ProxyPoolManager().connect(pool, "h", 1, _backoff_settings(retries=2), _Handshake(fail_first=1)) as sock:
        assert roundtrip(sock) == b"ping"
    assert pauses == []


def test_exhausted_pool_candidates_exit_without_sleep(pauses):  # F-003
    pool = http_pool(f"http://127.0.0.1:{free_port()}", f"http://127.0.0.1:{free_port()}")
    with pytest.raises(EgressConnectError):
        ProxyPoolManager().connect(pool, "h", 1, _backoff_settings(retries=5))
    assert pauses == []  # оба прокси испробованы: ретраить некуда, спать незачем


def test_retry_to_same_proxy_sleeps_base_times_factor_power(track, pauses):  # F-004
    proxy, attempts = flaky_proxy(track, FakeConnectProxy(), fail_first=99)
    settings = _backoff_settings(retries=3, retry_sleep_seconds=0.2, retry_backoff_factor=3.0)
    with pytest.raises(EgressConnectError):
        ProxyPoolManager().connect(http_pool(f"http://127.0.0.1:{proxy.port}"), "h", 1, settings)
    assert len(attempts) == 4
    # повтор №n спит retry_sleep_seconds * factor ** (n - 1): первый — ровно retry_sleep_seconds
    assert pauses == pytest.approx([0.2, 0.6, 1.8])


@pytest.mark.parametrize("retries, expected", [(0, []), (1, [0.5])])
def test_no_sleep_after_last_attempt(pauses, retries, expected):  # F-003
    with pytest.raises(EgressConnectError):
        ProxyPoolManager().connect(
            PoolConfig("direct", "direct"), "127.0.0.1", free_port(), _backoff_settings(retries=retries)
        )
    assert pauses == expected


def test_huge_retries_with_large_factor_do_not_overflow(pauses):  # F-002 (cycle 2)
    settings = _backoff_settings(retries=400, retry_backoff_factor=10.0, retry_max_sleep_seconds=30.0)
    with pytest.raises(EgressConnectError):
        ProxyPoolManager().connect(PoolConfig("direct", "direct"), "127.0.0.1", free_port(), settings)
    assert pauses and max(pauses) <= 30.0
    assert pauses[-1] == 30.0  # 0.5 * 10.0 ** 398 переполняет float → OverflowError → потолок


def test_gentle_backoff_reaches_max_sleep_without_exponent_cutoff(pauses):  # Cycle 4: без потолка степени
    settings = _backoff_settings(
        retries=41, retry_sleep_seconds=0.000001, retry_backoff_factor=1.5, retry_max_sleep_seconds=3.0
    )
    ProxyPoolManager._backoff_pause(PoolConfig("direct", "direct"), settings, 40, False, time.monotonic() + 60.0)
    # 1e-6 * 1.5 ** 40 ≈ 11.1 > 3.0 → потолок; при обрезке степени до 30 было бы ≈ 0.19
    assert pauses == [3.0]


# ───────────────────── буферизованный CONNECT и согласование таймаутов ─────────────────────


class _CountingSocket:
    """Обёртка сокета: считает вызовы `recv`, остальное делегирует."""

    def __init__(self, sock: socket.socket):
        self._sock = sock
        self.recv_calls: list[int] = []

    def recv(self, size: int, flags: int = 0) -> bytes:
        self.recv_calls.append(flags)
        return self._sock.recv(size, flags)

    def __getattr__(self, name: str):
        return getattr(self._sock, name)


@pytest.fixture
def connect_peer(monkeypatch):
    """`open_http_connect` поверх socketpair: тест пишет «ответ прокси» в `peer`, клиент читает свой конец."""
    client, peer = socket.socketpair()
    counting = _CountingSocket(client)
    monkeypatch.setattr("model_georouter.proxy_pool._dial", lambda host, port, deadline: counting)
    yield counting, peer
    for sock in (client, peer):
        sock.close()


def _connect(deadline_in: float = 3.0):
    endpoint = parse_proxy_url("http://user:s3cret@proxy.local:3128")
    return open_http_connect(endpoint, "api.example.com", 443, time.monotonic() + deadline_in)


def test_open_http_connect_buffered_fast(connect_peer):
    counting, peer = connect_peer
    peer.sendall(b"HTTP/1.1 200 Connection established\r\nProxy-Agent: fake\r\n\r\n")
    sock = _connect()
    assert sock is counting
    assert len(counting.recv_calls) <= 2  # один peek + один забор, не recv(1) на байт
    assert b"CONNECT api.example.com:443 HTTP/1.1" in peer.recv(4096)
    sock.sendall(b"ping")
    peer.settimeout(2)
    assert peer.recv(10) == b"ping"  # туннель рабочий


def test_open_http_connect_eof_peek_fails_fast(track):
    server = track(FakeServer(recv_head))  # вычитывает CONNECT и закрывает соединение, ничего не ответив
    endpoint = parse_proxy_url(f"http://127.0.0.1:{server.port}")
    started = time.monotonic()
    with pytest.raises(EgressConnectError) as excinfo:
        open_http_connect(endpoint, "h", 443, time.monotonic() + 5)
    assert excinfo.value.phase == "connect_read"
    assert time.monotonic() - started < 1  # EOF виден сразу, дедлайн не выжигается


def test_open_http_connect_fragmented_no_cpu_spin(connect_peer):
    counting, peer = connect_peer
    response = b"HTTP/1.1 200 Connection established\r\nX-Pad: " + b"a" * 30 + b"\r\n\r\n"
    chunks = [response[i : i + 10] for i in range(0, len(response), 10)]

    def feed() -> None:
        for piece in chunks:
            time.sleep(0.03)
            peer.sendall(piece)

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    assert _connect() is counting
    feeder.join()
    # на каждый кусок — peek + забор (плюс финальный): холостого цикла нет
    assert len(counting.recv_calls) <= 2 * len(chunks) + 2


def test_open_http_connect_oversized_limit(connect_peer):
    counting, peer = connect_peer
    flood = b"HTTP/1.1 200 OK\r\nX-Pad: " + b"a" * (CONNECT_RESPONSE_LIMIT + 100)  # разделителя нет

    def send_flood() -> None:
        try:
            peer.sendall(flood)
        except OSError:
            pass

    writer = threading.Thread(target=send_flood, daemon=True)
    writer.start()
    try:
        with pytest.raises(EgressConnectError, match="oversized") as excinfo:
            _connect()
        assert excinfo.value.phase == "connect_read"
    finally:
        counting.close()
        peer.close()
        writer.join(1.0)


def test_open_http_connect_preserves_trailing_tls_bytes(connect_peer):
    _, peer = connect_peer
    tls_hello = b"\x16\x03\x03\x00\x05hello"
    peer.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n" + tls_hello)
    sock = _connect()
    sock.settimeout(2)
    assert sock.recv(100) == tls_hello  # ни одного байта за \r\n\r\n не потеряно


@pytest.mark.parametrize("code", [403, 502])
def test_open_http_connect_rejects_non_2xx_and_closes_socket(connect_peer, code):
    counting, peer = connect_peer
    peer.sendall(f"HTTP/1.1 {code} X\r\n\r\n".encode())
    with pytest.raises(EgressConnectError, match=f"HTTP {code}") as excinfo:
        _connect()
    assert excinfo.value.phase == "connect_read"
    assert counting.fileno() == -1


def test_open_http_connect_accepts_2xx_range(connect_peer):
    counting, peer = connect_peer
    peer.sendall(b"HTTP/1.1 204 No Content\r\n\r\n")
    assert _connect() is counting
    assert counting.fileno() != -1


def test_proxy_pool_logs_safe_label_no_password(caplog):
    port = free_port()
    pool = http_pool(f"http://alice:Sup3rS3cret@127.0.0.1:{port}")
    with caplog.at_level(logging.WARNING, logger="model_georouter.proxy_pool"), pytest.raises(EgressConnectError):
        ProxyPoolManager().connect(pool, "h", 443, SETTINGS)
    assert f"http://127.0.0.1:{port}" in caplog.text
    assert "Sup3rS3cret" not in caplog.text and "alice" not in caplog.text


def test_tls_handshake_respects_remaining_attempt_deadline(track):
    upstream = track(FakeServer(echo_handler))
    seen: list[float | None] = []

    def wrap(raw: socket.socket) -> socket.socket:
        seen.append(raw.gettimeout())
        return raw

    settings = EgressSettings(connect_timeout=1.0, total_deadline=60.0, retries=0, penalty_seconds=30)
    ProxyPoolManager().connect(PoolConfig("direct", "direct"), "127.0.0.1", upstream.port, settings, wrap).close()
    assert seen[0] is not None and 0 < seen[0] <= 1.0  # остаток попытки, а не total_deadline


def test_no_penalty_when_attempt_timeout_before_tls(track):
    upstream = track(FakeServer(echo_handler))
    proxy = track(FakeConnectProxy(target=("127.0.0.1", upstream.port)))
    pool = http_pool(f"http://127.0.0.1:{proxy.port}")
    manager = ProxyPoolManager()
    real_open = manager._open

    def slow_open(*args):
        sock = real_open(*args)
        time.sleep(0.15)  # туннель поднят, но бюджет попытки (0.1 с) исчерпан
        return sock

    manager._open = slow_open  # type: ignore[method-assign]
    wrap = _Handshake(fail_first=0)
    settings = EgressSettings(connect_timeout=0.1, total_deadline=5.0, retries=0, penalty_seconds=30)
    with pytest.raises(EgressTimeoutError, match="before TLS"):
        manager.connect(pool, "h", 443, settings, wrap)
    assert wrap.raw == []  # хендшейк не начинался
    assert not manager.is_penalized(manager.endpoints(pool)[0])


def test_connect_overall_deadline_bounds_retries():
    settings = EgressSettings(
        connect_timeout=2.0,
        total_deadline=0.5,
        retries=20,
        penalty_seconds=30,
        retry_sleep_seconds=0.2,
        retry_backoff_factor=1.0,
        retry_max_sleep_seconds=0.2,
    )
    started = time.monotonic()
    with pytest.raises(EgressConnectError):
        ProxyPoolManager().connect(PoolConfig("direct", "direct"), "127.0.0.1", free_port(), settings)
    assert time.monotonic() - started < 0.5 + 0.4  # паузы backoff не выходят за общий дедлайн
