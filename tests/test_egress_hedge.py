"""Тесты модуля `model_georouter.egress_hedge` (карточка W1-A)."""

from __future__ import annotations

import socket
import ssl
import threading
import time
from collections.abc import Callable
from typing import cast

import pytest

from model_georouter.config import PoolConfig
from model_georouter.egress_hedge import (
    AttemptFn,
    EgressHedgeRunner,
    HedgeOutcome,
    HedgePolicy,
    HedgeRecorder,
    connect_pool_hedged,
    hedged_connect,
)
from model_georouter.errors import AllProxiesPenalizedError, EgressConnectError, EgressTimeoutError
from model_georouter.proxy_pool import EgressSettings, ProxyEndpoint, ProxyPoolManager
from tests.fake_server import FakeServer
from tests.helpers import FakeConnectProxy, echo_handler
from tests.v3_types import Track


def _route_de_pool(*urls: str) -> PoolConfig:
    proxies = tuple(urls) or ("http://127.0.0.1:19001", "http://127.0.0.1:19002")
    return PoolConfig("route-de", "http_connect", "first_available", True, proxies)


def _socket_pair_attempt(
    created: dict[str, socket.socket], stall_first_s: float = 0.0
) -> tuple[list[str], list[socket.socket], AttemptFn]:
    """Создаёт тестовую функцию попытки на базе `socket.socketpair()` с сохранением сокетов по метке узла."""
    called: list[str] = []
    peers: list[socket.socket] = []
    lock = threading.Lock()

    def attempt(endpoint: ProxyEndpoint | None, deadline: float) -> socket.socket:
        label = endpoint.label if endpoint is not None else "direct"
        with lock:
            idx = len(called)
            called.append(label)
        client, peer = socket.socketpair()
        with lock:
            created[label] = client
            peers.append(peer)
        if idx == 0 and stall_first_s > 0:
            time.sleep(min(stall_first_s, max(0.0, deadline - time.monotonic() + 0.05)))
            if time.monotonic() > deadline:
                client.close()
                raise TimeoutError("first node stalled past deadline")
        return client

    return called, peers, attempt


def test_flag_disabled_executes_legacy_single_attempt_only() -> None:
    """При выключенном флаге выполняется ровно одна попытка без запуска второго узла route-de."""
    pool = _route_de_pool("http://127.0.0.1:19001", "http://127.0.0.1:19002")
    created: dict[str, socket.socket] = {}
    called, peers, attempt = _socket_pair_attempt(created)
    recorder = HedgeRecorder()
    sock = hedged_connect(pool, attempt, policy=HedgePolicy(enabled=False), recorder=recorder)
    try:
        assert called == ["http://127.0.0.1:19001"]
        assert len(recorder.outcomes) == 1
        outcome = recorder.outcomes[0]
        assert (outcome.winner, outcome.attempts, outcome.hedged, outcome.failover) == (
            "http://127.0.0.1:19001",
            1,
            False,
            False,
        )
        assert recorder.winners == ["http://127.0.0.1:19001"] and recorder.failovers == []
    finally:
        sock.close()
        for peer in peers:
            peer.close()


def test_flag_disabled_single_attempt_failure_does_not_try_second_node() -> None:
    """При выключенном флаге падение первой попытки не запускает вторую попытку по пулу."""
    pool = _route_de_pool("http://127.0.0.1:19001", "http://127.0.0.1:19002")
    attempted: list[str] = []
    recorder = HedgeRecorder()

    def failing_attempt(endpoint: ProxyEndpoint | None, _deadline: float) -> socket.socket:
        assert endpoint is not None
        attempted.append(endpoint.label)
        raise EgressConnectError("primary node down", phase="connect")

    with pytest.raises(EgressConnectError, match="primary node down"):
        _ = hedged_connect(pool, failing_attempt, policy=HedgePolicy(enabled=False), recorder=recorder)
    assert attempted == ["http://127.0.0.1:19001"]
    assert recorder.winners == [] and len(recorder.failures) == 1


def test_hedge_enabled_stalled_first_node_fails_over_to_second_within_3s() -> None:
    """При зависании первого узла > hedge_delay_s запускается второй узел route-de, побеждает за < 3 с и закрывает проигравшего."""
    pool = _route_de_pool("http://127.0.0.1:19001", "http://127.0.0.1:19002")
    created: dict[str, socket.socket] = {}
    called, peers, attempt = _socket_pair_attempt(created, stall_first_s=0.22)
    recorder = HedgeRecorder()
    policy = HedgePolicy(enabled=True, hedge_delay_s=0.05, budget_s=2.0)

    started = time.monotonic()
    winner_sock = hedged_connect(pool, attempt, policy=policy, recorder=recorder)
    elapsed = time.monotonic() - started
    try:
        assert elapsed < 1.0  # укладывается с запасом в бюджет <= 3 с
        assert called == ["http://127.0.0.1:19001", "http://127.0.0.1:19002"]
        time.sleep(0.25)
        assert created["http://127.0.0.1:19001"].fileno() == -1
        assert created["http://127.0.0.1:19002"].fileno() != -1
        report = recorder.outcomes[0]
        assert (report.pool, report.winner, report.attempts) == ("route-de", "http://127.0.0.1:19002", 2)
        assert report.hedged is True and report.failover is True and report.loser_closed is True
        assert recorder.winners == ["http://127.0.0.1:19002"]
        assert recorder.failovers == ["http://127.0.0.1:19002"]
    finally:
        winner_sock.close()
        for peer in peers:
            peer.close()


def test_hedge_enabled_fast_first_node_wins_without_spawning_second() -> None:
    """Если первый узел route-de отвечает быстрее hedge_delay_s, второй узел не запускается."""
    pool = _route_de_pool("http://127.0.0.1:19001", "http://127.0.0.1:19002")
    created: dict[str, socket.socket] = {}
    called, peers, attempt = _socket_pair_attempt(created)
    captured: list[HedgeOutcome] = []
    recorder_fn: Callable[[HedgeOutcome], None] = captured.append
    sock = hedged_connect(pool, attempt, policy=HedgePolicy(enabled=True, hedge_delay_s=0.2), recorder=recorder_fn)
    try:
        assert called == ["http://127.0.0.1:19001"]
        assert len(captured) == 1 and captured[0].winner == "http://127.0.0.1:19001"
        assert captured[0].attempts == 1 and not captured[0].hedged and not captured[0].failover
    finally:
        sock.close()
        for peer in peers:
            peer.close()


def test_hedge_all_attempts_fail_or_stall_raises_within_budget() -> None:
    """При падении всех попыток или полном зависании выбрасывается ошибка соединения/таймаута в рамках <= 3 с."""
    pool = _route_de_pool("http://127.0.0.1:19001", "http://127.0.0.1:19002")
    recorder = HedgeRecorder()
    attempted: list[str] = []

    def always_fail(endpoint: ProxyEndpoint | None, _deadline: float) -> socket.socket:
        assert endpoint is not None
        attempted.append(endpoint.label)
        raise OSError("connection refused")

    with pytest.raises(EgressConnectError, match="OSError"):
        _ = hedged_connect(
            pool, always_fail, policy=HedgePolicy(enabled=True, hedge_delay_s=0.05, budget_s=1.0), recorder=recorder
        )
    assert attempted == ["http://127.0.0.1:19001", "http://127.0.0.1:19002"]
    assert recorder.winners == [] and len(recorder.failures) == 1

    def stall_forever(_endpoint: ProxyEndpoint | None, deadline: float) -> socket.socket:
        time.sleep(max(0.0, deadline - time.monotonic() + 0.05))
        raise TimeoutError("stalled in handshake")

    started = time.monotonic()
    with pytest.raises(EgressTimeoutError):
        _ = hedged_connect(
            pool, stall_forever, policy=HedgePolicy(enabled=True, hedge_delay_s=0.05, budget_s=0.25), recorder=recorder
        )
    assert time.monotonic() - started < 1.0


def test_runner_with_proxy_pool_manager_and_tls_wrap(track: Track, monkeypatch: pytest.MonkeyPatch) -> None:
    """Интеграция `EgressHedgeRunner` с `ProxyPoolManager`: TLS stall на 1-м прокси переключается на 2-й прокси."""
    upstream = track(FakeServer(echo_handler))
    p1 = track(FakeConnectProxy(target=("127.0.0.1", cast(int, upstream.port))))
    p2 = track(FakeConnectProxy(target=("127.0.0.1", cast(int, upstream.port))))
    pool = _route_de_pool(f"http://127.0.0.1:{cast(int, p1.port)}", f"http://127.0.0.1:{cast(int, p2.port)}")
    recorder = HedgeRecorder()
    manager = ProxyPoolManager()
    runner = EgressHedgeRunner(manager, HedgePolicy(enabled=True, hedge_delay_s=0.06, budget_s=1.5), recorder)
    wrap_calls, wrap_lock = 0, threading.Lock()
    ok_called: list[str] = []

    def mark_ok(ep: ProxyEndpoint) -> None:
        ok_called.append(ep.label)

    monkeypatch.setattr(manager, "mark_ok", mark_ok, raising=False)

    def wrap_tls(raw: socket.socket) -> socket.socket:
        nonlocal wrap_calls
        with wrap_lock:
            wrap_calls += 1
            current = wrap_calls
        if current == 1:
            time.sleep(0.25)
            raise ssl.SSLError("stalled handshake")
        return raw

    settings = EgressSettings(connect_timeout=1.5, total_deadline=3.0, retries=1, penalty_seconds=30)
    with connect_pool_hedged(runner, pool, ("127.0.0.1", cast(int, upstream.port)), settings, wrap=wrap_tls) as sock:
        sock.settimeout(2.0)
        sock.sendall(b"hedged-ok")
        assert sock.recv(32) == b"hedged-ok"
    assert recorder.winners == [f"http://127.0.0.1:{cast(int, p2.port)}"]
    assert recorder.failovers == [f"http://127.0.0.1:{cast(int, p2.port)}"]
    eps = manager.endpoints(pool)
    # Проигравший падает с SSLError уже после победы 2-го узла: штраф ставится асинхронно в `_attempt`.
    wait_until = time.monotonic() + 1.0
    while not manager.is_penalized(eps[0]) and time.monotonic() < wait_until:
        time.sleep(0.01)
    assert manager.is_penalized(eps[0]) is True and manager.is_penalized(eps[1]) is False
    assert ok_called == [f"http://127.0.0.1:{cast(int, p2.port)}"]


def test_hedge_win_does_not_penalize_slow_but_alive_loser() -> None:
    """Regression F-001/F-028: победа 2-го узла не штрафует живой, но медленный 1-й узел — пул route-de остаётся 2-узловым."""
    pool = _route_de_pool("http://127.0.0.1:19001", "http://127.0.0.1:19002")
    manager = ProxyPoolManager()
    eps = manager.endpoints(pool)
    created: dict[str, socket.socket] = {}
    called, peers, attempt = _socket_pair_attempt(created, stall_first_s=0.22)
    recorder = HedgeRecorder()
    policy = HedgePolicy(enabled=True, hedge_delay_s=0.05, budget_s=2.0)
    winner_sock = hedged_connect(pool, attempt, policy=policy, recorder=recorder, candidates=eps, manager=manager)
    try:
        time.sleep(0.3)  # дожидаемся завершения медленного проигравшего
        assert called == [eps[0].label, eps[1].label]
        assert recorder.winners == [eps[1].label] and recorder.failovers == [eps[1].label]
        assert manager.is_penalized(eps[0]) is False and manager.is_penalized(eps[1]) is False
        assert len(manager.available(pool)) == 2
        assert policy.should_hedge(pool.name, len(manager.available(pool))) is True
    finally:
        winner_sock.close()
        for peer in peers:
            peer.close()


def test_hedge_win_penalizes_only_failed_node() -> None:
    """Упавший с ошибкой узел штрафуется, победивший — нет."""
    pool = _route_de_pool("http://127.0.0.1:19001", "http://127.0.0.1:19002")
    manager = ProxyPoolManager()
    eps = manager.endpoints(pool)
    peers: list[socket.socket] = []

    def first_fails(endpoint: ProxyEndpoint | None, _deadline: float) -> socket.socket:
        assert endpoint is not None
        if endpoint.raw == eps[0].raw:
            raise OSError("connection refused")
        client, peer = socket.socketpair()
        peers.append(peer)
        return client

    recorder = HedgeRecorder()
    policy = HedgePolicy(enabled=True, hedge_delay_s=0.05, budget_s=2.0)
    winner_sock = hedged_connect(pool, first_fails, policy=policy, recorder=recorder, candidates=eps, manager=manager)
    try:
        assert recorder.winners == [eps[1].label]
        assert manager.is_penalized(eps[0]) is True and manager.is_penalized(eps[1]) is False
    finally:
        winner_sock.close()
        for peer in peers:
            peer.close()


def test_hedge_without_winner_penalizes_nodes_stalled_past_deadline() -> None:
    """Без победителя узлы, не ответившие до дедлайна, штрафуются."""
    pool = _route_de_pool("http://127.0.0.1:19001", "http://127.0.0.1:19002")
    manager = ProxyPoolManager()
    eps = manager.endpoints(pool)
    release = threading.Event()

    def stall(_endpoint: ProxyEndpoint | None, _deadline: float) -> socket.socket:
        _ = release.wait(2.0)
        raise TimeoutError("stalled")

    try:
        with pytest.raises(EgressTimeoutError):
            _ = hedged_connect(
                pool,
                stall,
                policy=HedgePolicy(enabled=True, hedge_delay_s=0.05, budget_s=0.2),
                candidates=eps,
                manager=manager,
            )
        assert manager.is_penalized(eps[0]) is True and manager.is_penalized(eps[1]) is True
    finally:
        release.set()


def test_policy_validation_and_empty_candidates() -> None:
    """Проверка валидации параметров `HedgePolicy` и пустого списка кандидатов."""
    with pytest.raises(ValueError, match="positive"):
        _ = HedgePolicy(hedge_delay_s=0.0)
    with pytest.raises(ValueError, match="exceed 3s"):
        _ = HedgePolicy(budget_s=3.5)
    recorder = HedgeRecorder()
    with pytest.raises(AllProxiesPenalizedError):
        _ = hedged_connect(
            _route_de_pool("http://127.0.0.1:19001"), lambda _ep, _dl: socket.socket(), recorder=recorder, candidates=[]
        )
    assert len(recorder.failures) == 1
