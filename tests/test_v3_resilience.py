"""Тесты отказоустойчивости v3 (W3-C): хеджирование route-de, дедлайн F-08, legacy-путь и config_guard."""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from typing import cast

import pytest

from model_georouter import config_guard
from model_georouter.config import PoolConfig, parse_config
from model_georouter.egress_admission import EgressAdmission
from model_georouter.egress_hedge import (
    MAX_HEDGE_BUDGET_S,
    ROUTE_DE_POOL,
    HedgePolicy,
    HedgeRecorder,
    hedged_connect,
)
from model_georouter.errors import ConfigError
from model_georouter.proxy_pool import ProxyEndpoint
from model_georouter.upstream_state import SCHEMA_VERSION, UpstreamStateWriter
from tests.fake_server import FakeServer
from tests.helpers import FakeConnectProxy, http_request
from tests.v3_types import ConfigData, StartBridge, Track

PATH = "/v1/chat/completions"
BODY = b'{"model":"claude-3-opus","messages":[{"role":"user","content":"ping"}]}'


def _base_config(
    *,
    listen: str = "127.0.0.1",
    port: int = 0,
    pools: dict[str, dict[str, object]] | None = None,
    rules: list[dict[str, object]] | None = None,
    upstreams: dict[str, object] | None = None,
    server_extra: dict[str, object] | None = None,
) -> ConfigData:
    """Формирует конфигурацию моста для интеграционных и модульных проверок v3."""
    cfg: ConfigData = {
        "server": {"listen": listen, "port": port, **(server_extra or {})},
        "pools": {"direct": {"type": "direct"}, **(pools or {})},
        "rules": rules or [],
    }
    if upstreams is not None:
        cfg["upstreams"] = upstreams
    return cfg


def _ok_upstream_handler(conn: socket.socket) -> None:
    """Читает HTTP-запрос и немедленно возвращает 200 OK с валидным JSON-ответом."""
    conn.settimeout(1.0)
    try:
        _ = conn.recv(4096)
        payload = b'{"id":"chatcmpl-v3","ok":true}'
        headers = (
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode("ascii")
        )
        conn.sendall(headers + payload)
    except OSError:
        return


def test_stalled_first_route_de_entry_second_wins_under_3s(
    track: Track, start_bridge: StartBridge, tmp_path: Path
) -> None:
    """Зависший первый вход route-de: второй вход побеждает за <= 3 с (unit + HTTP bridge)."""
    pool = PoolConfig(
        name=ROUTE_DE_POOL,
        type="http_connect",
        proxies=("http://127.0.0.1:10827", "http://127.0.0.1:10828"),
    )
    stall_gate = threading.Event()
    recorder = HedgeRecorder()
    state_file = tmp_path / "upstream_state.json"
    writer = UpstreamStateWriter(state_file, min_interval_s=0.0)

    def _attempt(ep: ProxyEndpoint | None, deadline: float) -> socket.socket:
        assert ep is not None
        if ep.port == 10827:
            _ = stall_gate.wait(timeout=max(0.0, deadline - time.monotonic() + 0.5))
            raise TimeoutError("first proxy stalled")
        left, right = socket.socketpair()
        right.close()
        return left

    t0 = time.monotonic()
    win_sock = hedged_connect(
        pool, _attempt, policy=HedgePolicy(enabled=True, hedge_delay_s=0.08, budget_s=2.0), recorder=recorder
    )
    elapsed_unit = time.monotonic() - t0
    stall_gate.set()
    win_sock.close()

    assert elapsed_unit <= MAX_HEDGE_BUDGET_S and len(recorder.outcomes) == 1
    outcome = recorder.outcomes[0]
    assert outcome.winner == "http://127.0.0.1:10828" and outcome.hedged is True and outcome.failover is True
    writer.record(outcome)
    persisted = cast(dict[str, object], json.loads(state_file.read_text(encoding="utf-8")))
    assert persisted["schema"] == SCHEMA_VERSION
    assert persisted["active_upstream"] == "127.0.0.1:10828" and persisted["failover_count"] == 1

    # Интеграционная проверка через HTTP-мост: первый прокси висит на CONNECT, второй отвечает 200 OK
    def _stall_connect(conn: socket.socket) -> None:
        conn.settimeout(1.0)
        try:
            _ = conn.recv(1024)
        except OSError:
            return
        _ = stalled_proxy.stop.wait(4.5)

    stalled_proxy = track(FakeServer(_stall_connect))
    healthy_upstream = track(FakeServer(_ok_upstream_handler))
    healthy_proxy = track(FakeConnectProxy(target=("127.0.0.1", cast(int, healthy_upstream.port))))
    proxies = [f"http://127.0.0.1:{cast(int, stalled_proxy.port)}", f"http://127.0.0.1:{cast(int, healthy_proxy.port)}"]
    cfg = _base_config(
        pools={ROUTE_DE_POOL: {"type": "http_connect", "proxies": proxies}},
        rules=[{"match_prefix": ["claude-"], "pool": ROUTE_DE_POOL}],
        upstreams={"opencode-ai": {"host": "127.0.0.1", "port": cast(int, healthy_upstream.port), "use_tls": False}},
        server_extra={"hedged_connect_enabled": True},
    )
    bridge = start_bridge(cfg)
    started = time.monotonic()
    resp = http_request(bridge.port, "POST", PATH, BODY)
    elapsed_http = time.monotonic() - started
    assert resp.status == 200 and cast(dict[str, object], resp.json)["ok"] is True
    assert elapsed_http <= 3.0, f"второй вход route-de должен побеждать за <= 3 с, получено {elapsed_http:.2f} с"


def test_all_route_de_entries_stalled_returns_502_within_f08_deadline(track: Track, start_bridge: StartBridge) -> None:
    """Если все входы route-de зависли, мост возвращает 502 fail-fast строго в пределах дедлайна F-08."""

    def _stall_tls(conn: socket.socket) -> None:
        conn.settimeout(1.0)
        try:
            _ = conn.recv(4096)
        except OSError:
            return
        _ = stalled_tls.stop.wait(0.8)

    stalled_tls = track(FakeServer(_stall_tls))
    proxy_a = track(FakeConnectProxy(target=("127.0.0.1", cast(int, stalled_tls.port))))
    proxy_b = track(FakeConnectProxy(target=("127.0.0.1", cast(int, stalled_tls.port))))
    f08_limit = config_guard.F08_V1_TOTAL_EGRESS_DEADLINE
    proxies = [f"http://127.0.0.1:{cast(int, proxy_a.port)}", f"http://127.0.0.1:{cast(int, proxy_b.port)}"]
    cfg = _base_config(
        pools={ROUTE_DE_POOL: {"type": "http_connect", "proxies": proxies}},
        rules=[{"match_prefix": ["claude-"], "pool": ROUTE_DE_POOL}],
        upstreams={"opencode-ai": {"host": "127.0.0.1", "port": cast(int, stalled_tls.port), "use_tls": True}},
        server_extra={"hedged_connect_enabled": True, "total_egress_deadline": f08_limit},
    )
    bridge = start_bridge(cfg)
    admission = EgressAdmission(config=config_guard.validate(parse_config(cfg)))
    with admission.admit(ROUTE_DE_POOL, PATH) as ticket:
        assert ticket.header_deadline_s <= f08_limit
        started = time.monotonic()
        resp = http_request(bridge.port, "POST", PATH, BODY)
        elapsed = time.monotonic() - started

    assert resp.status == 502 and cast(dict[str, dict[str, object]], resp.json)["error"]["retryable"] is True
    assert elapsed < 3.8 and elapsed < f08_limit


def test_hedged_connect_disabled_uses_legacy_single_attempt_path(track: Track, start_bridge: StartBridge) -> None:
    """При hedged_connect_enabled = False используется legacy-путь (одна попытка без параллельного hedge)."""
    pool = PoolConfig(
        name=ROUTE_DE_POOL,
        type="http_connect",
        proxies=("http://127.0.0.1:10827", "http://127.0.0.1:10828"),
    )
    called_ports: list[int] = []
    recorder = HedgeRecorder()

    def _attempt(ep: ProxyEndpoint | None, deadline: float) -> socket.socket:
        assert ep is not None and deadline > time.monotonic()
        called_ports.append(ep.port)
        left, right = socket.socketpair()
        right.close()
        return left

    legacy_policy = HedgePolicy(enabled=False, hedge_delay_s=0.05, budget_s=1.0)
    assert legacy_policy.should_hedge(ROUTE_DE_POOL, 2) is False
    sock = hedged_connect(pool, _attempt, policy=legacy_policy, recorder=recorder)
    sock.close()

    assert called_ports == [10827] and len(recorder.outcomes) == 1
    assert recorder.outcomes[0].hedged is False and recorder.outcomes[0].attempts == 1
    assert recorder.outcomes[0].failover is False

    upstream = track(FakeServer(_ok_upstream_handler))
    proxy1 = track(FakeConnectProxy(target=("127.0.0.1", cast(int, upstream.port))))
    second_touched = threading.Event()
    proxy2 = track(FakeServer(lambda conn: (second_touched.set(), conn.close())))
    proxies = [f"http://127.0.0.1:{cast(int, proxy1.port)}", f"http://127.0.0.1:{cast(int, proxy2.port)}"]
    cfg = _base_config(
        pools={ROUTE_DE_POOL: {"type": "http_connect", "proxies": proxies}},
        rules=[{"match_prefix": ["claude-"], "pool": ROUTE_DE_POOL}],
        upstreams={"opencode-ai": {"host": "127.0.0.1", "port": cast(int, upstream.port), "use_tls": False}},
        server_extra={"hedged_connect_enabled": False, "single_pool_max_retries": 0},
    )
    bridge = start_bridge(cfg)
    resp = http_request(bridge.port, "POST", PATH, BODY)
    assert resp.status == 200 and second_touched.is_set() is False


@pytest.mark.parametrize("forbidden_port", sorted(config_guard.FORBIDDEN_ROUTE_DE_PORTS))
def test_forbidden_ports_in_route_de_rejected_by_config_guard_and_startup(
    forbidden_port: int, start_bridge: StartBridge, tmp_path: Path
) -> None:
    """Порты :10820 и :10810 в пуле route-de отклоняются config_guard и при запуске моста."""
    pool_cfg: dict[str, object] = {
        "type": "http_connect",
        "egress_id": "de-dedicated",
        "proxies": [f"http://127.0.0.1:{forbidden_port}"],
    }
    cfg_data = _base_config(
        port=10830,
        pools={ROUTE_DE_POOL: pool_cfg},
        server_extra={"total_egress_deadline": config_guard.F08_V1_TOTAL_EGRESS_DEADLINE},
    )
    for check in (
        lambda: config_guard.validate_route_de_pool(pool_cfg),
        lambda: config_guard.validate_config(cfg_data),
        lambda: config_guard.validate(parse_config(cfg_data)),
        lambda: start_bridge(cfg_data),
    ):
        with pytest.raises(ConfigError):
            _ = check()

    cfg_path = tmp_path / f"bad_{forbidden_port}.json"
    _ = cfg_path.write_text(json.dumps(cfg_data), encoding="utf-8")
    with pytest.raises(ConfigError):
        _ = config_guard.load_and_validate_config(cfg_path)
    assert config_guard.main(["--config", str(cfg_path)]) == 1

    cred_pool: dict[str, object] = {
        "type": "http_connect",
        "egress_id": "de-dedicated",
        "proxies": [f"http://secret_user:secret_pass@127.0.0.1:{forbidden_port}"],
    }
    cred_cfg = _base_config(
        port=10830,
        pools={ROUTE_DE_POOL: cred_pool},
        server_extra={"total_egress_deadline": config_guard.F08_V1_TOTAL_EGRESS_DEADLINE},
    )
    with pytest.raises(ConfigError) as exc_info:
        _ = parse_config(cred_cfg)
    assert "secret_user" not in str(exc_info.value) and "secret_pass" not in str(exc_info.value)
