"""Красные репро-тесты W0 для целевого поведения v3 (TLS fail-fast и стартовая валидация конфига)."""

from __future__ import annotations

import socket
import time
from typing import cast

import pytest

from model_georouter.config import ConfigManager, parse_config
from model_georouter.errors import ConfigError
from tests.fake_server import FakeServer
from tests.helpers import FakeConnectProxy, http_request
from tests.v3_types import ConfigData, StartBridge, Track

PATH = "/v1/chat/completions"
BODY = b'{"model":"claude-3-opus","messages":[{"role":"user","content":"ping"}]}'


def _base_config(
    *,
    listen: str = "127.0.0.1",
    port: int = 10830,
    pools: dict[str, dict[str, object]] | None = None,
    rules: list[dict[str, object]] | None = None,
    upstreams: dict[str, object] | None = None,
    server_extra: dict[str, object] | None = None,
) -> ConfigData:
    """Собирает базовый словарь конфигурации моста для проверки стартовой валидации и рантайма."""
    server: dict[str, object] = {"listen": listen, "port": port}
    if server_extra:
        server.update(server_extra)
    cfg: ConfigData = {
        "server": server,
        "pools": {"direct": {"type": "direct"}, **(pools or {})},
        "rules": rules or [],
    }
    if upstreams is not None:
        cfg["upstreams"] = upstreams
    return cfg


def test_v3_tls_stall_over_3s_on_single_proxy_route_de_returns_502_fail_fast(
    track: Track, start_bridge: StartBridge
) -> None:
    """TLS stall > 3 с на единственном прокси в пуле route-de должен возвращать 502 (fail-fast вместо зависания)."""

    def _stall_tls_handshake(conn: socket.socket) -> None:
        # Принимаем TCP-соединение от CONNECT-прокси и читаем TLS ClientHello, но не шлём ServerHello (> 3 с).
        conn.settimeout(1.0)
        try:
            _ = conn.recv(4096)
        except OSError:
            return
        _ = stalled_tls.stop.wait(4.5)

    stalled_tls = track(FakeServer(_stall_tls_handshake))
    proxy = track(FakeConnectProxy(target=("127.0.0.1", cast(int, stalled_tls.port))))
    cfg = _base_config(
        port=0,
        pools={"route-de": {"type": "http_connect", "proxies": [f"http://127.0.0.1:{cast(int, proxy.port)}"]}},
        rules=[{"match_prefix": ["claude-"], "pool": "route-de"}],
        upstreams={"opencode-ai": {"host": "127.0.0.1", "port": cast(int, stalled_tls.port), "use_tls": True}},
    )
    bridge = start_bridge(cfg)

    started = time.monotonic()
    resp = http_request(bridge.port, "POST", PATH, BODY)
    elapsed = time.monotonic() - started

    assert resp.status == 502
    assert cast(dict[str, dict[str, object]], resp.json)["error"]["retryable"] is True
    assert elapsed < 3.8, f"ожидался fail-fast 502 при TLS stall > 3 с, фактически запрос висел {elapsed:.2f} с"


@pytest.mark.parametrize("forbidden_port", [10820, 10810])
def test_v3_route_de_rejects_forbidden_ports_10820_and_10810_on_startup(forbidden_port: int) -> None:
    """Наличие :10820 или :10810 в пуле route-de должно отклоняться при старте конфигурации."""
    cfg_data = _base_config(
        pools={
            "route-de": {
                "type": "http_connect",
                "proxies": [f"http://127.0.0.1:{forbidden_port}"],
            }
        }
    )
    with pytest.raises(ConfigError):
        _ = parse_config(cfg_data)
    with pytest.raises(ConfigError):
        _ = ConfigManager.from_dict(cfg_data)


def test_v3_non_loopback_listen_rejected_and_loopback_127_and_ipv6_accepted() -> None:
    """Не-loopback listen (не 127.0.0.1/::1) должен отклоняться при старте конфигурации."""
    for non_loopback in ("0.0.0.0", "192.168.1.10", "10.0.0.1", "::", "localhost"):
        with pytest.raises(ConfigError):
            _ = parse_config(_base_config(listen=non_loopback))
        with pytest.raises(ConfigError):
            _ = ConfigManager.from_dict(_base_config(listen=non_loopback))

    for loopback in ("127.0.0.1", "::1"):
        try:
            parsed = parse_config(_base_config(listen=loopback))
            managed = ConfigManager.from_dict(_base_config(listen=loopback)).get()
        except ConfigError as exc:
            raise AssertionError(f"loopback listen {loopback!r} должен приниматься при старте конфигурации") from exc
        assert parsed.server.listen == loopback
        assert managed.server.listen == loopback
