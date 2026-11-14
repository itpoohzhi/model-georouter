"""Зелёные тесты v3 (W3-C, D1-5): фиксируют целевые инварианты из замороженного test_v3_red_repro.py."""

from __future__ import annotations

import json
import socket
import time
from typing import Any

import pytest

from model_georouter.config import LOOPBACK, ConfigManager, parse_config
from model_georouter.config_guard import (
    ALLOWED_ROUTE_DE_PORTS,
    F08_V1_TOTAL_EGRESS_DEADLINE,
    FORBIDDEN_ROUTE_DE_PORTS,
    validate,
    validate_config,
    validate_listen_host,
    validate_route_de_pool,
)
from model_georouter.errors import ConfigError
from tests.helpers import FakeConnectProxy, FakeServer, HttpUpstream, http_request, make_config

PATH = "/v1/chat/completions"
BODY = b'{"model":"claude-3-opus","messages":[{"role":"user","content":"ping"}]}'


def test_chat_completions_uses_empty_message_sanitizer(track, start_bridge, monkeypatch):
    """HTTP-путь очищает пустые сообщения до отправки upstream."""
    from model_georouter import bridge_server

    original = bridge_server.sanitize_openai_empty_messages
    seen = []
    received = []

    def sanitize(body):
        seen.append(body)
        return original(body)

    def respond(conn, req):
        received.append(json.loads(req.body))
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")

    monkeypatch.setattr(bridge_server, "sanitize_openai_empty_messages", sanitize)
    upstream = track(HttpUpstream(respond))
    bridge = start_bridge(make_config(upstream.port))
    body = json.dumps(
        {
            "model": "gpt-4",
            "messages": [
                {"role": "assistant", "content": ""},
                {"role": "user", "content": ""},
                {"role": "user", "content": "ping"},
            ],
        }
    ).encode()
    response = http_request(bridge.port, "POST", PATH, body)
    assert response.status == 200
    assert seen == [body]
    assert received[0]["messages"] == [
        {"role": "assistant", "content": " "},
        {"role": "user", "content": " "},
        {"role": "user", "content": "ping"},
    ]


def _base_config(
    *,
    listen: str = "127.0.0.1",
    port: int = 10830,
    pools: dict[str, Any] | None = None,
    rules: list[dict[str, Any]] | None = None,
    upstreams: dict[str, Any] | None = None,
    server_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Собирает базовый словарь конфигурации моста для проверки стартовой валидации и рантайма."""
    server: dict[str, Any] = {
        "listen": listen,
        "port": port,
        "total_egress_deadline": F08_V1_TOTAL_EGRESS_DEADLINE,
    }
    if server_extra:
        server.update(server_extra)
    cfg: dict[str, Any] = {
        "server": server,
        "pools": {"direct": {"type": "direct"}, **(pools or {})},
        "rules": rules or [],
    }
    if upstreams is not None:
        cfg["upstreams"] = upstreams
    return cfg


def test_v3_green_tls_stall_over_3s_on_single_proxy_route_de_returns_502_fail_fast(track, start_bridge) -> None:
    """TLS stall > 3 с на единственном прокси в пуле route-de возвращает 502 fail-fast (< 3.8 с)."""

    def _stall_tls_handshake(conn: socket.socket) -> None:
        # Принимаем TCP-соединение от CONNECT-прокси и читаем TLS ClientHello, но не шлём ServerHello (> 3 с).
        conn.settimeout(1.0)
        try:
            _ = conn.recv(4096)
        except OSError:
            return
        _ = stalled_tls.stop.wait(4.5)

    stalled_tls = track(FakeServer(_stall_tls_handshake))
    proxy = track(FakeConnectProxy(target=("127.0.0.1", stalled_tls.port)))
    cfg = _base_config(
        port=0,
        pools={"route-de": {"type": "http_connect", "proxies": [f"http://127.0.0.1:{proxy.port}"]}},
        rules=[{"match_prefix": ["claude-"], "pool": "route-de"}],
        upstreams={"opencode-ai": {"host": "127.0.0.1", "port": stalled_tls.port, "use_tls": True}},
    )
    bridge = start_bridge(cfg)

    started = time.monotonic()
    resp = http_request(bridge.port, "POST", PATH, BODY)
    elapsed = time.monotonic() - started

    assert resp.status == 502
    assert resp.json["error"]["retryable"] is True
    assert elapsed < 3.8, f"ожидался fail-fast 502 при TLS stall > 3 с, фактически запрос висел {elapsed:.2f} с"


@pytest.mark.parametrize("forbidden_port", [10820, 10810])
def test_v3_green_route_de_rejects_forbidden_ports_10820_and_10810_on_startup(
    forbidden_port: int, start_bridge
) -> None:
    """Наличие :10820 или :10810 в пуле route-de отклоняется config_guard и при старте моста (D1-5)."""
    assert forbidden_port in FORBIDDEN_ROUTE_DE_PORTS
    cfg_data = _base_config(
        pools={
            "route-de": {
                "type": "http_connect",
                "egress_id": "de-main",
                "proxies": [f"http://127.0.0.1:{forbidden_port}"],
            }
        }
    )
    with pytest.raises(ConfigError):
        validate_route_de_pool(cfg_data["pools"]["route-de"])
    with pytest.raises(ConfigError):
        validate_config(cfg_data)
    with pytest.raises(ConfigError):
        validate(parse_config(cfg_data))
    with pytest.raises(ConfigError):
        start_bridge(cfg_data)

    # Выделенные порты DE (10827..10829) с валидным egress_id проходят стартовую валидацию
    valid_cfg = _base_config(
        pools={
            "route-de": {
                "type": "http_connect",
                "egress_id": "de-main",
                "proxies": [f"http://127.0.0.1:{p}" for p in sorted(ALLOWED_ROUTE_DE_PORTS)],
            }
        }
    )
    guarded = validate_config(valid_cfg)
    assert guarded.route_de_egress_id == "de-main"
    assert validate(parse_config(valid_cfg)).bridge.server.listen == LOOPBACK


def test_v3_green_non_loopback_listen_rejected_and_ipv4_loopback_accepted() -> None:
    """Не-loopback listen отклоняется; parse_config принимает 127.0.0.1 и ::1 ортогонально, config_guard требует 127.0.0.1 (D1-5)."""
    for non_loopback in ("0.0.0.0", "192.168.1.10", "10.0.0.1", "::", "localhost"):
        with pytest.raises(ConfigError):
            validate_listen_host(non_loopback)
        with pytest.raises(ConfigError):
            parse_config(_base_config(listen=non_loopback))
        with pytest.raises(ConfigError):
            ConfigManager.from_dict(_base_config(listen=non_loopback))
        with pytest.raises(ConfigError):
            validate_config(_base_config(listen=non_loopback))

    with pytest.raises(ConfigError):
        validate_listen_host("::1")
    with pytest.raises(ConfigError):
        validate_config(_base_config(listen="::1"))
    assert parse_config(_base_config(listen="::1")).server.listen == "::1"
    assert ConfigManager.from_dict(_base_config(listen="::1")).get().server.listen == "::1"

    parsed = parse_config(_base_config(listen=LOOPBACK))
    managed = ConfigManager.from_dict(_base_config(listen=LOOPBACK)).get()
    guarded = validate_config(_base_config(listen=LOOPBACK))
    assert validate_listen_host(LOOPBACK) == LOOPBACK
    assert parsed.server.listen == LOOPBACK
    assert managed.server.listen == LOOPBACK
    assert guarded.bridge.server.listen == LOOPBACK
