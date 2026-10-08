"""Регрессионные тесты замечаний Совета Тимлидов Cycle 2 (RW-001 .. RW-006)."""

from __future__ import annotations

import json
import re
import socket
import threading
import time
from pathlib import Path

import pytest

from tests.helpers import HttpUpstream, make_config, read_all, wait_for
from universal_ai_bridge import SERVICE_NAME, SERVICE_NAME_ALIASES
from universal_ai_bridge.bridge_server import BridgeServer
from universal_ai_bridge.config import ConfigError, ConfigManager, PoolConfig, parse_config
from universal_ai_bridge.errors import BadRequestError, EgressTimeoutError
from universal_ai_bridge.http_wire import parse_request_head
from universal_ai_bridge.model_router import BodyInspector
from universal_ai_bridge.proxy_pool import EgressSettings, ProxyPoolManager, _resolve

ROOT = Path(__file__).resolve().parent.parent
PATH = "/v1/chat/completions"


def _ok(conn, req) -> None:
    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\nConnection: close\r\n\r\nx")


def _raw_exchange(port: int, payload: bytes) -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(payload)
        return read_all(sock)


# ───────────────────────────── RW-001: отказ на этапе accept ─────────────────────────────


def test_rw001_ingress_overflow_rejected_without_spawning_threads(track, start_bridge):
    up = track(HttpUpstream(_ok))
    bridge = start_bridge(make_config(up.port, server={"max_connections": 1}))
    started: list[int] = []
    original = bridge.server.process_request_thread

    def counting(request, client_address):
        started.append(1)
        original(request, client_address)

    bridge.server.process_request_thread = counting
    holder = socket.create_connection(("127.0.0.1", bridge.port), timeout=5)
    try:
        holder.sendall(b"POST " + PATH.encode() + b" HTTP/1.1\r\nHost: h\r\n")  # голова не завершена: слот занят
        assert wait_for(lambda: started == [1])
        for _ in range(20):
            raw = _raw_exchange(bridge.port, b"GET /health HTTP/1.1\r\nHost: h\r\n\r\n")
            assert raw.startswith(b"HTTP/1.1 503 Service Unavailable\r\n")
            assert b"Retry-After: 1\r\n" in raw and b"Content-Type: text/plain\r\n" in raw
            assert raw.endswith(b"\r\n\r\nService Unavailable\n")
        assert started == [1]  # ни одного рабочего потока на отклонённые соединения
    finally:
        holder.close()
    assert bridge.server.metrics.snapshot()["rejected_503"] >= 20
    assert wait_for(lambda: bridge.server.ingress.stats()["in_use"] == 0)  # слот возвращён в finally потока


def test_rw001_ingress_slot_is_released_after_each_request(track, start_bridge):
    up = track(HttpUpstream(_ok))
    bridge = start_bridge(make_config(up.port, server={"max_connections": 1}))
    for _ in range(5):
        assert _raw_exchange(bridge.port, b"GET /health HTTP/1.1\r\nHost: h\r\n\r\n").startswith(b"HTTP/1.1 200")
        assert wait_for(lambda: bridge.server.ingress.stats()["in_use"] == 0)


# ───────────────────────────── RW-002: строгий разбор запроса ─────────────────────────────


@pytest.mark.parametrize(
    "block",
    [
        b"G@T / HTTP/1.1\r\nHost: h\r\n\r\n",  # метод не token
        b"GE(T / HTTP/1.1\r\nHost: h\r\n\r\n",
        b"GET /a\rb HTTP/1.1\r\nHost: h\r\n\r\n",  # bare CR в пути
        b"GET /a\nb HTTP/1.1\r\nHost: h\r\n\r\n",  # bare LF в пути
        b"GET /a?x=1\ry=2 HTTP/1.1\r\nHost: h\r\n\r\n",  # bare CR в query
        b"GET /a\x00b HTTP/1.1\r\nHost: h\r\n\r\n",  # управляющий символ
        b"GET /a\x7fb HTTP/1.1\r\nHost: h\r\n\r\n",  # DEL
        b"GET / HTTP/1.1\r\nBad Name: v\r\n\r\n",  # имя заголовка не token
        b"GET / HTTP/1.1\r\nBad(Name: v\r\n\r\n",
        b"GET / HTTP/1.1\r\nX\x00: v\r\n\r\n",
        b"GET / HTTP/1.1\r\nX: a\rb\r\n\r\n",  # bare CR в значении
        b"GET / HTTP/1.1\r\nX: a\nb: c\r\n\r\n",  # bare LF в значении
    ],
)
def test_rw002_malformed_request_head_is_rejected(block):
    with pytest.raises(BadRequestError):
        parse_request_head(block)


def test_rw002_valid_head_is_still_accepted():
    head = parse_request_head(b"post /v1/x?a=1&b=%20 HTTP/1.1\r\nHost: h\r\nX-Custom_1.a: ok value\r\n\r\n")
    assert head.method == "POST" and head.target == "/v1/x?a=1&b=%20"
    assert head.get("x-custom_1.a") == "ok value"


def test_rw002_bare_lf_in_query_gets_400_over_the_wire(track, start_bridge):
    up = track(HttpUpstream(_ok))
    bridge = start_bridge(make_config(up.port))
    raw = _raw_exchange(bridge.port, b"GET /health?a=1\nb=2 HTTP/1.1\r\nHost: h\r\n\r\n")
    assert raw.startswith(b"HTTP/1.1 400 ")
    raw = _raw_exchange(bridge.port, b"G@T /health HTTP/1.1\r\nHost: h\r\n\r\n")
    assert raw.startswith(b"HTTP/1.1 400 ")
    raw = _raw_exchange(bridge.port, b"GET /health HTTP/1.1\r\nHost: h\r\nX: a\rb\r\n\r\n")
    assert raw.startswith(b"HTTP/1.1 400 ")
    assert up.requests == []


# ───────────────────────────── RW-003: конфликтующие ключи model ─────────────────────────────


def _feed(body: bytes, step: int | None = None) -> BodyInspector:
    inspector = BodyInspector(1 << 20)
    for i in range(0, len(body), step or len(body)):
        inspector.feed(body[i : i + (step or len(body))])
    return inspector


@pytest.mark.parametrize("step", [None, 1, 7])
def test_rw003_conflicting_model_keys_are_rejected(step):
    with pytest.raises(BadRequestError):
        _feed(b'{"model":"claude-a","messages":[],"model":"gpt-b"}', step)
    with pytest.raises(BadRequestError):
        _feed(b'{"model":"a","x":{"model":"z"},"y":[1,{"a":"b"}],"model":"b"}', step)


def test_rw003_identical_duplicates_and_nested_model_are_fine():
    assert _feed(b'{"model":"a","model":"a","z":1}', 3).model == "a"
    assert _feed(b'{"model":"a","x":{"model":"z"}}', 2).model == "a"
    assert _feed(b'{"model":"","model":""}').model is None


def test_rw003_conflicting_model_gets_400_over_the_wire(track, start_bridge):
    up = track(HttpUpstream(_ok))
    bridge = start_bridge(make_config(up.port))
    body = b'{"model":"claude-x","model":"deepseek-y"}'
    raw = _raw_exchange(
        bridge.port,
        b"POST " + PATH.encode() + b" HTTP/1.1\r\nHost: h\r\nContent-Length: %d\r\n\r\n" % len(body) + body,
    )
    assert raw.startswith(b"HTTP/1.1 400 ")
    assert up.requests == []


# ───────────────────────────── RW-004: конфиг, алиасы, README ─────────────────────────────


def _json_blocks(text: str) -> list[dict]:
    blocks = re.findall(r"```json\n(.*?)\n```", text, re.DOTALL)
    return [json.loads(b) for b in blocks if '"pools"' in b]


def test_rw004_readme_and_example_configs_pass_parse_config():
    readme_configs = _json_blocks((ROOT / "README.md").read_text(encoding="utf-8"))
    assert readme_configs, "README.md must contain a config snippet"
    for data in [*readme_configs, json.loads((ROOT / "config.example.json").read_text(encoding="utf-8"))]:
        config = parse_config(data)
        assert config.server.listen == "127.0.0.1" and config.server.port == 10830


LEGACY = {
    "server": {"host": "127.0.0.1", "port": 10830, "max_connections": 8},
    "slots": {"direct_slots": 3, "proxy_slots": 4},
    "pools": {
        "direct": {"type": "direct"},
        "route-de": {"type": "http_connect", "urls": ["http://127.0.0.1:10820"]},
    },
    "routing": {
        "default_pool": "direct",
        "geo_fallback_pool": "route-de",
        "rules": [{"prefix": "claude-", "pool": "route-de"}, {"prefix": ["a-", "b-"], "pool": "route-de"}],
    },
}


def test_rw004_legacy_aliases_and_nested_sections():
    config = parse_config(LEGACY)
    assert config.server.listen == "127.0.0.1" and config.server.max_connections == 8
    assert (config.server.direct_slots, config.server.proxy_slots) == (3, 4)
    assert config.pools["route-de"].proxies == ("http://127.0.0.1:10820",)
    assert config.geo_fallback_pool == "route-de" and config.default_pool == "direct"
    assert [r.match_prefix for r in config.rules] == [("claude-",), ("a-", "b-")]


def test_rw004_flat_canonical_keys_win_over_aliases():
    data = json.loads(json.dumps(LEGACY))
    data["server"]["listen"] = "127.0.0.1"
    data["server"]["host"] = "10.0.0.1"
    assert parse_config(data).server.listen == "127.0.0.1"


def test_rw004_aliases_do_not_mask_validation_errors():
    data = json.loads(json.dumps(LEGACY))
    data["server"]["host"] = "0.0.0.0"
    with pytest.raises(ConfigError, match="listen"):
        parse_config(data)
    data = json.loads(json.dumps(LEGACY))
    data["pools"]["route-de"]["urls"] = []
    with pytest.raises(ConfigError, match="proxies"):
        parse_config(data)


def test_rw004_service_name_is_model_georouter_with_legacy_alias(track, start_bridge):
    assert SERVICE_NAME == "model-georouter" and "universal-ai-bridge" in SERVICE_NAME_ALIASES
    up = track(HttpUpstream(_ok))
    bridge = start_bridge(make_config(up.port))
    health = bridge.health()
    assert health["service"] == "model-georouter" and health["service_aliases"] == list(SERVICE_NAME_ALIASES)


# ───────────────────────────── RW-005: бюджет DNS + connect ─────────────────────────────


def test_rw005_slow_dns_respects_total_egress_deadline(monkeypatch):
    release = threading.Event()

    def slow_getaddrinfo(*args, **kwargs):
        release.wait(5)
        return []

    monkeypatch.setattr(socket, "getaddrinfo", slow_getaddrinfo)
    settings = EgressSettings(connect_timeout=5.0, total_deadline=0.4, retries=2, penalty_seconds=1)
    started = time.monotonic()
    try:
        with pytest.raises(EgressTimeoutError):
            ProxyPoolManager().connect(PoolConfig("direct", "direct"), "slow-dns.invalid", 80, settings)
        assert time.monotonic() - started < 2.0  # без бюджета DNS: ≥ 5 с на каждую попытку
    finally:
        release.set()


def test_rw005_resolve_is_bounded_by_deadline(monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: release.wait(5))
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            _resolve("slow-dns.invalid", 80, time.monotonic() + 0.3)
        assert time.monotonic() - started < 2.0
    finally:
        release.set()


def test_rw005_ip_literal_connects_without_resolver_thread():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        settings = EgressSettings(connect_timeout=2.0, total_deadline=3.0, retries=0, penalty_seconds=1)
        sock = ProxyPoolManager().connect(
            PoolConfig("direct", "direct"), "127.0.0.1", server.getsockname()[1], settings
        )
        sock.close()


# ───────────────────────────── RW-006: hot-reload max_connections ─────────────────────────────


def _write_config(path: Path, max_connections: int) -> None:
    data = make_config(1, server={"max_connections": max_connections})
    path.write_text(json.dumps(data), encoding="utf-8")


def test_rw006_max_connections_hot_reload_resizes_ingress(tmp_path):
    cfg = tmp_path / "config.json"
    _write_config(cfg, 2)
    server = BridgeServer(ConfigManager(cfg, poll_interval=0.0))
    try:
        assert server.ingress.stats()["limit"] == 2
        _write_config(cfg, 50)
        server._check_reload()
        assert server.ingress.stats()["limit"] == 50
        _write_config(cfg, 1)
        server._check_reload()
        assert server.ingress.stats()["limit"] == 1
    finally:
        server.server_close()


def test_rw006_lowered_limit_applies_to_new_connections_without_restart(tmp_path):
    up = HttpUpstream(_ok)
    cfg = tmp_path / "config.json"
    data = make_config(up.port, server={"max_connections": 4})
    cfg.write_text(json.dumps(data), encoding="utf-8")
    server = BridgeServer(ConfigManager(cfg, poll_interval=0.0))
    server.start_background()
    port = server.server_address[1]
    holder = socket.create_connection(("127.0.0.1", port), timeout=5)
    try:
        holder.sendall(b"POST " + PATH.encode() + b" HTTP/1.1\r\nHost: h\r\n")
        assert _raw_exchange(port, b"GET /health HTTP/1.1\r\nHost: h\r\n\r\n").startswith(b"HTTP/1.1 200")
        data["server"]["max_connections"] = 1
        time.sleep(0.01)
        cfg.write_text(json.dumps(data) + " ", encoding="utf-8")
        assert wait_for(
            lambda: _raw_exchange(port, b"GET /health HTTP/1.1\r\nHost: h\r\n\r\n").startswith(b"HTTP/1.1 503")
        )
        assert server.ingress.stats()["limit"] == 1
    finally:
        holder.close()
        server.stop()
        up.close()
