"""Сквозные тесты: настоящий BridgeServer + фейковые upstream/прокси на реальных сокетах."""

from __future__ import annotations

import json
import logging
import os
import threading
import time

import pytest

from tests.helpers import (
    FakeConnectProxy,
    FakeSocks5Proxy,
    HttpUpstream,
    chunk,
    free_port,
    http_request,
    make_config,
    open_request,
    parse_response,
    read_all,
    read_until,
    respond,
    send_sse_head,
    wait_for,
)
from universal_ai_bridge.bridge_server import BridgeServer
from universal_ai_bridge.config import ConfigManager

PATH = "/v1/chat/completions"
UPSTREAM_PATH = "/inference/openai/v1/chat/completions"
REGION_BODY = b'{"error":{"type":"RegionError","message":"not available in your region"}}'


def body_for(model: str, extra: str = "") -> bytes:
    return json.dumps({"model": model, "messages": [{"role": "user", "content": "hi" + extra}], "stream": True}).encode()


def ok_responder(label: bytes):
    def responder(conn, req):
        respond(conn, 200, label, {"Content-Type": "text/plain", "X-Request-Id": "abc"})

    return responder


def proxy_pools(proxy_port: int, name: str = "route-de") -> dict:
    return {name: {"type": "http_connect", "proxies": [f"http://127.0.0.1:{proxy_port}"]}}


# ───────────────────────────── стриминг ─────────────────────────────


def test_sse_stream_is_relayed_incrementally_not_buffered(track, start_bridge):
    release = threading.Event()

    def streaming(conn, req):
        send_sse_head(conn)
        conn.sendall(chunk(b"data: one\n\n"))
        release.wait(5)
        conn.sendall(chunk(b"data: two\n\n") + chunk(b"data: [DONE]\n\n") + b"0\r\n\r\n")

    upstream = track(HttpUpstream(streaming))
    bridge = start_bridge(make_config(upstream.port))
    sock = open_request(bridge.port, "POST", PATH, body_for("m"), {"Content-Type": "application/json"})
    first = read_until(sock, b"data: one")  # приходит, пока upstream ещё держит поток
    assert not release.is_set() and b"text/event-stream" in first and b"Transfer-Encoding: chunked" in first
    release.set()
    full = first + read_all(sock)
    response = parse_response(full)
    assert response.status == 200 and response.terminated
    assert response.body == b"data: one\n\ndata: two\n\ndata: [DONE]\n\n"
    assert response.headers["connection"] == "close"
    assert upstream.requests[0].path == UPSTREAM_PATH
    assert wait_for(bridge.slots_idle)


def test_content_length_response_and_headers_pass_through(track, start_bridge):
    upstream = track(HttpUpstream(ok_responder(b"hello world")))
    bridge = start_bridge(make_config(upstream.port))
    response = http_request(bridge.port, "POST", PATH, body_for("m"))
    assert response.status == 200 and response.body == b"hello world"
    assert response.headers["x-request-id"] == "abc" and response.headers["content-length"] == "11"


def test_mid_stream_upstream_failure_closes_without_done(track, start_bridge):
    def dies(conn, req):
        send_sse_head(conn)
        conn.sendall(chunk(b"data: partial\n\n"))
        time.sleep(0.1)  # затем соединение рвётся без терминирующего chunk

    upstream = track(HttpUpstream(dies))
    bridge = start_bridge(make_config(upstream.port))
    sock = open_request(bridge.port, "POST", PATH, body_for("m"))
    raw = read_all(sock)
    response = parse_response(raw)
    assert response.status == 200
    assert response.body == b"data: partial\n\n"
    assert not response.terminated and b"[DONE]" not in raw
    assert wait_for(lambda: bridge.server.metrics.snapshot().get("aborted_streams", 0) >= 1)
    assert wait_for(bridge.slots_idle)


def test_client_disconnect_closes_upstream_and_frees_slot(track, start_bridge):
    upstream_closed = threading.Event()

    def endless(conn, req):
        send_sse_head(conn)
        try:
            while True:
                conn.sendall(chunk(b"data: tick\n\n"))
                time.sleep(0.05)
        except OSError:
            upstream_closed.set()

    upstream = track(HttpUpstream(endless))
    bridge = start_bridge(make_config(upstream.port))
    sock = open_request(bridge.port, "POST", PATH, body_for("m"))
    read_until(sock, b"data: tick")
    assert bridge.health()["slots"]["direct"]["in_use"] == 1
    sock.close()
    assert upstream_closed.wait(5), "upstream-соединение должно закрыться после ухода клиента"
    assert wait_for(bridge.slots_idle)
    assert wait_for(lambda: bridge.server.metrics.snapshot().get("relay_client_cancelled", 0) == 1)


# ───────────────────────────── маршрутизация ─────────────────────────────


def test_rules_route_models_to_direct_or_proxy_pools(track, start_bridge):
    direct_up = track(HttpUpstream(ok_responder(b"direct")))
    proxied_up = track(HttpUpstream(ok_responder(b"via-proxy")))
    proxy = track(FakeConnectProxy(target=("127.0.0.1", proxied_up.port)))
    bridge = start_bridge(
        make_config(
            direct_up.port,
            pools=proxy_pools(proxy.port),
            rules=[{"match_prefix": ["claude-"], "pool": "route-de"}],
        )
    )
    claude = http_request(bridge.port, "POST", PATH, body_for("claude-3-opus"))
    assert claude.body == b"via-proxy" and len(proxy.connects) == 1
    assert proxy.connects[0]["authority"] == f"127.0.0.1:{direct_up.port}"
    gpt = http_request(bridge.port, "POST", PATH, body_for("gpt-5"))
    assert gpt.body == b"direct" and len(proxy.connects) == 1
    no_model = http_request(bridge.port, "POST", PATH, b'{"messages":[]}')
    assert no_model.body == b"direct"  # модель не найдена — default_pool
    not_json = http_request(bridge.port, "POST", PATH, b"plain text")
    assert not_json.body == b"direct"
    assert proxied_up.requests[0].body == body_for("claude-3-opus")  # тело дошло байт в байт


def test_socks5_pool_uses_remote_dns_end_to_end(track, start_bridge):
    up = track(HttpUpstream(ok_responder(b"via-socks")))
    socks = track(FakeSocks5Proxy(target=("127.0.0.1", up.port)))
    config = make_config(
        1,
        pools={"route-hy2": {"type": "socks5", "remote_dns": True, "proxies": [f"socks5h://127.0.0.1:{socks.port}"]}},
        rules=[{"match_regex": "^gemini", "pool": "route-hy2"}],
        upstreams={"opencode-ai": {"host": "model-gateway.invalid", "port": 80, "use_tls": False}},
    )
    bridge = start_bridge(config)
    response = http_request(bridge.port, "POST", PATH, body_for("gemini-2.5"))
    assert response.body == b"via-socks"
    assert (socks.requests[0]["atyp"], socks.requests[0]["host"]) == (0x03, "model-gateway.invalid")
    assert up.requests[0].headers["host"] == "model-gateway.invalid"


def test_chunked_request_body_is_decoded_and_model_extracted(track, start_bridge):
    direct_up = track(HttpUpstream(ok_responder(b"direct")))
    proxied_up = track(HttpUpstream(ok_responder(b"via-proxy")))
    proxy = track(FakeConnectProxy(target=("127.0.0.1", proxied_up.port)))
    bridge = start_bridge(
        make_config(direct_up.port, pools=proxy_pools(proxy.port), rules=[{"match_prefix": ["claude-"], "pool": "route-de"}])
    )
    payload = body_for("claude-3")
    wire = chunk(payload[:10]) + chunk(payload[10:]) + b"0\r\n\r\n"
    response = http_request(bridge.port, "POST", PATH, wire, {"Transfer-Encoding": "chunked"})
    assert response.body == b"via-proxy"
    seen = proxied_up.requests[0]
    assert seen.body == payload and seen.headers["content-length"] == str(len(payload))
    assert "transfer-encoding" not in seen.headers


def test_google_style_model_in_path_is_used_for_routing(track, start_bridge):
    direct_up = track(HttpUpstream(ok_responder(b"direct")))
    proxied_up = track(HttpUpstream(ok_responder(b"via-proxy")))
    proxy = track(FakeConnectProxy(target=("127.0.0.1", proxied_up.port)))
    bridge = start_bridge(
        make_config(direct_up.port, pools=proxy_pools(proxy.port), rules=[{"match_prefix": ["gemini-"], "pool": "route-de"}])
    )
    response = http_request(bridge.port, "POST", "/v1/v1beta/models/gemini-2.5-pro:generateContent", b"{}")
    assert response.body == b"via-proxy"
    assert proxied_up.requests[0].path == "/inference/google/v1beta/models/gemini-2.5-pro:generateContent"


# ───────────────────────────── replay при 403 RegionError ─────────────────────────────


def test_region_403_is_replayed_once_via_fallback_and_cached(track, start_bridge):
    def region_blocked(conn, req):
        respond(conn, 403, REGION_BODY, {"Content-Type": "application/json"})

    direct_up = track(HttpUpstream(region_blocked))
    proxied_up = track(HttpUpstream(ok_responder(b"served-via-proxy")))
    proxy = track(FakeConnectProxy(target=("127.0.0.1", proxied_up.port)))
    bridge = start_bridge(make_config(direct_up.port, pools=proxy_pools(proxy.port), geo_fallback_pool="route-de"))
    payload = body_for("gpt-5", "x" * 1000)

    first = http_request(bridge.port, "POST", PATH, payload)
    assert first.status == 200 and first.body == b"served-via-proxy"  # клиент 403 не видит
    assert len(direct_up.requests) == 1 and len(proxied_up.requests) == 1
    assert proxied_up.requests[0].body == payload  # replay повторяет тело байт в байт
    assert bridge.server.geo_cache.is_blocked("opencode-ai", "gpt-5")
    assert bridge.server.metrics.snapshot()["replays"] == 1

    second = http_request(bridge.port, "POST", PATH, payload)  # Geo-Cache: сразу через прокси, direct не трогаем
    assert second.status == 200 and second.body == b"served-via-proxy"
    assert len(direct_up.requests) == 1 and len(proxied_up.requests) == 2
    assert bridge.server.metrics.snapshot()["replays"] == 1
    assert bridge.server.metrics.snapshot()["geo_cache_hits"] == 1

    other = http_request(bridge.port, "POST", PATH, body_for("llama-3"))  # другая модель — снова пробует direct
    assert other.status == 200 and len(direct_up.requests) == 2

    assert http_request(bridge.port, "POST", "/cache/flush").json == {"flushed": 2}
    assert wait_for(bridge.slots_idle)


def test_replay_happens_exactly_once_even_if_fallback_also_says_region_error(track, start_bridge):
    def region_blocked(conn, req):
        respond(conn, 403, REGION_BODY)

    direct_up = track(HttpUpstream(region_blocked))
    proxied_up = track(HttpUpstream(region_blocked))
    proxy = track(FakeConnectProxy(target=("127.0.0.1", proxied_up.port)))
    bridge = start_bridge(make_config(direct_up.port, pools=proxy_pools(proxy.port), geo_fallback_pool="route-de"))
    response = http_request(bridge.port, "POST", PATH, body_for("gpt-5"))
    assert response.status == 403 and response.body == REGION_BODY  # ответ fallback-а отдаётся как есть
    assert len(direct_up.requests) == 1 and len(proxied_up.requests) == 1  # третьего запроса нет


def test_plain_auth_403_is_not_replayed_and_passes_through(track, start_bridge):
    def auth_denied(conn, req):
        respond(conn, 403, b'{"error":{"message":"Invalid API key"}}', {"Content-Type": "application/json"})

    direct_up = track(HttpUpstream(auth_denied))
    proxied_up = track(HttpUpstream(ok_responder(b"never")))
    proxy = track(FakeConnectProxy(target=("127.0.0.1", proxied_up.port)))
    bridge = start_bridge(make_config(direct_up.port, pools=proxy_pools(proxy.port), geo_fallback_pool="route-de"))
    response = http_request(bridge.port, "POST", PATH, body_for("gpt-5"))
    assert response.status == 403 and b"Invalid API key" in response.body
    assert response.headers["content-type"] == "application/json"
    assert proxy.connects == [] and len(direct_up.requests) == 1
    assert len(bridge.server.geo_cache) == 0
    assert bridge.server.metrics.snapshot()["classified_403_plain"] == 1
    assert wait_for(bridge.slots_idle)


def test_waf_403_with_chunked_html_body_is_relayed_untouched(track, start_bridge):
    html = b"<html><title>Attention Required! | Cloudflare</title></html>"

    def waf(conn, req):
        conn.sendall(
            b"HTTP/1.1 403 Forbidden\r\nContent-Type: text/html\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
            + chunk(html[:20])
            + chunk(html[20:])
            + b"0\r\n\r\n"
        )

    direct_up = track(HttpUpstream(waf))
    proxy = track(FakeConnectProxy())
    bridge = start_bridge(make_config(direct_up.port, pools=proxy_pools(proxy.port), geo_fallback_pool="route-de"))
    response = http_request(bridge.port, "POST", PATH, body_for("m"))
    assert response.status == 403 and response.body == html and response.terminated
    assert proxy.connects == []


def test_region_403_with_chunked_body_is_detected(track, start_bridge):
    def region_chunked(conn, req):
        conn.sendall(
            b"HTTP/1.1 403 Forbidden\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
            + chunk(REGION_BODY[:15])
            + chunk(REGION_BODY[15:])
            + b"0\r\n\r\n"
        )

    direct_up = track(HttpUpstream(region_chunked))
    proxied_up = track(HttpUpstream(ok_responder(b"proxy-ok")))
    proxy = track(FakeConnectProxy(target=("127.0.0.1", proxied_up.port)))
    bridge = start_bridge(make_config(direct_up.port, pools=proxy_pools(proxy.port), geo_fallback_pool="route-de"))
    assert http_request(bridge.port, "POST", PATH, body_for("m")).body == b"proxy-ok"


def test_proxy_routed_403_is_never_replayed(track, start_bridge):
    def region_blocked(conn, req):
        respond(conn, 403, REGION_BODY)

    up = track(HttpUpstream(region_blocked))
    proxy = track(FakeConnectProxy(target=("127.0.0.1", up.port)))
    bridge = start_bridge(
        make_config(
            up.port,
            pools=proxy_pools(proxy.port),
            rules=[{"match_prefix": ["claude-"], "pool": "route-de"}],
            geo_fallback_pool="route-de",
        )
    )
    response = http_request(bridge.port, "POST", PATH, body_for("claude-3"))
    assert response.status == 403 and len(up.requests) == 1


# ───────────────────────────── сбои прокси ─────────────────────────────


def test_dead_proxy_gives_502_json_then_fast_fail_and_direct_is_unaffected(track, start_bridge):
    direct_up = track(HttpUpstream(ok_responder(b"direct-ok")))
    dead = f"http://127.0.0.1:{free_port()}"
    bridge = start_bridge(
        make_config(
            direct_up.port,
            pools={"route-de": {"type": "http_connect", "proxies": [dead]}},
            rules=[{"match_prefix": ["claude-"], "pool": "route-de"}],
        )
    )
    first = http_request(bridge.port, "POST", PATH, body_for("claude-3"))
    assert first.status == 502 and first.status != 403
    assert first.json["error"]["type"] == "proxy_connect_failed" and first.json["error"]["retryable"] is True

    started = time.monotonic()
    second = http_request(bridge.port, "POST", PATH, body_for("claude-3"))
    assert second.status == 502 and second.json["error"]["type"] == "proxy_unavailable"
    assert second.json["error"]["retryable"] is True
    assert time.monotonic() - started < 1.0  # circuit breaker: без ожидания сети

    health = bridge.health()
    assert health["pools"]["route-de"]["proxies_penalized"] == 1
    assert health["slots"]["proxy"]["in_use"] == 0  # слот при fast-fail не удерживался
    assert http_request(bridge.port, "POST", PATH, body_for("gpt-5")).body == b"direct-ok"


def test_proxy_that_hangs_on_connect_gives_504_json(track, start_bridge):
    from tests.helpers import FakeServer

    hang = track(FakeServer(lambda conn: time.sleep(5)))
    direct_up = track(HttpUpstream(ok_responder(b"ok")))
    config = make_config(
        direct_up.port,
        pools={"route-de": {"type": "http_connect", "proxies": [f"http://127.0.0.1:{hang.port}"]}},
        rules=[{"match_prefix": ["claude-"], "pool": "route-de"}],
        server={"connect_timeout": 0.3, "pre_send_retries": 0},
    )
    bridge = start_bridge(config)
    response = http_request(bridge.port, "POST", PATH, body_for("claude-3"))
    assert response.status == 504 and response.json["error"]["type"] == "proxy_gateway_timeout"
    assert response.json["error"]["retryable"] is True
    assert wait_for(bridge.slots_idle)


def test_region_403_with_dead_fallback_returns_502_not_403(track, start_bridge):
    def region_blocked(conn, req):
        respond(conn, 403, REGION_BODY)

    direct_up = track(HttpUpstream(region_blocked))
    dead = f"http://127.0.0.1:{free_port()}"
    bridge = start_bridge(
        make_config(direct_up.port, pools={"route-de": {"type": "http_connect", "proxies": [dead]}}, geo_fallback_pool="route-de")
    )
    response = http_request(bridge.port, "POST", PATH, body_for("gpt-5"))
    assert response.status == 502 and response.json["error"]["retryable"] is True
    assert bridge.server.geo_cache.is_blocked("opencode-ai", "gpt-5")  # блокировка всё равно запомнена
    assert wait_for(bridge.slots_idle)


def test_upstream_without_response_headers_gives_504(track, start_bridge):
    from tests.helpers import FakeServer

    silent = track(FakeServer(lambda conn: time.sleep(5)))
    bridge = start_bridge(make_config(silent.port, server={"headers_timeout": 0.4}))
    started = time.monotonic()
    response = http_request(bridge.port, "POST", PATH, body_for("m"))
    assert response.status == 504 and response.json["error"]["type"] == "upstream_timeout"
    assert time.monotonic() - started < 3
    assert wait_for(bridge.slots_idle)


def test_unreachable_direct_upstream_gives_502(start_bridge):
    bridge = start_bridge(make_config(free_port()))
    response = http_request(bridge.port, "POST", PATH, body_for("m"))
    assert response.status == 502 and response.json["error"]["retryable"] is True


# ───────────────────────────── слоты ─────────────────────────────


def test_split_slots_slow_proxy_cannot_block_direct_models(track, start_bridge):
    release = threading.Event()

    def stuck(conn, req):
        release.wait(10)
        respond(conn, 200, b"late")

    direct_up = track(HttpUpstream(ok_responder(b"direct-ok")))
    stuck_up = track(HttpUpstream(stuck))
    proxy = track(FakeConnectProxy(target=("127.0.0.1", stuck_up.port)))
    bridge = start_bridge(
        make_config(
            direct_up.port,
            pools=proxy_pools(proxy.port),
            rules=[{"match_prefix": ["claude-"], "pool": "route-de"}],
            server={"direct_slots": 2, "proxy_slots": 1, "headers_timeout": 10.0},
        )
    )
    results = []
    holder = threading.Thread(
        target=lambda: results.append(http_request(bridge.port, "POST", PATH, body_for("claude-3"))), daemon=True
    )
    holder.start()
    assert wait_for(lambda: bridge.health()["slots"]["proxy"]["in_use"] == 1)

    overloaded = http_request(bridge.port, "POST", PATH, body_for("claude-3"))
    assert overloaded.status == 503
    assert overloaded.json["error"]["type"] == "bridge_overloaded" and overloaded.json["error"]["retryable"] is True
    assert overloaded.headers["retry-after"] == "5"

    started = time.monotonic()
    for _ in range(4):  # direct-слоты свободны, пока proxy-слот занят зависшим запросом
        assert http_request(bridge.port, "POST", PATH, body_for("gpt-5")).body == b"direct-ok"
    assert time.monotonic() - started < 2
    assert bridge.health()["slots"]["direct"]["limit"] == 2

    release.set()
    holder.join(10)
    assert results[0].status == 200 and results[0].body == b"late"
    assert wait_for(bridge.slots_idle)
    assert bridge.server.metrics.snapshot()["rejected_503"] == 1


def test_direct_slots_are_enforced_independently(track, start_bridge):
    release = threading.Event()

    def stuck(conn, req):
        release.wait(10)
        respond(conn, 200, b"late")

    up = track(HttpUpstream(stuck))
    bridge = start_bridge(make_config(up.port, server={"direct_slots": 1, "headers_timeout": 10.0}))
    holder = threading.Thread(target=lambda: http_request(bridge.port, "POST", PATH, body_for("m")), daemon=True)
    holder.start()
    assert wait_for(lambda: bridge.health()["slots"]["direct"]["in_use"] == 1)
    assert http_request(bridge.port, "POST", PATH, body_for("m")).status == 503
    release.set()
    holder.join(10)
    assert wait_for(bridge.slots_idle)


# ───────────────────────────── входные ограничения ─────────────────────────────


def test_oversized_content_length_is_rejected_with_413_before_forwarding(track, start_bridge):
    up = track(HttpUpstream(ok_responder(b"x")))
    bridge = start_bridge(make_config(up.port, server={"body_buffer_max_bytes": 1024}))
    response = http_request(bridge.port, "POST", PATH, b'{"model":"m","p":"' + b"z" * 5000 + b'"}')
    assert response.status == 413
    assert response.json["error"]["type"] == "payload_too_large" and response.json["error"]["limit_bytes"] == 1024
    assert up.requests == [] and wait_for(bridge.slots_idle)
    exact = http_request(bridge.port, "POST", PATH, b'{"model":"' + b"m" * 1000 + b'"}')
    assert exact.status == 200  # в пределах лимита


def test_oversized_chunked_body_is_rejected_with_413(track, start_bridge):
    up = track(HttpUpstream(ok_responder(b"x")))
    bridge = start_bridge(make_config(up.port, server={"body_buffer_max_bytes": 1024}))
    wire = b"".join(chunk(b"a" * 500) for _ in range(6)) + b"0\r\n\r\n"
    response = http_request(bridge.port, "POST", PATH, wire, {"Transfer-Encoding": "chunked"})
    assert response.status == 413 and up.requests == []


def test_malformed_requests_get_json_400(track, start_bridge):
    up = track(HttpUpstream(ok_responder(b"x")))
    bridge = start_bridge(make_config(up.port))
    both = http_request(bridge.port, "POST", PATH, b"{}", {"Transfer-Encoding": "chunked", "Content-Length": "2"})
    assert both.status == 400 and both.json["error"]["type"] == "bad_request"
    bad_length = http_request(bridge.port, "POST", PATH, b"{}", {"Content-Length": "abc"})
    assert bad_length.status == 400
    import socket

    with socket.create_connection(("127.0.0.1", bridge.port), timeout=5) as sock:
        sock.sendall(b"NONSENSE\r\n\r\n")
        assert parse_response(read_all(sock)).status == 400
    assert up.requests == []


def test_truncated_request_body_is_rejected(track, start_bridge):
    import socket

    up = track(HttpUpstream(ok_responder(b"x")))
    bridge = start_bridge(make_config(up.port))
    with socket.create_connection(("127.0.0.1", bridge.port), timeout=5) as sock:
        sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 100\r\n\r\n{\"model\":\"m\"")
        sock.shutdown(socket.SHUT_WR)
        assert parse_response(read_all(sock)).status == 400
    assert up.requests == [] and wait_for(bridge.slots_idle)


def test_slow_request_body_times_out_with_408_without_holding_slot(track, start_bridge):
    import socket

    up = track(HttpUpstream(ok_responder(b"x")))
    bridge = start_bridge(make_config(up.port, server={"body_timeout": 0.4}))
    with socket.create_connection(("127.0.0.1", bridge.port), timeout=5) as sock:
        sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 100\r\n\r\n{")
        response = parse_response(read_all(sock, 5))
    assert response.status == 408 and response.json["error"]["type"] == "request_timeout"
    assert up.requests == [] and bridge.health()["slots"]["direct"]["in_use"] == 0


def test_expect_100_continue_is_honoured(track, start_bridge):
    import socket

    up = track(HttpUpstream(ok_responder(b"ok")))
    bridge = start_bridge(make_config(up.port))
    body = body_for("m")
    with socket.create_connection(("127.0.0.1", bridge.port), timeout=5) as sock:
        sock.sendall(
            f"POST {PATH} HTTP/1.1\r\nHost: x\r\nContent-Length: {len(body)}\r\nExpect: 100-continue\r\n\r\n".encode()
        )
        assert read_until(sock, b"100 Continue")
        sock.sendall(body)
        assert parse_response(read_all(sock)).body == b"ok"
    assert up.requests[0].body == body and "expect" not in up.requests[0].headers


def test_secrets_do_not_reach_the_log(track, start_bridge, caplog):
    up = track(HttpUpstream(ok_responder(b"ok")))
    bridge = start_bridge(make_config(up.port))
    with caplog.at_level(logging.INFO, logger="universal_ai_bridge.server"):
        http_request(
            bridge.port,
            "POST",
            PATH + "?key=URLSECRET123",
            body_for("m"),
            {"Authorization": "Bearer sk-headersecret999"},
        )
        assert wait_for(lambda: "request method=POST" in caplog.text)
    assert "headersecret999" not in caplog.text and "URLSECRET123" not in caplog.text


# ───────────────────────────── hot-reload в работающем сервере ─────────────────────────────


def test_hot_reload_changes_routing_without_restart_and_survives_bad_config(track, tmp_path):
    direct_up = track(HttpUpstream(ok_responder(b"direct")))
    proxied_up = track(HttpUpstream(ok_responder(b"via-proxy")))
    proxy = track(FakeConnectProxy(target=("127.0.0.1", proxied_up.port)))
    path = tmp_path / "config.json"
    config = make_config(direct_up.port, pools=proxy_pools(proxy.port), rules=[])

    def write(data, stamp):
        path.write_text(json.dumps(data))
        os.utime(path, (stamp, stamp))

    write(config, 1_700_000_000)
    server = BridgeServer(ConfigManager(path, poll_interval=0))
    server.start_background()
    track(server)
    port = server.server_address[1]
    try:
        assert http_request(port, "POST", PATH, body_for("claude-3")).body == b"direct"

        config["rules"] = [{"match_prefix": ["claude-"], "pool": "route-de"}]
        write(config, 1_700_000_100)
        assert http_request(port, "POST", PATH, body_for("claude-3")).body == b"via-proxy"

        path.write_text("{definitely not json")
        os.utime(path, (1_700_000_200, 1_700_000_200))
        assert http_request(port, "POST", PATH, body_for("claude-3")).body == b"via-proxy"  # прежний снимок
        assert http_request(port, "GET", "/health").json["config_error"] is not None

        config["server"]["direct_slots"] = 3
        write(config, 1_700_000_300)
        health = http_request(port, "GET", "/health").json
        assert health["config_error"] is None
    finally:
        pass


# ───────────────────────────── Rework Cycle 1 ─────────────────────────────


def test_slowloris_body_trickle_hits_single_body_deadline(track, start_bridge):  # RW-001
    import select
    import socket

    up = track(HttpUpstream(ok_responder(b"x")))
    bridge = start_bridge(make_config(up.port, server={"body_timeout": 1.0}))
    started = time.monotonic()
    with socket.create_connection(("127.0.0.1", bridge.port), timeout=10) as sock:
        sock.sendall(b"POST " + PATH.encode() + b" HTTP/1.1\r\nHost: h\r\nContent-Length: 1000\r\n\r\n")
        try:
            while time.monotonic() - started < 6:
                sock.sendall(b"{")  # каждый recv сервера получает данные раньше 1 с — прежний таймаут сбрасывался
                if select.select([sock], [], [], 0.3)[0]:
                    break
        except OSError:
            pass
        response = parse_response(read_all(sock))
    assert response.status == 408 and response.json["error"]["type"] == "request_timeout"
    assert time.monotonic() - started < 3
    assert up.requests == [] and wait_for(bridge.slots_idle)


def test_ingress_semaphore_rejects_burst_with_503_and_recovers(track, start_bridge):  # RW-008
    import socket

    up = track(HttpUpstream(ok_responder(b"x")))
    bridge = start_bridge(make_config(up.port, server={"max_connections": 1, "ingress_wait_timeout": 0.3}))
    holder = socket.create_connection(("127.0.0.1", bridge.port), timeout=5)
    try:
        holder.sendall(b"POST " + PATH.encode() + b" HTTP/1.1\r\nHost: h\r\n")  # голова не завершена: поток занят
        time.sleep(0.2)
        rejected = http_request(bridge.port, "GET", "/health")
        assert rejected.status == 503 and rejected.body == b"Service Unavailable\n"  # RW-001 (Cycle 2): inline 503 в accept
        assert rejected.headers["retry-after"] == "1"
    finally:
        holder.close()
    assert wait_for(lambda: http_request(bridge.port, "GET", "/health").status == 200)
    assert bridge.server.metrics.snapshot()["rejected_503"] >= 1


def test_ingress_limit_is_configurable_and_validated():
    from universal_ai_bridge.config import ConfigError, parse_config

    base = {"server": {"listen": "127.0.0.1", "port": 1}, "pools": {}, "rules": []}
    server = parse_config(base).server
    assert server.max_connections == 256 and server.ingress_wait_timeout == 5.0
    for key, value in (("max_connections", 0), ("ingress_wait_timeout", 0)):
        bad = {**base, "server": {**base["server"], key: value}}
        try:
            parse_config(bad)
        except ConfigError as exc:
            assert key in str(exc)
        else:
            raise AssertionError(f"{key}={value} was accepted")


def test_truncated_403_body_without_signature_is_counted(track, start_bridge):  # RW-009
    long_plain = b"<html>" + b"x" * 40000 + b"</html>"
    direct_up = track(HttpUpstream(lambda conn, req: respond(conn, 403, long_plain)))
    proxied_up = track(HttpUpstream(ok_responder(b"proxy-ok")))
    proxy = track(FakeConnectProxy(target=("127.0.0.1", proxied_up.port)))
    bridge = start_bridge(make_config(direct_up.port, pools=proxy_pools(proxy.port), geo_fallback_pool="route-de"))
    response = http_request(bridge.port, "POST", PATH, body_for("m"))
    assert response.status == 403 and response.body == long_plain  # WAF-403 по-прежнему отдаётся как есть
    counters = http_request(bridge.port, "GET", "/metrics").json["counters"]
    assert counters["classified_403_truncated"] == 1 and counters["classified_403_plain"] == 1

    short = track(HttpUpstream(lambda conn, req: respond(conn, 403, b"<html>nope</html>")))
    bridge2 = start_bridge(make_config(short.port, pools=proxy_pools(proxy.port), geo_fallback_pool="route-de"))
    assert http_request(bridge2.port, "POST", PATH, body_for("m")).status == 403
    assert "classified_403_truncated" not in http_request(bridge2.port, "GET", "/metrics").json["counters"]


@pytest.mark.parametrize(
    "path, headers",
    [
        (PATH, {"Transfer-Encoding": "chunked", "Content-Length": "2"}),
        ("/health", {"Transfer-Encoding": "chunked", "Content-Length": "2"}),
    ],
)
def test_te_plus_cl_is_rejected_with_400_everywhere(track, start_bridge, path, headers):  # RW-010
    up = track(HttpUpstream(ok_responder(b"x")))
    bridge = start_bridge(make_config(up.port))
    response = http_request(bridge.port, "POST", path, b"{}", headers)
    assert response.status == 400 and up.requests == []


def test_conflicting_duplicate_content_length_is_rejected_with_400(track, start_bridge):  # RW-010
    import socket

    up = track(HttpUpstream(ok_responder(b"x")))
    bridge = start_bridge(make_config(up.port))
    for path in (PATH, "/health"):
        with socket.create_connection(("127.0.0.1", bridge.port), timeout=5) as raw:
            raw.sendall(
                f"POST {path} HTTP/1.1\r\nHost: h\r\nContent-Length: 2\r\nContent-Length: 5\r\n\r\n{{}}".encode()
            )
            assert parse_response(read_all(raw)).status == 400
    assert up.requests == []


def test_chunk_without_trailing_crlf_gets_400(track, start_bridge):  # RW-011
    up = track(HttpUpstream(ok_responder(b"x")))
    bridge = start_bridge(make_config(up.port))
    for wire in (b'5\r\n{"a":XX0\r\n\r\n', b'5\r\n{"a":\n\n0\r\n\r\n', b'5\r\n{"a":X\n0\r\n\r\n'):
        response = http_request(bridge.port, "POST", PATH, wire, {"Transfer-Encoding": "chunked"})
        assert response.status == 400 and response.json["error"]["type"] == "bad_request"
    assert up.requests == []
