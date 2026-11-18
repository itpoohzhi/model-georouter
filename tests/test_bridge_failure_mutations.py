"""Поведенческие тесты ветвей BridgeServer, выбранных мутационным QA."""

from __future__ import annotations

import json
import threading
import time
from typing import cast

from model_georouter.adapters import AdapterRoute
from model_georouter.bridge_server import (
    DIAGNOSTIC_BODY_LIMIT,
    BridgeServer,
    _breaker,
    _ensure_anthropic_beta,
    _Handler,
)
from model_georouter.config import ConfigManager
from model_georouter.http_wire import parse_request_head
from model_georouter.model_router import BodyInspector
from model_georouter.proxy_pool import ProxyPoolManager
from model_georouter.retry_budget import RetryBudgetSpec
from model_georouter.transport_failover import BreakerState
from tests.helpers import (
    FakeConnectProxy,
    HttpUpstream,
    chunk,
    free_port,
    http_request,
    make_config,
    open_request,
    read_all,
    read_until,
    respond,
    send_sse_head,
)
from tests.test_retry_budget_and_routes import (
    BODY,
    PATH,
    REGION_BODY,
    _diagnostic_metrics,
    _dual_pool_config,
    _ExplodingEmptyInspector,
    _RecordingProxyPoolManager,
    _sanitize,
    _transformed_head_response,
)


def test_bridge_server_keeps_configured_geo_cache_path_and_runtime_identity(tmp_path) -> None:
    cache_path = tmp_path / "geo-cache.json"
    manager = ConfigManager.from_dict(make_config(1, server={"geo_cache_file": str(cache_path)}))
    server = BridgeServer(manager)
    try:
        snapshot = manager.get()
        assert server.geo_cache._path == cache_path
        assert server.runtime_for(snapshot) is server.runtime_for(snapshot)
    finally:
        server.server_close()


def test_anthropic_beta_merge_ignores_other_headers_and_empty_tokens() -> None:
    req = parse_request_head(b"POST / HTTP/1.1\r\nHost: h\r\nX-Other: wrong\r\nAnthropic-Beta: old, , value\r\n\r\n")

    _ensure_anthropic_beta(req, "new")

    assert [(key, value) for key, value in req.headers if key.lower() == "anthropic-beta"] == [
        ("anthropic-beta", "old, value, new")
    ]


def test_google_post_body_is_sanitized_but_get_and_other_paths_are_not() -> None:
    raw = b'{"contents":[{"role":"model","parts":[{"functionCall":{"name":"a"}}]}]}'
    sanitized, _ = _sanitize("POST", "/v1beta/models/m:generateContent", raw, "gemini")
    get_body, _ = _sanitize("GET", "/v1beta/models/m:generateContent", raw, "gemini")
    other_body, _ = _sanitize("POST", "/v1/chat/completions", raw, "gemini")

    assert sanitized != raw and b"thoughtSignature" in sanitized
    assert get_body == raw and other_body == raw


def test_empty_post_body_returns_before_sanitizers() -> None:
    handler = object.__new__(_Handler)
    handler.model = "gemini"
    req = parse_request_head(b"POST / HTTP/1.1\r\nHost: h\r\n\r\n")

    handler._sanitize_request_body(
        req,
        AdapterRoute("test", "up", "/v1beta/models/m:generateContent"),
        cast(BodyInspector, cast(object, _ExplodingEmptyInspector())),
    )


def test_anthropic_body_is_rewritten_only_on_messages_route() -> None:
    raw = b'{"thinking":{"type":"adaptive"}}'
    changed, headers = _sanitize("POST", "/v1/messages", raw, "claude-3-opus")
    untouched, _ = _sanitize("POST", "/v1/chat/completions", raw, "claude-3-opus")

    assert b'"prefix_mismatch_behavior":"drop_block"' in changed
    assert untouched == raw
    assert any(key.lower() == "anthropic-beta" and "thinking-binding-controls" in value for key, value in headers)


def test_proxied_region_403_is_not_replayed_to_geo_fallback(track, start_bridge) -> None:
    primary_up = track(HttpUpstream(lambda conn, _req: respond(conn, 403, REGION_BODY)))
    primary_proxy = track(FakeConnectProxy(target=("127.0.0.1", primary_up.port)))
    fallback_up = track(HttpUpstream(lambda conn, _req: respond(conn, 200, b"unexpected-fallback")))
    fallback_proxy = track(FakeConnectProxy(target=("127.0.0.1", fallback_up.port)))
    bridge = start_bridge(_dual_pool_config(primary_up.port, primary_proxy.port, fallback_proxy.port))

    response = http_request(bridge.port, "POST", PATH, BODY)

    assert response.status == 403 and response.body == REGION_BODY
    assert fallback_proxy.connects == []


def test_geo_fallback_replay_requires_token_from_config_retry_budget(track, start_bridge) -> None:
    """F-004: live-путь берёт retry-токен из `ConfigManager.retry_budget`, пустой bucket не реплеит."""
    primary_up = track(HttpUpstream(lambda conn, _req: respond(conn, 403, REGION_BODY)))
    fallback_up = track(HttpUpstream(lambda conn, _req: respond(conn, 200, b"via-nl")))
    fallback_proxy = track(FakeConnectProxy(target=("127.0.0.1", fallback_up.port)))
    config = make_config(
        primary_up.port,
        pools={"route-nl": {"type": "http_connect", "proxies": [f"http://127.0.0.1:{fallback_proxy.port}"]}},
        geo_fallback_pool="route-nl",
    )
    config["failover"] = {"rps_peak": 5.0, "sla_p99_seconds": 1.0}
    bridge = start_bridge(config)
    budget = bridge.server.config_manager.retry_budget

    blocked = http_request(bridge.port, "POST", PATH, BODY)
    assert blocked.status == 403 and blocked.body == REGION_BODY
    assert fallback_proxy.connects == [] and budget.available_tokens == 0

    for _ in range(5):
        budget.mint_on_admitted_primary()
    assert budget.available_tokens == 1

    replayed = http_request(bridge.port, "POST", PATH, BODY)
    assert replayed.status == 200 and replayed.body == b"via-nl"
    assert budget.available_tokens == 0


def test_geo_fallback_equal_to_direct_pool_does_not_replay_same_request(track, start_bridge) -> None:
    direct = track(HttpUpstream(lambda conn, _req: respond(conn, 403, REGION_BODY)))
    bridge = start_bridge(make_config(direct.port, geo_fallback_pool="direct"))

    response = http_request(bridge.port, "POST", PATH, BODY)

    assert response.status == 403 and len(direct.requests) == 1


def test_direct_header_wait_uses_regular_timeout_without_proxy_failover(track, start_bridge) -> None:
    def delayed_direct(conn, _req) -> None:
        time.sleep(0.12)
        respond(conn, 200, b"direct")

    direct = track(HttpUpstream(delayed_direct))
    fallback_up = track(HttpUpstream(lambda conn, _req: respond(conn, 200, b"fallback")))
    fallback_proxy = track(FakeConnectProxy(target=("127.0.0.1", fallback_up.port)))
    bridge = start_bridge(
        make_config(
            direct.port,
            pools={"route-nl": {"type": "http_connect", "proxies": [f"http://127.0.0.1:{fallback_proxy.port}"]}},
            geo_fallback_pool="route-nl",
            server={"headers_timeout": 0.5, "total_egress_deadline": 1.0},
        )
    )
    bridge.server.retry_budget = RetryBudgetSpec(primary_first_header_timeout=0.03)

    response = http_request(bridge.port, "POST", PATH, BODY)

    assert response.status == 200 and response.body == b"direct" and fallback_proxy.connects == []
    assert not any(key.startswith("classified_403") for key in bridge.server.metrics.snapshot())


def test_all_penalized_fails_before_slot_acquisition(track, start_bridge) -> None:
    proxy = track(FakeConnectProxy(status=502))
    config = make_config(
        1,
        pools={"route-de": {"type": "http_connect", "proxies": [f"http://127.0.0.1:{proxy.port}"]}},
        rules=[{"match_prefix": ["claude-"], "pool": "route-de"}],
        server={"proxy_slots": 1},
    )
    manager = ProxyPoolManager()
    pool = ConfigManager.from_dict(config).get().pools["route-de"]
    manager.penalize(manager.endpoints(pool)[0], 30)
    bridge = start_bridge(config, proxy_manager=manager)
    lease = bridge.server.slots.acquire("proxy", 1)
    assert lease is not None
    try:
        response = http_request(bridge.port, "POST", PATH, BODY)
    finally:
        lease.release()
    assert response.status == 502 and response.json["error"]["type"] == "proxy_unavailable"


def test_proxy_slot_exhaustion_never_starts_available_proxy(track, start_bridge) -> None:
    upstream = track(HttpUpstream(lambda conn, _req: respond(conn, 200, b"must-not-run")))
    proxy = track(FakeConnectProxy(target=("127.0.0.1", upstream.port)))
    config = make_config(
        upstream.port,
        pools={"route-de": {"type": "http_connect", "proxies": [f"http://127.0.0.1:{proxy.port}"]}},
        rules=[{"match_prefix": ["claude-"], "pool": "route-de"}],
        server={"proxy_slots": 1},
    )
    bridge = start_bridge(config)
    lease = bridge.server.slots.acquire("proxy", 1)
    assert lease is not None
    try:
        response = http_request(bridge.port, "POST", PATH, BODY)
    finally:
        lease.release()
    assert response.status == 503 and response.json["error"]["type"] == "bridge_overloaded"
    assert proxy.connects == []


def test_connect_failure_is_penalized_once_before_single_pool_failover(track, start_bridge) -> None:
    rejected = track(FakeConnectProxy(status=502))
    fallback_up = track(HttpUpstream(lambda conn, _req: respond(conn, 200, b"fallback")))
    fallback_proxy = track(FakeConnectProxy(target=("127.0.0.1", fallback_up.port)))
    manager = _RecordingProxyPoolManager()
    bridge = start_bridge(
        _dual_pool_config(fallback_up.port, rejected.port, fallback_proxy.port),
        proxy_manager=manager,
    )
    bridge.prime_replay_tokens()

    response = http_request(bridge.port, "POST", PATH, BODY)

    assert response.status == 200 and response.body == b"fallback"
    assert len(manager.penalties) == 1


def test_transformed_head_strips_framing_headers_and_sends_no_body() -> None:
    wire = _transformed_head_response("HEAD", b"secret")
    head, _, body = wire.partition(b"\r\n\r\n")

    assert b"Transfer-Encoding" not in head
    assert head.count(b"Content-Length: 6") == 1 and b"X-Test: yes" in head
    assert body == b""


def test_non_transforming_adapter_streams_first_chunk_before_upstream_finishes(track, start_bridge) -> None:
    release = threading.Event()

    def streaming(conn, _req) -> None:
        send_sse_head(conn)
        conn.sendall(chunk(b"data: first\n\n"))
        release.wait(1)
        conn.sendall(b"0\r\n\r\n")

    upstream = track(HttpUpstream(streaming))
    bridge = start_bridge(make_config(upstream.port))
    sock = open_request(bridge.port, "POST", PATH, BODY)
    try:
        first = read_until(sock, b"data: first", timeout=0.5)
        assert b"data: first" in first and not release.is_set()
        release.set()
        _ = first + read_all(sock)
    finally:
        release.set()
        sock.close()


def test_cloudcode_transform_returns_one_normalized_response(track, start_bridge) -> None:
    payload = json.dumps(
        {
            "currentTier": {"id": "paid"},
            "ineligibleTiers": [{"reasonCode": "UNSUPPORTED_LOCATION", "tierId": "free"}],
        }
    ).encode()
    upstream = track(HttpUpstream(lambda conn, _req: respond(conn, 200, payload, {"Content-Type": "application/json"})))
    config = make_config(
        upstream.port,
        upstreams={"cloudcode-pa": {"host": "127.0.0.1", "port": upstream.port, "use_tls": False}},
        adapters={"cloudcode": {"prefixes": ["/cloudcode"], "upstream": "cloudcode-pa"}},
    )
    bridge = start_bridge(config)

    response = http_request(bridge.port, "POST", "/cloudcode/v1internal:loadCodeAssist", b"{}")

    assert response.status == 200 and json.loads(response.body)["ineligibleTiers"] == []
    assert response.raw.count(b"HTTP/1.1 200") == 1
    assert "transfer-encoding" not in response.headers


def test_direct_region_403_replays_once_to_fallback(track, start_bridge) -> None:
    direct = track(HttpUpstream(lambda conn, _req: respond(conn, 403, REGION_BODY)))
    fallback_up = track(HttpUpstream(lambda conn, _req: respond(conn, 200, b"fallback")))
    fallback_proxy = track(FakeConnectProxy(target=("127.0.0.1", fallback_up.port)))
    bridge = start_bridge(
        make_config(
            direct.port,
            pools={"route-nl": {"type": "http_connect", "proxies": [f"http://127.0.0.1:{fallback_proxy.port}"]}},
            geo_fallback_pool="route-nl",
        )
    )
    bridge.prime_replay_tokens()
    response = http_request(bridge.port, "POST", PATH, BODY)

    assert response.status == 200 and response.body == b"fallback"
    assert len(direct.requests) == 1 and len(fallback_proxy.connects) == 1


def test_403_diagnostic_truncation_counter_distinguishes_complete_boundary() -> None:
    oversized = _diagnostic_metrics(b"x" * (DIAGNOSTIC_BODY_LIMIT + 1))
    exact = _diagnostic_metrics(b"x" * DIAGNOSTIC_BODY_LIMIT)

    assert oversized["classified_403_truncated"] == 1
    assert "classified_403_truncated" not in exact


def test_successful_primary_neither_spends_retry_token_nor_skips_minting(track, start_bridge) -> None:
    """F-003: токен тратится только при фактически принятом replay, а не на входе в primary."""
    primary = track(HttpUpstream(lambda conn, _req: respond(conn, 200, b"primary")))
    fallback_up = track(HttpUpstream(lambda conn, _req: respond(conn, 200, b"via-nl")))
    fallback_proxy = track(FakeConnectProxy(target=("127.0.0.1", fallback_up.port)))
    bridge = start_bridge(
        make_config(
            primary.port,
            pools={"route-nl": {"type": "http_connect", "proxies": [f"http://127.0.0.1:{fallback_proxy.port}"]}},
            geo_fallback_pool="route-nl",
        )
    )
    bridge.prime_replay_tokens()
    budget = bridge.server.config_manager.retry_budget
    assert budget.available_tokens == 1 and budget.admitted_primary == 5

    response = http_request(bridge.port, "POST", PATH, BODY)

    assert response.status == 200 and response.body == b"primary" and fallback_proxy.connects == []
    assert budget.available_tokens == 1 and budget.consumed_total == 0
    assert budget.admitted_primary == 6


def test_open_breaker_sends_new_requests_to_fallback_instead_of_dead_primary(track, start_bridge) -> None:
    """F-002: OPEN-состояние на реальных тиках переключает выбор маршрута на fallback."""
    direct = track(HttpUpstream(lambda conn, _req: respond(conn, 500, b"dead-primary")))
    fallback_up = track(HttpUpstream(lambda conn, _req: respond(conn, 200, b"via-nl")))
    fallback_proxy = track(FakeConnectProxy(target=("127.0.0.1", fallback_up.port)))
    bridge = start_bridge(
        make_config(
            direct.port,
            pools={"route-nl": {"type": "http_connect", "proxies": [f"http://127.0.0.1:{fallback_proxy.port}"]}},
            geo_fallback_pool="route-nl",
        )
    )
    route = bridge.server.runtime_for(bridge.server.config_manager.get()).registry.resolve(PATH)
    breaker, now = _breaker(route.upstream, "direct"), int(time.monotonic())
    for _ in range(3):
        breaker.record_failure(now_tick=now)
    assert breaker.state_at(now) is BreakerState.OPEN

    response = http_request(bridge.port, "POST", PATH, BODY)

    assert response.status == 200 and response.body == b"via-nl" and len(fallback_proxy.connects) == 1
    assert direct.requests == []


def test_half_open_breaker_reprobes_primary_once_cooldown_elapsed(track, start_bridge) -> None:
    """F-002: реальный now_tick двигает OPEN→HALF_OPEN и возвращает трафик в primary."""
    direct = track(HttpUpstream(lambda conn, _req: respond(conn, 200, b"primary")))
    fallback_up = track(HttpUpstream(lambda conn, _req: respond(conn, 200, b"via-nl")))
    fallback_proxy = track(FakeConnectProxy(target=("127.0.0.1", fallback_up.port)))
    bridge = start_bridge(
        make_config(
            direct.port,
            pools={"route-nl": {"type": "http_connect", "proxies": [f"http://127.0.0.1:{fallback_proxy.port}"]}},
            geo_fallback_pool="route-nl",
        )
    )
    route = bridge.server.runtime_for(bridge.server.config_manager.get()).registry.resolve(PATH)
    breaker, opened_at = _breaker(route.upstream, "direct"), int(time.monotonic()) - 20
    for _ in range(3):
        breaker.record_failure(now_tick=opened_at)
    assert breaker.state_at(int(time.monotonic())) is BreakerState.HALF_OPEN

    response = http_request(bridge.port, "POST", PATH, BODY)

    assert response.status == 200 and response.body == b"primary" and fallback_proxy.connects == []
    assert len(direct.requests) == 1
    assert breaker.state_at(int(time.monotonic())) is BreakerState.CLOSED


def _replay_spy(monkeypatch) -> list[dict[str, object]]:
    import model_georouter.bridge_server as bridge_module

    calls: list[dict[str, object]] = []

    def spy(facts: dict[str, object]) -> dict[str, object]:
        calls.append(facts)
        return {"replay_allowed": True}

    monkeypatch.setattr(bridge_module, "evaluate_replay_for_endpoint", spy)
    return calls


def _precommit_config(direct_port: int, fallback_port: int, **server: object) -> dict[str, object]:
    return make_config(
        direct_port,
        pools={"route-nl": {"type": "http_connect", "proxies": [f"http://127.0.0.1:{fallback_port}"]}},
        geo_fallback_pool="route-nl",
        server={"headers_timeout": 0.1, "total_egress_deadline": 2.0, "connect_timeout": 0.5, **server},
    )


def test_precommit_connect_failure_asks_replay_evaluator_before_failing(track, start_bridge, monkeypatch) -> None:
    """F-005: connect-сбой до коммита доходит до `evaluate_replay_for_endpoint` и реплеится."""
    fallback_up = track(HttpUpstream(lambda conn, _req: respond(conn, 200, b"via-nl")))
    fallback_proxy = track(FakeConnectProxy(target=("127.0.0.1", fallback_up.port)))
    calls = _replay_spy(monkeypatch)
    bridge = start_bridge(_precommit_config(free_port(), fallback_proxy.port))
    bridge.prime_replay_tokens()

    response = http_request(bridge.port, "POST", PATH, BODY)

    assert response.status == 200 and response.body == b"via-nl" and len(fallback_proxy.connects) == 1
    assert len(calls) == 1 and calls[0]["connect_error"] is True
    assert calls[0]["endpoint"] == PATH and calls[0]["retry_owner"] == "GEOROUTER_OWNER"


def test_precommit_header_timeout_asks_replay_evaluator_before_failing(track, start_bridge, monkeypatch) -> None:
    """F-005: header-timeout до коммита доходит до `evaluate_replay_for_endpoint` и реплеится."""

    def silent(conn, _req) -> None:
        time.sleep(0.3)

    direct = track(HttpUpstream(silent))
    fallback_up = track(HttpUpstream(lambda conn, _req: respond(conn, 200, b"via-nl")))
    fallback_proxy = track(FakeConnectProxy(target=("127.0.0.1", fallback_up.port)))
    calls = _replay_spy(monkeypatch)
    bridge = start_bridge(_precommit_config(direct.port, fallback_proxy.port))
    bridge.prime_replay_tokens()

    response = http_request(bridge.port, "POST", PATH, BODY)

    assert response.status == 200 and response.body == b"via-nl" and len(fallback_proxy.connects) == 1
    assert len(calls) == 1 and calls[0]["timed_out"] is True
