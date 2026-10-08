from __future__ import annotations

import pytest

from tests.helpers import HttpUpstream, http_request, make_config, respond
from universal_ai_bridge.adapters import (
    AdapterRegistry,
    AdminAdapter,
    CordisAdapter,
    GenericAdapter,
    OpenCodeAdapter,
    clean_path,
)
from universal_ai_bridge.config import parse_config
from universal_ai_bridge.errors import BadRequestError

OPENCODE = OpenCodeAdapter(("/v1", "/go/v1"), "opencode-ai")
CORDIS = CordisAdapter(("/zen/v1", "/zen/go/v1"), "cordis-ai")
GENERIC = GenericAdapter(("/messages", "/chat/completions"), "openrouter-ai")


@pytest.mark.parametrize(
    "incoming, expected",
    [
        ("/v1/responses", "/inference/openai/v1/responses"),
        ("/v1/chat/completions", "/inference/openai/v1/chat/completions"),
        ("/v1/messages", "/inference/anthropic/v1/messages"),
        ("/v1/models", "/inference/openai/v1/models"),
        ("/go/v1/chat/completions", "/inference/go/openai/v1/chat/completions"),
        ("/go/v1/messages", "/inference/go/anthropic/v1/messages"),
        ("/v1/v1beta/models/gemini-pro:generateContent", "/inference/google/v1beta/models/gemini-pro:generateContent"),
        ("/go/v1/v1beta/models/gemini-pro:streamGenerateContent", "/inference/go/google/v1beta/models/gemini-pro:streamGenerateContent"),
        ("/v1/unknown/endpoint", "/v1/unknown/endpoint"),  # неизвестные пути идут как есть
        ("//v1//chat/completions/", "/inference/openai/v1/chat/completions"),
    ],
)
def test_opencode_normalization(incoming, expected):
    route = OPENCODE.route(incoming)
    assert route is not None and route.path == expected
    assert (route.adapter, route.upstream) == ("opencode", "opencode-ai")


@pytest.mark.parametrize(
    "incoming, expected",
    [
        ("/zen/v1/chat/completions", "/zen/v1/chat/completions"),
        ("/zen/go/v1/messages", "/zen/go/v1/messages"),
        ("//zen//v1/./chat/completions/", "/zen/v1/chat/completions"),
    ],
)
def test_cordis_normalization(incoming, expected):
    route = CORDIS.route(incoming)
    assert route is not None and route.path == expected
    assert route.adapter == "cordis"


@pytest.mark.parametrize(
    "incoming, expected",
    [
        ("/messages", "/v1/messages"),
        ("/chat/completions", "/v1/chat/completions"),
        ("/chat/completions/", "/v1/chat/completions"),
        ("/messages/count_tokens", "/v1/messages/count_tokens"),
    ],
)
def test_generic_normalization(incoming, expected):
    route = GENERIC.route(incoming)
    assert route is not None and route.path == expected
    assert route.upstream == "openrouter-ai"


def test_adapters_reject_foreign_paths_on_segment_boundaries():
    assert OPENCODE.route("/v10/x") is None
    assert OPENCODE.route("/zen/v1/x") is None
    assert CORDIS.route("/zen/v10") is None
    assert GENERIC.route("/messagesX") is None
    assert not OPENCODE.matches("/messages")


@pytest.mark.parametrize("path", ["/v1/../etc/passwd", "/v1/%2e%2e/x", "/v1/%2E%2E%2fx/..", "/v1/a\\b", "v1/x", "/v1/\x00"])
def test_path_traversal_and_garbage_rejected(path):
    with pytest.raises(BadRequestError):
        clean_path(path)


def test_registry_picks_longest_prefix_across_adapters():
    registry = AdapterRegistry.from_config(parse_config({"server": {"listen": "127.0.0.1", "port": 1}, "pools": {}, "rules": []}))
    assert registry.resolve("/v1/messages").adapter == "opencode"
    assert registry.resolve("/go/v1/messages").adapter == "opencode"
    assert registry.resolve("/zen/v1/messages").adapter == "cordis"
    assert registry.resolve("/zen/go/v1/messages").adapter == "cordis"
    assert registry.resolve("/messages").adapter == "generic"
    assert registry.resolve("/chat/completions").path == "/v1/chat/completions"
    assert registry.resolve("/nothing/here") is None
    with pytest.raises(BadRequestError):
        registry.resolve("/v1/../x")


def test_registry_honours_configured_prefixes():
    cfg = parse_config(
        {
            "server": {"listen": "127.0.0.1", "port": 1},
            "pools": {},
            "rules": [],
            "adapters": {"opencode": {"prefixes": ["/oc"], "upstream": "cordis-ai"}},
        }
    )
    registry = AdapterRegistry.from_config(cfg)
    route = registry.resolve("/oc/chat/completions")
    assert (route.adapter, route.upstream) == ("opencode", "cordis-ai")
    assert registry.resolve("/v1/chat/completions") is None


class FakeProvider:
    def health_payload(self):
        return {"status": "ok"}

    def metrics_payload(self):
        return {"counters": {}}

    def flush_cache(self):
        return 3


def test_admin_adapter_dispatch():
    admin = AdminAdapter()
    provider = FakeProvider()
    assert admin.matches("/health") and admin.matches("/metrics") and admin.matches("/cache/flush")
    assert not admin.matches("/v1/health") and not admin.matches("/../health")
    assert admin.handle("GET", "/health", provider) == (200, {"status": "ok"}, {})
    assert admin.handle("GET", "/metrics", provider)[1] == {"counters": {}}
    assert admin.handle("POST", "/cache/flush", provider)[1] == {"flushed": 3}
    status, payload, headers = admin.handle("GET", "/cache/flush", provider)
    assert status == 405 and headers["Allow"] == "POST" and payload["error"]["type"] == "method_not_allowed"
    assert admin.handle("POST", "/health", provider)[0] == 405


# ───────────────────────────── через живой сервер ─────────────────────────────


def echo_path(conn, req):
    respond(conn, 200, req.path.encode(), {"Content-Type": "text/plain"})


def test_health_reports_split_slots_and_pools(track, start_bridge):
    upstream = track(HttpUpstream(echo_path))
    bridge = start_bridge(
        make_config(upstream.port, pools={"route-de": {"type": "http_connect", "proxies": ["http://127.0.0.1:9"]}})
    )
    response = http_request(bridge.port, "GET", "/health")
    assert response.status == 200 and response.headers["content-type"] == "application/json"
    health = response.json
    assert health["status"] == "ok" and health["listen"].startswith("127.0.0.1:")
    assert health["slots"] == {
        "direct": {"limit": 16, "in_use": 0, "available": 16},
        "proxy": {"limit": 16, "in_use": 0, "available": 16},
    }
    assert health["pools"]["route-de"] == {"type": "http_connect", "proxies_total": 1, "proxies_penalized": 0}
    assert health["geo_cache"] == {"entries": 0}
    assert upstream.requests == []  # admin-пути не уходят в upstream


def test_metrics_and_cache_flush_endpoints(track, start_bridge):
    upstream = track(HttpUpstream(echo_path))
    bridge = start_bridge(make_config(upstream.port))
    bridge.server.geo_cache.mark_blocked("opencode-ai", "m1")
    bridge.server.geo_cache.mark_blocked("opencode-ai", "m2")
    assert http_request(bridge.port, "GET", "/cache/flush").status == 405
    flushed = http_request(bridge.port, "POST", "/cache/flush")
    assert flushed.status == 200 and flushed.json == {"flushed": 2}
    metrics = http_request(bridge.port, "GET", "/metrics").json
    assert metrics["counters"]["requests_total"] >= 3
    assert metrics["geo_cache_entries"] == 0


@pytest.mark.parametrize(
    "ingress, upstream_path",
    [
        ("/v1/chat/completions", "/inference/openai/v1/chat/completions"),
        ("/go/v1/messages", "/inference/go/anthropic/v1/messages"),
        ("/zen/v1/chat/completions", "/zen/v1/chat/completions"),
        ("/zen/go/v1/messages", "/zen/go/v1/messages"),
        ("/messages", "/v1/messages"),
        ("/chat/completions", "/v1/chat/completions"),
    ],
)
def test_adapters_rewrite_paths_toward_upstream(track, start_bridge, ingress, upstream_path):
    upstream = track(HttpUpstream(echo_path))
    bridge = start_bridge(make_config(upstream.port))
    response = http_request(
        bridge.port,
        "POST",
        ingress + "?beta=true",
        b'{"model":"m"}',
        {"Content-Type": "application/json", "Authorization": "Bearer sk-test", "X-Custom": "kept"},
    )
    assert response.status == 200
    seen = upstream.requests[0]
    assert seen.path == upstream_path + "?beta=true"
    assert seen.body == b'{"model":"m"}'
    assert seen.headers["authorization"] == "Bearer sk-test" and seen.headers["x-custom"] == "kept"
    assert seen.headers["accept-encoding"] == "identity" and seen.headers["connection"] == "close"
    assert seen.headers["content-length"] == "13"


def test_upstream_base_path_is_prepended(track, start_bridge):
    upstream = track(HttpUpstream(echo_path))
    config = make_config(upstream.port, upstreams={"openrouter-ai": {"host": "127.0.0.1", "port": upstream.port, "use_tls": False, "base_path": "/api"}})
    bridge = start_bridge(config)
    http_request(bridge.port, "POST", "/chat/completions", b"{}")
    assert upstream.requests[0].path == "/api/v1/chat/completions"


def test_unknown_path_and_traversal_get_json_errors(track, start_bridge):
    upstream = track(HttpUpstream(echo_path))
    bridge = start_bridge(make_config(upstream.port))
    missing = http_request(bridge.port, "GET", "/nothing")
    assert missing.status == 404 and missing.json["error"]["type"] == "not_found"
    traversal = http_request(bridge.port, "GET", "/v1/../health")
    assert traversal.status == 400 and traversal.json["error"]["type"] == "bad_request"
    assert upstream.requests == []
