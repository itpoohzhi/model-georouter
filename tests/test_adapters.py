from __future__ import annotations

import json

import pytest

from model_georouter.adapters import (
    AdapterRegistry,
    AdminAdapter,
    CordisAdapter,
    GenericAdapter,
    OpenCodeAdapter,
    clean_path,
)
from model_georouter.config import parse_config
from model_georouter.errors import BadRequestError
from model_georouter.model_router import SKIP_THOUGHT_SIGNATURE_VALIDATOR, sanitize_google_v1beta_body
from tests.helpers import HttpUpstream, http_request, make_config, respond

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
        (
            "/go/v1/v1beta/models/gemini-pro:streamGenerateContent",
            "/inference/go/google/v1beta/models/gemini-pro:streamGenerateContent",
        ),
        ("/go/v1/models", "/inference/go/openai/v1/models"),
        (
            "/go/v1/models/gemini-4-argon-contributor:streamGenerateContent",
            "/inference/go/google/v1beta/models/gemini-4-argon-contributor:streamGenerateContent",
        ),
        (
            "/v1/models/gemini-4-argon-contributor:generateContent",
            "/inference/google/v1beta/models/gemini-4-argon-contributor:generateContent",
        ),
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


@pytest.mark.parametrize(
    "path", ["/v1/../etc/passwd", "/v1/%2e%2e/x", "/v1/%2E%2E%2fx/..", "/v1/a\\b", "v1/x", "/v1/\x00"]
)
def test_path_traversal_and_garbage_rejected(path):
    with pytest.raises(BadRequestError):
        clean_path(path)


def test_registry_picks_longest_prefix_across_adapters():
    registry = AdapterRegistry.from_config(
        parse_config({"server": {"listen": "127.0.0.1", "port": 1}, "pools": {}, "rules": []})
    )
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
    config = make_config(
        upstream.port,
        upstreams={
            "openrouter-ai": {"host": "127.0.0.1", "port": upstream.port, "use_tls": False, "base_path": "/api"}
        },
    )
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


# ───────────────────────────── Rework Cycle 1 ─────────────────────────────


@pytest.mark.parametrize(
    "path",
    [
        "/v1/%2e%2e%2fsecret",  # RW-003: после одного decode сегмент содержит `/`
        "/v1/%252e%252e/secret",  # RW-003: двойное кодирование `..`
        "/v1/a%255cb",  # RW-003: двойное кодирование `\`
        "/v1/%252e",
        "/v1/a%2Fb",
    ],
)
def test_clean_path_rejects_double_encoded_traversal(path):
    with pytest.raises(BadRequestError):
        clean_path(path)


def test_clean_path_keeps_dots_inside_legit_segments():
    assert clean_path("/v1/gpt-4.1//x/") == "/v1/gpt-4.1/x"


def test_double_encoded_traversal_gets_400_from_live_server(track, start_bridge):
    upstream = track(HttpUpstream(echo_path))
    bridge = start_bridge(make_config(upstream.port))
    for path in ("/v1/%2e%2e%2fsecret", "/v1/%252e%252e/secret"):
        response = http_request(bridge.port, "GET", path)
        assert response.status == 400 and response.json["error"]["type"] == "bad_request"
    assert upstream.requests == []


@pytest.mark.parametrize("path", ["/health", "/metrics"])
def test_admin_head_returns_headers_without_body(track, start_bridge, path):  # RW-012
    upstream = track(HttpUpstream(echo_path))
    bridge = start_bridge(make_config(upstream.port))
    head = http_request(bridge.port, "HEAD", path)
    get = http_request(bridge.port, "GET", path)
    assert head.status == 200 and head.body_raw == b""
    assert head.headers["content-type"] == "application/json"
    assert int(head.headers["content-length"]) > 0 and "content-length" in get.headers
    assert head.headers["connection"] == "close"


def test_admin_head_on_post_only_endpoint_is_405_without_body(track, start_bridge):  # RW-012
    upstream = track(HttpUpstream(echo_path))
    bridge = start_bridge(make_config(upstream.port))
    head = http_request(bridge.port, "HEAD", "/cache/flush")
    assert head.status == 405 and head.body_raw == b"" and head.headers["allow"] == "POST"


@pytest.mark.parametrize(
    "method, path, status", [("POST", "/cache/flush", 200), ("POST", "/health", 405), ("POST", "/metrics", 405)]
)
def test_admin_early_response_drains_request_body(track, start_bridge, method, path, status):  # RW-002
    upstream = track(HttpUpstream(echo_path))
    bridge = start_bridge(make_config(upstream.port))
    response = http_request(bridge.port, method, path, b"x" * (8 * 1024 * 1024))
    assert response.status == status and response.json
    assert upstream.requests == []


# ───────────────────────────── Нормализация тела Google v1beta ─────────────────────────────

SKIP_SIG = "c2tpcF90aG91Z2h0X3NpZ25hdHVyZV92YWxpZGF0b3I="


def _dump(payload) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def test_sanitize_google_v1beta_body_injects_thought_signature_and_strips_loops():
    assert SKIP_THOUGHT_SIGNATURE_VALIDATOR == SKIP_SIG
    leaked = "<unused94>thought\nplan<unused95>Real text\nend.]]> Let's stop.]]> Let's end.]]> Let's stop.]]>"
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": "hi"}]},
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "bash", "args": {}}},
                    {"functionCall": {"name": "ls", "args": {}}, "thoughtSignature": "existing-sig"},
                    {"text": leaked},
                ],
            },
        ]
    }
    out = json.loads(sanitize_google_v1beta_body(_dump(payload)))
    parts = out["contents"][1]["parts"]
    assert parts[0]["thoughtSignature"] == SKIP_SIG
    assert parts[1]["thoughtSignature"] == "existing-sig"
    assert parts[2] == {"text": "Real text"}
    assert out["contents"][0] == payload["contents"][0]


def test_sanitize_google_v1beta_body_drops_empty_text_parts_but_keeps_one_part():
    payload = {"contents": [{"role": "model", "parts": [{"text": "<unused94>thought\nonly thinking<unused95>"}]}]}
    out = json.loads(sanitize_google_v1beta_body(_dump(payload)))
    assert out["contents"][0]["parts"] == [{"text": " "}]


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"not json",
        b"[1,2]",
        b'{"contents":"x"}',
        b'{"contents":[{"role":"user","parts":[{"functionCall":{"name":"a"}}]}]}',
        b'{"contents": [{"role": "model", "parts": [{"functionCall": {"name": "a"}, "thoughtSignature": "s"}]}]}',
        b'{"contents": [{"role": "model", "parts": [{"text": "plain"}]}]}',
    ],
)
def test_sanitize_google_v1beta_body_returns_same_bytes_without_mutation(raw):
    out = sanitize_google_v1beta_body(raw)
    assert raw is out


def test_google_v1beta_route_sanitizes_body_and_updates_content_length(track, start_bridge):
    upstream = track(HttpUpstream(echo_path))
    bridge = start_bridge(make_config(upstream.port))
    payload = {"contents": [{"role": "model", "parts": [{"functionCall": {"name": "bash", "args": {}}}]}]}
    raw = _dump(payload)
    response = http_request(
        bridge.port,
        "POST",
        "/go/v1/v1beta/models/gemini-4-argon-contributor:streamGenerateContent?alt=sse",
        raw,
        {"Content-Type": "application/json"},
    )
    assert response.status == 200
    seen = upstream.requests[0]
    assert seen.path.startswith("/inference/go/google/v1beta/models/gemini-4-argon-contributor:streamGenerateContent")
    sent = json.loads(seen.body)
    assert sent["contents"][0]["parts"][0]["thoughtSignature"] == SKIP_SIG
    assert seen.body != raw
    assert seen.headers["content-length"] == str(len(seen.body))
