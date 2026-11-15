from __future__ import annotations

import json
import socket
import time
from typing import TypedDict, cast

import pytest

from model_georouter.adapters import (
    AdapterRegistry,
    AdapterRoute,
    AdminAdapter,
    CloudCodeAdapter,
    CordisAdapter,
    GenericAdapter,
    OpenCodeAdapter,
    PassthroughAdapter,
    clean_path,
)
from model_georouter.config import parse_config
from model_georouter.errors import BadRequestError
from model_georouter.http_wire import ResponseHead
from model_georouter.model_router import (
    SKIP_THOUGHT_SIGNATURE_VALIDATOR,
    THINKING_BINDING_BETA,
    ensure_anthropic_thinking_drop_block,
    sanitize_google_v1beta_body,
)
from tests.helpers import FakeConnectProxy, HttpUpstream, Req, http_request, make_config, respond
from tests.v3_types import StartBridge, Track


class GooglePart(TypedDict):
    text: str
    thoughtSignature: str
    functionCall: object


class GoogleContent(TypedDict):
    role: str
    parts: list[GooglePart]


class AdapterPayload(TypedDict):
    contents: list[GoogleContent]
    messages: list[dict[str, object]]
    thinking: dict[str, object]
    currentTier: dict[str, str]
    paidTier: dict[str, str]
    allowedTiers: list[dict[str, str]]
    ineligibleTiers: list[dict[str, str]]
    status: str
    listen: str
    slots: object
    pools: dict[str, object]
    geo_cache: dict[str, int]
    counters: dict[str, int]
    geo_cache_entries: int
    error: dict[str, str]


def _load(raw: str | bytes) -> AdapterPayload:
    return cast(AdapterPayload, json.loads(raw))


def _require_route(route: AdapterRoute | None) -> AdapterRoute:
    assert route is not None
    return route


OPENCODE = OpenCodeAdapter(("/v1", "/go/v1"), "opencode-ai")
CORDIS = CordisAdapter(("/zen/v1", "/zen/go/v1"), "cordis-ai")
GENERIC = GenericAdapter(("/messages", "/chat/completions"), "openrouter-ai")
CLOUDCODE = CloudCodeAdapter(("/cloudcode",), "cloudcode-pa", "route-de")


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
def test_opencode_normalization(incoming: str, expected: str) -> None:
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
def test_cordis_normalization(incoming: str, expected: str) -> None:
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
def test_generic_normalization(incoming: str, expected: str) -> None:
    route = GENERIC.route(incoming)
    assert route is not None and route.path == expected
    assert route.upstream == "openrouter-ai"


def test_adapters_reject_foreign_paths_on_segment_boundaries() -> None:
    assert OPENCODE.route("/v10/x") is None
    assert OPENCODE.route("/zen/v1/x") is None
    assert CORDIS.route("/zen/v10") is None
    assert GENERIC.route("/messagesX") is None
    assert not OPENCODE.matches("/messages")


@pytest.mark.parametrize(
    "path", ["/v1/../etc/passwd", "/v1/%2e%2e/x", "/v1/%2E%2E%2fx/..", "/v1/a\\b", "v1/x", "/v1/\x00"]
)
def test_path_traversal_and_garbage_rejected(path: str) -> None:
    with pytest.raises(BadRequestError):
        _ = clean_path(path)


def test_registry_picks_longest_prefix_across_adapters() -> None:
    registry = AdapterRegistry.from_config(
        parse_config({"server": {"listen": "127.0.0.1", "port": 1}, "pools": {}, "rules": []})
    )
    assert _require_route(registry.resolve("/v1/messages")).adapter == "opencode"
    assert _require_route(registry.resolve("/go/v1/messages")).adapter == "opencode"
    assert _require_route(registry.resolve("/zen/v1/messages")).adapter == "cordis"
    assert _require_route(registry.resolve("/zen/go/v1/messages")).adapter == "cordis"
    assert _require_route(registry.resolve("/messages")).adapter == "generic"
    assert _require_route(registry.resolve("/chat/completions")).path == "/v1/chat/completions"
    assert registry.resolve("/nothing/here") is None
    with pytest.raises(BadRequestError):
        _ = registry.resolve("/v1/../x")


def test_registry_honours_configured_prefixes() -> None:
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
    assert route is not None
    assert (route.adapter, route.upstream) == ("opencode", "cordis-ai")
    assert registry.resolve("/v1/chat/completions") is None


# ───────────────────────────── новые адаптеры из конфига ─────────────────────────────

SG_CONFIG = {
    "upstreams": {"vendor-ai": {"host": "api.vendor.example", "base_path": "/v1"}},
    "pools": {"route-sg": {"type": "http_connect", "proxies": ["http://127.0.0.1:9"]}},
    "adapters": {"sg-vendor": {"prefixes": ["/sg-vendor"], "upstream": "vendor-ai", "pool": "route-sg"}},
}


def test_passthrough_adapter_strips_longest_prefix() -> None:
    adapter = PassthroughAdapter(("/sg", "/sg/kimi"), "vendor-ai", "route-sg", name="sg-kimi")
    assert _require_route(adapter.route("/sg/kimi/chat/completions")).path == "/chat/completions"
    assert _require_route(adapter.route("/sg/kimi")).path == "/"
    assert _require_route(adapter.route("/sg/kimi/")).path == "/"
    assert _require_route(adapter.route("/sg/other")).path == "/other"
    assert adapter.route("/other") is None
    route = adapter.route("/sg/kimi/v1/messages")
    assert route is not None
    assert (route.adapter, route.upstream, route.pool) == ("sg-kimi", "vendor-ai", "route-sg")


def test_registry_builds_custom_adapters_from_config() -> None:
    cfg = parse_config({"server": {"listen": "127.0.0.1", "port": 1}, "rules": [], **SG_CONFIG})
    registry = AdapterRegistry.from_config(cfg)
    route = registry.resolve("/sg-vendor/chat/completions")
    assert route is not None
    assert (route.adapter, route.upstream, route.path, route.pool) == (
        "sg-vendor",
        "vendor-ai",
        "/chat/completions",
        "route-sg",
    )
    assert registry.get("sg-vendor") is not None
    assert registry.get("opencode") is not None  # встроенные адаптеры остались
    assert registry.resolve("/nothing/here") is None


class FakeProvider:
    def health_payload(self) -> dict[str, object]:
        return {"status": "ok"}

    def metrics_payload(self) -> dict[str, object]:
        return {"counters": {}}

    def flush_cache(self) -> int:
        return 3


def test_admin_adapter_dispatch() -> None:
    admin = AdminAdapter()
    provider = FakeProvider()
    assert admin.matches("/health") and admin.matches("/metrics") and admin.matches("/cache/flush")
    assert not admin.matches("/v1/health") and not admin.matches("/../health")
    assert admin.handle("GET", "/health", provider) == (200, {"status": "ok"}, {})
    assert admin.handle("GET", "/metrics", provider)[1] == {"counters": {}}
    assert admin.handle("POST", "/cache/flush", provider)[1] == {"flushed": 3}
    status, payload, headers = admin.handle("GET", "/cache/flush", provider)
    assert status == 405 and headers["Allow"] == "POST"
    assert payload["error"] == {"type": "method_not_allowed", "message": "use POST", "retryable": False}
    assert admin.handle("POST", "/health", provider)[0] == 405


# ───────────────────────────── через живой сервер ─────────────────────────────


def echo_path(conn: socket.socket, req: Req) -> None:
    respond(conn, 200, req.path.encode(), {"Content-Type": "text/plain"})


def test_health_reports_split_slots_and_pools(track: Track, start_bridge: StartBridge) -> None:
    upstream = track(HttpUpstream(echo_path))
    bridge = start_bridge(
        make_config(
            cast(int, upstream.port), pools={"route-de": {"type": "http_connect", "proxies": ["http://127.0.0.1:9"]}}
        )
    )
    response = http_request(bridge.port, "GET", "/health")
    assert response.status == 200 and response.headers["content-type"] == "application/json"
    health = _load(response.body)
    assert health["status"] == "ok" and health["listen"].startswith("127.0.0.1:")
    assert health["slots"] == {
        "direct": {"limit": 16, "in_use": 0, "available": 16},
        "proxy": {"limit": 16, "in_use": 0, "available": 16},
    }
    assert health["pools"]["route-de"] == {"type": "http_connect", "proxies_total": 1, "proxies_penalized": 0}
    assert health["geo_cache"] == {"entries": 0}
    assert upstream.requests == []  # admin-пути не уходят в upstream


def test_metrics_and_cache_flush_endpoints(track: Track, start_bridge: StartBridge) -> None:
    upstream = track(HttpUpstream(echo_path))
    bridge = start_bridge(make_config(cast(int, upstream.port)))
    bridge.server.geo_cache.mark_blocked("opencode-ai", "m1")
    bridge.server.geo_cache.mark_blocked("opencode-ai", "m2")
    assert http_request(bridge.port, "GET", "/cache/flush").status == 405
    flushed = http_request(bridge.port, "POST", "/cache/flush")
    assert flushed.status == 200 and cast(object, flushed.json) == {"flushed": 2}
    metrics = _load(http_request(bridge.port, "GET", "/metrics").body)
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
def test_adapters_rewrite_paths_toward_upstream(
    track: Track, start_bridge: StartBridge, ingress: str, upstream_path: str
) -> None:
    upstream = track(HttpUpstream(echo_path))
    bridge = start_bridge(make_config(cast(int, upstream.port)))
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


def test_upstream_base_path_is_prepended(track: Track, start_bridge: StartBridge) -> None:
    upstream = track(HttpUpstream(echo_path))
    config = make_config(
        cast(int, upstream.port),
        upstreams={
            "openrouter-ai": {
                "host": "127.0.0.1",
                "port": cast(int, upstream.port),
                "use_tls": False,
                "base_path": "/api",
            }
        },
    )
    bridge = start_bridge(config)
    _ = http_request(bridge.port, "POST", "/chat/completions", b"{}")
    assert upstream.requests[0].path == "/api/v1/chat/completions"


def test_custom_adapter_rewrites_path_and_prepends_base_path(track: Track, start_bridge: StartBridge) -> None:
    upstream = track(HttpUpstream(echo_path))
    config = make_config(
        cast(int, upstream.port),
        upstreams={
            "vendor-ai": {"host": "127.0.0.1", "port": cast(int, upstream.port), "use_tls": False, "base_path": "/v1"},
        },
        adapters={
            "sg-vendor": {"prefixes": ["/sg-vendor"], "upstream": "vendor-ai"},
        },
    )
    bridge = start_bridge(config)
    response = http_request(bridge.port, "POST", "/sg-vendor/chat/completions", b'{"model":"vendor-1"}')
    assert response.status == 200
    assert upstream.requests[0].path == "/v1/chat/completions"


def test_unknown_path_and_traversal_get_json_errors(track: Track, start_bridge: StartBridge) -> None:
    upstream = track(HttpUpstream(echo_path))
    bridge = start_bridge(make_config(cast(int, upstream.port)))
    missing = http_request(bridge.port, "GET", "/nothing")
    assert missing.status == 404 and _load(missing.body)["error"]["type"] == "not_found"
    traversal = http_request(bridge.port, "GET", "/v1/../health")
    assert traversal.status == 400 and _load(traversal.body)["error"]["type"] == "bad_request"
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
def test_clean_path_rejects_double_encoded_traversal(path: str) -> None:
    with pytest.raises(BadRequestError):
        _ = clean_path(path)


def test_clean_path_keeps_dots_inside_legit_segments() -> None:
    assert clean_path("/v1/gpt-4.1//x/") == "/v1/gpt-4.1/x"


def test_double_encoded_traversal_gets_400_from_live_server(track: Track, start_bridge: StartBridge) -> None:
    upstream = track(HttpUpstream(echo_path))
    bridge = start_bridge(make_config(cast(int, upstream.port)))
    for path in ("/v1/%2e%2e%2fsecret", "/v1/%252e%252e/secret"):
        response = http_request(bridge.port, "GET", path)
        assert response.status == 400 and _load(response.body)["error"]["type"] == "bad_request"
    assert upstream.requests == []


@pytest.mark.parametrize("path", ["/health", "/metrics"])
def test_admin_head_returns_headers_without_body(track: Track, start_bridge: StartBridge, path: str) -> None:  # RW-012
    upstream = track(HttpUpstream(echo_path))
    bridge = start_bridge(make_config(cast(int, upstream.port)))
    head = http_request(bridge.port, "HEAD", path)
    get = http_request(bridge.port, "GET", path)
    assert head.status == 200 and head.body_raw == b""
    assert head.headers["content-type"] == "application/json"
    assert int(head.headers["content-length"]) > 0 and "content-length" in get.headers
    assert head.headers["connection"] == "close"


def test_admin_head_on_post_only_endpoint_is_405_without_body(
    track: Track, start_bridge: StartBridge
) -> None:  # RW-012
    upstream = track(HttpUpstream(echo_path))
    bridge = start_bridge(make_config(cast(int, upstream.port)))
    head = http_request(bridge.port, "HEAD", "/cache/flush")
    assert head.status == 405 and head.body_raw == b"" and head.headers["allow"] == "POST"


@pytest.mark.parametrize(
    "method, path, status", [("POST", "/cache/flush", 200), ("POST", "/health", 405), ("POST", "/metrics", 405)]
)
def test_admin_early_response_drains_request_body(
    track: Track, start_bridge: StartBridge, method: str, path: str, status: int
) -> None:  # RW-002
    upstream = track(HttpUpstream(echo_path))
    bridge = start_bridge(make_config(cast(int, upstream.port)))
    response = http_request(bridge.port, method, path, b"x" * (8 * 1024 * 1024))
    assert response.status == status and cast(object, response.json)
    assert upstream.requests == []


# ───────────────────────────── Нормализация тела Google v1beta ─────────────────────────────

SKIP_SIG = "c2tpcF90aG91Z2h0X3NpZ25hdHVyZV92YWxpZGF0b3I="


def _dump(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def test_sanitize_google_v1beta_body_injects_thought_signature_and_strips_loops() -> None:
    assert SKIP_THOUGHT_SIGNATURE_VALIDATOR == SKIP_SIG
    leaked = "<unused94>thought\nplan<unused95>Real text\nend.]]> Let's stop.]]> Let's end.]]> Let's stop.]]>"
    payload: dict[str, object] = {
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
    out = _load(sanitize_google_v1beta_body(_dump(payload)))
    parts = out["contents"][1]["parts"]
    assert parts[0]["thoughtSignature"] == SKIP_SIG
    assert parts[1]["thoughtSignature"] == "existing-sig"
    assert parts[2] == {"text": "Real text"}
    assert out["contents"][0] == cast(list[object], payload["contents"])[0]


def test_sanitize_google_v1beta_body_drops_empty_text_parts_but_keeps_one_part() -> None:
    payload: dict[str, object] = {
        "contents": [{"role": "model", "parts": [{"text": "<unused94>thought\nonly thinking<unused95>"}]}]
    }
    out = _load(sanitize_google_v1beta_body(_dump(payload)))
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
        # непарные `]]>` в «соседних» CDATA без петли: хвостовые пробелы не теряются
        _dump(
            {
                "contents": [
                    {
                        "role": "model",
                        "parts": [
                            {
                                "text": "Keep <x><![CDATA[a]]></x> <x><![CDATA[b]]></x> "
                                + "<x><![CDATA[c]]></x> <x><![CDATA[d]]></x>  "
                            }
                        ],
                    }
                ]
            }
        ),
        # < 4 маркеров `]]>`: порог не достигнут, короткая петля остаётся как есть
        _dump({"contents": [{"role": "model", "parts": [{"text": "Real]]> Let's stop.]]> Let's stop.]]>"}]}]}),
    ],
)
def test_sanitize_google_v1beta_body_returns_same_bytes_without_mutation(raw: bytes) -> None:
    out = sanitize_google_v1beta_body(raw)
    assert raw is out


def test_google_v1beta_route_sanitizes_body_and_updates_content_length(track: Track, start_bridge: StartBridge) -> None:
    upstream = track(HttpUpstream(echo_path))
    bridge = start_bridge(make_config(cast(int, upstream.port)))
    payload: dict[str, object] = {
        "contents": [{"role": "model", "parts": [{"functionCall": {"name": "bash", "args": {}}}]}]
    }
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
    sent = _load(seen.body)
    assert sent["contents"][0]["parts"][0]["thoughtSignature"] == SKIP_SIG
    assert seen.body != raw
    assert seen.headers["content-length"] == str(len(seen.body))


def test_sanitize_google_v1beta_body_whitespace_prefix_is_linear_and_unchanged() -> None:
    raw = _dump({"contents": [{"role": "model", "parts": [{"text": " " * 20000 + "x]]> x]]> x]]> x]]>"}]}]})
    started = time.perf_counter()
    out = sanitize_google_v1beta_body(raw)
    assert time.perf_counter() - started < 0.2
    assert out is raw


def test_sanitize_google_v1beta_body_strips_loop_after_long_whitespace_prefix_quickly() -> None:
    text = "Answer" + " " * 20000 + "end.]]> Let's stop.]]> stop.]]> end.]]>"
    raw = _dump({"contents": [{"role": "model", "parts": [{"text": text}]}]})
    started = time.perf_counter()
    out = _load(sanitize_google_v1beta_body(raw))
    assert time.perf_counter() - started < 0.2
    assert out["contents"][0]["parts"] == [{"text": "Answer"}]


@pytest.mark.parametrize(
    "text",
    [
        # петля начинается с голого `]]>`, повторы с литералом
        "Answer]]> Let's stop.]]> Let's stop.]]> Let's stop.]]>",
        # цепочка из одних `]]>` без литералов
        "Answer]]>]]>]]>]]>",
    ],
)
def test_sanitize_google_v1beta_body_strips_loop_with_bare_marker(text: str) -> None:
    raw = _dump({"contents": [{"role": "model", "parts": [{"text": text}]}]})
    out = _load(sanitize_google_v1beta_body(raw))
    assert out["contents"][0]["parts"] == [{"text": "Answer"}]


def test_sanitize_google_v1beta_body_survives_unpaired_surrogate() -> None:
    raw = (
        b'{"contents":[{"role":"model","parts":[{"functionCall":{"name":"f"}}]},'
        b'{"role":"user","parts":[{"text":"\\ud800"}]}]}'
    )
    out = sanitize_google_v1beta_body(raw)
    assert out is not raw
    parsed = _load(out)
    assert parsed["contents"][0]["parts"][0]["thoughtSignature"] == SKIP_SIG
    assert parsed["contents"][1]["parts"][0]["text"] == "\ud800"


def test_google_v1beta_replay_sends_normalized_body_with_correct_length_on_both_attempts(
    track: Track, start_bridge: StartBridge
) -> None:
    region_body = b'{"error":{"type":"RegionError","message":"not available in your region"}}'

    def region_blocked(conn: socket.socket, _req: Req) -> None:
        respond(conn, 403, region_body, {"Content-Type": "application/json"})

    def ok(conn: socket.socket, _req: Req) -> None:
        respond(conn, 200, b"served-via-proxy", {"Content-Type": "text/plain"})

    direct_up = track(HttpUpstream(region_blocked))
    proxied_up = track(HttpUpstream(ok))
    proxy = track(FakeConnectProxy(target=("127.0.0.1", cast(int, proxied_up.port))))
    bridge = start_bridge(
        make_config(
            cast(int, direct_up.port),
            pools={"route-de": {"type": "http_connect", "proxies": [f"http://127.0.0.1:{cast(int, proxy.port)}"]}},
            geo_fallback_pool="route-de",
            server={"geo_error_signatures": ["RegionError"]},
        )
    )
    payload: dict[str, object] = {
        "contents": [
            {"role": "user", "parts": [{"text": "Привет, мир"}]},
            {"role": "model", "parts": [{"functionCall": {"name": "bash", "args": {}}}]},
        ]
    }
    raw = _dump(payload)
    response = http_request(
        bridge.port,
        "POST",
        "/go/v1/v1beta/models/gemini-4-argon-contributor:generateContent",
        raw,
        {"Content-Type": "application/json"},
    )
    assert response.status == 200 and response.body == b"served-via-proxy"
    assert len(direct_up.requests) == 1 and len(proxied_up.requests) == 1
    first, second = direct_up.requests[0], proxied_up.requests[0]
    assert first.body == second.body != raw
    assert "Привет, мир" in first.body.decode("utf-8")
    assert _load(first.body)["contents"][1]["parts"][0]["thoughtSignature"] == SKIP_SIG
    assert first.headers["content-length"] == second.headers["content-length"] == str(len(first.body))


# ───────────────────────────── Anthropic thinking block_binding ─────────────────────────────


@pytest.mark.parametrize("thinking_type", ["adaptive", "enabled"])
def test_ensure_anthropic_thinking_drop_block_injects_binding(thinking_type: str) -> None:
    payload: dict[str, object] = {
        "model": "claude-haiku-5-5",
        "thinking": {"type": thinking_type, "display": "summarized"},
        "messages": [{"role": "user", "content": "hi"}],
    }
    raw = _dump(payload)
    out, needs_beta = ensure_anthropic_thinking_drop_block(raw)
    assert needs_beta is True and out is not raw
    sent = _load(out)
    assert sent["thinking"] == {
        "type": thinking_type,
        "display": "summarized",
        "block_binding": {"prefix_mismatch_behavior": "drop_block"},
    }
    assert sent["messages"] == payload["messages"]
    assert THINKING_BINDING_BETA == "thinking-binding-controls-2026-08-01"


def test_ensure_anthropic_thinking_drop_block_keeps_client_behavior_but_needs_beta() -> None:
    raw = _dump({"thinking": {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "error"}}})
    out, needs_beta = ensure_anthropic_thinking_drop_block(raw)
    assert out is raw and needs_beta is True


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"not json",
        b"[1,2]",
        b'{"messages":[]}',
        b'{"thinking":"adaptive"}',
        b'{"thinking":{"type":"disabled"}}',
        b'{"thinking":{"display":"summarized"}}',
    ],
)
def test_ensure_anthropic_thinking_drop_block_noop(raw: bytes) -> None:
    out, needs_beta = ensure_anthropic_thinking_drop_block(raw)
    assert out is raw and needs_beta is False


def test_ensure_anthropic_thinking_drop_block_survives_unpaired_surrogate() -> None:
    raw = b'{"thinking":{"type":"adaptive"},"messages":[{"role":"user","content":"\\ud800"}]}'
    out, needs_beta = ensure_anthropic_thinking_drop_block(raw)
    assert needs_beta is True and out is not raw
    parsed = _load(out)
    assert parsed["thinking"]["block_binding"] == {"prefix_mismatch_behavior": "drop_block"}
    assert parsed["messages"][0]["content"] == "\ud800"


def test_anthropic_messages_route_injects_drop_block_and_merges_beta_header(
    track: Track, start_bridge: StartBridge
) -> None:
    upstream = track(HttpUpstream(echo_path))
    bridge = start_bridge(make_config(cast(int, upstream.port)))
    payload: dict[str, object] = {
        "model": "claude-haiku-5-5",
        "max_tokens": 16,
        "thinking": {"type": "adaptive", "display": "summarized"},
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "", "signature": "sig"}]},
            {"role": "user", "content": "again"},
        ],
    }
    raw = _dump(payload)
    response = http_request(
        bridge.port,
        "POST",
        "/go/v1/messages",
        raw,
        {"Content-Type": "application/json", "anthropic-beta": "interleaved-thinking-2025-05-14"},
    )
    assert response.status == 200
    seen = upstream.requests[0]
    assert seen.path == "/inference/go/anthropic/v1/messages"
    sent = _load(seen.body)
    assert sent["thinking"]["block_binding"] == {"prefix_mismatch_behavior": "drop_block"}
    betas = [token.strip() for token in seen.headers["anthropic-beta"].split(",")]
    assert betas == ["interleaved-thinking-2025-05-14", THINKING_BINDING_BETA]
    assert seen.headers["content-length"] == str(len(seen.body))


@pytest.mark.parametrize(
    "payload",
    [
        # не-Claude на anthropic-протоколе: чужой upstream может не знать block_binding/beta
        {"model": "deepseek-v4.1-flash", "thinking": {"type": "adaptive"}, "messages": []},
        # Claude без включённого thinking: подписывать и дропать нечего
        {"model": "claude-haiku-5-5", "thinking": {"type": "disabled"}, "messages": []},
        {"model": "claude-haiku-5-5", "messages": []},
    ],
)
def test_anthropic_messages_route_leaves_other_requests_untouched(
    track: Track, start_bridge: StartBridge, payload: dict[str, object]
) -> None:
    upstream = track(HttpUpstream(echo_path))
    bridge = start_bridge(make_config(cast(int, upstream.port)))
    raw = _dump(payload)
    response = http_request(bridge.port, "POST", "/go/v1/messages", raw, {"Content-Type": "application/json"})
    assert response.status == 200
    seen = upstream.requests[0]
    assert seen.body == raw
    assert "anthropic-beta" not in seen.headers


def test_cloudcode_normalization() -> None:
    route = CLOUDCODE.route("/cloudcode/v1internal:loadCodeAssist")
    assert route is not None
    assert route.adapter == "cloudcode"
    assert route.upstream == "cloudcode-pa"
    assert route.path == "/v1internal:loadCodeAssist"
    assert route.pool == "route-de"

    route_root = CLOUDCODE.route("/cloudcode")
    assert route_root is not None
    assert route_root.path == "/"

    route_clean = CLOUDCODE.route("//cloudcode//v1internal:loadCodeAssist/")
    assert route_clean is not None
    assert route_clean.path == "/v1internal:loadCodeAssist"

    assert CLOUDCODE.route("/other/endpoint") is None


def test_cloudcode_wants_transform() -> None:
    route_assist = CLOUDCODE.route("/cloudcode/v1internal:loadCodeAssist")
    assert route_assist is not None
    head_200 = ResponseHead("HTTP/1.1", 200, "OK", [("Content-Type", "application/json")])
    head_403 = ResponseHead("HTTP/1.1", 403, "Forbidden", [("Content-Type", "application/json")])
    head_500 = ResponseHead("HTTP/1.1", 500, "Internal Server Error", [("Content-Type", "application/json")])

    assert CLOUDCODE.wants_transform(route_assist, head_200) is True
    assert CLOUDCODE.wants_transform(route_assist, head_403) is False
    assert CLOUDCODE.wants_transform(route_assist, head_500) is False

    route_other = CLOUDCODE.route("/cloudcode/v1/models")
    assert route_other is not None
    assert CLOUDCODE.wants_transform(route_other, head_200) is False


def test_cloudcode_transform_response_strips_unsupported_location() -> None:
    route = CLOUDCODE.route("/cloudcode/v1internal:loadCodeAssist")
    assert route is not None
    head = ResponseHead(
        "HTTP/1.1",
        200,
        "OK",
        [
            ("Content-Type", "application/json"),
            ("Content-Length", "999"),
            ("Transfer-Encoding", "chunked"),
        ],
    )
    original: dict[str, object] = {
        "currentTier": {"id": "standard-tier"},
        "paidTier": {"id": "g1-ultra-tier"},
        "allowedTiers": [{"tierId": "standard-tier"}],
        "ineligibleTiers": [{"reasonCode": "UNSUPPORTED_LOCATION", "tierId": "free-tier"}],
    }
    body = json.dumps(original).encode("utf-8")
    new_head, new_body = CLOUDCODE.transform_response(route, head, body)

    transformed = _load(new_body)
    assert transformed["currentTier"] == {"id": "standard-tier"}
    assert transformed["paidTier"] == {"id": "g1-ultra-tier"}
    assert transformed["allowedTiers"] == [{"tierId": "standard-tier"}]
    assert transformed["ineligibleTiers"] == []
    assert new_head.get("content-length") == str(len(new_body))
    assert new_head.get("transfer-encoding") is None


def test_cloudcode_transform_response_preserves_other_reasons() -> None:
    route = CLOUDCODE.route("/cloudcode/v1internal:loadCodeAssist")
    assert route is not None
    head = ResponseHead("HTTP/1.1", 200, "OK", [("Content-Type", "application/json")])
    original: dict[str, object] = {
        "currentTier": {"id": "standard-tier"},
        "ineligibleTiers": [{"reasonCode": "PAYMENT_REQUIRED", "tierId": "premium-tier"}],
    }
    body = json.dumps(original).encode("utf-8")
    new_head, new_body = CLOUDCODE.transform_response(route, head, body)
    assert new_body == body
    assert new_head == head


def test_cloudcode_transform_response_fail_safe_on_invalid_json() -> None:
    route = CLOUDCODE.route("/cloudcode/v1internal:loadCodeAssist")
    assert route is not None
    head = ResponseHead("HTTP/1.1", 200, "OK", [("Content-Type", "application/json")])
    invalid_body = b"not a valid json {{"
    new_head, new_body = CLOUDCODE.transform_response(route, head, invalid_body)
    assert new_body == invalid_body
    assert new_head == head


def test_cloudcode_transform_response_handles_gzip() -> None:
    import gzip

    route = CLOUDCODE.route("/cloudcode/v1internal:loadCodeAssist")
    assert route is not None
    head = ResponseHead(
        "HTTP/1.1",
        200,
        "OK",
        [
            ("Content-Type", "application/json"),
            ("Content-Encoding", "gzip"),
        ],
    )
    original: dict[str, object] = {
        "currentTier": {"id": "standard-tier"},
        "ineligibleTiers": [{"reasonCode": "UNSUPPORTED_LOCATION", "tierId": "free-tier"}],
    }
    compressed = gzip.compress(json.dumps(original).encode("utf-8"))
    new_head, new_body = CLOUDCODE.transform_response(route, head, compressed)

    transformed = _load(new_body)
    assert transformed["ineligibleTiers"] == []
    assert new_head.get("content-encoding") is None
    assert new_head.get("content-length") == str(len(new_body))


def test_cloudcode_bridge_end_to_end(track: Track, start_bridge: StartBridge) -> None:
    google_payload: dict[str, object] = {
        "currentTier": {"id": "standard-tier"},
        "paidTier": {"id": "g1-ultra-tier"},
        "ineligibleTiers": [{"reasonCode": "UNSUPPORTED_LOCATION", "tierId": "free-tier"}],
    }
    encoded = json.dumps(google_payload).encode("utf-8")

    def upstream_handler(conn: socket.socket, req: Req) -> None:
        assert req.path == "/v1internal:loadCodeAssist"
        respond(conn, 200, encoded, {"Content-Type": "application/json"})

    up = track(HttpUpstream(upstream_handler))
    cfg = make_config(
        cast(int, up.port),
        upstreams={"cloudcode-pa": {"host": "127.0.0.1", "port": cast(int, up.port), "use_tls": False}},
        adapters={"cloudcode": {"prefixes": ["/cloudcode"], "upstream": "cloudcode-pa"}},
    )
    bridge = start_bridge(cfg)

    response = http_request(
        bridge.port,
        "POST",
        "/cloudcode/v1internal:loadCodeAssist",
        b"{}",
        {"Content-Type": "application/json"},
    )
    assert response.status == 200
    data = _load(response.body)
    assert data["currentTier"] == {"id": "standard-tier"}
    assert data["paidTier"] == {"id": "g1-ultra-tier"}
    assert data["ineligibleTiers"] == []
    assert response.headers["content-length"] == str(len(response.body))


def test_cloudcode_transform_response_requires_tier() -> None:
    """Без currentTier и без paidTier ineligibleTiers НЕ удаляются (RW-002)."""
    route = CLOUDCODE.route("/cloudcode/v1internal:loadCodeAssist")
    assert route is not None
    head = ResponseHead("HTTP/1.1", 200, "OK", [("Content-Type", "application/json")])
    original: dict[str, object] = {
        "ineligibleTiers": [{"reasonCode": "UNSUPPORTED_LOCATION", "tierId": "free-tier"}],
    }
    body = json.dumps(original).encode("utf-8")
    new_head, new_body = CLOUDCODE.transform_response(route, head, body)
    assert new_body == body
    assert new_head == head


def test_cloudcode_transform_response_handles_corrupt_gzip() -> None:
    """Повреждённый gzip-поток (zlib.error / BadGzipFile) возвращает исходный ответ без исключения (RW-003)."""
    route = CLOUDCODE.route("/cloudcode/v1internal:loadCodeAssist")
    assert route is not None
    head = ResponseHead(
        "HTTP/1.1",
        200,
        "OK",
        [
            ("Content-Type", "application/json"),
            ("Content-Encoding", "gzip"),
        ],
    )
    # Валидный заголовок gzip, но повреждённый deflate-поток
    corrupt_gzip = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03\xff\xff\x00\x00"
    new_head, new_body = CLOUDCODE.transform_response(route, head, corrupt_gzip)
    assert new_body == corrupt_gzip
    assert new_head == head
