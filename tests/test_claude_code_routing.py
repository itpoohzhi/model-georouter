from __future__ import annotations

import json
import socket
import struct
import time
from dataclasses import replace
from typing import Any, cast

import pytest

from model_georouter.bridge_server import _Handler
from model_georouter.claude_code_routing import CatalogTicketStore
from tests.helpers import (
    BridgeHarness,
    HttpUpstream,
    Req,
    chunk,
    free_port,
    http_request,
    make_config,
    respond,
    send_sse_head,
)
from tests.v3_types import StartBridge, Track

STANDARD = "claude-standard"
SYNTHETIC = "claude-synthetic"
STREAM_PATH = "/cloudcode/v1internal:streamGenerateContent"
CATALOG_PATH = "/cloudcode/v1internal:fetchAvailableModels"
AUTH = "Bearer google-secret"


def _body(model: str = STANDARD) -> bytes:
    return json.dumps({"model": model, "request": {"contents": []}}, separators=(",", ":")).encode()


def _config(google_port: int, sidecar_port: int, *, enabled: bool = True) -> dict[str, object]:
    return make_config(
        google_port,
        upstreams={
            "cloudcode-pa": {"host": "127.0.0.1", "port": google_port, "use_tls": False},
            "cloudcode-claude-sidecar": {"host": "127.0.0.1", "port": sidecar_port, "use_tls": False},
        },
        adapters={"cloudcode": {"prefixes": ["/cloudcode"], "upstream": "cloudcode-pa"}},
        cloudcode_claude_code={
            "enabled": enabled,
            "sidecar_upstream": "cloudcode-claude-sidecar",
            "intercept_models": [STANDARD, SYNTHETIC],
            "model_mapping": {STANDARD: STANDARD, SYNTHETIC: SYNTHETIC},
            "google_fallback_models": [STANDARD],
        },
    )


def _google_ok(conn: socket.socket, _req: Req) -> None:
    respond(conn, 200, b"google", {"Content-Type": "text/plain"})


def _issue_catalog_ticket(store: CatalogTicketStore) -> str:
    grant = cast(Any, object())
    return store.issue(grant, grant, grant, grant, "", grant, grant, "direct")


def _launch(
    track: Track,
    start_bridge: StartBridge,
    sidecar_handler,
    *,
    enabled: bool = True,
    google_handler=_google_ok,
) -> tuple[HttpUpstream, HttpUpstream, BridgeHarness]:
    google = track(HttpUpstream(google_handler))
    sidecar = track(HttpUpstream(sidecar_handler))
    bridge = start_bridge(_config(google.port, sidecar.port, enabled=enabled))
    return google, sidecar, bridge


def test_exact_match_uses_normalized_path_and_sanitizes_headers(track: Track, start_bridge: StartBridge) -> None:
    def sidecar_ok(conn: socket.socket, _req: Req) -> None:
        respond(conn, 200, b"sidecar")

    google, sidecar, bridge = _launch(track, start_bridge, sidecar_ok)
    response = http_request(
        bridge.port,
        "POST",
        STREAM_PATH + "?alt=sse",
        _body(),
        {
            "Authorization": AUTH,
            "Proxy-Authorization": "Bearer proxy-secret",
            "Cookie": "session=secret",
            "X-Api-Key": "api-secret",
            "X-Goog-Api-Key": "goog-secret",
            "X-Custom": "kept",
            "X-Model-Georouter-Client-Header": "removed",
        },
    )

    assert response.status == 200 and response.body == b"sidecar"
    assert google.requests == [] and len(sidecar.requests) == 1
    seen = sidecar.requests[0]
    assert seen.path == "/v1internal:streamGenerateContent?alt=sse" and seen.body == _body()
    assert seen.headers["x-custom"] == "kept"
    assert "x-model-georouter-client-header" not in seen.headers
    assert seen.headers["x-model-georouter-request-id"]
    for name in ("authorization", "proxy-authorization", "cookie", "x-api-key", "x-goog-api-key"):
        assert name not in seen.headers


@pytest.mark.parametrize(
    ("path", "body"),
    [
        (STREAM_PATH + "-lookalike", _body()),
        ("/cloudcode/v1internal:loadCodeAssist", _body()),
        (STREAM_PATH, _body("other-model")),
        (STREAM_PATH, b"{}"),
    ],
)
def test_non_match_keeps_google_route(track: Track, start_bridge: StartBridge, path: str, body: bytes) -> None:
    def unused_sidecar(conn: socket.socket, _req: Req) -> None:
        respond(conn, 200, b"unexpected")

    google, sidecar, bridge = _launch(track, start_bridge, unused_sidecar)
    response = http_request(bridge.port, "POST", path, body, {"Authorization": AUTH})

    assert response.status == 200 and response.body == b"google"
    assert len(google.requests) == 1 and sidecar.requests == []
    assert google.requests[0].headers["authorization"] == AUTH and google.requests[0].body == body


def test_off_identity_for_cloudcode_catalog_load_assist_and_gemini(track: Track, start_bridge: StartBridge) -> None:
    def unused_sidecar(conn: socket.socket, _req: Req) -> None:
        respond(conn, 200, b"unexpected")

    google, sidecar, bridge = _launch(track, start_bridge, unused_sidecar, enabled=False)
    cases = (
        (CATALOG_PATH, b"{}"),
        (STREAM_PATH, _body()),
        ("/cloudcode/v1internal:loadCodeAssist", b"{}"),
        ("/v1/models/gemini-pro:generateContent", b'{"contents":[]}'),
    )
    for path, body in cases:
        response = http_request(bridge.port, "POST", path + "?original=1", body, {"Authorization": AUTH})
        assert response.status == 200 and response.body == b"google"

    assert sidecar.requests == [] and len(google.requests) == len(cases)
    assert [seen.body for seen in google.requests] == [body for _path, body in cases]
    assert all(seen.headers["authorization"] == AUTH for seen in google.requests)
    assert all(seen.path.endswith("?original=1") for seen in google.requests)

    response = http_request(
        bridge.port,
        "POST",
        CATALOG_PATH,
        b"{}",
        {"X-Model-Georouter-Catalog-Ticket": "off-client-value"},
    )
    assert response.status == 200 and response.body == b"google"
    assert google.requests[-1].headers["x-model-georouter-catalog-ticket"] == "off-client-value"
    assert sidecar.requests == []


def test_gemini_route_is_unaffected_when_sidecar_enabled(track: Track, start_bridge: StartBridge) -> None:
    def unused_sidecar(conn: socket.socket, _req: Req) -> None:
        respond(conn, 503, b"unexpected")

    google, sidecar, bridge = _launch(track, start_bridge, unused_sidecar)
    response = http_request(
        bridge.port,
        "POST",
        "/v1/models/gemini-pro:generateContent",
        b'{"contents":[]}',
        {"Authorization": AUTH},
    )

    assert response.status == 200 and response.body == b"google"
    assert len(google.requests) == 1 and sidecar.requests == []
    assert bridge.server.geo_cache.is_blocked("cloudcode-pa", STANDARD) is False


def test_expired_catalog_ticket_is_invalid_for_exists_and_consume() -> None:
    now = [0.0]
    store = CatalogTicketStore(lambda: now[0])
    token = _issue_catalog_ticket(store)

    assert store.exists(token)
    now[0] = 10.0
    assert not store.exists(token)
    assert store.consume(token) is None


def test_catalog_ticket_cannot_be_consumed_twice() -> None:
    store = CatalogTicketStore(lambda: 0.0)
    token = _issue_catalog_ticket(store)

    assert store.consume(token) is not None
    assert store.consume(token) is None


def test_catalog_callback_success_single_use_and_google_header_stripping(
    track: Track, start_bridge: StartBridge, caplog: pytest.LogCaptureFixture
) -> None:
    callback_statuses: list[int] = []
    tickets: list[str] = []

    def sidecar_merge(conn: socket.socket, req: Req) -> None:
        ticket = req.headers["x-model-georouter-catalog-ticket"]
        tickets.append(ticket)
        callback_path = req.headers["x-model-georouter-catalog-callback-path"]
        assert callback_path == CATALOG_PATH
        for body in (b'{"ignored":true}', b'{"ignored":"repeat"}'):
            callback = http_request(
                bridge.port,
                "POST",
                callback_path,
                body,
                {
                    "X-Model-Georouter-Catalog-Ticket": ticket,
                    "X-Model-Georouter-Internal": "remove-me",
                    "Authorization": "Bearer callback-secret",
                },
            )
            callback_statuses.append(callback.status)
        respond(conn, 200, b"merged-catalog")

    google, sidecar, bridge = _launch(track, start_bridge, sidecar_merge)
    response = http_request(bridge.port, "POST", CATALOG_PATH, b'{"original":true}', {"Authorization": AUTH})

    assert response.status == 200 and response.body == b"merged-catalog"
    assert callback_statuses == [200, 403]
    assert len(sidecar.requests) == 1 and len(google.requests) == 1
    seen = google.requests[0]
    assert seen.body == b'{"original":true}' and seen.headers["authorization"] == AUTH
    assert "x-model-georouter-catalog-ticket" not in seen.headers
    assert "x-model-georouter-internal" not in seen.headers
    assert tickets[0] not in caplog.text and AUTH not in caplog.text and "callback-secret" not in caplog.text


def test_catalog_callback_expiry_fails_closed_then_parent_fails_open(track: Track, start_bridge: StartBridge) -> None:
    now = [0.0]
    callback_statuses: list[int] = []

    def expired_callback(conn: socket.socket, req: Req) -> None:
        now[0] = 11.0
        callback = http_request(
            bridge.port,
            "POST",
            req.headers["x-model-georouter-catalog-callback-path"],
            b"{}",
            {"X-Model-Georouter-Catalog-Ticket": req.headers["x-model-georouter-catalog-ticket"]},
        )
        callback_statuses.append(callback.status)
        respond(conn, 502, b"catalog merge failed")

    google, _sidecar, bridge = _launch(track, start_bridge, expired_callback)
    bridge.server._catalog_clock = lambda: now[0]
    response = http_request(bridge.port, "POST", CATALOG_PATH, b'{"original":true}', {"Authorization": AUTH})

    assert callback_statuses == [403]
    assert response.status == 200 and response.body == b"google"
    assert len(google.requests) == 1 and google.requests[0].body == b'{"original":true}'


@pytest.mark.parametrize("mode", ["status", "transport"])
def test_catalog_precommit_fallback_revokes_ticket(track: Track, start_bridge: StartBridge, mode: str) -> None:
    callbacks: list[tuple[str, str]] = []

    def fail_before_commit(conn: socket.socket, req: Req) -> None:
        callbacks.append(
            (
                req.headers["x-model-georouter-catalog-callback-path"],
                req.headers["x-model-georouter-catalog-ticket"],
            )
        )
        if mode == "status":
            respond(conn, 503, b"sidecar error")

    google, _sidecar, bridge = _launch(track, start_bridge, fail_before_commit)
    response = http_request(bridge.port, "POST", CATALOG_PATH, b'{"original":true}', {"Authorization": AUTH})
    callback_path, ticket = callbacks[0]
    late_callback = http_request(
        bridge.port,
        "POST",
        callback_path,
        b"{}",
        {"X-Model-Georouter-Catalog-Ticket": ticket},
    )

    assert response.status == 200 and response.body == b"google"
    assert late_callback.status == 403
    assert len(google.requests) == 1 and google.requests[0].body == b'{"original":true}'


def test_catalog_ticket_is_bound_to_normalized_rpc(track: Track, start_bridge: StartBridge) -> None:
    callback_statuses: list[int] = []

    def wrong_rpc(conn: socket.socket, req: Req) -> None:
        callback = http_request(
            bridge.port,
            "POST",
            STREAM_PATH,
            _body(),
            {"X-Model-Georouter-Catalog-Ticket": req.headers["x-model-georouter-catalog-ticket"]},
        )
        callback_statuses.append(callback.status)
        respond(conn, 502, b"wrong callback")

    google, _sidecar, bridge = _launch(track, start_bridge, wrong_rpc)
    response = http_request(bridge.port, "POST", CATALOG_PATH, b'{"original":true}')

    assert callback_statuses == [403] and response.body == b"google"
    assert len(google.requests) == 1 and google.requests[0].path == "/v1internal:fetchAvailableModels"


def test_loopback_peer_check() -> None:
    assert _Handler._loopback_peer(("127.0.0.1", 12345))
    assert _Handler._loopback_peer(("::1", 12345))
    assert not _Handler._loopback_peer(("192.0.2.1", 12345))
    assert not _Handler._loopback_peer(object())


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_transient_status_falls_back_once_with_original_context(
    track: Track, start_bridge: StartBridge, status: int
) -> None:
    def transient(conn: socket.socket, _req: Req) -> None:
        respond(conn, status, b"sidecar error")

    google, sidecar, bridge = _launch(track, start_bridge, transient)
    response = http_request(
        bridge.port,
        "POST",
        STREAM_PATH + "?alt=sse",
        _body(),
        {"Authorization": AUTH, "Cookie": "original-cookie", "X-Api-Key": "original-key"},
    )

    assert response.status == 200 and response.body == b"google"
    assert len(sidecar.requests) == 1 and len(google.requests) == 1
    seen = google.requests[0]
    assert seen.path == "/v1internal:streamGenerateContent?alt=sse" and seen.body == _body()
    assert seen.headers["authorization"] == AUTH
    assert seen.headers["cookie"] == "original-cookie" and seen.headers["x-api-key"] == "original-key"
    assert not any(name.startswith("x-model-georouter-") for name in seen.headers)


@pytest.mark.parametrize("status", [400, 422, 501])
def test_pre_dispatch_rejection_falls_back_once(track: Track, start_bridge: StartBridge, status: int) -> None:
    def reject(conn: socket.socket, _req: Req) -> None:
        respond(conn, status, b"rejected", {"X-Sidecar-Pre-Dispatch": "1"})

    google, sidecar, bridge = _launch(track, start_bridge, reject)
    response = http_request(bridge.port, "POST", STREAM_PATH, _body())

    assert response.status == 200 and response.body == b"google"
    assert len(sidecar.requests) == 1 and len(google.requests) == 1


@pytest.mark.parametrize("status", [400, 422, 501])
def test_contract_status_without_pre_dispatch_marker_does_not_fall_back(
    track: Track, start_bridge: StartBridge, status: int
) -> None:
    def client_error(conn: socket.socket, _req: Req) -> None:
        respond(conn, status, b"client error")

    google, sidecar, bridge = _launch(track, start_bridge, client_error)
    response = http_request(bridge.port, "POST", STREAM_PATH, _body())

    assert response.status == status and response.body == b"client error"
    assert len(sidecar.requests) == 1 and google.requests == []


def test_connection_refused_falls_back_once(track: Track, start_bridge: StartBridge) -> None:
    google = track(HttpUpstream(_google_ok))
    bridge = start_bridge(_config(google.port, free_port()))

    response = http_request(bridge.port, "POST", STREAM_PATH, _body(), {"Authorization": AUTH})

    assert response.status == 200 and response.body == b"google"
    assert len(google.requests) == 1 and google.requests[0].headers["authorization"] == AUTH


@pytest.mark.parametrize("mode", ["eof", "reset"])
def test_pre_header_eof_and_reset_fall_back_once(track: Track, start_bridge: StartBridge, mode: str) -> None:
    def close_early(conn: socket.socket, _req: Req) -> None:
        if mode == "reset":
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))

    google, sidecar, bridge = _launch(track, start_bridge, close_early)
    response = http_request(bridge.port, "POST", STREAM_PATH, _body())

    assert response.status == 200 and response.body == b"google"
    assert len(sidecar.requests) == 1 and len(google.requests) == 1


def test_header_timeout_falls_back_once(track: Track, start_bridge: StartBridge) -> None:
    def silent(conn: socket.socket, _req: Req) -> None:
        time.sleep(0.2)
        respond(conn, 200, b"too late")

    google, sidecar, bridge = _launch(track, start_bridge, silent)
    snapshot = bridge.server.config_manager.get()
    bridge.server.config_manager._snapshot = replace(
        snapshot,
        cloudcode_claude_code=replace(snapshot.cloudcode_claude_code, sidecar_header_timeout_seconds=0.05),
    )
    started = time.monotonic()
    response = http_request(bridge.port, "POST", STREAM_PATH, _body())

    assert time.monotonic() - started < 1.0
    assert response.status == 200 and response.body == b"google"
    assert len(sidecar.requests) == 1 and len(google.requests) == 1


@pytest.mark.parametrize(
    ("status", "headers"),
    [(503, {}), (504, {}), (422, {"X-Sidecar-Pre-Dispatch": "1"})],
)
def test_synthetic_only_id_never_goes_to_google(
    track: Track, start_bridge: StartBridge, status: int, headers: dict[str, str]
) -> None:
    def failed(conn: socket.socket, _req: Req) -> None:
        respond(conn, status, b"sanitized sidecar error", headers)

    google, sidecar, bridge = _launch(track, start_bridge, failed)
    response = http_request(bridge.port, "POST", STREAM_PATH, _body(SYNTHETIC), {"Authorization": AUTH})

    assert response.status == status and response.body == b"sanitized sidecar error"
    assert len(sidecar.requests) == 1 and google.requests == []


def test_synthetic_transport_failure_never_goes_to_google(track: Track, start_bridge: StartBridge) -> None:
    google, sidecar, bridge = _launch(track, start_bridge, lambda _conn, _req: None)
    response = http_request(bridge.port, "POST", STREAM_PATH, _body(SYNTHETIC), {"Authorization": AUTH})

    assert response.status == 502
    assert len(sidecar.requests) == 1 and google.requests == []


@pytest.mark.parametrize("first_chunk", [b"data: first\n\n", b": keepalive\n\n"])
def test_post_commit_reset_has_zero_fallback(track: Track, start_bridge: StartBridge, first_chunk: bytes) -> None:
    def commit_then_reset(conn: socket.socket, _req: Req) -> None:
        send_sse_head(conn)
        conn.sendall(chunk(first_chunk))
        time.sleep(0.1)  # дать relay отправить первый chunk до TCP RST
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))

    google, sidecar, bridge = _launch(track, start_bridge, commit_then_reset)
    response = http_request(bridge.port, "POST", STREAM_PATH + "?alt=sse", _body())

    assert response.status == 200 and first_chunk in response.body
    assert len(sidecar.requests) == 1 and google.requests == []
