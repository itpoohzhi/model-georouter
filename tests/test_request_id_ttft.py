"""Тесты валидации/генерации X-Request-Id и замера ttft_ms в лог-строке request."""

from __future__ import annotations

import json
import logging
import re
import socket
import threading

import pytest

from model_georouter.bridge_server import _resolve_request_id
from tests.helpers import (
    LOCALHOST,
    HttpUpstream,
    chunk,
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

PATH = "/v1/chat/completions"
GENERATED_REQ_ID_RE = re.compile(r"^req_[0-9a-f]{16}$")
REQUEST_TAIL_RE = re.compile(
    r" replayed=(?:True|False) req_id=(?P<req_id>\S+) ttft_ms=(?P<ttft_ms>-|\d+) dur_ms=(?P<dur_ms>\d+)$"
)


def _body_for(model: str = "m", *, stream: bool = False) -> bytes:
    return json.dumps(
        {"model": model, "messages": [{"role": "user", "content": "hi"}], "stream": stream}
    ).encode()


def _ok_responder(label: bytes = b"ok"):
    def responder(conn, req):
        respond(conn, 200, label, {"Content-Type": "text/plain"})

    return responder


def _request_log_lines(caplog, prefix: str = "request method=") -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(prefix)]


# ───────────────────────────── юнит-тесты _resolve_request_id ─────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("req_abcdef0123456789", "req_abcdef0123456789"),
        ("a", "a"),
        ("A-b.c_9", "A-b.c_9"),
        ("a" * 64, "a" * 64),
        ("A0_-.z" * 10 + "1234", "A0_-.z" * 10 + "1234"),
        ("  req_trimmed_1  ", "req_trimmed_1"),
    ],
)
def test_resolve_request_id_accepts_valid_values(raw: str, expected: str) -> None:
    """Валидные X-Request-Id (1..64 символов из allowlist) возвращаются как есть."""
    assert _resolve_request_id(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "   ",
        "foo bar",
        "foo\nbar",
        "foo\r\nX-Injected: 1",
        "req_ok\n",
        "\nreq_ok",
        "id\x00x",
        "id\x1bx",
        "a" * 65,
        "../etc/passwd",
    ],
)
def test_resolve_request_id_rejects_invalid_and_generates(raw: str | None) -> None:
    """Невалидные и отсутствующие значения заменяются сгенерированным req_<16hex>."""
    resolved = _resolve_request_id(raw)
    assert GENERATED_REQ_ID_RE.fullmatch(resolved)
    if raw is not None:
        assert resolved != raw
        assert resolved != raw.strip()


# ───────────────────────────── E2E: X-Request-Id и log injection ─────────────────────────────


def test_e2e_valid_request_id_logged(track, start_bridge, caplog) -> None:
    """Валидный X-Request-Id попадает в лог-строку request вместе с ttft_ms и dur_ms."""
    up = track(HttpUpstream(_ok_responder(b"ok")))
    bridge = start_bridge(make_config(up.port))
    with caplog.at_level(logging.INFO, logger="model_georouter.server"):
        resp = http_request(bridge.port, "POST", PATH, _body_for(), {"X-Request-Id": "req_test123"})
        assert resp.status == 200
        assert wait_for(lambda: len(_request_log_lines(caplog, "request method=POST")) == 1)
    line = _request_log_lines(caplog, "request method=POST")[0]
    match = REQUEST_TAIL_RE.search(line)
    assert match is not None, line
    assert match.group("req_id") == "req_test123"
    assert match.group("ttft_ms").isdigit()
    assert 0 <= int(match.group("ttft_ms")) <= int(match.group("dur_ms"))


def test_e2e_missing_request_id_generates_id(track, start_bridge, caplog) -> None:
    """Без заголовка X-Request-Id мост генерирует correlation ID формата req_<16hex>."""
    up = track(HttpUpstream(_ok_responder(b"ok")))
    bridge = start_bridge(make_config(up.port))
    with caplog.at_level(logging.INFO, logger="model_georouter.server"):
        resp = http_request(bridge.port, "POST", PATH, _body_for())
        assert resp.status == 200
        assert wait_for(lambda: len(_request_log_lines(caplog, "request method=POST")) == 1)
    line = _request_log_lines(caplog, "request method=POST")[0]
    match = REQUEST_TAIL_RE.search(line)
    assert match is not None, line
    assert GENERATED_REQ_ID_RE.fullmatch(match.group("req_id"))
    assert match.group("ttft_ms").isdigit()


@pytest.mark.parametrize("bad_value", ["bad id", "a" * 65, "../etc/passwd", "id\x1bx"])
def test_e2e_invalid_request_id_replaced_and_never_logged(track, start_bridge, caplog, bad_value: str) -> None:
    """Невалидный X-Request-Id заменяется сгенерированным и не попадает в лог в сыром виде."""
    up = track(HttpUpstream(_ok_responder(b"ok")))
    bridge = start_bridge(make_config(up.port))
    with caplog.at_level(logging.INFO, logger="model_georouter.server"):
        resp = http_request(bridge.port, "POST", PATH, _body_for(), {"X-Request-Id": bad_value})
        assert resp.status == 200
        assert wait_for(lambda: len(_request_log_lines(caplog, "request method=POST")) == 1)
    line = _request_log_lines(caplog, "request method=POST")[0]
    match = REQUEST_TAIL_RE.search(line)
    assert match is not None, line
    assert GENERATED_REQ_ID_RE.fullmatch(match.group("req_id"))
    assert bad_value not in caplog.text


def test_e2e_bare_lf_in_request_id_rejected_without_logging_value(track, start_bridge, caplog) -> None:
    """Сырой LF в заголовке X-Request-Id отклоняется парсером, сырое значение не попадает в лог."""
    up = track(HttpUpstream(_ok_responder(b"ok")))
    bridge = start_bridge(make_config(up.port))
    raw_req = (
        f"POST {PATH} HTTP/1.1\r\n"
        f"Host: {LOCALHOST}\r\n"
        "X-Request-Id: evil_hdr\ninjected=true\r\n"
        "Content-Length: 0\r\n\r\n"
    ).encode("latin-1")
    with caplog.at_level(logging.INFO, logger="model_georouter.server"):
        with socket.create_connection((LOCALHOST, bridge.port), timeout=5) as sock:
            sock.sendall(raw_req)
            resp = parse_response(read_all(sock))
        assert resp.status == 400
        assert wait_for(lambda: len(_request_log_lines(caplog, "request method=-")) == 1)
    line = _request_log_lines(caplog, "request method=-")[0]
    match = REQUEST_TAIL_RE.search(line)
    assert match is not None, line
    assert match.group("req_id") == "-"
    assert match.group("ttft_ms") == "-"
    assert "evil_hdr" not in caplog.text
    assert "injected=true" not in caplog.text


# ───────────────────────────── E2E: ttft_ms ─────────────────────────────


def test_e2e_ttft_recorded_for_streaming_response(track, start_bridge, caplog) -> None:
    """Стриминговый SSE-ответ фиксирует числовой ttft_ms <= dur_ms при первом байте клиенту."""
    release = threading.Event()

    def streaming(conn, req):
        send_sse_head(conn)
        conn.sendall(chunk(b"data: first\n\n"))
        release.wait(timeout=5.0)
        conn.sendall(chunk(b"data: second\n\n"))
        conn.sendall(b"0\r\n\r\n")

    up = track(HttpUpstream(streaming))
    bridge = start_bridge(make_config(up.port))
    with caplog.at_level(logging.INFO, logger="model_georouter.server"):
        sock = open_request(bridge.port, "POST", PATH, _body_for(stream=True), {"X-Request-Id": "req_sse_1"})
        try:
            first = read_until(sock, b"data: first\n\n")
            release.set()
            tail = read_all(sock)
        finally:
            sock.close()
        resp = parse_response(first + tail)
        assert resp.status == 200
        assert b"data: second" in resp.body
        assert wait_for(lambda: len(_request_log_lines(caplog, "request method=POST")) == 1)
    line = _request_log_lines(caplog, "request method=POST")[0]
    match = REQUEST_TAIL_RE.search(line)
    assert match is not None, line
    assert match.group("req_id") == "req_sse_1"
    assert match.group("ttft_ms").isdigit()
    assert 0 <= int(match.group("ttft_ms")) <= int(match.group("dur_ms"))


def test_e2e_ttft_dash_when_no_upstream_response(track, start_bridge, caplog) -> None:
    """Локальные ответы и ошибки до upstream (404 без адаптера, /health) логируют ttft_ms=-."""
    up = track(HttpUpstream(_ok_responder(b"ok")))
    bridge = start_bridge(make_config(up.port))
    with caplog.at_level(logging.INFO, logger="model_georouter.server"):
        resp_404 = http_request(
            bridge.port, "GET", "/unknown/no-adapter", headers={"X-Request-Id": "req_404_test"}
        )
        assert resp_404.status == 404
        assert len(up.requests) == 0
        assert wait_for(lambda: len(_request_log_lines(caplog, "request method=GET path=/unknown/no-adapter")) == 1)
        line_404 = _request_log_lines(caplog, "request method=GET path=/unknown/no-adapter")[0]
        match_404 = REQUEST_TAIL_RE.search(line_404)
        assert match_404 is not None, line_404
        assert match_404.group("req_id") == "req_404_test"
        assert match_404.group("ttft_ms") == "-"
        assert match_404.group("dur_ms").isdigit()

        resp_health = http_request(bridge.port, "GET", "/health", headers={"X-Request-Id": "req_health_1"})
        assert resp_health.status == 200
        assert wait_for(lambda: len(_request_log_lines(caplog, "request method=GET path=/health")) == 1)
        line_health = _request_log_lines(caplog, "request method=GET path=/health")[0]
        match_health = REQUEST_TAIL_RE.search(line_health)
        assert match_health is not None, line_health
        assert match_health.group("req_id") == "req_health_1"
        assert match_health.group("ttft_ms") == "-"
        assert match_health.group("dur_ms").isdigit()
