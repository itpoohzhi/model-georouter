from __future__ import annotations

import json
import logging
import socket
import threading
import time

import pytest

from tests.helpers import chunk, dechunk
from universal_ai_bridge.error_handler import ErrorHandler
from universal_ai_bridge.errors import (
    AllProxiesPenalizedError,
    BadRequestError,
    BodyTooLargeError,
    EgressConnectError,
    EgressTimeoutError,
    FramingError,
    SlotsExhaustedError,
    UpstreamProtocolError,
    UpstreamTimeoutError,
)
from universal_ai_bridge.http_wire import (
    ChunkedFraming,
    EofFraming,
    LengthFraming,
    NoBodyFraming,
    build_client_head,
    framing_for_request,
    framing_for_response,
    parse_request_head,
    parse_response_head,
)
from universal_ai_bridge.logging_utils import get_logger, mask_secrets
from universal_ai_bridge.relay import (
    CLIENT_CANCELLED,
    CLIENT_WRITE_TIMEOUT,
    COMPLETE,
    INACTIVITY_TIMEOUT,
    UPSTREAM_TRUNCATED,
    SSERelay,
)

# ───────────────────────────── ErrorHandler ─────────────────────────────

HANDLER = ErrorHandler()


@pytest.mark.parametrize(
    "exc, status, etype, retryable",
    [
        (AllProxiesPenalizedError("route-de"), 502, "proxy_unavailable", True),
        (EgressTimeoutError("connect timed out"), 504, "proxy_gateway_timeout", True),
        (EgressConnectError("refused"), 502, "proxy_connect_failed", True),
        (UpstreamTimeoutError("no headers"), 504, "upstream_timeout", True),
        (UpstreamProtocolError("closed"), 502, "upstream_unreachable", True),
        (SlotsExhaustedError("proxy"), 503, "bridge_overloaded", True),
        (BodyTooLargeError(100), 413, "payload_too_large", False),
        (BadRequestError("bad"), 400, "bad_request", False),
        (RuntimeError("boom with Bearer sk-live123456789"), 502, "bridge_internal_error", False),
    ],
)
def test_error_mapping(exc, status, etype, retryable):
    response = HANDLER.convert(exc)
    assert response.status == status
    error = response.payload["error"]
    assert error["type"] == etype and error["retryable"] is retryable and isinstance(error["message"], str)
    assert "sk-live123456789" not in response.body.decode()
    assert response.status != 403


def test_transport_failures_never_map_to_403():
    transport = [
        AllProxiesPenalizedError("p"),
        EgressConnectError("x"),
        EgressTimeoutError("x"),
        UpstreamTimeoutError("x"),
        UpstreamProtocolError("x"),
        ConnectionResetError(),
        TimeoutError(),
    ]
    assert all(HANDLER.convert(exc).status in (502, 504) for exc in transport)
    assert HANDLER.build(403, "forbidden", "nope", retryable=False).status == 502  # инвариант


def test_error_response_wire_format_and_headers():
    response = HANDLER.convert(EgressTimeoutError("connect to h:1 via http://p:3128 timed out"))
    raw = response.to_bytes()
    head, _, body = raw.partition(b"\r\n\r\n")
    assert head.startswith(b"HTTP/1.1 504 Gateway Timeout")
    assert b"Content-Type: application/json" in head and b"Connection: close" in head
    assert f"Content-Length: {len(body)}".encode() in head
    assert json.loads(body) == {
        "error": {
            "type": "proxy_gateway_timeout",
            "message": "connect to h:1 via http://p:3128 timed out",
            "retryable": True,
        }
    }
    overload = HANDLER.convert(SlotsExhaustedError("proxy"))
    assert overload.headers["Retry-After"] == "5"
    assert overload.payload["error"]["retry_after_seconds"] == 5


def test_error_messages_mask_proxy_credentials():
    response = HANDLER.convert(EgressConnectError("proxy http://user:hunter2@10.0.0.1:3128 failed"))
    assert "hunter2" not in response.body.decode()


# ───────────────────────────── маскирование секретов ─────────────────────────────


@pytest.mark.parametrize(
    "raw, masked, secret",
    [
        ("Authorization: Bearer sk-abcdef1234567890", "Bearer sk-***", "abcdef1234567890"),
        ("header Bearer st-token_value_123", "Bearer st-***", "token_value_123"),
        ("header bearer st_token_value_123", "bearer st_***", "token_value_123"),
        ("Authorization: Bearer eyJhbGciOi.payload.sig", "Bearer ***", "eyJhbGciOi"),
        ("Proxy-Authorization: Basic dXNlcjpwYXNz", "Basic ***", "dXNlcjpwYXNz"),
        ("via http://user:s3cr3t@proxy.local:3128", "http://user:***@proxy.local:3128", "s3cr3t"),
        ("socks5h://bob:p%40ss@127.0.0.1:1080", "socks5h://bob:***@127.0.0.1:1080", "p%40ss"),
        ("GET /v1/x?key=AbC123&b=2", "?key=***&b=2", "AbC123"),
        ('{"password": "hunter2", "x": 1}', '"password": "***"', "hunter2"),
        ("x-api-key: sk-1234567890abcdef", "x-api-key: ***", "1234567890abcdef"),
        ("standalone sk-1234567890abcdef in text", "sk-***", "1234567890abcdef"),
    ],
)
def test_mask_secrets(raw, masked, secret):
    result = mask_secrets(raw)
    assert masked in result
    assert secret not in result
    assert mask_secrets(result) == result  # идемпотентно


def test_mask_secrets_leaves_ordinary_text_alone():
    text = "basic usage of the bridge; model=gpt-5 status=200 pool=route-de"
    assert mask_secrets(text) == text


def test_logger_filter_masks_secrets_in_records(caplog):
    logger = get_logger("universal_ai_bridge.test_masking")
    with caplog.at_level(logging.INFO, logger=logger.name):
        logger.info("upstream said: Authorization: Bearer sk-supersecret1234 via %s", "http://u:pw123@h:1")
    assert "Bearer sk-***" in caplog.text
    assert "supersecret1234" not in caplog.text and "pw123" not in caplog.text


# ───────────────────────────── обрамление тела ─────────────────────────────

CHUNKED = chunk(b"data: one\n\n") + chunk(b"data: two\n\n") + b"0\r\n\r\n"


def test_chunked_framing_at_every_split_point():
    wire = CHUNKED + b"EXTRA-NOT-PART-OF-BODY"
    for cut in range(len(CHUNKED) + 1):
        collected = bytearray()
        framing = ChunkedFraming(collected.extend)
        consumed = framing.feed(wire[:cut])
        if not framing.complete:
            consumed += framing.feed(wire[cut:])
        assert framing.complete
        assert consumed == len(CHUNKED), f"split at {cut}"
        assert bytes(collected) == b"data: one\n\ndata: two\n\n"


def test_chunked_framing_with_extensions_and_trailers():
    wire = b"4;ext=1\r\nabcd\r\n0\r\nX-Trailer: v\r\n\r\n"
    framing = ChunkedFraming()
    assert framing.feed(wire) == len(wire) and framing.complete


def test_chunked_framing_incomplete_is_not_complete_on_eof():
    framing = ChunkedFraming()
    framing.feed(chunk(b"hello"))
    assert not framing.complete and not framing.on_eof()


@pytest.mark.parametrize("bad", [b"zz\r\nabc\r\n", b"-1\r\n", b"\r\n"])
def test_chunked_framing_rejects_garbage(bad):
    with pytest.raises(FramingError):
        ChunkedFraming().feed(bad)


def test_length_eof_and_nobody_framing():
    out = bytearray()
    framing = LengthFraming(5, out.extend)
    assert framing.feed(b"abc") == 3 and not framing.complete
    assert framing.feed(b"defgh") == 2 and framing.complete
    assert bytes(out) == b"abcde"
    assert LengthFraming(0).complete and NoBodyFraming().complete
    eof = EofFraming()
    assert eof.feed(b"xyz") == 3 and not eof.complete and eof.on_eof()


def test_framing_selection_for_requests_and_responses():
    assert isinstance(framing_for_request([("Transfer-Encoding", "chunked")], None, 10), ChunkedFraming)
    assert isinstance(framing_for_request([("Content-Length", "5")], None, 10), LengthFraming)
    assert isinstance(framing_for_request([], None, 10), NoBodyFraming)
    with pytest.raises(BodyTooLargeError):
        framing_for_request([("Content-Length", "11")], None, 10)
    for headers in (
        [("Content-Length", "1"), ("Transfer-Encoding", "chunked")],
        [("Content-Length", "-1")],
        [("Content-Length", "1"), ("Content-Length", "1")],
        [("Transfer-Encoding", "gzip, chunked")],
    ):
        with pytest.raises(BadRequestError):
            framing_for_request(headers, None, 10)
    assert isinstance(framing_for_response("GET", 200, [("Transfer-Encoding", "chunked")]), ChunkedFraming)
    assert isinstance(framing_for_response("GET", 200, [("Content-Length", "3")]), LengthFraming)
    assert isinstance(framing_for_response("GET", 200, []), EofFraming)
    assert isinstance(framing_for_response("HEAD", 200, [("Content-Length", "3")]), NoBodyFraming)
    assert isinstance(framing_for_response("GET", 204, []), NoBodyFraming)


def test_head_parsing_and_client_head_rewrite():
    req = parse_request_head(b"POST /v1/x?a=1 HTTP/1.1\r\nHost: h\r\nX-A: b: c\r\n\r\n")
    assert (req.method, req.target, req.get("x-a")) == ("POST", "/v1/x?a=1", "b: c")
    for broken in (b"GARBAGE\r\n\r\n", b"GET / HTTP/2\r\n\r\n", b"GET / HTTP/1.1\r\nno-colon\r\n\r\n"):
        with pytest.raises(BadRequestError):
            parse_request_head(broken)
    resp = parse_response_head(
        b"HTTP/1.1 200 OK\r\nConnection: keep-alive, X-Hop\r\nX-Hop: 1\r\nKeep-Alive: timeout=5\r\n"
        b"Transfer-Encoding: chunked\r\nContent-Type: text/event-stream\r\n\r\n"
    )
    head = build_client_head(resp).decode()
    assert head.startswith("HTTP/1.1 200 OK\r\n")
    assert "Transfer-Encoding: chunked" in head and "Content-Type: text/event-stream" in head
    assert "X-Hop" not in head and "Keep-Alive" not in head and head.count("Connection:") == 1
    with pytest.raises(UpstreamProtocolError):
        parse_response_head(b"NOPE\r\n\r\n")


# ───────────────────────────── SSERelay ─────────────────────────────


class RelayRig:
    """upstream_peer → [relay] → client_peer на socketpair'ах; relay крутится в отдельном потоке."""

    def __init__(self, framing, *, inactivity=5.0, write_timeout=5.0, initial=b"", initial_fed=False):
        self.up_relay_side, self.up_peer = socket.socketpair()
        self.client_relay_side, self.client_peer = socket.socketpair()
        self.result = None
        relay = SSERelay(inactivity_timeout=inactivity, write_timeout=write_timeout, poll_interval=0.05)

        def run():
            self.result = relay.relay(
                self.up_relay_side, self.client_relay_side, framing, initial=initial, initial_fed=initial_fed
            )
            self.client_relay_side.close()  # как сервер: закрыть клиента «как есть»

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def read_client_until_eof(self, timeout=10.0) -> bytes:
        buf = b""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.client_peer.settimeout(max(0.05, deadline - time.monotonic()))
            data = self.client_peer.recv(65536)
            if not data:
                return buf
            buf += data
        raise TimeoutError

    def finish(self):
        self.thread.join(10)
        assert not self.thread.is_alive()
        for sock in (self.up_peer, self.client_peer):
            sock.close()
        return self.result


def upstream_is_closed(sock: socket.socket, timeout=3.0) -> bool:
    sock.settimeout(timeout)
    try:
        return sock.recv(1) == b""
    except ConnectionResetError:
        return True
    except TimeoutError:
        return False


def test_relay_streams_chunks_as_they_arrive():
    rig = RelayRig(ChunkedFraming())
    rig.up_peer.sendall(chunk(b"data: one\n\n"))
    rig.client_peer.settimeout(3)
    first = rig.client_peer.recv(65536)  # пришло до того, как upstream закончил
    assert first == chunk(b"data: one\n\n")
    time.sleep(0.05)
    rig.up_peer.sendall(chunk(b"data: two\n\n"))
    rig.up_peer.sendall(b"0\r\n\r\n")
    rest = rig.read_client_until_eof()
    assert dechunk(first + rest) == (b"data: one\n\ndata: two\n\n", True)
    result = rig.finish()
    assert result.ok and result.bytes_sent == len(CHUNKED)


def test_relay_sends_initial_bytes_first_and_stops_at_body_end():
    rig = RelayRig(LengthFraming(10), initial=b"hello")
    rig.up_peer.sendall(b"world-and-garbage")
    assert rig.read_client_until_eof() == b"helloworld"
    assert rig.finish().outcome == COMPLETE


def test_relay_initial_fed_does_not_double_count():
    framing = LengthFraming(6)
    framing.feed(b"abc")
    rig = RelayRig(framing, initial=b"abc", initial_fed=True)
    rig.up_peer.sendall(b"def")
    assert rig.read_client_until_eof() == b"abcdef"
    assert rig.finish().ok


def test_relay_eof_delimited_body_completes_on_close():
    rig = RelayRig(EofFraming())
    rig.up_peer.sendall(b"stream until close")
    rig.up_peer.close()
    assert rig.read_client_until_eof() == b"stream until close"
    assert rig.finish().outcome == COMPLETE


def test_mid_stream_upstream_failure_drops_socket_without_fake_done():
    rig = RelayRig(ChunkedFraming())
    rig.up_peer.sendall(chunk(b"data: partial\n\n"))
    time.sleep(0.1)
    rig.up_peer.close()  # upstream оборвался после 200 OK, без терминирующего chunk
    received = rig.read_client_until_eof()
    result = rig.finish()
    assert result.outcome == UPSTREAM_TRUNCATED
    assert received == chunk(b"data: partial\n\n")  # ровно то, что прислал upstream
    assert b"[DONE]" not in received and not received.endswith(b"0\r\n\r\n")


def test_content_length_truncation_is_detected():
    rig = RelayRig(LengthFraming(100))
    rig.up_peer.sendall(b"short")
    rig.up_peer.close()
    assert rig.read_client_until_eof() == b"short"
    assert rig.finish().outcome == UPSTREAM_TRUNCATED


def test_client_disconnect_closes_upstream_immediately():
    rig = RelayRig(ChunkedFraming())
    rig.up_peer.sendall(chunk(b"data: one\n\n"))
    rig.client_peer.settimeout(3)
    rig.client_peer.recv(65536)
    rig.client_peer.close()
    started = time.monotonic()
    assert upstream_is_closed(rig.up_peer, 3.0)  # upstream-сокет закрыт моментально
    assert time.monotonic() - started < 1.5
    assert rig.finish().outcome == CLIENT_CANCELLED


def test_client_reset_mid_write_is_cancel():
    rig = RelayRig(ChunkedFraming())
    rig.client_peer.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
    rig.client_peer.close()
    time.sleep(0.1)
    try:
        for _ in range(50):
            rig.up_peer.sendall(chunk(b"x" * 1000))
            time.sleep(0.01)
    except OSError:
        pass
    assert rig.finish().outcome == CLIENT_CANCELLED


def test_inactivity_timeout_closes_both_sides():
    rig = RelayRig(ChunkedFraming(), inactivity=0.3)
    started = time.monotonic()
    assert rig.read_client_until_eof() == b""
    assert time.monotonic() - started < 3
    assert upstream_is_closed(rig.up_peer)
    assert rig.finish().outcome == INACTIVITY_TIMEOUT


def test_backpressure_pauses_reading_upstream_until_client_drains():
    total = 24 * 1024 * 1024
    rig = RelayRig(LengthFraming(total))
    sent = [0]
    done = threading.Event()

    def producer():
        block = b"s" * 65536
        try:
            while sent[0] < total:
                rig.up_peer.sendall(block[: min(len(block), total - sent[0])])
                sent[0] += min(len(block), total - sent[0])
        except OSError:
            pass
        done.set()

    threading.Thread(target=producer, daemon=True).start()
    time.sleep(0.6)  # клиент ничего не читает
    assert not done.is_set()  # relay не вычерпал upstream в память: отправитель заблокирован
    assert sent[0] < total
    received = 0
    while received < total:
        rig.client_peer.settimeout(10)
        data = rig.client_peer.recv(1 << 20)
        assert data
        received += len(data)
    assert done.wait(10) and received == total
    assert rig.finish().ok


def test_client_write_timeout_when_client_never_reads():
    rig = RelayRig(EofFraming(), write_timeout=0.4)

    def flood():
        try:
            while True:
                rig.up_peer.sendall(b"f" * 65536)
        except OSError:
            pass

    threading.Thread(target=flood, daemon=True).start()
    result = rig.finish()
    assert result.outcome == CLIENT_WRITE_TIMEOUT


# ───────────────────────────── Rework Cycle 1 ─────────────────────────────


@pytest.mark.parametrize(
    "bad",
    [b"5\r\nhelloXX0\r\n\r\n", b"5\r\nhello\n\n0\r\n\r\n", b"5\r\nhello\rX0\r\n\r\n", b"5\r\nhelloX\n0\r\n\r\n"],
)
def test_chunked_framing_requires_crlf_after_chunk_data(bad):  # RW-011
    with pytest.raises(FramingError):
        ChunkedFraming().feed(bad)


def test_chunked_framing_crlf_split_across_feeds():  # RW-011
    framing = ChunkedFraming()
    for part in (b"5\r\nhello\r", b"\n0\r", b"\n\r\n"):
        framing.feed(part)
    assert framing.complete
    broken = ChunkedFraming()
    broken.feed(b"5\r\nhello\r")
    with pytest.raises(FramingError):
        broken.feed(b"X")


def test_parse_request_head_rejects_smuggling_headers():  # RW-010
    base = b"POST /x HTTP/1.1\r\nHost: h\r\n"
    for extra in (
        b"Content-Length: 2\r\nTransfer-Encoding: chunked\r\n",
        b"Transfer-Encoding: chunked\r\nContent-Length: 2\r\n",
        b"Content-Length: 2\r\nContent-Length: 5\r\n",
    ):
        with pytest.raises(BadRequestError):
            parse_request_head(base + extra + b"\r\n")
    assert parse_request_head(base + b"Content-Length: 2\r\n\r\n").get("content-length") == "2"
    assert parse_request_head(base + b"Transfer-Encoding: chunked\r\n\r\n").get("transfer-encoding") == "chunked"


def test_log_dir_and_rotated_files_are_private(tmp_path):  # RW-006
    import os
    import stat

    from universal_ai_bridge.logging_utils import add_file_handler

    logger = logging.getLogger("uab.test.private-log")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    directory = tmp_path / "logs"
    directory.mkdir()
    os.chmod(directory, 0o755)
    old_umask = os.umask(0)  # при umask 0 файлы без явного chmod получили бы 0666
    handler = None
    try:
        add_file_handler(logger, directory)
        handler = logger.handlers[-1]
        handler.maxBytes = 300
        for _ in range(40):
            logger.info("x" * 60)
    finally:
        os.umask(old_umask)
        if handler is not None:
            logger.removeHandler(handler)
            handler.close()
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    files = sorted(directory.glob("bridge.log*"))
    assert len(files) >= 3  # текущий лог и несколько архивов ротации
    assert {stat.S_IMODE(f.stat().st_mode) for f in files} == {0o600}
